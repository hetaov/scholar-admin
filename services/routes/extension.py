"""英文语句扩展路由（prefix=/english/extension）

契约：api-contract.md §3.18（E1~E3 + E4' / E6 + **E7 / E8 / E9 多轮**；E5 离线批量预生成本期不做）。
设计：docs_v1/扩展/第一期-scholar-admin接口与admin-web实验页-v1.md §3.3

2026-10-03（修订 2）：旧 E4 `PUT /points/{sentence_id}` 人工校对覆盖随 `source=manual`
全局校对层下线而整条删除（替代为 E4' `PUT /review/{sid}` / E6 `GET /review/{sid}`，
学习者判断只写 extension_review / extension_review_log，见红线 R9）。

信封风格 A：{"success": true, "data": ...}；业务失败 HTTP 200 + success=false + code；
任务不存在/过期 → HTTP 404。
··
鉴权：main.py 挂载在 _PAID_ROUTERS，require_paid_user 白名单。
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

import config
from services.dependencies import get_db
from services.english.extension import (
    get_cached_point,
    grade_l1_fill,
    resolve_effective_point,
)
from services.english.extension_review import get_review, merge_review_points, save_review
from services.english.extension_round import REGISTERS, DIFFICULTY_HARDER, DIFFICULTY_SAME
from services.learning.extension_round import STATUS_ACTIVE, get_round
from services.learning.extension_task import (
    create_extension_task,
    get_task,
    recover_task_if_stale,
    run_extension_task,
)
from services.providers.extension_llm import ExtensionError

logger = logging.getLogger("scholar-admin.routes.extension")

router = APIRouter(prefix="/english/extension", tags=["english-extension"])

# 后台任务强引用集合：防止 create_task 协程被 GC 回收（同 eval.py / dialogue.py）
_background_tasks: set[asyncio.Task] = set()

# 学习者自建点 id 前缀（契约 §3.18 `origin` 行）：命中即拒绝进判分，不静默过滤（B34）
_MINE_ID_PREFIX = "mine_"


async def _load_overlay(db, scholar_id: str, sentence_id: str) -> dict | None:
    """读侧取学习者 overlay 补丁（E1 合并用）。

    无记录 → `None`（→ `merge_review_points` 返回纯 AI 结果 + `review=null` 摘要）。
    AI 集合（`english_extension_point`）不参与、不回写（§4.26 红线②）。
    """
    res = await db.query(
        config.EXTENSION_REVIEW_COLLECTION,
        where={"scholar_id": scholar_id, "sentence_id": sentence_id},
        limit=1,
    )
    records = res.get("records", [])
    return records[0] if records else None


# ===========================================================================
# 统一响应 / 失败 helper（信封风格 A）
# ===========================================================================


def _fail(code: str, message: str) -> dict:
    """业务失败响应：HTTP 200 + success=false + code + message。"""
    return {"success": False, "code": code, "message": message}


def _disabled() -> dict:
    return _fail(
        "EXTENSION_DISABLED",
        "语点扩展功能未开启（请配置 EXTENSION_ENABLED=1）",
    )


def _round_disabled() -> dict:
    """E7 / E8 / E9 多轮开关关闭（红线 R13：禁静默降级，**不回退单轮 L2**）。"""
    return _fail(
        "EXTENSION_ROUND_DISABLED",
        "造句多轮未开启（请配置 EXTENSION_ROUND_ENABLED=1）",
    )


# ===========================================================================
# 请求体
# ===========================================================================


class ExtractPointsRequest(BaseModel):
    """E1 抽取请求体。"""

    sentence_id: str = Field(..., min_length=1)
    textbook_id: str = Field("", description="教材 ID")
    chapter_id: str = Field("", description="章节 ID")
    lesson_id: str = Field("", description="课时 ID")
    original: str = Field(..., min_length=1, description="原句")
    translation: str = Field("", description="译文")
    force_refresh: bool = Field(False, description="跳过缓存强制重抽")
    enable_fallback: bool = Field(
        True,
        description="LLM 失败时是否启用规则兜底（与服务端全局开关为「与」关系；"
        "置 false 用于实验页「关兜底」对照实验）",
    )
    scholar_id: str = Field(
        "",
        description="学习者 ID；传则读侧合并其 overlay 补丁（points 带 origin、出参加 review）",
    )


class EvaluateRequest(BaseModel):
    """E2 评测请求体。"""

    sentence_id: str = Field(..., min_length=1)
    selected_ids: list[str] = Field(default_factory=list)
    task_type: str = Field("l1_fill")  # l1_fill | l2_sentence
    input_mode: str = Field("text")  # text | audio
    user_input: str = Field("")
    audio_base64: str | None = Field(None)


class SaveReviewRequest(BaseModel):
    """E4' 提交「我的补丁」请求体（全量替换该 (scholar_id, sentence_id) 的 overlay）。"""

    scholar_id: str = Field("", description="学习者 ID（校对的作用域维度）")
    content_hash: str = Field("", description="与 E1 同源；与当前原句不一致 → 标 stale")
    removed_texts: list[str] = Field(
        default_factory=list, description="归一化后的 AI 点文本（提交全量：想恢复就去掉某条）"
    )
    added: list[dict] = Field(default_factory=list, description="自建点全量")
    note: str = Field("", description="备注")


class StartRoundRequest(BaseModel):
    """E7 开启多轮会话请求体。"""

    sentence_id: str = Field(..., min_length=1)
    textbook_id: str = Field("", description="教材 ID")
    lesson_id: str = Field("", description="课时 ID")
    scholar_id: str = Field(..., min_length=1, description="学习者 ID（会话作用域维度）")
    selected_ids: list[str] = Field(
        default_factory=list, description="勾选的语言点，长度 1~3"
    )
    points_snapshot: list[dict] = Field(
        default_factory=list, description="落快照，判分不回查 point 集合（红线 R11）"
    )
    original: str = Field(..., min_length=1, description="原句")
    translation: str = Field("", description="译文")
    max_turns: int | None = Field(None, description="轮次上限 1~6，缺省取服务端默认")
    # 字段名 `register` 与 BaseModel 的父类属性冲突（pydantic 告警）→ 用 alias 承接入参
    register_pref: str = Field(
        "auto", alias="register", description="语域偏好（F9：落会话供后续轮出题）"
    )
    difficulty: str = Field("same", description="后续轮难度策略（F9：落会话）")


class SubmitRoundTurnRequest(BaseModel):
    """E8 提交第 k 轮作答请求体。"""

    round_id: str = Field(..., min_length=1)
    scholar_id: str = Field(..., min_length=1, description="校验归属")
    # 空串 / 超长由路由显式判 INVALID_INPUT（契约 §3.18），不交给 pydantic 422
    user_input: str = Field("", description="本轮作答")
    client_turn_index: int | None = Field(
        None, description="乐观并发校验；与服务端不一致 → ROUND_TURN_MISMATCH"
    )


# ===========================================================================
# E1 POST /english/extension/points — 抽取（含缓存快路径）
# ===========================================================================


@router.post("/points")
async def extract_points(body: ExtractPointsRequest, db=Depends(get_db)):
    """抽取语言点（E1）。

    - 缓存命中（非 force_refresh）→ 直接返回 points，task_id=null；
    - 未命中 → 建 extract 任务，后台执行，返回 pending + task_id。
    """
    if not config.EXTENSION_ENABLED:
        return _disabled()

    # 句子过短 → 不抽取
    if len(body.original.split()) < 4:
        return _fail("SENTENCE_TOO_SHORT", "句子过短，无可用语言点（需 ≥4 词）")

    # 缓存快路径
    if not body.force_refresh:
        cached = await get_cached_point(
            db, sentence_id=body.sentence_id, original=body.original
        )
        if cached is not None:
            points = cached["points"]
            review = None
            # 读侧合并（契约 §3.18）：仅当传 scholar_id 且校对开关开时才合并，
            # 不传 → 纯 AI 结果（教研观察视角），review=null；AI 集合零写入。
            if body.scholar_id and config.EXTENSION_REVIEW_ENABLED:
                overlay = await _load_overlay(db, body.scholar_id, body.sentence_id)
                points, review = merge_review_points(
                    points, overlay, cached["meta"].get("content_hash", "")
                )
            return {
                "success": True,
                "data": {
                    "task_id": None,
                    "status": "success",
                    "cached": True,
                    "points": points,
                    "review": review,
                    "meta": cached["meta"],
                },
            }

    # 未命中 → 入队
    task = await create_extension_task(
        db,
        kind="extract",
        sentence_id=body.sentence_id,
        textbook_id=body.textbook_id or None,
        lesson_id=body.lesson_id or None,
        enable_fallback=body.enable_fallback,
    )
    bg = asyncio.create_task(
        run_extension_task(
            task["task_id"],
            kind="extract",
            sentence_id=body.sentence_id,
            textbook_id=body.textbook_id or None,
            lesson_id=body.lesson_id or None,
            original=body.original,
            translation=body.translation,
            enable_fallback=body.enable_fallback,
        )
    )
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return {
        "success": True,
        "data": {
            "task_id": task["task_id"],
            "status": "pending",
            "cached": False,
            "points": [],
            "review": None,
            "meta": None,
        },
    }


# ===========================================================================
# E2 POST /english/extension/evaluate — 提交评测
# ===========================================================================


@router.post("/evaluate")
async def evaluate(body: EvaluateRequest, db=Depends(get_db)):
    """提交评测（E2）。

    - l1_fill：后端确定性判分，多数情况同请求即 success；
    - l2_sentence：LLM rubric，异步 pending。
    """
    if not config.EXTENSION_ENABLED:
        return _disabled()

    # selected_ids 不能为空
    if not body.selected_ids:
        return _fail("INVALID_INPUT", "selected_ids 不能为空")

    # D2（2026-10-03）：学习者自建点无后端 l1 题面与答案键，**命中即报错**不静默过滤
    if any(str(pid).startswith(_MINE_ID_PREFIX) for pid in body.selected_ids):
        return _fail("INVALID_INPUT", "自建语言点暂不支持判分（origin=mine）")

    # 取该句「生效」记录（updated_at 最新）的 points —— 同时作为评测任务的
    # points_snapshot（契约 §3.18：提交时落快照，评测不回查 point 集合）
    rec = await resolve_effective_point(db, body.sentence_id)
    if not rec:
        return _fail("INVALID_INPUT", "该句子尚未抽取语言点，请先调用 E1")
    points = rec.get("points", [])
    selected = [p for p in points if p.get("id") in body.selected_ids]
    if not selected:
        return _fail("INVALID_INPUT", "selected_ids 不在语言点列表中")

    if body.task_type == "l1_fill":
        # L1 同步判分：确定性比对，零 LLM
        results = [grade_l1_fill(p, body.user_input) for p in selected]
        all_passed = all(r["passed"] for r in results)
        return {
            "success": True,
            "data": {
                "task_id": None,
                "status": "success",
                "result": {
                    "score": sum(r["score"] for r in results),
                    "passed": all_passed,
                    "correct_ids": [cid for r in results for cid in r["correct_ids"]],
                    "wrong_ids": [wid for r in results for wid in r["wrong_ids"]],
                    "answer_key": selected[0]["text"],
                },
            },
        }

    # l2_sentence：异步
    task = await create_extension_task(
        db,
        kind="evaluate",
        sentence_id=body.sentence_id,
        task_type="l2_sentence",
        selected_ids=body.selected_ids,
        input_mode=body.input_mode,
        user_input=body.user_input,
    )
    bg = asyncio.create_task(
        run_extension_task(
            task["task_id"],
            kind="evaluate",
            task_type="l2_sentence",
            selected_ids=body.selected_ids,
            points_snapshot=points,
            input_mode=body.input_mode,
            user_input=body.user_input,
            audio_base64=body.audio_base64,
        )
    )
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return {
        "success": True,
        "data": {"task_id": task["task_id"], "status": "pending", "result": None},
    }


# ===========================================================================
# E3 GET /english/extension/task/{task_id} — 统一轮询
# ===========================================================================


@router.get("/task/{task_id}")
async def get_task_status(task_id: str, db=Depends(get_db)):
    """查询任务状态（E3）。定点自愈 + 过期 404。"""
    if not config.EXTENSION_ENABLED:
        return _disabled()

    task = await get_task(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    # 过期检查
    now_ms = int(time.time() * 1000)
    if task.get("expires_at", 0) < now_ms:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")

    # 定点自愈卡死任务
    await recover_task_if_stale(db, task)

    # 前端只回 error_detail 字符串（契约红线）
    error = task.get("error")
    error_detail = error.get("error_detail") if error else None

    return {
        "success": True,
        "data": {
            "task_id": task["task_id"],
            "kind": task.get("kind"),
            "status": task["status"],
            "result": task.get("result"),
            "error": error_detail,
            "updated_at": task.get("updated_at"),
        },
    }


# ===========================================================================
# E4' PUT /english/extension/review/{sentence_id} — 提交「我的补丁」
# ===========================================================================


@router.put("/review/{sentence_id}")
async def put_review(sentence_id: str, body: SaveReviewRequest, db=Depends(get_db)):
    """提交学习者补丁（E4'）。

    全量替换 overlay → diff 出 add / remove / restore → 逐条 append 进学习历史。
    只写 `extension_review` / `extension_review_log`，**零 mastery 写入**（R9）。
    """
    try:
        data = await save_review(
            db,
            scholar_id=body.scholar_id,
            sentence_id=sentence_id,
            content_hash=body.content_hash,
            removed_texts=body.removed_texts,
            added=body.added,
            note=body.note,
        )
    except ExtensionError as e:
        return _fail(e.error_code, e.detail)
    return {"success": True, "data": data}


# ===========================================================================
# E6 GET /english/extension/review/{sentence_id} — 我的校对与学习历史
# ===========================================================================


@router.get("/review/{sentence_id}")
async def fetch_review(sentence_id: str, scholar_id: str = "", db=Depends(get_db)):
    """我的校对 + 学习历史时间线（E6）。**只读**，history 按 `at` 倒序最近 50 条。"""
    try:
        data = await get_review(db, scholar_id=scholar_id, sentence_id=sentence_id)
    except ExtensionError as e:
        return _fail(e.error_code, e.detail)
    return {"success": True, "data": data}


# ===========================================================================
# E7 POST /english/extension/round — 开启多轮会话 + 第 1 轮情景句
# ===========================================================================


@router.post("/round")
async def start_round(body: StartRoundRequest, db=Depends(get_db)):
    """开启多轮会话（E7）。**恒异步**（出题需 LLM），经 **E3** 轮询取结果。

    同步校验（不进任务即返回）：多轮开关 → `selected_ids`（空 / 超上限 / 含 `mine_`）
    → `max_turns`（1~6）→ `register` / `difficulty` 枚举 → `points_snapshot` 与
    `selected_ids` 一致性；越界一律 `INVALID_INPUT`。
    """
    if not config.EXTENSION_ROUND_ENABLED:
        return _round_disabled()

    if not body.selected_ids:
        return _fail("INVALID_INPUT", "selected_ids 不能为空")
    if len(body.selected_ids) > config.EXTENSION_ROUND_MAX_SELECTED:
        return _fail(
            "INVALID_INPUT",
            f"单次会话最多勾选 {config.EXTENSION_ROUND_MAX_SELECTED} 个语言点，"
            f"当前 {len(body.selected_ids)} 个",
        )
    if any(str(pid).startswith(_MINE_ID_PREFIX) for pid in body.selected_ids):
        return _fail("INVALID_INPUT", "自建语言点暂不支持造句多轮（origin=mine）")

    max_turns = (
        body.max_turns if body.max_turns is not None else config.EXTENSION_ROUND_MAX_TURNS
    )
    try:
        max_turns = int(max_turns)
    except (TypeError, ValueError):
        return _fail("INVALID_INPUT", "max_turns 必须为整数")
    if max_turns < 1 or max_turns > config.EXTENSION_ROUND_MAX_TURNS_HARD:
        return _fail(
            "INVALID_INPUT",
            f"max_turns 必须在 1~{config.EXTENSION_ROUND_MAX_TURNS_HARD} 之间，"
            f"当前 {max_turns}",
        )

    if body.register_pref not in REGISTERS:
        return _fail("INVALID_INPUT", f"register 取值须为 {' / '.join(REGISTERS)}")
    if body.difficulty not in (DIFFICULTY_SAME, DIFFICULTY_HARDER):
        return _fail("INVALID_INPUT", "difficulty 取值须为 same / harder")

    if not body.points_snapshot:
        return _fail("INVALID_INPUT", "points_snapshot 不能为空")
    if not any(
        p.get("id") in body.selected_ids for p in body.points_snapshot
    ):
        return _fail("INVALID_INPUT", "selected_ids 不在语言点列表中")

    task = await create_extension_task(
        db,
        kind="round",
        round_action="start",
        scholar_id=body.scholar_id,
        sentence_id=body.sentence_id,
        textbook_id=body.textbook_id or None,
        lesson_id=body.lesson_id or None,
        selected_ids=body.selected_ids,
        points_snapshot=body.points_snapshot,
    )
    bg = asyncio.create_task(
        run_extension_task(
            task["task_id"],
            kind="round",
            round_action="start",
            scholar_id=body.scholar_id,
            sentence_id=body.sentence_id,
            textbook_id=body.textbook_id or None,
            lesson_id=body.lesson_id or None,
            selected_ids=body.selected_ids,
            points_snapshot=body.points_snapshot,
            original=body.original,
            translation=body.translation,
            max_turns=max_turns,
            register=body.register_pref,
            difficulty=body.difficulty,
        )
    )
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return {"success": True, "data": {"task_id": task["task_id"], "status": "pending"}}


# ===========================================================================
# E8 POST /english/extension/round/turn — 提交第 k 轮作答（判分 + 推进）
# ===========================================================================


@router.post("/round/turn")
async def submit_round_turn(body: SubmitRoundTurnRequest, db=Depends(get_db)):
    """提交第 k 轮作答（E8）。**恒异步**，任务内**串行两次 LLM**（判分 → 出题）。

    同步校验：`user_input`（空 / 超 `EXTENSION_ROUND_MAX_INPUT_LEN`，红线 R14）
    → 归属（`get_round(scholar_id=)` 为 None → **HTTP 404**）→ 终态 → `ROUND_CLOSED`
    → 轮次 → `ROUND_TURN_MISMATCH`。
    """
    if not config.EXTENSION_ROUND_ENABLED:
        return _round_disabled()

    if not str(body.user_input or "").strip():
        return _fail("INVALID_INPUT", "user_input 不能为空")
    if len(body.user_input) > config.EXTENSION_ROUND_MAX_INPUT_LEN:
        return _fail(
            "INVALID_INPUT",
            f"user_input 超过 {config.EXTENSION_ROUND_MAX_INPUT_LEN} 字符",
        )

    doc = await get_round(db, body.round_id, scholar_id=body.scholar_id)
    if doc is None:
        # R12：归属不符与「不存在 / 已过期」同码，防 round_id 枚举
        raise HTTPException(status_code=404, detail="练习会话不存在或已过期")

    if doc.get("status") != STATUS_ACTIVE:
        return _fail("ROUND_CLOSED", "本组练习已结束，可「再来一组」")

    server_turn_index = int(doc.get("turn_index") or 0)
    if body.client_turn_index is not None and int(body.client_turn_index) != server_turn_index:
        return _fail(
            "ROUND_TURN_MISMATCH",
            f"轮次已变化，请按当前轮次重新提交（服务端当前第 {server_turn_index} 轮）",
        )

    task = await create_extension_task(
        db,
        kind="round",
        round_action="turn",
        scholar_id=body.scholar_id,
        sentence_id=doc.get("sentence_id"),
        user_input=body.user_input,
    )
    bg = asyncio.create_task(
        run_extension_task(
            task["task_id"],
            kind="round",
            round_action="turn",
            scholar_id=body.scholar_id,
            round_id=body.round_id,
            user_input=body.user_input,
            client_turn_index=body.client_turn_index,
        )
    )
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    return {"success": True, "data": {"task_id": task["task_id"], "status": "pending"}}


# ===========================================================================
# E9 GET /english/extension/round/{round_id} — 会话详情（同步只读）
# ===========================================================================


@router.get("/round/{round_id}")
async def fetch_round(round_id: str, scholar_id: str = "", db=Depends(get_db)):
    """会话详情（E9）。**同步只读，不推进状态机**（复盘 / 导出 / 断线恢复）。

    `turns[]` **含 `reference_en`**（复盘用；E7 / E8 下发的 `turn` 不含，防泄题 D5）。
    """
    if not config.EXTENSION_ROUND_ENABLED:
        return _round_disabled()

    if not scholar_id or not str(scholar_id).strip():
        return _fail("INVALID_INPUT", "scholar_id 不能为空")

    doc = await get_round(db, round_id, scholar_id=scholar_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="练习会话不存在或已过期")

    return {
        "success": True,
        "data": {
            "round_id": doc.get("round_id"),
            "status": doc.get("status"),
            "max_turns": doc.get("max_turns"),
            "turn_index": doc.get("turn_index"),
            "selected_ids": doc.get("selected_ids") or [],
            "points_snapshot": doc.get("points_snapshot") or [],
            "turns": doc.get("turns") or [],
            "summary": doc.get("summary"),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "expires_at": doc.get("expires_at"),
        },
    }
