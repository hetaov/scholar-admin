"""AI 对话生成域接口（批量面，设计稿 §5 / §7 / v2 调整稿）

POST /ai/dialogue/v1/generate — 提交批量对话生成任务。
- 毫秒级返回 `{ task_id: "dg_xxx", status: "pending", ... }`，不做任何 LLM 调用
  （生成由后台 `run_dialogue_gen_task` 执行）；
- 入参核心字段可选化 + 手动校验 → 业务失败 HTTP 200 + success=false + code：
  INVALID_INPUT（缺参/空参/超长/枚举非法）/ TASK_GROUP_REQUIRED（任务组缺失或空）/
  TYPE_NOT_SUPPORTED（preferred_type 非 auto/dialogue/non_dialogue）/
  DIALOGUE_GEN_DISABLED（总开关 DIALOGUE_GEN_ENABLED=0）；
- v2 入参：仅 `scholar_id + task_group`（必填），其它（`scenario/roles/recall/metrics/
  prompt_lang/preferred_type`）全部可选——`scenario/roles` 仍兼容透传进 context（旧
  路径在 `build_prompt` 阶段被使用，可继续生效），v2 新流程默认忽略；
- v2 删除：`generation` / `test_scenario` 入参及对应校验（场景候选机制取消）；
- 技术异常 → HTTP 500。

GET /ai/dialogue/v1/task/{task_id} — 查询生成结果。
- 状态枚举 pending/processing/success/failed；TTL 24h 过期 → 404；
- 卡死自愈：processing 超时 → 查询定点置 failed（全集合巡检由后台定时任务执行）。

GET /ai/dialogue/v1/task/{task_id}/checkpoints — checkpoint 轨迹（T4 / §5 ②'）。
- 返回 `{checkpoints: [{checkpoint_id, stage, retry_count, ts}]}`（旧→新）；
- 任务不存在/过期 → 404；开关关闭或无断点 → 空列表。

POST /ai/dialogue/v1/task/{task_id}/resume — 断点续写（T4 / §5 ③）。
- 仅 failed（含卡死自愈）可续跑，续跑**同一** task_id（不新建任务）；
- success/进行中 → TASK_NOT_RESUMABLE；无断点 → CHECKPOINT_NOT_FOUND；
- 校验通过 → 异步续跑，立即返回 status=processing。

POST /ai/dialogue/v1/task/{task_id}/user-input — 用户作答评测（T5 / §7.2 Tab3）。
- 入参 `{ text, target_sentence_id? }`；参考句按 `resolve_target_sentence` 解析；
- 调 `evaluate_text`（L1 规则 + L2 Judge）→ 回显 score/meaningful/faithfulness/anomaly/confidence；
- 评测指标**回写到任务文档** `user_inputs`（新集合、零侵入；不写 `skill_state`/`evaluation` 证据）；
- 空 text → INVALID_INPUT；任务不存在/过期 → 404；技术异常 → 500。

GET /ai/dialogue/v1/corpus — 本地 NC2 语料读取（T2 / §3.1、§3.2）。
- 返回任务组下拉数据（本地 NC2 全课）+ 本地模拟学者指标摘要；免 db、不触网；
- 语料缺失 → corpus_available=false + 空任务组（前端回落 mock，不阻断页面）。

零侵入（§9-7）：本路由为**新增**，不读写任何既有集合；开关默认关。
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from config import DIALOGUE_GEN_ENABLED, DIALOGUE_LOCAL_SCHOLAR_ID
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.learning import dialogue_gen_graph
from services.learning.dialogue_gen_task import (
    COLLECTION as DIALOGUE_TASK_COLLECTION,
    STATUS_FAILED,
    STATUS_PROCESSING,
    build_task_id,
    create_task,
    get_task,
    recover_task_if_stale,
    run_dialogue_gen_task,
)
from services.learning.local_corpus import (
    corpus_path,
    iter_lesson_groups,
    learner_summary,
    list_lessons,
    load_learner,
    try_load_corpus,
)
from services.evaluation_engine import LOW_CONFIDENCE_THRESHOLD, evaluate_text
from services.providers.dialogue_gen import (
    SUPPORTED_PREFERRED_TYPES,
    clamp_top_k,
    resolve_target_sentence,
)

logger = logging.getLogger("scholar-admin.dialogue_gen")

router = APIRouter(tags=["ai-dialogue"])

# 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消（同 ai.py）
_background_tasks: set[asyncio.Task] = set()

# 长度口径（对齐既有 §3.12 惯例）
_MAX_SCHOLAR_ID = 64
_MAX_SENTENCE_ID = 64
_MAX_SENTENCE_CONTENT = 500
_MAX_BACKGROUND = 500
_MAX_GROUP_LABEL = 100
# T5 用户作答：单条长度上限（对齐 ai.py turn user_input ≤2000）+ 任务内保留条数
_MAX_USER_INPUT = 2000
_MAX_USER_INPUTS_STORED = 20

# 形态白名单：A/B/C 对话与非对话回落；retell/fill/task 属「非对话子形态」，
# 只能作为生成结果出现，不作为请求 preferred_type。
_UNSUPPORTED_PREFERRED_TYPES = ("retell", "fill", "task")


class DialogueGenRequest(BaseModel):
    """批量对话生成提交请求（核心字段全部可选 + 手动校验）。"""

    # v2：extra="allow" 捕获已废弃 key（generation / test_scenario），
    # 否则 Pydantic 默认 ignore 会静默丢弃、废弃检测失效。
    model_config = ConfigDict(extra="allow")

    scholar_id: Optional[str] = None
    task_group: Optional[dict] = None  # {lesson_id, group_id, group_label, sentences[]}
    scenario: Optional[dict] = None  # {background, goal?}；v2 兼容透传（默认由 summarize 产出）
    roles: Optional[list] = None  # [{code, name, identity}]；v2 兼容透传（默认由 summarize 产出）
    recall: Optional[dict] = None  # {enabled, top_k}
    metrics: Optional[dict] = None  # {enabled, weak_skills[]}
    prompt_lang: Optional[str] = None  # zh / en，缺省 zh
    preferred_type: Optional[str] = None  # auto / dialogue / non_dialogue，缺省 auto
    # v2：删除 `generation` / `test_scenario`（场景候选机制取消；旧请求归一为 `INVALID_INPUT`）


class DialogueResumeRequest(BaseModel):
    """断点续写请求（T4；`from_checkpoint_id` 可选，缺省取最新 checkpoint）。"""

    from_checkpoint_id: Optional[str] = None


class DialogueUserInputRequest(BaseModel):
    """用户作答评测请求（T5；`target_sentence_id` 可选，缺省按结果/任务组推导）。"""

    text: Optional[str] = None
    target_sentence_id: Optional[str] = None


class DialogueGenResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


def _fail(code: str, message: str) -> DialogueGenResponse:
    return DialogueGenResponse(success=False, code=code, message=message)


# ---------------------------------------------------------------------------
# 手动校验（业务失败 200 + success=false）
# ---------------------------------------------------------------------------


def _validate_task_group(body: DialogueGenRequest) -> DialogueGenResponse | None:
    """任务组必填结构校验（缺失/空 → TASK_GROUP_REQUIRED；叶子非法 → INVALID_INPUT）。"""
    task_group = body.task_group
    if not isinstance(task_group, dict):
        return _fail("TASK_GROUP_REQUIRED", "task_group（object）必填")
    sentences = task_group.get("sentences")
    if not isinstance(sentences, list) or not sentences:
        return _fail("TASK_GROUP_REQUIRED", "task_group.sentences 不能为空")
    for si, s in enumerate(sentences):
        if not isinstance(s, dict):
            return _fail("INVALID_INPUT", f"task_group.sentences[{si}] 必须是 object")
        sid = str(s.get("sentence_id") or "").strip()
        content = str(s.get("content") or "").strip()
        if not sid:
            return _fail(
                "INVALID_INPUT", f"task_group.sentences[{si}].sentence_id 不能为空"
            )
        if len(sid) > _MAX_SENTENCE_ID:
            return _fail(
                "INVALID_INPUT",
                f"task_group.sentences[{si}].sentence_id 超长（≤{_MAX_SENTENCE_ID}）",
            )
        if not content:
            return _fail("INVALID_INPUT", f"task_group.sentences[{si}].content 不能为空")
        if len(content) > _MAX_SENTENCE_CONTENT:
            return _fail(
                "INVALID_INPUT",
                f"task_group.sentences[{si}].content 超长（≤{_MAX_SENTENCE_CONTENT}）",
            )
    return None


def _validate(body: DialogueGenRequest) -> DialogueGenResponse | None:
    """公共校验：scholar_id / task_group / roles / scenario / 枚举。

    v2：`generation` / `test_scenario` 已从 `DialogueGenRequest` 删除，模型开启
    `extra="allow"` 后旧请求的这两个 key 落在 `model_extra` — 这里检测命中即返回
    `INVALID_INPUT`（行为可观察，不再静默丢弃）。
    """
    raw = getattr(body, "model_extra", None) or getattr(body, "__dict__", {}) or {}
    for removed_key in ("generation", "test_scenario"):
        if raw.get(removed_key) is not None:
            return _fail(
                "INVALID_INPUT",
                f"{removed_key} 已废弃（v2 调整移除场景候选机制）；详见设计稿 v2",
            )

    if not body.scholar_id or not str(body.scholar_id).strip():
        return _fail("INVALID_INPUT", "scholar_id 不能为空")
    if len(str(body.scholar_id)) > _MAX_SCHOLAR_ID:
        return _fail("INVALID_INPUT", f"scholar_id 超长（≤{_MAX_SCHOLAR_ID}）")

    bad = _validate_task_group(body)
    if bad:
        return bad

    if body.roles is not None:
        if not isinstance(body.roles, list):
            return _fail("INVALID_INPUT", "roles 必须是 array")
        for ri, role in enumerate(body.roles):
            if not isinstance(role, dict) or not str(role.get("code") or "").strip():
                return _fail("INVALID_INPUT", f"roles[{ri}].code 不能为空")

    if body.scenario is not None:
        if not isinstance(body.scenario, dict):
            return _fail("INVALID_INPUT", "scenario 必须是 object")
        background = str(body.scenario.get("background") or "")
        if len(background) > _MAX_BACKGROUND:
            return _fail("INVALID_INPUT", f"scenario.background 超长（≤{_MAX_BACKGROUND}）")

    preferred_type = (body.preferred_type or "auto").strip().lower()
    if preferred_type in _UNSUPPORTED_PREFERRED_TYPES:
        return _fail(
            "TYPE_NOT_SUPPORTED",
            f"preferred_type={preferred_type} 为生成结果子形态，"
            f"请求仅支持 auto/dialogue/non_dialogue",
        )
    if preferred_type not in SUPPORTED_PREFERRED_TYPES:
        return _fail(
            "INVALID_INPUT",
            f"preferred_type 非法：{preferred_type or '(空)'}（auto/dialogue/non_dialogue）",
        )

    prompt_lang = (body.prompt_lang or "zh").strip().lower()
    if prompt_lang not in ("zh", "en"):
        return _fail("INVALID_INPUT", f"prompt_lang 非法：{prompt_lang or '(空)'}（zh/en）")
    return None


def _build_context(body: DialogueGenRequest) -> dict:
    """生成自包含快照（§6.1）：任务即执行唯一依据，可独立重跑。

    v2：删除 `generation` / `test_scenario` 字段写入；`scenario` / `roles` 仍兼容透传
    （旧请求仍能进 context，旧 build_prompt 行为可用；v2 新流程默认忽略）。
    """
    recall = body.recall or {}
    metrics = body.metrics or {}
    return {
        "task_group": body.task_group,
        "scenario": body.scenario or {},
        "roles": body.roles or [],
        "recall": {
            "enabled": bool(recall.get("enabled")),
            "top_k": clamp_top_k(recall.get("top_k")),
        },
        "metrics": {
            "enabled": bool(metrics.get("enabled")),
            "weak_skills": list(metrics.get("weak_skills") or []),
        },
        "prompt_lang": (body.prompt_lang or "zh").strip().lower(),
        "preferred_type": (body.preferred_type or "auto").strip().lower(),
    }


def _error_display(raw_error) -> str | None:
    """error 展示为可读字符串（完整 error 对象保留在任务文档中）。"""
    if isinstance(raw_error, dict):
        return raw_error.get("error_detail")
    return raw_error


def _task_payload(task: dict) -> dict:
    """任务文档 → 接口响应 data（结构与前端 `DialogueGenTask` 对齐）。"""
    status = task.get("status")
    checkpoint_id = task.get("checkpoint_id")
    # T4：有断点游标且任务失败 → 可续写；success 不可续写（卡死 processing 经查询
    # 定点自愈转 failed 后同样命中此判定）
    resumable = bool(checkpoint_id) and status == STATUS_FAILED
    return {
        "task_id": task["task_id"],
        "status": status,
        "checkpoint_id": checkpoint_id,
        "resumable": resumable,
        "result": task.get("result"),
        "error": _error_display(task.get("error")),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }


# ---------------------------------------------------------------------------
# 提交：POST /ai/dialogue/v1/generate
# ---------------------------------------------------------------------------


@router.post("/ai/dialogue/v1/generate", response_model=DialogueGenResponse)
async def submit_dialogue_gen(
    body: DialogueGenRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> DialogueGenResponse:
    """提交批量对话生成任务 — 毫秒级返回 task_id，生成在后台执行。"""
    if not DIALOGUE_GEN_ENABLED:
        return _fail(
            "DIALOGUE_GEN_DISABLED",
            "批量对话生成未开启（DIALOGUE_GEN_ENABLED=0）",
        )

    bad = _validate(body)
    if bad:
        return bad

    context = _build_context(body)
    try:
        task_doc = await create_task(
            db,
            task_id=build_task_id(),
            scholar_id=str(body.scholar_id).strip(),
            context=context,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[dialogue_gen] 任务创建异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务创建失败: {str(e)}")

    # 后台执行，不阻塞当前请求（必须持引用 + done_callback，防协程被 GC 回收）
    bg = asyncio.create_task(run_dialogue_gen_task(task_doc["task_id"]))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    logger.info(
        f"[dialogue_gen] 任务已提交 → task_id={task_doc['task_id']}, "
        f"scholar={body.scholar_id}"
    )
    return DialogueGenResponse(success=True, data=_task_payload(task_doc))


# ---------------------------------------------------------------------------
# 查询：GET /ai/dialogue/v1/task/{task_id}
# ---------------------------------------------------------------------------


@router.get("/ai/dialogue/v1/task/{task_id}", response_model=DialogueGenResponse)
async def get_dialogue_gen_task(
    task_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> DialogueGenResponse:
    """查询异步对话生成结果 — pending/processing/success/failed；不存在/过期 → 404。"""
    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 定点自愈：processing 且超时 → 置 failed
        if await recover_task_if_stale(db, task):
            task = await get_task(db, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="任务不存在")
        # TTL 过滤：expires_at 已过期的按不存在处理（物理清理由后台巡检执行）
        if task.get("expires_at", 0) <= int(time.time() * 1000):
            raise HTTPException(status_code=404, detail="任务已过期")
        return DialogueGenResponse(success=True, data=_task_payload(task))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"[dialogue_gen] 任务查询异常: task_id={task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务查询失败: {str(e)}")


# ---------------------------------------------------------------------------
# 断点轨迹：GET /ai/dialogue/v1/task/{task_id}/checkpoints（T4）
# ---------------------------------------------------------------------------


@router.get(
    "/ai/dialogue/v1/task/{task_id}/checkpoints",
    response_model=DialogueGenResponse,
)
async def list_dialogue_gen_checkpoints(
    task_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> DialogueGenResponse:
    """任务 checkpoint 时间线（旧→新；页面轨迹 Tab 渲染 stage/retry_count/ts）。

    - 复用 T3 查询的 TTL 口径：任务不存在/过期 → 404；
    - 断点开关关闭或无 checkpoint → 返回空列表（页面轨迹为空，不阻断演示）。
    """
    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.get("expires_at", 0) <= int(time.time() * 1000):
            raise HTTPException(status_code=404, detail="任务已过期")
        checkpoints = await dialogue_gen_graph.list_dialogue_checkpoints(db, task_id)
        return DialogueGenResponse(success=True, data={"checkpoints": checkpoints})
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"[dialogue_gen] checkpoint 查询异常: task_id={task_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"checkpoint 查询失败: {str(e)}")


# ---------------------------------------------------------------------------
# 断点续写：POST /ai/dialogue/v1/task/{task_id}/resume（T4）
# ---------------------------------------------------------------------------


@router.post(
    "/ai/dialogue/v1/task/{task_id}/resume",
    response_model=DialogueGenResponse,
)
async def resume_dialogue_gen(
    task_id: str,
    body: Optional[DialogueResumeRequest] = None,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> DialogueGenResponse:
    """从最近（或指定）checkpoint 续跑**同一**任务（不新建 task_id，§5 ③）。

    - 仅 `failed`（含卡死自愈转 failed）可续写；`success`/进行中 → `TASK_NOT_RESUMABLE`；
    - 无可用 checkpoint（缺失或开关关闭）→ `CHECKPOINT_NOT_FOUND`；
    - 校验通过 → `asyncio` 异步续跑，立即返回 `status=processing` 的任务快照
      （claim 在后台原子抢占 `failed → processing`，避免并发重复续写）。
    """
    if not DIALOGUE_GEN_ENABLED:
        return _fail(
            "DIALOGUE_GEN_DISABLED",
            "批量对话生成未开启（DIALOGUE_GEN_ENABLED=0）",
        )

    from_checkpoint_id = (body.from_checkpoint_id if body else None) or None
    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.get("expires_at", 0) <= int(time.time() * 1000):
            raise HTTPException(status_code=404, detail="任务已过期")

        # 卡死 processing 先定点自愈为 failed（与查询接口同口径），再判定可续写性
        if await recover_task_if_stale(db, task):
            task = await get_task(db, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="任务不存在")

        if task.get("status") != STATUS_FAILED:
            return _fail(
                "TASK_NOT_RESUMABLE",
                f"任务状态 {task.get('status')} 不可续写（仅 failed 可续写）",
            )

        if not await dialogue_gen_graph.dialogue_checkpoint_exists(
            db, task_id, from_checkpoint_id
        ):
            return _fail(
                "CHECKPOINT_NOT_FOUND",
                "无可用断点（checkpoint 缺失或 DIALOGUE_GEN_CHECKPOINT_ENABLED=0）",
            )

        bg = asyncio.create_task(
            run_dialogue_gen_task(
                task_id,
                resume=True,
                from_checkpoint_id=from_checkpoint_id,
            )
        )
        _background_tasks.add(bg)
        bg.add_done_callback(_background_tasks.discard)

        logger.info(
            f"[dialogue_gen] 任务续写已调度 → task_id={task_id}, "
            f"from_checkpoint_id={from_checkpoint_id or '(latest)'}"
        )
        # 乐观返回 processing（task 文档由后台 claim 真实改写为 processing）
        task["status"] = STATUS_PROCESSING
        task["error"] = None
        return DialogueGenResponse(success=True, data=_task_payload(task))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"[dialogue_gen] 任务续写异常: task_id={task_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"任务续写失败: {str(e)}")


# ---------------------------------------------------------------------------
# 用户作答评测：POST /ai/dialogue/v1/task/{task_id}/user-input（T5）
# ---------------------------------------------------------------------------


async def _write_back_user_input(db, task: dict, record: dict) -> bool:
    """把评测指标回写到任务文档 `user_inputs`（新集合字段，零侵入 §9-7）。

    采用「读-改-写」（FakeDB/真实客户端均支持 `$set`）；不触碰 `updated_at`，
    避免影响 processing 任务的卡死自愈判定。保留最近 `_MAX_USER_INPUTS_STORED` 条。
    """
    history = [r for r in (task.get("user_inputs") or []) if isinstance(r, dict)]
    history.append(record)
    history = history[-_MAX_USER_INPUTS_STORED:]
    res = await db.update(
        DIALOGUE_TASK_COLLECTION,
        where={"task_id": task["task_id"]},
        data={"$set": {"user_inputs": history}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


@router.post(
    "/ai/dialogue/v1/task/{task_id}/user-input",
    response_model=DialogueGenResponse,
)
async def submit_dialogue_user_input(
    task_id: str,
    body: DialogueUserInputRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> DialogueGenResponse:
    """用户作答 → `evaluate_text` 评测并回显（T5；指标回写任务 `user_inputs`）。

    - 参考句解析：显式 `target_sentence_id`（须在任务组内）→ 结果 turns 最后一个目标句
      → 任务组首句（`resolve_target_sentence`）；
    - `evaluate_text`（L1 规则 + L2 Judge）失败回落 L1，不静默；低置信（<0.6）标记
      `low_confidence`（不回写既有 `skill_state`，零侵入 §9-7）；
    - 空 `text` → `INVALID_INPUT`；任务不存在/过期 → 404；技术异常 → 500。
    """
    if not DIALOGUE_GEN_ENABLED:
        return _fail(
            "DIALOGUE_GEN_DISABLED",
            "批量对话生成未开启（DIALOGUE_GEN_ENABLED=0）",
        )

    text = str(body.text or "").strip()
    if not text:
        return _fail("INVALID_INPUT", "text 不能为空")
    if len(text) > _MAX_USER_INPUT:
        return _fail("INVALID_INPUT", f"text 超长（≤{_MAX_USER_INPUT}）")

    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        if task.get("expires_at", 0) <= int(time.time() * 1000):
            raise HTTPException(status_code=404, detail="任务已过期")

        target = resolve_target_sentence(
            task.get("context") or {}, body.target_sentence_id, task.get("result")
        )
        if target is None:
            if body.target_sentence_id:
                return _fail(
                    "INVALID_INPUT",
                    f"target_sentence_id={body.target_sentence_id} 不在任务组必用句中",
                )
            return _fail("TASK_GROUP_REQUIRED", "任务组无可评测句子（sentences 为空）")

        verdict = evaluate_text(target["content"], text)
        confidence = float(verdict.get("confidence") or 0.0)
        record = {
            "text": text,
            "target_sentence_id": target["sentence_id"],
            "reference": target["content"],
            "score": verdict.get("score"),
            "meaningful": bool(verdict.get("meaningful")),
            "faithfulness": bool(verdict.get("faithfulness")),
            "anomaly": bool(verdict.get("anomaly")),
            "confidence": confidence,
            "level": verdict.get("level"),
            "judge_model": verdict.get("judge_model"),
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "created_at": int(time.time() * 1000),
        }

        written_back = True
        try:
            written_back = await _write_back_user_input(db, task, record)
        except Exception as exc:  # noqa: BLE001 — 回写失败不影响评测回显
            written_back = False
            logger.warning(
                f"[dialogue_gen] 用户作答指标回写失败: task_id={task_id}: {exc}"
            )

        return DialogueGenResponse(success=True, data={**record, "written_back": written_back})
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"[dialogue_gen] 用户作答评测异常: task_id={task_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"用户作答评测失败: {str(e)}")


# ---------------------------------------------------------------------------
# 本地语料：GET /ai/dialogue/v1/corpus（T2）
# ---------------------------------------------------------------------------


@router.get("/ai/dialogue/v1/corpus", response_model=DialogueGenResponse)
async def get_dialogue_corpus(
    scholar_id: Optional[str] = None,
) -> DialogueGenResponse:
    """本地 NC2 语料 → 任务组下拉 + 模拟学者指标（§3.1/§3.2，免 db、不触网）。

    - 语料来源于文件（`DIALOGUE_CORPUS_DIR/corpus.json`），**不读真实库**；
    - 语料缺失/损坏 → `corpus_available=false` + 空 `task_groups`，
      前端据此回落 mock（不阻断页面演示）；
    - `learner` 取自 `learners/<scholar_id>.json` 的模拟指标摘要，供表单预填 weak_skills。
    """
    if not DIALOGUE_GEN_ENABLED:
        return _fail(
            "DIALOGUE_GEN_DISABLED",
            "批量对话生成未开启（DIALOGUE_GEN_ENABLED=0）",
        )

    try:
        corpus_file = corpus_path()
        data_dir = Path(corpus_file).parent
        corpus = try_load_corpus(corpus_file)
        groups = iter_lesson_groups(corpus)
        scholar = str(scholar_id or DIALOGUE_LOCAL_SCHOLAR_ID).strip() or DIALOGUE_LOCAL_SCHOLAR_ID
        learner = learner_summary(load_learner(scholar, data_dir))
        return DialogueGenResponse(
            success=True,
            data={
                "corpus_available": bool(groups),
                "book": corpus.get("book"),
                "lessons": list_lessons(corpus),
                "task_groups": groups,
                "scholar_id": scholar,
                "learner": learner,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[dialogue_gen] 语料读取异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"语料读取失败: {str(e)}")
