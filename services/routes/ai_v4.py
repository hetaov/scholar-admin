"""沉浸式 AI 会话 v4 接口（伪流式实验面 · api-contract §3.17）

新增独立路径（**v2 / v3 零改动**）：
- `POST /ai/session/v4` — 提交会话生成任务（mode=start 开场 / mode=turn 续轮），
  请求字段与业务码**对齐 v3**，并**追加 3 个可选字段**（`stream` / `hint_mode` / `model_tier`）；
  新增总开关 `SESSION_V4_ENABLED`（默认 0），关闭时提交返回 200 + success=false +
  code=SESSION_V4_DISABLED。
- `GET /ai/session/v4/task/{task_id}` — 查询生成结果（状态机/TTL/卡死自愈同 v3），
  `data` **追加** `partial_text` / `partial_seq` / `partial_updated_at` / `hint_status` / `timings`。

契约红线：请求只追加可选字段（缺省即 v3 行为）；响应只在 `data` 上追加字段，
`result` 结构 `{ content_type, ai_text, hint, suggested_targets }` **一字不改**。
集合 `ai_session_v4_task` / `ai_session_v4` / `ai_session_v4_checkpoint`，不写 v2/v3 集合。

业务失败 HTTP 200 + success=false + code：
  INVALID_INPUT（缺参/空参/超长/枚举非法）/ SESSION_NOT_FOUND / TURN_IN_PROGRESS /
  TYPE_NOT_SUPPORTED / SESSION_V4_DISABLED（总开关关闭）；
技术异常 → HTTP 500。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import SESSION_V4_ENABLED
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.learning import session_state_v4 as session_state
from services.learning.session_state_v4 import (
    build_session_id,
    create_session,
    get_session,
    set_pending,
)
from services.learning.session_task_v4 import (
    COLLECTION,
    build_task_id,
    create_session_task,
    get_task,
    recover_task_if_stale,
    run_session_task,
)

logger = logging.getLogger("scholar-admin.ai_v4")

router = APIRouter(tags=["ai"])

# 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消（同 v2/v3）
_background_tasks: set[asyncio.Task] = set()

# MVP 形态白名单（同 v3：auto/dialogue/fill；retell/task → TYPE_NOT_SUPPORTED）
_SUPPORTED_PREFERRED_TYPES = ("auto", "dialogue", "fill")
_UNSUPPORTED_PREFERRED_TYPES = ("retell", "task")
_VALID_MODES = ("start", "turn")
_VALID_KINDS = ("new", "review")

# v4 追加枚举（可选字段，非法 → INVALID_INPUT）
_VALID_HINT_MODES = ("sync", "on_demand")
_VALID_MODEL_TIERS = ("fast", "standard")

# 长度口径（同 v3 手动校验）
_MAX_SCHOLAR_ID = 64
_MAX_SESSION_ID = 64
_MAX_NAME = 100
_MAX_SCENE = 500
_MAX_CONTENT = 500
_MAX_USER_INPUT = 2000
_MAX_GROUPS = 10
_MAX_SENTENCES_PER_GROUP = 50


class AiSessionV4Request(BaseModel):
    """会话 v4 提交请求（字段与口径同 v3，追加 3 个可选字段）。"""

    scholar_id: Optional[str] = None
    mode: Optional[str] = None  # start / turn，缺省 start
    session_id: Optional[str] = None  # turn 必填
    scenario: Optional[dict] = None  # start 必填：{ scene_id?, title?, scene, goal?, constraints? }
    roles: Optional[dict] = None  # start 必填：{ ai_role, learner_role }
    groups: Optional[list] = None  # start 必填：素材组
    user_input: Optional[str] = None  # turn 必填 ≤2000
    preferred_type: Optional[str] = None  # auto / dialogue / fill，缺省 auto
    assisted: Optional[bool] = None  # turn 时前端上报本轮是否借助提示卡
    # ---- v4 追加（全部可选，缺省即 v3 行为）----
    stream: Optional[bool] = None  # true → 生成过程节流写 partial_text
    hint_mode: Optional[str] = None  # sync（默认，同 v3）/ on_demand（按需生成）
    model_tier: Optional[str] = None  # standard（默认）/ fast


class AiSessionResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


# ---------------------------------------------------------------------------
# 手动校验 helpers（业务失败 200 + success=false，同 v3）
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> AiSessionResponse:
    return AiSessionResponse(success=False, code=code, message=message)


def _validate_scholar_id(body: AiSessionV4Request) -> AiSessionResponse | None:
    if not body.scholar_id or not str(body.scholar_id).strip():
        return _fail("INVALID_INPUT", "scholar_id 不能为空")
    if len(str(body.scholar_id)) > _MAX_SCHOLAR_ID:
        return _fail("INVALID_INPUT", f"scholar_id 超长（≤{_MAX_SCHOLAR_ID}）")
    return None


def _validate_preferred_type(body: AiSessionV4Request) -> AiSessionResponse | None:
    preferred_type = (body.preferred_type or "auto").strip().lower()
    if preferred_type in _UNSUPPORTED_PREFERRED_TYPES:
        return _fail(
            "TYPE_NOT_SUPPORTED",
            f"preferred_type={preferred_type} 为后续扩展形态，本期仅支持 "
            f"auto/dialogue/fill",
        )
    if preferred_type not in _SUPPORTED_PREFERRED_TYPES:
        return _fail(
            "INVALID_INPUT",
            f"preferred_type 非法：{preferred_type or '(空)'}（auto/dialogue/fill）",
        )
    return None


def _validate_v4_options(body: AiSessionV4Request) -> AiSessionResponse | None:
    """v4 追加可选字段的枚举校验（缺省不校验）。"""
    if body.hint_mode is not None:
        hint_mode = str(body.hint_mode).strip().lower()
        if hint_mode not in _VALID_HINT_MODES:
            return _fail(
                "INVALID_INPUT",
                f"hint_mode 非法：{body.hint_mode}（{'/'.join(_VALID_HINT_MODES)}）",
            )
    if body.model_tier is not None:
        model_tier = str(body.model_tier).strip().lower()
        if model_tier not in _VALID_MODEL_TIERS:
            return _fail(
                "INVALID_INPUT",
                f"model_tier 非法：{body.model_tier}（{'/'.join(_VALID_MODEL_TIERS)}）",
            )
    return None


def _validate_start_payload(body: AiSessionV4Request) -> AiSessionResponse | None:
    """start 模式必填结构校验：scenario / roles / groups（含叶子长度口径，同 v3）。"""
    scenario = body.scenario
    if not isinstance(scenario, dict):
        return _fail("INVALID_INPUT", "mode=start 时 scenario（object）必填")
    scene = str(scenario.get("scene") or "").strip()
    if not scene:
        return _fail("INVALID_INPUT", "scenario.scene 不能为空")
    if len(scene) > _MAX_SCENE:
        return _fail("INVALID_INPUT", f"scenario.scene 超长（≤{_MAX_SCENE}）")

    roles = body.roles
    if not isinstance(roles, dict):
        return _fail("INVALID_INPUT", "mode=start 时 roles（object）必填")
    ai_role = roles.get("ai_role")
    learner_role = roles.get("learner_role")
    if not isinstance(ai_role, dict) or not str(ai_role.get("name") or "").strip():
        return _fail("INVALID_INPUT", "roles.ai_role.name 不能为空")
    if not isinstance(learner_role, dict) or not str(
        learner_role.get("name") or ""
    ).strip():
        return _fail("INVALID_INPUT", "roles.learner_role.name 不能为空")
    for label, role in (("ai_role", ai_role), ("learner_role", learner_role)):
        for key in ("name", "identity", "style", "goal"):
            if role.get(key) and len(str(role[key])) > _MAX_NAME:
                return _fail("INVALID_INPUT", f"roles.{label}.{key} 超长（≤{_MAX_NAME}）")

    groups = body.groups
    if not isinstance(groups, list) or not groups:
        return _fail("INVALID_INPUT", "mode=start 时 groups（array，1~10 组）必填")
    if len(groups) > _MAX_GROUPS:
        return _fail("INVALID_INPUT", f"groups 组数超上限（≤{_MAX_GROUPS}）")
    for gi, group in enumerate(groups):
        if not isinstance(group, dict):
            return _fail("INVALID_INPUT", f"groups[{gi}] 必须是 object")
        kind = str(group.get("kind") or "new")
        if kind not in _VALID_KINDS:
            return _fail(
                "INVALID_INPUT", f"groups[{gi}].kind 非法：{kind}（new/review）"
            )
        sentences = group.get("sentences")
        if not isinstance(sentences, list) or not sentences:
            return _fail("INVALID_INPUT", f"groups[{gi}].sentences 不能为空")
        if len(sentences) > _MAX_SENTENCES_PER_GROUP:
            return _fail(
                "INVALID_INPUT",
                f"groups[{gi}].sentences 超上限（≤{_MAX_SENTENCES_PER_GROUP}）",
            )
        for si, s in enumerate(sentences):
            if not isinstance(s, dict):
                return _fail("INVALID_INPUT", f"groups[{gi}].sentences[{si}] 必须是 object")
            sid = str(s.get("sentence_id") or "").strip()
            content = str(s.get("content") or "").strip()
            if not sid:
                return _fail(
                    "INVALID_INPUT", f"groups[{gi}].sentences[{si}].sentence_id 不能为空"
                )
            if len(sid) > _MAX_SESSION_ID:
                return _fail(
                    "INVALID_INPUT",
                    f"groups[{gi}].sentences[{si}].sentence_id 超长（≤{_MAX_SESSION_ID}）",
                )
            if not content:
                return _fail(
                    "INVALID_INPUT", f"groups[{gi}].sentences[{si}].content 不能为空"
                )
            if len(content) > _MAX_CONTENT:
                return _fail(
                    "INVALID_INPUT",
                    f"groups[{gi}].sentences[{si}].content 超长（≤{_MAX_CONTENT}）",
                )
    return None


def _build_start_context(scenario: dict, roles: dict, groups: list) -> dict:
    """start 上下文快照：mode=start + 场景/角色/素材，history 为空（同 v3）。"""
    return {
        "mode": "start",
        "scenario": scenario,
        "roles": roles,
        "materials": groups,
        "history": [],
        "user_input": None,
        "assisted": False,
        "target_sentence_ids": [],
    }


def _build_hint_status(task: dict) -> str:
    """hint 异步态（契约 §3.17）：sync 时 hint 随 result 同轮返回。"""
    status = task.get("status")
    hint_mode = str(task.get("hint_mode") or "sync")
    if status == "failed":
        return "failed"
    if status == "success":
        result = task.get("result") or {}
        return "ready" if result.get("hint") else "none"
    # pending / processing
    return "pending" if hint_mode == "on_demand" else "ready"


# ---------------------------------------------------------------------------
# 提交：POST /ai/session/v4
# ---------------------------------------------------------------------------


@router.post("/ai/session/v4", response_model=AiSessionResponse)
async def submit_ai_session_v4(
    body: AiSessionV4Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> AiSessionResponse:
    """提交会话 v4 生成任务 — 毫秒级返回 task_id/session_id，生成在后台执行。

    - 总开关关闭（SESSION_V4_ENABLED=0）→ SESSION_V4_DISABLED（零侵入可一键回退）；
    - mode=start：校验三要素 → 建会话态（ai_session_v4）+ 任务（占在途位）→ 调度；
    - mode=turn：装载会话态（归属校验 + 在途位 gate）→ 建任务（context 自包含快照）→ 调度；
    - v4 追加可选字段 `stream` / `hint_mode` / `model_tier` 透传至任务文档。
    """
    # 0. 总开关（默认 0 关闭；不影响已创建任务的查询）
    if not SESSION_V4_ENABLED:
        return _fail("SESSION_V4_DISABLED", "会话 v4 未启用（SESSION_V4_ENABLED=0）")

    # 1. 公共校验（业务失败 200 + success=false）
    bad = _validate_scholar_id(body)
    if bad:
        return bad
    mode = (body.mode or "start").strip().lower()
    if mode not in _VALID_MODES:
        return _fail("INVALID_INPUT", f"mode 非法：{mode or '(空)'}（start/turn）")
    bad = _validate_preferred_type(body)
    if bad:
        return bad
    bad = _validate_v4_options(body)
    if bad:
        return bad
    preferred_type = (body.preferred_type or "auto").strip().lower()

    request_stream = bool(body.stream)
    hint_mode = str(body.hint_mode or "sync").strip().lower()
    model_tier = str(body.model_tier or "standard").strip().lower()

    session_id: str | None = None
    task_id: str | None = None
    try:
        if mode == "start":
            bad = _validate_start_payload(body)
            if bad:
                return bad
            scenario = body.scenario
            roles = body.roles
            groups = body.groups
            session_id = build_session_id()
            context = _build_start_context(scenario, roles, groups)
            # 先建任务（含 session_id），再建会话态（pending_task=本任务，创建即占位）
            task_doc = await create_session_task(
                db,
                task_id=build_task_id(),
                scholar_id=body.scholar_id,
                session_id=session_id,
                mode=mode,
                preferred_type=preferred_type,
                context=context,
                stream=request_stream,
                hint_mode=hint_mode,
                model_tier=model_tier,
            )
            task_id = task_doc["task_id"]
            await create_session(
                db,
                session_id=session_id,
                scholar_id=body.scholar_id,
                scenario=scenario,
                roles=roles,
                materials=groups,
                pending_task=task_id,
            )
        else:  # turn
            user_input = str(body.user_input or "").strip()
            session_id = str(body.session_id or "").strip()
            if not session_id:
                return _fail("SESSION_NOT_FOUND", "mode=turn 时 session_id 必填")
            if not user_input:
                return _fail("INVALID_INPUT", "mode=turn 时 user_input 不能为空")
            if len(user_input) > _MAX_USER_INPUT:
                return _fail("INVALID_INPUT", f"user_input 超长（≤{_MAX_USER_INPUT}）")
            sess = await get_session(db, session_id)
            now_ms = int(time.time() * 1000)
            if sess is None or sess.get("expires_at", 0) <= now_ms:
                return _fail("SESSION_NOT_FOUND", "会话不存在或已过期")
            if sess.get("scholar_id") != body.scholar_id:
                return _fail(
                    "SESSION_NOT_FOUND", "会话归属学者不符（scholar_id 不一致）"
                )
            # 单在途任务 gate：被占用 → TURN_IN_PROGRESS（先占位后建任务，避免悬空任务）
            task_id = build_task_id()
            if not await set_pending(db, session_id=session_id, task_id=task_id):
                return _fail(
                    "TURN_IN_PROGRESS",
                    "该会话已有在途生成任务，请轮询到终态后再提交",
                )
            context = {
                "mode": "turn",
                "scenario": sess.get("scenario") or {},
                "roles": sess.get("roles") or {},
                "materials": sess.get("materials") or [],
                "history": sess.get("history") or [],
                "user_input": user_input,
                "assisted": bool(body.assisted),
                "target_sentence_ids": [],
            }
            task_doc = await create_session_task(
                db,
                task_id=task_id,
                scholar_id=body.scholar_id,
                session_id=session_id,
                mode=mode,
                preferred_type=preferred_type,
                context=context,
                stream=request_stream,
                hint_mode=hint_mode,
                model_tier=model_tier,
            )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[ai_v4] session v4 任务创建异常: {e}", exc_info=True)
        # 尽力回滚：会话态未建成功时清掉悬空任务；在途位占住时释放
        try:
            if mode == "start" and task_id:
                await db.delete(
                    COLLECTION, where={"task_id": task_id}, multi=False
                )
            elif mode == "turn" and session_id and task_id:
                await session_state.release_pending(
                    db, session_id=session_id, task_id=task_id
                )
        except Exception:  # noqa: BLE001
            pass
        raise HTTPException(status_code=500, detail=f"任务创建失败: {str(e)}")

    # 2. 后台执行，不阻塞当前请求（毫秒级返回）。
    bg = asyncio.create_task(run_session_task(task_id))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    logger.info(
        f"[ai_v4] session v4 任务已提交 → task_id={task_id}, mode={mode}, "
        f"session_id={session_id}, preferred_type={preferred_type}, stream={request_stream}, "
        f"scholar={body.scholar_id}"
    )
    return AiSessionResponse(
        success=True,
        data={
            "task_id": task_id,
            "status": task_doc["status"],
            "session_id": session_id,
        },
    )


# ---------------------------------------------------------------------------
# 查询：GET /ai/session/v4/task/{task_id}
# ---------------------------------------------------------------------------


@router.get("/ai/session/v4/task/{task_id}", response_model=AiSessionResponse)
async def get_ai_session_v4_task(
    task_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> AiSessionResponse:
    """查询异步会话 v4 生成结果 — pending/processing/success/failed（同 v3）+ 伪流式增量字段。"""
    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 定点自愈：被查询任务若卡死（processing 且超时）→ 置 failed + 释放会话在途位
        if await recover_task_if_stale(db, task):
            task = await get_task(db, task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="任务不存在")
        # TTL 过滤：expires_at 已过期的按不存在处理（物理清理由后台巡检执行）
        now_ms = int(time.time() * 1000)
        if task.get("expires_at", 0) <= now_ms:
            raise HTTPException(status_code=404, detail="任务已过期")

        raw_error = task.get("error")
        error_display = (
            raw_error.get("error_detail") if isinstance(raw_error, dict) else raw_error
        )
        return AiSessionResponse(
            success=True,
            data={
                "task_id": task["task_id"],
                "status": task["status"],
                "result": task.get("result"),
                "error": error_display,
                # ---- v4 追加（终态时 partial_text 已被清空）----
                "partial_text": task.get("partial_text"),
                "partial_seq": task.get("partial_seq"),
                "partial_updated_at": task.get("partial_updated_at"),
                "hint_status": _build_hint_status(task),
                "timings": task.get("timings"),
            },
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"[ai_v4] session v4 任务查询异常: task_id={task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务查询失败: {str(e)}")
