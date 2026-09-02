"""沉浸式 AI 会话 v2 接口（proposal 2026-09-02 / api-contract §3.12）

POST /ai/session/v2 — 提交会话生成任务（mode=start 开场 / mode=turn 续轮）。
- 提交毫秒级返回 { task_id: "st_xxx", status: "pending", session_id: "s_xxx" }，
  不做任何 LLM 调用（生成由后台 run_session_task 执行）；
- 入参核心字段可选化 + 手动校验 → 业务失败 HTTP 200 + success=false + code：
  INVALID_INPUT（缺参/空参/超长/枚举非法）/ SESSION_NOT_FOUND（turn 的 session_id
  缺失、不存在、已过期或归属 scholar 不符）/ TURN_IN_PROGRESS（会话已有在途任务，
  前端轮询到终态后再提交）/ TYPE_NOT_SUPPORTED（preferred_type=retell/task，MVP 边界）；
- 技术异常 → HTTP 500。

GET /ai/session/v2/task/{task_id} — 查询生成结果。
- 状态枚举 pending/processing/success/failed；TTL 24h 过期 → 404；
- 卡死自愈：processing 超时（SESSION_LLM_TIMEOUT_SECONDS）→ 查询定点置 failed
  并释放会话在途位；全集合巡检（recover_stale_tasks + cleanup_expired）由后台
  定时任务执行（services/background_tasks.py）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.learning import session_state
from services.learning.session_state import (
    build_session_id,
    create_session,
    get_session,
    set_pending,
)
from services.learning.session_task import (
    COLLECTION,
    build_task_id,
    create_session_task,
    get_task,
    recover_task_if_stale,
    run_session_task,
)

logger = logging.getLogger("scholar-admin.ai")

router = APIRouter(tags=["ai"])

# 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消（同 eval.py）
_background_tasks: set[asyncio.Task] = set()

# MVP 形态白名单（docs_v1 §11：auto/dialogue/fill；retell/task → TYPE_NOT_SUPPORTED）
_SUPPORTED_PREFERRED_TYPES = ("auto", "dialogue", "fill")
_UNSUPPORTED_PREFERRED_TYPES = ("retell", "task")
_VALID_MODES = ("start", "turn")
_VALID_KINDS = ("new", "review")

# 长度口径（api-contract §3.12 手动校验）
_MAX_SCHOLAR_ID = 64
_MAX_SESSION_ID = 64
_MAX_NAME = 100
_MAX_SCENE = 500
_MAX_CONTENT = 500
_MAX_USER_INPUT = 2000
_MAX_GROUPS = 10
_MAX_SENTENCES_PER_GROUP = 50


class AiSessionV2Request(BaseModel):
    """会话 v2 提交请求。

    核心字段全部可选 + 手动校验（契约 §3.12：业务失败 200 + success=false，无 4xx），
    缺参/空参/超长/枚举非法统一由路由返回 INVALID_INPUT 等业务码。
    """

    scholar_id: Optional[str] = None
    mode: Optional[str] = None  # start / turn，缺省 start
    session_id: Optional[str] = None  # turn 必填
    scenario: Optional[dict] = None  # start 必填：{ scene_id?, title?, scene, goal?, constraints? }
    roles: Optional[dict] = None  # start 必填：{ ai_role, learner_role }
    groups: Optional[list] = None  # start 必填：素材组
    user_input: Optional[str] = None  # turn 必填 ≤2000
    preferred_type: Optional[str] = None  # auto / dialogue / fill，缺省 auto
    assisted: Optional[bool] = None  # turn 时前端上报本轮是否借助提示卡


class AiSessionResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


# ---------------------------------------------------------------------------
# 手动校验 helpers（业务失败 200 + success=false，对齐契约 §3.12 / §3.4 惯例）
# ---------------------------------------------------------------------------


def _fail(code: str, message: str) -> AiSessionResponse:
    return AiSessionResponse(success=False, code=code, message=message)


def _validate_scholar_id(body: AiSessionV2Request) -> AiSessionResponse | None:
    if not body.scholar_id or not str(body.scholar_id).strip():
        return _fail("INVALID_INPUT", "scholar_id 不能为空")
    if len(str(body.scholar_id)) > _MAX_SCHOLAR_ID:
        return _fail("INVALID_INPUT", f"scholar_id 超长（≤{_MAX_SCHOLAR_ID}）")
    return None


def _validate_preferred_type(body: AiSessionV2Request) -> AiSessionResponse | None:
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


def _validate_start_payload(body: AiSessionV2Request) -> AiSessionResponse | None:
    """start 模式必填结构校验：scenario / roles / groups（含叶子长度口径）。"""
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


def _build_start_context(
    body: AiSessionV2Request, scenario: dict, roles: dict, groups: list
) -> dict:
    """start 上下文快照：mode=start + 场景/角色/素材，history 为空（§4.18）。"""
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


# ---------------------------------------------------------------------------
# 提交：POST /ai/session/v2
# ---------------------------------------------------------------------------


@router.post("/ai/session/v2", response_model=AiSessionResponse)
async def submit_ai_session_v2(
    body: AiSessionV2Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> AiSessionResponse:
    """提交会话生成任务 — 毫秒级返回 task_id/session_id，生成在后台执行。

    - mode=start：校验三要素（scenario/roles/groups）→ 建会话态 + 任务（占在途位）→ 调度；
    - mode=turn：按 session_id 装载会话态（归属校验 + 在途位 gate）→ 建任务（context
      自包含快照）→ 调度；user_input 为空 → INVALID_INPUT。
    """
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
    preferred_type = (body.preferred_type or "auto").strip().lower()

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
            context = _build_start_context(body, scenario, roles, groups)
            # 先建任务（含 session_id），再建会话态（pending_task=本任务，创建即占位）
            task_doc = await create_session_task(
                db,
                task_id=build_task_id(),
                scholar_id=body.scholar_id,
                session_id=session_id,
                mode=mode,
                preferred_type=preferred_type,
                context=context,
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
            if not await set_pending(
                db, session_id=session_id, task_id=task_id
            ):
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
            )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[ai] session v2 任务创建异常: {e}", exc_info=True)
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
    #    必须持有 task 引用并注册 done_callback，否则协程可能被 GC 回收而取消（同 eval.py）。
    bg = asyncio.create_task(run_session_task(task_id))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    logger.info(
        f"[ai] session v2 任务已提交 → task_id={task_id}, mode={mode}, "
        f"session_id={session_id}, preferred_type={preferred_type}, scholar={body.scholar_id}"
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
# 查询：GET /ai/session/v2/task/{task_id}
# ---------------------------------------------------------------------------


@router.get("/ai/session/v2/task/{task_id}", response_model=AiSessionResponse)
async def get_ai_session_v2_task(
    task_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> AiSessionResponse:
    """查询异步会话生成结果 — pending/processing/success/failed。

    返回：{ success: true, data: { task_id, status, result, error } }
    - result（success 时）：{ session_id, content_type, ai_text, hint, suggested_targets }
    - error（failed 时）：可读失败原因字符串；完整 error 对象留在任务文档
    - 任务不存在 / 过期（TTL 24h）→ HTTP 404
    """
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

        # error 展示为可读字符串（完整 error 对象保留在任务文档中）
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
            },
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"[ai] session v2 任务查询异常: task_id={task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务查询失败: {str(e)}")
