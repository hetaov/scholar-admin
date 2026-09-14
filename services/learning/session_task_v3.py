"""沉浸式 AI 会话 v3 异步生成任务模型 — `ai_session_v3_task` 集合 CRUD 与状态流转（§11.2）

同构复制 `services/learning/session_task.py`（v2）：状态机、并发口径、失败回写规则与
v2 逐条一致，**仅集合名参数化**（`ai_session_v3_task` / `ai_session_v3`）且生成核改为
`services.learning.dialogue_engine`（v3 新引擎），**不写 v2 集合**。

状态机（单向，禁止回退）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

- create_session_task    : 生成 `task_id`（`st_` 前缀），插入 pending 任务，返回任务文档
- claim_task             : 原子抢占 pending → processing（multi=False + modified_count>0 判成功）
- finish_task            : processing → success(+result) | failed(+error)
- get_task               : 按 task_id 查询（不过滤 TTL，接口层自行过滤）
- cleanup_expired        : 删除 expires_at <= now 的任务（TTL 清理，后台巡检用）
- recover_stale_tasks    : 巡检卡死 processing 任务（全集合，后台巡检每 60s）
- recover_task_if_stale  : 定点恢复单条卡死任务（查询热路径，避免全表扫描）
- run_session_task       : 后台执行器：claim → dialogue_engine 生成（超时 SESSION_LLM_TIMEOUT_SECONDS）
                           → 回写会话历史并释放在途位（session_state_v3）→ finish_task

与 v2 的差异仅两处（§11.4「差异仅在路径与集合」）：
- 集合名参数化为 `ai_session_v3_task` / `ai_session_v3`；
- 生成核为 `dialogue_engine`（新引擎：召回 clamp + LangGraph 会话图 + 断点续写），
  出口结构对齐 v2 `{ content_type, ai_text, hint, suggested_targets }`。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import SESSION_LLM_TIMEOUT_SECONDS, SESSION_V3_STATE_COLLECTION, SESSION_V3_TASK_COLLECTION
from services.dependencies import get_db
from services.learning import session_state_v3 as session_state
from services.learning import dialogue_engine
from services.providers.session_gen import (
    ERR_LLM_TIMEOUT,
    ERR_NETWORK_ERROR,
    STAGE_LLM,
    SessionGenError,
)

logger = logging.getLogger("scholar-admin.session_task_v3")

COLLECTION = SESSION_V3_TASK_COLLECTION
SESSION_COLLECTION = SESSION_V3_STATE_COLLECTION

# 任务默认保留时长：24h（与 session_state_v3 同口径，轮询窗口 + 容错重试绰绰有余）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`st_` + 32 位 uuid hex（与 v2 同前缀，§11.4）。"""
    return "st_" + uuid.uuid4().hex


async def create_session_task(
    db,
    *,
    task_id: str | None = None,
    scholar_id: str,
    session_id: str,
    mode: str,
    preferred_type: str,
    context: dict,
) -> dict:
    """创建 pending 任务并落库，返回任务文档。

    不做任何 LLM 调用，保证调用方（提交接口）耗时毫秒级（与 v2 逐条一致）。

    `task_id` 可缺省（自动生成 `st_<32hex>`）；turn 提交需先抢占会话在途位
    （session_state_v3.set_pending），故路由层预生成 task_id 后传入，保证占位与任务一致。

    字段对齐 v2 `ai_session_task`（§4.18）；`context` 为生成自包含快照：
    `{ mode, scenario, roles, materials(groups), history(≤20 截至提交时), user_input, assisted,
    target_sentence_ids }`——任务即执行唯一依据，不依赖会话态可独立重跑。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": task_id or build_task_id(),
        "scholar_id": scholar_id,
        "session_id": session_id,
        "mode": mode,
        "preferred_type": preferred_type,
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "context": context,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[session_v3] create → task_id={task_doc['task_id']}, session_id={session_id}, "
        f"mode={mode}, preferred_type={preferred_type}, scholar={scholar_id}"
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing（并发安全，同 v2）。"""
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（§11.4：卡死自愈 → LLM_TIMEOUT）。"""
    return {
        "error_code": ERR_LLM_TIMEOUT,
        "error_detail": "执行超时",
        "failure_stage": STAGE_LLM,
        "llm_timeout_seconds": SESSION_LLM_TIMEOUT_SECONDS,
        "raw": None,
    }


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | list | None = None,
    error: dict | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。"""
    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = error
        logger.info(
            f"[session_v3] fail → task_id={task_id}, error={error.get('error_code')}"
        )
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[session_v3] done → task_id={task_id}, status=success")
    await db.update(
        COLLECTION,
        where={"task_id": task_id},
        data={
            "$set": {
                "status": status,
                "result": result_value,
                "error": error_value,
                "updated_at": _now_ms(),
            }
        },
        multi=False,
    )


async def get_task(db, task_id: str) -> dict | None:
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL，接口层自行过滤）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期任务，返回删除数量（同 v2 TTL 口径）。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[session_v3] cleanup → 删除过期任务 {count} 条")
    return count


async def _release_session_slot(db, session_id: str | None, task_id: str | None) -> None:
    """释放会话在途位：`ai_session_v3.pending_task == task_id` 时置 null（§4.19）。"""
    if not session_id or not task_id:
        return
    try:
        await db.update(
            SESSION_COLLECTION,
            where={"session_id": session_id, "pending_task": task_id},
            data={"$set": {"pending_task": None, "updated_at": _now_ms()}},
            multi=False,
        )
    except Exception as e:  # noqa: BLE001 — 释放失败不影响任务收尾
        logger.error(f"[session_v3] 释放会话在途位失败 session={session_id}: {e}")


async def recover_stale_tasks(db, timeout_s: int | None = None) -> int:
    """巡检卡死的 processing 任务：超时未更新 → 置 failed，并同步释放会话在途位。"""
    timeout = timeout_s or SESSION_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    threshold = now - timeout * 1000
    stale_where = {
        "status": STATUS_PROCESSING,
        "updated_at": {"$lt": threshold},
    }
    res = await db.query(COLLECTION, where=stale_where, limit=1000)
    stale_tasks = res.get("records", [])
    if not stale_tasks:
        return 0
    upd = await db.update(
        COLLECTION,
        where=stale_where,
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": _build_stale_error(),
                "updated_at": now,
            }
        },
        multi=True,
    )
    count = upd.get("modified_count", 0)
    if count:
        logger.info(f"[session_v3] recover → 卡死任务标记 failed {count} 条")
    for t in stale_tasks:
        await _release_session_slot(db, t.get("session_id"), t.get("task_id"))
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int | None = None) -> bool:
    """定点恢复：单条卡死 processing 任务 → 置 failed，并同步释放会话在途位。"""
    if task.get("status") != STATUS_PROCESSING:
        return False
    timeout = timeout_s or SESSION_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    if task.get("updated_at", 0) > now - timeout * 1000:
        return False
    res = await db.update(
        COLLECTION,
        where={"task_id": task["task_id"], "status": STATUS_PROCESSING},
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": _build_stale_error(),
                "updated_at": now,
            }
        },
        multi=False,
    )
    if res.get("modified_count", 0):
        logger.info(
            f"[session_v3] recover → task_id={task['task_id']} 卡死任务标记 failed"
        )
        await _release_session_slot(db, task.get("session_id"), task["task_id"])
        return True
    return False


def _session_gen_error_to_dict(e: Exception) -> dict | None:
    """把生成域业务异常映射为任务 error 对象；非生成域异常返回 None（通用兜底）。"""
    if isinstance(e, SessionGenError):
        return e.to_dict(llm_timeout_seconds=SESSION_LLM_TIMEOUT_SECONDS)
    return None


async def run_session_task(task_id: str) -> None:
    """后台执行会话 v3 生成任务并写回结果（§11.2 / §11.3）。

    由提交接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim_task 原子抢占：被其他实例抢占则直接返回，避免重复执行
    - 生成：任务 `context` 自包含快照即执行唯一依据，走 v3 引擎
      （services.learning.dialogue_engine，超时 SESSION_LLM_TIMEOUT_SECONDS）
    - 成功：回写 AI 产出至 ai_session_v3.history 并释放 pending_task
      （services.learning.session_state_v3.complete_turn），随后 finish_task success
    - 失败：释放 pending_task（不污染 history，session_state_v3.release_pending），
      随后 finish_task failed（不降级、不静默）
    """
    db = get_db()
    task = await get_task(db, task_id)
    if task is None:
        logger.info(f"[session_v3] run skip → task_id={task_id} 不存在（已 TTL 清理）")
        return
    if not await claim_task(db, task_id):
        logger.info(f"[session_v3] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return
    session_id = task.get("session_id")
    result: dict | None = None
    error: dict | None = None
    try:
        payload = await dialogue_engine.generate_session_reply(
            db=db,
            session_id=session_id,
            context=task.get("context") or {},
            preferred_type=task.get("preferred_type", "auto"),
        )
        result = {
            "session_id": session_id,
            "content_type": payload["content_type"],
            "ai_text": payload["ai_text"],
            "hint": payload.get("hint"),
            "suggested_targets": payload.get("suggested_targets") or [],
        }
        # 仅 success 回写会话历史并释放在途位（§4.19 回写/失败口径）。
        if session_id:
            ctx = task.get("context") or {}
            await session_state.complete_turn(
                db,
                session_id=session_id,
                task_id=task_id,
                user_text=(
                    ctx.get("user_input") if task.get("mode") == "turn" else None
                ),
                assisted=bool(ctx.get("assisted")),
                ai_text=result["ai_text"],
                content_type=result["content_type"],
                suggested_targets=result.get("suggested_targets"),
            )
    except Exception as e:  # noqa: BLE001
        error = _session_gen_error_to_dict(e)
        if error is None:
            logger.error(f"[session_v3] run error → task_id={task_id}: {e}", exc_info=True)
            error = {
                "error_code": ERR_NETWORK_ERROR,
                "error_detail": str(e)[:500],
                "failure_stage": STAGE_LLM,
                "llm_timeout_seconds": SESSION_LLM_TIMEOUT_SECONDS,
                "raw": None,
            }
        # 失败：释放在途位（不污染 history）；自身异常仅记日志不阻断任务收尾
        if session_id:
            try:
                await session_state.release_pending(
                    db, session_id=session_id, task_id=task_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    f"[session_v3] 释放在途位失败 session_id={session_id}: {exc}",
                    exc_info=True,
                )
    await finish_task(db, task_id, result=result, error=error)
