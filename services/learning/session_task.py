"""沉浸式 AI 会话 v2 异步生成任务模型 — `ai_session_task` 集合 CRUD 与状态流转

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
- run_session_task       : 后台执行器：claim → 火山生成（session_gen，wait_for 超时）
                           → 回写会话历史并释放在途位（session_state）→ finish_task

与 `translation_task`（services/learning/translation_task.py）的差异（§4.18）：
- task_id 前缀 `st_`（data-model-contract §4.18）；
- 任务自包含 `context`（截至提交时的场景/角色/素材/history/user_input/assisted 快照），
  任务即执行唯一依据，不依赖会话态可独立重跑；
- 不写 `evaluation` 证据（会话域计费/质量对账为后续扩展，避免污染评估域）；
- 任务终态（success/failed）**同步释放其占用的会话在途位**（§4.19 `ai_session.pending_task`，
  防会话死锁）；卡死自愈/巡检置 failed 时同样释放；
- 失败（failed）**不污染会话历史**（该轮 AI 产出丢弃，仅 success 回写，§4.19 失败口径）。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import SESSION_LLM_TIMEOUT_SECONDS
from services.dependencies import get_db
from services.learning import session_state
from services.providers import session_gen
from services.providers.session_gen import (
    ERR_LLM_TIMEOUT,
    ERR_NETWORK_ERROR,
    STAGE_LLM,
    SessionGenError,
)

logger = logging.getLogger("scholar-admin.session_task")

COLLECTION = "ai_session_task"
SESSION_COLLECTION = "ai_session"

# 任务默认保留时长：24h（与 translation_task/dialogue_task 一致，轮询窗口 + 容错重试绰绰有余）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`st_` + 32 位 uuid hex。"""
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

    不做任何 LLM 调用，保证调用方（提交接口）耗时毫秒级（ADR-0022 决策 A 同款）。

    `task_id` 可缺省（自动生成 `st_<32hex>`）；turn 提交需先抢占会话在途位
    （session_state.set_pending），故路由层预生成 task_id 后传入，保证占位与任务一致。

    字段对齐 data-model-contract §4.18；`context` 为生成自包含快照：
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
        f"[session] create → task_id={task_doc['task_id']}, session_id={session_id}, "
        f"mode={mode}, preferred_type={preferred_type}, scholar={scholar_id}"
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing。

    并发安全：where 限定 status=pending，multi=False 只更新一条；
    `modified_count>0` 说明本实例抢占成功，否则已被其他实例抢占/状态已变。
    """
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（api-contract §3.12：卡死自愈 → LLM_TIMEOUT）。"""
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
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。

    error 为对象 { error_code, error_detail, failure_stage, llm_timeout_seconds, raw }。
    """
    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = error
        logger.info(
            f"[session] fail → task_id={task_id}, error={error.get('error_code')}"
        )
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[session] done → task_id={task_id}, status=success")
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
    """删除 expires_at <= now 的过期任务，返回删除数量。

    注意：任务过期即删除（不再需要释放会话在途位——ai_session 同 TTL 过期，
    会话态本身已清理；若因写序原因会话仍存活，其 pending_task 指向的任务已
    failed/不存在，不会造成二次占用）。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[session] cleanup → 删除过期任务 {count} 条")
    return count


async def _release_session_slot(db, session_id: str | None, task_id: str | None) -> None:
    """释放会话在途位：`ai_session.pending_task == task_id` 时置 null（§4.19）。

    where 条件带 pending_task=task_id：只清空指向本任务的占位，
    不会误清新任务占位（并发下后到任务不受影响）。落库失败仅记日志不阻断。
    """
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
        logger.error(f"[session] 释放会话在途位失败 session={session_id}: {e}")


async def recover_stale_tasks(db, timeout_s: int | None = None) -> int:
    """巡检卡死的 processing 任务：updated_at 超过 timeout_s 未更新 → 置为 failed，
    并**同步释放各卡死任务占用的会话在途位**（§4.18：卡死自愈/巡检置 failed 时同步释放）。

    兜底容器回收 / 进程崩溃导致的 processing 卡死（任务记录是唯一状态源，
    实例挂了没有 else 分支写 failed，必须靠巡检恢复）。

    Args:
        timeout_s: 卡死判定阈值（秒）。缺省取 SESSION_LLM_TIMEOUT_SECONDS——
            与 translation 巡检不同，会话生成单次 LLM 调用上限即该常量，
            阈值必须 ≥ LLM 上限，避免合法长调用被巡检误杀。

    Returns:
        修复（置为 failed）的任务数量
    """
    timeout = timeout_s or SESSION_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    threshold = now - timeout * 1000
    stale_where = {
        "status": STATUS_PROCESSING,
        "updated_at": {"$lt": threshold},
    }
    # 先取卡死任务明细（需 session_id 以释放会话在途位）。卡死是低频异常，
    # 全量条件查询可接受（后台巡检 60s 一轮，非查询热路径）。
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
        logger.info(f"[session] recover → 卡死任务标记 failed {count} 条")
    for t in stale_tasks:
        await _release_session_slot(db, t.get("session_id"), t.get("task_id"))
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int | None = None) -> bool:
    """定点恢复：单条卡死 processing 任务（updated_at 超时）→ 置 failed，
    并同步释放该任务占用的会话在途位。

    与 `recover_stale_tasks` 的区别：只针对传入任务的 task_id 做单点更新，
    避免查询热路径触发全集合条件更新（无索引时全表扫描会拖慢轮询）。

    Args:
        task: get_task 返回的任务文档
        timeout_s: 卡死判定阈值（秒），缺省 SESSION_LLM_TIMEOUT_SECONDS（同上）

    Returns:
        是否恢复成功（该任务被置为 failed）
    """
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
            f"[session] recover → task_id={task['task_id']} 卡死任务标记 failed"
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
    """后台执行会话生成任务并写回结果（§4.18/§4.19 / proposal §3.2）。

    由提交接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim_task 原子抢占：被其他实例抢占则直接返回，避免重复执行
    - 生成：任务 `context` 自包含快照即执行唯一依据（不依赖会话态可独立重跑），
      火山生成（services/providers/session_gen，超时 SESSION_LLM_TIMEOUT_SECONDS）
    - 成功：回写 AI 产出至 ai_session.history 并释放 pending_task
      （services.learning.session_state.complete_turn），随后 finish_task success
    - 失败：释放 pending_task（不污染 history，session_state.release_pending），
      随后 finish_task failed（不降级、不静默）
    - 不写 evaluation 证据（会话域计费/质量对账为后续扩展）
    """
    db = get_db()
    task = await get_task(db, task_id)
    if task is None:
        logger.info(f"[session] run skip → task_id={task_id} 不存在（已 TTL 清理）")
        return
    if not await claim_task(db, task_id):
        logger.info(f"[session] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return
    session_id = task.get("session_id")
    result: dict | None = None
    error: dict | None = None
    try:
        payload = await session_gen.generate_session_reply(
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
        # 仅 turn 模式回写 user 条（assisted 上报自 context 快照）；
        # start 开场同样回写——单条 ai 开场白（不入 user 条，§4.19 回写口径）。
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
            logger.error(f"[session] run error → task_id={task_id}: {e}", exc_info=True)
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
                    f"[session] 释放在途位失败 session_id={session_id}: {exc}",
                    exc_info=True,
                )
    await finish_task(db, task_id, result=result, error=error)
