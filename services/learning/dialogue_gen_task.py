"""AI 对话生成域任务模型 — `ai_dialogue_task` 集合 CRUD 与状态流转（设计稿 §6.1）。

状态机（单向，禁止回退；同构复用 `session_task` 范式）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

- create_task              : 生成 `task_id`（`dg_` 前缀），插入 pending 任务，返回任务文档
- claim_task               : 原子抢占 pending → processing（multi=False + modified_count>0）
- claim_resume_task        : 原子抢占 failed → processing（T4 断点续写入口）
- finish_task              : processing → success(+result) | failed(+error)
- get_task                 : 按 task_id 查询（不过滤 TTL，接口层自行过滤）
- cleanup_expired          : 删除 expires_at <= now 的任务及其 checkpoint（TTL 清理，后台巡检用）
- recover_stale_tasks      : 巡检卡死 processing 任务（全集合，后台巡检每 60s）
- recover_task_if_stale    : 定点恢复单条卡死任务（查询热路径，避免全表扫描）
- run_dialogue_gen_task    : 后台执行器：claim → 生成（图/直连）→ finish（含断点续写）

与 `ai_session_task` 的差异（设计稿 §6.1）：
- `task_id` 前缀 `dg_`；无 `session_id`/`mode`（批量生成，非逐轮会话）；
- `context` 快照为 `{ task_group, scenario, roles, recall, metrics, prompt_lang, preferred_type }`；
- 新增 `checkpoint_id`（断点续写游标，T1 恒为 None）与 `retry_count`（覆盖重试预算，T1 恒 0）；
- 新增 `user_inputs`（T5 用户作答评测回写，默认空数组；只写本集合，不写 skill_state/evaluation）；
- 不写 `evaluation` 证据（生成域产物，避免污染评估域）；
- 不写任何 `ai_session*` 集合（两面隔离，§8.4 第 16 条）。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import (
    DIALOGUE_CHECKPOINT_COLLECTION,
    DIALOGUE_GEN_CHECKPOINT_ENABLED,
    DIALOGUE_GEN_GRAPH_ENABLED,
    DIALOGUE_LLM_TIMEOUT_SECONDS,
)
from services.dependencies import get_db
from services.learning import dialogue_gen_graph
from services.providers import dialogue_gen
from services.providers.dialogue_gen import (
    ERR_LLM_TIMEOUT,
    ERR_NETWORK_ERROR,
    STAGE_LLM,
    DialogueGenError,
)

logger = logging.getLogger("scholar-admin.dialogue_gen_task")

COLLECTION = "ai_dialogue_task"

# 任务保留时长：24h（与 translation_task/session_task 一致）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`dg_` + 32 位 uuid hex。"""
    return "dg_" + uuid.uuid4().hex


async def create_task(
    db,
    *,
    task_id: str | None = None,
    scholar_id: str,
    context: dict,
) -> dict:
    """创建 pending 任务并落库，返回任务文档（不做任何 LLM 调用，毫秒级返回）。

    `context` 为生成自包含快照：`{ task_group, scenario, roles, recall, metrics,
    prompt_lang, preferred_type }`——任务即执行唯一依据，可独立重跑。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": task_id or build_task_id(),
        "scholar_id": scholar_id,
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "context": context,
        "checkpoint_id": None,
        "retry_count": 0,
        "user_inputs": [],
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[dialogue_gen] create → task_id={task_doc['task_id']}, scholar={scholar_id}"
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing（并发安全：where 限定 status=pending）。"""
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_PENDING},
        data={"$set": {"status": STATUS_PROCESSING, "updated_at": _now_ms()}},
        multi=False,
    )
    return res.get("modified_count", 0) > 0


async def claim_resume_task(db, task_id: str) -> bool:
    """原子抢占 failed → processing（T4 断点续写入口；并发安全：where 限定 status=failed）。

    续写会清除旧 error（进入 processing 后状态列不再展示历史失败原因）。
    """
    res = await db.update(
        COLLECTION,
        where={"task_id": task_id, "status": STATUS_FAILED},
        data={
            "$set": {
                "status": STATUS_PROCESSING,
                "error": None,
                "updated_at": _now_ms(),
            }
        },
        multi=False,
    )
    return res.get("modified_count", 0) > 0


def _build_stale_error() -> dict:
    """卡死任务恢复的 error 对象（对齐 session_task：卡死自愈 → LLM_TIMEOUT）。"""
    return {
        "error_code": ERR_LLM_TIMEOUT,
        "error_detail": "执行超时",
        "failure_stage": STAGE_LLM,
        "llm_timeout_seconds": DIALOGUE_LLM_TIMEOUT_SECONDS,
        "raw": None,
    }


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | None = None,
    error: dict | None = None,
    checkpoint_id: str | None = None,
    retry_count: int | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。

    同步更新 `checkpoint_id`（续写游标）与 `retry_count`（覆盖重试预算，T4 起生效）。
    """
    changes: dict[str, Any] = {"updated_at": _now_ms()}
    if error is not None:
        changes.update({"status": STATUS_FAILED, "result": None, "error": error})
        logger.info(
            f"[dialogue_gen] fail → task_id={task_id}, error={error.get('error_code')}"
        )
    else:
        changes.update({"status": STATUS_SUCCESS, "result": result, "error": None})
        logger.info(f"[dialogue_gen] done → task_id={task_id}, status=success")
    if retry_count is not None:
        changes["retry_count"] = retry_count
    if checkpoint_id is not None:
        changes["checkpoint_id"] = checkpoint_id
    await db.update(
        COLLECTION, where={"task_id": task_id}, data={"$set": changes}, multi=False
    )


async def get_task(db, task_id: str) -> dict | None:
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL，接口层自行过滤）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期任务及其 checkpoint，返回任务删除数量（后台巡检执行）。

    T4：任务过期后其 `ai_dialogue_checkpoint` 明细一并清理（thread_id = task_id），
    避免断点集合无界增长；checkpoint 清理为 best-effort，失败不影响任务清理结果。
    """
    now = now_ms if now_ms is not None else _now_ms()
    expired_where = {"expires_at": {"$lte": now}}
    # 先取过期 task_id（按 thread_id 清理对应 checkpoint），再删除任务
    task_ids: list[str] = []
    try:
        records = await db.query(COLLECTION, where=expired_where, limit=1000)
        task_ids = [
            str(r.get("task_id")) for r in records.get("records", []) if r.get("task_id")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[dialogue_gen] cleanup 读取过期任务失败（忽略）: {exc}")
    res = await db.delete(COLLECTION, where=expired_where)
    count = res.get("deleted_count", 0)
    if task_ids:
        try:
            await db.delete(
                DIALOGUE_CHECKPOINT_COLLECTION, where={"thread_id": {"$in": task_ids}}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[dialogue_gen] cleanup 清理 checkpoint 失败（忽略）: {exc}")
    if count:
        logger.info(f"[dialogue_gen] cleanup → 删除过期任务 {count} 条")
    return count


async def recover_stale_tasks(db, timeout_s: int | None = None) -> int:
    """巡检卡死的 processing 任务：updated_at 超时 → 置 failed（后台巡检）。

    Args:
        timeout_s: 卡死判定阈值（秒），缺省 `DIALOGUE_LLM_TIMEOUT_SECONDS`
            （必须 ≥ 单次 LLM 上限，避免合法长调用被误杀）。

    Returns:
        修复（置为 failed）的任务数量
    """
    timeout = timeout_s or DIALOGUE_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    stale_where = {
        "status": STATUS_PROCESSING,
        "updated_at": {"$lt": now - timeout * 1000},
    }
    upd = await db.update(
        COLLECTION,
        where=stale_where,
        data={"$set": {"status": STATUS_FAILED, "error": _build_stale_error(), "updated_at": now}},
        multi=True,
    )
    count = upd.get("modified_count", 0)
    if count:
        logger.info(f"[dialogue_gen] recover → 卡死任务标记 failed {count} 条")
    return count


async def recover_task_if_stale(db, task: dict, timeout_s: int | None = None) -> bool:
    """定点恢复：单条卡死 processing 任务 → 置 failed（查询热路径，避免全表扫描）。"""
    if task.get("status") != STATUS_PROCESSING:
        return False
    timeout = timeout_s or DIALOGUE_LLM_TIMEOUT_SECONDS
    now = _now_ms()
    if task.get("updated_at", 0) > now - timeout * 1000:
        return False
    res = await db.update(
        COLLECTION,
        where={"task_id": task["task_id"], "status": STATUS_PROCESSING},
        data={"$set": {"status": STATUS_FAILED, "error": _build_stale_error(), "updated_at": now}},
        multi=False,
    )
    if res.get("modified_count", 0):
        logger.info(f"[dialogue_gen] recover → task_id={task['task_id']} 卡死任务标记 failed")
        return True
    return False


def _dialogue_gen_error_to_dict(e: Exception) -> dict | None:
    """把生成域业务异常映射为任务 error 对象；非生成域异常返回 None（通用兜底）。"""
    if isinstance(e, DialogueGenError):
        return e.to_dict(llm_timeout_seconds=DIALOGUE_LLM_TIMEOUT_SECONDS)
    return None


async def run_dialogue_gen_task(
    task_id: str,
    *,
    resume: bool = False,
    from_checkpoint_id: str | None = None,
) -> None:
    """后台执行对话生成任务并写回结果。

    由提交/续写接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim 原子抢占（resume=False → pending；resume=True → failed），被其他实例抢占
      则直接返回，避免重复执行；
    - 生成编排：`DIALOGUE_GEN_GRAPH_ENABLED=1` → 走生成图（含覆盖校验/重试/降级 +
      断点续写），置 0 → 回退 T1 单函数直连生成（逐断言兼容、可一键回退）；
    - 成功 → finish_task success(+result，含 checkpoint_id/retry_count 回写)；
      失败 → finish_task failed(+error)，并刷新 checkpoint_id（供再次续写）。
    """
    db = get_db()
    task = await get_task(db, task_id)
    if task is None:
        logger.info(f"[dialogue_gen] run skip → task_id={task_id} 不存在（已 TTL 清理）")
        return
    claimed = await (claim_resume_task if resume else claim_task)(db, task_id)
    if not claimed:
        logger.info(
            f"[dialogue_gen] run skip → task_id={task_id} 已被抢占或状态不可执行"
            f"（resume={resume}）"
        )
        return

    result: dict | None = None
    error: dict | None = None
    checkpoint_id: str | None = None
    retry_count: int | None = None
    try:
        context = task.get("context") or {}
        preferred_type = str(context.get("preferred_type") or "auto")
        if DIALOGUE_GEN_GRAPH_ENABLED:
            result = await dialogue_gen_graph.run_dialogue_gen_graph(
                db=db,
                task_id=task_id,
                scholar_id=str(task.get("scholar_id") or ""),
                context=context,
                preferred_type=preferred_type,
                resume=resume,
                from_checkpoint_id=from_checkpoint_id,
            )
            checkpoint = result.get("checkpoint") or {}
            checkpoint_id = checkpoint.get("checkpoint_id")
            retry_count = result.get("retry_count")
        else:
            result = await dialogue_gen.generate_dialogue(
                context=context,
                preferred_type=preferred_type,
            )
    except Exception as e:  # noqa: BLE001
        error = _dialogue_gen_error_to_dict(e)
        if error is None:
            logger.error(f"[dialogue_gen] run error → task_id={task_id}: {e}", exc_info=True)
            error = {
                "error_code": ERR_NETWORK_ERROR,
                "error_detail": str(e)[:500],
                "failure_stage": STAGE_LLM,
                "llm_timeout_seconds": DIALOGUE_LLM_TIMEOUT_SECONDS,
                "raw": None,
            }
        # 失败也刷新断点游标（T4：failed 任务据此可续写）
        if DIALOGUE_GEN_GRAPH_ENABLED and DIALOGUE_GEN_CHECKPOINT_ENABLED:
            try:
                checkpoint_id = await dialogue_gen_graph.latest_checkpoint_id(db, task_id)
            except Exception:  # noqa: BLE001 — 游标刷新失败不掩盖原始错误
                logger.warning(
                    f"[dialogue_gen] 刷新 checkpoint 游标失败 → task_id={task_id}",
                    exc_info=True,
                )
    await finish_task(
        db,
        task_id,
        result=result,
        error=error,
        checkpoint_id=checkpoint_id,
        retry_count=retry_count,
    )
