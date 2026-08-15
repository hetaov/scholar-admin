"""对话匹配异步任务模型 — `dialogue_task` 集合 CRUD 与状态流转

状态机（单向，禁止回退）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

- create_task      : 生成 `task_id`，插入 pending 任务，返回任务文档
- claim_task       : 原子抢占 pending → processing（multi=False + modified_count>0 判成功）
- finish_task      : processing → success(+result) | failed(+error)
- get_task         : 按 task_id 查询（不过滤 TTL，接口层自行过滤）
- cleanup_expired  : 删除 expires_at <= now 的任务（TTL 清理，Phase 4 巡检用）

任务记录是唯一可靠状态源（不依赖进程内存），容器回收/重启后状态不丢。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from services.dependencies import get_db
from services.dialogue import load_learned_sentences, match_dialogue

logger = logging.getLogger("scholar-admin.dialogue_task")

COLLECTION = "dialogue_task"

# 任务默认保留时长：24h，保证客户端轮询窗口(30s)+容错重试绰绰有余
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`dt_` + 32 位 uuid hex。"""
    return "dt_" + uuid.uuid4().hex


async def create_task(db, *, scholar_id: str, sentence: str) -> dict:
    """创建 pending 任务并落库，返回任务文档。

    不做任何 LLM 调用，保证调用方（提交接口）耗时毫秒级。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": build_task_id(),
        "scholar_id": scholar_id,
        "sentence": sentence,
        "status": STATUS_PENDING,
        "result": None,
        "is_question": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[task] create → task_id={task_doc['task_id']}, scholar={scholar_id}"
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


async def finish_task(
    db,
    task_id: str,
    *,
    result: dict | list | None = None,
    is_question: bool | None = None,
    error: str | None = None,
) -> None:
    """写回执行结果：error 非空 → failed(+error, result 置 null)，否则 success(+result)。"""
    if error is not None:
        status = STATUS_FAILED
        result_value = None
        error_value = str(error)
        logger.info(f"[task] fail → task_id={task_id}, error={error_value[:200]}")
    else:
        status = STATUS_SUCCESS
        result_value = result
        error_value = None
        logger.info(f"[task] done → task_id={task_id}, status=success")
    await db.update(
        COLLECTION,
        where={"task_id": task_id},
        data={
            "$set": {
                "status": status,
                "result": result_value,
                "is_question": is_question,
                "error": error_value,
                "updated_at": _now_ms(),
            }
        },
        multi=False,
    )


async def get_task(db, task_id: str) -> dict | None:
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期任务，返回删除数量。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[task] cleanup → 删除过期任务 {count} 条")
    return count


async def recover_stale_tasks(db, timeout_s: int = 120) -> int:
    """巡检卡死的 processing 任务：updated_at 超过 timeout_s 未更新 → 置为 failed。

    兜底容器回收 / 进程崩溃导致的 processing 卡死（任务记录是唯一状态源，
    实例挂了没有 else 分支写 failed，必须靠巡检恢复）。

    Returns:
        修复（置为 failed）的任务数量
    """
    now = _now_ms()
    threshold = now - timeout_s * 1000
    res = await db.update(
        COLLECTION,
        where={
            "status": STATUS_PROCESSING,
            "updated_at": {"$lt": threshold},
        },
        data={
            "$set": {
                "status": STATUS_FAILED,
                "error": "执行超时",
                "updated_at": now,
            }
        },
        multi=True,
    )
    count = res.get("modified_count", 0)
    if count:
        logger.info(f"[task] recover → 卡死任务标记 failed {count} 条")
    return count


async def run_dialogue_task(task_id: str, scholar_id: str, sentence: str) -> None:
    """后台执行对话匹配任务并写回结果。

    由提交接口 `asyncio.create_task(...)` 调度，与请求解耦：
    - claim_task 原子抢占：被其他实例抢占则直接返回，避免重复执行
    - 加载已学语句 → 执行 LangGraph 匹配 → finish_task 写回
    - 业务失败（无已学语句/匹配失败）与异常统一走 failed 分支，绝不抛到调度方
    """
    db = get_db()
    if not await claim_task(db, task_id):
        logger.info(f"[task] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return
    try:
        learned = await load_learned_sentences(db, scholar_id)
        if not learned:
            await finish_task(db, task_id, error="该学者暂无已学语句")
            return

        result = await match_dialogue(
            input_sentence=sentence,
            scholar_id=scholar_id,
            learned_sentences=learned,
        )
        if result.get("success"):
            await finish_task(
                db,
                task_id,
                result=result.get("data"),
                is_question=result.get("is_question"),
            )
        else:
            await finish_task(
                db,
                task_id,
                error=result.get("error") or "对话匹配失败",
            )
    except Exception as e:  # noqa: BLE001
        logger.error(
            f"[task] run error → task_id={task_id}: {e}",
            exc_info=True,
        )
        await finish_task(db, task_id, error=f"对话匹配失败: {str(e)[:500]}")
