"""复习推荐异步任务模型 — `review_recommend_task` 集合 CRUD 与状态流转（v4 R4 第二段）

状态机（单向，禁止回退，口径同 `translation_task`）：
    pending ──(claim_task 原子抢占)──> processing ──┬──> success (+result)
                                                    └──> failed  (+error)

- create_review_recommend_task : 生成 `task_id`（`rr_` 前缀），插入 pending 任务，返回任务文档
- claim_task                   : 原子抢占 pending → processing（multi=False + modified_count>0 判成功）
- finish_task                  : processing → success(+result) | failed(+error)
- get_task                     : 按 task_id 查询（不过滤 TTL，接口层自行过滤）
- cleanup_expired              : 删除 expires_at <= now 的任务（TTL 清理）
- run_review_recommend_task    : 后台执行器：claim → LLM（wait_for 超时）→ **强校验** → finish

设计口径（`docs_v1/技能重构/技能逻辑重构-v4-完整设计文档.md` §3.5 A1~A6）：
- **候选集由规则层给出**（入参原样落库，供执行器构造提示词与校验返回值）；
- LLM 只排序 + 写 `reason_text`；返回值必须严格覆盖候选集（越界/重复/缺项 → failed）；
- 失败不降级（不产出部分结果）——**调用方（小程序）回退规则排序**，规则排序始终是基线。
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS
from services.dependencies import get_db
from services.providers.review_recommend import (
    ERR_LLM_UNAVAILABLE,
    ERR_INVALID_OUTPUT,
    STAGE_LLM,
    STAGE_PARSE,
    ReviewRecommendError,
    build_review_recommend_messages,
    call_review_recommend_llm,
    parse_review_recommend_output,
)

logger = logging.getLogger("scholar-admin.review_recommend_task")

COLLECTION = "review_recommend_task"

# 任务默认保留时长：24h（与 translation_task 一致，保证客户端轮询窗口 + 容错重试）
TASK_TTL_MS = 24 * 60 * 60 * 1000

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_task_id() -> str:
    """生成业务任务 ID：`rr_` + 32 位 uuid hex。"""
    return "rr_" + uuid.uuid4().hex


async def create_review_recommend_task(
    db,
    *,
    scholar_id: str,
    candidates: list[dict],
) -> dict:
    """创建 pending 任务并落库（不做任何 LLM 调用，提交接口毫秒级返回）。

    字段：`task_id / scholar_id / candidates / status / result / error / created_at / updated_at / expires_at`。
    """
    now = _now_ms()
    task_doc: dict[str, Any] = {
        "task_id": build_task_id(),
        "scholar_id": scholar_id,
        "candidates": list(candidates or []),
        "status": STATUS_PENDING,
        "result": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    await db.insert(COLLECTION, task_doc)
    logger.info(
        f"[review_recommend] create → task_id={task_doc['task_id']}, "
        f"scholar={scholar_id}, candidates={len(task_doc['candidates'])}"
    )
    return task_doc


async def claim_task(db, task_id: str) -> bool:
    """原子抢占 pending → processing（并发安全：where 限定 status=pending + multi=False）。"""
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
    result: dict | None = None,
    error: dict | None = None,
) -> None:
    """写回执行结果：error 非空 → failed（result 置 null），否则 success（error 置 null）。"""
    if error is not None:
        status, result_value, error_value = STATUS_FAILED, None, error
        logger.info(f"[review_recommend] fail → task_id={task_id}, error={error.get('error_code')}")
    else:
        status, result_value, error_value = STATUS_SUCCESS, result, None
        logger.info(f"[review_recommend] done → task_id={task_id}")
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
    """按 task_id 查询任务，未命中返回 None（不过滤 TTL）。"""
    res = await db.query(COLLECTION, where={"task_id": task_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的任务，返回删除条数（TTL 清理）。"""
    now = now_ms if isinstance(now_ms, int) else _now_ms()
    res = await db.delete(
        COLLECTION,
        where={"expires_at": {"$lte": now}},
        multi=True,
    )
    return int(res.get("deleted_count", 0) or 0)


async def run_review_recommend_task(task_id: str) -> None:
    """后台执行器（由提交接口 `asyncio.create_task` 调度）：

    claim 抢占 → 构造提示词（候选集来自任务文档）→ LLM（`asyncio.wait_for` 超时）→
    **强校验返回值**（越界/重复/缺项 → failed）→ finish_task。

    **失败不降级**：不产出部分结果；小程序侧保持规则排序（A5）。
    """
    db = get_db()
    if not await claim_task(db, task_id):
        logger.info(f"[review_recommend] run skip → task_id={task_id} 已被抢占或状态非 pending")
        return
    result: dict | None = None
    error: dict | None = None
    try:
        task = await get_task(db, task_id)
        candidates = (task or {}).get("candidates") or []
        candidate_ids = [str(c.get("group_id") or "") for c in candidates if c.get("group_id")]
        if not candidate_ids:
            raise ReviewRecommendError(ERR_INVALID_OUTPUT, STAGE_PARSE, "候选集为空，无法生成推荐")
        messages = build_review_recommend_messages(candidates)
        content = await call_review_recommend_llm(messages)
        if content is None:
            raise ReviewRecommendError(ERR_LLM_UNAVAILABLE, STAGE_LLM, "LLM 调用失败（凭据缺失或后端错误）")
        recommendations = parse_review_recommend_output(content, candidate_ids)
        if recommendations is None:
            raise ReviewRecommendError(
                ERR_INVALID_OUTPUT, STAGE_PARSE, "模型输出非法（越界/重复/缺项或非 JSON）"
            )
        result = {
            "recommendations": recommendations,
            "strategy": "ai_rank",
            "generated_at": _now_ms(),
        }
    except ReviewRecommendError as e:
        error = e.to_dict(llm_timeout_seconds=REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS)
    except Exception as e:  # noqa: BLE001
        logger.error(f"[review_recommend] run error → task_id={task_id}: {e}", exc_info=True)
        error = {
            "error_code": "NETWORK_ERROR",
            "error_detail": str(e)[:500],
            "failure_stage": STAGE_LLM,
            "llm_timeout_seconds": REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS,
            "raw": None,
        }
    await finish_task(db, task_id, result=result, error=error)
