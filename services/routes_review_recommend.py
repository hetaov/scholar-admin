"""复习推荐路由（v4 R4 第二段：AI 排序与理由）

契约：`api-contract.md §3.16`
- `POST /ai/review-recommend`          提交（异步，毫秒级返回 task_id）
- `GET  /ai/review-recommend/task/{id}` 轮询结果（pending/processing/success/failed）

门控：仅在 `REVIEW_RECOMMEND_ENABLED=1` 时由 main.py 条件 include；开关关 → 路由不注册（404）。
职责边界（设计文档 §3.5 A1~A6 / D7）：候选集由规则层给出，LLM 只排序 + 写理由，**不得增删候选**；
失败不降级，调用方（小程序）回退规则排序（A5）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import REVIEW_RECOMMEND_MAX_CANDIDATES
from services.dependencies import get_db
from services.learning.review_recommend_task import (
    create_review_recommend_task,
    get_task,
    run_review_recommend_task,
)

logger = logging.getLogger("scholar-admin.routes.review_recommend")

router = APIRouter()

# 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消（同 eval.py）
_background_tasks: set[asyncio.Task] = set()


class ReviewRecommendRequest(BaseModel):
    """提交请求（契约 §3.16）。"""

    scholar_id: Optional[str] = None  # 必填（业务校验：空 → INVALID_INPUT）
    now: Optional[int] = None  # 可选：调用方参考时间（毫秒），仅作提示词素材
    candidates: Optional[list] = None  # 必填：1~20 项（规则层确定）


class ReviewRecommendResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


def _normalize_candidates(raw: Any) -> tuple[list[dict], str | None]:
    """裁剪/校验候选集：返回 (candidates, error_code)。

    仅保留含 `group_id` 的项（其余字段原样透传作提示词素材）；空 → INVALID_INPUT；
    超出上限 → 截断至 REVIEW_RECOMMEND_MAX_CANDIDATES（不报错，契约允许截断）。
    """
    if not isinstance(raw, list):
        return [], "INVALID_INPUT"
    items = [c for c in raw if isinstance(c, dict) and str(c.get("group_id") or "")]
    if not items:
        return [], "INVALID_INPUT"
    return items[:REVIEW_RECOMMEND_MAX_CANDIDATES], None


@router.post("/ai/review-recommend", response_model=ReviewRecommendResponse)
async def submit_review_recommend(
    body: ReviewRecommendRequest,
    db=Depends(get_db),
) -> ReviewRecommendResponse:
    """提交复习推荐任务（异步）：创建 pending 任务 → 后台执行 → 毫秒级返回 task_id。"""
    scholar_id = (body.scholar_id or "").strip()
    if not scholar_id:
        return ReviewRecommendResponse(
            success=False, code="INVALID_INPUT", message="scholar_id 不能为空"
        )
    candidates, err = _normalize_candidates(body.candidates)
    if err:
        return ReviewRecommendResponse(
            success=False, code="INVALID_INPUT", message="candidates 不能为空（每项需含 group_id）"
        )
    try:
        task = await create_review_recommend_task(
            db, scholar_id=scholar_id, candidates=candidates
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[review_recommend] 任务创建异常: {e}", exc_info=True)
        raise
    # 后台执行，不阻塞当前请求
    bg = asyncio.create_task(run_review_recommend_task(task["task_id"]))
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)
    logger.info(
        f"[review_recommend] 任务已提交 → task_id={task['task_id']}, "
        f"scholar={scholar_id}, candidates={len(candidates)}"
    )
    return ReviewRecommendResponse(
        success=True, data={"task_id": task["task_id"], "status": task["status"]}
    )


@router.get("/ai/review-recommend/task/{task_id}", response_model=ReviewRecommendResponse)
async def get_review_recommend_task(
    task_id: str,
    db=Depends(get_db),
) -> ReviewRecommendResponse:
    """查询复习推荐任务（pending/processing/success/failed）；过期/不存在 → TASK_NOT_FOUND。"""
    task = await get_task(db, task_id)
    if not task:
        return ReviewRecommendResponse(
            success=False, code="TASK_NOT_FOUND", message="任务不存在或已过期"
        )
    return ReviewRecommendResponse(
        success=True,
        data={
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "result": task.get("result"),
            "error": task.get("error"),
        },
    )
