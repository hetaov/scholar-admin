"""S4.3 AI Planner 路由（契约 api-contract §3.9 第 9 个接口）

GET /planner/next-action — 下一次学习动作推荐（S4.3 AI Planner 闭环）

- 默认（PLANNER_ENABLED=1）：build_learning_context → build_plan →
  generate_learning_plan 幂等落库 learning_plan（scholar_id + plan_date）。
- 回退（PLANNER_ENABLED=0）：复用 S3.3 /training/recommend 同构响应
  （strategy / has_history / gate_suggestion / difficulty / mastery /
   evidence_sparse / activities / skill_states，无 review_items / rationale）。
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config import (
    PLANNER_ENABLED,
    PLANNER_TOP_ACTIVITIES,
    PLANNER_TOP_REVIEW_ITEMS,
)
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.learning_planner import generate_learning_plan
from services.pre_assessment import pre_assess
from services.models_learning import get_skill_states
from services.routes_training import _skill_states_from_records

logger = logging.getLogger("scholar-admin.routes.planner")

router = APIRouter(tags=["planner"])


class PlannerResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


@router.get("/planner/next-action", response_model=PlannerResponse)
async def planner_next_action(
    scholar_id: Optional[str] = None,
    date: Optional[str] = None,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> PlannerResponse:
    """AI Planner 下一次学习动作推荐。

    入参：scholar_id（必）、date（可选，YYYY-MM-DD，缺省今日）。
    出参 200：{ success, data: { next_action: { strategy, review_items,
    activities, difficulty, rationale } } }
    """
    scholar_id = str(scholar_id or "").strip()
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholar_id")

    if not PLANNER_ENABLED:
        # S3.3 回退：与 /training/recommend 同构响应（无 review_items / rationale）
        assess = await pre_assess(db, scholar_id=scholar_id)
        records = (
            (await get_skill_states(db, scholar_id=scholar_id)).get("records") or []
        )
        strategy = (
            "cold_start"
            if (not assess["has_history"] or assess["evidence_sparse"])
            else "weakness"
        )
        logger.info(
            "[planner/next-action] PLANNER_ENABLED=0 回退 S3.3 scholar_id=%s strategy=%s",
            scholar_id, strategy,
        )
        return PlannerResponse(
            success=True,
            data={
                "next_action": {
                    "strategy": strategy,
                    "has_history": assess["has_history"],
                    "gate_suggestion": assess["gate_suggestion"],
                    "difficulty": assess["difficulty"],
                    "mastery": assess["mastery"],
                    "evidence_sparse": assess["evidence_sparse"],
                    "activities": assess["activity_recommendation"],
                    "skill_states": _skill_states_from_records(records),
                }
            },
        )

    try:
        result = await generate_learning_plan(
            db,
            scholar_id=scholar_id,
            date=date,
            top_review=PLANNER_TOP_REVIEW_ITEMS,
            top_activities=PLANNER_TOP_ACTIVITIES,
        )
    except Exception as exc:  # pragma: no cover - 防御：建议层不阻断
        logger.exception(
            "[planner/next-action] 生成学习计划失败 scholar_id=%s", scholar_id
        )
        raise HTTPException(status_code=500, detail=f"学习计划生成失败: {exc}")

    logger.info(
        "[planner/next-action] scholar_id=%s plan_date=%s strategy=%s",
        scholar_id, result["plan_date"], result["plan"]["strategy"],
    )
    return PlannerResponse(
        success=True,
        data={
            "next_action": result["plan"],
            "plan_date": result["plan_date"],
        },
    )
