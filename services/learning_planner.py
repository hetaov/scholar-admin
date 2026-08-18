"""S4.3 AI Planner 决策核心（设计文档 §27/§28 / P2-3）

- `build_plan`：纯函数，基于 LearningContext 输出下一次学习动作。
  零外部依赖（建议层，LLM 可选注入 P2 后续）。
- `generate_learning_plan`：闭环接线——build_learning_context → build_plan →
  幂等落库 learning_plan（scholar_id + plan_date，upsert 覆盖）。
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from services.cold_start import COLD_START_SEQUENCE
from services.learning_context import build_learning_context

LEARNING_PLAN = "learning_plan"


def build_plan(
    context: dict,
    *,
    top_review: int | None = None,
    top_activities: int | None = None,
) -> dict:
    """输出下一次学习动作：strategy / review_items / activities / difficulty / rationale。

    - strategy 四态：
      - cold_start：无历史（回退标准引导序列）
      - review：有到期复习项（优先于新内容）
      - weakness：无到期复习但存在弱项（mastery < 0.6 的能力）
      - practice：无到期复习且无显著弱项（保持常规节奏）
    - review_items：context.reviewItems（top_review 截断）
    - activities：review/weakness/practice 态取 context.activityType
      （top_activities 截断）；cold_start 态回退 COLD_START_SEQUENCE
    - difficulty：context.difficulty（复用难度建议）
    - rationale：推荐理由（弱项/到期/冷启动原因）
    """
    learner = context.get("learner") or {}
    review_items = context.get("reviewItems") or []
    weak_skills = context.get("weakSkills") or []
    activities = context.get("activityType") or list(COLD_START_SEQUENCE)
    difficulty = context.get("difficulty")

    if learner.get("has_history") is False:
        strategy = "cold_start"
        act = list(COLD_START_SEQUENCE)
        rationale = "无学习历史（冷启动），建议按标准引导序列先熟悉内容与基础训练。"
    elif review_items:
        strategy = "review"
        act = activities[:top_activities] if top_activities else activities[:]
        rationale = (
            f"有 {len(review_items)} 个到期复习项（弱项驱动优先），"
            "建议先完成复习再进入新内容。"
        )
    elif weak_skills:
        strategy = "weakness"
        act = activities[:top_activities] if top_activities else activities[:]
        rationale = f"存在薄弱能力（{'、'.join(weak_skills)}），建议针对性练习这些能力。"
    else:
        strategy = "practice"
        act = activities[:top_activities] if top_activities else activities[:]
        rationale = "当前无到期复习且无显著弱项，建议保持常规练习节奏。"
    # 复习态截断
    review_out = review_items[:top_review] if top_review else review_items

    return {
        "strategy": strategy,
        "review_items": review_out,
        "activities": act,
        "difficulty": difficulty,
        "rationale": rationale,
    }


async def generate_learning_plan(
    db,
    *,
    scholar_id: str,
    date: str | None = None,
    top_review: int | None = None,
    top_activities: int | None = None,
) -> dict:
    """build_learning_context → build_plan → 幂等落库 learning_plan（scholar_id + plan_date）。

    消费 end_session_with_summary 产出的 review_schedule / skill_state 更新
    （会话结束 → 下次计划含新到期复习项），形成"学习→记录→调度→再学习"闭环。
    失败仅告警不阻断（建议层，不抛异常）。
    """
    plan_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    context = await build_learning_context(
        db, scholar_id=scholar_id, date=plan_date, top_review=top_review
    )
    plan = build_plan(
        context, top_review=top_review, top_activities=top_activities,
    )
    now_ms = int(time.time() * 1000)
    doc_id = f"{scholar_id}_{plan_date}_plan"
    result = await db.update(
        collection=LEARNING_PLAN,
        where={"_id": doc_id},
        data={
            "$set": {
                "scholar_id": scholar_id,
                "plan_date": plan_date,
                "strategy": plan["strategy"],
                "review_items": plan["review_items"],
                "activities": plan["activities"],
                "difficulty": plan["difficulty"],
                "rationale": plan["rationale"],
                "updated_at": now_ms,
            }
        },
        upsert=True,
        multi=False,
    )
    # upsert 新文档时补 created_at（首次写入）；已存在则保留原 created_at
    if result.get("upserted_id"):
        await db.update(
            collection=LEARNING_PLAN,
            where={"_id": doc_id},
            data={"$set": {"created_at": now_ms}},
            multi=False,
        )
    return {
        "plan": plan,
        "plan_date": plan_date,
        "doc_id": doc_id,
    }
