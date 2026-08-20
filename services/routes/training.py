"""P0 Training 受控任务接口（契约 api-contract §3.9）

GET  /training/recommend — 训练 Tab 推荐（S3.3）：前置评估 → 推荐 Activities + 能力状态 + 策略标注
POST /training/exercise — 生成受控任务（弱项驱动，§9-4）
POST /training/evaluate — 提交作答，返回逐题判定 + 即时反馈（§9-4）

原则：
- 弱项驱动（§3.2）：从 skill_state 取该 skill 最弱句；无历史（冷启动 §5.6）回退
  标准引导序列（content → shadowing → translation → listening），不报错不阻断（§9-9）。
- 结果并入 `learning_attempt`（mode='training'，附录 B-2），不建独立 training_session。
- 逐题判定（§3.3）：translation 类走 evaluate_text（LLM + Rule）；低置信（< 阈值）不回写
  SkillState（§9-2）；门控回写复用 upsert_skill_state + record_attempt。
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from services.cold_start import cold_start_flag, cold_start_prior, has_skill_history
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.evaluation_engine import evaluate_text
from services.events import record_attempt
from services.models_content import get_sentences_by_ids, query_all_pages
from services.models_conversation import DEFAULT_SKILL_CODE
from services.models_learning import (
    SKILL_STATE,
    get_skill_states,
    upsert_skill_state,
)
from services.models_training import (
    COLD_START_SEQUENCE,
    LOW_CONFIDENCE_THRESHOLD,
    TASK_ITEM_LIMIT,
    apply_results_to_task,
    build_item,
    build_training_task,
    create_task,
    evaluate_item,
    get_task,
    update_task,
)
from services.pre_assessment import aggregate_mastery, pre_assess

logger = logging.getLogger("scholar-admin.routes.training")
router = APIRouter(tags=["training"])


class TrainResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


class ExerciseRequest(BaseModel):
    scholar_id: Optional[str] = Field(None, description="必填（业务层校验）")
    skill_code: Optional[str] = Field(None, description="必填，如 translation / listening")
    difficulty: Optional[int] = Field(None, description="缺省取该 skill 当前档位（冷启动=1）")


class AnswerItem(BaseModel):
    item_id: str
    response: str


class EvaluateRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="必填（业务层校验）")
    answers: list[AnswerItem] = Field(default_factory=list, description="逐题作答")


# ---------------------------------------------------------------------------
# 弱项驱动选句
# ---------------------------------------------------------------------------


async def _pick_weak_sentences(
    db: CloudBaseNoSQLClient,
    scholar_id: str,
    skill_code: str,
    limit: int = TASK_ITEM_LIMIT,
) -> list[dict]:
    """按弱项驱动选句（§3.2 Select Sentence）。

    候选：该学者 + 该 skill 的 skill_state，按 mastery_score 升序取最弱 N 句；
    无该 skill 状态（冷启动）→ 标准引导序列任意句（先验默认，不阻断 §9-9）。
    """
    states = await query_all_pages(
        db,
        collection=SKILL_STATE,
        where={"scholar_id": scholar_id, "skill_code": skill_code},
    )
    if states:
        states.sort(key=lambda s: int(s.get("mastery_score") or 0))
        sentence_ids = [s.get("sentence_id") for s in states[:limit]]
        sentences = await get_sentences_by_ids(db, sentence_ids)
        by_id = {s.get("sentence_id"): s for s in sentences}
        return [by_id[sid] for sid in sentence_ids if sid in by_id]

    # 冷启动：标准引导序列（content → shadowing → translation → listening）——
    # P0 从任意已见句兜底，无历史直接返回空任务体（接口不报错，见 §9-9）。
    logger.info(
        "[training/exercise] 冷启动：scholar_id=%s 无 %s 状态，回退先验默认",
        scholar_id, skill_code,
    )
    return []


def _activity_for_skill(skill_code: str) -> str:
    """skill → 活动类型映射（§3.3 Evaluator 映射）。"""
    mapping = {
        "translation": "TRANSLATION",
        "pronunciation": "SHADOWING",
        "fluency": "SHADOWING",
        "listening": "LISTENING",
        "comprehension": "LISTENING",
        "vocabulary": "TRANSLATION",
        "grammar": "TRANSLATION",
    }
    return mapping.get(str(skill_code or "").lower(), "TRANSLATION")


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


def _skill_states_from_records(records: list[dict]) -> list[dict]:
    """把 skill_state 记录组装为能力状态列表（S3.3，训练 Tab 展示）。

    每能力一项：mastery（0~1，均值）、attempt_count（总尝试）、difficulty（历史均值）。
    """
    grouped: dict[str, list[dict]] = {}
    for record in records:
        code = str(record.get("skill_code") or DEFAULT_SKILL_CODE)
        grouped.setdefault(code, []).append(record)
    mastery_by_skill = aggregate_mastery(records)
    states: list[dict] = []
    for code, recs in grouped.items():
        attempts = sum(int(r.get("attempt_count") or 0) for r in recs)
        difficulties = [
            float(d)
            for r in recs
            if (d := r.get("difficulty")) is not None and str(d) != ""
        ]
        states.append(
            {
                "skill_code": code,
                "mastery": mastery_by_skill.get(code),
                "attempt_count": attempts,
                "difficulty": round(sum(difficulties) / len(difficulties), 2) if difficulties else None,
            }
        )
    states.sort(key=lambda s: (s["mastery"] if s["mastery"] is not None else 1.0))
    return states


@router.get("/training/recommend", response_model=TrainResponse)
async def training_recommend(
    scholar_id: Optional[str] = None,
    sentence_id: Optional[str] = None,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> TrainResponse:
    """训练 Tab 推荐（S3.3）：复用前置评估生成推荐 Activities + 能力状态 + 策略标注。

    - strategy："cold_start"（无历史 / 证据稀疏，回退标准引导序列）| "weakness"（弱项驱动）
    - activities：推荐 Activities 列表（小写名，供前端映射训练模式）
    - skill_states：能力状态列表（mastery / attempt_count / difficulty）
    """
    scholar_id = str(scholar_id or "").strip()
    if not scholar_id:
        return TrainResponse(
            success=False, code="INVALID_INPUT", message="缺少参数 scholar_id"
        )

    assess = await pre_assess(db, scholar_id=scholar_id, sentence_id=sentence_id)
    result = await get_skill_states(
        db, scholar_id=scholar_id, sentence_id=sentence_id
    )
    records = result.get("records") or []
    strategy = (
        "cold_start"
        if (not assess["has_history"] or assess["evidence_sparse"])
        else "weakness"
    )

    logger.info(
        "[training/recommend] scholar_id=%s strategy=%s activities=%s",
        scholar_id, strategy, assess["activity_recommendation"],
    )
    return TrainResponse(
        success=True,
        data={
            "strategy": strategy,
            "has_history": assess["has_history"],
            "gate_suggestion": assess["gate_suggestion"],
            "difficulty": assess["difficulty"],
            "mastery": assess["mastery"],
            "evidence_sparse": assess["evidence_sparse"],
            "activities": assess["activity_recommendation"],
            "skill_states": _skill_states_from_records(records),
        },
    )


@router.post("/training/exercise", response_model=TrainResponse)
async def training_exercise(
    data: ExerciseRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> TrainResponse:
    """生成受控任务：弱项驱动选句 + 任务落库（learning_attempt mode='training'）。"""
    scholar_id = str(data.scholar_id or "").strip()
    skill_code = str(data.skill_code or "").strip()
    if not scholar_id or not skill_code:
        return TrainResponse(
            success=False, code="INVALID_INPUT", message="缺少参数 scholar_id / skill_code"
        )

    difficulty = int(data.difficulty or 1)
    sentences = await _pick_weak_sentences(db, scholar_id, skill_code)
    activity = _activity_for_skill(skill_code)

    items = []
    for i, s in enumerate(sentences[:TASK_ITEM_LIMIT]):
        items.append(build_item(i, s, activity))

    task = build_training_task(
        scholar_id=scholar_id,
        skill_code=skill_code,
        difficulty=difficulty,
        items=items,
    )
    task = await create_task(db, task)

    # 冷启动标记（§5.6.5）：无该 skill 历史 → "cold_start": true + 先验默认
    has_history = await has_skill_history(db, scholar_id=scholar_id, skill_code=skill_code)
    is_cold = cold_start_flag(has_history=has_history)
    prior = cold_start_prior(difficulty=difficulty)

    logger.info(
        "[training/exercise] scholar_id=%s skill_code=%s items=%d cold_start=%s",
        scholar_id, skill_code, len(items), is_cold,
    )
    return TrainResponse(
        success=True,
        data={
            "task_id": task["task_id"],
            "items": items,
            "difficulty": difficulty,
            "cold_start": is_cold,
            "prior_defaults": prior,
        },
    )


@router.post("/training/evaluate", response_model=TrainResponse)
async def training_evaluate(
    data: EvaluateRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> TrainResponse:
    """提交作答：逐题判定 + 即时反馈；门控回写 SkillState + record_attempt。"""
    task_id = str(data.task_id or "").strip()
    if not task_id:
        return TrainResponse(
            success=False, code="INVALID_INPUT", message="缺少参数 task_id"
        )

    task = await get_task(db, task_id)
    if task is None:
        return TrainResponse(success=False, code="NOT_FOUND", message="任务不存在")
    if task.get("attempt_status") not in ("pending",):
        return TrainResponse(
            success=False, code="CONFLICT", message="任务已提交，不可重复评估"
        )

    answers = {a.item_id: a.response for a in data.answers}
    results = []
    for item in task.get("items", []):
        response = answers.get(item.get("item_id")) or ""
        results.append(evaluate_item(item, response, evaluate_text))

    task = apply_results_to_task(task, results)
    await update_task(db, task)

    # 门控回写（§9-2/§9-4）：正确且高置信才更新 SkillState；正确则记 attempt 事件
    for item in task.get("items", []):
        verdict = {
            "score": item.get("score") or 0,
            "meaningful": item.get("correct", False),
            "confidence": item.get("confidence") or 0.0,
        }
        sentence_id = item.get("sentence_id")
        if not sentence_id:
            continue
        if not verdict["meaningful"]:
            continue
        if (verdict["confidence"] or 0.0) < LOW_CONFIDENCE_THRESHOLD:
            continue  # 低置信不回写（§9-2）
        await upsert_skill_state(
            db,
            scholar_id=task["scholar_id"],
            sentence_id=sentence_id,
            skill_code=task["skill_code"],
            score=verdict["score"],
            sparse_discount=True,  # 证据稀疏保护（§5.6.2）：<3 次尝试更新量打折
            confidence=verdict["confidence"],  # S3.1 P1：置信度均值更新（§5.6.2）
            outcome="success" if verdict["score"] >= 60 else "fail",  # 稳定性方向
            difficulty=task.get("difficulty"),  # 当前难度档位
        )
        await record_attempt(
            db,
            scholar_id=task["scholar_id"],
            sentence_id=sentence_id,
            skill_code=task["skill_code"],
            attempt_type="training",
            status="correct" if verdict["score"] >= 60 else "incorrect",
            score=verdict["score"],
        )

    logger.info(
        "[training/evaluate] task_id=%s overall=%s", task_id, task.get("overall"),
    )
    return TrainResponse(
        success=True,
        data={
            "results": [
                {
                    "item_id": r["item_id"],
                    "correct": r["correct"],
                    "feedback": r["feedback"],
                    "score": r["score"],
                    "confidence": r["confidence"],
                }
                for r in results
            ],
            "overall": task.get("overall"),
        },
    )
