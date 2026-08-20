"""S4.3 / P2-2 LearningContextBuilder — 学习上下文组装（向量版，RAG Retriever 已接入）

设计文档 §2.1 统一流水线 / 草稿 §十九/§二十（LearningContext 结构）：
learner / book / lesson / sentences / learningObjectives / currentSkills /
weakSkills / recentAttempts / reviewItems / currentIntent / difficulty / activityType

- book / lesson / sentences：P2-2 RAG 已实施 —— 从最近 skill_state 推断当前学习位置，
  填充当前教材/课快照（Required knowledge）+ 跨课 RAG 召回句子（Optional knowledge）。
- learningObjectives：留待 P2-4 Sentence 富化。
- Retriever 为可替换基础设施（设计文档 §2.2 / ADR-0017）：默认 VolcanoEmbeddingRetriever，
  未配置或异常自动降级为空召回（不阻断 planner）。
"""
from __future__ import annotations

from config import MIN_EVIDENCE
from services.cold_start import COLD_START_SEQUENCE
from services.learning_scheduler import get_due_review_items
from services.models_content import (
    LESSON,
    get_sentences_by_lesson_ids,
    get_textbook_v2,
)
from services.models_learning import SKILL_STATE, get_skill_states
from services.pre_assessment import aggregate_mastery, pre_assess
from services.rag_retriever import CurriculumRetriever, get_curriculum_retriever


async def resolve_current_position(db, *, scholar_id: str) -> dict | None:
    """推断当前学习位置（最近 skill_state 的句子 → 课 → 教材）。

    Returns:
        {"lesson_id", "textbook_id", "lesson": 课快照, "textbook": 教材快照}
        或 None（无历史 / 无法定位，冷启动场景）。
    """
    result = await db.query(
        collection=SKILL_STATE,
        where={"scholar_id": scholar_id},
        order=[{"field": "updated_at", "direction": "desc"}],
        limit=1,
    )
    records = result.get("records") or []
    if not records:
        return None
    latest = records[0]
    lesson_id = latest.get("lesson_id")
    if not lesson_id:
        return None
    lesson_result = await db.query(
        collection=LESSON, where={"lesson_id": lesson_id}, limit=1
    )
    lesson_records = lesson_result.get("records") or []
    if not lesson_records:
        return None
    lesson = lesson_records[0]
    textbook_id = lesson.get("textbook_id")
    textbook = None
    if textbook_id:
        textbook = await get_textbook_v2(db, textbook_id)
    return {
        "lesson_id": lesson_id,
        "textbook_id": textbook_id,
        "lesson": lesson,
        "textbook": textbook,
    }


def _book_snapshot(textbook: dict | None) -> dict | None:
    """教材快照（LearningContext.book，草稿 §二十）。"""
    if not textbook:
        return None
    return {
        "textbook_id": textbook.get("textbook_id") or textbook.get("_id"),
        "title": textbook.get("title", ""),
        "grade": textbook.get("grade", ""),
        "level": textbook.get("level", ""),
    }


def _lesson_snapshot(lesson: dict) -> dict:
    """课快照（LearningContext.lesson，草稿 §二十）。"""
    return {
        "lesson_id": lesson.get("lesson_id") or lesson.get("_id"),
        "title": lesson.get("title", ""),
        "order": lesson.get("order"),
        "chapter_id": lesson.get("chapter_id", ""),
        "textbook_id": lesson.get("textbook_id", ""),
    }


def _sentence_snapshot(
    sentence: dict, *, source: str, score: float | None = None
) -> dict:
    """句子快照（LearningContext.sentences 元素）。

    source: "required"（当前课 Required knowledge）/ "optional"（跨课 RAG Optional knowledge）
    """
    snap = {
        "sentence_id": sentence.get("sentence_id") or sentence.get("_id"),
        "text": sentence.get("text", ""),
        "translation": sentence.get("translation", ""),
        "lesson_id": sentence.get("lesson_id", ""),
        "textbook_id": sentence.get("textbook_id", ""),
        "source": source,
    }
    if score is not None:
        snap["score"] = score
    return snap


def _build_query(lesson: dict, weak_skills: list[str]) -> str:
    """检索 query：当前课标题 + 弱项技能描述（对齐草稿 §十八 AI Training Context → Retriever）。"""
    parts = [lesson.get("title", "")] if lesson else []
    parts.extend(weak_skills or [])
    return " ".join(p for p in parts if p).strip()


async def build_learning_context(
    db,
    *,
    scholar_id: str,
    date: str | None = None,
    top_review: int | None = None,
    retriever: CurriculumRetriever | None = None,
) -> dict:
    """组装 LearningContext（skill_state 聚合 + 到期复习 + 前置评估 + RAG 课程上下文）。

    Args:
        retriever: 跨课召回器（可替换基础设施）。None → get_curriculum_retriever()
            （按配置：RAG_EMBEDDING_MODEL 未配置或开关关闭时自动降级 no-op）。

    返回 dict（与草稿 §二十 LearningContext 字段对齐）：
    - learner: scholar_id / has_history / evidence_sparse / cold_start
    - book / lesson / sentences: P2-2 RAG 已实施 —— 当前教材/课快照 + 当前课句子（required）
      与跨课召回句子（optional，附 score）
    - learningObjectives: 留待 P2-4 Sentence 富化
    - currentSkills / weakSkills: 按 skill_code 聚合的 mastery（aggregate_mastery）
    - recentAttempts: 近期 attempt 数（skill_state.attempt_count 求和）
    - reviewItems: get_due_review_items 到期复习项（P2-3 核心输入）
    - currentIntent: 由策略推断（review / weakness / practice）
    - difficulty / activityType: 复用 pre_assess 难度建议 / 弱项驱动 Activities
    """
    assess = await pre_assess(db, scholar_id=scholar_id)
    records = (await get_skill_states(db, scholar_id=scholar_id)).get("records") or []
    mastery_by_skill = aggregate_mastery(records)
    review_items = await get_due_review_items(
        db, scholar_id=scholar_id, date=date, limit=top_review
    )
    recent_attempts = sum(int(r.get("attempt_count") or 0) for r in records)

    if not records:
        # 无历史 → 先验默认（对齐 S2.4 cold_start_prior）
        has_history = False
        evidence_sparse = True
        current_skills = {}
        weak_skills = []
        current_intent = "cold_start"
        difficulty = assess["difficulty"]
        activities = list(COLD_START_SEQUENCE)
    else:
        has_history = True
        evidence_sparse = recent_attempts < MIN_EVIDENCE
        current_skills = dict(mastery_by_skill)
        weak_skills = [
            code
            for code, m in sorted(mastery_by_skill.items(), key=lambda kv: kv[1])[:3]
            if m < 0.6
        ]
        current_intent = (
            "review" if review_items else ("weakness" if weak_skills else "practice")
        )
        difficulty = assess["difficulty"]
        activities = assess["activity_recommendation"]

    # ---- P2-2 RAG：当前课（Required）+ 跨课召回（Optional），失败自动降级 ----
    book: dict | None = None
    lesson: dict | None = None
    sentences: list[dict] = []
    try:
        position = await resolve_current_position(db, scholar_id=scholar_id)
        if position and position.get("lesson"):
            lesson = _lesson_snapshot(position["lesson"])
            book = _book_snapshot(position.get("textbook"))
            lesson_sentences = await get_sentences_by_lesson_ids(
                db, [position["lesson_id"]]
            )
            sentences = [
                _sentence_snapshot(s, source="required")
                for s in lesson_sentences
                if s.get("text")
            ]
            # 跨课召回（Optional knowledge）：默认 retriever 按配置自动降级 no-op
            retriever = retriever or get_curriculum_retriever()
            optional = await retriever.retrieve(
                db,
                scholar_id=scholar_id,
                query=_build_query(lesson, weak_skills),
                exclude_lesson_ids=[position["lesson_id"]],
            )
            sentences.extend(
                _sentence_snapshot(s, source="optional", score=s.get("score"))
                for s in optional
                if s.get("sentence_id") and s.get("text")
            )
    except Exception:
        # 降级：不阻断 planner 主链路（book/lesson/sentences 保持占位）
        pass

    return {
        "learner": {
            "scholar_id": scholar_id,
            "has_history": has_history,
            "evidence_sparse": evidence_sparse,
            "cold_start": not has_history,
        },
        "book": book,
        "lesson": lesson,
        "sentences": sentences,
        "learningObjectives": [],  # P2-4 Sentence 富化
        "currentSkills": current_skills,
        "weakSkills": weak_skills,
        "recentAttempts": recent_attempts,
        "reviewItems": review_items,
        "currentIntent": current_intent,
        "difficulty": difficulty,
        "activityType": activities,
    }
