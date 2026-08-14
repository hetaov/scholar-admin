"""进度/掌握度聚合模块 — Sentence → Lesson → Chapter → Book 逐级向上（Phase 4）

数据源全部由调用方传入（纯函数，不触网），便于单测与复现：
- `states`   ：学者的 skill_state 记录（可选已按 skill_code 过滤）
- `sentences`：内容层级 sentence_v2 记录（含 lesson_id / chapter_id）
- `lessons`  ：内容层级 lesson 记录（含 chapter_id）
- `chapters` ：内容层级 chapter 记录（含 textbook_id）

约定：
- 未指定 skill_code 时，同一句子多条 skill_state 取 progress 最高者（乐观聚合）。
- 无学习记录的句子按 progress=0 计入 total，保证分母与内容结构一致。
- 输出同时含三级结构（book/chapters/lessons）与兼容字段（units/sentences）。
"""

from __future__ import annotations

import logging
from typing import Any

from services.models_learning import (
    STATUS_LEARNED,
    STATUS_MASTERED,
    STATUS_NOT_STARTED,
    STATUS_LEARNING,
    STATUS_REVIEW_DUE,
    clamp,
    derive_progress,
)
from services.tracking_stats import format_duration

logger = logging.getLogger("scholar-admin.progress")

# 状态枚举顺序（mastery_distribution 固定输出键）
_DISTRIBUTION_KEYS = [
    STATUS_NOT_STARTED,
    STATUS_LEARNING,
    STATUS_LEARNED,
    STATUS_MASTERED,
    STATUS_REVIEW_DUE,
]

_LEARNED_STATUSES = {STATUS_LEARNED, STATUS_MASTERED}


# ---------------------------------------------------------------------------
# 纯函数：句子级
# ---------------------------------------------------------------------------


def sentence_progress(state: dict | None) -> float:
    """由 skill_state（mastery_score / status）得该句进度 0-1；无记录为 0。"""
    if not state:
        return 0.0
    return derive_progress(state.get("status"), state.get("mastery_score"))


def pick_state(states: list[dict], skill_code: str | None = None) -> dict | None:
    """从同一句子的多条 skill_state 中选一条用于聚合。

    - 指定 skill_code：取该能力的状态（无则 None）。
    - 未指定：取 progress 最高者（乐观聚合，表示该句已掌握的最好能力）。
    """
    if not states:
        return None
    if skill_code:
        for s in states:
            if s.get("skill_code") == skill_code:
                return s
        return None
    return max(states, key=sentence_progress)


def mastery_distribution(states: list[dict]) -> dict:
    """掌握度分布：各状态计数 + learned/mastered 占比。"""
    counts = {k: 0 for k in _DISTRIBUTION_KEYS}
    for s in states:
        st = s.get("status")
        if st in counts:
            counts[st] += 1
    total = sum(counts.values())
    learned = counts[STATUS_LEARNED]
    mastered = counts[STATUS_MASTERED]
    return {
        **counts,
        "total": total,
        "learned_count": learned,
        "mastered_count": mastered,
        "learned_pct": round(learned / total, 4) if total else 0.0,
        "mastered_pct": round(mastered / total, 4) if total else 0.0,
    }


def merge_distributions(distributions: list[dict]) -> dict:
    """合并多个 mastery_distribution（按状态键累加，百分比重算）。"""
    counts = {k: 0 for k in _DISTRIBUTION_KEYS}
    total = learned = mastered = 0
    for d in distributions:
        for k in _DISTRIBUTION_KEYS:
            counts[k] += int(d.get(k) or 0)
        total += int(d.get("total") or 0)
        learned += int(d.get("learned_count") or 0)
        mastered += int(d.get("mastered_count") or 0)
    return {
        **counts,
        "total": total,
        "learned_count": learned,
        "mastered_count": mastered,
        "learned_pct": round(learned / total, 4) if total else 0.0,
        "mastered_pct": round(mastered / total, 4) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# 纯函数：课 / 章 / 书 逐级聚合
# ---------------------------------------------------------------------------


def lesson_progress(lesson: dict, sentence_items: list[dict]) -> dict:
    """课内进度：sentence_items = [{"sentence_id", "state"|None}, ...]。

    total_sentence_count 以课内句子数为准（含未学）；无记录句子 progress 计 0。
    """
    total = len(sentence_items)
    states = [it["state"] for it in sentence_items if it.get("state")]
    learned = sum(1 for s in states if s.get("status") in _LEARNED_STATUSES)
    avg = round(sum(sentence_progress(s) for s in states) / total, 4) if total else 0.0
    return {
        "lesson_id": lesson.get("lesson_id"),
        "lesson_title": lesson.get("title", lesson.get("lesson_title", "")),
        "order": lesson.get("order"),
        "progress": avg,
        "learned_sentence_count": learned,
        "total_sentence_count": total,
        "mastery_distribution": mastery_distribution(states),
    }


def chapter_progress(chapter: dict, lesson_items: list[dict]) -> dict:
    """章内进度：按课内句子数加权平均。"""
    total_sentences = sum(l["total_sentence_count"] for l in lesson_items)
    learned = sum(l["learned_sentence_count"] for l in lesson_items)
    if total_sentences:
        progress = round(
            sum(l["progress"] * l["total_sentence_count"] for l in lesson_items)
            / total_sentences,
            4,
        )
    else:
        progress = 0.0
    return {
        "chapter_id": chapter.get("chapter_id"),
        "chapter_title": chapter.get("title", chapter.get("chapter_title", "")),
        "order": chapter.get("order"),
        "progress": progress,
        "learned_sentence_count": learned,
        "total_sentence_count": total_sentences,
        "lesson_count": len(lesson_items),
        "mastery_distribution": merge_distributions(
            [l["mastery_distribution"] for l in lesson_items]
        ),
        "lessons": lesson_items,
    }


def book_progress(chapter_items: list[dict]) -> dict:
    """全书进度：按章内句子数加权平均。"""
    total_sentences = sum(c["total_sentence_count"] for c in chapter_items)
    learned = sum(c["learned_sentence_count"] for c in chapter_items)
    if total_sentences:
        progress = round(
            sum(c["progress"] * c["total_sentence_count"] for c in chapter_items)
            / total_sentences,
            4,
        )
    else:
        progress = 0.0
    return {
        "progress": progress,
        "learned_sentence_count": learned,
        "total_sentence_count": total_sentences,
        "chapter_count": len(chapter_items),
        "lesson_count": sum(c["lesson_count"] for c in chapter_items),
        "mastery_distribution": merge_distributions(
            [c["mastery_distribution"] for c in chapter_items]
        ),
    }


def book_progress_from_lessons(lesson_items: list[dict]) -> dict:
    """无章教材全书进度：lesson 直接挂在 book 下, 按课内句子数加权平均。"""
    total_sentences = sum(l["total_sentence_count"] for l in lesson_items)
    learned = sum(l["learned_sentence_count"] for l in lesson_items)
    if total_sentences:
        progress = round(
            sum(l["progress"] * l["total_sentence_count"] for l in lesson_items)
            / total_sentences,
            4,
        )
    else:
        progress = 0.0
    return {
        "progress": progress,
        "learned_sentence_count": learned,
        "total_sentence_count": total_sentences,
        "chapter_count": 0,
        "lesson_count": len(lesson_items),
        "mastery_distribution": merge_distributions(
            [l["mastery_distribution"] for l in lesson_items]
        ),
    }


# ---------------------------------------------------------------------------
# 学习时长聚合
# ---------------------------------------------------------------------------


def sum_time_spent(attempts: list[dict]) -> float:
    """study_attempt.time_spent 求和（秒）；缺省/非法值按 0。"""
    total = 0.0
    for a in attempts:
        try:
            v = float(a.get("time_spent") or 0)
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            total += v
    return round(total, 2)


# ---------------------------------------------------------------------------
# 顶层聚合（组装三级结构）
# ---------------------------------------------------------------------------


def aggregate_progress(
    *,
    scholar_id: str,
    textbook_id: str,
    states: list[dict],
    sentences: list[dict],
    lessons: list[dict],
    chapters: list[dict],
    skill_code: str | None = None,
    attempts: list[dict] | None = None,
    detail: str = "full",
) -> dict:
    """聚合全书进度。

    detail 控制返回粒度（默认 full，保持旧契约不变）：
    - "full":    summary + chapters + 平铺 lessons/units/sentences
    - "chapter": summary + chapters（含内嵌 lessons），省略平铺字段
    - "overview": summary + 章级列表（不含课明细）
    - "lesson":  summary + 课级统计列表（不含章节/句子明细），tracking/stats 默认形态
    - "summary": 仅 summary（教材列表场景，省去层级组装）

    无章教材（chapters 为空，lesson 直挂 book）：chapter/overview 粒度下
    chapters 恒为空数组，改由顶层 lessons 返回课级进度列表（不含句子明细），
    保证总览页/章节页仍能拿到层级结构。
    """
    # 1. 句子级：把 skill_state 按 sentence_id 分组并 pick
    states_by_sentence: dict[str, list[dict]] = {}
    for st in states:
        states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)
    sentence_state: dict[str, dict | None] = {
        sid: pick_state(s_list, skill_code)
        for sid, s_list in states_by_sentence.items()
    }

    # 2. 课级：为每个 lesson 组装其句子 items
    lesson_by_id = {l.get("lesson_id"): l for l in lessons}
    sentence_items_by_lesson: dict[str, list[dict]] = {}
    for s in sentences:
        lid = s.get("lesson_id")
        sentence_items_by_lesson.setdefault(lid, []).append(
            {"sentence_id": s.get("sentence_id"), "state": sentence_state.get(s.get("sentence_id"))}
        )

    lesson_items_by_id: dict[str, dict] = {}
    for lid, items in sentence_items_by_lesson.items():
        lesson = lesson_by_id.get(lid, {"lesson_id": lid, "title": "", "order": None})
        lesson_items_by_id[lid] = lesson_progress(lesson, items)

    # 3. 章级：按 chapter 分组 lessons；无章教材(无 chapter)则课直接挂 book 下
    chapter_items: list[dict] = []
    if chapters:
        for chapter in chapters:
            cid = chapter.get("chapter_id")
            own_lessons = [
                lesson_items_by_id[l.get("lesson_id")]
                for l in lessons
                if l.get("chapter_id") == cid and l.get("lesson_id") in lesson_items_by_id
            ]
            chapter_items.append(chapter_progress(chapter, own_lessons))
        book = book_progress(chapter_items)
    else:
        book = book_progress_from_lessons(list(lesson_items_by_id.values()))
    total_time_spent = sum_time_spent(attempts or [])

    summary = {
        "total_time_spent": total_time_spent,
        "total_time_spent_display": format_duration(int(total_time_spent)),
        "textbook_progress": book["progress"],
        "learned_sentence_count": book["learned_sentence_count"],
        "total_sentence_count": book["total_sentence_count"],
        "chapter_count": book["chapter_count"],
        "lesson_count": book["lesson_count"],
        "mastery_distribution": book["mastery_distribution"],
        "learned_sentence_count_all_skills": sum(
            1 for s in states if s.get("status") in _LEARNED_STATUSES
        ),
    }

    result: dict = {
        "scholar_id": scholar_id,
        "text_book_id": textbook_id,
        "skill_code": skill_code,
        "summary": summary,
    }

    # 5. 层级列表：
    #    - "lesson": 仅 summary + 课级统计列表（无章节/句子明细）
    #    - "full"/"chapter": chapters（含内嵌 lessons）
    #    - "overview": 章级列表（剥离 lessons）
    #    - 无章教材（chapters 为空）在 chapter/overview 下由顶层 lessons 承载课级进度
    if detail == "lesson":
        result["lessons"] = list(lesson_items_by_id.values())
    elif detail in ("full", "chapter"):
        result["chapters"] = chapter_items
    elif detail == "overview":
        result["chapters"] = [
            {k: v for k, v in c.items() if k != "lessons"} for c in chapter_items
        ]
    if not chapters and detail in ("chapter", "overview"):
        result["lessons"] = list(lesson_items_by_id.values())
    if detail == "full":
        flat_lessons = [
            {**l, "unit_id": l["lesson_id"], "unit_title": l["lesson_title"]}
            for l in lesson_items_by_id.values()
        ]
        result["lessons"] = flat_lessons
        result["units"] = flat_lessons
        result["sentences"] = [
            {
                "sentence_id": s.get("sentence_id"),
                "lesson_id": s.get("lesson_id"),
                "chapter_id": s.get("chapter_id"),
                "learned": bool(sentence_state.get(s.get("sentence_id"))),
                "progress": sentence_progress(sentence_state.get(s.get("sentence_id"))),
                "skill_code": (sentence_state.get(s.get("sentence_id")) or {}).get("skill_code"),
            }
            for s in sentences
        ]
    return result
