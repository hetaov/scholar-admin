"""M3 G1.1 组视图服务 — getLessonSentenceGroups（service-contract §8.5）

组视图接口 `GET /tracking/textbooks/{tid}/lessons/{lid}/groups` 的服务函数：

- 校验 lesson 存在（不存在 → `LessonNotFoundError`，404 `LESSON_NOT_FOUND`）；
- 有 `sentence_group` → 按组组织返回（group_id / group_title / group_type /
  order_in_lesson / sentences[]）；
- 无 group → **读兼容层**：逐句构造临时组 `legacy_{lesson_id}_{sentence_id}`
  （组标题 = 语句 text 前 20 字，type = `stand_alone`），**返回结构逐字一致**，
  调用方零改动；
- 组内句子 status / skills / weakest_skill / review_count / next_review_at 口径
  与 `/sentences` 接口**逐字一致**（M3 skill_state 写入键零变化，仍按 sentence_id 独立聚合）；
- `is_canonical`：`canonical_sentence_id` 为 null / 自身 → True，否则 False；
- 纯读，不审。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from services.english import LessonNotFoundError
from services.models_content import (
    LESSON,
    get_sentence_groups_by_lesson,
    get_sentences_by_lesson,
    query_all_pages,
)
from services.models_learning import (
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_MASTERED,
)
from services.progress import (
    mastery_distribution,
    mastery_ratio,
    pick_state,
    status_to_int,
)

logger = logging.getLogger("scholar-admin.english.group_view")

# 能力全集（与 routes/tracking.py /sentences 接口口径一致）
_SKILL_CODES = ("translation", "conversation", "listening", "reading", "speaking")

_SELECT_STATE_FIELDS = {
    "scholar_id": 1,
    "sentence_id": 1,
    "skill_code": 1,
    "status": 1,
    "mastery_score": 1,
    "attempt_count": 1,
    "next_review_at": 1,
}


def _to_iso(timestamp) -> str | None:
    """int 秒级时间戳 → ISO 8601 UTC 字符串；空 / 非法返回 None。"""
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
    except Exception:
        return None


async def _find_lesson(db, *, textbook_id: str, lesson_id: str) -> dict:
    """查 lesson 内容表（v2），不存在抛 LessonNotFoundError。"""
    result = await db.query(
        collection=LESSON,
        where={"textbook_id": textbook_id, "lesson_id": lesson_id},
        limit=1,
    )
    records = result.get("records", [])
    if not records:
        raise LessonNotFoundError(lesson_id)
    return records[0]


async def _load_states_by_sentence(
    db, *, scholar_id: str, sentence_ids: list[str]
) -> list[dict]:
    """按学者 + 句子 $in 分批拉取 skill_state（与 /sentences 接口同口径）。"""
    ids = [sid for sid in sentence_ids if sid]
    states: list[dict] = []
    for i in range(0, len(ids), 200):
        states.extend(await query_all_pages(
            db,
            collection=SKILL_STATE,
            where={
                "scholar_id": scholar_id,
                "sentence_id": {"$in": ids[i:i + 200]},
            },
            select=_SELECT_STATE_FIELDS,
        ))
    return states


def _build_sentence_entry(
    s: dict,
    *,
    states_by_sentence: dict[str, list[dict]],
) -> dict:
    """单句条目（与 /sentences 接口逐字一致 + is_canonical / canonical_sentence_id）。"""
    sid = s.get("sentence_id") or ""
    picked = pick_state(states_by_sentence.get(sid, []))
    s_states = states_by_sentence.get(sid, [])
    skills = {
        st.get("skill_code"): status_to_int(st.get("status"))
        for st in s_states
        if st.get("skill_code")
    }
    csid = s.get("canonical_sentence_id")
    return {
        "sentence_id": sid,
        "content": s.get("text", ""),
        "translation": s.get("translation", ""),
        "status": status_to_int(picked.get("status")) if picked else 0,
        "skills": skills,
        "weakest_skill": min(skills, key=skills.get) if skills else None,
        "review_count": int(picked.get("attempt_count") or 0) if picked else 0,
        "next_review_at": _to_iso(picked.get("next_review_at")) if picked else None,
        "is_canonical": not csid or csid == sid,
        "canonical_sentence_id": csid,
    }


async def getLessonSentenceGroups(
    db,
    *,
    textbook_id: str,
    lesson_id: str,
    scholar_id: str,
) -> dict:
    """课时语句组视图（service-contract §8.5，读兼容层）。

    Returns:
        {
          "lesson_id", "lesson_title",
          "summary": {mastery, skills, learned_sentence_count,
                      total_sentence_count, group_count},
          "groups": [{group_id, group_title, group_type, order_in_lesson, sentences[]}]
        }
    """
    lesson = await _find_lesson(db, textbook_id=textbook_id, lesson_id=lesson_id)

    sentences = await get_sentences_by_lesson(db, lesson_id)
    sentence_by_id = {
        s.get("sentence_id"): s for s in sentences if s.get("sentence_id")
    }
    sentence_ids = [s.get("sentence_id") for s in sentences if s.get("sentence_id")]

    states = await _load_states_by_sentence(db, scholar_id=scholar_id, sentence_ids=sentence_ids)
    states_by_sentence: dict[str, list[dict]] = {}
    for st in states:
        states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

    def _entry(s: dict) -> dict:
        return _build_sentence_entry(s, states_by_sentence=states_by_sentence)

    groups = await get_sentence_groups_by_lesson(db, lesson_id)

    if not groups:
        # 读兼容层：无任何分组 → 逐句构造临时组，返回结构逐字一致
        groups = [
            {
                "group_id": f"legacy_{lesson_id}_{s.get('sentence_id') or ''}",
                "group_title": (s.get("text") or "")[:20],
                "group_type": "stand_alone",
                "order_in_lesson": idx,
                "sentences": [_entry(s)],
            }
            for idx, s in enumerate(sentences)
        ]
    else:
        built = []
        for g in groups:
            members = []
            for sid in g.get("sentence_ids") or []:
                s = sentence_by_id.get(sid)
                if s:
                    members.append(_entry(s))
            built.append({
                "group_id": g.get("group_id"),
                "group_title": g.get("title") or "",
                "group_type": g.get("type"),
                "order_in_lesson": g.get("order_in_lesson"),
                "sentences": members,
            })
        groups = built

    # summary（口径与 /sentences 逐字一致）
    total_sentences = len(sentences)
    picked_states = [
        p for p in (
            pick_state(states_by_sentence.get(sid, [])) for sid in sentence_ids
        ) if p
    ]
    dist = mastery_distribution(picked_states)
    skill_dist: dict[str, float] = {}
    for code in _SKILL_CODES:
        code_states = [
            st for st in states
            if st.get("skill_code") == code and st.get("sentence_id") in sentence_ids
        ]
        if code_states:
            skill_dist[code] = mastery_ratio(
                mastery_distribution(code_states), total_sentences
            )
    learned = sum(
        1 for p in picked_states
        if p.get("status") in (STATUS_LEARNED, STATUS_MASTERED)
    )

    return {
        "lesson_id": lesson_id,
        "lesson_title": lesson.get("title", lesson.get("lesson_title", "")),
        "summary": {
            "mastery": mastery_ratio(dist, total_sentences),
            "skills": skill_dist,
            "learned_sentence_count": learned,
            "total_sentence_count": total_sentences,
            "group_count": len(groups),
        },
        "groups": groups,
    }
