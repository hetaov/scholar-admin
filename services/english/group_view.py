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
    PUBLIC_SKILL_CODES,
    canonical_skill_code,
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

# 对外统计能力集合；存量 listening 读取时归一化为 speaking。
_SKILL_CODES = PUBLIC_SKILL_CODES

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
        canonical_skill_code(st.get("skill_code")): status_to_int(st.get("status"))
        for st in s_states
        if canonical_skill_code(st.get("skill_code"))
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


def _is_review_due(state: dict, now_ts: int | None = None) -> bool:
    """句子的**记忆规律到期**判定：`next_review_at` 落在 (0, 当日末]。

    ⚠️ 与 `/tracking/review-plan` 的句级到期谓词**语义不同**（后者额外要求 `status != mastered`，
    即「已掌握句免于复习」）。本谓词**不排除已掌握句**——v4 的产品目标是学习者的**遗忘曲线**：
    已掌握的句子同样会随间隔到期而需要复习（用户 2026-09-21 拍板「已掌握组也按间隔推回来，
    先按简单规则」）。两个谓词各有消费方，勿互改。
    """
    if not state:
        return False
    ts = state.get("next_review_at")
    if not isinstance(ts, (int, float)) or ts <= 0:
        return False
    return ts <= (now_ts if now_ts is not None else _end_of_today_ts())


def _end_of_today_ts() -> int:
    """当日 23:59:59（UTC，与 review-plan 口径一致）。"""
    now = datetime.now(timezone.utc)
    end = now.replace(hour=23, minute=59, second=59, microsecond=0)
    return int(end.timestamp())


def aggregate_group_progress(
    *,
    groups: list[dict],
    sentences: list[dict],
    states: list[dict],
) -> dict:
    """组维度进度聚合（派生口径；契约 api-contract §3.1 组计数 / data-model-contract §4.22）。

    v4 需求 R1：统计只讲「已学习组 / 已掌握组 / 学习进度」，不再按能力（技能/阶段）维度。

    口径（与句级展示**同源**，复用 pick_state / status_to_int / mastery_distribution）：
    - 组状态 = 组内句状态取 **min**（木桶：组状态 = 最弱句，与组掌握同口径）
    - 已学习组 = 组状态 >= 1（组内至少 1 句已学）；已掌握组 = 组状态 >= 3
    - 无分组的课次按读兼容层逐句合成 legacy 组（与 getLessonSentenceGroups 同款），
      保证书级计数与端上组视图一致
    - 与能力维度无关：**不需要** skill 过滤，历史行缺 phase 也不影响本口径

    Args:
        groups: sentence_group 文档列表（含 group_id / lesson_id / sentence_ids / title）
        sentences: 教材内句子列表（含 sentence_id / lesson_id / text）
        states: 该教材句子的 skill_state 列表（全量，不过滤 skill_code）

    Returns:
        {"learned_group_count": int, "mastered_group_count": int, "total_group_count": int,
         "groups": [{"group_id", "group_title", "group_status", "group_mastery", "sentence_count"}]}
    """
    states_by_sentence: dict[str, list[dict]] = {}
    for st in states:
        states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

    sentence_by_id = {
        s.get("sentence_id"): s for s in sentences if s.get("sentence_id")
    }
    sentences_by_lesson: dict[str, list[dict]] = {}
    for s in sentences:
        sentences_by_lesson.setdefault(s.get("lesson_id"), []).append(s)
    groups_by_lesson: dict[str, list[dict]] = {}
    for g in groups:
        groups_by_lesson.setdefault(g.get("lesson_id"), []).append(g)

    rows: list[dict] = []
    for lesson_id, lesson_sentences in sentences_by_lesson.items():
        lesson_groups = groups_by_lesson.get(lesson_id) or []
        if lesson_groups:
            meta = [
                {
                    "group_id": g.get("group_id"),
                    "group_title": g.get("title") or "",
                    "order_in_lesson": g.get("order_in_lesson"),
                }
                for g in lesson_groups
            ]
            member_lists = [
                [
                    sentence_by_id[sid]
                    for sid in (g.get("sentence_ids") or [])
                    if sid in sentence_by_id
                ]
                for g in lesson_groups
            ]
        else:
            # 读兼容层：无任何分组 → 逐句临时组（与 getLessonSentenceGroups 同款）
            meta = [
                {
                    "group_id": f"legacy_{lesson_id}_{s.get('sentence_id') or ''}",
                    "group_title": (s.get("text") or "")[:20],
                    "order_in_lesson": idx,
                }
                for idx, s in enumerate(lesson_sentences)
            ]
            member_lists = [[s] for s in lesson_sentences]

        for info, members in zip(meta, member_lists):
            if not members:
                continue
            picked = [
                p
                for p in (
                    pick_state(states_by_sentence.get(m.get("sentence_id"), []))
                    for m in members
                )
                if p
            ]
            # 无 skill_state 的句子计为未学（status 0，与句级展示一致）
            statuses = [status_to_int(p.get("status")) for p in picked]
            statuses += [0] * (len(members) - len(picked))
            rows.append({
                "group_id": info["group_id"],
                "group_title": info["group_title"],
                "lesson_id": lesson_id,
                "order_in_lesson": info.get("order_in_lesson"),
                "group_status": min(statuses) if statuses else 0,
                "learned_sentence_count": sum(1 for s in statuses if s >= 1),
                "group_due_count": sum(1 for p in picked if _is_review_due(p)),
                "group_mastery": (
                    mastery_ratio(mastery_distribution(picked), len(members))
                    if picked
                    else 0.0
                ),
                "sentence_count": len(members),
            })

    return {
        "learned_group_count": sum(1 for r in rows if r["group_status"] >= 1),
        "mastered_group_count": sum(1 for r in rows if r["group_status"] >= 3),
        "total_group_count": len(rows),
        "groups": rows,
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
            if canonical_skill_code(st.get("skill_code")) == code
            and st.get("sentence_id") in sentence_ids
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
