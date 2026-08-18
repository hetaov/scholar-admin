"""S4.3 Review Scheduler — 到期复习调度（复用 /tracking/review-plan 到期口径）

口径（ADR-0004 间隔算法）：
- pick_state（乐观，progress 最高者）后 next_review_at 存在且 ≤ 当日 23:59:59
- 且 status ≠ mastered
- 排序：next_review_at 升序 + 同到期日 mastery_score 升序（薄弱优先）

供 S4.3 LearningContextBuilder 组装 reviewItems 输入（P2-3 核心输入），
亦可供其他调度方（如复习计划）复用，避免与 /tracking/review-plan 口径漂移。
"""
from __future__ import annotations

from datetime import datetime, timezone

from services.models_content import get_sentences_by_ids, query_all_pages
from services.models_learning import SKILL_STATE, STATUS_MASTERED
from services.progress import pick_state


def _to_iso(timestamp) -> str | None:
    """unix 秒 → UTC ISO8601；非法/空返回 None。"""
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


async def get_due_review_items(
    db,
    *,
    scholar_id: str,
    date: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """到期复习项（供 LearningContext 组装与 Planner 消费）。

    与 GET /tracking/review-plan 同一到期口径：
    - 一次 skill_state 查询（按学者拉取必要字段）+ 内容 $in 批量加载 + 内存过滤排序；
    - 输出：sentence 快照（content/translation/lesson_id）+ 到期时间 + 上次结果 +
      mastery/review_count。
    - 无到期记录返回空列表，不报错。
    """
    if date is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = date
    end_ts = int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        .timestamp()
    )

    states = await query_all_pages(
        db,
        collection=SKILL_STATE,
        where={"scholar_id": scholar_id},
        select={
            "scholar_id": 1,
            "sentence_id": 1,
            "skill_code": 1,
            "status": 1,
            "mastery_score": 1,
            "attempt_count": 1,
            "next_review_at": 1,
            "last_outcome": 1,
        },
    )
    states_by_sentence: dict[str, list[dict]] = {}
    for st in states:
        states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

    candidates: list[tuple[str, dict]] = []  # (sentence_id, picked_state)
    for sid, s_states in states_by_sentence.items():
        picked = pick_state(s_states)
        if not picked:
            continue
        if picked.get("status") == STATUS_MASTERED:
            continue
        try:
            review_ts = int(picked.get("next_review_at") or 0)
        except (TypeError, ValueError):
            continue
        if review_ts <= 0 or review_ts > end_ts:
            continue
        candidates.append((sid, picked))

    if not candidates:
        return []

    sentences = await get_sentences_by_ids(db, [sid for sid, _ in candidates])
    content_by_id = {s.get("sentence_id"): s for s in sentences}

    queue = []
    for sid, picked in candidates:
        s = content_by_id.get(sid) or {}
        queue.append({
            "sentence_id": sid,
            "content": s.get("text", ""),
            "translation": s.get("translation", ""),
            "lesson_id": s.get("lesson_id", ""),
            "next_review_at": _to_iso(picked.get("next_review_at")),
            "last_result": picked.get("last_outcome"),
            "mastery_score": int(picked.get("mastery_score") or 0),
            "review_count": int(picked.get("attempt_count") or 0),
            "_sort_ts": int(picked.get("next_review_at") or 0),
            "_sort_mastery": int(picked.get("mastery_score") or 0),
        })
    queue.sort(key=lambda it: (it.pop("_sort_ts"), it.pop("_sort_mastery")))
    if limit:
        queue = queue[:limit]
    return queue
