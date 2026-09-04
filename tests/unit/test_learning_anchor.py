"""services/learning/anchor.py 单测（§7 阶段 1 验收）

验收用例（方案 §7 阶段 1）：
  新开始 / 正常续学不误伤 / 孤立尾部 / 真全通关 / 末课通关前方缺课
另补：存量锚点信任（幂等不跳动）、孤立尾部无更早进行中课不回退、无课教材。
"""
from __future__ import annotations

import pytest

from services.learning.anchor import (
    REASON_COMPLETED,
    REASON_FRONTIER,
    REASON_NEW_START,
    REASON_ORPHAN_TAIL,
    REASON_RESUME,
    lesson_key,
    resolve_book_anchor,
)


def _lessons(n: int, prefix: str = "l") -> list[dict]:
    """构造 n 个按序课（order=1..n）。"""
    return [
        {"lesson_id": f"{prefix}{i}", "order": i, "title": f"Lesson {i}"}
        for i in range(1, n + 1)
    ]


def _tags(*items) -> dict:
    """items: ("l3","started") / ("l8","done") → {lesson_id: tag}。"""
    return {k: v for k, v in items}


# ---------------------------------------------------------------------------
# 1. 新开始
# ---------------------------------------------------------------------------


def test_new_start_all_none():
    lessons = _lessons(8)
    res = resolve_book_anchor(lessons_sorted=lessons, tags={})
    assert res["reason"] == REASON_NEW_START
    assert res["lesson_id"] == "l1"
    assert res["chapter_id"] == "l1"
    assert res["group_id"] is None
    assert res["book_finished"] is False
    assert res["order"] == 1


def test_new_start_overrides_dirty_stored_anchor():
    """全书零真实证据 + 存量锚点指向末课（幽灵批量写入残留形态）→ 覆盖为第 1 课。"""
    lessons = _lessons(8)
    stored = {"current_lesson_id": "l8", "current_chapter_id": "l8"}
    res = resolve_book_anchor(lessons_sorted=lessons, tags={}, stored=stored)
    assert res["reason"] == REASON_NEW_START
    assert res["lesson_id"] == "l1"


def test_no_lessons_returns_none():
    assert resolve_book_anchor(lessons_sorted=[], tags={}) is None


# ---------------------------------------------------------------------------
# 2. 正常续学不误伤
# ---------------------------------------------------------------------------


def test_resume_adopts_stored_in_progress_lesson():
    """存量锚点课真实进行中 → 采纳（不按末课/百分比重算）。"""
    lessons = _lessons(8)
    tags = _tags(("l3", "started"), ("l7", "started"))
    stored = {"current_lesson_id": "l3", "current_chapter_id": "l3"}
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags, stored=stored)
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l3"


def test_resume_no_stored_picks_last_started():
    lessons = _lessons(8)
    tags = _tags(("l2", "started"), ("l5", "started"))
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l5"


def test_resume_single_real_lesson_in_middle():
    """全书仅一课有真实证据且不在首课 → 正常续学该课（无孤立回退）。"""
    lessons = _lessons(8)
    tags = _tags(("l6", "started"))
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l6"


# ---------------------------------------------------------------------------
# 3. 孤立尾部
# ---------------------------------------------------------------------------


def test_orphan_tail_rolls_back_to_earliest_started():
    """l1 进行中 + l2-7 无证据 + l8 有真实证据（顺序缺口尾部）→ 回退 l1。"""
    lessons = _lessons(8)
    tags = _tags(("l1", "started"), ("l8", "started"))
    stored = {"current_lesson_id": "l8", "current_chapter_id": "l8"}
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags, stored=stored)
    assert res["reason"] == REASON_ORPHAN_TAIL
    assert res["lesson_id"] == "l1"


def test_isolated_tail_without_stored_no_rollback_matches_read_side():
    """无存量锚点时孤立尾部不回退：与读侧简单规则同口径取最后一个进行中课（l8）。"""
    lessons = _lessons(8)
    tags = _tags(("l2", "done"), ("l3", "started"), ("l8", "started"))
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l8"


def test_isolated_tail_without_earlier_started_no_rollback():
    """末课孤立但有真实证据、无更早进行中课 → 不回退，按末课续学（真实跳学不误伤）。"""
    lessons = _lessons(8)
    tags = _tags(("l1", "done"), ("l8", "started"))
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l8"


# ---------------------------------------------------------------------------
# 4. 真全通关
# ---------------------------------------------------------------------------


def test_all_done_book_completed():
    lessons = _lessons(8)
    tags = {f"l{i}": "done" for i in range(1, 9)}
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_COMPLETED
    assert res["lesson_id"] == "l8"
    assert res["book_finished"] is True


def test_partial_done_run_frontier_next_lesson():
    """l1-3 全通关、l4 起未开始 → 锚点推进到 l4（_enterNextLesson 语义）。"""
    lessons = _lessons(8)
    tags = {f"l{i}": "done" for i in range(1, 4)}
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_FRONTIER
    assert res["lesson_id"] == "l4"
    assert res["book_finished"] is False


# ---------------------------------------------------------------------------
# 5. 末课通关、前方缺课
# ---------------------------------------------------------------------------


def test_last_lesson_done_with_front_gap_stays_completed():
    """末课已通关、前方存在缺课（l2-7 无证据）→ 停留末课完成态（不因缺课回退）。"""
    lessons = _lessons(8)
    tags = _tags(("l1", "done"), ("l8", "done"))
    res = resolve_book_anchor(lessons_sorted=lessons, tags=tags)
    assert res["reason"] == REASON_COMPLETED
    assert res["lesson_id"] == "l8"
    assert res["book_finished"] is True


# ---------------------------------------------------------------------------
# 幂等 / 健壮性
# ---------------------------------------------------------------------------


def test_resolve_is_idempotent_same_evidence():
    """相同证据 + 已校准锚点 → 结果稳定（脚本重复执行不跳动）。"""
    lessons = _lessons(8)
    tags = _tags(("l3", "started"))
    first = resolve_book_anchor(lessons_sorted=lessons, tags=tags, stored={"current_lesson_id": "l3"})
    second = resolve_book_anchor(lessons_sorted=lessons, tags=tags, stored={"current_lesson_id": "l3"})
    assert first == second
    assert first["lesson_id"] == "l3"


def test_invalid_tag_falls_back_to_none():
    lessons = _lessons(3)
    res = resolve_book_anchor(lessons_sorted=lessons, tags={"l2": "weird"})
    assert res["reason"] == REASON_NEW_START
    assert res["lesson_id"] == "l1"


def test_lesson_key_fallback_to_id():
    lesson = {"_id": "legacy_x", "order": 1}
    assert lesson_key(lesson) == "legacy_x"


def test_stored_chapter_only_hint_accepted():
    """旧记录只有 current_chapter_id（无 current_lesson_id）也能被信任。"""
    lessons = _lessons(8)
    tags = _tags(("l4", "started"))
    res = resolve_book_anchor(
        lessons_sorted=lessons,
        tags=tags,
        stored={"current_chapter_id": "l4"},
    )
    assert res["reason"] == REASON_RESUME
    assert res["lesson_id"] == "l4"
