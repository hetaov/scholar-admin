"""单元测试：scripts/repair_book_anchor 存量锚点重算脚本纯函数

覆盖（§7 阶段 4，docs_v1《沉浸式锚点与learning写入证据化重构方案》§5.5.4/§10.8）：
- _is_result_row：结果性状态 / mastery_score 阈值判定（§5.5.1 强证据）
- sort_lessons：order 已修复(1..N 唯一)保持 DB 序；脏 order 回退标题课号重排
- assign_row：行归属课（lesson_id 优先 → sentence 反查）
- compute_lesson_tags：none / started（带分 attempt-only、无结果行、句子目录缺失、
  行不可归属、部分句达标）/ done（整课句全覆盖结果态）边界
- plan_single_book：stored vs resolved diff（零变化幂等 / 无锚点补第 1 课 /
  group 非空归一置 null / completed 联动 anchor 决策）
"""
from __future__ import annotations

from scripts.repair_book_anchor import (
    _is_result_row,
    assign_row,
    compute_lesson_tags,
    plan_single_book,
    sort_lessons,
)
from services.learning.anchor import (
    REASON_COMPLETED,
    REASON_NEW_START,
    REASON_RESUME,
    TAG_DONE,
    TAG_NONE,
    TAG_STARTED,
)

TH = 60  # result mastery threshold


def _lesson(lid: str, title: str, order: int, chapter_id: str | None = "c1") -> dict:
    return {
        "_id": lid,
        "lesson_id": lid,
        "chapter_id": chapter_id,
        "title": title,
        "order": order,
    }


def _row(sid: str, status: str = "learning", mastery=None) -> dict:
    return {"sentence_id": sid, "status": status, "mastery_score": mastery}


def _chapter(cid: str = "c1", order: int = 1) -> dict:
    return {"_id": cid, "chapter_id": cid, "title": "Chapter", "order": order}


class TestIsResultRow:
    def test_result_statuses(self):
        assert _is_result_row(_row("s1", "learned"), TH)
        assert _is_result_row(_row("s1", "mastered"), TH)
        assert not _is_result_row(_row("s1", "learning"), TH)
        assert not _is_result_row(_row("s1", "review_due"), TH)

    def test_mastery_threshold(self):
        assert _is_result_row(_row("s1", "learning", mastery=60), TH)
        assert _is_result_row(_row("s1", "learning", mastery=100), TH)
        assert not _is_result_row(_row("s1", "learning", mastery=59), TH)
        assert not _is_result_row(_row("s1", "learning", mastery=None), TH)


class TestSortLessons:
    def test_keeps_fixed_order(self):
        """order 已修复为 1..N 唯一 → 保持 DB order ASC 返回序（repair_lesson_order 幂等）。"""
        lessons = [
            _lesson("l1", "Unit 1", 1),
            _lesson("l2", "Unit 2", 2),
            _lesson("l3", "Unit 3", 3),
        ]
        assert [l["lesson_id"] for l in sort_lessons(lessons, [_chapter()])] == ["l1", "l2", "l3"]

    def test_falls_back_to_title_when_order_dirty(self):
        """脏 order（全 1）→ 按标题课号重排（repair_lesson_order.plan_book_order 同源）。"""
        lessons = [
            _lesson("u8", "Unit 8 Joy", 1),
            _lesson("u1", "Unit 1 Come on in", 1),
            _lesson("u2", "Unit 2 Help", 1),
        ]
        ordered = sort_lessons(lessons, [_chapter()])
        assert [l["lesson_id"] for l in ordered] == ["u1", "u2", "u8"]

    def test_duplicate_order_falls_back(self):
        """order 重复非唯一 → 也回退标题课号重排。"""
        lessons = [
            _lesson("u2", "Unit 2 Help", 1),
            _lesson("u1", "Unit 1 Come on in", 1),
        ]
        assert [l["lesson_id"] for l in sort_lessons(lessons, [_chapter()])] == ["u1", "u2"]

    def test_empty(self):
        assert sort_lessons([], [_chapter()]) == []


class TestAssignRow:
    def test_lesson_id_direct_hit(self):
        lesson_ids = {"l1", "l2"}
        assert assign_row({"lesson_id": "l2", "sentence_id": "s9"}, lesson_ids, {}) == "l2"

    def test_sentence_fallback(self):
        lesson_ids = {"l1", "l2"}
        sent2lesson = {"s3": "l1"}
        assert assign_row({"lesson_id": "l_ghost", "sentence_id": "s3"}, lesson_ids, sent2lesson) == "l1"

    def test_no_hit_returns_none(self):
        lesson_ids = {"l1"}
        sent2lesson = {"s3": "l1"}
        assert assign_row({"lesson_id": "l_ghost", "sentence_id": "s9"}, lesson_ids, sent2lesson) is None
        assert assign_row({}, lesson_ids, sent2lesson) is None


class TestComputeLessonTags:
    LESSONS = [
        _lesson("l1", "Unit 1", 1),
        _lesson("l2", "Unit 2", 2),
        _lesson("l3", "Unit 3", 3),
        _lesson("l4", "Unit 4", 4),
        _lesson("l5", "Unit 5", 5),
        _lesson("l6", "Unit 6", 6),
        _lesson("l7", "Unit 7", 7),
    ]

    def _run(self, rows, scored, sents):
        return compute_lesson_tags(
            self.LESSONS,
            rows_by_lesson=rows,
            scored_attempts_by_lesson=scored,
            sentences_by_lesson=sents,
            result_mastery_threshold=TH,
        )

    def test_done_when_all_sentences_result(self):
        """课内每句都有结果态行且行全部可归属 → done（读侧 progress=100 同义）。"""
        rows = {"l1": [_row("s1", "learned"), _row("s2", "mastered")]}
        sents = {"l1": ["s1", "s2"]}
        tags, _ = self._run(rows, {}, sents)
        assert tags["l1"] == TAG_DONE

    def test_multi_skill_sentence_result_counted(self):
        """句内多 skill 行：任一结果态 → 该句达标（读侧 pick_state 乐观口径）。"""
        rows = {"l2": [_row("s1", "learned"), _row("s1", "learning"), _row("s2", "learning", 80)]}
        sents = {"l2": ["s1", "s2"]}
        tags, _ = self._run(rows, {}, sents)
        assert tags["l2"] == TAG_DONE

    def test_partial_sentences_started(self):
        """仅部分句达结果态（2/3）→ started。"""
        rows = {"l3": [_row("s1", "learned"), _row("s2", "learning")]}
        sents = {"l3": ["s1", "s2", "s3"]}
        tags, detail = self._run(rows, {}, sents)
        assert tags["l3"] == TAG_STARTED
        assert "1/3" in detail["l3"]

    def test_sentence_never_studied_started(self):
        """课有行但只覆盖部分句子（其余句从未学）→ 读侧 progress<100 → started。"""
        rows = {"l4": [_row("s1", "learned")]}
        sents = {"l4": ["s1", "s2"]}
        tags, _ = self._run(rows, {}, sents)
        assert tags["l4"] == TAG_STARTED

    def test_rows_but_no_sentence_catalog_conservative_started(self):
        """有 skill_state 行但课句子目录缺失 → 保守不判通关（防 vacuous done）。"""
        rows = {"l5": [_row("s1", "learned")]}
        tags, detail = self._run(rows, {}, {})
        assert tags["l5"] == TAG_STARTED
        assert "目录缺失" in detail["l5"]

    def test_none_when_no_evidence(self):
        """无行无带分 attempt → none。"""
        tags, _ = self._run({}, {}, {})
        assert tags["l5"] == TAG_NONE
        assert tags["l6"] == TAG_NONE

    def test_scored_attempt_only_started(self):
        """无 skill_state 行但有带分 attempt → 真实已开始（不判通关）。"""
        rows = {}
        scored = {"l6": 2}
        tags, detail = self._run(rows, scored, {"l6": ["s1"]})
        assert tags["l6"] == TAG_STARTED
        assert "带分 attempt 2" in detail["l6"]

    def test_unassignable_row_blocks_done(self):
        """行无法归属课句子（sentence 不在课目录）→ 保守 started，即使其余句全达标。"""
        rows = {"l7": [_row("s1", "learned"), _row("s2", "learned"), _row("sX", "learned")]}
        sents = {"l7": ["s1", "s2"]}
        tags, detail = self._run(rows, {}, sents)
        assert tags["l7"] == TAG_STARTED
        assert "无法归属" in detail["l7"]

    def test_low_mastery_not_result(self):
        """mastery 低于阈值不构成结果态 → 未通关。"""
        rows = {"l1": [_row("s1", "learning", 40), _row("s2", "learning", 40)]}
        sents = {"l1": ["s1", "s2"]}
        tags, _ = self._run(rows, {}, sents)
        assert tags["l1"] == TAG_STARTED


class TestPlanSingleBook:
    LESSONS = [
        _lesson("l1", "Unit 1", 1),
        _lesson("l2", "Unit 2", 2),
        _lesson("l3", "Unit 3", 3),
    ]

    def _book(self, lesson=None, chapter=None, group=None) -> dict:
        return {
            "_id": "sid_tid",
            "current_lesson_id": lesson,
            "current_chapter_id": chapter,
            "current_group_id": group,
        }

    def test_stored_matches_resolved_no_change(self):
        """存量锚点课真实进行中 → 信任采纳，零变化（幂等）。"""
        tags = {"l1": TAG_DONE, "l2": TAG_STARTED, "l3": TAG_NONE}
        plan = plan_single_book(self._book(lesson="l2", chapter="l2"), self.LESSONS, tags)
        assert plan["changed"] is False
        assert plan["reason"] == REASON_RESUME
        assert plan["resolved_lesson"] == "l2"

    def test_no_anchor_all_none_resolves_first_lesson(self):
        """无存量锚点 + 全书零证据 → 第 1 课（new_start 覆盖脏锚点语义）。"""
        tags = {"l1": TAG_NONE, "l2": TAG_NONE, "l3": TAG_NONE}
        plan = plan_single_book(self._book(lesson=None, chapter="l9"), self.LESSONS, tags)
        assert plan["changed"] is True
        assert plan["reason"] == REASON_NEW_START
        assert plan["resolved_lesson"] == "l1"

    def test_stale_anchor_replaced(self):
        """存量锚点课无真实证据（如旧入口盲写）→ 重算到真实课。"""
        tags = {"l1": TAG_STARTED, "l2": TAG_NONE, "l3": TAG_NONE}
        plan = plan_single_book(self._book(lesson="l3", chapter="l3"), self.LESSONS, tags)
        assert plan["changed"] is True
        assert plan["reason"] == REASON_RESUME
        assert plan["resolved_lesson"] == "l1"

    def test_stale_group_normalized_to_null(self):
        """group 级锚非空（旧契约残留）→ 归一置 null，即使 lesson 已一致。"""
        tags = {"l1": TAG_DONE, "l2": TAG_STARTED, "l3": TAG_NONE}
        plan = plan_single_book(self._book(lesson="l2", chapter="l2", group="g1"), self.LESSONS, tags)
        assert plan["changed"] is True  # group 非空未归一 → 需写回置 null
        assert plan["resolved_lesson"] == "l2"

    def test_book_finished_keeps_last_lesson(self):
        """末课已通关 → completed 停留末课；存量锚点已是末课 → 零变化。"""
        lessons = [
            _lesson("l1", "Unit 1", 1),
            _lesson("l2", "Unit 2", 2),
        ]
        tags = {"l1": TAG_DONE, "l2": TAG_DONE}
        plan = plan_single_book(self._book(lesson="l2", chapter="l2"), lessons, tags)
        assert plan["changed"] is False
        assert plan["reason"] == REASON_COMPLETED
        assert plan["book_finished"] is True
        assert plan["resolved_lesson"] == "l2"

    def test_chapter_lesson_alias_normalization(self):
        """存量只有 chapter 级锚（lesson 空）→ 归一补 lesson 同写课 id（C2 同一契约）。"""
        tags = {"l1": TAG_DONE, "l2": TAG_STARTED, "l3": TAG_NONE}
        plan = plan_single_book(self._book(lesson=None, chapter="l2"), self.LESSONS, tags)
        assert plan["changed"] is True  # lesson 空 → 补写
        assert plan["resolved_lesson"] == "l2"
