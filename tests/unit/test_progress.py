"""单元测试:进度/掌握度聚合模块(Phase 4) — services/progress

覆盖:
- 句子级:sentence_progress / pick_state / mastery_distribution / merge_distributions
- 课/章/书逐级聚合:lesson_progress / chapter_progress / book_progress
- 顶层聚合:aggregate_progress(三级结构 + 兼容字段)
- 学习时长:sum_time_spent
- 核心原则:同一输入可复现;按 skill_code 过滤后只反映该能力
"""

from __future__ import annotations

import pytest

from services.progress import (
    aggregate_progress,
    book_progress,
    chapter_progress,
    lesson_progress,
    mastery_distribution,
    mastery_ratio,
    merge_distributions,
    pick_state,
    sentence_progress,
    status_distribution_array,
    status_to_int,
    sum_time_spent,
)

STATUS_LEARNED = "learned"
STATUS_MASTERED = "mastered"
STATUS_LEARNING = "learning"
STATUS_REVIEW_DUE = "review_due"


# ===========================================================================
# 句子级
# ===========================================================================


class TestSentenceProgress:
    def test_score_based(self):
        assert sentence_progress({"mastery_score": 90}) == 0.9
        assert sentence_progress({"mastery_score": 0}) == 0.0
        assert sentence_progress({"mastery_score": 150}) == 1.0  # 封顶

    def test_status_based(self):
        assert sentence_progress({"status": STATUS_MASTERED}) == 1.0
        assert sentence_progress({"status": STATUS_LEARNED}) == 1.0
        assert sentence_progress({"status": STATUS_LEARNING}) == 0.5
        assert sentence_progress({"status": STATUS_REVIEW_DUE}) == 0.5
        assert sentence_progress({"status": "not_started"}) == 0.0

    def test_mastery_score_takes_priority(self):
        assert sentence_progress({"status": STATUS_LEARNED, "mastery_score": 30}) == 0.3

    def test_empty(self):
        assert sentence_progress(None) == 0.0
        assert sentence_progress({}) == 0.0


class TestPickState:
    def test_single(self):
        st = {"sentence_id": "s1", "skill_code": "translation", "status": STATUS_LEARNED}
        assert pick_state([st]) == st

    def test_with_skill_code_filter(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation", "mastery_score": 40},
            {"sentence_id": "s1", "skill_code": "listening", "mastery_score": 90},
        ]
        picked = pick_state(states, skill_code="translation")
        assert picked["skill_code"] == "translation"
        assert pick_state(states, skill_code="speaking") is None

    def test_optimistic_without_filter(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation", "mastery_score": 40},
            {"sentence_id": "s1", "skill_code": "listening", "mastery_score": 90},
        ]
        assert pick_state(states)["mastery_score"] == 90  # 取 progress 最高

    def test_empty(self):
        assert pick_state([]) is None
        assert pick_state(None) is None


class TestMasteryDistribution:
    def test_counts(self):
        states = [
            {"status": STATUS_LEARNED},
            {"status": STATUS_MASTERED},
            {"status": STATUS_LEARNING},
            {"status": STATUS_LEARNED},
            {},  # 未知状态不统计
        ]
        dist = mastery_distribution(states)
        assert dist["learned"] == 2
        assert dist["mastered"] == 1
        assert dist["learning"] == 1
        assert dist["total"] == 4
        assert dist["learned_count"] == 2
        assert dist["mastered_count"] == 1
        assert dist["learned_pct"] == 0.5
        assert dist["mastered_pct"] == 0.25

    def test_empty(self):
        dist = mastery_distribution([])
        assert dist["total"] == 0
        assert dist["learned_pct"] == 0.0
        assert dist["mastered_pct"] == 0.0

    def test_merge(self):
        merged = merge_distributions(
            [mastery_distribution([{"status": STATUS_LEARNED}]),
             mastery_distribution([{"status": STATUS_MASTERED}, {"status": STATUS_LEARNED}])]
        )
        assert merged["learned"] == 2
        assert merged["mastered"] == 1
        assert merged["total"] == 3
        assert merged["learned_pct"] == round(2 / 3, 4)


# ===========================================================================
# 课 / 章 / 书 逐级聚合
# ===========================================================================


class TestLessonProgress:
    LESSON = {"lesson_id": "lesson_1", "title": "Lesson 1", "order": 1}

    def test_mixed_learned_and_unlearned(self):
        items = [
            {"sentence_id": "s1", "state": {"status": STATUS_LEARNED, "mastery_score": 80}},
            {"sentence_id": "s2", "state": {"status": STATUS_LEARNING, "mastery_score": 40}},
            {"sentence_id": "s3", "state": None},  # 未学
        ]
        item = lesson_progress(self.LESSON, items)
        assert item["lesson_id"] == "lesson_1"
        assert item["total_sentence_count"] == 3
        assert item["learned_sentence_count"] == 1
        assert item["progress"] == round((0.8 + 0.4 + 0.0) / 3, 4)
        assert item["mastery_distribution"]["total"] == 2  # 未学不计分布

    def test_empty_lesson(self):
        item = lesson_progress(self.LESSON, [])
        assert item["progress"] == 0.0
        assert item["total_sentence_count"] == 0


class TestChapterProgress:
    def test_weighted_by_sentences(self):
        lesson_items = [
            {"lesson_id": "l1", "total_sentence_count": 3, "learned_sentence_count": 3,
             "progress": 1.0, "mastery_distribution": mastery_distribution(
                 [{"status": STATUS_LEARNED} for _ in range(3)])},
            {"lesson_id": "l2", "total_sentence_count": 1, "learned_sentence_count": 0,
             "progress": 0.0, "mastery_distribution": mastery_distribution([])},
        ]
        item = chapter_progress({"chapter_id": "c1", "title": "Ch1", "order": 1}, lesson_items)
        assert item["progress"] == 0.75  # (3*1.0 + 1*0.0)/4
        assert item["total_sentence_count"] == 4
        assert item["learned_sentence_count"] == 3
        assert item["lesson_count"] == 2
        assert item["mastery_distribution"]["learned"] == 3

    def test_empty(self):
        item = chapter_progress({"chapter_id": "c1"}, [])
        assert item["progress"] == 0.0
        assert item["lesson_count"] == 0


class TestBookProgress:
    def test_weighted(self):
        chapters = [
            {"total_sentence_count": 4, "learned_sentence_count": 4, "progress": 1.0,
             "lesson_count": 2, "mastery_distribution": mastery_distribution(
                 [{"status": STATUS_LEARNED} for _ in range(4)])},
            {"total_sentence_count": 4, "learned_sentence_count": 0, "progress": 0.0,
             "lesson_count": 1, "mastery_distribution": mastery_distribution([])},
        ]
        book = book_progress(chapters)
        assert book["progress"] == 0.5
        assert book["total_sentence_count"] == 8
        assert book["learned_sentence_count"] == 4
        assert book["chapter_count"] == 2
        assert book["lesson_count"] == 3


# ===========================================================================
# 学习时长
# ===========================================================================


class TestSumTimeSpent:
    def test_valid(self):
        assert sum_time_spent([
            {"time_spent": 120},
            {"time_spent": 30.5},
            {"time_spent": None},
        ]) == 150.5

    def test_dirty(self):
        assert sum_time_spent([
            {"time_spent": "abc"},
            {"time_spent": -5},
            {},
        ]) == 0.0

    def test_empty(self):
        assert sum_time_spent([]) == 0.0


# ===========================================================================
# 顶层聚合
# ===========================================================================


class TestAggregateProgress:
    SENTENCES = [
        {"sentence_id": "s1", "lesson_id": "l1", "chapter_id": "c1"},
        {"sentence_id": "s2", "lesson_id": "l1", "chapter_id": "c1"},
        {"sentence_id": "s3", "lesson_id": "l2", "chapter_id": "c1"},
        {"sentence_id": "s4", "lesson_id": "l2", "chapter_id": "c1"},
    ]
    LESSONS = [
        {"lesson_id": "l1", "chapter_id": "c1", "title": "L1", "order": 1},
        {"lesson_id": "l2", "chapter_id": "c1", "title": "L2", "order": 2},
    ]
    CHAPTERS = [
        {"chapter_id": "c1", "textbook_id": "tb_1", "title": "C1", "order": 1},
    ]

    def test_full_aggregation(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
            {"sentence_id": "s2", "skill_code": "translation",
             "status": STATUS_LEARNING, "mastery_score": 40},
            {"sentence_id": "s3", "skill_code": "translation",
             "status": STATUS_MASTERED, "mastery_score": 95},
        ]
        attempts = [{"time_spent": 120}, {"time_spent": 30}]
        stats = aggregate_progress(
            scholar_id="scholar_1",
            textbook_id="tb_1",
            states=states,
            sentences=self.SENTENCES,
            lessons=self.LESSONS,
            chapters=self.CHAPTERS,
            attempts=attempts,
        )
        summary = stats["summary"]
        assert summary["textbook_progress"] == round((0.8 + 0.4 + 0.95 + 0.0) / 4, 4)
        assert summary["learned_sentence_count"] == 2  # s1, s3
        assert summary["total_sentence_count"] == 4
        assert summary["chapter_count"] == 1
        assert summary["lesson_count"] == 2
        assert summary["total_time_spent"] == 150.0
        assert summary["total_time_spent_display"] == "2分30秒"
        assert summary["mastery_distribution"]["learned"] == 1
        assert summary["mastery_distribution"]["mastered"] == 1

        # 章级
        assert stats["chapters"][0]["chapter_id"] == "c1"
        assert stats["chapters"][0]["lessons"][0]["lesson_id"] == "l1"
        assert stats["chapters"][0]["lessons"][0]["progress"] == 0.6

        # 平铺字段(命名统一: lesson_id)
        assert stats["units"][0]["lesson_id"] == "l1"
        assert len(stats["sentences"]) == 4
        assert stats["sentences"][0]["learned"] is True

    def test_skill_filter_only_reflects_that_skill(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
            {"sentence_id": "s1", "skill_code": "listening",
             "status": STATUS_LEARNING, "mastery_score": 30},
            {"sentence_id": "s2", "skill_code": "listening",
             "status": STATUS_LEARNED, "mastery_score": 90},
        ]
        # 只算 translation: 只有 s1 有一条 translation 状态
        stats = aggregate_progress(
            scholar_id="scholar_1",
            textbook_id="tb_1",
            states=states,
            sentences=self.SENTENCES,
            lessons=self.LESSONS,
            chapters=self.CHAPTERS,
            skill_code="translation",
        )
        summary = stats["summary"]
        assert stats["skill_code"] == "translation"
        assert summary["learned_sentence_count"] == 1  # s1
        assert summary["textbook_progress"] == round(0.8 / 4, 4)

    def test_reproducible(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
            {"sentence_id": "s4", "skill_code": "translation",
             "status": STATUS_LEARNING, "mastery_score": 50},
        ]
        attempts = [{"time_spent": 100}, {"time_spent": 20}]
        a = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS, attempts=attempts,
        )
        b = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS, attempts=attempts,
        )
        assert a == b  # 同一输入两次调用完全一致

    def test_detail_overview_strips_lessons(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
        ]
        overview = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS,
            detail="overview",
        )
        # 章级存在且不含 lessons，summary 与 full 一致
        assert "chapters" in overview
        assert "lessons" not in overview and "units" not in overview
        assert "sentences" not in overview
        assert "lessons" not in overview["chapters"][0]
        assert overview["chapters"][0]["chapter_id"] == "c1"
        assert overview["chapters"][0]["lesson_count"] == 2
        full = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS,
        )
        assert overview["summary"] == full["summary"]

    def test_no_chapter_textbook_lessons_under_book(self):
        """无章教材: 不传 chapters, lesson 直接挂在 book 下聚合。"""
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
            {"sentence_id": "s2", "skill_code": "translation",
             "status": STATUS_LEARNING, "mastery_score": 40},
        ]
        stats = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_noch",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=[],
        )
        summary = stats["summary"]
        # book 级聚合, 无 chapter
        assert summary["chapter_count"] == 0
        assert summary["lesson_count"] == 2
        assert summary["total_sentence_count"] == 4
        assert summary["learned_sentence_count"] == 1
        assert summary["textbook_progress"] == round((0.8 + 0.4 + 0.0 + 0.0) / 4, 4)
        # chapters 为空数组
        assert stats["chapters"] == []
        # full 模式下 lessons/units 平铺仍返回(命名统一: lesson_id)
        assert [u["lesson_id"] for u in stats["units"]] == ["l1", "l2"]

    def test_no_chapter_overview(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
        ]
        stats = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_noch",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=[],
            detail="overview",
        )
        assert stats["chapters"] == []
        assert stats["summary"]["lesson_count"] == 2
        assert stats["summary"]["chapter_count"] == 0
        # 无章教材在 overview 粒度下由顶层 lessons 承载课级进度
        assert [l["lesson_id"] for l in stats["lessons"]] == ["l1", "l2"]
        assert stats["lessons"][0]["total_sentence_count"] == 2
        assert stats["lessons"][0]["learned_sentence_count"] == 1
        assert "sentences" not in stats
        assert "units" not in stats

    def test_detail_lesson_returns_summary_and_lesson_list(self):
        """detail="lesson"(默认): 仅 summary + 课级统计, 无章节/句子明细。"""
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
            {"sentence_id": "s3", "skill_code": "translation",
             "status": STATUS_MASTERED, "mastery_score": 95},
        ]
        stats = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS,
            detail="lesson",
        )
        # 无章节/平铺字段
        assert "chapters" not in stats
        assert "units" not in stats
        assert "sentences" not in stats
        # 课级统计列表(不含句子明细)
        lessons = stats["lessons"]
        assert [l["lesson_id"] for l in lessons] == ["l1", "l2"]
        assert lessons[0]["total_sentence_count"] == 2
        assert lessons[0]["learned_sentence_count"] == 1
        assert lessons[0]["progress"] == 0.4  # s1=0.8, s2 无状态=0, 均值
        assert "mastery_distribution" in lessons[0]
        # 有章教材下 lesson_count 与 summary 一致
        assert stats["summary"]["lesson_count"] == 2

    def test_no_chapter_chapter_detail_returns_lesson_list(self):
        states = [
            {"sentence_id": "s1", "skill_code": "translation",
             "status": STATUS_LEARNED, "mastery_score": 80},
        ]
        stats = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_noch",
            states=states, sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=[],
            detail="chapter",
        )
        assert stats["chapters"] == []
        assert [l["lesson_id"] for l in stats["lessons"]] == ["l1", "l2"]
        assert "sentences" not in stats
        assert "units" not in stats

    def test_no_states(self):
        stats = aggregate_progress(
            scholar_id="scholar_1", textbook_id="tb_1",
            states=[], sentences=self.SENTENCES,
            lessons=self.LESSONS, chapters=self.CHAPTERS,
        )
        assert stats["summary"]["textbook_progress"] == 0.0
        assert stats["summary"]["learned_sentence_count"] == 0
        assert stats["summary"]["total_sentence_count"] == 4
        # 未学习时每课仍返回条目(进度 0)
        assert len(stats["chapters"]) == 1
        assert len(stats["chapters"][0]["lessons"]) == 2


# ===========================================================================
# 查询接口 2/3 输出辅助(Phase 6)
# ===========================================================================


class TestQueryOutputHelpers:
    """接口 2/3 输出转换: 状态→数字 / 综合掌握度 / 6 级分布数组。"""

    def test_status_to_int(self):
        assert status_to_int("not_started") == 0
        assert status_to_int("learning") == 1
        assert status_to_int("learned") == 2
        assert status_to_int("mastered") == 3
        assert status_to_int("review_due") == 4
        assert status_to_int(None) == 0
        assert status_to_int("unknown") == 0

    def test_mastery_ratio_weighted_4_levels(self):
        # 全未学 → 0
        assert mastery_ratio({"total": 4, "not_started": 4}) == 0.0
        # 全掌握 → 1.0
        assert mastery_ratio({"total": 4, "mastered": 4}) == 1.0
        # learned 折半权重: learned=1 → 2/3 ≈ 0.6667(4 位小数)
        assert mastery_ratio({"total": 3, "learned": 3}) == pytest.approx(0.6667, abs=1e-4)
        # 文档示例 [40,0,20,40,0,0] → (2*20 + 3*40)/(3*100) ≈ 0.5333
        assert mastery_ratio(
            {"total": 100, "not_started": 40, "learned": 20, "mastered": 40}
        ) == pytest.approx(0.5333, abs=1e-4)
        # review_due 并入 mastered 档
        assert mastery_ratio({"total": 2, "review_due": 2}) == 1.0
        # 空分布 → 0
        assert mastery_ratio({}) == 0.0

    def test_mastery_ratio_content_total_includes_unlearned(self):
        # 3 句有记录(learning/learned/mastered), 内容共 4 句 → 未学 1 句按 0 档计入分母
        d = {"total": 3, "learning": 1, "learned": 1, "mastered": 1}
        assert mastery_ratio(d, 4) == pytest.approx(0.5)  # (1+2+3)/(3*4)
        # 不传 content_total 时仅按有记录句数 → 会虚高(0.6667)
        assert mastery_ratio(d) == pytest.approx(0.6667, abs=1e-4)
        # content_total 小于记录数时以记录数为准
        assert mastery_ratio(d, 2) == pytest.approx(0.6667, abs=1e-4)

    def test_status_distribution_array_6_slots(self):
        d = {"not_started": 4, "learning": 0, "learned": 2, "mastered": 4, "review_due": 0}
        assert status_distribution_array(d) == [4, 0, 2, 4, 0, 0]
        assert status_distribution_array({}) == [0, 0, 0, 0, 0, 0]
        # 5 级计数透传, 第 6 位恒 0
        assert status_distribution_array({"learned": 1, "review_due": 3}) == [0, 0, 1, 0, 3, 0]
