"""单元测试：scripts/repair_lesson_order 一次性修复脚本纯函数

覆盖（2026-09-03 统一修复「换教材即复现首个任务=第 8 课」的数据层）：
- parse_lesson_no：与小程序前端 parseLessonNumber 同口径的课号解析
- plan_book_order：
  · 单章教材（Unit 1..8 + Review）→ Unit1..8 后 Review（无课号保底排桶末）
  · 多章教材（chapter 顺序乱排）→ 章序 → 章内课号
  · 无章教材 → 全书标题课号
  · 孤儿课（chapter_id 不在章表）排最后
- build_plan：order 全 1 脏数据 → 1..N 唯一递增
"""
from __future__ import annotations

from scripts.repair_lesson_order import build_plan, parse_lesson_no, plan_book_order


def _lesson(lesson_id: str, chapter_id: str | None, title: str, order: int = 1) -> dict:
    return {
        "_id": lesson_id,
        "lesson_id": lesson_id,
        "chapter_id": chapter_id,
        "title": title,
        "order": order,
    }


def _chapter(chapter_id: str, order: int, title: str = "Chapter") -> dict:
    return {
        "_id": chapter_id,
        "chapter_id": chapter_id,
        "title": title,
        "order": order,
    }


class TestParseLessonNo:
    """与前端 parseLessonNumber 对齐的课号解析。"""

    def test_keyword_formats(self):
        assert parse_lesson_no("Lesson 4") == 4
        assert parse_lesson_no("Lesson 4: 商务谈判与报价") == 4
        assert parse_lesson_no("第 12 课") == 12
        assert parse_lesson_no("UNIT 3") == 3
        assert parse_lesson_no("Unit 8 Joy in the Air") == 8
        assert parse_lesson_no("Module 3: Food") == 3
        assert parse_lesson_no("Module 1 Unit 2") == 1  # 取 Module

    def test_leading_number_formats(self):
        assert parse_lesson_no("1. Greeting") == 1
        assert parse_lesson_no("8、标题") == 8
        assert parse_lesson_no("12：标题") == 12

    def test_unparseable_returns_none(self):
        assert parse_lesson_no("Review Let's Go Camping!") is None
        assert parse_lesson_no("Come on in") is None
        assert parse_lesson_no("") is None
        assert parse_lesson_no(None) is None


class TestPlanBookOrder:
    def test_single_chapter_unit_order_with_review_last(self):
        """tb_5aefa2 形态：单章 + Unit 1..8 + Review，输入乱序也能还原。"""
        ch = _chapter("c1", 1)
        lessons = [
            _lesson("u8", "c1", "Unit 8 Joy in the Air"),
            _lesson("u1", "c1", "Unit 1 Come on in"),
            _lesson("u5", "c1", "Unit 5 She Helps Me a Lot"),
            _lesson("review", "c1", "Review Let's Go Camping!"),
            _lesson("u2", "c1", "Unit 2 Help yourself"),
            _lesson("u3", "c1", "Unit 3 I'm the Chef Today"),
            _lesson("u4", "c1", "Unit 4 Help out in the Kitchen"),
            _lesson("u6", "c1", "Unit 6 I Love My Family"),
            _lesson("u7", "c1", "Unit 7 Family Time"),
        ]
        ordered = plan_book_order([ch], lessons)
        titles = [l["title"] for l in ordered]
        assert titles[0] == "Unit 1 Come on in"
        assert titles[-1] == "Review Let's Go Camping!"  # 无课号排桶末
        assert titles[0:8] == [
            "Unit 1 Come on in",
            "Unit 2 Help yourself",
            "Unit 3 I'm the Chef Today",
            "Unit 4 Help out in the Kitchen",
            "Unit 5 She Helps Me a Lot",
            "Unit 6 I Love My Family",
            "Unit 7 Family Time",
            "Unit 8 Joy in the Air",
        ]

    def test_multi_chapter_by_chapter_order_then_lesson_no(self):
        """新概念形态：多章（chapter 输入乱序），章间按 chapter.order，章内按 Lesson N。"""
        chapters = [
            _chapter("c2", 2),
            _chapter("c1", 1),
            _chapter("c3", 3),
        ]
        lessons = [
            _lesson("l10", "c2", "Lesson 10"),
            _lesson("l9", "c2", "Lesson 9"),
            _lesson("l1", "c1", "Lesson 1: Excuse me!"),
            _lesson("l2", "c1", "Lesson 2: Excuse me?"),
            _lesson("l50", "c3", "Lesson 50"),
        ]
        ordered = plan_book_order(chapters, lessons)
        assert [l["title"] for l in ordered] == [
            "Lesson 1: Excuse me!",
            "Lesson 2: Excuse me?",
            "Lesson 9",
            "Lesson 10",
            "Lesson 50",
        ]

    def test_chapterless_book_sorted_by_title(self):
        """无章教材：lesson 直挂 book（chapter_id 为空），全书按标题课号。"""
        lessons = [
            _lesson("u3", None, "Unit 3"),
            _lesson("u1", None, "Unit 1"),
            _lesson("u2", None, "Unit 2"),
        ]
        ordered = plan_book_order([], lessons)
        assert [l["title"] for l in ordered] == ["Unit 1", "Unit 2", "Unit 3"]

    def test_orphan_lessons_pushed_last(self):
        """孤儿课（chapter_id 不在章表）排在有章课程之后。"""
        chapters = [_chapter("c1", 1)]
        lessons = [
            _lesson("o1", "c_ghost", "Unit 99 Orphan"),
            _lesson("u1", "c1", "Unit 1"),
            _lesson("u2", "c1", "Unit 2"),
        ]
        ordered = plan_book_order(chapters, lessons)
        assert [l["title"] for l in ordered] == ["Unit 1", "Unit 2", "Unit 99 Orphan"]

    def test_stable_when_all_unparseable(self):
        """标题全部无课号：保持原相对顺序（DB order ASC 返回序）。"""
        lessons = [
            _lesson("a", None, "Alpha"),
            _lesson("b", None, "Beta"),
            _lesson("c", None, "Gamma"),
        ]
        ordered = plan_book_order([], lessons)
        assert [l["title"] for l in ordered] == ["Alpha", "Beta", "Gamma"]


class TestBuildPlan:
    def test_dirty_order_all_ones_rewritten_unique(self):
        """线上脏数据（order 全 1）→ 计划重写为 1..N 唯一递增。"""
        ch = _chapter("c1", 1)
        lessons = [
            _lesson("u1", "c1", "Unit 1 Come on in", order=1),
            _lesson("u2", "c1", "Unit 2 Help yourself", order=1),
            _lesson("u8", "c1", "Unit 8 Joy in the Air", order=1),
        ]
        plan = build_plan([ch], lessons)
        assert [p["old_order"] for p in plan] == [1, 1, 1]
        assert [p["new_order"] for p in plan] == [1, 2, 3]
        assert [p["lesson_id"] for p in plan] == ["u1", "u2", "u8"]

    def test_unique_order_unchanged(self):
        """order 已正确（1..N）→ 计划零变更（幂等，可重复执行）。"""
        ch = _chapter("c1", 1)
        lessons = [
            _lesson("u1", "c1", "Unit 1", order=1),
            _lesson("u2", "c1", "Unit 2", order=2),
        ]
        plan = build_plan([ch], lessons)
        assert [(p["old_order"], p["new_order"]) for p in plan] == [(1, 1), (2, 2)]
