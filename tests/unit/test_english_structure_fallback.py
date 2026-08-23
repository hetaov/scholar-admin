"""E1.1/E1.2 兼容回退单元测试：textbook_v2 无内嵌结构时回退独立 chapter/lesson 集合

背景：标准内容管线 `write_content_v2` 产出的 textbook_v2 只有冗余计数，章节/课时
写在独立 `chapter` / `lesson` 集合（见 services/english/structure.py）。本文件覆盖
英语管理端在该形态下的行为（回归：内嵌结构教材行为不变）。

覆盖入口：
- _find_lesson（list/create 共用）
- list_english_lesson_sentences（lesson_title 从独立集合取）
- get_english_chapter_tree（树组装）
- list_english_textbook_stats（计数字段回退）
- validate_english_sentences（orphan 检测用独立集合索引）
"""
from __future__ import annotations

import asyncio

import pytest

from services.english.sentence_management import (
    LessonNotFoundError,
    _find_lesson,
    list_english_lesson_sentences,
)
from services.english.validation import (
    get_english_chapter_tree,
    list_english_textbook_stats,
    validate_english_sentences,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _seed_flat_textbook(
    db: FakeDB,
    *,
    textbook_id: str = "tb_std",
    chapter_id: str = "ch_1",
    lessons: tuple[tuple[str, str], ...] = (("ls_1", "L1"), ("ls_2", "L2")),
) -> None:
    """标准管线形态：textbook_v2 只有计数，层级在独立 chapter/lesson 集合。"""
    db.add(
        "textbook_v2",
        {
            "textbook_id": textbook_id,
            "title": "标准教材",
            "subject_type": "english",
            "chapter_count": 1,
            "lesson_count": len(lessons),
            "sentence_count": 0,
        },
    )
    db.add(
        "chapter",
        {
            "chapter_id": chapter_id,
            "textbook_id": textbook_id,
            "title": "Ch1",
            "order": 1,
        },
    )
    for order, (lid, title) in enumerate(lessons, start=1):
        db.add(
            "lesson",
            {
                "lesson_id": lid,
                "chapter_id": chapter_id,
                "textbook_id": textbook_id,
                "title": title,
                "order": order,
            },
        )


def _seed_embedded_textbook(db: FakeDB, *, textbook_id: str = "tb_emb") -> None:
    """内嵌结构形态：textbook_v2.chapters[].lessons[]（回归基线）。"""
    db.add(
        "textbook_v2",
        {
            "textbook_id": textbook_id,
            "title": "内嵌教材",
            "subject_type": "english",
            "chapters": [
                {
                    "chapter_id": "ch_1",
                    "title": "Unit 1",
                    "lessons": [{"lesson_id": "ls_1", "title": "L1"}],
                }
            ],
        },
    )


# ===========================================================================
# _find_lesson 回退（list/create 共用入口）
# ===========================================================================


class TestFindLessonFallback:
    def test_flat_fallback_finds_lesson(self):
        db = FakeDB()
        _seed_flat_textbook(db)
        _tb, lesson, chapter_id = _run(_find_lesson(db, "tb_std", "ls_2"))
        assert chapter_id == "ch_1"
        assert lesson["lesson_id"] == "ls_2"
        assert lesson["title"] == "L2"

    def test_flat_fallback_missing_raises(self):
        db = FakeDB()
        _seed_flat_textbook(db)
        with pytest.raises(LessonNotFoundError):
            _run(_find_lesson(db, "tb_std", "ls_nope"))

    def test_embedded_priority_unchanged(self):
        db = FakeDB()
        _seed_embedded_textbook(db)
        _tb, lesson, chapter_id = _run(_find_lesson(db, "tb_emb", "ls_1"))
        assert chapter_id == "ch_1"
        assert lesson["title"] == "L1"


# ===========================================================================
# list_english_lesson_sentences（lesson_title 从独立集合取）
# ===========================================================================


class TestListSentencesFallback:
    def test_list_sentences_flat(self):
        db = FakeDB()
        _seed_flat_textbook(db)
        db.add(
            "sentence_v2",
            {
                "sentence_id": "s1",
                "text": "Hello!",
                "textbook_id": "tb_std",
                "lesson_id": "ls_1",
                "chapter_id": "ch_1",
            },
        )
        data = _run(
            list_english_lesson_sentences(
                db, textbook_id="tb_std", lesson_id="ls_1"
            )
        )
        assert data["lesson_title"] == "L1"
        assert data["total"] == 1
        assert data["sentences"][0]["text"] == "Hello!"


# ===========================================================================
# get_english_chapter_tree（树组装回退）
# ===========================================================================


class TestChapterTreeFallback:
    def test_chapter_tree_flat(self):
        db = FakeDB()
        _seed_flat_textbook(db)
        data = _run(get_english_chapter_tree(db, textbook_id="tb_std"))
        assert data["title"] == "标准教材"
        assert len(data["chapters"]) == 1
        ch = data["chapters"][0]
        assert ch["chapter_id"] == "ch_1"
        assert ch["title"] == "Ch1"
        assert [ls["lesson_id"] for ls in ch["lessons"]] == ["ls_1", "ls_2"]

    def test_chapter_tree_embedded_unchanged(self):
        db = FakeDB()
        _seed_embedded_textbook(db)
        data = _run(get_english_chapter_tree(db, textbook_id="tb_emb"))
        assert len(data["chapters"]) == 1
        assert [ls["lesson_id"] for ls in data["chapters"][0]["lessons"]] == ["ls_1"]


# ===========================================================================
# list_english_textbook_stats（计数字段回退）
# ===========================================================================


class TestStatsFallback:
    def test_stats_counts_fallback(self):
        db = FakeDB()
        _seed_flat_textbook(
            db, lessons=(("ls_1", "L1"), ("ls_2", "L2"), ("ls_3", "L3"))
        )
        data = _run(list_english_textbook_stats(db))
        tb = next(t for t in data["textbooks"] if t["textbook_id"] == "tb_std")
        assert tb["chapter_count"] == 1
        assert tb["lesson_count"] == 3

    def test_stats_embedded_counts(self):
        db = FakeDB()
        _seed_embedded_textbook(db)
        data = _run(list_english_textbook_stats(db))
        tb = next(t for t in data["textbooks"] if t["textbook_id"] == "tb_emb")
        assert tb["chapter_count"] == 1
        assert tb["lesson_count"] == 1


# ===========================================================================
# validate_english_sentences（orphan 检测用独立集合索引）
# ===========================================================================


class TestValidateFallback:
    def test_validate_orphan_uses_flat_index(self):
        db = FakeDB()
        _seed_flat_textbook(db)
        db.add(
            "sentence_v2",
            {
                "sentence_id": "s1",
                "text": "Hello!",
                "translation": "你好",
                "textbook_id": "tb_std",
                "lesson_id": "ls_1",
                "chapter_id": "ch_1",
            },
        )
        db.add(
            "sentence_v2",
            {
                "sentence_id": "s2",
                "text": "Bye!",
                "translation": "再见",
                "textbook_id": "tb_std",
                "lesson_id": "ls_999",
                "chapter_id": "",
            },
        )
        result = _run(validate_english_sentences(db, textbook_id="tb_std"))
        orphan_ids = [i["sentence_id"] for i in result["issues"]["orphan_lesson"]]
        assert orphan_ids == ["s2"]
        assert result["summary"]["error_count"] == 1
