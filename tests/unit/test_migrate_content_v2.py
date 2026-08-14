"""services/migrate_content_v2.py 迁移逻辑单元测试(经 FakeDB)

覆盖 Phase 1 存量迁移:
- migrate_textbook_to_v2: 旧 textbook → textbook_v2(全量复制 + version=1)
- migrate_content_to_v2: 旧 unit → chapter/lesson + 旧 sentence → sentence_v2(回填引用)
- 幂等: 重复执行不产生重复数据
- 安全: 旧表只读, 绝不修改旧表数据
"""

from __future__ import annotations

import asyncio

from services.migrate_content_v2 import (
    NO_CHAPTER_TEXTBOOKS,
    migrate_content_to_v2,
    migrate_textbook_to_v2,
)
from tests.fakes.fake_db import FakeDB

# 旧表种子数据
TEXTBOOK_SEED = [
    {"_id": "tb_1", "title": "NCE Book 1", "grade": "1", "semester": "Book 1"},
    {"_id": "tb_noch", "title": "NCE Book 2", "grade": "1", "semester": "Book 2"},
]
UNIT_SEED = [
    {"unit_id": "unit_a", "title": "Unit A", "text_book_id": "tb_1", "unit_index": 1, "total_sentences": 2},
    {"unit_id": "unit_b", "title": "Unit B", "text_book_id": "tb_1", "unit_index": 2, "total_sentences": 2},
    {"unit_id": "unit_c", "title": "Unit C", "text_book_id": "tb_noch", "unit_index": 1, "total_sentences": 2},
    {"unit_id": "unit_d", "title": "Unit D", "text_book_id": "tb_noch", "unit_index": 2, "total_sentences": 2},
]
SENTENCE_SEED = [
    {"sentence_id": "sent_1", "unit_id": "unit_a", "index": 1, "text": "Hello", "text_book_id": "tb_1"},
    {"sentence_id": "sent_2", "unit_id": "unit_a", "index": 2, "text": "World", "text_book_id": "tb_1"},
    {"sentence_id": "sent_3", "unit_id": "unit_b", "index": 1, "text": "Goodbye", "text_book_id": "tb_1"},
    {"sentence_id": "sent_4", "unit_id": "unit_b", "index": 2, "text": "Friend", "text_book_id": "tb_1"},
    {"sentence_id": "sent_5", "unit_id": "unit_c", "index": 1, "text": "Nihao", "text_book_id": "tb_noch"},
    {"sentence_id": "sent_6", "unit_id": "unit_c", "index": 2, "text": "Zaijian", "text_book_id": "tb_noch"},
    {"sentence_id": "sent_7", "unit_id": "unit_d", "index": 1, "text": "Xiexie", "text_book_id": "tb_noch"},
    {"sentence_id": "sent_8", "unit_id": "unit_d", "index": 2, "text": "Buqi", "text_book_id": "tb_noch"},
]


def _make_db() -> FakeDB:
    return FakeDB(seed={
        "textbook": TEXTBOOK_SEED,
        "unit": UNIT_SEED,
        "sentence": SENTENCE_SEED,
    })


class TestMigrateTextbookToV2:
    def test_full_copy_with_version(self):
        db = _make_db()
        stats = asyncio.run(migrate_textbook_to_v2(db))
        assert stats["textbook_v2_created"] == 2
        v2 = db.all("textbook_v2")
        assert len(v2) == 2
        tb1 = next(t for t in v2 if t["_id"] == "tb_1")
        assert tb1["version"] == 1
        assert tb1["title"] == "NCE Book 1"

    def test_idempotent(self):
        db = _make_db()
        asyncio.run(migrate_textbook_to_v2(db))
        stats2 = asyncio.run(migrate_textbook_to_v2(db))
        assert stats2["textbook_v2_created"] == 0
        assert stats2["textbook_v2_skipped"] == 2
        assert len(db.all("textbook_v2")) == 2

    def test_old_table_not_modified(self):
        db = _make_db()
        asyncio.run(migrate_textbook_to_v2(db))
        assert db.all("textbook") == TEXTBOOK_SEED  # 旧表原样保留


class TestMigrateContentToV2:
    def test_creates_chapter_lesson_sentence(self):
        db = _make_db()
        stats = asyncio.run(migrate_content_to_v2(db))
        # 默认配置下 tb_1 与 tb_noch 各 1 章(每章 2 unit)
        assert stats["chapter_created"] == 2
        assert stats["lesson_created"] == 4
        assert stats["sentence_v2_created"] == 8
        assert len(db.all("chapter")) == 2
        assert len(db.all("lesson")) == 4
        assert len(db.all("sentence_v2")) == 8

    def test_lesson_id_reuses_unit_id(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        lessons = db.all("lesson")
        assert {l["lesson_id"] for l in lessons} == {"unit_a", "unit_b", "unit_c", "unit_d"}

    def test_sentence_v2_has_hierarchy_refs(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        sentences = db.all("sentence_v2")
        assert len(sentences) == 8
        for s in sentences:
            assert s["chapter_id"]
            assert s["lesson_id"] == s["unit_id"]
            assert s["textbook_id"] in ("tb_1", "tb_noch")

    def test_idempotent(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        stats2 = asyncio.run(migrate_content_to_v2(db))
        assert stats2["chapter_created"] == 0
        assert stats2["lesson_created"] == 0
        assert stats2["sentence_v2_created"] == 0
        assert len(db.all("chapter")) == 2
        assert len(db.all("lesson")) == 4
        assert len(db.all("sentence_v2")) == 8

    def test_old_tables_not_modified(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        assert db.all("textbook") == TEXTBOOK_SEED
        assert db.all("unit") == UNIT_SEED
        assert db.all("sentence") == SENTENCE_SEED


class TestNoChapterTextbook:
    def test_no_chapter_lessons_under_book(self, monkeypatch):
        """无章教材: 不创建 chapter, lesson 直接挂 book 下。"""
        monkeypatch.setattr("services.migrate_content_v2.NO_CHAPTER_TEXTBOOKS", {"tb_noch"})
        db = _make_db()
        stats = asyncio.run(migrate_content_to_v2(db))
        # tb_noch 无章: 不为其创建 chapter
        assert stats["chapter_created"] == 1  # 仅 tb_1 有 1 章
        chapters = db.all("chapter")
        assert all(c["textbook_id"] == "tb_1" for c in chapters)
        # tb_noch 的课挂在 book 下(chapter_id 为空)
        noch_lessons = [l for l in db.all("lesson") if l["textbook_id"] == "tb_noch"]
        assert len(noch_lessons) == 2
        assert all(l["chapter_id"] == "" for l in noch_lessons)
        # sentence_v2 同样 chapter_id 为空
        noch_sents = [s for s in db.all("sentence_v2") if s["textbook_id"] == "tb_noch"]
        assert len(noch_sents) == 4
        assert all(s["chapter_id"] == "" for s in noch_sents)

    def test_repair_removes_existing_chapters_and_clears_refs(self, monkeypatch):
        """修复无章教材历史数据: 删除既有 chapter, 清空 lesson/sentence_v2 的 chapter_id。"""
        monkeypatch.setattr("services.migrate_content_v2.NO_CHAPTER_TEXTBOOKS", {"tb_noch"})
        db = _make_db()
        # 先按有章迁移(tb_noch 不在配置中)产生历史数据
        monkeypatch.setattr("services.migrate_content_v2.NO_CHAPTER_TEXTBOOKS", set())
        asyncio.run(migrate_content_to_v2(db))
        assert len(db.all("chapter")) == 2  # tb_1 + tb_noch 各 1 章
        assert any(c["textbook_id"] == "tb_noch" for c in db.all("chapter"))
        # 再按无章配置重跑: 应删除 tb_noch 的 chapter 并置空引用
        monkeypatch.setattr("services.migrate_content_v2.NO_CHAPTER_TEXTBOOKS", {"tb_noch"})
        stats = asyncio.run(migrate_content_to_v2(db))
        assert stats.get("chapter_removed", 0) >= 1
        assert all(c["textbook_id"] == "tb_1" for c in db.all("chapter"))
        noch_lessons = [l for l in db.all("lesson") if l["textbook_id"] == "tb_noch"]
        assert all(l["chapter_id"] == "" for l in noch_lessons)
        noch_sents = [s for s in db.all("sentence_v2") if s["textbook_id"] == "tb_noch"]
        assert all(s["chapter_id"] == "" for s in noch_sents)

    def test_idempotent_no_chapter(self, monkeypatch):
        """无章教材迁移幂等: 重跑不重复创建 lesson/sentence。"""
        monkeypatch.setattr("services.migrate_content_v2.NO_CHAPTER_TEXTBOOKS", {"tb_noch"})
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        stats2 = asyncio.run(migrate_content_to_v2(db))
        assert stats2["lesson_created"] == 0
        assert stats2["sentence_v2_created"] == 0
        assert len(db.all("lesson")) == 4
        assert len(db.all("sentence_v2")) == 8
