"""services/migrate_content_v2.py 迁移逻辑单元测试(经 FakeDB)

覆盖 Phase 1 存量迁移:
- migrate_textbook_to_v2: 旧 textbook → textbook_v2(全量复制 + version=1)
- migrate_content_to_v2: 旧 unit → chapter/lesson + 旧 sentence → sentence_v2(回填引用)
- 幂等: 重复执行不产生重复数据
- 安全: 旧表只读, 绝不修改旧表数据
"""

from __future__ import annotations

import asyncio

from services.migrate_content_v2 import migrate_content_to_v2, migrate_textbook_to_v2
from tests.fakes.fake_db import FakeDB

# 旧表种子数据
TEXTBOOK_SEED = [
    {"_id": "tb_1", "title": "NCE Book 1", "grade": "1", "semester": "Book 1"},
]
UNIT_SEED = [
    {"unit_id": "unit_a", "title": "Unit A", "text_book_id": "tb_1", "unit_index": 1, "total_sentences": 2},
    {"unit_id": "unit_b", "title": "Unit B", "text_book_id": "tb_1", "unit_index": 2, "total_sentences": 2},
]
SENTENCE_SEED = [
    {"sentence_id": "sent_1", "unit_id": "unit_a", "index": 1, "text": "Hello", "text_book_id": "tb_1"},
    {"sentence_id": "sent_2", "unit_id": "unit_a", "index": 2, "text": "World", "text_book_id": "tb_1"},
    {"sentence_id": "sent_3", "unit_id": "unit_b", "index": 1, "text": "Goodbye", "text_book_id": "tb_1"},
    {"sentence_id": "sent_4", "unit_id": "unit_b", "index": 2, "text": "Friend", "text_book_id": "tb_1"},
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
        assert stats["textbook_v2_created"] == 1
        v2 = db.all("textbook_v2")
        assert len(v2) == 1
        assert v2[0]["_id"] == "tb_1"
        assert v2[0]["version"] == 1
        assert v2[0]["title"] == "NCE Book 1"

    def test_idempotent(self):
        db = _make_db()
        asyncio.run(migrate_textbook_to_v2(db))
        stats2 = asyncio.run(migrate_textbook_to_v2(db))
        assert stats2["textbook_v2_created"] == 0
        assert stats2["textbook_v2_skipped"] == 1
        assert len(db.all("textbook_v2")) == 1

    def test_old_table_not_modified(self):
        db = _make_db()
        asyncio.run(migrate_textbook_to_v2(db))
        assert db.all("textbook") == TEXTBOOK_SEED  # 旧表原样保留


class TestMigrateContentToV2:
    def test_creates_chapter_lesson_sentence(self):
        db = _make_db()
        stats = asyncio.run(migrate_content_to_v2(db))
        assert stats["chapter_created"] == 1  # 2 units → 1 章(默认每章 8 unit)
        assert stats["lesson_created"] == 2
        assert stats["sentence_v2_created"] == 4
        assert len(db.all("chapter")) == 1
        assert len(db.all("lesson")) == 2
        assert len(db.all("sentence_v2")) == 4

    def test_lesson_id_reuses_unit_id(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        lessons = db.all("lesson")
        assert {l["lesson_id"] for l in lessons} == {"unit_a", "unit_b"}

    def test_sentence_v2_has_hierarchy_refs(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        sentences = db.all("sentence_v2")
        assert len(sentences) == 4
        for s in sentences:
            assert s["chapter_id"]
            assert s["lesson_id"] == s["unit_id"]
            assert s["textbook_id"] == "tb_1"

    def test_idempotent(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        stats2 = asyncio.run(migrate_content_to_v2(db))
        assert stats2["chapter_created"] == 0
        assert stats2["lesson_created"] == 0
        assert stats2["sentence_v2_created"] == 0
        assert len(db.all("chapter")) == 1
        assert len(db.all("lesson")) == 2
        assert len(db.all("sentence_v2")) == 4

    def test_old_tables_not_modified(self):
        db = _make_db()
        asyncio.run(migrate_content_to_v2(db))
        assert db.all("textbook") == TEXTBOOK_SEED
        assert db.all("unit") == UNIT_SEED
        assert db.all("sentence") == SENTENCE_SEED
