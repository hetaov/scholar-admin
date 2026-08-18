"""Phase 1 内容模型分层集成测试

覆盖:
- _write_to_db(构建流程): 只写新表(textbook_v2/chapter/lesson/sentence_v2), 旧表已下线(Phase 6)
- _store_recognition_result(视觉识别): 只写新表, 层级完整
"""

from __future__ import annotations

import asyncio
import logging

from services.routes_build import _write_to_db
from services.routes_vision import _store_recognition_result
from tests.fakes.fake_db import FakeDB

logging.disable(logging.CRITICAL)

BUILD_CONTENT = {
    "textbook_info": {
        "name": "Test English",
        "grade": "3",
        "semester": "Book 3",
    },
    "units": [
        {
            "unit_index": 1,
            "unit_title": "Lesson One",
            "topic": "Greetings",
            "sentences": [
                {"text": "Hello world", "translation": "你好世界"},
                {"text": "How are you", "translation": "你好吗"},
            ],
        },
        {
            "unit_index": 2,
            "unit_title": "Lesson Two",
            "topic": "Numbers",
            "sentences": [
                {"text": "One two three", "translation": "一二三"},
            ],
        },
    ],
}


class TestBuildNewTables:
    def test_build_writes_new_tables(self, fake_db):
        # fake_db 由 tests/integration/conftest.py autouse 注入(方式 A),无需手动 setattr
        result = asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))

        # 只写新表(旧表 Phase 6 已下线, 不再写入)
        assert len(fake_db.all("textbook")) == 0
        assert len(fake_db.all("unit")) == 0
        assert len(fake_db.all("paragraph")) == 0
        assert len(fake_db.all("sentence")) == 0
        assert len(fake_db.all("textbook_v2")) == 1
        assert len(fake_db.all("chapter")) >= 1
        assert len(fake_db.all("lesson")) == 2
        assert len(fake_db.all("sentence_v2")) == 3

        # 返回结构
        assert result["unit_count"] == 2
        assert result["total_sentences"] == 3
        assert result["v2"]["sentence_count"] == 3
        assert all("lesson_id" in u for u in result["units"])
        assert all("unit_id" not in u for u in result["units"])

    def test_sentence_v2_hierarchy_consistent(self, fake_db):
        asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))
        for s in fake_db.all("sentence_v2"):
            assert s["chapter_id"]
            lesson = next(l for l in fake_db.all("lesson") if l["lesson_id"] == s["lesson_id"])
            assert lesson["chapter_id"] == s["chapter_id"]
            assert "unit_id" not in s
            assert "text_book_id" not in s


class TestVisionNewTables:
    def test_vision_writes_new_tables(self):
        db = FakeDB()
        result = {
            "title": "Photo Lesson",
            "sentences": [
                {"index": 1, "text": "A", "translation": "甲"},
                {"index": 2, "text": "B", "translation": "乙"},
            ],
        }
        info = asyncio.run(_store_recognition_result(db, result, "upload", "tb_1"))

        assert info["sentence_count"] == 2
        assert "lesson_id" in info
        assert "unit_id" not in info
        # 只写新表(旧表 Phase 6 已下线)
        assert len(db.all("unit")) == 0
        assert len(db.all("paragraph")) == 0
        assert len(db.all("sentence")) == 0
        assert len(db.all("textbook_v2")) == 1
        assert len(db.all("chapter")) == 1
        assert len(db.all("lesson")) == 1
        assert len(db.all("sentence_v2")) == 2

    def test_vision_without_textbook_skips_textbook_v2(self):
        db = FakeDB()
        result = {"title": "Free Image", "sentences": [{"index": 1, "text": "X", "translation": "未知"}]}
        asyncio.run(_store_recognition_result(db, result, "upload", None))
        assert len(db.all("textbook_v2")) == 0  # 无教材不建 textbook_v2
        assert len(db.all("lesson")) == 1  # 但层级仍完整
        assert len(db.all("sentence_v2")) == 1
