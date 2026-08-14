"""Phase 1 内容模型分层集成测试

覆盖:
- _write_to_db(构建流程)双写: 旧表(unit/paragraph/sentence)照旧 + 新表(textbook_v2/chapter/lesson/sentence_v2)
- _store_recognition_result(视觉识别)双写: 旧表照旧 + 新表层级完整
- 旧接口行为不变(由既有 test_tracking_endpoints 兜底)
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


class TestBuildDoubleWrite:
    def test_build_writes_old_and_new_tables(self, monkeypatch):
        db = FakeDB()
        monkeypatch.setattr("services.routes_build.get_db", lambda: db)
        result = asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))

        # 旧表照旧
        assert len(db.all("textbook")) == 1
        assert len(db.all("unit")) == 2
        assert len(db.all("paragraph")) == 2
        assert len(db.all("sentence")) == 3

        # 新表双写
        assert len(db.all("textbook_v2")) == 1
        assert len(db.all("chapter")) >= 1
        assert len(db.all("lesson")) == 2
        assert len(db.all("sentence_v2")) == 3

        # 返回兼容旧接口结构
        assert result["unit_count"] == 2
        assert result["total_sentences"] == 3
        assert result["v2"]["sentence_count"] == 3

    def test_lesson_id_reuses_unit_id(self, monkeypatch):
        db = FakeDB()
        monkeypatch.setattr("services.routes_build.get_db", lambda: db)
        asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))
        unit_ids = {u["unit_id"] for u in db.all("unit")}
        lesson_ids = {l["lesson_id"] for l in db.all("lesson")}
        assert lesson_ids == unit_ids

    def test_sentence_v2_hierarchy_consistent(self, monkeypatch):
        db = FakeDB()
        monkeypatch.setattr("services.routes_build.get_db", lambda: db)
        asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))
        for s in db.all("sentence_v2"):
            assert s["chapter_id"]
            lesson = next(l for l in db.all("lesson") if l["lesson_id"] == s["lesson_id"])
            assert lesson["chapter_id"] == s["chapter_id"]

    def test_old_tables_behavior_unchanged(self, monkeypatch):
        """旧表写入内容与 Phase 1 之前完全一致(不新增/删除旧表字段逻辑)。"""
        db = FakeDB()
        monkeypatch.setattr("services.routes_build.get_db", lambda: db)
        asyncio.run(_write_to_db(BUILD_CONTENT, "Test English"))
        sent = db.all("sentence")[0]
        assert set(sent) >= {"sentence_id", "unit_id", "paragraph_id", "index", "text", "text_book_id"}


class TestVisionDoubleWrite:
    def test_vision_writes_old_and_new_tables(self):
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
        # 旧表照旧
        assert len(db.all("unit")) == 1
        assert len(db.all("paragraph")) == 1
        assert len(db.all("sentence")) == 2
        # 新表双写
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
