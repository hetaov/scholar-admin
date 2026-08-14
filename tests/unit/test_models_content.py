"""services/models_content.py 纯函数单元测试

覆盖 Phase 1 内容模型分层:
- group_units_into_chapters: unit → chapter 分组规则
- build_textbook_v2_doc / build_chapter_doc / build_lesson_doc / build_sentence_v2_doc: 字段与层级引用
- write_content_v2: 双写新表(经 FakeDB 验证)
"""

from __future__ import annotations

import asyncio

from services.models_content import (
    build_chapter_doc,
    build_lesson_doc,
    build_sentence_v2_doc,
    build_textbook_v2_doc,
    get_lessons_by_textbook,
    group_units_into_chapters,
    write_content_v2,
)
from tests.fakes.fake_db import FakeDB


# ---------------------------------------------------------------------------
# group_units_into_chapters
# ---------------------------------------------------------------------------


class TestGroupUnitsIntoChapters:
    def test_empty_units(self):
        assert group_units_into_chapters([]) == []

    def test_less_than_one_group(self):
        units = [{"unit_id": "u1"}, {"unit_id": "u2"}]
        groups = group_units_into_chapters(units)
        assert len(groups) == 1
        assert groups[0]["chapter_index"] == 1
        assert [u["unit_id"] for u in groups[0]["units"]] == ["u1", "u2"]

    def test_multiple_groups(self):
        units = [{"unit_id": f"u{i}"} for i in range(1, 10)]
        groups = group_units_into_chapters(units, units_per_chapter=3)
        assert len(groups) == 3
        assert [g["chapter_index"] for g in groups] == [1, 2, 3]
        assert [len(g["units"]) for g in groups] == [3, 3, 3]

    def test_remainder_group(self):
        units = [{"unit_id": f"u{i}"} for i in range(1, 11)]  # 10 units
        groups = group_units_into_chapters(units, units_per_chapter=3)
        assert len(groups) == 4
        assert [len(g["units"]) for g in groups] == [3, 3, 3, 1]

    def test_zero_units_per_chapter_all_in_one(self):
        units = [{"unit_id": f"u{i}"} for i in range(1, 10)]
        groups = group_units_into_chapters(units, units_per_chapter=0)
        assert len(groups) == 1
        assert len(groups[0]["units"]) == 9


# ---------------------------------------------------------------------------
# build_* 文档字段
# ---------------------------------------------------------------------------


class TestBuildDocs:
    def test_build_textbook_v2_doc(self):
        doc = build_textbook_v2_doc(
            "tb_1", "NCE", grade="1", level="Book 1",
            chapter_count=2, lesson_count=10, sentence_count=50, now=1000,
        )
        assert doc["_id"] == "tb_1"
        assert doc["textbook_id"] == "tb_1"
        assert doc["version"] == 1
        assert doc["chapter_count"] == 2
        assert doc["lesson_count"] == 10
        assert doc["sentence_count"] == 50

    def test_build_chapter_doc(self):
        doc = build_chapter_doc("chapter_x", "tb_1", 1, "Chapter 1", 3, now=1000)
        assert doc["_id"] == "chapter_x"
        assert doc["chapter_id"] == "chapter_x"
        assert doc["textbook_id"] == "tb_1"
        assert doc["order"] == 1
        assert doc["lesson_count"] == 3

    def test_build_lesson_doc(self):
        doc = build_lesson_doc("unit_3", "chapter_x", "tb_1", 1, "Lesson 1", 5, now=1000)
        assert doc["_id"] == "unit_3"  # lesson_id 沿用旧 unit_id
        assert doc["lesson_id"] == "unit_3"
        assert doc["chapter_id"] == "chapter_x"
        assert doc["textbook_id"] == "tb_1"
        assert doc["sentence_count"] == 5

    def test_build_sentence_v2_doc_keeps_old_fields(self):
        old = {
            "sentence_id": "sent_1",
            "unit_id": "unit_3",
            "index": 2,
            "text": "Hello",
            "translation": "你好",
            "text_book_id": "tb_1",
            "keywords": ["a"],
        }
        doc = build_sentence_v2_doc(old, "chapter_x", "unit_3", "tb_1", now=1000)
        assert doc["_id"] == "sent_1"
        assert doc["chapter_id"] == "chapter_x"
        assert doc["lesson_id"] == "unit_3"
        assert doc["textbook_id"] == "tb_1"
        assert doc["unit_id"] == "unit_3"  # 过渡期保留
        assert doc["text_book_id"] == "tb_1"  # 过渡期保留
        assert doc["order"] == 2


# ---------------------------------------------------------------------------
# write_content_v2(经 FakeDB)
# ---------------------------------------------------------------------------


class TestWriteContentV2:
    def _units(self, n=3, per=2):
        units = []
        for i in range(1, n + 1):
            units.append({
                "unit_id": f"unit_{i}",
                "unit_title": f"Lesson {i}",
                "sentences": [
                    {
                        "sentence_id": f"sent_{i}_{j}",
                        "unit_id": f"unit_{i}",
                        "index": j,
                        "text": f"Text {i}-{j}",
                        "text_book_id": "tb_1",
                    }
                    for j in range(1, per + 1)
                ],
            })
        return units

    async def _write(self, db, textbook_id="tb_1", n=3):
        return await write_content_v2(
            db,
            textbook_id=textbook_id,
            textbook_title="NCE",
            grade="1",
            level="Book 1",
            units=self._units(n),
            now=1000,
            units_per_chapter=2,
        )

    def test_writes_all_new_tables(self):
        db = FakeDB()
        asyncio.run(self._write(db))
        assert len(db.all("textbook_v2")) == 1
        assert len(db.all("chapter")) == 2  # 3 units / 2 per chapter
        assert len(db.all("lesson")) == 3
        assert len(db.all("sentence_v2")) == 6

    def test_sentence_v2_has_lesson_and_chapter_ref(self):
        db = FakeDB()
        asyncio.run(self._write(db))
        sentences = db.all("sentence_v2")
        for s in sentences:
            assert s["chapter_id"]
            assert s["lesson_id"] == s["unit_id"]  # lesson_id 与旧 unit_id 一致

    def test_lesson_chapter_mapping(self):
        db = FakeDB()
        asyncio.run(self._write(db))
        lessons = db.all("lesson")
        # 前 2 个 unit 在同一章, 第 3 个 unit 在下一章
        assert lessons[0]["chapter_id"] == lessons[1]["chapter_id"]
        assert lessons[1]["chapter_id"] != lessons[2]["chapter_id"]

    def test_idempotent_upsert_textbook_v2(self):
        db = FakeDB()
        asyncio.run(self._write(db, n=2))
        stats2 = asyncio.run(self._write(db, n=1))
        assert stats2["chapter_count"] == 1
        assert stats2["lesson_count"] == 1
        assert len(db.all("textbook_v2")) == 1  # 不重复创建
        assert len(db.all("lesson")) == 3

    def test_chapterless_writes_no_chapters(self):
        """无章教材: Book → Lesson → Sentence, 不创建 chapter。"""
        db = FakeDB()
        asyncio.run(write_content_v2(
            db,
            textbook_id="tb_noch",
            textbook_title="NCE",
            grade="1",
            level="Book 1",
            units=self._units(n=3),
            now=1000,
            chapterless=True,
        ))
        assert len(db.all("chapter")) == 0
        lessons = db.all("lesson")
        assert len(lessons) == 3
        for l in lessons:
            assert l["chapter_id"] == ""
        for s in db.all("sentence_v2"):
            assert s["chapter_id"] == ""
        tbs = asyncio.run(db.query(collection="textbook_v2", where={"_id": "tb_noch"}))
        tb = tbs["records"][0]
        assert tb["chapter_count"] == 0
        assert tb["lesson_count"] == 3

    def test_get_lessons_by_textbook_returns_chapterless_lessons(self):
        db = FakeDB()
        asyncio.run(write_content_v2(
            db,
            textbook_id="tb_noch",
            textbook_title="NCE",
            grade="1",
            level="Book 1",
            units=self._units(n=3),
            now=1000,
            chapterless=True,
        ))
        lessons = asyncio.run(get_lessons_by_textbook(db, "tb_noch"))
        assert [l["lesson_id"] for l in lessons] == ["unit_1", "unit_2", "unit_3"]
        # 无章教材: 不按 chapter_id 过滤也能查到全部课
        assert all(l["chapter_id"] == "" for l in lessons)
