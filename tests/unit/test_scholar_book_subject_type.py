"""scholar_book subject_type 多学科扩展 + getter 兼容层

覆盖：
  1. build_scholar_book_doc 默认写入 subject_type='english'（英语 0 迁移）
  2. build_scholar_book_doc 显式传 subject_type=math → 写入 math
  3. normalize_scholar_book_doc getter：存量无字段 → 注入 english（零写回）
  4. normalize_scholar_book_doc 不修改传入对象（返回副本）
  5. upsert_scholar_book 接受 subject_type 参数并落库
  6. list_scholar_books 支持 subject_type 过滤
"""
from __future__ import annotations

import pytest

from services.models_scholar_book import (
    SCHOLAR_BOOK,
    build_scholar_book_doc,
    list_scholar_books,
    normalize_scholar_book_doc,
    scholar_book_id,
    upsert_scholar_book,
)


# ===========================================================================
# 1. build_scholar_book_doc 默认 subject_type=english
# ===========================================================================


class TestBuildScholarBookDocSubjectType:
    def test_default_subject_type_english(self):
        """不传 subject_type → 默认写入 english（英语 0 迁移）。"""
        doc = build_scholar_book_doc(
            scholar_id="s1", textbook_id="tb_1", now=1000,
        )
        assert doc["subject_type"] == "english"
        # 原字段不变（向后兼容零破坏）
        assert doc["_id"] == "s1_tb_1"
        assert doc["scholar_id"] == "s1"
        assert doc["textbook_id"] == "tb_1"

    def test_explicit_subject_type_math(self):
        """显式传 subject_type=math → 写入 math。"""
        doc = build_scholar_book_doc(
            scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=1000,
        )
        assert doc["subject_type"] == "math"

    def test_explicit_subject_type_chinese(self):
        """显式传 subject_type=chinese → 写入 chinese（未来学科扩展）。"""
        doc = build_scholar_book_doc(
            scholar_id="s1", textbook_id="tb_yw_1",
            subject_type="chinese", now=1000,
        )
        assert doc["subject_type"] == "chinese"


# ===========================================================================
# 2. normalize_scholar_book_doc getter 兼容层
# ===========================================================================


class TestNormalizeScholarBookDocGetter:
    """normalize_scholar_book_doc 是读侧兼容入口：对 scholar_book 单条记录打标。"""

    def test_missing_subject_type_defaults_to_english(self):
        """存量老数据（完全无 subject_type）→ 返回 subject_type='english'。"""
        legacy = {
            "_id": "s1_tb_1",
            "scholar_id": "s1",
            "textbook_id": "tb_1",
            "status": "learning",
        }
        normalized = normalize_scholar_book_doc(legacy)
        assert normalized["subject_type"] == "english"

    def test_subject_type_none_defaults_to_english(self):
        """明确 subject_type=None → 返回 english。"""
        doc = {
            "scholar_id": "s1",
            "textbook_id": "tb_1",
            "subject_type": None,
        }
        normalized = normalize_scholar_book_doc(doc)
        assert normalized["subject_type"] == "english"

    def test_subject_type_empty_string_defaults_to_english(self):
        """subject_type='' 空串 → 返回 english。"""
        doc = {
            "scholar_id": "s1",
            "textbook_id": "tb_1",
            "subject_type": "",
        }
        normalized = normalize_scholar_book_doc(doc)
        assert normalized["subject_type"] == "english"

    def test_subject_type_explicit_keeps_value_math(self):
        """显式传 math 保留原值不覆盖。"""
        doc = {
            "scholar_id": "s1",
            "textbook_id": "tb_math_1",
            "subject_type": "math",
        }
        normalized = normalize_scholar_book_doc(doc)
        assert normalized["subject_type"] == "math"

    def test_subject_type_explicit_keeps_value_chinese(self):
        """显式传 chinese 保留原值（未来学科扩展）。"""
        normalized = normalize_scholar_book_doc(
            {"scholar_id": "s1", "textbook_id": "tb_yw_1", "subject_type": "chinese"}
        )
        assert normalized["subject_type"] == "chinese"

    def test_normalize_returns_copy_does_not_mutate_input(self):
        """兼容层不应修改传入对象（避免副作用污染调用方原始对象）。"""
        legacy = {"scholar_id": "s1", "textbook_id": "tb_1"}
        legacy_keys_before = set(legacy.keys())
        out = normalize_scholar_book_doc(legacy)
        assert out is not legacy
        assert set(legacy.keys()) == legacy_keys_before
        assert "subject_type" not in legacy


# ===========================================================================
# 3. upsert_scholar_book 接受 subject_type 参数
# ===========================================================================


class TestUpsertScholarBookSubjectType:
    @pytest.mark.asyncio
    async def test_upsert_default_subject_type_english(self, fake_db):
        """不传 subject_type → upsert 写入 english。"""
        doc = await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", now=1000,
        )
        assert doc["subject_type"] == "english"
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["subject_type"] == "english"

    @pytest.mark.asyncio
    async def test_upsert_explicit_subject_type_math(self, fake_db):
        """显式传 subject_type=math → upsert 写入 math。"""
        doc = await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=1000,
        )
        assert doc["subject_type"] == "math"
        records = fake_db.all(SCHOLAR_BOOK)
        assert records[0]["subject_type"] == "math"

    @pytest.mark.asyncio
    async def test_upsert_preserves_subject_type_on_update(self, fake_db):
        """更新断点时不丢失已写入的 subject_type。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=1000,
        )
        # 第二次 upsert 不传 subject_type → 保留原 math
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            current_lesson_id="l2", now=2000,
        )
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["subject_type"] == "math"


# ===========================================================================
# 4. list_scholar_books 支持 subject_type 过滤
# ===========================================================================


class TestListScholarBooksSubjectTypeFilter:
    @pytest.mark.asyncio
    async def test_filter_by_subject_type_math(self, fake_db):
        """subject_type=math 过滤：仅返回 math 教材关联。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_en_1",
            subject_type="english", now=1000,
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=2000,
        )
        # 过滤 math
        math_books = await list_scholar_books(
            fake_db, scholar_id="s1", subject_type="math"
        )
        assert len(math_books) == 1
        assert math_books[0]["textbook_id"] == "tb_math_1"

    @pytest.mark.asyncio
    async def test_filter_by_subject_type_english(self, fake_db):
        """subject_type=english 过滤：仅返回 english 教材关联。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_en_1",
            subject_type="english", now=1000,
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=2000,
        )
        en_books = await list_scholar_books(
            fake_db, scholar_id="s1", subject_type="english"
        )
        assert len(en_books) == 1
        assert en_books[0]["textbook_id"] == "tb_en_1"

    @pytest.mark.asyncio
    async def test_no_subject_type_returns_all(self, fake_db):
        """不传 subject_type → 返回全部（向后兼容）。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_en_1",
            subject_type="english", now=1000,
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_math_1",
            subject_type="math", now=2000,
        )
        all_books = await list_scholar_books(fake_db, scholar_id="s1")
        assert len(all_books) == 2
