"""单元测试:学者×教材关联模型(Phase 5) — services.models_scholar_book

覆盖:
- 复合键生成(scholar_book_id)
- 文档构建(build_scholar_book_doc)
- upsert 语义:首次插入 / 断点更新 / 重复加入幂等(同一 学者×教材 仅一条)
- total_time_spent 增量累加 / last_studied_at 刷新
- 会话结算回写(touch_scholar_book,无 textbook_id 时不落库)
"""

from __future__ import annotations

import pytest

from services.models_scholar_book import (
    BOOK_STATUS_LEARNING,
    SCHOLAR_BOOK,
    build_scholar_book_doc,
    list_scholar_books,
    scholar_book_id,
    touch_scholar_book,
    upsert_scholar_book,
)
from tests.fakes.fake_db import FakeDB


class TestScholarBookId:
    def test_composite_key(self):
        assert scholar_book_id("s1", "tb_1") == "s1_tb_1"

    def test_distinguishes_order(self):
        assert scholar_book_id("s1", "tb_2") != scholar_book_id("s2", "tb_1")


class TestBuildDoc:
    def test_full_doc(self):
        doc = build_scholar_book_doc(
            scholar_id="s1",
            textbook_id="tb_1",
            current_chapter_id="c1",
            current_lesson_id="l1",
            total_time_spent=60,
            last_studied_at=1000,
            started_at=900,
            now=2000,
        )
        assert doc["_id"] == "s1_tb_1"
        assert doc["scholar_book_id"] == "s1_tb_1"
        assert doc["scholar_id"] == "s1"
        assert doc["textbook_id"] == "tb_1"
        assert doc["current_chapter_id"] == "c1"
        assert doc["current_lesson_id"] == "l1"
        assert doc["total_time_spent"] == 60
        assert doc["last_studied_at"] == 1000
        assert doc["started_at"] == 900
        assert doc["status"] == BOOK_STATUS_LEARNING
        assert doc["created_at"] == 2000
        assert doc["updated_at"] == 2000

    def test_defaults(self):
        doc = build_scholar_book_doc(scholar_id="s1", textbook_id="tb_1", now=3000)
        assert doc["total_time_spent"] == 0
        assert doc["last_studied_at"] == 3000
        assert doc["current_chapter_id"] is None
        assert doc["current_lesson_id"] is None


class TestUpsert:
    @pytest.mark.asyncio
    async def test_first_insert(self, fake_db):
        doc = await upsert_scholar_book(
            fake_db,
            scholar_id="s1",
            textbook_id="tb_1",
            current_chapter_id="c1",
            current_lesson_id="l1",
            last_studied_at=1000,
            now=2000,
        )
        assert doc["_id"] == "s1_tb_1"
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    @pytest.mark.asyncio
    async def test_repeated_join_is_idempotent(self, fake_db):
        """重复加入(更新断点)不产生第二条记录。"""
        await upsert_scholar_book(
            fake_db,
            scholar_id="s1",
            textbook_id="tb_1",
            current_chapter_id="c1",
            current_lesson_id="l1",
            now=1000,
        )
        await upsert_scholar_book(
            fake_db,
            scholar_id="s1",
            textbook_id="tb_1",
            current_chapter_id="c2",
            current_lesson_id="l2",
            now=2000,
        )
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["current_chapter_id"] == "c2"
        assert records[0]["current_lesson_id"] == "l2"

    @pytest.mark.asyncio
    async def test_time_delta_accumulates(self, fake_db):
        """多次结算 total_time_spent 增量累加。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", time_delta_sec=30, now=1000
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", time_delta_sec=45, now=2000
        )
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["total_time_spent"] == 75

    @pytest.mark.asyncio
    async def test_refresh_last_studied_at(self, fake_db):
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", last_studied_at=100, now=1000
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", last_studied_at=500, now=2000
        )
        records = fake_db.all(SCHOLAR_BOOK)
        assert records[0]["last_studied_at"] == 500

    @pytest.mark.asyncio
    async def test_scholar_book_isolated_by_scholar(self, fake_db):
        """不同学者对同一教材各自独立成记录。"""
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", now=1000
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s2", textbook_id="tb_1", now=1000
        )
        assert len(fake_db.all(SCHOLAR_BOOK)) == 2


class TestListAndTouch:
    @pytest.mark.asyncio
    async def test_list_scholar_books(self, fake_db):
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_1", now=1000
        )
        await upsert_scholar_book(
            fake_db, scholar_id="s1", textbook_id="tb_2", now=2000
        )
        books = await list_scholar_books(fake_db, scholar_id="s1")
        assert len(books) == 2
        # 按 last_studied_at 降序: tb_2 在前
        assert books[0]["textbook_id"] == "tb_2"

    @pytest.mark.asyncio
    async def test_touch_updates_time_and_stamp(self, fake_db):
        book = await touch_scholar_book(
            fake_db,
            scholar_id="s1",
            textbook_id="tb_1",
            last_studied_at=1000,
            time_delta_sec=60,
            now=1000,
        )
        assert book is not None
        assert book["total_time_spent"] == 60
        assert book["last_studied_at"] == 1000
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    @pytest.mark.asyncio
    async def test_touch_without_textbook_returns_none(self, fake_db):
        book = await touch_scholar_book(
            fake_db, scholar_id="s1", textbook_id=None, last_studied_at=1000, now=1000
        )
        assert book is None
        assert fake_db.all(SCHOLAR_BOOK) == []
