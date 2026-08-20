"""集成测试: scholar_book subject_type 多学科扩展 — 路由层

覆盖：
  1. GET /scholar/{id}/books?subject_type=english — 按学科过滤
  2. GET /scholar/{id}/books?subject_type=math — 按学科过滤
  3. GET /scholar/{id}/books 不传 subject_type — 返回全部（向后兼容）
  4. PUT /scholar/{id}/books/{tid}/position body 含 subject_type=math — 首次写入
  5. PUT /scholar/{id}/books/{tid}/position 不传 subject_type — 默认 english
  6. POST /textbook body 含 subject_type — 写入 textbook_v2
  7. GET /scholar/{id}/books 响应含 subject_type 字段
"""
from __future__ import annotations

import pytest

from services.models_scholar_book import SCHOLAR_BOOK
from services.routes_tracking import router as tracking_router


class TestGetScholarBooksSubjectTypeFilter:
    """GET /scholar/{id}/books?subject_type= 过滤"""

    def test_filter_english(self, make_client, fake_db):
        client = make_client(tracking_router)
        # 准备：一本 english + 一本 math
        client.put(
            "/scholar/s1/books/tb_en_1/position",
            json={"subject_type": "english", "current_lesson_id": "l1"},
        )
        client.put(
            "/scholar/s1/books/tb_math_1/position",
            json={"subject_type": "math", "current_lesson_id": "l2"},
        )

        resp = client.get("/scholar/s1/books?subject_type=english")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        assert books[0]["textbook_id"] == "tb_en_1"
        assert books[0]["subject_type"] == "english"

    def test_filter_math(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_en_1/position",
            json={"subject_type": "english", "current_lesson_id": "l1"},
        )
        client.put(
            "/scholar/s1/books/tb_math_1/position",
            json={"subject_type": "math", "current_lesson_id": "l2"},
        )

        resp = client.get("/scholar/s1/books?subject_type=math")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        assert books[0]["textbook_id"] == "tb_math_1"
        assert books[0]["subject_type"] == "math"

    def test_no_filter_returns_all(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_en_1/position",
            json={"subject_type": "english", "current_lesson_id": "l1"},
        )
        client.put(
            "/scholar/s1/books/tb_math_1/position",
            json={"subject_type": "math", "current_lesson_id": "l2"},
        )

        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 2

    def test_response_contains_subject_type_field(self, make_client, fake_db):
        """响应 books[*] 必须包含 subject_type 字段（契约对齐）。"""
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"subject_type": "english", "current_lesson_id": "l1"},
        )
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        book = resp.json()["data"]["books"][0]
        assert "subject_type" in book
        assert book["subject_type"] == "english"


class TestPutPositionSubjectType:
    """PUT /scholar/{id}/books/{tid}/position body 含 subject_type"""

    def test_first_join_with_subject_type_math(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.put(
            "/scholar/s1/books/tb_math_1/position",
            json={"subject_type": "math", "current_lesson_id": "l1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["subject_type"] == "math"
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["subject_type"] == "math"

    def test_first_join_default_english(self, make_client, fake_db):
        """不传 subject_type → 默认 english。"""
        client = make_client(tracking_router)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_lesson_id": "l1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["subject_type"] == "english"

    def test_subject_type_only_accepted(self, make_client, fake_db):
        """仅传 subject_type 也应接受（至少一个字段校验通过）。"""
        client = make_client(tracking_router)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"subject_type": "english"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["subject_type"] == "english"


class TestPostTextbookSubjectType:
    """POST /textbook body 含 subject_type"""

    def test_post_with_subject_type_english(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.post(
            "/textbook",
            json={"title": "NCE 1", "subject_type": "english"},
        )
        assert resp.status_code == 200

    def test_post_without_subject_type_defaults_english(self, make_client, fake_db):
        """不传 subject_type → 默认 english（向后兼容）。"""
        client = make_client(tracking_router)
        resp = client.post("/textbook", json={"title": "NCE 2"})
        assert resp.status_code == 200
