"""集成测试: session/start → scholar_book 创建 + subject_type 透传链路

覆盖 Bug 2 修复：start_session 时立即创建 scholar_book，使教材即时出现在学者列表中
覆盖 Bug 1 修复：subject_type 从 session/start 透传到 scholar_book，保证学科标识正确

场景：
  1. POST /tracking/session/start 带 subject_type=math → study_session 含 subject_type + scholar_book 创建
  2. POST /tracking/session/start 不传 subject_type → 默认 english
  3. POST /tracking/session/end → touch_scholar_book 从 session 读取 subject_type
  4. GET /scholar/{id}/books?subject_type=math → 只返回 math 学者的教材
  5. GET /scholar/{id}/books?subject_type=english → 只返回 english 学者的教材
"""
from __future__ import annotations

import pytest

from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from services.models_scholar_book import SCHOLAR_BOOK
from services.events import STUDY_SESSION


class TestSessionStartCreatesScholarBook:
    """POST /tracking/session/start 带 textbook_id 时立即创建 scholar_book"""

    def test_start_with_subject_type_math_creates_scholar_book(self, make_client, fake_db):
        """数学教材开始学习 → session 创建 + scholar_book 创建（subject_type=math）。"""
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/session/start",
            json={
                "scholar_id": "s1",
                "textbook_id": "tb_math_1",
                "subject_type": "math",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["subject_type"] == "math"
        # scholar_book 应立即创建
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["textbook_id"] == "tb_math_1"
        assert records[0]["subject_type"] == "math"

    def test_start_without_subject_type_defaults_english(self, make_client, fake_db):
        """不传 subject_type → 默认 english（向后兼容）。"""
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_en_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        # session 中 subject_type 为 None（未显式传），但 scholar_book 默认 english
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["subject_type"] == "english"

    def test_start_without_textbook_id_no_scholar_book(self, make_client, fake_db):
        """不传 textbook_id → 不创建 scholar_book（无教材关联）。"""
        client = make_client(state_router, tracking_router)
        resp = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1"},
        )
        assert resp.status_code == 200
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 0


class TestSessionEndPropagatesSubjectType:
    """POST /tracking/session/end → touch_scholar_book 从 session 读取 subject_type"""

    def test_end_session_propagates_subject_type_math(self, make_client, fake_db):
        """数学 session 结算 → scholar_book 保留 math 标识。"""
        client = make_client(state_router, tracking_router)
        # start
        start_resp = client.post(
            "/tracking/session/start",
            json={
                "scholar_id": "s1",
                "textbook_id": "tb_math_1",
                "subject_type": "math",
            },
        )
        session_id = start_resp.json()["data"]["session_id"]
        # end
        end_resp = client.post(
            "/tracking/session/end",
            json={"session_id": session_id},
        )
        assert end_resp.status_code == 200
        # scholar_book 应保留 math
        records = fake_db.all(SCHOLAR_BOOK)
        assert len(records) == 1
        assert records[0]["subject_type"] == "math"


class TestScholarBooksFilterBySubjectType:
    """GET /scholar/{id}/books?subject_type= 端到端过滤"""

    def test_math_books_only(self, make_client, fake_db):
        """?subject_type=math 只返回数学教材关联。"""
        client = make_client(state_router, tracking_router)
        # 英语教材
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_en_1", "subject_type": "english"},
        )
        # 数学教材
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_math_1", "subject_type": "math"},
        )

        resp = client.get("/scholar/s1/books?subject_type=math")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        assert books[0]["textbook_id"] == "tb_math_1"
        assert books[0]["subject_type"] == "math"

    def test_english_books_only(self, make_client, fake_db):
        """?subject_type=english 只返回英语教材关联。"""
        client = make_client(state_router, tracking_router)
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_en_1", "subject_type": "english"},
        )
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_math_1", "subject_type": "math"},
        )

        resp = client.get("/scholar/s1/books?subject_type=english")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        assert books[0]["textbook_id"] == "tb_en_1"
        assert books[0]["subject_type"] == "english"

    def test_no_filter_returns_all(self, make_client, fake_db):
        """不传 subject_type → 返回全部。"""
        client = make_client(state_router, tracking_router)
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_en_1", "subject_type": "english"},
        )
        client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_math_1", "subject_type": "math"},
        )

        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 2

    def test_legacy_record_without_subject_type_filtered_as_english(self, make_client, fake_db):
        """存量 scholar_book 记录（无 subject_type 字段）→ normalize 为 english，
        ?subject_type=math 不返回，?subject_type=english 返回。"""
        client = make_client(state_router, tracking_router)
        # 手动插入一条无 subject_type 的存量记录
        fake_db.add(SCHOLAR_BOOK, {
            "_id": "s1_tb_legacy",
            "scholar_id": "s1",
            "textbook_id": "tb_legacy",
            "status": "learning",
            "last_studied_at": 1000,
        })

        # math 过滤 → 不返回 legacy 记录
        resp = client.get("/scholar/s1/books?subject_type=math")
        books = resp.json()["data"]["books"]
        assert len(books) == 0

        # english 过滤 → 返回 legacy 记录（normalize 为 english）
        resp = client.get("/scholar/s1/books?subject_type=english")
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        assert books[0]["textbook_id"] == "tb_legacy"
        assert books[0]["subject_type"] == "english"
