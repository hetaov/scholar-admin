"""集成测试:学者×教材关联接口(Phase 5)

覆盖:
- GET /scholar/{scholar_id}/books — 我的教材列表(含进度),空列表/有记录
- PUT /scholar/{scholar_id}/books/{textbook_id}/position — 更新断点:
  首次加入 / 断点更新 / 重复加入幂等(同一 学者×教材 仅一条)
- 断点更新后重新获取列表能取回 current_lesson_id(验收标准)
- 会话结算回写:end 后 scholar_book 的 last_studied_at 与 total_time_spent 更新
- 参数校验(缺参 / 至少一个字段)
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.models_scholar_book import SCHOLAR_BOOK
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router


def _tracking_client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_tracking.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(tracking_router)
    return TestClient(app)


def _state_client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_state.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(state_router)
    return TestClient(app)


def _seed_content(fake_db):
    """预置 tb_1 内容层级: 1 章 1 课 2 句。"""
    fake_db.add("chapter", {"chapter_id": "c1", "textbook_id": "tb_1", "title": "Ch1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l1", "chapter_id": "c1", "title": "L1", "order": 1})
    fake_db.add(
        "sentence_v2",
        {"sentence_id": "s1", "lesson_id": "l1", "chapter_id": "c1", "textbook_id": "tb_1", "order": 1},
    )
    fake_db.add(
        "sentence_v2",
        {"sentence_id": "s2", "lesson_id": "l1", "chapter_id": "c1", "textbook_id": "tb_1", "order": 2},
    )


class TestPutPosition:
    def test_first_join_creates_record(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["scholar_id"] == "s1"
        assert data["textbook_id"] == "tb_1"
        assert data["current_chapter_id"] == "c1"
        assert data["current_lesson_id"] == "l1"
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_update_position_overwrites(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c2", "current_lesson_id": "l2"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["current_chapter_id"] == "c2"
        assert resp.json()["data"]["current_lesson_id"] == "l2"
        # 幂等: 同一 学者×教材 只有一条
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_repeated_join_idempotent(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        for _ in range(3):
            resp = client.put(
                "/scholar/s1/books/tb_1/position",
                json={"current_lesson_id": "l1"},
            )
            assert resp.status_code == 200
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_only_last_studied_at_allowed(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"last_studied_at": 1234567890},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["last_studied_at"] == 1234567890

    def test_empty_body_rejected(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        resp = client.put("/scholar/s1/books/tb_1/position", json={})
        assert resp.status_code == 400
        assert "current_chapter_id" in resp.json()["detail"]


class TestGetBooks:
    def test_empty_list(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["scholar_id"] == "s1"
        assert data["books"] == []

    def test_list_with_progress(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s1", "skill_code": "translation",
             "status": "learned", "mastery_score": 80, "attempt_count": 2},
        )
        client = _tracking_client(monkeypatch, fake_db)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        book = books[0]
        assert book["textbook_id"] == "tb_1"
        assert book["current_lesson_id"] == "l1"
        summary = book["summary"]
        assert summary["total_sentence_count"] == 2
        assert summary["learned_sentence_count"] == 1
        assert summary["textbook_progress"] == pytest.approx(0.4)  # (0.8 + 0.0) / 2

    def test_breakpoint_retrieved_after_update(self, monkeypatch, fake_db):
        """验收标准: 断点更新后重新获取列表能取回 current_lesson_id。"""
        client = _tracking_client(monkeypatch, fake_db)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_lesson_id": "l5"},
        )
        resp = client.get("/scholar/s1/books")
        book = resp.json()["data"]["books"][0]
        assert book["current_chapter_id"] == "c1"
        assert book["current_lesson_id"] == "l5"

    def test_multiple_books_isolated(self, monkeypatch, fake_db):
        client = _tracking_client(monkeypatch, fake_db)
        client.put("/scholar/s1/books/tb_1/position", json={"current_lesson_id": "l1"})
        client.put("/scholar/s1/books/tb_2/position", json={"current_lesson_id": "l2"})
        resp = client.get("/scholar/s1/books")
        books = resp.json()["data"]["books"]
        assert {b["textbook_id"] for b in books} == {"tb_1", "tb_2"}
        # 每本教材独立断点
        by_id = {b["textbook_id"]: b["current_lesson_id"] for b in books}
        assert by_id == {"tb_1": "l1", "tb_2": "l2"}


class TestSessionSettlementWriteback:
    @pytest.mark.asyncio
    async def test_end_session_writeback_time_and_stamp(self, monkeypatch, fake_db):
        """会话结算回写: scholar_book 的 last_studied_at 与 total_time_spent 被更新。"""
        client = _state_client(monkeypatch, fake_db)
        session = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_1"},
        ).json()["data"]
        # 模拟已学习 30 秒,保证 duration_sec 非零,验证 total_time_spent 回写
        await fake_db.update(
            "study_session",
            where={"_id": session["session_id"]},
            data={"$set": {"started_at": 1}},
        )
        client.post("/tracking/session/end", json={"session_id": session["session_id"]})

        books = fake_db.all(SCHOLAR_BOOK)
        assert len(books) == 1
        book = books[0]
        assert book["scholar_id"] == "s1"
        assert book["textbook_id"] == "tb_1"
        assert book["last_studied_at"] is not None
        assert book["total_time_spent"] > 0

    @pytest.mark.asyncio
    async def test_end_session_accumulates_over_sessions(self, monkeypatch, fake_db):
        client = _state_client(monkeypatch, fake_db)
        for _ in range(2):
            session = client.post(
                "/tracking/session/start",
                json={"scholar_id": "s1", "textbook_id": "tb_1"},
            ).json()["data"]
            # 模拟已学习 30 秒,再结算,保证 duration_sec 非零
            await fake_db.update(
                "study_session",
                where={"_id": session["session_id"]},
                data={"$set": {"started_at": 1}},
            )
            client.post("/tracking/session/end", json={"session_id": session["session_id"]})

        books = fake_db.all(SCHOLAR_BOOK)
        assert len(books) == 1  # 幂等: 多次会话仍只有一条
        # 两次会话各自结算, total_time_spent 为两次 duration 之和(均 > 0)
        assert books[0]["total_time_spent"] > 0

    @pytest.mark.asyncio
    async def test_end_session_without_textbook_no_book(self, monkeypatch, fake_db):
        client = _state_client(monkeypatch, fake_db)
        session = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1"},
        ).json()["data"]
        client.post("/tracking/session/end", json={"session_id": session["session_id"]})
        assert fake_db.all(SCHOLAR_BOOK) == []
