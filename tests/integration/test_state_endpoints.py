"""集成测试:学习状态上报接口 + 查询改走 skill_state（Phase 2）+ 事件写入（Phase 3）

被测链路:FastAPI TestClient + FakeDB,覆盖:
- POST /tracking/state        上报单句单能力状态(创建 / 累加 / 参数校验)
- POST /tracking/state        同时写一条 study_attempt 事件(append-only)
- GET  /tracking/{scholar_id} 只查 skill_state, 无记录打日志不回退旧表
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router


def _client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_state.get_db", lambda: fake_db)
    monkeypatch.setattr("services.routes_tracking.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(state_router)
    app.include_router(tracking_router)
    return TestClient(app)


class TestPostTrackingState:
    def test_create_new(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "skill_code": "translation",
                "status": "已学",
                "score": 85,
                "time_spent": 120,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        state = data["state"]
        assert state["attempt_count"] == 1
        assert state["status"] == "learned"
        assert state["mastery_score"] == 85.0
        assert state["skill_code"] == "translation"
        # Phase 3: 同时写入一条 study_attempt 事件
        attempt = data["attempt"]
        assert attempt["scholar_id"] == "s1"
        assert attempt["sentence_id"] == "sent_1"
        assert attempt["skill_code"] == "translation"
        assert attempt["time_spent"] == 120
        assert attempt["attempt_id"]
        assert fake_db.all("study_attempt").__len__() == 1

    def test_repeat_accumulates_state_but_appends_events(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        for _ in range(2):
            resp = client.post(
                "/tracking/state",
                json={"scholar_id": "s1", "sentence_id": "sent_1"},
            )
            assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["state"]["attempt_count"] == 2
        # skill_state 只一条(同复合键 upsert), study_attempt 每报一条追加一条
        assert fake_db.all("skill_state").__len__() == 1
        assert fake_db.all("study_attempt").__len__() == 2

    def test_missing_scholar_id(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/state", json={"sentence_id": "sent_1"})
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]

    def test_missing_sentence_id(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/state", json={"scholar_id": "s1"})
        assert resp.status_code == 400
        assert "sentence_id" in resp.json()["detail"]

    def test_default_skill_code(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_1"}
        )
        assert resp.json()["data"]["state"]["skill_code"] == "translation"

    def test_attempt_type_inferred_from_skill_code(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "skill_code": "listening"},
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["attempt_type"] == "listen"

    def test_attempt_explicit_fields(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": "s1",
                "sentence_id": "sent_1",
                "attempt_type": "quiz",
                "attempt_status": "correct",
                "score": 100,
                "lesson_id": "unit_1",
                "session_id": "ses_abc",
            },
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["attempt_type"] == "quiz"
        assert attempt["status"] == "correct"
        assert attempt["lesson_id"] == "unit_1"
        assert attempt["session_id"] == "ses_abc"

    def test_invalid_attempt_status_falls_back(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "attempt_status": "???"},
        )
        assert resp.json()["data"]["attempt"]["status"] == "completed"


class TestGetTrackingByScholarV2:
    def test_prefers_skill_state(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_1", "status": "learned"},
        )
        # 旧表也放一条, 验证查询只走 skill_state（旧表数据不被返回）
        fake_db.add(
            "learning_mastery_tracking",
            {"scholar_id": "s1", "sentence_id": "sent_old", "status": "learned"},
        )
        resp = client.get("/tracking/s1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["records"][0]["sentence_id"] == "sent_1"
        assert body["records"][0]["attempt_count"] == 1

    def test_no_skill_state_returns_empty_no_fallback(self, monkeypatch, fake_db):
        """skill_state 无记录时不回退旧表: 返回空 + 打日志"""
        client = _client(monkeypatch, fake_db)
        fake_db.add(
            "learning_mastery_tracking",
            {"scholar_id": "legacy_user", "sentence_id": "sent_old", "status": "learned"},
        )
        resp = client.get("/tracking/legacy_user")
        assert resp.status_code == 200
        body = resp.json()
        # 旧表数据不再被返回（不回退）
        assert body["total"] == 0
        assert body["records"] == []
