"""集成测试:学习会话接口（Phase 3 事件模型）

被测链路:FastAPI TestClient + FakeDB,覆盖:
- POST /tracking/session/start  创建 study_session(active)
- POST /tracking/session/end    结算会话(回填 duration_sec / attempt_count / status)
- 会话内事件隔离(重复会话互不干扰)
- 参数校验(缺 scholar_id / 缺 session_id / 会话不存在)
"""

from __future__ import annotations

from services.routes_state import router as state_router


def _start(client, **overrides) -> dict:
    payload = {"scholar_id": "s1", "textbook_id": "tb_1", "device": "ios"}
    payload.update(overrides)
    resp = client.post("/tracking/session/start", json=payload)
    assert resp.status_code == 200
    return resp.json()["data"]


class TestSessionStart:
    def test_create_active_session(self, make_client, fake_db):
        client = make_client(state_router)
        session = _start(client)
        assert session["scholar_id"] == "s1"
        assert session["textbook_id"] == "tb_1"
        assert session["device"] == "ios"
        assert session["status"] == "active"
        assert session["session_id"]
        assert session["started_at"]
        assert session["ended_at"] is None
        assert session["duration_sec"] == 0
        assert session["attempt_count"] == 0
        assert fake_db.all("study_session").__len__() == 1

    def test_missing_scholar_id(self, make_client, fake_db):
        client = make_client(state_router)
        resp = client.post("/tracking/session/start", json={"textbook_id": "tb_1"})
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]

    def test_multiple_sessions_isolated(self, make_client, fake_db):
        client = make_client(state_router)
        _start(client, scholar_id="s1")
        _start(client, scholar_id="s1")
        _start(client, scholar_id="s2")
        sessions = fake_db.all("study_session")
        assert len(sessions) == 3
        # 每次 start 生成独立 session_id
        assert len({s["session_id"] for s in sessions}) == 3


class TestSessionEnd:
    def test_settles_session_with_attempt_count(self, make_client, fake_db):
        client = make_client(state_router)
        session = _start(client)
        session_id = session["session_id"]
        # 会话内上报 2 次学习事件
        for _ in range(2):
            resp = client.post(
                "/tracking/state",
                json={
                    "scholar_id": "s1",
                    "sentence_id": "sent_1",
                    "session_id": session_id,
                },
            )
            assert resp.status_code == 200
        resp = client.post("/tracking/session/end", json={"session_id": session_id})
        assert resp.status_code == 200
        ended = resp.json()["data"]
        assert ended["session_id"] == session_id
        assert ended["status"] == "ended"
        assert ended["attempt_count"] == 2
        assert ended["ended_at"] is not None
        assert ended["duration_sec"] >= 0

    def test_end_session_missing_id(self, make_client, fake_db):
        client = make_client(state_router)
        resp = client.post("/tracking/session/end", json={})
        assert resp.status_code == 400
        assert "session_id" in resp.json()["detail"]

    def test_end_session_not_found(self, make_client, fake_db):
        client = make_client(state_router)
        resp = client.post("/tracking/session/end", json={"session_id": "ses_unknown"})
        assert resp.status_code == 404


class TestSessionAttemptIsolation:
    def test_attempts_belongs_to_own_session(self, make_client, fake_db):
        """重复会话互不干扰:各会话的 attempt_count 只统计本会话事件。"""
        client = make_client(state_router)
        ses_a = _start(client, scholar_id="s1")
        ses_b = _start(client, scholar_id="s1")

        # 会话 A 上报 2 次, 会话 B 上报 1 次
        for _ in range(2):
            client.post(
                "/tracking/state",
                json={"scholar_id": "s1", "sentence_id": "sent_a", "session_id": ses_a["session_id"]},
            )
        client.post(
            "/tracking/state",
            json={"scholar_id": "s1", "sentence_id": "sent_b", "session_id": ses_b["session_id"]},
        )

        resp_a = client.post("/tracking/session/end", json={"session_id": ses_a["session_id"]})
        resp_b = client.post("/tracking/session/end", json={"session_id": ses_b["session_id"]})
        assert resp_a.json()["data"]["attempt_count"] == 2
        assert resp_b.json()["data"]["attempt_count"] == 1
        # 事件带上各自的 session_id
        attempts = fake_db.all("study_attempt")
        assert sum(1 for a in attempts if a["session_id"] == ses_a["session_id"]) == 2
        assert sum(1 for a in attempts if a["session_id"] == ses_b["session_id"]) == 1

    def test_events_outside_session_not_counted(self, make_client, fake_db):
        """无 session 的事件不进入任何会话的 attempt_count。"""
        client = make_client(state_router)
        session = _start(client)
        client.post(
            "/tracking/state", json={"scholar_id": "s1", "sentence_id": "sent_outside"}
        )
        resp = client.post("/tracking/session/end", json={"session_id": session["session_id"]})
        assert resp.json()["data"]["attempt_count"] == 0
