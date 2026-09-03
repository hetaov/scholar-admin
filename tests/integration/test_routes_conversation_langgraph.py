"""集成测试：S4.2 L2 ConversationGraph 路由接入（HTTP 契约不变，api-contract §3.9）

被测链路：FastAPI TestClient + FakeDB，覆盖：
- POST /conversation/scenario：L2 图初始化（会话落 graph_state + checkpoint_id + checkpoint 落库）
- POST /conversation/turn：L2 图路径（checkpoint 恢复 → 评估 → 推进 → 回复），契约与 L1 一致
- 多轮断点续聊：turn_index / consecutive_failures 递增恢复
- 回退：CONVERSATION_GRAPH_ENABLED=0 → L1 轻量状态机（无 checkpoint）；图执行失败 → L1 兜底

原则：不触网（no_external_calls 屏蔽 _call_judge → 评估回落 L1 规则；_generate_reply 注入 fake）。
"""
from __future__ import annotations

import pytest

from services.routes_conversation import router as conversation_router


def _client(make_client, monkeypatch):
    import services.routes_conversation as rc

    monkeypatch.setattr(
        rc, "_generate_reply", lambda **k: {"reply": "OK, continue.", "next_target": ""}
    )
    return make_client(conversation_router)


def _create_session(client) -> str:
    resp = client.post("/conversation/scenario", json={"scholar_id": "u1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    return body["data"]["session_id"]


class TestScenarioGraphInit:
    def test_scenario_returns_immersive_opening(self, make_client, monkeypatch, fake_db):
        """开场返回引导性英文回复 + 会话初始状态（混元评估 conversation 维度要求）。"""
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/conversation/scenario",
            json={"scholar_id": "u1", "topic": "Product pricing negotiation"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["reply"] == "OK, continue."  # 注入回复（真实链路为 LLM 开场白）
        state = data["state"]
        assert state["stage"] == "opening"
        assert state["hint"] is None
        assert state["rephrased"] is None
        assert state["suggestion"] is None
        assert state["difficulty"] in (1, 2)

    def test_scenario_writes_graph_state_and_checkpoint(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        stored = fake_db.all("conversation_session")[0]
        # L2：会话落图状态快照 + checkpoint 引用
        assert stored["graph_state"]["stage"] == "ready_for_utterance"
        assert stored["checkpoint_id"]
        # checkpoint 落库（thread_id=session_id）
        docs = fake_db.all("conversation_graph_checkpoint")
        assert docs
        assert all(d["thread_id"] == sid for d in docs)


class TestTurnGraphPath:
    def test_turn_returns_contract_and_updates_session(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        resp = client.post("/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."})
        assert resp.status_code == 200
        data = resp.json()["data"]
        # HTTP 契约与 L1 一致（§3.9）
        assert data["turn_id"].startswith("cvt_")
        assert data["reply"] == "OK, continue."
        verdict = data["eval_verdict"]
        assert "meaningful" in verdict and "confidence" in verdict
        assert data["state"]["stage"] in ("answer", "hint", "rephrase", "downgrade")

        # 会话图状态回写（turn_index 推进 + checkpoint 更新）
        stored = fake_db.all("conversation_session")[0]
        assert stored["graph_state"]["turn_index"] == 1
        assert stored["checkpoint_id"]
        assert stored["turn_index"] == 1

        # 本轮 checkpoint 落库
        docs = fake_db.all("conversation_graph_checkpoint")
        assert len(docs) > 1

    def test_multi_turn_resumes_via_checkpoint(self, make_client, monkeypatch, fake_db):
        """断点续聊：两轮 turn 状态递增恢复（turn_index / consecutive_failures）。"""
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        for _ in range(2):
            resp = client.post(
                "/conversation/turn", json={"session_id": sid, "utterance": "hello there"}
            )
            assert resp.status_code == 200
            assert resp.json()["success"] is True

        stored = fake_db.all("conversation_session")[0]
        assert stored["graph_state"]["turn_index"] == 2
        # 两轮均未达意 → 连续失败累计
        assert stored["graph_state"]["consecutive_failures"] >= 1

        # 轮次记录均含 eval_verdict
        turns = fake_db.all("conversation_turn")
        assert len(turns) == 2
        assert all("eval_verdict" in t for t in turns)


class TestGraphFallback:
    def test_disabled_uses_l1_no_checkpoint(self, make_client, monkeypatch, fake_db):
        """CONVERSATION_GRAPH_ENABLED=0 → 回退 L1 轻量状态机：不落 graph_state/checkpoint。"""
        import services.routes_conversation as rc

        monkeypatch.setattr(rc, "CONVERSATION_GRAPH_ENABLED", False)
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)

        stored = fake_db.all("conversation_session")[0]
        assert "graph_state" not in stored
        assert "checkpoint_id" not in stored
        assert fake_db.all("conversation_graph_checkpoint") == []

        resp = client.post("/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["reply"] == "OK, continue."
        assert data["state"]["stage"] in ("answer", "hint", "rephrase", "downgrade")

    def test_graph_failure_falls_back_to_l1(self, make_client, monkeypatch, fake_db):
        """图执行异常 → 回退 L1（HTTP 契约仍成功返回）。"""
        import services.routes_conversation as rc

        async def boom(*a, **k):
            raise RuntimeError("graph broken")

        monkeypatch.setattr(rc, "run_turn_graph", boom)
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)

        resp = client.post("/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["reply"] == "OK, continue."  # L1 兜底正常
        # L1 路径不更新 checkpoint
        stored = fake_db.all("conversation_session")[0]
        assert stored["graph_state"]["turn_index"] == 0
