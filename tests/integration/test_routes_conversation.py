"""集成测试：Conversation MVP 接口（契约 api-contract §3.9）— scenario / turn / history

被测链路：FastAPI TestClient + FakeDB，覆盖：
- POST /conversation/scenario：创建会话，难度档位默认 1，前置评估为建议层不阻断
- POST /conversation/turn：AI 回复 + 每轮 eval_verdict + 降级路径（hint → rephrase → 降档）
- GET /conversation/history：轮次 + 会话结束小结 + 门控 SkillState 更新 + ReviewSchedule
- 错误：缺参 / 会话不存在 / 会话已结束
- 原则：每轮必返 eval_verdict；低置信不回写（§9-2）；建议层不硬阻断（§9-5）
"""
from __future__ import annotations

import pytest

from services.routes_conversation import router as conversation_router


def _client(make_client, monkeypatch):
    # 屏蔽真实 LLM：AI 回复回落规则兜底（不触网、确定性）；评估回落由 no_external_calls 兜底
    import services.routes_conversation as rc

    monkeypatch.setattr(rc, "_generate_reply", lambda **k: {"reply": "OK, continue.", "next_target": ""})
    return make_client(conversation_router)


def _create_session(client: TestClient, scholar: str = "u1") -> str:
    resp = client.post("/conversation/scenario", json={"scholar_id": scholar})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    return body["data"]["session_id"]


class TestScenario:
    def test_create_session_cold_start_default(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/conversation/scenario", json={"scholar_id": "u1", "topic": "travel"}
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["session_id"].startswith("cvs_")
        assert data["difficulty"] == 1  # 冷启动先验默认（§5.6.1）
        # 前置评估为建议层（§0.2/§9-5）：gate_suggestion 不硬阻断
        assert data["pre_assessment"]["gate_suggestion"] == "pass"
        # §5.6.5 冷启动标记 + 先验默认
        assert data["cold_start"] is True
        assert data["prior_defaults"]["mastery"] == pytest.approx(0.35)
        assert data["prior_defaults"]["difficulty"] == 1
        assert data["prior_defaults"]["confidence"] == 0.0
        # 沉浸式开场扩展字段：引导开口的英文开场白 + 会话初始状态（供会话页渲染首条气泡）
        assert data["reply"] == "OK, continue."  # 注入回复（真实链路为 LLM 开场白）
        assert data["state"]["stage"] == "opening"
        assert data["state"]["hint"] is None
        assert data["state"]["rephrased"] is None
        assert data["state"]["difficulty"] == 1

    def test_create_session_with_history_not_cold(self, make_client, monkeypatch, fake_db):
        """已有 skill_state 历史 → 非冷启动（§5.6.5）。"""
        fake_db.add("skill_state", {
            "_id": "u1_sent_1_translation",
            "scholar_id": "u1",
            "sentence_id": "sent_1",
            "skill_code": "translation",
            "status": "learning",
        })
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/scenario", json={"scholar_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["cold_start"] is False

    def test_scenario_with_history_uses_pre_assessment(self, make_client, monkeypatch, fake_db):
        """S3.2 §6.2：有历史 → 前置评估按 skill_state 聚合生成 Gate 建议 + Activity 推荐。

        证据充分（attempt=3 ≥ MIN_EVIDENCE）→ 弱项驱动推荐；难度档位写入会话
        （对 ConversationGraph 生效）；建议层不阻断（会话仍创建成功）。
        """
        fake_db.add("skill_state", {
            "_id": "u1_sent_1_translation",
            "scholar_id": "u1",
            "sentence_id": "sent_1",
            "skill_code": "translation",
            "status": "learning",
            "mastery_score": 40,  # mastery=0.4 → Gate 建议 "training"
            "attempt_count": 3,
        })
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/scenario", json={"scholar_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["cold_start"] is False
        # 非冷启动不返回先验默认（§5.6 仅冷启动展示引导）
        assert "prior_defaults" not in data
        # Gate 建议（建议层不阻断：会话仍创建成功）
        assert data["pre_assessment"]["gate_suggestion"] == "training"
        # 弱项驱动：translation 最弱 → translation 排最前
        assert data["pre_assessment"]["activity_recommendation"][0] == "translation"
        assert data["difficulty"] == 1  # 0.4 < 0.7 → 最低档

    def test_scenario_history_difficulty_band_2(self, make_client, monkeypatch, fake_db):
        """S3.2 §6.2：mastery ∈ [0.7, 0.85) → 难度档位 2 写入会话。"""
        fake_db.add("skill_state", {
            "_id": "u1_sent_1_translation",
            "scholar_id": "u1",
            "sentence_id": "sent_1",
            "skill_code": "translation",
            "status": "learning",
            "mastery_score": 75,
            "attempt_count": 3,
        })
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/scenario", json={"scholar_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["pre_assessment"]["gate_suggestion"] == "pass"
        assert data["difficulty"] == 2  # 0.75 ∈ [0.7, 0.85) → band 2

    def test_missing_scholar_id(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/scenario", json={})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"


class TestTurn:
    def test_first_turn_returns_verdict(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        resp = client.post(
            "/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."}
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["turn_id"].startswith("cvt_")
        assert data["reply"] == "OK, continue."
        # 每轮必返 eval_verdict（§9-2）
        verdict = data["eval_verdict"]
        assert "meaningful" in verdict
        assert "faithfulness" in verdict
        assert "anomaly" in verdict
        assert 0 <= verdict["confidence"] <= 1
        assert data["state"]["stage"] in ("answer", "hint", "rephrase", "downgrade")

    def test_missing_session_id(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/turn", json={"session_id": "", "utterance": "hi"})
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_nonexistent_session(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/conversation/turn", json={"session_id": "cvs_nope", "utterance": "hi"}
        )
        assert resp.json()["code"] == "NOT_FOUND"

    def test_conflict_ended_session(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        client.post(
            "/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."}
        )
        client.get(f"/conversation/history?session_id={sid}")  # 结束会话
        resp = client.post(
            "/conversation/turn", json={"session_id": sid, "utterance": "again"}
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == "CONFLICT"


class TestHistory:
    def test_history_with_summary_and_gate(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        client.post(
            "/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."}
        )
        client.post(
            "/conversation/turn", json={"session_id": sid, "utterance": "It is a watch."}
        )
        resp = client.get(f"/conversation/history?session_id={sid}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["turns"]) == 2
        session = data["session"]
        # 会话结束：生成小结（§9-3）
        assert session["ended_at"] is not None
        assert session["summary"]["total_turns"] == 2

        # 门控 SkillState 更新 + ReviewSchedule 记录（§9-3）：达意且高置信才回写
        stored = fake_db.all("conversation_session")[0]
        assert stored["summary"]["meaningful_rate"] >= 0
        assert stored["ended_at"] is not None
        assert stored["review_schedule"] is not None

    def test_history_empty_session(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        sid = _create_session(client)
        resp = client.get(f"/conversation/history?session_id={sid}")
        data = resp.json()["data"]
        assert data["turns"] == []
        assert data["session"]["summary"] is None  # 无轮次不强制结束

    def test_history_missing_session(self, make_client, monkeypatch, fake_db):
        client = _client(make_client, monkeypatch)
        resp = client.get("/conversation/history?session_id=cvs_nope")
        assert resp.json()["code"] == "NOT_FOUND"
