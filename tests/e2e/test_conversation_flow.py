"""P0 e2e 验收：会话链路（conversation）全闭环。

前端契约（F7/7.4 情景对话）：POST /conversation/scenario（创建会话 + 冷启动建议）
→ POST /conversation/turn（逐轮对话 + eval_verdict）→ GET /conversation/history
（历史 + 会话结束生成小结，门控回写 SkillState）。

被测链路（模拟真实前端调用顺序）：
    POST /conversation/scenario（绑定目标句，无历史 → cold_start=true）
        → POST /conversation/turn × N（达意成功 / 连续失败降级路径）
        → GET /conversation/history（触发 end_session_with_summary）
        → GET /tracking/{scholar_id}（验证 SkillState 门控回写闭环）

断言重点：
- 冷启动契约：无历史 → cold_start=true + prior_defaults + difficulty=1 + 建议序列；
- 状态机推进：达意成功 → answer；连续失败 → hint → rephrase → downgrade（附录 B-1）；
- 每轮 eval_verdict 内联（score/meaningful/confidence/level），L1 规则兜底可用；
- 会话结束小结：summary.total_turns / meaningful_rate 落库；
- 门控回写（§9-2）：confidence ≥ 0.6 且 meaningful → upsert_skill_state
  （skill_code=translation，score≥80 → mastered）+ record_attempt 事件；
  低置信（<0.6）不回写 SkillState。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from services.routes_conversation import router as conversation_router
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.fake_db import FakeDB

SCHOLAR_ID = "e2e_scholar_conv_001"
SENTENCE_ID = "sent_e2e_conv_001"
LESSON_ID = "lesson_e2e_001"
TARGET_TEXT = "It is a watch."
TOPIC = "daily conversation"
SCENARIO = "free_talk"


def _seed_sentence(fake_db: FakeDB) -> None:
    """预置内容库 sentence_v2，使 scenario/turn 能命中绑定句。"""
    fake_db.add(
        "sentence_v2",
        {
            "_id": SENTENCE_ID,
            "sentence_id": SENTENCE_ID,
            "lesson_id": LESSON_ID,
            "text": TARGET_TEXT,
            "translation": "这是一块手表。",
        },
    )


def _client(make_client, monkeypatch) -> TestClient:
    """构建会话链路 TestClient：conversation + state + tracking。

    - LLM 回复生成（services.dialogue.call_volcano）默认屏蔽：monkeypatch
      _generate_reply 返回固定回复，保证不触网；
    - evaluate_text 的 _call_judge 已由 e2e conftest autouse 屏蔽 → 回落 L1 规则。
    """
    monkeypatch.setattr(
        "services.routes_conversation._generate_reply",
        lambda **kw: {"reply": "Good job! Keep going.", "next_target": ""},
    )
    return make_client(conversation_router, state_router, tracking_router)


class TestConversationColdStart:
    """会话创建 + 冷启动契约。"""

    def test_scenario_cold_start_defaults(self, make_client, monkeypatch, fake_db):
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={
                "scholar_id": SCHOLAR_ID,
                "scenario": SCENARIO,
                "topic": TOPIC,
                "sentence_id": SENTENCE_ID,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["session_id"].startswith("cvs_")
        assert data["difficulty"] == 1
        assert data["cold_start"] is True  # 无历史 → 冷启动标记（§5.6.5）
        assert data["prior_defaults"]["difficulty"] == 1
        assert data["pre_assessment"]["gate_suggestion"] == "pass"
        # 冷启动先验默认引导序列
        assert data["pre_assessment"]["activity_recommendation"] == [
            "content",
            "shadowing",
            "translation",
            "listening",
        ]

    def test_scenario_missing_scholar_invalid(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/scenario", json={"topic": TOPIC})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"


class TestConversationTurnFlow:
    """逐轮对话 + 状态机推进。"""

    def test_turn_meaningful_advances_answer(self, make_client, monkeypatch, fake_db):
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={
                "scholar_id": SCHOLAR_ID,
                "scenario": SCENARIO,
                "topic": TOPIC,
                "sentence_id": SENTENCE_ID,
            },
        )
        session_id = resp.json()["data"]["session_id"]

        # 达意成功：输出与目标句一致（L1 规则 → score=100 / meaningful / confidence=0.95）
        resp = client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": TARGET_TEXT},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["turn_id"].startswith("cvt_")
        assert data["state"]["stage"] == "answer"  # 达意成功 → answer
        assert data["state"]["difficulty"] == 1
        verdict = data["eval_verdict"]
        assert verdict["score"] == 100
        assert verdict["meaningful"] is True
        assert verdict["confidence"] == 0.95
        assert verdict["level"] == "l1"
        assert data["reply"] == "Good job! Keep going."  # 注入回复

        # 会话仍为 active，可继续
        turns = fake_db.all("conversation_turn")
        assert len(turns) == 1
        assert turns[0]["eval_verdict"]["score"] == 100

    def test_turn_consecutive_failures_downgrade_path(self, make_client, monkeypatch, fake_db):
        """连续不达意 → hint → rephrase → downgrade 降级路径（附录 B-1）。"""
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={"scholar_id": SCHOLAR_ID, "sentence_id": SENTENCE_ID},
        )
        session_id = resp.json()["data"]["session_id"]

        # 完全不相关输出：L1 → score=15 / meaningful=False / confidence=0.5（低置信）
        stages: list[str] = []
        for _ in range(3):
            resp = client.post(
                "/conversation/turn",
                json={"session_id": session_id, "utterance": "zzzz nothing related"},
            )
            assert resp.status_code == 200
            stages.append(resp.json()["data"]["state"]["stage"])

        assert stages == ["hint", "rephrase", "downgrade"]
        # 降档后 difficulty 归一（clamp 到 MIN_DIFFICULTY=1）
        assert resp.json()["data"]["state"]["difficulty"] == 1
        # 低置信（<0.6）不回写 SkillState
        assert len(fake_db.all("skill_state")) == 0

    def test_turn_unknown_session_not_found(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/conversation/turn",
            json={"session_id": "cvs_nonexistent", "utterance": "hello"},
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "NOT_FOUND"

    def test_turn_missing_params_invalid(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/conversation/turn", json={"utterance": "hello"})
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"


class TestConversationHistorySummary:
    """会话结束小结 + 门控 SkillState 回写闭环。"""

    def test_history_generates_summary_and_writes_state(self, make_client, monkeypatch, fake_db):
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={
                "scholar_id": SCHOLAR_ID,
                "scenario": SCENARIO,
                "topic": TOPIC,
                "sentence_id": SENTENCE_ID,
            },
        )
        session_id = resp.json()["data"]["session_id"]

        # 两轮：达意成功（100 分）→ 未达意（低置信）
        client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": TARGET_TEXT},
        )
        client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": "zzzz wrong"},
        )

        # 会话历史：首次查询触发 end_session_with_summary
        resp = client.get(f"/conversation/history?session_id={session_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["session"]["session_id"] == session_id
        assert data["session"]["ended_at"] is not None
        summary = data["session"]["summary"]
        assert summary["total_turns"] == 2
        assert summary["meaningful_rate"] == 0.5
        assert len(data["turns"]) == 2

        # 门控回写：仅达意且高置信的那轮回写 SkillState
        # （第 1 轮 meaningful+confidence=0.95 → 回写；第 2 轮 low_confidence → 跳过）
        states = fake_db.all("skill_state")
        assert len(states) == 1
        state = states[0]
        assert state["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_translation"
        assert state["skill_code"] == "translation"  # 会话默认落 translation
        assert state["status"] == "mastered"  # score=100 ≥ 80 → mastered
        assert state["mastery_score"] == 100.0
        assert state["attempt_count"] == 1
        # S3.1 P1：SkillState 新字段落库（契约 §4.11.4）
        assert state["confidence"] == pytest.approx(0.95 / 3, abs=1e-4)  # attempt=1 证据打折
        assert state["difficulty"] == 1
        assert state["last_outcome"] == "success"
        assert state["stable_streak"] == 1
        assert "stability" not in state  # 首次不判稳定性
        # 回写伴随 learning attempt 事件（会话关联）
        attempts = fake_db.all("study_attempt")
        assert len(attempts) == 1
        assert attempts[0]["skill_code"] == "translation"
        # 会话文档落小结与 skill_updates
        session_doc = fake_db.all("conversation_session")[0]
        assert session_doc["stage"] == "ended"
        assert session_doc["summary"]["total_turns"] == 2
        # S3.1 P1：会话级门控快照 + 冷启动豁免（§9-2）
        gate = session_doc["summary"]["gate"]
        assert gate["consecutive_low_conf"] == 1
        assert gate["downgrade_factor"] == 1.0  # 未达连续 2 轮低置信，且冷启动豁免
        assert session_doc["cold_start"] is True
        assert session_doc["skill_updates"][0]["sentence_id"] == SENTENCE_ID

        # 追踪查询：验证 SkillState 闭环可见
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1
        assert resp.json()["records"][0]["status"] == "mastered"

    def test_history_non_cold_consecutive_low_conf_downgrades(
        self, make_client, monkeypatch, fake_db
    ):
        """S3.1 P1：非冷启动会话连续 2 轮低置信 → 整会话降权 ×0.5（§9-2）。"""
        _seed_sentence(fake_db)
        # 预置既有 skill_state → has_skill_history=True → 非冷启动
        fake_db.add(
            "skill_state",
            {
                "_id": f"{SCHOLAR_ID}_{SENTENCE_ID}_translation",
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "mastery_score": 60.0,
                "attempt_count": 1,
                "status": "learning",
                "progress": 0.5,
            },
        )
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={"scholar_id": SCHOLAR_ID, "sentence_id": SENTENCE_ID},
        )
        assert resp.json()["data"]["cold_start"] is False
        session_id = resp.json()["data"]["session_id"]

        # 达意成功 1 轮 + 连续 2 轮未达意（低置信）
        client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": TARGET_TEXT},
        )
        for _ in range(2):
            client.post(
                "/conversation/turn",
                json={"session_id": session_id, "utterance": "zzzz nothing related"},
            )

        resp = client.get(f"/conversation/history?session_id={session_id}")
        gate = resp.json()["data"]["session"]["summary"]["gate"]
        assert gate["consecutive_low_conf"] == 2
        assert gate["downgrade_factor"] == 0.5  # 非冷启动 → 降权生效
        assert gate["alert"] is False

        # 达意轮回写且增量打折（60 + (100-60)×0.5×2/3 = 73.33）
        state = fake_db.all("skill_state")[0]
        assert state["mastery_score"] == pytest.approx(73.3333, abs=1e-3)
        # 降权会话的置信度仍正常更新（attempt=2：ewma=(0+0.95)/2 × 2/3）
        assert state["confidence"] == pytest.approx(0.95 / 3, abs=1e-4)

    def test_history_low_faithfulness_not_precipitated(self, make_client, monkeypatch, fake_db):
        """S3.1 P1：忠实率 < 0.7 → 标记 AI 内容偏差，不沉淀为能力证据（§9-2）。"""
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        verdicts = iter(
            [
                {
                    "score": 85,
                    "meaningful": True,
                    "faithfulness": False,  # 达意但语义偏差
                    "anomaly": False,
                    "confidence": 0.9,
                    "level": "sentence",
                },
                {
                    "score": 80,
                    "meaningful": True,
                    "faithfulness": False,
                    "anomaly": False,
                    "confidence": 0.85,
                    "level": "sentence",
                },
            ]
        )
        monkeypatch.setattr(
            "services.routes_conversation.evaluate_text",
            lambda original, utterance: next(verdicts),
        )

        resp = client.post(
            "/conversation/scenario",
            json={"scholar_id": SCHOLAR_ID, "sentence_id": SENTENCE_ID},
        )
        session_id = resp.json()["data"]["session_id"]
        client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": "any utterance"},
        )
        client.post(
            "/conversation/turn",
            json={"session_id": session_id, "utterance": "another utterance"},
        )

        resp = client.get(f"/conversation/history?session_id={session_id}")
        gate = resp.json()["data"]["session"]["summary"]["gate"]
        assert gate["faithfulness_rate"] == 0.0
        assert gate["ai_content_bias"] is True

        # AI 内容偏差：不沉淀任何 SkillState / attempt（§9-2）
        assert len(fake_db.all("skill_state")) == 0
        assert len(fake_db.all("study_attempt")) == 0

    def test_history_low_confidence_not_written(self, make_client, monkeypatch, fake_db):
        """整场会话全部低置信 → 小结生成但不回写任何 SkillState（§9-2）。"""
        _seed_sentence(fake_db)
        client = _client(make_client, monkeypatch)

        resp = client.post(
            "/conversation/scenario",
            json={"scholar_id": SCHOLAR_ID, "sentence_id": SENTENCE_ID},
        )
        session_id = resp.json()["data"]["session_id"]

        for _ in range(2):
            client.post(
                "/conversation/turn",
                json={"session_id": session_id, "utterance": "zzzz nothing related"},
            )

        resp = client.get(f"/conversation/history?session_id={session_id}")
        assert resp.json()["success"] is True
        summary = resp.json()["data"]["session"]["summary"]
        assert summary["meaningful_rate"] == 0.0

        assert len(fake_db.all("skill_state")) == 0
        assert len(fake_db.all("study_attempt")) == 0
        # 会话正常结束
        assert fake_db.all("conversation_session")[0]["stage"] == "ended"

    def test_history_nonexistent_not_found(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.get("/conversation/history?session_id=cvs_nonexistent")
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "NOT_FOUND"
