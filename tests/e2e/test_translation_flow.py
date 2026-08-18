"""P0 e2e 验收：翻译链路（translation）全闭环。

被测链路（模拟真实前端调用顺序）：
    POST /eval/translate（翻译评估，文字直评 / 语音 ASR）
        → POST /tracking/state（状态上报，skill_code=translation）
        → GET /tracking/{scholar_id}（追踪查询，验证闭环）

断言重点：
- 评估结果 transcription/status 与前端直评语义一致；
- 状态上报按复合键 {scholar_id}_{sentence_id}_{skill_code} 幂等 upsert，
  attempt_count 累加、study_attempt 事件 append-only；
- 追踪查询返回的 records 与上报的状态字段一致；
- 会话（session）联动：上报可挂 session_id，end 结算回填 attempt_count；
- 错题标记：error_type 仅 attempt_status=incorrect 时写入。
"""

from __future__ import annotations

import base64

from services.routes_eval import get_asr_service, router as eval_router
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.fake_providers import FakeAsrService

SCHOLAR_ID = "e2e_scholar_trans_001"
SENTENCE_ID = "sent_e2e_trans_001"
LESSON_ID = "lesson_e2e_001"
ORIGINAL_TEXT = "It is a watch."
FAKE_AUDIO = base64.b64encode(b"fake-mp3-audio-bytes").decode()


def _client(make_client, asr=None):
    """构建翻译链路 TestClient：eval + state + tracking，ASR 走 dependency_overrides。"""
    if asr is None:
        asr = FakeAsrService()
    return make_client(
        eval_router,
        state_router,
        tracking_router,
        overrides={get_asr_service: lambda: asr},
    )


class TestTranslationTextFullFlow:
    """文字直评翻译链路闭环。"""

    def test_text_translation_closed_loop(self, make_client, fake_db):
        client = _client(make_client)

        # 1. 翻译评估：文字直评（模型不可用 → levenshtein 兜底，精确匹配 status=5）
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": ORIGINAL_TEXT,
                "user_input": "it is a watch",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 5

        # 2. 状态上报：skill_code=translation
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "lesson_id": LESSON_ID,
                "status": "learned",
                "score": 90,
                "mastery": 0.9,
                "time_spent": 120,
                "attempt_type": "translate",
                "attempt_status": "completed",
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        state, attempt = data["state"], data["attempt"]
        # state：复合键 + 状态 + 掌握度 + 尝试计数
        assert state["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_translation"
        assert state["skill_code"] == "translation"
        assert state["status"] == "learned"
        assert state["mastery_score"] == 90.0
        assert state["attempt_count"] == 1
        # attempt：事件类型由 skill_code 推断为 translate
        assert attempt["attempt_type"] == "translate"
        assert attempt["status"] == "completed"
        assert attempt["time_spent"] == 120

        # 3. 追踪查询：闭环验证
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["total"] == 1
        record = result["records"][0]
        assert record["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_translation"
        assert record["status"] == "learned"

        # 4. 落库断言：skill_state 一条 + study_attempt 一条
        assert len(fake_db.all("skill_state")) == 1
        assert len(fake_db.all("study_attempt")) == 1

    def test_unknown_scholar_returns_empty(self, make_client):
        client = _client(make_client)
        resp = client.get("/tracking/e2e_nobody")
        assert resp.status_code == 200
        assert resp.json()["records"] == []
        assert resp.json()["total"] == 0


class TestTranslationAudioFullFlow:
    """语音 ASR 翻译链路闭环。"""

    def test_audio_translation_closed_loop(self, make_client):
        asr = FakeAsrService()  # 默认转写 "it is a watch"
        client = _client(make_client, asr=asr)

        # 1. 语音翻译评估：audio → ASR 转写 → 评分
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 5
        assert asr.call_count == 1

        # 2. 状态上报 → 3. 查询闭环
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "status": "learned",
                "score": 85,
            },
        )
        assert resp.json()["success"] is True

        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1

    def test_asr_unavailable_falls_back(self, make_client):
        """ASR 不可用 → 业务降级码 ASR_UNAVAILABLE（前端可回退云函数评估）。"""
        client = _client(make_client, asr=FakeAsrService.unavailable())
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "ASR_UNAVAILABLE"


class TestStateUpsertAccumulation:
    """复合键幂等：重复上报仅累加，不产生重复 skill_state。"""

    def test_repeated_report_accumulates_attempt_count(self, make_client, fake_db):
        client = _client(make_client)
        payload = {
            "scholar_id": SCHOLAR_ID,
            "sentence_id": SENTENCE_ID,
            "skill_code": "translation",
            "status": "learned",
            "score": 90,
        }
        for _ in range(3):
            resp = client.post("/tracking/state", json=payload)
            assert resp.json()["success"] is True

        # skill_state 仍仅 1 条，attempt_count 累加；study_attempt 事件 3 条
        states = fake_db.all("skill_state")
        assert len(states) == 1
        assert states[0]["attempt_count"] == 3
        assert len(fake_db.all("study_attempt")) == 3

        # 查询接口返回聚合后单条记录
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1


class TestTranslationSessionFlow:
    """翻译链路与会话（session）联动。"""

    def test_session_start_state_end_flow(self, make_client, fake_db):
        client = _client(make_client)

        # 1. 会话开始
        resp = client.post(
            "/tracking/session/start",
            json={"scholar_id": SCHOLAR_ID, "textbook_id": "tb_e2e_001"},
        )
        assert resp.status_code == 200
        session_id = resp.json()["data"]["session_id"]
        assert session_id

        # 2. 状态上报挂 session_id
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "status": "learned",
                "score": 90,
                "session_id": session_id,
            },
        )
        assert resp.json()["success"] is True
        assert resp.json()["data"]["attempt"]["session_id"] == session_id

        # 3. 会话结束：结算回填 attempt_count
        resp = client.post("/tracking/session/end", json={"session_id": session_id})
        assert resp.status_code == 200
        ended = resp.json()["data"]
        assert ended["status"] == "ended"
        assert ended["attempt_count"] == 1
        assert ended["duration_sec"] >= 0

        # 落库：会话 1 条，会话内事件可统计
        sessions = fake_db.all("study_session")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["status"] == "ended"


class TestTranslationErrorMark:
    """错题标记：error_type 仅 incorrect 时写入事件。"""

    def test_error_type_only_written_when_incorrect(self, make_client, fake_db):
        client = _client(make_client)
        base = {
            "scholar_id": SCHOLAR_ID,
            "sentence_id": SENTENCE_ID,
            "skill_code": "translation",
        }

        # correct + error_type：不写入 error_type
        resp = client.post(
            "/tracking/state",
            json={**base, "attempt_status": "correct", "error_type": "grammar"},
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["status"] == "correct"
        assert "error_type" not in attempt

        # incorrect + error_type：写入
        resp = client.post(
            "/tracking/state",
            json={**base, "attempt_status": "incorrect", "error_type": "grammar"},
        )
        attempt = resp.json()["data"]["attempt"]
        assert attempt["status"] == "incorrect"
        assert attempt["error_type"] == "grammar"
