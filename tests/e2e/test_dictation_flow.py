"""P0 e2e 验收：听写链路（dictation）全闭环。

前端契约（pages/task/detail/chapter）：听写模式 cardField=original（写英文原句），
落库 Skill=translation（skillCode='translation'），评估走 /eval/translate。

被测链路（模拟真实前端调用顺序）：
    POST /eval/translate（听写评估：文字直评原句 / 语音 ASR 转写比对）
        → POST /tracking/state（状态上报，skill_code=translation）
        → GET /tracking/{scholar_id}（追踪查询，验证闭环）

断言重点：
- 听写文字路径：original_text(英文原句) 与 user_input 比对评分，transcription 原样回传；
- 听写语音路径：audio → ASR 转写 → 与英文原句比对，ASR 调用计数正确；
- 状态推导：低分无显式状态 → review_due；高分显式 learned → 保留；
- 会话（session）联动：听写练习 start → 上报挂 session_id → end 结算；
- 错误契约：INVALID_INPUT / INVALID_AUDIO / ASR_UNAVAILABLE。
"""

from __future__ import annotations

import base64

from services.routes_eval import get_asr_service, router as eval_router
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.fake_providers import FakeAsrService

SCHOLAR_ID = "e2e_scholar_dict_001"
SENTENCE_ID = "sent_e2e_dict_001"
LESSON_ID = "lesson_e2e_001"
ORIGINAL_TEXT = "It is a watch."  # 英文原句（听写参考文本）
FAKE_AUDIO = base64.b64encode(b"fake-mp3-audio-bytes").decode()


def _client(make_client, asr=None):
    """构建听写链路 TestClient：eval + state + tracking，ASR 走 dependency_overrides。"""
    if asr is None:
        asr = FakeAsrService()
    return make_client(
        eval_router,
        state_router,
        tracking_router,
        overrides={get_asr_service: lambda: asr},
    )


class TestDictationTextFlow:
    """听写文字路径：看英文原句 → 写出英文 → 比对 → 上报 → 查询。"""

    def test_text_dictation_closed_loop(self, make_client, fake_db):
        client = _client(make_client)

        # 1. 听写评估：写出的英文原句与参考文本一致（levenshtein 兜底精确匹配 → status=5）
        resp = client.post(
            "/eval/translate",
            json={"original_text": ORIGINAL_TEXT, "user_input": "It is a watch."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "It is a watch."
        assert body["data"]["status"] == 5

        # 2. 状态上报：听写落库 skill_code=translation
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "lesson_id": LESSON_ID,
                "status": "learned",
                "score": 95,
                "time_spent": 45,
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        state = data["state"]
        assert state["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_translation"
        assert state["status"] == "learned"
        assert state["mastery_score"] == 95.0
        assert state["attempt_count"] == 1

        # 3. 追踪查询：闭环
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["total"] == 1
        assert result["records"][0]["status"] == "learned"

        # 落库断言
        assert len(fake_db.all("skill_state")) == 1
        assert len(fake_db.all("study_attempt")) == 1

    def test_low_score_derives_review_due(self, make_client, fake_db):
        """听写低分且无显式状态 → 状态推导 review_due（复习队列）。"""
        client = _client(make_client)

        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "score": 40,
                "time_spent": 60,
            },
        )
        assert resp.status_code == 200
        state = resp.json()["data"]["state"]
        assert state["status"] == "review_due"
        assert state["mastery_score"] == 40.0
        assert state["attempt_count"] == 1


class TestDictationAudioFlow:
    """听写语音路径：听英文 → 跟读录音 → ASR 转写比对 → 上报 → 查询。"""

    def test_audio_dictation_closed_loop(self, make_client):
        asr = FakeAsrService()  # 默认转写 "it is a watch"
        client = _client(make_client, asr=asr)

        # 1. 语音听写评估：audio → ASR 转写 → 与英文原句比对
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
        assert body["data"]["status"] == 5  # 归一化后与原句一致
        assert asr.call_count == 1

        # 2. 上报 → 3. 查询闭环
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "translation",
                "status": "learned",
                "score": 90,
            },
        )
        assert resp.json()["success"] is True
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1

    def test_asr_unavailable_returns_code(self, make_client):
        client = _client(make_client, asr=FakeAsrService.unavailable())
        resp = client.post(
            "/eval/translate",
            json={"original_text": ORIGINAL_TEXT, "audio_base64": FAKE_AUDIO},
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "ASR_UNAVAILABLE"

    def test_invalid_audio_returns_code(self, make_client):
        client = _client(make_client)
        resp = client.post(
            "/eval/translate",
            json={"original_text": ORIGINAL_TEXT, "audio_base64": "not-base64!!"},
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_missing_inputs_returns_invalid(self, make_client):
        client = _client(make_client)
        resp = client.post("/eval/translate", json={"original_text": ORIGINAL_TEXT})
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"


class TestDictationSessionFlow:
    """听写练习与会话（session）联动。"""

    def test_session_start_state_end_flow(self, make_client, fake_db):
        client = _client(make_client)

        # 会话开始
        resp = client.post(
            "/tracking/session/start",
            json={"scholar_id": SCHOLAR_ID, "textbook_id": "tb_e2e_dict"},
        )
        session_id = resp.json()["data"]["session_id"]

        # 听写上报挂 session_id（多句听写 → 多次 attempt）
        for i, sent in enumerate(["sent_e2e_dict_001", "sent_e2e_dict_002"]):
            resp = client.post(
                "/tracking/state",
                json={
                    "scholar_id": SCHOLAR_ID,
                    "sentence_id": sent,
                    "skill_code": "translation",
                    "status": "learned",
                    "score": 92,
                    "session_id": session_id,
                },
            )
            assert resp.json()["data"]["attempt"]["session_id"] == session_id

        # 会话结束：结算回填 attempt_count
        resp = client.post("/tracking/session/end", json={"session_id": session_id})
        assert resp.status_code == 200
        ended = resp.json()["data"]
        assert ended["status"] == "ended"
        assert ended["attempt_count"] == 2

        # 落库：会话 1 条 + 2 个 skill_state + 2 条事件
        sessions = fake_db.all("study_session")
        assert len(sessions) == 1
        assert sessions[0]["status"] == "ended"
        assert len(fake_db.all("skill_state")) == 2
        assert len(fake_db.all("study_attempt")) == 2
