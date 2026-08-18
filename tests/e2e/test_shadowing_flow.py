"""P0 e2e 验收：Shadowing 跟读链路（speaking）全闭环。

前端契约（F4/4.3 Shadowing）：听原声 → 跟读录音 → evaluateSpeech(POST /eval/speech,
SOE-N 句级口语评测) → 三指标反馈（准确率/流利度/完整度 + 词级 MatchTag 高亮）；
Skill 落库 speaking，走 reportSpeaking → POST /tracking/state。

被测链路（模拟真实前端调用顺序）：
    POST /eval/speech（SOE-N 口语评测，落库 speech_evaluation 存档）
        → POST /tracking/state（状态上报，skill_code=speaking）
        → GET /tracking/{scholar_id}（追踪查询，验证闭环）

断言重点：
- parsed 三指标归一：accuracy/fluency(×100)/completion/suggested_score 0~100，
  words[].match_tag 0=命中 / 2=未命中（前端词级高亮语义）；
- speech_evaluation 原始 JSON 存档落库（raw + parsed + provider=soe_n）；
- SOE 不可用 / 参数非法 / 音频非法 → 200 + success=false + code 错误契约；
- speaking 状态上报复合键幂等、attempt_count 累加；
- 追踪查询 records 与上报状态一致。
"""

from __future__ import annotations

import base64

from services.routes_eval import get_speech_provider, router as eval_router
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.fake_providers import FakeSpeechProvider

SCHOLAR_ID = "e2e_scholar_shadow_001"
SENTENCE_ID = "sent_e2e_shadow_001"
LESSON_ID = "lesson_e2e_001"
ORIGINAL_TEXT = "The quick brown fox jumps over the lazy dog."
FAKE_AUDIO = base64.b64encode(b"fake-mp3-audio-bytes").decode()


def _client(make_client, provider=None):
    """构建 Shadowing 链路 TestClient：eval + state + tracking，Provider 走 dependency_overrides。"""
    if provider is None:
        provider = FakeSpeechProvider()
    return make_client(
        eval_router,
        state_router,
        tracking_router,
        overrides={get_speech_provider: lambda: provider},
    )


class TestSpeechEvalFullFlow:
    """SOE-N 口语评测 → 状态上报 → 追踪查询闭环。"""

    def test_shadowing_closed_loop(self, make_client, fake_db):
        client = _client(make_client)

        # 1. SOE-N 口语评测（FakeSpeechProvider 固定原始 JSON）
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        parsed = body["data"]
        # 三指标归一（PronFluency=0.85 → 85.0；其余原值 0~100）
        assert parsed["accuracy"] == 78.9
        assert parsed["fluency"] == 85.0
        assert parsed["completion"] == 90.0
        assert parsed["suggested_score"] == 82.5
        # 词级 MatchTag：0=命中 / 2=未命中
        assert parsed["words"][0] == {"word": "the", "match_tag": 0}
        assert parsed["words"][2] == {"word": "brown", "match_tag": 2}

        # 2. speech_evaluation 原始 JSON 存档落库
        archives = fake_db.all("speech_evaluation")
        assert len(archives) == 1
        arch = archives[0]
        assert arch["scholar_id"] == SCHOLAR_ID
        assert arch["sentence_id"] == SENTENCE_ID
        assert arch["provider"] == "soe_n"
        assert arch["parsed"]["accuracy"] == 78.9
        assert arch["raw"]["PronAccuracy"] == 78.9  # 原始 JSON 保留

        # 3. 状态上报：Shadowing 落库 skill_code=speaking
        resp = client.post(
            "/tracking/state",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "skill_code": "speaking",
                "lesson_id": LESSON_ID,
                "status": "learned",
                "score": 82,
                "time_spent": 30,
            },
        )
        assert resp.status_code == 200
        state = resp.json()["data"]["state"]
        assert state["_id"] == f"{SCHOLAR_ID}_{SENTENCE_ID}_speaking"
        assert state["skill_code"] == "speaking"
        assert state["status"] == "learned"
        assert state["mastery_score"] == 82.0
        assert state["attempt_count"] == 1

        # 4. 追踪查询：闭环
        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.status_code == 200
        result = resp.json()
        assert result["total"] == 1
        assert result["records"][0]["skill_code"] == "speaking"
        assert result["records"][0]["status"] == "learned"

    def test_speech_state_upsert_accumulates(self, make_client, fake_db):
        """同一复合键重复上报 → attempt_count 累加、事件 append-only。"""
        client = _client(make_client)
        payload = {
            "scholar_id": SCHOLAR_ID,
            "sentence_id": SENTENCE_ID,
            "skill_code": "speaking",
            "status": "learning",
            "score": 70,
        }
        for _ in range(2):
            resp = client.post("/tracking/state", json=payload)
            assert resp.json()["success"] is True

        states = fake_db.all("skill_state")
        assert len(states) == 1
        assert states[0]["attempt_count"] == 2
        assert len(fake_db.all("study_attempt")) == 2

        resp = client.get(f"/tracking/{SCHOLAR_ID}")
        assert resp.json()["total"] == 1


class TestSpeechEvalErrorContract:
    """SOE-N 错误契约：200 + success=false + code（业务失败不走 4xx/5xx）。"""

    def test_soe_unavailable_returns_code(self, make_client):
        client = _client(make_client, provider=FakeSpeechProvider.unavailable())
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "SOE_UNAVAILABLE"

    def test_provider_failure_returns_code(self, make_client):
        """Provider 可用但调用失败（raw=None）→ 同样降级 SOE_UNAVAILABLE。"""
        client = _client(make_client, provider=FakeSpeechProvider.failing())
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
            },
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "SOE_UNAVAILABLE"

    def test_missing_required_field_returns_422(self, make_client):
        """Pydantic 必填字段缺失 → FastAPI 422（技术契约），不进入业务分支。"""
        client = _client(make_client)
        resp = client.post(
            "/eval/speech",
            json={"sentence_id": SENTENCE_ID, "original_text": ORIGINAL_TEXT, "audio_base64": FAKE_AUDIO},
        )
        assert resp.status_code == 422

    def test_empty_required_field_returns_invalid_input(self, make_client):
        """字段存在但为空字符串 → 业务校验 INVALID_INPUT（200 + success=false）。"""
        client = _client(make_client)
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": "  ",
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_invalid_audio_returns_code(self, make_client):
        client = _client(make_client)
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": "not-base64!!",
            },
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_unsupported_voice_format_returns_invalid(self, make_client):
        client = _client(make_client)
        resp = client.post(
            "/eval/speech",
            json={
                "scholar_id": SCHOLAR_ID,
                "sentence_id": SENTENCE_ID,
                "original_text": ORIGINAL_TEXT,
                "audio_base64": FAKE_AUDIO,
                "voice_format": "ogg",
            },
        )
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"
