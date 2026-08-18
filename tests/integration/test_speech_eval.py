"""集成测试:SOE-N 句级口语评测(F2/2.3) — POST /eval/speech

被测链路:FastAPI TestClient + Fake SpeechProvider + FakeDB,覆盖:
- 正常评测:校验 → Provider → 归一化 → 落库 speech_evaluation
- 参数校验:缺必填 / 非法 voice_format → INVALID_INPUT
- 音频校验:非法 base64 / 超 5MB → INVALID_AUDIO
- 降级:Provider 不可用 / 评测失败(返回 None) → SOE_UNAVAILABLE
- 契约:业务失败 200 + success=false + code,与 /eval/translate 对齐;仅落库成功路径
"""
from __future__ import annotations

import base64
import time

import pytest

from services.routes_eval import (
    MAX_AUDIO_BYTES,
    router as eval_router,
)
from services.speech_eval import get_speech_provider
from tests.fakes.fake_providers import FakeSpeechProvider
from tests.fakes.seed_factory import speech_payload


def _client(make_client, provider=None):
    """构建 TestClient：make_client 注入 FakeDB，SpeechProvider 走 dependency_overrides。"""
    if provider is None:
        provider = FakeSpeechProvider()
    return make_client(eval_router, overrides={get_speech_provider: lambda: provider})


class TestSpeechEvalSuccess:
    def test_normal_evaluation_and_persist(self, make_client, fake_db):
        """正常链路:校验 → Provider → 归一化 → 落库 speech_evaluation。"""
        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["code"] == "OK"

        # 归一化结果返回给前端
        data = body["data"]
        assert data["accuracy"] == 78.9
        assert data["fluency"] == 85.0  # PronFluency 0.85 × 100
        assert data["completion"] == 90.0
        assert data["suggested_score"] == 82.5
        assert data["words"] == [
            {"word": "the", "match_tag": 0},
            {"word": "quick", "match_tag": 0},
            {"word": "brown", "match_tag": 2},
        ]

        # 原始 JSON + parsed 落库 speech_evaluation
        rows = fake_db.all("speech_evaluation")
        assert len(rows) == 1
        doc = rows[0]
        assert doc["scholar_id"] == "scholar_1001"
        assert doc["sentence_id"] == "sent_0001"
        assert doc["original_text"] == "The quick brown fox jumps over the lazy dog"
        assert doc["audio_ref"] is None  # P0 不存音频本体
        assert doc["provider"] == "soe_n"
        assert doc["raw"] == FakeSpeechProvider.DEFAULT_RAW
        assert doc["parsed"] == data
        assert isinstance(doc["created_at"], int) and doc["created_at"] > 0

    def test_data_url_audio_base64(self, make_client, fake_db):
        """兼容 data URL 前缀的音频 base64(前端录音直传形态)。"""
        client = _client(make_client)
        b64 = base64.b64encode(b"fake-mp3-bytes").decode()
        resp = client.post(
            "/eval/speech",
            json=speech_payload(audio_base64=f"data:audio/mp3;base64,{b64}"),
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert len(fake_db.all("speech_evaluation")) == 1

    def test_wav_voice_format(self, make_client, fake_db):
        """voice_format=wav 应透传给 Provider 并成功。"""
        seen = {}

        class RecordingProvider(FakeSpeechProvider):
            def evaluate(self, audio_bytes, ref_text, voice_format="mp3"):
                seen["voice_format"] = voice_format
                return self.raw

        client = _client(make_client, provider=RecordingProvider())
        resp = client.post("/eval/speech", json=speech_payload(voice_format="wav"))
        assert resp.json()["success"] is True
        assert seen["voice_format"] == "wav"

    def test_provider_receives_decoded_audio(self, make_client, fake_db):
        """Provider 收到的应为解码后的原始字节。"""
        seen = {}

        class RecordingProvider(FakeSpeechProvider):
            def evaluate(self, audio_bytes, ref_text, voice_format="mp3"):
                seen["audio_bytes"] = audio_bytes
                seen["ref_text"] = ref_text
                return self.raw

        client = _client(make_client, provider=RecordingProvider())
        payload = speech_payload(audio_base64=base64.b64encode(b"real-bytes-123").decode())
        resp = client.post("/eval/speech", json=payload)
        assert resp.json()["success"] is True
        assert seen["audio_bytes"] == b"real-bytes-123"
        assert seen["ref_text"] == payload["original_text"]


class TestSpeechEvalValidation:
    def test_missing_required_audio(self, make_client, fake_db):
        """必填 audio_base64 为空 → INVALID_INPUT(与 /eval/translate 区分:
        speech 的 audio_base64 是必填字段,空串走必填校验而非 INVALID_AUDIO)。"""
        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(audio_base64=""))
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"
        assert fake_db.all("speech_evaluation") == []

    def test_blank_scholar_id(self, make_client, fake_db):
        """scholar_id 为纯空格(pydantic 长度合法但 strip 后为空)→ INVALID_INPUT。"""
        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(scholar_id="   "))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_invalid_voice_format(self, make_client, fake_db):
        """voice_format 非 mp3/wav → INVALID_INPUT。"""
        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(voice_format="ogg"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_missing_scholar_id_422(self, make_client, fake_db):
        """缺 scholar_id → pydantic min_length 校验 422(契约外的技术校验)。"""
        client = _client(make_client)
        payload = speech_payload()
        payload.pop("scholar_id")
        resp = client.post("/eval/speech", json=payload)
        assert resp.status_code == 422

    def test_invalid_base64(self, make_client, fake_db):
        """非法 base64 → INVALID_AUDIO。"""
        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(audio_base64="!!!not-base64!!!"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_AUDIO"
        assert fake_db.all("speech_evaluation") == []

    def test_audio_too_large(self, make_client, fake_db):
        """音频超 5MB(16k/60s 定标)→ INVALID_AUDIO。"""
        client = _client(make_client)
        oversized = base64.b64encode(b"x" * (MAX_AUDIO_BYTES + 1)).decode()
        resp = client.post("/eval/speech", json=speech_payload(audio_base64=oversized))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_AUDIO"
        assert fake_db.all("speech_evaluation") == []


class TestSpeechEvalFallback:
    def test_provider_unavailable(self, make_client, fake_db):
        """Provider 无凭据 → SOE_UNAVAILABLE(前端可回退旧 /eval/translate)。"""
        client = _client(make_client, provider=FakeSpeechProvider.unavailable())
        resp = client.post("/eval/speech", json=speech_payload())
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SOE_UNAVAILABLE"
        assert fake_db.all("speech_evaluation") == []

    def test_provider_failure(self, make_client, fake_db):
        """Provider 可用但评测失败(返回 None)→ SOE_UNAVAILABLE。"""
        client = _client(make_client, provider=FakeSpeechProvider.failing())
        resp = client.post("/eval/speech", json=speech_payload())
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SOE_UNAVAILABLE"
        assert fake_db.all("speech_evaluation") == []

    def test_persist_failure_does_not_block_result(self, make_client, fake_db, monkeypatch):
        """落库异常仅记日志,不阻塞评测结果返回。"""
        client = _client(make_client)

        async def _boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(fake_db, "insert", _boom)
        resp = client.post("/eval/speech", json=speech_payload())
        body = resp.json()
        assert resp.status_code == 200
        assert body["success"] is True  # 评测成功,落库失败不降级
        assert body["data"]["accuracy"] == 78.9

    def test_created_at_uses_ms_timestamp(self, make_client, fake_db):
        """落库 created_at 为毫秒时间戳(与既有 data-model 口径一致)。"""
        before = int(time.time() * 1000)
        client = _client(make_client)
        client.post("/eval/speech", json=speech_payload())
        after = int(time.time() * 1000)
        doc = fake_db.all("speech_evaluation")[0]
        assert before <= doc["created_at"] <= after
