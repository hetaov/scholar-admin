"""集成测试:评估接口(4.6.5a) — POST /eval/translate

被测链路:FastAPI TestClient + 假 ASR/假模型,覆盖:
- 文字路径:user_input 直评 → { transcription, status }
- 语音路径:audio_base64 → ASR → 转写 → 评分
- 参数校验:缺入参 / 非法 base64 / 空音频
- 降级:ASR 未配置凭据 → ASR_UNAVAILABLE(前端可回退云函数评估)
- 原则:仅评估不落库,状态写入仍走 POST /tracking/state
"""
from __future__ import annotations

import base64
import json

import pytest

from services.routes_eval import get_asr_service, router as eval_router
from tests.fakes.fake_providers import FakeAsrService


def _client(make_client, monkeypatch, asr=None, model_output=None) -> TestClient:
    """构建 TestClient,eval 路由的 ASR 依赖与模型输出均可注入。

    - 默认不触网:no_external_calls 已屏蔽 _call_volcano(返回 None → levenshtein 兜底)
    - 传 model_output 时覆盖为模拟模型返回
    - ASR 依赖走 dependency_overrides 注入(Depend 定义期捕获,必须 override)
    """
    if asr is None:
        asr = FakeAsrService()
    if model_output is not None:
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: model_output
        )
    return make_client(eval_router, overrides={get_asr_service: lambda: asr})


class TestEvalTranslateText:
    def test_exact_match(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, model_output='{"status": 5}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "It is a watch.", "user_input": "it is a watch"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 5

    def test_fallback_when_model_unavailable(self, make_client, monkeypatch):
        # 模型不可用(默认返回 None) → 兜底 levenshtein
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "It is a watch.", "user_input": "It is a watch."},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == 5  # 精确匹配兜底

    def test_empty_text_input(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, model_output='{"status": 3}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "user_input": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == 0  # 空输入 → 0


class TestEvalTranslateVoice:
    def test_voice_asr_and_score(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, model_output='{"status": 4}')
        fake_audio = base64.b64encode(b"fake-mp3-bytes").decode()
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": "It is a watch.",
                "audio_base64": fake_audio,
                "voice_format": "mp3",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 4

    def test_asr_unavailable(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, asr=FakeAsrService.unavailable())
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": "It is a watch.",
                "audio_base64": base64.b64encode(b"x").decode(),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "ASR_UNAVAILABLE"

    def test_asr_recognition_failure(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, asr=FakeAsrService.failing())
        resp = client.post(
            "/eval/translate",
            json={
                "original_text": "It is a watch.",
                "audio_base64": base64.b64encode(b"x").decode(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == "ASR_UNAVAILABLE"


class TestEvalTranslateValidation:
    def test_missing_both_inputs(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/translate", json={"original_text": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_invalid_base64(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "audio_base64": "!!!not-base64!!!"},
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_empty_audio(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "audio_base64": ""},
        )
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_missing_original_text(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/translate", json={"user_input": "hello"})
        assert resp.status_code == 422  # pydantic min_length 校验

    def test_does_not_touch_db(self, make_client, monkeypatch, fake_db):
        """仅评估不落库:不访问任何数据库。"""
        client = _client(make_client, monkeypatch, model_output='{"status": 5}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "user_input": "hello"},
        )
        assert resp.json()["success"] is True
        assert fake_db.all("skill_state") == []
        assert fake_db.all("study_attempt") == []


class TestEvalTranscribe:
    """纯语音转写（2026-09-03）— POST /eval/transcribe：ASR → transcription，只转写不评分。

    会话自由语音无标准答案:v1 /eval/translate 语音路径的 evaluate()(LLM 评分,
    失败兜底 levenshtein)从不被消费 → 本端点去除评分环节,无需 original_text。
    错误契约与 /eval/translate 对齐:业务失败 200 + success=false + code。
    """

    def _audio(self, raw=b"fake-mp3-bytes"):
        return base64.b64encode(raw).decode()

    def _block_scoring(self, monkeypatch):
        """评分一旦被调用立即失败 → 证明转写端点不进入评估链路。"""

        def _boom(*args, **kwargs):  # pragma: no cover
            raise AssertionError("POST /eval/transcribe 不应调用 evaluate()（纯转写无评分）")

        monkeypatch.setattr("services.routes.eval.evaluate", _boom)

    def test_transcribe_audio_only(self, make_client, monkeypatch):
        """语音转写成功:仅返回 transcription(无 status 评分字段),不触发评估。"""
        self._block_scoring(monkeypatch)
        asr = FakeAsrService()
        client = _client(make_client, monkeypatch, asr=asr)
        resp = client.post(
            "/eval/transcribe",
            json={"audio_base64": self._audio(), "voice_format": "mp3"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"] == {"transcription": "it is a watch"}  # 无 status
        assert asr.call_count == 1
        assert asr.calls[0][0] == b"fake-mp3-bytes"
        assert asr.calls[0][1] == "mp3"

    def test_no_original_text_needed(self, make_client, monkeypatch):
        """纯转写不要求 original_text:AI 会话自由语音无标准答案(与 v1 关键差异)。"""
        self._block_scoring(monkeypatch)
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/transcribe", json={"audio_base64": self._audio()})
        assert resp.status_code == 200
        assert resp.json()["data"]["transcription"] == "it is a watch"

    def test_empty_audio(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/transcribe", json={"audio_base64": ""})
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_missing_audio(self, make_client, monkeypatch):
        # audio_base64 为必填字段 → pydantic 422（同 /eval/speech）
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/transcribe", json={})
        assert resp.status_code == 422

    def test_invalid_base64(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch)
        resp = client.post("/eval/transcribe", json={"audio_base64": "!!!not-base64!!!"})
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_audio_too_large(self, make_client, monkeypatch):
        # 超过 MAX_AUDIO_BYTES(5MB) → INVALID_AUDIO
        client = _client(make_client, monkeypatch)
        big = b"x" * (5 * 1024 * 1024 + 1)
        resp = client.post("/eval/transcribe", json={"audio_base64": self._audio(big)})
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_asr_unavailable(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, asr=FakeAsrService.unavailable())
        resp = client.post("/eval/transcribe", json={"audio_base64": self._audio()})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "ASR_UNAVAILABLE"

    def test_asr_recognition_failure(self, make_client, monkeypatch):
        client = _client(make_client, monkeypatch, asr=FakeAsrService.failing())
        resp = client.post("/eval/transcribe", json={"audio_base64": self._audio()})
        assert resp.status_code == 200
        assert resp.json()["code"] == "ASR_UNAVAILABLE"
