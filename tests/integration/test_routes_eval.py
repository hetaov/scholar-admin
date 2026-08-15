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
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.routes_eval import get_asr_service, router as eval_router


class FakeASR:
    """假 ASR:返回固定转写文本;可通过结果控制是否模拟识别失败。"""

    available = True
    result = "it is a watch"

    def recognize(self, audio_bytes: bytes, voice_format: str = "mp3") -> str | None:
        return self.result


class UnavailableASR:
    """无凭据的 ASR:available=False。"""

    available = False

    def recognize(self, audio_bytes: bytes, voice_format: str = "mp3") -> None:
        return None


class FailASR:
    """可用但识别失败(返回 None)。"""

    available = True

    def recognize(self, audio_bytes: bytes, voice_format: str = "mp3") -> None:
        return None


def _client(monkeypatch, asr=None, model_output=None) -> TestClient:
    """构建 TestClient,eval 路由的 ASR 依赖与模型输出均可注入。

    - 默认屏蔽模型调用(返回 None → 走 levenshtein 兜底),避免测试环境真实触网
    - 传 model_output 时模拟模型返回
    """
    if asr is None:
        asr = FakeASR()
    # 用 FastAPI 官方 dependency_overrides 注入假 ASR（Depends 在定义期已捕获原对象，
    # monkeypatch 模块属性无效，必须走 override 机制）
    app = FastAPI()
    app.include_router(eval_router)
    app.dependency_overrides[get_asr_service] = lambda: asr
    if model_output is not None:
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: model_output
        )
    else:
        monkeypatch.setattr("services.evaluator._call_volcano", lambda *a, **k: None)
    return TestClient(app)


class TestEvalTranslateText:
    def test_exact_match(self, monkeypatch):
        client = _client(monkeypatch, model_output='{"status": 5}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "It is a watch.", "user_input": "it is a watch"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["transcription"] == "it is a watch"
        assert body["data"]["status"] == 5

    def test_fallback_when_model_unavailable(self, monkeypatch):
        # 模型不可用(默认返回 None) → 兜底 levenshtein
        client = _client(monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "It is a watch.", "user_input": "It is a watch."},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == 5  # 精确匹配兜底

    def test_empty_text_input(self, monkeypatch):
        client = _client(monkeypatch, model_output='{"status": 3}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "user_input": ""},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == 0  # 空输入 → 0


class TestEvalTranslateVoice:
    def test_voice_asr_and_score(self, monkeypatch):
        client = _client(monkeypatch, model_output='{"status": 4}')
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

    def test_asr_unavailable(self, monkeypatch):
        client = _client(monkeypatch, asr=UnavailableASR())
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

    def test_asr_recognition_failure(self, monkeypatch):
        client = _client(monkeypatch, asr=FailASR())
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
    def test_missing_both_inputs(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post("/eval/translate", json={"original_text": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_invalid_base64(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "audio_base64": "!!!not-base64!!!"},
        )
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_empty_audio(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "audio_base64": ""},
        )
        assert resp.json()["code"] == "INVALID_AUDIO"

    def test_missing_original_text(self, monkeypatch):
        client = _client(monkeypatch)
        resp = client.post("/eval/translate", json={"user_input": "hello"})
        assert resp.status_code == 422  # pydantic min_length 校验

    def test_does_not_touch_db(self, monkeypatch, fake_db):
        """仅评估不落库:不访问任何数据库。"""
        client = _client(monkeypatch, model_output='{"status": 5}')
        resp = client.post(
            "/eval/translate",
            json={"original_text": "Hello", "user_input": "hello"},
        )
        assert resp.json()["success"] is True
        assert fake_db.all("skill_state") == []
        assert fake_db.all("study_attempt") == []
