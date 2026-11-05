""""语音描述出题 · 阶段 A 转写接口验收（POST /math/asr/transcribe）

覆盖场景（契约对齐 /eval/transcribe：业务失败 200 + success=false + code）：
- 音源全空 → INVALID_AUDIO（不触达 ASR）
- ASR 未配置凭据 → ASR_UNAVAILABLE
- 转写成功 → { success: true, data: { transcription } }
- ASR 返回空 → ASR_UNAVAILABLE（提示重试）
- voice_file_id 优先于 audio_base64（音源取舍走 resolve_audio_bytes）
"""
from __future__ import annotations

import pytest

from services.routes_math import router as math_router


class _StubASR:
    """ASR 服务桩：available + recognize 返回值可控"""

    def __init__(self, *, available: bool = True, text: str = "三年级分数加减出五道应用题"):
        self.available = available
        self._text = text
        self.calls: list[dict] = []

    def recognize(self, audio_bytes, voice_format="mp3"):  # noqa: ANN001
        self.calls.append({"bytes": audio_bytes, "voice_format": voice_format})
        return self._text


@pytest.fixture()
def stub_asr(monkeypatch):
    stub = _StubASR()
    monkeypatch.setattr(
        "services.routes.math.get_asr_service_for", lambda _kind: stub
    )
    return stub


def test_invalid_audio_when_both_sources_empty(make_client, fake_db, stub_asr):
    client = make_client(math_router)
    res = client.post("/math/asr/transcribe", json={"voice_file_id": "", "audio_base64": ""})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["code"] == "INVALID_AUDIO"
    assert stub_asr.calls == []  # 音源无效不触达 ASR


def test_asr_unavailable(make_client, fake_db, monkeypatch):
    monkeypatch.setattr(
        "services.routes.math.get_asr_service_for",
        lambda _kind: _StubASR(available=False),
    )
    client = make_client(math_router)
    res = client.post(
        "/math/asr/transcribe", json={"audio_base64": "QUJD"}  # "ABC"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is False
    assert body["code"] == "ASR_UNAVAILABLE"


def test_transcribe_success(make_client, fake_db, stub_asr):
    client = make_client(math_router)
    res = client.post("/math/asr/transcribe", json={"audio_base64": "QUJD"})
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    assert body["data"]["transcription"] == "三年级分数加减出五道应用题"
    assert stub_asr.calls[0]["bytes"] == b"ABC"
    assert stub_asr.calls[0]["voice_format"] == "mp3"  # 缺省 mp3


def test_empty_transcription_reports_retry(make_client, fake_db, monkeypatch):
    monkeypatch.setattr(
        "services.routes.math.get_asr_service_for",
        lambda _kind: _StubASR(text=""),
    )
    client = make_client(math_router)
    res = client.post("/math/asr/transcribe", json={"audio_base64": "QUJD"})
    body = res.json()
    assert body["success"] is False
    assert body["code"] == "ASR_UNAVAILABLE"
    assert "重试" in body["message"]


def test_voice_file_id_preferred_over_base64(make_client, fake_db, stub_asr, monkeypatch):
    """fileID 优先（绕开 callContainer 100KB 上限）：resolve 收到 fileID 分支"""
    seen: dict = {}

    async def _fake_resolve(voice_file_id, audio_base64, **kwargs):  # noqa: ANN001
        seen["voice_file_id"] = voice_file_id
        seen["audio_base64"] = audio_base64
        return b"FILEDATA"

    monkeypatch.setattr("services.routes.math.resolve_audio_bytes", _fake_resolve)
    client = make_client(math_router)
    res = client.post(
        "/math/asr/transcribe",
        json={"voice_file_id": "cloud://x.mp3", "audio_base64": "QUJD"},
    )
    assert res.json()["success"] is True
    assert seen["voice_file_id"] == "cloud://x.mp3"
    assert seen["audio_base64"] == "QUJD"
    assert stub_asr.calls[0]["bytes"] == b"FILEDATA"
