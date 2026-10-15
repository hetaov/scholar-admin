"""aac 音频支持（2026-10-15 真机缺陷：微信小程序在部分真机上录不出 mp3）

背景：
  - 微信 `RecorderManager` 的原生/默认格式是 **aac**，mp3 需额外转码；
    部分真机该转码失败 —— 起录正常（onStart 有）、能录满时长，
    但 `stop()` 后微信 `onError` 报 **file error**（编码器产不出文件）。
  - 客户端已改为「默认 mp3，file error 时自适应降级 aac」，故服务端必须能吃下 aac。
  - SOE-N 的 `VoiceFileType` 只有 0=pcm / 1=wav / 2=mp3 / 4=speex —— **不含 aac**
    ⇒ Provider 在送 SOE 前必须把 aac 转成 mp3。
  - ASR 侧（services/providers/asr.py）原生支持 aac，直接透传，无需转码。

覆盖：
  - 白名单放行 aac（路由层不再 INVALID_INPUT）
  - aac → mp3 转码被真正调用，且之后按 mp3 送 SOE
  - 转码失败 ⇒ `transcode_failed`（且不再往下走 SDK）
  - mp3 / wav 不受影响（不触发转码）
  - ffmpeg 缺失 ⇒ `_transcode_to_mp3` 返回 None
"""

from __future__ import annotations

import base64

import pytest

from services.providers import speech_eval as se


# --------------------------------------------------------------------------- #
# 路由层：白名单
# --------------------------------------------------------------------------- #
class TestAacAcceptedByRoute:
    def test_aac_is_no_longer_invalid_input(self, make_client, fake_db):
        """aac 必须通过 voice_format 白名单（此前只有 mp3/wav）。"""
        from tests.integration.test_speech_eval import _client, speech_payload

        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(voice_format="aac"))
        body = resp.json()
        assert body["code"] != "INVALID_INPUT", body
        assert body["success"] is True, body

    def test_ogg_still_invalid_input(self, make_client, fake_db):
        """回归：白名单放宽到 aac 后，其它非法格式仍应被拒。"""
        from tests.integration.test_speech_eval import _client, speech_payload

        client = _client(make_client)
        resp = client.post("/eval/speech", json=speech_payload(voice_format="ogg"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"


# --------------------------------------------------------------------------- #
# Provider 层：转码接线
# --------------------------------------------------------------------------- #
def _provider() -> se.TencentSoeNProvider:
    """构造一个「凭据齐备」的 Provider（不触网，转码后才可能走到 _load_sdk）。"""
    return se.TencentSoeNProvider(
        appid="appid", secret_id="sid", secret_key="skey", session_token=None
    )


class TestAacTranscoding:
    def test_aac_is_transcoded_before_soe(self, monkeypatch):
        """aac ⇒ 先转 mp3，且转换器收到原始 aac 字节与源格式。"""
        seen = {}

        def fake_transcode(audio_bytes: bytes, src_format: str):
            seen["src_format"] = src_format
            seen["src_bytes"] = audio_bytes
            return b"MP3-BYTES"

        monkeypatch.setattr(se, "_transcode_to_mp3", fake_transcode)
        monkeypatch.setattr(se, "_load_sdk", lambda: False)  # 转码后即停，不触网

        raw, reason = _provider().evaluate_with_reason(b"AAC-BYTES", "hello world", "aac")

        assert seen["src_format"] == "aac"
        assert seen["src_bytes"] == b"AAC-BYTES"
        assert raw is None
        # 走到 sdk 检查说明转码已通过（未因 transcode_failed 提前返回）
        assert reason == "sdk_missing"

    def test_transcode_failure_short_circuits(self, monkeypatch):
        """转码失败 ⇒ transcode_failed，且不再往下加载 SDK（快速失败）。"""
        called = {"sdk": False}

        monkeypatch.setattr(se, "_transcode_to_mp3", lambda b, f: None)
        monkeypatch.setattr(se, "_load_sdk", lambda: called.__setitem__("sdk", True))

        raw, reason = _provider().evaluate_with_reason(b"AAC-BYTES", "hello world", "aac")

        assert raw is None
        assert reason == "transcode_failed"
        assert called["sdk"] is False

    def test_mp3_and_wav_do_not_trigger_transcode(self, monkeypatch):
        """回归：mp3 / wav 是 SOE 原生格式，不得被送去转码。"""
        calls = []
        monkeypatch.setattr(
            se, "_transcode_to_mp3", lambda b, f: calls.append(f) or b"X"
        )
        monkeypatch.setattr(se, "_load_sdk", lambda: False)

        for fmt in ("mp3", "wav"):
            _provider().evaluate_with_reason(b"BYTES", "hello world", fmt)

        assert calls == []

    def test_transcode_failed_has_readable_text(self):
        """失败原因 token 必须有可读文案（路由层拼进 message）。"""
        assert "transcode_failed" in se.SPEECH_FAIL_REASON_TEXT
        assert se.SPEECH_FAIL_REASON_TEXT["transcode_failed"]


class TestTranscodeToMp3:
    def test_missing_ffmpeg_returns_none(self, monkeypatch):
        """镜像内缺 ffmpeg ⇒ 返回 None（上层据此报 transcode_failed，不必等 SOE 超时）。"""
        monkeypatch.setattr(se.shutil, "which", lambda _: None)
        assert se._transcode_to_mp3(b"AAC", "aac") is None

    def test_ffmpeg_failure_returns_none(self, monkeypatch):
        """ffmpeg 非零退出 ⇒ 返回 None。"""
        monkeypatch.setattr(se.shutil, "which", lambda _: "/usr/bin/ffmpeg")

        class _Proc:
            returncode = 1
            stdout = b""
            stderr = b"Invalid data found when processing input"

        monkeypatch.setattr(
            se.subprocess, "run", lambda *a, **k: _Proc()
        )
        assert se._transcode_to_mp3(b"AAC", "aac") is None

    def test_ffmpeg_success_returns_mp3_bytes(self, monkeypatch):
        """ffmpeg 成功 ⇒ 返回 mp3 字节。"""
        monkeypatch.setattr(se.shutil, "which", lambda _: "/usr/bin/ffmpeg")

        class _Proc:
            returncode = 0
            stdout = b"MP3!"
            stderr = b""

        monkeypatch.setattr(se.subprocess, "run", lambda *a, **k: _Proc())
        assert se._transcode_to_mp3(b"AAC", "aac") == b"MP3!"

    def test_uses_16k_mono_lame(self, monkeypatch):
        """转码参数必须对齐 SOE 引擎口径：16k / 单声道 / libmp3lame。"""
        monkeypatch.setattr(se.shutil, "which", lambda _: "/usr/bin/ffmpeg")
        captured = {}

        class _Proc:
            returncode = 0
            stdout = b"MP3!"
            stderr = b""

        def run(cmd, *a, **k):
            captured["cmd"] = cmd
            return _Proc()

        monkeypatch.setattr(se.subprocess, "run", run)
        se._transcode_to_mp3(b"AAC", "aac")

        cmd = captured["cmd"]
        assert "libmp3lame" in cmd
        assert "16000" in cmd
        assert "1" in cmd
