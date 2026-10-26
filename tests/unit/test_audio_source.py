"""单元测试：`services/infra/audio_source`（音源统一取值，**fileID 优先**）。

背景：微信 `wx.cloud.callContainer` **请求包体上限 100KB**（官方《调用服务问题排查指引》
`-606001 请求包大小超过 100KB`）⇒ 音频不能走 base64 进请求体，必须先入对象存储、
请求体只带 fileID，由服务端自行拉取。本模块就是这套「二选一」取值的唯一实现。

故这里的用例同时是**全项目所有语音入口**的安全网（/eval/transcribe、/eval/speech、
/eval/translate、/eval/translate/v2），改动本模块请先看这些用例。
"""

from __future__ import annotations

import asyncio

import pytest

from services.infra.audio_source import (
    MAX_AUDIO_BYTES,
    InvalidAudio,
    decode_audio_base64,
    resolve_audio_bytes,
)
from services.infra.tcb_storage import CloudBaseStorageClient, StorageAPIError


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def patch_storage(monkeypatch):
    """返回安装「对象存储下载桩」的函数：result 为 bytes 或待抛异常。"""

    def _install(result):
        async def _fake_download(self, file_id, timeout=30.0):
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(CloudBaseStorageClient, "download_file", _fake_download)

    return _install


class TestResolvePriority:
    """fileID 优先 —— 这是绕开 100KB 上限的关键，任何时候都不得回退。"""

    def test_file_id_preferred_over_base64(self, patch_storage):
        patch_storage(b"from-cloud")
        got = _run(
            resolve_audio_bytes(
                "cloud://env.x/a.mp3", "ZmFrZS1iYXNlNjQ=", route="/unit-test"
            )
        )
        assert got == b"from-cloud"

    def test_file_id_only(self, patch_storage):
        patch_storage(b"only-file")
        got = _run(resolve_audio_bytes("cloud://env.x/a.mp3", "", route="/unit-test"))
        assert got == b"only-file"

    def test_base64_only_fallback(self):
        got = _run(
            resolve_audio_bytes(None, "ZmFrZS1iYXNlNjQ=", route="/unit-test")
        )
        assert got == b"fake-base64"

    def test_neither_provided_returns_none(self):
        """两者都缺 → None（交由各路由自己的「文本/语音二选一」或必填校验处理）。"""
        assert _run(resolve_audio_bytes(None, None, route="/unit-test")) is None
        assert _run(resolve_audio_bytes("", "", route="/unit-test")) is None
        assert _run(resolve_audio_bytes("   ", "   ", route="/unit-test")) is None


class TestResolveFailures:
    """各种「给了音频但取不到」的失败分支，都要落成 InvalidAudio（不 5xx）。"""

    def test_storage_api_error(self, patch_storage):
        patch_storage(StorageAPIError("STORAGE_FILE_NONEXIST"))
        with pytest.raises(InvalidAudio):
            _run(resolve_audio_bytes("cloud://env.x/gone.mp3", "", route="/unit-test"))

    def test_generic_exception_maps_to_invalid_audio(self, patch_storage):
        """非预期异常也要收敛为业务错误（HTTPError / 网络抖动等），不能让请求 5xx。"""
        patch_storage(RuntimeError("boom"))
        with pytest.raises(InvalidAudio):
            _run(resolve_audio_bytes("cloud://env.x/a.mp3", "", route="/unit-test"))

    def test_empty_download(self, patch_storage):
        patch_storage(b"")
        with pytest.raises(InvalidAudio):
            _run(resolve_audio_bytes("cloud://env.x/a.mp3", "", route="/unit-test"))

    def test_oversize_via_file_id(self, patch_storage):
        """fileID 路径**必须**自设上限：否则后端会被动下载超大对象。"""
        patch_storage(b"x" * (MAX_AUDIO_BYTES + 1))
        seen = {}
        with pytest.raises(InvalidAudio) as exc:
            _run(
                resolve_audio_bytes(
                    "cloud://env.x/big.mp3",
                    "",
                    route="/unit-test",
                    oversize_message=seen.setdefault("noop", "定制文案"),
                )
            )
        assert exc.value.message == "定制文案"

    def test_oversize_via_base64(self):
        import base64 as _b64

        huge = _b64.b64encode(b"x" * (MAX_AUDIO_BYTES + 1)).decode()
        with pytest.raises(InvalidAudio):
            _run(
                resolve_audio_bytes(
                    None, huge, route="/unit-test", oversize_message="超限啦"
                )
            )

    def test_invalid_base64(self):
        with pytest.raises(InvalidAudio):
            _run(resolve_audio_bytes(None, "!!!not-base64!!!", route="/unit-test"))


class TestDecodeAudioBase64:
    def test_data_url_prefix(self):
        assert decode_audio_base64("data:audio/mp3;base64,ZmFrZQ==") == b"fake"

    def test_plain_base64(self):
        assert decode_audio_base64("ZmFrZQ==") == b"fake"

    def test_invalid_returns_none(self):
        assert decode_audio_base64("!!!") is None

    def test_empty_returns_none(self):
        assert decode_audio_base64("   ") is None
