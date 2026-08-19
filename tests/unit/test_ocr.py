"""F4.2 单元测试：OCR Provider 抽象与腾讯云实现（无网络，mock SDK client）。"""

import config
import pytest

from services.math import ocr as ocr_module
from services.math.ocr import (
    OcrConfigError,
    OcrError,
    OcrProvider,
    OcrResult,
    TencentGeneralOcrProvider,
    get_provider,
)


class _FakeClient:
    """按响应顺序返回结果；若响应为异常则抛出（用于模拟失败/重试）。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def GeneralAccurateOCR(self, req):
        self.calls += 1
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _fake_resp(texts):
    resp = type("FakeResp", (), {})()
    resp.TextDetections = [
        {
            "DetectedText": text,
            "Polygon": [{"X": 0.0, "Y": float(i)}, {"X": 1.0, "Y": float(i)}],
        }
        for i, text in enumerate(texts)
    ]
    return resp


def _make_provider(monkeypatch, secret_id="ak", secret_key="sk", client=None):
    monkeypatch.setattr(config, "TENCENT_OCR_SECRET_ID", secret_id)
    monkeypatch.setattr(config, "TENCENT_OCR_SECRET_KEY", secret_key)
    provider = TencentGeneralOcrProvider()
    if client is not None:
        monkeypatch.setattr(provider, "_build_client", lambda: client)
    return provider


# ---------------------------------------------------------------------------
# 抽象与单例
# ---------------------------------------------------------------------------


def test_ocr_provider_is_abstract():
    assert OcrProvider.__abstractmethods__ == {"available", "recognize"}


def test_get_provider_returns_singleton():
    assert get_provider() is get_provider()


def test_dependencies_get_ocr_shared_singleton(monkeypatch):
    from services import dependencies

    monkeypatch.setattr(ocr_module, "_provider", None)
    monkeypatch.setattr(dependencies, "_get_ocr_provider", ocr_module.get_provider)
    assert dependencies.get_ocr() is ocr_module.get_provider()


# ---------------------------------------------------------------------------
# 凭据可用性
# ---------------------------------------------------------------------------


def test_available_with_credentials(monkeypatch):
    provider = _make_provider(monkeypatch)
    assert provider.available is True


def test_available_without_credentials(monkeypatch):
    provider = _make_provider(monkeypatch, secret_id="", secret_key="")
    assert provider.available is False


# ---------------------------------------------------------------------------
# recognize 正常路径
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recognize_success_builds_text_and_blocks(monkeypatch):
    client = _FakeClient([_fake_resp(["第 1 题", "解方程 x+1=2"])])
    provider = _make_provider(monkeypatch, client=client)

    result = await provider.recognize(b"fake-image-bytes")

    assert isinstance(result, OcrResult)
    assert result.text == "第 1 题\n解方程 x+1=2"
    assert len(result.blocks) == 2
    assert result.blocks[0]["block_id"] == "blk_0001"
    assert result.blocks[0]["text"] == "第 1 题"
    assert result.blocks[0]["bbox"] == [[0.0, 0.0], [1.0, 0.0]]
    assert result.blocks[0]["image_url_crop"] is None
    assert result.blocks[1]["block_id"] == "blk_0002"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_recognize_skips_blank_detections(monkeypatch):
    client = _FakeClient([_fake_resp(["题目", "", "   ", "答案"])])
    provider = _make_provider(monkeypatch, client=client)

    result = await provider.recognize(b"img")

    assert result.text == "题目\n答案"
    assert len(result.blocks) == 2


# ---------------------------------------------------------------------------
# 失败重试与降级
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recognize_retries_once_then_succeeds(monkeypatch):
    client = _FakeClient([RuntimeError("timeout"), _fake_resp(["第 3 题"])])
    provider = _make_provider(monkeypatch, client=client)

    result = await provider.recognize(b"img")

    assert result.text == "第 3 题"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_recognize_raises_after_retry(monkeypatch):
    client = _FakeClient([RuntimeError("boom-a"), RuntimeError("boom-b")])
    provider = _make_provider(monkeypatch, client=client)

    with pytest.raises(OcrError):
        await provider.recognize(b"img")
    assert client.calls == 2


@pytest.mark.asyncio
async def test_recognize_without_credentials_raises_without_network(monkeypatch):
    provider = _make_provider(monkeypatch, secret_id="", secret_key="")
    monkeypatch.setattr(
        provider, "_build_client", lambda: pytest.fail("无凭据时不应触网")
    )

    with pytest.raises(OcrConfigError):
        await provider.recognize(b"img")


@pytest.mark.asyncio
async def test_recognize_empty_image_raises(monkeypatch):
    provider = _make_provider(monkeypatch, client=_FakeClient([]))

    with pytest.raises(OcrError):
        await provider.recognize(b"")
