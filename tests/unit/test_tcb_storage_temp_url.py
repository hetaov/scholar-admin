"""get_temp_file_url 的 file_id 传递（2026-10-23 修复，线上 image_url 恒空根因）

被测对象：services/infra/tcb_storage.py

背景（线上实测）：云端真实 fileID 形如 `cloud://{env}.{bucket}/{path}`，
按 `cloud://{env}/{path}` 拼出的 ID 微信返回 `status=1 STORAGE_FILE_NONEXIST`
→ 换链失败 → `image_url` 恒为空串 → 前端错题本缩略图落成失效的本地临时路径。

覆盖用例：
  - 传 file_id → **原样**发给微信（不再用 env_id 拼接）
  - 不传 file_id → 回落旧拼法（兼容既有调用）
  - 微信返回 status≠0 → 空串（不抛异常，image_file_id 仍可永久引用）
"""
from __future__ import annotations

import httpx
import pytest

from services.infra.tcb_storage import CloudBaseStorageClient

REAL_FILE_ID = "cloud://knowlege-graph-env-x.6b6e-knowlege-graph-env-x-1380427893/scan/s1.png"


class _FakeResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def client(monkeypatch):
    c = CloudBaseStorageClient(env_id="env-x", appid="appid", secret="secret")

    async def _tok() -> str:
        return "tok"

    monkeypatch.setattr(c, "_get_access_token", _tok)
    return c


@pytest.mark.asyncio
async def test_uses_given_file_id(client, monkeypatch):
    """传入真实 fileID 时应原样使用（修复点）"""
    captured = {}

    async def _post(self, url, **kwargs):  # noqa: ANN001
        captured.update(kwargs.get("json") or {})
        return _FakeResp({
            "errcode": 0,
            "file_list": [{"fileid": REAL_FILE_ID, "download_url": "https://tmp/x.png", "status": 0}],
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    url = await client.get_temp_file_url("scan/s1.png", file_id=REAL_FILE_ID)

    assert url == "https://tmp/x.png"
    assert captured["file_list"][0]["fileid"] == REAL_FILE_ID  # 未被 env_id 改写


@pytest.mark.asyncio
async def test_fallback_when_no_file_id(client, monkeypatch):
    """不传 file_id 时回落旧拼法（兼容既有调用，防回归）"""
    captured = {}

    async def _post(self, url, **kwargs):  # noqa: ANN001
        captured.update(kwargs.get("json") or {})
        return _FakeResp({
            "errcode": 0,
            "file_list": [{"fileid": "cloud://env-x/scan/s1.png", "download_url": "https://tmp/y.png", "status": 0}],
        })

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    url = await client.get_temp_file_url("scan/s1.png")

    assert url == "https://tmp/y.png"
    assert captured["file_list"][0]["fileid"] == "cloud://env-x/scan/s1.png"


@pytest.mark.asyncio
async def test_nonzero_status_returns_empty(client, monkeypatch):
    """微信返回 STORAGE_FILE_NONEXIST → 空串且**不抛异常**（渲染/上传不阻断）"""

    async def _post(self, url, **kwargs):  # noqa: ANN001
        return _FakeResp({"errcode": 0, "file_list": [
            {"fileid": REAL_FILE_ID, "download_url": "", "status": 1, "errmsg": "STORAGE_FILE_NONEXIST"},
        ]})

    monkeypatch.setattr(httpx.AsyncClient, "post", _post)

    assert await client.get_temp_file_url("scan/s1.png", file_id=REAL_FILE_ID) == ""
