"""回归测试：TCB 云存储客户端（微信云开发 HTTP API 实现，F4 上传 500 修复②）

背景（2026-08-21 真实故障链，逐步暴露）：
- 403：wx.uploadFile 直传云托管公网域名缺 X-WX-OPENID（前端 ensureOpenId 已修）
- UnsupportedProtocol：tcb.tencentcloudapi.com 的 UploadFile 拒绝 application/json
- AuthFailure.SignatureFailure：改 multipart + TC3 签名后仍验签失败
  （该 action 无官方文档，验签规则不明）
→ 弃用 tcb OpenAPI 存储 action，改走微信云开发 HTTP API：
  1. POST /tcb/uploadfile 换取 COS 上传链接
  2. POST COS url 表单直传（key/Signature/x-cos-security-token/x-cos-meta-fileid/file）
  3. POST /tcb/batchdownloadfile 换取临时下载链接
"""
from __future__ import annotations

import asyncio
import json
from unittest import mock

import pytest

from services.infra.tcb_storage import CloudBaseStorageClient, StorageAPIError, _TOKEN_CACHE


def _run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text if text else (json.dumps(payload) if payload is not None else "")

    def json(self):
        return self.payload


class FakeAsyncClient:
    """按预设响应队列返回；记录 GET/POST 的 url 与 kwargs"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.gets: list = []
        self.posts: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self._responses.pop(0)

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self._responses.pop(0)


def _make_client(**kwargs):
    return CloudBaseStorageClient(
        env_id=kwargs.get("env_id", "test-env"),
        appid=kwargs.get("appid", "wx-appid-test"),
        secret=kwargs.get("secret", "wx-secret-test"),
    )


TOKEN_RESP = FakeResponse({"access_token": "TOKEN_ABC", "expires_in": 7200})

UPLOAD_RESP = FakeResponse(
    {
        "errcode": 0,
        "errmsg": "ok",
        "url": "https://cos.ap-shanghai.myqcloud.com/bucket/scan/a.jpg",
        "token": "COS_TOKEN_XYZ",
        "authorization": "q-sign-algorithm=sha1&q-sign-time=1;2",
        "cos_file_id": "COS_FILE_ID_9",
        "file_id": "cloud://test-env/scan/a.jpg",
    }
)

COS_OK_RESP = FakeResponse({}, status_code=204)

DOWNLOAD_RESP = FakeResponse(
    {
        "errcode": 0,
        "errmsg": "ok",
        "file_list": [
            {
                "fileid": "cloud://test-env/scan/a.jpg",
                "download_url": "https://7465-test-env-1258717764.tcb.qcloud.la/a.jpg",
                "status": 0,
                "errmsg": "ok",
            }
        ],
    }
)


@pytest.fixture(autouse=True)
def _reset_token_cache():
    _TOKEN_CACHE["token"] = ""
    _TOKEN_CACHE["expires_at"] = 0
    yield
    _TOKEN_CACHE["token"] = ""
    _TOKEN_CACHE["expires_at"] = 0


def test_upload_file_two_step_cos_form_upload():
    """upload_file：拿上传链接 → COS 表单直传，字段齐全，file_id 返回"""
    fake = FakeAsyncClient([TOKEN_RESP, UPLOAD_RESP, COS_OK_RESP])
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        result = _run(client.upload_file("scan/a.jpg", b"fakejpeg-bytes"))

    assert result == {
        "file_id": "cloud://test-env/scan/a.jpg",
        "cloud_path": "scan/a.jpg",
    }
    # 步骤 1：/tcb/uploadfile 携带 access_token 与 env/path
    assert fake.gets, "应调用 cgi-bin/token 获取 access_token"
    up_url, up_kwargs = fake.posts[0]
    assert up_url == "https://api.weixin.qq.com/tcb/uploadfile"
    assert up_kwargs["params"]["access_token"] == "TOKEN_ABC"
    assert up_kwargs["json"] == {"env": "test-env", "path": "scan/a.jpg"}
    # 步骤 2：COS 表单直传（multipart 由 httpx 构造，签名字段在 data 中）
    cos_url, cos_kwargs = fake.posts[1]
    assert cos_url == "https://cos.ap-shanghai.myqcloud.com/bucket/scan/a.jpg"
    assert cos_kwargs["data"]["key"] == "scan/a.jpg"
    assert cos_kwargs["data"]["Signature"] == "q-sign-algorithm=sha1&q-sign-time=1;2"
    assert cos_kwargs["data"]["x-cos-security-token"] == "COS_TOKEN_XYZ"
    assert cos_kwargs["data"]["x-cos-meta-fileid"] == "COS_FILE_ID_9"
    filename, file_bytes = cos_kwargs["files"]["file"]
    assert filename == "a.jpg"
    assert file_bytes == b"fakejpeg-bytes"


def test_access_token_cached_across_calls():
    """access_token 模块级缓存：连续两次上传只换取一次"""
    fake = FakeAsyncClient(
        [TOKEN_RESP, UPLOAD_RESP, COS_OK_RESP, UPLOAD_RESP, COS_OK_RESP]
    )
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        _run(client.upload_file("scan/a.jpg", b"bytes-1"))
        _run(client.upload_file("scan/b.jpg", b"bytes-2"))

    assert len(fake.gets) == 1, "第二次上传应命中缓存，不重复换取 access_token"
    assert fake.posts[0][1]["json"]["path"] == "scan/a.jpg"
    assert fake.posts[2][1]["json"]["path"] == "scan/b.jpg"


def test_upload_file_missing_credentials():
    """缺少 WX_APPID/WX_SECRET 时明确报错（不发起网络请求）"""
    client = _make_client(appid="", secret="")
    with mock.patch("httpx.AsyncClient") as mocked:
        with pytest.raises(StorageAPIError, match="WX_APPID / WX_SECRET"):
            _run(client.upload_file("scan/a.jpg", b"x"))
    mocked.assert_not_called()


def test_upload_file_upload_link_errcode():
    """换取上传链接失败（errcode != 0）→ StorageAPIError"""
    fake = FakeAsyncClient([TOKEN_RESP, FakeResponse({"errcode": 40097, "errmsg": "bad"})])
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        with pytest.raises(StorageAPIError, match="获取文件上传链接失败"):
            _run(client.upload_file("scan/a.jpg", b"x"))


def test_upload_file_cos_failure():
    """COS 直传 HTTP >= 400 → StorageAPIError（含状态码与响应摘要）"""
    fake = FakeAsyncClient(
        [TOKEN_RESP, UPLOAD_RESP, FakeResponse({}, status_code=500, text="<Error>boom</Error>")]
    )
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        with pytest.raises(StorageAPIError, match="COS 上传失败 HTTP 500"):
            _run(client.upload_file("scan/a.jpg", b"x"))


def test_get_temp_file_url():
    """get_temp_file_url：batchdownloadfile 返回 download_url"""
    fake = FakeAsyncClient([TOKEN_RESP, DOWNLOAD_RESP])
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        url = _run(client.get_temp_file_url("scan/a.jpg"))

    assert url == "https://7465-test-env-1258717764.tcb.qcloud.la/a.jpg"
    dl_url, dl_kwargs = fake.posts[0]
    assert dl_url == "https://api.weixin.qq.com/tcb/batchdownloadfile"
    assert dl_kwargs["params"]["access_token"] == "TOKEN_ABC"
    assert dl_kwargs["json"]["env"] == "test-env"
    assert dl_kwargs["json"]["file_list"] == [
        {"fileid": "cloud://test-env/scan/a.jpg", "max_age": 3600}
    ]


def test_get_temp_file_url_errcode_returns_empty():
    """batchdownloadfile errcode != 0 → 返回空串（调用方降级 image_file_id）"""
    fake = FakeAsyncClient([TOKEN_RESP, FakeResponse({"errcode": -1, "errmsg": "sys error"})])
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        url = _run(client.get_temp_file_url("scan/a.jpg"))

    assert url == ""


def test_get_temp_file_url_empty_list_returns_empty():
    """batchdownloadfile 无 file_list → 返回空串"""
    fake = FakeAsyncClient([TOKEN_RESP, FakeResponse({"errcode": 0, "errmsg": "ok", "file_list": []})])
    client = _make_client()

    with mock.patch("httpx.AsyncClient", lambda **kw: fake):
        url = _run(client.get_temp_file_url("scan/a.jpg"))

    assert url == ""
