"""回归测试：TCB 云存储 UploadFile 的 multipart/form-data 请求构造（F4.5 上传 500 修复）

背景（2026-08-21 真实故障）：
- tcb_storage.upload_file 复用 DB 客户端的 _request，固定 Content-Type: application/json；
- TCB UploadFile 接口要求 multipart/form-data，JSON 请求返回
  UnsupportedProtocol「this action does not support Content-Type=`application/json`」；
- 修复：upload_file 改发 multipart 字节，Content-Type 含 boundary 且随 body 参与 TC3 签名。
"""
from __future__ import annotations

import asyncio
from unittest import mock

from services.infra.tcb_storage import (
    CloudBaseStorageClient,
    _build_multipart_upload,
)

_CLIENT = CloudBaseStorageClient(
    env_id="test-env",
    secret_id="AKID-test-secret-id",
    secret_key="test-secret-key",
    region="ap-shanghai",
)


def test_build_multipart_upload_structure():
    """multipart body 含表单字段 + 文件部分，文件二进制原样嵌入（非 base64）"""
    body, content_type = _build_multipart_upload(
        {"EnvId": "test-env", "FilePath": "scan/a.jpg", "CloudPath": "scan/a.jpg"},
        file_field="FileContent",
        file_bytes=b"\xff\xd8fakejpeg",
        filename="a.jpg",
    )
    boundary = content_type.split("boundary=")[1]
    assert content_type == f"multipart/form-data; boundary={boundary}"
    text = body.decode("utf-8", "replace")
    assert 'name="EnvId"' in text
    assert 'name="FilePath"' in text
    assert 'name="CloudPath"' in text
    assert 'name="FileContent"; filename="a.jpg"' in text
    assert text.count(f"--{boundary}") >= 2
    assert body.endswith(f"--{boundary}--\r\n".encode("utf-8"))
    # 文件二进制必须原样嵌入（此前 JSON 方案用 base64 导致 Content-Type 失配）
    assert b"\xff\xd8fakejpeg" in body


def test_upload_file_passes_multipart_body_with_matching_boundary():
    """upload_file 调用 _request：action=UploadFile、body 为 multipart 字节、
    Content-Type boundary 与 body 内 boundary 一致"""
    captured = {}

    async def fake_request(
        action, payload=None, *, body=None, content_type="application/json"
    ):
        captured["action"] = action
        captured["payload"] = payload
        captured["body"] = body
        captured["content_type"] = content_type
        return {"FileId": "cloud://test-env/scan/a.jpg"}

    async def run():
        with mock.patch.object(_CLIENT, "_request", fake_request):
            return await _CLIENT.upload_file("scan/a.jpg", b"fakejpeg-bytes")

    result = asyncio.run(run())

    assert result["file_id"] == "cloud://test-env/scan/a.jpg"
    assert captured["action"] == "UploadFile"
    assert captured["payload"] is None
    boundary = captured["content_type"].split("boundary=")[1]
    assert captured["body"].startswith(f"--{boundary}".encode("utf-8"))
    assert captured["body"].endswith(f"--{boundary}--\r\n".encode("utf-8"))
    assert b"fakejpeg-bytes" in captured["body"]
    assert b'name="FileContent"' in captured["body"]


def test_request_sends_raw_body_with_custom_content_type():
    """_request 支持 body 字节 + 自定义 Content-Type（含 boundary），签名与之对齐"""
    captured = {}

    class FakeResp:
        def json(self):
            return {"Response": {"FileId": "cloud://test-env/scan/a.jpg"}}

    async def fake_post(self, url, **kwargs):
        captured["content"] = kwargs.get("content")
        captured["headers"] = kwargs.get("headers", {})
        captured["json"] = kwargs.get("json")
        return FakeResp()

    async def run():
        with mock.patch("httpx.AsyncClient.post", fake_post):
            return await _CLIENT._request(
                "UploadFile",
                None,
                body=b"multipart-bytes",
                content_type="multipart/form-data; boundary=xxxx",
            )

    resp = asyncio.run(run())

    assert resp["FileId"] == "cloud://test-env/scan/a.jpg"
    assert captured["content"] == b"multipart-bytes"
    assert captured["json"] is None, "multipart 禁止用 json= 发送"
    headers = captured["headers"]
    assert headers["Content-Type"] == "multipart/form-data; boundary=xxxx"
    assert headers["X-TC-Action"] == "UploadFile"
    assert "SignedHeaders=content-type;host" in headers["Authorization"]
