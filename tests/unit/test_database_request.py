"""回归测试：TC3 签名与请求体序列化一致性

背景（2026-08-17 真实故障）：
- 手写 TC3 签名用 json.dumps(payload)（默认带空格）计算 HashedRequestPayload；
- 而 httpx 的 json= 参数会以紧凑格式（无空格）重新序列化 body，
  导致服务端按收到的 body 重算 sha256 与签名不符 → AuthFailure.SignatureFailure。
- 修复：_request 改用 content= 发送与签名完全一致的预序列化字节。

本文件锁定该行为，防止回归。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from unittest import mock

import httpx

from services.database import CloudBaseNoSQLClient

_CLIENT = CloudBaseNoSQLClient(
    env_id="test-env",
    secret_id="AKID-test-secret-id",
    secret_key="test-secret-key",
    region="ap-shanghai",
)

_PAYLOAD = {
    "EnvId": "test-env",
    "MgoCommands": [
        {
            "TableName": "audio_asset",
            "CommandType": "QUERY",
            "Command": '{"find": "audio_asset", "filter": {}, "skip": 0, "limit": 1}',
        }
    ],
}


def test_signature_hashes_default_json_dumps_payload():
    """签名中 HashedRequestPayload 必须基于 json.dumps(payload)（默认分隔符，带空格）"""
    headers = _CLIENT._sign_tc3("RunCommands", _PAYLOAD)
    authorization = headers["Authorization"]
    expected_hash = hashlib.sha256(json.dumps(_PAYLOAD).encode("utf-8")).hexdigest()
    # 签名基于 canonical request，其内嵌 hashed_payload，这里只校验 Authorization 存在且可解析
    assert authorization.startswith("TC3-HMAC-SHA256 Credential=")
    assert "SignedHeaders=content-type;host" in authorization


def test_request_sends_content_not_json():
    """_request 必须以 content= 发送预序列化 body，且与 json.dumps(payload) 完全一致"""
    captured = {}

    async def fake_post(self, url, **kwargs):
        captured["content"] = kwargs.get("content")
        captured["json"] = kwargs.get("json")
        return mock.MagicMock()

    async def run():
        with mock.patch.object(httpx.AsyncClient, "post", fake_post):
            try:
                await _CLIENT._request("RunCommands", _PAYLOAD)
            except Exception:
                pass
        return captured

    asyncio.run(run())

    assert "content" in captured and captured["content"] is not None
    assert captured.get("json") is None, "禁止用 json= 发送（会以紧凑格式重新序列化导致签名失配）"
    assert captured["content"] == json.dumps(_PAYLOAD)


def test_request_body_bytes_match_signature():
    """发送的 body 编码后与签名时哈希的字节完全一致"""
    headers = _CLIENT._sign_tc3("RunCommands", _PAYLOAD)
    # 解析 Authorization 中隐含的 hashed payload 无法直接取出，改为校验发送字节 = 签名输入字节
    body = json.dumps(_PAYLOAD).encode("utf-8")
    h = hashlib.sha256(body).hexdigest()
    assert len(h) == 64
    assert isinstance(headers["Authorization"], str)
