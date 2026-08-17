"""对比官方 SDK 与手写签名的请求头差异（抓取官方 SDK 实际发送的 headers）"""
import json
import os
from unittest import mock

from dotenv import load_dotenv

load_dotenv()

import requests

captured = {}


def spy_request(self, method, url, **kwargs):
    captured["method"] = method
    captured["url"] = url
    captured["headers"] = dict(kwargs.get("headers") or {})
    captured["body"] = kwargs.get("data") or kwargs.get("json")
    return mock.MagicMock()

from tencentcloud.common import credential as cred_lib
from tencentcloud.tcb.v20180608 import models as tcb_models
from tencentcloud.tcb.v20180608 import tcb_client as tcb_client_lib

cred = cred_lib.Credential(
    os.getenv("TENCENTCLOUD_SECRETID"),
    os.getenv("TENCENTCLOUD_SECRETKEY"),
    os.getenv("TENCENTCLOUD_SESSIONTOKEN") or None,
)
client = tcb_client_lib.TcbClient(cred, os.getenv("TCB_REGION", "ap-shanghai"))

req = tcb_models.RunCommandsRequest()
req.EnvId = os.getenv("TCB_ENV_ID")
req.MgoCommands = [
    {
        "TableName": "audio_asset",
        "CommandType": "QUERY",
        "Command": json.dumps({"find": "audio_asset", "filter": {}, "skip": 0, "limit": 1}),
    }
]

with mock.patch.object(requests.Session, "request", spy_request):
    try:
        client.RunCommands(req)
    except Exception:
        pass

print("URL:", captured.get("url"))
print("BODY:", captured.get("body"))
for k, v in captured.get("headers", {}).items():
    print(f"  {k}: {v}")
