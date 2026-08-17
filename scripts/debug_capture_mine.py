"""抓取手写 CloudBaseNoSQLClient 实际发送的 headers/body，与官方 SDK 对比"""
import asyncio
import json
from unittest import mock

import httpx

from services.database import CloudBaseNoSQLClient

captured = {}


async def spy_post(self, url, **kwargs):
    captured["url"] = url
    captured["headers"] = dict(kwargs.get("headers") or {})
    captured["body"] = kwargs.get("json") or kwargs.get("content")
    return mock.MagicMock()


async def main():
    client = CloudBaseNoSQLClient()
    payload = {
        "EnvId": client.env_id,
        "MgoCommands": [
            {
                "TableName": "audio_asset",
                "CommandType": "QUERY",
                "Command": json.dumps({"find": "audio_asset", "filter": {}, "skip": 0, "limit": 1}),
            }
        ],
    }
    with mock.patch.object(httpx.AsyncClient, "post", spy_post):
        try:
            await client._request("RunCommands", payload)
        except Exception as e:
            print("EXC:", e)

    print("URL:", captured.get("url"))
    print("BODY:", captured.get("body"))
    for k, v in captured.get("headers", {}).items():
        print(f"  {k}: {v}")


asyncio.run(main())
