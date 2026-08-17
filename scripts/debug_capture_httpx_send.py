"""抓取 httpx 编码后的实际 request（content 字节与最终 headers）"""
import asyncio
import json
from unittest import mock

import httpx

from services.database import CloudBaseNoSQLClient

captured = {}


async def spy_send(self, request, **kwargs):
    captured["url"] = str(request.url)
    captured["headers"] = dict(request.headers)
    captured["content"] = request.content
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
    with mock.patch.object(httpx.AsyncClient, "send", spy_send):
        try:
            await client._request("RunCommands", payload)
        except Exception as e:
            print("EXC:", e)

    print("URL:", captured.get("url"))
    print("CONTENT:", captured.get("content"))
    for k, v in captured.get("headers", {}).items():
        print(f"  {k}: {v}")


asyncio.run(main())
