"""对比 httpx 与 requests 对同一 payload 的最终序列化差异"""
import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

import httpx
import requests

from services.database import CloudBaseNoSQLClient


async def gen():
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
    return client._sign_tc3("RunCommands", payload), payload


headers, payload = asyncio.run(gen())

# httpx 编码
httpx_req = httpx.Request("POST", "https://tcb.tencentcloudapi.com", json=payload, headers=headers)
print("HTTPX content:", httpx_req.content)
print("HTTPX content-length:", len(httpx_req.content))

# requests 编码
r = requests.Request("POST", "https://tcb.tencentcloudapi.com", json=payload, headers=headers)
prepared = r.prepare()
print("REQUESTS body:", prepared.body)
print("REQUESTS body-length:", len(prepared.body or b""))

print("SAME:", httpx_req.content == prepared.body)
