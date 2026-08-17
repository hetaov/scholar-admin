"""决定性实验：同一份签名 headers，分别用 httpx 与 requests 发送"""
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

# 1) httpx
async def httpx_send():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post("https://tcb.tencentcloudapi.com", json=payload, headers=headers)
        return r.status_code, r.text[:300]

status, text = asyncio.run(httpx_send())
print("HTTPX status:", status)
print("HTTPX body:", text)

# 2) requests
r2 = requests.post("https://tcb.tencentcloudapi.com", json=payload, headers=headers, timeout=30)
print("REQUESTS status:", r2.status_code)
print("REQUESTS body:", r2.text[:300])
