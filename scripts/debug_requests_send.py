"""隔离实验：用 requests 发送手写签名，判断问题在签名内容还是 httpx 层"""
import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

import requests

from services.database import CloudBaseNoSQLClient


async def get_headers():
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
    headers = client._sign_tc3("RunCommands", payload)
    return headers, payload


headers, payload = asyncio.run(get_headers())
resp = requests.post(
    "https://tcb.tencentcloudapi.com",
    json=payload,
    headers=headers,
    timeout=30,
)
print("status:", resp.status_code)
print("body:", resp.text[:400])
