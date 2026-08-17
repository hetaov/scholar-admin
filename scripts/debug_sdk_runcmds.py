"""用官方 SDK 调用 tcb.RunCommands（真正的目标接口），验证接口本身可用"""
import json
import os

from dotenv import load_dotenv

load_dotenv()

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
try:
    resp = client.RunCommands(req)
    print("SDK RunCommands OK:", resp.to_json_string()[:400])
except Exception as e:
    print("SDK RunCommands FAIL:", e)
