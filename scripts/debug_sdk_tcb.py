"""最小复现：用官方 SDK 调用 CloudBase tcb.RunCommands，对比手写签名的失败原因"""
import os

from dotenv import load_dotenv

load_dotenv()

from tencentcloud.common import credential as cred_lib
from tencentcloud.tcb.v20180608 import models as tcb_models
from tencentcloud.tcb.v20180608 import tcb_client as tcb_client_lib

print("TCB_API_HOST =", os.getenv("TCB_API_HOST"))
print("REGION =", os.getenv("TCB_REGION"))
print("ENV_ID =", os.getenv("TCB_ENV_ID"))

cred = cred_lib.Credential(
    os.getenv("TENCENTCLOUD_SECRETID"),
    os.getenv("TENCENTCLOUD_SECRETKEY"),
    os.getenv("TENCENTCLOUD_SESSIONTOKEN") or None,
)
client = tcb_client_lib.TcbClient(cred, os.getenv("TCB_REGION"))

req = tcb_models.DescribeEnvsRequest()
try:
    resp = client.DescribeEnvs(req)
    print("SDK DescribeEnvs OK:", resp.to_json_string()[:300])
except Exception as e:
    print("SDK DescribeEnvs FAIL:", e)
