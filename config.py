"""CloudBase 项目配置"""
import os

# CloudBase 环境 ID
ENV_ID = os.environ.get("TCB_ENV_ID", "knowlege-graph-env-d7cwud346b70b")

# 区域
REGION = os.environ.get("TCB_REGION", "ap-shanghai")

# 腾讯云 API 密钥（CloudRun 会自动注入）
# 本地运行请通过环境变量设置，切勿将密钥硬编码到代码中
SECRET_ID = os.environ.get("TENCENTCLOUD_SECRETID", "")
SECRET_KEY = os.environ.get("TENCENTCLOUD_SECRETKEY", "")
SESSION_TOKEN = os.environ.get("TENCENTCLOUD_SESSIONTOKEN", "")

# CloudBase HTTP API 基础地址
TCB_API_HOST = "tcb.tencentcloudapi.com"

# 服务端口
PORT = int(os.environ.get("PORT", 8080))
