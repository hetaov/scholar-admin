"""CloudBase 项目配置"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 自动加载项目根目录下的 .env 文件（本地开发用，生产环境通过平台注入）
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# CloudBase 环境 ID
ENV_ID = os.environ.get("TCB_ENV_ID", "knowlege-graph-env-d7cwud346b70b")

# 腾讯云账号 AppID（SOE-N WSS 鉴权 URL 必需，控制台右上角头像 → 账号信息）
# 与 scripts/soe_n_verify.py 同源（TCB_APPID），本地可写 scholar-admin/.env
TCB_APPID = os.environ.get("TCB_APPID", "")

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

# ==================== 火山引擎方舟模型配置 ====================

# 火山方舟 API 地址（OpenAI 兼容接口）
VOLCANO_BASE_URL = os.environ.get(
    "VOLCANO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"
)

# 火山方舟 API Key（在 https://console.volcengine.com/ark 创建）
VOLCANO_API_KEY = os.environ.get("VOLCANO_API_KEY", "")

# 【重要】模型 ID 必须是推理接入点 ID（Endpoint ID），而非模型名称
# 步骤：控制台 → 在线推理 → 创建接入点 → 选择 doubao-1.5-vision-pro-32k → 获得 ep-xxx 格式 ID
# 获取地址：https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint
VOLCANO_VISION_MODEL = os.environ.get("VOLCANO_VISION_MODEL", "")

# 图片最大大小（字节，默认 10MB）
VOLCANO_MAX_IMAGE_SIZE = int(os.environ.get("VOLCANO_MAX_IMAGE_SIZE", 10 * 1024 * 1024))

# 支持的图片格式
VOLCANO_IMAGE_FORMATS = ["png", "jpg", "jpeg", "webp", "bmp"]

# 火山方舟对话模型（文本推理接入点 ID，用于对话匹配等纯文本场景）
# 获取地址：https://console.volcengine.com/ark/region:ark+cn-beijing/endpoint
VOLCANO_CHAT_MODEL = os.environ.get(
    "VOLCANO_CHAT_MODEL",
    os.environ.get("VOLCANO_VISION_MODEL", ""),
)
