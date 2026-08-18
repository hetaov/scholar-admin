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

# LLM-as-a-Judge 模型（后置评估 L2，设计文档 §5.3 / 附录 B-4）
# 约束：Judge ≠ Generator（同厂商不同型号），独立配置项可随时切换；
# 未配置时回退 VOLCANO_CHAT_MODEL（生成模型），保证评估可用性但不保证独立性（低置信双判兜底）
LLM_JUDGE_MODEL = os.environ.get("LLM_JUDGE_MODEL", "") or VOLCANO_CHAT_MODEL

# 低置信门控阈值（设计文档 §9-2）：confidence < 阈值 不回写 SkillState
EVAL_CONFIDENCE_THRESHOLD = float(os.environ.get("EVAL_CONFIDENCE_THRESHOLD", 0.6))

# 冷启动先验默认（设计文档 §5.6.1）：无历史时 SkillState 返回先验默认（零外部调用）
COLD_START_MASTERY = float(os.environ.get("COLD_START_MASTERY", 0.35))  # 未知偏保守
COLD_START_DIFFICULTY = int(os.environ.get("COLD_START_DIFFICULTY", 1))  # 最低档起步

# 证据稀疏阈值（设计文档 §5.6.2）：attempt_count < MIN_EVIDENCE 时更新权重打折
MIN_EVIDENCE = int(os.environ.get("MIN_EVIDENCE", 3))

# ==================== L3 批量评估配置（S4.1） ====================

# 周报异常率告警阈值（设计文档 §9-6）：anomaly_rate > 阈值 触发告警
EVAL_BATCH_ALERT_RATE = float(os.environ.get("EVAL_BATCH_ALERT_RATE", 0.1))

# 周报抽样率（附录 B-3）：默认 10% 确定性抽样（seed 固定可复现）
EVAL_BATCH_SAMPLE_RATE = float(os.environ.get("EVAL_BATCH_SAMPLE_RATE", 0.1))

# 冷启动样本门槛（设计文档 §5.6.4）：抽样样本数 < 该值不启用异常率告警
EVAL_BATCH_MIN_SAMPLES = int(os.environ.get("EVAL_BATCH_MIN_SAMPLES", 100))

# ==================== S4.2 ConversationGraph L2 配置 ====================

# L2 LangGraph 会话图开关：默认开启（1）；置 0 回退 L1 轻量状态机（§5 兼容回退）
CONVERSATION_GRAPH_ENABLED = os.environ.get("CONVERSATION_GRAPH_ENABLED", "1") != "0"

# LangGraph checkpointer 持久化集合（契约 data-model-contract §4.11.4）
# 以 thread_id=session_id 维度保存每轮 checkpoint，支持断点续聊
CONVERSATION_CHECKPOINT_COLLECTION = os.environ.get(
    "CONVERSATION_CHECKPOINT_COLLECTION", "conversation_graph_checkpoint"
)

# ==================== S4.3 AI Planner 配置 ====================

# AI Planner 开关：默认开启（1）；置 0 回退 S3.3 /training/recommend 简单推荐
PLANNER_ENABLED = os.environ.get("PLANNER_ENABLED", "1") != "0"

# 推荐条数上限（默认：复习 5 / 活动 4）
PLANNER_TOP_REVIEW_ITEMS = int(os.environ.get("PLANNER_TOP_REVIEW_ITEMS", 5))
PLANNER_TOP_ACTIVITIES = int(os.environ.get("PLANNER_TOP_ACTIVITIES", 4))

# ==================== P2-2 RAG Retriever 配置 ====================

# RAG 开关：默认开启（1）；置 0 回退非向量版（book/lesson/sentences 占位，S4.3 行为）
RAG_RETRIEVER_ENABLED = os.environ.get("RAG_RETRIEVER_ENABLED", "1") != "0"

# 方舟 embedding 模型（推理接入点/模型 ID，OpenAI 兼容接口 /embeddings）。
# 未配置时 retriever 降级 no-op（返回空召回，不阻断 planner 主链路）
RAG_EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "")

# 句子向量缓存集合（契约 data-model-contract §4.13 sentence_embedding）
RAG_EMBEDDING_COLLECTION = os.environ.get("RAG_EMBEDDING_COLLECTION", "sentence_embedding")

# 跨课召回条数（Optional knowledge top-K）
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", 5))

# embedding 批量大小（每批文本数，防单请求过大）
RAG_EMBED_BATCH_SIZE = int(os.environ.get("RAG_EMBED_BATCH_SIZE", 16))
