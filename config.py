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

# LLM 教材总结模型（数学 F1/F2：教材描述草稿生成 / AI 知识总结，契约 §4.12.8）
# 约束：Summary ≠ Judge ≠ Generator（同厂商不同型号），独立配置项可随时切换；
# 未配置时回退 VOLCANO_CHAT_MODEL，保证生成可用性
LLM_SUMMARY_MODEL = os.environ.get("LLM_SUMMARY_MODEL", "") or VOLCANO_CHAT_MODEL

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

# ==================== 付费白名单鉴权配置 ====================

# 鉴权模式：
#   dev（默认）    —— 请求无 X-WX-OPENID 时放行，仅本地开发/测试使用
#   enforce（生产）—— 所有请求必须携带 X-WX-OPENID 且 openid 在
#                     app_whitelist 白名单内，否则 403 拒绝
AUTH_MODE = os.environ.get("AUTH_MODE", "dev")

# 付费能力白名单集合（文档 _id 固定为 "paid"，字段 openids: string[]）
WHITELIST_COLLECTION = os.environ.get("WHITELIST_COLLECTION", "app_whitelist")

# ==================== F3.2 A4 练习纸渲染配置 ====================

# 渲染产物输出目录（PDF/PNG/预览图）。CloudRun 容器 /tmp 可写；生产可挂载持久盘后改环境变量。
# 产物通过 main.py 挂载的 StaticFiles（/static/sheets）对外提供，file_refs 引用相对 URL。
RENDER_OUTPUT_DIR = os.environ.get("RENDER_OUTPUT_DIR", "/tmp/scholar_sheets")

# 产物静态访问 URL 前缀（file_refs.pdf/png 填相对路径，如 /static/sheets/ps_xxx/sheet.pdf）
RENDER_STATIC_URL_PREFIX = os.environ.get("RENDER_STATIC_URL_PREFIX", "/static/sheets")

# 家长核对二维码（ADR-0010 A-13：≥20×20mm 含签名与有效期）
# 签名密钥：HMAC-SHA256(sheet_id + expires_at)。生产必须配置，缺失时二维码降级（qr_url 为空）。
SHEET_QR_SECRET = os.environ.get("SHEET_QR_SECRET", "")
# 二维码有效期（秒，默认 7 天）
SHEET_QR_TTL_SECONDS = int(os.environ.get("SHEET_QR_TTL_SECONDS", 7 * 24 * 3600))
# 二维码扫码落地页（家长核对 H5；MVP 由前端配置，服务端仅填 qr_url 前缀）
SHEET_QR_SCAN_PAGE = os.environ.get("SHEET_QR_SCAN_PAGE", "/scan/sheet")

# 单张渲染超时（秒，任务卡验收：单张 ≤10s）
RENDER_TIMEOUT_SECONDS = int(os.environ.get("RENDER_TIMEOUT_SECONDS", 10))

# ==================== F4.2 微信云开发 HTTP API 配置（云存储） ====================

# 微信云开发 HTTP API（api.weixin.qq.com/tcb/...）需小程序 access_token，
# 由 WX_APPID + WX_SECRET 换取（2 小时过期，tcb_storage 模块级缓存自动刷新）。
# WX_APPID 即小程序 appid（project.config.json 中 appid 字段）；
# WX_SECRET 在微信公众平台 → 开发管理 → 开发设置 → AppSecret 获取，
# 生产经云托管环境变量注入，本地开发可写 scholar-admin/.env。
# 背景：tcb.tencentcloudapi.com 的 UploadFile/GetTempFileURL 为无文档非标准 action
# （multipart 签名校验失败），云存储统一改走微信 HTTP API（见 tcb_storage.py）。
WX_APPID = os.environ.get("WX_APPID", "")
WX_SECRET = os.environ.get("WX_SECRET", "")

# ==================== F4.2 腾讯云 OCR 配置 ====================

# 腾讯云 OCR 密钥：默认复用 CloudRun 注入的 TENCENTCLOUD_SECRETID/SECRETKEY
# （与 services/asr.py 同源），可用 TENCENT_OCR_SECRET_ID/SECRET_KEY 独立覆盖。
TENCENT_OCR_SECRET_ID = os.environ.get("TENCENT_OCR_SECRET_ID", "") or SECRET_ID
TENCENT_OCR_SECRET_KEY = os.environ.get("TENCENT_OCR_SECRET_KEY", "") or SECRET_KEY

# OCR 服务区域（腾讯云 OCR 支持 ap-guangzhou / ap-shanghai 等）
TENCENT_OCR_REGION = os.environ.get("TENCENT_OCR_REGION", "ap-shanghai")

# OCR 引擎（ADR-0020：MVP 通用印刷体二选一）：
#   general_accurate → GeneralAccurateOCR（更准，适合中文数学题文本）
#   general_fast     → GeneralFastOCR（更快更省）
TENCENT_OCR_ENGINE = os.environ.get("TENCENT_OCR_ENGINE", "general_accurate")

# 单次 OCR 调用超时（秒，超时进入重试/降级，不阻断上传链路）
TENCENT_OCR_TIMEOUT_SECONDS = int(os.environ.get("TENCENT_OCR_TIMEOUT_SECONDS", 10))

# ==================== F4.3 扫描归类 Judge 配置 ====================

# Judge 单次调用超时（秒）：缩短同步阻塞时长，缓解 callContainer 15s 上限冲突
# （超时后 classify_status=failed，前端轮询重试会重新触发 Judge）
LLM_JUDGE_TIMEOUT_SECONDS = int(os.environ.get("LLM_JUDGE_TIMEOUT_SECONDS", 30))
# 知识点候选集上限（prompt token 控制，减少单次生成时间）
LLM_JUDGE_CANDIDATE_LIMIT = int(os.environ.get("LLM_JUDGE_CANDIDATE_LIMIT", 40))
# OCR 全文送入 Judge 的最大字符数（超出截断，控制 token）
LLM_JUDGE_OCR_TEXT_MAX = int(os.environ.get("LLM_JUDGE_OCR_TEXT_MAX", 4000))

# ==================== F1 知识总结 LangGraph 图编排配置（2026-08-21 SOP ⑤） ====================

# LangGraph 知识总结图开关：默认开启（true）；置 false 回退原直接调用路径
USE_LANGGRAPH_SUMMARY = os.environ.get("USE_LANGGRAPH_SUMMARY", "true").lower() == "true"

# 混元模型评估配置（混元走腾讯云 OpenAPI，独立鉴权）
HUNYUAN_APP_ID = os.environ.get("HUNYUAN_APP_ID", "")
HUNYUAN_SECRET_ID = os.environ.get("HUNYUAN_SECRET_ID", "") or SECRET_ID
HUNYUAN_SECRET_KEY = os.environ.get("HUNYUAN_SECRET_KEY", "") or SECRET_KEY
HUNYUAN_EVAL_MODEL = os.environ.get("HUNYUAN_EVAL_MODEL", "hunyuan-pro")
HUNYUAN_BASE_URL = os.environ.get(
    "HUNYUAN_BASE_URL", "https://hunyuan.tencentcloudapi.com"
)
# 混元评估调用超时（秒，超时降级跳过不阻塞）
HUNYUAN_TIMEOUT_SECONDS = int(os.environ.get("HUNYUAN_TIMEOUT_SECONDS", 15))
# 评估通过阈值（score ≥ 此值直接落盘，否则重试 ≤ 2 次）
HUNYUAN_EVAL_PASS_THRESHOLD = float(os.environ.get("HUNYUAN_EVAL_PASS_THRESHOLD", 0.7))
# 评估最大重试次数（不达标时回到生成节点重试）
HUNYUAN_EVAL_MAX_RETRIES = int(os.environ.get("HUNYUAN_EVAL_MAX_RETRIES", 2))
