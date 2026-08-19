"""数学学科服务包（scholar-admin /math 模块）

承载数学学科 F1~F4 后端能力：
- F1  AI 知识总结生成（knowledge_summary）
- F2  教材描述四接口（curriculum_description，本任务 F2.1）
- F3  错题练习纸 A4 选题源扩展（沿用 ADR-0010）
- F4  错题扫描上传与归类（scan_upload）

路由统一挂载在 services/routes_math.py（prefix="/math"）。
"""
from __future__ import annotations

# ===========================================================================
# curriculum_node 模型常量（契约 data-model-contract §4.12.1 / §4.12.8）
# ===========================================================================

# 教材描述适用节点类型（F2：仅 unit / lesson / knowledge_point 三类节点生效；
# 教材版本/年级节点不要求，对应 §4.12.8(a)）
DESCRIPTION_NODE_TYPES = ("unit", "lesson", "knowledge_point")

# description_source 枚举（契约 §4.12.1）
DESCRIPTION_SOURCE_MANUAL = "manual"        # 人工编辑
DESCRIPTION_SOURCE_AI_DRAFT = "ai_draft"    # AI 草稿（不入正式 description_history）
DESCRIPTION_SOURCE_AI_ADOPTED = "ai_adopted"  # 采纳的 AI 草稿

# 描述历史保留最近 N 个版本（契约 §4.12.8(a)）
DESCRIPTION_HISTORY_LIMIT = 10

# 幂等键前缀：{node_id}:v{description_version}（契约 §4.12.8(a)）
def description_idempotency_key(node_id: str, version: int) -> str:
    return f"{node_id}:v{version}"


# ===========================================================================
# AI 知识总结（F1）常量（契约 data-model-contract §4.12.8(b) / ADR-0019）
# ===========================================================================

from services.math.knowledge_summary import (
    ABILITY_DIMENSIONS,
    EXTENDED_DIFFICULTY_BANDS,
    SUMMARY_NODE_TYPES,
    SUMMARY_STATUS_DEGRADED,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_GENERATING,
    SUMMARY_STATUS_NOT_GENERATED,
    SUMMARY_STATUS_PENDING,
    SUMMARY_STATUS_SUCCESS,
)


# ===========================================================================
# 错题扫描上传（F4）常量（契约 data-model-contract §4.12.9）
# ===========================================================================

from services.math.error_scanner import (
    CLASSIFY_METHOD_AUTO_SCAN,
    CLASSIFY_METHOD_MANUAL_CORRECTED,
    CLASSIFY_STATUS_CLASSIFYING,
    CLASSIFY_STATUS_FAILED,
    CLASSIFY_STATUS_NEEDS_REVIEW,
    CLASSIFY_STATUS_PENDING,
    CLASSIFY_STATUS_SUCCESS,
    OCR_STATUS_FAILED,
    OCR_STATUS_PENDING,
    OCR_STATUS_PROCESSING,
    OCR_STATUS_SUCCESS,
    SCAN_STORAGE_PREFIX,
    SOURCE_AUTO_SCAN,
    SOURCE_MANUAL_CORRECTED,
)
