"""家长操作审计（契约 data-model-contract §4.12.7 / §4.12.10 / ADR-0011）

audit_log 集合：
- **append-only**：只插入，不可修改 / 删除；保留 ≥6 个月，到期归档
- 字段：log_id、action、object_ref、actor、occurred_at、result、context

action 枚举（以契约为准，任务卡命名冲突处契约优先）：
- 既有 5 类：generate / share / download / return / modify（F3 沿用 ADR-0011）
- 数学新增 7 类：
  - F2：edit_description / draft_description / adopt_description
  - F1：generate_knowledge_summary
  - F4：scan_upload / scan_classify / scan_correct
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any, Optional

logger = logging.getLogger("scholar-admin.audit")

# 既有必审动作（F3 沿用 ADR-0011）
AUDIT_ACTION_GENERATE = "generate"
AUDIT_ACTION_SHARE = "share"
AUDIT_ACTION_DOWNLOAD = "download"
AUDIT_ACTION_RETURN = "return"
AUDIT_ACTION_MODIFY = "modify"

# 数学学科新增动作
AUDIT_ACTION_EDIT_DESCRIPTION = "edit_description"        # F2 人工编辑教材描述
AUDIT_ACTION_DRAFT_DESCRIPTION = "draft_description"      # F2 AI 草稿生成
AUDIT_ACTION_ADOPT_DESCRIPTION = "adopt_description"      # F2 草稿采纳
AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY = "generate_knowledge_summary"  # F1 AI 知识总结
AUDIT_ACTION_SCAN_UPLOAD = "scan_upload"                  # F4 错题扫描上传
AUDIT_ACTION_SCAN_CLASSIFY = "scan_classify"              # F4 错题自动归类
AUDIT_ACTION_SCAN_CORRECT = "scan_correct"                # F4 人工修正归类

# 必审动作全集（13 类）
MUST_AUDIT_ACTIONS = frozenset(
    {
        AUDIT_ACTION_GENERATE,
        AUDIT_ACTION_SHARE,
        AUDIT_ACTION_DOWNLOAD,
        AUDIT_ACTION_RETURN,
        AUDIT_ACTION_MODIFY,
        AUDIT_ACTION_EDIT_DESCRIPTION,
        AUDIT_ACTION_DRAFT_DESCRIPTION,
        AUDIT_ACTION_ADOPT_DESCRIPTION,
        AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
        AUDIT_ACTION_SCAN_UPLOAD,
        AUDIT_ACTION_SCAN_CLASSIFY,
        AUDIT_ACTION_SCAN_CORRECT,
    }
)

# 结果枚举（契约 §4.12.7：success / failed / denied + 原因码）
AUDIT_RESULT_SUCCESS = "success"
AUDIT_RESULT_FAILED = "failed"
AUDIT_RESULT_DENIED = "denied"

# 审计集合名（与 database.py COLLECTION_NAMES 保持一致）
AUDIT_LOG_COLLECTION = "audit_log"


def _gen_log_id() -> str:
    """生成 log_id：毫秒时间戳 + 随机后缀（幂等唯一）"""
    return f"al_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


async def write_audit(
    db,
    *,
    action: str,
    object_ref: str = "",
    actor: str = "",
    result: str = AUDIT_RESULT_SUCCESS,
    context: Optional[dict[str, Any]] = None,
    reason: Optional[str] = None,
) -> dict:
    """写入一条审计记录（append-only，不可修改 / 删除）

    Args:
        db: CloudBaseNoSQLClient
        action: 必审动作枚举（MUST_AUDIT_ACTIONS）
        object_ref: 关联对象 ID（练习纸 ID / curriculum_node_id / scan_id 等）
        actor: 家长/教师账号 ID（openid）
        result: success / failed / denied
        context: 扩展上下文 {device, entry, ip?, ...}
        reason: result != success 时的原因码
    """
    if action not in MUST_AUDIT_ACTIONS:
        logger.warning(f"[audit] 未登记的动作 {action!r}，仍按必审写入")

    entry = {
        "log_id": _gen_log_id(),
        "action": action,
        "object_ref": object_ref,
        "actor": actor,
        "occurred_at": int(time.time() * 1000),
        "result": result,
        "context": context or {},
    }
    if reason:
        entry["result"] = f"{result}:{reason}"

    try:
        await db.insert(AUDIT_LOG_COLLECTION, entry)
        return entry
    except Exception as e:  # 审计写入失败不阻断业务（仅记录告警）
        logger.error(f"[audit] 写入失败 action={action}, object_ref={object_ref}: {e}")
        raise
