"""家长操作审计（契约 data-model-contract §4.12.7 / §4.12.10 DM-4 / ADR-0011）

audit_log 集合：
- **append-only**：只插入，不可修改 / 删除；保留 ≥6 个月，到期归档
- 字段：log_id、action、object_ref、actor、occurred_at、result、context

action 枚举（以契约 DM-4 必审清单 5 分组为准，5+7+5=17 类）：
① 既有 5 类（ADR-0011，F3 练习纸）：
  generate / share / download / return / modify
② F2 描述 3 类：
  edit_description / draft_description / adopt_description
③ F1 总结 1 类：
  generate_knowledge_summary
④ F4 扫描 3 类：
  scan_upload / scan_classify / scan_correct
⑤ G 管理端 5 类（G0.2 新增，SOP §4.12.10(b) 定稿）：
  create_math_textbook / update_math_textbook / delete_math_textbook
  import_math_nodes / manual_edit_summary
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

# 数学学科新增动作（F2/F1/F4 链路，7 类）
AUDIT_ACTION_EDIT_DESCRIPTION = "edit_description"        # F2 人工编辑教材描述
AUDIT_ACTION_DRAFT_DESCRIPTION = "draft_description"      # F2 AI 草稿生成
AUDIT_ACTION_ADOPT_DESCRIPTION = "adopt_description"      # F2 草稿采纳
AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY = "generate_knowledge_summary"  # F1 AI 知识总结
AUDIT_ACTION_SCAN_UPLOAD = "scan_upload"                  # F4 错题扫描上传
AUDIT_ACTION_SCAN_CLASSIFY = "scan_classify"              # F4 错题自动归类
AUDIT_ACTION_SCAN_CORRECT = "scan_correct"                # F4 人工修正归类

# —— G0.2 新增：G 管理端 5 类（契约 DM-4 必审清单第 ⑤ 组） —— #
AUDIT_ACTION_CREATE_MATH_TEXTBOOK = "create_math_textbook"    # G1.3 数学生效：教材创建
AUDIT_ACTION_UPDATE_MATH_TEXTBOOK = "update_math_textbook"    # G1.3 数学生效：教材元数据修改
AUDIT_ACTION_DELETE_MATH_TEXTBOOK = "delete_math_textbook"    # G1.3 数学生效：教材删除
AUDIT_ACTION_IMPORT_MATH_NODES = "import_math_nodes"          # G2 知识点/目录导入生效
AUDIT_ACTION_MANUAL_EDIT_SUMMARY = "manual_edit_summary"      # G5.3/G5.6 人工修正总结结果

# 必审动作全集（DM-4 定稿：5+7+5 = 17 类）
# — 命名约定：双常量等价，REQUIRED_AUDIT_ACTIONS 为契约对外命名；MUST_AUDIT_ACTIONS 为历史命名（向后兼容）
MUST_AUDIT_ACTIONS = frozenset(
    {
        # ① ADR-0011：F3 练习纸（5 类）
        AUDIT_ACTION_GENERATE,
        AUDIT_ACTION_SHARE,
        AUDIT_ACTION_DOWNLOAD,
        AUDIT_ACTION_RETURN,
        AUDIT_ACTION_MODIFY,
        # ② F2 描述（3 类）
        AUDIT_ACTION_EDIT_DESCRIPTION,
        AUDIT_ACTION_DRAFT_DESCRIPTION,
        AUDIT_ACTION_ADOPT_DESCRIPTION,
        # ③ F1 总结（1 类）
        AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
        # ④ F4 扫描（3 类）
        AUDIT_ACTION_SCAN_UPLOAD,
        AUDIT_ACTION_SCAN_CLASSIFY,
        AUDIT_ACTION_SCAN_CORRECT,
        # ⑤ G 管理端（G0.2 新增，5 类）
        AUDIT_ACTION_CREATE_MATH_TEXTBOOK,
        AUDIT_ACTION_UPDATE_MATH_TEXTBOOK,
        AUDIT_ACTION_DELETE_MATH_TEXTBOOK,
        AUDIT_ACTION_IMPORT_MATH_NODES,
        AUDIT_ACTION_MANUAL_EDIT_SUMMARY,
    }
)
REQUIRED_AUDIT_ACTIONS = MUST_AUDIT_ACTIONS  # 契约命名别名（DM-4 REQUIRED_AUDIT_ACTIONS）

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
