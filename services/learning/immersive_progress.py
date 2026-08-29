"""沉浸式五步进度持久化 — `immersive_progress` 集合 CRUD（后端只做不透明 JSON 存取）

契约：`docs_v2/03-change/proposals/2026-08-29-沉浸式五步进度持久化后端接口.md`
接口：GET / PUT / DELETE `/scholar/{scholar_id}/textbooks/{textbook_id}/groups/{group_id}/immersive-progress`

设计要点（契约 §2 G3「后端实现尽量简单」）：
- **payload 视为不透明 JSON**：字段语义全部由小程序服务层
  `services/task/immersive-progress.js` 负责（serialize/deserialize + 版本校验），
  后端零业务理解、原样存取（仅校验 `version` 与主键字段存在性）。
- 单行 upsert（复合键 `{scholar_id, textbook_id, group_id}` last-write-wins）/ 单查 / 单删。
- 与既有 `skill_state`（POST /tracking/state，SkillState 技能级状态 0-5）**互不混存**，
  既有接口/集合零改动（契约 G2）。
- 不写 `audit_log`（高频学习遥测，非管理操作）；不触发 AI 计费；
  单条 payload < 5KB（契约 §3.2.4）。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("scholar-admin.immersive_progress")

COLLECTION = "immersive_progress"

# 当前契约版本（与小程序 services/task/immersive-progress.js PROGRESS_VERSION 对齐；
# 结构变更时前后端同步 +1，deserialize 以版本校验拒绝旧数据）
PROGRESS_VERSION = 1

# 单条 payload 上限（契约 §3.2.4：< 5KB）
MAX_PAYLOAD_BYTES = 5 * 1024

# 主键字段（路径三件套 + 断点失效判定主键）
KEY_FIELDS = ("scholar_id", "textbook_id", "group_id", "sentence_id")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _require_non_empty_str(body: dict, field: str) -> str:
    value = body.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"缺少或非法字段 {field}")
    return value.strip()


def validate_progress_payload(body: dict) -> dict:
    """校验并归一化 PUT 入参（契约 §3.2.2）。

    后端只校验：`version` 匹配 + 主键/`sentence_id` 存在性 + `challenge_active`
    /`saved_at` 类型 + `payload` 为合法 JSON 且 < 5KB；不解析 payload 内部结构。

    Args:
        body: PUT 请求体（应为 dict）

    Returns:
        归一化后的存储字段（不含 created_at/updated_at，由 save 时补充）

    Raises:
        ValueError: 校验不通过（路由层转 400）
    """
    if not isinstance(body, dict):
        raise ValueError("请求体必须为 JSON 对象")

    version = body.get("version")
    if version != PROGRESS_VERSION:
        raise ValueError(f"version 非法或缺失（当前契约版本 {PROGRESS_VERSION}）")

    fields: dict[str, Any] = {"version": PROGRESS_VERSION}
    for key in KEY_FIELDS:
        fields[key] = _require_non_empty_str(body, key)

    challenge_active = body.get("challenge_active")
    if not isinstance(challenge_active, bool):
        raise ValueError("challenge_active 必须为 boolean")
    fields["challenge_active"] = challenge_active

    saved_at = body.get("saved_at")
    if not isinstance(saved_at, (int, float)) or isinstance(saved_at, bool):
        raise ValueError("saved_at 必须为 int(ms) 时间戳")
    fields["saved_at"] = int(saved_at)

    payload = body.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload 必须为 JSON 对象（不透明存储）")
    try:
        size = len(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (TypeError, ValueError):
        raise ValueError("payload 不是合法的 JSON 对象")
    if size > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload 超过 {MAX_PAYLOAD_BYTES} 字节上限（当前 {size} 字节）")
    fields["payload"] = payload

    return fields


def _build_key(scholar_id: str, textbook_id: str, group_id: str) -> dict:
    return {
        "scholar_id": scholar_id,
        "textbook_id": textbook_id,
        "group_id": group_id,
    }


async def get_progress(db, scholar_id: str, textbook_id: str, group_id: str) -> dict | None:
    """按复合键单查五步进度明细；无记录返回 None（契约 §3.2.1 无记录 → data:null）。"""
    res = await db.query(
        COLLECTION,
        where=_build_key(scholar_id, textbook_id, group_id),
        limit=1,
    )
    records = res.get("records", [])
    return records[0] if records else None


async def save_progress(db, body: dict) -> dict:
    """覆盖式 upsert（last-write-wins），返回最新文档（契约 §3.2.2）。

    - 校验失败抛 ValueError（路由层转 400）；
    - 不校验 `group_id` 存在性（宽松 upsert，group 删除场景由前端 sentenceId 失效兜底）；
    - 写入字段：复合键 + sentence_id + version + challenge_active + saved_at + payload
      + created_at（首插）/ updated_at。
    """
    fields = validate_progress_payload(body)
    scholar_id = fields["scholar_id"]
    textbook_id = fields["textbook_id"]
    group_id = fields["group_id"]

    now = _now_ms()
    existing = await get_progress(db, scholar_id, textbook_id, group_id)
    created_at = existing.get("created_at") if existing else now
    data = {"$set": {**fields, "created_at": created_at, "updated_at": now}}
    await db.update(
        COLLECTION,
        where=_build_key(scholar_id, textbook_id, group_id),
        data=data,
        upsert=True,
        multi=False,
    )
    logger.info(
        f"[immersive-progress] save → scholar_id={scholar_id}, "
        f"textbook_id={textbook_id}, group_id={group_id}, "
        f"sentence_id={fields['sentence_id']}, upserted={existing is None}"
    )
    saved = await get_progress(db, scholar_id, textbook_id, group_id)
    return saved or fields


async def clear_progress(db, scholar_id: str, textbook_id: str, group_id: str) -> bool:
    """幂等清除：删除该复合键记录，不存在也返回成功（契约 §3.2.3）。

    Returns:
        True = 确实删除了记录；False = 无记录（幂等成功）
    """
    res = await db.delete(
        COLLECTION,
        where=_build_key(scholar_id, textbook_id, group_id),
        multi=False,
    )
    deleted = bool(res.get("deleted_count", 0))
    logger.info(
        f"[immersive-progress] clear → scholar_id={scholar_id}, "
        f"textbook_id={textbook_id}, group_id={group_id}, deleted={deleted}"
    )
    return deleted
