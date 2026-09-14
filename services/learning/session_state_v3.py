"""沉浸式 AI 会话 v3 会话态模型 — `ai_session_v3` 集合（§11.2 / §11.6）

同构复制 `services/learning/session_state.py`（v2），**仅集合名参数化**为
`ai_session_v3`；语义、并发口径、history 裁剪规则与 v2 逐条一致，避免与 v2 集合混用。

- 会话 ID：`s_<32hex>`；TTL 24h（与 `ai_session_v3_task` 同口径）
- **单会话单在途任务**：start 创建即占位（pending_task=本任务）；turn 提交前须空闲
- history 轮次 ≤20（失败不写 history，仅释放在途位）
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import SESSION_V3_STATE_COLLECTION

logger = logging.getLogger("scholar-admin.session_state_v3")

COLLECTION = SESSION_V3_STATE_COLLECTION

# 会话保留时长：24h（与 ai_session_v3_task 同口径，start/turn 同生命周期）
SESSION_TTL_MS = 24 * 60 * 60 * 1000

STATUS_ACTIVE = "active"

# history 轮次上限：20 轮（每轮 ≤2 条：user+ai；开场 ai 单条也计入条数）
MAX_HISTORY_TURNS = 20
MAX_HISTORY_ENTRIES = MAX_HISTORY_TURNS * 2


def _now_ms() -> int:
    return int(time.time() * 1000)


def build_session_id() -> str:
    """生成会话 ID：`s_` + 32 位 uuid hex。"""
    return "s_" + uuid.uuid4().hex


def _trim_history(history: list[dict]) -> list[dict]:
    """裁剪 history 至上限（40 条）：从最旧丢弃，尽量保持 [user, ai] 配对完整。

    开头若为 ai（开场白单条）先单独丢弃一条，再成对丢弃，避免裁出孤立的 ai 回复。
    """
    trimmed = history
    while len(trimmed) > MAX_HISTORY_ENTRIES:
        if trimmed and trimmed[0].get("role") == "ai":
            trimmed = trimmed[1:]
        else:
            trimmed = trimmed[2:]
    return trimmed


async def create_session(
    db,
    *,
    session_id: str,
    scholar_id: str,
    scenario: dict,
    roles: dict,
    materials: list,
    pending_task: str,
) -> dict:
    """创建会话态（mode=start 提交即创建）并落库，返回会话文档。"""
    now = _now_ms()
    doc: dict[str, Any] = {
        "session_id": session_id,
        "scholar_id": scholar_id,
        "scenario": scenario,
        "roles": roles,
        "materials": materials,
        "history": [],
        "assisted_count": 0,
        "pending_task": pending_task,
        "status": STATUS_ACTIVE,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + SESSION_TTL_MS,
    }
    await db.insert(COLLECTION, doc)
    logger.info(
        f"[session_v3] create → session_id={session_id}, scholar={scholar_id}, "
        f"pending_task={pending_task}"
    )
    return doc


async def get_session(db, session_id: str) -> dict | None:
    """按 session_id 查询会话态，未命中返回 None（不过滤 TTL，接口层自行过滤）。"""
    res = await db.query(COLLECTION, where={"session_id": session_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期会话（TTL 清理，后台巡检执行）。"""
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[session_v3] cleanup → 删除过期会话 {count} 条")
    return count


async def set_pending(db, *, session_id: str, task_id: str) -> bool:
    """抢占单在途位：pending_task 空闲 → 置 task_id（并发下仅一个提交成功）。"""
    res = await db.update(
        COLLECTION,
        where={"session_id": session_id, "pending_task": None},
        data={"$set": {"pending_task": task_id, "updated_at": _now_ms()}},
        multi=False,
    )
    ok = res.get("modified_count", 0) > 0
    if ok:
        logger.info(f"[session_v3] set_pending → session_id={session_id}, task_id={task_id}")
    else:
        logger.info(f"[session_v3] set_pending 失败（在途位被占用）→ session_id={session_id}")
    return ok


async def release_pending(db, *, session_id: str, task_id: str) -> bool:
    """释放在途位：`pending_task == task_id` 时置 null（失败路径 / 终态兜底）。"""
    if not session_id or not task_id:
        return False
    res = await db.update(
        COLLECTION,
        where={"session_id": session_id, "pending_task": task_id},
        data={"$set": {"pending_task": None, "updated_at": _now_ms()}},
        multi=False,
    )
    ok = res.get("modified_count", 0) > 0
    if ok:
        logger.info(f"[session_v3] release_pending → session_id={session_id}, task_id={task_id}")
    return ok


async def complete_turn(
    db,
    *,
    session_id: str,
    task_id: str,
    ai_text: str,
    content_type: str,
    suggested_targets: list | None = None,
    user_text: str | None = None,
    assisted: bool = False,
) -> bool:
    """turn 成功回写：history 追加 [user(可选), ai] 并释放 pending_task。"""
    sess = await get_session(db, session_id)
    if sess is None:
        logger.warning(f"[session_v3] complete_turn → session 不存在 session_id={session_id}")
        return False
    if sess.get("pending_task") != task_id:
        logger.warning(
            f"[session_v3] complete_turn → 在途位已不属于本任务 "
            f"session_id={session_id}, task_id={task_id}, pending={sess.get('pending_task')}"
        )
        return False
    now = _now_ms()
    history = list(sess.get("history") or [])
    if user_text is not None and str(user_text).strip():
        history = _trim_history(
            history
            + [
                {
                    "role": "user",
                    "text": str(user_text),
                    "assisted": bool(assisted),
                    "created_at": now,
                }
            ]
        )
    history = _trim_history(
        history
        + [
            {
                "role": "ai",
                "text": ai_text,
                "content_type": content_type,
                "suggested_targets": suggested_targets or [],
                "created_at": now,
            }
        ]
    )
    res = await db.update(
        COLLECTION,
        where={"session_id": session_id, "pending_task": task_id},
        data={
            "$set": {
                "history": history,
                "pending_task": None,
                "assisted_count": int(sess.get("assisted_count") or 0)
                + (1 if assisted and user_text else 0),
                "updated_at": now,
            }
        },
        multi=False,
    )
    ok = res.get("modified_count", 0) > 0
    if ok:
        logger.info(
            f"[session_v3] complete_turn → session_id={session_id}, task_id={task_id}, "
            f"history_len={len(history)}"
        )
    else:
        logger.warning(
            f"[session_v3] complete_turn 写回失败（占位已被并发修改）→ session_id={session_id}"
        )
    return ok
