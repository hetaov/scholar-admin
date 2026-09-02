"""沉浸式 AI 会话态模型 — `ai_session` 集合（data-model-contract §4.19）

承载会话级上下文：`mode=start` 提交即创建（此时 history 为空、pending_task=本任务），
`mode=turn` 由后端按 session_id 装载上下文续轮；AI 产出由后台执行器回写 history
并释放 pending_task。

- 会话 ID：`s_<32hex>`；TTL 24h（与 `ai_session_task` 同口径，过期即删除）
- **单会话单在途任务**（§4.19 并发口径）：start 创建即占位（pending_task=本任务）；
  turn 提交前须空闲（pending_task=null），否则 → 200 + success=false + TURN_IN_PROGRESS
  （前端轮询到终态后再提交）
- history 轮次记录（≤20 轮，超出丢弃最旧，防 prompt 溢出）：
  - mode=start 开场产出记为首条**单 ai 记录**（role=ai 开场白，不入 user 条；
    续轮 context.history 快照含开场，保证剧情延续与素材埋伏可追溯）；
  - mode=turn 成功时由执行器回写 [user, ai]（user 条带 assisted 上报）；
  - 裁剪优先丢开场单条（_trim_history：开头 ai 先单独丢弃，再成对丢弃，避免孤立 ai）；
  - **生成失败不写 history**（该轮 AI 产出丢弃，前端可重试/重开轮次），仅释放在途位
- assisted_count：会话内「借助提示」累计（前端 assisted 上报，供形态选择启发式）
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from services.dependencies import get_db

logger = logging.getLogger("scholar-admin.session_state")

COLLECTION = "ai_session"

# 会话保留时长：24h（与 ai_session_task 同口径，start/turn 同生命周期）
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
    """创建会话态（mode=start 提交即创建）并落库，返回会话文档。

    history 初始为空；`pending_task` 为 start 首任务的 task_id（创建即占位，
    杜绝乱序回写，§4.19 并发口径）。materials 为 groups 原样（kind=new/review + sentences）。
    """
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
        f"[session_state] create → session_id={session_id}, scholar={scholar_id}, "
        f"pending_task={pending_task}"
    )
    return doc


async def get_session(db, session_id: str) -> dict | None:
    """按 session_id 查询会话态，未命中返回 None（不过滤 TTL，接口层自行过滤）。"""
    res = await db.query(COLLECTION, where={"session_id": session_id}, limit=1)
    records = res.get("records", [])
    return records[0] if records else None


async def cleanup_expired(db, now_ms: int | None = None) -> int:
    """删除 expires_at <= now 的过期会话（TTL 清理，后台巡检执行）。

    会话与其任务同 TTL（24h，同批创建），到期几乎同步；任务先删、会话后删，
    运行中的任务（≤ SESSION_LLM_TIMEOUT_SECONDS）远小于 TTL，不会被误删。
    """
    now = now_ms if now_ms is not None else _now_ms()
    res = await db.delete(COLLECTION, where={"expires_at": {"$lte": now}})
    count = res.get("deleted_count", 0)
    if count:
        logger.info(f"[session_state] cleanup → 删除过期会话 {count} 条")
    return count


async def set_pending(db, *, session_id: str, task_id: str) -> bool:
    """抢占单在途位：pending_task 空闲 → 置 task_id。

    where 限定 pending_task=null + multi=False：并发下仅一个提交成功
    （modified_count>0），失败方由提交接口返回 TURN_IN_PROGRESS。

    Returns:
        是否抢占成功
    """
    res = await db.update(
        COLLECTION,
        where={"session_id": session_id, "pending_task": None},
        data={"$set": {"pending_task": task_id, "updated_at": _now_ms()}},
        multi=False,
    )
    ok = res.get("modified_count", 0) > 0
    if ok:
        logger.info(f"[session_state] set_pending → session_id={session_id}, task_id={task_id}")
    else:
        logger.info(f"[session_state] set_pending 失败（在途位被占用）→ session_id={session_id}")
    return ok


async def release_pending(db, *, session_id: str, task_id: str) -> bool:
    """释放在途位：`pending_task == task_id` 时置 null（失败路径 / 终态兜底）。

    where 带 pending_task=task_id：只清空指向本任务的占位，不误清新任务占位。

    Returns:
        是否释放成功（会话存在且占位确属本任务）
    """
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
        logger.info(f"[session_state] release_pending → session_id={session_id}, task_id={task_id}")
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
    """turn 成功回写：history 追加 [user(可选), ai] 并释放 pending_task（§4.19）。

    仅由后台执行器在生成成功时调用（失败走 release_pending，不写 history）。
    - user_text 非空时先追加 user 条（带 assisted 上报）；
    - 始终追加 ai 条（role=ai, text/content_type/suggested_targets）；
    - 超出轮次上限由 _trim_history 丢弃最旧；
    - 同一事务内 assisted_count 累计（读改写，单会话单在途任务下无并发竞争）。

    Returns:
        是否执行成功（会话存在且在途位确属本任务）
    """
    sess = await get_session(db, session_id)
    if sess is None:
        logger.warning(f"[session_state] complete_turn → session 不存在 session_id={session_id}")
        return False
    if sess.get("pending_task") != task_id:
        logger.warning(
            f"[session_state] complete_turn → 在途位已不属于本任务 "
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
            f"[session_state] complete_turn → session_id={session_id}, task_id={task_id}, "
            f"history_len={len(history)}"
        )
    else:
        logger.warning(
            f"[session_state] complete_turn 写回失败（占位已被并发修改）→ session_id={session_id}"
        )
    return ok
