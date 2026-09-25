"""课文断点持久化模型 — lesson_resume_point（详情页重构 v1 · D5）

集合：
- `lesson_resume_point`：一个用户对一节课只存一条断点（复合键 `{openid}_{lesson_id}`），
  记录最后停留位置（tab / sentence_id / microflow_node）与更新时间（updated_at）。

写入语义（D5）：
- `upsert_resume_point`：存在则 `$set` 覆盖 tab/sentence_id/microflow_node/updated_at，
  不存在则插入。每用户每课只保留一条断点（只记最后位置，不记历史栈）。
- `clean_resume_point`：全课掌握或脏数据清理时删除断点。

与 skill_state 关系（红线 §5.7）：语义不同，独立存储，不复用 skill_state。
详情页训练调沉浸式技能流时，技能步走 skill_state，导航断点走 lesson_resume_point。

字段（与实现契约 §2.1 对齐）：
- `tab`：content / mastery / training / conversation（必填）
- `sentence_id`：内容/训练/会话 Tab 有值；掌握 Tab 无（可空）
- `microflow_node`：listen / shadowing / recall / compare / ai_followup（可空；
  未进微流程无节点）
- `updated_at`：最后更新时间（秒级时间戳，与 scholar_book 风格一致）
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("scholar-admin.models.lesson_resume")

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

LESSON_RESUME_POINT = "lesson_resume_point"

# tab 枚举
RESUME_TAB_CONTENT = "content"
RESUME_TAB_MASTERY = "mastery"
RESUME_TAB_TRAINING = "training"
RESUME_TAB_CONVERSATION = "conversation"
VALID_RESUME_TABS = {
    RESUME_TAB_CONTENT,
    RESUME_TAB_MASTERY,
    RESUME_TAB_TRAINING,
    RESUME_TAB_CONVERSATION,
}

# microflow_node 枚举
VALID_MICROFLOW_NODES = {"listen", "shadowing", "recall", "compare", "ai_followup"}


# ---------------------------------------------------------------------------
# 主键生成（纯函数）
# ---------------------------------------------------------------------------


def lesson_resume_point_id(openid: str, lesson_id: str) -> str:
    """lesson_resume_point 复合键：{openid}_{lesson_id}，保证每用户每课唯一。"""
    return f"{openid}_{lesson_id}"


# ---------------------------------------------------------------------------
# 入参校验（纯函数）
# ---------------------------------------------------------------------------


def _validate_resume_point_fields(tab: str, microflow_node: str | None) -> None:
    """校验 tab 必填且合法、microflow_node 非空时合法；违例抛 ValueError。"""
    if not tab or tab not in VALID_RESUME_TABS:
        raise ValueError(
            f"invalid tab={tab!r}; must be one of {sorted(VALID_RESUME_TABS)}"
        )
    if microflow_node is not None and microflow_node not in VALID_MICROFLOW_NODES:
        raise ValueError(
            f"invalid microflow_node={microflow_node!r}; "
            f"must be one of {sorted(VALID_MICROFLOW_NODES)} or None"
        )


# ---------------------------------------------------------------------------
# 文档构建（纯函数）
# ---------------------------------------------------------------------------


def build_resume_point_doc(
    *,
    openid: str,
    lesson_id: str,
    tab: str,
    sentence_id: str | None = None,
    microflow_node: str | None = None,
    now: int | None = None,
) -> dict:
    """构建 lesson_resume_point 文档（新插入用，纯函数只生成不落库）。

    - `_id = f"{openid}_{lesson_id}"`（与 skill_state 复合键风格一致，保证 upsert 幂等）
    - `updated_at = now`（缺省当前时间，秒级时间戳）
    - tab 必填且 ∈ VALID_RESUME_TABS，否则抛 ValueError
    - microflow_node 非空时须 ∈ VALID_MICROFLOW_NODES，否则抛 ValueError
    """
    _validate_resume_point_fields(tab, microflow_node)
    now = int(now or time.time())
    _id = lesson_resume_point_id(openid, lesson_id)
    return {
        "_id": _id,
        "openid": openid,
        "lesson_id": lesson_id,
        "tab": tab,
        "sentence_id": sentence_id,
        "microflow_node": microflow_node,
        "updated_at": now,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def get_resume_point(
    db,
    *,
    openid: str,
    lesson_id: str,
) -> dict | None:
    """读取某用户某课的断点；无则 None。"""
    result = await db.query(
        collection=LESSON_RESUME_POINT,
        where={"_id": lesson_resume_point_id(openid, lesson_id)},
        limit=1,
    )
    records = result.get("records", [])
    return records[0] if records else None


async def upsert_resume_point(
    db,
    *,
    openid: str,
    lesson_id: str,
    tab: str,
    sentence_id: str | None = None,
    microflow_node: str | None = None,
) -> dict:
    """按 (openid, lesson_id) 唯一键 upsert 断点；返回最新文档。

    - tab 必填且 ∈ VALID_RESUME_TABS，否则抛 ValueError
    - sentence_id / microflow_node 可空（掌握 Tab 无句；未进微流程无节点）
    - microflow_node 非空时须 ∈ VALID_MICROFLOW_NODES，否则抛 ValueError
    - 存在则 `$set` 覆盖 tab/sentence_id/microflow_node/updated_at（断点只记最后位置）
    - 不存在则 insert
    """
    _validate_resume_point_fields(tab, microflow_node)
    now = int(time.time())
    _id = lesson_resume_point_id(openid, lesson_id)
    existing = await get_resume_point(db, openid=openid, lesson_id=lesson_id)

    if existing:
        changes: dict[str, Any] = {
            "tab": tab,
            "sentence_id": sentence_id,
            "microflow_node": microflow_node,
            "updated_at": now,
        }
        update_result = await db.update(
            collection=LESSON_RESUME_POINT,
            where={"_id": _id},
            data={"$set": changes},
            multi=False,
        )
        matched = (
            update_result.get("matched_count", 0) if isinstance(update_result, dict) else 0
        )
        logger.info(
            f"[upsert_resume_point] UPDATE openid={openid} "
            f"lesson_id={lesson_id} matched={matched}"
        )
        # 防御：existing 误报时回退到 insert（与 upsert_scholar_book 一致）
        if matched == 0:
            logger.warning(
                f"[upsert_resume_point] UPDATE matched=0 (existing was truthy but "
                f"_id not found), falling back to INSERT: openid={openid} "
                f"lesson_id={lesson_id}"
            )
        else:
            latest = await get_resume_point(db, openid=openid, lesson_id=lesson_id)
            return latest or {**existing, **changes}

    doc = build_resume_point_doc(
        openid=openid,
        lesson_id=lesson_id,
        tab=tab,
        sentence_id=sentence_id,
        microflow_node=microflow_node,
        now=now,
    )
    await db.insert(collection=LESSON_RESUME_POINT, data=doc)
    logger.info(
        f"[upsert_resume_point] INSERT openid={openid} "
        f"lesson_id={lesson_id} _id={doc.get('_id')}"
    )
    return doc


async def clean_resume_point(
    db,
    *,
    openid: str,
    lesson_id: str,
) -> bool:
    """删除断点（全课掌握 / 脏数据清理）。

    返回是否删除成功：deleted_count >= 1 返回 True，无记录可删返回 False（不抛错）。
    """
    _id = lesson_resume_point_id(openid, lesson_id)
    result = await db.delete(
        collection=LESSON_RESUME_POINT,
        where={"_id": _id},
        multi=False,
    )
    deleted = result.get("deleted_count", 0) if isinstance(result, dict) else 0
    logger.info(
        f"[clean_resume_point] DELETE openid={openid} "
        f"lesson_id={lesson_id} deleted={deleted}"
    )
    return deleted >= 1
