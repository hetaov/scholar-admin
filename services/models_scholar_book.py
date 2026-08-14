"""学者 × 教材 关联模型 — scholar_book（Phase 5 关联与学习计划）

集合：
- `scholar_book`：一个学者对一本教材只有一条记录（复合键 `{scholar_id}_{textbook_id}`），
  记录断点（current_chapter_id / current_lesson_id）、最后学习时间（last_studied_at）
  与累计学习时长（total_time_spent）。

写入语义：
- `upsert_scholar_book`：存在则更新断点/时间并增量累加 total_time_spent，不存在则插入。
  用于两个场景：
  1. 断点续学（PUT /scholar/{scholar_id}/books/{textbook_id}/position）
  2. 会话结算回写（end_tracking_session 内调用，刷新 last_studied_at + 累加时长）

字段（与 references/target-model.md 对齐）：
- `status`：not_started / learning / completed（首次加入即 learning）
- `total_time_spent`：累计学习时长（秒），端侧结算时增量累加，可周期重算
- `last_studied_at`：最后学习时间（毫秒时间戳）
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

SCHOLAR_BOOK = "scholar_book"

# 状态枚举
BOOK_STATUS_NOT_STARTED = "not_started"
BOOK_STATUS_LEARNING = "learning"
BOOK_STATUS_COMPLETED = "completed"
VALID_BOOK_STATUSES = {
    BOOK_STATUS_NOT_STARTED,
    BOOK_STATUS_LEARNING,
    BOOK_STATUS_COMPLETED,
}


# ---------------------------------------------------------------------------
# 主键生成（纯函数）
# ---------------------------------------------------------------------------


def scholar_book_id(scholar_id: str, textbook_id: str) -> str:
    """scholar_book 复合键：{scholar_id}_{textbook_id}，保证一对多唯一。"""
    return f"{scholar_id}_{textbook_id}"


# ---------------------------------------------------------------------------
# 文档构建（纯函数）
# ---------------------------------------------------------------------------


def build_scholar_book_doc(
    *,
    scholar_id: str,
    textbook_id: str,
    current_chapter_id: str | None = None,
    current_lesson_id: str | None = None,
    status: str = BOOK_STATUS_LEARNING,
    total_time_spent: int | float = 0,
    last_studied_at: int | None = None,
    started_at: int | None = None,
    completed_at: int | None = None,
    now: int | None = None,
) -> dict:
    """构建 scholar_book 文档（纯函数，只生成不落库）。"""
    now = int(now or time.time())
    _id = scholar_book_id(scholar_id, textbook_id)
    return {
        "_id": _id,
        "scholar_book_id": _id,
        "scholar_id": scholar_id,
        "textbook_id": textbook_id,
        "status": status if status in VALID_BOOK_STATUSES else BOOK_STATUS_LEARNING,
        "current_chapter_id": current_chapter_id,
        "current_lesson_id": current_lesson_id,
        "total_time_spent": int(total_time_spent or 0),
        "last_studied_at": int(last_studied_at) if last_studied_at is not None else now,
        "started_at": int(started_at) if started_at is not None else now,
        "completed_at": completed_at,
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def get_scholar_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
) -> dict | None:
    """按 学者×教材 查询单条 scholar_book；不存在返回 None。"""
    result = await db.query(
        collection=SCHOLAR_BOOK,
        where={"_id": scholar_book_id(scholar_id, textbook_id)},
        limit=1,
    )
    records = result.get("records", [])
    return records[0] if records else None


async def upsert_scholar_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
    current_chapter_id: str | None = None,
    current_lesson_id: str | None = None,
    last_studied_at: int | None = None,
    time_delta_sec: int | float = 0,
    now: int | None = None,
) -> dict:
    """upsert scholar_book：存在则更新断点/时间并增量累加时长，不存在则插入。

    参数：
    - `current_chapter_id` / `current_lesson_id`：新断点，传 None 表示不更新。
    - `last_studied_at`：最后学习时间（毫秒）；会话结算时传 ended_at。
    - `time_delta_sec`：本次学习时长增量（秒），累加到 total_time_spent。

    返回最新 scholar_book 文档。
    """
    now = int(now or time.time())
    existing = await get_scholar_book(
        db, scholar_id=scholar_id, textbook_id=textbook_id
    )

    if existing:
        changes: dict[str, Any] = {
            "updated_at": now,
        }
        if current_chapter_id is not None:
            changes["current_chapter_id"] = current_chapter_id
        if current_lesson_id is not None:
            changes["current_lesson_id"] = current_lesson_id
        if last_studied_at is not None:
            changes["last_studied_at"] = int(last_studied_at)
        if time_delta_sec:
            changes["total_time_spent"] = int(existing.get("total_time_spent") or 0) + int(
                time_delta_sec
            )
        await db.update(
            collection=SCHOLAR_BOOK,
            where={"_id": scholar_book_id(scholar_id, textbook_id)},
            data={"$set": changes},
            multi=False,
        )
        latest = await get_scholar_book(
            db, scholar_id=scholar_id, textbook_id=textbook_id
        )
        return latest or {**existing, **changes}

    doc = build_scholar_book_doc(
        scholar_id=scholar_id,
        textbook_id=textbook_id,
        current_chapter_id=current_chapter_id,
        current_lesson_id=current_lesson_id,
        total_time_spent=int(time_delta_sec or 0),
        last_studied_at=last_studied_at,
        now=now,
    )
    await db.insert(collection=SCHOLAR_BOOK, data=doc)
    return doc


async def touch_scholar_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
    last_studied_at: int | None = None,
    time_delta_sec: int | float = 0,
    now: int | None = None,
) -> dict | None:
    """会话结算回写：刷新 last_studied_at 并增量累加 total_time_spent。

    返回回写后的 scholar_book；textbook_id 为空时返回 None（无法归属教材）。
    """
    if not textbook_id:
        return None
    return await upsert_scholar_book(
        db,
        scholar_id=scholar_id,
        textbook_id=textbook_id,
        last_studied_at=last_studied_at,
        time_delta_sec=time_delta_sec,
        now=now,
    )


async def list_scholar_books(db, *, scholar_id: str) -> list[dict]:
    """查询某学者全部教材关联（按 last_studied_at 降序）。"""
    result = await db.query(
        collection=SCHOLAR_BOOK,
        where={"scholar_id": scholar_id},
        order=[{"field": "last_studied_at", "direction": "desc"}],
        limit=1000,
    )
    return result.get("records", [])
