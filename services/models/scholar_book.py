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
- `subject_type`：学科标识（english/math/chinese，缺省 english；与 textbook_v2.subject_type
  对齐，用于 scholar/{id}/books 按学科过滤）。读侧 getter 兼容无字段存量记录（零写回）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("scholar-admin.models.scholar_book")

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
# subject_type 多学科常量（与 models_content.normalize_textbook_doc 对齐）
# 存量 scholar_book 记录无 subject_type 字段时，读侧 getter 兜底 english（零写回）。
# ---------------------------------------------------------------------------
SCHOLAR_BOOK_SUBJECT_TYPE_ENGLISH = "english"
SCHOLAR_BOOK_SUBJECT_TYPE_MATH = "math"
SCHOLAR_BOOK_SUBJECT_TYPE_CHINESE = "chinese"
_SCHOLAR_BOOK_DEFAULT_SUBJECT_TYPE = SCHOLAR_BOOK_SUBJECT_TYPE_ENGLISH
_SCHOLAR_BOOK_VALID_SUBJECT_TYPES = frozenset({
    SCHOLAR_BOOK_SUBJECT_TYPE_ENGLISH,
    SCHOLAR_BOOK_SUBJECT_TYPE_MATH,
    SCHOLAR_BOOK_SUBJECT_TYPE_CHINESE,
})


# ---------------------------------------------------------------------------
# 读侧 getter 兼容（scholar_book 记录 → 规范化）
# ---------------------------------------------------------------------------


def normalize_scholar_book_doc(doc: dict) -> dict:
    """读取 scholar_book 记录后的 getter 兼容层。

    **契约对齐 §4.1（scholar_book 扩展）**：存量记录无 `subject_type` 字段时，
    读侧透明注入 `english`，不回写 DB，保证零迁移。

    实现原则（避免副作用）：
    - 返回新字典，不修改传入 doc；
    - `subject_type` 缺失 / None / 空串 / 非法 → 注入 `english`；
    - 显式合法值（english/math/chinese）→ 保留原值。
    """
    out = dict(doc)
    st = out.get("subject_type")
    if not st or st not in _SCHOLAR_BOOK_VALID_SUBJECT_TYPES:
        out["subject_type"] = _SCHOLAR_BOOK_DEFAULT_SUBJECT_TYPE
    return out


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
    subject_type: str | None = None,
    now: int | None = None,
) -> dict:
    """构建 scholar_book 文档（纯函数，只生成不落库）。

    subject_type 缺省 english（与 textbook_v2 对齐）；显式合法值 math/chinese 保留。
    """
    now = int(now or time.time())
    _id = scholar_book_id(scholar_id, textbook_id)
    # subject_type 兜底：None/空/非法 → english
    st = subject_type if isinstance(subject_type, str) and subject_type.strip() else None
    if st is None or st not in _SCHOLAR_BOOK_VALID_SUBJECT_TYPES:
        st = _SCHOLAR_BOOK_DEFAULT_SUBJECT_TYPE
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
        "subject_type": st,
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
    """按 学者×教材 查询单条 scholar_book；不存在返回 None。

    返回值经 normalize_scholar_book_doc 兜底（存量无 subject_type → english）。
    """
    result = await db.query(
        collection=SCHOLAR_BOOK,
        where={"_id": scholar_book_id(scholar_id, textbook_id)},
        limit=1,
    )
    records = result.get("records", [])
    return normalize_scholar_book_doc(records[0]) if records else None


async def upsert_scholar_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
    current_chapter_id: str | None = None,
    current_lesson_id: str | None = None,
    last_studied_at: int | None = None,
    time_delta_sec: int | float = 0,
    subject_type: str | None = None,
    now: int | None = None,
) -> dict:
    """upsert scholar_book：存在则更新断点/时间并增量累加时长，不存在则插入。

    参数：
    - `current_chapter_id` / `current_lesson_id`：新断点，传 None 表示不更新。
    - `last_studied_at`：最后学习时间（毫秒）；会话结算时传 ended_at。
    - `time_delta_sec`：本次学习时长增量（秒），累加到 total_time_spent。
    - `subject_type`：学科标识（首次插入时写入；更新时不传则保留原值）。

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
        # subject_type 仅在显式传入合法值时更新（保留原值语义）
        if isinstance(subject_type, str) and subject_type.strip() in _SCHOLAR_BOOK_VALID_SUBJECT_TYPES:
            changes["subject_type"] = subject_type.strip()
        update_result = await db.update(
            collection=SCHOLAR_BOOK,
            where={"_id": scholar_book_id(scholar_id, textbook_id)},
            data={"$set": changes},
            multi=False,
        )
        matched = update_result.get("matched_count", 0) if isinstance(update_result, dict) else 0
        logger.info(
            f"[upsert_scholar_book] UPDATE scholar_id={scholar_id} "
            f"textbook_id={textbook_id} matched={matched} changes={list(changes.keys())}"
        )
        # 防御：如果 update 未匹配到记录（get_scholar_book 误报 existing），
        # 回退到 insert 分支，保证数据一定写入
        if matched == 0:
            logger.warning(
                f"[upsert_scholar_book] UPDATE matched=0 (existing was truthy but _id not found), "
                f"falling back to INSERT: scholar_id={scholar_id} textbook_id={textbook_id}"
            )
        else:
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
        subject_type=subject_type,
        now=now,
    )
    await db.insert(collection=SCHOLAR_BOOK, data=doc)
    logger.info(
        f"[upsert_scholar_book] INSERT scholar_id={scholar_id} "
        f"textbook_id={textbook_id} subject_type={doc.get('subject_type')} _id={doc.get('_id')}"
    )
    return doc


async def touch_scholar_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
    last_studied_at: int | None = None,
    time_delta_sec: int | float = 0,
    subject_type: str | None = None,
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
        subject_type=subject_type,
        now=now,
    )


async def list_scholar_books(
    db,
    *,
    scholar_id: str,
    subject_type: str | None = None,
) -> list[dict]:
    """查询某学者全部教材关联（按 last_studied_at 降序）。

    subject_type 可选过滤（english/math/chinese）；不传返回全部（向后兼容）。
    返回值经 normalize_scholar_book_doc 兜底（存量无 subject_type → english）。

    **注意**：subject_type 过滤仅在内存侧执行，DB where 不带 subject_type 条件。
    原因：存量 scholar_book 记录可能无 subject_type 字段，CloudBase（MongoDB 风格）
    where 条件 `subject_type=math` 会漏掉无字段记录，导致 `?subject_type=english`
    无法返回存量英语记录。改为 DB 只按 scholar_id 查询，内存侧 normalize 后过滤，
    保证存量与新增记录均能正确过滤。
    """
    where: dict[str, Any] = {"scholar_id": scholar_id}
    result = await db.query(
        collection=SCHOLAR_BOOK,
        where=where,
        order=[{"field": "last_studied_at", "direction": "desc"}],
        limit=1000,
    )
    records = result.get("records", [])
    # 读侧 normalize：存量无 subject_type → english
    records = [normalize_scholar_book_doc(r) for r in records]
    # 内存侧过滤：按 normalize 后的 subject_type 精确匹配
    if subject_type:
        records = [r for r in records if r.get("subject_type") == subject_type]
    return records
