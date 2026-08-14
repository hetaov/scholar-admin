"""task → scholar_book 幂等迁移（Phase 5 学者×教材关联）

- **只写新表、只读旧表**：任何路径均不修改/删除 task。
- 幂等：按复合键 `_id = {scholar_id}_{textbook_id}` 判重，已存在则跳过。
- 字段映射：
  - task.scholar_id   → scholar_book.scholar_id
  - task.text_book_id → scholar_book.textbook_id
  - task.created_at   → last_studied_at / started_at（自动识别 秒/毫秒，缺省 now）
  - (缺省)            → status = learning, total_time_spent = 0

命令行运行（真实 CloudBase 环境，小批量可回滚）：
    python3 -m services.migrate_scholar_book
"""

from __future__ import annotations

import asyncio
import logging
import time

from services.dependencies import get_db
from services.models_scholar_book import (
    BOOK_STATUS_LEARNING,
    SCHOLAR_BOOK,
    build_scholar_book_doc,
    scholar_book_id,
)

logger = logging.getLogger("scholar-admin.migrate.scholar_book")

# 旧集合名（过渡期保留）
LEGACY_TASK = "task"

_BATCH = 100


def _to_ms(value) -> int | None:
    """把 task.created_at 规整为毫秒时间戳（秒/毫秒/微秒自动识别，缺省 None）。"""
    if value is None or value == "":
        return None
    try:
        ts = int(value)
    except (TypeError, ValueError):
        return None
    if ts > 10**13:  # 微秒 → 毫秒
        ts = ts // 1000
    elif ts < 10**12:  # 秒 → 毫秒
        ts = ts * 1000
    return ts


async def migrate_task_to_scholar_book(
    db,
    *,
    batch_size: int = _BATCH,
) -> dict:
    """幂等迁移 task → scholar_book，返回统计。"""
    stats = {"processed": 0, "created": 0, "skipped": 0, "failed": 0}
    offset = 0

    while True:
        page = await db.query(
            collection=LEGACY_TASK,
            where={},
            offset=offset,
            limit=batch_size,
        )
        records = page.get("records", [])
        if not records:
            break

        for rec in records:
            stats["processed"] += 1
            scholar_id = rec.get("scholar_id")
            textbook_id = rec.get("text_book_id")
            if not scholar_id or not textbook_id:
                stats["skipped"] += 1
                continue

            _id = scholar_book_id(scholar_id, textbook_id)
            existing = await db.query(collection=SCHOLAR_BOOK, where={"_id": _id}, limit=1)
            if existing.get("records"):
                stats["skipped"] += 1
                continue

            started_at = _to_ms(rec.get("created_at"))
            doc = build_scholar_book_doc(
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                status=BOOK_STATUS_LEARNING,
                total_time_spent=0,
                last_studied_at=started_at,
                started_at=started_at,
            )
            try:
                await db.insert(collection=SCHOLAR_BOOK, data=doc)
                stats["created"] += 1
            except Exception:  # noqa: BLE001
                stats["failed"] += 1

        if len(records) < batch_size:
            break
        offset += batch_size

    logger.info(
        f"[migrate_scholar_book] processed={stats['processed']}, "
        f"created={stats['created']}, skipped={stats['skipped']}, failed={stats['failed']}"
    )
    return stats


async def main() -> None:
    """命令行入口：python3 -m services.migrate_scholar_book"""
    logging.basicConfig(level=logging.INFO)
    db = get_db()
    stats = await migrate_task_to_scholar_book(db)
    print(stats)


if __name__ == "__main__":
    asyncio.run(main())
