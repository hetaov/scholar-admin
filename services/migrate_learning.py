"""learning_mastery_tracking → skill_state 幂等迁移（Phase 2 能力模型）

- **只写新表、只读旧表**：任何路径均不修改/删除 learning_mastery_tracking。
- 幂等：按复合键 `{scholar_id}_{sentence_id}_{skill_code}` 判重，已存在则跳过。
- 同一句多技能：旧表无 skill_code 字段，迁移默认按 `translation` 写入一条
  （可在调用时指定 skill_code 参数分批迁移）。
- 字段映射：
  - status（中文/英文）→ normalize_status 收敛为英文枚举
  - score / mastery → mastery_score(0-100)
  - study_count → attempt_count（缺省 1）
  - last_study_time → last_studied_at（缺省 now）
  - next_review_at → 按滚动调度公式初算
  - lesson_id → 按 sentence_id 从旧 sentence 回填 unit_id（旧 unit_id 即新 lesson_id）

命令行运行（真实 CloudBase 环境，小批量可回滚）：
    python3 -m services.migrate_learning
"""

from __future__ import annotations

import asyncio
import logging
import time

from services.dependencies import get_db
from services.models_content import SENTENCE_V2
from services.models_learning import (
    DEFAULT_SKILL_CODE,
    SKILL_STATE,
    build_skill_state_doc,
    derive_status,
    skill_state_id,
    to_mastery_score,
)

logger = logging.getLogger("scholar-admin.migrate.learning")

# 旧集合名（过渡期保留）
LEGACY_TRACKING = "learning_mastery_tracking"
LEGACY_SENTENCE = "sentence"

_BATCH = 100


async def _legacy_lesson_id(db, sentence_id: str) -> str | None:
    """按 sentence_id 从旧 sentence 回填 lesson_id（旧 unit_id 即新 lesson_id）。

    优先查新表 sentence_v2，未迁移时回退旧 sentence（只读）。
    """
    try:
        page = await db.query(
            collection=SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1
        )
        recs = page.get("records") or []
        if recs and recs[0].get("lesson_id"):
            return recs[0]["lesson_id"]
    except Exception:  # noqa: BLE001 - 迁移期间任何失败都回退旧表
        pass
    try:
        page = await db.query(
            collection=LEGACY_SENTENCE, where={"sentence_id": sentence_id}, limit=1
        )
        recs = page.get("records") or []
        if recs and recs[0].get("unit_id"):
            return recs[0]["unit_id"]
    except Exception:  # noqa: BLE001
        pass
    return None


async def migrate_tracking_to_skill_state(
    db,
    *,
    skill_code: str = DEFAULT_SKILL_CODE,
    batch_size: int = _BATCH,
) -> dict:
    """幂等迁移 learning_mastery_tracking → skill_state，返回统计。"""
    stats = {"processed": 0, "created": 0, "skipped": 0, "failed": 0}
    offset = 0

    while True:
        page = await db.query(
            collection=LEGACY_TRACKING,
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
            sentence_id = rec.get("sentence_id")
            if not scholar_id or not sentence_id:
                stats["skipped"] += 1
                continue

            state_key = skill_state_id(scholar_id, sentence_id, skill_code)
            existing = await db.query(collection=SKILL_STATE, where={"_id": state_key}, limit=1)
            if existing.get("records"):
                stats["skipped"] += 1
                continue

            now = int(time.time())
            last_studied_at = rec.get("last_study_time") or rec.get("last_studied_at") or now
            try:
                last_studied_at = int(last_studied_at)
            except (TypeError, ValueError):
                last_studied_at = now
            attempt_count = rec.get("study_count") or rec.get("attempt_count") or 1
            try:
                attempt_count = max(1, int(attempt_count))
            except (TypeError, ValueError):
                attempt_count = 1
            mastery_score = to_mastery_score(rec.get("score"), rec.get("mastery"))
            status = derive_status(
                rec.get("status"),
                mastery_score,
                has_mastery=mastery_score is not None,
            )

            doc = build_skill_state_doc(
                scholar_id=scholar_id,
                sentence_id=sentence_id,
                skill_code=skill_code,
                lesson_id=await _legacy_lesson_id(db, sentence_id),
                status=status,
                mastery_score=mastery_score,
                attempt_count=attempt_count,
                last_studied_at=last_studied_at,
                now=now,
            )
            try:
                await db.insert(collection=SKILL_STATE, data=doc)
                stats["created"] += 1
            except Exception:  # noqa: BLE001
                stats["failed"] += 1

        if len(records) < batch_size:
            break
        offset += batch_size

    logger.info(
        f"[migrate_learning] processed={stats['processed']}, "
        f"created={stats['created']}, skipped={stats['skipped']}, failed={stats['failed']}"
    )
    return stats


async def main() -> None:
    """命令行入口：python3 -m services.migrate_learning"""
    logging.basicConfig(level=logging.INFO)
    db = get_db()
    stats = await migrate_tracking_to_skill_state(db)
    print(stats)


if __name__ == "__main__":
    asyncio.run(main())
