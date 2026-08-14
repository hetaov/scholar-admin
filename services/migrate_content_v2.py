"""数据迁移脚本 —— 旧内容表 → v2 新表(Phase 1 存量迁移)

迁移内容:
1. 旧 textbook → textbook_v2(全量复制 + version=1 + 冗余计数)
2. 旧 unit → chapter / lesson(lesson_id 沿用旧 unit_id 值, 便于新旧核对)
3. 旧 sentence → sentence_v2(回填 chapter_id / lesson_id, 过渡期保留 unit_id / text_book_id)

安全约束(清理红线, 见 SKILL.md 核心原则 8):
- 本脚本**只写新表、只读旧表**, 绝不删除/修改旧表数据。
- 幂等: 新表已存在(按 _id / lesson_id / sentence_id 判重)则跳过, 可重复执行。
- 小批量: batch_size 默认 100; 回滚方式 = 删除新表数据即可(旧表未动)。

用法:
    python -m services.migrate_content_v2 [--batch 100]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid

from services.dependencies import get_db
from services.models_content import (
    CHAPTER,
    LESSON,
    SENTENCE_V2,
    TEXTBOOK_V2,
    build_chapter_doc,
    build_lesson_doc,
    build_sentence_v2_doc,
    build_textbook_v2_doc,
    group_units_into_chapters,
)

logger = logging.getLogger("scholar-admin.migrate_content_v2")

# 旧表集合名
OLD_TEXTBOOK = "textbook"
OLD_UNIT = "unit"
OLD_SENTENCE = "sentence"


async def _has_record(db, collection: str, key: str, value: str) -> bool:
    result = await db.query(collection=collection, where={key: value}, limit=1)
    return bool(result.get("records"))


async def migrate_textbook_to_v2(db, batch_size: int = 100) -> dict:
    """旧 textbook → textbook_v2(全量复制 + version=1), 幂等。"""
    stats = {"textbook_v2_created": 0, "textbook_v2_skipped": 0}
    offset = 0
    while True:
        result = await db.query(collection=OLD_TEXTBOOK, where={}, limit=batch_size, offset=offset)
        records = result.get("records", [])
        if not records:
            break
        for doc in records:
            tb_id = doc.get("_id") or doc.get("textbook_id")
            if not tb_id:
                continue
            if await _has_record(db, TEXTBOOK_V2, "_id", tb_id):
                stats["textbook_v2_skipped"] += 1
                continue
            v2 = build_textbook_v2_doc(
                tb_id,
                doc.get("title", ""),
                grade=doc.get("grade", ""),
                level=doc.get("semester", doc.get("level", "")),
                chapter_count=0,
                lesson_count=0,
                sentence_count=0,
                now=doc.get("created_at") or int(time.time()),
            )
            await db.insert(collection=TEXTBOOK_V2, data=v2)
            stats["textbook_v2_created"] += 1
        offset += len(records)
    logger.info(f"[migrate] textbook_v2: {stats}")
    return stats


async def migrate_content_to_v2(db, batch_size: int = 100) -> dict:
    """旧 unit → chapter/lesson + 旧 sentence → sentence_v2, 幂等。"""
    stats = {
        "chapter_created": 0, "chapter_skipped": 0,
        "lesson_created": 0, "lesson_skipped": 0,
        "sentence_v2_created": 0, "sentence_v2_skipped": 0,
    }
    offset = 0
    while True:
        result = await db.query(collection=OLD_TEXTBOOK, where={}, limit=batch_size, offset=offset)
        textbooks = result.get("records", [])
        if not textbooks:
            break
        for tb in textbooks:
            tb_id = tb.get("_id") or tb.get("textbook_id")
            if not tb_id:
                continue
            await _migrate_one_textbook(db, tb_id, stats)
        offset += len(textbooks)

    # 兜底: 无教材归属的孤立 unit(text_book_id 为空), 全部归入独立 orphan chapter
    await _migrate_orphan_units(db, stats)
    logger.info(f"[migrate] content: {stats}")
    return stats


async def _migrate_one_textbook(db, tb_id: str, stats: dict) -> None:
    """迁移单本教材下的 unit/sentence 到 chapter/lesson/sentence_v2。"""
    now = int(time.time())

    # 1. 取该教材全部 unit
    units_result = await db.query(
        collection=OLD_UNIT,
        where={"text_book_id": tb_id},
        limit=5000,
    )
    units = units_result.get("records", [])
    if not units:
        return

    # 2. 分组建 chapter(幂等: textbook_id + order 唯一)
    groups = group_units_into_chapters(units)
    chapter_lesson_map: dict[str, str] = {}  # lesson_id -> chapter_id

    for g in groups:
        chapter_order = g["chapter_index"]
        existing = await db.query(
            collection=CHAPTER,
            where={"textbook_id": tb_id, "order": chapter_order},
            limit=1,
        )
        if existing.get("records"):
            chapter = existing["records"][0]
            chapter_id = chapter["chapter_id"]
            stats["chapter_skipped"] += 1
        else:
            chapter_id = f"chapter_{uuid.uuid4().hex[:16]}"
            await db.insert(collection=CHAPTER, data=build_chapter_doc(
                chapter_id, tb_id, chapter_order,
                f"Chapter {chapter_order}", len(g["units"]), now,
            ))
            stats["chapter_created"] += 1

        for u in g["units"]:
            lesson_id = u.get("unit_id", "")
            if not lesson_id:
                continue
            if await _has_record(db, LESSON, "lesson_id", lesson_id):
                stats["lesson_skipped"] += 1
            else:
                await db.insert(collection=LESSON, data=build_lesson_doc(
                    lesson_id, chapter_id, tb_id,
                    u.get("unit_index", 1),
                    u.get("title", ""),
                    u.get("total_sentences", 0),
                    now,
                ))
                stats["lesson_created"] += 1
            chapter_lesson_map[lesson_id] = chapter_id

    # 3. 迁移句子: 按 unit_id 查旧 sentence
    for u in units:
        lesson_id = u.get("unit_id", "")
        chapter_id = chapter_lesson_map.get(lesson_id, "")
        if not lesson_id or not chapter_id:
            continue
        sent_offset = 0
        while True:
            sents = await db.query(
                collection=OLD_SENTENCE,
                where={"unit_id": lesson_id},
                limit=500,
                offset=sent_offset,
            )
            records = sents.get("records", [])
            if not records:
                break
            for s in records:
                sent_id = s.get("sentence_id", "")
                if not sent_id:
                    continue
                if await _has_record(db, SENTENCE_V2, "sentence_id", sent_id):
                    stats["sentence_v2_skipped"] += 1
                    continue
                v2 = build_sentence_v2_doc(
                    s, chapter_id, lesson_id, tb_id,
                    now=s.get("created_at") or now,
                )
                await db.insert(collection=SENTENCE_V2, data=v2)
                stats["sentence_v2_created"] += 1
            sent_offset += len(records)


async def _migrate_orphan_units(db, stats: dict) -> None:
    """处理无教材归属的孤立 unit(text_book_id 为空), 归属到全局 orphan chapter。

    幂等: 重跑时复用已存在的 orphan chapter(textbook_id 为空), 不新建空壳章。
    """
    now = int(time.time())

    units_result = await db.query(
        collection=OLD_UNIT,
        where={"text_book_id": ""},
        limit=5000,
    )
    orphan_units = [u for u in units_result.get("records", []) if u.get("unit_id")]
    if not orphan_units:
        return

    # 复用已存在的 orphan chapter, 避免重跑产生重复空壳章
    existing = await db.query(collection=CHAPTER, where={"textbook_id": ""}, limit=1)
    records = existing.get("records", [])
    if records:
        orphan_chapter_id = records[0]["chapter_id"]
        stats["chapter_skipped"] += 1
    else:
        orphan_chapter_id = f"chapter_orphan_{uuid.uuid4().hex[:8]}"
        await db.insert(collection=CHAPTER, data=build_chapter_doc(
            orphan_chapter_id, "", 1, "Orphan Chapter", len(orphan_units), now,
        ))
        stats["chapter_created"] += 1

    for u in orphan_units:
        lesson_id = u.get("unit_id", "")
        if await _has_record(db, LESSON, "lesson_id", lesson_id):
            stats["lesson_skipped"] += 1
        else:
            await db.insert(collection=LESSON, data=build_lesson_doc(
                lesson_id, orphan_chapter_id, "",
                u.get("unit_index", 1),
                u.get("title", ""),
                u.get("total_sentences", 0),
                now,
            ))
            stats["lesson_created"] += 1
        # 句子迁移与正常路径一致
        sent_offset = 0
        while True:
            sents = await db.query(
                collection=OLD_SENTENCE,
                where={"unit_id": lesson_id},
                limit=500,
                offset=sent_offset,
            )
            records = sents.get("records", [])
            if not records:
                break
            for s in records:
                sent_id = s.get("sentence_id", "")
                if not sent_id:
                    continue
                if await _has_record(db, SENTENCE_V2, "sentence_id", sent_id):
                    stats["sentence_v2_skipped"] += 1
                    continue
                v2 = build_sentence_v2_doc(
                    s, orphan_chapter_id, lesson_id, "",
                    now=s.get("created_at") or now,
                )
                await db.insert(collection=SENTENCE_V2, data=v2)
                stats["sentence_v2_created"] += 1
            sent_offset += len(records)


async def run_migration(db=None, batch_size: int = 100) -> dict:
    """执行完整迁移(旧表只读, 新表写入)。返回统计。"""
    db = db or get_db()
    result = {}
    result.update(await migrate_textbook_to_v2(db, batch_size))
    result.update(await migrate_content_to_v2(db, batch_size))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 内容模型存量迁移(旧表只读)")
    parser.add_argument("--batch", type=int, default=100, help="批量大小, 默认 100")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    stats = asyncio.run(run_migration(batch_size=args.batch))
    print("迁移完成:", stats)


if __name__ == "__main__":
    main()
