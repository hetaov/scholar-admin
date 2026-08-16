"""内容模型分层数据访问辅助 —— 全部指向新表(textbook_v2 / chapter / lesson / sentence_v2)

Phase 1 目标:
    textbook_v2 → chapter → lesson → sentence_v2
新表独立创建, 旧表 textbook/sentence/unit/paragraph 迁移结束后下线(Phase 6)。
"""

from __future__ import annotations

import logging
import time
import uuid

# 新表集合名
TEXTBOOK_V2 = "textbook_v2"
CHAPTER = "chapter"
LESSON = "lesson"
SENTENCE_V2 = "sentence_v2"

DEFAULT_UNITS_PER_CHAPTER = 8

logger = logging.getLogger("scholar-admin.models.content")


# ---------------------------------------------------------------------------
# 查询辅助(全部指向新表)
# ---------------------------------------------------------------------------


async def get_chapters(db, textbook_id: str, limit: int = 1000) -> list[dict]:
    """按教材查章节, 按 order 升序。"""
    result = await db.query(
        collection=CHAPTER,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def get_lessons_by_textbook(db, textbook_id: str, limit: int = 5000) -> list[dict]:
    """按教材查全部课（无章教材: lesson 直接挂 book 下, chapter_id 为空）, 按 order 升序。"""
    result = await db.query(
        collection=LESSON,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def get_sentences_by_lesson(db, lesson_id: str, limit: int = 1000) -> list[dict]:
    """按课查句子, 按 order 升序。"""
    result = await db.query(
        collection=SENTENCE_V2,
        where={"lesson_id": lesson_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def query_all_pages(
    db,
    *,
    collection: str,
    where: dict | None = None,
    order: list[dict] | None = None,
    select: dict | None = None,
    page_size: int = 1000,
) -> list[dict]:
    """分页拉取集合全部匹配文档（规避单次 limit 上限）。

    供批量 $in 查询与全量学习数据拉取复用，避免 N+1 逐条查询。
    """
    records: list[dict] = []
    offset = 0
    while True:
        page = await db.query(
            collection=collection,
            where=where,
            order=order,
            select=select,
            offset=offset,
            limit=page_size,
        )
        recs = page.get("records", [])
        records.extend(recs)
        if len(recs) < page_size:
            break
        offset += page_size
    return records


async def get_lessons_by_chapter_ids(
    db,
    chapter_ids: list[str],
    page_size: int = 1000,
) -> list[dict]:
    """按多个章节批量查课（$in，每批 200 防 $in 数组过大），按 order 升序。

    替代逐章 get_lessons 的 N+1 查询。
    """
    if not chapter_ids:
        return []
    records: list[dict] = []
    for i in range(0, len(chapter_ids), 200):
        records.extend(await query_all_pages(
            db,
            collection=LESSON,
            where={"chapter_id": {"$in": chapter_ids[i:i + 200]}},
            order=[{"field": "order", "direction": "asc"}],
            page_size=page_size,
        ))
    return records


async def get_sentences_by_lesson_ids(
    db,
    lesson_ids: list[str],
    page_size: int = 1000,
) -> list[dict]:
    """按多个课批量查句子（$in，每批 200 防 $in 数组过大），按 order 升序。

    替代逐课 get_sentences_by_lesson 的 N+1 查询。
    """
    if not lesson_ids:
        return []
    records: list[dict] = []
    for i in range(0, len(lesson_ids), 200):
        records.extend(await query_all_pages(
            db,
            collection=SENTENCE_V2,
            where={"lesson_id": {"$in": lesson_ids[i:i + 200]}},
            order=[{"field": "order", "direction": "asc"}],
            page_size=page_size,
        ))
    return records


async def get_textbook_v2(db, textbook_id: str) -> dict | None:
    """按主键查教材 v2, 不存在返回 None。"""
    result = await db.query(collection=TEXTBOOK_V2, where={"_id": textbook_id}, limit=1)
    records = result.get("records", [])
    return records[0] if records else None


# ---------------------------------------------------------------------------
# 纯函数: 文档构建(可单测, 不触网)
# ---------------------------------------------------------------------------


def group_units_into_chapters(
    units: list[dict],
    units_per_chapter: int = DEFAULT_UNITS_PER_CHAPTER,
) -> list[dict]:
    """把 units 按顺序分成若干章。

    Returns:
        [{"chapter_index": 1, "units": [unit, ...]}, ...]
        units_per_chapter <= 0 时全部归入第 1 章。
    """
    if not units:
        return []
    size = units_per_chapter if units_per_chapter and units_per_chapter > 0 else len(units)
    groups = []
    for start in range(0, len(units), size):
        groups.append({
            "chapter_index": len(groups) + 1,
            "units": units[start:start + size],
        })
    return groups


def build_textbook_v2_doc(
    textbook_id: str,
    title: str,
    grade: str = "",
    level: str = "",
    chapter_count: int = 0,
    lesson_count: int = 0,
    sentence_count: int = 0,
    now: int | None = None,
) -> dict:
    """旧 textbook → textbook_v2(全量复制 + version=1 + 冗余计数)。"""
    now = now or int(time.time())
    return {
        "_id": textbook_id,
        "textbook_id": textbook_id,
        "title": title,
        "grade": grade,
        "level": level,
        "version": 1,
        "chapter_count": chapter_count,
        "lesson_count": lesson_count,
        "sentence_count": sentence_count,
        "created_at": now,
        "updated_at": now,
    }


def build_chapter_doc(
    chapter_id: str,
    textbook_id: str,
    order: int,
    title: str,
    lesson_count: int,
    now: int | None = None,
) -> dict:
    now = now or int(time.time())
    return {
        "_id": chapter_id,
        "chapter_id": chapter_id,
        "textbook_id": textbook_id,
        "order": order,
        "title": title,
        "lesson_count": lesson_count,
        "created_at": now,
    }


def build_lesson_doc(
    lesson_id: str,
    chapter_id: str,
    textbook_id: str,
    order: int,
    title: str,
    sentence_count: int,
    now: int | None = None,
) -> dict:
    now = now or int(time.time())
    return {
        "_id": lesson_id,
        "lesson_id": lesson_id,
        "chapter_id": chapter_id,
        "textbook_id": textbook_id,
        "order": order,
        "title": title,
        "sentence_count": sentence_count,
        "created_at": now,
        "updated_at": now,
    }


def build_sentence_v2_doc(
    sentence_doc: dict,
    chapter_id: str,
    lesson_id: str,
    textbook_id: str,
    now: int | None = None,
) -> dict:
    """sentence → sentence_v2(全量复制 + 回填 chapter_id / lesson_id / textbook_id)。"""
    now = now or int(time.time())
    return {
        "_id": sentence_doc["sentence_id"],
        "sentence_id": sentence_doc["sentence_id"],
        "textbook_id": textbook_id,
        "chapter_id": chapter_id,
        "lesson_id": lesson_id,
        "order": sentence_doc.get("index", sentence_doc.get("order", 1)),
        "text": sentence_doc.get("text", ""),
        "translation": sentence_doc.get("translation", ""),
        "audio_url": sentence_doc.get("audio_url", ""),
        "knowledge_point_ids": sentence_doc.get("knowledge_point_ids", []),
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# 双写辅助: 将一次构建的结果写入新表
# ---------------------------------------------------------------------------


async def write_content_v2(
    db,
    *,
    textbook_id: str,
    textbook_title: str,
    grade: str = "",
    level: str = "",
    units: list[dict],
    now: int | None = None,
    units_per_chapter: int = DEFAULT_UNITS_PER_CHAPTER,
    chapterless: bool = False,
) -> dict:
    """将构建的内容双写进新表(textbook_v2 + chapter + lesson + sentence_v2)。

    Args:
        textbook_id: 教材 ID; 为空时不写 textbook_v2(视觉识别等无教材场景)。
        units: 每个单元含 lesson_id / lesson_title / sentences(句子 doc 列表)。
               句子 doc 需含 sentence_id / index / text / translation 等字段。
        units_per_chapter: 每章包含的课数; 仅 chapterless=False 时生效。
        chapterless: True 时不创建 chapter, lesson 直接挂在 book 下(chapter_id 为空)。

    Returns:
        {"chapter_count": n, "lesson_count": n, "sentence_count": n}
    """
    now = now or int(time.time())

    # 分批构建时 order 不与已有记录冲突: 以已有 lesson 数量为偏移
    existing_lessons = await db.query(
        collection=LESSON, where={"textbook_id": textbook_id}, select={"_id": 1},
    )
    lesson_offset = len(existing_lessons.get("records", []))

    # 2. 构建 chapter / lesson / sentence_v2 文档
    chapter_docs: list[dict] = []
    lesson_docs: list[dict] = []
    sentence_docs: list[dict] = []

    def _append_lesson(u: dict, chapter_id: str) -> None:
        lesson_id = u["lesson_id"]
        unit_sentences = u.get("sentences", [])
        lesson_docs.append(build_lesson_doc(
            lesson_id,
            chapter_id,
            textbook_id,
            lesson_offset + len(lesson_docs) + 1,
            u.get("lesson_title", f"Lesson {len(lesson_docs) + 1}"),
            len(unit_sentences),
            now,
        ))
        for s in unit_sentences:
            sentence_docs.append(build_sentence_v2_doc(
                s, chapter_id, lesson_id, textbook_id, now,
            ))

    if chapterless:
        # 无章教材: Book → Lesson → Sentence
        for u in units:
            _append_lesson(u, "")
    else:
        # 有章教材: Book → Chapter → Lesson → Sentence
        existing_chapters = await db.query(
            collection=CHAPTER, where={"textbook_id": textbook_id}, select={"_id": 1},
        )
        chapter_offset = len(existing_chapters.get("records", []))

        groups = group_units_into_chapters(units, units_per_chapter)
        for g in groups:
            chapter_id = f"chapter_{uuid.uuid4().hex[:16]}"
            chapter_docs.append(build_chapter_doc(
                chapter_id,
                textbook_id,
                chapter_offset + g["chapter_index"],
                f"Chapter {chapter_offset + g['chapter_index']}",
                len(g["units"]),
                now,
            ))
            for u in g["units"]:
                _append_lesson(u, chapter_id)

    # 3. 写 textbook_v2(幂等 upsert, 计数累加)
    tb_doc = build_textbook_v2_doc(
        textbook_id, textbook_title, grade=grade, level=level,
        chapter_count=len(chapter_docs),
        lesson_count=len(lesson_docs),
        sentence_count=len(sentence_docs),
        now=now,
    )
    if textbook_id:
        existing_tb = await get_textbook_v2(db, textbook_id)
        if existing_tb:
            await db.update(
                collection=TEXTBOOK_V2,
                where={"_id": textbook_id},
                data={"$set": {
                    "chapter_count": int(existing_tb.get("chapter_count", 0) or 0) + len(chapter_docs),
                    "lesson_count": int(existing_tb.get("lesson_count", 0) or 0) + len(lesson_docs),
                    "sentence_count": int(existing_tb.get("sentence_count", 0) or 0) + len(sentence_docs),
                    "updated_at": now,
                }},
                multi=False,
            )
        else:
            await db.insert(collection=TEXTBOOK_V2, data=tb_doc)

    # 4. 写 chapter / lesson / sentence_v2
    if chapter_docs:
        await db.insert(collection=CHAPTER, data=chapter_docs)
    if lesson_docs:
        await db.insert(collection=LESSON, data=lesson_docs)
    if sentence_docs:
        await db.insert(collection=SENTENCE_V2, data=sentence_docs)

    logger.info(
        f"[models_content] 新表写入完成: textbook_v2={textbook_id}, "
        f"chapters={len(chapter_docs)}, lessons={len(lesson_docs)}, "
        f"sentences={len(sentence_docs)}"
    )
    return {
        "chapter_count": len(chapter_docs),
        "lesson_count": len(lesson_docs),
        "sentence_count": len(sentence_docs),
    }
