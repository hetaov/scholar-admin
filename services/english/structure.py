"""英语教材层级读取：兼容 textbook_v2 内嵌结构与独立 chapter/lesson 集合两种形态

背景（2026-08-23 修复）：标准内容管线 `models.content.write_content_v2` 产出的
`textbook_v2` 文档只有冗余计数（chapter_count / lesson_count / sentence_count），
章节 / 课时写在独立的 `chapter` / `lesson` 集合（data-model-contract §4.2）；
而英语管理端（E1.1/E1.2）此前只读 `textbook_v2.chapters[].lessons[]` /
`textbook_v2.lessons[]` 内嵌结构，导致标准管线建的教材在管理端查不到课时
（LESSON_NOT_FOUND 404）。

本模块提供统一入口 `load_lesson_entries`：内嵌结构优先，为空时回退查独立
`chapter` / `lesson` 集合，使英语管理端两种教材形态均可读。
"""

from __future__ import annotations

CHAPTER_COLLECTION = "chapter"
LESSON_COLLECTION = "lesson"


async def load_lesson_entries(db, textbook: dict) -> list[dict]:
    """遍历教材的课时条目，兼容两种存储形态。

    Args:
        db: 数据库客户端（CloudBaseNoSQLClient / FakeDB 同构接口）。
        textbook: textbook_v2 文档（含 textbook_id）。

    Returns:
        [{"chapter_id", "chapter_title", "lesson"}, ...]
        - 内嵌形态：
          - 标准：textbook.chapters[].lessons[]（契约 §4.2）
          - 无章教材：textbook.lessons[]（chapter_id=''，chapter_title='未分章'）
        - 回退形态：textbook_v2 无内嵌结构（标准内容管线 write_content_v2 产物）时，
          查独立 chapter/lesson 集合按 order 组装；chapter_id 查不到时归入「未分章」。
    """
    entries: list[dict] = []
    for ch in textbook.get("chapters") or []:
        for ls in ch.get("lessons") or []:
            entries.append(
                {
                    "chapter_id": ch.get("chapter_id") or "",
                    "chapter_title": ch.get("title") or "",
                    "lesson": ls,
                }
            )
    for ls in textbook.get("lessons") or []:
        entries.append(
            {"chapter_id": "", "chapter_title": "未分章", "lesson": ls}
        )
    if entries:
        return entries

    # 回退：独立 chapter/lesson 集合（textbook_v2 仅冗余计数，无内嵌结构）
    textbook_id = textbook.get("textbook_id") or ""
    if not textbook_id:
        return entries
    ch_q = await db.query(
        CHAPTER_COLLECTION,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=2000,
    )
    ls_q = await db.query(
        LESSON_COLLECTION,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=5000,
    )
    chapters = {c.get("chapter_id") or "": c for c in ch_q["records"]}
    for ls in ls_q["records"]:
        cid = ls.get("chapter_id") or ""
        ch = chapters.get(cid) or {}
        entries.append(
            {
                "chapter_id": cid,
                "chapter_title": ch.get("title") or "未分章",
                "lesson": ls,
            }
        )
    return entries
