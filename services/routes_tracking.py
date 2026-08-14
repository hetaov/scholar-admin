"""学习追踪 + 教材接口"""

from __future__ import annotations

import json as json_lib
import logging

from fastapi import APIRouter, HTTPException

from services.dependencies import get_db
from services.events import STUDY_ATTEMPT
from services.models_learning import SKILL_STATE
from services.models_content import (
    get_chapters,
    get_lessons,
    get_lessons_by_textbook,
    get_sentences_by_lesson,
    get_textbook_v2,
)
from services.models_scholar_book import (
    list_scholar_books,
    touch_scholar_book,
    upsert_scholar_book,
)
from services.progress import aggregate_progress
from services.tracking_stats import compute_tracking_stats

logger = logging.getLogger("scholar-admin.routes.tracking")
router = APIRouter(tags=["追踪 & 教材"])


# ==================== 学习追踪 ====================


@router.get("/tracking/{scholar_id}")
async def get_tracking_by_scholar(scholar_id: str):
    """根据 scholar_id 查询学习追踪记录

    优先查询 skill_state 集合（Phase 2 能力模型）；若尚无迁移数据，回退查询
    learning_mastery_tracking 旧表（过渡兼容，旧表只读，Phase 6 前不清理）。
    """
    try:
        db = get_db()
        result = await db.query(
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
        )
        if not result.get("records"):
            result = await db.query(
                collection="learning_mastery_tracking",
                where={"scholar_id": scholar_id},
            )
            logger.info(
                f"[查询] skill_state 无记录, 回退 learning_mastery_tracking, "
                f"scholar_id={scholar_id}, 结果={result}"
            )
        else:
            logger.info(
                f"[查询] 查询 skill_state 集合, scholar_id={scholar_id}, 结果={result}"
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/tracking/stats")
async def get_tracking_stats(data: dict):
    """统计学习进度 — Phase 4 起服务端聚合（skill_state + 内容层级逐级聚合）

    请求体（Phase 4 推荐）：
    {
      "scholar_id": "scholar_xxx",   // 必填
      "textbook_id": "tb_xxx",       // 教材（兼容别名 text_book_id）
      "skill_code": "translation",   // 可选, 按能力维度独立聚合
      "detail": "lesson"             // 可选: lesson(默认, summary+课级统计) / full(全量,兼容) / chapter(章节树含课) / overview(章级) / summary(仅汇总)
    }

    兼容入口（Phase 2/3，客户端上报，仍可用）：
    {
      "scholar_id": "scholar_xxx",
      "text_book_id": "tb_xxx",      // 兼容入口下必填
      "record_list": [               // 客户端学习记录
        {"sentence_id": "sent_xxx", "time_spent": 120, "status": "learned", "score": 90}
      ]
    }

    返回 data 结构（默认 detail="lesson"，仅汇总 + 课级统计，不含章节/句子明细）：
    {
      "scholar_id": "...",
      "text_book_id": "...",
      "skill_code": "...",
      "summary": {
        "total_time_spent": 秒,
        "total_time_spent_display": "1小时2分5秒",
        "textbook_progress": 0.xx,
        "learned_sentence_count": n,
        "total_sentence_count": n,
        "chapter_count": n,       // 无章教材恒为 0
        "lesson_count": n,
        "mastery_distribution": {...}
      },
      "lessons": [
        {
          "lesson_id": "...",
          "lesson_title": "...",
          "order": 1,
          "progress": 0.xx,
          "learned_sentence_count": n,
          "total_sentence_count": n,
          "mastery_distribution": {...}
        }
      ]
    }

    detail 变化说明：
    - "lesson"(默认): 仅 summary + lessons 课级统计列表（无 chapters / units / sentences）
    - "full"(兼容): 追加 chapters / 平铺 units / sentences（含句子明细）
    - "chapter"(兼容): chapters 含内嵌 lessons，省略平铺字段
    - "overview"(兼容): 章级列表（剥离 lessons 明细）
    - "summary": 仅 summary（教材列表场景）
    """
    scholar_id = str(data.get("scholar_id") or "").strip()
    record_list = data.get("record_list")

    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholar_id")

    # ── 兼容入口：客户端上报 record_list（Phase 2/3）──
    if record_list is not None:
        return await _stats_from_record_list(data, scholar_id, record_list)

    # ── Phase 4：服务端聚合（不依赖客户端上报）──
    textbook_id = str(
        data.get("textbook_id") or data.get("text_book_id") or ""
    ).strip()
    if not textbook_id:
        raise HTTPException(status_code=400, detail="缺少参数 text_book_id")
    skill_code = str(data.get("skill_code") or "").strip() or None
    detail = str(data.get("detail") or "lesson").strip() or "lesson"
    if detail not in ("full", "chapter", "overview", "lesson", "summary"):
        detail = "lesson"

    try:
        db = get_db()
        stats = await _aggregate_progress_for_book(
            db,
            scholar_id=scholar_id,
            textbook_id=textbook_id,
            skill_code=skill_code,
            detail=detail,
        )
        logger.info(
            f"[tracking/stats] scholar_id={scholar_id}, textbook_id={textbook_id}, "
            f"skill_code={skill_code}, "
            f"progress={stats['summary']['textbook_progress']}"
        )
        return {"success": True, "data": stats}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/stats] 统计异常: {e}")
        raise HTTPException(status_code=500, detail=f"统计失败: {str(e)}")


async def _aggregate_progress_for_book(
    db,
    *,
    scholar_id: str,
    textbook_id: str,
    skill_code: str | None = None,
    detail: str = "full",
) -> dict:
    """服务端聚合一本教材的进度（Phase 4 逐级聚合，stats 与 books 列表复用）。

    数据源：skill_state（可选按 skill_code 过滤）+ 内容层级（chapter → lesson →
    sentence_v2；无章教材则 book → lesson）+ study_attempt 时长，全部经
    aggregate_progress 逐级加权。
    detail 透传 aggregate_progress："full" / "chapter" / "overview" / "summary"；
    无章教材在 chapter/overview 粒度下 chapters 为空、由 lessons 返回课级进度。
    """
    where: dict = {"scholar_id": scholar_id}
    if skill_code:
        where["skill_code"] = skill_code

    # 1. 拉学者 skill_state
    state_page = await db.query(
        collection=SKILL_STATE,
        where=where,
        limit=10000,
        select={
            "scholar_id": 1,
            "sentence_id": 1,
            "skill_code": 1,
            "status": 1,
            "mastery_score": 1,
            "attempt_count": 1,
        },
    )
    states = state_page.get("records", [])

    # 2. 拉该教材内容层级（chapter → lesson → sentence_v2；无章教材则 book → lesson）
    chapters = await get_chapters(db, textbook_id)
    lessons: list[dict] = []
    sentences: list[dict] = []
    if chapters:
        for ch in chapters:
            ch_lessons = await get_lessons(db, ch.get("chapter_id"))
            lessons.extend(ch_lessons)
    else:
        lessons = await get_lessons_by_textbook(db, textbook_id)
    for le in lessons:
        sentences.extend(
            await get_sentences_by_lesson(db, le.get("lesson_id"))
        )

    # 3. 学习时长：study_attempt.time_spent 聚合
    attempt_page = await db.query(
        collection=STUDY_ATTEMPT,
        where=where,
        limit=10000,
        select={"sentence_id": 1, "skill_code": 1, "time_spent": 1},
    )
    attempts = attempt_page.get("records", [])

    return aggregate_progress(
        scholar_id=scholar_id,
        textbook_id=textbook_id,
        states=states,
        sentences=sentences,
        lessons=lessons,
        chapters=chapters,
        skill_code=skill_code,
        attempts=attempts,
        detail=detail,
    )


async def _stats_from_record_list(
    data: dict, scholar_id: str, record_list: object
) -> dict:
    """兼容入口（Phase 2/3）：由客户端 record_list 计算统计，接口契约不变。"""
    text_book_id = str(data.get("text_book_id") or "").strip()
    if not text_book_id:
        raise HTTPException(status_code=400, detail="缺少参数 text_book_id")
    if not isinstance(record_list, list):
        raise HTTPException(status_code=400, detail="record_list 必须为数组")

    try:
        db = get_db()
        page_size = 100

        # 1. 分页拉取该教材下的全部 sentence（按 unit + index 排序）
        sentences: list[dict] = []
        offset = 0
        while True:
            page = await db.query(
                collection="sentence",
                where={"text_book_id": text_book_id},
                order=[{"field": "unit_id", "direction": "asc"}, {"field": "index", "direction": "asc"}],
                offset=offset,
                limit=page_size,
                select={
                    "sentence_id": 1,
                    "unit_id": 1,
                    "index": 1,
                    "text": 1,
                    "text_book_id": 1,
                },
            )
            records = page.get("records", [])
            sentences.extend(records)
            if len(records) < page_size:
                break
            offset += page_size

        # 2. 分页拉取该教材下的全部 unit
        units: list[dict] = []
        offset = 0
        while True:
            page = await db.query(
                collection="unit",
                where={"text_book_id": text_book_id},
                offset=offset,
                limit=page_size,
                select={
                    "unit_id": 1,
                    "title": 1,
                    "text_book_id": 1,
                    "total_sentences": 1,
                },
            )
            records = page.get("records", [])
            units.extend(records)
            if len(records) < page_size:
                break
            offset += page_size

        stats = compute_tracking_stats(
            scholar_id=scholar_id,
            text_book_id=text_book_id,
            record_list=record_list,
            sentences=sentences,
            units=units,
        )
        logger.info(
            f"[tracking/stats] scholar_id={scholar_id}, text_book_id={text_book_id}, "
            f"records={len(record_list)}, "
            f"total_time={stats['summary']['total_time_spent']}s, "
            f"progress={stats['summary']['textbook_progress']}"
        )
        return {"success": True, "data": stats}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/stats] 统计异常: {e}")
        raise HTTPException(status_code=500, detail=f"统计失败: {str(e)}")


# ==================== 教材管理 ====================


@router.get("/textbook")
async def get_textbook_all():
    """查询所有教材列表 — textbook 集合全部数据"""
    try:
        db = get_db()
        result = await db.query(collection="textbook", where={})
        logger.info(f"[查询] 查询 textbook 集合全部数据，结果={result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/textbook")
async def add_textbook(data: dict):
    """添加教材 — 请求体 {"title": "新概念2"}"""
    try:
        db = get_db()
        result = await db.insert(collection="textbook", data=data)
        logger.info(
            f"[插入] textbook 添加成功: {json_lib.dumps(data, ensure_ascii=False)}"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")


# ==================== 学者 × 教材 关联（Phase 5） ====================


async def _fetch_textbook_title(db, textbook_id: str) -> str | None:
    """按 textbook_id 查书名：优先 textbook_v2，迁移过渡期回退旧表 textbook。"""
    tb = await get_textbook_v2(db, textbook_id)
    if tb and tb.get("title"):
        return tb.get("title")
    old = await db.query(
        collection="textbook",
        where={"_id": textbook_id},
        limit=1,
    )
    old_records = old.get("records", [])
    if old_records and old_records[0].get("title"):
        return old_records[0].get("title")
    return None


@router.get("/scholar/{scholar_id}/books")
async def get_scholar_books(scholar_id: str, skill_code: str | None = None):
    """我的教材列表 — 该学者全部 scholar_book 关联（含教材级进度）。

    对每本教材复用服务端聚合（skill_state + 内容层级 + study_attempt），
    返回每本书的断点（current_chapter_id / current_lesson_id）、
    累计时长（total_time_spent）与进度摘要（summary）。

    返回：
    {
      "success": true,
      "data": {
        "scholar_id": "...",
        "books": [
          {
            "textbook_id": "...",
            "title": "教材名称",
            "current_chapter_id": "...",
            "current_lesson_id": "...",
            "last_studied_at": 123,
            "total_time_spent": 60,
            "status": "learning",
            "summary": { ...aggregate_progress summary... }
          }
        ]
      }
    }
    """
    try:
        db = get_db()
        books = await list_scholar_books(db, scholar_id=scholar_id)
        enriched = []
        for book in books:
            textbook_id = book.get("textbook_id")
            if not textbook_id:
                continue
            title = await _fetch_textbook_title(db, textbook_id)
            stats = await _aggregate_progress_for_book(
                db,
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                skill_code=skill_code,
                detail="summary",
            )
            enriched.append(
                {
                    "textbook_id": textbook_id,
                    "title": title,
                    "current_chapter_id": book.get("current_chapter_id"),
                    "current_lesson_id": book.get("current_lesson_id"),
                    "last_studied_at": book.get("last_studied_at"),
                    "total_time_spent": book.get("total_time_spent"),
                    "status": book.get("status"),
                    "summary": stats.get("summary", {}),
                }
            )
        logger.info(
            f"[scholar/{scholar_id}/books] scholar_id={scholar_id}, "
            f"skill_code={skill_code}, books={len(enriched)}"
        )
        return {
            "success": True,
            "data": {"scholar_id": scholar_id, "books": enriched},
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[scholar/{scholar_id}/books] 教材列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"教材列表失败: {str(e)}")


@router.put("/scholar/{scholar_id}/books/{textbook_id}/position")
async def update_book_position(
    scholar_id: str,
    textbook_id: str,
    data: dict,
):
    """更新断点 — upsert scholar_book（同一 学者×教材 只保留一条记录）。

    请求体（均可选，至少提供一个）：
    {
      "current_chapter_id": "chapter_xxx",
      "current_lesson_id": "lesson_xxx",
      "last_studied_at": 1234567890
    }

    返回最新 scholar_book 文档。
    """
    current_chapter_id = data.get("current_chapter_id")
    current_lesson_id = data.get("current_lesson_id")
    last_studied_at = data.get("last_studied_at")
    if (
        current_chapter_id is None
        and current_lesson_id is None
        and last_studied_at is None
    ):
        raise HTTPException(
            status_code=400,
            detail="至少提供一个字段: current_chapter_id / current_lesson_id / last_studied_at",
        )
    try:
        db = get_db()
        book = await upsert_scholar_book(
            db,
            scholar_id=scholar_id,
            textbook_id=textbook_id,
            current_chapter_id=current_chapter_id,
            current_lesson_id=current_lesson_id,
            last_studied_at=last_studied_at,
        )
        logger.info(
            f"[scholar/{scholar_id}/books/{textbook_id}/position] "
            f"chapter={current_chapter_id}, lesson={current_lesson_id}, "
            f"last_studied_at={last_studied_at}"
        )
        return {"success": True, "data": book}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[scholar/{scholar_id}/books/{textbook_id}/position] 更新断点失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新断点失败: {str(e)}")
