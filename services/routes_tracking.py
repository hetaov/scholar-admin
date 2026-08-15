"""学习追踪 + 教材接口"""

from __future__ import annotations

import json as json_lib
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from services.dependencies import get_db
from services.events import STUDY_ATTEMPT
from services.models_learning import (
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_MASTERED,
)
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
from services.progress import (
    aggregate_progress,
    mastery_distribution,
    mastery_ratio,
    pick_state,
    status_distribution_array,
    status_to_int,
)
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


# ==================== 查询接口拆分（tracking/stats → 按页拆分，Phase 6） ====================


_SKILL_CODES = ("translation", "listening", "reading", "speaking")


def _to_iso(timestamp) -> str | None:
    """int 秒级时间戳 → ISO 8601 UTC 字符串；空 / 非法返回 None。"""
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
    except Exception:
        return None


async def _find_lesson_by_id(db, textbook_id: str, lesson_id: str) -> dict | None:
    """按教材 + lesson_id 找内容层级 lesson（有章教材经 chapters → lessons；无章直接按教材）。"""
    chapters = await get_chapters(db, textbook_id)
    lessons: list[dict] = []
    if chapters:
        for ch in chapters:
            lessons.extend(await get_lessons(db, ch.get("chapter_id")))
    else:
        lessons = await get_lessons_by_textbook(db, textbook_id)
    for le in lessons:
        if le.get("lesson_id") == lesson_id:
            return le
    return None


@router.get("/scholar/{scholar_id}/textbooks/{textbook_id}/lessons")
async def get_textbook_lessons(scholar_id: str, textbook_id: str, skill_code: str | None = None):
    """教材详情（lesson 列表 + 顶部三概念）— 查询接口拆分后（接口 2）。

    对教材复用服务端聚合（_aggregate_progress_for_book, detail="lesson"），
    并按能力独立聚合构造每课 skills（口径与 summary.mastery 一致）。

    返回：
    {
      "success": true,
      "data": {
        "summary": {
          "textbook_progress": 0.6,            // 0-1
          "learned_sentence_count": 60,
          "total_sentence_count": 100,
          "mastery": 0.5                       // 0-1（4 级档位加权，见 progress.mastery_ratio）
        },
        "lessons": [
          {
            "lesson_id": "...",
            "lesson_title": "...",
            "progress": {
              "overall_percent": 60,           // 0-100
              "mastery": 0.5,                  // 0-1
              "skills": {"translation": 0.5, ...},   // 各能力 0-1
              "status_distribution": [4, 0, 2, 4, 0, 0]  // 6 级计数（not_started…review_due, 0）
            }
          }
        ]
      }
    }
    """
    try:
        db = get_db()
        base = await _aggregate_progress_for_book(
            db,
            scholar_id=scholar_id,
            textbook_id=textbook_id,
            skill_code=skill_code,
            detail="lesson",
        )
        summary_raw = base.get("summary", {})

        # 各能力独立聚合（复用同一聚合路径），构造每课 skills
        skill_views: dict[str, dict[str, dict]] = {}
        for code in _SKILL_CODES:
            view = await _aggregate_progress_for_book(
                db,
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                skill_code=code,
                detail="lesson",
            )
            skill_views[code] = {
                l.get("lesson_id"): l for l in view.get("lessons", [])
            }

        lessons_out = []
        for lesson in base.get("lessons", []):
            lid = lesson.get("lesson_id")
            skills = {}
            for code, by_lid in skill_views.items():
                target = by_lid.get(lid)
                # 仅该能力有学习记录(total>0)时输出, 避免全 0 能力刷屏
                if target and (target.get("mastery_distribution") or {}).get("total"):
                    skills[code] = mastery_ratio(
                        target.get("mastery_distribution", {}),
                        lesson.get("total_sentence_count", 0),
                    )
            lessons_out.append({
                "lesson_id": lid,
                "lesson_title": lesson.get("lesson_title", ""),
                "progress": {
                    "overall_percent": round(lesson.get("progress", 0) * 100),
                    "mastery": mastery_ratio(
                        lesson.get("mastery_distribution", {}),
                        lesson.get("total_sentence_count", 0),
                    ),
                    "skills": skills,
                    "status_distribution": status_distribution_array(
                        lesson.get("mastery_distribution", {})
                    ),
                },
            })

        data = {
            "summary": {
                "textbook_progress": summary_raw.get("textbook_progress", 0.0),
                "learned_sentence_count": summary_raw.get("learned_sentence_count", 0),
                "total_sentence_count": summary_raw.get("total_sentence_count", 0),
                "mastery": mastery_ratio(
                    summary_raw.get("mastery_distribution", {}),
                    summary_raw.get("total_sentence_count", 0),
                ),
            },
            "lessons": lessons_out,
        }
        logger.info(
            f"[scholar/{scholar_id}/textbooks/{textbook_id}/lessons] "
            f"scholar_id={scholar_id}, textbook_id={textbook_id}, "
            f"skill_code={skill_code}, lessons={len(lessons_out)}"
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[scholar/.../textbooks/.../lessons] 教材详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"教材详情失败: {str(e)}")


@router.get("/tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences")
async def get_lesson_sentences(scholar_id: str, textbook_id: str, lesson_id: str):
    """章节句子明细 + 顶部概览 — 查询接口拆分后（接口 3）。

    硬约束：请求粒度=返回粒度（lesson 级），仅返回该 lesson 的句子；
    每句输出 status / skills / weakest_skill / review_count / next_review_at。
    summary 为该 lesson 内（乐观聚合后）的掌握度分布与各能力掌握度。

    返回：
    {
      "success": true,
      "data": {
        "lesson_id": "...",
        "lesson_title": "...",
        "summary": {
          "mastery": 0.5,                     // 0-1
          "skills": {"translation": 0.5, ...},     // 各能力 0-1
          "learned_sentence_count": 6,
          "total_sentence_count": 10
        },
        "sentences": [
          {
            "sentence_id": "...",
            "content": "What's he like?",
            "translation": "他是什么样的人？",
            "status": 3,                       // 0-5：not_started=0 … review_due=4
            "skills": {"translation": 3, "listening": 4, ...},  // {code: 0-5}
            "weakest_skill": "reading",
            "review_count": 2,
            "next_review_at": "2026-08-16T10:00:00Z"
          }
        ]
      }
    }
    """
    try:
        db = get_db()
        # 1. 校验 lesson 存在并取标题
        lesson = await _find_lesson_by_id(db, textbook_id, lesson_id)
        if lesson is None:
            raise HTTPException(status_code=404, detail=f"lesson 不存在: {lesson_id}")

        # 2. 句子明细（含 text / translation）
        sentences = await get_sentences_by_lesson(db, lesson_id)

        # 3. 该学者全部 skill_state（含 next_review_at）
        state_page = await db.query(
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
            limit=10000,
            select={
                "scholar_id": 1,
                "sentence_id": 1,
                "skill_code": 1,
                "status": 1,
                "mastery_score": 1,
                "attempt_count": 1,
                "next_review_at": 1,
            },
        )
        states = state_page.get("records", [])
        states_by_sentence: dict[str, list[dict]] = {}
        for st in states:
            states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

        # 4. 句子列表 + 已学计数（乐观聚合：无指定能力取 progress 最高者）
        sentence_ids = [s.get("sentence_id") for s in sentences]
        picked_by_sentence = {
            sid: pick_state(states_by_sentence.get(sid, [])) for sid in sentence_ids
        }
        learned = 0
        sentence_list = []
        for s in sentences:
            sid = s.get("sentence_id")
            picked = picked_by_sentence.get(sid)
            s_states = states_by_sentence.get(sid, [])
            skills = {
                st.get("skill_code"): status_to_int(st.get("status"))
                for st in s_states
                if st.get("skill_code")
            }
            if picked and picked.get("status") in (STATUS_LEARNED, STATUS_MASTERED):
                learned += 1
            sentence_list.append({
                "sentence_id": sid,
                "content": s.get("text", ""),
                "translation": s.get("translation", ""),
                "status": status_to_int(picked.get("status")) if picked else 0,
                "skills": skills,
                "weakest_skill": min(skills, key=skills.get) if skills else None,
                "review_count": int(picked.get("attempt_count") or 0) if picked else 0,
                "next_review_at": _to_iso(picked.get("next_review_at")) if picked else None,
            })

        # 5. summary：乐观聚合后的分布 + 各能力掌握度
        total_sentences = len(sentences)
        picked_states = [p for p in picked_by_sentence.values() if p]
        dist = mastery_distribution(picked_states)
        skill_dist = {}
        for code in _SKILL_CODES:
            code_states = [
                s for s in states
                if s.get("skill_code") == code and s.get("sentence_id") in sentence_ids
            ]
            if code_states:
                skill_dist[code] = mastery_ratio(mastery_distribution(code_states), total_sentences)

        data = {
            "lesson_id": lesson_id,
            "lesson_title": lesson.get("title", lesson.get("lesson_title", "")),
            "summary": {
                "mastery": mastery_ratio(dist, total_sentences),
                "skills": skill_dist,
                "learned_sentence_count": learned,
                "total_sentence_count": total_sentences,
            },
            "sentences": sentence_list,
        }
        logger.info(
            f"[tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences] "
            f"scholar_id={scholar_id}, sentences={len(sentence_list)}, learned={learned}"
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/.../lessons/.../sentences] 章节句子明细失败: {e}")
        raise HTTPException(status_code=500, detail=f"章节句子明细失败: {str(e)}")


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
