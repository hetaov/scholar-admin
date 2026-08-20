"""学习追踪 + 教材接口"""

from __future__ import annotations

import json as json_lib
import logging
from datetime import datetime, time, timedelta, timezone

import uuid

from fastapi import APIRouter, HTTPException, Query

from services.dependencies import get_db
from services.events import STUDY_ATTEMPT
from services.models_learning import (
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_MASTERED,
    STATUS_NOT_STARTED,
)
from services.models_content import (
    TEXTBOOK_V2,
    build_textbook_v2_doc,
    get_chapters,
    get_lessons_by_chapter_ids,
    get_lessons_by_textbook,
    get_sentences_by_ids,
    get_sentences_by_lesson,
    get_sentences_by_lesson_ids,
    query_all_pages,
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
from services.gamification import (
    aggregate_wrong_book,
    build_daily_goal,
    build_leaderboard,
    evaluate_badges,
)

logger = logging.getLogger("scholar-admin.routes.tracking")
router = APIRouter(tags=["追踪 & 教材"])


# ==================== 学习追踪 ====================


@router.get("/tracking/leaderboard")
async def get_leaderboard(
    period: str = "week",
    metric: str = "minutes",
    limit: int = 10,
    scholar_id: str | None = None,
):
    """排行榜（P2/F8）— 契约 §3.7。

    入参：`{ period?: week|month|all, metric?: minutes|sentences, limit?: 10(≤50), scholar_id? }`
    出参：`{ success, data: { period, metric, items: [{rank, scholar_id, name, value, is_me}], my_rank } }`

    口径：
    - 按周期过滤 `study_attempt.created_at` 窗口（week=近7天 / month=近30天 / all=全量）；
    - 按 scholar_id 分组：`minutes` = time_spent 求和（秒÷60 四舍五入）、
      `sentences` = 去重 sentence_id 数；内存排序降序取 TopN；
    - `name` 批量加载自 `scholars` 集合（`_id`=scholar_id、`name`=昵称），不返回 openid；
    - `is_me`/`my_rank`：请求带 scholar_id 时计算；未上榜 `my_rank=null`；
    - 无记录 → `success:true, items:[], my_rank:null`，不报错。
    """
    if period not in ("week", "month", "all"):
        raise HTTPException(status_code=400, detail=f"period 非法: {period}（应为 week/month/all）")
    if metric not in ("minutes", "sentences"):
        raise HTTPException(status_code=400, detail=f"metric 非法: {metric}（应为 minutes/sentences）")
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail=f"limit 应在 1-50 之间: {limit}")
    try:
        db = get_db()
        data = await build_leaderboard(
            db,
            period=period,
            metric=metric,
            limit=limit,
            scholar_id=scholar_id,
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/leaderboard] 排行榜失败: {e}")
        raise HTTPException(status_code=500, detail=f"排行榜失败: {str(e)}")


@router.get("/tracking/{scholar_id}")
async def get_tracking_by_scholar(scholar_id: str):
    """根据 scholar_id 查询学习追踪记录（skill_state 能力模型，不再回退旧表）

    只查询 skill_state（Phase 2 能力模型）；无记录时直接打印日志提示。
    """
    try:
        db = get_db()
        result = await db.query(
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
        )
        if not result.get("records"):
            # 不回退旧表：无记录直接打日志（可能未迁移或该学者确无学习数据）
            logger.warning(
                f"[查询] skill_state 无记录, 不回退旧表, scholar_id={scholar_id} "
                f"(请检查迁移或该学者是否确无学习数据)"
            )
        else:
            logger.info(
                f"[查询] 查询 skill_state 集合, scholar_id={scholar_id}, 结果={result}"
            )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


async def _load_book_content(
    db,
    textbook_id: str,
    *,
    with_sentences: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """一次拉取教材内容层级（chapters / lessons / sentences），批量 $in 避免 N+1。

    有章教材：chapter → lesson → sentence_v2；无章教材：book → lesson → sentence_v2。
    返回 (chapters, lessons, sentences)。
    """
    chapters = await get_chapters(db, textbook_id)
    if chapters:
        lessons = await get_lessons_by_chapter_ids(
            db, [c.get("chapter_id") for c in chapters]
        )
    else:
        lessons = await get_lessons_by_textbook(db, textbook_id)
    sentences: list[dict] = []
    if with_sentences:
        sentences = await get_sentences_by_lesson_ids(
            db, [le.get("lesson_id") for le in lessons]
        )
    return chapters, lessons, sentences


# ==================== 教材管理 ====================


@router.get("/textbook")
async def get_textbook_all(subject_type: str = None):
    """查询教材列表 — textbook_v2 集合
    
    Args:
        subject_type: 学科过滤（english / math / chinese），缺省返回全部
    """
    try:
        db = get_db()
        where = {}
        if subject_type:
            where["subject_type"] = subject_type
        result = await db.query(collection=TEXTBOOK_V2, where=where)
        logger.info(f"[查询] 查询 textbook_v2 集合，subject_type={subject_type}，结果={result}")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.post("/textbook")
async def add_textbook(data: dict):
    """添加教材 — 请求体 {"title": "新概念2"}（写入 textbook_v2）"""
    try:
        db = get_db()
        textbook_id = str(data.get("textbook_id") or data.get("_id") or "").strip() \
            or f"tb_{uuid.uuid4().hex[:8]}"
        doc = build_textbook_v2_doc(
            textbook_id=textbook_id,
            title=str(data.get("title") or "").strip(),
            grade=str(data.get("grade") or "").strip(),
            level=str(data.get("level") or "").strip(),
        )
        result = await db.insert(collection=TEXTBOOK_V2, data=doc)
        logger.info(
            f"[插入] textbook_v2 添加成功: {json_lib.dumps(doc, ensure_ascii=False)}"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")


# ==================== 学者 × 教材 关联（Phase 5） ====================


async def _fetch_textbook_titles(db, textbook_ids: list[str]) -> dict[str, str]:
    """批量查书名：textbook_v2 一次 $in 取回。

    替代逐本 _fetch_textbook_title 的 N+1 查询。返回 {textbook_id: title}。
    """
    ids = [t for t in textbook_ids if t]
    titles: dict[str, str] = {}
    if not ids:
        return titles
    recs = await query_all_pages(
        db,
        collection=TEXTBOOK_V2,
        where={"_id": {"$in": ids}},
        select={"_id": 1, "title": 1},
    )
    for r in recs:
        if r.get("title"):
            titles[r.get("_id")] = r.get("title")
    return titles


@router.get("/scholar/{scholar_id}/books")
async def get_scholar_books(scholar_id: str, skill_code: str | None = None):
    """我的教材列表 — 该学者全部 scholar_book 关联（含教材级进度）。

    内容层级按书批量 $in 加载一次，学习数据（skill_state / study_attempt）全量
    仅查询一次后按各教材句子集合在内存内过滤聚合，书名一次 $in 批量取回；
    查询次数 = 1(books) + 2(states/attempts) + 3×N(内容) + 2(书名)，与学者级
    数据规模无关，避免每本书重复拉取 states/attempts 导致的 N+1 慢查询。

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
        textbook_ids = [b.get("textbook_id") for b in books if b.get("textbook_id")]

        # 1. 内容层级：每本书批量加载一次（chapters + lessons + sentences），
        #    并记录各教材的句子集合用于内存过滤学习数据
        content_by_book: dict[str, tuple[list[dict], list[dict], list[dict]]] = {}
        sentence_ids_by_book: dict[str, set[str]] = {}
        for tid in textbook_ids:
            chapters, lessons, sentences = await _load_book_content(db, tid)
            content_by_book[tid] = (chapters, lessons, sentences)
            sentence_ids_by_book[tid] = {
                s.get("sentence_id") for s in sentences if s.get("sentence_id")
            }

        # 2. 书名批量 $in（textbook_v2 一次取回）
        titles = await _fetch_textbook_titles(db, textbook_ids)

        # 3. 学习数据仅拉一次，按各教材句子集合在内存内过滤聚合，
        #    不再每本书重复查询学者级 states/attempts
        states = await query_all_pages(
            db,
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
            select={
                "scholar_id": 1,
                "sentence_id": 1,
                "skill_code": 1,
                "status": 1,
                "mastery_score": 1,
                "attempt_count": 1,
            },
        )
        attempts = await query_all_pages(
            db,
            collection=STUDY_ATTEMPT,
            where={"scholar_id": scholar_id},
            select={"sentence_id": 1, "skill_code": 1, "time_spent": 1},
        )

        enriched = []
        for book in books:
            textbook_id = book.get("textbook_id")
            if not textbook_id:
                continue
            sids = sentence_ids_by_book.get(textbook_id, set())
            # 全量书内学习数据（供分能力聚合，不受 skill_code 查询参数影响，与接口 2/3 一致）
            all_book_states = [
                st for st in states if st.get("sentence_id") in sids
            ]
            book_states = [
                st for st in all_book_states
                if skill_code is None or st.get("skill_code") == skill_code
            ]
            book_attempts = [
                a for a in attempts
                if a.get("sentence_id") in sids
                and (skill_code is None or a.get("skill_code") == skill_code)
            ]
            chapters, lessons, sentences = content_by_book[textbook_id]
            stats = aggregate_progress(
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                states=book_states,
                sentences=sentences,
                lessons=lessons,
                chapters=chapters,
                skill_code=skill_code,
                attempts=book_attempts,
                detail="summary",
            )
            summary = dict(stats.get("summary", {}))
            # 综合掌握度：与接口 2/3 同口径（mastery_ratio 4 级档位加权 ÷3）
            summary["mastery"] = mastery_ratio(
                summary.get("mastery_distribution", {}),
                summary.get("total_sentence_count", 0),
            )
            # 分能力掌握度：内存内过滤（不触库），仅该能力有记录时输出该键
            skills: dict[str, float] = {}
            for code in _SKILL_CODES:
                code_states = [
                    st for st in all_book_states
                    if st.get("skill_code") == code
                ]
                if code_states:
                    skills[code] = mastery_ratio(
                        mastery_distribution(code_states),
                        summary.get("total_sentence_count", 0),
                    )
            summary["skills"] = skills
            enriched.append(
                {
                    "textbook_id": textbook_id,
                    "title": titles.get(textbook_id),
                    "current_chapter_id": book.get("current_chapter_id"),
                    "current_lesson_id": book.get("current_lesson_id"),
                    "last_studied_at": book.get("last_studied_at"),
                    "total_time_spent": book.get("total_time_spent"),
                    "status": book.get("status"),
                    "summary": summary,
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


# ==================== 查询接口拆分（原 POST /tracking/stats 已移除，Phase 6 按页拆分） ====================


# 能力全集：含对话能力（前端 SKILL_ORDER 四能力 = translation/conversation/listening/speaking，
# 另有内部 reading）。接口 2 每课 skills 与接口 3 summary.skills 均按此聚合，保证概览与
# 句子级 skills（全量 skill_state）口径一致，避免「列表项有对话、概览缺对话」。
_SKILL_CODES = ("translation", "conversation", "listening", "reading", "speaking")


def _to_iso(timestamp) -> str | None:
    """int 秒级时间戳 → ISO 8601 UTC 字符串；空 / 非法返回 None。"""
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat()
    except Exception:
        return None


async def _find_lesson_by_id(db, textbook_id: str, lesson_id: str) -> dict | None:
    """按教材 + lesson_id 找内容层级 lesson（批量加载，有章经 chapters → lessons；无章直接按教材）。"""
    _, lessons, _ = await _load_book_content(db, textbook_id, with_sentences=False)
    for le in lessons:
        if le.get("lesson_id") == lesson_id:
            return le
    return None


@router.get("/scholar/{scholar_id}/textbooks/{textbook_id}/lessons")
async def get_textbook_lessons(scholar_id: str, textbook_id: str, skill_code: str | None = None):
    """教材详情（lesson 列表 + 顶部三概念）— 查询接口拆分后（接口 2）。

    内容层级与学习数据仅查询一次（批量 $in + 分页），按能力在内存内独立
    聚合构造每课 skills（口径与 summary.mastery 一致），避免 N+1 逐章/逐课
    查询与 5 次重复聚合触库导致的慢查询。

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
        # 一次拉取内容层级 + 全量学习数据，内存内按能力过滤聚合，
        # 将原 5 次重复聚合查询（含逐章/逐课 N+1）压缩为固定 5 次查询，
        # 查询次数与教材规模无关。
        chapters, lessons, sentences = await _load_book_content(db, textbook_id)
        states = await query_all_pages(
            db,
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
            select={
                "scholar_id": 1,
                "sentence_id": 1,
                "skill_code": 1,
                "status": 1,
                "mastery_score": 1,
                "attempt_count": 1,
            },
        )
        attempts = await query_all_pages(
            db,
            collection=STUDY_ATTEMPT,
            where={"scholar_id": scholar_id},
            select={"sentence_id": 1, "skill_code": 1, "time_spent": 1},
        )

        def _aggregate(
            states_subset: list[dict],
            code: str | None,
            attempts_subset: list[dict],
        ) -> dict:
            return aggregate_progress(
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                states=states_subset,
                sentences=sentences,
                lessons=lessons,
                chapters=chapters,
                skill_code=code,
                attempts=attempts_subset,
                detail="lesson",
            )

        # base：按请求 skill_code 过滤（None 时使用全部学习数据）
        base_states = [
            st for st in states
            if skill_code is None or st.get("skill_code") == skill_code
        ]
        base_attempts = [
            a for a in attempts
            if skill_code is None or a.get("skill_code") == skill_code
        ]
        base = _aggregate(base_states, skill_code, base_attempts)
        summary_raw = base.get("summary", {})

        # 各能力独立聚合：仅内存内按 skill_code 过滤，不再重复触库
        states_by_skill: dict[str, list[dict]] = {c: [] for c in _SKILL_CODES}
        for st in states:
            c = st.get("skill_code")
            if c in states_by_skill:
                states_by_skill[c].append(st)
        attempts_by_skill: dict[str, list[dict]] = {c: [] for c in _SKILL_CODES}
        for a in attempts:
            c = a.get("skill_code")
            if c in attempts_by_skill:
                attempts_by_skill[c].append(a)

        skill_views: dict[str, dict[str, dict]] = {}
        for code in _SKILL_CODES:
            view = _aggregate(
                states_by_skill[code], code, attempts_by_skill[code]
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

        total_attempt_count = summary_raw.get("total_attempt_count", 0)
        learned_sentence_count = summary_raw.get("learned_sentence_count", 0)
        data = {
            "summary": {
                "textbook_progress": summary_raw.get("textbook_progress", 0.0),
                "learned_sentence_count": learned_sentence_count,
                "total_sentence_count": summary_raw.get("total_sentence_count", 0),
                "total_attempt_count": total_attempt_count,
                "avg_attempt_count": (
                    round(total_attempt_count / learned_sentence_count, 2)
                    if learned_sentence_count
                    else 0.0
                ),
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
        sentence_ids = [s.get("sentence_id") for s in sentences if s.get("sentence_id")]

        # 3. 该课句子的 skill_state：按 sentence_id $in 分批查询（每批 200），
        #    仅拉取必要数据，替代原「该学者全部状态」全量分页拉取，
        #    查询次数与本课句子数挂钩、与学者总学习量解耦
        states: list[dict] = []
        if sentence_ids:
            for i in range(0, len(sentence_ids), 200):
                states.extend(await query_all_pages(
                    db,
                    collection=SKILL_STATE,
                    where={
                        "scholar_id": scholar_id,
                        "sentence_id": {"$in": sentence_ids[i:i + 200]},
                    },
                    select={
                        "scholar_id": 1,
                        "sentence_id": 1,
                        "skill_code": 1,
                        "status": 1,
                        "mastery_score": 1,
                        "attempt_count": 1,
                        "next_review_at": 1,
                    },
                ))
        states_by_sentence: dict[str, list[dict]] = {}
        for st in states:
            states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

        # 4. 句子列表 + 已学计数（乐观聚合：无指定能力取 progress 最高者）
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


# ==================== 学习调度与统计（f4 复习队列 / f5 补漏清单 / f6 日历） ====================


@router.post("/tracking/review-plan")
async def get_review_plan(data: dict):
    """今日复习队列（f4）— 契约 §3.6。

    入参：`{ scholar_id(必), date?: "YYYY-MM-DD"（默认今日） }`
    出参：`{ success, data: { date, total, review_queue: [...] } }`

    口径：
    - 一次 skill_state 查询（按学者拉取必要字段）+ 内容 $in 批量加载 + 内存过滤排序；
    - 到期判定：`pick_state`（乐观）后 `next_review_at` 存在且 ≤ 当日 23:59:59，
      且 `status ≠ mastered`（与 ADR-0004 间隔算法一致）；
    - 排序：`next_review_at` 升序，同到期日按 `mastery_score` 升序（薄弱优先）；
    - `skills`/`weakest_skill`/`status`/`review_count`/`next_review_at` 与
      §3.1 sentences 接口派生字段一致。
    - 无到期记录 → `success:true, total:0` 空队列，不报错。
    """
    scholar_id = data.get("scholar_id")
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    date_str = data.get("date")
    if date_str is None:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"date 格式非法: {date_str}（应为 YYYY-MM-DD）",
            )
    end_of_day = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc,
    )
    end_ts = int(end_of_day.timestamp())
    try:
        db = get_db()
        # 1. 一次拉取该学者全部 skill_state（必要字段），内存内按句子分组
        states = await query_all_pages(
            db,
            collection=SKILL_STATE,
            where={"scholar_id": scholar_id},
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
        states_by_sentence: dict[str, list[dict]] = {}
        for st in states:
            states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

        # 2. 内存过滤：pick_state（乐观）后到期且未掌握
        candidates: list[tuple[str, dict]] = []  # (sentence_id, picked_state)
        for sid, s_states in states_by_sentence.items():
            picked = pick_state(s_states)
            if not picked:
                continue
            if picked.get("status") == STATUS_MASTERED:
                continue
            try:
                review_ts = int(picked.get("next_review_at") or 0)
            except (TypeError, ValueError):
                continue
            if review_ts <= 0 or review_ts > end_ts:
                continue
            candidates.append((sid, picked))
        if not candidates:
            return {
                "success": True,
                "data": {"date": date_str, "total": 0, "review_queue": []},
            }

        # 3. 候选句子内容 $in 批量加载（跨课/跨教材一次取回）
        sentences = await get_sentences_by_ids(db, [sid for sid, _ in candidates])
        content_by_id = {s.get("sentence_id"): s for s in sentences}

        # 4. 组装（与 sentences 接口派生字段同构）+ 排序
        queue = []
        for sid, picked in candidates:
            s = content_by_id.get(sid) or {}
            s_states = states_by_sentence.get(sid, [])
            skills = {
                st.get("skill_code"): status_to_int(st.get("status"))
                for st in s_states
                if st.get("skill_code")
            }
            queue.append({
                "sentence_id": sid,
                "content": s.get("text", ""),
                "translation": s.get("translation", ""),
                "lesson_id": s.get("lesson_id", ""),
                "skills": skills,
                "weakest_skill": min(skills, key=skills.get) if skills else None,
                "status": status_to_int(picked.get("status")),
                "review_count": int(picked.get("attempt_count") or 0),
                "next_review_at": _to_iso(picked.get("next_review_at")),
                "_sort_ts": int(picked.get("next_review_at") or 0),
                "_sort_mastery": int(picked.get("mastery_score") or 0),
            })
        queue.sort(key=lambda it: (it.pop("_sort_ts"), it.pop("_sort_mastery")))

        data = {"date": date_str, "total": len(queue), "review_queue": queue}
        logger.info(
            f"[tracking/review-plan] scholar_id={scholar_id}, "
            f"date={date_str}, total={len(queue)}"
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/review-plan] 复习队列失败: {e}")
        raise HTTPException(status_code=500, detail=f"复习队列失败: {str(e)}")


@router.post("/tracking/weakness-plan")
async def get_weakness_plan(data: dict):
    """薄弱点补漏清单（f5）— 契约 §3.6。

    入参：`{ scholar_id(必), textbook_id?, lesson_id? }`
    出参：`{ success, data: { total, weakness_queue: [...] } }`

    口径：
    - 候选 = `pick_state`（乐观）后 `status = not_started`（未学）或
      `mastery_score < 60`（低分）的句子（仅含有 skill_state 记录的句子，
      与掌握度聚合口径一致）；
    - 范围：不传 → 学者全量一次状态查询；传 `textbook_id` → 先加载该教材
      句子再查状态；传 `lesson_id` → 先加载该课句子再查状态（此时必须同传
      `textbook_id`）；
    - 排序：`weakest_skill` 升序，同最弱能力按 `chapter_id` / `order` 章节顺序；
    - 字段与 review-plan 同构（sentence_id/content/translation/lesson_id/
      chapter_id/skills/weakest_skill/mastery_score/status/review_count）；
    - 无候选 → `success:true, total:0` 空队列，不报错。
    """
    scholar_id = data.get("scholar_id")
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    textbook_id = data.get("textbook_id")
    lesson_id = data.get("lesson_id")
    if lesson_id and not textbook_id:
        raise HTTPException(
            status_code=400,
            detail="指定 lesson_id 时必须同时指定 textbook_id",
        )
    try:
        db = get_db()
        # 1. 内容范围（可选）：lesson_id → 单课；textbook_id → 整本；均无 → 学者级
        scope_sentences: list[dict] | None = None
        if lesson_id:
            scope_sentences = await get_sentences_by_lesson(db, lesson_id)
        elif textbook_id:
            _, _, scope_sentences = await _load_book_content(db, textbook_id)
        scope_ids = (
            {s.get("sentence_id") for s in scope_sentences}
            if scope_sentences is not None else None
        )

        # 2. skill_state 查询：限定范围按范围句子 $in 分批，学者级一次全量
        select_fields = {
            "scholar_id": 1,
            "sentence_id": 1,
            "skill_code": 1,
            "status": 1,
            "mastery_score": 1,
            "attempt_count": 1,
        }
        if scope_ids is not None:
            states: list[dict] = []
            ids = list(scope_ids)
            for i in range(0, len(ids), 200):
                states.extend(await query_all_pages(
                    db,
                    collection=SKILL_STATE,
                    where={
                        "scholar_id": scholar_id,
                        "sentence_id": {"$in": ids[i:i + 200]},
                    },
                    select=select_fields,
                ))
        else:
            states = await query_all_pages(
                db,
                collection=SKILL_STATE,
                where={"scholar_id": scholar_id},
                select=select_fields,
            )
        states_by_sentence: dict[str, list[dict]] = {}
        for st in states:
            states_by_sentence.setdefault(st.get("sentence_id"), []).append(st)

        # 3. 内存过滤候选：pick_state（乐观）后未学或低分
        candidates: list[tuple[str, dict]] = []
        for sid, s_states in states_by_sentence.items():
            picked = pick_state(s_states)
            if not picked:
                continue
            try:
                score = int(picked.get("mastery_score") or 0)
            except (TypeError, ValueError):
                score = 0
            if picked.get("status") == STATUS_NOT_STARTED or score < 60:
                candidates.append((sid, picked))
        if not candidates:
            return {
                "success": True,
                "data": {"total": 0, "weakness_queue": []},
            }

        # 4. 内容：范围已加载则复用，否则候选内容 $in 批量加载
        if scope_sentences is not None:
            content_by_id = {s.get("sentence_id"): s for s in scope_sentences}
        else:
            sentences = await get_sentences_by_ids(db, [sid for sid, _ in candidates])
            content_by_id = {s.get("sentence_id"): s for s in sentences}

        # 5. 组装（与 review-plan 字段同构）+ 排序（最弱能力升序 / 章节顺序）
        queue = []
        for sid, picked in candidates:
            s = content_by_id.get(sid) or {}
            s_states = states_by_sentence.get(sid, [])
            skills = {
                st.get("skill_code"): status_to_int(st.get("status"))
                for st in s_states
                if st.get("skill_code")
            }
            weakest = min(skills, key=skills.get) if skills else None
            queue.append({
                "sentence_id": sid,
                "content": s.get("text", ""),
                "translation": s.get("translation", ""),
                "lesson_id": s.get("lesson_id", ""),
                "chapter_id": s.get("chapter_id", ""),
                "skills": skills,
                "weakest_skill": weakest,
                "mastery_score": int(picked.get("mastery_score") or 0),
                "status": status_to_int(picked.get("status")),
                "review_count": int(picked.get("attempt_count") or 0),
                "_sort_weak": weakest or "\uffff",
                "_sort_chapter": s.get("chapter_id") or "",
                "_sort_order": int(s.get("order") or 0),
            })
        queue.sort(key=lambda it: (
            it.pop("_sort_weak"), it.pop("_sort_chapter"), it.pop("_sort_order"),
        ))

        data = {"total": len(queue), "weakness_queue": queue}
        logger.info(
            f"[tracking/weakness-plan] scholar_id={scholar_id}, "
            f"textbook_id={textbook_id}, lesson_id={lesson_id}, total={len(queue)}"
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/weakness-plan] 补漏清单失败: {e}")
        raise HTTPException(status_code=500, detail=f"补漏清单失败: {str(e)}")


@router.get("/tracking/{scholar_id}/calendar")
async def get_study_calendar(scholar_id: str, days: int = Query(30)):
    """学习日历热力图 / 连续打卡（f6）— 契约 §3.6。

    入参：`{ scholar_id(路径), days?: int（默认 30，上限 90） }`
    出参：`{ success, data: { streak_days, heatmap: [{ date, attempt_count }] } }`

    口径：
    - 按 `study_attempt.created_at`（秒级时间戳）按天聚合（UTC 日期）；
    - 窗口：从今天起往前 `days` 天（含今天）；
    - `heatmap` 仅含有 attempt 记录的日期（升序），`attempt_count` 为当日事件数；
    - `streak_days`：从今天起往前连续 ≥1 条 attempt 的天数；今天无记录 → 0（不回溯）；
    - 无记录 → `success:true, heatmap:[], streak_days:0`，不报错。
    """
    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail=f"days 应在 1-90 之间: {days}")
    try:
        db = get_db()
        today = datetime.now(timezone.utc).date()
        start_date = today - timedelta(days=days - 1)
        start_ts = int(datetime.combine(start_date, time.min, tzinfo=timezone.utc).timestamp())
        end_ts = int(datetime.combine(today, time.max, tzinfo=timezone.utc).timestamp())

        # 1. 窗口内 study_attempt 时间戳（只取 created_at，量小）
        attempts = await query_all_pages(
            db,
            collection=STUDY_ATTEMPT,
            where={
                "scholar_id": scholar_id,
                "created_at": {"$gte": start_ts, "$lte": end_ts},
            },
            select={"created_at": 1},
        )

        # 2. 按天聚合（UTC 日期）
        counts: dict[str, int] = {}
        for att in attempts:
            try:
                ts = int(att.get("created_at") or 0)
            except (TypeError, ValueError):
                continue
            if ts <= 0:
                continue
            day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            counts[day] = counts.get(day, 0) + 1
        heatmap = [
            {"date": day, "attempt_count": counts[day]}
            for day in sorted(counts)
        ]

        # 3. 连续打卡：从今天起往前连续 ≥1 条 attempt；今天无记录 → 0（不回溯）
        today_str = today.strftime("%Y-%m-%d")
        streak_days = 0
        if counts.get(today_str, 0) > 0:
            cur = today
            while counts.get(cur.strftime("%Y-%m-%d"), 0) > 0:
                streak_days += 1
                cur -= timedelta(days=1)

        logger.info(
            f"[tracking/{scholar_id}/calendar] days={days}, "
            f"attempt_days={len(heatmap)}, streak_days={streak_days}"
        )
        return {"success": True, "data": {"streak_days": streak_days, "heatmap": heatmap}}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/{scholar_id}/calendar] 日历失败: {e}")
        raise HTTPException(status_code=500, detail=f"日历失败: {str(e)}")


@router.post("/tracking/daily-goal")
async def get_daily_goal(data: dict):
    """每日学习目标（P2/F6）— 契约 §3.7。

    入参：`{ scholar_id(必), date?: "YYYY-MM-DD"（默认今日） }`
    出参：`{ success, data: { date, goal, progress, completed, percent } }`

    口径：
    - 目标生成（规则引擎 `build_daily_goal`）：
      `goal = clamp(round(avg7 × 1.2), floor, cap)`；近 7 天（不含今日）无记录 → floor；
    - 进度：`new_sentences` = 今日 learned/mastered 去重句数、`minutes` = 今日
      time_spent 求和（秒÷60）、`attempts` = 今日事件数；
    - `percent` = 三指标达标率均值；`completed = percent ≥ 100`；
    - 无记录 → `success:true` 且 goal 回落 floor、percent=0，不报错。
    """
    scholar_id = data.get("scholar_id")
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    date_str = data.get("date")
    if date_str is not None:
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"date 格式非法: {date_str}（应为 YYYY-MM-DD）",
            )
    try:
        db = get_db()
        goal = await build_daily_goal(db, scholar_id=scholar_id, date_str=date_str)
        return {"success": True, "data": goal}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/daily-goal] 每日目标失败: {e}")
        raise HTTPException(status_code=500, detail=f"每日目标失败: {str(e)}")


@router.get("/tracking/{scholar_id}/badges")
async def get_scholar_badges(scholar_id: str):
    """徽章墙（P2/F8）— 契约 §3.7。

    出参：`{ success, data: { earned: [...], locked: [...] } }`

    口径：
    - 服务层 `evaluate_badges` 按 `badge.condition_type` 聚合 current；
      `current ≥ target` 且未发放 → 幂等插入 `scholar_badge`，再读回已获得列表；
    - `enabled=false` 的徽章不返回；无徽章定义 → `success:true, earned:[], locked:[]`。
    """
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    try:
        db = get_db()
        data = await evaluate_badges(db, scholar_id=scholar_id)
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/{scholar_id}/badges] 徽章失败: {e}")
        raise HTTPException(status_code=500, detail=f"徽章失败: {str(e)}")


@router.get("/tracking/{scholar_id}/wrong-book")
async def get_wrong_book(
    scholar_id: str,
    limit: int = Query(50),
    error_type: str | None = Query(None),
):
    """错题本（P2/F11）— 契约 §3.7。

    入参：`{ scholar_id(路径), limit?: int（默认 50，上限 200）, error_type?: string }`
    出参：`{ success, data: { total, items: [...] } }`

    口径：
    - 服务层 `aggregate_wrong_book`：一次学者级 `study_attempt`（status=incorrect）
      查询 → 按句子内存分组聚合（error_count / last_error_at / error_types 分布）→
      内容 `$in` 批量加载 sentence_v2 → 按 last_error_at 降序；
    - `error_type` 入参过滤：仅统计该类型错误（error_count 为该类型计数），
      `error_types` 仍为全量分布；
    - 无错题 → `success:true, total:0, items:[]`，不报错。
    """
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail=f"limit 应在 1-200 之间: {limit}")
    try:
        db = get_db()
        data = await aggregate_wrong_book(
            db,
            scholar_id=scholar_id,
            limit=limit,
            error_type=error_type,
        )
        return {"success": True, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/{scholar_id}/wrong-book] 错题本失败: {e}")
        raise HTTPException(status_code=500, detail=f"错题本失败: {str(e)}")
