"""学习状态上报接口 — POST /tracking/state + 学习会话接口（Phase 2 + Phase 3 + Phase 5）

Phase 2：每次调用按 `{scholar_id}_{sentence_id}_{skill_code}` 复合键 upsert 一条
skill_state，重复上报只累加 attempt_count、刷新 last_studied_at。
Phase 3：上报同时追加一条 study_attempt 事件（append-only）；并提供会话
start/end 接口（study_session），会话结算时回填 duration_sec 与 attempt_count。
Phase 5：会话结算时回写 scholar_book（刷新 last_studied_at、增量累加
total_time_spent），供"我的教材列表/断点续学"使用。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from services.dependencies import get_db
from services.english import SentenceNotFoundError
from services.english.sentence_management import ensureSentenceSemanticKey
from services.events import end_session, record_attempt, start_session
from services.models_learning import DEFAULT_SKILL_CODE, upsert_skill_state
from services.models_scholar_book import touch_scholar_book, upsert_scholar_book

logger = logging.getLogger("scholar-admin.routes.state")
router = APIRouter(tags=["学习状态"])


@router.post("/tracking/state")
async def report_tracking_state(data: dict):
    """上报单句单能力学习状态，返回最新状态与本次写入的事件。

    请求体：
    {
      "scholar_id": "scholar_xxx",   // 必填
      "sentence_id": "sent_xxx",     // 必填
      "skill_code": "translation",   // 可选，默认 translation
      "lesson_id": "unit_xxx",       // 可选，便于按课聚合
      "status": "learned",           // 可选，支持中英文（如 "已学"）
      "score": 90,                   // 可选，0-100
      "mastery": 0.9,                // 可选，0-1
      "time_spent": 120,             // 可选，学习时长（秒），归入事件聚合
      "attempt_type": "translate",   // 可选，事件类型（read/translate/listen/speak/quiz）
      "attempt_status": "completed", // 可选，事件结果（correct/incorrect/completed/abandoned）
      "session_id": "ses_xxx",       // 可选，所属会话（由 POST /tracking/session/start 创建）
      "error_type": "grammar"        // 可选（P2/F11）：仅 attempt_status=incorrect 时生效
                                     //   vocabulary/grammar/pronunciation/comprehension/other
    }

    返回：
    {
      "success": true,
      "data": {
        "state":   { ...最新 skill_state 文档... },
        "attempt": { ...本次写入的 study_attempt 事件... }
      }
    }
    """
    scholar_id = str(data.get("scholar_id") or "").strip()
    sentence_id = str(data.get("sentence_id") or "").strip()
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholar_id")
    if not sentence_id:
        raise HTTPException(status_code=400, detail="缺少参数 sentence_id")

    skill_code = str(data.get("skill_code") or DEFAULT_SKILL_CODE).strip()
    time_spent = data.get("time_spent")

    try:
        db = get_db()
        # M3 G1.2 + M5（service-contract §8.5 + data-model §4.15）：Lazy dedup —
        # 惰性补齐语义键并落 sentence_semantic_key（registry 成为 canonical/duplicate
        # 权威源，sentence_v2 字段保持同步）；skill_state 写入键零变化。
        try:
            semantic = await ensureSentenceSemanticKey(db, sentence_id=sentence_id)
            logger.info(
                f"[tracking/state] 语义键补齐 sentence_id={sentence_id}, "
                f"canonical={semantic.get('canonical_sentence_id')}"
            )
        except SentenceNotFoundError as exc:
            # 句子不在内容库（契约 §3.2 错误仅 400/500）：跳过补齐，不影响状态写入
            logger.warning(f"[tracking/state] 跳过语义键补齐（句子不在内容库）: {exc}")
        state = await upsert_skill_state(
            db,
            scholar_id=scholar_id,
            sentence_id=sentence_id,
            skill_code=skill_code,
            lesson_id=data.get("lesson_id"),
            status=data.get("status"),
            score=data.get("score"),
            mastery=data.get("mastery"),
        )
        attempt = await record_attempt(
            db,
            scholar_id=scholar_id,
            sentence_id=sentence_id,
            skill_code=skill_code,
            attempt_type=data.get("attempt_type"),
            status=data.get("attempt_status") or data.get("status"),
            score=data.get("score"),
            mastery=data.get("mastery"),
            time_spent=time_spent,
            lesson_id=data.get("lesson_id"),
            session_id=data.get("session_id"),
            error_type=data.get("error_type"),
        )
        logger.info(
            f"[tracking/state] scholar_id={scholar_id}, sentence_id={sentence_id}, "
            f"skill_code={skill_code}, status={state.get('status')}, "
            f"attempt_count={state.get('attempt_count')}, time_spent={time_spent}, "
            f"attempt_id={attempt.get('attempt_id')}, "
            f"error_type={attempt.get('error_type')}"
        )
        return {"success": True, "data": {"state": state, "attempt": attempt}}
    except Exception as e:
        logger.error(f"[tracking/state] 状态上报失败: {e}")
        raise HTTPException(status_code=500, detail=f"状态上报失败: {str(e)}")


@router.post("/tracking/session/start")
async def start_tracking_session(data: dict):
    """创建一个学习会话（study_session，status=active），返回会话文档。

    请求体：
    {
      "scholar_id": "scholar_xxx",   // 必填
      "textbook_id": "tb_xxx",       // 可选，当前学习教材
      "subject_type": "math",        // 可选，学科标识（english/math/chinese，缺省 english）
      "device": "ios",               // 可选，设备标识
      "source": "app"                // 可选，来源（app/web/...）
    }

    返回：
    {
      "success": true,
      "data": { ...新创建的 study_session 文档... }
    }
    """
    scholar_id = str(data.get("scholar_id") or "").strip()
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholar_id")
    textbook_id = data.get("textbook_id")
    subject_type = data.get("subject_type")
    logger.info(
        f"[tracking/session/start] 收到请求 scholar_id={scholar_id} "
        f"textbook_id={textbook_id!r} subject_type={subject_type!r}"
    )
    try:
        db = get_db()
        session = await start_session(
            db,
            scholar_id=scholar_id,
            textbook_id=textbook_id,
            device=data.get("device"),
            source=data.get("source"),
            subject_type=subject_type,
        )
        # 同时创建/更新 scholar_book，使教材立即出现在学者 books 列表中（「学习中」状态）
        if textbook_id:
            book = await upsert_scholar_book(
                db,
                scholar_id=scholar_id,
                textbook_id=textbook_id,
                subject_type=subject_type,
            )
            logger.info(
                f"[tracking/session/start] scholar_book 已创建/更新: "
                f"_id={book.get('_id') if book else 'None'} "
                f"subject_type={book.get('subject_type') if book else 'None'}"
            )
        else:
            logger.warning(
                f"[tracking/session/start] textbook_id 为空，跳过 scholar_book 创建"
            )
        logger.info(
            f"[tracking/session/start] scholar_id={scholar_id}, "
            f"session_id={session.get('session_id')}, "
            f"subject_type={subject_type}"
        )
        return {"success": True, "data": session}
    except Exception as e:
        logger.error(f"[tracking/session/start] 会话创建失败: {e}")
        raise HTTPException(status_code=500, detail=f"会话创建失败: {str(e)}")


@router.post("/tracking/session/end")
async def end_tracking_session(data: dict):
    """结算一个学习会话：回填 ended_at / duration_sec / attempt_count。

    请求体：
    {
      "session_id": "ses_xxx"        // 必填，会话 ID
    }

    返回：
    {
      "success": true,
      "data": { ...结算后的 study_session 文档... }
    }
    """
    session_id = str(data.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少参数 session_id")
    try:
        db = get_db()
        session = await end_session(db, session_id=session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")

        # Phase 5：会话结算回写 scholar_book（刷新 last_studied_at、增量累加 total_time_spent）
        # subject_type 从 session 透传，保证 scholar_book 学科标识一致
        book = await touch_scholar_book(
            db,
            scholar_id=session.get("scholar_id"),
            textbook_id=session.get("textbook_id"),
            last_studied_at=session.get("ended_at"),
            time_delta_sec=session.get("duration_sec") or 0,
            subject_type=session.get("subject_type"),
        )
        logger.info(
            f"[tracking/session/end] session_id={session_id}, "
            f"duration_sec={session.get('duration_sec')}, "
            f"attempt_count={session.get('attempt_count')}, "
            f"status={session.get('status')}, "
            f"book_textbook={session.get('textbook_id')}, "
            f"book_total_time={(book or {}).get('total_time_spent')}"
        )
        return {"success": True, "data": session}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[tracking/session/end] 会话结算失败: {e}")
        raise HTTPException(status_code=500, detail=f"会话结算失败: {str(e)}")
