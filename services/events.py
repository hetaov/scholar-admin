"""学习事件模型 — study_attempt + study_session 写入辅助（Phase 3 事件模型）

集合：
- `study_attempt`：append-only 学习事件日志，每次学习行为写一条，只插入不修改
- `study_session`：学习会话，start 创建 / end 结算，duration_sec 与 attempt_count 在结算时回填

字段映射（旧学习追踪模型 → study_attempt）：
- `time_spent` → `time_spent`（归入事件聚合与 scholar_book.total_time_spent）
- `status/score/mastery` → 事件级状态与得分
- `study_count` → 多次上报产生多条 attempt（聚合时再统计次数）

原则：
- study_attempt 只增不改（append-only，无 update 调用路径）。
- study_session 结算时 `duration_sec = ended_at - started_at`，
  `attempt_count` 与该会话内 study_attempt 事件数一致。
- attempt_type 由前端传，未传时按 skill_code 推断（translation→translate 等）。
"""

from __future__ import annotations

import time
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

STUDY_ATTEMPT = "study_attempt"
STUDY_SESSION = "study_session"

# ---------------------------------------------------------------------------
# 事件枚举
# ---------------------------------------------------------------------------

# attempt_type：本次学习行为的类型
ATTEMPT_TYPE_READ = "read"
ATTEMPT_TYPE_TRANSLATE = "translate"
ATTEMPT_TYPE_LISTEN = "listen"
ATTEMPT_TYPE_SPEAK = "speak"
ATTEMPT_TYPE_QUIZ = "quiz"
VALID_ATTEMPT_TYPES = {
    ATTEMPT_TYPE_READ,
    ATTEMPT_TYPE_TRANSLATE,
    ATTEMPT_TYPE_LISTEN,
    ATTEMPT_TYPE_SPEAK,
    ATTEMPT_TYPE_QUIZ,
}

# attempt 状态：本次尝试的结果
ATTEMPT_STATUS_CORRECT = "correct"
ATTEMPT_STATUS_INCORRECT = "incorrect"
ATTEMPT_STATUS_COMPLETED = "completed"
ATTEMPT_STATUS_ABANDONED = "abandoned"
VALID_ATTEMPT_STATUSES = {
    ATTEMPT_STATUS_CORRECT,
    ATTEMPT_STATUS_INCORRECT,
    ATTEMPT_STATUS_COMPLETED,
    ATTEMPT_STATUS_ABANDONED,
}

# 会话状态
SESSION_STATUS_ACTIVE = "active"
SESSION_STATUS_ENDED = "ended"
VALID_SESSION_STATUSES = {SESSION_STATUS_ACTIVE, SESSION_STATUS_ENDED}

# skill_code → 默认 attempt_type
_SKILL_TO_ATTEMPT_TYPE: dict[str, str] = {
    "translation": ATTEMPT_TYPE_TRANSLATE,
    "listening": ATTEMPT_TYPE_LISTEN,
    "speaking": ATTEMPT_TYPE_SPEAK,
    "reading": ATTEMPT_TYPE_READ,
}


# ---------------------------------------------------------------------------
# 归一化 / 推断（纯函数）
# ---------------------------------------------------------------------------


def infer_attempt_type(skill_code: Any) -> str:
    """由 skill_code 推断默认 attempt_type；未知能力回落 quiz。"""
    if not skill_code:
        return ATTEMPT_TYPE_QUIZ
    return _SKILL_TO_ATTEMPT_TYPE.get(str(skill_code).strip(), ATTEMPT_TYPE_QUIZ)


def normalize_attempt_type(attempt_type: Any) -> str:
    """把前端传入的 attempt_type 收敛为枚举值；无效回落 quiz。"""
    if not attempt_type:
        return ATTEMPT_TYPE_QUIZ
    t = str(attempt_type).strip().lower()
    if t in VALID_ATTEMPT_TYPES:
        return t
    return ATTEMPT_TYPE_QUIZ


def normalize_attempt_status(status: Any) -> str:
    """把前端传入的 attempt 状态收敛为枚举值；无效回落 completed。"""
    if not status:
        return ATTEMPT_STATUS_COMPLETED
    s = str(status).strip().lower()
    if s in VALID_ATTEMPT_STATUSES:
        return s
    return ATTEMPT_STATUS_COMPLETED


# ---------------------------------------------------------------------------
# 主键生成
# ---------------------------------------------------------------------------


def new_attempt_id(now: int | None = None) -> str:
    """生成 study_attempt 主键：att_{毫秒时间戳}_{随机短id}。"""
    now = int(now or time.time())
    return f"att_{now}_{uuid.uuid4().hex[:8]}"


def new_session_id(now: int | None = None) -> str:
    """生成 study_session 主键：ses_{毫秒时间戳}_{随机短id}。"""
    now = int(now or time.time())
    return f"ses_{now}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 文档构建（纯函数）
# ---------------------------------------------------------------------------


def build_attempt_doc(
    *,
    scholar_id: str,
    sentence_id: str,
    skill_code: str,
    attempt_type: str | None = None,
    status: str | None = None,
    score: float | None = None,
    mastery: float | None = None,
    time_spent: int | None = None,
    lesson_id: str | None = None,
    session_id: str | None = None,
    attempt_id: str | None = None,
    now: int | None = None,
) -> dict:
    """构建 study_attempt 事件文档（纯函数，只生成不落库）。"""
    now = int(now or time.time())
    _id = attempt_id or new_attempt_id(now)
    return {
        "_id": _id,
        "attempt_id": _id,
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "lesson_id": lesson_id,
        "skill_code": skill_code,
        "attempt_type": normalize_attempt_type(attempt_type or infer_attempt_type(skill_code)),
        "status": normalize_attempt_status(status),
        "score": score,
        "mastery": mastery,
        "time_spent": int(time_spent) if time_spent is not None else None,
        "session_id": session_id,
        "created_at": now,
    }


def build_session_doc(
    *,
    scholar_id: str,
    textbook_id: str | None = None,
    device: str | None = None,
    source: str | None = None,
    session_id: str | None = None,
    started_at: int | None = None,
    now: int | None = None,
) -> dict:
    """构建 study_session 文档（start 态：ended_at/duration_sec/attempt_count 为空）。"""
    now = int(now or time.time())
    started_at = int(started_at or now)
    _id = session_id or new_session_id(now)
    return {
        "_id": _id,
        "session_id": _id,
        "scholar_id": scholar_id,
        "textbook_id": textbook_id,
        "device": device,
        "source": source,
        "status": SESSION_STATUS_ACTIVE,
        "started_at": started_at,
        "ended_at": None,
        "duration_sec": 0,
        "attempt_count": 0,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def record_attempt(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    skill_code: str,
    attempt_type: str | None = None,
    status: str | None = None,
    score: float | None = None,
    mastery: float | None = None,
    time_spent: int | None = None,
    lesson_id: str | None = None,
    session_id: str | None = None,
    now: int | None = None,
) -> dict:
    """写入一条 study_attempt 事件（append-only，只插入不修改）。

    返回写入的事件文档。
    """
    doc = build_attempt_doc(
        scholar_id=scholar_id,
        sentence_id=sentence_id,
        skill_code=skill_code,
        attempt_type=attempt_type,
        status=status,
        score=score,
        mastery=mastery,
        time_spent=time_spent,
        lesson_id=lesson_id,
        session_id=session_id,
        now=now,
    )
    await db.insert(collection=STUDY_ATTEMPT, data=doc)
    return doc


async def start_session(
    db,
    *,
    scholar_id: str,
    textbook_id: str | None = None,
    device: str | None = None,
    source: str | None = None,
    now: int | None = None,
) -> dict:
    """创建一个 study_session（status=active），返回会话文档。"""
    doc = build_session_doc(
        scholar_id=scholar_id,
        textbook_id=textbook_id,
        device=device,
        source=source,
        now=now,
    )
    await db.insert(collection=STUDY_SESSION, data=doc)
    return doc


async def end_session(db, *, session_id: str, ended_at: int | None = None) -> dict | None:
    """结算 study_session：回填 ended_at / duration_sec / attempt_count。

    - duration_sec = ended_at - started_at
    - attempt_count 与当前会话内 study_attempt 事件数一致
    返回最新会话文档；会话不存在返回 None。
    """
    now = int(ended_at or time.time())
    result = await db.query(collection=STUDY_SESSION, where={"_id": session_id}, limit=1)
    records = result.get("records", [])
    if not records:
        return None
    session = records[0]

    attempt_count = await db.count(
        collection=STUDY_ATTEMPT,
        where={"session_id": session_id},
    )
    started_at = int(session.get("started_at") or now)
    duration_sec = max(0, now - started_at)

    await db.update(
        collection=STUDY_SESSION,
        where={"_id": session_id},
        data={"$set": {
            "status": SESSION_STATUS_ENDED,
            "ended_at": now,
            "duration_sec": duration_sec,
            "attempt_count": attempt_count,
        }},
        multi=False,
    )
    latest = await db.query(collection=STUDY_SESSION, where={"_id": session_id}, limit=1)
    return latest["records"][0]


async def count_session_attempts(db, *, session_id: str) -> int:
    """统计指定会话内的 study_attempt 事件数。"""
    return await db.count(collection=STUDY_ATTEMPT, where={"session_id": session_id})
