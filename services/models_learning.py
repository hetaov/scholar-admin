"""学习状态模型 — skill 定义 + skill_state 当前状态读写（Phase 2 能力模型）

集合：
- `skill`      ：能力定义（种子数据：translation / listening / speaking / reading）
- `skill_state`：学者 × 句子 × 能力 的当前状态，主键 `{scholar_id}_{sentence_id}_{skill_code}`

字段映射（旧 learning_mastery_tracking → skill_state）：
- `time_spent`  → 归入事件聚合与 scholar_book.total_time_spent（Phase 3/5 落点）
- `status/score/mastery` → `status` / `mastery_score` / `progress`
- `study_count` → `attempt_count`
- `last_study_time` → `last_studied_at`

原则：
- upsert 按复合键幂等：同一学者对同一句子同能力只产生一条，重复上报只累加 attempt_count。
- 新写入统一英文状态枚举；中文状态词（旧数据）经 `normalize_status` 收敛。
- `next_review_at` 由滚动调度公式计算：`last_studied_at + interval(attempt_count, mastery_score)`。
"""

from __future__ import annotations

import time
from typing import Any

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

SKILL = "skill"
SKILL_STATE = "skill_state"

# ---------------------------------------------------------------------------
# 状态枚举（新写入统一英文）
# ---------------------------------------------------------------------------

STATUS_NOT_STARTED = "not_started"
STATUS_LEARNING = "learning"
STATUS_LEARNED = "learned"
STATUS_MASTERED = "mastered"
STATUS_REVIEW_DUE = "review_due"

VALID_STATUSES = {
    STATUS_NOT_STARTED,
    STATUS_LEARNING,
    STATUS_LEARNED,
    STATUS_MASTERED,
    STATUS_REVIEW_DUE,
}

# 中文状态词 → 英文枚举（旧数据兼容）
_STATUS_CN_MAP = {
    "已学": STATUS_LEARNED,
    "已学会": STATUS_LEARNED,
    "已学完": STATUS_LEARNED,
    "已完成": STATUS_LEARNED,
    "已掌握": STATUS_MASTERED,
    "掌握": STATUS_MASTERED,
    "学习中": STATUS_LEARNING,
    "未学": STATUS_NOT_STARTED,
    "未开始": STATUS_NOT_STARTED,
    "未学习": STATUS_NOT_STARTED,
}

# 英文状态词别名 → 标准枚举
_STATUS_EN_ALIASES = {
    "learned": STATUS_LEARNED,
    "complete": STATUS_LEARNED,
    "completed": STATUS_LEARNED,
    "done": STATUS_LEARNED,
    "mastered": STATUS_MASTERED,
    "master": STATUS_MASTERED,
    "learning": STATUS_LEARNING,
    "in_progress": STATUS_LEARNING,
    "studying": STATUS_LEARNING,
    "not_started": STATUS_NOT_STARTED,
    "new": STATUS_NOT_STARTED,
    "todo": STATUS_NOT_STARTED,
    "unlearned": STATUS_NOT_STARTED,
    "review_due": STATUS_REVIEW_DUE,
    "due": STATUS_REVIEW_DUE,
}

DEFAULT_SKILL_CODE = "translation"

# ---------------------------------------------------------------------------
# skill 种子数据（预置能力定义与阈值）
# ---------------------------------------------------------------------------

SKILL_SEEDS: list[dict] = [
    {
        "_id": "translation",
        "skill_code": "translation",
        "name": "翻译",
        "mastery_threshold": 0.8,
        "learned_threshold": 0.6,
    },
    {
        "_id": "listening",
        "skill_code": "listening",
        "name": "听力",
        "mastery_threshold": 0.8,
        "learned_threshold": 0.6,
    },
    {
        "_id": "speaking",
        "skill_code": "speaking",
        "name": "口语",
        "mastery_threshold": 0.8,
        "learned_threshold": 0.6,
    },
    {
        "_id": "reading",
        "skill_code": "reading",
        "name": "阅读",
        "mastery_threshold": 0.8,
        "learned_threshold": 0.6,
    },
]


def get_default_skills() -> list[dict]:
    """返回 skill 种子数据的深拷贝，避免调用方修改共享常量。"""
    return [dict(s) for s in SKILL_SEEDS]


# ---------------------------------------------------------------------------
# 状态归一化（中文 → 英文枚举）
# ---------------------------------------------------------------------------


def normalize_status(status: Any) -> str:
    """把中文/英文状态词收敛为统一英文枚举。

    无法识别时回落 `learning`（保持新写入始终有效）。
    """
    if not status:
        return STATUS_LEARNING
    s = str(status).strip()
    if s in _STATUS_CN_MAP:
        return _STATUS_CN_MAP[s]
    low = s.lower()
    return _STATUS_EN_ALIASES.get(low, STATUS_LEARNING)


# ---------------------------------------------------------------------------
# 复合键与分数换算
# ---------------------------------------------------------------------------


def skill_state_id(scholar_id: str, sentence_id: str, skill_code: str) -> str:
    """skill_state 主键：{scholar_id}_{sentence_id}_{skill_code}。"""
    return f"{scholar_id}_{sentence_id}_{skill_code}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def to_mastery_score(score: Any, mastery: Any) -> float | None:
    """score(0-100) 与 mastery(0-1) 归一为 mastery_score(0-100)。

    两者都给出时优先 score；均无效返回 None。
    """
    if score is not None and score != "":
        try:
            return clamp(float(score), 0.0, 100.0)
        except (TypeError, ValueError):
            pass
    if mastery is not None and mastery != "":
        try:
            return clamp(float(mastery) * 100.0, 0.0, 100.0)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# 滚动学习调度公式（Phase 2 落点）
# ---------------------------------------------------------------------------

REVIEW_BASE_INTERVALS_DAYS: list[int] = [1, 3, 7, 14, 30]

_DAY_SECONDS = 86400


def review_interval_seconds(attempt_count: int, mastery_score: float | None) -> int:
    """滚动学习间隔（秒）：基础 1/3/7/14/30 天随 attempt_count 递增。

    - mastery_score ≥ 80：间隔 × 1.5
    - mastery_score < 60：间隔 ÷ 2（至少 1 天）
    """
    idx = min(max(int(attempt_count), 1) - 1, len(REVIEW_BASE_INTERVALS_DAYS) - 1)
    days = float(REVIEW_BASE_INTERVALS_DAYS[idx])
    if mastery_score is not None:
        if mastery_score >= 80:
            days = days * 1.5
        elif mastery_score < 60:
            days = max(1.0, days / 2.0)
    return int(days * _DAY_SECONDS)


def compute_next_review_at(
    last_studied_at: int,
    attempt_count: int,
    mastery_score: float | None,
) -> int:
    """next_review_at = last_studied_at + interval(attempt_count, mastery_score)。"""
    return int(last_studied_at) + review_interval_seconds(attempt_count, mastery_score)


def derive_status(status: Any, mastery_score: float | None, has_mastery: bool) -> str:
    """推断状态：显式已掌握/已学保留；低掌握度 → review_due；高分无显式 → mastered。

    - mastery_score < 60 → review_due（除非显式 mastered / learned）
    - 显式 status → 尊重（normalize_status）
    - mastery_score ≥ 80 且无显式 → mastered
    - 其余 → learning
    """
    if has_mastery and mastery_score is not None and mastery_score < 60:
        norm = normalize_status(status) if status else None
        if norm in (STATUS_MASTERED, STATUS_LEARNED):
            return norm
        return STATUS_REVIEW_DUE
    if status:
        return normalize_status(status)
    if has_mastery and mastery_score is not None and mastery_score >= 80:
        return STATUS_MASTERED
    return STATUS_LEARNING


def derive_progress(status: str, mastery_score: float | None) -> float:
    """由 status / mastery_score 得 progress(0-1)。"""
    if mastery_score is not None:
        return round(clamp(mastery_score / 100.0, 0.0, 1.0), 4)
    if status == STATUS_MASTERED:
        return 1.0
    if status == STATUS_LEARNED:
        return 1.0
    if status == STATUS_LEARNING:
        return 0.5
    if status == STATUS_REVIEW_DUE:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# skill_state 文档构建（纯函数）
# ---------------------------------------------------------------------------


def build_skill_state_doc(
    *,
    scholar_id: str,
    sentence_id: str,
    skill_code: str,
    lesson_id: str | None = None,
    status: str = STATUS_LEARNING,
    mastery_score: float | None = None,
    progress: float | None = None,
    attempt_count: int = 1,
    last_studied_at: int | None = None,
    now: int | None = None,
) -> dict:
    """构建 skill_state 文档（新插入用）。"""
    now = int(now or time.time())
    last_studied_at = int(last_studied_at or now)
    status = normalize_status(status)
    if progress is None:
        progress = derive_progress(status, mastery_score)
    return {
        "_id": skill_state_id(scholar_id, sentence_id, skill_code),
        "state_id": skill_state_id(scholar_id, sentence_id, skill_code),
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "lesson_id": lesson_id,
        "skill_code": skill_code,
        "status": status,
        "mastery_score": mastery_score,
        "progress": progress,
        "attempt_count": int(attempt_count),
        "last_studied_at": last_studied_at,
        "next_review_at": compute_next_review_at(last_studied_at, attempt_count, mastery_score),
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def seed_skills(db) -> dict:
    """幂等预置 skill 种子数据（按 _id 存在则跳过）。"""
    created = 0
    skipped = 0
    for seed in get_default_skills():
        existing = await db.query(collection=SKILL, where={"_id": seed["_id"]}, limit=1)
        if existing.get("records"):
            skipped += 1
            continue
        await db.insert(collection=SKILL, data=seed)
        created += 1
    return {"created": created, "skipped": skipped}


async def upsert_skill_state(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    skill_code: str = DEFAULT_SKILL_CODE,
    **update: Any,
) -> dict:
    """按复合键 upsert 一条 skill_state；重复上报只累加 attempt_count、刷新 last_studied_at。

    update 可含：status / score(0-100) / mastery(0-1) / lesson_id / last_studied_at / now。
    返回最新状态文档。
    """
    now = int(update.get("now") or time.time())
    last_studied_at = int(update.get("last_studied_at") or now)
    state_key = skill_state_id(scholar_id, sentence_id, skill_code)

    existing = await db.query(collection=SKILL_STATE, where={"_id": state_key}, limit=1)
    records = existing.get("records", [])

    if records:
        doc = records[0]
        attempt_count = int(doc.get("attempt_count") or 0) + 1
        old_score = doc.get("mastery_score")
        mastery_score = to_mastery_score(update.get("score"), update.get("mastery"))
        if mastery_score is None:
            mastery_score = old_score
        has_mastery = mastery_score is not None
        status = derive_status(update.get("status"), mastery_score, has_mastery)
        progress = derive_progress(status, mastery_score)
        lesson_id = update.get("lesson_id") or doc.get("lesson_id")
        changes = {
            "status": status,
            "mastery_score": mastery_score,
            "progress": progress,
            "lesson_id": lesson_id,
            "attempt_count": attempt_count,
            "last_studied_at": last_studied_at,
            "next_review_at": compute_next_review_at(last_studied_at, attempt_count, mastery_score),
            "updated_at": now,
        }
        await db.update(
            collection=SKILL_STATE,
            where={"_id": state_key},
            data={"$set": changes},
            multi=False,
        )
        latest = await db.query(collection=SKILL_STATE, where={"_id": state_key}, limit=1)
        return latest["records"][0]

    mastery_score = to_mastery_score(update.get("score"), update.get("mastery"))
    has_mastery = mastery_score is not None
    status = derive_status(update.get("status"), mastery_score, has_mastery)
    doc = build_skill_state_doc(
        scholar_id=scholar_id,
        sentence_id=sentence_id,
        skill_code=skill_code,
        lesson_id=update.get("lesson_id"),
        status=status,
        mastery_score=mastery_score,
        attempt_count=1,
        last_studied_at=last_studied_at,
        now=now,
    )
    await db.insert(collection=SKILL_STATE, data=doc)
    return doc


async def get_skill_states(
    db,
    *,
    scholar_id: str,
    sentence_id: str | None = None,
    skill_code: str | None = None,
) -> dict:
    """查询学者的 skill_state 记录（可选按句子 / 能力过滤）。"""
    where: dict = {"scholar_id": scholar_id}
    if sentence_id:
        where["sentence_id"] = sentence_id
    if skill_code:
        where["skill_code"] = skill_code
    return await db.query(collection=SKILL_STATE, where=where)
