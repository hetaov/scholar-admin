"""学习状态模型 — skill 定义 + skill_state 当前状态读写（Phase 2 能力模型）

集合：
- `skill`      ：能力定义（种子数据：translation / listening / speaking / reading）
- `skill_state`：学者 × 句子 × 能力 的当前状态，主键 `{scholar_id}_{sentence_id}_{skill_code}`

字段映射（旧能力状态模型 → skill_state）：
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

from config import MIN_EVIDENCE

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
# S3.1 P1：SkillState 置信度 / 稳定性 / 难度策略常量（契约 §4.11.4 启用，设计文档 §5.4/§5.6.2）
# ---------------------------------------------------------------------------

CONFIDENCE_HISTORY_SIZE = 5  # confidence = 近 N 次评估置信度均值（EWMA 窗口）
STABILITY_MIN_STREAK = 2  # stability 至少 2 次同方向才上升（§5.6.2）
STABILITY_UP_STEP = 0.2  # 稳定性上升步长
STABILITY_DOWN_STEP = 0.1  # 稳定性衰减步长（反方向立即回落）
DIFFICULTY_MIN = 1  # 难度档位下限（对齐冷启动先验）
DIFFICULTY_MAX = 5  # 难度档位上限

# S3.1 P1：Activity → Skill 权重配置种子（契约 §4.11.5，草稿 §二十四）
ACTIVITY_SKILL_WEIGHT = "activity_skill_weight"
ACTIVITY_SKILL_WEIGHT_SEEDS: dict[str, dict[str, float]] = {
    "SHADOWING": {
        "Pronunciation": 0.45,
        "Fluency": 0.30,
        "Listening": 0.15,
        "Speaking": 0.10,
    },
    "TRANSLATION": {
        "Recall": 0.50,
        "Usage": 0.30,
        "Grammar": 0.20,
    },
    "DICTATION": {
        "Listening": 0.60,
        "Spelling": 0.20,
        "Recall": 0.20,
    },
    "CONVERSATION": {
        "Speaking": 0.40,
        "Usage": 0.30,
        "Listening": 0.20,
        "Fluency": 0.10,
    },
}

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


# 默认分级阈值（0-1 比例）；未在 SKILL_SEEDS 中找到 skill_code 时回落。
# 设计文档已为每个能力单独配置 learned_threshold=0.6 / mastered_threshold=0.8，
# 但原 derive_status 未消费，导致 60-80 区间永远停留在 learning。
# 现新增第三个产出档位：learned（介于 learning 与 mastered 之间），
# 配合 SKILL_SEEDS.per_skill 阈值使得"已学但未掌握"可被显式表示，不再卡死 learning。
_DEFAULT_LEARNED_THRESHOLD = 0.6
_DEFAULT_MASTERED_THRESHOLD = 0.8


def _resolve_thresholds(skill_code: str | None) -> tuple[float, float]:
    """按 skill_code 查 SKILL_SEEDS 的 (learned_threshold, mastered_threshold)。

    未匹配（未知能力/None）回落默认值；阈值以 0-1 比例返回（独立于 mastery_score 0-100 入参）。
    """
    if skill_code:
        for seed in SKILL_SEEDS:
            if seed.get("skill_code") == skill_code:
                try:
                    learned = float(seed.get("learned_threshold") or _DEFAULT_LEARNED_THRESHOLD)
                except (TypeError, ValueError):
                    learned = _DEFAULT_LEARNED_THRESHOLD
                try:
                    mastered = float(seed.get("mastery_threshold") or _DEFAULT_MASTERED_THRESHOLD)
                except (TypeError, ValueError):
                    mastered = _DEFAULT_MASTERED_THRESHOLD
                return clamp(learned, 0.0, 1.0), clamp(mastered, 0.0, 1.0)
    return _DEFAULT_LEARNED_THRESHOLD, _DEFAULT_MASTERED_THRESHOLD


def derive_status(
    status: Any,
    mastery_score: float | None,
    has_mastery: bool,
    *,
    skill_code: str | None = None,
) -> str:
    """推断状态：显式状态优先；分数自动分级（learned/mastered 双阈值，未达 learned 为 learning/review_due）。

    阈值来源：SKILL_SEEDS.per_skill 的 learned_threshold / mastery_threshold（0-1 比例），
    mastery_score 入参为 0-100；未匹配的能力回落 0.6 / 0.8。

    决策表（has_mastery=True 且无显式状态词时）：
    - score < learned_threshold×100  → review_due   （除非显式 mastered / learned）
    - learned_threshold×100 ≤ score < mastered_threshold×100  → learned   ← 新增
    - score ≥ mastered_threshold×100  → mastered
    - 无 mastery_score                → learning
    """
    learned_thr, mastered_thr = _resolve_thresholds(skill_code)
    if has_mastery and mastery_score is not None and mastery_score < learned_thr * 100.0:
        norm = normalize_status(status) if status else None
        if norm in (STATUS_MASTERED, STATUS_LEARNED):
            return norm
        return STATUS_REVIEW_DUE
    if status:
        return normalize_status(status)
    if has_mastery and mastery_score is not None:
        if mastery_score >= mastered_thr * 100.0:
            return STATUS_MASTERED
        if mastery_score >= learned_thr * 100.0:
            return STATUS_LEARNED
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
# S3.1 P1：confidence / stability / difficulty 更新策略（§5.6.2，契约 §4.11.4）
# ---------------------------------------------------------------------------


def update_confidence(
    prev: float | None, new_confidence: float, attempt_count: int
) -> float:
    """confidence = 近 N 次评估置信度均值(EWMA) × 证据系数（§5.6.2）。

    - alpha = 1 / min(attempt_count, CONFIDENCE_HISTORY_SIZE)：证据越多越接近均值
    - 证据系数 = min(1, attempt_count / MIN_EVIDENCE)：证据稀疏时整体打折（冷启动保护）
    """
    attempt = max(int(attempt_count), 1)
    alpha = 1.0 / min(attempt, CONFIDENCE_HISTORY_SIZE)
    ewma = float(prev or 0.0) * (1.0 - alpha) + float(new_confidence) * alpha
    evidence = min(1.0, attempt / MIN_EVIDENCE) if MIN_EVIDENCE > 0 else 1.0
    return round(clamp(ewma * evidence, 0.0, 1.0), 4)


def update_stability(
    prev: float | None,
    outcome: str,
    last_outcome: str | None,
    stable_streak: int,
) -> tuple[float, str, int]:
    """stability 至少 2 次同方向才上升，反方向立即衰减（§5.6.2）。

    返回 (new_stability, last_outcome, stable_streak)。
    - 首次（无 last_outcome）：只记录方向，不改变稳定性
    - 连续 ≥2 次同方向：stability + STABILITY_UP_STEP
    - 反方向：stability - STABILITY_DOWN_STEP，streak 重置为 1
    """
    outcome = str(outcome or "")
    streak = int(stable_streak or 0)
    stability = float(prev or 0.0)
    if not last_outcome:
        return round(stability, 4), outcome, 1
    if outcome == str(last_outcome):
        streak += 1
        if streak >= STABILITY_MIN_STREAK:
            stability = min(1.0, stability + STABILITY_UP_STEP)
        return round(stability, 4), outcome, streak
    return round(max(0.0, stability - STABILITY_DOWN_STEP), 4), outcome, 1


def update_difficulty(prev: int | float | None, new_difficulty: int | float | None) -> int:
    """difficulty 取当前档位（会话/训练使用），clamp 到 [DIFFICULTY_MIN, DIFFICULTY_MAX]。"""
    if new_difficulty is None:
        return int(prev or DIFFICULTY_MIN)
    return int(clamp(float(new_difficulty), DIFFICULTY_MIN, DIFFICULTY_MAX))


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
    confidence: float | None = None,
    stability: float | None = None,
    difficulty: int | None = None,
) -> dict:
    """构建 skill_state 文档（新插入用）。

    S3.1 P1：可选写入 confidence / stability / difficulty（契约 §4.11.4），
    未提供时保持 nullable 不写入（向后兼容存量数据）。
    """
    now = int(now or time.time())
    last_studied_at = int(last_studied_at or now)
    status = normalize_status(status)
    if progress is None:
        progress = derive_progress(status, mastery_score)
    doc = {
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
    if confidence is not None:
        doc["confidence"] = round(clamp(float(confidence), 0.0, 1.0), 4)
    if stability is not None:
        doc["stability"] = round(clamp(float(stability), 0.0, 1.0), 4)
    if difficulty is not None:
        doc["difficulty"] = update_difficulty(None, difficulty)
    return doc


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


async def seed_activity_skill_weights(db) -> dict:
    """幂等预置 Activity → Skill 权重配置（契约 §4.11.5，草稿 §二十四）。"""
    created = 0
    skipped = 0
    for activity, weights in ACTIVITY_SKILL_WEIGHT_SEEDS.items():
        existing = await db.query(
            collection=ACTIVITY_SKILL_WEIGHT, where={"activity": activity}, limit=1
        )
        if existing.get("records"):
            skipped += 1
            continue
        await db.insert(
            collection=ACTIVITY_SKILL_WEIGHT,
            data={"activity": activity, "skill_weights": dict(weights)},
        )
        created += 1
    return {"created": created, "skipped": skipped}


async def get_activity_skill_weights(db, activity: str) -> dict:
    """读取某 Activity 的权重配置；未预置回退内置默认（草稿 §二十四）。"""
    activity = str(activity or "").strip()
    result = await db.query(
        collection=ACTIVITY_SKILL_WEIGHT, where={"activity": activity}, limit=1
    )
    records = result.get("records", [])
    if records:
        return records[0].get("skill_weights") or {}
    return dict(ACTIVITY_SKILL_WEIGHT_SEEDS.get(activity, {}))


async def upsert_skill_state(
    db,
    *,
    scholar_id: str,
    sentence_id: str,
    skill_code: str = DEFAULT_SKILL_CODE,
    sparse_discount: bool = False,
    confidence: float | None = None,
    outcome: str | None = None,
    difficulty: int | None = None,
    weight: float = 1.0,
    **update: Any,
) -> dict:
    """按复合键 upsert 一条 skill_state；重复上报只累加 attempt_count、刷新 last_studied_at。

    update 可含：status / score(0-100) / mastery(0-1) / lesson_id / last_studied_at / now。

    sparse_discount（证据稀疏保护，设计文档 §5.6.2，冷启动路径启用）：
    - attempt_count < MIN_EVIDENCE 时，增量更新量按 `attempt_count / MIN_EVIDENCE` 打折
      （第 1 次后续更新只贡献 1/3，防单次偶然污染）；
    - 默认 False 保持既有调用行为不变（旧调用不受影响）。

    S3.1 P1（契约 §4.11.4 启用）：
    - confidence：本轮评估置信度，写入前经 `update_confidence`（近 N 次均值 × 证据系数）
    - outcome：本次结果方向（"success" / "fail"），用于 `update_stability` 稳定性更新
    - difficulty：当前难度档位，直接落档（clamp [1, 5]）
    - weight：增量权重（门控降权用，如整会话 ×0.5），与 sparse_discount 叠加

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
        raw_score = to_mastery_score(update.get("score"), update.get("mastery"))
        effective_weight = float(weight) if weight else 1.0
        if raw_score is None:
            mastery_score = old_score
        elif sparse_discount and attempt_count < MIN_EVIDENCE:
            # 证据稀疏打折：只按证据比例贡献增量（§5.6.2）
            effective_weight *= attempt_count / MIN_EVIDENCE
            mastery_score = (old_score or 0.0) + (raw_score - (old_score or 0.0)) * effective_weight
        elif effective_weight != 1.0:
            # 门控降权（如整会话 ×0.5）：增量打折而非全量打折
            mastery_score = (old_score or 0.0) + (raw_score - (old_score or 0.0)) * effective_weight
        else:
            mastery_score = raw_score
        has_mastery = mastery_score is not None
        status = derive_status(
            update.get("status"), mastery_score, has_mastery, skill_code=skill_code
        )
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
        # S3.1 P1：confidence / stability / difficulty 更新（契约 §4.11.4）
        if confidence is not None:
            changes["confidence"] = update_confidence(
                doc.get("confidence"), confidence, attempt_count
            )
        if outcome:
            new_stability, last_outcome, stable_streak = update_stability(
                doc.get("stability"),
                outcome,
                doc.get("last_outcome"),
                doc.get("stable_streak", 0),
            )
            changes["stability"] = new_stability
            changes["last_outcome"] = last_outcome
            changes["stable_streak"] = stable_streak
        if difficulty is not None:
            changes["difficulty"] = update_difficulty(doc.get("difficulty"), difficulty)
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
    status = derive_status(
        update.get("status"), mastery_score, has_mastery, skill_code=skill_code
    )
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
        difficulty=difficulty,
    )
    # S3.1 P1：首次置信度同样经证据系数打折（§5.6.2：attempt=1 → ×1/3）
    if confidence is not None:
        doc["confidence"] = update_confidence(None, confidence, 1)
    # S3.1 P1：首次带 outcome 时记录方向（稳定性首次不升降，§5.6.2）
    if outcome:
        doc["last_outcome"] = str(outcome)
        doc["stable_streak"] = 1
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
