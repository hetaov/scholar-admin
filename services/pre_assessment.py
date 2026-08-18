"""P1 前置评估（设计文档 §6.2 / §9-5；执行计划 S3.2）— Gate 建议 + 难度档位 + Activity 推荐

原则：
- 建议层不硬阻断（§0.2/§9-5）：gate_suggestion 仅建议，前端可展示但不得阻断会话创建。
- 纯规则、零外部输入依赖（P1 约束）：仅基于 skill_state 聚合，无新增评测调用。
- 冷启动回退（§5.6/§6.2 冷启动约束）：无历史 → 先验默认（gate="pass" 不触发推荐）
  + 标准引导序列，不报错不拒绝。
- 证据稀疏保护（§5.6.3）：总尝试数 < MIN_EVIDENCE 时弱项不可信，Activity 推荐回退
  标准引导序列（非弱项驱动）。

数据来源：skill_state 聚合（mastery_score / score / mastery / difficulty / attempt_count）。
"""
from __future__ import annotations

from config import MIN_EVIDENCE
from services.cold_start import COLD_START_SEQUENCE
from services.models_conversation import DEFAULT_DIFFICULTY, DEFAULT_SKILL_CODE
from services.models_learning import (
    ACTIVITY_SKILL_WEIGHT_SEEDS,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    clamp,
    get_skill_states,
    to_mastery_score,
)

# Gate 建议取值（§6.2，建议层不阻断）
GATE_PASS = "pass"  # 可自由进入会话
GATE_TRAINING = "training"  # 建议先做 Shadowing/翻译等训练再进自由会话
GATE_CONTENT = "content"  # 完全陌生，建议先看内容 Tab

# §6.2 Gate 阈值
GATE_CONTENT_THRESHOLD = 0.3  # mastery < 0.3 → 完全陌生，先看内容
GATE_TRAINING_THRESHOLD = 0.5  # mastery < 0.5 → 建议先训练

# §6.2 难度分档（按 mastery；示例 0.5~0.7→1、0.7~0.85→2、>0.85→3）
DIFFICULTY_BAND_3 = 0.85
DIFFICULTY_BAND_2 = 0.7

# skill_state.skill_code → ACTIVITY_SKILL_WEIGHT_SEEDS 技能名映射（草稿 §二十四）
# 用于把"某能力弱"归因到各 Activity 覆盖的具体技能（reading 无对应受控技能，匹配 0）
SKILL_CODE_TO_SKILL_NAMES: dict[str, tuple[str, ...]] = {
    "translation": ("Recall", "Usage", "Grammar"),
    "listening": ("Listening",),
    "speaking": ("Speaking", "Fluency", "Pronunciation"),
    "reading": (),
}


def _record_mastery(record: dict) -> float | None:
    """单条 skill_state → mastery（0~1）；无有效值返回 None。

    优先 mastery_score（0~100），否则经 `to_mastery_score` 归一（score / mastery）。
    """
    raw = record.get("mastery_score")
    if raw is not None and raw != "":
        try:
            return clamp(float(raw), 0.0, 100.0) / 100.0
        except (TypeError, ValueError):
            pass
    score = to_mastery_score(record.get("score"), record.get("mastery"))
    if score is None:
        return None
    return score / 100.0


def aggregate_mastery(records: list[dict]) -> dict[str, float]:
    """按 skill_code 聚合 mastery（0~1）：同能力取均值；无有效 mastery 的记录跳过。"""
    buckets: dict[str, list[float]] = {}
    for record in records:
        skill = str(record.get("skill_code") or DEFAULT_SKILL_CODE)
        m = _record_mastery(record)
        if m is None:
            continue
        buckets.setdefault(skill, []).append(m)
    return {skill: round(sum(v) / len(v), 4) for skill, v in buckets.items()}


def _history_difficulty(records: list[dict]) -> float | None:
    """历史难度档位（§6.2"参考历史难度档位"）：有效 difficulty 均值；无 → None。"""
    values: list[float] = []
    for record in records:
        d = record.get("difficulty")
        if d is None or d == "":
            continue
        try:
            values.append(float(d))
        except (TypeError, ValueError):
            continue
    return round(sum(values) / len(values), 2) if values else None


def _total_attempts(records: list[dict]) -> int:
    """总尝试数（证据量，§5.6.3）。"""
    return sum(int(record.get("attempt_count") or 0) for record in records)


def gate_suggestion(mastery: float | None) -> str:
    """Gate 建议（§6.2）：完全陌生→先看内容；偏弱→先训练；否则 pass（不阻断）。

    无可用数据 → "pass"（§5.6.1 先验默认不触发推荐）。
    """
    if mastery is None:
        return GATE_PASS
    if mastery < GATE_CONTENT_THRESHOLD:
        return GATE_CONTENT
    if mastery < GATE_TRAINING_THRESHOLD:
        return GATE_TRAINING
    return GATE_PASS


def suggest_difficulty(mastery: float | None, history_difficulty: float | None = None) -> int:
    """难度档位（§6.2）：按 mastery 分档；无 mastery 时参考历史难度档位；均无 → 先验默认 1。"""
    if mastery is not None:
        if mastery >= DIFFICULTY_BAND_3:
            return 3
        if mastery >= DIFFICULTY_BAND_2:
            return 2
        return 1
    if history_difficulty is not None:
        return int(clamp(round(history_difficulty), DIFFICULTY_MIN, DIFFICULTY_MAX))
    return DEFAULT_DIFFICULTY


def recommend_activities(mastery_by_skill: dict[str, float]) -> list[str]:
    """弱项驱动 Activity 推荐（草稿 §二十四/§二十五 AI Recommended Activities）。

    对各候选 Activity（ACTIVITY_SKILL_WEIGHT_SEEDS 的 4 个受控任务），按其覆盖技能与
    弱项（低 mastery）的匹配度加权得分降序排序，返回推荐列表（小写 activity 名）。

    无弱项数据 → 回退标准引导序列（§5.6.3）。
    """
    if not mastery_by_skill:
        return list(COLD_START_SEQUENCE)
    scored: list[tuple[float, str]] = []
    for activity, weights in ACTIVITY_SKILL_WEIGHT_SEEDS.items():
        score = 0.0
        for skill, m in mastery_by_skill.items():
            matched = sum(
                weights.get(name, 0.0)
                for name in SKILL_CODE_TO_SKILL_NAMES.get(skill, ())
            )
            score += (1.0 - m) * matched  # 技能越弱（mastery 越低）得分越高
        scored.append((round(score, 4), activity.lower()))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [name for _, name in scored]


async def pre_assess(
    db,
    *,
    scholar_id: str,
    sentence_id: str | None = None,
) -> dict:
    """前置评估主入口（S3.2）：基于 skill_state 聚合生成建议（纯规则、零外部依赖）。

    返回：
    - has_history：是否有学习历史（§5.6.5，决定会话 "cold_start" 标记）
    - gate_suggestion："pass" | "training" | "content"（建议层，不阻断）
    - difficulty：会话初始难度档位（写入会话，对 ConversationGraph 生效）
    - activity_recommendation：推荐 Activities 列表（弱项驱动；证据稀疏/无历史回退引导序列）
    - mastery：聚合 mastery（0~1；无可用数据为 None）
    - evidence_sparse：证据是否稀疏（总尝试数 < MIN_EVIDENCE，§5.6.3）
    """
    result = await get_skill_states(
        db, scholar_id=scholar_id, sentence_id=sentence_id
    )
    records = result.get("records") or []
    if not records:
        # 冷启动回退（§5.6 / §6.2）：先验默认，不报错不拒绝
        return {
            "has_history": False,
            "gate_suggestion": GATE_PASS,
            "difficulty": DEFAULT_DIFFICULTY,
            "activity_recommendation": list(COLD_START_SEQUENCE),
            "mastery": None,
            "evidence_sparse": True,
        }

    mastery_by_skill = aggregate_mastery(records)
    overall = (
        round(sum(mastery_by_skill.values()) / len(mastery_by_skill), 4)
        if mastery_by_skill
        else None
    )
    history_difficulty = _history_difficulty(records)
    total_attempts = _total_attempts(records)
    evidence_sparse = total_attempts < MIN_EVIDENCE

    return {
        "has_history": True,
        "gate_suggestion": gate_suggestion(overall),
        "difficulty": suggest_difficulty(overall, history_difficulty),
        "activity_recommendation": (
            list(COLD_START_SEQUENCE)
            if evidence_sparse
            else recommend_activities(mastery_by_skill)
        ),
        "mastery": overall,
        "evidence_sparse": evidence_sparse,
    }
