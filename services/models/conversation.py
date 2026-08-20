"""P0 Conversation MVP 数据层 — conversation_session + conversation_turn + L1 轻量状态机

设计文档 §4.2 / 契约 data-model-contract §4.11.3 / 附录 B-1 降级路径。

集合：
- `conversation_session`：会话实例（scenario/topic/difficulty/起止/summary）
- `conversation_turn`：轮次（utterance/reply/stage/hint/rephrased/eval_verdict_ref）

L1 轻量状态机（MVP，FastAPI 实现；P1 演进 L2 LangGraph + checkpointer）：
- 会话级：stage = active / ended；difficulty 档位（默认 1，冷启动先验默认）
- 轮级降级路径（附录 B-1）：consecutive_failures 累计 →
    1 次 → hint（提示）
    2 次 → rephrase（重述）
    3 次 → downgrade（降档，difficulty-1，重置计数）
  降档至 difficulty < 1 → 输出转训练建议（suggestion），P0 不终止会话（终止/转训练 P1 补全）
- 达意成功（meaningful=True 且 confidence ≥ 阈值）→ 重置计数，stage 回落 answer

门控回写（§9-2/§9-3）：每轮 eval_verdict 低置信（< EVAL_CONFIDENCE_THRESHOLD）不回写
SkillState；会话结束生成小结 + 门控 SkillState 更新记录 + ReviewSchedule 生成记录。
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from config import EVAL_CONFIDENCE_THRESHOLD
from services.events import record_attempt
from services.models_learning import upsert_skill_state

# ---------------------------------------------------------------------------
# 集合名（顶层常量，供 check_schema.py 扫描）
# ---------------------------------------------------------------------------

CONVERSATION_SESSION = "conversation_session"
CONVERSATION_TURN = "conversation_turn"

# ---------------------------------------------------------------------------
# 枚举
# ---------------------------------------------------------------------------

# 会话状态
SESSION_STAGE_ACTIVE = "active"
SESSION_STAGE_ENDED = "ended"
VALID_SESSION_STAGES = {SESSION_STAGE_ACTIVE, SESSION_STAGE_ENDED}

# 轮级阶段（降级路径推进；答案正常时为 answer）
TURN_STAGE_ANSWER = "answer"
TURN_STAGE_HINT = "hint"
TURN_STAGE_REPHRASE = "rephrase"
TURN_STAGE_DOWNGRADE = "downgrade"
VALID_TURN_STAGES = {
    TURN_STAGE_ANSWER,
    TURN_STAGE_HINT,
    TURN_STAGE_REPHRASE,
    TURN_STAGE_DOWNGRADE,
}

# 降级路径触发阈值（连续不达意次数）
FAILURES_FOR_HINT = 1
FAILURES_FOR_REPHRASE = 2
FAILURES_FOR_DOWNGRADE = 3

# 默认档位（冷启动先验默认，设计文档 §5.6.1）
DEFAULT_DIFFICULTY = 1
MIN_DIFFICULTY = 1
# 默认 skill（会话按达意/忠实评估，回写统一记 translation 对齐既有默认能力）
DEFAULT_SKILL_CODE = "translation"

# 低置信门控（§9-2）：confidence < 阈值 不回写 SkillState
LOW_CONFIDENCE_THRESHOLD = EVAL_CONFIDENCE_THRESHOLD

# S3.1 P1：会话级门控（§9-2，契约 §4.11.4）
LOW_CONF_STREAK_FOR_DOWNGRADE = 2  # 连续 2 轮低置信 → 整会话降权 ×0.5
SESSION_DOWNGRADE_FACTOR = 0.5  # 整会话降权系数
FAITHFULNESS_BIAS_THRESHOLD = 0.7  # 忠实率 < 0.7 → 标记"AI 内容偏差"，不沉淀
ANOMALY_ALERT_THRESHOLD = 0.1  # 异常率 > 10% → 告警并触发调优工单

# 会话轮数上限（防无限会话）
MAX_TURNS = 30


# ---------------------------------------------------------------------------
# 主键生成
# ---------------------------------------------------------------------------


def new_session_id(now: int | None = None) -> str:
    """生成会话主键：cvs_{毫秒时间戳}_{随机短id}。"""
    now = int(now or time.time())
    return f"cvs_{now}_{uuid.uuid4().hex[:8]}"


def new_turn_id(now: int | None = None) -> str:
    """生成轮次主键：cvt_{毫秒时间戳}_{随机短id}。"""
    now = int(now or time.time())
    return f"cvt_{now}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# 文档构建（纯函数）
# ---------------------------------------------------------------------------


def build_session_doc(
    *,
    scholar_id: str,
    scenario: str | None = None,
    topic: str | None = None,
    difficulty: int = DEFAULT_DIFFICULTY,
    sentence_ids: list[str] | None = None,
    session_id: str | None = None,
    now: int | None = None,
    cold_start: bool = False,
) -> dict:
    """构建 conversation_session 文档（active 态）。

    S3.1 P1：cold_start 记录会话创建时的冷启动判定（§9-2 冷启动期豁免降权用）。
    """
    now = int(now or time.time())
    _id = session_id or new_session_id(now)
    return {
        "_id": _id,
        "session_id": _id,
        "scholar_id": scholar_id,
        "scenario": scenario,
        "topic": topic,
        "difficulty": difficulty,
        "stage": SESSION_STAGE_ACTIVE,
        "sentence_ids": sentence_ids or [],
        "cold_start": bool(cold_start),
        "started_at": now,
        "ended_at": None,
        "summary": None,
        "skill_updates": [],  # 门控 SkillState 更新记录（§9-3）
        "review_schedule": [],  # ReviewSchedule 生成记录（§9-3）
        "created_at": now,
        "updated_at": now,
    }


def build_turn_doc(
    *,
    session_id: str,
    sentence_id: str | None,
    original_text: str,
    translation: str | None,
    utterance: str,
    reply: str,
    stage: str,
    hint: str | None = None,
    rephrased: str | None = None,
    suggestion: str | None = None,
    eval_verdict_ref: str | None = None,
    turn_id: str | None = None,
    now: int | None = None,
) -> dict:
    """构建 conversation_turn 文档（证据快照不可变：original_text/translation/utterance 落库）。"""
    now = int(now or time.time())
    _id = turn_id or new_turn_id(now)
    return {
        "_id": _id,
        "turn_id": _id,
        "session_id": session_id,
        "sentence_id": sentence_id,
        "original_text": original_text,
        "translation": translation,
        "utterance": utterance,
        "reply": reply,
        "stage": stage,
        "hint": hint,
        "rephrased": rephrased,
        "suggestion": suggestion,
        "eval_verdict_ref": eval_verdict_ref,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# L1 轻量状态机（纯函数）
# ---------------------------------------------------------------------------


def next_turn_stage(
    *,
    consecutive_failures: int,
    difficulty: int,
    meaningful: bool,
    low_confidence: bool,
) -> dict:
    """状态机推进：返回 { stage, hint, rephrased, suggestion, difficulty, reset_failures }。

    consecutive_failures 语义 = **本轮失败后的累计失败次数**（达意成功传 0）。

    规则（附录 B-1，P0 落降档分支；终止/转训练 P1 补全）：
    - 达意成功（meaningful 且非低置信）→ answer，重置计数；
    - 累计失败 1 → hint（提示）；2 → rephrase（重述）；3 → downgrade（降档，重置计数）；
    - 降档后 difficulty < MIN_DIFFICULTY → suggestion（转训练建议文案，P0 不终止会话）。
    """
    if meaningful and not low_confidence:
        return {
            "stage": TURN_STAGE_ANSWER,
            "hint": None,
            "rephrased": None,
            "suggestion": None,
            "difficulty": difficulty,
            "reset_failures": True,
        }

    if consecutive_failures >= FAILURES_FOR_DOWNGRADE:
        new_difficulty = max(MIN_DIFFICULTY, difficulty - 1)
        if new_difficulty < difficulty:
            return {
                "stage": TURN_STAGE_DOWNGRADE,
                "hint": None,
                "rephrased": None,
                "suggestion": None,
                "difficulty": new_difficulty,
                "reset_failures": True,  # 降档后重新计数
            }
        # 已到最低档：不再降档，转训练建议（P0 输出建议不终止）
        return {
            "stage": TURN_STAGE_DOWNGRADE,
            "hint": None,
            "rephrased": None,
            "suggestion": (
                "这个句子对你偏难了，建议先回到训练模块巩固这个知识点，再回来挑战。"
            ),
            "difficulty": difficulty,
            "reset_failures": True,
        }

    if consecutive_failures >= FAILURES_FOR_REPHRASE:
        return {
            "stage": TURN_STAGE_REPHRASE,
            "hint": None,
            "rephrased": True,  # 由 AI 回复承载重述文本
            "suggestion": None,
            "difficulty": difficulty,
            "reset_failures": False,
        }

    if consecutive_failures >= FAILURES_FOR_HINT:
        return {
            "stage": TURN_STAGE_HINT,
            "hint": True,  # 由 AI 回复承载提示文本
            "rephrased": None,
            "suggestion": None,
            "difficulty": difficulty,
            "reset_failures": False,
        }

    return {
        "stage": TURN_STAGE_ANSWER,
        "hint": None,
        "rephrased": None,
        "suggestion": None,
        "difficulty": difficulty,
        "reset_failures": False,
    }


def build_session_summary(turns: list[dict]) -> dict:
    """会话小结（§9-3）：汇总达意率/平均分/平均置信度/忠实率/异常率。

    S3.1 P1：新增 faithfulness_rate（忠实率）与 anomaly_rate（异常率），
    供会话级门控判定（§9-2）。
    """
    total = len(turns)
    if total == 0:
        return {
            "total_turns": 0,
            "meaningful_rate": 0.0,
            "avg_score": 0,
            "avg_confidence": 0.0,
            "faithfulness_rate": 0.0,
            "anomaly_rate": 0.0,
        }
    meaningful = sum(
        1 for t in turns if (t.get("eval_verdict") or {}).get("meaningful")
    )
    scores = [
        (t.get("eval_verdict") or {}).get("score") or 0 for t in turns
    ]
    confs = [
        (t.get("eval_verdict") or {}).get("confidence") or 0.0 for t in turns
    ]
    # 忠实率分母 = 达意轮数（faithfulness 只在达意轮有意义，未达意轮不稀释）
    faithful = sum(
        1 for t in turns if (t.get("eval_verdict") or {}).get("faithfulness")
    )
    anomaly = sum(
        1 for t in turns if (t.get("eval_verdict") or {}).get("anomaly")
    )
    faithfulness_rate = round(faithful / meaningful, 4) if meaningful else 1.0
    return {
        "total_turns": total,
        "meaningful_rate": round(meaningful / total, 4),
        "avg_score": int(round(sum(scores) / total)),
        "avg_confidence": round(sum(confs) / total, 4),
        "faithfulness_rate": faithfulness_rate,
        "anomaly_rate": round(anomaly / total, 4),
    }


def session_gate(turns: list[dict], *, is_cold: bool = False) -> dict:
    """会话级门控（§9-2，P1 启用）。

    - 连续 ≥2 轮低置信 → downgrade_factor=0.5（整会话降权 ×0.5；冷启动期豁免不惩罚）
    - 忠实率 < 0.7 → ai_content_bias=True（标记"AI 内容偏差"，不沉淀为能力证据）
    - 异常率 > 10% → alert=True（告警并触发调优工单）

    返回：{consecutive_low_conf, downgrade_factor, ai_content_bias,
           faithfulness_rate, anomaly_rate, alert}
    """
    verdicts = [(t.get("eval_verdict") or {}) for t in turns]
    total = len(verdicts)
    if total == 0:
        return {
            "consecutive_low_conf": 0,
            "downgrade_factor": 1.0,
            "ai_content_bias": False,
            "faithfulness_rate": 0.0,
            "anomaly_rate": 0.0,
            "alert": False,
        }

    # 连续低置信轮数
    streak = best = 0
    for v in verdicts:
        if (v.get("confidence") or 0.0) < LOW_CONFIDENCE_THRESHOLD:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    consecutive_low_conf = best

    meaningful = sum(1 for v in verdicts if v.get("meaningful"))
    faithful = sum(1 for v in verdicts if v.get("faithfulness"))
    anomaly = sum(1 for v in verdicts if v.get("anomaly"))
    # 忠实率分母 = 达意轮数（faithfulness 只在达意轮有意义；无达意轮视为无偏差）
    faithfulness_rate = round(faithful / meaningful, 4) if meaningful else 1.0
    anomaly_rate = round(anomaly / total, 4)

    downgrade = (
        consecutive_low_conf >= LOW_CONF_STREAK_FOR_DOWNGRADE and not is_cold
    )
    return {
        "consecutive_low_conf": consecutive_low_conf,
        "downgrade_factor": SESSION_DOWNGRADE_FACTOR if downgrade else 1.0,
        "ai_content_bias": faithfulness_rate < FAITHFULNESS_BIAS_THRESHOLD,
        "faithfulness_rate": faithfulness_rate,
        "anomaly_rate": anomaly_rate,
        "alert": anomaly_rate > ANOMALY_ALERT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def get_session(db, session_id: str) -> dict | None:
    """按主键取会话；不存在返回 None。"""
    result = await db.query(
        collection=CONVERSATION_SESSION, where={"_id": session_id}, limit=1
    )
    records = result.get("records", [])
    return records[0] if records else None


async def list_turns(db, session_id: str, limit: int = MAX_TURNS) -> list[dict]:
    """会话内轮次（时间升序，含 eval_verdict 内联）。"""
    result = await db.query(
        collection=CONVERSATION_TURN,
        where={"session_id": session_id},
        order=[{"field": "created_at", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def create_session(db, doc: dict) -> dict:
    """创建会话并落库，返回会话文档（含 _id）。"""
    inserted = await db.insert(CONVERSATION_SESSION, doc)
    doc["_id"] = (inserted.get("ids") or [None])[0]
    return doc


async def append_turn(db, doc: dict) -> dict:
    """追加轮次并落库，返回轮次文档（含 _id）。"""
    inserted = await db.insert(CONVERSATION_TURN, doc)
    doc["_id"] = (inserted.get("ids") or [None])[0]
    return doc


async def end_session_with_summary(
    db,
    session: dict,
    turns: list[dict],
    eval_by_turn: dict[str, dict],
    *,
    is_cold: bool = False,
) -> dict:
    """结束会话：生成小结 + 会话级门控 + SkillState 更新 + ReviewSchedule 生成记录。

    - 会话级门控（§9-2，S3.1 P1）：
      - 连续 ≥2 轮低置信 → 整会话降权 ×0.5（冷启动期豁免）
      - 忠实率 < 0.7 → 标记"AI 内容偏差"，跳过 SkillState 回写（不沉淀为能力证据）
      - 异常率 > 10% → alert 告警（summary 内标记，触发调优工单）
    - 每轮 eval_verdict 置信度 ≥ 阈值 且 meaningful → 回写 upsert_skill_state（门控 §9-2）
    - 回写同时追加一条 learning attempt 事件（record_attempt，会话关联）
    - 小结写入 summary（含 gate 快照）；skill_updates / review_schedule 落会话文档
    """
    gate = session_gate(turns, is_cold=is_cold)
    summary = build_session_summary(turns)
    summary["gate"] = gate
    skill_updates: list[dict] = []
    review_schedule: list[dict] = []

    for turn in turns:
        verdict = turn.get("eval_verdict") or {}
        sentence_id = turn.get("sentence_id")
        if not sentence_id:
            continue
        if not verdict.get("meaningful"):
            continue
        if (verdict.get("confidence") or 0.0) < LOW_CONFIDENCE_THRESHOLD:
            continue  # 低置信不回写（§9-2）
        if gate["ai_content_bias"]:
            continue  # AI 内容偏差：不沉淀为能力证据（§9-2）

        score = verdict.get("score") or 0
        state = await upsert_skill_state(
            db,
            scholar_id=session["scholar_id"],
            sentence_id=sentence_id,
            skill_code=DEFAULT_SKILL_CODE,
            score=score,
            status="mastered" if score >= 80 else None,
            sparse_discount=True,  # 证据稀疏保护（§5.6.2）：<3 次尝试更新量打折
            weight=gate["downgrade_factor"],  # 连续低置信 → 整会话降权 ×0.5
            confidence=verdict.get("confidence"),
            outcome="success" if score >= 60 else "fail",
            difficulty=session.get("difficulty"),
        )
        skill_updates.append(
            {
                "sentence_id": sentence_id,
                "mastery_score": state.get("mastery_score"),
                "status": state.get("status"),
                "attempt_count": state.get("attempt_count"),
                "next_review_at": state.get("next_review_at"),
                "confidence": state.get("confidence"),
                "stability": state.get("stability"),
                "difficulty": state.get("difficulty"),
            }
        )
        review_schedule.append(
            {
                "sentence_id": sentence_id,
                "next_review_at": state.get("next_review_at"),
            }
        )
        await record_attempt(
            db,
            scholar_id=session["scholar_id"],
            sentence_id=sentence_id,
            skill_code=DEFAULT_SKILL_CODE,
            attempt_type="speak",
            status="correct" if score >= 60 else "incorrect",
            score=score,
            session_id=session["session_id"],
        )

    now = int(time.time() * 1000)
    changes = {
        "stage": SESSION_STAGE_ENDED,
        "ended_at": now,
        "summary": summary,
        "skill_updates": skill_updates,
        "review_schedule": review_schedule,
        "updated_at": now,
    }
    await db.update(
        collection=CONVERSATION_SESSION,
        where={"_id": session["_id"]},
        data={"$set": changes},
        multi=False,
    )
    updated = await get_session(db, session["session_id"])
    return updated or session


async def update_session_stage(
    db, session: dict, *, stage: str, difficulty: int | None = None
) -> None:
    """轻量更新会话 stage / difficulty（轮次推进时使用）。"""
    changes: dict[str, Any] = {
        "stage": stage,
        "updated_at": int(time.time() * 1000),
    }
    if difficulty is not None:
        changes["difficulty"] = difficulty
    await db.update(
        collection=CONVERSATION_SESSION,
        where={"_id": session["_id"]},
        data={"$set": changes},
        multi=False,
    )
