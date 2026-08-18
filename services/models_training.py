"""P0 Training 受控任务数据层 — learning_attempt（mode='training'）+ 弱项驱动 + 逐题判定

设计文档 §3.2 TrainingGraph / 契约 data-model-contract §4.11.1 / 附录 B-2。

集合：
- `learning_attempt`：统一尝试记录（Training 并入本表，不建独立 `training_session`）。
  任务创建时写入一条 `attempt_status='pending'` 的 task 快照（task_id + items）；
  evaluate 时在同一文档上补判定结果并回写 SkillState。

弱项驱动（§9-4）：基于 `skill_state` 该 skill 的 mastery 排序，取最弱句；
无历史（冷启动 §5.6）回退标准引导序列（content → shadowing → translation → listening），
从当前教材任意句选取，不报错不阻断。

逐题判定（§3.3 Evaluator 映射）：
- 翻译类：LLM + Rule（evaluate_text，达意/忠实/异常/置信度）
- 判定正确：score ≥ 60 且 meaningful；低置信（< EVAL_CONFIDENCE_THRESHOLD）不回写 SkillState
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from config import EVAL_CONFIDENCE_THRESHOLD

logger = logging.getLogger("scholar-admin.models.training")

LEARNING_ATTEMPT = "learning_attempt"

# 任务生成数量（P0：单任务最多 3 题）
TASK_ITEM_LIMIT = 3
# 判定通过线（0~100）
PASS_SCORE = 60

# 低置信门控（§9-2）
LOW_CONFIDENCE_THRESHOLD = EVAL_CONFIDENCE_THRESHOLD

# 标准引导序列（冷启动，§5.6.3）
COLD_START_SEQUENCE = ("content", "shadowing", "translation", "listening")

# Activity → Skill 权重表（配置集合 activity_skill_weight 的 P0 内置默认值，草稿 §二十四）
ACTIVITY_SKILL_WEIGHTS: dict[str, dict[str, float]] = {
    "SHADOWING": {"pronunciation": 0.45, "fluency": 0.30, "intonation": 0.25},
    "TRANSLATION": {"translation": 0.60, "vocabulary": 0.40},
    "LISTENING": {"listening": 0.55, "comprehension": 0.45},
    "CONVERSATION": {"translation": 0.35, "fluency": 0.30, "comprehension": 0.35},
}


def new_task_id(now: int | None = None) -> str:
    """受控任务主键：trn_{毫秒时间戳}_{随机短id}。"""
    now = int(now or time.time())
    return f"trn_{now}_{uuid.uuid4().hex[:8]}"


def build_training_task(
    *,
    scholar_id: str,
    skill_code: str,
    difficulty: int,
    items: list[dict],
    task_id: str | None = None,
    now: int | None = None,
) -> dict:
    """构建 learning_attempt（mode='training'）任务快照文档。

    items: [{ item_id, sentence_id, content(英文), prompt(引导/刺激), translation }]
    """
    now = int(now or time.time())
    _id = task_id or new_task_id(now)
    return {
        "_id": _id,
        "task_id": _id,
        "scholar_id": scholar_id,
        "mode": "training",
        "skill_code": skill_code,
        "difficulty": difficulty,
        "attempt_status": "pending",
        "items": items,
        "created_at": now,
        "updated_at": now,
    }


def build_item(item_index: int, sentence: dict, activity: str) -> dict:
    """按活动类型构建受控任务 item（stimulus + 作答引导）。

    P0 覆盖三种 Evaluator 映射（§3.3）：
    - translation：中文刺激 → 英文作答（LLM + Rule）
    - shadowing：跟读目标英文（SOE-N，P0 以文本近似）
    - listening：听力（TTS 音频，P0 以文本近似）
    """
    text = str(sentence.get("text") or "").strip()
    translation = str(sentence.get("translation") or "").strip()
    activity = str(activity or "translation").upper()
    prompt_map = {
        "TRANSLATION": (
            f"翻译为英文：{translation}" if translation else f"说出英文：{text}"
        ),
        "SHADOWING": f"跟读这句英文：{text}",
        "LISTENING": f"听音频后复述或作答（参考：{text}）",
    }
    return {
        "item_id": f"it_{item_index + 1}",
        "sentence_id": sentence.get("sentence_id") or "",
        "content": text,
        "translation": translation,
        "prompt": prompt_map.get(activity, prompt_map["TRANSLATION"]),
        "activity": activity,
    }


def evaluate_item(
    item: dict, response: str, judge_fn
) -> dict:
    """逐题判定（§3.3）：返回 { item_id, correct, feedback, score, confidence }。

    judge_fn: 文本评估函数（默认可注入 evaluation_engine.evaluate_text，
    测试可替换为确定性桩）。判定口径：score ≥ 60 且 meaningful 即 correct；
    低置信（< 阈值）判为不确定（correct=False，feedback 提示重试）。
    """
    response = str(response or "").strip()
    original = item.get("content") or ""
    verdict = judge_fn(original, response)
    score = int(verdict.get("score") or 0)
    confidence = float(verdict.get("confidence") or 0.0)
    meaningful = bool(verdict.get("meaningful"))
    anomaly = bool(verdict.get("anomaly"))

    if anomaly or not response:
        correct = False
        feedback = "未检测到有效作答，请重新尝试。"
    elif confidence < LOW_CONFIDENCE_THRESHOLD:
        correct = False
        feedback = "作答无法可靠判定，请再试一次（说完整一些）。"
    elif meaningful and score >= PASS_SCORE:
        correct = True
        feedback = "很好！达意且忠实，继续加油。"
    else:
        correct = False
        feedback = "还需努力：再对照目标句练习一次。"

    return {
        "item_id": item.get("item_id"),
        "correct": correct,
        "feedback": feedback,
        "score": score,
        "confidence": confidence,
    }


def summarize_results(results: list[dict]) -> dict:
    """任务整体汇总：correct / total / 平均分。"""
    total = len(results)
    if total == 0:
        return {"correct": 0, "total": 0, "avg_score": 0}
    correct = sum(1 for r in results if r.get("correct"))
    avg = int(round(sum(r.get("score") or 0 for r in results) / total))
    return {"correct": correct, "total": total, "avg_score": avg}


def apply_results_to_task(task: dict, results: list[dict]) -> dict:
    """把判定结果合并回任务快照（items 补 correct/score/feedback/confidence）。"""
    by_item = {r["item_id"]: r for r in results}
    updated_items = []
    error_types: set[str] = set()
    for it in task.get("items", []):
        merged = {**it, **by_item.get(it.get("item_id"), {})}
        if not merged.get("correct") and not (merged.get("score") or 0) >= PASS_SCORE:
            # 未通过 → 按活动推断 error_type（草稿 §九；P0 简化为 comprehensiom 兜底）
            error_types.add("comprehension")
        updated_items.append(merged)
    overall = summarize_results(results)
    status = "completed" if overall["correct"] == overall["total"] else "failed"
    return {
        **task,
        "items": updated_items,
        "attempt_status": status,
        "overall": overall,
        "error_type": sorted(error_types)[0] if error_types else None,
        "updated_at": int(time.time() * 1000),
    }


# ---------------------------------------------------------------------------
# 读写（经 db）
# ---------------------------------------------------------------------------


async def get_task(db, task_id: str) -> dict | None:
    """按主键取受控任务（learning_attempt 集合）；不存在返回 None。"""
    result = await db.query(collection=LEARNING_ATTEMPT, where={"_id": task_id}, limit=1)
    records = result.get("records", [])
    return records[0] if records else None


async def create_task(db, doc: dict) -> dict:
    """创建受控任务并落库（learning_attempt，mode='training'）。"""
    inserted = await db.insert(LEARNING_ATTEMPT, doc)
    doc["_id"] = (inserted.get("ids") or [None])[0]
    return doc


async def update_task(db, task: dict) -> dict:
    """更新任务（evaluate 后回写判定结果）。"""
    now = int(time.time() * 1000)
    changes = {
        "items": task.get("items"),
        "attempt_status": task.get("attempt_status"),
        "overall": task.get("overall"),
        "error_type": task.get("error_type"),
        "updated_at": now,
    }
    await db.update(
        collection=LEARNING_ATTEMPT,
        where={"_id": task["_id"]},
        data={"$set": changes},
        multi=False,
    )
    task["updated_at"] = now
    return task


# ---------------------------------------------------------------------------
# 弱项驱动选句（§9-4 / §3.2 Get WeakSkill + Select Sentence）
# ---------------------------------------------------------------------------


def sort_sentences_by_weakness(
    sentences: list[dict], skill_code: str
) -> list[dict]:
    """内存弱项排序：含该 skill 的 mastery 越低越靠前；无该 skill 状态排末尾（冷启动）。"""
    def key_fn(s: dict) -> tuple[Any, Any, Any]:
        states = s.get("_states") or []
        picked = None
        for st in states:
            if st.get("skill_code") == skill_code:
                picked = st
                break
        if picked is None:
            # 冷启动：无该 skill 状态 → 排最后（先学已见句），但保持可返回不阻断（§9-9）
            return (2, 0, s.get("order") or 0)
        mastery = int(picked.get("mastery_score") or 0)
        status = str(picked.get("status") or "")
        # 未学（not_started）视为最弱；其次低分；最后已掌握
        if status == "not_started":
            return (0, -1, s.get("order") or 0)
        return (1, mastery, s.get("order") or 0)

    return sorted(sentences, key=key_fn)
