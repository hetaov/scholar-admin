"""P0 Conversation MVP 接口（契约 api-contract §3.9）

POST /conversation/scenario — 创建会话（前置评估建议 + 难度档位，建议层不硬阻断）
POST /conversation/turn     — 提交一轮（AI 回复 + 状态机推进 + 每轮 eval_verdict）
GET  /conversation/history  — 会话历史与小结

原则：
- 前置评估为建议层（§0.2/§6.2/§9-5）：无历史回退先验默认（difficulty=1 + 标准引导序列），
  gate_suggestion 仅建议不阻断（附录 B-5 不配初始诊断）。
- 每轮必返 eval_verdict（§9-2），低置信（< EVAL_CONFIDENCE_THRESHOLD）不回写 SkillState。
- 会话结束（history 触发）生成小结 + 门控 SkillState 更新 + ReviewSchedule（§9-3）。
- 降级路径（附录 B-1）：Hint → Rephrase → 降档；P0 落这三分支，终止/转训练建议 P1 补全。

LLM 依赖：AI 回复生成复用 services.dialogue.call_volcano（测试中 monkeypatch 注入）。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from config import CONVERSATION_GRAPH_ENABLED
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.dialogue import call_volcano
from services.cold_start import cold_start_flag, cold_start_prior
from services.pre_assessment import pre_assess
from services.evaluation_engine import evaluate_text
from services.conversation_graph import run_turn_graph, start_scenario_graph
from services.models_conversation import (
    DEFAULT_DIFFICULTY,
    LOW_CONFIDENCE_THRESHOLD,
    SESSION_STAGE_ACTIVE,
    SESSION_STAGE_ENDED,
    append_turn,
    build_session_doc,
    build_turn_doc,
    create_session,
    end_session_with_summary,
    get_session,
    list_turns,
    next_turn_stage,
    update_session_stage,
)
from services.models_content import get_sentences_by_ids

logger = logging.getLogger("scholar-admin.routes.conversation")
router = APIRouter(tags=["conversation"])


class ConvResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


class ScenarioRequest(BaseModel):
    scholar_id: Optional[str] = Field(None, description="必填（业务层校验，缺参返回 INVALID_INPUT）")
    scenario: Optional[str] = Field(None, description="场景名，如 daily_conversation")
    topic: Optional[str] = Field(None, description="话题，如 travel")
    sentence_id: Optional[str] = Field(None, description="绑定目标句（有则回写该句 SkillState）")


class TurnRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="必填（业务层校验）")
    utterance: Optional[str] = Field(None, description="用户本轮输出（英文）")
    mode: Optional[str] = Field("text", description="text | voice")


# ---------------------------------------------------------------------------
# AI 回复生成（可注入，测试 monkeypatch services.routes_conversation._generate_reply）
# ---------------------------------------------------------------------------

_REPLY_SYSTEM_PROMPT = (
    "你是英语对话教练。围绕当前话题与目标句，用自然英文引导学习者输出。\n"
    "输出必须是 JSON 对象，不要任何解释，格式：\n"
    '{"reply": "对用户输出的自然回应（含鼓励或纠正提示，英文）", '
    '"next_target": "达成后引导的下一句英文目标句（可为空字符串表示暂不换句）"}'
)


def _generate_reply(
    *,
    topic: str,
    target: str,
    utterance: str,
    stage: str,
    difficulty: int,
) -> dict:
    """调用 LLM 生成 AI 回复（{reply, next_target}）；失败回落规则兜底。

    - answer / hint / rephrase / downgrade 均由 stage 描述引导方向。
    """
    stage_hint = {
        "opening": "会话开场：AI 扮演会话中的对方角色（如客户/朋友），用自然英文开启一段围绕话题的练习对话："
                   "简要交代情景并点明学习者立场，以开放问题/邀请结尾引导学习者开口；"
                   "不要直接写出目标句全文，不要使用中文",
        "answer": "用户输出正确或接近，给出自然回应；若 target 为空则生成一句新目标句",
        "hint": "用户卡住：给出目标句的首词/关键短语提示，帮助他继续",
        "rephrase": "用户仍不会：用更简单的英文重述目标句（同义替换）",
        "downgrade": "降档：换一句更简单、更短的目标句，降低难度",
    }[stage]
    prompt = (
        f"话题：{topic}\n当前目标句：{target or '(无，请生成)'}\n"
        f"难度档位：{difficulty}\n用户输出：{utterance}\n"
        f"本轮任务（{stage}）：{stage_hint}\n请给出 JSON："
    )
    try:
        content = call_volcano(prompt, system_prompt=_REPLY_SYSTEM_PROMPT, temperature=0.4)
        import json as json_lib

        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        parsed = json_lib.loads(text)
        return {
            "reply": str(parsed.get("reply") or "").strip(),
            "next_target": str(parsed.get("next_target") or "").strip(),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("[conversation] AI 回复生成失败，回落规则兜底: %s", e)
        return _fallback_reply(topic, target, utterance, stage)


def _fallback_reply(*, topic: str, target: str, utterance: str, stage: str) -> dict:
    """规则兜底回复（LLM 不可用时保证接口可用，P0 验收可测试）。"""
    if stage == "opening":
        topic_label = topic or "this topic"
        return {
            "reply": (
                f"Thanks for joining this conversation! Today, let's practise a short "
                f"dialogue about {topic_label}. I'll play the other side — could you begin "
                f"by telling me your first thoughts in a complete English sentence?"
            ),
            "next_target": "",
        }
    if stage == "hint":
        words = target.split()[:2]
        return {"reply": f"提示：以 {(' '.join(words))} 开头试试？", "next_target": ""}
    if stage == "rephrase":
        return {"reply": f"换个说法：{target}（意思是：说得更简单些）", "next_target": ""}
    if stage == "downgrade":
        return {"reply": "我们换一个更简单的句子，慢慢来。", "next_target": "It is a book."}
    if target:
        return {"reply": f"不错，继续。试着说：{target}", "next_target": ""}
    return {"reply": "很好！我们进入下一句。", "next_target": "I like apples."}


# ---------------------------------------------------------------------------
# 轮次执行：L1 轻量状态机 / S4.2 L2 LangGraph 图路径（HTTP 契约一致，可回退）
# ---------------------------------------------------------------------------


async def _l1_turn(db: CloudBaseNoSQLClient, session: dict, utterance: str) -> ConvResponse:
    """L1 轻量状态机路径（CONVERSATION_GRAPH_ENABLED=0 或图不可用时的回退）。"""
    session_id = session["session_id"]
    topic = session.get("topic") or "daily conversation"
    difficulty = int(session.get("difficulty") or DEFAULT_DIFFICULTY)
    current_target = str(session.get("current_target") or "").strip()

    # 目标句：绑定句优先（evidence 快照），否则用会话当前引导句
    original_text = current_target
    sentence_id = None
    if session.get("sentence_ids"):
        # 取首个绑定句作为参考（P0 简化：单句会话）
        sid = session["sentence_ids"][0]
        sentences = await get_sentences_by_ids(db, [sid])
        if sentences:
            sentence_id = sid
            original_text = str(sentences[0].get("text") or original_text or "")

    # 1. 评估用户输出 vs 目标表达
    verdict = evaluate_text(original_text, utterance)
    confidence = float(verdict.get("confidence") or 0.0)
    meaningful = bool(verdict.get("meaningful"))
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    # 2. 状态机推进（降级路径：hint → rephrase → 降档，附录 B-1）
    # consecutive_failures 传「本轮失败后的累计失败次数」：达意成功记 0，否则 +1
    consecutive_failures = int(session.get("consecutive_failures") or 0)
    failed_this_turn = not (meaningful and not low_confidence)
    failures_after_this = consecutive_failures + (1 if failed_this_turn else 0)
    step = next_turn_stage(
        consecutive_failures=failures_after_this,
        difficulty=difficulty,
        meaningful=meaningful,
        low_confidence=low_confidence,
    )
    new_failures = 0 if step["reset_failures"] else failures_after_this

    # 3. AI 回复生成（按 stage 引导；失败时由规则兜底）
    ai = _generate_reply(
        topic=topic,
        target=original_text,
        utterance=utterance,
        stage=step["stage"],
        difficulty=step["difficulty"],
    )
    next_target = ai.get("next_target") or ""
    if not next_target and original_text:
        next_target = original_text  # 保持当前目标句（未换句）

    # 4. 写轮次（证据快照不可变）+ 更新会话状态
    now = int(time.time() * 1000)
    turn = build_turn_doc(
        session_id=session_id,
        sentence_id=sentence_id,
        original_text=original_text,
        translation=session.get("translation"),
        utterance=utterance,
        reply=ai.get("reply") or "",
        stage=step["stage"],
        hint=step.get("hint"),
        rephrased=step.get("rephrased"),
        suggestion=step.get("suggestion"),
        now=now,
    )
    turn["eval_verdict"] = verdict  # 内联供小结/历史使用（§9-2）
    await append_turn(db, turn)

    await update_session_stage(
        db, session, stage=SESSION_STAGE_ACTIVE, difficulty=step["difficulty"]
    )
    await db.update(
        collection="conversation_session",
        where={"_id": session["_id"]},
        data={
            "$set": {
                "consecutive_failures": new_failures,
                "current_target": next_target,
                "translation": session.get("translation"),
                "updated_at": now,
            }
        },
        multi=False,
    )

    logger.info(
        "[conversation/turn][L1] session_id=%s stage=%s meaningful=%s confidence=%.2f",
        session_id, step["stage"], meaningful, confidence,
    )
    return ConvResponse(
        success=True,
        data={
            "turn_id": turn["turn_id"],
            "reply": ai.get("reply") or "",
            "state": {
                "stage": step["stage"],
                "hint": step.get("hint"),
                "rephrased": step.get("rephrased"),
                "suggestion": step.get("suggestion"),
                "difficulty": step["difficulty"],
            },
            "eval_verdict": verdict,
        },
    )


async def _l2_turn(db: CloudBaseNoSQLClient, session: dict, utterance: str) -> ConvResponse:
    """S4.2 L2：LangGraph 图路径（checkpoint 恢复 → 评估 → 推进 → 回复）。"""
    result = await run_turn_graph(
        db, session, utterance, evaluator=evaluate_text, reply_generator=_generate_reply
    )
    gs = result["graph_state"]
    reply = result["reply"]
    verdict = result["verdict"]
    turn_stage = result["turn_stage"]

    current_target = str(gs.get("current_target") or "")
    sentence_id = None
    original_text = current_target
    if session.get("sentence_ids"):
        sid = session["sentence_ids"][0]
        sentences = await get_sentences_by_ids(db, [sid])
        if sentences:
            sentence_id = sid
            original_text = str(sentences[0].get("text") or original_text or "")

    now = int(time.time() * 1000)
    turn = build_turn_doc(
        session_id=session["session_id"],
        sentence_id=sentence_id,
        original_text=original_text,
        translation=session.get("translation"),
        utterance=utterance,
        reply=reply,
        stage=turn_stage,
        hint=gs.get("hint"),
        rephrased=gs.get("rephrased"),
        suggestion=gs.get("suggestion"),
        now=now,
    )
    turn["eval_verdict"] = verdict  # 内联供小结/历史使用（§9-2）
    await append_turn(db, turn)

    await update_session_stage(
        db,
        session,
        stage=SESSION_STAGE_ACTIVE,
        difficulty=int(gs.get("difficulty") or DEFAULT_DIFFICULTY),
    )
    await db.update(
        collection="conversation_session",
        where={"_id": session["_id"]},
        data={
            "$set": {
                "consecutive_failures": int(gs.get("consecutive_failures") or 0),
                "current_target": current_target,
                "turn_index": int(gs.get("turn_index") or 0),
                "graph_state": gs,
                "checkpoint_id": result["checkpoint_id"],
                "translation": session.get("translation"),
                "updated_at": now,
            }
        },
        multi=False,
    )

    logger.info(
        "[conversation/turn][L2] session_id=%s stage=%s turn_index=%s meaningful=%s",
        session["session_id"], turn_stage, gs.get("turn_index"),
        bool(verdict.get("meaningful")),
    )
    return ConvResponse(
        success=True,
        data={
            "turn_id": turn["turn_id"],
            "reply": reply,
            "state": {
                "stage": turn_stage,
                "hint": gs.get("hint"),
                "rephrased": gs.get("rephrased"),
                "suggestion": gs.get("suggestion"),
                "difficulty": gs.get("difficulty"),
            },
            "eval_verdict": verdict,
        },
    )


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


@router.post("/conversation/scenario", response_model=ConvResponse)
async def conversation_scenario(
    data: ScenarioRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> ConvResponse:
    """创建会话：返回 session_id + difficulty + 前置评估建议（建议层，不阻断）。"""
    scholar_id = str(data.scholar_id or "").strip()
    if not scholar_id:
        return ConvResponse(success=False, code="INVALID_INPUT", message="缺少参数 scholar_id")

    topic = str(data.topic or "").strip() or "daily conversation"
    scenario = str(data.scenario or "").strip() or "free_talk"
    sentence_ids: list[str] = []
    bound_sentence_text = ""  # 绑定目标句文本：作为开场白牵引的目标句上下文（P0 单句）
    if data.sentence_id:
        sentences = await get_sentences_by_ids(db, [data.sentence_id])
        if sentences:
            sentence_ids = [data.sentence_id]
            bound_sentence_text = str(sentences[0].get("text") or "")

    # 前置评估（S3.2 §6.2/§9-5）：基于 skill_state 聚合生成 Gate 建议 + 难度档位 +
    # Activity 推荐（建议层不阻断）；无历史 → 冷启动回退（§5.6），不报错不拒绝
    assessment = await pre_assess(db, scholar_id=scholar_id, sentence_id=data.sentence_id)
    is_cold = cold_start_flag(has_history=assessment["has_history"])  # §5.6.5 标记
    prior = cold_start_prior(difficulty=assessment["difficulty"]) if is_cold else None

    # 难度档位（§6.2）写入会话文档，对 ConversationGraph 生效；无历史回退先验默认 1
    doc = build_session_doc(
        scholar_id=scholar_id,
        scenario=scenario,
        topic=topic,
        difficulty=assessment["difficulty"],
        sentence_ids=sentence_ids,
        cold_start=is_cold,  # S3.1 P1：记录冷启动判定（§9-2 降权豁免）
    )
    session = await create_session(db, doc)

    # S4.2 L2：图初始化（落 checkpoint 支持断点续聊）；失败不阻断 → 回退 L1 轻量状态机
    graph_init: dict | None = None
    if CONVERSATION_GRAPH_ENABLED:
        try:
            graph_init = await start_scenario_graph(
                db, session, evaluator=evaluate_text, reply_generator=_generate_reply
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("[conversation/scenario] 图初始化失败，回退 L1: %s", e)
    if graph_init is not None:
        gs = graph_init["graph_state"]
        now = int(time.time() * 1000)
        await db.update(
            collection="conversation_session",
            where={"_id": session["_id"]},
            data={
                "$set": {
                    "graph_state": gs,
                    "checkpoint_id": graph_init["checkpoint_id"],
                    "current_target": str(gs.get("current_target") or ""),
                    "updated_at": now,
                }
            },
            multi=False,
        )
        session = {**session, "graph_state": gs, "checkpoint_id": graph_init["checkpoint_id"]}

    pre_assessment = {
        "gate_suggestion": assessment["gate_suggestion"],  # 建议层：不硬阻断（§0.2/§9-5）
        "activity_recommendation": assessment["activity_recommendation"],
    }
    logger.info(
        "[conversation/scenario] scholar_id=%s session_id=%s topic=%s cold_start=%s "
        "gate=%s difficulty=%s",
        scholar_id, session["session_id"], topic, is_cold,
        assessment["gate_suggestion"], assessment["difficulty"],
    )
    data: dict = {
        "session_id": session["session_id"],
        "difficulty": session["difficulty"],
        "pre_assessment": pre_assessment,
        "cold_start": is_cold,  # §5.6.5 标记
    }
    if prior is not None:
        data["prior_defaults"] = prior  # 先验默认（仅冷启动返回，前端可展示引导）

    # 沉浸式开场（扩展字段，向后兼容）：生成引导学习者开口的英文开场白 + 会话初始状态，
    # 供会话页直接渲染首条 AI 气泡。LLM 失败回落规则兜底（_generate_reply 内部处理），不阻断创建。
    opening = _generate_reply(
        topic=topic,
        target=bound_sentence_text,
        utterance="",
        stage="opening",
        difficulty=int(session["difficulty"] or assessment["difficulty"]),
    )
    data["reply"] = opening.get("reply") or ""
    data["state"] = {
        "stage": "opening",
        "hint": None,
        "rephrased": None,
        "suggestion": None,
        "difficulty": int(session["difficulty"] or assessment["difficulty"]),
    }
    return ConvResponse(success=True, data=data)


@router.post("/conversation/turn", response_model=ConvResponse)
async def conversation_turn(
    data: TurnRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> ConvResponse:
    """提交一轮：AI 回复 + 状态机推进 + 每轮 eval_verdict（含降级路径分支）。"""
    session_id = str(data.session_id or "").strip()
    utterance = str(data.utterance or "").strip()
    if not session_id or not utterance:
        return ConvResponse(
            success=False, code="INVALID_INPUT", message="缺少参数 session_id / utterance"
        )

    session = await get_session(db, session_id)
    if session is None:
        return ConvResponse(success=False, code="NOT_FOUND", message="会话不存在")
    if session.get("stage") != SESSION_STAGE_ACTIVE:
        return ConvResponse(success=False, code="CONFLICT", message="会话已结束")

    # S4.2 L2：图路径优先（checkpoint 恢复断点续聊）；图不可用/执行失败 → 回退 L1 轻量状态机
    if CONVERSATION_GRAPH_ENABLED and session.get("graph_state"):
        try:
            return await _l2_turn(db, session, utterance)
        except Exception as e:  # noqa: BLE001
            logger.warning("[conversation/turn] 图执行失败，回退 L1: %s", e)
    return await _l1_turn(db, session, utterance)


@router.get("/conversation/history", response_model=ConvResponse)
async def conversation_history(
    session_id: str = Query(..., description="必填"),
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> ConvResponse:
    """会话历史：轮次列表 + 会话小结；已结束会话在首次查询时生成小结并落库。"""
    session_id = str(session_id or "").strip()
    if not session_id:
        return ConvResponse(success=False, code="INVALID_INPUT", message="缺少参数 session_id")

    session = await get_session(db, session_id)
    if session is None:
        return ConvResponse(success=False, code="NOT_FOUND", message="会话不存在")

    turns = await list_turns(db, session_id)
    for turn in turns:
        verdict = turn.pop("eval_verdict", None)
        if verdict is not None:
            turn["eval_verdict"] = verdict  # 原样保留（历史视图内联）

    # 会话结束：首次查询生成小结 + 门控 SkillState 更新 + ReviewSchedule（§9-3）
    if session.get("stage") != SESSION_STAGE_ENDED and turns:
        session = await end_session_with_summary(
            db,
            session,
            turns,
            {},
            is_cold=bool(session.get("cold_start")),  # 冷启动会话豁免降权（§9-2）
        )
    summary = session.get("summary")

    return ConvResponse(
        success=True,
        data={
            "session": {
                "session_id": session["session_id"],
                "scenario": session.get("scenario"),
                "topic": session.get("topic"),
                "difficulty": session.get("difficulty"),
                "started_at": session.get("started_at"),
                "ended_at": session.get("ended_at"),
                "summary": summary,
            },
            "turns": turns,
        },
    )
