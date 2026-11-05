"""数学语音出题「语义整编图」（方案：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §4）

数学自有逻辑，**不 import 英语域 asr_repair_graph**（§2.2 范式不匹配）：

  START → sanitize（零 token：整句重复折叠）
            ↓
        slot_extract（唯一 LLM 节点：原文 → 槽位 JSON + 置信度 + 证据）
            │ 失败/超时/非法 JSON ──→ fallback（原文直进既有单步意图解析）→ END
            ↓
        slot_validate（零 token：S1~S4 + 教材 grounding + 上下文补位）
            ↓
        rebuild（零 token：槽位 → 规范描述回显 canonical_text）→ END

下游衔接（§4.6 intent_assemble）：槽位完整且置信达标 → 纯函数 slots→intent（快路径）；
否则由调用方以 canonical_text 走既有单步意图解析（慢路径）。

红线：R16 节点级永不抛异常（LLM 失败走 fallback）；R18 `DESC_INTAKE_ENABLED=0`
时 run_intake 返回 None（调用方出参不加任何字段）；本图**不写库**（溯源字段由
调用方随意图卡流程落库）。
"""
from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import DESC_INTAKE_SLOT_CONFIDENCE_MIN
from services.math.desc_validators import (
    SLOT_DIFFICULTY,
    SLOT_EXTENDED,
    SLOT_GRADE,
    SLOT_KPS,
    SLOT_QTYPE,
    SLOT_SCENARIO,
    SLOT_SEMESTER,
    SLOT_COUNT,
    SlotValidation,
    rebuild_canonical_text,
    sanitize_text,
    slot_validate,
)
from services.providers.desc_intake_llm import call_slot_extract

logger = logging.getLogger("scholar-admin.math.desc_intake_graph")

# 图版本号（落库溯源）
_GRAPH_VERSION = 1


class DescIntakeState(TypedDict, total=False):
    """语义整编图状态"""

    # 输入
    raw_text: str
    kp_candidates: list          # 教材候选节点 [{name, grade, node_code, summary}]
    context: dict                # 页面上下文 {grade, semester}
    confidence_min: float
    # 各节点产物
    raw_text_sanitized: str
    sanitize_meta: dict
    slots: dict                  # 槽位表 {槽名: {value, confidence, evidence, optional}}
    intake_meta: dict
    canonical_text: str
    # fallback / 输出
    fallback: bool
    fallback_reason: str
    # intent_assemble 产物（§4.6）
    intent: dict | None
    intent_source: str           # slots | llm | fallback


# ---------------------------------------------------------------------------
# 图节点
# ---------------------------------------------------------------------------


def _node_sanitize(state: DescIntakeState) -> dict:
    """节点 1：整句重复折叠 + 空白/标点归一（纯函数，零 token）。"""
    result = sanitize_text(state.get("raw_text") or "")
    return {
        "raw_text_sanitized": result.text,
        "sanitize_meta": {
            "collapsed_sentence_repeats": result.collapsed_sentence_repeats,
            "collapsed_run_repeats": result.collapsed_run_repeats,
        },
    }


async def _node_slot_extract(state: DescIntakeState) -> dict:
    """节点 2：LLM 槽位抽取（唯一 LLM 节点）。失败置 fallback 标记（不抛出）。"""
    out = await call_slot_extract(
        state.get("raw_text_sanitized") or "",
        kp_candidates=state.get("kp_candidates"),
        context=state.get("context"),
    )
    if out.get("slots") is None:
        return {"fallback": True, "fallback_reason": str(out.get("error") or "unknown")}
    return {"slots": out["slots"]}


def _route_after_extract(state: DescIntakeState) -> str:
    if state.get("fallback"):
        return "fallback"
    return "slot_validate"


def _node_slot_validate(state: DescIntakeState) -> dict:
    """节点 3：S1~S4 校验 + 教材 grounding + 上下文补位（纯函数，零 token）。"""
    v = slot_validate(
        state.get("slots"),
        kp_candidates=state.get("kp_candidates"),
        context=state.get("context"),
        confidence_min=state.get("confidence_min", DESC_INTAKE_SLOT_CONFIDENCE_MIN),
    )
    return {
        "slots": _slots_to_json(v),
        "intake_meta": v.to_meta(),
    }


def _slots_to_json(v: SlotValidation) -> dict:
    """SlotValidation.slots（Slot 对象表）→ 可序列化槽位 JSON。"""
    return {
        name: {
            "value": s.value,
            "confidence": s.confidence,
            "evidence": s.evidence,
            "optional": s.optional,
        }
        for name, s in v.slots.items()
    }


def _node_rebuild(state: DescIntakeState) -> dict:
    """节点 4：规范描述回显（纯模板，零 token，§4.4）。"""
    return {"canonical_text": rebuild_canonical_text(state.get("slots") or {})}


def _node_fallback(state: DescIntakeState) -> dict:
    """fallback 节点：不产出槽位（调用方用原文走既有单步意图解析，§4.1）。"""
    logger.warning(
        f"[desc_intake_graph] slot_extract 失败走 fallback: {state.get('fallback_reason')}"
    )
    return {
        "fallback": True,
        "intake_meta": {
            "slot_confidence": 0.0,
            "completeness": "fallback",
            "missing_required": [],
            "issues": [],
            "not_grounded_kps": [],
            "low_confidence_slots": [],
            "dropped_slots": [],
            "context_filled": [],
        },
    }


# ---------------------------------------------------------------------------
# intent_assemble（§4.6：快/慢双路径）
# ---------------------------------------------------------------------------

# 快路径采纳所需的非 optional 槽位（缺任一 → 慢路径）
_FAST_PATH_REQUIRED = (SLOT_KPS, SLOT_COUNT)


def _slot_plain(slots: dict, name: str) -> dict:
    item = slots.get(name)
    if item is None:
        return {"value": None, "confidence": 0.0, "evidence": "", "optional": False}
    if isinstance(item, dict):
        return item
    return {"value": item, "confidence": 0.0, "evidence": "", "optional": False}


def intent_assemble(
    slots: dict, intake_meta: dict, canonical_text: str
) -> tuple[dict | None, str]:
    """槽位 → 出题意图（纯函数）。

    Returns:
        (intent, source)：
        - 快路径（槽位完整、无低置信必填槽、无 S4 丢弃、kp 已 grounding）：
          纯函数映射，source="slots"；
        - 其余：返回 (None, "llm")，由调用方以 canonical_text 走既有单步意图解析。
    """
    if intake_meta.get("completeness") != "complete":
        return None, "llm"
    if intake_meta.get("low_confidence_slots"):
        # 低置信槽位待意图卡人工确认，快路径不采纳
        if any(n in _FAST_PATH_REQUIRED for n in intake_meta["low_confidence_slots"]):
            return None, "llm"
    if intake_meta.get("issues"):
        return None, "llm"

    kp_slot = _slot_plain(slots, SLOT_KPS)
    count_slot = _slot_plain(slots, SLOT_COUNT)
    if not kp_slot.get("value") or count_slot.get("value") is None:
        return None, "llm"

    grade = _slot_plain(slots, SLOT_GRADE).get("value")
    semester = _slot_plain(slots, SLOT_SEMESTER).get("value")
    intent: dict[str, Any] = {
        "knowledge_points": list(kp_slot["value"]),
        "question_count": int(count_slot["value"]),
    }
    if grade is not None:
        intent["grade"] = int(grade)
    if semester:
        intent["semester"] = semester
    qtype = _slot_plain(slots, SLOT_QTYPE).get("value")
    if qtype:
        intent["question_type"] = qtype
    difficulty = _slot_plain(slots, SLOT_DIFFICULTY).get("value")
    if difficulty:
        intent["difficulty_hint"] = difficulty
    scenario = _slot_plain(slots, SLOT_SCENARIO).get("value")
    if scenario:
        intent["scenario"] = scenario
    extended = _slot_plain(slots, SLOT_EXTENDED).get("value")
    if extended is not None:
        intent["include_extended"] = bool(extended)
    return intent, "slots"


# ---------------------------------------------------------------------------
# 图构建与编译（对齐 summary_graph：模块级编译实例缓存）
# ---------------------------------------------------------------------------


def _build_intake_graph() -> Any:
    """构建并编译语义整编图（无 db 依赖，无需 partial 绑定）。"""
    graph = StateGraph(DescIntakeState)
    graph.add_node("sanitize", _node_sanitize)
    graph.add_node("slot_extract", _node_slot_extract)
    graph.add_node("slot_validate", _node_slot_validate)
    graph.add_node("rebuild", _node_rebuild)
    graph.add_node("fallback", _node_fallback)

    graph.add_edge(START, "sanitize")
    graph.add_edge("sanitize", "slot_extract")
    graph.add_conditional_edges(
        "slot_extract",
        _route_after_extract,
        {"fallback": "fallback", "slot_validate": "slot_validate"},
    )
    graph.add_edge("slot_validate", "rebuild")
    graph.add_edge("rebuild", END)
    graph.add_edge("fallback", END)
    return graph.compile()


_compiled_graph: Any = None


def _get_compiled_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_intake_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# 运行入口（T12）
# ---------------------------------------------------------------------------


def _fallback_result(raw_text: str, reason: str) -> dict:
    """LLM 失败时的兜底产物：调用方拿原文走既有单步意图解析（§4.1）。"""
    sanitized = sanitize_text(raw_text or "").text
    return {
        "graph_version": _GRAPH_VERSION,
        "raw_text_sanitized": sanitized,
        "slots": {},
        "intake_meta": {
            "slot_confidence": 0.0,
            "completeness": "fallback",
            "missing_required": [],
            "issues": [],
            "not_grounded_kps": [],
            "low_confidence_slots": [],
            "dropped_slots": [],
            "context_filled": [],
        },
        "canonical_text": "",
        "intent": None,
        "intent_source": "fallback",
        "fallback": True,
        "fallback_reason": reason,
    }


async def run_intake(
    raw_text: str,
    *,
    kp_candidates: list[dict] | None = None,
    context: dict | None = None,
    confidence_min: float | None = None,
    enabled: bool | None = None,
) -> dict | None:
    """语义整编运行入口（永不抛异常）。

    Args:
        raw_text: ASR 原文（sanitize 在图内完成）。
        kp_candidates: 教材候选节点（grounding 白名单）。
        context: 页面上下文 {grade, semester}（textbook_id 已知时提供）。
        confidence_min: 覆盖 DESC_INTAKE_SLOT_CONFIDENCE_MIN。
        enabled: 覆盖 DESC_INTAKE_ENABLED（测试注入用）。

    Returns:
        - `DESC_INTAKE_ENABLED=0` → **None**（R18：调用方出参不加任何字段，
          按既有单步意图解析处理）；
        - 正常/降级 → 结果 dict（含 fallback 标记与 fallback_reason）。
    """
    import config as _cfg

    if enabled is None:
        enabled = _cfg.DESC_INTAKE_ENABLED
    if not enabled:
        return None

    try:
        graph = _get_compiled_graph()
        state: DescIntakeState = {
            "raw_text": raw_text or "",
            "kp_candidates": kp_candidates or [],
            "context": context or {},
            "confidence_min": (
                confidence_min
                if confidence_min is not None
                else _cfg.DESC_INTAKE_SLOT_CONFIDENCE_MIN
            ),
        }
        final: DescIntakeState = await graph.ainvoke(state)

        if final.get("fallback"):
            result = _fallback_result(
                final.get("raw_text") or "", str(final.get("fallback_reason") or "unknown")
            )
        else:
            result = {
                "graph_version": _GRAPH_VERSION,
                "raw_text_sanitized": final.get("raw_text_sanitized") or "",
                "slots": final.get("slots") or {},
                "intake_meta": final.get("intake_meta") or {},
                "canonical_text": final.get("canonical_text") or "",
                "intent": None,
                "intent_source": "llm",
                "fallback": False,
                "fallback_reason": "",
            }

        # intent_assemble（§4.6）：仅非 fallback 且槽位可用时尝试快路径
        if not result["fallback"]:
            intent, source = intent_assemble(
                result["slots"], result["intake_meta"], result["canonical_text"]
            )
            result["intent"] = intent
            result["intent_source"] = source
        return result
    except Exception as e:  # noqa: BLE001 —— R16：永不抛出
        err = f"{type(e).__name__}: {e}"
        logger.warning(f"[desc_intake_graph] run_intake 异常走 fallback: {err}")
        return _fallback_result(raw_text, err)
