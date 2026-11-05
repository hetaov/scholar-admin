"""语音描述出题·意图解析服务（POST /math/quiz/intent · 方案 §4.6/§7.1）

快/慢双路径衔接：
- **快路径（零 token）**：前端把整编图产物 `intake_meta`（含快路径 intent）随请求
  带回，服务层直接采纳（`intent_assemble` 已在图内完成）；
- **慢路径（LLM）**：`intake_meta` 缺失/fallback/无 intent → 以 `canonical_text`
  （优先）或 `asr_raw_text` / `description` 做单步 LLM 意图解析——这正是「语义整编
  对下游的直接增益」：解析不再面对噪声文本。

两路径共用教材 grounding：`knowledge_points` 名称 → `curriculum_node.title` 精确
匹配 → `grounded_kps[]`（出参，供意图卡展示「已定位教材考点」+ 出题主图 prompt）。

红线：R16 意图解析失败不抛异常（返回 fallback 标记）；审计 `quiz_intent` 成功/失败均写。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from services.audit import AUDIT_RESULT_FAILED, AUDIT_RESULT_SUCCESS, write_audit
from services.database import CURRICULUM_NODE_COLLECTION
from services.math.desc_quiz_graph import _call_chat
from services.providers.desc_intake_llm import (
    parse_slot_extract_response as _parse_llm_json,
)

try:
    from services.metrics.audit import AUDIT_ACTION_QUIZ_INTENT
except ImportError:  # pragma: no cover — 常量后补时自动切换（对齐 quiz.py 模式）
    AUDIT_ACTION_QUIZ_INTENT = "quiz_intent"

logger = logging.getLogger("scholar-admin.math.quiz_intent")

_MAX_GROUNDED_NAMES = 10  # 送 grounding 的考点名上限（防爆量）

_INTENT_SYSTEM = (
    "你是小学数学出题意图解析器。从出题需求文本解析结构化意图。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_INTENT_USER_TEMPLATE = """请解析下面的出题需求：

{text}

输出 JSON（未提及的字段必须为 null 或空数组，禁止编造）：
{{
  "knowledge_points": ["考点名称", …],   // 空数组表示未提及
  "question_count": 正整数或 null,
  "question_type_hint": "口算/竖式/应用题/填空/判断/选择题等，或 null",
  "difficulty_hint": "难度提示原话，或 null",
  "scenario": "情境词，或 null",
  "include_extended": true/false/null    // 用户是否要求拔高/引入新知识点
}}"""


def parse_intent_response(text: str) -> dict:
    """解析意图 JSON（容错）；`knowledge_points` 归一为 `list[str]`。"""
    parsed = _parse_llm_json(text)
    kps = parsed.get("knowledge_points")
    if isinstance(kps, str):
        parsed["knowledge_points"] = [kps] if kps.strip() else []
    elif not isinstance(kps, list):
        parsed["knowledge_points"] = []
    else:
        parsed["knowledge_points"] = [str(k).strip() for k in kps if str(k).strip()]
    return parsed


def _node_summary(node: dict) -> str:
    """节点 → 知识点 summary（对齐 quiz.py `_summary_from_node` 退化口径）。"""
    ai = node.get("ai_summary") or {}
    return str(ai.get("summary") or node.get("title") or "")


async def load_grounded_kps(db, names: list[str]) -> list[dict]:
    """考点名 → 教材节点（`curriculum_node.title` 精确匹配）→ grounded_kps。

    未命中的名称**不出现在** grounded_kps（前端标签显示「未定位」），
    但保留在 intent.knowledge_points 里，不丢弃用户意图。
    """
    grounded: list[dict] = []
    seen: set[str] = set()
    for name in (names or [])[:_MAX_GROUNDED_NAMES]:
        name = str(name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            res = await db.query(CURRICULUM_NODE_COLLECTION, where={"title": name}, limit=1)
        except Exception as e:  # noqa: BLE001 —— 图谱 IO 失败降级为未定位
            logger.warning(f"[quiz_intent] grounding 查询失败 name={name!r}: {e}")
            continue
        records = res.get("records") or []
        if not records:
            continue
        node = records[0]
        grounded.append(
            {
                "name": name,
                "node_code": str(node.get("code") or ""),
                "summary": _node_summary(node),
            }
        )
    return grounded


def _fast_path_intent(intake_meta: dict | None) -> dict | None:
    """整编图快路径产物 → intent（`intake_meta` 无 fallback 且带 intent 时）。"""
    if not isinstance(intake_meta, dict):
        return None
    if intake_meta.get("fallback"):
        return None
    intent = intake_meta.get("intent")
    return intent if isinstance(intent, dict) else None


async def _slow_path_intent(text: str) -> tuple[dict, str]:
    """LLM 单步意图解析（永不抛出）。返回 (intent, error)。"""
    try:
        content = await _call_chat(
            system=_INTENT_SYSTEM,
            user=_INTENT_USER_TEMPLATE.format(text=text),
        )
        intent = parse_intent_response(content)
        return intent, ""
    except Exception as e:  # noqa: BLE001 —— R16
        err = f"{type(e).__name__}: {e}"
        logger.warning(f"[quiz_intent] 意图解析失败走空意图: {err}")
        return {
            "knowledge_points": [],
            "question_count": None,
            "question_type_hint": None,
            "difficulty_hint": None,
            "scenario": None,
            "include_extended": None,
        }, err


async def parseQuizIntent(
    db,
    *,
    scholar_id: str,
    description: str = "",
    asr_raw_text: str = "",
    intake_meta: dict | None = None,
    actor: str = "",
) -> dict:
    """意图解析入口（永不抛异常；`scholar_id` 缺失由路由层 400 拦截）。

    Returns（方案 §7.1 出参口径）：
        {"intent", "grounded_kps", "intent_suggestions", "intent_missing",
         "intent_confidence", "canonical_text", "intake_meta", "fallback"}
    """
    t0 = time.time()
    meta = intake_meta if isinstance(intake_meta, dict) else {}
    text = (
        str(meta.get("canonical_text") or "")
        or asr_raw_text
        or description
    ).strip()

    intent = _fast_path_intent(intake_meta)
    fallback = False
    if intent is not None:
        source = "intake_fast_path"
    else:
        fallback = bool(meta.get("fallback")) or not text
        intent, err = await _slow_path_intent(text)
        source = "llm_slow_path" + (f"（降级: {err}）" if err else "")
        if err:
            fallback = True

    # 快路径产物缺 knowledge_points / question_count 时用慢路径补齐
    if intent is not None and (
        not intent.get("knowledge_points") or not intent.get("question_count")
    ):
        slow, _err = await _slow_path_intent(
            text or (asr_raw_text or description).strip()
        )
        if not intent.get("knowledge_points"):
            intent["knowledge_points"] = slow.get("knowledge_points") or []
        if not intent.get("question_count"):
            intent["question_count"] = slow.get("question_count")

    # 教材 grounding
    grounded = await load_grounded_kps(db, intent.get("knowledge_points") or [])

    meta_out = dict(meta)
    result = {
        "intent": intent,
        "grounded_kps": grounded,
        "intent_suggestions": [],  # 追问话术由前端依 intent_missing 生成（§4.5）
        "intent_missing": list(meta.get("missing_required") or []),
        "intent_confidence": float(meta.get("slot_confidence") or 0.0),
        "canonical_text": str(meta.get("canonical_text") or ""),
        "intake_meta": meta_out,
        "fallback": fallback,
    }

    # 审计（成功/失败均写；不含原文，防 PII 入审计日志）
    try:
        await write_audit(
            db,
            action=AUDIT_ACTION_QUIZ_INTENT,
            object_ref=scholar_id,
            actor=actor,
            result=AUDIT_RESULT_SUCCESS if not fallback else AUDIT_RESULT_FAILED,
            context={
                "scholar_id": scholar_id,
                "source": source,
                "has_intake_meta": bool(meta),
                "kp_count": len(intent.get("knowledge_points") or []),
                "grounded_count": len(grounded),
                "fallback": fallback,
                "elapsed_ms": int((time.time() - t0) * 1000),
            },
        )
    except Exception as e:  # noqa: BLE001 —— 审计失败不影响主流程
        logger.warning(f"[quiz_intent] 审计写入失败: {e}")
    return result
