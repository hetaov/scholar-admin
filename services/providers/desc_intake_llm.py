"""语义整编图 slot_extract 节点的 LLM 调用（方案：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §4）

唯一 LLM 节点：ASR 原文（sanitize 后）→ 槽位 JSON + 逐槽置信度 + 原话证据。
范式约定（§4.0）：**不修文本**——LLM 在语义层理解，同音错词/重复/中文数字
噪声由槽位抽取吸收；value 未提及必须为 null，禁止编造默认值。

失败语义（对齐 summary_graph._call_hunyuan_evaluate 的容错风格，红线 R16）：
任何异常（网络/超时/非法 JSON）→ 返回 {"slots": None, "error": "<原因>"}，
**永不抛出**；上层据此走 fallback（原文直进既有单步意图解析，§4.1）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import (
    DESC_INTAKE_LLM_MODEL,
    DESC_INTAKE_LLM_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)

logger = logging.getLogger("scholar-admin.math.desc_intake_llm")

# 送入 prompt 的教材候选考点名上限（控制 token；grounding 兜底在 slot_validate）
MAX_KP_CANDIDATE_NAMES = 50

_SLOT_EXTRACT_SYSTEM_PROMPT = (
    "你是小学数学出题需求解析器。用户会给你一段语音转写的口语化出题需求，"
    "请抽取结构化槽位。只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_SLOT_EXTRACT_USER_TEMPLATE = """请从下面的出题需求中抽取槽位。

出题需求（语音转写，可能有口语噪声）：
{raw_text}

{context_block}教材候选考点（考点槽位优先从中选名称最接近的一个或多个；都不像则输出原话词）：
{kp_candidates}

槽位定义（每槽输出 {{\"value\": …, \"confidence\": 0~1, \"evidence\": \"原话依据\"}}）：
- grade：年级，1~6 整数
- semester：学期，只能是 \"上\" 或 \"下\"
- knowledge_points：考点名称数组（可多个）
- question_count：出题数量，正整数
- question_type：题型（口算/竖式/应用题/填空/判断/选择题等）
- difficulty_hint：难度提示（原话有才填，如\"难一点\"）
- scenario：情境词（原话有才填，如\"分蛋糕\"）
- extended_request：是否要求拔高/引入新知识点，true/false，原话未提及填 null

硬性规则：
1. 原话未提及的槽位 value 必须为 null，**禁止编造默认值**（宁可 null 不可猜）；
2. 同音错字按语义理解（如\"家减法\"→加减法），evidence 保留**原话原文**；
3. 中文数字换算为阿拉伯数字（\"五道\"→5）；
4. 原话有重复没关系，同一槽位只取一次值；
5. evidence 为 value 对应的原话片段，纯推断（原话没有对应词）时填 \"\"。

输出 JSON：
{{\"grade\": {{…}}, \"semester\": {{…}}, \"knowledge_points\": {{…}}, \"question_count\": {{…}}, \"question_type\": {{…}}, \"difficulty_hint\": {{…}}, \"scenario\": {{…}}, \"extended_request\": {{…}}}}"""


def build_slot_extract_messages(
    raw_text: str,
    *,
    kp_candidates: list[dict] | None = None,
    context: dict | None = None,
) -> list[dict]:
    """构造 slot_extract 的 messages（纯函数，可独立单测）。"""
    names = [
        str(c.get("name") or "")
        for c in (kp_candidates or [])[:MAX_KP_CANDIDATE_NAMES]
        if c.get("name")
    ]
    ctx = context or {}
    ctx_items = [
        f"- {k}: {v}" for k, v in ctx.items() if v is not None
    ]
    context_block = (
        "页面已知上下文（可直接作为对应槽位依据，无需重复抽取）：\n"
        + "\n".join(ctx_items)
        + "\n\n"
        if ctx_items
        else ""
    )
    return [
        {"role": "system", "content": _SLOT_EXTRACT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _SLOT_EXTRACT_USER_TEMPLATE.format(
                raw_text=raw_text,
                context_block=context_block,
                kp_candidates="\n".join(f"- {n}" for n in names) if names else "（无候选，输出原话词）",
            ),
        },
    ]


def parse_slot_extract_response(text: str) -> dict[str, Any]:
    """解析 LLM 返回的槽位 JSON（容错：markdown fence / 首尾花括号提取）。

    解析失败抛 ValueError（由 call_slot_extract 捕获降级，不外泄）。
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"slot_extract 输出非 JSON 对象: {type(parsed).__name__}")
    return parsed


async def call_slot_extract(
    raw_text: str,
    *,
    kp_candidates: list[dict] | None = None,
    context: dict | None = None,
) -> dict[str, Any]:
    """调 LLM 抽取槽位（永不抛出；失败降级返回 error 字段）。

    Returns:
        {"slots": dict | None, "error": str}：
        - 成功：{"slots": {槽位名: {value, confidence, evidence}}, "error": None}
        - 失败：{"slots": None, "error": "<异常类型>: <信息>"}
    """
    messages = build_slot_extract_messages(
        raw_text, kp_candidates=kp_candidates, context=context
    )
    try:
        async with httpx.AsyncClient(timeout=DESC_INTAKE_LLM_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                # 2026-11-05：混元 tokenhub → 火山方舟（该网关不认 hunyuan-pro，恒 400）
                f"{VOLCANO_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {VOLCANO_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DESC_INTAKE_LLM_MODEL,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            slots = parse_slot_extract_response(content)
            return {"slots": slots, "error": None}
    except Exception as e:  # noqa: BLE001 —— 红线 R16：任何失败不抛出
        err = f"{type(e).__name__}: {e}"
        logger.warning(f"[desc_intake_llm] slot_extract 失败，走 fallback: {err}")
        return {"slots": None, "error": err}
