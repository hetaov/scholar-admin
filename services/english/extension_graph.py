"""英文语句扩展 — 非连续短语 LangGraph 编排

设计：docs_v1/扩展/非连续短语-LangGraph编排方案-v1.md

定位：
- 仅负责**非连续**多词表达（可分短语动词 turn the light off → turn off）的发现与富化；
- 既有连续短语抽取（generate_candidates + 候选口径 LLM）保持原样，由主流水线并行执行后 merge。

图结构：
  START → discover_structures ──LLM──► validate_spans ──后端R7──┐
                        (空)                                       │
                         └──────────────► assemble ◄──────────────┘
                                          (assemble 内部：有 validated 则需 enrich，否则直出空)

  实际为线性：discover → validate → (validated 非空 → enrich) → assemble

降级：
- discover / enrich 任一 LLM 失败 → 该节点 error 记录，assemble 出空 non_contiguous_points；
- 主流水线仍用连续点产出结果，不整体失败（R8 不变）。
"""
from __future__ import annotations

import logging
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from services.providers.extension_llm import (
    ExtensionError,
    call_extension_llm,
    extract_json,
)

logger = logging.getLogger("scholar-admin.extension_graph")

_GRAPH_VERSION = 1


# ===========================================================================
# 图状态
# ===========================================================================


class ExtensionGraphState(TypedDict, total=False):
    # 输入
    sentence_id: str
    original: str
    translation: str
    # discover
    raw_structures: list[dict]
    discover_error: str
    # validate
    validated_structures: list[dict]
    discarded_count: int
    # enrich
    enriched_points: list[dict]
    enrich_error: str
    # assemble
    final_points: list[dict]
    # 输出
    status: str  # success | no_structures | failed
    error: str


# ===========================================================================
# Prompt 常量
# ===========================================================================

_DISCOVER_SYSTEM_PROMPT = """你是英语语言学点抽取助手。任务：从英文句子中识别**非连续**多词表达。

「非连续」指构成表达的词被其他词隔开，典型：
- 可分短语动词（separable phrasal verbs）：turn the light off 中的 turn...off；pick it up 中的 pick...up
- 含插入语的习语：look the word up 中的 look...up

输出要求（严格 JSON）：
{
  "structures": [
    {
      "text": "turn off",          // 规范形式（动词+小品词直接拼接，不带中间词）
      "spans": [[0, 4], [15, 18]], // 原句中的字符区间，多组；text 必须等于这些区间切出字符的拼接
      "type": "phrase",            // phrase 或 idiom
      "confidence": 0.9            // 0~1
    }
  ]
}

约束：
1. 只报**非连续**表达；连续短语（如 take part in）由候选层处理，不要重复。
2. spans 必须是原句的真实字符偏移；text = ''.join(original[s:e] for s,e in spans)。
3. 无非连续表达时返回 {"structures": []}。
4. 不要返回单个单词。
"""

_ENRICH_SYSTEM_PROMPT = """你是英语教学助手。为下列非连续语言点补充教学信息。

输入是已通过坐标校验的结构清单（编号 n1, n2...）。
对每个你认为值得教学的结构输出一条，用 structure_id 指代；**不要改写 text / spans**。

输出要求（严格 JSON）：
{
  "points": [
    {
      "structure_id": "n1",
      "type": "phrase",
      "meaning_zh": "关掉",
      "register": "neutral",       // formal | neutral | informal
      "example_en": "Please turn off the light before you leave.",
      "example_zh": "离开前请关灯。",
      "collocations": ["turn off the light", "turn off the TV"],
      "synonyms": ["switch off"],
      "confusable": ["turn on"],
      "cefr": "A2",                // A1|A2|B1|B2|C1|C2 或空
      "confidence": 0.9,
      "reason": "常用及物可分短语动词"
    }
  ]
}

约束：
- meaning_zh 必填，≤30字；example_en 必填且必须包含 text（大小写不敏感）；
- phrase 必填 collocations ≥1；
- reason 必填，≤40字。
"""


def _build_discover_messages(original: str, translation: str) -> list[dict]:
    user = f"英文原句：{original}\n"
    if translation:
        user += f"中文译文：{translation}\n"
    user += "请识别其中的非连续多词表达（动词与小品词/名词被其他词隔开）。"
    return [
        {"role": "system", "content": _DISCOVER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _build_enrich_messages(
    original: str, translation: str, structures: list[dict]
) -> list[dict]:
    lines = []
    for i, s in enumerate(structures, start=1):
        sid = f"n{i}"
        spans_str = ",".join(f"[{a},{b}]" for a, b in s["spans"])
        lines.append(
            f"- {sid}｜text={s['text']}｜spans={spans_str}｜type={s.get('type','phrase')}"
        )
    user = f"英文原句：{original}\n"
    if translation:
        user += f"中文译文：{translation}\n"
    user += "结构清单（已通过坐标校验）：\n" + "\n".join(lines) + "\n"
    user += "请为值得教学的结构补充教学信息。"
    return [
        {"role": "system", "content": _ENRICH_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ===========================================================================
# 节点：discover_structures（LLM 发现非连续结构）
# ===========================================================================


def _validate_raw_structures(raw: Any, original: str) -> list[dict]:
    """校验 discover LLM 输出的 structures，返回合法项。"""
    if not isinstance(raw, list):
        return []
    valid: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        spans = item.get("spans")
        ptype = item.get("type", "phrase")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(spans, list) or len(spans) < 2:
            continue  # 非连续必须 ≥2 个区间
        parsed_spans: list[list[int]] = []
        ok = True
        for s in spans:
            if not isinstance(s, (list, tuple)) or len(s) != 2:
                ok = False
                break
            try:
                parsed_spans.append([int(s[0]), int(s[1])])
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        if ptype not in ("phrase", "idiom"):
            ptype = "phrase"
        valid.append({"text": text.strip(), "spans": parsed_spans, "type": ptype})
    return valid


async def discover_structures(state: ExtensionGraphState) -> dict:
    """Node 1：LLM 识别非连续结构（text + spans + type）。"""
    original = state["original"]
    translation = state.get("translation", "")
    try:
        messages = _build_discover_messages(original, translation)
        content = await call_extension_llm(messages)
        parsed = extract_json(content)
        raw = (parsed or {}).get("structures", []) if parsed else []
        structures = _validate_raw_structures(raw, original)
        return {"raw_structures": structures, "discover_error": ""}
    except ExtensionError as e:
        logger.warning("[extension_graph] discover LLM 失败: %s", e.error_code)
        return {"raw_structures": [], "discover_error": e.error_code}
    except Exception as e:  # noqa: BLE001
        logger.error("[extension_graph] discover 异常: %s", e, exc_info=True)
        return {"raw_structures": [], "discover_error": "PROVIDER_UNAVAILABLE"}


# ===========================================================================
# 节点：validate_spans（后端 R7 多区间校验，零 LLM）
# ===========================================================================


async def validate_spans(state: ExtensionGraphState) -> dict:
    """Node 2：后端校验 spans 坐标（R7：拼接 == text），丢无效。"""
    original = state["original"]
    raw = state.get("raw_structures", [])
    validated: list[dict] = []
    discarded = 0
    for s in raw:
        spans = s["spans"]
        # 区间合法性
        prev_end = -1
        ok = True
        for a, b in spans:
            if not (0 <= a < b <= len(original)):
                ok = False
                break
            if a <= prev_end:
                ok = False
                break
            prev_end = b
        if not ok:
            discarded += 1
            continue
        # R7：拼接（去空白）== text（去空白）
        joined = "".join(original[a:b] for a, b in spans)
        if re.sub(r"\s+", "", joined) != re.sub(r"\s+", "", s["text"]):
            discarded += 1
            continue
        validated.append(s)
    return {"validated_structures": validated, "discarded_count": discarded}


# ===========================================================================
# 节点：enrich_points（LLM 富化释义）
# ===========================================================================


def _validate_enriched_items(raw: Any, structure_ids: set[str]) -> list[dict]:
    if not isinstance(raw, list):
        return []
    valid: list[dict] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid = item.get("structure_id")
        if not isinstance(sid, str) or sid not in structure_ids or sid in seen:
            continue
        seen.add(sid)
        valid.append(item)
    return valid


async def enrich_points(state: ExtensionGraphState) -> dict:
    """Node 3：LLM 为校验通过的结构补充释义/例句/搭配。"""
    structures = state.get("validated_structures", [])
    if not structures:
        return {"enriched_points": [], "enrich_error": ""}
    original = state["original"]
    translation = state.get("translation", "")
    structure_ids = {f"n{i}" for i in range(1, len(structures) + 1)}
    try:
        messages = _build_enrich_messages(original, translation, structures)
        content = await call_extension_llm(messages)
        parsed = extract_json(content)
        raw = (parsed or {}).get("points", []) if parsed else []
        items = _validate_enriched_items(raw, structure_ids)
        return {"enriched_points": items, "enrich_error": ""}
    except ExtensionError as e:
        logger.warning("[extension_graph] enrich LLM 失败: %s", e.error_code)
        return {"enriched_points": [], "enrich_error": e.error_code}
    except Exception as e:  # noqa: BLE001
        logger.error("[extension_graph] enrich 异常: %s", e, exc_info=True)
        return {"enriched_points": [], "enrich_error": "PROVIDER_UNAVAILABLE"}


# ===========================================================================
# 节点：assemble（后端组装最终 points）
# ===========================================================================


async def assemble(state: ExtensionGraphState) -> dict:
    """Node 4：把 validated_structures 的 text/spans 与 enriched_points 合并为统一 point。"""
    structures = state.get("validated_structures", [])
    enriched = state.get("enriched_points", [])
    enriched_map: dict[str, dict] = {}
    for e in enriched:
        enriched_map[e["structure_id"]] = e

    points: list[dict] = []
    for i, s in enumerate(structures, start=1):
        sid = f"n{i}"
        e = enriched_map.get(sid, {})
        points.append(
            {
                "id": "",
                "type": e.get("type", s.get("type", "phrase")),
                "text": s["text"],
                "span": s["spans"][0],
                "spans": s["spans"],
                "discontinuous": len(s["spans"]) > 1,
                "pos": e.get("pos", ""),
                "meaning_zh": e.get("meaning_zh", ""),
                "literal_zh": e.get("literal_zh", ""),
                "register": e.get("register", "neutral"),
                "example_en": e.get("example_en", ""),
                "example_zh": e.get("example_zh", ""),
                "collocations": e.get("collocations", []),
                "synonyms": e.get("synonyms", []),
                "confusable": e.get("confusable", []),
                "cefr": e.get("cefr", ""),
                "reason": e.get("reason", ""),
                "confidence": e.get("confidence", 0.0),
                "risk": "",
                "excluding": False,
            }
        )

    discover_err = state.get("discover_error", "")
    enrich_err = state.get("enrich_error", "")
    if discover_err and not points:
        status = "failed"
        error = discover_err
    elif enrich_err and not points:
        status = "failed"
        error = enrich_err
    elif not points:
        status = "no_structures"
        error = ""
    else:
        status = "success"
        error = ""

    return {"status": status, "error": error, "final_points": points}


# ===========================================================================
# 条件边
# ===========================================================================


def _route_after_validate(state: ExtensionGraphState) -> str:
    """validated 非空 → enrich；否则跳过 enrich 直 assemble。"""
    if state.get("validated_structures"):
        return "enrich"
    return "assemble"


# ===========================================================================
# 图编译
# ===========================================================================


def build_extension_graph():
    """构建并编译非连续短语抽取图。"""
    workflow = StateGraph(ExtensionGraphState)
    workflow.add_node("discover", discover_structures)
    workflow.add_node("validate", validate_spans)
    workflow.add_node("enrich", enrich_points)
    workflow.add_node("assemble", assemble)

    workflow.add_edge(START, "discover")
    workflow.add_edge("discover", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"enrich": "enrich", "assemble": "assemble"},
    )
    workflow.add_edge("enrich", "assemble")
    workflow.add_edge("assemble", END)
    return workflow.compile()


# 模块级编译实例（复用）
_extension_graph = build_extension_graph()


async def run_extension_graph(
    *,
    original: str,
    translation: str = "",
    sentence_id: str = "",
) -> dict:
    """执行非连续短语抽取图，返回 {points, status, error}。

    points 已含 spans / discontinuous，但尚未走 _validate_point / 配额 / L1，
    由主流水线 merge 后统一处理。
    """
    state: ExtensionGraphState = {
        "sentence_id": sentence_id,
        "original": original,
        "translation": translation,
    }
    result = await _extension_graph.ainvoke(state)
    return {
        "points": result.get("final_points", []),
        "status": result.get("status", "failed"),
        "error": result.get("error", ""),
    }
