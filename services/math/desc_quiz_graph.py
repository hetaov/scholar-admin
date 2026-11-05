"""LangGraph 语音描述出题主图——基础题段（方案：docs_v1/数学/出题-语音描述出题-LangGraph编排方案.md §5）

两段式破「只改个数」（§5.1）：解剖（deconstruct，产出可程序校验的题目结构卡）
与生成（generate_basic，以结构卡为硬约束）解耦；validate_basic 零 token 程序校验
（V1~V6，实现在 desc_validators）做「约束」主体。

  START → deconstruct ─┬(卡非法 & 未重解剖)→ deconstruct（重解剖 ≤1 次）
                       ├(卡非法 & 已重解剖)→ degrade（无卡直出 + STRUCTURE_CARD_DEGRADED）
                       └(ok)→ generate_basic → validate_basic
                                  ↑ (fail & retry<max) 定向重试（附失败原因）
                                  ↓ (fail & retry≥max) 题目级剔除 + warnings
                                END

拔高段（extend_plan/generate_extended/validate_extended，§6）由 T16/T17 在本图追加。

红线：R16 节点级永不抛异常（LLM 失败降级）；落库/出参投影由 quiz.py 负责（R17）；
`USE_LANGGRAPH_DESC_QUIZ` 开关在路由层判断（T20/T21），本模块不感知。
"""
from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from config import (
    DESC_QUIZ_BASIC_MAX_RETRIES,
    DESC_QUIZ_LLM_TIMEOUT_SECONDS,
    DESC_INTAKE_LLM_MODEL,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
from services.math.desc_validators import (
    check_answer,
    check_answer_leakage,
    check_kp_in_scope,
    check_structure_card,
    check_trap_assignment,
    near_duplicate,
    same_structure,
    validate_extended_item,
)
from services.providers.desc_intake_llm import (
    parse_slot_extract_response as _parse_llm_json,  # 容错 JSON 解析复用（同族模块）
)

logger = logging.getLogger("scholar-admin.math.desc_quiz_graph")

_GRAPH_VERSION = 1

W_CARD_DEGRADED = "STRUCTURE_CARD_DEGRADED"
W_ITEM_DROPPED = "DESC_BASIC_ITEM_DROPPED"
W_TRAP_NOT_HIT = "V3_TRAP_NOT_HIT"
W_EXTEND_PLAN_DEGRADED = "EXTEND_PLAN_DEGRADED"
W_EXTEND_ITEM_DROPPED = "DESC_EXTEND_ITEM_DROPPED"


# ---------------------------------------------------------------------------
# Prompts（§5.2 / §5.3）
# ---------------------------------------------------------------------------

_DECONSTRUCT_SYSTEM = (
    "你是小学数学教研专家。请对给定的同类型题出题需求做『题目解剖』，"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_DECONSTRUCT_USER_TEMPLATE = """请解剖以下出题需求，产出「题目结构卡」。

知识点（只允许围绕这些考点）：
{grounded_kps}
题型提示：{question_type_hint}

输出 JSON：
{{
  "core_rationale": "本类题的算理/解题原理（一句话）",
  "step_sequence": ["步骤1", "步骤2", …],   // 3~6 步，每步一个动作短语
  "givent_unknown": {{"givens": ["已知量结构"], "unknown": "未知量"}},
  "error_prone_steps": [{{"step": 2, "trap": "易错点（含典型 misconception）", "weight": 0.6}}],  // ≥1 项，step 为 step_sequence 下标（从 1 起）
  "difficulty_band": 1~5 的整数,
  "variant_axes": ["可变维度，如：数值", "情境"]
}}"""

_BASIC_SYSTEM = (
    "你是小学数学命题专家。只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_BASIC_USER_TEMPLATE = """依据「题目结构卡」出 {question_count} 道同类型题。

结构卡：
{structure_card}

知识点（只允许用这些考点与表述）：{grounded_kps}
{ref_block}{feedback_block}硬性要求：
1. 题目步骤数与结构卡 step_sequence 一致（说法可换，步骤不可增删）；
2. 至少 {trap_min} 道题把「数值设点」放在 error_prone_steps 指定的易错位上
   （如：让易错解恰好是错误方法算出的结果），其余题易错位随机分布；
3. 数值范围与教材难度档 ±1；答案必须是可计算的确定值；
4. 每题输出字段：
   {{"question": "题干", "answer": "确定答案", "answer_type": "fraction|decimal|integer",
     "difficulty": 1~5, "variant_level": "L1|L2", "knowledge_point_name": "考点名",
     "rationale": "为什么这么出（≤80 字，禁止包含答案数值或解题步骤）",
     "target_error_step": 易错位步骤序号（未设点为 0）}}

输出 JSON：{{"items": [每题一个对象]}}"""


# ---------------------------------------------------------------------------
# LLM 调用 seam（测试经 monkeypatch 本函数注入 stub）
# ---------------------------------------------------------------------------


async def _call_chat(*, system: str, user: str) -> str:
    """火山方舟 chat 调用（返回原始 content 字符串；网络/超时异常由调用方捕获降级）。

    2026-11-05 由混元 tokenhub 切换：该网关不认 hunyuan-pro 等服务 ID（恒 400
    Bad Request）→ 慢路径恒降级、意图卡全空。对齐既有出题链路（VOLCANO_*）。
    读超时 DESC_QUIZ_LLM_TIMEOUT_SECONDS（默认 180s）：推理型模型单次出 5~10 题
    实测 > 30s（真机 ReadTimeout → 0 题 failed 复现）。
    """
    async with httpx.AsyncClient(timeout=DESC_QUIZ_LLM_TIMEOUT_SECONDS) as client:
        resp = await client.post(
            f"{VOLCANO_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {VOLCANO_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DESC_INTAKE_LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 响应解析（容错，非法即抛 ValueError 由节点捕获降级）
# ---------------------------------------------------------------------------


def parse_card_response(text: str) -> dict:
    """解析结构卡 JSON（复用 slot_extract 的容错解析）。"""
    card = _parse_llm_json(text)
    if not isinstance(card.get("step_sequence"), list):
        raise ValueError("结构卡缺 step_sequence")
    return card


def _tolerant_json(text: str) -> Any:
    """容错 JSON 解析（对象或数组；fence/噪声剥离）。失败抛 ValueError。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    if not (text.startswith("{") or text.startswith("[")):
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or start >= end:
            start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def parse_basic_items_response(text: str) -> list[dict]:
    """解析基础题 items JSON：接受 {"items": [...]} 或裸数组；剔除非 dict 项。"""
    parsed = _tolerant_json(text)
    raw_items = parsed.get("items") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_items, list):
        raise ValueError("items 输出非数组")
    return [it for it in raw_items if isinstance(it, dict) and it.get("question")]


# ---------------------------------------------------------------------------
# 单题程序校验（零 token，§5.4 V1/V2/V4/V5/V6；V3 为批次级）
# ---------------------------------------------------------------------------


def validate_basic_item(
    item: dict,
    *,
    ref_question: str = "",
    allowed_kps: list[str] | None = None,
    prior_questions: list[str] | None = None,
) -> list[str]:
    """单题校验，返回拒绝码列表（空 = 通过）。永不抛异常。"""
    codes: list[str] = []
    question = str(item.get("question") or "")

    # V1 结构同构：与参照题（源题/首题）指纹比对（L2 换情境不改步骤序列）
    if ref_question and not same_structure(question, ref_question):
        codes.append("V1_STRUCTURE_MISMATCH")

    # V2 答案可计算性：可程序求解且错误 → 拒；不可程序求解（应用题）→ 交双评不拒
    verdict = check_answer(question, item.get("answer"))
    if verdict is False:
        codes.append("V2_ANSWER_WRONG")

    # V4 查重：与参照题、与先前已产题两两
    for other in [ref_question] + list(prior_questions or []):
        if other and near_duplicate(question, other):
            codes.append("V4_DUPLICATE")
            break

    # V5 答案泄露
    if check_answer_leakage(item):
        codes.append("V5_ANSWER_LEAK")

    # V6 考点越界
    if not check_kp_in_scope(item.get("knowledge_point_name"), allowed_kps):
        codes.append("V6_KP_OUT_OF_SCOPE")
    return codes


# ---------------------------------------------------------------------------
# 图状态
# ---------------------------------------------------------------------------


class DescQuizState(TypedDict, total=False):
    # 输入
    grounded_kps: list          # [{name, summary?, node_code?}]（grounding 后）
    question_count: int
    question_type_hint: str
    source_question: str        # 源题样例（structure_mode 有；纯描述模式可空）
    max_retries: int
    # 拔高（§6）：include_extended 开启时启用
    include_extended: bool
    neighbors: list             # 教材图谱一跳邻居 [{name, node_code, summary?}]
    # 解剖
    card: dict | None
    card_degraded: bool
    deconstruct_retried: bool
    # 生成与校验
    items: list
    failure_feedback: str
    basic_retry: int
    validation_failed: bool
    # 拔高段
    extend_plan: dict           # {"ladder": [...], "degraded": bool}
    extend_step: int
    extend_pool: list           # 剩余可用邻居（降档时收缩）
    extend_degrade_count: int
    extended_items: list
    extend_item: dict           # 当前档生成产物（未校验）
    extend_done: bool
    extend_route: str           # next | replan | done
    # 输出
    warnings: list
    status: str
    error: str


def _kps_block(grounded_kps: list) -> str:
    lines = []
    for k in grounded_kps or []:
        name = k.get("name") if isinstance(k, dict) else str(k)
        summary = k.get("summary") if isinstance(k, dict) else ""
        lines.append(f"- {name}" + (f"（{summary}）" if summary else ""))
    return "\n".join(lines) or "（无）"


# ---------------------------------------------------------------------------
# 图节点
# ---------------------------------------------------------------------------


async def _node_deconstruct(state: DescQuizState) -> dict:
    """解剖节点：LLM 产出结构卡；非法 → 重解剖 ≤1 次 → 降级无卡直出。"""
    user = _DECONSTRUCT_USER_TEMPLATE.format(
        grounded_kps=_kps_block(state.get("grounded_kps")),
        question_type_hint=state.get("question_type_hint") or "不限",
    )
    try:
        content = await _call_chat(system=_DECONSTRUCT_SYSTEM, user=user)
        card = parse_card_response(content)
    except Exception as e:  # noqa: BLE001
        card, err = None, f"{type(e).__name__}: {e}"
        logger.warning(f"[desc_quiz_graph] 解剖失败: {err}")
    else:
        issues = check_structure_card(card)
        if not issues:
            return {"card": card, "card_degraded": False}
        card, err = None, ",".join(issues)

    if not state.get("deconstruct_retried"):
        return {"card": None, "deconstruct_retried": True}
    # 重解剖仍失败 → 降级：无卡直出（prompt 走既有模板语义）
    return {
        "card": None,
        "card_degraded": True,
        "warnings": [W_CARD_DEGRADED],
        "failure_feedback": "",
        "error": err,
    }


def _route_after_deconstruct(state: DescQuizState) -> str:
    if state.get("card"):
        return "generate_basic"
    if state.get("card_degraded"):
        return "generate_basic"  # 降级直出
    return "deconstruct"  # 首次失败，重解剖


async def _node_generate_basic(state: DescQuizState) -> dict:
    """生成节点：结构卡为硬约束；降级时 structure_card 段置「无（按常规出题）」。"""
    card = state.get("card")
    structure_block = (
        json.dumps(card, ensure_ascii=False) if card else "无结构卡（按知识点常规出同类型题）"
    )
    ref = state.get("source_question") or ""
    ref_block = f"源题样例（结构必须与之一致）：{ref}\n" if ref else ""
    feedback = state.get("failure_feedback") or ""
    feedback_block = (
        f"上一轮校验失败原因（本轮必须修正）：\n{feedback}\n" if feedback else ""
    )
    count = int(state.get("question_count") or 0)
    user = _BASIC_USER_TEMPLATE.format(
        question_count=count,
        structure_card=structure_block,
        grounded_kps=_kps_block(state.get("grounded_kps")),
        ref_block=ref_block,
        feedback_block=feedback_block,
        trap_min=-(-count // 2),  # ceil(count/2)
    )
    try:
        content = await _call_chat(system=_BASIC_SYSTEM, user=user)
        items = parse_basic_items_response(content)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[desc_quiz_graph] 生成失败: {type(e).__name__}: {e}")
        items = []
        # 留痕：0 题时前端只见「未生成有效题目」，凭此码可区分超时/解析失败
        state_warnings = list(state.get("warnings") or [])
        state_warnings.append(f"{W_ITEM_DROPPED}[GEN_FAILED]: {type(e).__name__}: {str(e)[:80]}")
        return {"items": items, "warnings": state_warnings}
    return {"items": items}


def _node_validate_basic(state: DescQuizState) -> dict:
    """校验节点（零 token）：单题 V1/V2/V4/V5/V6 + 批次 V3；路由重试/剔除。"""
    items = list(state.get("items") or [])
    ref = state.get("source_question") or ""
    allowed = [
        k.get("name") if isinstance(k, dict) else str(k)
        for k in state.get("grounded_kps") or []
    ]
    max_retries = int(state.get("max_retries", DESC_QUIZ_BASIC_MAX_RETRIES))
    retry = int(state.get("basic_retry", 0))

    issues_per_item: list[list[str]] = []
    prior: list[str] = []
    for it in items:
        codes = validate_basic_item(
            it, ref_question=ref, allowed_kps=allowed, prior_questions=prior
        )
        issues_per_item.append(codes)
        if not codes:
            prior.append(str(it.get("question") or ""))

    failed_pairs = [(i, c) for i, c in enumerate(issues_per_item) if c]
    # 空产出（LLM 超时/输出不合法）：零 token 校验放行会「看似成功实则 0 题」→
    # 触发一次重出（仅一次，防超时×重试把总时长拖爆；仍空则落 failed 带原因）
    if not items:
        if retry < 1:
            return {
                "validation_failed": True,
                "failure_feedback": "上一轮未返回任何题目（超时或输出不合法），请严格按要求的 JSON 格式输出全部题目",
                "basic_retry": retry + 1,
            }
        # 不再重试：标记失败原因（前端 error_message 会展示）
        warnings = list(state.get("warnings") or [])
        if not any(w.startswith(W_ITEM_DROPPED) for w in warnings):
            warnings.append(f"{W_ITEM_DROPPED}[GEN_EMPTY]: 连续两次未返回题目")
        return {
            "validation_failed": False,
            "failure_feedback": "",
            "items": [],
            "warnings": warnings,
            "status": "basic_ok",
        }
    if failed_pairs and retry < max_retries:
        feedback = "\n".join(
            f"- 第{i + 1}题「{str(items[i].get('question'))[:40]}…」: {','.join(c)}"
            for i, c in failed_pairs
        )
        return {
            "validation_failed": True,
            "failure_feedback": feedback,
            "basic_retry": retry + 1,
        }

    # 重试上限或全部通过：题目级剔除（不整体失败）
    kept: list[dict] = []
    warnings = list(state.get("warnings") or [])  # 合并既有告警（如 STRUCTURE_CARD_DEGRADED）
    for it, codes in zip(items, issues_per_item):
        if codes:
            warnings.append(f"{W_ITEM_DROPPED}[{','.join(codes)}]: {str(it.get('question'))[:40]}")
        else:
            kept.append(it)

    # 批次级 V3（易错位占比）
    ok3, _ = check_trap_assignment(kept)
    if not ok3:
        warnings.append(W_TRAP_NOT_HIT)

    return {
        "validation_failed": False,
        "failure_feedback": "",
        "items": kept,
        "warnings": warnings,
        "status": "basic_ok",
    }


def _route_after_validate(state: DescQuizState) -> str:
    if state.get("validation_failed"):
        return "generate_basic"
    return _route_to_extend(state)


def _route_to_extend(state: DescQuizState) -> str:
    """基础段通过后的拔高门：include_extended 开且有基础题才进计划节点。"""
    if state.get("include_extended") and state.get("items"):
        return "extend_plan"
    return END


# ---------------------------------------------------------------------------
# 拔高段（§6：extend_plan 零 token → generate_extended → validate_extended 降档路由）
# ---------------------------------------------------------------------------

_EXTEND_MAX_DEGRADE = 2  # 降档总上限（防震荡，§6.4）


_EXTENDED_USER_TEMPLATE = """你是小学数学命题专家。请基于结构卡出一道**拔高题**：在基础考点上引入一个新知识点。

结构卡：
{structure_card}

基础考点（题目主体必须仍在其上）：{grounded_kps}
基础题样例：{base_sample}
本档引入：
- 新知识点：{new_kp}（必须让题目**自然运用**它，禁止「附加小问」式拼接）
- 与基础考点组合方式：{combine_with}
- 难度：{base_difficulty} + {difficulty_delta}（禁止跳档）
- 情境提示：{axis_hint}

硬性要求：
1. 题干必须真实用到新知识点的核心算理（术语自然出现在题干）；
2. 禁止引入计划外的任何知识点；
3. 答案必须是可计算的确定值；rationale ≤80 字且禁止包含答案。

输出 JSON：{{"items": [{{"question": "…", "answer": "…", "answer_type": "…",
  "difficulty": {base_difficulty_plus_delta}, "knowledge_point_name": "{new_kp}",
  "rationale": "…"}}]}}"""


def build_extend_plan(
    pool: list,
    *,
    base_kp_names: list[str],
    base_difficulty: int,
    question_count: int,
    used_kps: list[str] | None = None,
) -> dict:
    """渐进计划（零 token，§6.2）：候选只来自一跳邻居白名单，每档 ≤1 个新 kp。"""
    used = set(used_kps or [])
    cands = [n for n in pool or [] if str(n.get("name") or "") not in used and n.get("name")]
    max_steps = 2 if (question_count or 0) > 2 else 1
    ladder: list[dict] = []
    for i in range(min(max_steps, len(cands))):
        step_no = i + 1
        ladder.append(
            {
                "step": step_no,
                "new_kp": {
                    "name": str(cands[i].get("name")),
                    "node_code": str(cands[i].get("node_code") or ""),
                },
                "combine_with": (
                    base_kp_names[0] if step_no == 1 and base_kp_names
                    else str(ladder[-1]["new_kp"]["name"])
                ),
                "difficulty_delta": step_no,  # 第一档 +1、第二档 +2，单调不跳档
                "axis_hint": str(cands[i].get("summary") or ""),
            }
        )
    return {"ladder": ladder, "degraded": len(ladder) < max_steps}


def _node_extend_plan(state: DescQuizState) -> dict:
    """计划节点（零 token）：从剩余邻居池组阶梯；空池 → 降级跳过拔高。"""
    base_names = [
        k.get("name") if isinstance(k, dict) else str(k)
        for k in state.get("grounded_kps") or []
    ]
    base_diff = _base_difficulty(state)
    plan = build_extend_plan(
        state.get("extend_pool"),
        base_kp_names=[str(n) for n in base_names if n],
        base_difficulty=base_diff,
        question_count=state.get("question_count") or 0,
    )
    warnings = list(state.get("warnings") or [])
    if not plan["ladder"]:
        warnings.append(W_EXTEND_PLAN_DEGRADED)
    return {"extend_plan": plan, "extend_step": 1, "warnings": warnings}


def _route_after_plan(state: DescQuizState) -> str:
    ladder = (state.get("extend_plan") or {}).get("ladder") or []
    return "generate_extended" if ladder else END


def _base_difficulty(state: DescQuizState) -> int:
    items = state.get("items") or []
    for it in items:
        try:
            return int(it.get("difficulty"))
        except (TypeError, ValueError):
            continue
    card = state.get("card") or {}
    band = card.get("difficulty_band")
    return band if isinstance(band, int) else 2


async def _node_generate_extended(state: DescQuizState) -> dict:
    """拔高生成节点：当前档 1 题；is_extended/extended_meta 随题打标（沿用既有口径）。"""
    plan = state.get("extend_plan") or {}
    ladder = plan.get("ladder") or []
    step_idx = int(state.get("extend_step", 1))
    step = ladder[step_idx - 1]
    new_kp = step["new_kp"]["name"]
    base_diff = _base_difficulty(state)
    items_basic = state.get("items") or []
    base_sample = str(items_basic[0].get("question")) if items_basic else "（无）"
    user = _EXTENDED_USER_TEMPLATE.format(
        structure_card=json.dumps(state.get("card") or {}, ensure_ascii=False),
        grounded_kps=_kps_block(state.get("grounded_kps")),
        base_sample=base_sample,
        new_kp=new_kp,
        combine_with=step.get("combine_with") or "",
        base_difficulty=base_diff,
        difficulty_delta=step.get("difficulty_delta", 1),
        axis_hint=step.get("axis_hint") or "（无）",
        base_difficulty_plus_delta=base_diff + int(step.get("difficulty_delta", 1)),
    )
    try:
        content = await _call_chat(system=_BASIC_SYSTEM, user=user)
        gen = parse_basic_items_response(content)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[desc_quiz_graph] 拔高生成失败: {type(e).__name__}: {e}")
        gen = []
    item = gen[0] if gen else {"question": "", "answer": ""}
    item["is_extended"] = True
    item["extended_meta"] = {"new_kp": new_kp, "step": step_idx}
    item["knowledge_point_name"] = new_kp
    return {"extend_item": item}


def _node_validate_extended(state: DescQuizState) -> dict:
    """拔高校验节点（零 token，E1~E5）；失败走降档而非重试（§6.4）。"""
    item = state.get("extend_item") or {}
    plan = state.get("extend_plan") or {}
    ladder = plan.get("ladder") or []
    step_idx = int(state.get("extend_step", 1))
    step = ladder[step_idx - 1] if step_idx <= len(ladder) else {}
    base_names = [
        str(k.get("name") if isinstance(k, dict) else k)
        for k in state.get("grounded_kps") or []
    ]
    base_questions = [str(it.get("question") or "") for it in state.get("items") or []]
    warnings = list(state.get("warnings") or [])
    extended = list(state.get("extended_items") or [])
    degrade_count = int(state.get("extend_degrade_count", 0))

    codes = validate_extended_item(
        item,
        base_kps=base_names,
        plan_new_kp=str(step.get("new_kp", {}).get("name") or ""),
        plan_difficulty_delta=int(step.get("difficulty_delta", 1)),
        base_difficulty=_base_difficulty(state),
        base_questions=base_questions,
    )
    if not codes:
        extended.append(item)
        has_more = step_idx < len(ladder)
        return {
            "extended_items": extended,
            "extend_step": step_idx + 1 if has_more else step_idx,
            "extend_route": "next" if has_more else "done",
        }

    # 失败 → 降档：把失败 new_kp 移出候选池重建计划（水位收缩），不重试同档
    failed_kp = str(step.get("new_kp", {}).get("name") or "")
    pool = [
        n for n in state.get("extend_pool") or []
        if str(n.get("name") or "") != failed_kp
    ]
    if degrade_count >= _EXTEND_MAX_DEGRADE or not pool:
        warnings.append(f"{W_EXTEND_ITEM_DROPPED}[{','.join(codes)}]: {str(item.get('question'))[:40]}")
        return {"extended_items": extended, "warnings": warnings, "extend_route": "done"}
    return {
        "extend_pool": pool,
        "extend_degrade_count": degrade_count + 1,
        "extended_items": extended,
        "warnings": warnings,
        "extend_step": 1,
        "extend_route": "replan",
    }


def _route_after_validate_extended(state: DescQuizState) -> str:
    route = state.get("extend_route")
    if route == "next":
        return "generate_extended"
    if route == "replan":
        return "extend_plan"
    return END


# ---------------------------------------------------------------------------
# 图构建与编译（基础段 + 拔高段）
# ---------------------------------------------------------------------------


def _build_desc_quiz_graph() -> Any:
    graph = StateGraph(DescQuizState)
    graph.add_node("deconstruct", _node_deconstruct)
    graph.add_node("generate_basic", _node_generate_basic)
    graph.add_node("validate_basic", _node_validate_basic)
    graph.add_node("extend_plan", _node_extend_plan)
    graph.add_node("generate_extended", _node_generate_extended)
    graph.add_node("validate_extended", _node_validate_extended)

    graph.add_edge(START, "deconstruct")
    graph.add_conditional_edges(
        "deconstruct",
        _route_after_deconstruct,
        {"deconstruct": "deconstruct", "generate_basic": "generate_basic"},
    )
    graph.add_edge("generate_basic", "validate_basic")
    graph.add_conditional_edges(
        "validate_basic",
        _route_after_validate,
        {"generate_basic": "generate_basic", "extend_plan": "extend_plan", END: END},
    )
    graph.add_conditional_edges(
        "extend_plan",
        _route_after_plan,
        {"generate_extended": "generate_extended", END: END},
    )
    graph.add_edge("generate_extended", "validate_extended")
    graph.add_conditional_edges(
        "validate_extended",
        _route_after_validate_extended,
        {"generate_extended": "generate_extended", "extend_plan": "extend_plan", END: END},
    )
    return graph.compile()


_compiled_graph: Any = None


def _get_compiled_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_desc_quiz_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# 运行入口
# ---------------------------------------------------------------------------


async def run_desc_quiz_basic(
    *,
    grounded_kps: list,
    question_count: int,
    question_type_hint: str = "",
    source_question: str = "",
    max_retries: int | None = None,
    include_extended: bool = False,
    neighbors: list | None = None,
) -> dict:
    """出题主图运行入口（永不抛异常；落库与 flag 回退由 quiz.py 负责）。

    Args:
        include_extended: 是否追加拔高题（§6）；为真时 `neighbors` 提供教材
            图谱一跳邻居候选（查询由调用方完成，本图零图谱 IO）。
        neighbors: [{name, node_code, summary?}]。

    Returns:
        {"items"（基础 + 拔高，拔高带 is_extended/extended_meta）, "warnings",
         "card", "card_degraded", "graph_version", "retries_used", "status",
         "extended_items", "extend_degrades"}
        异常降级：items=[]、warnings 带 DESC_BASIC_GRAPH_ERROR。
    """
    try:
        graph = _get_compiled_graph()
        state: DescQuizState = {
            "grounded_kps": grounded_kps or [],
            "question_count": int(question_count or 0),
            "question_type_hint": question_type_hint or "",
            "source_question": source_question or "",
            "max_retries": (
                max_retries if max_retries is not None else DESC_QUIZ_BASIC_MAX_RETRIES
            ),
            "include_extended": bool(include_extended),
            "neighbors": neighbors or [],
            "extend_pool": neighbors or [],
            "extended_items": [],
            "extend_degrade_count": 0,
            "warnings": [],
        }
        final: DescQuizState = await graph.ainvoke(state)
        basic = final.get("items") or []
        extended = final.get("extended_items") or []
        return {
            "items": basic + extended,
            "extended_items": extended,
            "extend_degrades": int(final.get("extend_degrade_count", 0)),
            "warnings": final.get("warnings") or [],
            "card": final.get("card"),
            "card_degraded": bool(final.get("card_degraded")),
            "graph_version": _GRAPH_VERSION,
            "retries_used": int(final.get("basic_retry", 0)),
            "status": final.get("status") or "basic_ok",
        }
    except Exception as e:  # noqa: BLE001 —— R16
        err = f"{type(e).__name__}: {e}"
        logger.warning(f"[desc_quiz_graph] run_desc_quiz_basic 异常: {err}")
        return {
            "items": [],
            "extended_items": [],
            "extend_degrades": 0,
            "warnings": [f"DESC_BASIC_GRAPH_ERROR: {err}"],
            "card": None,
            "card_degraded": True,
            "graph_version": _GRAPH_VERSION,
            "retries_used": 0,
            "status": "error",
        }
