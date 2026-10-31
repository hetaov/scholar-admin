"""ASR 判前修复 — LangGraph 子图（三级流水线 + 纯函数护栏）

设计：docs_v1/扩展/第六期-ASR断句修复与判句回显-修改方案-v1.md §4
触发：docs_v1/扩展/issue.md（语音造句被 ASR 切成碎片 → 判分反馈失真）

问题：学习者在造句阶段用语音作答，因不熟悉目标句，ASR 把一句话切成几段
（"He finally. Took up the question."），或把结巴如实转写（"he he finally"）。
碎片文本直接进判分 ⇒ rubric 的 semantics 维度失真、确定性 `must_use_hit`
子串匹配被插入的标点打断 ⇒ 反馈"不是自己想看到的"。

方案：**判分前**插入一个整理器，只做三件事 —— 合并误断句、删紧邻重复、补最小书写格式。

## 三级流水线（成本与延迟护栏）
  Tier 0 `precheck`   纯函数，零 token：判定是否需要修复
  Tier 1 `rule_fix`   纯函数，零 token：复用 `text_normalize.smart_punctuate`
  Tier 2 `llm_repair` 仅当出现「断句」或「紧邻重复」时才调 LLM

## 红线
- **R15 只减不增**：不得加词 / 换词 / 改词形 / 重排 ⇒ 由护栏 **G2 子序列判据**结构性保证；
- **R16 失败不阻断**：LLM 失败 / 超时 / 护栏不过 → 回落，继续判分，且 `fallback=true` 留痕；
- **R17 原文不覆盖**：本模块**不写库**，只返回文本；落库由 B04/B05 负责；
- **R18 关开关逐字不变**：`EXTENSION_ROUND_ASR_REPAIR_ENABLED=0` 时 `run_asr_repair`
  返回 `repaired_input=None`，调用方据此不加字段。

图结构：
  START → precheck ──(无需修复)────────────────► finalize
              ├─(仅标点/空白)─► rule_fix ──────► finalize
              └─(断句/重复)──► llm_repair ──► guard ─┬─► finalize
                                        失败/不过 ──┴─► fallback ─► finalize
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any, TypedDict

import config
from langgraph.graph import END, START, StateGraph
from services.infra.text_normalize import smart_punctuate
from services.providers.extension_asr_repair_llm import (
    PROMPT_VERSION,
    AsrRepairError,
    build_asr_repair_messages,
    call_asr_repair_llm,
    parse_asr_repair,
)
from services.providers.extension_llm import ExtensionError

logger = logging.getLogger("scholar-admin.asr_repair_graph")

_GRAPH_VERSION = 1

# 词序列归一化：小写 + 只留字母数字与撇号（`don't` 保持完整）。
# 护栏 G2 与退化注入器（B09 A 轨）**必须共用这一套**，否则指标失真。
_TOKEN_RE = re.compile(r"[a-z0-9']+")

# 合法语法重复白名单（G4）：这些紧邻重复是**正确的英语**，去重就是改坏句子。
# 典型：`She said she had had enough.`（过去完成时）、`the book that that boy read`。
GRAMMATIC_REDUP_WORDS = frozenset({"had", "that"})

# 口语填充词（G3）：ASR 偶尔会把 um / uh 之类转写出来。它们从来不是句子的词，
# 删除属于"整理"而非"改写作"，故与「紧邻重复富余」同列进可删额度。
FILLER_WORDS = frozenset({"um", "uh", "er", "ah", "mm", "hmm", "eh"})

# 句末标点
_TERMINAL = ".!?"

# 句末标点后可能跟的收尾符号：`."` `.'` `.)` 等。
# ⚠️ 判"是否缺句末标点"前必须先剥掉它们 —— 否则
#   `"Macaroni has been grown ... years."` 末字符是 `"` 而非 `.`，会被误判 unpunctuated
#   （A 轨 120 条实测命中 1 条）。
_TRAILING_WRAP = "\"'”’)]}>"

# 误断句特征：句中出现终结标点且其后还有内容。
#
# ⚠️ 初版用 `[.!?]\s+\S` 裸匹配，A 轨实测 30 条真实教材句**误报率 10%**
#   （`Mrs. Smith's living room is large.` 里的 `Mrs.` 被当成断句）。
#   代价是白白多一次 LLM 调用（约 2s），故按「缩写 / 首字母」排除后逐个候选判定。
_FRAGMENT_CANDIDATE_RE = re.compile(r"([A-Za-z][A-Za-z.'’]*)([.!?])(\s+\S)")

# 称谓与常用缩写：其后的句点**不是**断句。
# ⚠️ 正则 `[A-Za-z][A-Za-z.'’]*` 是**贪婪**的，会把 `B.C` 整个吃进来变成 `b.c`
#   （A 轨 120 条实测：`800 B.C. has just had` 被误判断句），故多点缩写必须逐个列全。
_ABBREV_TITLES = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc",
        "e.g", "i.e", "inc", "ltd", "no", "fig", "approx", "esp",
        # 多点缩写（贪婪匹配后 token 含点）
        "b.c", "a.d", "a.m", "p.m", "u.s", "u.k", "u.s.a", "ph.d",
    }
)

_WS_RE = re.compile(r"\s+")

# source 取值
SOURCE_NONE = "none"
SOURCE_RULE = "rule"
SOURCE_LLM = "llm"
SOURCE_FALLBACK = "fallback"
SOURCE_DISABLED = "disabled"


# ===========================================================================
# 纯函数：词序列工具
# ===========================================================================


def norm_tokens(text: str | None) -> list[str]:
    """文本 → 归一化词序列（小写、去标点）。护栏与脚本共用的唯一口径。"""
    return _TOKEN_RE.findall((text or "").lower())


def _is_subsequence(small: list[str], big: list[str]) -> bool:
    """`small` 是否为 `big` 的**按序子序列**（可跳过元素，不可重排）。"""
    it = iter(big)
    return all(tok in it for tok in small)


def _adjacent_dup_count(tokens: list[str], word: str) -> int:
    """统计 `word` 的紧邻重复对数（重叠计数：`had had had` → 2）。"""
    return sum(
        1 for i in range(len(tokens) - 1) if tokens[i] == word and tokens[i + 1] == word
    )


def dropped_tokens(raw: str, repaired: str) -> list[str]:
    """原文 → 修复文被删掉的词（多重集差，排序保证确定性）。"""
    diff = Counter(norm_tokens(raw)) - Counter(norm_tokens(repaired))
    return sorted(diff.elements())


def dup_excess(tokens: list[str]) -> Counter:
    """每个**紧邻重复** run 中"可删的富余副本"：run 长 L → 富余 L-1 个。

    这是 G3 的删除额度来源：结巴 `he he` 富余 1 个 `he`，删掉它属于"整理"；
    而删掉任何**非富余**的词，就超出了额度 ⇒ 判为改写作。

    `took took up` → `{took: 1}`；`had had had` → `{had: 2}`。
    """
    excess: Counter = Counter()
    if not tokens:
        return excess
    run_word, run_len = tokens[0], 1
    for tok in tokens[1:]:
        if tok == run_word:
            run_len += 1
        else:
            if run_len > 1:
                excess[run_word] += run_len - 1
            run_word, run_len = tok, 1
    if run_len > 1:
        excess[run_word] += run_len - 1
    return excess


def deletion_allowance(tokens: list[str]) -> Counter:
    """G3 的可删额度 = 紧邻重复富余 + 口语填充词。"""
    return dup_excess(tokens) + Counter(w for w in tokens if w in FILLER_WORDS)


# ===========================================================================
# Tier 0：precheck（零 token）
# ===========================================================================


def has_fragment(text: str) -> bool:
    """是否存在**误断句**：句中的终结标点，且不是称谓缩写 / 首字母缩写后的句点。

    | 输入 | 结果 | 说明 |
    |---|---|---|
    | `He finally. Took up the question.` | `True` | 真断句 |
    | `Mrs. Smith's living room is large.` | `False` | `Mrs.` 是称谓 |
    | `J. Smith went home.` | `False` | 单字母 = 首字母缩写 |
    | `He finally took up the question.` | `False` | 句末标点不算 |
    """
    for m in _FRAGMENT_CANDIDATE_RE.finditer(text or ""):
        low = m.group(1).lower().rstrip(".'’")
        if low in _ABBREV_TITLES:
            continue
        if len(low) <= 1:  # 首字母缩写：`J. Smith`
            continue
        return True
    return False


def precheck(raw: str) -> list[str]:
    """判定是否需要修复，返回 `reasons[]`（空列表 = 无需修复）。

    | reason | 判据 |
    |---|---|
    | `fragmented` | 句中（非末尾）出现终结标点且其后仍有内容 |
    | `repeated` | 存在紧邻重复 token，且**不在** `GRAMMATIC_REDUP_WORDS` 内 |
    | `unpunctuated` | 末尾无终结标点 |
    | `whitespace` | 存在连续空白或首尾空白 |
    """
    text = raw or ""
    reasons: list[str] = []

    if has_fragment(text):
        reasons.append("fragmented")

    tokens = norm_tokens(text)
    for i in range(len(tokens) - 1):
        if tokens[i] == tokens[i + 1] and tokens[i] not in GRAMMATIC_REDUP_WORDS:
            reasons.append("repeated")
            break

    stripped = text.strip()
    # 先剥掉句末可能的收尾引号/括号，再看真正的句末字符
    tail = stripped.rstrip(_TRAILING_WRAP)
    if tail and tail[-1] not in _TERMINAL:
        reasons.append("unpunctuated")

    if text != _WS_RE.sub(" ", text).strip():
        reasons.append("whitespace")

    return reasons


def needs_llm(reasons: list[str]) -> bool:
    """是否必须动用 LLM：只有「断句」与「紧邻重复」是规则层修不了的。"""
    return "fragmented" in reasons or "repeated" in reasons


# ===========================================================================
# Tier 1：规则快修（零 token）
# ===========================================================================


def rule_fix_text(raw: str) -> str:
    """空白归一 + 首字母大写 + 句末补标点。

    复用 `services/infra/text_normalize.smart_punctuate`（与 `/eval/transcribe`
    同一实现，但**不受** `ASR_SMART_PUNCTUATION` 开关影响 —— 那个开关管挂载点，
    本处是直接调函数）。
    """
    text = raw or ""
    if not text.strip():
        return text
    return smart_punctuate(text) or text


# ===========================================================================
# 护栏（R15 的结构性保证）
# ===========================================================================


def guard(raw: str, repaired: str) -> tuple[bool, str]:
    """修复结果 5 判据校验。任一不过 → 拒绝（调用方回落）。

    | # | 判据 | 拒绝码 |
    |---|---|---|
    | G1 | 修复文本非空、且含至少一个词 | `G1_EMPTY` |
    | G2 | 修复文词序列是原文词序列的**子序列** | `G2_NOT_SUBSEQUENCE` |
    | G3 | 删除**有据**：被删词 ⊆（紧邻重复富余 + 填充词）；且净删除比例 ≤ `MAX_DROP_RATIO` | `G3_UNJUSTIFIED_DELETION` / `G3_DROP_TOO_MUCH` |
    | G4 | 未破坏 `GRAMMATIC_REDUP_WORDS` 的紧邻重复 | `G4_GRAMMATIC_REDUP_BROKEN` |
    | G5 | 长度 ≤ `EXTENSION_ROUND_MAX_INPUT_LEN` | `G5_TOO_LONG` |

    **G2 是本方案的基石**：子序列天然蕴含「未加词（长度不增）、未换词（元素同源）、
    未重排（顺序一致）」，把"不自动纠正、不加词"从 prompt 祈求升级为代码可证。

    **G3 为什么必须按"是否有据"判、而不是按删除比例判**：短句里删一个结巴就是高比例 ——
    `he he finally` → `He finally.` 净删 33%，若只卡比例（0.3）会把**正确的**去重一起拒掉。
    故主判据改为"每个被删词是否落在可删额度内"，比例只作兜底。
    """
    rep = (repaired or "").strip()
    if not rep:
        return False, "G1_EMPTY"
    if len(rep) > config.EXTENSION_ROUND_MAX_INPUT_LEN:
        return False, "G5_TOO_LONG"

    raw_tokens = norm_tokens(raw)
    rep_tokens = norm_tokens(rep)
    if not rep_tokens:
        return False, "G1_EMPTY"

    if not _is_subsequence(rep_tokens, raw_tokens):
        return False, "G2_NOT_SUBSEQUENCE"

    if raw_tokens:
        # G3① 删除必须有据：被删的每个词都得落在「紧邻重复富余 + 填充词」额度内。
        dropped = Counter(raw_tokens) - Counter(rep_tokens)
        if dropped:
            allowance = deletion_allowance(raw_tokens)
            if any(dropped[w] > allowance[w] for w in dropped):
                return False, "G3_UNJUSTIFIED_DELETION"
        # G3② 兜底：净删除比例上限（防整句被吞掉一半以上）
        drop_ratio = 1 - (len(rep_tokens) / len(raw_tokens))
        if drop_ratio > config.EXTENSION_ROUND_ASR_REPAIR_MAX_DROP_RATIO:
            return False, "G3_DROP_TOO_MUCH"

    for word in GRAMMATIC_REDUP_WORDS:
        if _adjacent_dup_count(rep_tokens, word) < _adjacent_dup_count(
            raw_tokens, word
        ):
            return False, "G4_GRAMMATIC_REDUP_BROKEN"

    return True, ""


# ===========================================================================
# 图状态
# ===========================================================================


class AsrRepairState(TypedDict, total=False):
    # 输入
    raw_input: str
    context: dict[str, Any]
    # precheck
    reasons: list[str]
    needs_repair: bool
    need_llm: bool
    # Tier 1
    rule_fixed: str
    # Tier 2
    llm_repaired: str
    dropped: list[str]
    repair_error: str
    # guard
    guard_ok: bool
    guard_reason: str
    # 输出
    repaired_input: str
    source: str
    applied: bool
    fallback: bool
    asr_repair: dict[str, Any]


# ===========================================================================
# 节点
# ===========================================================================


async def precheck_node(state: AsrRepairState) -> dict:
    """Node 1：Tier 0 预检（零 token）。"""
    reasons = precheck(state.get("raw_input", ""))
    return {
        "reasons": reasons,
        "needs_repair": bool(reasons),
        "need_llm": needs_llm(reasons),
    }


async def rule_fix_node(state: AsrRepairState) -> dict:
    """Node 2：Tier 1 规则快修（零 token）。"""
    return {"rule_fixed": rule_fix_text(state.get("raw_input", ""))}


async def llm_repair_node(state: AsrRepairState) -> dict:
    """Node 3：Tier 2 LLM 精修。

    ⚠️ 与 `extension_graph` 同款约定：LLM 失败**写 error 字段、不抛出**，
       由 `guard` → `fallback` 承接，绝不阻断判分（R16）。
    """
    raw = state.get("raw_input", "")
    try:
        messages = build_asr_repair_messages(raw, context=state.get("context") or {})
        content = await call_asr_repair_llm(messages)
        parsed = parse_asr_repair(content)
    except (ExtensionError, AsrRepairError) as e:
        logger.warning("[asr_repair] 修复 LLM 失败: %s", e.error_code)
        return {"llm_repaired": "", "dropped": [], "repair_error": e.error_code}
    except Exception as e:  # noqa: BLE001
        logger.error("[asr_repair] 修复异常: %s", e, exc_info=True)
        return {
            "llm_repaired": "",
            "dropped": [],
            "repair_error": "PROVIDER_UNAVAILABLE",
        }

    return {
        "llm_repaired": parsed["repaired"],
        "dropped": parsed["dropped"],
        "repair_error": "",
    }


async def guard_node(state: AsrRepairState) -> dict:
    """Node 4：护栏校验（纯函数，零 token）。"""
    raw = state.get("raw_input", "")
    err = state.get("repair_error", "")
    repaired = state.get("llm_repaired", "")

    if err or not repaired:
        # LLM 失败 → 视为"未产出可用修复"，交给 fallback
        return {"guard_ok": False, "guard_reason": err or "LLM_NO_OUTPUT"}

    ok, reason = guard(raw, repaired)
    if not ok:
        logger.warning("[asr_repair] 护栏拒绝 → reason=%s", reason)
    return {"guard_ok": ok, "guard_reason": reason}


async def fallback_node(state: AsrRepairState) -> dict:
    """Node 5：回落。基线取规则快修结果（它最多只改标点，永远安全）。"""
    raw = state.get("raw_input", "")
    baseline = state.get("rule_fixed") or rule_fix_text(raw)
    if not baseline.strip():
        baseline = raw
    return {"llm_repaired": baseline, "dropped": [], "guard_ok": False}


async def finalize_node(state: AsrRepairState) -> dict:
    """Node 6：收敛出最终文本与元信息。**唯一**决定 `source` / `applied` 的地方。"""
    raw = state.get("raw_input", "") or ""
    reasons = list(state.get("reasons") or [])
    needs = bool(state.get("needs_repair"))
    need_llm_flag = bool(state.get("need_llm"))
    err = state.get("repair_error", "") or ""
    guard_reason = state.get("guard_reason", "") or ""

    if not needs:
        repaired, source, is_fallback = raw, SOURCE_NONE, False
    elif not need_llm_flag:
        repaired = state.get("rule_fixed") or rule_fix_text(raw) or raw
        source, is_fallback = SOURCE_RULE, False
    elif not state.get("guard_ok", False):
        repaired = state.get("llm_repaired") or state.get("rule_fixed") or raw
        source, is_fallback = SOURCE_FALLBACK, True
    else:
        repaired = state.get("llm_repaired") or raw
        source, is_fallback = SOURCE_LLM, False

    if not (repaired or "").strip():
        repaired = raw

    applied = bool((repaired or "").strip()) and repaired.strip() != raw.strip()

    meta: dict[str, Any] = {
        "applied": applied,
        "source": source,
        "fallback": is_fallback,
        "reasons": reasons,
        "dropped": dropped_tokens(raw, repaired) if applied else [],
        "prompt_version": PROMPT_VERSION,
    }
    # 观测字段：仅在非空时下发，避免干净路径的载荷噪音
    if guard_reason:
        meta["guard_reason"] = guard_reason
    if err:
        meta["error"] = err

    logger.info(
        f"[asr_repair] 收敛 → source={source}, applied={applied}, "
        f"reasons={reasons}, fallback={is_fallback}"
    )
    return {"repaired_input": repaired, "source": source, "applied": applied,
            "fallback": is_fallback, "asr_repair": meta}


# ===========================================================================
# 条件边
# ===========================================================================


def _route_after_precheck(state: AsrRepairState) -> str:
    if not state.get("needs_repair"):
        return "finalize"
    if state.get("need_llm"):
        return "llm"
    return "rule"


def _route_after_guard(state: AsrRepairState) -> str:
    return "finalize" if state.get("guard_ok") else "fallback"


# ===========================================================================
# 图编译
# ===========================================================================


def build_asr_repair_graph():
    """构建并编译 ASR 判前修复图。"""
    workflow = StateGraph(AsrRepairState)
    workflow.add_node("precheck", precheck_node)
    workflow.add_node("rule_fix", rule_fix_node)
    workflow.add_node("llm_repair", llm_repair_node)
    workflow.add_node("guard", guard_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("finalize", finalize_node)

    workflow.add_edge(START, "precheck")
    workflow.add_conditional_edges(
        "precheck",
        _route_after_precheck,
        {"finalize": "finalize", "rule": "rule_fix", "llm": "llm_repair"},
    )
    workflow.add_edge("rule_fix", "finalize")
    workflow.add_edge("llm_repair", "guard")
    workflow.add_conditional_edges(
        "guard",
        _route_after_guard,
        {"finalize": "finalize", "fallback": "fallback"},
    )
    workflow.add_edge("fallback", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()


# 模块级编译实例（复用）
_asr_repair_graph = build_asr_repair_graph()


def _disabled_meta() -> dict[str, Any]:
    """关开关时的元信息（R18：调用方据此不加字段，故主体为 None）。"""
    return {
        "applied": False,
        "source": SOURCE_DISABLED,
        "fallback": False,
        "reasons": [],
        "dropped": [],
        "prompt_version": PROMPT_VERSION,
    }


async def run_asr_repair(raw: str, *, context: dict | None = None) -> dict:
    """执行 ASR 判前修复，返回 `{"repaired_input", "asr_repair"}`。

    Args:
        raw: 原始 ASR 转写文本（可能为空）。
        context: 可选语境 `{prompt_zh, must_use, ...}`，**只作意图参考、不作对齐模板**。

    Returns:
        - 开关**关**（R18）→ `{"repaired_input": None, "asr_repair": None}`；
          调用方（B04）据此**不**往返回体加字段 ⇒ 今日行为逐字不变。
        - 其余情况 → `repaired_input` 恒为可用文本（不可能是空串），
          `asr_repair` 含 `applied` / `source` / `fallback` / `reasons` / `dropped`。

    保证：**永不抛异常**。任何意外都回落为原文继续判分（R16）。
    """
    text = raw or ""

    if not config.EXTENSION_ROUND_ASR_REPAIR_ENABLED:
        return {"repaired_input": None, "asr_repair": None}

    if not text.strip():
        return {"repaired_input": text, "asr_repair": _disabled_meta() | {
            "source": SOURCE_NONE}}

    try:
        state: AsrRepairState = {"raw_input": text, "context": context or {}}
        result = await _asr_repair_graph.ainvoke(state)
        repaired = result.get("repaired_input")
        if not (repaired or "").strip():
            repaired = text
        return {
            "repaired_input": repaired,
            "asr_repair": result.get("asr_repair") or _disabled_meta(),
        }
    except Exception as e:  # noqa: BLE001
        logger.error("[asr_repair] 子图异常，回落原文: %s", e, exc_info=True)
        return {
            "repaired_input": text,
            "asr_repair": {
                "applied": False,
                "source": SOURCE_FALLBACK,
                "fallback": True,
                "reasons": [],
                "dropped": [],
                "prompt_version": PROMPT_VERSION,
                "error": "PROVIDER_UNAVAILABLE",
            },
        }


__all__ = [
    "GRAMMATIC_REDUP_WORDS",
    "AsrRepairState",
    "build_asr_repair_graph",
    "dropped_tokens",
    "guard",
    "norm_tokens",
    "precheck",
    "rule_fix_text",
    "run_asr_repair",
]
