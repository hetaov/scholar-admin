"""AI 对话生成域（批量面 `/ai/dialogue/v1`）生成引擎 — v2 调整稿。

设计稿：`docs_v1/AI会话/AI英语对话生成设计-调整稿（每句学习语句扇出评估·自动选优·背景与角色后置）.md`

职责：根据任务 `context`（自包含快照，§6.1），承担两类入口：
1. **T1 单函数直连**（`generate_dialogue`，`DIALOGUE_GEN_GRAPH_ENABLED=0` 时回退）：一次 LLM 出全量；
2. **T3 LangGraph 节点函数**（`build_per_sentence_evaluate_messages` /
   `parse_per_sentence_evaluation` / `build_summarize_messages` / `parse_summary` /
   `select_best` 等）：由 `services/learning/dialogue_gen_graph.py` 编排。

v2 调整要点（替代 v1 场景候选方案）：
- 取消候选场景机制；
- 角色不再页面预设，改由 `summarize` 节点对选中 turns 后置反推；
- 流程：`load_context → Send(evaluate_per_sentence × N) → select_best → summarize →
  validate_coverage → (evaluate | refine | fallback) → evaluate → persist`；
- LangGraph 节点契约见 §3.1 / §3.2；选优纯函数见 §2.5（`select_best`）。

T1 边界（保持直连可用；不接图/不接 checkpoint/不接覆盖校验重试环/不接 L2 Judge）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable

from starlette.concurrency import run_in_threadpool

from config import (
    DIALOGUE_LLM_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_CHAT_MODEL,
)

logger = logging.getLogger("scholar-admin.dialogue_gen")

# LLM 输出 JSON 提取（容忍代码块包裹，同 session_gen / translation_eval）
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# 失败阶段（对齐 §4.16/§4.18 error 对象）
STAGE_LLM = "llm"
STAGE_PARSE = "parse"
STAGE_COVERAGE = "coverage"

# 错误码（设计稿 §5 error_code 枚举）
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
ERR_COVERAGE_FAILED = "COVERAGE_FAILED"
ERR_NETWORK_ERROR = "NETWORK_ERROR"

# 形态白名单（§4.5：A/B/C 对话优先，不适配回落非对话）
SUPPORTED_CONTENT_TYPES = ("dialogue", "non_dialogue")
SUPPORTED_PREFERRED_TYPES = ("auto", "dialogue", "non_dialogue")
NON_DIALOGUE_SUB_TYPES = ("retell", "fill", "task")
SUPPORTED_PROMPT_LANGS = ("zh", "en")

# 渲染进 prompt 的任务组句子上限（控 token）
MAX_RENDER_SENTENCES = 24

# 召回条数硬约束（§4.2：clamp 到 [2, 6]）
RECALL_MIN = 2
RECALL_MAX = 6

# v2 设计稿 §2.5: 选优策略（仅 natural_first / score_first；当前等价自然度最高）
DEFAULT_SELECT_STRATEGY = "natural_first"
SELECT_STRATEGIES = ("natural_first", "score_first")

# v2 设计稿 §3.1 / §3.2 输出契约约束
MAX_BACKGROUND_CHARS = 500
MIN_ROLES = 2
MAX_ROLES = 3
ROLE_CODES = ("A", "B", "C")
NATURALNESS_MIN = 0.0
NATURALNESS_MAX = 1.0

# 生成器签名：`async (messages) -> content | None`（单测经注入传 fake）
LLMGenerator = Callable[[list[dict]], Awaitable[str | None]]


class DialogueGenError(Exception):
    """对话生成业务失败（不降级，直接置任务 failed）。

    Attributes:
        error_code: LLM_TIMEOUT / LLM_UNAVAILABLE / LLM_PARSE_ERROR / COVERAGE_FAILED
        failure_stage: llm / parse / coverage
    """

    def __init__(self, error_code: str, failure_stage: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.failure_stage = failure_stage
        self.detail = detail

    def to_dict(self, llm_timeout_seconds: int | None = None, raw=None) -> dict:
        """转为任务 error 对象（对齐 data-model-contract §4.16/§4.18）。"""
        return {
            "error_code": self.error_code,
            "error_detail": self.detail,
            "failure_stage": self.failure_stage,
            "llm_timeout_seconds": llm_timeout_seconds,
            "raw": raw,
        }


# ---------------------------------------------------------------------------
# 入参规整
# ---------------------------------------------------------------------------


def clamp_top_k(top_k: Any) -> int:
    """召回条数收敛到 [2, 6]（§4.2 需求硬约束）。非法值按缺省 4。"""
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 4
    return max(RECALL_MIN, min(RECALL_MAX, value))


def normalize_roles(raw_roles: Any) -> list[dict]:
    """角色表规整：仅保留含 `code` 的项，统一为 `{code, name, identity}`。"""
    roles: list[dict] = []
    for item in raw_roles or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        roles.append(
            {
                "code": code,
                "name": str(item.get("name") or code).strip(),
                "identity": str(item.get("identity") or "").strip(),
            }
        )
    return roles


def task_group_sentence_ids(task_group: dict) -> list[str]:
    """任务组句子 id 有序清单（去重保序；覆盖校验的必用集合）。"""
    ids: list[str] = []
    for s in (task_group or {}).get("sentences") or []:
        sid = str((s or {}).get("sentence_id") or "").strip()
        if sid and sid not in ids:
            ids.append(sid)
    return ids


def resolve_target_sentence(
    context: dict,
    target_sentence_id: str | None = None,
    result: dict | None = None,
) -> dict | None:
    """解析用户作答要对照的参考句（T5 用户作答评测）。

    优先级（保证「评的是学习者刚被引导的那句」）：
    1. 显式 `target_sentence_id`：必须命中任务组必用句，否则返回 None
       （调用方据此返回 INVALID_INPUT，不静默换句）；
    2. 生成结果 `turns` 中**最后一个**带 `target_sentence_id` 的必用句；
    3. 任务组首句。

    Returns:
        `{sentence_id, content}`；任务组无有效句子时 None。
    """
    sentences: dict[str, dict] = {}
    for item in (context or {}).get("task_group", {}).get("sentences") or []:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("sentence_id") or "").strip()
        content = str(item.get("content") or "").strip()
        if sid and content and sid not in sentences:
            sentences[sid] = {"sentence_id": sid, "content": content}

    explicit = str(target_sentence_id or "").strip()
    if explicit:
        return sentences.get(explicit)

    for turn in reversed((result or {}).get("turns") or []):
        sid = str((turn or {}).get("target_sentence_id") or "").strip()
        if sid and sid in sentences:
            return sentences[sid]

    return next(iter(sentences.values()), None)


def choose_content_type(preferred_type: str, roles: list[dict]) -> str:
    """形态选择（§4.5，T1 简化版）：

    - 角色缺失（<2）→ 强制 `non_dialogue`（多角色对话无从谈起）；
    - 否则按 `preferred_type`（auto/dialogue/non_dialogue），auto 取 `dialogue`。
    """
    if len(roles) < 2:
        return "non_dialogue"
    if preferred_type in SUPPORTED_CONTENT_TYPES:
        return preferred_type
    if preferred_type not in ("auto", None):
        logger.warning(
            "[dialogue_gen] preferred_type=%r 不在支持集，按 auto 兜底", preferred_type
        )
    return "dialogue"


# ---------------------------------------------------------------------------
# Prompt 构建（§4.4 重构要点 + v2 evaluate / summarize）
# ---------------------------------------------------------------------------


def _render_task_group(
    task_group: dict, *, exclude_sid: str | None = None
) -> str:
    """把任务组句子渲染为 prompt 文本（编号 + id + 原文，受上限约束）。

    Args:
        task_group: 任务组字典
        exclude_sid: 若给定则排除该 sentence_id（v2 evaluate_per_sentence 用于
                     "其余必用句仅作上下文，不在本分支写出"）。
    """
    lines: list[str] = []
    for i, s in enumerate((task_group or {}).get("sentences") or [], 1):
        if i > MAX_RENDER_SENTENCES:
            break
        content = str((s or {}).get("content") or "").strip()
        sid = str((s or {}).get("sentence_id") or "").strip()
        if not content or sid == (exclude_sid or ""):
            continue
        lines.append(f"{i}. [{sid}] {content}")
    return "\n".join(lines) if lines else "（无）"


def _render_recall(recalled: Any) -> str:
    """渲染召回学习语句（§4.2：跨课已学句，仅用于背景融合，非必用）。"""
    lines: list[str] = []
    for i, s in enumerate(recalled or [], 1):
        if not isinstance(s, dict):
            continue
        content = str(s.get("content") or s.get("text") or "").strip()
        sid = str(s.get("sentence_id") or "").strip()
        if not content:
            continue
        lines.append(f"{i}. [{sid}] {content}")
        if i >= RECALL_MAX:
            break
    return "\n".join(lines) if lines else "（无）"


def _render_missing(missing_ids: Any, required_ids: list[str]) -> str:
    """渲染上一轮遗漏的必用句（refine 重试时回灌，§4.3 覆盖校验环）。"""
    missing = [str(x) for x in (missing_ids or []) if str(x) in set(required_ids)]
    return "、".join(missing) if missing else "（无）"


def _render_roles(roles: list[dict]) -> str:
    """角色表渲染：`A = Tom（student）` 逐行。"""
    if not roles:
        return "（未提供，非对话形态）"
    lines = []
    for role in roles:
        suffix = f"（{role['identity']}）" if role["identity"] else ""
        lines.append(f"{role['code']} = {role['name']}{suffix}")
    return "\n".join(lines)


def build_dialogue_messages(
    context: dict,
    *,
    content_type: str,
    roles: list[dict],
    prompt_lang: str,
    recalled: Any = None,
    refine_missing: Any = None,
) -> list[dict]:
    """构建生成 messages（system + user 两段，§4.4）。T1/T3 直连复用。

    Args:
        context: 任务 context 快照（§6.1）：task_group/scenario/roles/recall/metrics/
                 prompt_lang/preferred_type
        content_type: 已定形态（dialogue / non_dialogue）
        roles: 规整后的角色表
        prompt_lang: 轮间提示语言（zh / en）
        recalled: 召回学习语句（§4.2，仅背景融合、非必用；缺省不注入）
        refine_missing: 上一轮遗漏的必用句 id（覆盖校验环 refine 回灌；缺省不注入）
    """
    lang = prompt_lang if prompt_lang in SUPPORTED_PROMPT_LANGS else "zh"
    scenario = context.get("scenario") or {}
    background = str(scenario.get("background") or "").strip()
    goal = str(scenario.get("goal") or "").strip()
    weak_skills = (context.get("metrics") or {}).get("weak_skills") or []
    task_group = context.get("task_group") or {}
    group_label = str(task_group.get("group_label") or task_group.get("group_id") or "").strip()

    if content_type == "dialogue":
        form_rule = (
            "编写一段**多角色英语对话**（A/B/C 你来我往）：整体 6~14 轮，口语自然、"
            "有情境推进；任务组每个句子都必须原样出现在某角色台词中，"
            "并在该轮用 target_sentence_id 标注。"
            'JSON 的 "content_type" 固定为 "dialogue"，"sub_type" 为 null。'
        )
    else:
        form_rule = (
            "任务组不适配多角色对话，改用**非对话形态**（retell 情景复述 / fill 补全 / "
            "task 任务应答三选一）：给出情境与提示，把任务组句子作为学习目标融入题目。"
            'JSON 的 "content_type" 固定为 "non_dialogue"，'
            '"sub_type" 取 retell / fill / task 之一且必填。'
        )

    system = (
        "你是资深英语教学对话编剧（服务《新概念英语2》学习者）。"
        "基于「任务组（必用）+ 会话背景 + 角色表」，产出可直接展示的英语学习内容。\n"
        "【硬约束】\n"
        "- 角色说话人用代码表示（A/B/C），遵守角色设定；\n"
        f"- {form_rule}\n"
        "- 背景不为空时，先给出一句 background_intro（说明情境），再进入正文；\n"
        f"- 轮间提示 prompts：每 2~3 轮可插入一条，语言为「{lang}」，"
        "引导学习者复述/用出目标句，**不得直接给出整句答案**；\n"
        "- 严禁编造任务组之外的学习目标句（used_sentence_ids/recalled_sentence_ids 只填真实出现过的 id）；\n"
        "- 只输出一个 JSON 对象，不要任何解释或代码块外文本。\n"
        "【输出 JSON 结构】\n"
        '{{"content_type": "{content_type}", "sub_type": null, "background_intro": "...", '
        '"turns": [{{"speaker": "A", "text": "...", "target_sentence_id": "<sid 或 null>"}}], '
        '"prompts": [{{"after_turn": 2, "lang": "{lang}", "text": "..."}}], '
        '"used_sentence_ids": ["<sid>"], "recalled_sentence_ids": [], '
        '"difficulty": 2, "notes": null}}'
    ).format(content_type=content_type, lang=lang)

    scene_parts = [p for p in (group_label, background) if p]
    if goal:
        scene_parts.append(f"目标：{goal}")
    weak_text = "、".join(str(x) for x in weak_skills) if weak_skills else "（无）"

    user_msg = (
        "【会话背景】{scene}\n"
        "【角色表】\n{roles}\n"
        "【任务组（必用）】\n{task_group}\n"
        "【召回学习语句（背景融合，非必用）】\n{recall}\n"
        "【弱项约束（仅用于难度决策，不写进台词）】{weak}\n"
        "请按上述 JSON 结构输出。"
    ).format(
        scene="；".join(scene_parts) if scene_parts else "（未提供，按通用学习场景）",
        roles=_render_roles(roles),
        task_group=_render_task_group(task_group),
        recall=_render_recall(recalled),
        weak=weak_text,
    )
    # refine 回灌：上一轮遗漏的必用句（§4.3 覆盖校验环）
    missing_text = _render_missing(refine_missing, task_group_sentence_ids(task_group))
    if missing_text != "（无）":
        user_msg += (
            "\n【上一轮遗漏（本必须原样出现，请补齐）】" + missing_text
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# v2 evaluate_per_sentence（每条学习语句 1 次 LLM，§3.1）
# ---------------------------------------------------------------------------


def build_per_sentence_evaluate_messages(
    *,
    context: dict,
    learning_sentence: dict,
    recalled: Any = None,
) -> list[dict]:
    """构建 evaluate_per_sentence 的 messages（v2 设计稿 §3.1）。

    Args:
        context: 任务 context 快照（§6.1）
        learning_sentence: `{sentence_id, content}` 单条学习语句
        recalled: 召回学习语句（仅作语境融合，非必用）
    """
    task_group = context.get("task_group") or {}
    ls_sid = str(learning_sentence.get("sentence_id") or "").strip()
    ls_content = str(learning_sentence.get("content") or "").strip()
    weak_skills = (context.get("metrics") or {}).get("weak_skills") or []
    weak_text = "、".join(str(x) for x in weak_skills) if weak_skills else "（无）"

    system = (
        "你是资深英语教学评估者（服务《新概念英语2》学习者）。\n"
        "判断「学习语句」能否在真实人际对话里自然地用出，"
        "并给出对应的对话草案与自然度评分。\n"
        "【硬约束】\n"
        "- 若学习语句为纯定义型 / 孤立语法陈述（无明显对话场景可承接）→ viable=False；\n"
        "- 若 viable=True，必须在 turns 中以 target_sentence_id 标注该学习句；\n"
        "- 召回学习语句为可融合的语境材料，非必用（recalled_used 可为空）；\n"
        "- speaker 代码暂时用 A/B/C 即可（最终角色由后续步骤产出）；\n"
        "- 只输出一个 JSON 对象，不要任何解释或代码块外文本。\n"
        "【输出 JSON 结构】\n"
        '{"viable": true, "recalled_used": ["<sid>"], '
        '"turns": [{"speaker": "A", "text": "...", "target_sentence_id": "<sid 或 null>"}], '
        '"naturalness": 0.85, "reason": "..."}'
    )

    user_msg = (
        "【学习语句】\n"
        "[{sid}] {content}\n"
        "【任务组（其余必用，仅作上下文，不在本分支写出）】\n{task_group}\n"
        "【可召回学习语句（语境融合，非必用）】\n{recall}\n"
        "【弱项约束（仅用于难度决策，不写进台词）】{weak}\n"
        "请按上述 JSON 结构输出。"
    ).format(
        sid=ls_sid or "unknown",
        content=ls_content or "（无）",
        task_group=_render_task_group(task_group, exclude_sid=ls_sid),
        recall=_render_recall(recalled),
        weak=weak_text,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# v2 summarize（1 次 LLM，§3.2）：选中 turns → {background, roles}
# ---------------------------------------------------------------------------


def build_summarize_messages(
    *,
    selected: dict,
    learning_sentence: dict | None = None,
    recalled: Any = None,
    task_group: dict | None = None,
) -> list[dict]:
    """构建 summarize 的 messages（v2 设计稿 §3.2）：对选中 turns 反推背景与角色。"""
    turns = selected.get("turns") or []
    turns_text_lines = []
    for i, t in enumerate(turns, 1):
        if not isinstance(t, dict):
            continue
        speaker = str(t.get("speaker") or "?")
        text = str(t.get("text") or "").strip()
        if not text:
            continue
        turns_text_lines.append(f"{i}. [{speaker}] {text}")
    turns_text = "\n".join(turns_text_lines) or "（空）"

    ls_text = ""
    if isinstance(learning_sentence, dict):
        ls_text = (
            f"[{str(learning_sentence.get('sentence_id') or '').strip()}] "
            f"{str(learning_sentence.get('content') or '').strip()}"
        )

    system = (
        "你是资深英语教学对话编剧（服务《新概念英语2》学习者）。\n"
        "【任务】基于给定的对话 turns，反推「一句中文背景 + 2~3 位角色」。\n"
        "【硬约束】\n"
        f"- background 中文，≤{MAX_BACKGROUND_CHARS} 字，说明时间/地点/情境与学习者的关系；\n"
        f"- roles 数量 ∈ [{MIN_ROLES}, {MAX_ROLES}]，code ∈ {list(ROLE_CODES)}，"
        "name/identity 由 turns 说话人反推；\n"
        "- 只输出一个 JSON 对象，不要任何解释或代码块外文本。\n"
        "【输出 JSON 结构】\n"
        '{"background": "...", '
        '"roles": [{"code": "A", "name": "Tom", "identity": "student"}]}'
    )

    user_msg = (
        "【对话 turns（待总结）】\n{turns}\n"
        "【对应学习语句】\n{learning_sentence}\n"
        "【可召回学习语句（仅用于语境，非必用）】\n{recall}\n"
        "【任务组（上下文）】\n{task_group}\n"
        "请按上述 JSON 结构输出。"
    ).format(
        turns=turns_text,
        learning_sentence=ls_text or "（无）",
        recall=_render_recall(recalled),
        task_group=_render_task_group(task_group) if task_group else "（无）",
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# 输出解析（容忍代码块包裹）
# ---------------------------------------------------------------------------


def _as_str_list(value: Any, *, limit: int | None = None) -> list[str]:
    """把任意值规整为去空字符串列表（保序去重，可选截断）。"""
    result: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
    return result[:limit] if limit is not None else result


def _parse_turns(raw_turns: Any, role_codes: list[str]) -> list[dict]:
    """解析 turns：保留 speaker/text 非空项；dialogue 形态校验说话人合法。"""
    turns: list[dict] = []
    if not isinstance(raw_turns, list):
        return turns
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker") or "").strip()
        text = str(item.get("text") or "").strip()
        if not speaker or not text:
            continue
        if role_codes and speaker not in role_codes:
            continue
        target = str(item.get("target_sentence_id") or "").strip() or None
        turns.append({"speaker": speaker, "text": text, "target_sentence_id": target})
    return turns


def _parse_prompts(raw_prompts: Any, *, default_lang: str) -> list[dict]:
    """解析轮间提示：`{after_turn, lang, text}`；lang 缺省用请求 prompt_lang。"""
    prompts: list[dict] = []
    if not isinstance(raw_prompts, list):
        return prompts
    for item in raw_prompts:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        try:
            after_turn = int(item.get("after_turn"))
        except (TypeError, ValueError):
            continue
        lang = str(item.get("lang") or default_lang).strip()
        if lang not in SUPPORTED_PROMPT_LANGS:
            lang = default_lang
        prompts.append({"after_turn": after_turn, "lang": lang, "text": text})
    return prompts


def parse_dialogue_output(
    content: str | None,
    *,
    required_ids: list[str],
    role_codes: list[str],
    prompt_lang: str,
    recall_ids: Any = None,
) -> dict | None:
    """解析模型输出 JSON 并规整为 §4.6 子结构（不含 coverage/metrics/session_id）。

    Args:
        content: 模型原始输出（容忍代码块包裹）
        required_ids: 任务组必用句 id（used_sentence_ids 的白名单）
        role_codes: 合法说话人代码（dialogue 形态校验）
        prompt_lang: 轮间提示缺省语言
        recall_ids: 召回候选句 id（recalled_sentence_ids 的白名单；缺省仅任务组内）

    Returns:
        dict | None（解析失败返回 None，由调用方转 LLM_PARSE_ERROR）
    """
    if not content:
        return None
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    content_type = str(parsed.get("content_type") or "").strip()
    if content_type not in SUPPORTED_CONTENT_TYPES:
        logger.warning(
            "[dialogue_gen] content_type=%r 不在支持集 → 解析失败", content_type
        )
        return None

    sub_type: str | None = None
    if content_type == "non_dialogue":
        candidate = str(parsed.get("sub_type") or "").strip()
        sub_type = candidate if candidate in NON_DIALOGUE_SUB_TYPES else "retell"

    turns = _parse_turns(parsed.get("turns"), role_codes)
    if not turns:
        return None
    if content_type == "dialogue" and len(turns) < 2:
        return None

    required_set = set(required_ids)
    # 只保留真实存在的句子 id（任务组集合内），杜绝编造 id
    used_ids = [sid for sid in _as_str_list(parsed.get("used_sentence_ids")) if sid in required_set]
    # recalled 白名单 = 任务组 ∪ 召回候选（无召回时仅任务组内，杜绝编造）
    recall_set = required_set | {str(x) for x in (recall_ids or [])}
    recalled_ids = [
        sid for sid in _as_str_list(parsed.get("recalled_sentence_ids")) if sid in recall_set
    ]

    try:
        difficulty = int(parsed.get("difficulty"))
    except (TypeError, ValueError):
        difficulty = None

    notes = str(parsed.get("notes") or "").strip() or None
    background_intro = str(parsed.get("background_intro") or "").strip() or None

    return {
        "content_type": content_type,
        "sub_type": sub_type,
        "background_intro": background_intro,
        "turns": turns,
        "prompts": _parse_prompts(parsed.get("prompts"), default_lang=prompt_lang),
        "used_sentence_ids": used_ids,
        "recalled_sentence_ids": recalled_ids,
        "difficulty": difficulty,
        "notes": notes,
    }


def parse_per_sentence_evaluation(
    content: str | None,
    *,
    learning_sentence_id: str,
    required_set: set[str],
    recall_set: set[str],
) -> dict | None:
    """解析 evaluate_per_sentence 输出（v2 设计稿 §3.1）。

    校验：
    - 严格 JSON，容忍代码块包裹；
    - `viable` 为 bool；
    - viable=True 时 `turns` 非空且**必须包含** `learning_sentence_id` 的 `target_sentence_id`；
    - `naturalness` ∈ [0, 1]；
    - `recalled_used` 仅允许 `recall_set` 内的 id；
    - turns 中所有 `target_sentence_id` 仅允许在 `required_set ∪ recall_set` 内（杜绝编造）。

    Returns:
        dict `{learning_sentence_id, viable, recalled_used, turns, naturalness, reason}` 或 None
    """
    if not content:
        return None
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    viable_raw = parsed.get("viable")
    if not isinstance(viable_raw, bool):
        return None

    result: dict = {
        "learning_sentence_id": str(learning_sentence_id or "").strip(),
        "viable": viable_raw,
    }

    if not viable_raw:
        result["reason"] = str(parsed.get("reason") or "").strip() or "unviable"
        result["recalled_used"] = []
        result["turns"] = []
        result["naturalness"] = 0.0
        return result

    # viable=True 分支：解析 turns
    raw_turns = parsed.get("turns")
    if not isinstance(raw_turns, list) or not raw_turns:
        return None

    whitelist = required_set | recall_set
    turns: list[dict] = []
    hit_learning = False
    for t in raw_turns:
        if not isinstance(t, dict):
            continue
        speaker = str(t.get("speaker") or "").strip()
        text = str(t.get("text") or "").strip()
        if not speaker or not text:
            continue
        target_raw = t.get("target_sentence_id")
        target: str | None = None
        if target_raw is not None:
            target = str(target_raw).strip() or None
            if target and target not in whitelist:
                return None  # 杜绝编造
            if target == result["learning_sentence_id"]:
                hit_learning = True
        turns.append({"speaker": speaker, "text": text, "target_sentence_id": target})

    if not turns or not hit_learning:
        return None

    # recalled_used 白名单校验
    recalled_used: list[str] = []
    for sid in _as_str_list(parsed.get("recalled_used")):
        if sid in recall_set and sid not in recalled_used:
            recalled_used.append(sid)

    # naturalness 范围校验
    try:
        naturalness = float(parsed.get("naturalness"))
    except (TypeError, ValueError):
        return None
    naturalness = round(max(NATURALNESS_MIN, min(NATURALNESS_MAX, naturalness)), 4)

    result["recalled_used"] = recalled_used
    result["turns"] = turns
    result["naturalness"] = naturalness
    result["reason"] = str(parsed.get("reason") or "").strip() or None
    return result


def parse_summary(content: str | None) -> dict | None:
    """解析 summarize 输出（v2 设计稿 §3.2）。

    校验：
    - 严格 JSON，容忍代码块包裹；
    - `background` 非空且 ≤ MAX_BACKGROUND_CHARS；
    - `roles` 长度 ∈ [MIN_ROLES, MAX_ROLES]、`code ∈ ROLE_CODES`、`name` 非空、code 唯一；

    Returns:
        dict `{background, roles}` 或 None。roles 长度不达标时返回 None
        （调用方走 `fallback_non_dialogue`，沿用 T3 语义）。
    """
    if not content:
        return None
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    background = str(parsed.get("background") or "").strip()
    if not background or len(background) > MAX_BACKGROUND_CHARS:
        return None

    raw_roles = parsed.get("roles")
    if not isinstance(raw_roles, list):
        return None

    roles: list[dict] = []
    seen_codes: set[str] = set()
    for item in raw_roles:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip().upper()
        name = str(item.get("name") or "").strip()
        identity = str(item.get("identity") or "").strip()
        if code not in ROLE_CODES or not name or code in seen_codes:
            continue
        roles.append({"code": code, "name": name, "identity": identity})
        seen_codes.add(code)

    if len(roles) < MIN_ROLES or len(roles) > MAX_ROLES:
        return None

    return {"background": background, "roles": roles}


def select_best(
    candidates: list[dict],
    *,
    strategy: str = DEFAULT_SELECT_STRATEGY,
) -> tuple[dict | None, dict]:
    """选优纯函数（v2 设计稿 §2.5）：

    - 仅 `viable=True` 的候选参与；
    - 默认 `natural_first`：naturalness 降序；同分按 turn_count（更丰富优先）；
    - `score_first`：当前等价 `natural_first`（预留多维扩展）；
    - 非法 strategy 按 `DEFAULT_SELECT_STRATEGY` 兜底。

    Args:
        candidates: evaluate_per_sentence 产出列表
        strategy: natural_first / score_first

    Returns:
        `(selected, selection)`；无可用候选时 `selected=None`。
    """
    viable = [c for c in (candidates or []) if c.get("viable")]
    effective_strategy = (
        strategy if strategy in SELECT_STRATEGIES else DEFAULT_SELECT_STRATEGY
    )
    if not viable:
        return None, {
            "strategy": effective_strategy,
            "reason": "no viable candidates",
            "selected_index": None,
            "naturalness": None,
            "turn_count": 0,
        }

    # natural_first 与 score_first 当前同实现：naturalness 降序，turn_count 降序保稳
    viable.sort(
        key=lambda c: (
            -float(c.get("naturalness") or 0.0),
            -len(c.get("turns") or []),
        )
    )
    chosen = viable[0]
    chosen_index: int | None = None
    for idx, c in enumerate(candidates or []):
        if c is chosen:
            chosen_index = idx
            break

    selection = {
        "strategy": effective_strategy,
        "reason": chosen.get("reason") or "naturalness highest",
        "selected_index": chosen_index,
        "naturalness": chosen.get("naturalness"),
        "turn_count": len(chosen.get("turns") or []),
    }
    return chosen, selection


# ---------------------------------------------------------------------------
# 覆盖统计 与 L1 指标（T1/T3 共用：只读统计，不做重试/降级）
# ---------------------------------------------------------------------------


def compute_coverage(required_ids: list[str], used_ids: list[str]) -> dict:
    """覆盖口径：任务组必用句命中数 / 总数（§4.6 `coverage`）。"""
    total = len(required_ids)
    used_set = set(used_ids)
    used = sum(1 for sid in required_ids if sid in used_set)
    ratio = round(used / total, 4) if total else 0.0
    return {"required_total": total, "required_used": used, "ratio": ratio}


def basic_metrics(coverage: dict, turn_count: int) -> dict:
    """本地 L1 规则指标（T3 接 L2 Judge 前的占位口径，标注 level=l1）。"""
    ratio = float(coverage.get("ratio") or 0.0)
    score = int(round(60 + 40 * ratio)) if turn_count else 0
    return {
        "score": score,
        "meaningful": turn_count >= 2,
        "faithfulness": ratio >= 0.5,
        "anomaly": False,
        "confidence": round(0.5 + 0.4 * ratio, 2),
        "level": "l1",
        "judge_model": None,
    }


def compute_missing_ids(required_ids: list[str], used_ids: list[str]) -> list[str]:
    """任务组必用句里「未被使用」的 id（覆盖校验环的 refine 依据，保序）。"""
    used_set = set(used_ids)
    return [sid for sid in required_ids if sid not in used_set]


def merge_notes(*notes: Any) -> str | None:
    """合并多来源 notes（状态列表 / 模型 notes 串）为契约字段（逗号分隔，去重保序）。"""
    merged: list[str] = []
    for item in notes:
        if item is None:
            continue
        candidates = item if isinstance(item, (list, tuple, set)) else str(item).split(",")
        for raw in candidates:
            text = str(raw or "").strip()
            if text and text not in merged:
                merged.append(text)
    return ",".join(merged) if merged else None


def build_non_dialogue_fallback(
    parsed: dict | None,
    *,
    required_sentences: list[dict],
    recalled_sentences: list[dict] | None = None,
    prompt_lang: str = "zh",
    background_intro: str | None = None,
    roles: list[dict] | None = None,
) -> dict:
    """覆盖重试用尽 / viable 全空后的**规则降级**：收拢为非对话（retell），保证可产出（§4.5）。

    以「情景复述」把全部必用句原样纳入题面（题面即出现 → 覆盖视为完整），
    不额外消耗 LLM 调用（单任务 LLM 上限 = 1 + MAX_RETRY）。
    v2：`roles` 可来自 `summarize` 产出或调用方传入，缺省取 `parsed['roles']`。
    """
    sentences = [s for s in (required_sentences or []) if str(s.get("content") or "").strip()]
    lines = [f"{i}. {str(s.get('content')).strip()}" for i, s in enumerate(sentences, 1)]
    header = (
        "Retell the following sentences in your own words:"
        if prompt_lang == "en"
        else "请用英语复述下列句子（可组织成一段连贯表达）："
    )
    instruction = header + "\n" + "\n".join(lines) if lines else header
    parsed = parsed or {}
    fallback_roles: list[dict] = []
    if roles:
        fallback_roles = list(roles)
    elif isinstance(parsed.get("roles"), list):
        fallback_roles = normalize_roles(parsed.get("roles"))

    return {
        "content_type": "non_dialogue",
        "sub_type": str(parsed.get("sub_type") or "retell"),
        "background_intro": background_intro or parsed.get("background_intro"),
        "roles": fallback_roles,
        "turns": [{"speaker": "T", "text": instruction, "target_sentence_id": None}],
        "prompts": list(parsed.get("prompts") or []),
        "used_sentence_ids": [str(s.get("sentence_id")) for s in sentences],
        "recalled_sentence_ids": [
            str(r.get("sentence_id")) for r in (recalled_sentences or []) if r.get("sentence_id")
        ],
        "difficulty": parsed.get("difficulty"),
        "notes": "fallback_non_dialogue",
    }


def build_result(
    parsed: dict,
    *,
    required_ids: list[str],
    roles: list[dict],
    retry_count: int = 0,
    metrics: dict | None = None,
    notes: Any = None,
    checkpoint: dict | None = None,
    summary: dict | None = None,
) -> dict:
    """把已校验/已评估的 parsed 组装为 §4.6 成功 result（图 persist 节点与直连共用）。

    v2：
    - 删除 `scene_candidates / selection / generation` 写入；
    - 新增 `summary: {background, roles}` 字段（由 `summarize` 节点产出后注入；
      缺省不落 key，前端按字段缺失处理）。
    """
    coverage = compute_coverage(required_ids, parsed.get("used_sentence_ids") or [])
    turn_count = len(parsed.get("turns") or [])
    result = {
        "session_id": "s_" + uuid.uuid4().hex,
        "content_type": parsed.get("content_type"),
        "sub_type": parsed.get("sub_type"),
        "background_intro": parsed.get("background_intro"),
        "roles": roles,
        "turns": parsed.get("turns") or [],
        "prompts": parsed.get("prompts") or [],
        "used_sentence_ids": parsed.get("used_sentence_ids") or [],
        "recalled_sentence_ids": parsed.get("recalled_sentence_ids") or [],
        "coverage": coverage,
        "retry_count": int(retry_count or 0),
        "checkpoint": checkpoint,
        "metrics": metrics or basic_metrics(coverage, turn_count),
        "notes": merge_notes(notes, parsed.get("notes")),
    }
    if summary is not None:
        result["summary"] = summary
    return result


# ---------------------------------------------------------------------------
# 单次 LLM 调用（图 generate / evaluate_per_sentence / summarize 节点共用）
# ---------------------------------------------------------------------------


def _call_dialogue_llm_sync(messages: list[dict], temperature: float = 0.8) -> str | None:
    """同步调用火山方舟对话模型；凭据缺失 / 调用失败返回 None。"""
    if not (VOLCANO_API_KEY and VOLCANO_CHAT_MODEL):
        logger.warning("[dialogue_gen] 未配置火山方舟凭据，无法生成")
        return None
    import requests

    resp = requests.post(
        f"{VOLCANO_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": VOLCANO_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=DIALOGUE_LLM_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        logger.error(
            "[dialogue_gen] 火山方舟返回 %s: %s", resp.status_code, resp.text[:200]
        )
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def _default_generator(messages: list[dict]) -> str | None:
    """默认生成器：同步请求丢线程池（超时由 `generate_dialogue` 外层 wait_for 兜底）。"""
    return await run_in_threadpool(_call_dialogue_llm_sync, messages)


async def invoke_dialogue_llm(
    messages: list[dict],
    *,
    generator: LLMGenerator | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """单次调用 LLM 并返回原始文本（T3 图所有节点复用，直连路径亦复用）。

    统一超时/空值语义（避免图与直连两套口径漂移）：
    - 超过 `timeout_seconds`（缺省 `DIALOGUE_LLM_TIMEOUT_SECONDS`）→ `LLM_TIMEOUT`；
    - 返回空（凭据缺失 / 调用失败 / 空串）→ `LLM_UNAVAILABLE`。
    """
    gen = generator or _default_generator
    timeout = timeout_seconds or DIALOGUE_LLM_TIMEOUT_SECONDS
    try:
        content = await asyncio.wait_for(gen(messages), timeout=timeout)
    except asyncio.TimeoutError:
        logger.error("[dialogue_gen] LLM 调用超过 %ss 未返回 → LLM_TIMEOUT", timeout)
        raise DialogueGenError(
            ERR_LLM_TIMEOUT,
            STAGE_LLM,
            f"LLM 调用超过 {timeout}s 未返回（DIALOGUE_LLM_TIMEOUT_SECONDS={timeout}）",
        )

    if not content:
        raise DialogueGenError(
            ERR_LLM_UNAVAILABLE, STAGE_LLM, "LLM 调用失败（模型不可用或返回空）"
        )
    return str(content)


# ---------------------------------------------------------------------------
# 对外生成入口（T1 单函数直连；不复用 v2 evaluate/summarize 链路）
# ---------------------------------------------------------------------------


async def generate_dialogue(
    *,
    context: dict,
    preferred_type: str = "auto",
    timeout_seconds: int | None = None,
    generator: LLMGenerator | None = None,
) -> dict:
    """直连生成对话内容（T1 单函数，不接图）：形态选择 → LLM → 解析 → 覆盖/指标。

    Args:
        context: 任务 context 快照（§6.1）
        preferred_type: auto / dialogue / non_dialogue（非法值按 auto 兜底）
        timeout_seconds: 缺省 `DIALOGUE_LLM_TIMEOUT_SECONDS`
        generator: 可注入生成器（单测传 fake；缺省走火山方舟）

    Returns:
        §4.6 结构（`checkpoint=None`、`retry_count=0`）

    Raises:
        DialogueGenError: LLM_UNAVAILABLE / LLM_TIMEOUT / LLM_PARSE_ERROR / COVERAGE_FAILED
    """
    task_group = context.get("task_group") or {}
    required_ids = task_group_sentence_ids(task_group)
    if not required_ids:
        raise DialogueGenError(
            ERR_COVERAGE_FAILED, STAGE_COVERAGE, "task_group 无可校验句子（sentences 为空）"
        )

    roles = normalize_roles(context.get("roles"))
    raw_lang = str(context.get("prompt_lang") or "zh").strip().lower()
    prompt_lang = raw_lang if raw_lang in SUPPORTED_PROMPT_LANGS else "zh"
    content_type = choose_content_type(preferred_type, roles)

    messages = build_dialogue_messages(
        context, content_type=content_type, roles=roles, prompt_lang=prompt_lang
    )

    content = await invoke_dialogue_llm(
        messages, generator=generator, timeout_seconds=timeout_seconds
    )

    parsed = parse_dialogue_output(
        content,
        required_ids=required_ids,
        role_codes=[r["code"] for r in roles],
        prompt_lang=prompt_lang,
    )
    if parsed is None:
        raise DialogueGenError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, f"模型输出解析失败: {content[:200]}"
        )

    coverage = compute_coverage(required_ids, parsed["used_sentence_ids"])
    # 覆盖不足只留痕（T3 才引入重试/降级环），不静默改写
    notes = parsed["notes"]
    if coverage["required_used"] < coverage["required_total"] and not notes:
        notes = "coverage_incomplete"

    result = {
        "session_id": "s_" + uuid.uuid4().hex,
        "content_type": parsed["content_type"],
        "sub_type": parsed["sub_type"],
        "background_intro": parsed["background_intro"],
        "roles": roles,
        "turns": parsed["turns"],
        "prompts": parsed["prompts"],
        "used_sentence_ids": parsed["used_sentence_ids"],
        "recalled_sentence_ids": parsed["recalled_sentence_ids"],
        "coverage": coverage,
        "retry_count": 0,
        "checkpoint": None,
        "metrics": basic_metrics(coverage, len(parsed["turns"])),
        "notes": notes,
    }
    logger.info(
        "[dialogue_gen] generate → content_type=%s, turns=%d, coverage=%d/%d, notes=%s",
        result["content_type"],
        len(result["turns"]),
        coverage["required_used"],
        coverage["required_total"],
        result["notes"],
    )
    return result
