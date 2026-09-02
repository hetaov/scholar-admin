"""沉浸式 AI 会话 v2 生成引擎（proposal 2026-09-02 / api-contract §3.12：不降级、不静默）

职责：根据任务 context（自包含快照，§4.18）生成
`{ content_type, ai_text, hint, suggested_targets }`。

- content_type：`dialogue`（角色扮演对话）/ `fill`（半成品对话补全，目标句空位由学习者补上）。
  `preferred_type=auto` 按 §11 启发式：**存在复习素材且复习句数 ≥ 新句 → fill，其余 → dialogue**。
  `retell`/`task` 为后续扩展点：显式请求由提交接口返回 `TYPE_NOT_SUPPORTED`
  （本模块只接收 auto/dialogue/fill）。
- hint：按「本轮意图引导用出的素材句」（suggested_targets）由 LLM **一次性生成 L1~L3**
  （§5.2/§11：L1 词义/词块线索 → L2 句式骨架 → L3 中英对照引导），客户端按
  「卡住 ≥15s 自动 / 💡 帮帮我」逐级展示；**L4「查看答案」为客户端本地动作，不经本接口**。
- suggested_targets：本轮意图引导用出的素材句 id（≤1，须在 context 素材句集合内，非法即剔除）。
- 生成模型：火山方舟 `VOLCANO_CHAT_MODEL`（与 /match/dialogue 生成同源，§4.18）；
  Judge=混元不变（`scripts/ai_session_eval.py`）。
- 错误语义（`SessionGenError`，error_code 枚举见 api-contract §3.12）：
  - `LLM_TIMEOUT`：`asyncio.wait_for` 达到 `SESSION_LLM_TIMEOUT_SECONDS`（默认 300s）强制取消；
  - `EVAL_UNAVAILABLE`：凭据缺失 / 调用失败 / 返回空；
  - `LLM_PARSE_ERROR`：输出非法 JSON / 字段缺失 / content_type 不在支持集合。
  - `NETWORK_ERROR` 由任务执行器（session_task）对非生成域异常通用兜底。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from config import (
    SESSION_LLM_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_CHAT_MODEL,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("scholar-admin.session_gen")

# LLM 输出 JSON 提取（容忍代码块包裹，同 translation_eval）
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# 失败阶段（对齐 §4.16/§4.18 error 对象）
STAGE_LLM = "llm"
STAGE_PARSE = "parse"

# 错误码（api-contract §3.12 error 枚举；NETWORK_ERROR 由任务执行器对非生成域异常兜底）
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_EVAL_UNAVAILABLE = "EVAL_UNAVAILABLE"
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
ERR_NETWORK_ERROR = "NETWORK_ERROR"

# 形态白名单（MVP：docs_v1 §11 收敛 auto/dialogue/fill）
SUPPORTED_CONTENT_TYPES = ("dialogue", "fill")

# 开场/每轮渲染进 prompt 的素材句上限（控 token：doubao 32k 上下文留足输出余量）
MAX_RENDER_SENTENCES = 24

# 提示档位上限（§5.2：L1~L3；L4 查看答案为客户端本地动作）
HINT_MAX_LEVEL = 3
HINT_LEVELS_COUNT = 3


class SessionGenError(Exception):
    """会话生成业务失败（不降级，直接置任务 failed）。

    Attributes:
        error_code: LLM_TIMEOUT / EVAL_UNAVAILABLE / LLM_PARSE_ERROR
        failure_stage: llm / parse
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
# 形态选择
# ---------------------------------------------------------------------------


def _group_kind(group: dict) -> str:
    """素材组 kind：`review` / `new`（缺省 new，对齐契约入参默认）。"""
    return str(group.get("kind") or "new")


def count_review_new(materials: list) -> tuple[int, int]:
    """统计素材组内句子数：(review_count, new_count)。

    kind 缺省按 new 计；group.sentences 缺省按空组计（不入计数）。
    """
    review = new = 0
    for group in materials or []:
        sentences = group.get("sentences") or []
        if _group_kind(group) == "review":
            review += len(sentences)
        else:
            new += len(sentences)
    return review, new


def choose_content_type(materials: list, preferred_type: str = "auto") -> str:
    """内容形态选择（§11 启发式，MVP 收敛 dialogue/fill）。

    - preferred_type 显式指定（dialogue/fill）→ 直接采用；
    - auto：存在复习素材且复习句数 ≥ 新句数 → fill（巩固负担低），其余 → dialogue；
    - 其它取值（理论上被路由 TYPE_NOT_SUPPORTED 拦截）→ 记日志并按 auto 兜底，
      不静默改写显式形态（日志可见）。
    """
    if preferred_type in SUPPORTED_CONTENT_TYPES:
        return preferred_type
    if preferred_type not in ("auto", None):
        logger.warning(f"[session_gen] preferred_type={preferred_type} 不在 MVP 支持集，按 auto 兜底")
    review, new = count_review_new(materials)
    if review > 0 and review >= new:
        logger.info(f"[session_gen] auto → fill（review={review} ≥ new={new}）")
        return "fill"
    logger.info(f"[session_gen] auto → dialogue（review={review} < new={new} 或无复习）")
    return "dialogue"


def collect_sentence_ids(materials: list) -> set[str]:
    """收集素材句 id 集合（suggested_targets 合法性校验用）。"""
    ids: set[str] = set()
    for group in materials or []:
        for s in group.get("sentences") or []:
            if s.get("sentence_id"):
                ids.add(str(s["sentence_id"]))
    return ids


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------


def _fmt_role(role: dict, default_name: str) -> str:
    """角色卡单行格式化（name + identity/style/goal 等附加字段）。"""
    name = str((role or {}).get("name") or default_name)
    extras = []
    for key in ("identity", "style", "goal"):
        if (role or {}).get(key):
            extras.append(f"{key}: {(role or {})[key]}")
    return f"{name}" + (f"（{'；'.join(extras)}）" if extras else "")


def _render_sentences(materials: list) -> str:
    """把素材组渲染为 prompt 文本（kind 标注 + 句子清单，受 MAX_RENDER_SENTENCES 上限约束）。

    新句在前（开场须引新句情境），复习句在后（作埋伏引子）；超限丢弃尾部。
    """
    lines: list[str] = []
    order: list[dict] = []
    for group in materials or []:
        kind = _group_kind(group)
        for s in group.get("sentences") or []:
            if not str(s.get("content") or "").strip():
                continue
            order.append({"kind": kind, "sid": s.get("sentence_id"), "content": str(s["content"]).strip()})
    order = order[:MAX_RENDER_SENTENCES]
    for i, item in enumerate(order, 1):
        kind_label = "复习" if item["kind"] == "review" else "新学"
        sid = f"[{item['sid']}] " if item.get("sid") else ""
        lines.append(f"{i}. （{kind_label}）{sid}{item['content']}")
    return "\n".join(lines) if lines else "（无素材）"


def _render_history(history: list, ai_name: str, learner_name: str) -> str:
    """把会话历史渲染为剧本对白（start 无历史 → 空串）。"""
    if not history:
        return ""
    lines = []
    for h in history:
        role = h.get("role")
        text = str(h.get("text") or "").strip()
        if not text:
            continue
        speaker = ai_name if role == "ai" else learner_name
        suffix = "（借助提示作答）" if h.get("assisted") else ""
        lines.append(f"{speaker}：{text}{suffix}")
    return "\n".join(lines)


def build_session_messages(context: dict, content_type: str) -> list[dict]:
    """构建会话生成 prompt（system + user 两段）。

    Args:
        context: 任务 context 快照（§4.18）：scenario/roles/materials(groups)/history/
                 user_input/assisted
        content_type: 已定形态（dialogue / fill）
    """
    scenario = context.get("scenario") or {}
    roles = context.get("roles") or {}
    ai_role = _fmt_role(roles.get("ai_role"), "AI")
    learner_role = _fmt_role(roles.get("learner_role"), "学习者")
    mode = "start" if context.get("mode") == "start" else "turn"
    user_input = str(context.get("user_input") or "").strip()
    assisted = bool(context.get("assisted"))
    history = context.get("history") or []
    scene_title = str(scenario.get("title") or "").strip()
    scene_desc = str(scenario.get("scene") or "").strip()
    goal = str(scenario.get("goal") or "").strip()
    constraints = str(scenario.get("constraints") or "").strip()

    scene_parts = [p for p in (scene_title, scene_desc) if p]
    if goal:
        scene_parts.append(f"目标：{goal}")
    if constraints:
        scene_parts.append(f"约束：{constraints}")

    material_text = _render_sentences(context.get("materials") or [])
    history_text = _render_history(history, ai_role.split("（")[0], learner_role.split("（")[0])

    if mode == "start":
        prompt_instruction = (
            "当前为会话开场（无历史）。请按所选形态展开开场情境："
            "用场景口吻开场并埋设情境缺口，引诱学习者用出第一个「新学」目标句；"
            "开场文本中不直接出现任何句子原文。"
        )
    else:
        assisted_mark = "（本轮学习者借助了提示卡）" if assisted else ""
        prompt_instruction = (
            f"当前输入为学习者上一条作答 —— {user_input}{assisted_mark}。\n"
            "请判断其是否用出了上一轮引导的目标句（对照会话记录与素材）：\n"
            "- 已自然用出 → 以角色身份肯定并收束该句，然后开启下一目标句的情境缺口；\n"
            "- 明显跑偏（不含任何目标词块）→ 回复中给 L1 词义线索 + 纠正方向，引导同一句；\n"
            "- 内容空或无法判断 → 顺着话题自然推进并留出可接话空间。"
        )

    system = (
        "你是沉浸式英语会话导演兼「{ai_role}」角色的扮演者。目标：通过沉浸式角色扮演，"
        "引导学习者「{learner_role}」自然用出本轮目标句（新学/复习句），巩固英语表达。"
        "你始终以角色身份说话，不跳出人设做语法点评（语法点评留给会话后报告）。\n"
        "【规则】\n"
        "- 不直接展示任何句子原文作为引导（「查看答案」是客户端本地动作，你绝不提供整句答案式提示）；\n"
        "- 每轮最多引导 1 个目标句；新学句与复习句不要在连续 2 轮内同时压给学习者；\n"
        "- 复习句必须能被场景合理容纳，容纳不了就不强行融入；\n"
        "- 开场与推进要制造「情境缺口」，让目标句成为学习者达成目标唯一自然的表达。\n"
        "【提示分级】按本轮目标句一次性生成 L1~L3（学习者卡住时逐级展示，任何档位都不给整句答案）：\n"
        "- L1 词义/词块线索：目标表达关键词块的中文意思 + 首字母；\n"
        "- L2 句式骨架：挖空的句式骨架，保留功能词；\n"
        "- L3 中英对照引导：整句中文意思 + 逐段对照，并给出句式开头（仍让学习者自己组句）。\n"
        "【输出】只输出一个 JSON 对象（不要任何解释或代码块外文本）：\n"
        '{{"content_type": "{content_type}", "ai_text": "<当前轮 AI 文本>", '
        '"hint": {{"levels": ["<L1>", "<L2>", "<L3>"]}} 或 null, '
        '"suggested_targets": ["<sentence_id>"]}}\n'
        "- content_type 必须与给定的 {content_type} 一致；\n"
        "- ai_text：dialogue = AI 角色台词（口语自然，1~3 句）；fill = 半成品对话任务卡"
        "（情景文本 + AI 侧台词，把要求学习者补全的目标句位置留空，不写出该句）；\n"
        "- hint：本轮需要学习者作答就给出目标句 L1~L3，否则 null；\n"
        "- suggested_targets：≤1 个，必须是上述素材中的句子 id。"
    ).format(ai_role=ai_role, learner_role=learner_role, content_type=content_type)

    user_msg = (
        "【场景】{scene}\n"
        "【角色】AI：{ai_role}｜学习者：{learner_role}\n"
        "【素材】（第 1 句起按序编号）\n{materials}\n"
        "【会话记录】\n{history}\n"
        "{instruction}"
    ).format(
        scene="；".join(scene_parts) if scene_parts else "（未提供，按通用自由会话）",
        ai_role=ai_role,
        learner_role=learner_role,
        materials=material_text,
        history=history_text or "（开场，尚无历史）",
        instruction=prompt_instruction,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# LLM 调用（火山方舟，OpenAI 兼容，JSON 输出）
# ---------------------------------------------------------------------------


def _call_session_llm(messages: list[dict], temperature: float = 0.7) -> str | None:
    """同步调用火山方舟对话模型（OpenAI 兼容）；凭据缺失 / 调用失败返回 None。

    超时由外层 `call_session_llm` 的 `asyncio.wait_for` 兜底（强取消），
    本函数仅设置 requests 软超时（防止线程池悬挂占满）。
    """
    if not (VOLCANO_API_KEY and VOLCANO_CHAT_MODEL):
        logger.warning("[session_gen] 未配置火山方舟凭据，无法生成")
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
            # 强制 JSON 输出（OpenAI 兼容字段，方舟支持）
            "response_format": {"type": "json_object"},
        },
        timeout=SESSION_LLM_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        logger.error(
            "[session_gen] 火山方舟返回 %s: %s", resp.status_code, resp.text[:200]
        )
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_session_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
) -> str | None:
    """LLM 调用异步包装：同步请求丢线程池 + `asyncio.wait_for` 超时强制取消。

    Raises:
        SessionGenError: 达到超时上限仍未返回 → error_code=LLM_TIMEOUT（stage=llm）
    """
    timeout = timeout_seconds or SESSION_LLM_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_call_session_llm, messages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[session_gen] LLM 调用超过 {timeout}s 未返回 → LLM_TIMEOUT"
        )
        raise SessionGenError(
            ERR_LLM_TIMEOUT,
            STAGE_LLM,
            f"LLM 调用超过 {timeout}s 未返回（SESSION_LLM_TIMEOUT_SECONDS={timeout}）",
        )


# ---------------------------------------------------------------------------
# 输出解析
# ---------------------------------------------------------------------------


def parse_session_output(content: str | None, valid_ids: set[str]) -> dict | None:
    """解析模型输出 JSON 并规整（容忍代码块包裹）。

    Returns:
        { content_type, ai_text, hint, suggested_targets } | None（解析失败）
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
        logger.warning(f"[session_gen] content_type={content_type!r} 不在支持集 → 解析失败")
        return None
    ai_text = str(parsed.get("ai_text") or "").strip()
    if not ai_text:
        return None

    # hint：{ levels: [L1,L2,L3], max_level: 3 }；空/缺省 → null（本轮无需作答）
    raw_hint = parsed.get("hint")
    hint = None
    if isinstance(raw_hint, dict):
        raw_levels = raw_hint.get("levels")
        if isinstance(raw_levels, list):
            levels = [str(x).strip() for x in raw_levels[:HINT_LEVELS_COUNT] if str(x).strip()]
            if levels:
                # 契约 levels=[L1,L2,L3] 三档：不足补空串，客户端逐级索引稳定
                levels += [""] * (HINT_LEVELS_COUNT - len(levels))
                hint = {"levels": levels, "max_level": HINT_MAX_LEVEL}

    # suggested_targets：≤1 个，剔除非法 id（不在素材句集合内）
    raw_targets = parsed.get("suggested_targets")
    suggested_targets: list[str] = []
    if isinstance(raw_targets, list):
        for t in raw_targets:
            tid = str(t).strip()
            if tid and tid in valid_ids and len(suggested_targets) < 1:
                suggested_targets.append(tid)

    return {
        "content_type": content_type,
        "ai_text": ai_text,
        "hint": hint,
        "suggested_targets": suggested_targets,
    }


# ---------------------------------------------------------------------------
# 对外生成入口
# ---------------------------------------------------------------------------


async def generate_session_reply(
    *,
    context: dict,
    preferred_type: str = "auto",
    timeout_seconds: int | None = None,
) -> dict:
    """会话生成（不降级）：形态选择 → LLM 调用 → 解析 → 规整结果。

    Args:
        context: 任务 context 快照（§4.18），含 mode/materials/scenario/roles/history/
                 user_input/assisted
        preferred_type: auto / dialogue / fill（retell/task 由路由层 TYPE_NOT_SUPPORTED 拦截）

    Returns:
        { content_type, ai_text, hint, suggested_targets }

    Raises:
        SessionGenError:
        - 模型不可用 / 调用失败 / 返回空 → EVAL_UNAVAILABLE（stage=llm）
        - 超时未返回 → LLM_TIMEOUT（stage=llm）
        - 输出无法解析 / 形态不支持 → LLM_PARSE_ERROR（stage=parse）
    """
    materials = context.get("materials") or []
    content_type = choose_content_type(materials, preferred_type)
    context = {**context, "mode": context.get("mode") or "start"}
    messages = build_session_messages(context, content_type)
    content = await call_session_llm(messages, timeout_seconds)
    if not content:
        raise SessionGenError(
            ERR_EVAL_UNAVAILABLE, STAGE_LLM, "LLM 调用失败（模型不可用或返回空）"
        )
    valid_ids = collect_sentence_ids(materials)
    parsed = parse_session_output(content, valid_ids)
    if parsed is None:
        raise SessionGenError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, f"模型输出解析失败: {content[:200]}"
        )
    logger.info(
        f"[session_gen] generate → content_type={parsed['content_type']}, "
        f"ai_text_len={len(parsed['ai_text'])}, "
        f"hint={'有' if parsed['hint'] else '无'}, "
        f"targets={parsed['suggested_targets']}"
    )
    return parsed
