"""数学同类型题出题服务（quiz）· 常量与骨架

契约：
- api-contract.md §3.10 接口 28~31（`POST /math/scan/quiz-recognize` / `POST /math/quiz/generate`
  / `POST /math/quiz/{quiz_id}/render` / `GET /math/quiz/{quiz_id}`）
- data-model-contract.md §4.12.11 `math_quiz`
- service-contract.md §7.11（scholar-admin 后端）/ §7.12（wx-knowlege-graph 小程序端）
- ADR-0035（拍照语义隔离 + 原题多选 + 两阶段 A4 打印，D1~D9）
- 设计文档 `docs_v1/数学/同类型题.md` §4.8（骨架与常量）

红线：
- **R1** quiz 识别链路**零写 `error_record`**（落 `services/math/quiz_recognizer.py`，B04）
- **R2** 不改动 `POST /math/scan/classify` / `POST /math/practice-sheet` / `services/math/error_scanner.py`
- **R3** `answer` / `hint` **落库不出参**（`_to_public_quiz`，B03）
- **R4** 不依赖 `/math/drill/*`（后端未上线，见设计文档 §2.5 现状③）

说明：
- 本模块按原子任务增量构建：**B01** 常量与 ID 生成 → **B02** 签名 / 落库 / 查询
  → **B06** 出题 prompt；其余函数（出题主流程 / 防背题校验）由 B07~B09 追加。
- 集合名常量风格对齐 `services.infra.database.ERROR_RECORD_COLLECTION`。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

from config import (
    LLM_DISABLE_THINKING,
    LLM_SUMMARY_MODEL,
)
from services.math.knowledge_summary import (
    LLMNotConfiguredError,
    LLMResponseError,
    _get_llm_client,
)
from services.audit import (
    AUDIT_RESULT_FAILED,
    AUDIT_RESULT_SUCCESS,
    write_audit,
)
from services.database import CURRICULUM_NODE_COLLECTION
from services.math.practice_sheet import (
    SHEET_LLM_CALL_TIMEOUT_SEC,  # 出题单次调用时延预算，沿用练习纸口径
    _find_extended_point,  # P1 拔高题：取节点奥数扩展点（practice_sheet.py:306）
    _validate_question_items,  # 结构校验（`practice_sheet.py:510`），B08 复用
)

try:  # B13 之后由 services.metrics.audit 提供常量；在此之前用同名字面量，口径一致
    from services.metrics.audit import AUDIT_ACTION_QUIZ_GENERATE
except ImportError:  # pragma: no cover — B13 实施后自动切到正式常量
    AUDIT_ACTION_QUIZ_GENERATE = "quiz_generate"

logger = logging.getLogger("scholar-admin.math.quiz")

# ---------------------------------------------------------------------------
# 出题排障诊断工具（2026-10-14 加）
#
# 背景：真机出现「LLM HTTP 200 但 [quiz] 出题返回空 items」→ 每题重试 2 次均空
# → 全部源题出题失败（500）。原日志只有一行「出题返回空 items」，看不出是
# ①输出被 max_tokens 截断（finish_reason=length）②模型把 JSON 吐在
# reasoning_content 而 content 为空 ③prompt 要求裸数组而代码只认 {"items": [...]}
# ④模型返回了别的键（questions/items 缺失）。以下工具把上述字段全部落日志。
#
# 红线：诊断日志**绝不能影响出题主流程**——取值与写日志全部 try/except 兜底。
# ---------------------------------------------------------------------------


def _preview(value: Any, limit: int = 500) -> str:
    """任意值 → 截断后的可打印字符串（日志用；永不抛异常）。"""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:  # noqa: BLE001 — 诊断工具不得影响主流程
        return "<unprintable>"
    return text if len(text) <= limit else f"{text[:limit]}…(截断, 总长 {len(text)})"


def _resp_attr(obj: Any, name: str, default: Any = "") -> Any:
    """安全取响应字段：兼容对象（OpenAI SDK）与 dict（裸响应/mock），永不抛异常。"""
    try:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
    except Exception:  # noqa: BLE001
        return default


def _log_quiz_llm_diag(resp: Any, *, model: str, kwargs: dict, prompt: str) -> None:
    """记录单次出题 LLM 调用的诊断字段。

    必看字段：finish_reason（length = 输出被 max_tokens 截断）/ usage（token 是否打满）/
    content 长度与原文 / reasoning_content 长度（推理型模型是否把正文吐到别处）/
    实际下发的 model·max_tokens·response_format·thinking·timeout。
    """
    try:
        choices = _resp_attr(resp, "choices", []) or []
        first = choices[0] if choices else None
        message = _resp_attr(first, "message", None)
        content = _resp_attr(message, "content", "")
        reasoning = _resp_attr(message, "reasoning_content", "")
        usage = _resp_attr(resp, "usage", None)
        logger.info(
            f"[quiz][llm] model={model!r} max_tokens={kwargs.get('max_tokens')} "
            f"timeout={kwargs.get('timeout')} response_format={kwargs.get('response_format')} "
            f"thinking_disabled={bool(kwargs.get('extra_body'))} "
            f"finish_reason={_resp_attr(first, 'finish_reason', '')!r} "
            f"prompt_tokens={_resp_attr(usage, 'prompt_tokens', None)} "
            f"completion_tokens={_resp_attr(usage, 'completion_tokens', None)} "
            f"total_tokens={_resp_attr(usage, 'total_tokens', None)} "
            f"prompt_len={len(prompt or '')} "
            f"content_len={len(content) if isinstance(content, str) else -1} "
            f"reasoning_len={len(reasoning) if isinstance(reasoning, str) else -1} "
            f"content_preview={_preview(content, 800)!r}"
        )
    except Exception as e:  # noqa: BLE001 — 诊断日志失败不得中断出题
        logger.warning(f"[quiz][llm] 诊断日志写入失败（已忽略）: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 集合名
# ---------------------------------------------------------------------------

MATH_QUIZ_COLLECTION = "math_quiz"  # 同类型题出题（本页落库，不动 error_record）

# ---------------------------------------------------------------------------
# 题量与防背题常量（设计文档 §4.8 / D6 / D9）
# ---------------------------------------------------------------------------

QUIZ_MAX_SOURCE_COUNT = 5  # 单次最多选几道源题（D9）
QUIZ_DEFAULT_COUNT_PER_SOURCE = 2  # 每道源题默认出几道
QUIZ_MAX_COUNT_PER_SOURCE = 3  # 每道源题上限 → 单次总量上限 5×3 = 10
QUIZ_REPEAT_WINDOW_MS = 10 * 60 * 1000  # 幂等窗口，沿用 SHEET_REPEAT_WINDOW_MS 口径（D8）
QUIZ_SIM_THRESHOLD_L1 = 0.3  # 数值重合阈值（ADR-0010 口径，本页首次落地）
QUIZ_SIM_THRESHOLD_L2 = 0.5  # 情境相似度阈值（源题内 + 跨源题）
QUIZ_EXTENDED_COUNT_PER_SOURCE = 1  # P1 拔高题：每道源题追加 1 道（`include_extended=true`）

# ---------------------------------------------------------------------------
# 状态常量（D3 两阶段：generate → render）
# ---------------------------------------------------------------------------

QUIZ_STATUS_GENERATED = "generated"
QUIZ_STATUS_RENDERING = "rendering"
QUIZ_STATUS_RENDERED = "rendered"
QUIZ_STATUS_FAILED = "failed"

# ---------------------------------------------------------------------------
# ID 生成（沿用 `_gen_sheet_id`（practice_sheet.py:143）形态）
# ---------------------------------------------------------------------------


def _gen_quiz_id() -> str:
    """生成 quiz_id：`qz_{ms}_{hex8}`。"""
    return f"qz_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 出题 Prompt（D4：以原题题干为唯一输入源，**不含 `error_type`**）
# ---------------------------------------------------------------------------

_QUIZ_QUESTION_SYSTEM_PROMPT = (
    "你是小学数学命题助手。请参考给定的原题，出与它题型完全相同、题意结构一致的练习题。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

# 逐字搬设计文档 §3 D4 的 prompt 正文（仅把「出 N 道」参数化为 {count}）。
# 知识点已知时使用；`summary` 由 `_build_quiz_prompt`（B07）传入（可为空串）。
_QUIZ_QUESTION_USER_TEMPLATE = """请参考下面这道“原题”，出 {count} 道与它【题型完全相同、题意结构一致】的练习题。

原题：{source_question}
所属知识点：{knowledge_point_name}（{summary}）
要求：
1. 必须与原题同题型（同为应用题/竖式计算/图形题等）、同考查点；仅替换数值（L1）、或等价替换情境（L2）。
2. 不改变题目考查的算理/步骤结构；难度贴合原题（±1 档）。
3. 每道题输出 {{"question": "题干", "answer": "参考答案", "answer_type": "number|text", "difficulty": 1~5 整数, "hint": "提示语"}}；**禁止输出原题原文**。

输出 JSON 结构（**必须是含 "items" 键的对象，不要直接输出顶层数组**）：
{{
  "items": [
    {{"question": "...", "answer": "...", "answer_type": "number", "difficulty": 3, "hint": "..."}}
  ]
}}"""

# D4 兜底：`knowledge_point_name` 为空（识别未命中且用户未填）→ 省略知识点段，
# 退化为「纯同题型模仿」，由调用方返回 warnings「未识别到知识点，已按原题题型出题」。
_QUIZ_QUESTION_USER_TEMPLATE_NO_KP = """请参考下面这道“原题”，出 {count} 道与它【题型完全相同、题意结构一致】的练习题。

原题：{source_question}
要求：
1. 必须与原题同题型（同为应用题/竖式计算/图形题等）、同考查点；仅替换数值（L1）、或等价替换情境（L2）。
2. 不改变题目考查的算理/步骤结构；难度贴合原题（±1 档）。
3. 每道题输出 {{"question": "题干", "answer": "参考答案", "answer_type": "number|text", "difficulty": 1~5 整数, "hint": "提示语"}}；**禁止输出原题原文**。

输出 JSON 结构（**必须是含 "items" 键的对象，不要直接输出顶层数组**）：
{{
  "items": [
    {{"question": "...", "answer": "...", "answer_type": "number", "difficulty": 3, "hint": "..."}}
  ]
}}"""

# P1 拔高题（`include_extended=true` 且节点存在 `extended_points` 时使用）：
# 沿用 `practice_sheet._build_question_prompt`(473) 的奥数口径 —— 以原题题型为起点，
# 按扩展点难度档（入门 / 普及 / 竞赛）适度拔高，难度 4~5，不超纲。
_QUIZ_EXTENDED_USER_TEMPLATE = """请参考下面这道“原题”与其所属知识点的奥数拔高点，出 {count} 道奥数拔高题。

原题：{source_question}
所属知识点：{knowledge_point_name}（{summary}）
拔高点难度档位：{extended_band}
要求：
1. 在原题题型基础上适度拔高（{extended_band} 档），可引入一步推理或等价变换，**不超纲**。
2. 难度 4~5；不改变原题考查的算理主线。
3. 每道题输出 {{"question": "题干", "answer": "参考答案", "answer_type": "number|text", "difficulty": 4~5 整数, "hint": "提示语"}}；**禁止输出原题原文**。

输出 JSON 结构（**必须是含 "items" 键的对象，不要直接输出顶层数组**）：
{{
  "items": [
    {{"question": "...", "answer": "...", "answer_type": "text", "difficulty": 4, "hint": "..."}}
  ]
}}"""

_QUIZ_EXTENDED_USER_TEMPLATE_NO_KP = """请参考下面这道“原题”，出 {count} 道奥数拔高题。

原题：{source_question}
拔高点难度档位：{extended_band}
要求：
1. 在原题题型基础上适度拔高（{extended_band} 档），可引入一步推理或等价变换，**不超纲**。
2. 难度 4~5；不改变原题考查的算理主线。
3. 每道题输出 {{"question": "题干", "answer": "参考答案", "answer_type": "number|text", "difficulty": 4~5 整数, "hint": "提示语"}}；**禁止输出原题原文**。

输出 JSON 结构（**必须是含 "items" 键的对象，不要直接输出顶层数组**）：
{{
  "items": [
    {{"question": "...", "answer": "...", "answer_type": "text", "difficulty": 4, "hint": "..."}}
  ]
}}"""


def _build_quiz_prompt(
    *,
    source_question: str,
    knowledge_point_name: str = "",
    summary: str = "",
    count: int = QUIZ_DEFAULT_COUNT_PER_SOURCE,
    extended: bool = False,
    extended_band: str = "",
) -> str:
    """渲染单源题出题 prompt（参考 `_build_question_prompt`（practice_sheet.py:468））。

    知识点为空（识别未命中且用户未填）→ 走 `*_NO_KP` 模板省略知识点段
    （D4 兜底，纯同题型模仿）；`extended=True` → 走 P1 拔高题模板。
    """
    source_question = (source_question or "").strip()
    knowledge_point_name = (knowledge_point_name or "").strip()
    summary = (summary or "").strip()
    band = (extended_band or "").strip() or "入门"
    if extended:
        if knowledge_point_name:
            return _QUIZ_EXTENDED_USER_TEMPLATE.format(
                count=int(count),
                source_question=source_question,
                knowledge_point_name=knowledge_point_name,
                summary=summary,
                extended_band=band,
            )
        return _QUIZ_EXTENDED_USER_TEMPLATE_NO_KP.format(
            count=int(count),
            source_question=source_question,
            extended_band=band,
        )
    if knowledge_point_name:
        return _QUIZ_QUESTION_USER_TEMPLATE.format(
            count=int(count),
            source_question=source_question,
            knowledge_point_name=knowledge_point_name,
            summary=summary,
        )
    return _QUIZ_QUESTION_USER_TEMPLATE_NO_KP.format(
        count=int(count),
        source_question=source_question,
    )


def _call_quiz_chat_sync(client: Any, model: str, system: str, prompt: str) -> str:
    """同步 chat 调用（线程池中执行）—— 出题专用。

    与 `practice_sheet._call_chat_sync`(540) 的差异：**system prompt 由调用方传入**
    （后者硬编码 `_QUESTION_SYSTEM_PROMPT`，那是「按知识点出题」的语义，不符合 D4
    「以原题为唯一输入源」）；temperature / thinking / 单次超时预算均沿用练习纸口径。
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "timeout": SHEET_LLM_CALL_TIMEOUT_SEC,
    }
    if LLM_DISABLE_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(**kwargs)
    # 2026-10-14 加：记录 finish_reason / usage / content 预览，定位「200 但 items 为空」
    _log_quiz_llm_diag(resp, model=model, kwargs=kwargs, prompt=prompt)
    return (resp.choices[0].message.content or "").strip()


def _extract_quiz_items(result: Any) -> list | None:
    """从 LLM 解析结果里取 `items`（**兼容模型跑偏的几种形态**）。

    2026-10-14 真机故障：LLM 返回 HTTP 200，但每题都「出题结果 items 为空」，两次重试
    均失败 → 全部源题出题失败（500）。根因是 prompt 只写了「输出 `[{...}]`」（裸数组），
    而解析只认 `{"items": [...]}` —— 模型越听话越失败。本次修法：
      ① prompt 明确要求 `{"items": [...]}`（与 `practice_sheet._QUESTION_USER_TEMPLATE`
         的「输出 JSON 结构」段同口径）；
      ② 本函数对**历史/跑偏形态**兜底，避免再出现「200 但整页失败」。

    兼容形态（按优先级）：
      1. `[...]` —— 裸数组（模型按旧 prompt 或自由发挥直接输出数组）
      2. `{"items": [...]}` —— 契约形态
      3. `{"questions": [...]}` —— 模型常用的同义键

    Returns: 取到且为 list → 该 list；否则 None（调用方按「空 items」重试）。
    """
    if isinstance(result, list):
        logger.warning(
            f"[quiz] LLM 返回顶层数组（非 {{\"items\": [...]}}），已兜底兼容 "
            f"items_len={len(result)}"
        )
        return result
    if not isinstance(result, dict):
        return None
    items = result.get("items")
    if isinstance(items, list):
        return items
    for key in ("questions",):
        candidate = result.get(key)
        if isinstance(candidate, list):
            logger.warning(
                f"[quiz] LLM 返回键 {key!r} 而非 'items'，已兜底兼容 items_len={len(candidate)}"
            )
            return candidate
    return None


async def _generate_quiz_questions(
    *,
    source_question: str,
    knowledge_point_name: str = "",
    summary: str = "",
    count: int = QUIZ_DEFAULT_COUNT_PER_SOURCE,
    extended: bool = False,
    extended_band: str = "",
) -> list[dict]:
    """单源题一次 LLM 调用出 `count` 道同类型题（D4 多源题调用：每源题独立一次）。

    沿用 `_generate_questions`(565) 的重试口径：**首次 + 重试 1 次**；
    `APITimeoutError` / `TimeoutError` **不重试**（2×timeout 会突破容器上限）。
    返回 LLM 原始 items（≤`count` 条），结构与与源题查重由 `_validate_quiz_items`（B08）负责。
    `extended=True` → 走 P1 拔高题 prompt（`extended_band`：入门 / 普及 / 竞赛）。

    Raises:
        LLMNotConfiguredError: LLM_SUMMARY_MODEL 未配置（→ 500）
        LLMResponseError: 两次均失败 / 返回空（→ 500）
    """
    if not LLM_SUMMARY_MODEL:
        raise LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法出题")
    from services.dialogue import _parse_json_response

    prompt = _build_quiz_prompt(
        source_question=source_question,
        knowledge_point_name=knowledge_point_name,
        summary=summary,
        count=count,
        extended=extended,
        extended_band=extended_band,
    )
    client = _get_llm_client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_quiz_chat_sync,
                client,
                LLM_SUMMARY_MODEL,
                _QUIZ_QUESTION_SYSTEM_PROMPT,
                prompt,
            )
            result = _parse_json_response(response)
            items = _extract_quiz_items(result)  # 兼容裸数组 / {"questions": [...]}（见函数注释）
            if isinstance(items, list) and items:
                return items[:count]
            last_err = LLMResponseError("出题结果 items 为空")
            # 2026-10-14 加：把「为什么空」所需的字段一次打全
            # —— result_type=list 说明模型按 prompt 返回了裸数组（代码只认 {"items": [...]}）；
            #    result_keys 能看到模型用了别的键（如 questions）；raw 预览能看到截断/前言。
            logger.warning(
                f"[quiz] 出题返回空 items（第 {attempt} 次）model={LLM_SUMMARY_MODEL!r} "
                f"count={count} extended={extended} kp={knowledge_point_name!r} "
                f"result_type={type(result).__name__} "
                f"result_keys={list(result.keys()) if isinstance(result, dict) else None} "
                f"items_type={type(items).__name__} "
                f"items_len={len(items) if isinstance(items, list) else None} "
                f"source_question_preview={_preview(source_question, 120)!r} "
                f"raw_response_preview={_preview(response, 800)!r}"
            )
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 LLMResponseError
            last_err = e
            logger.warning(
                f"[quiz] 出题失败（第 {attempt} 次）model={LLM_SUMMARY_MODEL!r} "
                f"kp={knowledge_point_name!r} count={count}: {type(e).__name__}: {e} "
                f"| source_question_preview={_preview(source_question, 120)!r}"
            )
            if type(e).__name__ in ("APITimeoutError", "TimeoutError"):
                break
    raise LLMResponseError(f"同类型题出题失败: {last_err}")


# ---------------------------------------------------------------------------
# 出题结果校验（D6 防背题：结构 + 源题重合度 + 跨源题查重）
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def _gen_quiz_item_id() -> str:
    """生成 quiz_item_id：`qi_{ms}_{hex8}`。"""
    return f"qi_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _numeric_overlap(text_a: str, text_b: str) -> float:
    """数值重合度 = 数值**多集**交集 / 并集（0.0 ~ 1.0；任一题无数值 → 0.0）。

    例：「0.36 + 2.7」与「2.7 + 0.36」→ 数值完全相同 → 1.0（视为雷同，答案也相同）。
    """
    ca = Counter(_NUM_RE.findall(str(text_a or "")))
    cb = Counter(_NUM_RE.findall(str(text_b or "")))
    if not ca or not cb:
        return 0.0
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 0.0


def _text_similarity(text_a: str, text_b: str) -> float:
    """情境相似度：去空白小写后 SequenceMatcher 相似度（0.0 ~ 1.0）。"""
    a = _WS_RE.sub("", str(text_a or "")).lower()
    b = _WS_RE.sub("", str(text_b or "")).lower()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_near_duplicate(question: str, reference: str) -> bool:
    """雷同判定（D6，沿用 ADR-0010 附录 A-15 口径）：

    **情境相似度 ≥0.5 且 数值重合 ≥0.3** → 雷同。二者缺一即放行：
    - 换过情境（相似度低）但数值碰巧相近 → 放行；
    - 表述几乎一致但数值全换（真 L1 变式）→ 放行。
    """
    return (
        _text_similarity(question, reference) >= QUIZ_SIM_THRESHOLD_L2
        and _numeric_overlap(question, reference) >= QUIZ_SIM_THRESHOLD_L1
    )


def _difficulty_of(raw: dict) -> int:
    """`difficulty` 非法兜底 3（与 `_validate_question_items` 同口径）。"""
    try:
        difficulty = int(raw.get("difficulty"))
    except (TypeError, ValueError):
        return 3
    return difficulty if 1 <= difficulty <= 5 else 3


def _validate_quiz_items(
    raw_items: list[dict],
    *,
    source_question: str,
    source_index: int = 0,
    accepted: list[dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """单源题出题结果校验 —— 结构 + 与**对应源题**重合度 + **跨源题查重**（D6）。

    - 结构：复用 `_validate_question_items`（`practice_sheet.py:510`），非法/缺
      `question` / `answer` → `LLMResponseError`；`difficulty` 非法兜底 3。
    - 雷同：对源题与已放行项（含其它源题已产生的题）逐一比对，
      命中的题**剔除并写 `warnings[]`**，**不抛错**（个别题去重不应使整次出题失败）。
    - `variant_level`：由**与源题的情境相似度**推导 —— 表述基本一致（≥0.5）即
      「只换了数值」→ `L1`；表述已换 → `L2`（与 ADR-0010 §2 变式分级一致）。

    Returns: `(cleaned_items, warnings)`。注：**「超限重试 1 次」在 B09 的逐源题生成
    循环里做**（本函数是纯校验，不触发 LLM）。
    """
    _validate_question_items({"items": raw_items})
    cleaned: list[dict] = []
    warnings: list[str] = []
    references = [it.get("question") or "" for it in (accepted or [])]

    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        question = str(raw.get("question") or "")
        if _is_near_duplicate(question, source_question):
            warnings.append(f"第 {idx + 1} 道题与原题重复，已剔除")
            continue
        if any(_is_near_duplicate(question, ref) for ref in references if ref):
            warnings.append(f"第 {idx + 1} 道题与已出题重复，已剔除")
            continue
        answer_type = str(raw.get("answer_type") or "number").strip().lower()
        if answer_type not in ("number", "text"):
            answer_type = "number"
        cleaned.append(
            {
                "quiz_item_id": _gen_quiz_item_id(),
                "source_index": source_index,
                "question": question,
                "answer": str(raw.get("answer") or ""),
                "answer_type": answer_type,
                "variant_level": (
                    "L1"
                    if _text_similarity(question, source_question) >= QUIZ_SIM_THRESHOLD_L2
                    else "L2"
                ),
                "difficulty": _difficulty_of(raw),
                "hint": str(raw.get("hint") or ""),
            }
        )
        references.append(question)
    return cleaned, warnings


# ---------------------------------------------------------------------------
# 幂等签名（D8）
# ---------------------------------------------------------------------------


def _quiz_signature(
    *,
    scholar_id: str,
    source_questions: list,
    count_per_source: int,
    model: str,
    include_extended: bool = False,
) -> str:
    """幂等签名：同 scholar + 同（勾选题集合 / 每源题题数 / 模型 / 拔高题开关）→ 同签名。

    沿用 `_sheet_signature`（practice_sheet.py:159）的 sorted-json + sha256 口径。
    **`source_questions` 按 `source_index` 稳定排序后取签** ⇒ 勾选顺序变化不影响签名
    （设计文档 D8）；题干 / 知识点 / 节点码变化则签名随之变化（需重新出题）。
    """
    sig = {
        "scholar_id": scholar_id,
        "source_questions": [
            {
                "source_index": q.get("source_index"),
                "question_text": q.get("question_text"),
                "knowledge_point_name": q.get("knowledge_point_name") or "",
                "node_code": q.get("node_code") or "",
            }
            for q in sorted(
                source_questions or [],
                key=lambda q: (q.get("source_index") is None, q.get("source_index")),
            )
        ],
        "count_per_source": int(count_per_source),
        "model": model or "",
        # P1 拔高题开关：true / false 产出不同（追加拔高题），须参与取签，
        # 否则开关注解不同也会幂等命中（沿用 `_sheet_signature` 把 `include_extended_points`
        # 纳入签名的口径，practice_sheet.py:183）。
        "include_extended": bool(include_extended),
    }
    raw = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 落库 + 查询（data-model-contract §4.12.11 / 设计文档 §4.6）
# ---------------------------------------------------------------------------


async def _persist_quiz(db, doc: dict) -> dict:
    """落库 `math_quiz`（字段见设计文档 §4.6）。

    `doc` 由调用方（B06 出题）按 §4.6 组装；本函数补齐 `_id` 与缺省值后 insert，
    并原样返回落库文档。注意：**未勾选的源题不入库**（D9，由调用方保证）。
    """
    now = _now_ms()
    record = {
        **doc,
        "_id": doc.get("quiz_id"),
        "status": doc.get("status") or QUIZ_STATUS_GENERATED,
        "file_refs": doc.get("file_refs") or {"pdf": "", "png": "", "preview": ""},
        "warnings": doc.get("warnings") or [],
        "render_status": doc.get("render_status") or "",
        "created_at": doc.get("created_at") or now,
    }
    await db.insert(MATH_QUIZ_COLLECTION, record)
    return record


def _to_public_quiz(quiz: dict) -> dict:
    """落库文档 → 接口出参（api-contract §3.10 接口 29 / 31）。

    **红线 R3**：`items[].answer` / `items[].hint` **恒不出参**（`answer` 仅供渲染与
    日后判题使用）。返回**新对象**，不就地修改落库文档。
    """
    return {
        "quiz_id": quiz.get("quiz_id"),
        "status": quiz.get("status"),
        "source_count": quiz.get("source_count") or 0,
        "count_per_source": quiz.get("count_per_source") or 0,
        "sources": [dict(s) for s in (quiz.get("sources") or [])],
        "items": [
            {
                **{
                    k: it.get(k)
                    for k in (
                        "quiz_item_id",
                        "source_index",
                        "question",
                        "answer_type",
                        "variant_level",
                        "difficulty",
                    )
                    if it.get(k) not in (None, "")
                },
                # P1 拔高题标记（前端 / 打印稿据此打「奥数拔高」角标）
                "is_extended": bool(it.get("is_extended")),
            }
            for it in (quiz.get("items") or [])
        ],
        "file_refs": quiz.get("file_refs") or {"pdf": "", "png": "", "preview": ""},
        "warnings": quiz.get("warnings") or [],
        "render_status": quiz.get("render_status") or "",
        # 渲染结果的用户可见原因（失败 / 降级时非空，供前端直接展示）
        "render_message": quiz.get("render_message") or "",
    }


async def getQuizById(db, *, quiz_id: str) -> dict | None:
    """按 quiz_id 查询同类型题公开详情（api-contract §3.10 接口 31：轮询渲染状态与产物）。

    返回 `_to_public_quiz` 结构（红线 R3：`answer` / `hint` 不出参），quiz 不存在返回 None。
    `render_status` 取自落库字段（由渲染任务回写，缺省 `""`）。
    """
    res = await db.query(MATH_QUIZ_COLLECTION, where={"quiz_id": quiz_id}, limit=1)
    records = res.get("records") or []
    if not records:
        return None
    return _to_public_quiz(records[0])


# ---------------------------------------------------------------------------
# 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class QuizError(Exception):
    """同类型题出题业务错误基类"""


class InvalidQuizSourceError(QuizError):
    """入参非法 → 400（`source_questions` / `count_per_source` / `scholar_id`）"""


class NodeCodeNotFoundError(QuizError):
    """`source_questions[].node_code` 在 `curriculum_node` 中不存在 → 404"""


# ---------------------------------------------------------------------------
# generateQuiz 主入口（api-contract §3.10 接口 29 / 设计文档 §4.3）
# ---------------------------------------------------------------------------


def _validate_source_questions(
    source_questions: list,
    count_per_source: int,
) -> list[dict]:
    """入参校验（§4.3 口径）：非空、≤5、`question_text` 非空、`source_index` 不重复
    （另有 `count_per_source` ∈ [1, 3]），任一非法 → `InvalidQuizSourceError`（400）。

    Returns: 归一化后的源题快照，按 `source_index` 升序。
    """
    if not isinstance(source_questions, list) or not source_questions:
        raise InvalidQuizSourceError("source_questions 不能为空（至少勾选 1 道原题）")
    if len(source_questions) > QUIZ_MAX_SOURCE_COUNT:
        raise InvalidQuizSourceError(
            f"source_questions 最多 {QUIZ_MAX_SOURCE_COUNT} 道，当前 {len(source_questions)} 道"
        )
    if not isinstance(count_per_source, int) or isinstance(count_per_source, bool):
        raise InvalidQuizSourceError("count_per_source 必须为整数")
    if not 1 <= count_per_source <= QUIZ_MAX_COUNT_PER_SOURCE:
        raise InvalidQuizSourceError(
            f"count_per_source 须为 1~{QUIZ_MAX_COUNT_PER_SOURCE} 的整数，当前 {count_per_source}"
        )

    sources: list[dict] = []
    seen: set = set()
    for src in source_questions:
        if not isinstance(src, dict):
            raise InvalidQuizSourceError("source_questions 每项必须为对象")
        idx = src.get("source_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            raise InvalidQuizSourceError("source_questions[].source_index 必须为整数")
        if idx in seen:
            raise InvalidQuizSourceError(f"source_questions[].source_index 重复: {idx}")
        seen.add(idx)
        question_text = str(src.get("question_text") or "").strip()
        if not question_text:
            raise InvalidQuizSourceError(f"source_questions[{idx}].question_text 不能为空")
        sources.append(
            {
                "source_index": idx,
                "question_text": question_text,
                "knowledge_point_name": str(src.get("knowledge_point_name") or "").strip(),
                "node_code": str(src.get("node_code") or "").strip(),
            }
        )
    return sorted(sources, key=lambda s: s["source_index"])


async def _load_node_by_code(db, node_code: str) -> dict:
    """按 `code` 读 `curriculum_node`（P1 拔高题需取 `ai_summary.extended_points`）。

    不存在 → `NodeCodeNotFoundError`（契约 §4.3：404）。
    """
    res = await db.query(CURRICULUM_NODE_COLLECTION, where={"code": node_code}, limit=1)
    records = res.get("records") or []
    if not records:
        raise NodeCodeNotFoundError(f"node_code 不存在: {node_code}")
    return records[0]


def _summary_from_node(node: dict, kp_name: str = "") -> str:
    """节点 → 注入 prompt 的知识点总结：**同名知识点 summary 优先**，退化节点级 summary。"""
    ai = node.get("ai_summary") or {}
    for kp in ai.get("knowledge_points") or []:
        if (kp.get("name") or "").strip() == (kp_name or "").strip() and kp_name:
            return str(kp.get("summary") or "")
    return str(ai.get("summary") or node.get("title") or "")


async def _load_summary_by_node_code(db, node_code: str, kp_name: str = "") -> str:
    """`_load_node_by_code` + `_summary_from_node` 的便捷封装（B09 起沿用）。"""
    node = await _load_node_by_code(db, node_code)
    return _summary_from_node(node, kp_name)


async def _generate_extended_items(
    *,
    node: dict | None,
    src: dict,
    label: str,
    summary: str = "",
    accepted: list[dict],
) -> tuple[int, list[str]]:
    """P1 拔高题：`include_extended=true` 且该源题节点存在 `extended_points` 时追加。

    沿用 `practice_sheet._find_extended_point`(306) 的匹配口径（同名
    `related_knowledge_name` 优先，否则取该节点第一个扩展点）；难度档位
    （入门 / 普及 / 竞赛）注入 prompt。拔高题同样过 D6 防背题校验，并打
    `is_extended=True` 标记（落库 + 出参，打印稿带「奥数拔高」角标）。

    Returns: `(追加题数, warnings)` —— 拔高题失败**不影响主流程**（仅写 warnings）。
    """
    warnings: list[str] = []
    if node is None:
        warnings.append(f"{label}未提供 node_code，无法出拔高题，已跳过")
        return 0, warnings
    ep = _find_extended_point(node, src.get("knowledge_point_name") or "")
    if not ep:
        warnings.append(f"{label}所属节点无奥数扩展点，已跳过拔高题")
        return 0, warnings

    band = str(ep.get("difficulty_band") or "").strip()
    try:
        raw_items = await _generate_quiz_questions(
            source_question=src["question_text"],
            knowledge_point_name=src.get("knowledge_point_name") or "",
            summary=summary,
            count=QUIZ_EXTENDED_COUNT_PER_SOURCE,
            extended=True,
            extended_band=band,
        )
        items, warn = _validate_quiz_items(
            raw_items,
            source_question=src["question_text"],
            source_index=src["source_index"],
            accepted=accepted,
        )
    except Exception as e:  # noqa: BLE001 — 拔高题失败仅降级，不拖累基础题
        logger.warning(f"[quiz] {label} 拔高题出题失败，已跳过: {type(e).__name__}: {e}")
        warnings.append(f"{label}拔高题出题失败，已跳过")
        return 0, warnings

    if not items:
        warnings.append(f"{label}拔高题出题结果不可用，已跳过")
        return 0, warnings
    for it in items:
        it["is_extended"] = True
    accepted.extend(items)
    warnings.extend([f"{label}拔高题：{w}" for w in warn])
    return len(items), warnings


async def _find_recent_quiz(
    db, scholar_id: str, signature: str, within_ms: int = QUIZ_REPEAT_WINDOW_MS
) -> dict | None:
    """同 scholar 幂等窗口内签名一致的最近 quiz（沿用 `_find_recent_sheet`(625) 口径）。"""
    res = await db.query(
        MATH_QUIZ_COLLECTION,
        where={"scholar_id": scholar_id},
        order=[{"field": "created_at", "direction": "desc"}],
        limit=5,
    )
    now = _now_ms()
    for r in res.get("records") or []:
        if r.get("signature") != signature:
            continue
        if r.get("status") != QUIZ_STATUS_GENERATED:
            continue
        if now - int(r.get("created_at") or 0) <= within_ms:
            return r
    return None


async def generateQuiz(
    db,
    *,
    scholar_id: str,
    source_questions: list,
    scan_id: str | None = None,
    count_per_source: int = QUIZ_DEFAULT_COUNT_PER_SOURCE,
    include_extended: bool = False,
    actor: str = "",
) -> dict:
    """出同类型题（api-contract §3.10 接口 29 · 设计文档 §4.3 / D6 / D8 / D9）。

    流程：入参校验 → 幂等签名命中则返回既有 quiz_id（**零 LLM 调用**）→
    **逐源题独立调用**出题 prompt（未勾选的题零 LLM 调用）→ 防背题校验（雷同剔除，
    全被剔除则重试 1 次）→ 落库 + 审计 `quiz_generate`。

    Returns: `_to_public_quiz` 结构（`items[]` 无 `answer` / `hint`，红线 R3）。

    Raises:
        InvalidQuizSourceError: 入参非法 / 缺 `scholar_id`（→ 400）
        NodeCodeNotFoundError: `node_code` 不存在（→ 404）
        LLMNotConfiguredError: `LLM_SUMMARY_MODEL` 未配置（→ 400，与练习纸链路同口径）
        LLMResponseError: 全部源题出题失败（→ 500）
    """
    if not scholar_id:
        raise InvalidQuizSourceError("缺少 scholar_id")
    sources = _validate_source_questions(source_questions, count_per_source)
    model = LLM_SUMMARY_MODEL or ""
    if not model:
        # 配置类错误**快速失败**（→ 400）：不逐源题走一遍「出题失败 → 跳过」的
        # 降级分支，否则会被 `generateQuiz` 末端的「全部失败」吞成 500（语义偏差）。
        raise LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法出题")
    signature = _quiz_signature(
        scholar_id=scholar_id,
        source_questions=sources,
        count_per_source=count_per_source,
        model=model,
        include_extended=include_extended,
    )

    t0 = time.time()
    logger.info(
        f"[quiz] 出题开始 scholar={scholar_id} 源题数={len(sources)} "
        f"count_per_source={count_per_source} include_extended={include_extended} "
        f"model={model!r} signature={signature[:12]} scan_id={scan_id or ''}"
    )

    recent = await _find_recent_quiz(db, scholar_id, signature)
    if recent is not None:
        logger.info(f"[quiz] 幂等命中 quiz_id={recent.get('quiz_id')} scholar={scholar_id}（零 LLM 调用）")
        return _to_public_quiz(recent)

    accepted: list[dict] = []
    extended_count = 0
    warnings: list[str] = []
    failed_sources = 0

    for src in sources:
        idx = src["source_index"]
        label = f"第 {idx + 1} 道原题"
        kp_name = src["knowledge_point_name"]
        summary = ""
        node = None
        if src["node_code"]:
            # node_code 不存在属契约 404，**不被下方 LLM 兜底吞掉**
            node = await _load_node_by_code(db, src["node_code"])
            summary = _summary_from_node(node, kp_name)
        if not kp_name:
            warnings.append(f"{label}未识别到知识点，已按原题题型出题")

        try:
            raw_items = await _generate_quiz_questions(
                source_question=src["question_text"],
                knowledge_point_name=kp_name,
                summary=summary,
                count=count_per_source,
            )
            items, warn = _validate_quiz_items(
                raw_items,
                source_question=src["question_text"],
                source_index=idx,
                accepted=accepted,
            )
            if not items and raw_items:
                # D6：该源题首轮全部被判雷同 → 重出 1 次
                logger.info(f"[quiz] {label} 首轮全被判雷同，重出 1 次")
                raw_items = await _generate_quiz_questions(
                    source_question=src["question_text"],
                    knowledge_point_name=kp_name,
                    summary=summary,
                    count=count_per_source,
                )
                items, warn = _validate_quiz_items(
                    raw_items,
                    source_question=src["question_text"],
                    source_index=idx,
                    accepted=accepted,
                )
        except Exception as e:  # noqa: BLE001 — 单源题失败跳过，不整体失败
            logger.warning(
                f"[quiz] {label} 出题失败，已跳过: {type(e).__name__}: {e} "
                f"| model={model!r} count_per_source={count_per_source} kp={kp_name!r} "
                f"node_code={src.get('node_code') or ''} "
                f"question_preview={_preview(src['question_text'], 120)!r}"
            )
            warnings.append(f"{label}出题失败，已跳过")
            failed_sources += 1
            continue

        if not items:
            logger.warning(
                f"[quiz] {label} 出题结果不可用（校验后为空）raw_items={len(raw_items) if raw_items else 0} "
                f"warnings={warn} question_preview={_preview(src['question_text'], 120)!r}"
            )
            warnings.append(f"{label}出题结果不可用，已跳过")
            failed_sources += 1
            continue
        warnings.extend([f"{label}：{w}" for w in warn])
        accepted.extend(items)

        # ---- P1 拔高题：`include_extended=true` 且该节点存在 extended_points ---- #
        if include_extended:
            added, ext_warn = await _generate_extended_items(
                node=node,
                src=src,
                label=label,
                summary=summary,
                accepted=accepted,
            )
            warnings.extend(ext_warn)
            extended_count += added

    if failed_sources >= len(sources):
        logger.error(
            f"[quiz] 全部源题出题失败 scholar={scholar_id} 源题数={len(sources)} "
            f"model={model!r} count_per_source={count_per_source} "
            f"include_extended={include_extended} signature={signature[:12]} "
            f"耗时={time.time() - t0:.1f}s warnings={warnings}"
        )
        await write_audit(
            db,
            action=AUDIT_ACTION_QUIZ_GENERATE,
            object_ref="",
            actor=actor,
            result=AUDIT_RESULT_FAILED,
            context={
                "scholar_id": scholar_id,
                "source_count": len(sources),
                "count_per_source": count_per_source,
                "generated_count": 0,
                "extended_count": 0,
                "include_extended": bool(include_extended),
                "model": model,
            },
        )
        raise LLMResponseError("全部源题出题失败，请稍后重试")

    quiz_id = _gen_quiz_id()
    logger.info(
        f"[quiz] 出题完成 quiz_id={quiz_id} scholar={scholar_id} 源题数={len(sources)} "
        f"生成={len(accepted)} 拔高={extended_count} 失败源题={failed_sources} "
        f"model={model!r} 耗时={time.time() - t0:.1f}s"
    )
    audit = await write_audit(
        db,
        action=AUDIT_ACTION_QUIZ_GENERATE,
        object_ref=quiz_id,
        actor=actor,
        result=AUDIT_RESULT_SUCCESS,
        context={
            "scholar_id": scholar_id,
            "source_count": len(sources),
            "count_per_source": count_per_source,
            "generated_count": len(accepted),
            "extended_count": extended_count,
            "include_extended": bool(include_extended),
            "model": model,
        },
    )
    saved = await _persist_quiz(
        db,
        {
            "quiz_id": quiz_id,
            "scholar_id": scholar_id,
            "source_scan_id": scan_id or "",
            "sources": sources,
            "source_count": len(sources),
            "count_per_source": count_per_source,
            "items": accepted,
            "include_extended": bool(include_extended),
            "extended_count": extended_count,
            "signature": signature,
            "warnings": warnings,
            "audit_log_id": (audit or {}).get("log_id") or "",
            "created_at": _now_ms(),
        },
    )
    logger.info(
        f"[quiz] generate 完成 quiz_id={quiz_id} 源题={len(sources)} "
        f"出题={len(accepted)}（含拔高 {extended_count}）降级源题={failed_sources}"
    )
    return _to_public_quiz(saved)
