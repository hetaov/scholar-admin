"""翻译评估 v2 评分引擎（ADR-0022 决策 B/D：LLM 超时可配置 + 不降级）

职责：对「标准句 + 用户译文」产出 { status(0-5), feedback, confidence }。
- mode=ec（英译中）：original_text=英文原句，用户输入=中文译文（docs_v1 §6.1）
- mode=ce（中译英）：original_text=中文原句，用户输入=英文译文（docs_v1 §6.2）
- **不降级**（docs_v1 §9）：模型不可用 / 网络异常 / 输出非法 → 抛
  `TranslationEvalError`，由任务执行器统一置为 failed（不回退 levenshtein /
  混元 / 云函数 updateTrackingStatus）。
- LLM 调用以 `TRANSLATION_LLM_TIMEOUT_SECONDS`（默认 300s）为上限，用
  `asyncio.wait_for` 包裹同步 `requests.post`（丢线程池，`run_in_threadpool`）。

对外函数：
- `infer_translation_mode(original_text)` → "ec" | "ce"
  （契约 v2 入参无 mode 字段，按原句语言推导：含中文字符 → ce，否则 ec）
- `build_translation_prompt(mode, original, reference, user_input)` → messages
  （reference 为「标准答案译文」，契约未提供时可传 None，仅以 original_text 评分）
- `call_translation_llm(messages, timeout_seconds)` → content（异步，超时抛错）
- `evaluate_translation_v2(mode, original, user_input)` → { status, feedback, confidence, raw }
- `parse_translation_output(content)` → 解析结果 | None
- `TranslationEvalError`：{ error_code, failure_stage }，error_code ∈
  ASR_UNAVAILABLE / LLM_TIMEOUT / EVAL_UNAVAILABLE / LLM_PARSE_ERROR / NETWORK_ERROR
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from config import (
    TRANSLATION_LLM_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_CHAT_MODEL,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("scholar-admin.translation_eval")

# LLM 输出 JSON 提取（容忍代码块包裹，docs_v1 §6.3）
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# feedback 缺省通用文案（docs_v1 §6.3）
DEFAULT_FEEDBACK = "请对照标准答案再练习一次"

# 默认置信度（docs_v1 §6.3：由模型输出或固定 0.8，仅作记录不做门控）
DEFAULT_CONFIDENCE = 0.8

# 失败阶段（ADR-0022 决策 C，data-model-contract §4.11.2）
STAGE_ASR = "asr"
STAGE_LLM = "llm"
STAGE_PARSE = "parse"

# 错误码（api-contract §3.4 error 枚举）
ERR_ASR_UNAVAILABLE = "ASR_UNAVAILABLE"
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_EVAL_UNAVAILABLE = "EVAL_UNAVAILABLE"
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
ERR_NETWORK_ERROR = "NETWORK_ERROR"

# 中文（CJK）字符检测：original_text 含中文 → ce（中译英），否则 → ec（英译中）
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


class TranslationEvalError(Exception):
    """翻译评估业务失败（不降级，直接置任务 failed）。

    Attributes:
        error_code: ASR_UNAVAILABLE / LLM_TIMEOUT / EVAL_UNAVAILABLE /
                    LLM_PARSE_ERROR / NETWORK_ERROR
        failure_stage: asr / llm / parse
    """

    def __init__(self, error_code: str, failure_stage: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.failure_stage = failure_stage
        self.detail = detail

    def to_dict(self, llm_timeout_seconds: int | None = None, raw=None) -> dict:
        """转为任务 error 对象（data-model-contract §4.16）。"""
        return {
            "error_code": self.error_code,
            "error_detail": self.detail,
            "failure_stage": self.failure_stage,
            "llm_timeout_seconds": llm_timeout_seconds,
            "raw": raw,
        }


def infer_translation_mode(original_text: str) -> str:
    """推导翻译方向：original_text 含中文字符 → ce（中译英），否则 → ec（英译中）。

    契约 v2 入参（api-contract §3.4）无 mode 字段，仅 original_text 可作依据：
    英译中原句为英文（无中文），中译英原句为中文（含中文），启发式可靠。
    """
    return "ce" if _CJK_RE.search(original_text or "") else "ec"


def build_translation_prompt(
    mode: str,
    original: str,
    reference: str | None,
    user_input: str,
) -> list[dict]:
    """构建翻译评分 prompt（docs_v1 §6.1/6.2 骨架）。

    Args:
        mode: "ec"（英译中）/"ce"（中译英）
        original: 原句（ec=英文原句；ce=中文原句）
        reference: 标准答案译文（ec=标准中文释义；ce=标准英文原句）。
            契约 v2 入参未提供（仅 original_text），调用方可传 None——
            此时省略「标准答案」行，仅以 original_text 为评分基准。
        user_input: 用户译文（文字或 ASR 转写）
    """
    if mode == "ce":
        system = (
            "你是英语学习场景的翻译评分专家。请根据「中文原句」与「标准英文原句」，"
            "对学习者的英文译文评分。0-5 整数："
            "5=完全正确；4=基本正确仅轻微瑕疵；3=意思正确但用词或语法有偏差；"
            "2=仅个别词正确；1=几乎不相关；0=未作答或完全错误。"
            "同时用一句话指出问题所在（语法/用词/漏译/中式英语等）并给出改进建议。"
            "只输出 JSON：{\"status\": <0-5>, \"feedback\": \"<中英双语点评>\"}"
        )
        context = f"中文原句：{original}"
        reference_label = "标准英文原句"
    else:
        system = (
            "你是英语学习场景的翻译评分专家。请根据「英文原句」与「标准中文释义」，"
            "对学习者的中文译文评分。0-5 整数："
            "5=完全正确；4=基本正确仅轻微瑕疵；3=意思正确但表达有偏差/遗漏次要信息；"
            "2=仅部分词义正确；1=几乎不相关；0=未作答或完全错误。"
            "同时用一句话指出问题所在（漏译/误译/语序/不自然等）并给出改进建议。"
            "只输出 JSON：{\"status\": <0-5>, \"feedback\": \"<中文点评>\"}"
        )
        context = f"英文原句：{original}"
        reference_label = "标准中文释义"
    if reference:
        context += f"\n{reference_label}：{reference}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context}\n用户译文：{user_input}\n请评分："},
    ]


def _call_translation_llm(messages: list[dict], temperature: float = 0.2) -> str | None:
    """同步调用火山方舟对话模型（OpenAI 兼容）；凭据缺失 / 调用失败返回 None。

    超时由外层 `call_translation_llm` 的 `asyncio.wait_for` 兜底（强取消），
    本函数仅设置 requests 软超时（防止线程池悬挂占满）。
    """
    if not (VOLCANO_API_KEY and VOLCANO_CHAT_MODEL):
        logger.warning("[translation_eval] 未配置火山方舟凭据，无法评分")
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
            # 强制 JSON 输出（OpenAI 兼容字段，方舟支持，docs_v1 §6.3）
            "response_format": {"type": "json_object"},
        },
        timeout=TRANSLATION_LLM_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        logger.error(
            "[translation_eval] 火山方舟返回 %s: %s", resp.status_code, resp.text[:200]
        )
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_translation_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
) -> str | None:
    """LLM 调用异步包装：同步请求丢线程池 + `asyncio.wait_for` 超时强制取消。

    Args:
        timeout_seconds: 超时上限（秒），缺省取 config.TRANSLATION_LLM_TIMEOUT_SECONDS

    Raises:
        TranslationEvalError: 达到超时上限仍未返回 → error_code=LLM_TIMEOUT（stage=llm）
    """
    timeout = timeout_seconds or TRANSLATION_LLM_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_call_translation_llm, messages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[translation_eval] LLM 调用超过 {timeout}s 未返回 → LLM_TIMEOUT"
        )
        raise TranslationEvalError(
            ERR_LLM_TIMEOUT,
            STAGE_LLM,
            f"LLM 调用超过 {timeout}s 未返回（TRANSLATION_LLM_TIMEOUT_SECONDS={timeout}）",
        )


def parse_translation_output(content: str | None) -> dict | None:
    """解析模型输出 JSON（容忍代码块包裹，docs_v1 §6.3）。

    Returns:
        { status: 0-5 int, feedback: str, confidence: float } | None（解析失败）
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
    raw_status = parsed.get("status", parsed.get("score"))
    if not isinstance(raw_status, (int, float)):
        return None
    feedback = str(parsed.get("feedback") or DEFAULT_FEEDBACK).strip()
    try:
        confidence = float(parsed.get("confidence") or DEFAULT_CONFIDENCE)
    except (TypeError, ValueError):
        confidence = DEFAULT_CONFIDENCE
    return {
        "status": max(0, min(5, int(raw_status))),  # 越界 clamp
        "feedback": feedback,
        "confidence": max(0.0, min(1.0, confidence)),
    }


async def evaluate_translation_v2(
    mode: str,
    original: str,
    user_input: str,
    reference: str | None = None,
    timeout_seconds: int | None = None,
) -> dict:
    """翻译评分（不降级）：LLM 调用 → 解析 → { status, feedback, confidence, raw }。

    Raises:
        TranslationEvalError:
        - 模型不可用 / 调用失败 → EVAL_UNAVAILABLE（stage=llm）
        - 超时未返回 → LLM_TIMEOUT（stage=llm）
        - 输出无法解析 → LLM_PARSE_ERROR（stage=parse）
    """
    messages = build_translation_prompt(mode, original, reference, user_input)
    content = await call_translation_llm(messages, timeout_seconds)
    if not content:
        raise TranslationEvalError(
            ERR_EVAL_UNAVAILABLE, STAGE_LLM, "LLM 调用失败（模型不可用或返回空）"
        )
    parsed = parse_translation_output(content)
    if parsed is None:
        raise TranslationEvalError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, f"模型输出解析失败: {content[:200]}"
        )
    parsed["raw"] = content
    return parsed
