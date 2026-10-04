"""英文语句扩展 LLM Provider（抽取 + L2 评测）

契约：api-contract.md §3.18；设计：第一期设计 §3.4。

职责：
- `_call_extension_llm(messages, temperature=0.2, thinking_disabled)`：同步调用火山方舟
  （OpenAI 兼容），`response_format=json_object`，`requests` 软超时；
  凭据缺失 / 非 200 → 返回 None。
- `call_extension_llm(messages, timeout_seconds, thinking_disabled)`：异步包装，
  `asyncio.wait_for(run_in_threadpool(...))` 强超时 → 抛 ExtensionError(LLM_TIMEOUT)。
- `thinking_disabled`（默认取 config.EXTENSION_THINKING_DISABLED=1）：透传
  `thinking.type=disabled`，与会话面 SESSION_V4_THINKING_DISABLED 同一结论——
  推理型模型先吐 reasoning_content 会显著拉长 content 产出时间，抽取任务无需推理。
- Provider 可插拔：模型取 config.EXTENSION_LLM_MODEL（缺省回落 VOLCANO_CHAT_MODEL）。
- Prompt v2（分层抽取）：模块级常量 + build_extension_messages(original, translation, candidates)。
  模型只对**候选清单**逐条判定并回 `candidate_id`，不自造 text/span
  （策略见 `docs_v1/扩展/重点词汇短语抽取策略.md` §3/§5）。

错误阶段：llm / parse；错误码：LLM_TIMEOUT / LLM_PARSE_ERROR / PROVIDER_UNAVAILABLE。
"""
from __future__ import annotations

import asyncio
import logging
import re

import config
from config import (
    EXTENSION_LLM_TIMEOUT_SECONDS,
    EXTENSION_THINKING_DISABLED,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
from starlette.concurrency import run_in_threadpool

from services.english.extension_candidates import expand_abbreviations

logger = logging.getLogger("scholar-admin.extension_llm")

# LLM 输出 JSON 提取（容忍代码块包裹）
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# 失败阶段
STAGE_LLM = "llm"
STAGE_PARSE = "parse"

# 错误码（api-contract §3.18 业务码表）
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_LLM_PARSE_ERROR = "LLM_PARSE_ERROR"
ERR_PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class ExtensionError(Exception):
    """扩展链路业务失败（不降级，直接置任务 failed）。

    Attributes:
        error_code: LLM_TIMEOUT / LLM_PARSE_ERROR / PROVIDER_UNAVAILABLE
        failure_stage: llm / parse
    """

    def __init__(self, error_code: str, failure_stage: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.failure_stage = failure_stage
        self.detail = detail

    def to_dict(self, llm_timeout_seconds: int | None = None, raw=None) -> dict:
        return {
            "error_code": self.error_code,
            "error_detail": self.detail,
            "failure_stage": self.failure_stage,
            "llm_timeout_seconds": llm_timeout_seconds,
            "raw": raw,
        }


# ===========================================================================
# Prompt v2（分层抽取；升版改 EXTENSION_PROMPT_VERSION 即自然失效缓存）
# 策略：docs_v1/扩展/重点词汇短语抽取策略.md §3/§5
# ===========================================================================

EXTENSION_SYSTEM_PROMPT = """你是英语学习内容审核器。给定英文原句与**已生成好的候选清单**，从候选中筛出最值得学习的语言点，并给出释义。

## 最重要的一条
**只能从候选清单中选择**：每条语言点用 `candidate_id`（如 c3）指代候选。
- **不得新增候选、不得自造文本或 span**（后端会以候选自带的 span 切原文落库）；
- **不要输出 `text` / `span` 字段**——只输出 `candidate_id`。

## 判定标准
1. 不提取基础词，也不提取可直接按字面理解的临时组合（如 "the discussion"）。
2. 短语优先固定搭配、习语、phrasal verbs 和高复用表达。
3. 同一字符区间只能归一类，固定度优先级 idiom > slang > phrase > word（重叠时取高优先级）。
4. 释义必须基于**当前上下文**，不是词典首义。
5. slang 必须区分常见口语 / 地区性 / 过时 / 粗俗冒犯，并填 `literal_zh`（字面义）、`register=informal`。
6. 慎用表达（粗俗 / 冒犯 / 强地域性 / 已过时）必须填 `risk`，取值 offensive | vulgar | regional | dated。

## 配额（合计 ≤ 5）
- word ≤ 3、phrase ≤ 2、slang ≤ 1、idiom ≤ 1

## 字段约束
- `meaning_zh` ≤ 30 字；`reason` ≤ 40 字
- `example_en` 必须包含该语言点在原句中的原文（大小写不敏感）
- `phrase` 必须填 `collocations`（≥1）
- `confidence` 为 0–1 的置信度；结果按学习价值从高到低排序
- `confusable` 必填 **≥3 个**易混淆表达（近义 / 形近 / 同词根），用于生成 L1 填空干扰项；
  确实难凑也要尽量给满 3 个，**不要留空数组**；不要与 `text` 自身互为子串

## 输出格式（严格 JSON，不要任何解释文字）
{
  "points": [
    {
      "candidate_id": "c3",
      "type": "word|phrase|slang|idiom",
      "pos": "词性（word 必填）",
      "meaning_zh": "...",
      "literal_zh": "字面义（slang 必填，其余可空）",
      "register": "neutral|formal|informal",
      "example_en": "...",
      "example_zh": "...",
      "collocations": ["..."],
      "synonyms": ["..."],
      "confusable": ["..."],
      "cefr": "A1|A2|B1|B2|C1|C2",
      "confidence": 0.93,
      "risk": "",
      "reason": "..."
    }
  ]
}

若候选中没有值得学习的语言点，返回 {"points": []}。
"""


def build_extension_messages(
    original: str,
    translation: str = "",
    candidates: list[dict] | None = None,
) -> list[dict]:
    """构建抽取 prompt messages（含编号候选清单与缩写提示）。

    Args:
        candidates: `services/english/extension_candidates.py::generate_candidates`
            产出的 `[{candidate_id, text, span, hint_type, source}]`。
    """
    candidates = candidates or []

    user = f"英文原句：{original}\n"
    if translation:
        user += f"中文译文：{translation}\n"

    # 缩写提示（gonna → going to）：仅作上下文，不参与返回
    hints = expand_abbreviations(original)
    if hints:
        user += "缩写提示（仅供理解，不得据此改写返回文本）：" + "；".join(
            f"{a} = {b}" for a, b in hints
        ) + "\n"

    if candidates:
        lines = [
            f"- {c['candidate_id']}｜{c['text']}｜span={c['span'][0]}:{c['span'][1]}"
            f"｜提示类型={c['hint_type']}"
            for c in candidates
        ]
        user += "候选清单（只能从下列候选中选择，用 candidate_id 指代）：\n"
        user += "\n".join(lines) + "\n"
    else:
        user += "候选清单为空。\n"

    user += "请从候选中筛选并输出语言点。"
    return [
        {"role": "system", "content": EXTENSION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ===========================================================================
# LLM 调用
# ===========================================================================


def _thinking_field(thinking_disabled: bool) -> dict:
    """方舟「思考」开关字段：关思考时返回 `{"thinking": {"type": "disabled"}}`，否则空。

    同 `services/providers/dialogue_gen.py::_thinking_field`（会话面 v4 已验证）：
    推理型模型会先产出 reasoning_content，content 几乎到最后才出；抽取只需分类与
    填字段，关思考可把单次调用耗时压到秒级，输出结构不变。
    """
    return {"thinking": {"type": "disabled"}} if thinking_disabled else {}


def _call_extension_llm(
    messages: list[dict],
    temperature: float = 0.2,
    thinking_disabled: bool = False,
) -> str | None:
    """同步调用火山方舟对话模型（OpenAI 兼容）；凭据缺失 / 调用失败返回 None。

    超时由外层 call_extension_llm 的 asyncio.wait_for 兜底。
    """
    model = config.EXTENSION_LLM_MODEL
    if not (VOLCANO_API_KEY and model):
        logger.warning("[extension_llm] 未配置火山方舟凭据或模型，无法调用")
        return None
    import requests

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    payload.update(_thinking_field(thinking_disabled))

    resp = requests.post(
        f"{VOLCANO_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=EXTENSION_LLM_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        logger.error(
            "[extension_llm] 火山方舟返回 %s: %s", resp.status_code, resp.text[:200]
        )
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_extension_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
    thinking_disabled: bool | None = None,
) -> str:
    """LLM 调用异步包装：同步请求丢线程池 + asyncio.wait_for 超时强制取消。

    Args:
        thinking_disabled: 是否透传 `thinking.type=disabled`；None → 取
            `config.EXTENSION_THINKING_DISABLED`（默认 1 = 关思考）。

    Raises:
        ExtensionError: 超时 → LLM_TIMEOUT(stage=llm)；返回 None → PROVIDER_UNAVAILABLE
    """
    timeout = timeout_seconds or EXTENSION_LLM_TIMEOUT_SECONDS
    if thinking_disabled is None:
        thinking_disabled = bool(EXTENSION_THINKING_DISABLED)
    try:
        content = await asyncio.wait_for(
            run_in_threadpool(
                _call_extension_llm, messages, thinking_disabled=thinking_disabled
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"[extension_llm] LLM 调用超过 {timeout}s 未返回 → LLM_TIMEOUT"
        )
        raise ExtensionError(
            ERR_LLM_TIMEOUT,
            STAGE_LLM,
            f"LLM 调用超过 {timeout}s 未返回",
        )
    if content is None:
        raise ExtensionError(
            ERR_PROVIDER_UNAVAILABLE,
            STAGE_LLM,
            "LLM 服务不可用（凭据缺失或调用失败）",
        )
    return content


def extract_json(content: str) -> dict | None:
    """从模型输出中提取 JSON 对象（容忍代码块 / 前后缀）。"""
    if not content:
        return None
    import json

    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
