"""语言点造句「多轮」出题 LLM Provider（中文情景句 + `reference_en`）

契约：api-contract.md §3.18（E7 / E8「中文情景句规格」）
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §3.3（出题 prompt v1）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B04）

职责（设计 §3.1：新增 provider **只包 prompt 与解析**，通道复用一期）：
- `ROUND_SYSTEM_PROMPT`：出题 system prompt **v1**（JSON-only + 5 条硬约束）；
- `call_round_prompt_llm(messages, ...)`：异步调用 —— 复用 `extension_llm` 的火山方舟
  通道（`json_object` + 关思考 + `wait_for` 强超时），超时取
  `EXTENSION_ROUND_LLM_TIMEOUT_SECONDS`，温度取 `ROUND_TEMPERATURE`（0.3，设计 §3.3）；
- `extract_json(content)`：复用一期实现（**不重复实现**，直接 re-export）；
- `parse_round_prompt(content)`：出题结果强校验，缺 `prompt_zh` → LLM_PARSE_ERROR。

边界：
- **只产出文本，不写库**（R10 / R11 由 B05 链路守护）；
- **失败即抛错、不静默降级**（R13）：超时 → LLM_TIMEOUT、解析失败 → LLM_PARSE_ERROR、
  通道不可用 → PROVIDER_UNAVAILABLE；
- `prompt_version` 由 `config.EXTENSION_ROUND_PROMPT_VERSION` 独立升版（**与抽取版本
  互不牵动**：抽取的幂等键不受出题改版影响）。

Prompt 版本：**v1**（`config.EXTENSION_ROUND_PROMPT_VERSION`）。
"""
from __future__ import annotations

import logging

import config
from services.providers.extension_llm import (
    ERR_LLM_PARSE_ERROR,
    STAGE_PARSE,
    ExtensionError,
    call_extension_llm,
    extract_json,
)

logger = logging.getLogger("scholar-admin.extension_round_llm")

# 出题采样温度（设计 §3.3：0.3 —— 情景句需要多样性，高于判分的 0.2）
ROUND_TEMPERATURE = 0.3

# 出题超时（缺省取 config，120s，对齐 EXTENSION_LLM_TIMEOUT_SECONDS）
ROUND_TIMEOUT_SECONDS = config.EXTENSION_ROUND_LLM_TIMEOUT_SECONDS

# 出题 prompt 版本（与 EXTENSION_PROMPT_VERSION 独立）
PROMPT_VERSION = config.EXTENSION_ROUND_PROMPT_VERSION

# 注册语域枚举（`register` 字段白名单；不在白名单内 → 归一为 neutral）
REGISTERS = ("neutral", "formal", "informal")

# 设计稿写作 ExtensionRoundError：与一期 ExtensionError 同错误码 / 同阶段语义，
# 这里直接复用（路由与任务层只需 catch ExtensionError 一种）。
ExtensionRoundError = ExtensionError


# ===========================================================================
# Prompt v1（出题：中文情景句 + 参考英文句）
# ===========================================================================

ROUND_SYSTEM_PROMPT = """你是英语造句题出题器。给定**目标语言点**与上下文，产出一条**中文情景句**，要求学习者能用该语言点自然地把它译成英文。

## 最重要的一条
**只输出 JSON**，不要任何解释文字、不要 Markdown 代码块。

## 硬约束（违反即不可用）
1. 中文情景句必须能**自然**用目标语言点表达（不要生硬套用、不要凑字数）。
2. 中文里**不得出现目标语言点的英文原文**（后端会二次校验，命中即判失败）。
3. **不得与「已用情景」重复**：场景 / 人物 / 动作**至少换一项**；已用情景见用户输入。
4. 第 1 轮情景贴近原句语境；后续轮按 `difficulty` —— `same` 换场景但保持语域，
   `harder` 换场景**并**升语域（neutral → formal / informal 视情景而定）。
5. `focus_hint` **必须针对上一轮 `errors[0]`**（第 1 轮**恒为空串**）——
   写「该怎么改」，不写正确答案；上一轮无错误时也为空串。

## 字段约束
- `prompt_zh`：中文情景句，一句话，**20~40 字**，口语自然，不出现书名号 / 引号包裹的英文。
- `register`：`neutral` / `formal` / `informal`，与情景匹配；用户输入给了语域偏好时以其为准。
- `scene_tag`：2~3 个场景标签（**小写英文**，如 `work` / `meeting` / `family`），参与去重。
- `reference_en`：该中文情景的**参考英文句**，必须**包含目标语言点原文**，
  语法正确、语域与 `register` 一致；它**不会下发给学习者**（只作判分锚点与示范句兜底）。

## 输出格式（严格 JSON）
{
  "prompt_zh": "会议拖得太久，他终于把话题转到了预算上。",
  "register": "neutral",
  "focus_hint": "注意 take up 后直接接名词，不要加 on。",
  "scene_tag": ["work", "meeting"],
  "reference_en": "He finally took up the question of the budget."
}
"""


async def call_round_prompt_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
    thinking_disabled: bool | None = None,
) -> str:
    """出题 LLM 异步调用（通道复用一期 `extension_llm`）。

    Args:
        messages: `build_round_prompt_messages` 装配的 messages（B05）。
        timeout_seconds: 缺省取 `config.EXTENSION_ROUND_LLM_TIMEOUT_SECONDS`。
        thinking_disabled: 缺省取 `config.EXTENSION_THINKING_DISABLED`（默认关思考）。

    Raises:
        ExtensionError(LLM_TIMEOUT / PROVIDER_UNAVAILABLE)
    """
    timeout = timeout_seconds or ROUND_TIMEOUT_SECONDS
    logger.info(
        f"[extension_round_llm] 出题调用 → prompt_version={PROMPT_VERSION}, "
        f"timeout={timeout}s, temperature={ROUND_TEMPERATURE}"
    )
    return await call_extension_llm(
        messages,
        timeout_seconds=timeout,
        thinking_disabled=thinking_disabled,
        temperature=ROUND_TEMPERATURE,
    )


def parse_round_prompt(content: str) -> dict:
    """解析并强校验出题输出；不合格 → `LLM_PARSE_ERROR`（stage=parse）。

    归一化：`register` 不在白名单 → `neutral`；`scene_tag` 非列表 → `[]`；
    `focus_hint` / `reference_en` 缺失 → 空串。`prompt_zh` 缺失或空 → 判解析失败
    （**不静默降级**：没有情景句就没法出题，R13）。
    """
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ExtensionError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, "出题输出不是 JSON 对象"
        )

    prompt_zh = str(data.get("prompt_zh") or "").strip()
    if not prompt_zh:
        logger.error(
            f"[extension_round_llm] 出题输出缺 prompt_zh → raw={str(content)[:200]}"
        )
        raise ExtensionError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, "出题输出缺少 prompt_zh"
        )

    register = str(data.get("register") or "").strip().lower()
    if register not in REGISTERS:
        register = "neutral"

    scene_tag = data.get("scene_tag") or []
    if not isinstance(scene_tag, list):
        scene_tag = []

    return {
        "prompt_zh": prompt_zh,
        "register": register,
        "focus_hint": str(data.get("focus_hint") or "").strip(),
        "scene_tag": [str(t).strip().lower() for t in scene_tag if str(t).strip()],
        "reference_en": str(data.get("reference_en") or "").strip(),
    }


__all__ = [
    "ROUND_SYSTEM_PROMPT",
    "ROUND_TEMPERATURE",
    "ROUND_TIMEOUT_SECONDS",
    "PROMPT_VERSION",
    "ExtensionRoundError",
    "call_round_prompt_llm",
    "extract_json",
    "parse_round_prompt",
]
