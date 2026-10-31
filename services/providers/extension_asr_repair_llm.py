"""ASR 判前修复 LLM Provider（合并误断句 + 删紧邻重复 + 最小书写格式）

设计：docs_v1/扩展/第六期-ASR断句修复与判句回显-修改方案-v1.md §4.2 Tier 2
触发：docs_v1/扩展/issue.md（语音造句被 ASR 切成碎片 → 判分反馈失真）

职责（照 `extension_round_llm.py` 的分工：**只包 prompt 与解析**，通道复用一期）：
- `ASR_REPAIR_SYSTEM_PROMPT`：修复 system prompt **v1**（§附录 A，白/黑名单直译）；
- `build_asr_repair_messages(...)`：装配 messages（本场景只需 system + 一条 user，
  比出题的 `build_round_prompt_messages` 简单得多，故就近放 provider，不再多一层）；
- `call_asr_repair_llm(messages, ...)`：异步调用 —— 复用 `extension_llm` 的火山方舟
  通道（`json_object` + 关思考 + `wait_for` 强超时），超时取
  `EXTENSION_ROUND_ASR_REPAIR_TIMEOUT_SECONDS`（30s，短档），温度取
  `EXTENSION_ROUND_ASR_REPAIR_TEMPERATURE`（0）；
- `extract_json(content)`：复用一期实现（**不重复实现**，直接 re-export）；
- `parse_asr_repair(content)`：修复结果强校验，缺 `repaired` → LLM_PARSE_ERROR。

边界：
- **只产出文本，不写库**（落库由 `extension_round.py` B04 / `learning/extension_round.py` B05 负责）；
- **失败即抛错、不静默降级**：超时 → LLM_TIMEOUT、解析失败 → LLM_PARSE_ERROR、
  通道不可用 → PROVIDER_UNAVAILABLE。是否回落由调用方（图 `guard` 节点 / B04）决定 ——
  本 provider 不自己吞异常，以便上层区分「失败」与「无需修复」；
- `prompt_version` 由 `config.EXTENSION_ROUND_ASR_REPAIR_PROMPT_VERSION` 独立升版。

⚠️ 本 provider **不负责**判断「有没有改坏」——那是 `asr_repair_graph.guard` 的职责（R15）。
   prompt 里的禁止条款是**第一道防线**，`guard` 的子序列判据是**结构性第二道防线**。
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

logger = logging.getLogger("scholar-admin.extension_asr_repair_llm")

# 修复采样温度（默认 0，config 可调）：整理任务是确定性行为，低于判分 0.2 / 出题 0.3。
ASR_REPAIR_TEMPERATURE = config.EXTENSION_ROUND_ASR_REPAIR_TEMPERATURE

# 修复超时（默认 30s，config 可调）：**刻意短于**出题/判分的 120s。
# 修复是可选增强，超时即回落原文继续判分（R16），不让学习者多等。
ASR_REPAIR_TIMEOUT_SECONDS = config.EXTENSION_ROUND_ASR_REPAIR_TIMEOUT_SECONDS

# 修复 prompt 版本（独立键，不复用 EXTENSION_ROUND_PROMPT_VERSION）
PROMPT_VERSION = config.EXTENSION_ROUND_ASR_REPAIR_PROMPT_VERSION

# 设计稿写作 AsrRepairError：与一期 ExtensionError 同错误码 / 同阶段语义，
# 这里直接复用（图与链路只需 catch ExtensionError 一种）。
AsrRepairError = ExtensionError


# ===========================================================================
# Prompt v1（修复：整理器，不是批改器）
# ===========================================================================

ASR_REPAIR_SYSTEM_PROMPT = """你是英语学习者「语音作答」的转写整理器。你只做整理，绝不做批改。

## 背景
学习者在做造句练习时用语音作答，因不熟悉目标句，语音识别（ASR）常把一句话切成几段，
或把结巴、重复朗读的内容如实转写下来。你的任务是把这段转写文本整理成**学习者本想说的那一句**。

## 允许的操作，且仅这三类
1. **合并误断句**：一句话被切成多段时，合并为一句；段首字母改为小写
   （`I` 与专有名词除外）；补充或修正标点。
2. **删除紧邻重复**：因结巴、重复朗读造成的**相邻**重复单词或词块
   （`he he` → `he`；`took took up` → `took up`）。
3. **最小书写格式**：空白归一、首字母大写、句末补 `.` `?` 或 `!`（按语义选择）。

## 禁止 —— 违反任一条即原样输出输入文本
1. **增加**任何单词，包括冠词、介词、助动词；
2. **替换**任何单词，或改变词形（时态、单复数、人称、拼写）；
3. **调整词序**；
4. 删除**非紧邻**的重复，或删除**合法语法重复**（`had had`、`that that`、`had had had`）；
5. 数字、缩写的展开或改写（ITN）。

## 特别注意
- 学习者是**初学者**，句子多半不完美。**语法错误必须原样保留**，你的职责不是改作文。
- 拿不准就原样输出 —— 不修复的代价远小于修错。

## 输出格式（严格 JSON，不要解释文字、不要 Markdown 代码块）
{
  "repaired": "整理后的英文句子",
  "dropped": ["被删除的词"],
  "notes": "一句话说明做了什么；无需修复时写 none"
}

无需修复时，`repaired` 与输入**逐字相同**，`dropped` 为空数组。
"""


def build_asr_repair_messages(
    user_input: str,
    *,
    context: dict | None = None,
) -> list[dict]:
    """装配修复 messages（system + 一条 user）。

    Args:
        user_input: 原始 ASR 转写文本（碎片）。
        context: 可选语境 `{prompt_zh, must_use, original, translation}`。
            只作**语境**帮助理解，绝不可当作对齐模板 —— 否则会退化成"照抄参考句"，
            与「不自动纠正」直接冲突（设计 §4.4）。

    Returns:
        OpenAI 风格 messages。
    """
    ctx = context or {}
    lines = [f"待整理的转写文本：\n{user_input}"]

    # 语境只给"要表达什么"，不给"应该怎么写"，避免诱导改写
    prompt_zh = str(ctx.get("prompt_zh") or "").strip()
    if prompt_zh:
        lines.append(f"该轮要求表达的中文情景（仅供参考其意图，不要据此改写英文）：\n{prompt_zh}")

    must_use = ctx.get("must_use") or []
    if isinstance(must_use, list) and must_use:
        points = ", ".join(str(m) for m in must_use if str(m).strip())
        if points:
            lines.append(f"本轮要求用上的语言点（若原文已含，务必保留原样）：\n{points}")

    return [
        {"role": "system", "content": ASR_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(lines)},
    ]


async def call_asr_repair_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
    thinking_disabled: bool | None = None,
) -> str:
    """修复 LLM 异步调用（通道复用一期 `extension_llm`）。

    Args:
        messages: `build_asr_repair_messages` 装配的 messages。
        timeout_seconds: 缺省取 `config.EXTENSION_ROUND_ASR_REPAIR_TIMEOUT_SECONDS`。
        thinking_disabled: 缺省取 `config.EXTENSION_THINKING_DISABLED`（默认关思考）。

    Raises:
        ExtensionError(LLM_TIMEOUT / PROVIDER_UNAVAILABLE)
    """
    timeout = timeout_seconds or ASR_REPAIR_TIMEOUT_SECONDS
    logger.info(
        f"[extension_asr_repair_llm] 修复调用 → prompt_version={PROMPT_VERSION}, "
        f"timeout={timeout}s, temperature={ASR_REPAIR_TEMPERATURE}"
    )
    return await call_extension_llm(
        messages,
        timeout_seconds=timeout,
        thinking_disabled=thinking_disabled,
        temperature=ASR_REPAIR_TEMPERATURE,
    )


def parse_asr_repair(content: str) -> dict:
    """解析并强校验修复输出；不合格 → `LLM_PARSE_ERROR`（stage=parse）。

    归一化：`dropped` 非列表 → `[]`；`notes` 缺失 → 空串。
    `repaired` 缺失或空 → 判解析失败（**不静默降级**：没有修复文本就无法判分）。

    ⚠️ 这里**只做形态校验**，不做"有没有改坏"的语义校验 —— 后者由
    `asr_repair_graph.guard` 的 5 条判据负责（R15）。
    """
    data = extract_json(content)
    if not isinstance(data, dict):
        raise ExtensionError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, "修复输出不是 JSON 对象"
        )

    repaired = str(data.get("repaired") or "").strip()
    if not repaired:
        logger.error(
            f"[extension_asr_repair_llm] 修复输出缺 repaired → raw={str(content)[:200]}"
        )
        raise ExtensionError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, "修复输出缺少 repaired"
        )

    dropped = data.get("dropped") or []
    if not isinstance(dropped, list):
        dropped = []

    return {
        "repaired": repaired,
        "dropped": [str(d).strip() for d in dropped if str(d).strip()],
        "notes": str(data.get("notes") or "").strip(),
    }


__all__ = [
    "ASR_REPAIR_SYSTEM_PROMPT",
    "ASR_REPAIR_TEMPERATURE",
    "ASR_REPAIR_TIMEOUT_SECONDS",
    "PROMPT_VERSION",
    "AsrRepairError",
    "build_asr_repair_messages",
    "call_asr_repair_llm",
    "extract_json",
    "parse_asr_repair",
]
