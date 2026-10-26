"""ASR 转写文本后处理（智能标点）。

背景：腾讯云 ASR「一句话识别」（`services/providers/asr.py` 的 `recognize`）返回的是
**无标点裸文本** —— SentenceRecognition 接口不提供标点/ITN 参数，要拿到真·标点要么换
厂家长音频产品（`ResTextFormat` 异步接口）、要么换引擎。故先做一层**低成本后处理**，
补齐最基础的书面格式，让「你的回答」这类展示区可读。

⚠️ 刻意**不做**这两件事（规则式极易误伤，且可能污染下游判分）：
  - ITN（逆文本正则化，如 "twenty two" → "22"）：需引擎层支持，规则化会把
    "I have two books" 之类的文本改坏；
  - 句内逗号 / 引号等韵律标点：缺停顿信息，无法可靠还原，宁可不加。

开关：`config.ASR_SMART_PUNCTUATION`（默认 0 关 ⇒ 今日行为逐字不变），
      2026-10-26 起在 `POST /eval/transcribe` 挂载，待实测确认收益后再决定是否全量。
"""
from __future__ import annotations

import re

_TERMINAL = ".!?"
_WS = re.compile(r"\s+")


def smart_punctuate(text: str | None) -> str | None:
    """给 ASR 裸文本补最基础的书面标点：空白归一 + 首字母大写 + 句末标点。

    Args:
        text: ASR 原始转写文本（可能为 None / 空串 / 全空白）。

    Returns:
        补齐后的文本；入参为 falsy 或纯空白时**原样返回**
        （不无中生有地造标点，避免下游把空白误判成有内容）。
    """
    if not text:
        return text
    compact = _WS.sub(" ", text).strip()
    if not compact:
        return text  # 纯空白：保持原样，交由调用方按「空」处理
    titled = compact[0].upper() + compact[1:]
    if titled[-1] not in _TERMINAL:
        titled += "."
    return titled
