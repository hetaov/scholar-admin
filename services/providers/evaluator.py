"""口语/翻译评估管线（由云函数 updateTrackingStatus 平移而来）

职责：对「标准句 + 用户输入」产出 { transcription, status }。
- status: 0-5 掌握度档位（5 = 完全掌握），与既有 state 语义一致
- 评分优先走火山方舟对话模型（VOLCANO_CHAT_MODEL），失败回退 levenshtein 兜底

与云函数的差异：
- 评分模型由混元（HUNYUAN）切换为火山方舟（VOLCANO_CHAT_MODEL，scholar-admin 既有范式）
- ASR 由 TC3 手写签名切换为官方 SDK（见 asr.py）
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional, Tuple

from config import VOLCANO_BASE_URL, VOLCANO_API_KEY, VOLCANO_CHAT_MODEL

logger = logging.getLogger("scholar-admin.evaluator")

# 与云函数 updateTrackingStatus 保持一致：
# 0-5 档位，>=5 视为匹配（前端以此判定 isMatch / 移出队列）
MATCH_STATUS = 5
DEFAULT_STATUS = 3

# 标点/空白归一化（与云函数 normalizeText 等价）
_PUNCT_RE = re.compile(r"[^\w\s']")


def _levenshtein(a: str, b: str) -> int:
    """编辑距离（与云函数 levenshtein 等价，DP 实现）"""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def normalize_text(text: str) -> str:
    """去除标点、统一大小写（评分前归一化，与云函数一致）"""
    return _PUNCT_RE.sub("", text).strip().lower()


def _call_volcano(messages: list[dict], temperature: float = 0.2) -> Optional[str]:
    """调用火山方舟对话模型（OpenAI 兼容），失败返回 None"""
    if not (VOLCANO_API_KEY and VOLCANO_CHAT_MODEL):
        logger.warning("[evaluator] 未配置火山方舟凭据，跳过模型评分")
        return None
    try:
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
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(f"[evaluator] 火山方舟返回 {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        logger.error(f"[evaluator] 火山方舟调用异常: {e}")
        return None


def _parse_status(content: str) -> Optional[int]:
    """从模型输出中解析 0-5 档位（容忍 JSON / 裸数字 / 中文）"""
    if not content:
        return None
    # 尝试 JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            for key in ("status", "score", "mastery"):
                val = parsed.get(key)
                if isinstance(val, (int, float)):
                    return max(0, min(5, int(val)))
        elif isinstance(parsed, (int, float)):
            return max(0, min(5, int(parsed)))
    except (json.JSONDecodeError, TypeError):
        pass
    # 裸数字（含中文数字）
    match = re.search(r"(\d+)(?:\.\d+)?", content)
    if match:
        return max(0, min(5, int(match.group(1))))
    return None


def calc_fallback_status(original: str, user_input: str) -> int:
    """无模型可用时的兜底评分（与云函数 calcFallbackStatus 语义一致）：
    - 归一化后完全一致 → 5
    - 编辑距离在容差内 → 4
    - 单词重叠率 >= 0.5 → 3
    - 否则 → 2
    """
    norm_original = normalize_text(original)
    norm_input = normalize_text(user_input)
    if not norm_original or not norm_input:
        return 2
    if norm_original == norm_input:
        return 5
    distance = _levenshtein(norm_original, norm_input)
    # 容差 = max(原始长度 20%, 2)（与云函数一致）
    tolerance = max(int(len(norm_original) * 0.2), 2)
    if distance <= tolerance:
        return 4
    original_words = set(norm_original.split())
    input_words = set(norm_input.split())
    if original_words and input_words:
        overlap = len(original_words & input_words) / len(original_words)
        if overlap >= 0.5:
            return 3
    return 2


def build_assessment_prompt(original: str, user_input: str) -> list[dict]:
    """构建 0-5 掌握度评分 prompt（与云函数 buildAssessmentPrompt 语义一致）"""
    return [
        {
            "role": "system",
            "content": (
                "你是英语口语/翻译水平评估专家。请根据用户对标准句的还原程度给出 0-5 的整数评分："
                "5=完全正确，4=基本正确仅轻微瑕疵，3=意思正确但用词或语法有偏差，"
                "2=仅个别词正确，1=几乎不相关，0=未作答或完全错误。"
                "只输出一个 JSON 对象，不要任何解释，格式：{\"status\": <0-5>}"
            ),
        },
        {
            "role": "user",
            "content": f"标准句：{original}\n用户输入：{user_input}\n请评分：",
        },
    ]


def evaluate(original: str, user_input: str) -> Tuple[int, Optional[str]]:
    """对一段文字输入评分（模型优先，失败回退 levenshtein）。

    Returns:
        (status, raw_model_output) — raw_model_output 用于日志/调试，可为 None
    """
    if not user_input or not user_input.strip():
        return 0, None
    content = _call_volcano(build_assessment_prompt(original, user_input))
    status = _parse_status(content) if content else None
    if status is not None:
        return status, content
    # 模型不可用 / 输出无法解析 → 兜底
    return calc_fallback_status(original, user_input), content


def evaluate_with_asr(
    original: str,
    transcription: str,
) -> Tuple[int, Optional[str]]:
    """对 ASR 转写结果评分（语音路径入口，与文字路径共用 evaluate）"""
    if not transcription or not transcription.strip():
        return 0, None
    return evaluate(original, transcription)
