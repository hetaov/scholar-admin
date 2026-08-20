"""P0 后置评估 v1：EvaluationEngine（L1 规则 + L2 LLM-as-a-Judge）

职责（设计文档 §5.2/§5.3，契约 data-model-contract §4.11.2）：
- L1 规则：轻量规则评估（无语义成本、可离线），产出 score/verdict 与置信度；
- L2 LLM Judge：Judge ≠ Generator（LLM_JUDGE_MODEL 独立配置，附录 B-4），
  基于评分卡 rubric 产出结构化 JSON（score / meaningful / faithfulness / anomaly / confidence）；
- 证据不可改、评价可重算：原始证据（raw）落库后不可变，重算只改评价部分（幂等）。

对外函数：
- evaluate_text(original, response)      → dict（L1+L2 综合评估，异常/空输入走 L1）
- evaluate_speech(parsed)                → dict（基于 SOE-N parsed 的 L1 规则评估）
- build_judge_prompt / _call_judge       → L2 Judge 链路（可注入替换，便于测试）
- confidence 门控常量 LOW_CONFIDENCE_THRESHOLD = 0.6（§9-2：低置信不回写 SkillState）

与既有 evaluator.py 的关系：evaluator.py 面向 /eval/translate 的 0-5 档位评分；
本模块面向「达意/忠实/异常/置信度」四维 eval_verdict，为会话/训练闭环的证据层。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from config import (
    EVAL_CONFIDENCE_THRESHOLD,
    LLM_JUDGE_MODEL,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
from services.evaluator import normalize_text

logger = logging.getLogger("scholar-admin.evaluation_engine")

# evaluation 证据集合（data-model-contract §4.11.2）
EVALUATION_COLLECTION = "evaluation"

# 低置信门控阈值（§9-2，config.py EVAL_CONFIDENCE_THRESHOLD，默认 0.6）：
# confidence < 阈值 不回写 SkillState
LOW_CONFIDENCE_THRESHOLD = EVAL_CONFIDENCE_THRESHOLD

# L2 评分卡 rubric（评分维度说明，随 prompt 下发）
_RUBRIC = {
    "score": "0-100 整数：与参考表达整体契合度（内容+语言质量）",
    "meaningful": "布尔：用户输出是否达意（与参考语义一致，允许措辞差异）",
    "faithfulness": "布尔：用户输出是否忠实于参考（无关键信息遗漏/偏离）",
    "anomaly": "布尔：是否异常（空输入/答非所问/乱码/完全不相关）",
    "confidence": "0-1 小数：本次判定置信度",
}

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _word_overlap(original: str, response: str) -> float:
    """参考与作答的词级重叠率（基于归一化分词，L1 规则核心信号）。"""
    o_words = set(original.split())
    r_words = set(response.split())
    if not o_words or not r_words:
        return 0.0
    return len(o_words & r_words) / len(o_words)


def l1_rule_evaluate(original: str, response: str) -> dict:
    """L1 规则评估：空/异常输入 → 0 分 + anomaly；词重叠+长度比例 → 分数与置信度。

    Returns:
        {score, meaningful, faithfulness, anomaly, confidence, level: "l1"}
    """
    norm_o = normalize_text(original)
    norm_r = normalize_text(response)
    if not norm_o or not norm_r:
        return {
            "score": 0,
            "meaningful": False,
            "faithfulness": False,
            "anomaly": True,
            "confidence": 0.9,
            "level": "l1",
        }

    if norm_o == norm_r:
        return {
            "score": 100,
            "meaningful": True,
            "faithfulness": True,
            "anomaly": False,
            "confidence": 0.95,
            "level": "l1",
        }

    overlap = _word_overlap(norm_o, norm_r)
    # 长度比例惩罚：作答过短（明显漏译）降分
    len_ratio = len(norm_r) / max(len(norm_o), 1)
    if overlap >= 0.8 and len_ratio >= 0.5:
        score = 90
        meaningful, faithfulness = True, True
        confidence = 0.8
    elif overlap >= 0.5 and len_ratio >= 0.3:
        score = 70
        meaningful, faithfulness = True, False
        confidence = 0.65
    elif overlap >= 0.2:
        score = 40
        meaningful, faithfulness = False, False
        confidence = 0.55
    else:
        score = 15
        meaningful, faithfulness = False, False
        # 完全不相关 → 置信度低于门控（<0.6），低置信不回写 SkillState（§9-2）
        confidence = 0.5

    return {
        "score": score,
        "meaningful": meaningful,
        "faithfulness": faithfulness,
        "anomaly": False,
        "confidence": confidence,
        "level": "l1",
    }


# ---------------------------------------------------------------------------
# L2 LLM-as-a-Judge
# ---------------------------------------------------------------------------


def build_judge_prompt(original: str, response: str) -> list[dict]:
    """构建 L2 Judge 评分卡 prompt（结构化 JSON 输出，禁止解释）。"""
    rubric_lines = "\n".join(f"- {k}: {v}" for k, v in _RUBRIC.items())
    return [
        {
            "role": "system",
            "content": (
                "你是英语学习场景的客观评估裁判（Judge）。请依据评分卡对「参考表达」与「用户输出」"
                "逐项判定，只输出一个 JSON 对象，不要任何解释。\n评分卡：\n" + rubric_lines
            ),
        },
        {
            "role": "user",
            "content": f"参考表达：{original}\n用户输出：{response}\n请判定：",
        },
    ]


def _call_judge(messages: list[dict], temperature: float = 0.0) -> Optional[str]:
    """调用 LLM_JUDGE_MODEL（Judge ≠ Generator 独立配置）；失败返回 None（回落 L1）。"""
    if not (VOLCANO_API_KEY and LLM_JUDGE_MODEL):
        logger.warning("[evaluation_engine] 未配置 LLM_JUDGE_MODEL，回落 L1 规则")
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
                "model": LLM_JUDGE_MODEL,
                "messages": messages,
                "temperature": temperature,
                # 强制 JSON 输出（OpenAI 兼容字段，方舟支持）
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            logger.error(
                "[evaluation_engine] Judge 模型返回 %s: %s", resp.status_code, resp.text[:200]
            )
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:  # noqa: BLE001
        logger.error("[evaluation_engine] Judge 调用异常: %s", e)
        return None


def _parse_judge_output(content: Optional[str]) -> Optional[dict]:
    """解析 Judge 输出 JSON，容忍代码块包裹；非法/缺关键字段返回 None（回落 L1）。"""
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

    try:
        return {
            "score": int(max(0, min(100, float(parsed.get("score", 0))))),
            "meaningful": bool(parsed.get("meaningful", False)),
            "faithfulness": bool(parsed.get("faithfulness", False)),
            "anomaly": bool(parsed.get("anomaly", False)),
            "confidence": _clamp(float(parsed.get("confidence", 0.5))),
            "level": "l2",
            "judge_model": LLM_JUDGE_MODEL,
        }
    except (TypeError, ValueError):
        return None


def evaluate_text(original: str, response: str) -> dict:
    """综合评估（L1 规则 → L2 Judge 补充；L2 不可用/输出非法回落 L1）。

    空输入/异常输入直接走 L1（不消耗 Judge 调用）。低置信（<0.6）由
    LOW_CONFIDENCE_THRESHOLD 常量暴露给上层路由做门控，本函数不决定回写。
    """
    if not response or not response.strip():
        verdict = l1_rule_evaluate(original, response or "")
        verdict["judge_model"] = None
        return verdict

    content = _call_judge(build_judge_prompt(original, response))
    judge = _parse_judge_output(content)
    if judge is not None:
        return judge

    # Judge 不可用/解析失败 → 回落 L1，并保留 judge_model=None 表明未走 L2
    verdict = l1_rule_evaluate(original, response)
    verdict["judge_model"] = None
    return verdict


def evaluate_speech(parsed: dict) -> dict:
    """基于 SOE-N 归一化结果（§4.9 parsed）的 L1 规则评估（不调 Judge）。

    parsed 字段：accuracy / fluency / completion / suggested_score（0~100）。
    - 综合分 = suggested_score（优先）或三项加权（0.5/0.25/0.25）
    - confidence 由 completion 支撑（完成度越高越可信）
    """
    suggested = float(parsed.get("suggested_score") or 0.0)
    if suggested > 0:
        score = suggested
    else:
        score = (
            0.5 * float(parsed.get("accuracy") or 0.0)
            + 0.25 * float(parsed.get("fluency") or 0.0)
            + 0.25 * float(parsed.get("completion") or 0.0)
        )
    score = max(0.0, min(100.0, score))
    completion = float(parsed.get("completion") or 0.0)
    meaningful = score >= 60
    anomaly = completion < 10
    confidence = _clamp(0.4 + completion / 100.0 * 0.55)  # 0.40 ~ 0.95

    return {
        "score": int(round(score)),
        "meaningful": meaningful,
        "faithfulness": meaningful,  # 语音评测以达意为主，忠实取同值
        "anomaly": anomaly,
        "confidence": confidence,
        "level": "l1_speech",
        "judge_model": None,
    }
