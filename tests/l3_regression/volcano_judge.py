"""DeepEval 自定义 judge LLM：火山 OpenAI 兼容接口（LLM_JUDGE_MODEL，S4.1-⑨）。

默认不启用：设 EVAL_VOLCANO_LLM=1 时 regression_weekly.py 使用本模型，
否则 DeepEval 使用内置默认模型（需 OPENAI_API_KEY）。
"""
from __future__ import annotations

import logging
import os

import requests

import config

logger = logging.getLogger("l3_regression.volcano_judge")

try:
    # deepeval 4.x
    from deepeval.llm import DeepEvalBaseLLM
except ImportError:  # pragma: no cover - 旧版本兼容
    from deepeval.models import DeepEvalBaseLLM  # type: ignore


class VolcanoJudgeLLM(DeepEvalBaseLLM):
    """基于火山方舟 OpenAI 兼容接口的 DeepEval judge（模型 = LLM_JUDGE_MODEL）。"""

    def __init__(self, model: str | None = None):
        self._model = model or config.LLM_JUDGE_MODEL

    def load_model(self):
        return self._model

    def generate(self, prompt: str) -> str:
        if not (config.VOLCANO_API_KEY and self._model):
            raise RuntimeError("未配置 VOLCANO_API_KEY / LLM_JUDGE_MODEL，无法使用火山 judge")
        resp = requests.post(
            f"{config.VOLCANO_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.VOLCANO_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            },
            timeout=60,
        )
        if resp.status_code != 200:
            logger.error("火山 judge 返回 %s: %s", resp.status_code, resp.text[:200])
            resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"volcano:{self._model}"


def make_judge_if_enabled():
    """EVAL_VOLCANO_LLM=1 时返回火山 judge，否则 None（DeepEval 默认模型）。"""
    if os.environ.get("EVAL_VOLCANO_LLM") == "1":
        return VolcanoJudgeLLM()
    return None
