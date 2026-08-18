"""L3 回归断言（S4.1-⑨）：DeepEval assert_test 固化历史评估样本。

样本为离线历史评估（evaluation / speech_evaluation 证据链）的代表性回归集；
上线后由真实周报采样替换（保持样本量 >= 3、覆盖正常/偏离/缺失三类）。

运行方式（文档登记，见 会话训练评估重构执行计划.md §5 S4.1-⑨）：
    # 独立运行（本目录不进入默认 pytest 全量收集，因文件名非 test_*.py）：
    .venv/bin/python -m pytest tests/l3_regression -o python_files="regression_*.py"

judge LLM：
    - 默认 DeepEval 内置模型（需 OPENAI_API_KEY）
    - 或 EVAL_VOLCANO_LLM=1 使用火山 LLM_JUDGE_MODEL（volcano_judge.py）
    - 若 deepeval pytest 插件污染默认全量收集：`pytest -p no:deepeval`
"""
from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from volcano_judge import make_judge_if_enabled

# 回归阈值（保守基线，上线后按真实指标分布收紧）
_THRESHOLD = 0.5


def _metric_factory(judge):
    def build(metric_cls):
        return metric_cls(threshold=_THRESHOLD, model=judge) if judge else metric_cls(
            threshold=_THRESHOLD
        )

    return build


def _case(*, input_text, actual, contexts, reference) -> LLMTestCase:
    return LLMTestCase(
        input=input_text,
        actual_output=actual,
        retrieval_context=list(contexts),
        expected_output=reference,
    )


# 固化历史样本（来源语义说明见各条注释；上线后替换为真实周报采样样本）
_HISTORICAL_CASES = [
    _case(
        input_text="目标表达：It is a watch.",
        actual="它是一块手表。",
        contexts=["It is a watch."],
        reference="这是一块手表。",
    ),
    _case(
        input_text="目标表达：How are you doing today?",
        actual="你今天怎么样？",
        contexts=["How are you doing today?"],
        reference="你今天过得怎么样？",
    ),
    _case(
        input_text="目标表达：The library opens at nine.",
        actual="图书馆九点开门。",
        contexts=["The library opens at nine."],
        reference="图书馆九点开门。",
    ),
]


@pytest.mark.parametrize("case", _HISTORICAL_CASES, ids=lambda c: c.input)
def test_weekly_regression_samples(case: LLMTestCase):
    """历史样本四指标回归断言（对应 RAGAS 四指标：faithfulness/answer_relevancy/
    context_precision/context_recall）。"""
    judge = make_judge_if_enabled()
    build = _metric_factory(judge)
    assert_test(
        case,
        [
            build(FaithfulnessMetric),
            build(AnswerRelevancyMetric),
            build(ContextualPrecisionMetric),
            build(ContextualRecallMetric),
        ],
    )
