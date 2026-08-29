"""单元测试：翻译评估 v2 评分引擎 services/translation_eval（ADR-0022 决策 B/D）

覆盖（docs_v1 §6.1/6.2/6.3 + §9 不降级 + §12 测试要点）：
- infer_translation_mode  ：原句含中文 → ce，否则 → ec
- build_translation_prompt：ec/ce 分型；reference 可选（None 省略标准答案行）
- parse_translation_output：JSON/代码块/越界 clamp/缺省 feedback 与 confidence/非法输出
- evaluate_translation_v2  ：成功 / EVAL_UNAVAILABLE / LLM_PARSE_ERROR / LLM_TIMEOUT（wait_for 强制取消）
"""
from __future__ import annotations

import asyncio

import pytest

from services.translation_eval import (
    ERR_EVAL_UNAVAILABLE,
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT,
    TranslationEvalError,
    build_translation_prompt,
    evaluate_translation_v2,
    infer_translation_mode,
    parse_translation_output,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# infer_translation_mode
# ---------------------------------------------------------------------------


class TestInferTranslationMode:
    def test_english_original_is_ec(self):
        assert infer_translation_mode("It is a watch.") == "ec"

    def test_chinese_original_is_ce(self):
        assert infer_translation_mode("这是一块手表。") == "ce"

    def test_empty_or_mixed_defaults_to_ec(self):
        assert infer_translation_mode("") == "ec"
        assert infer_translation_mode(None) == "ec"


# ---------------------------------------------------------------------------
# build_translation_prompt
# ---------------------------------------------------------------------------


class TestBuildTranslationPrompt:
    def test_ec_prompt_contains_english_original(self):
        messages = build_translation_prompt(
            "ec", "It is a watch.", "它是一块手表。", "一块表"
        )
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert "英文原句" in system and "标准中文释义" in system
        assert "英文原句：It is a watch." in user
        assert "标准中文释义：它是一块手表。" in user
        assert "用户译文：一块表" in user
        assert '"status"' in system and '"feedback"' in system

    def test_ce_prompt_contains_chinese_original(self):
        messages = build_translation_prompt(
            "ce", "这是一块手表。", "It is a watch.", "It's a watch"
        )
        system = messages[0]["content"]
        user = messages[1]["content"]
        assert "中文原句" in system and "标准英文原句" in system
        assert "中文原句：这是一块手表。" in user
        assert "标准英文原句：It is a watch." in user
        assert "用户译文：It's a watch" in user

    def test_reference_none_omits_standard_answer_line(self):
        """契约 v2 入参无 reference 字段，传 None 时省略标准答案行。"""
        messages = build_translation_prompt("ec", "It is a watch.", None, "一块表")
        user = messages[1]["content"]
        assert "英文原句：It is a watch." in user
        assert "标准中文释义" not in user
        assert "用户译文：一块表" in user


# ---------------------------------------------------------------------------
# parse_translation_output
# ---------------------------------------------------------------------------


class TestParseTranslationOutput:
    def test_plain_json(self):
        parsed = parse_translation_output('{"status": 5, "feedback": "很棒"}')
        assert parsed == {
            "status": 5,
            "feedback": "很棒",
            "confidence": 0.8,  # 缺省固定 0.8
        }

    def test_code_block_wrapped(self):
        content = '```json\n{"status": 3, "feedback": "基本正确"}\n```'
        parsed = parse_translation_output(content)
        assert parsed["status"] == 3
        assert parsed["feedback"] == "基本正确"

    def test_status_clamped(self):
        assert parse_translation_output('{"status": 99}')["status"] == 5
        assert parse_translation_output('{"status": -3}')["status"] == 0

    def test_missing_feedback_uses_default(self):
        parsed = parse_translation_output('{"status": 4}')
        assert parsed["status"] == 4
        assert parsed["feedback"] == "请对照标准答案再练习一次"

    def test_confidence_from_model(self):
        parsed = parse_translation_output('{"status": 2, "confidence": 0.5}')
        assert parsed["confidence"] == 0.5

    def test_invalid_output_returns_none(self):
        assert parse_translation_output(None) is None
        assert parse_translation_output("not json at all") is None
        assert parse_translation_output('{"foo": "bar"}') is None  # 缺 status
        assert parse_translation_output("[1,2,3]") is None  # 非对象


# ---------------------------------------------------------------------------
# evaluate_translation_v2（不降级）
# ---------------------------------------------------------------------------


class TestEvaluateTranslationV2:
    def test_success(self, monkeypatch):
        content = '{"status": 4, "feedback": "用词准确，注意时态", "confidence": 0.85}'
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: content,
        )
        result = _run(evaluate_translation_v2("ec", "It is a watch.", "它是一块表"))
        assert result["status"] == 4
        assert "用词准确" in result["feedback"]
        assert result["confidence"] == 0.85
        assert result["raw"] == content

    def test_llm_unavailable_raises_eval_unavailable(self, monkeypatch):
        # no_external_calls 同款：LLM 返回 None → 不降级，抛 EVAL_UNAVAILABLE
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: None,
        )
        with pytest.raises(TranslationEvalError) as ei:
            _run(evaluate_translation_v2("ec", "It is a watch.", "一块表"))
        assert ei.value.error_code == ERR_EVAL_UNAVAILABLE
        assert ei.value.failure_stage == "llm"

    def test_parse_error_raises_llm_parse_error(self, monkeypatch):
        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm",
            lambda *a, **k: "完全无法解析的输出",
        )
        with pytest.raises(TranslationEvalError) as ei:
            _run(evaluate_translation_v2("ec", "It is a watch.", "一块表"))
        assert ei.value.error_code == ERR_LLM_PARSE_ERROR
        assert ei.value.failure_stage == "parse"

    def test_llm_timeout_raises_llm_timeout(self, monkeypatch):
        """wait_for 强制取消：LLM 挂起超过配置上限 → LLM_TIMEOUT（不无限等待）。"""
        import time as _time

        def slow_llm(*a, **k):
            _time.sleep(0.2)  # 模拟挂起，远超 0.01s 上限

        monkeypatch.setattr(
            "services.translation_eval._call_translation_llm", slow_llm
        )
        with pytest.raises(TranslationEvalError) as ei:
            _run(evaluate_translation_v2("ec", "It is a watch.", "一块表", timeout_seconds=0.01))
        assert ei.value.error_code == ERR_LLM_TIMEOUT
        assert ei.value.failure_stage == "llm"
