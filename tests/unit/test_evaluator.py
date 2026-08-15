"""单元测试:评估管线(4.6.5a) — services/evaluator

覆盖:
- normalize_text: 标点/大小写归一化
- _levenshtein / calc_fallback_status: 无模型兜底评分(平移云函数语义)
- _parse_status: 模型输出解析(JSON/裸数字/中文数字)
- build_assessment_prompt: prompt 结构
- evaluate: 模型优先, 失败回退兜底
"""
from __future__ import annotations

from services.evaluator import (
    _call_volcano,
    _levenshtein,
    _parse_status,
    build_assessment_prompt,
    calc_fallback_status,
    evaluate,
    normalize_text,
)


class TestNormalizeText:
    def test_strip_punctuation_lowercase(self):
        assert normalize_text("Hello, World!") == "hello world"
        assert normalize_text("It's a watch.") == "it's a watch"
        assert normalize_text("  HELLO  ") == "hello"
        assert normalize_text("") == ""
        assert normalize_text("你好，世界") == "你好世界"


class TestLevenshtein:
    def test_basic(self):
        assert _levenshtein("", "") == 0
        assert _levenshtein("abc", "") == 3
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("kitten", "sitting") == 3
        assert _levenshtein("watch", "watch") == 0
        assert _levenshtein("watch", "whatch") == 1  # 相邻交换记 2, 替换记 1


class TestCalcFallbackStatus:
    def test_exact_match_status_5(self):
        assert calc_fallback_status("What is this?", "What is this?") == 5
        # 标点/大小写差异仍算完全一致(归一化后)
        assert calc_fallback_status("It is a watch.", "it is a watch") == 5

    def test_close_match_status_4(self):
        # 单字符替换在容差内 → 4
        assert calc_fallback_status("It is a watch.", "It is a witch.") == 4
        # 轻微遗漏 → 4
        assert calc_fallback_status("This is a book.", "This is book.") == 4

    def test_partial_overlap_status_3(self):
        # 单词重叠 >= 50% → 3
        assert calc_fallback_status("This is a red car.", "It is a car.") == 3

    def test_low_overlap_status_2(self):
        assert calc_fallback_status("This is a red car.", "I like apples.") == 2
        # 空输入 → 2(调用方对空输入另有 0 处理, 见 evaluate)
        assert calc_fallback_status("Hello", "") == 2

    def test_consistent_with_evaluate_when_model_unavailable(self, monkeypatch):
        monkeypatch.setattr("services.evaluator._call_volcano", lambda *a, **k: None)
        assert evaluate("What is this?", "What is this?")[0] == 5
        assert evaluate("Hello", "")[0] == 0  # 空输入直接 0


class TestParseStatus:
    def test_json_object(self):
        assert _parse_status('{"status": 5}') == 5
        assert _parse_status('{"score": 3}') == 3
        assert _parse_status('{"mastery": 2}') == 2

    def test_clamped(self):
        assert _parse_status('{"status": 7}') == 5
        assert _parse_status('{"status": -1}') == 0

    def test_bare_number(self):
        assert _parse_status("5") == 5
        assert _parse_status("  4  ") == 4
        assert _parse_status("评分：3分") == 3

    def test_invalid(self):
        assert _parse_status("") is None
        assert _parse_status("完全无法解析的文本") is None


class TestBuildAssessmentPrompt:
    def test_structure(self):
        msgs = build_assessment_prompt("It is a watch.", "It is a witch.")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert "0-5" in msgs[0]["content"]
        assert msgs[1]["role"] == "user"
        assert "It is a watch." in msgs[1]["content"]
        assert "It is a witch." in msgs[1]["content"]


class TestEvaluate:
    def test_model_priority(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: '{"status": 5}'
        )
        assert evaluate("Hello", "hello")[0] == 5

    def test_empty_input_returns_0_even_with_model(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: '{"status": 5}'
        )
        assert evaluate("Hello", "   ")[0] == 0

    def test_unparseable_model_output_falls_back(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: "我拒绝回答"
        )
        assert evaluate("It is a watch.", "it is a watch")[0] == 5  # 兜底精确匹配

    def test_returns_raw_output_for_debug(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluator._call_volcano", lambda *a, **k: '{"status": 4}'
        )
        status, raw = evaluate("Hello", "hello world")
        assert status == 4
        assert raw == '{"status": 4}'
