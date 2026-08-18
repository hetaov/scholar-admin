"""单元测试：EvaluationEngine（P0 后置评估 v1，设计文档 §5.2/§5.3）

覆盖：
- L1 规则：精确匹配 / 部分重叠 / 空输入（异常）/ 低重叠
- L2 Judge：prompt 结构 / 输出解析（JSON、代码块包裹、非法回退）/ 模型不可用回退 L1
- 语音评估：SOE-N parsed → score/confidence/异常标记
- 低置信门控常量（§9-2：<0.6 不回写 SkillState）
"""
from __future__ import annotations

from services.evaluation_engine import (
    LOW_CONFIDENCE_THRESHOLD,
    _call_judge,
    _parse_judge_output,
    build_judge_prompt,
    evaluate_speech,
    evaluate_text,
    l1_rule_evaluate,
)


class TestL1RuleEvaluate:
    def test_exact_match(self):
        v = l1_rule_evaluate("It is a watch.", "it is a watch")
        assert v["score"] == 100
        assert v["meaningful"] is True
        assert v["anomaly"] is False
        assert v["confidence"] >= 0.9

    def test_partial_overlap(self):
        v = l1_rule_evaluate("It is a watch on the desk.", "it is a watch")
        # 词重叠 100%，长度比例 < 0.5 → 落入 0.5 档（faithfulness False）
        assert v["score"] == 70
        assert v["meaningful"] is True
        assert v["faithfulness"] is False

    def test_empty_response_is_anomaly(self):
        v = l1_rule_evaluate("It is a watch.", "")
        assert v["score"] == 0
        assert v["anomaly"] is True
        assert v["confidence"] == 0.9

    def test_low_overlap(self):
        v = l1_rule_evaluate("The weather is nice today.", "banana apple")
        assert v["score"] <= 15
        assert v["meaningful"] is False


class TestJudge:
    def test_prompt_contains_rubric(self):
        msgs = build_judge_prompt("ref", "ans")
        assert msgs[0]["role"] == "system"
        assert "评分卡" in msgs[0]["content"]
        assert msgs[1]["content"] == "参考表达：ref\n用户输出：ans\n请判定："

    def test_parse_valid_json(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluation_engine.LLM_JUDGE_MODEL", "ep-judge-1"
        )
        parsed = _parse_judge_output(
            '{"score": 88, "meaningful": true, "faithfulness": true, '
            '"anomaly": false, "confidence": 0.92}'
        )
        assert parsed is not None
        assert parsed["score"] == 88
        assert parsed["meaningful"] is True
        assert parsed["confidence"] == 0.92
        assert parsed["level"] == "l2"
        assert parsed["judge_model"] == "ep-judge-1"

    def test_parse_fenced_json(self):
        parsed = _parse_judge_output(
            '```json\n{"score": 75, "meaningful": true, "faithfulness": false, '
            '"anomaly": false, "confidence": 0.7}\n```'
        )
        assert parsed is not None
        assert parsed["score"] == 75

    def test_parse_invalid_falls_back(self):
        assert _parse_judge_output("not json at all") is None
        assert _parse_judge_output(None) is None
        assert _parse_judge_output('{"score": "abc"}') is None

    def test_score_clamped(self):
        parsed = _parse_judge_output(
            '{"score": 150, "meaningful": true, "faithfulness": true, '
            '"anomaly": false, "confidence": 1.5}'
        )
        assert parsed["score"] == 100
        assert parsed["confidence"] == 1.0


class TestEvaluateText:
    def test_judge_used_when_available(self, monkeypatch):
        monkeypatch.setattr(
            "services.evaluation_engine._call_judge",
            lambda *a, **k: (
                '{"score": 90, "meaningful": true, "faithfulness": true, '
                '"anomaly": false, "confidence": 0.9}'
            ),
        )
        v = evaluate_text("reference text", "user answer")
        assert v["level"] == "l2"
        assert v["score"] == 90

    def test_fallback_l1_when_judge_unavailable(self, monkeypatch):
        monkeypatch.setattr("services.evaluation_engine._call_judge", lambda *a, **k: None)
        v = evaluate_text("It is a watch.", "it is a watch")
        assert v["level"] == "l1"
        assert v["judge_model"] is None
        assert v["score"] == 100

    def test_empty_response_skips_judge(self, monkeypatch):
        called = {"n": 0}

        def fake_judge(*a, **k):
            called["n"] += 1
            return None

        monkeypatch.setattr("services.evaluation_engine._call_judge", fake_judge)
        v = evaluate_text("ref", "   ")
        assert called["n"] == 0  # 空输入不消耗 Judge
        assert v["anomaly"] is True


class TestEvaluateSpeech:
    def test_suggested_score_priority(self):
        parsed = {
            "accuracy": 80.0,
            "fluency": 70.0,
            "completion": 90.0,
            "suggested_score": 85.0,
        }
        v = evaluate_speech(parsed)
        assert v["score"] == 85
        assert v["meaningful"] is True
        assert v["anomaly"] is False
        assert v["level"] == "l1_speech"

    def test_weighted_when_no_suggested(self):
        parsed = {"accuracy": 60.0, "fluency": 40.0, "completion": 80.0}
        v = evaluate_speech(parsed)
        assert v["score"] == 60  # 0.5*60 + 0.25*40 + 0.25*80 = 60
        assert v["meaningful"] is True

    def test_low_completion_is_anomaly(self):
        parsed = {"accuracy": 10.0, "fluency": 5.0, "completion": 5.0}
        v = evaluate_speech(parsed)
        assert v["anomaly"] is True
        assert v["score"] < 20


class TestLowConfidenceGate:
    def test_threshold_constant(self):
        # §9-2 门控：confidence < 0.6 不回写 SkillState
        assert LOW_CONFIDENCE_THRESHOLD == 0.6

    def test_l1_low_overlap_below_threshold(self):
        v = l1_rule_evaluate("The weather is nice today.", "banana apple")
        assert v["confidence"] < LOW_CONFIDENCE_THRESHOLD
