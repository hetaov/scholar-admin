"""单元测试:SOE-N 原始结果归一化(F2/2.2,契约 api-contract §3.4.2)

被测函数:services.speech_eval.normalize_soe_result
覆盖:
- 顶层扁平字段形态(F1-2 实测:F1-2 scripts/soe_n_verify.py 输出顶层字段)
- result 子对象嵌套形态(SDK 另一返回路径)
- PronFluency 0~1 → 0~100 归一
- Words 词级 MatchTag 透传(0=命中/2=未命中)、异常条目跳过
- 空/缺字段兜底为 0
"""
from __future__ import annotations

from services.speech_eval import normalize_soe_result


# F1-2 实测形态:顶层扁平字段(见 scripts/soe_n_verify.py 输出)
FLAT_RAW = {
    "voice_id": "soe_n_voice_demo",
    "SuggestedScore": 82.5,
    "PronAccuracy": 78.9,
    "PronFluency": 0.85,  # SOE-N 原值 0~1,归一后 ×100
    "PronCompletion": 90.0,
    "Words": [
        {"Word": "the", "MatchTag": 0},
        {"Word": "quick", "MatchTag": 0},
        {"Word": "brown", "MatchTag": 2},
    ],
    "Pronunciation": {"AvgPronunciation": 0.81},
}


def test_flat_top_level_fields():
    parsed = normalize_soe_result(FLAT_RAW)
    assert parsed["accuracy"] == 78.9
    assert parsed["fluency"] == 85.0  # 0.85 × 100
    assert parsed["completion"] == 90.0
    assert parsed["suggested_score"] == 82.5
    assert parsed["words"] == [
        {"word": "the", "match_tag": 0},
        {"word": "quick", "match_tag": 0},
        {"word": "brown", "match_tag": 2},
    ]


def test_nested_result_fields():
    """兼容形态:字段全部内嵌在 result 子对象中。"""
    raw = {"result": {k: v for k, v in FLAT_RAW.items() if k != "voice_id"}}
    parsed = normalize_soe_result(raw)
    assert parsed["accuracy"] == 78.9
    assert parsed["fluency"] == 85.0
    assert parsed["completion"] == 90.0
    assert parsed["suggested_score"] == 82.5
    assert len(parsed["words"]) == 3


def test_fluency_scaling_from_int():
    """PronFluency 为整数 1(原值 0~1)时 ×100 = 100。"""
    parsed = normalize_soe_result({"PronFluency": 1})
    assert parsed["fluency"] == 100.0


def test_words_skip_malformed_entries():
    """Words 中非 dict / 缺 Word 的条目应跳过。"""
    raw = {
        "Words": [
            {"Word": "ok", "MatchTag": 0},
            "not-a-dict",
            {"MatchTag": 2},  # 缺 Word
            {"Word": "", "MatchTag": 0},  # 空 Word
            {"Word": "keep", "MatchTag": 2},
        ]
    }
    parsed = normalize_soe_result(raw)
    assert parsed["words"] == [
        {"word": "ok", "match_tag": 0},
        {"word": "keep", "match_tag": 2},
    ]


def test_match_tag_default_zero():
    """Word 存在但 MatchTag 缺失 → 默认 0(命中)。"""
    parsed = normalize_soe_result({"Words": [{"Word": "the"}]})
    assert parsed["words"] == [{"word": "the", "match_tag": 0}]


def test_empty_and_missing_fields_fallback_zero():
    assert normalize_soe_result({}) == {
        "accuracy": 0.0,
        "fluency": 0.0,
        "completion": 0.0,
        "suggested_score": 0.0,
        "words": [],
    }
