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

from pathlib import Path

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

class TestSoeNFailReason:
    """失败原因可区分（2026-09-21 后修）：`evaluate_with_reason` 逐因给出 token。

    这些分支都在 `_load_sdk()`/网络调用之前，用假凭据即可命中，不触网。
    """

    def _provider(self, **kw):
        from services.providers.speech_eval import TencentSoeNProvider

        # 假凭据 → available=True，但不会真正发起 SOE 调用（前置校验先失败）
        return TencentSoeNProvider(appid="appid-x", secret_id="sid-x", secret_key="skey-x", **kw)

    def test_no_credentials(self):
        from services.providers.speech_eval import TencentSoeNProvider

        raw, reason = TencentSoeNProvider(appid="", secret_id="", secret_key="").evaluate_with_reason(
            b"bytes", "hello world"
        )
        assert (raw, reason) == (None, "no_credentials")

    def test_empty_audio(self):
        raw, reason = self._provider().evaluate_with_reason(b"", "hello world")
        assert (raw, reason) == (None, "empty_audio")

    def test_ref_text_too_long(self):
        """超**接受上限 90 词** → ref_text_too_long（2026-09-21 用户拍板 30 → 90）。"""
        long_text = " ".join(f"w{i}" for i in range(91))
        raw, reason = self._provider().evaluate_with_reason(b"bytes", long_text)
        assert (raw, reason) == (None, "ref_text_too_long")

    def test_31_words_no_longer_too_long(self):
        """31 词（旧上限之上）不再判过长 —— 走段落模式，前置校验应放行到 SDK 阶段之后。

        这里只断言「不再因词数被拒」：假凭据 + 31 词时不应返回 ref_text_too_long
        （真实模式选择由 select_eval_mode 单测覆盖；此处不触发网络）。
        """
        text_31 = " ".join(f"w{i}" for i in range(31))
        raw, reason = self._provider().evaluate_with_reason(b"", text_31)  # 空音频先拦，验证词数未拦
        assert reason == "empty_audio"

    def test_ref_text_empty(self):
        raw, reason = self._provider().evaluate_with_reason(b"bytes", "")
        assert (raw, reason) == (None, "ref_text_empty")

    def test_evaluate_legacy_signature_still_returns_raw_only(self):
        """兼容旧签名：`evaluate` 仍只返回 raw（存量调用方零改动）。"""
        assert self._provider().evaluate(b"", "hello world") is None

    def test_fail_reason_text_covers_all_tokens(self):
        from services.providers.speech_eval import SPEECH_FAIL_REASON_TEXT

        for token in ("no_credentials", "empty_audio", "ref_text_too_long", "ref_text_empty",
                      "sdk_missing", "eval_failed", "timeout", "call_error"):
            assert SPEECH_FAIL_REASON_TEXT.get(token), token

class TestSelectEvalMode:
    """评测模式按句长自适应（2026-09-21 后修；SOE 官方：句子 ≤30 词 / 段落 ≤120 词）。"""

    def _mode(self, words, override=None):
        from services.providers.speech_eval import select_eval_mode

        return select_eval_mode(" ".join(f"w{i}" for i in range(words)), override)

    def test_short_text_uses_sentence_mode(self):
        assert self._mode(1) == 1
        assert self._mode(30) == 1          # 官方句子模式上限内

    def test_long_text_switches_to_paragraph_mode(self):
        assert self._mode(31) == 2          # 超句子上限 → 段落模式（不再判过长）
        assert self._mode(90) == 2          # 接受上限内

    def test_override_forces_mode(self):
        assert self._mode(5, "2") == 2      # 强制段落（排障）
        assert self._mode(90, "1") == 1     # 强制句子（排障）
        assert self._mode(5, "auto") == 1   # 非 1/2 → 回落自适应


class TestSdkDir:
    """SDK 源码目录必须指向**仓库根**下的 vendor/（2026-09-22 修复）。

    失效形态：文件由 services/speech_eval.py 搬到 services/providers/speech_eval.py 后，
    `parent.parent` 只到 services/，于是去找不存在的 services/vendor/...，`_load_sdk()`
    在 `SDK_DIR.is_dir()` 处直接返回 False → 评测恒报 `sdk_missing`（本地与线上同命中）。

    只断言路径、不断言目录存在：`vendor/` 被 .gitignore 排除，CI 检出后没有该目录。
    """

    def test_sdk_dir_is_repo_root_vendor(self):
        from services.providers.speech_eval import SDK_DIR

        repo_root = Path(__file__).resolve().parents[2]  # tests/unit/test_x.py → 仓库根
        assert SDK_DIR == repo_root / "vendor" / "tencentcloud-speech-sdk-python"
