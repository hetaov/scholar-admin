"""共享假 Provider（T1.4）：TTS / SOE-N / ASR 的内存替身。

统一三态模型（与 services 接口语义一致）：
- available=False      → 无真实凭据，业务层返回 *_UNAVAILABLE
- result=None 且可用   → 调用失败，业务层同样降级
- 其他 result          → 正常返回（可注入自定义 raw 以覆盖断言场景）

所有实现记录调用次数与入参，便于断言"被调用了几次 / 收到什么"。
测试文件不应再自行定义 Fake*Provider，统一 import 本模块。
"""

from __future__ import annotations


class FakeProviderBase:
    """基类：available 标志 + result + 调用记录。"""

    def __init__(self, result, available: bool = True):
        self.raw = result
        self.available = available
        self.calls: list[tuple] = []
        self.call_count = 0

    def _invoke(self, *args):
        self.calls.append(args)
        self.call_count += 1
        return self.raw if self.available else None

    @classmethod
    def unavailable(cls):
        """无凭据替身：available=False，业务走 *_UNAVAILABLE。"""
        return cls(result=None, available=False)

    @classmethod
    def failing(cls):
        """可用但调用失败替身：返回 None，业务同样降级。"""
        return cls(result=None, available=True)


class FakeAsrService(FakeProviderBase):
    """假 ASR：返回固定转写文本（默认 "it is a watch"）。"""

    DEFAULT_RESULT = "it is a watch"

    def __init__(self, result: str | None = DEFAULT_RESULT, available: bool = True):
        super().__init__(result, available)

    def recognize(self, audio_bytes: bytes, voice_format: str = "mp3") -> str | None:
        return self._invoke(audio_bytes, voice_format)


class FakeTtsProvider(FakeProviderBase):
    """假 TTS Provider：返回固定合成结果，记录合成文本。"""

    DEFAULT_RAW = {
        "Audio": "ZmFrZS1tcDMtYXVkaW8tYnl0ZXM=",  # base64("fake-mp3-audio-bytes")
        "RequestId": "req-tts-001",
        "SessionId": "sess-abc-001",
    }

    def __init__(self, result=DEFAULT_RAW, available: bool = True):
        super().__init__(result, available)
        self.called_texts: list[str] = []

    def synthesize(self, text):
        self.called_texts.append(text)
        return self._invoke(text)


class FakeSpeechProvider(FakeProviderBase):
    """假 SOE-N Provider：返回固定原始评测 JSON。"""

    DEFAULT_RAW = {
        "voice_id": "soe_n_voice_demo",
        "SuggestedScore": 82.5,
        "PronAccuracy": 78.9,
        "PronFluency": 0.85,
        "PronCompletion": 90.0,
        "Words": [
            {"Word": "the", "MatchTag": 0},
            {"Word": "quick", "MatchTag": 0},
            {"Word": "brown", "MatchTag": 2},
        ],
    }

    def __init__(self, result=DEFAULT_RAW, available: bool = True, fail_reason: str = "eval_failed"):
        super().__init__(result, available)
        # 失败原因 token（2026-09-21 后修）：真实 Provider 的 evaluate_with_reason 会区分成因，
        # 替身需同样能给出原因，才能覆盖「本句过长（ref_text_too_long）」这类可分流场景。
        self.fail_reason = fail_reason

    def evaluate(self, audio_bytes, ref_text, voice_format="mp3"):
        return self._invoke(audio_bytes, ref_text, voice_format)

    def evaluate_with_reason(self, audio_bytes, ref_text, voice_format="mp3"):
        """带原因的评测替身：成功 → (raw, None)；失败 → (None, fail_reason)。

        与 ABC 默认实现同构（**委托 `self.evaluate`**）——这样子类覆写 `evaluate` 的记录型替身
        （测试里常见的 RecordingProvider）依然被走到，不必逐个改造。
        """
        raw = self.evaluate(audio_bytes, ref_text, voice_format)
        if raw is None:
            return None, self.fail_reason
        return raw, None

    @classmethod
    def failing_with(cls, reason: str):
        """可用但按指定原因失败替身（如 reason='ref_text_too_long' → 本句过长）。"""
        return cls(result=None, available=True, fail_reason=reason)
