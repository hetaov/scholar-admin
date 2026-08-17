"""SOE-N 句级口语评测 Provider（F2/2.2）

SpeechProvider 抽象接口 + TencentSoeNProvider 实现（WSS 录音模式 rec_mode=1）。

- 调用面复用 F1-2 已验证的 scripts/soe_n_verify.py：官方 tencentcloud-speech-sdk-python
  为纯源码分发（PyPI 无包，2026-08-17 实测），vendor 目录随镜像发布，运行时 sys.path 引用仓库根目录
- 注入模式对齐 services/asr.py 的 get_asr_service（模块级单例）
- evaluate 返回 SOE-N 完整原始 JSON（供路由层落 speech_evaluation.raw）；
  normalize_soe_result 按 api-contract §3.4.2 给出 parsed 归一口径
"""
from __future__ import annotations

import logging
import sys
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import (
    TCB_APPID,
    SESSION_TOKEN,
    SECRET_ID,
    SECRET_KEY,
)

logger = logging.getLogger("scholar-admin.speech_eval")

# 官方 SDK 源码分发目录（随镜像发布，见 scripts/soe_n_verify.py 安装说明）
SDK_DIR = (
    Path(__file__).resolve().parent.parent / "vendor" / "tencentcloud-speech-sdk-python"
)

# SOE-N 评测原始结果存档集合（data-model-contract §4.9；需 scripts/init_speech_evaluation_collection.py 幂等建表）
SPEECH_EVALUATION_COLLECTION = "speech_evaluation"

# SOE-N 语音评测参数（与 F1-2 实测一致）
SOE_ENGINE = "16k_en"          # 英语 16k
SOE_EVAL_MODE = 1              # 1 = 句子
SOE_REC_MODE = 1               # 1 = 录音模式（一次性上传完整音频，≤60s）
SOE_TEXT_MODE = 0              # 0 = 普通文本
SOE_MAX_REF_WORDS = 30         # 句级 ref_text ≤30 词（F1-2 定标）
EVALUATE_TIMEOUT = 65.0        # 录音 ≤60s + 缓冲，超出视为评测失败

# 契约 voice_format 字符串 → SOE-N int（0=pcm / 1=wav / 2=mp3 / 4=speex）
VOICE_FORMAT_MAP = {"mp3": 2, "wav": 1}
DEFAULT_VOICE_FORMAT = "mp3"

# SDK 模块惰性加载缓存
_sdk_credential = None
_sdk_speaking_assessment = None


def _load_sdk() -> bool:
    """加载官方 SOE-N SDK（源码分发，vendor 目录加入 sys.path）。成功返回 True。"""
    global _sdk_credential, _sdk_speaking_assessment
    if _sdk_credential is not None:
        return True
    if SDK_DIR.is_dir() and str(SDK_DIR) not in sys.path:
        sys.path.insert(0, str(SDK_DIR))
    try:
        from common import credential as _c
        from soe import speaking_assessment as _sa

        _sdk_credential = _c
        _sdk_speaking_assessment = _sa
        return True
    except ImportError:
        logger.error(
            "[speech_eval] 缺少官方 SDK 源码，请先 git clone 到 vendor/tencentcloud-speech-sdk-python"
        )
        return False


class SpeechProvider(ABC):
    """语音评测 Provider 抽象接口（供路由层依赖注入）"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """是否具备可用凭据"""

    @abstractmethod
    def evaluate(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> Optional[dict]:
        """对一段音频做句级口语评测，返回 SOE-N 完整原始 JSON；失败返回 None。"""


class _SoeNListener:
    """评测结果回调：最终结果在 on_recognition_complete 中返回（对齐 F1-2 SoeNListener）"""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Optional[dict] = None
        self.fail_reason: Optional[str] = None

    def on_recognition_start(self, response) -> None:
        logger.info("[speech_eval] 评测开始 voice_id=%s", response.get("voice_id"))

    def on_intermediate_result(self, response) -> None:
        pass  # 录音模式（rec_mode=1）不关注中间结果

    def on_recognition_complete(self, response) -> None:
        self.result = response
        self.event.set()

    def on_fail(self, response) -> None:
        self.fail_reason = str(response)
        self.event.set()


class TencentSoeNProvider(SpeechProvider):
    """SOE-N 口语评测客户端（WSS + HMAC-SHA1 签名，官方 SDK 内部处理）"""

    def __init__(
        self,
        appid: str = TCB_APPID,
        secret_id: str = SECRET_ID,
        secret_key: str = SECRET_KEY,
        session_token: str = SESSION_TOKEN,
    ) -> None:
        self._appid = appid
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._session_token = session_token

    @property
    def available(self) -> bool:
        """需要 AppID + SecretId + SecretKey（CloudRun 注入 / 本地 .env）"""
        return bool(self._appid and self._secret_id and self._secret_key)

    def evaluate(
        self, audio_bytes: bytes, ref_text: str, voice_format: str = DEFAULT_VOICE_FORMAT
    ) -> Optional[dict]:
        """录音模式（rec_mode=1）一次性上传完整音频做句级评测，返回原始 JSON。"""
        if not self.available:
            logger.warning(
                "[speech_eval] 未配置 TCB_APPID / TENCENTCLOUD_SECRETID/SECRETKEY，无法调用 SOE-N"
            )
            return None
        if not audio_bytes:
            logger.warning("[speech_eval] 音频内容为空")
            return None
        if not ref_text or len(ref_text.split()) > SOE_MAX_REF_WORDS:
            logger.warning(
                "[speech_eval] ref_text 为空或超 %d 词（句级评测上限）", SOE_MAX_REF_WORDS
            )
            return None
        if not _load_sdk():
            return None

        voice_format_int = VOICE_FORMAT_MAP.get(voice_format)
        if voice_format_int is None:
            logger.warning(
                "[speech_eval] 未知 voice_format=%r，回退 mp3", voice_format
            )
            voice_format_int = VOICE_FORMAT_MAP[DEFAULT_VOICE_FORMAT]

        try:
            cred = _sdk_credential.Credential(
                self._secret_id, self._secret_key, self._session_token or None
            )
            listener = _SoeNListener()
            recognizer = _sdk_speaking_assessment.SpeakingAssessment(
                self._appid, cred, SOE_ENGINE, listener
            )
            recognizer.set_eval_mode(SOE_EVAL_MODE)        # 1 = 句子
            recognizer.set_rec_mode(SOE_REC_MODE)          # 1 = 录音模式
            recognizer.set_ref_text(ref_text)              # 句级 ≤30 词
            recognizer.set_text_mode(SOE_TEXT_MODE)        # 0 = 普通文本
            recognizer.set_voice_format(voice_format_int)  # 2=mp3 / 1=wav

            recognizer.start()
            try:
                recognizer.write(audio_bytes)
            finally:
                recognizer.stop()

            if listener.event.wait(EVALUATE_TIMEOUT):
                if listener.result is not None:
                    logger.info(
                        "[speech_eval] 评测成功 size=%dB ref_text=%r",
                        len(audio_bytes), ref_text[:40],
                    )
                    return listener.result
                logger.warning("[speech_eval] 评测失败: %s", listener.fail_reason)
                return None
            logger.error("[speech_eval] 评测超时（%ss）", EVALUATE_TIMEOUT)
            return None
        except Exception as e:  # noqa: BLE001 — 任何 SDK/网络异常都降级为 None（路由层回退旧链路）
            logger.error("[speech_eval] SOE-N 调用异常: %s", e)
            return None


def _pick(raw: dict, key: str):
    """兼容 SOE-N 响应两种形态：顶层扁平字段 或 result 子对象内嵌。"""
    if key in raw:
        return raw[key]
    result = raw.get("result")
    if isinstance(result, dict) and key in result:
        return result[key]
    return None


def normalize_soe_result(raw: dict) -> dict:
    """把 SOE-N 原始 JSON 归一为 api-contract §3.4.2 / data-model §4.9 parsed 口径。

    - accuracy / completion / suggested_score：0~100（SOE-N 原值）
    - fluency：0~100（= SOE-N PronFluency × 100 归一，官方原值为 0~1）
    - words[].match_tag：0=命中 / 2=未命中（SOE-N 原始语义，前端词级高亮）
    """
    pron_accuracy = _pick(raw, "PronAccuracy")
    pron_fluency = _pick(raw, "PronFluency")
    pron_completion = _pick(raw, "PronCompletion")
    suggested_score = _pick(raw, "SuggestedScore")

    fluency = None
    if isinstance(pron_fluency, (int, float)):
        fluency = float(pron_fluency) * 100.0

    words = _pick(raw, "Words")
    normalized_words = []
    if isinstance(words, list):
        for w in words:
            if not isinstance(w, dict):
                continue
            word = w.get("Word")
            if not word:
                continue
            normalized_words.append(
                {"word": word, "match_tag": int(w.get("MatchTag", 0) or 0)}
            )

    return {
        "accuracy": float(pron_accuracy or 0.0),
        "fluency": fluency if fluency is not None else 0.0,
        "completion": float(pron_completion or 0.0),
        "suggested_score": float(suggested_score or 0.0),
        "words": normalized_words,
    }


# 模块级单例（与 asr.py get_asr_service 注入风格一致）
_speech_provider: TencentSoeNProvider | None = None


def get_speech_provider() -> SpeechProvider:
    global _speech_provider
    if _speech_provider is None:
        _speech_provider = TencentSoeNProvider()
    return _speech_provider
