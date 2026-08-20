"""TTS 语音合成 Provider（F3/3.2）

TTSProvider 抽象接口 + TencentTtsProvider 实现（腾讯云 TextToVoice，TC3 REST）。

- 与 speech_eval.py 的 SOE-N（WSS + 源码分发 SDK）不同：TTS 走官方 PyPI 包
  tencentcloud-sdk-python-tts（标准 TencentCloud SDK 全家桶成员，TC3 HTTP REST）
- 参数对齐契约 api-contract §3.5 / data-model-contract §4.10：
  PrimaryLanguage=2（英文）、Codec=mp3、SampleRate=16000、ModelType=1（普通音色）
- synthesize 返回 TTS 完整原始响应（Audio base64 / RequestId / SessionId），
  供路由层提取音频并落 audio_asset（文本键控缓存）
- 文本键控工具：normalize_text（trim + 合并连续空白）+ hash_text（sha256），
  即 audio_asset.text_hash 唯一键来源（唯一索引命中 = 缓存命中，见步骤 3.3）
- 注入模式对齐 services/asr.py 的 get_asr_service / speech_eval.py 的 get_speech_provider
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Optional

from config import (
    REGION,
    SECRET_ID,
    SECRET_KEY,
    SESSION_TOKEN,
)

logger = logging.getLogger("scholar-admin.tts")

# 音频资产集合（data-model-contract §4.10；需幂等建表脚本，见步骤 3.3）
AUDIO_ASSET_COLLECTION = "audio_asset"

# TTS 合成参数（契约 api-contract §3.5 / data-model §4.10）
TTS_MODEL_TYPE = 1          # 1 = 普通音色（标准版）
TTS_VOICE_TYPE = 101001     # 英文默认音色（PrimaryLanguage=2 时生效，跑通后按效果更换）
TTS_CODEC = "mp3"
TTS_SAMPLE_RATE = 16000
TTS_PRIMARY_LANGUAGE = 2    # 2 = 英文
TTS_MAX_TEXT_CHARS = 200    # 契约 §3.5 上限（路由层校验 → INVALID_TEXT）
TTS_MAX_TEXT_BYTES = 150    # 腾讯 TextToVoice Text 硬限制（UTF-8 字节，Provider 层防线）

# 语义召回复用预留（F3/3.4，契约 api-contract §3.5 语义检索子节）
SEMANTIC_ENGINE_RESERVED = "reserved"       # 当前预留引擎标识（未接入 RAG，恒返回空 hits）
SEMANTIC_SEARCH_MAX_TEXT_CHARS = 500        # 语义检索入参 text 上限（>500 → INVALID_INPUT）
SEMANTIC_SEARCH_DEFAULT_TOP_K = 5
SEMANTIC_SEARCH_MAX_TOP_K = 20

# SDK 模块惰性加载缓存
_sdk_credential = None
_sdk_models = None
_sdk_tts_client = None


def _load_sdk() -> bool:
    """加载官方 TTS SDK（PyPI 包 tencentcloud-sdk-python-tts）。成功返回 True。"""
    global _sdk_credential, _sdk_models, _sdk_tts_client
    if _sdk_tts_client is not None:
        return True
    try:
        from tencentcloud.common import credential as _c
        from tencentcloud.tts.v20190823 import models as _m
        from tencentcloud.tts.v20190823 import tts_client as _tc

        _sdk_credential = _c
        _sdk_models = _m
        _sdk_tts_client = _tc
        return True
    except ImportError:
        logger.error("[tts] 缺少官方 SDK，请先执行: pip install tencentcloud-sdk-python-tts")
        return False


class TTSProvider(ABC):
    """语音合成 Provider 抽象接口（供路由层依赖注入）"""

    @property
    @abstractmethod
    def available(self) -> bool:
        """是否具备可用凭据"""

    @abstractmethod
    def synthesize(self, text: str) -> Optional[dict]:
        """合成一段英文语音，返回 TTS 完整原始响应（含 Audio base64）；失败返回 None。"""


class TencentTtsProvider(TTSProvider):
    """腾讯云语音合成客户端（TextToVoice，TC3 REST 同步调用，路由层丢线程池）"""

    def __init__(
        self,
        secret_id: str = SECRET_ID,
        secret_key: str = SECRET_KEY,
        session_token: str = SESSION_TOKEN,
        region: str = REGION,
    ) -> None:
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._session_token = session_token
        self._region = region

    @property
    def available(self) -> bool:
        """TTS 走 TC3 REST，仅需 SecretId + SecretKey（SOE-N 才额外要 AppID）"""
        return bool(self._secret_id and self._secret_key)

    def synthesize(self, text: str) -> Optional[dict]:
        """合成一段英文语音（mp3/16k），返回原始响应 {Audio, RequestId, SessionId}。"""
        if not self.available:
            logger.warning("[tts] 未配置 TENCENTCLOUD_SECRETID/SECRETKEY，无法调用 TTS")
            return None
        normalized = normalize_text(text)
        if not normalized:
            logger.warning("[tts] 合成文本为空")
            return None
        if len(normalized) > TTS_MAX_TEXT_CHARS:
            logger.warning("[tts] 文本超契约上限 %d 字符", TTS_MAX_TEXT_CHARS)
            return None
        if len(normalized.encode("utf-8")) > TTS_MAX_TEXT_BYTES:
            logger.warning(
                "[tts] 文本超腾讯 %d 字节硬限制，拒绝合成（utf-8 bytes=%d）",
                TTS_MAX_TEXT_BYTES, len(normalized.encode("utf-8")),
            )
            return None
        if not _load_sdk():
            return None

        try:
            cred = _sdk_credential.Credential(
                self._secret_id, self._secret_key, self._session_token or None
            )
            client = _sdk_tts_client.TtsClient(cred, self._region)

            req = _sdk_models.TextToVoiceRequest()
            req.Text = normalized
            req.SessionId = uuid.uuid4().hex
            req.ModelType = TTS_MODEL_TYPE
            req.VoiceType = TTS_VOICE_TYPE
            req.Codec = TTS_CODEC
            req.SampleRate = TTS_SAMPLE_RATE
            req.PrimaryLanguage = TTS_PRIMARY_LANGUAGE

            resp = client.TextToVoice(req)
            raw = json.loads(resp.to_json_string())
            audio = extract_audio_base64(raw)
            if audio is None:
                logger.error("[tts] 响应缺少 Audio 字段: %s", str(raw)[:200])
                return None
            logger.info(
                "[tts] 合成成功 text=%r audio_len=%dB", normalized[:40], len(audio),
            )
            return raw
        except Exception as e:  # noqa: BLE001 — 任何 SDK/网络异常都降级为 None（路由层回退"看文字"）
            logger.error("[tts] TextToVoice 调用异常: %s", e)
            return None


def normalize_text(text: str) -> str:
    """契约 §3.5 文本归一：trim + 合并连续空白（text_hash 唯一键的前置）。"""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip())


def hash_text(text: str) -> str:
    """audio_asset.text_hash 唯一键：sha256(normalize_text(text))。"""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def extract_audio_base64(raw: Optional[dict]) -> Optional[str]:
    """从 TTS 原始响应提取音频 base64；缺失/为空返回 None（路由层 → TTS_UNAVAILABLE）。"""
    if not isinstance(raw, dict):
        return None
    audio = raw.get("Audio")
    return audio if isinstance(audio, str) and audio else None


def semantic_lookup(
    text: str,
    db=None,  # noqa: ANN001 — 预留参数：后续向量查询需访问数据库，保持签名稳定
    top_k: int = SEMANTIC_SEARCH_DEFAULT_TOP_K,
) -> Optional[dict]:
    """（预留）语义召回复用 — audio_asset 语义检索占位（F3/3.4，契约 §3.5 子节）。

    当前为预留位：恒返回 None（未接入 RAG，见提案非目标）。接入后实现为
    embedding 向量检索 → 返回语义相近的 audio_asset 文档列表，供：
    - POST /audio/tts 缓存 miss 时复用语义相近音频（避免重复合成）
    - POST /audio/assets/search 填充 hits（engine 切换为具体引擎）
    """
    logger.debug("[tts] semantic_lookup 为预留占位（engine=%s），未接入 RAG", SEMANTIC_ENGINE_RESERVED)
    return None


# 模块级单例（与 asr.py / speech_eval.py 注入风格一致）
_tts_provider: TencentTtsProvider | None = None


def get_tts_provider() -> TTSProvider:
    global _tts_provider
    if _tts_provider is None:
        _tts_provider = TencentTtsProvider()
    return _tts_provider
