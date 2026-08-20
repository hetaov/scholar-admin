"""腾讯云 ASR（一句话识别）服务

封装腾讯云官方 SDK（tencentcloud-sdk-python-asr）的 SentenceRecognition 接口，
复用 config 中 CloudRun 自动注入的 TENCENTCLOUD_SECRETID/SECRETKEY/SESSIONTOKEN。

与云函数 updateTrackingStatus 的 TC3 手写签名等价，但更简洁：
- 官方 SDK 内部处理 TC3 签名与请求重试
- 支持临时密钥（SESSION_TOKEN 透传）
"""
from __future__ import annotations

import base64
import logging
from typing import Optional

from config import (
    SESSION_TOKEN,
    SECRET_ID,
    SECRET_KEY,
)

logger = logging.getLogger("scholar-admin.asr")

# 一句话识别（英语 16k，与云函数参数保持一致）
ASR_SERVICE_TYPE = "16k_en"
ASR_REGION = "ap-guangzhou"
ASR_VERSION = "2019-06-14"
# 微信小程序录音默认格式（由前端 recorderManager.format 指定）
DEFAULT_VOICE_FORMAT = "mp3"


class ASRService:
    """腾讯云一句话语音识别客户端"""

    def __init__(
        self,
        secret_id: str = SECRET_ID,
        secret_key: str = SECRET_KEY,
        session_token: str = SESSION_TOKEN,
        region: str = ASR_REGION,
        engine_type: str = ASR_SERVICE_TYPE,
    ) -> None:
        self._secret_id = secret_id
        self._secret_key = secret_key
        self._session_token = session_token
        self._region = region
        self._engine_type = engine_type

    @property
    def available(self) -> bool:
        """是否有可用的腾讯云凭据（CloudRun 注入 / 本地 .env）"""
        return bool(self._secret_id and self._secret_key)

    def recognize(self, audio_bytes: bytes, voice_format: str = DEFAULT_VOICE_FORMAT) -> Optional[str]:
        """对一段音频做一句话识别，返回转写文本；失败返回 None。

        Args:
            audio_bytes: 音频原始字节（mp3/wav/m4a 等，由 voice_format 指定）
            voice_format: 音频格式，与腾讯云 ASR VoiceFormat 一致（mp3/wav/m4a/aac/pcm...）
        """
        if not self.available:
            logger.warning("[asr] 未配置腾讯云凭据，无法调用 ASR")
            return None
        if not audio_bytes:
            logger.warning("[asr] 音频内容为空")
            return None

        try:
            # 延迟导入，避免未安装 SDK 时拖垮整个服务
            from tencentcloud.common import credential
            from tencentcloud.common.profile.client_profile import ClientProfile
            from tencentcloud.common.profile.http_profile import HttpProfile
            from tencentcloud.asr.v20190614 import asr_client, models

            cred = credential.Credential(
                self._secret_id, self._secret_key, self._session_token or None
            )
            http_profile = HttpProfile()
            http_profile.endpoint = "asr.tencentcloudapi.com"
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile

            client = asr_client.AsrClient(cred, self._region, client_profile)

            req = models.SentenceRecognitionRequest()
            params = {
                "EngSerViceType": self._engine_type,
                "SourceType": 1,  # 1 = 音频数据 base64
                "VoiceFormat": voice_format,
                "Data": base64.b64encode(audio_bytes).decode("ascii"),
                "DataLen": len(audio_bytes),
            }
            req.from_json_string(__import__("json").dumps(params))

            resp = client.SentenceRecognition(req)
            text = (getattr(resp, "Result", None) or "").strip()
            logger.info(f"[asr] 识别结果: {text!r} (size={len(audio_bytes)})")
            return text or None
        except Exception as e:  # noqa: BLE001 — 任何 SDK/网络异常都降级为 None
            logger.error(f"[asr] ASR 调用异常: {e}")
            return None


# 模块级单例（与 volcano.py / dialogue.py 的 _get_client 风格一致）
_asr_service: ASRService | None = None


def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service
