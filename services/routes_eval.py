"""评估接口（4.6.5a：翻译 Tab 试点，评估管线整体迁移）

POST /eval/translate — 对一段文字/语音输入产出 { transcription, status }。
- 仅做「评估」，不落库：状态写入仍由前端走 reportTranslation → POST /tracking/state
- 双入参：user_input（文字直评）或 audio_base64（语音 ASR → 转写 → 评分）
- ASR 不可用（未配置凭据 / 识别失败）时返回 success=false + code=ASR_UNAVAILABLE，
  前端可回退云函数评估（灰度兜底，见 execution-guide 4.6.5c）
"""
from __future__ import annotations

import base64
import binascii
import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from services.asr import get_asr_service, ASRService
from services.evaluator import evaluate

logger = logging.getLogger("scholar-admin.eval")

router = APIRouter(tags=["eval"])


class EvalTranslateRequest(BaseModel):
    """评估请求：文字与语音二选一（至少提供其一）"""

    original_text: str = Field(..., min_length=1, max_length=2000, description="标准句/参考答案")
    user_input: Optional[str] = Field(
        None, max_length=2000, description="用户文字输入（文字直评路径）"
    )
    audio_base64: Optional[str] = Field(
        None, description="音频 base64（语音路径，ASR 转写后评分）"
    )
    voice_format: Optional[str] = Field(
        "mp3", description="音频格式（mp3/wav/m4a/aac/pcm），默认 mp3"
    )


class EvalTranslateResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


@router.post("/eval/translate", response_model=EvalTranslateResponse)
async def eval_translate(
    body: EvalTranslateRequest,
    asr: ASRService = Depends(get_asr_service),
) -> EvalTranslateResponse:
    # 区分「未提供」与「提供为空」：None = 未提供；"" = 提供了空内容
    has_text = body.user_input is not None
    has_audio = body.audio_base64 is not None
    if not has_text and not has_audio:
        return EvalTranslateResponse(
            success=False,
            code="INVALID_INPUT",
            message="user_input 与 audio_base64 至少提供其一",
        )

    transcription = body.user_input or ""
    if has_audio:
        # 语音路径：ASR 转写
        if not body.audio_base64.strip():
            return EvalTranslateResponse(
                success=False, code="INVALID_AUDIO", message="音频内容为空"
            )
        if not asr.available:
            return EvalTranslateResponse(
                success=False,
                code="ASR_UNAVAILABLE",
                message="ASR 服务未配置凭据（可回退云函数评估）",
            )
        try:
            audio_bytes = base64.b64decode(body.audio_base64, validate=True)
        except (binascii.Error, ValueError):
            return EvalTranslateResponse(
                success=False, code="INVALID_AUDIO", message="audio_base64 不是合法 base64"
            )
        if not audio_bytes:
            return EvalTranslateResponse(
                success=False, code="INVALID_AUDIO", message="音频内容为空"
            )
        result = asr.recognize(audio_bytes, voice_format=body.voice_format or "mp3")
        if not result:
            return EvalTranslateResponse(
                success=False,
                code="ASR_UNAVAILABLE",
                message="语音识别失败（可回退云函数评估）",
            )
        transcription = result

    # 评分（模型优先，失败回退 levenshtein 兜底；空文字输入 → 0）
    status, _model_output = evaluate(body.original_text, transcription)

    return EvalTranslateResponse(
        success=True,
        data={
            "transcription": transcription,
            "status": status,
        },
    )
