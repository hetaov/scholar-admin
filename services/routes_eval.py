"""评估接口（4.6.5a：翻译 Tab 试点，评估管线整体迁移）

POST /eval/translate — 对一段文字/语音输入产出 { transcription, status }。
- 仅做「评估」，不落库：状态写入仍由前端走 reportTranslation → POST /tracking/state
- 双入参：user_input（文字直评）或 audio_base64（语音 ASR → 转写 → 评分）
- ASR 不可用（未配置凭据 / 识别失败）时返回 success=false + code=ASR_UNAVAILABLE，
  前端可回退云函数评估（灰度兜底，见 execution-guide 4.6.5c）

POST /eval/speech — SOE-N 句级口语评测（F2/2.3，契约 api-contract §3.4.2）。
- 仅评测不落 skill 状态（评审 R5：Shadowing 的 speaking 上报仍走 /tracking/state）
- 原始 JSON 落 speech_evaluation 集合（raw + 归一化 parsed），与 skill_state 解耦
- 错误契约对齐 /eval/translate：业务失败 200 + success=false + code
  （INVALID_INPUT / INVALID_AUDIO / SOE_UNAVAILABLE），仅技术异常走 5xx
"""
from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.asr import get_asr_service, ASRService
from services.dependencies import get_db
from services.database import CloudBaseNoSQLClient
from services.evaluator import evaluate
from services.speech_eval import (
    get_speech_provider,
    normalize_soe_result,
    SpeechProvider,
    SPEECH_EVALUATION_COLLECTION,
)

logger = logging.getLogger("scholar-admin.eval")

router = APIRouter(tags=["eval"])

# F1-3 定标：16k / 单声道 / mp3|wav / ≤60s。16k mp3 60s 上限约 5MB，
# 作为 P0 时长近似校验（精确时长解析依赖音频解码，留 F6 真机走查；SOE-N 服务端最终兜底超长）
MAX_AUDIO_BYTES = 5 * 1024 * 1024


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


class EvalSpeechRequest(BaseModel):
    """语音评测请求（SOE-N 句级，契约 api-contract §3.4.2）"""

    scholar_id: str = Field(..., min_length=1, max_length=64, description="学者 ID")
    sentence_id: str = Field(..., min_length=1, max_length=64, description="语句 ID")
    original_text: str = Field(
        ..., min_length=1, max_length=500, description="参考文本（句级 ≤30 词）"
    )
    audio_base64: str = Field(..., description="音频 base64（16k/单声道/mp3 或 wav，≤60s）")
    voice_format: Optional[str] = Field(
        "mp3", description="mp3（默认，SOE-N=2）或 wav（=1）"
    )


def _decode_audio_base64(raw_b64: str) -> Optional[bytes]:
    """解码音频 base64（兼容 data URL 前缀）；非法返回 None。"""
    s = raw_b64.strip()
    if "," in s and s.split(",", 1)[0].startswith("data:"):
        s = s.split(",", 1)[1]
    if not s:
        return None
    try:
        return base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError):
        return None


@router.post("/eval/speech", response_model=EvalTranslateResponse)
async def eval_speech(
    body: EvalSpeechRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
    provider: SpeechProvider = Depends(get_speech_provider),
) -> EvalTranslateResponse:
    # 1. 必填参数校验（契约：业务失败走 200 + success=false，无 4xx）
    for name in ("scholar_id", "sentence_id", "original_text", "audio_base64"):
        if not getattr(body, name).strip():
            return EvalTranslateResponse(
                success=False, code="INVALID_INPUT", message=f"{name} 不能为空"
            )
    if body.voice_format not in ("mp3", "wav"):
        return EvalTranslateResponse(
            success=False,
            code="INVALID_INPUT",
            message="voice_format 仅支持 mp3 / wav",
        )

    # 2. 音频校验（INVALID_AUDIO）
    audio_bytes = _decode_audio_base64(body.audio_base64)
    if audio_bytes is None:
        return EvalTranslateResponse(
            success=False, code="INVALID_AUDIO", message="audio_base64 不是合法 base64"
        )
    if not audio_bytes:
        return EvalTranslateResponse(
            success=False, code="INVALID_AUDIO", message="音频内容为空"
        )
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return EvalTranslateResponse(
            success=False,
            code="INVALID_AUDIO",
            message="音频过大（16k/60s 上限约 5MB），请分段评测",
        )

    # 3. SOE-N 评测（同步 WSS 阻塞，丢线程池避免卡事件循环）
    if not provider.available:
        return EvalTranslateResponse(
            success=False,
            code="SOE_UNAVAILABLE",
            message="SOE-N 凭据未配置（可回退旧 /eval/translate 链路）",
        )
    raw = await run_in_threadpool(
        provider.evaluate, audio_bytes, body.original_text, body.voice_format or "mp3"
    )
    if raw is None:
        return EvalTranslateResponse(
            success=False,
            code="SOE_UNAVAILABLE",
            message="语音评测失败（可回退旧 /eval/translate 链路）",
        )

    # 4. 归一化 + 落库 speech_evaluation（原始 JSON 存档；落库失败不阻塞返回，仅记日志）
    parsed = normalize_soe_result(raw)
    try:
        await db.insert(
            SPEECH_EVALUATION_COLLECTION,
            {
                "scholar_id": body.scholar_id,
                "sentence_id": body.sentence_id,
                "original_text": body.original_text,
                "audio_ref": None,  # P0 不存音频本体，留待对象存储迁移时回填（契约 §4.9）
                "provider": "soe_n",
                "raw": raw,
                "parsed": parsed,
                "created_at": int(time.time() * 1000),
            },
        )
    except Exception as e:  # noqa: BLE001 — 存档失败不影响评测结果返回
        logger.error("[eval] speech_evaluation 落库失败: %s", e)

    return EvalTranslateResponse(
        success=True,
        data=parsed,
    )
