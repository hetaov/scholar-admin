"""评估接口（4.6.5a：翻译 Tab 试点，评估管线整体迁移）

POST /eval/translate — 对一段文字/语音输入产出 { transcription, status }。
- 仅做「评估」，不落库：状态写入仍由前端走 reportTranslation → POST /tracking/state
- 双入参：user_input（文字直评）或 audio_base64（语音 ASR → 转写 → 评分）
- ASR 不可用（未配置凭据 / 识别失败）时返回 success=false + code=ASR_UNAVAILABLE，
  前端可回退云函数评估（灰度兜底，见 execution-guide 4.6.5c）

POST /eval/translate/v2 — 翻译评估异步提交（2026-08-29，ADR-0022 / docs_v1《翻译功能优化.md》）。
- 提交毫秒级返回 { task_id, status: pending }，不做任何 LLM/ASR 调用；
- 评分后台执行（run_translation_task），失败即 failed，不降级（不回退 levenshtein/混元/云函数）；
- 终态无条件写 evaluation 证据（成功/失败均写，失败全量留痕）；
- 参数与 v1 一致（original_text + user_input/audio_base64 二选一），
  可选 scholar_id/sentence_id 仅落库关联、不参与评分。

GET /eval/translate/v2/task/{task_id} — 查询异步翻译评估结果。
- 状态枚举 pending/processing/success/failed；TTL 24h 过期 → 404；
- 卡死自愈：processing 超时 → 查询定点置 failed；
  全集合巡检（recover_stale_tasks + cleanup_expired）由后台定时任务执行（services/background_tasks.py）。

POST /eval/speech — SOE-N 句级口语评测（F2/2.3，契约 api-contract §3.4.2）。
- 仅评测不落 skill 状态（评审 R5：Shadowing 的 speaking 上报仍走 /tracking/state）
- 原始 JSON 落 speech_evaluation 集合（raw + 归一化 parsed），与 skill_state 解耦
- 错误契约对齐 /eval/translate：业务失败 200 + success=false + code
  （INVALID_INPUT / INVALID_AUDIO / SOE_UNAVAILABLE），仅技术异常走 5xx
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from config import SECRET_ID, SECRET_KEY, SESSION_TOKEN, TCB_APPID
from services.asr import get_asr_service, ASRService
from services.dependencies import get_db
from services.database import CloudBaseNoSQLClient
from services.evaluator import evaluate
from services.learning.translation_task import (
    STATUS_FAILED,
    create_translation_task,
    get_task,
    recover_task_if_stale,
    run_translation_task,
)
from services.providers.translation_eval import infer_translation_mode
from services.speech_eval import (
    get_speech_provider,
    normalize_soe_result,
    SpeechProvider,
    SPEECH_EVALUATION_COLLECTION,
)

logger = logging.getLogger("scholar-admin.eval")

router = APIRouter(tags=["eval"])

# v2 后台任务强引用集合：防止 create_task 的协程被 GC 回收导致任务中途取消（同 dialogue.py）
_background_tasks: set[asyncio.Task] = set()
# 全集合 TTL/卡死巡检（recover_stale_tasks + cleanup_expired）已移出提交热路径，
# 由后台定时任务执行（services/background_tasks.py，lifespan 启动，每 60s）。

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


# ---------------------------------------------------------------------------
# 翻译评估 v2（2026-08-29，ADR-0022 / docs_v1《翻译功能优化.md》/ api-contract §3.4）
# ---------------------------------------------------------------------------


class EvalTranslateV2Request(BaseModel):
    """翻译评估异步提交请求（参数与 v1 一致 + 可选落库关联字段）。

    original_text 设为可选并手动校验 → 缺失/为空统一返回 INVALID_INPUT
    （契约 §3.4：业务失败 200 + success=false，无 4xx）。
    """

    original_text: Optional[str] = Field(
        None, max_length=2000, description="标准句/参考答案"
    )
    user_input: Optional[str] = Field(
        None, max_length=2000, description="用户文字输入（文字直评路径）"
    )
    audio_base64: Optional[str] = Field(
        None, description="音频 base64（语音路径，ASR 转写后评分）"
    )
    voice_format: Optional[str] = Field(
        "mp3", description="音频格式（mp3/wav/m4a/aac/pcm），默认 mp3"
    )
    scholar_id: Optional[str] = Field(
        None, max_length=64, description="学者 ID（仅落库关联，不参与评分）"
    )
    sentence_id: Optional[str] = Field(
        None, max_length=64, description="语句 ID（仅落库关联，不参与评分）"
    )


@router.post("/eval/translate/v2", response_model=EvalTranslateResponse)
async def eval_translate_v2(
    body: EvalTranslateV2Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> EvalTranslateResponse:
    """提交异步翻译评估 — 毫秒级返回 task_id，评分在后台执行（不做任何 LLM/ASR 调用）。

    返回：{ success: true, data: { task_id: "tr_xxx", status: "pending" } }
    """
    # 1. 参数校验（业务失败 200 + success=false，对齐契约 §3.4）
    if not body.original_text or not body.original_text.strip():
        return EvalTranslateResponse(
            success=False, code="INVALID_INPUT", message="original_text 不能为空"
        )
    has_text = body.user_input is not None
    has_audio = body.audio_base64 is not None
    if not has_text and not has_audio:
        return EvalTranslateResponse(
            success=False,
            code="INVALID_INPUT",
            message="user_input 与 audio_base64 至少提供其一",
        )

    input_mode = "voice" if has_audio else "text"
    user_input = body.user_input
    if has_audio:
        # 音频同步校验（不调用 ASR，仅 base64/大小校验）
        if not body.audio_base64.strip():
            return EvalTranslateResponse(
                success=False, code="INVALID_AUDIO", message="音频内容为空"
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
        if len(audio_bytes) > MAX_AUDIO_BYTES:
            return EvalTranslateResponse(
                success=False,
                code="INVALID_AUDIO",
                message="音频过大（16k/60s 上限约 5MB），请分段评测",
            )

    # 2. 创建 pending 任务（落库，毫秒级返回）
    try:
        # mode 推导：契约无 mode 入参，按原句语言推导（含中文 → ce，否则 ec）
        mode = infer_translation_mode(body.original_text)
        task = await create_translation_task(
            db,
            original_text=body.original_text,
            input_mode=input_mode,
            mode=mode,
            scholar_id=body.scholar_id,
            sentence_id=body.sentence_id,
            user_input=user_input,
            audio_base64=body.audio_base64 if has_audio else None,
            voice_format=body.voice_format or "mp3",
        )
    except Exception as e:  # noqa: BLE001
        logger.error(f"[eval] v2 任务创建异常: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务创建失败: {str(e)}")

    # 3. 后台执行，不阻塞当前请求（毫秒级返回）。
    #    必须持有 task 引用并注册 done_callback，否则协程可能被 GC 回收而取消（同 dialogue.py）。
    bg = asyncio.create_task(
        run_translation_task(
            task["task_id"],
            original_text=body.original_text,
            mode=mode,
            input_mode=input_mode,
            user_input=user_input,
            audio_base64=body.audio_base64 if has_audio else None,
            voice_format=body.voice_format or "mp3",
            scholar_id=body.scholar_id,
            sentence_id=body.sentence_id,
        )
    )
    _background_tasks.add(bg)
    bg.add_done_callback(_background_tasks.discard)

    # 4. 全集合巡检（恢复卡死 + 清理过期）由后台定时任务执行（services/background_tasks.py），
    #    提交热路径不承担任何全表扫描，保证毫秒级返回稳定。
    logger.info(f"[eval] v2 任务已提交 → task_id={task['task_id']}, mode={mode}, input_mode={input_mode}")
    return EvalTranslateResponse(
        success=True,
        data={
            "task_id": task["task_id"],
            "status": task["status"],
        },
    )


@router.get("/eval/translate/v2/task/{task_id}", response_model=EvalTranslateResponse)
async def get_eval_translate_v2_task(
    task_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> EvalTranslateResponse:
    """查询异步翻译评估结果 — pending/processing/success/failed。

    返回：{ success: true, data: { task_id, status, result, error } }
    - result：{ transcription, status(0-5), feedback, confidence }（success 时）
    - error：可读失败原因字符串（failed 时）；任务不存在/过期 → 404
    """
    try:
        task = await get_task(db, task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        # 定点自愈：被查询任务若卡死（processing 且超时）→ 置 failed（error=LLM_TIMEOUT）
        if await recover_task_if_stale(db, task):
            task = {
                **task,
                "status": STATUS_FAILED,
                "error": {
                    "error_code": "LLM_TIMEOUT",
                    "error_detail": "执行超时",
                    "failure_stage": "llm",
                    "llm_timeout_seconds": None,
                    "raw": None,
                },
            }
        # TTL 过滤：expires_at 已过期的按不存在处理（物理清理由提交接口概率巡检执行）
        now_ms = int(time.time() * 1000)
        if task.get("expires_at", 0) <= now_ms:
            raise HTTPException(status_code=404, detail="任务已过期")

        # error 展示为可读字符串（完整 error 对象保留在任务文档与 evaluation 证据中）
        raw_error = task.get("error")
        error_display = (
            raw_error.get("error_detail") if isinstance(raw_error, dict) else raw_error
        )
        return EvalTranslateResponse(
            success=True,
            data={
                "task_id": task["task_id"],
                "status": task["status"],
                "result": task.get("result"),
                "error": error_display,
            },
        )
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error(f"[eval] v2 任务查询异常: task_id={task_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"任务查询失败: {str(e)}")


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
        # 详细诊断日志：区分缺失项（不打印密钥值，仅打印存在性）
        logger.warning(
            "[eval] /eval/speech SOE_UNAVAILABLE 诊断：TCB_APPID=%s, SECRET_ID=%s, "
            "SECRET_KEY=%s, SESSION_TOKEN=%s；"
            "来源=CloudRun 自动注入（TENCENTCLOUD_*）或控制台环境变量，改后需重启服务生效",
            "已配置" if TCB_APPID else "缺失",
            "已配置" if SECRET_ID else "缺失",
            "已配置" if SECRET_KEY else "缺失",
            "已配置" if SESSION_TOKEN else "缺失",
        )
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
