"""TTS 语音合成接口（F3/3.3，契约 api-contract §3.5）

POST /audio/tts — 文本键控缓存：同一文本二次请求直接命中 `audio_asset`，不调用 TTS。
- 读路径：`text_hash` 精确命中 → `ref_count` +1（读改写 $set）→ 直接返回缓存
- 写路径：miss → TTS 合成 → 落库 `audio_asset`（含 `text_hash` 唯一键）→ `from_cache=false`
- 错误契约对齐 /eval/translate：业务失败 200 + success=false + code
  （INVALID_TEXT / TTS_UNAVAILABLE），仅技术异常（数据库错误等）走 5xx
- 落库失败不阻塞返回（对齐 /eval/speech：缓存尽力而为，下次请求重新合成）
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from services.dependencies import get_db
from services.database import CloudBaseNoSQLClient
from services.tts import (
    AUDIO_ASSET_COLLECTION,
    SEMANTIC_ENGINE_RESERVED,
    SEMANTIC_SEARCH_DEFAULT_TOP_K,
    SEMANTIC_SEARCH_MAX_TEXT_CHARS,
    SEMANTIC_SEARCH_MAX_TOP_K,
    TTS_CODEC,
    TTS_MAX_TEXT_CHARS,
    TTS_SAMPLE_RATE,
    TTS_VOICE_TYPE,
    TTSProvider,
    extract_audio_base64,
    get_tts_provider,
    hash_text,
    normalize_text,
    semantic_lookup,
)

logger = logging.getLogger("scholar-admin.tts")

router = APIRouter(tags=["tts"])


class TtsRequest(BaseModel):
    """语音合成请求（契约 api-contract §3.5）"""

    text: str = Field(..., description="待合成英文文本（≤200 字符）")
    sentence_id: Optional[str] = Field(
        None, max_length=64, description="关联语句 ID（透传落库，便于运营对账）"
    )


class SemanticSearchRequest(BaseModel):
    """语义检索请求（契约 api-contract §3.5 语义检索子节，F3/3.4 预留）"""

    text: str = Field(..., description="语义查询文本（≤500 字符）")
    top_k: int = Field(
        SEMANTIC_SEARCH_DEFAULT_TOP_K,
        description=f"期望返回条数（1-{SEMANTIC_SEARCH_MAX_TOP_K}，越界 → INVALID_INPUT）",
    )


class TtsResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


@router.post("/audio/tts", response_model=TtsResponse)
async def audio_tts(
    body: TtsRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
    provider: TTSProvider = Depends(get_tts_provider),
) -> TtsResponse:
    # 1. 文本校验（契约：INVALID_TEXT —— 缺失/空/超长，业务失败走 200 + success=false）
    text = normalize_text(body.text)
    if not text:
        return TtsResponse(success=False, code="INVALID_TEXT", message="text 不能为空")
    if len(text) > TTS_MAX_TEXT_CHARS:
        return TtsResponse(
            success=False,
            code="INVALID_TEXT",
            message=f"text 超长（>{TTS_MAX_TEXT_CHARS} 字符）",
        )
    text_hash = hash_text(text)

    # 2. 缓存读路径：text_hash 精确命中 → ref_count +1 → 直接返回（不调 TTS）
    cached = await db.query(AUDIO_ASSET_COLLECTION, {"text_hash": text_hash}, limit=1)
    if cached.get("records"):
        doc = cached["records"][0]
        audio = doc.get("audio_base64")
        if audio:
            # ref_count 命中 +1（P0 用读改写 $set：FakeDB 与真实 CloudBase 的 $inc 行为不一致，
            # 且缓存计数非关键路径，读多写少，并发竞态可接受）
            try:
                await db.update(
                    AUDIO_ASSET_COLLECTION,
                    {"text_hash": text_hash},
                    {"$set": {"ref_count": int(doc.get("ref_count") or 0) + 1}},
                )
            except Exception as e:  # noqa: BLE001 — 计数失败不影响缓存返回
                logger.warning("[tts] audio_asset ref_count 更新失败: %s", e)
            return TtsResponse(
                success=True,
                data={
                    "audio_base64": audio,
                    "codec": doc.get("codec", TTS_CODEC),
                    "sample_rate": doc.get("sample_rate", TTS_SAMPLE_RATE),
                    "from_cache": True,
                },
            )

    # 2.5 (预留) 语义召回复用（F3/3.4，契约 §3.5 语义检索子节）：精确 miss 时尝试语义召回——
    # 当前未接入 RAG，semantic_lookup 恒返回 None，直接走合成；接入后命中语义相近音频在此复用，
    # 避免重复合成（audio_asset 复用与 text_hash 精确命中走同一 from_cache 语义）
    _ = semantic_lookup(text, db, top_k=1)

    # 3. TTS 合成（TC3 REST 同步阻塞，丢线程池避免卡事件循环）
    if not provider.available:
        return TtsResponse(
            success=False,
            code="TTS_UNAVAILABLE",
            message="TTS 凭据未配置（前端可回退看文字模式）",
        )
    raw = await run_in_threadpool(provider.synthesize, text)
    audio = extract_audio_base64(raw)
    if audio is None:
        return TtsResponse(
            success=False,
            code="TTS_UNAVAILABLE",
            message="语音合成失败（前端可回退看文字模式）",
        )

    # 4. 落库 audio_asset（文本键控缓存；落库失败不阻塞返回，仅记日志）
    doc: dict = {
        "text_hash": text_hash,
        "text": text,
        "audio_base64": audio,
        "codec": TTS_CODEC,
        "sample_rate": TTS_SAMPLE_RATE,
        "voice": str(TTS_VOICE_TYPE),
        "tts_request_id": (raw or {}).get("RequestId"),
        "ref_count": 0,
        "created_at": int(time.time() * 1000),
    }
    if body.sentence_id:
        doc["sentence_id"] = body.sentence_id
    try:
        await db.insert(AUDIO_ASSET_COLLECTION, doc)
    except Exception as e:  # noqa: BLE001 — 缓存写入失败不影响本次合成结果返回
        logger.error("[tts] audio_asset 落库失败: %s", e)

    return TtsResponse(
        success=True,
        data={
            "audio_base64": audio,
            "codec": TTS_CODEC,
            "sample_rate": TTS_SAMPLE_RATE,
            "from_cache": False,
        },
    )


@router.post("/audio/assets/search", response_model=TtsResponse)
async def audio_assets_search(
    body: SemanticSearchRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> TtsResponse:
    """audio_asset 语义检索（预留接口，F3/3.4，契约 §3.5 语义检索子节）。

    当前 engine=reserved：仅对齐契约形态，恒返回空 hits（本步不实现召回）。
    接入 RAG 后 engine 切换为具体引擎并填充 hits（数据源即 semantic_lookup 语义召回结果）。
    """
    text = normalize_text(body.text)
    if not text or len(text) > SEMANTIC_SEARCH_MAX_TEXT_CHARS:
        return TtsResponse(
            success=False,
            code="INVALID_INPUT",
            message=f"text 缺失/空或超长（>{SEMANTIC_SEARCH_MAX_TEXT_CHARS} 字符）",
        )
    if not 1 <= body.top_k <= SEMANTIC_SEARCH_MAX_TOP_K:
        return TtsResponse(
            success=False,
            code="INVALID_INPUT",
            message=f"top_k 越界（1-{SEMANTIC_SEARCH_MAX_TOP_K}）",
        )
    # 预留：语义召回（当前 engine=reserved，hits 恒空）
    hits = semantic_lookup(text, db, top_k=body.top_k)
    return TtsResponse(
        success=True,
        data={
            "engine": SEMANTIC_ENGINE_RESERVED,
            "hits": hits or [],
        },
    )
