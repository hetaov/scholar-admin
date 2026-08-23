"""英语教材语句管理路由（prefix=/english）

契约：api-contract.md §3.11 E-API-1~E-API-7（SOP §5 E1.1 + E1.2）：
1. GET    /english/textbook/stats                           教材统计概览（E-API-1）
2. GET    /english/textbook/{tid}/chapters                  章节课时树（E-API-2）
3. POST   /english/textbook/{tid}/validate-sentences        语句归属校验（E-API-3）
4. GET    /english/textbook/{tid}/lessons/{lid}/sentences   语句列表（E-API-4）
5. POST   /english/textbook/{tid}/lessons/{lid}/sentences   新增语句（批量，E-API-7）
6. PUT    /english/sentence/{sid}                           编辑语句（E-API-6）
7. DELETE /english/sentence/{sid}                           删除语句 + 级联清理（E-API-5）

鉴权：管理端（main.py 挂载在 _PAID_ROUTERS，require_paid_user 白名单）。
异常映射统一走 _english_error_to_http（service-contract §8.2）。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.auth import get_request_openid
from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.english import (
    ConfirmTextMismatchError,
    EnglishManagementError,
    LessonNotFoundError,
    SentenceNotFoundError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.english.sentence_management import (
    create_english_sentences,
    delete_english_sentence,
    edit_english_sentence,
    list_english_lesson_sentences,
)
from services.english.validation import (
    get_english_chapter_tree,
    list_english_textbook_stats,
    validate_english_sentences,
)

logger = logging.getLogger("scholar-admin.routes.english")

router = APIRouter(prefix="/english", tags=["english"])


# ===========================================================================
# 请求体 Pydantic 模型
# ===========================================================================


class EnglishSentenceItem(BaseModel):
    """POST 单条语句入参（E-API-7）。"""

    text: str = Field(..., min_length=1, description="语句原文（必填）")
    translation: str = Field("", description="译文")
    audio_url: str | None = Field(None, description="音频 URL")
    knowledge_point_ids: list[str] = Field(default_factory=list, description="关联知识点")


class EnglishCreateSentencesRequest(BaseModel):
    """POST 批量新增语句请求体。"""

    sentences: list[EnglishSentenceItem] = Field(..., min_length=1)


class EnglishEditSentenceRequest(BaseModel):
    """PUT 编辑语句请求体（全部字段可选，仅传需更新项）。"""

    text: str | None = None
    translation: str | None = None
    audio_url: str | None = None
    knowledge_point_ids: list[str] | None = None


class EnglishDeleteSentenceRequest(BaseModel):
    """DELETE 删除语句请求体（二次确认 + 可选开关）。"""

    confirm_text: str = Field(..., min_length=1, description="语句完整原文确认")
    delete_audio_asset: bool = Field(False, description="是否同时删除 audio_asset 缓存")
    delete_duplicates: bool = Field(False, description="是否同时删除同 text_hash 重复句")


class EnglishValidateSentencesRequest(BaseModel):
    """POST 语句归属校验请求体（E-API-3，全部可选）。"""

    scope: str = Field("full", description="校验范围：full / chapter / lesson")
    chapter_id: str | None = Field(None, description="scope=chapter 时必填")
    lesson_id: str | None = Field(None, description="scope=lesson 时必填")
    check_types: list[str] | None = Field(None, description="校验类型白名单（缺省全部）")


# ===========================================================================
# 异常映射（service-contract §8.2：404 三查无 / 400 两校验 / 其余 500）
# ===========================================================================


def _english_error_to_http(exc: Exception) -> HTTPException:
    """英语域业务异常 → HTTP 状态码（与 F1/F2/G 的 _xxx_error_to_http 模式一致）。"""
    if isinstance(exc, (TextbookNotFoundError, LessonNotFoundError, SentenceNotFoundError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (SentencePayloadError, ConfirmTextMismatchError)):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# 1. GET /english/textbook/stats — 教材统计概览（E-API-1）
# ===========================================================================


@router.get("/textbook/stats")
async def english_textbook_stats(
    grade: str | None = Query(None, description="年级过滤"),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 教材统计概览（E-API-1，纯读不审，validation_status 从缓存读取）。"""
    data = await list_english_textbook_stats(db, grade=grade)
    return {"success": True, "data": data}


# ===========================================================================
# 2. GET /english/textbook/{textbook_id}/chapters — 章节课时树（E-API-2）
# ===========================================================================


@router.get("/textbook/{textbook_id}/chapters")
async def english_textbook_chapters(
    textbook_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 章节课时树（E-API-2，管理端专用，无 scholar_id 依赖）。"""
    try:
        data = await get_english_chapter_tree(db, textbook_id=textbook_id)
        return {"success": True, "data": data}
    except TextbookNotFoundError as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 3. POST /english/textbook/{textbook_id}/validate-sentences — 语句校验（E-API-3）
# ===========================================================================


@router.post("/textbook/{textbook_id}/validate-sentences")
async def english_textbook_validate(
    textbook_id: str,
    body: EnglishValidateSentencesRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 语句归属校验（E-API-3，5 类异常 + summary + 缓存 TTL 1h）。"""
    try:
        data = await validate_english_sentences(
            db,
            textbook_id=textbook_id,
            scope=body.scope,
            chapter_id=body.chapter_id,
            lesson_id=body.lesson_id,
            check_types=body.check_types,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 4. GET /english/textbook/{tid}/lessons/{lid}/sentences — 语句列表
# ===========================================================================


@router.get("/textbook/{textbook_id}/lessons/{lesson_id}/sentences")
async def english_lesson_sentences_list(
    textbook_id: str,
    lesson_id: str,
    keyword: str | None = Query(None, description="text 模糊匹配"),
    duplicate_only: bool = Query(False, description="只返回重复句"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=200, description="每页条数"),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 管理端语句列表（E-API-4，含重复标记 + 关联数据计数）。"""
    try:
        data = await list_english_lesson_sentences(
            db,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            keyword=keyword,
            duplicate_only=duplicate_only,
            page=page,
            page_size=page_size,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 5. POST /english/textbook/{tid}/lessons/{lid}/sentences — 新增语句
# ===========================================================================


@router.post("/textbook/{textbook_id}/lessons/{lesson_id}/sentences")
async def english_lesson_sentences_create(
    textbook_id: str,
    lesson_id: str,
    body: EnglishCreateSentencesRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 批量新增语句（E-API-7，自动生成 sentence_id + text_hash，重复跳过）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await create_english_sentences(
            db,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            sentences=[item.model_dump() for item in body.sentences],
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 6. PUT /english/sentence/{sentence_id} — 编辑语句
# ===========================================================================


@router.put("/sentence/{sentence_id}")
async def english_sentence_edit(
    sentence_id: str,
    body: EnglishEditSentenceRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """PUT 编辑语句（E-API-6，字段白名单 + text 变更重算 text_hash）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await edit_english_sentence(
            db,
            sentence_id=sentence_id,
            text=body.text,
            translation=body.translation,
            audio_url=body.audio_url,
            knowledge_point_ids=body.knowledge_point_ids,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (SentenceNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 7. DELETE /english/sentence/{sentence_id} — 删除 + 级联清理
# ===========================================================================


@router.delete("/sentence/{sentence_id}")
async def english_sentence_delete(
    sentence_id: str,
    body: EnglishDeleteSentenceRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """DELETE 删除语句 + 级联清理 6 表（E-API-5，二次确认 confirm_text）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await delete_english_sentence(
            db,
            sentence_id=sentence_id,
            confirm_text=body.confirm_text,
            delete_audio_asset=body.delete_audio_asset,
            delete_duplicates=body.delete_duplicates,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (SentenceNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e
