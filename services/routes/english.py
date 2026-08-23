"""英语教材语句管理路由（prefix=/english）

契约：api-contract.md §3.11 E-API-1~E-API-12（SOP §5 E1.1 + E1.2 + M3 G1.3 + E-API-12 去重）：
1. GET    /english/textbook/stats                           教材统计概览（E-API-1）
2. GET    /english/textbook/{tid}/chapters                  章节课时树（E-API-2）
3. POST   /english/textbook/{tid}/validate-sentences        语句归属校验（E-API-3）
4. GET    /english/textbook/{tid}/lessons/{lid}/sentences   语句列表（E-API-4）
5. POST   /english/textbook/{tid}/lessons/{lid}/sentences   新增语句（批量，E-API-7）
6. PUT    /english/sentence/{sid}                           编辑语句（E-API-6）
7. DELETE /english/sentence/{sid}                           删除语句 + 级联清理（E-API-5）
8. GET    /english/textbook/{tid}/lessons/{lid}/groups      分组列表（E-API-8，M3 G1.3）
9. POST   /english/textbook/{tid}/lessons/{lid}/groups      新建分组（E-API-9，M3 G1.3）
10. PUT   /english/group/{gid}                              编辑分组（E-API-10，M3 G1.3）
11. DELETE /english/group/{gid}                             删除分组（E-API-11，M3 G1.3）
12. POST  /english/textbook/{tid}/deduplicate               批量去重（E-API-12，dry_run 预览 / 执行清理）

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
    GroupNotFoundError,
    LessonNotFoundError,
    SentenceNotFoundError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.english.sentence_group import (
    createSentenceGroup,
    deleteSentenceGroup,
    editSentenceGroup,
    listSentenceGroups,
)
from services.english.deduplication import deduplicateEnglishSentences
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


class EnglishCreateGroupRequest(BaseModel):
    """POST 新建分组请求体（E-API-9）。"""

    title: str = Field(..., min_length=1, description="组标题（必填）")
    type: str = Field(..., description="组类型枚举（VALID_SENTENCE_GROUP_TYPES）")
    sentence_ids: list[str] | None = Field(None, description="组成员（可先建组后填句）")
    order_in_lesson: int | None = Field(None, ge=0, description="课内序号（缺省 = 当前最大 + 1）")


class EnglishEditGroupRequest(BaseModel):
    """PUT 编辑分组请求体（E-API-10，全部可选；sentence_ids 全量替换）。"""

    title: str | None = Field(None, min_length=1, description="组标题")
    type: str | None = Field(None, description="组类型枚举")
    sentence_ids: list[str] | None = Field(None, description="组成员（全量替换语义）")
    order_in_lesson: int | None = Field(None, ge=0, description="课内序号")


class EnglishDeleteGroupRequest(BaseModel):
    """DELETE 删除分组请求体（E-API-11，二次确认组内语句数）。"""

    confirm_sentence_count: int = Field(..., ge=0, description="组内当前语句数（二次确认）")


class EnglishDeduplicateRequest(BaseModel):
    """POST 批量去重请求体（E-API-12，全部可选）。"""

    lesson_id: str | None = Field(None, description="限定课时（缺省 = 整本教材）")
    dry_run: bool = Field(True, description="true 仅预览（零写入）；false 执行级联清理")


# ===========================================================================
# 异常映射（service-contract §8.2：404 四查无 / 400 两校验 / 其余 500）
# ===========================================================================


def _english_error_to_http(exc: Exception) -> HTTPException:
    """英语域业务异常 → HTTP 状态码（与 F1/F2/G 的 _xxx_error_to_http 模式一致）。"""
    if isinstance(exc, (
        TextbookNotFoundError,
        LessonNotFoundError,
        SentenceNotFoundError,
        GroupNotFoundError,
    )):
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


# ===========================================================================
# 8. GET /english/textbook/{tid}/lessons/{lid}/groups — 分组列表（E-API-8，M3 G1.3）
# ===========================================================================


@router.get("/textbook/{textbook_id}/lessons/{lesson_id}/groups")
async def english_lesson_groups_list(
    textbook_id: str,
    lesson_id: str,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=200, description="每页条数"),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 管理端分组列表（E-API-8，含未分组计数 + 组内句子详情，纯读不审）。"""
    try:
        data = await listSentenceGroups(
            db,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            page=page,
            page_size=page_size,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 9. POST /english/textbook/{tid}/lessons/{lid}/groups — 新建分组（E-API-9，M3 G1.3）
# ===========================================================================


@router.post("/textbook/{textbook_id}/lessons/{lesson_id}/groups")
async def english_lesson_groups_create(
    textbook_id: str,
    lesson_id: str,
    body: EnglishCreateGroupRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 新建分组（E-API-9，成员写回 group_id + role_in_group，审计 create_sentence_group）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await createSentenceGroup(
            db,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            title=body.title,
            type=body.type,
            sentence_ids=body.sentence_ids,
            order_in_lesson=body.order_in_lesson,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 10. PUT /english/group/{group_id} — 编辑分组（E-API-10，M3 G1.3）
# ===========================================================================


@router.put("/group/{group_id}")
async def english_group_edit(
    group_id: str,
    body: EnglishEditGroupRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """PUT 编辑分组（E-API-10，sentence_ids 全量替换，审计 edit_sentence_group）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await editSentenceGroup(
            db,
            group_id=group_id,
            title=body.title,
            type=body.type,
            sentence_ids=body.sentence_ids,
            order_in_lesson=body.order_in_lesson,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (GroupNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 12. POST /english/textbook/{textbook_id}/deduplicate — 批量去重（E-API-12）
# ===========================================================================


@router.post("/textbook/{textbook_id}/deduplicate")
async def english_textbook_deduplicate(
    textbook_id: str,
    body: EnglishDeduplicateRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 批量去重（E-API-12）：扫描 text_hash 重复组 → dry_run 预览 / 确认清理。

    两段式：先 dry_run=true 预览（含关联计数，零写入），确认后 dry_run=false
    执行级联清理（保留 canonical = 组内 created_at 最早 / 已有 canonical 自指者）。
    """
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await deduplicateEnglishSentences(
            db,
            textbook_id=textbook_id,
            lesson_id=body.lesson_id,
            dry_run=body.dry_run,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 11. DELETE /english/group/{group_id} — 删除分组（E-API-11，M3 G1.3）
# ===========================================================================


@router.delete("/group/{group_id}")
async def english_group_delete(
    group_id: str,
    body: EnglishDeleteGroupRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """DELETE 删除分组（E-API-11，成员 group_id 置 null 不物理删句，审计 delete_sentence_group）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await deleteSentenceGroup(
            db,
            group_id=group_id,
            confirm_sentence_count=body.confirm_sentence_count,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (GroupNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e


# ===========================================================================
# 12. POST /english/textbook/{textbook_id}/deduplicate — 批量去重（E-API-12）
# ===========================================================================


@router.post("/textbook/{textbook_id}/deduplicate")
async def english_textbook_deduplicate(
    textbook_id: str,
    body: EnglishDeduplicateRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 批量去重（E-API-12）：扫描 text_hash 重复组 → dry_run 预览 / 确认清理。

    两段式：先 dry_run=true 预览（含关联计数，零写入），确认后 dry_run=false
    执行级联清理（保留 canonical = 组内 created_at 最早 / 已有 canonical 自指者）。
    """
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await deduplicateEnglishSentences(
            db,
            textbook_id=textbook_id,
            lesson_id=body.lesson_id,
            dry_run=body.dry_run,
            editor_id=actor,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, LessonNotFoundError, EnglishManagementError) as e:
        raise _english_error_to_http(e) from e
