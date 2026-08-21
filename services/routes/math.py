"""数学学科路由（prefix=/math）

挂载数学学科 F1~F4 后端接口：
- F2  教材描述四接口（curriculum_description）
- F1  教材 AI 知识总结两接口（knowledge_summary）
- F3  练习纸生成接口（practice_sheet）
- F4  错题扫描上传与归类（scan_upload，后续任务）

契约：api-contract.md §3.10（F2 四接口路径 /math/curriculum-node/{id}/...，
id 即 curriculum_node.node_id；F1 两接口 /math/knowledge-summary/...；
F3 接口 /math/practice-sheet；鉴权归管理端付费组 require_paid_user）。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field
from typing import Literal

from services.auth import get_request_openid
from services.database import CURRICULUM_NODE_COLLECTION, CloudBaseNoSQLClient
from services.dependencies import get_db
from services.math.curriculum_description import (
    DescriptionError,
    DescriptionValidationError,
    LLMGenerationError,
    LLMNotConfiguredError,
    NodeNotFoundError,
    NodeTypeUnsupportedError,
    adopt_draft,
    generate_draft,
    get_description,
    save_description,
)
from services.math.knowledge_summary import (
    KnowledgeSummaryError,
    LLMResponseError,
    NoDescriptionError,
    generateKnowledgeSummary,
    getKnowledgeSummary,
    LLMNotConfiguredError as SummaryLLMNotConfiguredError,
    NodeNotFoundError as SummaryNodeNotFoundError,
    NodeTypeUnsupportedError as SummaryNodeTypeUnsupportedError,
)
from services.math.practice_sheet import (
    InvalidSourceError,
    KnowledgePointNotMatchedError,
    MissingScholarError,
    NoQuestionsAvailableError,
    PracticeSheetError,
    generatePracticeSheet,
    getPracticeSheetById,
    listKnowledgeSummaries,
)
from services.math.error_scanner import (
    ImageTooLargeError,
    ImageValidationError,
    JudgeNotConfiguredError,
    JudgeResponseError,
    MissingScholarError as ScanMissingScholarError,
    OcrNotReadyError,
    ScanClassifyError,
    ScanCorrectError,
    ScanCorrectValidationError,
    ScanNotFoundError,
    ScanUploadError,
    StorageError,
    classify_scan_upload,
    correct_scan_classify,
    create_scan_upload,
)
from services.math import (
    ConfirmationMismatchError,
    ERR_NODE_NOT_FOUND,
    TextbookNotFoundError,
    TextbookPayloadError,
    BatchJobNotFoundError,
)
from services.math.knowledge_summary import (
    KnowledgeSummaryError,
    batchGenerateKnowledgeSummary,
    getBatchSummaryStatus,
    manualEditKnowledgeSummary,
    NodeNotFoundError as SummaryNodeNotFoundError,
)
from services.math.textbook_management import (
    create_math_textbook,
    delete_math_textbook_cleanup,
    get_textbook_overview,
    import_curriculum_nodes,
    list_math_textbooks,
    update_math_textbook,
)

logger = logging.getLogger("scholar-admin.routes.math")

router = APIRouter(prefix="/math", tags=["math"])


@router.get("")
async def math_health():
    """数学学科模块占位健康检查（T0.1 骨架端点）"""
    return {"success": True, "data": {"module": "math", "status": "ok"}}


# ---------------------------------------------------------------------------
# F2 教材描述四接口（契约 api-contract.md §3.10）
# ---------------------------------------------------------------------------


class DescriptionContent(BaseModel):
    """教材描述内容（契约 §4.12.8(a)）

    长度约束（summary≤800 字、列表项≤120 字、note≤200 字）由服务层
    _validate_description 统一校验，超长返回 400（避免 Pydantic 422 语义偏差）。
    """

    summary: str = Field(..., description="一段总结，不超过 800 字")
    key_points: list[str] = Field(default_factory=list, description="关键要点列表（每项 ≤120 字）")
    typical_examples: list[dict] = Field(
        default_factory=list, description="典型例题引用 [{ref, note}]"
    )
    prerequisites: list[str] = Field(default_factory=list, description="先修要点列表")
    teaching_tips: list[str] = Field(default_factory=list, description="教学提示列表")


class SaveDescriptionRequest(BaseModel):
    """POST /math/curriculum-node/{id}/description 请求体

    description 用宽松 dict 接收，结构 / 类型 / 长度校验由服务层
    _validate_description 统一完成（非法统一返回 400，避免 Pydantic 422 语义偏差）。
    """

    description: dict = Field(..., description="教材描述内容")


class DraftDescriptionRequest(BaseModel):
    """POST /math/curriculum-node/{id}/description/draft 请求体"""

    force_regenerate: bool = Field(False, description="是否强制重新生成（忽略缓存）")


class AdoptDescriptionRequest(BaseModel):
    """POST /math/curriculum-node/{id}/description/adopt 请求体"""

    description: dict = Field(..., description="待采纳的教材描述内容")


def _description_error_to_http(exc: DescriptionError) -> HTTPException:
    """业务异常 → HTTP 状态码（契约：400 参数/节点类型不支持/LLM 未配置，404 节点不存在，500 生成失败）"""
    if isinstance(exc, NodeNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(
        exc, (DescriptionValidationError, NodeTypeUnsupportedError, LLMNotConfiguredError)
    ):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, LLMGenerationError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.get("/curriculum-node/{id}/description")
async def math_get_description(
    id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 取当前教材描述（含 description_version / description_history）"""
    try:
        data = await get_description(db, "", id)
        return {"success": True, "data": data}
    except DescriptionError as e:
        raise _description_error_to_http(e) from e


@router.post("/curriculum-node/{id}/description")
async def math_save_description(
    id: str,
    body: SaveDescriptionRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 保存人工教材描述（source=manual，版本化 + 历史，写审计）"""
    updated_by = get_request_openid(request) or "anonymous"
    try:
        data = await save_description(
            db,
            "",
            id,
            content=body.description,
            updated_by=updated_by,
        )
        return {"success": True, "data": data}
    except DescriptionError as e:
        raise _description_error_to_http(e) from e


@router.post("/curriculum-node/{id}/description/draft")
async def math_generate_draft(
    id: str,
    body: DraftDescriptionRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST AI 草稿生成：调用 LLM_SUMMARY_MODEL，不写正式描述，写审计"""
    try:
        data = await generate_draft(db, "", id, force_regenerate=body.force_regenerate)
        return {"success": True, "data": data}
    except DescriptionError as e:
        raise _description_error_to_http(e) from e


@router.post("/curriculum-node/{id}/description/adopt")
async def math_adopt_draft(
    id: str,
    body: AdoptDescriptionRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 草稿采纳：source=ai_adopted，版本化 + 历史，写审计"""
    updated_by = get_request_openid(request) or "anonymous"
    try:
        data = await adopt_draft(
            db,
            "",
            id,
            content=body.description,
            updated_by=updated_by,
        )
        return {"success": True, "data": data}
    except DescriptionError as e:
        raise _description_error_to_http(e) from e


# ---------------------------------------------------------------------------
# F1 知识总结两接口（契约 api-contract.md §3.10）
# ---------------------------------------------------------------------------


class GenerateKnowledgeSummaryRequest(BaseModel):
    """POST /math/knowledge-summary/generate 请求体"""

    curriculum_node_id: str = Field(
        ..., description="教材节点 id（curriculum_node.node_id）"
    )
    force_regenerate: bool = Field(
        False, description="是否强制重新生成（忽略已有总结的幂等命中）"
    )
    include_extended_points: bool = Field(
        True, description="是否包含奥数拓展点 extended_points"
    )


def _summary_error_to_http(exc: KnowledgeSummaryError) -> HTTPException:
    """业务异常 → HTTP 状态码（契约：400 参数/无描述/节点类型不支持/LLM 未配置，
    404 节点不存在，500 AI 调用失败）"""
    if isinstance(exc, SummaryNodeNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(
        exc,
        (NoDescriptionError, SummaryNodeTypeUnsupportedError, SummaryLLMNotConfiguredError),
    ):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, LLMResponseError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


# ===========================================================================
# G1.2 批量总结 3 接口（batch-generate / batch-status / manual-edit · 契约 §3.10）
# 注：本节路由放在 F1 两条 knowledge-summary 路由「之前」注册，保证静态路径
# /batch-generate /batch-status 不会被先定义的 GET /{curriculum_node_id} path-param 路由误匹配。
# ===========================================================================


class BatchGenerateRequest(BaseModel):
    """POST /math/knowledge-summary/batch-generate 请求体。"""

    textbook_id: str = Field(..., description="教材 ID（必须 math）", min_length=1)
    scope: Literal["not_generated_only", "all", "force"] = Field(
        "not_generated_only",
        description="调度范围：not_generated_only 跳已生成；all 过 F1 幂等；force 强制重生成",
    )
    node_ids: list[str] | None = Field(
        None,
        description="可选：仅处理指定 node_id 列表（空或 None=该教材全部节点）",
    )


def _batch_summary_error_to_http(exc: Exception) -> HTTPException:
    """G1.2 异常 → HTTP 映射（404/400/500 三段式，batch/manual-edit 共用）。"""
    # 404 族：仅白名单内的「明确 Not Found 异常」（注意 KeyError/IndexError 也是 LookupError 子类但属程序 bug → 走 500）
    if isinstance(exc, (TextbookNotFoundError, BatchJobNotFoundError)):
        code = getattr(exc, "code", None) or getattr(type(exc), "code", None) or "LOOKUP_NOT_FOUND"
        msg = getattr(exc, "message", None) or str(exc)
        return HTTPException(status_code=404, detail={"code": code, "message": msg})
    if isinstance(exc, SummaryNodeNotFoundError):
        return HTTPException(
            status_code=404,
            detail={"code": ERR_NODE_NOT_FOUND, "message": str(exc)},
        )
    # 400 族：业务校验异常（ValueError 语义 + KnowledgeSummaryError 其余 + ConfirmationMismatchError）
    if isinstance(exc, (TextbookPayloadError, ConfirmationMismatchError)):
        return HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": exc.message},
        )
    if isinstance(exc, KnowledgeSummaryError):
        code = f"KNOWLEDGE_SUMMARY_{type(exc).__name__}"
        return HTTPException(status_code=400, detail={"code": code, "message": str(exc)})
    # 500 兜底（含 KeyError/IndexError 等程序 bug）
    logger.exception("Unhandled batch-summary error: %s", exc)
    return HTTPException(
        status_code=500,
        detail={"code": "INTERNAL_ERROR", "message": f"{type(exc).__name__}: {exc}"},
    )


@router.post("/knowledge-summary/batch-generate")
async def math_batch_summary_generate(
    body: BatchGenerateRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST /math/knowledge-summary/batch-generate — 批量调度 AI 总结（G1.2 接口 1）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await batchGenerateKnowledgeSummary(
            db,
            textbook_id=body.textbook_id,
            scope=body.scope,
            node_ids=body.node_ids,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (
        TextbookPayloadError,
        ConfirmationMismatchError,
        KnowledgeSummaryError,
        Exception,
    ) as e:
        raise _batch_summary_error_to_http(e) from e


@router.get("/knowledge-summary/batch-status")
async def math_batch_summary_status(
    job_id: str = Query(..., description="批任务 ID（来自 batch-generate 响应）", min_length=1),
):
    """GET /math/knowledge-summary/batch-status — 查询批任务进度与 items（G1.2 接口 2）。"""
    try:
        data = await getBatchSummaryStatus(job_id=job_id)
        return {"success": True, "data": data}
    except (LookupError, Exception) as e:
        raise _batch_summary_error_to_http(e) from e


@router.post("/knowledge-summary/generate")
async def math_generate_knowledge_summary(
    body: GenerateKnowledgeSummaryRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST AI 知识总结生成（写回 ai_summary + 审计 generate_knowledge_summary；
    服务层幂等命中直接返回已有结果，不重复调用 LLM）"""
    try:
        data = await generateKnowledgeSummary(
            db,
            curriculum_node_id=body.curriculum_node_id,
            force_regenerate=body.force_regenerate,
            include_extended_points=body.include_extended_points,
        )
        return {"success": True, "data": data}
    except KnowledgeSummaryError as e:
        raise _summary_error_to_http(e) from e


@router.get("/knowledge-summary/list")
async def math_list_knowledge_summaries(
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 全部已总结节点摘要（F3.3 小程序选题：勾选知识点）

    复用与生成选题同一口径（_load_all_summarized_nodes），
    返回节点信息 + knowledge_points/extended_points 清单。
    注意：须先于 /knowledge-summary/{curriculum_node_id} 注册，
    避免动态路由吞掉 /list。
    """
    try:
        data = await listKnowledgeSummaries(db)
        return {"success": True, "data": data}
    except KnowledgeSummaryError as e:
        raise _summary_error_to_http(e) from e


@router.get("/knowledge-summary/{curriculum_node_id}")
async def math_get_knowledge_summary(
    curriculum_node_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 取节点最新 AI 知识总结（未生成返回 status=not_generated，不报错）"""
    try:
        data = await getKnowledgeSummary(db, curriculum_node_id=curriculum_node_id)
        return {"success": True, "data": data}
    except KnowledgeSummaryError as e:
        raise _summary_error_to_http(e) from e


# ---------------------------------------------------------------------------
# F3 练习纸生成接口（契约 api-contract.md §3.10，2026-08-19 扩展）
# ---------------------------------------------------------------------------


class GeneratePracticeSheetRequest(BaseModel):
    """POST /math/practice-sheet 请求体

    source / knowledge_points / include_extended_points 为 F3.1 扩展入参
    （契约 api-contract.md §3.10；data-model-contract.md §4.12.10(a)）。
    """

    scholar_id: str = Field(..., description="学生 id")
    template_type: str = Field("standard", description="练习纸版式（MVP 仅 standard）")
    node_codes: list[str] | None = Field(
        None, description="教材知识点编码列表（curriculum_node.code，可选）"
    )
    primary_errors: list[dict] | None = Field(
        None, description="错因筛选 [{node_code, type}]（可选）"
    )
    difficulty_bands: list[dict] | None = Field(
        None, description="难度档位 [{node_code, band}]（可选）"
    )
    source: str = Field(
        "wrong_book",
        description="选题源：wrong_book（错题本，默认）/ ai_knowledge（F1 知识点清单）/ mixed",
    )
    knowledge_points: list[dict] | None = Field(
        None,
        description="AI 知识点清单（source=ai_knowledge/mixed 必填，每项 {name, ability_dimensions[]?, extended_point?}）",
    )
    include_extended_points: bool = Field(
        False, description="是否含奥数扩展题（仅 source=ai_knowledge/mixed 生效）"
    )
    wrong_book_ratio: float = Field(
        0.5, description="错题占比（source=mixed 时有效，默认 0.5）"
    )


def _practice_sheet_error_to_http(exc: PracticeSheetError) -> HTTPException:
    """业务异常 → HTTP 状态码（契约：400 参数/知识点未匹配/无可用选题/LLM 未配置，500 出题失败）"""
    if isinstance(
        exc,
        (
            InvalidSourceError,
            MissingScholarError,
            KnowledgePointNotMatchedError,
            NoQuestionsAvailableError,
            SummaryLLMNotConfiguredError,
        ),
    ):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, LLMResponseError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/practice-sheet")
async def math_generate_practice_sheet(
    body: GeneratePracticeSheetRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 生成 A4 练习纸（三种选题源；同参数 10 分钟内幂等；
    落库 practice_sheet + 渲染任务入队 + 审计 action=generate）"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await generatePracticeSheet(
            db,
            scholar_id=body.scholar_id,
            template_type=body.template_type,
            node_codes=body.node_codes,
            primary_errors=body.primary_errors,
            difficulty_bands=body.difficulty_bands,
            source=body.source,
            knowledge_points=body.knowledge_points,
            include_extended_points=body.include_extended_points,
            wrong_book_ratio=body.wrong_book_ratio,
            actor=actor,
        )
        return {"success": True, "data": data}
    except PracticeSheetError as e:
        raise _practice_sheet_error_to_http(e) from e


@router.get("/practice-sheet/{sheet_id}")
async def math_get_practice_sheet(
    sheet_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 练习纸公开详情（F3.3 小程序端轮询渲染状态；契约 §3.10）

    返回 _to_public_sheet 结构 + render_status（渲染任务最新状态），
    供前端区分"渲染中 / 已成功 / 渲染失败"后展示 PDF/PNG 下载入口。
    二维码扫码校验（签名/过期/账号绑定）由 S3 家长核对页按契约实现，
    本接口先支持小程序生成后轮询场景。
    """
    try:
        data = await getPracticeSheetById(db, sheet_id=sheet_id)
    except PracticeSheetError as e:
        raise _practice_sheet_error_to_http(e) from e
    if data is None:
        raise HTTPException(status_code=404, detail="练习纸不存在")
    return {"success": True, "data": data}


# ---------------------------------------------------------------------------
# F4 错题扫描上传接口（契约 api-contract.md §3.10）
# ---------------------------------------------------------------------------


def _scan_upload_error_to_http(exc: ScanUploadError) -> HTTPException:
    """业务异常 → HTTP 状态码（契约：400 缺参/格式不合法，413 体积超限，500 存储/未知）"""
    if isinstance(exc, (ScanMissingScholarError, ImageValidationError)):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ImageTooLargeError):
        return HTTPException(status_code=413, detail=str(exc))
    if isinstance(exc, StorageError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/scan/upload")
async def math_scan_upload(
    request: Request,
    scholar_id: str | None = Form(None, description="学生 id"),
    image: UploadFile = File(..., description="错题图片（multipart 文件）"),
    note: str | None = Form(None, description="备注（可选）"),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 错题图片上传（F4.1）

    scholar_id 允许缺省由服务层统一校验返回 400（沿用 F2 惯例：
    避免 Pydantic 422 语义偏差）。

    链路：sha256 去重（同图幂等返回既有 scan_id，不新增记录）→ 图片落
    CloudBase 云存储 → 落库 math_scan_upload（ocr_status=pending）→
    写审计 scan_upload → 后台异步触发 OCR（F4.2 接入 provider 前保持 pending）。

    出参（契约 §3.10）：{scan_id, status, image_url, image_file_id, image_hash, deduped}
    """
    actor = get_request_openid(request) or "anonymous"
    image_bytes = await image.read()
    try:
        data = await create_scan_upload(
            db,
            scholar_id=scholar_id,
            image_bytes=image_bytes,
            filename=image.filename or "",
            note=note,
            actor=actor,
        )
        return {"success": True, "data": data}
    except ScanUploadError as e:
        raise _scan_upload_error_to_http(e) from e


# ---------------------------------------------------------------------------
# F4.3 扫描归类接口（契约 api-contract.md §3.10 POST /math/scan/classify）
# ---------------------------------------------------------------------------


class ClassifyScanRequest(BaseModel):
    """POST /math/scan/classify 请求体

    scan_id 用宽松 str 接收（默认空串），缺参统一由路由层判 400 返回，
    避免 Pydantic 422 语义偏差（沿用 F2/F4.1 惯例）。
    """

    scan_id: str = Field("", description="扫描记录 id（必）")
    force_reclassify: bool = Field(
        False, description="是否强制重新归类（默认 false）"
    )


def _scan_classify_error_to_http(exc: ScanClassifyError) -> HTTPException:
    """业务异常 → HTTP 状态码（契约 §3.10：404 scan 不存在/OCR 未完成，500 Judge 不可用/解析失败）"""
    if isinstance(exc, (ScanNotFoundError, OcrNotReadyError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (JudgeNotConfiguredError, JudgeResponseError)):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/scan/classify")
async def math_scan_classify(
    body: ClassifyScanRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 扫描归类（F4.3）

    链路：拉取 ocr_text / ocr_blocks → 调用 LLM_JUDGE_MODEL 做
    "题目识别 + 知识点定位 + 错因判定" → 置信度门控（>=0.6 写 error_record，
    <0.6 / 知识点无法定位 → needs_review 不写 error_record）→ 写审计 scan_classify。

    幂等：classify_status ∈ {success, needs_review} 且非 force_reclassify
    → 直接返回已有结果（不重复调用 Judge，不重复写 error_record）。

    出参（契约 §3.10）：{scan_id, status, items[{error_record_id, knowledge_point_name,
    error_type, confidence, ocr_block_id}]}
    """
    actor = get_request_openid(request) or "anonymous"
    if not body.scan_id:
        raise HTTPException(status_code=400, detail="缺少 scan_id")
    try:
        data = await classify_scan_upload(
            db,
            scan_id=body.scan_id,
            force_reclassify=body.force_reclassify,
            actor=actor,
        )
        return {"success": True, "data": data}
    except ScanClassifyError as e:
        raise _scan_classify_error_to_http(e) from e


# ---------------------------------------------------------------------------
# F4.4 人工修正归类接口（契约 api-contract.md §3.10 POST /math/scan/{scan_id}/correct）
# ---------------------------------------------------------------------------


class CorrectItemRequest(BaseModel):
    """修正项（契约 §3.10 items[] 单项）

    所有字段可选：已有 error_record_id → 更新；无 → 新建。
    """

    error_record_id: str = Field("", description="已有错因记录 id（提供则更新，否则新建）")
    knowledge_point_name: str = Field("", description="修正后的知识点 name")
    error_type: str = Field(
        "", description="错因四分类 concept/method/computation/reading"
    )
    raw_text_corrected: str = Field("", description="人工修正后的原始题干文本（可选）")


class CorrectScanRequest(BaseModel):
    """POST /math/scan/{scan_id}/correct 请求体"""

    items: list[CorrectItemRequest] = Field(
        ..., description="修正项列表（至少 1 项）"
    )


def _scan_correct_error_to_http(
    exc: ScanCorrectError | ScanNotFoundError,
) -> HTTPException:
    """业务异常 → HTTP 状态码（契约 §3.10：400 参数校验，404 scan 不存在，500 未知）"""
    if isinstance(exc, ScanNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ScanCorrectValidationError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/scan/{scan_id}/correct")
async def math_scan_correct(
    scan_id: str,
    body: CorrectScanRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 人工修正归类（F4.4）

    链路：读取 scan → 遍历 items → 已有 error_record_id 更新归类
    （classify_method=manual_corrected），无 error_record_id 新建 error_record
    （classify_method=manual_corrected, source=manual_corrected）→
    更新 classify_status=success → 写审计 scan_correct。

    出参（契约 §3.10）：{scan_id, corrected[{error_record_id, knowledge_point_name, error_type}]}
    """
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await correct_scan_classify(
            db,
            scan_id=scan_id,
            items=[item.model_dump() for item in body.items],
            actor=actor,
        )
        return {"success": True, "data": data}
    except (ScanCorrectError, ScanNotFoundError) as e:
        raise _scan_correct_error_to_http(e) from e


# ===========================================================================
# G1.1 数学教材管理 6 接口（SOP §5，契约 §3.10 API-3）
# ===========================================================================


# ------------- 请求体 Pydantic 模型 ------------- #


class MathTextbookCreateRequest(BaseModel):
    """POST /math/textbook 入参（对齐 math textbook_v2 DM-2）。"""
    title: str = Field(..., min_length=1, description="教材全名")
    grade: str = Field(..., description="年级，如 七年级")
    semester: str | None = Field(None, description="数学生效：up/down")
    subject_type: str | None = Field("math", description="固定 math（本接口禁止 english）")
    publisher: str | None = Field(None)
    cover_url: str | None = Field(None)
    isbn: str | None = Field(None)
    textbook_id: str | None = Field(None, description="自定义教材ID，缺省自动生成")
    chapters: list[dict] = Field(default_factory=list)


class MathTextbookUpdateRequest(BaseModel):
    """PUT /math/textbook/{id} 入参。改标题/跨学科改 subject_type 需 confirm_title 二次确认。"""
    title: str | None = Field(None)
    grade: str | None = Field(None)
    semester: str | None = Field(None)
    subject_type: str | None = Field(None)
    publisher: str | None = Field(None)
    cover_url: str | None = Field(None)
    isbn: str | None = Field(None)
    chapters: list[dict] | None = Field(None)
    confirm_title: str = Field("", description="改标题/改 subject_type 时必须传入确认标题")


class MathTextbookImportNodesRequest(BaseModel):
    """POST /math/textbook/import-nodes 入参。"""
    textbook_id: str = Field(..., min_length=1)
    nodes: list[dict] = Field(..., description="curriculum_node 列表，code 幂等")
    on_duplicate: str = Field("skip", description="遇到同 code 时：skip 跳过（默认）/ update 覆盖")


class MathTextbookDeleteCleanupRequest(BaseModel):
    """DELETE /math/textbook/{id} 入参：需 confirm_textbook_title 二次确认。"""
    confirm_textbook_title: str = Field(..., min_length=1, description="必须与当前教材标题完全相等")


# ------------- 业务异常 → HTTP 映射 ------------- #


def _textbook_error_to_http(exc: Exception) -> HTTPException:
    """Textbook* 系列错误 → HTTP 状态码（契约 §3.10：404 不存在 / 400 校验失败 / 500 兜底）。"""
    if isinstance(exc, TextbookNotFoundError):
        return HTTPException(status_code=404, detail={
            "code": exc.code, "message": str(exc),
        })
    if isinstance(exc, (TextbookPayloadError, ConfirmationMismatchError)):
        return HTTPException(status_code=400, detail={
            "code": getattr(exc, "code", "BAD_REQUEST"), "message": str(exc),
        })
    return HTTPException(status_code=500, detail={"code": "INTERNAL_ERROR", "message": str(exc)})


# ------------- 6 路由端点 ------------- #


@router.get("/textbook")
async def math_textbook_list(
    subject_type: str = "math",
    grade: str | None = None,
    semester: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
    limit: int = 100,
    request: Request = None,  # type: ignore[assignment]
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET /math/textbook — 数学教材列表。"""
    try:
        data = await list_math_textbooks(
            db,
            subject_type=subject_type,
            grade=grade,
            semester=semester,
            keyword=keyword,
            offset=offset,
            limit=limit,
        )
        return {"success": True, "data": data}
    except TextbookPayloadError as e:
        raise _textbook_error_to_http(e) from e


@router.post("/textbook")
async def math_textbook_create(
    body: MathTextbookCreateRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST /math/textbook — 新增数学教材。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await create_math_textbook(db, payload=body.model_dump(), actor=actor)
        return {"success": True, "data": data}
    except (TextbookPayloadError, ConfirmationMismatchError) as e:
        raise _textbook_error_to_http(e) from e


@router.put("/textbook/{textbook_id}")
async def math_textbook_update(
    textbook_id: str,
    body: MathTextbookUpdateRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """PUT /math/textbook/{textbook_id} — 更新教材（改标题/学科需二次确认）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await update_math_textbook(
            db,
            textbook_id=textbook_id,
            update=body.model_dump(exclude_unset=False),
            confirm_title=body.confirm_title,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (TextbookPayloadError, TextbookNotFoundError, ConfirmationMismatchError) as e:
        raise _textbook_error_to_http(e) from e


@router.get("/textbook/overview")
async def math_textbook_overview(
    textbook_id: str,
    request: Request = None,  # type: ignore[assignment]
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET /math/textbook/overview — 教材 + 节点统计概览（G-API-4）。"""
    try:
        data = await get_textbook_overview(db, textbook_id=textbook_id)
        return {"success": True, "data": data}
    except TextbookNotFoundError as e:
        raise _textbook_error_to_http(e) from e


@router.post("/textbook/import-nodes")
async def math_textbook_import_nodes(
    body: MathTextbookImportNodesRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST /math/textbook/import-nodes — 批量导入 curriculum_node（G-API-5，code 幂等）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await import_curriculum_nodes(
            db,
            textbook_id=body.textbook_id,
            nodes=body.nodes,
            on_duplicate=body.on_duplicate,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (TextbookPayloadError, TextbookNotFoundError, ConfirmationMismatchError) as e:
        raise _textbook_error_to_http(e) from e


@router.delete("/textbook/{textbook_id}")
async def math_textbook_delete_cleanup(
    textbook_id: str,
    body: MathTextbookDeleteCleanupRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """DELETE /math/textbook/{id} — 清理 description + ai_summary（不删节点结构，G-API-6）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await delete_math_textbook_cleanup(
            db,
            textbook_id=textbook_id,
            confirm_textbook_title=body.confirm_textbook_title,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (TextbookNotFoundError, ConfirmationMismatchError) as e:
        raise _textbook_error_to_http(e) from e


# ===========================================================================
# G1.2 第 3 接口：人工修正 AI 知识点（DM-3 manual_edited* 字段 · 契约 §3.10 + §4.12.8(b)）
# batch-generate / batch-status 两路由 + 异常映射函数已放在文件前段（F1 两接口之前）注册，
# 保证静态路径 /knowledge-summary/batch-{generate,status} 不会被先定义的
# GET /knowledge-summary/{curriculum_node_id} 路径参数路由误抢占匹配。
# ===========================================================================


class ManualEditSummaryRequest(BaseModel):
    """POST /math/curriculum-node/{id}/manual-edit-summary 请求体（DM-3 manual_edited*）。"""

    knowledge_points: list[dict] | None = Field(
        None,
        description="替换 ai_summary.knowledge_points（显式传 [] 清空，None 不改）",
    )
    extended_points: list[dict] | None = Field(
        None,
        description="替换 ai_summary.extended_points（显式传 [] 清空，None 不改）",
    )
    overwrite_ai: bool = Field(
        False,
        description="MVP 仅打标；true 表示后续 force 重生成允许覆盖人工编辑内容（SOP 暂未实现重生成行为）",
    )
    editor_id: str = Field(
        "",
        description="人工修正者标识；空串或缺省时使用请求 openid（fallback anonymous）",
    )


@router.post("/curriculum-node/{curriculum_node_id}/manual-edit-summary")
async def math_curriculum_node_manual_edit_summary(
    curriculum_node_id: str,
    body: ManualEditSummaryRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST /math/curriculum-node/{id}/manual-edit-summary — 人工修正 AI 知识点（G1.2 接口 3，DM-3）。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await manualEditKnowledgeSummary(
            db,
            curriculum_node_id=curriculum_node_id,
            editor_id=body.editor_id,
            knowledge_points=body.knowledge_points,
            extended_points=body.extended_points,
            overwrite_ai=body.overwrite_ai,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (
        LookupError,
        KnowledgeSummaryError,
        ConfirmationMismatchError,
        TextbookPayloadError,
        Exception,
    ) as e:
        raise _batch_summary_error_to_http(e) from e


# ===========================================================================
# 接口 26：GET /math/textbook/{textbook_id}/knowledge-points（2026-08-21 SOP ⑤ K5）
# 按教材聚合全部节点的 ai_summary 知识点，返回扁平知识点列表（小程序知识点列表页消费）
# 契约：api-contract.md §3.10 接口 26
# ===========================================================================


@router.get("/textbook/{textbook_id}/knowledge-points")
async def math_get_textbook_knowledge_points(
    textbook_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET /math/textbook/{textbook_id}/knowledge-points — 按教材聚合知识点列表（接口 26）"""
    from services.math.textbook_management import _get_math_textbook_or_404

    # 1. 校验教材存在且为 math
    tb = await _get_math_textbook_or_404(db, textbook_id)
    tb_title = tb.get("title") or ""

    # 2. 查该教材所有 curriculum_node（含 ai_summary）
    query = await db.query(
        CURRICULUM_NODE_COLLECTION,
        where={"textbook_id": textbook_id},
        limit=1000,
    )
    nodes = query.get("records", [])

    # 3. 扁平化：遍历节点 → 展开 ai_summary.knowledge_points[]
    kps: list[dict] = []
    for node in nodes:
        ai = node.get("ai_summary") or {}
        if ai.get("status") != "success":
            continue
        node_id = node.get("node_id") or ""
        node_title = node.get("title") or ""
        quality_score = ai.get("quality_score", -1)
        extended_points = ai.get("extended_points") or []

        for kp in ai.get("knowledge_points") or []:
            kps.append({
                "node_id": node_id,
                "node_title": node_title,
                **kp,
                "quality_score": quality_score,
                "extended_points": extended_points,
            })

    return {
        "success": True,
        "data": {
            "textbook_id": textbook_id,
            "title": tb_title,
            "knowledge_points": kps,
            "total": len(kps),
        },
    }
