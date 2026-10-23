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

import asyncio
import json
import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Literal

import httpx

from services.audit import (
    AUDIT_ACTION_QUIZ_RENDER,
    AUDIT_RESULT_SUCCESS,
    write_audit,
)

from services.auth import get_request_openid
from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    CloudBaseNoSQLClient,
)
from services.dependencies import get_db
from services.math.a4_renderer import (
    RENDER_JOB_RENDERING,
    QuizNotFoundError,
    renderQuizJob,
)
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
from services.math.quiz import (
    QUIZ_DEFAULT_COUNT_PER_SOURCE,
    QUIZ_STATUS_GENERATED,
    QUIZ_STATUS_RENDERED,
    QUIZ_STATUS_RENDERING,
    InvalidQuizSourceError,
    NodeCodeNotFoundError,
    generateQuiz,
    getQuizById,
)
from services.math.quiz_recognizer import recognize_quiz_scan
from services.math.skill_progress import build_skill_progress
from services.math.error_scanner import (
    EXTRA_AI_TEXTBOOK_ID,
    KP_SOURCE_EXAM_PAPER,
    KP_SOURCE_TEXTBOOK,
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
    list_curriculum_nodes,
    create_curriculum_node,
    update_curriculum_node,
    delete_curriculum_node,
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
    skill_weakness: list[dict] | None = Field(
        None,
        description="技能条弱信号（M14 双源出题驱动入参，[{display_group, kp_ids?}]；落库/出参回显并纳入幂等签名）",
    )
    include_exam_variants: bool = Field(
        False,
        description="真题同款变式开关（M14 双源：按学者真题错题 exam_backlink_to 回链本次选中教材点出变式，items[].source_type=exam_paper）",
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
            skill_weakness=body.skill_weakness,
            include_exam_variants=body.include_exam_variants,
            wrong_book_ratio=body.wrong_book_ratio,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (PracticeSheetError, LLMResponseError) as e:
        # LLMResponseError（AI 出题失败，契约 500）继承自 KnowledgeSummaryError 而非
        # PracticeSheetError，需一并映射，避免未处理异常吞掉 detail
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


class ScanUploadJsonRequest(BaseModel):
    """POST /math/scan/upload 的 **JSON 通道**入参（云开发中转，2026-10-23）

    背景：小程序 `wx.uploadFile` 直连云托管公网域名会校验「合法域名」，而该域名须
    ICP 备案（云托管默认域名仅供测试）。故新增通道 A：前端先把图传到**云开发存储**
    （`wx.cloud.uploadFile`，免白名单）→ `getTempFileURL` 取临时链接 → 以 JSON 提交
    `image_url`，后端下载字节后复用同一条 F4.1 主流程。

    与 multipart 通道**二选一**，后端以 `image` 文件是否存在判定通道，出参完全一致。
    """

    scholar_id: str = Field("", description="学生 id")
    image_url: str = Field("", description="云存储临时下载链接（http/https）")
    note: str = Field("", description="备注（可选）")
    photo_context: str = Field("", description="教材上下文 JSON 字符串（可选）")


async def _parse_json_upload_payload(request: Request) -> ScanUploadJsonRequest | None:
    """解析 JSON 通道入参

    ⚠️ FastAPI 在**同时存在 File/Form 参数**时不会解析 JSON body 参数（body 被当作表单
    处理，`Body(None)` 恒为 None，实测确认）。故这里按 content-type 手动解析：
    仅 `application/json` 才读原始 body（multipart 时流已被 Starlette 消费，读会抛
    RuntimeError）。
    """
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" not in content_type:
        return None
    raw = await request.body()
    if not raw:
        return None
    try:
        return ScanUploadJsonRequest.model_validate_json(raw)
    except ValidationError as e:
        errs = e.errors()
        msg = errs[0].get("msg", "入参不合法") if errs else "入参不合法"
        raise HTTPException(status_code=400, detail=f"请求体不合法: {msg}") from e


async def _fetch_image_bytes(url: str) -> bytes:
    """下载 `image_url` → bytes（云开发中转通道）

    体积/格式校验统一交给服务层 `_validate_image`（保持与 multipart 通道同一口径），
    这里只负责网络与状态码；失败按 400 返回（入参不可用，非服务故障）。
    """
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="image_url 非法：必须是 http(s) 链接")
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"下载图片失败: {e}") from e
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=400, detail=f"下载图片失败: HTTP {resp.status_code}"
        )
    return resp.content


def _filename_from_image_url(url: str) -> str:
    """从临时链接推导文件名（供扩展名校验）；无法识别时回落 `scan.jpg`"""
    name = url.split("?", 1)[0].rsplit("/", 1)[-1]
    return name if "." in name else "scan.jpg"


@router.post("/scan/upload")
async def math_scan_upload(
    request: Request,
    scholar_id: str | None = Form(None, description="学生 id（multipart 通道）"),
    image: UploadFile | None = File(
        None, description="错题图片（multipart 通道，与 JSON 通道 image_url 二选一）"
    ),
    note: str | None = Form(None, description="备注（可选，multipart 通道）"),
    photo_context: str | None = Form(
        None, description="教材上下文 JSON 字符串（可选，multipart 通道）"
    ),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 错题图片上传（F4.1）

    两条通道（后端按是否有 `image` 文件自动判定，出参与落库完全一致）：
    - **multipart**：`image` 文件 + `scholar_id` / `note`（原通道，保留兼容）；
    - **JSON（云开发中转）**：`{scholar_id, image_url, note, photo_context}`，
      后端下载 `image_url` 得到字节后走同一条主流程（2026-10-23 新增，免备案域名）。

    scholar_id 允许缺省由服务层统一校验返回 400（沿用 F2 惯例：
    避免 Pydantic 422 语义偏差）。

    链路：sha256 去重（同图幂等返回既有 scan_id，不新增记录）→ 图片落
    CloudBase 云存储 → 落库 math_scan_upload（ocr_status=pending）→
    写审计 scan_upload → 后台异步触发 OCR（F4.2 接入 provider 前保持 pending）。

    出参（契约 §3.10）：{scan_id, status, image_url, image_file_id, image_hash, deduped}
    """
    actor = get_request_openid(request) or "anonymous"
    if image is not None:
        image_bytes = await image.read()
        filename = image.filename or ""
        sid = scholar_id
        note_value = note
        context_value = photo_context
    else:
        payload = await _parse_json_upload_payload(request)
        if payload is None or not payload.image_url:
            raise HTTPException(
                status_code=400, detail="缺少图片：请上传 image 文件或提交 image_url"
            )
        image_bytes = await _fetch_image_bytes(payload.image_url)
        filename = _filename_from_image_url(payload.image_url)
        sid = payload.scholar_id
        note_value = payload.note
        context_value = payload.photo_context or None
    try:
        data = await create_scan_upload(
            db,
            scholar_id=sid,
            image_bytes=image_bytes,
            filename=filename,
            note=note_value,
            actor=actor,
            photo_context=context_value,
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
    photo_context: str = Field(
        "",
        description="拍照携带的教材上下文（可选，M9）：JSON 字符串 "
        '{"textbook_id": "...", "unit_title": "...", "lesson_title": "..."}；'
        "存在且链校验通过时作为归类优先锚点，缺省不携带 = 现状自动猜",
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
            photo_context=body.photo_context,
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
    new_kp_name: str = Field(
        "",
        description="图谱外 → 新建知识点专用名（B1.5）：picker 未命中任何教材图谱时 "
        "LLM 预填、用户可编辑，提交后按 EXTRA_AI 幂等规则新建/复用节点并锚定",
    )
    error_type: str = Field(
        "", description="错因四分类 concept/method/computation/reading"
    )
    question_text: str = Field(
        "",
        description="人工修正后的题干原文（可选；与 raw_text_corrected 语义合并，"
        "落库统一写 question_text，raw_text_corrected 保留兼容字段）",
    )
    raw_text_corrected: str = Field("", description="人工修正后的原始题干文本（可选，兼容字段）")


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
# F4.6 学者错题列表（契约 api-contract.md §3.10 接口 25 · MVP 原始列表版）
# ===========================================================================


@router.get("/error-stats")
async def math_error_stats(
    scholar_id: str = Query("", description="学者 id（必）"),
    limit: int = Query(100, ge=1, le=500, description="最多返回条数，按 created_at 倒序"),
    knowledge_point_name: str | None = Query(
        None, description="按知识点名过滤（可选）"
    ),
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 学者历史错题列表（接口 25 · MVP 简单列表版）

    后端直接查 error_record，不做聚合；前端在 JS 层聚合统计。
    MVP 出参：{ success, data: { items: [{ error_record_id, knowledge_point_name,
      error_type, source, created_at, question_text, occurrence,
      textbook_id, grade, semester, unit_title, lesson_title, node_title,
      node_code, drill_stats, last_drill_result }], total } }
    字段映射：record_id → error_record_id，primary_error → error_type；
    knowledge_point_name 为识别/修正落库的冗余字段（缺失时回退 node_code）。
    B2 追加透传（B1/B1.5 已随落库，缺省空串/0/{} 向后兼容历史数据）：
    question_text 题干原文、occurrence 错题次数、链锚点
    （textbook_id/grade/semester/unit_title/lesson_title/node_title）、
    drill_stats/last_drill_result 巩固证据 —— 供前端列表行「原题概述」、
    教材链归集与先修/拔高判定直接渲染。
    """
    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少 scholar_id")

    where: dict = {"scholar_id": scholar_id}
    if knowledge_point_name:
        where["knowledge_point_name"] = knowledge_point_name

    res = await db.query(
        ERROR_RECORD_COLLECTION,
        where=where,
        order=[{"field": "created_at", "direction": "desc"}],
        limit=limit,
    )
    records = res.get("records") or []
    items = [
        {
            "error_record_id": r.get("record_id") or "",
            "knowledge_point_name": (
                r.get("knowledge_point_name") or r.get("node_code") or ""
            ),
            "error_type": r.get("primary_error") or "",
            "source": r.get("source") or "",
            "created_at": r.get("created_at") or 0,
            # ── B2：题干 / 链锚点 / 巩固证据透传（缺省兜底，前端 normalize 零改动）──
            "question_text": r.get("question_text") or "",
            "occurrence": r.get("occurrence") or 0,
            "textbook_id": r.get("textbook_id") or "",
            "grade": r.get("grade") or "",
            "semester": r.get("semester") or "",
            "unit_title": r.get("unit_title") or "",
            "lesson_title": r.get("lesson_title") or "",
            "node_title": r.get("node_title") or "",
            "node_code": r.get("node_code") or "",
            # ── M13：双源透传（knowledge_point 双源治理，方案 §3.2 / O3=A）──
            # kp_source 缺省按数据事实兜底：EXTRA_AI 真题/图谱外锚定记录 = exam_paper，
            # 正式教材链/无链 = textbook——存量 EXTRA_AI 记录零回填即可识别，前端零改动。
            # exam_backlink_to 软回链透传（真题题簇命中教材点才落库，缺省 None）。
            "kp_source": r.get("kp_source")
            or (
                KP_SOURCE_EXAM_PAPER
                if (r.get("textbook_id") or "") == EXTRA_AI_TEXTBOOK_ID
                else KP_SOURCE_TEXTBOOK
            ),
            "exam_backlink_to": r.get("exam_backlink_to") or None,
            "drill_stats": r.get("drill_stats") or {},
            "last_drill_result": r.get("last_drill_result") or {},
        }
        for r in records
    ]
    # total 取全量计数（不受 limit 截断影响）
    try:
        total = await db.count(ERROR_RECORD_COLLECTION, where=where) or len(items)
    except Exception:
        total = len(items)
    # M11 ②：学者维度技能条统计（与 items 同源同批；无单元技能条/无可映射记录 → [] 兼容存量）
    try:
        skill_progress = await build_skill_progress(db, records)
    except Exception:
        logger.warning("error-stats skill_progress 聚合失败，降级空数组", exc_info=True)
        skill_progress = []
    return {
        "success": True,
        "data": {
            "items": items,
            "total": total,
            "skill_progress": skill_progress,
        },
    }


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


# ===========================================================================
# Curriculum Node CRUD（管理页面用）
# ===========================================================================


@router.get("/textbook/{textbook_id}/nodes")
async def math_list_curriculum_nodes(
    textbook_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET /math/textbook/{textbook_id}/nodes — 列出教材下所有课程节点。"""
    try:
        nodes = await list_curriculum_nodes(db, textbook_id)
        return {"success": True, "data": nodes, "total": len(nodes)}
    except TextbookNotFoundError as e:
        raise _textbook_error_to_http(e) from e


@router.post("/textbook/{textbook_id}/nodes")
async def math_create_curriculum_node(
    textbook_id: str,
    body: dict,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST /math/textbook/{textbook_id}/nodes — 创建单个课程节点。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        doc = await create_curriculum_node(db, textbook_id=textbook_id, node=body, actor=actor)
        return {"success": True, "data": doc}
    except (TextbookPayloadError, TextbookNotFoundError) as e:
        raise _textbook_error_to_http(e) from e


@router.put("/curriculum-node/{node_id}")
async def math_update_curriculum_node(
    node_id: str,
    body: dict,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """PUT /math/curriculum-node/{node_id} — 更新单个课程节点。"""
    actor = get_request_openid(request) or "anonymous"
    try:
        result = await update_curriculum_node(db, node_id=node_id, updates=body, actor=actor)
        return {"success": True, "data": result}
    except TextbookPayloadError as e:
        raise _textbook_error_to_http(e) from e


@router.delete("/curriculum-node/{node_id}")
async def math_delete_curriculum_node(
    node_id: str,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """DELETE /math/curriculum-node/{node_id} — 删除单个课程节点。"""
    actor = get_request_openid(request) or "anonymous"
    result = await delete_curriculum_node(db, node_id=node_id, actor=actor)
    return {"success": True, "data": result}


# ---------------------------------------------------------------------------
# 🆕 数学同类型题出题 4 接口（SOP ④ · api-contract §3.10 接口 28~31 · ADR-0035）
# ---------------------------------------------------------------------------

# 渲染后台任务集合：持有强引用，避免 create_task 被 GC 回收
_QUIZ_RENDER_TASKS: set = set()


class QuizRecognizeRequest(BaseModel):
    """POST /math/scan/quiz-recognize 请求体（接口 28）

    字段**宽松接收**（`str` 默认值），缺参 / `photo_context` 非法 JSON 由路由层
    统一判 400（沿用 F2/F4 惯例，避免 Pydantic 422 语义偏差）。
    """

    scholar_id: str = Field("", description="学者 id（必）")
    scan_id: str = Field(
        "", description="POST /math/scan/upload（接口 13）返回的 scan_id（OCR 须已完成，必）"
    )
    photo_context: str = Field(
        "",
        description="教材上下文 JSON（`{textbook_id?, unit_title?, lesson_title?}`）；"
        "缺省 \"\" = 全量自动猜；非法 JSON → 400",
    )


class QuizSourceQuestion(BaseModel):
    """`source_questions[]` 单项（接口 29 入参）

    `source_index` 回指接口 28 出参 `items[]` 下标，用于结果项标注「来自原题 k」
    与幂等签名稳定排序（D9 / D8）。类型校验由服务层统一完成 → 非法一律 400。
    """

    source_index: Any = Field(None, description="回指接口 28 items[] 的下标（必，整数）")
    question_text: str = Field("", description="该题题干（识别结果，可编辑后回传；空串 → 400）")
    knowledge_point_name: str = Field("", description="该题知识点名；缺省 \"\" → prompt 省略知识点段")
    node_code: str = Field("", description="该题知识点节点码；有值时取 ai_summary.summary 注入 prompt")


class GenerateQuizRequest(BaseModel):
    """POST /math/quiz/generate 请求体（接口 29）"""

    scholar_id: str = Field("", description="学者 id（必）")
    scan_id: str = Field("", description="溯源用，记入 math_quiz.source_scan_id")
    source_questions: list[QuizSourceQuestion] = Field(
        default_factory=list, description="被勾选的源题（D9：未勾选题不出现在数组里）"
    )
    count_per_source: Any = Field(
        QUIZ_DEFAULT_COUNT_PER_SOURCE, description="每道源题出几道（1~3 整数，非法 → 400）"
    )
    include_extended: bool = Field(False, description="P1 拔高题开关（本期前端不传）")


class RenderQuizRequest(BaseModel):
    """POST /math/quiz/{quiz_id}/render 请求体（接口 30）"""

    scholar_id: str = Field("", description="学者 id（必）")


def _quiz_recognize_error_to_http(exc: Exception) -> HTTPException:
    """接口 28 异常 → HTTP 状态码（契约：404 scan 不存在/OCR 未就绪，500 Judge/存储失败）"""
    if isinstance(exc, (ScanNotFoundError, OcrNotReadyError)):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, (JudgeNotConfiguredError, JudgeResponseError)):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _quiz_generate_error_to_http(exc: Exception) -> HTTPException:
    """接口 29 异常 → HTTP 状态码（契约：400 入参非法，404 node_code 不存在，500 出题失败）"""
    if isinstance(exc, InvalidQuizSourceError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, NodeCodeNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


@router.post("/scan/quiz-recognize")
async def math_quiz_recognize(
    body: QuizRecognizeRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 拍照题识别（接口 28 · 契约 §3.10）

    链路：读 `math_scan_upload` 的 OCR 结果 → 加载知识点候选集（`photo_context` 命中则收敛）
    → 一次 Judge 调用「题目切分 + 知识点定位」→ 返回**全部**题干供前端勾选。

    **红线 R1**：全程零写库 —— 不写 `error_record`、不回写 OCR/归类状态、不新建 `curriculum_node`。
    审计 `quiz_recognize`（成功 / 失败均写）由服务层写入。
    """
    actor = get_request_openid(request) or "anonymous"
    if not body.scholar_id.strip():
        raise HTTPException(status_code=400, detail="缺少 scholar_id")
    if not body.scan_id.strip():
        raise HTTPException(status_code=400, detail="缺少 scan_id")
    if body.photo_context.strip():
        try:
            json.loads(body.photo_context)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"photo_context 非法 JSON: {e}") from e
    try:
        data = await recognize_quiz_scan(
            db,
            scan_id=body.scan_id,
            scholar_id=body.scholar_id,
            photo_context=body.photo_context,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (ScanNotFoundError, OcrNotReadyError, JudgeNotConfiguredError, JudgeResponseError) as e:
        raise _quiz_recognize_error_to_http(e) from e


@router.post("/quiz/generate")
async def math_quiz_generate(
    body: GenerateQuizRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 出同类型题（接口 29 · 契约 §3.10 · D9）

    只对 `source_questions[]` 内的题出题（**未勾选题零 LLM 调用**），逐源题独立调用
    出题 prompt → 防背题校验（D6）→ 落库 `math_quiz`；出参经 `_to_public_quiz` 净化，
    **`answer` / `hint` 不出参**（红线 R3）。幂等：同参数 10 分钟窗口内返回既有 `quiz_id`（D8）。
    """
    actor = get_request_openid(request) or "anonymous"
    try:
        data = await generateQuiz(
            db,
            scholar_id=body.scholar_id,
            source_questions=[q.model_dump() for q in body.source_questions],
            scan_id=body.scan_id or None,
            count_per_source=body.count_per_source,
            include_extended=body.include_extended,
            actor=actor,
        )
        return {"success": True, "data": data}
    except (InvalidQuizSourceError, NodeCodeNotFoundError) as e:
        raise _quiz_generate_error_to_http(e) from e
    except (LLMResponseError, SummaryLLMNotConfiguredError) as e:
        # LLMResponseError：全部源题出题失败（500）；LLMNotConfiguredError：模型未配置（400）
        if isinstance(e, SummaryLLMNotConfiguredError):
            raise HTTPException(status_code=400, detail=str(e)) from e
        raise _quiz_generate_error_to_http(e) from e


async def _quiz_render_task(db, quiz_id: str) -> None:
    """渲染后台任务骨架：吞掉所有异常（结果由 `math_quiz` 状态承载，供接口 31 轮询）"""
    try:
        await renderQuizJob(db, quiz_id)
    except QuizNotFoundError as e:
        logger.warning("[quiz] render 后台任务中止（quiz 不存在）: %s", e)
    except Exception as e:  # noqa: BLE001
        logger.error("[quiz] render 后台任务异常 quiz_id=%s: %s", quiz_id, e)


def _spawn_quiz_render_task(db, quiz_id: str) -> None:
    """启动渲染后台任务并持有强引用（防 GC）"""
    task = asyncio.create_task(_quiz_render_task(db, quiz_id))
    _QUIZ_RENDER_TASKS.add(task)
    task.add_done_callback(_QUIZ_RENDER_TASKS.discard)


@router.post("/quiz/{quiz_id}/render")
async def math_quiz_render(
    quiz_id: str,
    body: RenderQuizRequest,
    request: Request,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """POST 确认打印（接口 30 · 契约 §3.10 · D7）

    异步渲染 A4（`a4_renderer.renderQuizJob`，**不生成二维码**）：
    `asyncio.create_task` 后立即返回 `status: "rendering"`，前端轮询接口 31 取 `file_refs`。
    审计 `action=quiz_render` 在这里写（成功 / 触发失败均留痕）。
    """
    actor = get_request_openid(request) or "anonymous"
    if not body.scholar_id.strip():
        raise HTTPException(status_code=400, detail="缺少 scholar_id")

    quiz = await getQuizById(db, quiz_id=quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail=f"同类型题不存在: quiz_id={quiz_id}")

    status = quiz.get("status") or ""
    file_refs = quiz.get("file_refs") or {"pdf": "", "png": "", "preview": ""}
    render_message = quiz.get("render_message") or ""
    if status in (QUIZ_STATUS_RENDERING, QUIZ_STATUS_RENDERED):
        # 幂等：渲染中 / 已渲染 → 直接返回既有产物，不重复触发
        return {
            "success": True,
            "data": {
                "quiz_id": quiz_id,
                "status": status,
                "file_refs": file_refs,
                "render_status": quiz.get("render_status") or "",
                "render_message": render_message,
            },
        }
    if status != QUIZ_STATUS_GENERATED:
        # 失败原因对用户可见：failed 时带上 `render_message`（如“渲染依赖未安装”）
        detail = f"当前状态不允许渲染（status={status}），请重新出题"
        if render_message:
            detail = f"{detail}：{render_message}"
        raise HTTPException(status_code=409, detail=detail)

    _spawn_quiz_render_task(db, quiz_id)
    await write_audit(
        db,
        action=AUDIT_ACTION_QUIZ_RENDER,
        object_ref=quiz_id,
        actor=actor,
        result=AUDIT_RESULT_SUCCESS,
        context={
            "scholar_id": body.scholar_id,
            "item_count": len(quiz.get("items") or []),
        },
    )
    return {
        "success": True,
        "data": {
            "quiz_id": quiz_id,
            "status": QUIZ_STATUS_RENDERING,
            "file_refs": file_refs,
            "render_status": RENDER_JOB_RENDERING,
            "render_message": "",
        },
    }


@router.get("/quiz/{quiz_id}")
async def math_get_quiz(
    quiz_id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
):
    """GET 同类型题详情 / 轮询渲染产物（接口 31 · 契约 §3.10）

    出参为 `_to_public_quiz` 结构 —— `items[].answer` / `hint` **恒不出参**（红线 R3）；
    `render_status` 由渲染任务回写，前端据此区分「渲染中 / 已成功 / 渲染失败」。
    """
    data = await getQuizById(db, quiz_id=quiz_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"同类型题不存在: quiz_id={quiz_id}")
    return {"success": True, "data": data}
