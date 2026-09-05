"""错题扫描上传与归类服务（F4.1 / F4.3）

契约：
- api-contract.md §3.10  POST /math/scan/upload、POST /math/scan/classify
- data-model-contract.md §4.12.9  math_scan_upload 集合 + error_record 扩展字段
- ADR-0020（MVP 腾讯云通用印刷体 OCR；LLM Judge 错因四分类 + 置信度门控）

F4.1 链路：上传图片 → sha256 去重（同图幂等返回既有 scan_id，不新增记录）
→ 图片落对象存储（CloudBase 云存储，复用 TCB 凭据）→ 落库 math_scan_upload
（ocr_status=pending）→ 写审计 scan_upload → 异步触发 OCR 任务。

F4.3 链路：拉取 ocr_text / ocr_blocks → 调用 LLM_JUDGE_MODEL 做
"题目识别 + 知识点定位 + 错因判定" → 置信度门控（>=0.6 写 error_record，
<0.6 / 知识点无法定位 → needs_review 不写 error_record）→ 写审计 scan_classify。
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import time
from typing import Any, Optional

from config import (
    EVAL_CONFIDENCE_THRESHOLD,
    LLM_JUDGE_CANDIDATE_LIMIT,
    LLM_JUDGE_DISABLE_THINKING,
    LLM_JUDGE_MODEL,
    LLM_JUDGE_OCR_TEXT_MAX,
    LLM_JUDGE_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_IMAGE_FORMATS,
    VOLCANO_MAX_IMAGE_SIZE,
)
from services.audit import (
    AUDIT_ACTION_SCAN_CLASSIFY,
    AUDIT_ACTION_SCAN_CORRECT,
    AUDIT_ACTION_SCAN_UPLOAD,
    AUDIT_RESULT_FAILED,
    write_audit,
)
from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
    TEXTBOOK_V2,
)
from services.tcb_storage import CloudBaseStorageClient

logger = logging.getLogger("scholar-admin.math.error_scanner")

# ---------------------------------------------------------------------------
# 状态常量（契约 §4.12.9）
# ---------------------------------------------------------------------------

# ocr_status 状态机（契约 §4.12.9：pending → processing → success / failed）
OCR_STATUS_PENDING = "pending"
OCR_STATUS_PROCESSING = "processing"
OCR_STATUS_SUCCESS = "success"
OCR_STATUS_FAILED = "failed"

# classify_status 状态机（契约 §4.12.9：pending → classifying → success / failed / needs_review）
# 注：F4.1 初版用 pending/classified/reviewed 三态偏离契约，F4.3 按契约 §0.4 对齐为 5 态
CLASSIFY_STATUS_PENDING = "pending"
CLASSIFY_STATUS_CLASSIFYING = "classifying"
CLASSIFY_STATUS_SUCCESS = "success"
CLASSIFY_STATUS_FAILED = "failed"
CLASSIFY_STATUS_NEEDS_REVIEW = "needs_review"

# 云存储路径前缀
SCAN_STORAGE_PREFIX = "scan"

# ---------------------------------------------------------------------------
# B1.5 图谱外知识点（EXTRA_AI 虚拟教材，主文档 §4.3 / 契约 §1.2b）
#   未命中正式教材候选集的高置信知识点 → 挂「课外补充 · AI 归类」虚拟教材下
#   幂等新建/复用节点，保证每条错题都有完整链锚点（不强行改挂正式节点）。
# ---------------------------------------------------------------------------

EXTRA_AI_TEXTBOOK_ID = "EXTRA_AI"
EXTRA_AI_TEXTBOOK_TITLE = "课外补充 · AI 归类"
EXTRA_AI_UNCLASSIFIED = "未分类"
EXTRA_AI_NODE_CODE_PREFIX = "xai_"
EXTRA_AI_CANDIDATE_HITS_LIMIT = 3  # 疑似正式教材匹配候选 ≤3（契约 §1.2b）


# ---------------------------------------------------------------------------
# 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class ScanUploadError(Exception):
    """错题扫描上传业务错误基类"""


class MissingScholarError(ScanUploadError):
    """缺少 scholar_id"""


class ImageValidationError(ScanUploadError):
    """图片格式不合法 / 空文件"""


class ImageTooLargeError(ScanUploadError):
    """图片体积超限"""


class StorageError(ScanUploadError):
    """对象存储上传失败"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _gen_scan_id() -> str:
    """scan_id 生成：scan_{毫秒时间戳}_{随机hex}"""
    return f"scan_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _calc_image_hash(image_bytes: bytes) -> str:
    """图片指纹：sha256(image_bytes)，用于幂等去重"""
    return hashlib.sha256(image_bytes).hexdigest()


def _detect_ext(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _validate_image(filename: str, image_bytes: bytes) -> None:
    """校验图片格式与体积（契约错误码：格式不合法→400，体积超限→413）"""
    ext = _detect_ext(filename)
    if ext not in VOLCANO_IMAGE_FORMATS:
        raise ImageValidationError(
            f"不支持的图片格式: .{ext or '?'}，仅支持 {', '.join(VOLCANO_IMAGE_FORMATS)}"
        )
    if not image_bytes:
        raise ImageValidationError("图片内容为空")
    if len(image_bytes) > VOLCANO_MAX_IMAGE_SIZE:
        raise ImageTooLargeError(
            f"图片体积 {len(image_bytes)} 字节超过上限 {VOLCANO_MAX_IMAGE_SIZE} 字节"
        )


# ---------------------------------------------------------------------------
# 异步 OCR 任务钩子（F4.2 接入后填充真实 provider）
# ---------------------------------------------------------------------------


async def _run_ocr_job(db: Any, scan_id: str, image_bytes: bytes) -> None:
    """后台 OCR 任务（MVP 占位：F4.2 未接入 provider 前保持 ocr_status=pending）

    F4.2 接入后：导入 services.math.ocr 的 provider，识别后回写
    ocr_text / ocr_blocks / ocr_status=success（失败置 failed）。
    """
    try:
        from services.math import ocr  # F4.2 后存在

        provider = ocr.get_provider()
        result = await provider.recognize(image_bytes)
        await db.update(
            MATH_SCAN_UPLOAD_COLLECTION,
            where={"scan_id": scan_id},
            data={
                "$set": {
                    "ocr_status": OCR_STATUS_SUCCESS,
                    "ocr_text": result.text,
                    "ocr_blocks": result.blocks,
                    "completed_at": int(time.time() * 1000),
                }
            },
        )
        logger.info(f"[scan] OCR 完成 scan_id={scan_id}")
    except ImportError:
        logger.info(f"[scan] OCR Provider 待 F4.2 接入，scan_id={scan_id} 保持 pending")
    except Exception as e:
        logger.error(f"[scan] OCR 失败 scan_id={scan_id}: {e}", exc_info=True)
        try:
            await db.update(
                MATH_SCAN_UPLOAD_COLLECTION,
                where={"scan_id": scan_id},
                data={"$set": {"ocr_status": OCR_STATUS_FAILED, "ocr_error": str(e)[:500]}},
            )
        except Exception:
            pass


def _schedule_ocr(db: Any, scan_id: str, image_bytes: bytes) -> None:
    """创建后台 OCR 任务（不阻塞上传响应；无运行中事件循环时仅打日志）"""
    try:
        asyncio.get_running_loop().create_task(_run_ocr_job(db, scan_id, image_bytes))
    except RuntimeError:
        logger.warning(f"[scan] 无运行中事件循环，跳过后台 OCR 调度 scan_id={scan_id}")


# ---------------------------------------------------------------------------
# 出参结构
# ---------------------------------------------------------------------------


def _to_public_upload(record: dict, *, deduped: bool) -> dict:
    """出参结构（契约 §3.10：scan_id / status / image_url / image_hash / deduped）"""
    return {
        "scan_id": record.get("scan_id"),
        "status": record.get("ocr_status", OCR_STATUS_PENDING),
        "image_url": record.get("image_url", ""),
        "image_file_id": record.get("image_file_id", ""),
        "image_hash": record.get("image_hash", ""),
        "deduped": deduped,
    }


# ---------------------------------------------------------------------------
# F4.1 主流程
# ---------------------------------------------------------------------------


async def create_scan_upload(
    db: Any,
    *,
    scholar_id: str,
    image_bytes: bytes,
    filename: str,
    note: str | None = None,
    actor: str = "",
    storage: CloudBaseStorageClient | None = None,
) -> dict:
    """创建错题扫描上传记录（F4.1 主流程）

    1. 校验入参（缺 scholar_id / 图片格式 / 体积）
    2. sha256 去重：同 hash 直接返回既有 scan_id（幂等，不新增记录）
    3. 图片落对象存储（CloudBase 云存储），image_file_id + image_url
    4. 落库 math_scan_upload（ocr_status=pending）
    5. 写审计 scan_upload
    6. 后台异步触发 OCR 任务
    """
    if not scholar_id:
        raise MissingScholarError("缺少 scholar_id")
    _validate_image(filename, image_bytes)

    image_hash = _calc_image_hash(image_bytes)

    # 幂等去重：同图（同 sha256）直接返回既有记录
    existing = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION, where={"image_hash": image_hash}, limit=1
    )
    records = existing.get("records") or []
    if records:
        rec = records[0]
        logger.info(
            f"[scan] 幂等命中 image_hash={image_hash[:12]} scan_id={rec.get('scan_id')}"
        )
        return _to_public_upload(rec, deduped=True)

    # 对象存储上传
    scan_id = _gen_scan_id()
    ext = _detect_ext(filename) or "jpg"
    cloud_path = f"{SCAN_STORAGE_PREFIX}/{scan_id}.{ext}"
    storage = storage or CloudBaseStorageClient()
    try:
        upload_info = await storage.upload_file(cloud_path, image_bytes)
    except Exception as e:
        logger.error(f"[scan] 云存储上传失败 cloud_path={cloud_path}: {e}", exc_info=True)
        raise StorageError(f"图片上传失败: {e}") from e

    image_file_id = upload_info["file_id"]
    try:
        image_url = await storage.get_temp_file_url(cloud_path)
    except Exception as e:
        logger.warning(f"[scan] 获取临时 URL 失败（image_file_id 仍可用）: {e}")
        image_url = image_file_id

    # 落库（字段对齐契约 §4.12.9）
    now_ms = int(time.time() * 1000)
    record = {
        "scan_id": scan_id,
        "scholar_id": scholar_id,
        "image_url": image_url,
        "image_file_id": image_file_id,
        "image_hash": image_hash,
        "ocr_status": OCR_STATUS_PENDING,
        "ocr_text": "",
        "ocr_blocks": [],
        "classify_status": CLASSIFY_STATUS_PENDING,
        "note": note or "",
        "created_at": now_ms,
        "completed_at": None,
        "audit_log_id": "",
    }
    await db.insert(MATH_SCAN_UPLOAD_COLLECTION, record)

    # 写审计 scan_upload（必审；失败仅记录日志，不阻断上传主流程）
    try:
        audit_entry = await write_audit(
            db,
            action=AUDIT_ACTION_SCAN_UPLOAD,
            object_ref=scan_id,
            actor=actor,
            context={
                "scholar_id": scholar_id,
                "image_hash": image_hash,
                "image_size": len(image_bytes),
                "deduped": False,
            },
        )
        await db.update(
            MATH_SCAN_UPLOAD_COLLECTION,
            where={"scan_id": scan_id},
            data={"$set": {"audit_log_id": audit_entry.get("log_id", "")}},
        )
    except Exception as e:
        logger.error(f"[scan] 审计写入失败 scan_id={scan_id}: {e}")

    # 异步触发 OCR（MVP：后台执行，响应先行返回）
    _schedule_ocr(db, scan_id, image_bytes)

    return _to_public_upload(record, deduped=False)


# ===========================================================================
# F4.3 扫描归类（契约 api-contract.md §3.10 POST /math/scan/classify）
# ===========================================================================
#
# 链路：拉取 ocr_text / ocr_blocks → 调用 LLM_JUDGE_MODEL 做
# "题目识别 + 知识点定位 + 错因判定" → 置信度门控（>=0.6 写 error_record，
# <0.6 / 知识点无法定位 → needs_review 不写 error_record）→ 写审计 scan_classify。
#
# Judge 模型沿用 LLM_JUDGE_MODEL（与 LLM_SUMMARY_MODEL 解耦，ADR-0020 决策 C）；
# 知识点候选集来自 F1 生成的 curriculum_node.ai_summary.knowledge_points[]，
# 按学者当前教材（scholar_book → textbook_id）过滤，无记录时回退全量候选。


# 错因四分类（契约 §4.12.2 / ADR-0020：concept / method / computation / reading）
ERROR_TYPES = ("concept", "method", "computation", "reading")

# error_record 扩展枚举（契约 §4.12.9(b)）
CLASSIFY_METHOD_AUTO_SCAN = "auto_scan"
CLASSIFY_METHOD_MANUAL_CORRECTED = "manual_corrected"
SOURCE_AUTO_SCAN = "auto_scan"
SOURCE_MANUAL_CORRECTED = "manual_corrected"

# 学者×教材关联集合（不在 database.py 常量中，按字面量引用）
_SCHOLAR_BOOK_COLLECTION = "scholar_book"

# Judge LLM 客户端单例（独立于 LLM_SUMMARY_MODEL 的总结客户端）
_judge_client: Optional[Any] = None


def _get_judge_client() -> Any:
    """获取 Judge 模型客户端（单次调用最长 LLM_JUDGE_TIMEOUT_SECONDS 秒；单例）"""
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI

        _judge_client = OpenAI(
            api_key=VOLCANO_API_KEY,
            base_url=VOLCANO_BASE_URL,
            timeout=LLM_JUDGE_TIMEOUT_SECONDS,
        )
    return _judge_client


# ---------------------------------------------------------------------------
# F4.3 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class ScanClassifyError(Exception):
    """错题扫描归类业务错误基类"""


class ScanNotFoundError(ScanClassifyError):
    """scan_id 不存在（路由层映射 404）"""


class OcrNotReadyError(ScanClassifyError):
    """OCR 未完成（路由层映射 404，契约 §3.10）"""


class JudgeNotConfiguredError(ScanClassifyError):
    """LLM_JUDGE_MODEL 未配置（路由层映射 500）"""


class JudgeResponseError(ScanClassifyError):
    """Judge 响应解析失败（路由层映射 500）"""


# ---------------------------------------------------------------------------
# 知识点候选集（契约：当前年级教材知识点候选集，来自 F1 ai_summary）
# ---------------------------------------------------------------------------


async def _load_scholar_textbook_ids(db, scholar_id: str) -> list[str]:
    """读取学者的教材关联（scholar_book → textbook_id），用于按年级过滤候选集"""
    if not scholar_id:
        return []
    res = await db.query(
        _SCHOLAR_BOOK_COLLECTION,
        where={"scholar_id": scholar_id},
        limit=50,
    )
    records = res.get("records") or []
    return [r.get("textbook_id") for r in records if r.get("textbook_id")]


async def _load_knowledge_point_candidates(
    db, scholar_id: str
) -> list[dict[str, Any]]:
    """加载知识点候选集（F1 生成的 ai_summary.knowledge_points[]）

    优先按学者当前教材（scholar_book → textbook_id）过滤；无关联记录时
    回退全量已总结节点（保证新学者也能归类，Judge 按 OCR 内容匹配）。

    返回 [{node_id, node_code, kp_name, grade, semester, textbook_id, title}]，
    title/semester 供 B1.5 链锚点冗余落库（命中候选时 node_title/grade/semester 直接可取）。
    """
    _SELECT = {
        "node_id": 1,
        "code": 1,
        "grade": 1,
        "semester": 1,
        "textbook_id": 1,
        "title": 1,
        "ai_summary": 1,
    }
    textbook_ids = await _load_scholar_textbook_ids(db, scholar_id)

    nodes: list[dict] = []
    for offset in range(0, 2000, 500):
        where: dict[str, Any] = {}
        if textbook_ids:
            where = {"textbook_id": {"$in": textbook_ids}}
        res = await db.query(
            CURRICULUM_NODE_COLLECTION,
            where=where,
            select=_SELECT,
            offset=offset,
            limit=500,
        )
        batch = res.get("records") or []
        nodes.extend(batch)
        if len(batch) < 500:
            break

    # 若按教材过滤后无节点（学者无 scholar_book 或教材未总结），回退全量
    if not nodes and textbook_ids:
        for offset in range(0, 2000, 500):
            res = await db.query(
                CURRICULUM_NODE_COLLECTION,
                select=_SELECT,
                offset=offset,
                limit=500,
            )
            batch = res.get("records") or []
            nodes.extend(batch)
            if len(batch) < 500:
                break

    candidates: list[dict[str, Any]] = []
    for node in nodes:
        ai = node.get("ai_summary")
        if not isinstance(ai, dict) or ai.get("status") != "success":
            continue
        for kp in ai.get("knowledge_points") or []:
            name = (kp.get("name") or "").strip()
            if not name:
                continue
            candidates.append(
                {
                    "node_id": node.get("node_id") or "",
                    "node_code": node.get("code") or "",
                    "kp_name": name,
                    "grade": node.get("grade") or "",
                    "semester": node.get("semester") or "",
                    "textbook_id": node.get("textbook_id") or "",
                    "title": node.get("title") or "",
                }
            )
    return candidates


# ---------------------------------------------------------------------------
# Judge Prompt（题目识别 + 知识点定位 + 错因判定）
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = (
    "你是小学数学错题归类助手。根据给定的 OCR 文本与知识点候选集，"
    "识别每道错题并判定其错因与对应知识点。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_CLASSIFY_USER_TEMPLATE = """请对以下 OCR 识别出的错题文本进行归类。

OCR 全文：
{ocr_text}

知识点候选集（name 即知识点名，grade 为年级，node_code 为知识点编码）：
{candidates}

要求：
1. 识别 OCR 文本中的每道错题（按题号或独立题干切分），每道题输出一项。
2. knowledge_point_name：优先从候选集中选最匹配的知识点 name；候选集无匹配项时，按 OCR 内容给出最贴近的知识点名称（不要填空字符串，尽量具体，如"大数加减法""两位数乘两位数"）。
3. error_type：错因四分类，取值 concept（概念错）/ method（方法错）/ computation（计算错）/ reading（审题错）。
4. confidence：本道题归类置信度 0~1，按「题干语义是否明确」给分，不以「是否命中候选集」为限（B1.5）：
   命中候选集且匹配明确 → ≥0.7；候选集外但题干完整、知识点可明确命名且与题面语义吻合 → ≥0.6
   （这类将自动走课外图谱 EXTRA_AI 新建直落，属正常高置信，不必压到低置信）；
   题干残缺/语义模糊/错因不明/知识点名无从把握 → <0.6（进 needs_review 人工修正）。
5. ocr_block_id：本道题在 OCR 检测块中对应的 block_id（取题干所在块，无则填空字符串 ""）。
6. question_text：从上方 OCR 全文**原文截取**本道题的完整题干（含条件与问题，逐字保留不改写、不补全）；一道大题拆成多个 item 时各自截取对应小题题干；无法截取（OCR 缺失/边界不明）填空字符串 ""。
7. candidate_hits：当本道题 knowledge_point_name **未出现在候选集**时，从候选集中挑 1~3 个**最疑似匹配**的知识点 name（数组，按疑似度降序）；knowledge_point_name 已在候选集内或候选集为空/无疑似项时输出 []。

输出 JSON 结构：
{{
  "items": [
    {{"knowledge_point_name": "...", "error_type": "concept|method|computation|reading", "confidence": 0.85, "ocr_block_id": "blk_0001", "question_text": "本道题题干原文", "candidate_hits": ["疑似正式教材知识点名1"]}}
  ]
}}"""


def _format_candidates(
    candidates: list[dict[str, Any]], limit: int = LLM_JUDGE_CANDIDATE_LIMIT
) -> str:
    """格式化候选集为 prompt 文本（限制条目数控制 token）"""
    if not candidates:
        return "（候选集为空，请按 OCR 内容自行推断每道题的知识点名称）"
    lines: list[str] = []
    for c in candidates[:limit]:
        lines.append(
            f"- name={c['kp_name']}, grade={c['grade']}, node_code={c['node_code']}"
        )
    if len(candidates) > limit:
        lines.append(f"（另有 {len(candidates) - limit} 个候选知识点省略）")
    return "\n".join(lines)


def _call_judge_sync(
    client: Any, model: str, system: str, prompt: str
) -> str:
    """同步 chat 调用（在线线程池中执行）

    推理模型（doubao-seed-2-1-pro 等）默认开启深度推理，真实规模下
    推理耗时 >120s 远超超时限制且占用 max_tokens；默认禁用 thinking，
    实测真实规模从 >120s 降到 ~3s（LLM_JUDGE_DISABLE_THINKING=0 可保留推理）。
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    if LLM_JUDGE_DISABLE_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


async def _call_classify_judge(
    ocr_text: str, candidates: list[dict[str, Any]]
) -> dict:
    """调用 LLM_JUDGE_MODEL 归类（JSON 解析失败重试 1 次，仍失败抛 JudgeResponseError）

    Judge 模型未配置抛 JudgeNotConfiguredError（路由层映射 500）。
    """
    if not (VOLCANO_API_KEY and LLM_JUDGE_MODEL):
        raise JudgeNotConfiguredError(
            "LLM_JUDGE_MODEL 未配置，无法归类（需配置 VOLCANO_API_KEY + LLM_JUDGE_MODEL）"
        )
    from services.dialogue import _parse_json_response

    # 截断 OCR 全文控制 token（LLM_JUDGE_OCR_TEXT_MAX 字符）
    ocr_text = (ocr_text or "")[:LLM_JUDGE_OCR_TEXT_MAX]
    prompt = _CLASSIFY_USER_TEMPLATE.format(
        ocr_text=ocr_text or "（OCR 文本为空）",
        candidates=_format_candidates(candidates),
    )
    client = _get_judge_client()
    last_err: Exception | None = None
    logger.info(
        f"[scan] Judge 调用开始 model={LLM_JUDGE_MODEL} "
        f"候选集={len(candidates)} 条 OCR文本={len(ocr_text)} 字 prompt={len(prompt)} 字符"
    )
    t0 = time.time()
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_judge_sync, client, LLM_JUDGE_MODEL,
                _CLASSIFY_SYSTEM_PROMPT, prompt,
            )
            result = _parse_json_response(response)
            _validate_classify_result(result)
            logger.info(
                f"[scan] Judge 调用成功 耗时={time.time() - t0:.1f}s "
                f"items={len(result.get('items') or [])} 条（第 {attempt} 次）"
            )
            return result
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 JudgeResponseError
            last_err = e
            logger.warning(
                f"[scan] 归类 Judge 调用失败（第 {attempt} 次）耗时={time.time() - t0:.1f}s: "
                f"{type(e).__name__}: {e}"
            )
    raise JudgeResponseError(f"扫描归类 Judge 调用失败: {last_err}")


def _validate_classify_result(result: dict) -> None:
    """校验 Judge 输出结构（契约 §3.10 items[]），非法抛 JudgeResponseError"""
    if not isinstance(result, dict):
        raise JudgeResponseError("Judge 输出必须为 JSON 对象")
    items = result.get("items")
    if not isinstance(items, list):
        raise JudgeResponseError("Judge 输出 items 必须为数组")
    for item in items:
        if not isinstance(item, dict):
            raise JudgeResponseError("items 每项必须为对象")
        et = item.get("error_type", "")
        if et and et not in ERROR_TYPES:
            raise JudgeResponseError(
                f"error_type 必须为 {list(ERROR_TYPES)} 之一，得到 {et!r}"
            )
        conf = item.get("confidence", 0)
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            raise JudgeResponseError(f"confidence 必须为数值，得到 {conf!r}")
        if conf_val < 0 or conf_val > 1:
            raise JudgeResponseError(f"confidence 必须在 0~1 之间，得到 {conf_val}")
        item["confidence"] = conf_val
        # question_text：题干原文，允许为空（OCR 不完整/未截取时走 needs_review，
        # 语义不变）；非字符串值归一为空串，不阻断归类（B1 契约 §1.2a）
        qt = item.get("question_text")
        item["question_text"] = qt.strip() if isinstance(qt, str) else ""
        # candidate_hits：疑似正式教材匹配名（≤3，B1.5 契约 §1.2b）；非法值归一 []
        raw_hits = item.get("candidate_hits")
        hits: list[str] = []
        if isinstance(raw_hits, list):
            for h in raw_hits:
                if isinstance(h, str) and h.strip() and h.strip() not in hits:
                    hits.append(h.strip())
        item["candidate_hits"] = hits[:EXTRA_AI_CANDIDATE_HITS_LIMIT]


# ---------------------------------------------------------------------------
# error_record 写入（仅高置信 + 知识点已定位项）
# ---------------------------------------------------------------------------


def _gen_record_id() -> str:
    """error_record ID 生成：er_{毫秒时间戳}_{随机hex}"""
    return f"er_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _match_candidate(
    kp_name: str, candidates: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """按 name 精确匹配知识点候选；无匹配返回 None（→ needs_review）"""
    if not kp_name:
        return None
    for c in candidates:
        if c["kp_name"] == kp_name:
            return c
    return None


def _kp_bigrams(text: str) -> set[str]:
    """知识点名的 2-gram 字符集（语义近似基础，中文无需分词）"""
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _strip_common_suffix(a: str, b: str) -> tuple[str, str]:
    """剥离两名的公共尾部（泛化模板后缀，如「的实际应用」/「计算规则」）。

    中文知识点名常以模板后缀收尾，bigram Jaccard 会因此虚高——「小数加法的
    实际应用」与「统计的实际应用」共享「的实际应用」gram 得 j≈0.4 却语义无关
    （真实候选集实测）。就近比较前剥掉公共尾，让交集只反映主题词相似。
    """
    i = 0
    while i < len(a) and i < len(b) and a[-1 - i] == b[-1 - i]:
        i += 1
    return (a[:-i] if i else a, b[:-i] if i else b)


# B1.5a 就近改名护栏：互斥运算语义词。中文知识点名 bigram 对单字操作符差异
# 不敏感——「小数加法的实际应用」vs「小数减法的实际应用」字符 2-gram
# Jaccard≈0.56 ≥0.25，机械就近会把加法题误挂到减法知识点（报告「加法↔减法
# 一字差误挂」风险）。就近改名前先做操作符互斥检查：两边各含运算语义词但
# 无交集 → 运算不同，禁止就近（共享语义词如「乘法…」≈「乘除…」不受影响）。
_OPERATOR_CONFLICT_TERMS = ("加", "减", "乘", "除")


def _operator_terms_in(name: str) -> set[str]:
    """知识点名中的运算语义词（加/减/乘/除），供操作符互斥护栏使用"""
    return {t for t in _OPERATOR_CONFLICT_TERMS if t in name}


def _operator_conflict(a: str, b: str) -> bool:
    """a/b 是否含互斥运算语义（如「加法…」vs「减法…」）。"""
    ops_a = _operator_terms_in(a)
    ops_b = _operator_terms_in(b)
    return bool(ops_a and ops_b) and not (ops_a & ops_b)


def _nearest_candidates(
    kp_name: str, candidates: list[dict[str, Any]], limit: int = 1
) -> list[dict[str, Any]]:
    """语义就近：2-gram Jaccard ≥0.25 的近似候选（精确恒排最前，按相似度降序）。

    B1.5「候选内修正命名」：correct 传入的自拟名未精确命中候选时，若与某候选
    语义接近（如「加法简便运算」≈ 候选「加法运算」：字符 2-gram 交集非空且
    Jaccard≥0.25），就近锚定该候选并修正为标准知识点名，避免图谱外误建 /
    机械改挂首个（报告 manual_correct 根因）。「单位换算」与教材候选交集为空 →
    不就近，走图谱外 EXTRA_AI 新建。

    B1.5a：就近前过两道护栏，防 bigram 相似但语义无关的误挂：
    1) 操作符互斥（_operator_conflict）——加减乘除互斥名（「加法…」vs
       「减法…」一字差）一律不就近；
    2) 公共模板后缀剥离（_strip_common_suffix）——先剥「的实际应用」等
       泛化尾再算 Jaccard，避免共享模板后缀虚高（真实候选集实测
       「小数加法的实际应用」vs「统计的实际应用」j≈0.4 但语义无关）。
    classify 直落 / correct 修正 / candidate_hits 建议共用此函数，天然同口径。
    """
    if not kp_name or not candidates:
        return []
    name = kp_name.strip()
    if len(name) < 2:
        return []
    exact: list[dict[str, Any]] = []
    scored: list[tuple[float, dict[str, Any]]] = []
    for c in candidates:
        cand = (c.get("kp_name") or "").strip()
        if not cand or len(cand) < 2:
            continue
        if cand == name:
            exact.append(c)
            continue
        if _operator_conflict(name, cand):
            continue  # 护栏 1：操作符互斥（加法↔减法）→ 不可就近
        core_a, core_cand = _strip_common_suffix(name, cand)
        ta = _kp_bigrams(core_a)
        cg = _kp_bigrams(core_cand)
        inter = len(ta & cg)
        if inter == 0:
            continue
        j = inter / len(ta | cg)
        if j >= 0.25:
            scored.append((j, c))
    # 同相似度按候选集原序（稳定排序）
    scored.sort(key=lambda t: -t[0])
    return (exact + [c for _, c in scored])[:limit]


def _candidate_hits_struct(
    judge_names: list[str], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """疑似正式教材改挂候选 → 结构化 [{textbook_id, grade, kp_name}]（≤3，契约 §1.2b）

    Judge 可按名回传 candidate_hits；后端用候选集映射为带教材/年级的结构化条目
    （随响应下发仅供前端建议，本期不持久化）。Judge 名精确/就近匹配不到候选，
    或 Judge 未回传时，用题面主名（judge_names[0]）就近子串兜底给建议。
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(c: dict[str, Any]) -> None:
        kp = c.get("kp_name") or ""
        if not kp or kp in seen:
            return
        seen.add(kp)
        out.append(
            {
                "textbook_id": c.get("textbook_id") or "",
                "grade": c.get("grade") or "",
                "kp_name": kp,
            }
        )

    for nm in judge_names or []:
        name = str(nm).strip()
        if not name:
            continue
        exact = _match_candidate(name, candidates)
        if exact:
            _push(exact)
        else:
            for near in _nearest_candidates(name, candidates, limit=1):
                _push(near)
        if len(out) >= EXTRA_AI_CANDIDATE_HITS_LIMIT:
            break
    return out[:EXTRA_AI_CANDIDATE_HITS_LIMIT]


# ---------------------------------------------------------------------------
# B1.5 EXTRA_AI 虚拟教材与图谱外节点（主文档 §4.3，幂等）
# ---------------------------------------------------------------------------


async def _ensure_extra_ai_textbook(db) -> None:
    """幂等 seed 虚拟教材「课外补充 · AI 归类」（textbook_v2）

    只作图谱外知识点挂靠锚点；不进入学者教材列表 / 教材三选 / 选材（§4.3）。
    """
    res = await db.query(
        TEXTBOOK_V2, where={"textbook_id": EXTRA_AI_TEXTBOOK_ID}, limit=1
    )
    if res.get("records"):
        return
    now_ms = int(time.time() * 1000)
    await db.insert(
        TEXTBOOK_V2,
        {
            "textbook_id": EXTRA_AI_TEXTBOOK_ID,
            "title": EXTRA_AI_TEXTBOOK_TITLE,
            "grade": "",
            "semester": "",
            "subject_type": "math",
            "publisher": "",
            "cover_url": "",
            "isbn": "",
            "chapters": [],
            "created_at": now_ms,
            "updated_at": now_ms,
            "note": "B1.5 图谱外知识点虚拟挂靠教材（主文档 §4.3），不进入选材与进度",
        },
    )


def _extra_ai_node_code(kp_name: str) -> str:
    """EXTRA_AI 节点 code：xai_{kp名 md5 前12}（稳定幂等，避免与正式教材 code 冲突）"""
    return f"{EXTRA_AI_NODE_CODE_PREFIX}{hashlib.md5(kp_name.encode('utf-8')).hexdigest()[:12]}"


async def _ensure_extra_ai_node(db, kp_name: str) -> dict[str, Any]:
    """图谱外知识点节点幂等新建/复用（title 精确匹配，§4.3）

    返回链锚点 dict（与候选 dict 同构，可直接喂 _write_error_record）：
    {node_code, kp_name, title, textbook_id, grade, semester, unit_title, lesson_title}
    unit/lesson 缺省「未分类」；textbook_id=EXTRA_AI。
    """
    name = (kp_name or "").strip()
    if not name:
        raise ValueError("图谱外知识点名不能为空")
    await _ensure_extra_ai_textbook(db)
    res = await db.query(
        CURRICULUM_NODE_COLLECTION,
        where={"textbook_id": EXTRA_AI_TEXTBOOK_ID, "title": name},
        limit=1,
    )
    existing = (res.get("records") or [])
    if existing:
        node = existing[0]
        return {
            "node_code": node.get("code") or _extra_ai_node_code(name),
            "kp_name": name,
            "title": name,
            "textbook_id": EXTRA_AI_TEXTBOOK_ID,
            "grade": "",
            "semester": "",
            "unit_title": EXTRA_AI_UNCLASSIFIED,
            "lesson_title": EXTRA_AI_UNCLASSIFIED,
        }
    code = _extra_ai_node_code(name)
    now_ms = int(time.time() * 1000)
    await db.insert(
        CURRICULUM_NODE_COLLECTION,
        {
            "textbook_id": EXTRA_AI_TEXTBOOK_ID,
            "node_id": f"{EXTRA_AI_TEXTBOOK_ID}_{code}",
            "code": code,
            "title": name,
            "node_type": "knowledge",
            "grade": "",
            "semester": "",
            "unit_title": EXTRA_AI_UNCLASSIFIED,
            "lesson_title": EXTRA_AI_UNCLASSIFIED,
            "created_at": now_ms,
            "updated_at": now_ms,
            "note": "B1.5 图谱外知识点（AI 归类），title 精确幂等",
        },
    )
    return {
        "node_code": code,
        "kp_name": name,
        "title": name,
        "textbook_id": EXTRA_AI_TEXTBOOK_ID,
        "grade": "",
        "semester": "",
        "unit_title": EXTRA_AI_UNCLASSIFIED,
        "lesson_title": EXTRA_AI_UNCLASSIFIED,
    }


def _is_extra_ai_anchor(matched: dict[str, Any]) -> bool:
    """链锚点是否挂在 EXTRA_AI 虚拟教材（图谱外）"""
    return (matched.get("textbook_id") or "") == EXTRA_AI_TEXTBOOK_ID


def _chain_anchor_fields(matched: dict[str, Any] | None) -> dict[str, Any]:
    """从链锚点（正式候选 / EXTRA_AI 节点）提取 error_record 链字段（§4.1）。

    与 _write_error_record 落库口径一致，correct 的 update/insert 分支复用，
    保证人工修正/改挂后链锚点与 node_code 同步刷新。
    """
    if not matched:
        return {}
    return {
        "node_code": matched.get("node_code") or "",
        "textbook_id": matched.get("textbook_id") or "",
        "grade": matched.get("grade") or "",
        "semester": matched.get("semester") or "",
        "unit_title": matched.get("unit_title") or "",
        "lesson_title": matched.get("lesson_title") or "",
        "node_title": matched.get("node_title") or matched.get("title")
        or matched.get("kp_name") or "",
    }


async def _write_error_record(
    db,
    *,
    scan_id: str,
    scholar_id: str,
    ocr_block_id: str,
    matched: dict[str, Any],
    error_type: str,
    confidence: float,
    knowledge_point_name: str = "",
    question_text: str = "",
    original_kp_name: str = "",
) -> str:
    """写一条 error_record（契约 §4.12.2 + §4.12.9(b) 扩展字段）

    matched：链锚点 dict —— 命中正式候选（_load_knowledge_point_candidates 单项）
    或图谱外 EXTRA_AI 节点（_ensure_extra_ai_node 返回值，textbook_id=EXTRA_AI）。

    knowledge_point_name：识别命中/人工修正时的知识点名（冗余落库，
    供错题列表直接展示，免 join；与 node_code 并存）。

    original_kp_name：B1.5a classify 直落就近改名时的原判名（Judge 自拟、
    与 knowledge_point_name 不同的名字）。仅就近改名分支传入并落库——eval
    评估与前端展示借此识别「自动就近改名」（对称于 EXTRA_AI 分支的 new_kp_name，
    因为 classify_result 项里存的已是改名后标准名，原判名需在记录上保留追踪）。

    question_text：题干原文（Judge 从 OCR 截取，B1 契约 §1.2a）；允许为空。

    链锚点冗余（textbook_id/grade/semester/unit_title/lesson_title/node_title）
    与 drill_stats/last_drill_result 初始化随库落（B1.5 §4.1，B2 免 join 聚合）。

    返回 record_id（= _id）。
    """
    now_ms = int(time.time() * 1000)
    record_id = _gen_record_id()
    record = {
        "record_id": record_id,
        "scholar_id": scholar_id,
        "attempt_ref": "",  # F4 扫描错题无 learning_attempt 关联
        "node_code": matched.get("node_code") or "",
        "knowledge_point_name": knowledge_point_name,
        "primary_error": error_type,
        "secondary_error": None,
        "stuck_step": None,
        "occurrence": 1,  # 首次扫描归类
        "created_at": now_ms,
        # §4.12.9(b) 扩展字段
        "scan_upload_id": scan_id,
        "classify_method": CLASSIFY_METHOD_AUTO_SCAN,
        "ocr_block_id": ocr_block_id,
        "source": SOURCE_AUTO_SCAN,
        "confidence": confidence,
        "question_text": question_text,
        # B1.5 §4.1：链锚点冗余（命中正式候选 / 图谱外 EXTRA_AI）
        "textbook_id": matched.get("textbook_id") or "",
        "grade": matched.get("grade") or "",
        "semester": matched.get("semester") or "",
        "unit_title": matched.get("unit_title") or "",
        "lesson_title": matched.get("lesson_title") or "",
        "node_title": matched.get("node_title") or matched.get("title")
        or matched.get("kp_name") or knowledge_point_name or "",
        # B1.5 §4.1：巩固证据初始化（B3 drill 回写）
        "drill_stats": {},
        "last_drill_result": {},
    }
    if original_kp_name:
        record["original_kp_name"] = original_kp_name  # B1.5a 就近改名追踪
    await db.insert(ERROR_RECORD_COLLECTION, record)
    return record_id


# ---------------------------------------------------------------------------
# F4.3 出参结构
# ---------------------------------------------------------------------------


def _to_public_classify(
    scan_id: str, status: str, items: list[dict[str, Any]]
) -> dict:
    """出参结构（契约 §3.10：scan_id / status / items[]）

    status ∈ {success, needs_review}；items[].error_record_id 为空表示
    该题未落 error_record（低置信/未定位，需人工修正）。
    """
    return {
        "scan_id": scan_id,
        "status": status,
        "items": items,
    }


# ---------------------------------------------------------------------------
# F4.3 主入口：classify_scan_upload
# ---------------------------------------------------------------------------


async def classify_scan_upload(
    db,
    *,
    scan_id: str,
    force_reclassify: bool = False,
    actor: str = "",
) -> dict:
    """扫描归类（F4.3 主流程，契约 §3.10 POST /math/scan/classify）

    前置校验：
    - scan_id 不存在 → ScanNotFoundError（404）
    - ocr_status ≠ success → OcrNotReadyError（404，契约 §3.10）
    - LLM_JUDGE_MODEL 未配置 → JudgeNotConfiguredError（500）

    幂等：classify_status ∈ {success, needs_review} 且非 force_reclassify
    → 直接返回已有结果（不重复调用 Judge，不重复写 error_record）。

    流程：
    1. 加载知识点候选集（F1 ai_summary，按学者教材过滤）
    2. 调用 LLM_JUDGE_MODEL 归类（题目识别 + 知识点定位 + 错因判定）
    3. 置信度门控：confidence >= 0.6 且知识点匹配 → 写 error_record；
       否则不写 error_record（needs_review）
    4. 更新 math_scan_upload.classify_status
    5. 写审计 scan_classify（成功/失败均落库）
    """
    if not scan_id:
        raise ScanNotFoundError("缺少 scan_id")

    # 1. 读取 scan 记录
    logger.info(f"[scan] classify 入口 scan_id={scan_id} force_reclassify={force_reclassify}")
    res = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION, where={"scan_id": scan_id}, limit=1
    )
    records = res.get("records") or []
    if not records:
        raise ScanNotFoundError(f"scan_id 不存在: {scan_id}")
    scan = records[0]

    # 2. OCR 完成性校验
    if scan.get("ocr_status") != OCR_STATUS_SUCCESS:
        detail = f"OCR 未完成（status={scan.get('ocr_status')}），无法归类"
        ocr_error = (scan.get("ocr_error") or "").strip()
        if ocr_error:
            detail += f"：{ocr_error[:200]}"
        logger.warning(
            f"[scan] classify 被 OCR 阻塞 scan_id={scan_id} "
            f"ocr_status={scan.get('ocr_status')} ocr_error={(ocr_error or '')[:200]} "
            f"classify_status={scan.get('classify_status')}"
        )
        raise OcrNotReadyError(detail)

    scholar_id = scan.get("scholar_id") or ""
    ocr_text = scan.get("ocr_text") or ""
    ocr_blocks = scan.get("ocr_blocks") or []

    # 3. 幂等：已归类且非强制重判 → 直接返回已有结果
    existing_status = scan.get("classify_status")
    if existing_status in (CLASSIFY_STATUS_SUCCESS, CLASSIFY_STATUS_NEEDS_REVIEW) and not force_reclassify:
        logger.info(
            f"[scan] 归类幂等命中 scan_id={scan_id} status={existing_status}"
        )
        existing_items = scan.get("classify_result") or []
        return _to_public_classify(
            scan_id,
            "success" if existing_status == CLASSIFY_STATUS_SUCCESS else "needs_review",
            existing_items,
        )

    # 3.5 处理中：classify_status=classifying → 返回 status=processing（前端轮询语义），
    # 避免前端轮询期间重复 POST 并发触发多个 Judge 调用（首次请求仍在后端执行）。
    if existing_status == CLASSIFY_STATUS_CLASSIFYING:
        logger.info(f"[scan] 归类处理中 scan_id={scan_id}，返回 processing")
        return _to_public_classify(scan_id, "processing", [])

    # 4. 标记 classifying（契约状态机：pending → classifying → success/failed/needs_review）
    await db.update(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        data={"$set": {"classify_status": CLASSIFY_STATUS_CLASSIFYING}},
    )
    logger.info(f"[scan] classify 标记 classifying scan_id={scan_id}，开始调 Judge")

    # 5. 加载知识点候选 + 调用 Judge
    try:
        candidates = await _load_knowledge_point_candidates(db, scholar_id)
        judge_result = await _call_classify_judge(ocr_text, candidates)
    except (JudgeNotConfiguredError, JudgeResponseError) as e:
        # Judge 不可用/解析失败 → classify_status=failed + 失败审计
        logger.warning(
            f"[scan] Judge 不可用，classify_status=failed scan_id={scan_id}: {type(e).__name__}"
        )
        await db.update(
            MATH_SCAN_UPLOAD_COLLECTION,
            where={"scan_id": scan_id},
            data={"$set": {"classify_status": CLASSIFY_STATUS_FAILED}},
        )
        await write_audit(
            db,
            action=AUDIT_ACTION_SCAN_CLASSIFY,
            object_ref=scan_id,
            actor=actor,
            result=AUDIT_RESULT_FAILED,
            context={
                "scholar_id": scholar_id,
                "reason": type(e).__name__,
                "candidates_count": 0,
            },
        )
        raise

    # 6. 置信度门控 + 写 error_record（三态分支，B1.5 契约 §1.2b）
    #    - 命中候选且 conf≥阈值且错因 → 直落 error_record（锚定正式教材节点）
    #    - 未命中候选但 conf≥阈值且错因且 kp 可命名 → EXTRA_AI 幂等新建/复用节点
    #      （图谱外自动归链，不再 needs_review 空置；修复报告「强行改挂」根因）
    #    - 其余（低置信/无错因/无 kp）→ needs_review，修正页提供新建/疑似改挂建议
    public_items: list[dict[str, Any]] = []
    all_confident = True
    for item in judge_result.get("items") or []:
        kp_name = (item.get("knowledge_point_name") or "").strip()
        error_type = item.get("error_type") or ""
        confidence = float(item.get("confidence") or 0)
        ocr_block_id = item.get("ocr_block_id") or ""
        question_text = item.get("question_text") or ""

        exact = _match_candidate(kp_name, candidates)
        matched = exact
        extra_ai = False
        final_kp = kp_name  # 落库/出参用名（就近改名时修正为候选标准名）
        renamed_from = ""   # B1.5a 就近改名原判名（仅近改名分支非空）
        if not exact and kp_name and confidence >= EVAL_CONFIDENCE_THRESHOLD and error_type:
            # B1.5a：候选外高置信先查语义就近（含操作符互斥护栏，口径同 correct 分支）
            # → 就近改名挂正式候选；无就近候选才 EXTRA_AI 幂等新建/复用（§4.3）。
            # 例：「小数乘法的实际应用」≈候选「小数乘除的实际应用」(Jaccard≥0.25)
            # 自动修正为标准名挂 formal，不再一律图谱外新建（评审 manual_correct
            # 「图谱外恰当性 0.8」修复）。
            near = _nearest_candidates(kp_name, candidates, limit=1)
            if near:
                matched = near[0]
                final_kp = near[0]["kp_name"]
                # 原判名保留追踪（对称 EXTRA_AI 分支的 new_kp_name）：
                # classify_result 项与 error_record 落库均为改名后标准名，
                # eval/前端需借此识别「自动就近改名」并核对改名目标是否贴切。
                renamed_from = kp_name
            else:
                matched = await _ensure_extra_ai_node(db, kp_name)
                extra_ai = True

        # 疑似正式教材改挂候选（仅供前端建议，不持久化；needs_review 修正页同用）
        candidate_hits: list[dict[str, Any]] = []
        if not exact:
            candidate_hits = _candidate_hits_struct(
                [n for n in (item.get("candidate_hits") or []) if n] or [kp_name],
                candidates,
            )

        if matched and confidence >= EVAL_CONFIDENCE_THRESHOLD and error_type:
            record_id = await _write_error_record(
                db,
                scan_id=scan_id,
                scholar_id=scholar_id,
                ocr_block_id=ocr_block_id,
                matched=matched,
                error_type=error_type,
                confidence=confidence,
                knowledge_point_name=final_kp,
                question_text=question_text,
                original_kp_name=renamed_from,
            )
            public_item: dict[str, Any] = {
                "error_record_id": record_id,
                "knowledge_point_name": final_kp,
                "error_type": error_type,
                "confidence": confidence,
                "ocr_block_id": ocr_block_id,
                "question_text": question_text,
            }
            if renamed_from:
                # 就近改名分支：original_kp_name 记录 Judge 原判名（对称 new_kp_name）
                public_item["original_kp_name"] = renamed_from
            if extra_ai:
                # 图谱外新建分支：new_kp_name 与 knowledge_point_name 同值（契约 §1.2b）
                public_item["new_kp_name"] = kp_name
            if candidate_hits:
                public_item["candidate_hits"] = candidate_hits
            public_items.append(public_item)
        else:
            # 低置信/无错因/无 kp → needs_review，不写 error_record
            all_confident = False
            review_item: dict[str, Any] = {
                "error_record_id": "",
                "knowledge_point_name": kp_name,
                "error_type": error_type,
                "confidence": confidence,
                "ocr_block_id": ocr_block_id,
                "question_text": question_text,
            }
            if kp_name:
                # 修正页「图谱外 → 新建知识点」预填名（与 knowledge_point_name 同值）
                review_item["new_kp_name"] = kp_name
            if candidate_hits:
                review_item["candidate_hits"] = candidate_hits
            public_items.append(review_item)

    # 7. 更新 math_scan_upload.classify_status + classify_result
    final_status = (
        CLASSIFY_STATUS_SUCCESS if all_confident and public_items else CLASSIFY_STATUS_NEEDS_REVIEW
    )
    await db.update(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        data={
            "$set": {
                "classify_status": final_status,
                "classify_result": public_items,
                "completed_at": int(time.time() * 1000),
            }
        },
    )
    logger.info(
        f"[scan] 归类落库 scan_id={scan_id} status={final_status} "
        f"items={len(public_items)} 全部高置信={all_confident} 候选集={len(candidates)}"
    )

    # 8. 写审计 scan_classify（必审）
    try:
        await write_audit(
            db,
            action=AUDIT_ACTION_SCAN_CLASSIFY,
            object_ref=scan_id,
            actor=actor,
            context={
                "scholar_id": scholar_id,
                "status": final_status,
                "items_count": len(public_items),
                "candidates_count": len(candidates),
                "force_reclassify": bool(force_reclassify),
            },
        )
    except Exception as e:
        logger.error(f"[scan] 归类审计写入失败 scan_id={scan_id}: {e}")

    return _to_public_classify(
        scan_id,
        "success" if final_status == CLASSIFY_STATUS_SUCCESS else "needs_review",
        public_items,
    )


# ===========================================================================
# F4.4 人工修正归类（契约 api-contract.md §3.10 POST /math/scan/{scan_id}/correct）
# ===========================================================================
#
# 链路：读取 scan → 遍历 items → 已有 error_record_id 更新归类
# （classify_method=manual_corrected），无 error_record_id 新建 error_record
# （classify_method=manual_corrected, source=manual_corrected）→ 更新
# classify_status=success（契约五态无 corrected，§0.4 以 success 代之）→ 写审计 scan_correct。
#
# 知识点 name → node_code 匹配复用 F4.3 候选集加载逻辑；
# 修正后的 error_record 作为后续 Judge few-shot 正样本来源（先留字段，不实现学习闭环）。


# ---------------------------------------------------------------------------
# F4.4 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class ScanCorrectError(Exception):
    """人工修正业务错误基类"""


class ScanCorrectValidationError(ScanCorrectError):
    """修正参数校验失败（路由层映射 400）"""


# ---------------------------------------------------------------------------
# F4.4 出参结构
# ---------------------------------------------------------------------------


def _to_public_correct(
    scan_id: str, corrected: list[dict[str, Any]]
) -> dict:
    """出参结构（契约 §3.10：{scan_id, corrected[{error_record_id, knowledge_point_name, error_type}]}）"""
    return {
        "scan_id": scan_id,
        "corrected": corrected,
    }


# ---------------------------------------------------------------------------
# F4.4 主入口：correct_scan_classify
# ---------------------------------------------------------------------------


def _validate_correct_items(items: list[dict[str, Any]]) -> None:
    """校验修正项列表（契约 §3.10 items[]）

    - items 非空（至少 1 项）
    - 每项 error_type 若提供则必须为四分类之一
    - 无 error_record_id 的新建项必须有 knowledge_point_name 或 error_type（否则无修正意义）
    """
    if not items:
        raise ScanCorrectValidationError("items 不能为空")
    for i, item in enumerate(items):
        et = item.get("error_type") or ""
        if et and et not in ERROR_TYPES:
            raise ScanCorrectValidationError(
                f"items[{i}].error_type 必须为 {list(ERROR_TYPES)} 之一，得到 {et!r}"
            )
        record_id = item.get("error_record_id") or ""
        kp_name = (item.get("knowledge_point_name") or "").strip()
        if not record_id and not kp_name and not et:
            raise ScanCorrectValidationError(
                f"items[{i}] 无 error_record_id 时须提供 knowledge_point_name 或 error_type"
            )


async def correct_scan_classify(
    db,
    *,
    scan_id: str,
    items: list[dict[str, Any]],
    actor: str = "",
) -> dict:
    """人工修正归类（F4.4 主流程，契约 §3.10 POST /math/scan/{scan_id}/correct）

    前置校验：
    - scan_id 不存在 → ScanNotFoundError（404）
    - items 为空 / 格式不合法 → ScanCorrectValidationError（400）

    流程：
    1. 读取 scan 记录
    2. 加载知识点候选集（用于 knowledge_point_name → node_code 匹配）
    3. 遍历 items：
       - 已有 error_record_id → 更新该 error_record（classify_method=manual_corrected，
         覆盖 knowledge_point_name/error_type/node_code）
       - 无 error_record_id → 新建 error_record（classify_method=manual_corrected,
         source=manual_corrected）
    4. 更新 math_scan_upload.classify_status=success（契约五态无 corrected，
       按 §0.4 以 success 代之）
    5. 写审计 scan_correct（必审）
    """
    if not scan_id:
        raise ScanNotFoundError("缺少 scan_id")

    _validate_correct_items(items)

    # 1. 读取 scan 记录
    res = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION, where={"scan_id": scan_id}, limit=1
    )
    records = res.get("records") or []
    if not records:
        raise ScanNotFoundError(f"scan_id 不存在: {scan_id}")
    scan = records[0]
    scholar_id = scan.get("scholar_id") or ""

    # 2. 加载知识点候选集（复用 F4.3 逻辑，用于 name → node_code 匹配）
    candidates = await _load_knowledge_point_candidates(db, scholar_id)

    # 3. 遍历 items：更新或新建 error_record
    corrected: list[dict[str, Any]] = []
    now_ms = int(time.time() * 1000)

    for item in items:
        record_id = item.get("error_record_id") or ""
        kp_name = (item.get("knowledge_point_name") or "").strip()
        # B1.5（契约 §1.3）：图谱外专用新建名；picker 未命中任何教材图谱时 LLM 预填
        new_kp_name = (item.get("new_kp_name") or "").strip()
        error_type = item.get("error_type") or ""
        raw_text_corrected = item.get("raw_text_corrected") or ""
        # B1（契约 §1.3）：题干统一落 question_text；raw_text_corrected 保留兼容字段。
        # 语义合并：question_text 优先，未传则回退 raw_text_corrected（老客户端向上兼容）。
        question_text = (item.get("question_text") or "").strip()
        merged_text = question_text or raw_text_corrected

        # ---- 锚点决策链（B1.5：图谱外新建 / 候选内精确或就近修正命名）----
        # 1) new_kp_name 非空 → 图谱外 EXTRA_AI 幂等新建/复用；
        # 2) kp_name 精确命中候选 → 锚定正式节点（标准名落库）；
        # 3) kp_name 就近命中（子串）→ 修正命名为候选标准名（报告「候选内修正命名」）；
        # 4) 其余 kp_name → 图谱外 EXTRA_AI（服务端兜底，老客户端无需 new_kp_name 也能归链）；
        # 5) 完全无名字（仅改错因）→ matched=None，不动 node_code/链锚点（update）或置空（新建）。
        matched: dict[str, Any] | None = None
        final_kp = ""
        if new_kp_name:
            matched = await _ensure_extra_ai_node(db, new_kp_name)
            final_kp = new_kp_name
        elif kp_name:
            exact = _match_candidate(kp_name, candidates)
            if exact:
                matched = exact
                final_kp = exact["kp_name"]
            else:
                near = _nearest_candidates(kp_name, candidates, limit=1)
                if near:
                    matched = near[0]
                    final_kp = near[0]["kp_name"]
                else:
                    matched = await _ensure_extra_ai_node(db, kp_name)
                    final_kp = kp_name
        anchor_fields = _chain_anchor_fields(matched)

        if record_id:
            # 已有 error_record_id → 更新归类（契约：classify_method=manual_corrected）
            update_data: dict[str, Any] = {
                "classify_method": CLASSIFY_METHOD_MANUAL_CORRECTED,
            }
            if final_kp:
                # 知识节点变更（含改挂 EXTRA_AI / 正式候选）→ 链锚点同步刷新
                update_data["node_code"] = anchor_fields["node_code"]
                update_data["knowledge_point_name"] = final_kp
                for k in ("textbook_id", "grade", "semester",
                          "unit_title", "lesson_title", "node_title"):
                    update_data[k] = anchor_fields[k]
            if error_type:
                update_data["primary_error"] = error_type
            if merged_text:
                update_data["question_text"] = merged_text
                # 兼容字段同步：老读取方读 raw_text_corrected 也能拿到最新题干
                update_data["raw_text_corrected"] = merged_text
            update_data["corrected_at"] = now_ms

            await db.update(
                ERROR_RECORD_COLLECTION,
                where={"record_id": record_id},
                data={"$set": update_data},
            )
            # 回读 node_code / primary_error 用于出参（回退到更新值或既有值）
            corrected.append(
                {
                    "error_record_id": record_id,
                    "knowledge_point_name": final_kp or kp_name,
                    "error_type": error_type,
                    "new_kp_name": new_kp_name or (
                        kp_name if matched and _is_extra_ai_anchor(matched) else ""
                    ),
                }
            )
        else:
            # 无 error_record_id → 新建 error_record
            # （契约：classify_method=manual_corrected, source=manual_corrected）
            new_id = _gen_record_id()
            record = {
                "record_id": new_id,
                "scholar_id": scholar_id,
                "attempt_ref": "",
                "node_code": anchor_fields.get("node_code") or "",
                "knowledge_point_name": final_kp or kp_name,
                "primary_error": error_type,
                "secondary_error": None,
                "stuck_step": None,
                "occurrence": 1,
                "created_at": now_ms,
                # §4.12.9(b) 扩展字段
                "scan_upload_id": scan_id,
                "classify_method": CLASSIFY_METHOD_MANUAL_CORRECTED,
                "ocr_block_id": "",
                "source": SOURCE_MANUAL_CORRECTED,
                "confidence": 1.0,  # 人工修正置信度固定 1.0
                "question_text": merged_text,
                "raw_text_corrected": merged_text,
                "corrected_at": now_ms,
                # B1.5 §4.1：链锚点冗余 + 巩固证据初始化
                "textbook_id": anchor_fields.get("textbook_id") or "",
                "grade": anchor_fields.get("grade") or "",
                "semester": anchor_fields.get("semester") or "",
                "unit_title": anchor_fields.get("unit_title") or "",
                "lesson_title": anchor_fields.get("lesson_title") or "",
                "node_title": anchor_fields.get("node_title") or "",
                "drill_stats": {},
                "last_drill_result": {},
            }
            await db.insert(ERROR_RECORD_COLLECTION, record)
            corrected.append(
                {
                    "error_record_id": new_id,
                    "knowledge_point_name": final_kp or kp_name,
                    "error_type": error_type,
                    "new_kp_name": new_kp_name or (
                        kp_name if matched and _is_extra_ai_anchor(matched) else ""
                    ),
                }
            )

    # 3.5 回写 classify_result（状态自洽）：correct 成功后 classify 幂等返回应能看到
    #     已落库/更新的 error_record_id；否则该题在 classify_result 中 error_record_id
    #     恒为空串，前端与重跑流程无法感知「已人工修正」。
    #     映射口径：classify_result 中 error_record_id 为空的槽位按序回填本次「新建」
    #     项的 id（corrected 与请求 items 同序，只取请求中无 error_record_id 的新建项；
    #     请求带 error_record_id 的 update 项不回填——槽位本就是直落题，id 已存在）。
    #     注：classify_result 无稳定逐题 id，若调用方未按空槽顺序提交则可能错位，
    #     MVP 接受该位置约定（B1 已落 question_text，后续可升级为题干锚定替代位置约定）。
    write_back: list[dict[str, Any]] = []
    existing_items = scan.get("classify_result") or []
    if existing_items and corrected:
        def _merged_qt(req: dict) -> str:
            qt = (req.get("question_text") or "").strip()
            return qt or (req.get("raw_text_corrected") or "")

        pending = [
            {
                "error_record_id": c.get("error_record_id", ""),
                "knowledge_point_name": c.get("knowledge_point_name") or "",
                "error_type": c.get("error_type") or "",
                "question_text": _merged_qt(req),
            }
            for c, req in zip(corrected, items)
            if not (req.get("error_record_id") or "")
        ]
        backfilled = False
        for it in existing_items:
            merged = dict(it)
            if not merged.get("error_record_id") and pending:
                nxt = pending.pop(0)
                merged["error_record_id"] = nxt["error_record_id"]
                if nxt["knowledge_point_name"]:
                    merged["knowledge_point_name"] = nxt["knowledge_point_name"]
                if nxt["error_type"]:
                    merged["error_type"] = nxt["error_type"]
                if nxt["question_text"]:
                    merged["question_text"] = nxt["question_text"]
                backfilled = True
            write_back.append(merged)

    # 4. 更新 classify_status=success（契约五态无 corrected，§0.4 以 success 代之）
    set_fields: dict[str, Any] = {
        "classify_status": CLASSIFY_STATUS_SUCCESS,
        "completed_at": now_ms,
    }
    if write_back and backfilled:
        set_fields["classify_result"] = write_back
    await db.update(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        data={"$set": set_fields},
    )

    # 5. 写审计 scan_correct（必审）
    try:
        await write_audit(
            db,
            action=AUDIT_ACTION_SCAN_CORRECT,
            object_ref=scan_id,
            actor=actor,
            context={
                "scholar_id": scholar_id,
                "items_count": len(corrected),
                "updated_count": sum(
                    1 for i in items if i.get("error_record_id")
                ),
                "created_count": sum(
                    1 for i in items if not i.get("error_record_id")
                ),
            },
        )
    except Exception as e:
        logger.error(f"[scan] 修正审计写入失败 scan_id={scan_id}: {e}")

    return _to_public_correct(scan_id, corrected)
