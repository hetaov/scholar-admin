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
    LLM_JUDGE_MODEL,
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
    """获取 Judge 模型客户端（单次调用最长 60 秒；单例）"""
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI

        _judge_client = OpenAI(
            api_key=VOLCANO_API_KEY,
            base_url=VOLCANO_BASE_URL,
            timeout=60.0,
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

    返回 [{node_id, node_code, kp_name, grade, textbook_id}]。
    """
    _SELECT = {
        "node_id": 1,
        "code": 1,
        "grade": 1,
        "textbook_id": 1,
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
                    "textbook_id": node.get("textbook_id") or "",
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
2. knowledge_point_name：从候选集中选最匹配的知识点 name；无法定位时填空字符串 ""。
3. error_type：错因四分类，取值 concept（概念错）/ method（方法错）/ computation（计算错）/ reading（审题错）。
4. confidence：本道题归类置信度 0~1（知识点无法定位或错因不明时 ≤0.5）。
5. ocr_block_id：本道题在 OCR 检测块中对应的 block_id（取题干所在块，无则填空字符串 ""）。

输出 JSON 结构：
{{
  "items": [
    {{"knowledge_point_name": "...", "error_type": "concept|method|computation|reading", "confidence": 0.85, "ocr_block_id": "blk_0001"}}
  ]
}}"""


def _format_candidates(candidates: list[dict[str, Any]], limit: int = 80) -> str:
    """格式化候选集为 prompt 文本（限制条目数控制 token）"""
    if not candidates:
        return "（候选集为空，请尽量按 OCR 内容推断知识点并降低 confidence）"
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
    """同步 chat 调用（在线程池中执行）"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
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

    prompt = _CLASSIFY_USER_TEMPLATE.format(
        ocr_text=ocr_text or "（OCR 文本为空）",
        candidates=_format_candidates(candidates),
    )
    client = _get_judge_client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_judge_sync, client, LLM_JUDGE_MODEL,
                _CLASSIFY_SYSTEM_PROMPT, prompt,
            )
            result = _parse_json_response(response)
            _validate_classify_result(result)
            return result
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 JudgeResponseError
            last_err = e
            logger.warning(
                f"[scan] 归类 Judge 调用失败（第 {attempt} 次）: "
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
) -> str:
    """写一条 error_record（契约 §4.12.2 + §4.12.9(b) 扩展字段）

    knowledge_point_name：识别命中/人工修正时的知识点名（冗余落库，
    供错题列表直接展示，免 join；与 node_code 并存）。

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
    }
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

    # 4. 标记 classifying（契约状态机：pending → classifying → success/failed/needs_review）
    await db.update(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        data={"$set": {"classify_status": CLASSIFY_STATUS_CLASSIFYING}},
    )

    # 5. 加载知识点候选 + 调用 Judge
    try:
        candidates = await _load_knowledge_point_candidates(db, scholar_id)
        judge_result = await _call_classify_judge(ocr_text, candidates)
    except (JudgeNotConfiguredError, JudgeResponseError) as e:
        # Judge 不可用/解析失败 → classify_status=failed + 失败审计
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

    # 6. 置信度门控 + 写 error_record（仅高置信 + 知识点匹配项）
    public_items: list[dict[str, Any]] = []
    all_confident = True
    for item in judge_result.get("items") or []:
        kp_name = (item.get("knowledge_point_name") or "").strip()
        error_type = item.get("error_type") or ""
        confidence = float(item.get("confidence") or 0)
        ocr_block_id = item.get("ocr_block_id") or ""

        matched = _match_candidate(kp_name, candidates)
        if matched and confidence >= EVAL_CONFIDENCE_THRESHOLD and error_type:
            record_id = await _write_error_record(
                db,
                scan_id=scan_id,
                scholar_id=scholar_id,
                ocr_block_id=ocr_block_id,
                matched=matched,
                error_type=error_type,
                confidence=confidence,
                knowledge_point_name=kp_name,
            )
            public_items.append(
                {
                    "error_record_id": record_id,
                    "knowledge_point_name": kp_name,
                    "error_type": error_type,
                    "confidence": confidence,
                    "ocr_block_id": ocr_block_id,
                }
            )
        else:
            # 低置信/知识点未匹配/无错因 → needs_review，不写 error_record
            all_confident = False
            public_items.append(
                {
                    "error_record_id": "",
                    "knowledge_point_name": kp_name,
                    "error_type": error_type,
                    "confidence": confidence,
                    "ocr_block_id": ocr_block_id,
                }
            )

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
        error_type = item.get("error_type") or ""
        raw_text_corrected = item.get("raw_text_corrected") or ""

        # name → node_code 匹配（匹配不到时 node_code 留空，不阻断人工修正）
        matched = _match_candidate(kp_name, candidates) if kp_name else None
        node_code = matched.get("node_code") or "" if matched else ""

        if record_id:
            # 已有 error_record_id → 更新归类（契约：classify_method=manual_corrected）
            update_data: dict[str, Any] = {
                "classify_method": CLASSIFY_METHOD_MANUAL_CORRECTED,
            }
            if kp_name:
                update_data["node_code"] = node_code
                update_data["knowledge_point_name"] = kp_name
            if error_type:
                update_data["primary_error"] = error_type
            if raw_text_corrected:
                update_data["raw_text_corrected"] = raw_text_corrected
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
                    "knowledge_point_name": kp_name,
                    "error_type": error_type,
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
                "node_code": node_code,
                "knowledge_point_name": kp_name,
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
                "raw_text_corrected": raw_text_corrected,
                "corrected_at": now_ms,
            }
            await db.insert(ERROR_RECORD_COLLECTION, record)
            corrected.append(
                {
                    "error_record_id": new_id,
                    "knowledge_point_name": kp_name,
                    "error_type": error_type,
                }
            )

    # 4. 更新 classify_status=success（契约五态无 corrected，§0.4 以 success 代之）
    await db.update(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        data={
            "$set": {
                "classify_status": CLASSIFY_STATUS_SUCCESS,
                "completed_at": now_ms,
            }
        },
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
