"""数学错题识别 Admin 调试干跑（api-contract §3.15 · 不落库）

管理台调试专用，与生产错题扫描链路（services/math/error_scanner.py）严格隔离：
- **不写任何集合**（math_scan_upload / error_record / curriculum_node / textbook / audit_log）
- **不传云存储**（不调 CloudBaseStorageClient）
- **不写审计**（不调 services.audit）
- OCR 与 Judge 为**真实付费调用**（会产生费用）

本模块只读复用 error_scanner.py 的纯函数（_validate_image / _match_candidate /
_nearest_candidates / _format_candidates / _call_classify_judge 等），不修改其代码。

设计文档：docs_v1/AI错题/数学AI错题识别-设计文档.md
实施拆分：docs_v1/AI错题/数学AI错题识别-任务拆分与断点.md（B03~B06 步）
"""
from __future__ import annotations

import logging
import time
from typing import Any

from services.database import CURRICULUM_NODE_COLLECTION
from services.math.error_scanner import (
    EVAL_CONFIDENCE_THRESHOLD,
    ImageTooLargeError,
    ImageValidationError,
    JudgeNotConfiguredError,
    JudgeResponseError,
    LLM_JUDGE_DISABLE_THINKING,
    LLM_JUDGE_MODEL,
    LLM_JUDGE_OCR_TEXT_MAX,
    _call_classify_judge,
    _detect_ext,
    _gen_record_id,
    _match_candidate,
    _nearest_candidates,
    _validate_classify_result,
    _validate_image,
)
from services.math.error_scanner import _SCHOLAR_BOOK_COLLECTION  # noqa: F401 — 仅供未来 dry_run 内省引用

logger = logging.getLogger("scholar-admin.math.error_scan_debug")

# 候选项 select 口径与生产 _load_knowledge_point_candidates 完全一致
_KP_SELECT = {
    "node_id": 1,
    "code": 1,
    "grade": 1,
    "semester": 1,
    "textbook_id": 1,
    "title": 1,
    "unit_title": 1,
    "lesson_title": 1,
    "ai_summary": 1,
}


async def _load_knowledge_point_candidates_by_textbook(
    db: Any,
    textbook_id: str,
    scholar_id: str = "",
) -> dict[str, Any]:
    """干跑专用：按 textbook_id 直接加载知识点候选集（只读）。

    与生产 ``_load_knowledge_point_candidates(db, scholar_id)`` 的差异：
    - **直接按 textbook_id 过滤**，不走 scholar_book → textbook_id 间接路径；
    - textbook_id 为空时**回退全量候选**（与生产回退全量口径一致）；
    - **不读 scholar_book 集合**（干跑不依赖学者绑定关系）；
    - 返回结构增加 ``source`` / ``count`` / ``prompt_included`` / ``truncated``
      用于 debug.candidates 段（B06 步组装）。

    Returns:
        {
            "candidates": [{node_id, node_code, kp_name, grade, semester,
                             textbook_id, title, unit_title, lesson_title}],
            "source": "textbook_id" | "all",
            "count": int,
            "prompt_included": int,
            "truncated": bool,
        }
    """
    from config import LLM_JUDGE_CANDIDATE_LIMIT

    nodes: list[dict] = []
    where: dict[str, Any] = {}
    source = "all"

    if textbook_id:
        where = {"textbook_id": textbook_id}
        source = "textbook_id"

    for offset in range(0, 2000, 500):
        res = await db.query(
            CURRICULUM_NODE_COLLECTION,
            where=where,
            select=_KP_SELECT,
            offset=offset,
            limit=500,
        )
        batch = res.get("records") or []
        nodes.extend(batch)
        if len(batch) < 500:
            break

    # textbook_id 过滤无结果时回退全量（与生产口径一致）
    if not nodes and textbook_id:
        source = "all"
        for offset in range(0, 2000, 500):
            res = await db.query(
                CURRICULUM_NODE_COLLECTION,
                select=_KP_SELECT,
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
                    "unit_title": node.get("unit_title") or "",
                    "lesson_title": node.get("lesson_title") or "",
                }
            )

    prompt_included = min(len(candidates), LLM_JUDGE_CANDIDATE_LIMIT)
    return {
        "candidates": candidates,
        "source": source,
        "count": len(candidates),
        "prompt_included": prompt_included,
        "truncated": len(candidates) > LLM_JUDGE_CANDIDATE_LIMIT,
    }


# ---------------------------------------------------------------------------
# B04：干跑主干 —— _validate_image → OCR → Judge → 组装 items[]
# ---------------------------------------------------------------------------


def _gen_debug_scan_id() -> str:
    """干跑专用 scan_id：debug_{毫秒时间戳}_{随机hex8}。

    与生产 _gen_scan_id（scan_ 前缀）区分，日志/审计追溯时可立即识别为干跑。
    """
    import secrets

    return f"debug_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


async def _run_ocr_for_debug(image_bytes: bytes) -> dict[str, Any]:
    """干跑 OCR：调用真实 provider，返回 debug.ocr 段所需的元信息 + text/blocks。

    **真实付费调用**（不落库、不传云存储）。
    """
    from services.math import ocr as ocr_mod
    from services.math.ocr import OcrError

    provider = ocr_mod.get_provider()
    provider_name = type(provider).__name__
    available = provider.available
    engine = getattr(provider, "_engine", "unknown")

    if not available:
        return {
            "provider": provider_name,
            "engine": engine,
            "available": False,
            "blocks_count": 0,
            "text_length": 0,
            "text_truncated": False,
            "text": "",
            "blocks": [],
            "error": "OCR 未配置凭据",
        }

    t0 = time.time()
    try:
        result = await provider.recognize(image_bytes)
        ocr_ms = int((time.time() - t0) * 1000)
        text = result.text or ""
        truncated = len(text) > LLM_JUDGE_OCR_TEXT_MAX
        return {
            "provider": provider_name,
            "engine": engine,
            "available": True,
            "blocks_count": len(result.blocks or []),
            "text_length": len(text),
            "text_truncated": truncated,
            "text": text,
            "blocks": result.blocks or [],
            "ocr_ms": ocr_ms,
        }
    except OcrError as e:
        raise OcrError(f"干跑 OCR 失败: {e}") from e


async def _assemble_dry_run_items(
    judge_result: dict,
    candidates: list[dict[str, Any]],
    textbook_id: str,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """组装干跑 items[]（与生产 classify_scan_upload 的 items[] 结构对齐）。

    与生产差异：
    - **不调 _ensure_extra_ai_node**（生产会建 EXTRA_AI 节点）；干跑用纯函数
      _extra_ai_node_code 生成预演 node_code（xai_ 前缀），但**不落库**。
    - **不调 _write_error_record**；error_record_id 恒为空串（不落库）。
    - match_type 四值：exact / renamed / extra_ai / none（B05 门控组装用）。
    """
    from services.math.error_scanner import (
        EXTRA_AI_NODE_CODE_PREFIX,
        _candidate_hits_struct,
        _chain_anchor_fields,
        _extra_ai_node_code,
        _is_extra_ai_anchor,
    )

    items: list[dict[str, Any]] = []
    for item in judge_result.get("items") or []:
        kp_name = (item.get("knowledge_point_name") or "").strip()
        error_type = item.get("error_type") or ""
        confidence = float(item.get("confidence") or 0)
        ocr_block_id = item.get("ocr_block_id") or ""
        question_text = item.get("question_text") or ""

        exact = _match_candidate(kp_name, candidates)
        matched = exact
        extra_ai = False
        final_kp = kp_name
        renamed_from = ""

        if not exact and kp_name and confidence >= confidence_threshold and error_type:
            near = _nearest_candidates(kp_name, candidates, limit=1)
            if near:
                matched = near[0]
                final_kp = near[0]["kp_name"]
                renamed_from = kp_name
            else:
                # 干跑：纯函数预演 EXTRA_AI node_code，不调 _ensure_extra_ai_node
                extra_ai = True
                matched = {
                    "node_code": _extra_ai_node_code(kp_name),
                    "kp_name": kp_name,
                    "title": "未分类",
                    "textbook_id": EXTRA_AI_NODE_CODE_PREFIX.rstrip("_"),
                    "grade": "",
                    "semester": "",
                    "unit_title": "未分类",
                    "lesson_title": "未分类",
                }

        # 疑似正式教材改挂候选（与生产同构，不持久化）
        candidate_hits: list[dict[str, Any]] = []
        if not exact:
            candidate_hits = _candidate_hits_struct(
                [n for n in (item.get("candidate_hits") or []) if n] or [kp_name],
                candidates,
            )

        # 组装 item（与生产 _to_public_classify 字段对齐）
        item_out: dict[str, Any] = {
            "error_record_id": "",  # 干跑不落库，恒为空串
            "knowledge_point_name": final_kp,
            "error_type": error_type,
            "confidence": confidence,
            "ocr_block_id": ocr_block_id,
            "question_text": question_text,
            "textbook_id": textbook_id or (matched.get("textbook_id") or "") if matched else textbook_id,
            "candidate_hits": candidate_hits,
        }

        # renamed 分支追加 original_kp_name
        if renamed_from:
            item_out["original_kp_name"] = renamed_from

        # extra_ai 分支追加 new_kp_name
        if extra_ai:
            item_out["new_kp_name"] = kp_name

        items.append(item_out)

    return items


async def recognize_error_scan_dry_run(
    db: Any,
    *,
    image_bytes: bytes,
    filename: str,
    textbook_id: str = "",
    scholar_id: str = "scholar_debug_01",
    confidence_threshold: float | None = None,
    compare_all_candidates: bool = False,
    include_ocr_text: bool = True,
) -> dict[str, Any]:
    """B04 干跑主干入口（api-contract §3.15 POST /math/scan/debug/recognize）。

    一次调用完成：校验 → OCR → 加载候选 → Judge → 组装 items[]。
    **不写任何集合、不传云存储、不写审计**；OCR 与 Judge 为真实付费调用。

    B04 仅实现主干（items[] + 基础 debug）；B05 追加门控四分支 + decisions[]；
    B06 追加 debug 其余段 + compare 段。
    """
    from services.math.ocr import OcrError

    # 0. 置信度阈值（默认用全局，可覆盖）
    threshold = (
        confidence_threshold if confidence_threshold is not None else EVAL_CONFIDENCE_THRESHOLD
    )

    # 1. 校验图片（复用生产 _validate_image 纯函数）
    _validate_image(filename, image_bytes)
    ext = _detect_ext(filename)

    # 2. 生成 debug scan_id
    scan_id = _gen_debug_scan_id()

    # 3. OCR（真实付费调用）
    ocr_info = await _run_ocr_for_debug(image_bytes)
    if not ocr_info.get("available"):
        raise OcrError(ocr_info.get("error", "OCR 不可用"))
    ocr_text = ocr_info["text"]
    ocr_ms = ocr_info.get("ocr_ms", 0)

    # 4. 加载候选集（B03 只读加载）
    cand_result = await _load_knowledge_point_candidates_by_textbook(
        db, textbook_id, scholar_id
    )
    candidates = cand_result["candidates"]

    # 5. Judge（真实付费调用）
    t_judge_start = time.time()
    judge_result = await _call_classify_judge(ocr_text, candidates)
    judge_ms = int((time.time() - t_judge_start) * 1000)

    # 6. 组装 items[]（B04 主干）
    items = await _assemble_dry_run_items(
        judge_result, candidates, textbook_id, threshold
    )

    # 7. 组装响应（B04 基础版，B05/B06 会补充 debug 段）
    data = {
        "scan_id": scan_id,
        "status": "success",
        "items": items,
        "debug": {
            "dry_run": True,
            "persisted": False,
            "request": {
                "textbook_id": textbook_id,
                "scholar_id": scholar_id,
                "confidence_threshold": threshold,
                "compare_all_candidates": compare_all_candidates,
                "image_bytes": len(image_bytes),
                "image_ext": ext,
            },
            "timings": {
                "total_ms": ocr_ms + judge_ms,
                "ocr_ms": ocr_ms,
                "judge_ms": judge_ms,
            },
            "candidates": {
                "textbook_id": textbook_id,
                "source": cand_result["source"],
                "count": cand_result["count"],
                "prompt_included": cand_result["prompt_included"],
                "truncated": cand_result["truncated"],
            },
            "ocr": {
                "provider": ocr_info["provider"],
                "engine": ocr_info["engine"],
                "available": ocr_info["available"],
                "blocks_count": ocr_info["blocks_count"],
                "text_length": ocr_info["text_length"],
                "text_truncated": ocr_info["text_truncated"],
                "text": ocr_text if include_ocr_text else "",
            },
            "judge": {
                "model": LLM_JUDGE_MODEL,
                "prompt_chars": 0,  # B06 填充
                "ocr_text_max": LLM_JUDGE_OCR_TEXT_MAX,
                "candidate_limit": cand_result["prompt_included"],
                "attempts": 1,  # B06 填充
                "disable_thinking": LLM_JUDGE_DISABLE_THINKING,
            },
            "decisions": [],  # B05 填充
            "compare": None,  # B06 填充
        },
    }
    return data
