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
    EXTRA_AI_TEXTBOOK_ID,
    EXTRA_AI_UNCLASSIFIED,
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


async def _call_judge_with_meta(
    ocr_text: str, candidates: list[dict[str, Any]]
) -> tuple[dict, dict[str, Any]]:
    """干跑专用 Judge 包装：调用生产 _call_classify_judge + 记录 meta 信息。

    返回 (judge_result, meta)，meta 含 prompt_chars / attempts。
    attempts 记录实际调用次数（_call_classify_judge 内部有重试，但无法从外部
    获知；此处用 1 表示成功返回，如果抛 JudgeResponseError 则说明重试后仍失败）。
    """
    from services.math.error_scanner import _CLASSIFY_USER_TEMPLATE, _format_candidates

    # 计算 prompt_chars（与生产 _call_classify_judge 内部构造一致）
    truncated_ocr = (ocr_text or "")[:LLM_JUDGE_OCR_TEXT_MAX]
    prompt = _CLASSIFY_USER_TEMPLATE.format(
        ocr_text=truncated_ocr or "（OCR 文本为空）",
        candidates=_format_candidates(candidates),
    )
    prompt_chars = len(prompt)

    attempts = 1
    try:
        result = await _call_classify_judge(ocr_text, candidates)
        return result, {"prompt_chars": prompt_chars, "attempts": attempts}
    except JudgeResponseError:
        # _call_classify_judge 内部已重试 1 次仍失败 → attempts=2
        attempts = 2
        raise



async def _assemble_dry_run_items(
    judge_result: dict,
    candidates: list[dict[str, Any]],
    textbook_id: str,
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """组装干跑 items[] + decisions[]（与生产 classify_scan_upload 对齐）。

    与生产差异：
    - **不调 _ensure_extra_ai_node**（生产会建 EXTRA_AI 节点）；干跑用纯函数
      _extra_ai_node_code 生成预演 node_code（xai_ 前缀），但**不落库**。
    - **不调 _write_error_record**；error_record_id 恒为空串（不落库）。
    - decisions[] 追加 match_type 四值 + would_write/would_create 预演标志，
      让管理台直观看到「生产会怎样落库」而不真的落。

    Returns:
        (items, decisions) —— items 与生产 _to_public_classify 字段对齐；
        decisions 每项含 judge_* 原始字段 + match_type + 门控结果 + 预演标志。
    """
    from services.math.error_scanner import (
        EXTRA_AI_NODE_CODE_PREFIX,
        _candidate_hits_struct,
        _chain_anchor_fields,
        _extra_ai_node_code,
        _is_extra_ai_anchor,
    )

    items: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for idx, item in enumerate(judge_result.get("items") or []):
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
        match_type = "none"
        preview_node_code = ""

        if exact:
            match_type = "exact"
        elif kp_name and confidence >= confidence_threshold and error_type:
            near = _nearest_candidates(kp_name, candidates, limit=1)
            if near:
                matched = near[0]
                final_kp = near[0]["kp_name"]
                renamed_from = kp_name
                match_type = "renamed"
            else:
                # 干跑：纯函数预演 EXTRA_AI node_code，不调 _ensure_extra_ai_node
                extra_ai = True
                preview_node_code = _extra_ai_node_code(kp_name)
                matched = {
                    "node_code": preview_node_code,
                    "kp_name": kp_name,
                    "title": "未分类",
                    "textbook_id": EXTRA_AI_TEXTBOOK_ID,
                    "grade": "",
                    "semester": "",
                    "unit_title": EXTRA_AI_UNCLASSIFIED,
                    "lesson_title": EXTRA_AI_UNCLASSIFIED,
                }
                match_type = "extra_ai"

        # 疑似正式教材改挂候选（与生产同构，不持久化）
        candidate_hits: list[dict[str, Any]] = []
        if not exact:
            candidate_hits = _candidate_hits_struct(
                [n for n in (item.get("candidate_hits") or []) if n] or [kp_name],
                candidates,
            )

        # 门控判定（与生产口径一致）
        passed_gate = bool(matched) and confidence >= confidence_threshold and bool(error_type)

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

        # 组装 decision（debug.decisions[] 每项）
        chain_anchor = _chain_anchor_fields(matched) if matched else {}
        decisions.append({
            "index": idx,
            "judge_kp_name": kp_name,
            "judge_error_type": error_type,
            "judge_confidence": confidence,
            "judge_question_text": question_text,
            "judge_ocr_block_id": ocr_block_id,
            "match_type": match_type,
            "matched_kp_name": matched.get("kp_name") or "" if matched else "",
            "matched_node_code": matched.get("node_code") or "" if matched else "",
            "confidence": confidence,
            "threshold": confidence_threshold,
            "passed_gate": passed_gate,
            "would_write_error_record": passed_gate,
            "would_create_extra_ai_node": extra_ai,
            "preview_node_code": preview_node_code,
            "chain_anchor": chain_anchor,
            "exam_backlink_to": None,  # M13 双源知识锚，干跑不计算
            "candidate_hits": candidate_hits,
        })

    return items, decisions


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
    """B04~B06 干跑入口（api-contract §3.15 POST /math/scan/debug/recognize）。

    一次调用完成：校验 → OCR → 加载候选 → Judge → 组装 items[] + decisions[]。
    **不写任何集合、不传云存储、不写审计**；OCR 与 Judge 为真实付费调用。

    compare_all_candidates=true 时再跑一次全量候选 Judge（**双倍 Judge 消耗**，
    风险 R-5），产出 compare.baseline/all/diff_summary。
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

    # 5. Judge（真实付费调用，B06 用 _call_judge_with_meta 记录 prompt_chars/attempts）
    t_judge_start = time.time()
    judge_result, judge_meta = await _call_judge_with_meta(ocr_text, candidates)
    judge_ms = int((time.time() - t_judge_start) * 1000)

    # 6. 组装 items[] + decisions[]（B04 主干 + B05 门控四分支）
    items, decisions = await _assemble_dry_run_items(
        judge_result, candidates, textbook_id, threshold
    )

    # 7. 顶层 status：任一 decision 未通过门控 → needs_review
    all_passed = all(d["passed_gate"] for d in decisions) if decisions else True
    top_status = "success" if all_passed else "needs_review"

    # 8. compare 段（B06：compare_all_candidates=true 时再跑全量候选 Judge）
    compare_data: dict[str, Any] | None = None
    if compare_all_candidates:
        logger.info("[scan][debug] compare_all_candidates=true，启动全量候选对照 Judge（双倍消耗）")
        # 加载全量候选（textbook_id="" 等价全量）
        all_cand_result = await _load_knowledge_point_candidates_by_textbook(
            db, "", scholar_id
        )
        all_candidates = all_cand_result["candidates"]

        t_compare_start = time.time()
        compare_judge_result, compare_judge_meta = await _call_judge_with_meta(
            ocr_text, all_candidates
        )
        compare_judge_ms = int((time.time() - t_compare_start) * 1000)

        compare_items, _ = await _assemble_dry_run_items(
            compare_judge_result, all_candidates, textbook_id, threshold
        )

        # diff_summary：对比 baseline（textbook_id 过滤）与 all（全量）的 kp 差异
        baseline_kps = {it["knowledge_point_name"] for it in items}
        all_kps = {it["knowledge_point_name"] for it in compare_items}
        only_in_baseline = sorted(baseline_kps - all_kps)
        only_in_all = sorted(all_kps - baseline_kps)
        # kp_changed：同题号但 kp 名变化的（按 index 对齐）
        kp_changed: list[dict[str, Any]] = []
        max_idx = min(len(items), len(compare_items))
        for i in range(max_idx):
            b_kp = items[i].get("knowledge_point_name", "")
            a_kp = compare_items[i].get("knowledge_point_name", "")
            if b_kp and a_kp and b_kp != a_kp:
                kp_changed.append({"index": i, "baseline_kp": b_kp, "all_kp": a_kp})

        compare_data = {
            "baseline": {
                "candidates_count": cand_result["count"],
                "items_count": len(items),
                "judge_ms": judge_ms,
            },
            "all": {
                "candidates_count": all_cand_result["count"],
                "items_count": len(compare_items),
                "judge_ms": compare_judge_ms,
                "items": compare_items,
            },
            "diff_summary": {
                "only_in_baseline": only_in_baseline,
                "only_in_all": only_in_all,
                "kp_changed": kp_changed,
            },
        }

    # 9. 组装响应（B04 基础 + B05 decisions + B06 judge meta + compare）
    data = {
        "scan_id": scan_id,
        "status": top_status,
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
                "total_ms": ocr_ms + judge_ms + (compare_data["all"]["judge_ms"] if compare_data else 0),
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
                "prompt_chars": judge_meta["prompt_chars"],  # B06 填充
                "ocr_text_max": LLM_JUDGE_OCR_TEXT_MAX,
                "candidate_limit": cand_result["prompt_included"],
                "attempts": judge_meta["attempts"],  # B06 填充
                "disable_thinking": LLM_JUDGE_DISABLE_THINKING,
            },
            "decisions": decisions,  # B05 门控四分支
            "compare": compare_data,  # B06 填充（null 或 compare 结构）
        },
    }
    return data
