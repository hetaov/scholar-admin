"""数学同类型题「拍照识别」服务（quiz recognize · 接口 28）

契约：
- api-contract.md §3.10 接口 28 `POST /math/scan/quiz-recognize`
- data-model-contract.md §4.12.9 `math_scan_upload`（只读）、§4.12.11 `math_quiz`（本模块不写）
- service-contract.md §7.11 `recognize_quiz_scan`
- ADR-0035 D2（拍照语义隔离）/ §2.6（后端无「单题识别模式」，故全量返回由前端多选）

**红线 R1 —— 本模块零写库**：
不写错题记录集合、不回写 `math_scan_upload` 的归类状态 / 结果字段、不新建 `curriculum_node`
（含 EXTRA_AI）。未命中候选 → `knowledge_point_name: ""` / `node_code: ""`，**不报错、不建节点**。

与 `services/math/error_scanner.py` 的关系：**只读复用、零改动** —— 复用其候选集加载 /
格式化 / Judge 同步调用与 OCR 完成性校验口径；**刻意跳过**错题链路的写记录函数以及
全部针对上传集合的 `db.update(...)` 回写分支（本模块只读，不产生任何 `db.insert` / `db.update`）。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from config import (
    LLM_JUDGE_MODEL,
    LLM_JUDGE_OCR_TEXT_MAX,
    VOLCANO_API_KEY,
)
from services.audit import AUDIT_RESULT_FAILED, AUDIT_RESULT_SUCCESS, write_audit
from services.database import MATH_SCAN_UPLOAD_COLLECTION
from services.math.error_scanner import (
    OCR_STATUS_SUCCESS,
    JudgeNotConfiguredError,
    JudgeResponseError,
    OcrNotReadyError,
    ScanNotFoundError,
    _call_judge_sync,
    _format_candidates,
    _get_judge_client,
    _load_knowledge_point_candidates,
    _parse_photo_context,
    _photo_context_anchor,
)

try:  # B13 之后由 services.metrics.audit 提供常量；在此之前用同名字面量，口径一致
    from services.metrics.audit import AUDIT_ACTION_QUIZ_RECOGNIZE
except ImportError:  # pragma: no cover — B13 实施后自动切到正式常量
    AUDIT_ACTION_QUIZ_RECOGNIZE = "quiz_recognize"

logger = logging.getLogger("scholar-admin.math.quiz_recognizer")

# ---------------------------------------------------------------------------
# Judge Prompt（只做「题目切分 + 知识点定位」，去掉错因四分类）
# ---------------------------------------------------------------------------

_QUIZ_RECOGNIZE_SYSTEM_PROMPT = (
    "你是小学数学题目识别助手。根据给定的 OCR 文本与知识点候选集，"
    "把文本切分成一道道独立的题目，并为每道题定位最匹配的知识点。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_QUIZ_RECOGNIZE_USER_TEMPLATE = """请把以下 OCR 文本切分成一道道独立的题目，并为每道题定位知识点。

OCR 全文：
{ocr_text}

知识点候选集（name 即知识点名，grade 为年级，node_code 为知识点编码）：
{candidates}

要求：
1. 按题号（1. 2. 3. 或（1）（2）等）或独立题干切分，每道题输出一项；会过滤卷头、姓名栏、分数栏、页码等非题目文本。
2. knowledge_point_name：只从候选集中选择最匹配的知识点的 name；候选集为空或无匹配项时输出空字符串 ""，不要自行发明知识点名。
3. node_code：所选知识点在候选集中对应的 node_code；第 2 项为空字符串时同样输出 ""。
4. question_text：从上方 OCR 全文**原文截取**本道题的完整题干（含条件与问题，逐字保留、不改写、不补全）；无法截取时输出 ""。
5. confidence：0~1，按「题干是否完整清晰 + 知识点是否明确」给分；题干残缺 / 语义模糊 → <0.6。

输出 JSON 结构：
{{
  "items": [
    {{"knowledge_point_name": "小数加减法", "node_code": "5-3-1-kp2", "question_text": "0.36 + 2.7 = ?", "confidence": 0.88}}
  ]
}}"""


def _validate_quiz_recognize_result(result: dict) -> None:
    """校验 Judge 输出结构（接口 28 items[]）；非法抛 JudgeResponseError。

    与 `_validate_classify_result` 的差别：无 `error_type` / `candidate_hits`，
    且**不校验条数、不截断**（一张卷子上的题要全部返回，筛选由前端多选完成，D9）。
    """
    if not isinstance(result, dict):
        raise JudgeResponseError("Judge 输出必须为 JSON 对象")
    items = result.get("items")
    if not isinstance(items, list):
        raise JudgeResponseError("Judge 输出 items 必须为数组")
    for item in items:
        if not isinstance(item, dict):
            raise JudgeResponseError("items 每项必须为对象")
        conf = item.get("confidence", 0)
        try:
            conf_val = float(conf)
        except (TypeError, ValueError):
            raise JudgeResponseError(f"confidence 必须为数值，得到 {conf!r}")
        if conf_val < 0 or conf_val > 1:
            raise JudgeResponseError(f"confidence 必须在 0~1 之间，得到 {conf_val}")
        item["confidence"] = conf_val
        qt = item.get("question_text")
        item["question_text"] = qt.strip() if isinstance(qt, str) else ""
        kp = item.get("knowledge_point_name")
        item["knowledge_point_name"] = kp.strip() if isinstance(kp, str) else ""
        nc = item.get("node_code")
        item["node_code"] = nc.strip() if isinstance(nc, str) else ""


async def _call_quiz_recognize_judge(
    ocr_text: str, candidates: list[dict[str, Any]]
) -> dict:
    """调用 LLM_JUDGE_MODEL 切题 + 知识点定位（JSON 解析失败重试 1 次）。

    Judge 未配置 → JudgeNotConfiguredError（路由层 500）；两次均失败 → JudgeResponseError。
    """
    if not (VOLCANO_API_KEY and LLM_JUDGE_MODEL):
        raise JudgeNotConfiguredError(
            "LLM_JUDGE_MODEL 未配置，无法识别题目（需配置 VOLCANO_API_KEY + LLM_JUDGE_MODEL）"
        )
    from services.dialogue import _parse_json_response

    ocr_text = (ocr_text or "")[:LLM_JUDGE_OCR_TEXT_MAX]
    prompt = _QUIZ_RECOGNIZE_USER_TEMPLATE.format(
        ocr_text=ocr_text or "（OCR 文本为空）",
        candidates=_format_candidates(candidates),
    )
    client = _get_judge_client()
    last_err: Exception | None = None
    logger.info(
        f"[quiz] Judge 调用开始 model={LLM_JUDGE_MODEL} "
        f"候选集={len(candidates)} 条 OCR文本={len(ocr_text)} 字 prompt={len(prompt)} 字符"
    )
    t0 = time.time()
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_judge_sync,
                client,
                LLM_JUDGE_MODEL,
                _QUIZ_RECOGNIZE_SYSTEM_PROMPT,
                prompt,
            )
            result = _parse_json_response(response)
            _validate_quiz_recognize_result(result)
            logger.info(
                f"[quiz] Judge 调用成功 耗时={time.time() - t0:.1f}s "
                f"items={len(result.get('items') or [])} 条（第 {attempt} 次）"
            )
            return result
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 JudgeResponseError
            last_err = e
            logger.warning(
                f"[quiz] 识别 Judge 调用失败（第 {attempt} 次）耗时={time.time() - t0:.1f}s: "
                f"{type(e).__name__}: {e}"
            )
    raise JudgeResponseError(f"同类型题识别 Judge 调用失败: {last_err}")


def _resolve_knowledge_point(
    kp_name: str, node_code: str, candidates: list[dict[str, Any]]
) -> tuple[str, str]:
    """知识点锚定：只在候选集内解析；**未命中 → 返回空串**（不建节点、不报错）。

    与错题链路的区别：错题链路未命中会走 EXTRA_AI 新建节点，
    本链路按 ADR-0035 附带决策「quiz 不新建 EXTRA_AI 节点」，交由用户在原题卡手填。
    """
    name = (kp_name or "").strip()
    if not name:
        return "", ""
    for cand in candidates:
        if (cand.get("kp_name") or "").strip() == name:
            return cand.get("kp_name") or name, (cand.get("node_code") or "").strip()
    # node_code 命中但 name 未命中（Judge 给了近义名）→ 也认可
    code = (node_code or "").strip()
    if code:
        for cand in candidates:
            if (cand.get("node_code") or "").strip() == code:
                return cand.get("kp_name") or name, cand.get("node_code") or code
    return "", ""


def _assemble_quiz_items(
    raw_items: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Judge items → 接口 28 出参 items[]（丢弃无 question_text 的项）。"""
    items: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            continue
        question_text = raw.get("question_text") or ""
        if not question_text:
            continue
        kp_name, node_code = _resolve_knowledge_point(
            raw.get("knowledge_point_name") or "",
            raw.get("node_code") or "",
            candidates,
        )
        items.append(
            {
                "block_index": idx,
                "question_text": question_text,
                "knowledge_point_name": kp_name,
                "node_code": node_code,
                "confidence": float(raw.get("confidence") or 0),
            }
        )
    return items


async def recognize_quiz_scan(
    db: Any,
    *,
    scan_id: str,
    scholar_id: str = "",
    photo_context: str = "",
    actor: str = "",
) -> dict:
    """拍照题识别（api-contract §3.10 接口 28，**红线 R1：零写库**）。

    只读 `math_scan_upload`（OCR 结果）+ `curriculum_node`（知识点候选集），
    一次 Judge 调用返回**全部**题目 `items[]`（D9：勾选在前端做）。

    Returns:
        {scan_id, status: "success" | "empty", ocr_text, items[{
            block_index, question_text, knowledge_point_name, node_code, confidence}]}

    Raises:
        ScanNotFoundError: scan_id 缺失 / 不存在（→ 404）
        OcrNotReadyError: OCR 未完成（→ 404）
        JudgeNotConfiguredError / JudgeResponseError: Judge 不可用（→ 500）
    """
    if not scan_id:
        raise ScanNotFoundError("缺少 scan_id")

    logger.info(f"[quiz] recognize 入口 scan_id={scan_id}")
    res = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION, where={"scan_id": scan_id}, limit=1
    )
    records = res.get("records") or []
    if not records:
        raise ScanNotFoundError(f"scan_id 不存在: {scan_id}")
    scan = records[0]

    # OCR 完成性校验（与 error_scanner.classify_scan_upload 同口径）
    if scan.get("ocr_status") != OCR_STATUS_SUCCESS:
        detail = f"OCR 未完成（status={scan.get('ocr_status')}），无法识别题目"
        ocr_error = (scan.get("ocr_error") or "").strip()
        if ocr_error:
            detail += f"：{ocr_error[:200]}"
        logger.warning(
            f"[quiz] recognize 被 OCR 阻塞 scan_id={scan_id} "
            f"ocr_status={scan.get('ocr_status')} ocr_error={(ocr_error or '')[:200]}"
        )
        raise OcrNotReadyError(detail)

    ocr_text = scan.get("ocr_text") or ""
    candidate_scholar = scan.get("scholar_id") or scholar_id or ""

    # 候选集：photo_context 命中时收敛到该教材链（与 F4 同口径），否则全量自动猜
    candidates = await _load_knowledge_point_candidates(db, candidate_scholar)
    ctx = _parse_photo_context(photo_context)
    anchor_candidates = _photo_context_anchor(candidates, ctx)
    used_candidates = anchor_candidates or candidates
    logger.info(
        f"[quiz] 候选集 scholar={candidate_scholar} 全量={len(candidates)} "
        f"photo_context={'命中' if anchor_candidates else '未命中/缺省'} 采用={len(used_candidates)}"
    )

    try:
        judge_result = await _call_quiz_recognize_judge(ocr_text, used_candidates)
    except (JudgeNotConfiguredError, JudgeResponseError) as e:
        logger.warning(
            f"[quiz] Judge 不可用 scan_id={scan_id}: {type(e).__name__}（零写库，不回写 classify_status）"
        )
        await write_audit(
            db,
            action=AUDIT_ACTION_QUIZ_RECOGNIZE,
            object_ref=scan_id,
            actor=actor,
            result=AUDIT_RESULT_FAILED,
            context={
                "scholar_id": scholar_id or candidate_scholar,
                "reason": type(e).__name__,
                "candidates_count": len(used_candidates),
                "model": LLM_JUDGE_MODEL,
            },
        )
        raise

    items = _assemble_quiz_items(judge_result.get("items") or [], used_candidates)
    status = "success" if items else "empty"
    await write_audit(
        db,
        action=AUDIT_ACTION_QUIZ_RECOGNIZE,
        object_ref=scan_id,
        actor=actor,
        result=AUDIT_RESULT_SUCCESS,
        context={
            "scholar_id": scholar_id or candidate_scholar,
            "item_count": len(items),
            "candidates_count": len(used_candidates),
            "model": LLM_JUDGE_MODEL,
        },
    )
    logger.info(f"[quiz] recognize 完成 scan_id={scan_id} status={status} items={len(items)}")
    return {
        "scan_id": scan_id,
        "status": status,
        "ocr_text": ocr_text,
        "items": items,
    }
