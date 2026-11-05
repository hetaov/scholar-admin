"""语音描述出题·描述模式出题服务（POST /math/quiz/generate 描述分支 · 方案 §7.2）

与拍照出题（quiz.py `generateQuiz`，source_questions 驱动）并行的入口：
`intent`（含 `grounded_kps`）驱动 → ① 图路径（`USE_LANGGRAPH_DESC_QUIZ=1` 且
`structure_mode=1`）：两段式编排图 `run_desc_quiz_basic`；
② 单步直出（flag 关 / `structure_mode=0`，请求级旁路 = R18）：一次 LLM 调用。

落库 `math_quiz` 新增：`graph_version` / `intake_meta`（溯源链）/
`validation_reasons[]`（拒绝码聚合）。出参投影：R3 红线 `answer`/`rationale` 恒
不出参；`target_error_step`/`extended_meta` 出参；`structure_fingerprint` 仅落库。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from config import USE_LANGGRAPH_DESC_QUIZ
from services.audit import AUDIT_RESULT_FAILED, AUDIT_RESULT_SUCCESS, write_audit
from services.database import CURRICULUM_NODE_COLLECTION
from services.math.desc_quiz_graph import (
    _call_chat,
    parse_basic_items_response,
    run_desc_quiz_basic,
)
from services.math.math_text import normalize_math_text
from services.math.quiz import (
    AUDIT_ACTION_QUIZ_GENERATE,
    InvalidQuizSourceError,
    _gen_quiz_id,
    _gen_quiz_item_id,
    _now_ms,
    _persist_quiz,
)

logger = logging.getLogger("scholar-admin.math.quiz_desc")

_DEFAULT_QUESTION_COUNT = 5
_MAX_QUESTION_COUNT = 20


async def load_neighbor_kps(
    db, grounded_kps: list[dict], *, max_neighbors: int = 2
) -> list[dict]:
    """一跳邻居（§6.2 图谱层白名单）：首个 grounded 节点同 `grade` 的其它节点。

    节点无 `grade` / 图谱查询失败 → 返回 []（拔高计划器自行 degraded 不阻塞）。
    """
    anchor = next((g for g in grounded_kps or [] if g.get("node_code")), None)
    if anchor is None:
        return []
    try:
        res = await db.query(
            CURRICULUM_NODE_COLLECTION, where={"code": anchor["node_code"]}, limit=1
        )
        records = res.get("records") or []
        if not records:
            return []
        grade = records[0].get("grade")
        if grade in (None, ""):
            return []
        res2 = await db.query(
            CURRICULUM_NODE_COLLECTION, where={"grade": grade}, limit=30
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[quiz_desc] 一跳邻居查询失败: {e}")
        return []

    exclude = {str(g.get("node_code") or "") for g in grounded_kps} | {""}
    out: list[dict] = []
    for r in res2.get("records") or []:
        code = str(r.get("code") or "")
        title = str(r.get("title") or "")
        if code in exclude or not title:
            continue
        ai = r.get("ai_summary") or {}
        out.append(
            {"name": title, "node_code": code, "summary": str(ai.get("summary") or title)}
        )
        if len(out) >= max_neighbors:
            break
    return out


_SINGLE_STEP_TEMPLATE = """请依据以下信息出 {question_count} 道小学数学练习题。

知识点：{kp_names}
题型：{question_type_hint}
难度提示：{difficulty_hint}

要求：每题输出 {{"question", "answer", "answer_type", "difficulty", "knowledge_point_name", "rationale"}}。
输出 JSON：{{"items": [...]}}"""


async def generate_desc_quiz_single_step(
    intent: dict, grounded_kps: list[dict]
) -> tuple[list[dict], list[str]]:
    """单步直出（flag 关 / structure_mode=0 请求级旁路，R18）。永不抛异常。"""
    kp_names = [str(g.get("name")) for g in grounded_kps] or [
        str(k) for k in (intent.get("knowledge_points") or [])
    ]
    try:
        content = await _call_chat(
            system="你是小学数学命题专家。只输出合法 JSON。",
            user=_SINGLE_STEP_TEMPLATE.format(
                question_count=int(intent.get("question_count") or _DEFAULT_QUESTION_COUNT),
                kp_names="、".join(kp_names) or "（未指定）",
                question_type_hint=str(intent.get("question_type_hint") or "不限"),
                difficulty_hint=str(intent.get("difficulty_hint") or "无"),
            ),
        )
        items = parse_basic_items_response(content)
    except Exception as e:  # noqa: BLE001 —— R16
        logger.warning(f"[quiz_desc] 单步直出失败: {type(e).__name__}: {e}")
        return [], [f"DESC_SINGLE_STEP_ERROR: {e}"]
    for it in items:
        it.setdefault("is_extended", False)
    return items, []


def _desc_quiz_signature(
    *, scholar_id: str, intent: dict, structure_mode: bool, include_extended: bool
) -> str:
    raw = json.dumps(
        {
            "scholar_id": scholar_id,
            "intent": intent,
            "structure_mode": bool(structure_mode),
            "include_extended": bool(include_extended),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _normalize_count(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise InvalidQuizSourceError("question_count 非法") from None
    if not 1 <= n <= _MAX_QUESTION_COUNT:
        raise InvalidQuizSourceError(f"question_count 超界（1~{_MAX_QUESTION_COUNT}）")
    return n


def _normalized_items(items: list[dict]) -> list[dict]:
    """图产物 → 落库口径：item_id / 题干清洗 / 拔高标记 / fingerprint 仅落库。"""
    out: list[dict] = []
    for it in items:
        rec = {
            "quiz_item_id": _gen_quiz_item_id(),
            "question": normalize_math_text(str(it.get("question") or "")),
            "answer": normalize_math_text(str(it.get("answer") or "")),
            "answer_type": str(it.get("answer_type") or "number"),
            "difficulty": it.get("difficulty"),
            "knowledge_point_name": str(it.get("knowledge_point_name") or ""),
            "rationale": str(it.get("rationale") or ""),
            "variant_level": str(it.get("variant_level") or "L1"),
            "is_extended": bool(it.get("is_extended")),
        }
        try:
            rec["target_error_step"] = int(it.get("target_error_step") or 0)
        except (TypeError, ValueError):
            rec["target_error_step"] = 0
        if it.get("extended_meta"):
            rec["extended_meta"] = it["extended_meta"]
        if it.get("structure_fingerprint"):
            rec["structure_fingerprint"] = str(it["structure_fingerprint"])  # 仅落库
        out.append(rec)
    return out


def _validation_reasons(warnings: list[str]) -> list[str]:
    """warnings 中的拒绝码聚合（`DESC_BASIC_ITEM_DROPPED[V2_…]: …` → 码列表）。"""
    codes: list[str] = []
    for w in warnings or []:
        if "[" in w and "]:" in w:
            inner = w[w.find("[") + 1 : w.rfind("]:")]
            for c in inner.split(","):
                c = c.strip()
                if c and c not in codes:
                    codes.append(c)
    return codes


def _to_public_desc_quiz(quiz: dict) -> dict:
    """落库文档 → 出参（§7.2）。R3：`answer`/`rationale`/fingerprint 不出参。"""
    return {
        "quiz_id": quiz.get("quiz_id"),
        "status": quiz.get("status"),
        "items": [
            {
                **{
                    k: it.get(k)
                    for k in (
                        "quiz_item_id",
                        "question",
                        "answer_type",
                        "difficulty",
                        "knowledge_point_name",
                        "target_error_step",
                        "is_extended",
                        "extended_meta",
                    )
                    if it.get(k) not in (None, "")
                }
            }
            for it in (quiz.get("items") or [])
        ],
        "warnings": quiz.get("warnings") or [],
        "graph_version": quiz.get("graph_version") or 0,
        "intent_missing": list(
            (quiz.get("intake_meta") or {}).get("missing_required") or []
        ),
    }


async def generateQuizFromDescription(
    db,
    *,
    scholar_id: str,
    intent: dict,
    grounded_kps: list[dict],
    intake_meta: dict | None = None,
    include_extended: bool = False,
    structure_mode: bool = True,
    scan_id: str | None = None,
    actor: str = "",
) -> dict:
    """描述模式出题（LLM/图异常永不抛出；入参非法 400 由路由层拦截）。"""
    count = _normalize_count(intent.get("question_count") or _DEFAULT_QUESTION_COUNT)
    use_graph = bool(USE_LANGGRAPH_DESC_QUIZ and structure_mode)
    neighbors: list[dict] = []
    if use_graph and include_extended:
        neighbors = await load_neighbor_kps(db, grounded_kps)

    t0 = time.time()
    if use_graph:
        try:
            result = await run_desc_quiz_basic(
                grounded_kps=grounded_kps
                or [{"name": k} for k in (intent.get("knowledge_points") or [])],
                question_count=count,
                question_type_hint=str(intent.get("question_type_hint") or ""),
                include_extended=include_extended,
                neighbors=neighbors,
            )
            raw_items = result["items"]
            warnings = result["warnings"]
            graph_version = result["graph_version"]
        except Exception as e:  # noqa: BLE001 —— R16 双保险
            logger.warning(f"[quiz_desc] 图路径异常: {e}")
            raw_items, warnings, graph_version = [], [f"DESC_GRAPH_ERROR: {e}"], 0
    else:
        raw_items, warnings = await generate_desc_quiz_single_step(intent, grounded_kps)
        graph_version = 0

    items = _normalized_items(raw_items)
    ok = bool(items)
    audit = await write_audit(
        db,
        action=AUDIT_ACTION_QUIZ_GENERATE,
        object_ref="",
        actor=actor,
        result=AUDIT_RESULT_SUCCESS if ok else AUDIT_RESULT_FAILED,
        context={
            "scholar_id": scholar_id,
            "mode": "desc_graph" if use_graph else "desc_single_step",
            "question_count": count,
            "generated_count": len(items),
            "extended_count": sum(1 for i in items if i["is_extended"]),
            "include_extended": bool(include_extended),
            "graph_version": graph_version,
            "elapsed_ms": int((time.time() - t0) * 1000),
        },
    )

    if not ok:
        return {
            "quiz_id": "",
            "status": "failed",
            "items": [],
            "warnings": warnings,
            "graph_version": graph_version,
            "intent_missing": list((intake_meta or {}).get("missing_required") or []),
        }

    quiz_id = _gen_quiz_id()
    saved = await _persist_quiz(
        db,
        {
            "quiz_id": quiz_id,
            "scholar_id": scholar_id,
            "source_scan_id": scan_id or "",
            "sources": [],
            "source_count": 0,
            "count_per_source": count,
            "items": items,
            "include_extended": bool(include_extended),
            "extended_count": sum(1 for i in items if i["is_extended"]),
            "signature": _desc_quiz_signature(
                scholar_id=scholar_id,
                intent=intent,
                structure_mode=structure_mode,
                include_extended=include_extended,
            ),
            "warnings": warnings,
            "audit_log_id": (audit or {}).get("log_id") or "",
            "graph_version": graph_version,
            "intake_meta": dict(intake_meta or {}),
            "validation_reasons": _validation_reasons(warnings),
            "created_at": _now_ms(),
        },
    )
    logger.info(
        f"[quiz_desc] 出题完成 quiz_id={quiz_id} mode={'graph' if use_graph else 'single'} 出题={len(items)}"
    )
    return _to_public_desc_quiz(saved)
