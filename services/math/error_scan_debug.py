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
from typing import Any

from services.database import CURRICULUM_NODE_COLLECTION
from services.math.error_scanner import _SCHOLAR_BOOK_COLLECTION

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
