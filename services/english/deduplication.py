"""E-API-12 英语语句批量去重（L1 hash 去重的管理端清理入口）

定位（service-contract §8 续编）：
- 既有 E-API-3 只能"发现"重复（duplicate_in_textbook / cross_textbook_duplicate）；
  E-API-5 仅支持单句删除时连带 `delete_duplicates`。本模块提供**独立批量去重入口**：
  扫描 `text_hash` 重复组 → `dry_run` 预览（含关联计数）→ 确认后级联清理（保留 canonical）。
- canonical 保留口径与 M3 建簇完全一致（复用 `_pick_canonical_id`：组内已有
  canonical 自指优先，否则 `created_at` 最早者）；
- 删除语义与 E-API-5 一致（复用 `_cascade_delete_sentence`：6 表级联 + 组引用移除 +
  sentence_semantic_key registry 维护），每次执行去重写一条审计
  `deduplicate_english_sentences`（必审 24）。

契约：api-contract.md §3.11 E-API-12；service-contract.md §8.1（E-API-12 行）。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from services.audit import AUDIT_ACTION_DEDUPLICATE_ENGLISH_SENTENCES, write_audit
from services.database import SENTENCE_V2, TEXTBOOK_V2
from services.english import LessonNotFoundError, TextbookNotFoundError
from services.english.sentence_management import (
    _RELATED_COUNT_COLLECTIONS,
    _cascade_delete_sentence,
    _find_lesson,
    _pick_canonical_id,
)

logger = logging.getLogger("scholar-admin.english.deduplication")


async def _find_textbook(db, textbook_id: str) -> dict:
    """教材存在性校验（404 TEXTBOOK_NOT_FOUND）。"""
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    return q["records"][0]


async def _load_related_counts(
    db, sentence_ids: list[str]
) -> dict[str, dict[str, int]]:
    """批量统计各句在 5 张关联表的计数（E-API-4 同款策略，单表 $in 查询后内存分组）。

    任一关联表缺失/查询失败不阻断去重（沿用 E-API-4 列表兜底语义）。
    """
    counts: dict[str, dict[str, int]] = {
        sid: {f"{c}_count": 0 for c in _RELATED_COUNT_COLLECTIONS}
        for sid in sentence_ids
    }
    if not sentence_ids:
        return counts
    for coll in _RELATED_COUNT_COLLECTIONS:
        try:
            res = await db.query(
                coll,
                where={"sentence_id": {"$in": sentence_ids}},
                select={"sentence_id": 1},
                limit=5000,
            )
            for r in res["records"]:
                sid = r.get("sentence_id")
                if sid in counts:
                    counts[sid][f"{coll}_count"] += 1
        except Exception as exc:
            logger.warning(
                f"[english.deduplicate] 关联表 {coll!r} 计数失败: {exc!r}"
            )
    return counts


async def deduplicateEnglishSentences(
    db,
    *,
    textbook_id: str,
    lesson_id: str | None = None,
    dry_run: bool = True,
    editor_id: str = "system",
) -> dict:
    """批量去重（E-API-12）：扫描 `text_hash` 重复组，`dry_run` 预览 / 确认执行清理。

    Args:
        db: CloudBaseNoSQLClient
        textbook_id: 教材 ID（必填，不存在 → 404）
        lesson_id: 可选课时 ID（传则仅清理该课时内重复；课时不存在 → 404）
        dry_run: True 仅返回预览（零写入）；False 执行级联清理并写审计
        editor_id: 操作者（审计 actor）

    Returns:
        {
          "scope": "textbook" | "lesson",
          "textbook_id": ...,
          "lesson_id": ...,
          "dry_run": bool,
          "total_groups": 重复组数,
          "total_duplicates": 待删重复句数,
          "groups": [
            {
              "text_hash", "text", "count",
              "canonical_sentence_id", "canonical_lesson_id",
              "duplicates": [{sentence_id, lesson_id, created_at, related_data}],
            }
          ],
          "deleted": {...} / None,   # dry_run=False 时：8 表级联汇总 + sentence_v2
          "deleted_count": 0,        # dry_run=False 时为实际删除句数
        }
    """
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    await _find_textbook(db, textbook_id)
    if lesson_id:
        await _find_lesson(db, textbook_id, lesson_id)

    where: dict[str, Any] = {"textbook_id": textbook_id}
    if lesson_id:
        where["lesson_id"] = lesson_id
    q = await db.query(SENTENCE_V2, where=where, limit=5000)
    rows = q["records"]

    # ---- 1) 按 text_hash 分组（text_hash 由 getter 惰性注入，存量无字段记录同样可靠）---- #
    hash_groups: dict[str, list[dict]] = {}
    for r in rows:
        h = r.get("text_hash") or ""
        if h and r.get("sentence_id"):
            hash_groups.setdefault(h, []).append(r)

    # ---- 2) 组装重复组（canonical 保留 + 待删列表，M3 建簇口径）---- #
    plan: list[dict] = []
    pending_delete: list[str] = []
    for _h, group in hash_groups.items():
        if len(group) <= 1:
            continue
        canonical_id = _pick_canonical_id(group, group[0].get("sentence_id") or "")
        canonical = next(
            (r for r in group if r.get("sentence_id") == canonical_id), group[0]
        )
        duplicates = [r for r in group if r.get("sentence_id") != canonical_id]
        duplicates.sort(key=lambda r: r.get("created_at") or 0)
        plan.append(
            {
                "text_hash": _h,
                "text": canonical.get("text") or group[0].get("text") or "",
                "count": len(group),
                "canonical_sentence_id": canonical_id,
                "canonical_lesson_id": canonical.get("lesson_id") or "",
                "duplicates": [
                    {
                        "sentence_id": r.get("sentence_id") or "",
                        "lesson_id": r.get("lesson_id") or "",
                        "created_at": int(r.get("created_at") or 0),
                    }
                    for r in duplicates
                ],
            }
        )
        pending_delete.extend(r.get("sentence_id") for r in duplicates)

    # ---- 3) 关联计数（预览展示 + 审计上下文共用）---- #
    related_counts = await _load_related_counts(
        db, [sid for sid in pending_delete if sid]
    )
    for g in plan:
        for d in g["duplicates"]:
            d["related_data"] = related_counts.get(
                d["sentence_id"],
                {f"{c}_count": 0 for c in _RELATED_COUNT_COLLECTIONS},
            )

    result: dict = {
        "scope": "lesson" if lesson_id else "textbook",
        "textbook_id": textbook_id,
        "lesson_id": lesson_id,
        "dry_run": dry_run,
        "total_groups": len(plan),
        "total_duplicates": len(pending_delete),
        "groups": plan,
        "deleted": None,
        "deleted_count": 0,
    }

    if dry_run or not pending_delete:
        return result

    # ---- 4) 执行：逐句级联删除（E-API-5 同款语义；audio_asset 默认保留）---- #
    deleted: dict[str, int] = {
        "study_attempt": 0,
        "skill_state": 0,
        "speech_evaluation": 0,
        "learning_attempt": 0,
        "audio_asset": 0,
        "conversation_turn_marked": 0,
        "sentence_group_refs_removed": 0,
        "semantic_registry_refs_removed": 0,
    }
    for sid in pending_delete:
        sub = await _cascade_delete_sentence(
            db, sentence_id=sid, delete_audio_asset=False
        )
        await db.delete(SENTENCE_V2, where={"sentence_id": sid})
        sub["sentence_v2"] = 1
        for k, v in sub.items():
            if k != "sentence_v2":
                deleted[k] = deleted.get(k, 0) + v

    result["deleted"] = deleted
    result["deleted_count"] = len(pending_delete)

    await write_audit(
        db,
        action=AUDIT_ACTION_DEDUPLICATE_ENGLISH_SENTENCES,
        object_ref=textbook_id,
        actor=editor_id,
        context={
            "scope": result["scope"],
            "lesson_id": lesson_id,
            "groups": len(plan),
            "duplicates_deleted": len(pending_delete),
            "deleted": deleted,
        },
    )
    logger.info(
        f"[english.deduplicate] textbook={textbook_id} lesson={lesson_id} "
        f"groups={len(plan)} duplicates={len(pending_delete)}"
    )
    return result
