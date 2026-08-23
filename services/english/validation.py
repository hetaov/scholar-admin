"""E1.2 英语语句管理：3 服务函数（stats/chapter_tree/validate + L1 hash 重复检测 + 校验缓存）

对应 3 个 HTTP 路由（routes/english.py）：
1. GET  /english/textbook/stats                      → list_english_textbook_stats
2. GET  /english/textbook/{tid}/chapters             → get_english_chapter_tree
3. POST /english/textbook/{tid}/validate-sentences   → validate_english_sentences

规格：service-contract.md §8.1；契约：api-contract.md §3.11 E-API-1~E-API-3；
L1 hash 重复检测规则：service-contract §8.3；校验缓存：§8.4（TTL 1h）。
"""
from __future__ import annotations

import logging
from typing import Any

from cachetools import TTLCache

from services.database import SENTENCE_V2, TEXTBOOK_V2
from services.english import (
    MissingFieldError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.english.structure import load_lesson_entries

logger = logging.getLogger("scholar-admin.english.validation")

# ===========================================================================
# 常量（service-contract §8.1 / §8.3 / §8.4）
# ===========================================================================

_VALID_SCOPES = {"full", "chapter", "lesson"}
_DEFAULT_CHECK_TYPES = (
    "orphan_lesson",
    "chapter_mismatch",
    "duplicate_in_textbook",
    "cross_textbook_duplicate",
    "empty_content",
)
_VALID_CHECK_TYPES = set(_DEFAULT_CHECK_TYPES)

# 校验缓存：进程内 TTLCache，TTL 3600s（1h），key = f"{textbook_id}:{scope}:{chapter_id}:{lesson_id}"
_VALIDATION_TTL = 3600
_validation_cache: TTLCache = TTLCache(maxsize=256, ttl=_VALIDATION_TTL)


# ===========================================================================
# 校验缓存辅助（service-contract §8.4）
# ===========================================================================


def _validation_key(
    textbook_id: str, scope: str, chapter_id: str | None, lesson_id: str | None
) -> str:
    return f"{textbook_id}:{scope}:{chapter_id or ''}:{lesson_id or ''}"


def clear_validation_cache() -> None:
    """清空校验缓存（测试隔离用，生产无调用）。"""
    _validation_cache.clear()


def _read_cached_status(
    textbook_id: str, scope: str, chapter_id: str | None, lesson_id: str | None
) -> str:
    """从缓存读取 validation_status，缺省 pending（E-API-1/E-API-2 复用）。"""
    entry = _validation_cache.get(_validation_key(textbook_id, scope, chapter_id, lesson_id))
    if entry:
        return entry.get("validation_status") or "pending"
    return "pending"


# ===========================================================================
# 结构辅助
# ===========================================================================


def _iter_lesson_entries(textbook: dict):
    """遍历教材的 (chapter_id, chapter_title, lesson) 三元组。

    兼容两种结构（与 sentence_management._find_lesson 对齐）：
    - 标准：textbook.chapters[].lessons[]
    - 无章教材：textbook.lessons[]（chapter_id=''，chapter_title='未分章'）
    """
    for ch in textbook.get("chapters") or []:
        for ls in ch.get("lessons") or []:
            yield ch.get("chapter_id") or "", ch.get("title") or "", ls
    for ls in textbook.get("lessons") or []:
        yield "", "未分章", ls


def _build_lesson_index(entries: list[dict]) -> dict[str, dict]:
    """lesson_id → {chapter_id, title} 索引（orphan/chapter_mismatch 校验用）。

    Args:
        entries: load_lesson_entries 返回的课时条目列表
        （[{"chapter_id", "chapter_title", "lesson"}, ...]，含独立集合回退形态）。
    """
    index: dict[str, dict] = {}
    for e in entries:
        ls = e["lesson"]
        index[ls.get("lesson_id") or ""] = {
            "chapter_id": e["chapter_id"],
            "title": ls.get("title") or "",
        }
    return index


def _duplicate_groups(rows: list[dict]) -> dict[str, list[dict]]:
    """按 text_hash 分组（hash 非空），仅返回组内 count>1 的组。

    getter（E0.1）已在读侧注入 text_hash，对存量无字段记录同样可靠。
    """
    groups: dict[str, list[dict]] = {}
    for r in rows:
        h = r.get("text_hash") or ""
        if h:
            groups.setdefault(h, []).append(r)
    return {h: g for h, g in groups.items() if len(g) > 1}


def _duplicate_instance_count(rows: list[dict]) -> int:
    """教材/课时重复数 = 冗余实例数（同 hash 组内除首条外的条数之和）。

    例如 2 条同 hash → 1；3 条同 hash → 2。语义 = 「可删除的冗余句数」。
    """
    return sum(len(g) - 1 for g in _duplicate_groups(rows).values())


# ===========================================================================
# 1. GET /english/textbook/stats — 教材统计概览（E-API-1）
# ===========================================================================


async def list_english_textbook_stats(db, *, grade: str | None = None) -> dict:
    """教材统计概览（纯读不审）。

    统计聚合在内存完成（教材/语句量级有限，避免跨集合复杂聚合）；
    subject_type 过滤读侧做（getter 对存量注入 english，与 G0.1 语义一致）；
    duplicate_count = 冗余实例数；validation_status 从缓存读取（缺省 pending）。
    """
    q = await db.query(TEXTBOOK_V2, where={}, limit=2000)
    textbooks: list[dict] = []
    for tb in q["records"]:
        if (tb.get("subject_type") or "english") != "english":
            continue
        tbid = tb.get("textbook_id") or ""
        if grade and (tb.get("grade") or "") != grade:
            continue
        sq = await db.query(SENTENCE_V2, where={"textbook_id": tbid}, limit=5000)
        rows = sq["records"]
        # 内嵌为空（标准管线只写计数，章节/课时在独立集合）→ 回退计数字段
        embedded_chapters = tb.get("chapters") or []
        embedded_lesson_count = sum(1 for _ in _iter_lesson_entries(tb))
        if embedded_chapters or embedded_lesson_count:
            chapter_count = len(embedded_chapters)
            lesson_count = embedded_lesson_count
        else:
            chapter_count = int(tb.get("chapter_count") or 0)
            lesson_count = int(tb.get("lesson_count") or 0)
        textbooks.append(
            {
                "textbook_id": tbid,
                "title": tb.get("title") or "",
                "grade": tb.get("grade") or "",
                "level": tb.get("level"),
                "chapter_count": chapter_count,
                "lesson_count": lesson_count,
                "sentence_count": len(rows),
                "duplicate_count": _duplicate_instance_count(rows),
                "validation_status": _read_cached_status(tbid, "full", None, None),
                "updated_at": tb.get("updated_at") or 0,
            }
        )
    return {"textbooks": textbooks}


# ===========================================================================
# 2. GET /english/textbook/{tid}/chapters — 章节课时树（E-API-2）
# ===========================================================================


async def get_english_chapter_tree(db, *, textbook_id: str) -> dict:
    """章节课时树（管理端，纯读不审）。

    返回 textbook.chapters → lessons 结构，每 lesson 聚合 sentence_count /
    duplicate_count / validation_status（缓存读取，缺省 pending）。
    orphan 句子（lesson_id 不存在）不归属任何 lesson，不出现在树中。
    """
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    tb = q["records"][0]

    sq = await db.query(SENTENCE_V2, where={"textbook_id": textbook_id}, limit=5000)
    by_lesson: dict[str, list[dict]] = {}
    for r in sq["records"]:
        by_lesson.setdefault(r.get("lesson_id") or "", []).append(r)

    def _lesson_entry(ls: dict) -> dict:
        lsid = ls.get("lesson_id") or ""
        lrows = by_lesson.get(lsid, [])
        return {
            "lesson_id": lsid,
            "title": ls.get("title") or "",
            "sentence_count": len(lrows),
            "duplicate_count": _duplicate_instance_count(lrows),
            "validation_status": _read_cached_status(textbook_id, "lesson", None, lsid),
        }

    chapters: list[dict] = []
    # 统一从 load_lesson_entries 取课时条目（兼容内嵌结构 + 独立集合回退）
    entries = await load_lesson_entries(db, tb)
    chapter_map: dict[str, dict] = {}
    chapter_order: list[str] = []
    for e in entries:
        cid = e["chapter_id"]
        if cid not in chapter_map:
            chapter_map[cid] = {
                "chapter_id": cid,
                "title": e["chapter_title"],
                "lessons": [],
            }
            chapter_order.append(cid)
        chapter_map[cid]["lessons"].append(_lesson_entry(e["lesson"]))
    chapters = [chapter_map[cid] for cid in chapter_order]

    return {
        "textbook_id": textbook_id,
        "title": tb.get("title") or "",
        "chapters": chapters,
    }


# ===========================================================================
# 3. POST /english/textbook/{tid}/validate-sentences — 语句归属校验（E-API-3）
# ===========================================================================


async def validate_english_sentences(
    db,
    *,
    textbook_id: str,
    scope: str = "full",
    chapter_id: str | None = None,
    lesson_id: str | None = None,
    check_types: list[str] | None = None,
) -> dict:
    """语句归属校验（E-API-3，结果写入缓存 TTL 1h）。

    5 类异常（service-contract §8.3 / api-contract E-API-3 校验维度）：
    - orphan_lesson             lesson_id 不存在                    → error
    - chapter_mismatch          sentence.chapter_id ≠ lesson 所属 chapter → warning
    - duplicate_in_textbook     教材内同 text_hash 多条              → warning
    - cross_textbook_duplicate  其他英语教材同 text_hash             → info
    - empty_content             text / translation 为空             → error
    """
    # 1) 教材存在性
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    tb = q["records"][0]

    # 2) scope 参数校验
    scope = (scope or "full").strip()
    if scope not in _VALID_SCOPES:
        raise SentencePayloadError(
            f"scope 取值仅限 {sorted(_VALID_SCOPES)}，收到 {scope!r}"
        )
    if scope == "chapter" and not chapter_id:
        raise MissingFieldError("scope=chapter 时 chapter_id 必填")
    if scope == "lesson" and not lesson_id:
        raise MissingFieldError("scope=lesson 时 lesson_id 必填")

    # 3) 校验目标句子
    where: dict[str, Any] = {"textbook_id": textbook_id}
    if scope == "chapter":
        where["chapter_id"] = chapter_id
    elif scope == "lesson":
        where["lesson_id"] = lesson_id
    sq = await db.query(SENTENCE_V2, where=where, limit=5000)
    rows = sq["records"]

    # 4) check_types 白名单
    types = set(check_types) if check_types else set(_DEFAULT_CHECK_TYPES)
    invalid = types - _VALID_CHECK_TYPES
    if invalid:
        raise SentencePayloadError(f"check_types 含非法取值：{sorted(invalid)}")

    # 5) 各维度校验
    issues: dict[str, list[dict]] = {t: [] for t in _DEFAULT_CHECK_TYPES}
    entries = await load_lesson_entries(db, tb)
    lesson_index = _build_lesson_index(entries)

    # orphan_lesson：lesson_id 不在教材 lesson 集合
    if "orphan_lesson" in types:
        for r in rows:
            lid = r.get("lesson_id") or ""
            if lid and lid not in lesson_index:
                issues["orphan_lesson"].append(
                    {
                        "sentence_id": r.get("sentence_id") or "",
                        "text": r.get("text") or "",
                        "lesson_id": lid,
                        "reason": "lesson_id not found in lesson collection",
                    }
                )

    # chapter_mismatch：sentence.chapter_id ≠ lesson 所属 chapter（orphan 不重复报）
    if "chapter_mismatch" in types:
        for r in rows:
            lid = r.get("lesson_id") or ""
            ls = lesson_index.get(lid)
            if not ls:
                continue
            scid = r.get("chapter_id") or ""
            lcid = ls["chapter_id"]
            if scid and scid != lcid:
                issues["chapter_mismatch"].append(
                    {
                        "sentence_id": r.get("sentence_id") or "",
                        "text": r.get("text") or "",
                        "sentence_chapter_id": scid,
                        "lesson_chapter_id": lcid,
                    }
                )

    # duplicate_in_textbook：教材内同 text_hash 多条（L1，service-contract §8.3）
    if "duplicate_in_textbook" in types:
        for h, g in _duplicate_groups(rows).items():
            issues["duplicate_in_textbook"].append(
                {
                    "text_hash": h,
                    "text": (g[0].get("text") or ""),
                    "count": len(g),
                    "sentence_ids": [r.get("sentence_id") or "" for r in g],
                    "lessons": [r.get("lesson_id") or "" for r in g],
                }
            )

    # cross_textbook_duplicate：其他英语教材同 text_hash（信息级）
    if "cross_textbook_duplicate" in types:
        other_ids = await _list_other_english_textbook_ids(db, textbook_id)
        other_hash_tbs: dict[str, set[str]] = {}
        if other_ids:
            oq = await db.query(
                SENTENCE_V2,
                where={"textbook_id": {"$in": other_ids}},
                limit=5000,
            )
            for r in oq["records"]:
                h = r.get("text_hash") or ""
                tbid = r.get("textbook_id") or ""
                if h and tbid:
                    other_hash_tbs.setdefault(h, set()).add(tbid)
        local_hashes = {r.get("text_hash") or "" for r in rows if r.get("text_hash")}
        for h in sorted(local_hashes):
            tbs = other_hash_tbs.get(h)
            if tbs:
                sample = next(
                    (r for r in rows if (r.get("text_hash") or "") == h), {}
                )
                issues["cross_textbook_duplicate"].append(
                    {
                        "text_hash": h,
                        "text": sample.get("text") or "",
                        "count": len(tbs) + 1,
                        "in_textbooks": sorted([textbook_id, *tbs]),
                    }
                )

    # empty_content：text / translation 为空
    if "empty_content" in types:
        for r in rows:
            sid = r.get("sentence_id") or ""
            text = r.get("text") or ""
            if not text or not text.strip():
                issues["empty_content"].append(
                    {"sentence_id": sid, "field": "text", "text": text}
                )
            elif not (r.get("translation") or "").strip():
                issues["empty_content"].append(
                    {"sentence_id": sid, "field": "translation", "text": text}
                )

    # 6) summary 分级 + validation_status
    error_count = len(issues["orphan_lesson"]) + len(issues["empty_content"])
    warning_count = len(issues["chapter_mismatch"]) + len(issues["duplicate_in_textbook"])
    info_count = len(issues["cross_textbook_duplicate"])
    total_issues = error_count + warning_count + info_count
    if error_count > 0:
        status = "error"
    elif warning_count > 0:
        status = "warning"
    else:
        status = "passed"

    result = {
        "textbook_id": textbook_id,
        "total_sentences": len(rows),
        "issues": issues,
        "summary": {
            "total_issues": total_issues,
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
        },
    }

    # 7) 写入校验缓存（service-contract §8.4，TTL 1h）
    _validation_cache[_validation_key(textbook_id, scope, chapter_id, lesson_id)] = {
        "validation_status": status,
        "result": result,
    }
    return result


async def _list_other_english_textbook_ids(db, textbook_id: str) -> list[str]:
    """查其他英语教材（subject_type=english，排除当前教材）的 textbook_id 列表。"""
    q = await db.query(TEXTBOOK_V2, where={}, limit=2000)
    return [
        t.get("textbook_id")
        for t in q["records"]
        if (t.get("subject_type") or "english") == "english"
        and t.get("textbook_id")
        and t.get("textbook_id") != textbook_id
    ]
