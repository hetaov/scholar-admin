"""G1.1 数学教材管理：共用 6 函数（CRUD + 概览 + 批量节点导入 + 清理）

对应 6 个 HTTP 路由（routes_math.py）：
1. GET    /math/textbook              → list_math_textbooks
2. POST   /math/textbook              → create_math_textbook
3. PUT    /math/textbook/{id}         → update_math_textbook
4. GET    /math/textbook/overview     → get_textbook_overview
5. POST   /math/textbook/import-nodes → import_curriculum_nodes
6. DELETE /math/textbook/{id}         → delete_math_textbook_cleanup

依赖：
- models_content.validate_textbook_payload / build_textbook_v2_doc（G0.1 底座）
- audit.write_audit（G0.2 审计 action 扩展后落库）
- math/__init__.py TextbookPayloadError / TextbookNotFoundError / ConfirmationMismatchError
- database.CloudBaseNoSQLClient（query / insert / update / count）
"""
from __future__ import annotations

import logging
import time
from typing import Any

from services import math as _math_pkg  # 避免与函数参数 math 命名冲突
from services.audit import (
    AUDIT_ACTION_CREATE_MATH_TEXTBOOK,
    AUDIT_ACTION_DELETE_MATH_TEXTBOOK,
    AUDIT_ACTION_IMPORT_MATH_NODES,
    AUDIT_ACTION_UPDATE_MATH_TEXTBOOK,
    write_audit,
)
from services.database import CURRICULUM_NODE_COLLECTION, TEXTBOOK_V2
from services.math import (
    ConfirmationMismatchError,
    CURRICULUM_NODE_IDEMPOTENCY_FIELD,
    ERR_DELETE_CONFIRM_MISMATCH,
    ERR_IMPORT_NODES_DUPLICATE_CODE,
    ERR_IMPORT_ON_DUPLICATE_INVALID,
    ERR_SUBJECT_TYPE_CHANGE_CONFIRM_REQUIRED,
    ERR_TEXTBOOK_NOT_FOUND,
    ERR_TEXTBOOK_SUBJECT_TYPE_NOT_MATH,
    ERR_UPDATE_TITLE_CONFIRM_MISMATCH,
    SUBJECT_TYPE_MATH,
    TextbookNotFoundError,
    TextbookPayloadError,
    VALID_IMPORT_ON_DUPLICATE_SET,
)
from services.models_content import (
    _TextbookPayloadError,  # validate_textbook_payload 抛的本地别名，需转 math.* 对外版
    build_textbook_v2_doc,
    validate_textbook_payload,
)

logger = logging.getLogger("scholar-admin.math.textbook_management")


# ===========================================================================
# 1. GET /math/textbook — 列表
# ===========================================================================


async def list_math_textbooks(
    db,
    *,
    subject_type: str = SUBJECT_TYPE_MATH,
    grade: str | None = None,
    semester: str | None = None,
    keyword: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> dict:
    """数学教材列表（默认 subject_type=math，永远不返回英语教材）。

    Returns:
        {"items": [...textbook_doc], "total": N}
    """
    where: dict[str, Any] = {"subject_type": subject_type or SUBJECT_TYPE_MATH}
    if grade:
        where["grade"] = grade
    if semester:
        where["semester"] = semester

    q = await db.query(
        TEXTBOOK_V2,
        where=where,
        order=[{"field": "created_at", "direction": "desc"}, {"field": "title", "direction": "asc"}],
        offset=offset,
        limit=limit,
    )
    items = q["records"]
    total = q["total"]

    if keyword:
        kw = keyword.strip().lower()
        items = [r for r in items if kw in (r.get("title") or "").lower()]
        total = len(items)

    return {"items": items, "total": total, "offset": offset, "limit": limit}


# ===========================================================================
# 辅助：按 textbook_id 查单条数学教材（其余函数共享）
# ===========================================================================


async def _get_math_textbook_or_404(db, textbook_id: str) -> dict:
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    doc = q["records"][0]
    if doc.get("subject_type") != SUBJECT_TYPE_MATH:
        # 非 math 教材在 /math 管理端按「不存在」处理（禁止操作）
        raise TextbookNotFoundError(
            textbook_id,
            message=f"教材 {textbook_id!r} 不是数学学科，数学管理端不可操作",
        )
    return doc


# ===========================================================================
# 2. POST /math/textbook — 新增
# ===========================================================================


async def create_math_textbook(
    db,
    *,
    payload: dict,
    actor: str = "",
) -> dict:
    """创建数学教材（只允许 subject_type=math 或缺省 math，禁止显式 english）。"""
    # ---- 契约 §3.1 POST /math/textbook：强制数学管理端仅创建 math 教材 ---- #
    payload = payload or {}
    st_raw = payload.get("subject_type") if isinstance(payload, dict) else None
    if isinstance(st_raw, str) and st_raw.strip() and st_raw.strip() != SUBJECT_TYPE_MATH:
        raise TextbookPayloadError(
            ERR_TEXTBOOK_SUBJECT_TYPE_NOT_MATH,
            f"数学管理端新建仅允许 subject_type={SUBJECT_TYPE_MATH!r}；"
            f"英语教材请走原生 /textbook 接口。收到 {st_raw!r}",
        )

    # validate_textbook_payload 内部抛的是 _TextbookPayloadError（models_content 本地别名），
    # 这里统一转成 math 包对外的 TextbookPayloadError，保证 routes 层 except 单一类型。
    try:
        cleaned = validate_textbook_payload(**payload)
    except _TextbookPayloadError as exc:
        raise TextbookPayloadError(exc.code, exc.message) from exc

    # 再强制一次 math：validate 缺省 subject_type → english（符合读侧 getter 契约）；
    # /math/textbook 写侧必须 math
    cleaned["subject_type"] = SUBJECT_TYPE_MATH

    chapters = cleaned.get("chapters") or []
    cleaned_tbid = cleaned.get("textbook_id") or _gen_default_textbook_id(cleaned)

    # build_doc 接受 math 相关字段 (subject_type/semester/publisher/cover_url/isbn) 以及 textbook_id
    # 但不接受 chapters / created_at / updated_at —— 手动追加（保证后向兼容）
    doc = build_textbook_v2_doc(
        title=cleaned["title"],
        grade=cleaned["grade"],
        subject_type=cleaned["subject_type"],
        semester=cleaned.get("semester"),
        publisher=cleaned.get("publisher"),
        cover_url=cleaned.get("cover_url"),
        isbn=cleaned.get("isbn"),
        textbook_id=cleaned_tbid,
        chapter_count=len(chapters),
    )
    now_ms = int(time.time() * 1000)
    doc["chapters"] = chapters
    doc["created_at"] = now_ms
    doc["updated_at"] = now_ms

    await db.insert(TEXTBOOK_V2, doc)
    await write_audit(
        db,
        action=AUDIT_ACTION_CREATE_MATH_TEXTBOOK,
        object_ref=doc["textbook_id"],
        actor=actor,
        context={
            "subject_type": doc["subject_type"],
            "grade": doc.get("grade"),
            "semester": doc.get("semester"),
        },
    )
    return doc


def _gen_default_textbook_id(cleaned: dict) -> str:
    """未传 textbook_id 时按 (grade+semester+publisher_or_title_hash) 生成。"""
    import hashlib

    grade = (cleaned.get("grade") or "").strip()
    sem = (cleaned.get("semester") or "").strip()
    pub = (cleaned.get("publisher") or cleaned.get("title") or "").strip()
    raw = f"math::{grade}::{sem}::{pub}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"tb_math_{grade}_{sem}_{h}".replace(" ", "_")


# ===========================================================================
# 3. PUT /math/textbook/{id} — 更新（二次确认保护）
# ===========================================================================


async def update_math_textbook(
    db,
    *,
    textbook_id: str,
    update: dict,
    confirm_title: str = "",
    actor: str = "",
) -> dict:
    """更新教材字段；跨学科改 subject_type 或改标题需二次确认 confirm_title。"""
    existing = await _get_math_textbook_or_404(db, textbook_id)
    update = {k: v for k, v in (update or {}).items() if v is not None or isinstance(v, (list, dict))}

    # ---- 危险字段：subject_type 跨学科变更 → 必须严格二次确认 ---- #
    new_st = update.get("subject_type")
    if new_st is not None and str(new_st) != existing.get("subject_type"):
        if not (confirm_title or "").strip():
            raise ConfirmationMismatchError(
                ERR_SUBJECT_TYPE_CHANGE_CONFIRM_REQUIRED,
                "变更学科类型（subject_type）需二次确认：入参 confirm_title 必须等于当前教材标题",
            )
        if (confirm_title or "").strip() != existing.get("title"):
            raise ConfirmationMismatchError(
                ERR_SUBJECT_TYPE_CHANGE_CONFIRM_REQUIRED,
                "变更 subject_type 失败：confirm_title 与当前教材标题不匹配",
            )

    # ---- 危险字段：改标题 → 需 confirm_title 匹配新标题 ---- #
    new_title = update.get("title")
    if new_title is not None and str(new_title) != existing.get("title"):
        if (confirm_title or "").strip() != str(new_title).strip():
            raise ConfirmationMismatchError(
                ERR_UPDATE_TITLE_CONFIRM_MISMATCH,
                "修改标题需二次确认：confirm_title 必须与新标题完全一致",
            )

    # ---- 允许更新的字段白名单（其余字段一律忽略，不直接更新 created_at/审计字段） ---- #
    ALLOWED_FIELDS = {
        "title", "grade", "subject_type", "semester",
        "publisher", "cover_url", "isbn", "chapters",
    }
    changes: dict[str, Any] = {}
    for k, v in update.items():
        if k in ALLOWED_FIELDS:
            # subject_type 合法性：如有修改，校验合法集合
            if k == "subject_type":
                v_norm = (v or "").strip() or _math_pkg.DEFAULT_SUBJECT_TYPE
                if v_norm not in _math_pkg.VALID_SUBJECT_TYPES_SET:
                    raise TextbookPayloadError(_math_pkg.ERR_INVALID_SUBJECT_TYPE, f"subject_type 取值仅限 {sorted(_math_pkg.VALID_SUBJECT_TYPES_SET)}")
                # math → 同时校验 semester 存在
                if v_norm == SUBJECT_TYPE_MATH and not update.get("semester") and not existing.get("semester"):
                    raise TextbookPayloadError(_math_pkg.ERR_MATH_SEMESTER_REQUIRED, "切换到数学必须提供 semester(up/down)")
                if v_norm == SUBJECT_TYPE_MATH:
                    sem = (update.get("semester") or existing.get("semester") or "").strip()
                    if sem not in _math_pkg.VALID_MATH_SEMESTERS_SET:
                        raise TextbookPayloadError(_math_pkg.ERR_INVALID_MATH_SEMESTER, f"数学 semester 取值仅限 {sorted(_math_pkg.VALID_MATH_SEMESTERS_SET)}")
                v = v_norm
            # semester：math 必传合法性
            if k == "semester":
                v_norm = (v or "").strip()
                st_for_validate = update.get("subject_type") or existing.get("subject_type")
                if st_for_validate == SUBJECT_TYPE_MATH:
                    if not v_norm:
                        raise TextbookPayloadError(_math_pkg.ERR_MATH_SEMESTER_REQUIRED, "数学必须指定 semester")
                    if v_norm not in _math_pkg.VALID_MATH_SEMESTERS_SET:
                        raise TextbookPayloadError(_math_pkg.ERR_INVALID_MATH_SEMESTER, f"semester 取值仅限 {sorted(_math_pkg.VALID_MATH_SEMESTERS_SET)}")
                v = v_norm
            changes[k] = v

    if not changes:
        # 无变更 → 直接返回原文档（不写审计）
        return existing

    changes["updated_at"] = int(time.time() * 1000)
    await db.update(TEXTBOOK_V2, where={"textbook_id": textbook_id}, data={"$set": changes}, multi=False)

    # 查回最新
    refreshed = await _get_math_textbook_or_404(db, textbook_id)

    # 审计 changed_fields：仅记录「新旧值不同」的字段名
    changed_fields = sorted([k for k in changes.keys() if k != "updated_at" and existing.get(k) != refreshed.get(k)])
    await write_audit(
        db,
        action=AUDIT_ACTION_UPDATE_MATH_TEXTBOOK,
        object_ref=textbook_id,
        actor=actor,
        context={"changed_fields": changed_fields},
    )
    return refreshed


# ===========================================================================
# 4. GET /math/textbook/overview — 概览（6 个 node_stats 字段聚合）
# ===========================================================================


async def get_textbook_overview(db, *, textbook_id: str) -> dict:
    """教材概览：聚合 curriculum_node，返回 6 个统计字段。"""
    tb = await _get_math_textbook_or_404(db, textbook_id)
    q = await db.query(CURRICULUM_NODE_COLLECTION, where={"textbook_id": textbook_id}, limit=5000)
    nodes = q["records"]

    def _has_description(n: dict) -> bool:
        d = n.get("description")
        return bool(d) and (not isinstance(d, dict) or any(d.values()))

    def _has_summary(n: dict) -> bool:
        s = n.get("ai_summary")
        if not s or not isinstance(s, dict):
            return False
        return s.get("status") == "generated" or s.get("manual_edited") in (True, False) and "status" in s

    unit_count = sum(1 for n in nodes if n.get("node_type") == "unit")
    lesson_count = sum(1 for n in nodes if n.get("node_type") == "lesson")
    kp_count = sum(1 for n in nodes if n.get("node_type") == "knowledge_point")
    described_count = sum(1 for n in nodes if _has_description(n))
    summarized_count = sum(1 for n in nodes if _has_summary(n))
    needs_review_count = sum(1 for n in nodes if bool(n.get("needs_review")))

    return {
        "textbook_id": textbook_id,
        "title": tb.get("title"),
        "grade": tb.get("grade"),
        "semester": tb.get("semester"),
        "publisher": tb.get("publisher"),
        "node_stats": {
            "unit_count": unit_count,
            "lesson_count": lesson_count,
            "kp_count": kp_count,
            "described_count": described_count,
            "summarized_count": summarized_count,
            "needs_review_count": needs_review_count,
        },
        "total_nodes": len(nodes),
    }


# ===========================================================================
# 5. POST /math/textbook/import-nodes — 批量导入 curriculum_node（幂等 code）
# ===========================================================================


async def import_curriculum_nodes(
    db,
    *,
    textbook_id: str,
    nodes: list[dict],
    on_duplicate: str = "skip",
    actor: str = "",
) -> dict:
    """批量导入知识点/目录节点。

    幂等键 = code（CURRICULUM_NODE_IDEMPOTENCY_FIELD）：
    - on_duplicate == "skip"（默认）：同 code 不写入
    - on_duplicate == "update"：同 code 覆盖原文档（保留 textbook_id）
    """
    tb = await _get_math_textbook_or_404(db, textbook_id)
    # 1) on_duplicate 合法性
    if (on_duplicate or "").strip() not in VALID_IMPORT_ON_DUPLICATE_SET:
        raise TextbookPayloadError(
            ERR_IMPORT_ON_DUPLICATE_INVALID,
            f"on_duplicate 取值仅限 {sorted(VALID_IMPORT_ON_DUPLICATE_SET)}，收到 {on_duplicate!r}",
        )
    on_duplicate = (on_duplicate or "skip").strip()

    nodes = list(nodes or [])
    errors: list[dict] = []

    # 2) payload 内 code 重复 → 一次性报错（不写任何）
    codes_in_payload = []
    for idx, node in enumerate(nodes):
        code = (node or {}).get(CURRICULUM_NODE_IDEMPOTENCY_FIELD)
        if not code:
            errors.append({"row": idx, "error": f"缺少幂等字段 {CURRICULUM_NODE_IDEMPOTENCY_FIELD!r}"})
            continue
        codes_in_payload.append(str(code))
    if errors:
        raise TextbookPayloadError(
            ERR_IMPORT_NODES_DUPLICATE_CODE,
            f"导入节点缺少 code：{errors}",
        )
    if len(codes_in_payload) != len(set(codes_in_payload)):
        dup = [c for c in codes_in_payload if codes_in_payload.count(c) > 1]
        raise TextbookPayloadError(
            ERR_IMPORT_NODES_DUPLICATE_CODE,
            f"单次导入 payload 中 code 重复：{sorted(set(dup))}",
        )

    # 3) 按每个 code 计算现有节点
    existing_q = await db.query(
        CURRICULUM_NODE_COLLECTION,
        where={CURRICULUM_NODE_IDEMPOTENCY_FIELD: {"$in": codes_in_payload}},
        limit=max(len(codes_in_payload), 1),
    )
    existing_by_code: dict[str, dict] = {}
    for n in existing_q["records"]:
        existing_by_code[str(n.get(CURRICULUM_NODE_IDEMPOTENCY_FIELD, ""))] = n

    inserted_count = skipped_count = updated_count = 0
    now_ms = int(time.time() * 1000)
    # 继承父教材的 grade/semester（节点未指定时）
    inherited = {"grade": tb.get("grade"), "semester": tb.get("semester")}

    for idx, node in enumerate(nodes):
        code = str(node[CURRICULUM_NODE_IDEMPOTENCY_FIELD])
        existing = existing_by_code.get(code)
        # 构建待写 doc
        merged = dict(node)
        merged["textbook_id"] = textbook_id
        for k, v in inherited.items():
            merged.setdefault(k, v)
        merged.setdefault("created_at", existing.get("created_at") if existing else now_ms)
        merged["updated_at"] = now_ms

        if existing is None:
            await db.insert(CURRICULUM_NODE_COLLECTION, merged)
            inserted_count += 1
        elif on_duplicate == "skip":
            skipped_count += 1
        else:  # update
            # 保留原 _id（如有）
            keep_id = existing.get("_id")
            if keep_id:
                merged["_id"] = keep_id
            await db.update(
                CURRICULUM_NODE_COLLECTION,
                where={CURRICULUM_NODE_IDEMPOTENCY_FIELD: code},
                data={"$set": {k: v for k, v in merged.items() if k != "_id"}},
                multi=False,
            )
            updated_count += 1

    stats = {
        "inserted": inserted_count,
        "skipped": skipped_count,
        "updated": updated_count,
        "errors": [],
    }
    await write_audit(
        db,
        action=AUDIT_ACTION_IMPORT_MATH_NODES,
        object_ref=textbook_id,
        actor=actor,
        context={"on_duplicate": on_duplicate, "stats": stats},
    )
    return {"textbook_id": textbook_id, "stats": stats}


# ===========================================================================
# 6. DELETE /math/textbook/{id} — 清理 description+ai_summary（不删节点结构）
# ===========================================================================


async def delete_math_textbook_cleanup(
    db,
    *,
    textbook_id: str,
    confirm_textbook_title: str,
    actor: str = "",
) -> dict:
    """高危清理：仅清 curriculum_node.description + ai_summary，不碰结构字段；需标题二次确认。"""
    tb = await _get_math_textbook_or_404(db, textbook_id)
    cur_title = str(tb.get("title") or "").strip()
    if str(confirm_textbook_title or "").strip() != cur_title:
        raise ConfirmationMismatchError(
            ERR_DELETE_CONFIRM_MISMATCH,
            f"二次确认失败：confirm_textbook_title={confirm_textbook_title!r} ≠ 当前标题={cur_title!r}",
        )

    q = await db.query(
        CURRICULUM_NODE_COLLECTION,
        where={"textbook_id": textbook_id},
        limit=5000,
    )
    nodes = q["records"]
    cleared = 0

    for n in nodes:
        clear_fields: dict[str, Any] = {}
        if n.get("description") is not None and n.get("description") != {}:
            clear_fields["description"] = None
            cleared += 1
        if n.get("ai_summary") is not None:
            clear_fields["ai_summary"] = None
            cleared += 1
        if clear_fields:
            clear_fields["updated_at"] = int(time.time() * 1000)
            await db.update(
                CURRICULUM_NODE_COLLECTION,
                where={"node_id": n["node_id"]},
                data={"$set": clear_fields},
                multi=False,
            )

    await write_audit(
        db,
        action=AUDIT_ACTION_DELETE_MATH_TEXTBOOK,
        object_ref=textbook_id,
        actor=actor,
        context={"cleared_count": cleared, "total_nodes": len(nodes)},
    )
    return {
        "textbook_id": textbook_id,
        "cleared_count": cleared,
        "total_nodes": len(nodes),
    }
