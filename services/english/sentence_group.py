"""M3 G1.3 英语语句分组管理：4 服务函数（list/create/edit/delete + 审计）

对应 4 个 HTTP 路由（routes/english.py，E-API-8~11）：
1. GET    /english/textbook/{tid}/lessons/{lid}/groups → listSentenceGroups
2. POST   /english/textbook/{tid}/lessons/{lid}/groups → createSentenceGroup
3. PUT    /english/group/{group_id}                    → editSentenceGroup
4. DELETE /english/group/{group_id}                    → deleteSentenceGroup

规格：service-contract.md §8.5；契约：api-contract.md §3.11 E-API-8~E-API-11；
数据模型：data-model-contract.md §4.14（sentence_group 集合）。
级联语义：删组 → 成员 sentence_v2.group_id 置 null（**不物理删句**）；
删句（sentence_management.py）→ 从所属组 sentence_ids[] 移除（DM-G7）。
"""
from __future__ import annotations

import logging
import time

from services.audit import (
    AUDIT_ACTION_CREATE_SENTENCE_GROUP,
    AUDIT_ACTION_DELETE_SENTENCE_GROUP,
    AUDIT_ACTION_EDIT_SENTENCE_GROUP,
    write_audit,
)
from services.database import TEXTBOOK_V2
from services.english import (
    ConfirmTextMismatchError,
    GroupNotFoundError,
    LessonNotFoundError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.models_content import (
    SENTENCE_GROUP,
    SENTENCE_V2,
    VALID_SENTENCE_GROUP_TYPES,
    build_sentence_group_doc,
    build_sentence_group_id,
    get_sentence_group,
    get_sentence_groups_by_lesson,
)
from services.english.structure import load_lesson_entries

logger = logging.getLogger("scholar-admin.english.sentence_group")


async def _find_lesson(db, textbook_id: str, lesson_id: str):
    """查英语教材 + 课时，任一不存在抛 404（与 sentence_management._find_lesson 同款）。

    lesson 挂载结构兼容三种形态（统一走 load_lesson_entries）：
    - 内嵌标准：textbook.chapters[].lessons[]（chapter_id 从所属 chapter 取）
    - 内嵌无章：textbook.lessons[]（lesson 直接挂 book 下，chapter_id=''）
    - 独立集合：textbook_v2 无内嵌结构时回退查 chapter/lesson 集合
      （标准内容管线 write_content_v2 产物，见 services/english/structure.py）
    """
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    tb = q["records"][0]
    for entry in await load_lesson_entries(db, tb):
        ls = entry["lesson"]
        if ls.get("lesson_id") == lesson_id:
            return tb, ls, entry["chapter_id"]
    raise LessonNotFoundError(lesson_id)


def _infer_role_in_group(group_type: str, index: int) -> str:
    """按组类型推断句内角色（service-contract §8.5：role_in_group 枚举）。

    - `dialogue_pair`：首句 `question`、次句 `answer_A`、其余 `statement`；
    - `grammar_family` / `vocab_family` / `stand_alone`：全 `statement`。
    """
    if group_type == "dialogue_pair":
        if index == 0:
            return "question"
        if index == 1:
            return "answer_A"
    return "statement"


def _unique_ids(sentence_ids: list[str] | None) -> list[str]:
    """去重保序（空串剔除），None → []。"""
    seen: set[str] = set()
    out: list[str] = []
    for sid in sentence_ids or []:
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


async def _load_lesson_sentences(db, textbook_id: str, lesson_id: str) -> dict[str, dict]:
    """拉取某 lesson 全部 sentence_v2，返回 {sentence_id: doc}（跨 lesson 校验基线）。"""
    q = await db.query(
        SENTENCE_V2,
        where={"textbook_id": textbook_id, "lesson_id": lesson_id},
        limit=2000,
    )
    return {
        r.get("sentence_id"): r
        for r in q["records"]
        if r.get("sentence_id")
    }


async def _write_members(db, sentence_docs: list[dict], *, group_id: str, group_type: str) -> None:
    """对组成员 sentence_v2 写回 group_id + 按 type 推断 role_in_group（建组/改组复用）。"""
    for index, doc in enumerate(sentence_docs):
        await db.update(
            SENTENCE_V2,
            where={"sentence_id": doc.get("sentence_id")},
            data={"$set": {
                "group_id": group_id,
                "role_in_group": _infer_role_in_group(group_type, index),
                "updated_at": int(time.time() * 1000),
            }},
        )


async def _clear_members(db, sentence_ids: list[str]) -> None:
    """成员 sentence_v2.group_id 置 null（删组/改组移除成员，不物理删句）。"""
    for sid in sentence_ids:
        await db.update(
            SENTENCE_V2,
            where={"sentence_id": sid},
            data={"$set": {
                "group_id": None,
                "role_in_group": None,
                "updated_at": int(time.time() * 1000),
            }},
        )


# ===========================================================================
# 1. GET /english/textbook/{tid}/lessons/{lid}/groups — 分组列表（E-API-8）
# ===========================================================================


async def listSentenceGroups(
    db,
    *,
    textbook_id: str,
    lesson_id: str,
    page: int = 1,
    page_size: int = 20,
) -> dict:
    """管理端分组列表（E-API-8，纯读不审）。

    返回 lesson 下全部分组（按 order_in_lesson 升序）+ 组内句子详情 +
    未分组句子计数；groups 内存分页。
    """
    _tb, lesson, _ch_id = await _find_lesson(db, textbook_id, lesson_id)

    groups = await get_sentence_groups_by_lesson(db, lesson_id)
    sentence_by_id = await _load_lesson_sentences(db, textbook_id, lesson_id)

    # 未分组句子数：lesson 内 group_id 为 null / 缺失
    ungrouped_sentences = sum(
        1 for s in sentence_by_id.values()
        if not s.get("group_id")
    )

    total = len(groups)
    start = max((page - 1) * page_size, 0)
    page_groups = groups[start : start + page_size]

    built = []
    for g in page_groups:
        members = []
        for sid in g.get("sentence_ids") or []:
            s = sentence_by_id.get(sid)
            if not s:
                continue
            csid = s.get("canonical_sentence_id")
            members.append({
                "sentence_id": sid,
                "text": s.get("text") or "",
                "translation": s.get("translation") or "",
                "role_in_group": s.get("role_in_group"),
                "is_canonical": not csid or csid == sid,
                "canonical_sentence_id": csid,
            })
        built.append({
            "group_id": g.get("group_id"),
            "title": g.get("title") or "",
            "type": g.get("type"),
            "order_in_lesson": g.get("order_in_lesson"),
            "sentence_ids": list(g.get("sentence_ids") or []),
            "sentences": members,
            "sentence_count": len(g.get("sentence_ids") or []),
            "created_at": g.get("created_at"),
            "updated_at": g.get("updated_at"),
        })

    return {
        "lesson_id": lesson_id,
        "lesson_title": lesson.get("title") or "",
        "ungrouped_sentences": ungrouped_sentences,
        "groups": built,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ===========================================================================
# 2. POST /english/textbook/{tid}/lessons/{lid}/groups — 新建分组（E-API-9）
# ===========================================================================


async def createSentenceGroup(
    db,
    *,
    textbook_id: str,
    lesson_id: str,
    title: str,
    type: str,
    sentence_ids: list[str] | None = None,
    order_in_lesson: int | None = None,
    editor_id: str = "system",
) -> dict:
    """新建分组（E-API-9）。

    title / type 必填（type 枚举 VALID_SENTENCE_GROUP_TYPES）；sentence_ids 可选
    （可先建组后填句），必须全部属于该 lesson，跨 lesson → 400；order_in_lesson
    缺省 = 当前最大 + 1；对成员写回 group_id + 按 type 推断 role_in_group。
    必审 create_sentence_group（必审 21）。
    """
    _tb, _lesson, _ch_id = await _find_lesson(db, textbook_id, lesson_id)

    if not title or not str(title).strip():
        raise SentencePayloadError("title 不能为空")
    title = str(title).strip()
    if type not in VALID_SENTENCE_GROUP_TYPES:
        raise SentencePayloadError(
            f"type={type!r} 仅限 {sorted(VALID_SENTENCE_GROUP_TYPES)}"
        )

    member_ids = _unique_ids(sentence_ids)
    sentence_by_id = await _load_lesson_sentences(db, textbook_id, lesson_id)
    member_docs: list[dict] = []
    for sid in member_ids:
        s = sentence_by_id.get(sid)
        if not s:
            # 契约 §8.5：sentence_ids 必须全部属于该 lesson，跨 lesson / 不存在 → 400
            raise SentencePayloadError(
                f"sentence_id={sid!r} 不属于该 lesson（不存在或跨 lesson）"
            )
        member_docs.append(s)

    if order_in_lesson is None:
        existing = await get_sentence_groups_by_lesson(db, lesson_id)
        order_in_lesson = (
            max((g.get("order_in_lesson") or 0) for g in existing) + 1
            if existing else 0
        )

    now_ms = int(time.time() * 1000)
    group_id = build_sentence_group_id(textbook_id, lesson_id, now=now_ms)
    doc = build_sentence_group_doc(
        group_id=group_id,
        textbook_id=textbook_id,
        lesson_id=lesson_id,
        title=title,
        type_=type,
        sentence_ids=member_ids,
        order_in_lesson=order_in_lesson,
        chapter_id=_ch_id,
        now=now_ms,
    )
    await db.insert(SENTENCE_GROUP, doc)

    if member_docs:
        await _write_members(db, member_docs, group_id=group_id, group_type=type)

    await write_audit(
        db,
        action=AUDIT_ACTION_CREATE_SENTENCE_GROUP,
        object_ref=group_id,
        actor=editor_id,
        context={
            "textbook_id": textbook_id,
            "lesson_id": lesson_id,
            "type": type,
            "member_count": len(member_ids),
        },
    )
    logger.info(
        f"[english.create_group] group_id={group_id}, type={type}, "
        f"order_in_lesson={order_in_lesson}, members={len(member_ids)}"
    )

    return {
        "group_id": group_id,
        "title": title,
        "type": type,
        "order_in_lesson": order_in_lesson,
        "sentence_ids": member_ids,
        "created_at": now_ms,
    }


# ===========================================================================
# 3. PUT /english/group/{group_id} — 编辑分组（E-API-10）
# ===========================================================================


async def editSentenceGroup(
    db,
    *,
    group_id: str,
    title: str | None = None,
    type: str | None = None,
    sentence_ids: list[str] | None = None,
    order_in_lesson: int | None = None,
    editor_id: str = "system",
) -> dict:
    """编辑分组（E-API-10，全可选；sentence_ids 全量替换）。

    - group 不存在 → GroupNotFoundError（404）；
    - `sentence_ids` 为**全量替换**：旧成员 group_id 置 null → 新成员写回
      group_id + 重推 role_in_group；新成员必须属于组的 lesson，跨 lesson → 400；
    - 必审 edit_sentence_group（必审 22，context 含 changed_fields + members 增删）。
    """
    group = await get_sentence_group(db, group_id)
    if not group:
        raise GroupNotFoundError(group_id)

    changed: list[str] = []
    set_fields: dict = {}

    if title is not None:
        if not str(title).strip():
            raise SentencePayloadError("title 不能为空")
        new_title = str(title).strip()
        if new_title != (group.get("title") or ""):
            set_fields["title"] = new_title
            changed.append("title")

    new_type = group.get("type")
    if type is not None:
        if type not in VALID_SENTENCE_GROUP_TYPES:
            raise SentencePayloadError(
                f"type={type!r} 仅限 {sorted(VALID_SENTENCE_GROUP_TYPES)}"
            )
        if type != (group.get("type")):
            set_fields["type"] = type
            new_type = type
            changed.append("type")

    if order_in_lesson is not None:
        if int(order_in_lesson) != (group.get("order_in_lesson") or 0):
            set_fields["order_in_lesson"] = int(order_in_lesson)
            changed.append("order_in_lesson")

    # ---- sentence_ids 全量替换 ---- #
    old_ids = _unique_ids(group.get("sentence_ids") or [])
    members_added: list[str] = []
    members_removed: list[str] = []
    if sentence_ids is not None:
        new_ids = _unique_ids(sentence_ids)
        if new_ids != old_ids:
            members_added = [sid for sid in new_ids if sid not in old_ids]
            members_removed = [sid for sid in old_ids if sid not in new_ids]
            set_fields["sentence_ids"] = new_ids
            changed.append("sentence_ids")

            # 新成员必须属于组的 lesson（跨 lesson → 400）
            sentence_by_id = await _load_lesson_sentences(
                db, group.get("textbook_id") or "", group.get("lesson_id") or ""
            )
            member_docs: list[dict] = []
            for sid in new_ids:
                s = sentence_by_id.get(sid)
                if not s:
                    raise SentencePayloadError(
                        f"sentence_id={sid!r} 不属于该 lesson（不存在或跨 lesson）"
                    )
                member_docs.append(s)
            # 旧成员 group_id 置 null → 新成员写回 + 重推 role_in_group
            if members_removed:
                await _clear_members(db, members_removed)
            if member_docs:
                await _write_members(db, member_docs, group_id=group_id, group_type=new_type or group.get("type"))

    if set_fields:
        set_fields["updated_at"] = int(time.time() * 1000)
        await db.update(
            SENTENCE_GROUP,
            where={"group_id": group_id},
            data={"$set": set_fields},
        )

    await write_audit(
        db,
        action=AUDIT_ACTION_EDIT_SENTENCE_GROUP,
        object_ref=group_id,
        actor=editor_id,
        context={
            "changed_fields": changed,
            "members_added": members_added,
            "members_removed": members_removed,
        },
    )

    updated = await get_sentence_group(db, group_id)
    return {
        "group_id": group_id,
        "title": (updated or group).get("title") or "",
        "type": (updated or group).get("type"),
        "order_in_lesson": (updated or group).get("order_in_lesson"),
        "sentence_ids": list((updated or group).get("sentence_ids") or []),
        "updated_at": (updated or group).get("updated_at") or 0,
    }


# ===========================================================================
# 4. DELETE /english/group/{group_id} — 删除分组（E-API-11）
# ===========================================================================


async def deleteSentenceGroup(
    db,
    *,
    group_id: str,
    confirm_sentence_count: int,
    editor_id: str = "system",
) -> dict:
    """删除分组（E-API-11）。

    - group 不存在 → GroupNotFoundError（404）；
    - confirm_sentence_count 必须等于组内当前语句数，不匹配 → ConfirmTextMismatchError（400）；
    - 成员 sentence_v2.group_id 置 null（**回退未分组，不物理删句**）；
    - 必审 delete_sentence_group（必审 23，context 含 released_sentences + group_title）。
    """
    group = await get_sentence_group(db, group_id)
    if not group:
        raise GroupNotFoundError(group_id)

    member_ids = _unique_ids(group.get("sentence_ids") or [])
    if int(confirm_sentence_count) != len(member_ids):
        raise ConfirmTextMismatchError(
            f"confirm_sentence_count 与组内当前语句数不匹配（当前 {len(member_ids)} 条）"
        )

    if member_ids:
        await _clear_members(db, member_ids)

    await db.delete(SENTENCE_GROUP, where={"group_id": group_id})

    await write_audit(
        db,
        action=AUDIT_ACTION_DELETE_SENTENCE_GROUP,
        object_ref=group_id,
        actor=editor_id,
        context={
            "released_sentences": len(member_ids),
            "group_title": group.get("title") or "",
        },
    )
    logger.info(
        f"[english.delete_group] group_id={group_id}, released_sentences={len(member_ids)}"
    )

    return {
        "group_id": group_id,
        "released_sentences": len(member_ids),
    }
