"""E1.1 英语语句管理：4 服务函数（list/create/edit/delete + 级联清理）

对应 4 个 HTTP 路由（routes/english.py）：
1. GET    /english/textbook/{tid}/lessons/{lid}/sentences → list_english_lesson_sentences
2. POST   /english/textbook/{tid}/lessons/{lid}/sentences → create_english_sentences
3. PUT    /english/sentence/{sid}                          → edit_english_sentence
4. DELETE /english/sentence/{sid}                          → delete_english_sentence

规格：service-contract.md §8.1；契约：api-contract.md §3.11 E-API-4~E-API-7；
级联删除范围：data-model-contract.md §4.3.1（DM-2，6 表处理方式）。
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from services.audit import (
    AUDIT_ACTION_CREATE_ENGLISH_SENTENCES,
    AUDIT_ACTION_DELETE_ENGLISH_SENTENCE,
    AUDIT_ACTION_EDIT_ENGLISH_SENTENCE,
    write_audit,
)
from services.database import SENTENCE_V2, TEXTBOOK_V2
from services.english import (
    ConfirmTextMismatchError,
    LessonNotFoundError,
    SentenceNotFoundError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.models_content import compute_text_hash

logger = logging.getLogger("scholar-admin.english.sentence_management")

# 级联物理删除的关联集合（where sentence_id 精确匹配）
_CASCADE_DELETE_COLLECTIONS = (
    "study_attempt",
    "skill_state",
    "speech_evaluation",
    "learning_attempt",
)
# 关联数据计数集合（E-API-4 related_data）
_RELATED_COUNT_COLLECTIONS = (
    "study_attempt",
    "skill_state",
    "speech_evaluation",
    "learning_attempt",
    "audio_asset",
)


def _gen_sentence_id() -> str:
    """生成 sentence_id：毫秒时间戳 + 随机后缀（幂等唯一）。"""
    return f"s_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


async def _find_lesson(db, textbook_id: str, lesson_id: str):
    """查英语教材 + 课时，任一不存在抛 404。返回 (textbook, lesson, chapter_id)。

    lesson 挂载结构兼容两种形态：
    - 标准：textbook.chapters[].lessons[]（契约 §4.2，chapter_id 从所属 chapter 取）
    - 无章教材：textbook.lessons[]（lesson 直接挂 book 下，chapter_id=''）
    """
    if not textbook_id:
        raise TextbookNotFoundError(textbook_id)
    q = await db.query(TEXTBOOK_V2, where={"textbook_id": textbook_id}, limit=1)
    if not q["records"]:
        raise TextbookNotFoundError(textbook_id)
    tb = q["records"][0]
    for ch in tb.get("chapters") or []:
        for ls in ch.get("lessons") or []:
            if ls.get("lesson_id") == lesson_id:
                return tb, ls, ch.get("chapter_id") or ""
    for ls in tb.get("lessons") or []:
        if ls.get("lesson_id") == lesson_id:
            return tb, ls, ""
    raise LessonNotFoundError(lesson_id)


# ===========================================================================
# 1. GET /english/textbook/{tid}/lessons/{lid}/sentences — 语句列表
# ===========================================================================


async def list_english_lesson_sentences(
    db,
    *,
    textbook_id: str,
    lesson_id: str,
    keyword: str | None = None,
    duplicate_only: bool = False,
    page: int = 1,
    page_size: int = 50,
) -> dict:
    """管理端语句列表（E-API-4）。

    返回 lesson 下全部语句 + duplicate 统计 + 关联数据计数 + 分页。
    text_hash 由 getter（E0.1）在读侧注入；keyword 模糊匹配 / duplicate_only
    过滤 / 分页在内存完成（句子量级 ≤ 数百条，避免 DB 正则与分页耦合）。
    """
    _tb, lesson, _ch_id = await _find_lesson(db, textbook_id, lesson_id)

    q = await db.query(
        SENTENCE_V2,
        where={"textbook_id": textbook_id, "lesson_id": lesson_id},
        limit=2000,
    )
    rows = q["records"]

    # ---- duplicate 统计（基于全量，不受 keyword 过滤影响）---- #
    hash_groups: dict[str, list[str]] = {}
    for r in rows:
        h = r.get("text_hash") or ""
        if h:
            hash_groups.setdefault(h, []).append(r.get("sentence_id") or "")
    duplicate_of: dict[str, list[str]] = {}
    for h, sids in hash_groups.items():
        if len(sids) > 1:
            for sid in sids:
                duplicate_of[sid] = [x for x in sids if x != sid]

    # ---- keyword 过滤（text 模糊，大小写不敏感）---- #
    if keyword and keyword.strip():
        kw = keyword.strip().lower()
        rows = [r for r in rows if kw in (r.get("text") or "").lower()]

    # ---- duplicate_only 过滤 ---- #
    if duplicate_only:
        rows = [r for r in rows if duplicate_of.get(r.get("sentence_id") or "")]

    total = len(rows)
    start = max((page - 1) * page_size, 0)
    page_rows = rows[start : start + page_size]

    # ---- related_data 各表计数（每条 $in 批量查询后内存分组）---- #
    page_sids = [r.get("sentence_id") or "" for r in page_rows]
    related_counts: dict[str, dict[str, int]] = {
        sid: {c: 0 for c in _RELATED_COUNT_COLLECTIONS} for sid in page_sids
    }
    if page_sids:
        for coll in _RELATED_COUNT_COLLECTIONS:
            try:
                res = await db.query(
                    coll,
                    where={"sentence_id": {"$in": page_sids}},
                    select={"sentence_id": 1},
                    limit=5000,
                )
                for r in res["records"]:
                    sid = r.get("sentence_id")
                    if sid in related_counts:
                        related_counts[sid][coll] += 1
            except Exception as exc:  # 关联表缺失/查询失败不阻断列表
                logger.warning(
                    f"[english.list] 关联表 {coll!r} 计数失败: {exc!r}"
                )

    sentences = []
    for r in page_rows:
        sid = r.get("sentence_id") or ""
        sentences.append(
            {
                "sentence_id": sid,
                "text": r.get("text") or "",
                "translation": r.get("translation") or "",
                "audio_url": r.get("audio_url") or "",
                "knowledge_point_ids": list(r.get("knowledge_point_ids") or []),
                "text_hash": r.get("text_hash") or "",
                "duplicate_count": len(duplicate_of.get(sid, [])),
                "duplicate_sentence_ids": duplicate_of.get(sid, []),
                "related_data": {
                    f"{c}_count": related_counts[sid].get(c, 0)
                    for c in _RELATED_COUNT_COLLECTIONS
                },
            }
        )

    return {
        "lesson_id": lesson_id,
        "lesson_title": lesson.get("title") or "",
        "sentences": sentences,
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ===========================================================================
# 2. POST /english/textbook/{tid}/lessons/{lid}/sentences — 新增语句
# ===========================================================================


async def create_english_sentences(
    db,
    *,
    textbook_id: str,
    lesson_id: str,
    sentences: list[dict],
    editor_id: str = "system",
) -> dict:
    """批量新增语句（E-API-7）。

    自动生成 sentence_id + text_hash；同 lesson 内 text_hash 重复 → 跳过并计数。
    """
    _tb, _lesson, _ch_id = await _find_lesson(db, textbook_id, lesson_id)

    if not isinstance(sentences, list) or not sentences:
        raise SentencePayloadError("sentences 必须是非空数组")

    # 该 lesson 已存在的 text_hash 集合（重复检测基线）
    existing = await db.query(
        SENTENCE_V2,
        where={"textbook_id": textbook_id, "lesson_id": lesson_id},
        limit=2000,
    )
    seen_hashes: set[str] = {
        r.get("text_hash") or "" for r in existing["records"] if r.get("text_hash")
    }

    now_ms = int(time.time() * 1000)
    docs: list[dict] = []
    skipped = 0
    for item in sentences:
        raw = item.get("text") if isinstance(item, dict) else None
        if not raw or not str(raw).strip():
            raise SentencePayloadError("每条语句 text 不能为空")
        text = str(raw).strip()
        text_hash = compute_text_hash(text)
        if not text_hash or text_hash in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(text_hash)
        docs.append(
            {
                "sentence_id": _gen_sentence_id(),
                "text": text,
                "translation": (item.get("translation") or "").strip(),
                "audio_url": item.get("audio_url") or "",
                "knowledge_point_ids": list(item.get("knowledge_point_ids") or []),
                "text_hash": text_hash,
                "textbook_id": textbook_id,
                "lesson_id": lesson_id,
                "chapter_id": _ch_id,
                "created_at": now_ms,
                "updated_at": now_ms,
            }
        )

    if docs:
        await db.insert(SENTENCE_V2, docs)

    await write_audit(
        db,
        action=AUDIT_ACTION_CREATE_ENGLISH_SENTENCES,
        object_ref=f"{textbook_id}+{lesson_id}",
        actor=editor_id,
        context={
            "textbook_id": textbook_id,
            "lesson_id": lesson_id,
            "created": len(docs),
            "skipped_duplicates": skipped,
        },
    )

    return {
        "created": len(docs),
        "skipped_duplicates": skipped,
        "sentences": [
            {"sentence_id": d["sentence_id"], "text": d["text"], "text_hash": d["text_hash"]}
            for d in docs
        ],
    }


# ===========================================================================
# 3. PUT /english/sentence/{sentence_id} — 编辑语句
# ===========================================================================


async def edit_english_sentence(
    db,
    *,
    sentence_id: str,
    text: str | None = None,
    translation: str | None = None,
    audio_url: str | None = None,
    knowledge_point_ids: list[str] | None = None,
    editor_id: str = "system",
) -> dict:
    """编辑语句（E-API-6）。字段白名单 {text, translation, audio_url, knowledge_point_ids}，
    text 变更后重算 text_hash。
    """
    if not sentence_id:
        raise SentenceNotFoundError(sentence_id)
    q = await db.query(SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1)
    if not q["records"]:
        raise SentenceNotFoundError(sentence_id)
    existing = q["records"][0]

    set_fields: dict[str, Any] = {}
    changed: list[str] = []
    text_hash_changed = False

    if text is not None:
        if not str(text).strip():
            raise SentencePayloadError("text 不能为空")
        new_text = str(text).strip()
        if new_text != (existing.get("text") or ""):
            set_fields["text"] = new_text
            changed.append("text")
            new_hash = compute_text_hash(new_text)
            if new_hash != (existing.get("text_hash") or ""):
                set_fields["text_hash"] = new_hash
                text_hash_changed = True

    if translation is not None:
        new_val = str(translation).strip() if translation else ""
        if new_val != (existing.get("translation") or ""):
            set_fields["translation"] = new_val
            changed.append("translation")

    if audio_url is not None:
        new_val = str(audio_url).strip() if audio_url else ""
        if new_val != (existing.get("audio_url") or ""):
            set_fields["audio_url"] = new_val
            changed.append("audio_url")

    if knowledge_point_ids is not None:
        new_val = list(knowledge_point_ids or [])
        if new_val != (existing.get("knowledge_point_ids") or []):
            set_fields["knowledge_point_ids"] = new_val
            changed.append("knowledge_point_ids")

    if set_fields:
        set_fields["updated_at"] = int(time.time() * 1000)
        await db.update(
            SENTENCE_V2,
            where={"sentence_id": sentence_id},
            data={"$set": set_fields},
        )

    await write_audit(
        db,
        action=AUDIT_ACTION_EDIT_ENGLISH_SENTENCE,
        object_ref=sentence_id,
        actor=editor_id,
        context={"changed_fields": changed, "text_hash_changed": text_hash_changed},
    )

    # 读回最新记录返回
    q2 = await db.query(SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1)
    updated = q2["records"][0]
    return {
        "sentence_id": updated.get("sentence_id") or sentence_id,
        "text": updated.get("text") or "",
        "translation": updated.get("translation") or "",
        "audio_url": updated.get("audio_url") or "",
        "knowledge_point_ids": list(updated.get("knowledge_point_ids") or []),
        "text_hash": updated.get("text_hash") or "",
        "updated_at": updated.get("updated_at") or 0,
    }


# ===========================================================================
# 4. DELETE /english/sentence/{sentence_id} — 删除 + 级联清理
# ===========================================================================


async def delete_english_sentence(
    db,
    *,
    sentence_id: str,
    confirm_text: str,
    delete_audio_asset: bool = False,
    delete_duplicates: bool = False,
    editor_id: str = "system",
) -> dict:
    """删除语句 + 级联清理 6 表（E-API-5，data-model §4.3.1 DM-2）。

    - 二次确认：confirm_text.strip() 必须等于 text.strip()，否则 400。
    - 级联：study_attempt / skill_state / speech_evaluation / learning_attempt 物理删；
      audio_asset 仅当 delete_audio_asset=true；conversation_turn 标记 deleted_sentence_ref。
    - delete_duplicates=true：同 text_hash 的其他语句递归级联删除。
    """
    if not sentence_id:
        raise SentenceNotFoundError(sentence_id)
    q = await db.query(SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1)
    if not q["records"]:
        raise SentenceNotFoundError(sentence_id)
    sentence = q["records"][0]

    if (confirm_text or "").strip() != (sentence.get("text") or "").strip():
        raise ConfirmTextMismatchError(
            "confirm_text 与语句原文不匹配，需输入完整句子文本"
        )

    deleted = await _cascade_delete_sentence(
        db, sentence_id=sentence_id, delete_audio_asset=delete_audio_asset
    )
    deleted["sentence_v2"] = 1
    await db.delete(SENTENCE_V2, where={"sentence_id": sentence_id})

    duplicates_deleted = 0
    text_hash = sentence.get("text_hash") or ""
    if delete_duplicates and text_hash:
        # 按 textbook 查全量后内存过滤 hash（getter 注入的 text_hash 不参与 DB where 过滤，
        # 对存量无 text_hash 字段的记录同样可靠）
        dup_q = await db.query(
            SENTENCE_V2,
            where={"textbook_id": sentence.get("textbook_id") or ""},
            limit=5000,
        )
        dup_sids = [
            r.get("sentence_id")
            for r in dup_q["records"]
            if (r.get("text_hash") or "") == text_hash
            and r.get("sentence_id")
            and r.get("sentence_id") != sentence_id
        ]
        for sid in dup_sids:
            sub = await _cascade_delete_sentence(
                db, sentence_id=sid, delete_audio_asset=delete_audio_asset
            )
            sub["sentence_v2"] = 1
            await db.delete(SENTENCE_V2, where={"sentence_id": sid})
            duplicates_deleted += 1
            for k, v in sub.items():
                if k != "sentence_v2":
                    deleted[k] = deleted.get(k, 0) + v

    await write_audit(
        db,
        action=AUDIT_ACTION_DELETE_ENGLISH_SENTENCE,
        object_ref=sentence_id,
        actor=editor_id,
        context={
            "deleted": deleted,
            "duplicates_deleted": duplicates_deleted,
            "confirm_text_match": True,
        },
    )

    return {
        "sentence_id": sentence_id,
        "deleted": deleted,
        "duplicates_deleted": duplicates_deleted,
    }


async def _cascade_delete_sentence(
    db, *, sentence_id: str, delete_audio_asset: bool = False
) -> dict:
    """单条语句的级联清理（不含 sentence_v2 本身）。

    Returns:
        {"study_attempt": n, "skill_state": n, "speech_evaluation": n,
         "learning_attempt": n, "audio_asset": n, "conversation_turn_marked": n}
    """
    deleted: dict[str, int] = {
        "study_attempt": 0,
        "skill_state": 0,
        "speech_evaluation": 0,
        "learning_attempt": 0,
        "audio_asset": 0,
        "conversation_turn_marked": 0,
    }

    # 物理删 4 表（where sentence_id 精确匹配）
    for coll in _CASCADE_DELETE_COLLECTIONS:
        try:
            res = await db.delete(coll, where={"sentence_id": sentence_id})
            deleted[coll] = int(res.get("deleted_count", 0) or 0)
        except Exception as exc:
            logger.warning(
                f"[english.delete] 级联删除 {coll!r} 失败 sentence_id={sentence_id}: {exc!r}"
            )

    # audio_asset：仅当 delete_audio_asset=true 物理删（默认保留，其他句可能复用音频）
    if delete_audio_asset:
        try:
            res = await db.delete("audio_asset", where={"sentence_id": sentence_id})
            deleted["audio_asset"] = int(res.get("deleted_count", 0) or 0)
        except Exception as exc:
            logger.warning(
                f"[english.delete] 级联删除 audio_asset 失败 sentence_id={sentence_id}: {exc!r}"
            )

    # conversation_turn：标记 deleted_sentence_ref=true，不物理删（保留会话上下文）
    try:
        turns = await db.query("conversation_turn", where={}, limit=5000)
        for t in turns["records"]:
            utterance = t.get("utterance") or ""
            reply = t.get("reply") or ""
            if sentence_id in utterance or sentence_id in reply:
                if not t.get("deleted_sentence_ref"):
                    await db.update(
                        "conversation_turn",
                        where={"turn_id": t.get("turn_id")},
                        data={"$set": {"deleted_sentence_ref": True}},
                    )
                    deleted["conversation_turn_marked"] += 1
    except Exception as exc:
        logger.warning(
            f"[english.delete] conversation_turn 标记失败 sentence_id={sentence_id}: {exc!r}"
        )

    return deleted
