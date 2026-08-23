"""E1.1 英语语句管理：4 服务函数（list/create/edit/delete + 级联清理）+ M5 语义去重表

对应 4 个 HTTP 路由（routes/english.py）：
1. GET    /english/textbook/{tid}/lessons/{lid}/sentences → list_english_lesson_sentences
2. POST   /english/textbook/{tid}/lessons/{lid}/sentences → create_english_sentences
3. PUT    /english/sentence/{sid}                          → edit_english_sentence
4. DELETE /english/sentence/{sid}                          → delete_english_sentence

M3 G1.2 + M5：ensureSentenceSemanticKey（Lazy dedup，service-contract §8.5；
data-model-contract §4.15 sentence_semantic_key 表），由 POST /tracking/state
上报路径调用（api-contract §3.2 API-G3）。M5 起 registry（sentence_semantic_key）
成为 canonical/duplicate 权威源，sentence_v2 字段保持同步；改句/删句级联维护
registry（脱离旧簇 / canonical 提升 / 簇空删表）。

规格：service-contract.md §8.1 + §8.5；契约：api-contract.md §3.11 E-API-4~E-API-7 + §3.2；
级联删除范围：data-model-contract.md §4.3.1（DM-2，6 表处理方式）+ §4.15。
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
from services.english.structure import load_lesson_entries
from services.models_content import (
    SENTENCE_GROUP,
    SENTENCE_SEMANTIC_KEY,
    build_sentence_semantic_key_doc,
    compute_text_hash,
    get_sentence_semantic_key,
)

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

    lesson 挂载结构兼容三种形态（统一走 load_lesson_entries）：
    - 内嵌标准：textbook.chapters[].lessons[]（契约 §4.2，chapter_id 从所属 chapter 取）
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
                # M5 §4.15：文本变更 → 脱离旧语义簇（registry 去引用 + 清 semantic_key；
                # 新文本的簇由下次上报惰性补齐，Lazy dedup 口径）
                if existing.get("semantic_key"):
                    await _unlink_from_semantic_registry(db, existing)
                    set_fields["semantic_key"] = None
                    set_fields["canonical_sentence_id"] = None

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
         "learning_attempt": n, "audio_asset": n, "conversation_turn_marked": n,
         "sentence_group_refs_removed": n, "semantic_registry_refs_removed": n}
    """
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

    # sentence_group：删除句子的组内引用（data-model §4.3.1 DM-G7：从 sentence_ids[]
    # 移除，**不删组**）。全量拉取 + 内存过滤（与 conversation_turn 同款，避免 DB
    # 数组包含匹配语义差异）。
    try:
        groups = await db.query(SENTENCE_GROUP, where={}, limit=5000)
        for g in groups["records"]:
            sids = list(g.get("sentence_ids") or [])
            if sentence_id not in sids:
                continue
            new_sids = [sid for sid in sids if sid != sentence_id]
            await db.update(
                SENTENCE_GROUP,
                where={"group_id": g.get("group_id")},
                data={"$set": {
                    "sentence_ids": new_sids,
                    "updated_at": int(time.time() * 1000),
                }},
            )
            deleted["sentence_group_refs_removed"] += 1
    except Exception as exc:
        logger.warning(
            f"[english.delete] sentence_group 引用移除失败 sentence_id={sentence_id}: {exc!r}"
        )

    # sentence_semantic_key（M5 §4.15）：删除句脱离语义簇；
    # canonical 被删 → 提升剩余最早者为新 canonical，簇空 → 删 registry。
    try:
        sent = await db.query(SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1)
        if sent["records"]:
            if await _unlink_from_semantic_registry(db, sent["records"][0]):
                deleted["semantic_registry_refs_removed"] = 1
    except Exception as exc:
        logger.warning(
            f"[english.delete] sentence_semantic_key 清理失败 sentence_id={sentence_id}: {exc!r}"
        )

    return deleted


# ===========================================================================
# 5. M5 — ensureSentenceSemanticKey（Lazy dedup + sentence_semantic_key 落表）
# ===========================================================================


async def _load_peers_by_hash(db, key: str, *, exclude: str) -> list[dict]:
    """全量拉取 sentence_v2 后内存过滤同 text_hash 成员（不含 exclude）。

    复用 M3 / delete_duplicates 同款策略：getter（E0.1）惰性注入的 text_hash
    参与过滤，存量无 text_hash 字段的记录同样可靠（真实 DB 的 getter 兼容层
    与 FakeDB 行为一致，见 services.database）。
    """
    all_rows = await db.query(SENTENCE_V2, where={}, limit=5000)
    return [
        r for r in all_rows["records"]
        if (r.get("text_hash") or "") == key
        and r.get("sentence_id")
        and r.get("sentence_id") != exclude
    ]


async def _load_members_by_ids(db, sentence_ids: list[str]) -> dict[str, int]:
    """批量查句子 created_at 映射 {sentence_id: created_at}（canonical 提升选最早者）。"""
    out: dict[str, int] = {}
    if not sentence_ids:
        return out
    for i in range(0, len(sentence_ids), 200):
        res = await db.query(
            SENTENCE_V2,
            where={"sentence_id": {"$in": sentence_ids[i:i + 200]}},
            select={"sentence_id": 1, "created_at": 1},
            limit=5000,
        )
        for r in res.get("records", []):
            sid = r.get("sentence_id")
            if sid:
                out[sid] = int(r.get("created_at") or 0)
    return out


async def _backfill_registry_from_peers(db, s: dict, key: str) -> dict:
    """以 sentence_v2 现有字段为基线构建 registry 文档并落库（冷启动 / M3 存量回填复用）。

    canonical 选取 = M3 口径（`_pick_canonical_id`：已有 canonical 优先，否则
    created_at 最早者；句子自身已带 canonical 时尊重之）；`duplicate_sentence_ids`
    = 同 hash 成员（不含 canonical，含本句若本句为重复句）。

    Returns:
        构建并写入的 registry doc。
    """
    sid = s.get("sentence_id") or ""
    peers = await _load_peers_by_hash(db, key, exclude=sid)
    existing_canonical = s.get("canonical_sentence_id")
    if existing_canonical and existing_canonical != sid:
        canonical_id = existing_canonical
    else:
        canonical_id = _pick_canonical_id([s] + peers, sid)
    dup_ids = [
        p.get("sentence_id") for p in peers
        if p.get("sentence_id") and p.get("sentence_id") != canonical_id
    ]
    if sid and sid != canonical_id and sid not in dup_ids:
        dup_ids.append(sid)
    doc = build_sentence_semantic_key_doc(
        semantic_key=key,
        canonical_sentence_id=canonical_id,
        duplicate_sentence_ids=dup_ids,
        text_hash=key,
        now=int(time.time() * 1000),
    )
    await db.insert(SENTENCE_SEMANTIC_KEY, doc)
    logger.info(
        f"[english.ensure_semantic_key] registry 落库 semantic_key={key[:12]}..., "
        f"canonical={canonical_id}, duplicates={len(dup_ids)}"
    )
    return doc


async def _ensure_registry_for_sentence(db, s: dict) -> None:
    """句子已带 semantic_key → registry 惰性回填（M5 存量兼容，只写 registry 表，不改 sentence_v2）。

    M3 存量：`sentence_v2.semantic_key` / `canonical_sentence_id` 已补齐但
    `sentence_semantic_key` 未建（M3 不落库本表）。本函数按句子字段 + 同 hash 成员
    回填一条 registry，使 M5 读写键统一；registry 已存在则跳过（幂等）。
    """
    key = s.get("semantic_key")
    if not key:
        return
    existing = await get_sentence_semantic_key(db, key)
    if existing:
        return
    await _backfill_registry_from_peers(db, s, key)


async def _unlink_from_semantic_registry(db, s: dict) -> bool:
    """句子（改文本 / 删除）脱离原语义簇，维护 registry（data-model §4.15）。

    - 句子在 `duplicate_sentence_ids` → 移除；
    - 句子是 canonical → 提升剩余成员中 `created_at` 最早者为新 canonical
      （写回其 `sentence_v2.canonical_sentence_id` = 自指）；无剩余成员 → 删 registry；
    - 句子无 semantic_key 或 registry 不存在 → 零操作（幂等）。

    Returns:
        True 表示发生 registry 变更（供级联计数）。
    """
    key = s.get("semantic_key")
    if not key:
        return False
    registry = await get_sentence_semantic_key(db, key)
    if not registry:
        return False
    sid = s.get("sentence_id") or ""
    dup_ids = [x for x in (registry.get("duplicate_sentence_ids") or []) if x != sid]
    canonical = registry.get("canonical_sentence_id")
    now_ms = int(time.time() * 1000)

    if canonical == sid:
        # 本句是 canonical：提升剩余最早者，或删表
        if dup_ids:
            members_created = await _load_members_by_ids(db, dup_ids)
            new_canonical = min(
                dup_ids, key=lambda x: members_created.get(x) or 0
            )
            # canonical 不入 duplicate_sentence_ids（与建簇/登记口径一致）
            new_dup_ids = [x for x in dup_ids if x != new_canonical]
            await db.update(
                SENTENCE_SEMANTIC_KEY,
                where={"_id": key},
                data={"$set": {
                    "canonical_sentence_id": new_canonical,
                    "duplicate_sentence_ids": new_dup_ids,
                    "updated_at": now_ms,
                }},
            )
            await db.update(
                SENTENCE_V2,
                where={"sentence_id": new_canonical},
                data={"$set": {
                    "canonical_sentence_id": new_canonical,
                    "updated_at": now_ms,
                }},
            )
        else:
            await db.delete(SENTENCE_SEMANTIC_KEY, where={"_id": key})
        return True

    if dup_ids != list(registry.get("duplicate_sentence_ids") or []):
        # 本句是重复句：仅从 duplicate_sentence_ids 移除
        await db.update(
            SENTENCE_SEMANTIC_KEY,
            where={"_id": key},
            data={"$set": {
                "duplicate_sentence_ids": dup_ids,
                "updated_at": now_ms,
            }},
        )
        return True
    return False


async def ensureSentenceSemanticKey(
    db,
    *,
    sentence_id: str,
    text: str | None = None,
) -> dict:
    """惰性补齐语句语义键 + 落 `sentence_semantic_key` 表（M5，service-contract §8.5 + data-model §4.15）。

    **Lazy dedup**：仅当 `sentence_v2.semantic_key` 缺失时计算 L1 hash
    （`semantic_key = compute_text_hash(text)`，与 E0.1 `text_hash` 同源）写回：
    - `sentence_semantic_key`（registry）成为 canonical / duplicate 权威源：
      - 簇不存在 → 按 M3 口径建簇（已有 canonical 优先，否则同 hash 组
        `created_at` 最早者为 canonical），本句指向 canonical；
      - 簇已存在 → canonical = registry.canonical_sentence_id，本句指向之并
        登记为重复句（`duplicate_sentence_ids`）；
    - `canonical_sentence_id` 自指 = 本句是 canonical（data-model §4.3）。
    同时写 `sentence_v2` 字段（与 registry 保持一致）；已补齐的句子**零写
    sentence_v2**，仅当 registry 缺失时惰性回填（M3 存量兼容，只写 registry 表）。
    由 `POST /tracking/state` 上报路径调用，冷门句零成本。

    Returns:
        {"sentence_id", "semantic_key", "canonical_sentence_id"}

    Raises:
        SentenceNotFoundError: sentence 不存在（404 SENTENCE_NOT_FOUND）。
    """
    if not sentence_id:
        raise SentenceNotFoundError(sentence_id)
    q = await db.query(SENTENCE_V2, where={"sentence_id": sentence_id}, limit=1)
    if not q["records"]:
        raise SentenceNotFoundError(sentence_id)
    s = q["records"][0]

    # 已补齐 → 零写 sentence_v2（幂等，冷门句零成本）；registry 缺失时回填一次
    if s.get("semantic_key"):
        await _ensure_registry_for_sentence(db, s)
        return {
            "sentence_id": sentence_id,
            "semantic_key": s["semantic_key"],
            "canonical_sentence_id": s.get("canonical_sentence_id"),
        }

    content = (text or "").strip() if text else (s.get("text") or "")
    key = compute_text_hash(content)
    if not key:
        # 空文本：不写语义键（避免空字符串误判重复，契约 §8.3），返回现状
        return {
            "sentence_id": sentence_id,
            "semantic_key": s.get("semantic_key"),
            "canonical_sentence_id": s.get("canonical_sentence_id"),
        }

    # M5：registry（sentence_semantic_key）成为 canonical/duplicate 权威源
    registry = await get_sentence_semantic_key(db, key)
    if registry is None:
        # 冷启动 / M3 存量：扫描 sentence_v2 收集同 hash 成员建簇（仅此一次，后续命中 registry）
        registry = await _backfill_registry_from_peers(db, s, key)
    canonical_id = registry.get("canonical_sentence_id") or sentence_id
    duplicate_ids = list(registry.get("duplicate_sentence_ids") or [])
    if canonical_id != sentence_id and sentence_id not in duplicate_ids:
        duplicate_ids.append(sentence_id)
        await db.update(
            SENTENCE_SEMANTIC_KEY,
            where={"_id": key},
            data={"$set": {
                "duplicate_sentence_ids": duplicate_ids,
                "updated_at": int(time.time() * 1000),
            }},
        )

    await db.update(
        SENTENCE_V2,
        where={"sentence_id": sentence_id},
        data={"$set": {
            "semantic_key": key,
            "canonical_sentence_id": canonical_id,
            "updated_at": int(time.time() * 1000),
        }},
    )
    logger.info(
        f"[english.ensure_semantic_key] sentence_id={sentence_id}, "
        f"semantic_key={key[:12]}..., canonical={canonical_id}, "
        f"duplicate_group_size={len(duplicate_ids) + 1}"
    )
    return {
        "sentence_id": sentence_id,
        "semantic_key": key,
        "canonical_sentence_id": canonical_id,
    }


def _pick_canonical_id(group: list[dict], self_id: str) -> str:
    """同 hash 组内选取 canonical：已有 canonical → 指向之；否则 created_at 最早者。

    规则（service-contract §8.5 / data-model §4.3）：
    - 组成员 `canonical_sentence_id` 自指（== 自身）→ 该句即 canonical，直接采用；
    - 组成员指向组内他句（已加入既有簇）→ 解析到该句；
    - 均无 → 组内 `created_at` 最早者（M3 口径），缺失时回退自身。
    """
    member_ids = {r.get("sentence_id") for r in group}
    for r in group:
        csid = r.get("canonical_sentence_id")
        if csid:
            if csid == r.get("sentence_id"):
                return csid
            if csid in member_ids:
                return csid
    earliest = min(group, key=lambda r: r.get("created_at") or 0)
    return earliest.get("sentence_id") or self_id
