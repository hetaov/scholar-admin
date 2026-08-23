"""内容模型分层数据访问辅助 —— 全部指向新表(textbook_v2 / chapter / lesson / sentence_v2)

Phase 1 目标:
    textbook_v2 → chapter → lesson → sentence_v2
新表独立创建, 旧表 textbook/sentence/unit/paragraph 迁移结束后下线(Phase 6)。

2026-08-20 SOP G0.1 扩展：
    textbook_v2 多学科化 — subject_type（english/math/chinese）默认 english，
    读侧 getter 兼容（normalize_textbook_doc）、写侧 build_doc 默认值、
    入参校验 validate_textbook_payload（契约 §4.1 + api-contract §3.1）。
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

# 新表集合名
TEXTBOOK_V2 = "textbook_v2"
CHAPTER = "chapter"
LESSON = "lesson"
SENTENCE_V2 = "sentence_v2"

DEFAULT_UNITS_PER_CHAPTER = 8

# G0.1: textbook 多学科常量（避免循环依赖 math.__init__ → math 会 import models_content 中的内容）
# 如后续需要对 math 可见，在 math/__init__.py 重新声明主值；此处保留底层副本，
# 保证 models_content 可被 math 依赖路径上的任何模块使用，不产生循环 import。
_SUBJECT_TYPE_ENGLISH = "english"
_SUBJECT_TYPE_MATH = "math"
_SUBJECT_TYPE_CHINESE = "chinese"
_DEFAULT_SUBJECT_TYPE = _SUBJECT_TYPE_ENGLISH
_VALID_SUBJECT_TYPES_SET = frozenset({_SUBJECT_TYPE_ENGLISH, _SUBJECT_TYPE_MATH, _SUBJECT_TYPE_CHINESE})
_VALID_MATH_SEMESTERS_SET = frozenset({"up", "down"})

_ERR_INVALID_SUBJECT_TYPE = "INVALID_SUBJECT_TYPE"
_ERR_MATH_SEMESTER_REQUIRED = "MATH_TEXTBOOK_SEMESTER_REQUIRED"
_ERR_INVALID_MATH_SEMESTER = "INVALID_MATH_SEMESTER"

logger = logging.getLogger("scholar-admin.models.content")


# ---------------------------------------------------------------------------
# G0.1 — 读侧 getter 兼容（textbook_v2 记录 → 规范化）
# ---------------------------------------------------------------------------


def normalize_textbook_doc(doc: dict) -> dict:
    """读取 textbook_v2 记录后的 getter 兼容层。

    **契约 §4.1**：存量记录无 `subject_type` 字段时，读侧透明注入 `english`，
    不回写 DB，保证零迁移。

    实现原则（避免副作用）：
    - 返回新字典，不修改传入 doc；
    - `subject_type` 缺失 → 注入 `english`；
    - `subject_type` 为 `None` / 空字符串 → 注入 `english`；
    - 显式合法值（english/math/chinese）→ 保留原值；
    - 非法 subject_type → **读侧不抛错**（避免破坏存量场景下的 GET 列表），
      仅注入 `english` 兜底；非法值写入侧由 `validate_textbook_payload` 把关。
    """
    out = dict(doc)  # shallow copy 够了，调用方不依赖引用
    st = out.get("subject_type")
    if not st or st not in _VALID_SUBJECT_TYPES_SET:
        out["subject_type"] = _DEFAULT_SUBJECT_TYPE
    return out


# ---------------------------------------------------------------------------
# E0.1 — sentence_v2 text_hash 惰性计算（读侧 getter 兼容，2026-08-22 SOP ④ DM-1）
# ---------------------------------------------------------------------------

# 句子文本归一化时移除的中英文标点（全角 + 半角）
_SENTENCE_PUNCT_CHARS = "！？。，、!?,."
_SENTENCE_PUNCT_TABLE = {ord(c): None for c in _SENTENCE_PUNCT_CHARS}


def normalize_sentence_text(text: str | None) -> str:
    """句子文本归一化（契约 §4.3 DM-1）。

    规则：strip + toLowerCase + 移除中英文标点(！？。，、!?,.) + 压缩连续空白为单空格。

    用于 L1 hash 重复检测：标点/大小写/空白差异的句子归一为同一指纹。
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = s.translate(_SENTENCE_PUNCT_TABLE)
    s = " ".join(s.split())  # 压缩连续空白（含 tab/newline）为单空格
    return s


def compute_text_hash(text: str | None) -> str:
    """计算句子文本的 sha256 指纹（契约 §4.3 DM-1）。

    `text_hash = sha256(normalize_sentence_text(text).encode('utf-8')).hexdigest()`（64 字符 hex）。
    空文本 → 返回 ''（不计算 hash，避免空字符串误判重复）。
    """
    if not text:
        return ""
    return hashlib.sha256(normalize_sentence_text(text).encode("utf-8")).hexdigest()


def normalize_sentence_doc(doc: dict) -> dict:
    """读取 sentence_v2 记录后的 getter 兼容层（契约 §4.3 DM-1）。

    **存量记录无 `text_hash` 字段时，读侧惰性计算注入（不写回 DB，零迁移）**。

    实现原则（避免副作用，与 normalize_textbook_doc 一致）：
    - 返回新字典，不修改传入 doc；
    - `text_hash` 缺失 / None / 空串 → 按 `text` 惰性计算注入；
    - `text_hash` 有值（非空）→ 保留原值不覆盖；
    - `text` 缺失或空串 → `text_hash = ''`（不计算 hash）。
    """
    out = dict(doc)  # shallow copy 够了，调用方不依赖引用
    th = out.get("text_hash")
    if not th:  # None / "" / missing → 惰性计算
        out["text_hash"] = compute_text_hash(out.get("text", "") or "")
    return out


# ---------------------------------------------------------------------------
# G0.1 — 入参校验（POST/PUT textbook 前）
# ---------------------------------------------------------------------------


class _TextbookPayloadError(ValueError):
    """校验异常（私有本地别名，routes 层如已 import math.TextbookPayloadError 也可直接 except ValueError(code in ...)）。

    为便于测试同时兼容两种风格：
    1. 异常对象带 `code` 属性；
    2. `str(err)` 以 `{CODE}: ` 开头，因此 assert "CODE" in str(err) 也成立。
    """

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def validate_textbook_payload(
    *,
    title: str = "",
    grade: str = "",
    level: str = "",
    subject_type: str | None = None,
    semester: str | None = None,
    publisher: str | None = None,
    cover_url: str | None = None,
    isbn: str | None = None,
    textbook_id: str | None = None,
    chapters: list | None = None,
) -> dict:
    """教材 CRUD 入参校验 + 规范化（契约 api-contract §3.1 POST /textbook）。

    校验规则：
      1. subject_type 未传（None/空串）→ 默认 english；
         传了但 ∉ {english,math,chinese} → 抛 ValueError(code=INVALID_SUBJECT_TYPE)。
      2. subject_type='math' 且缺 semester → 抛 MATH_TEXTBOOK_SEMESTER_REQUIRED。
      3. subject_type='math' 且 semester ∉ {up,down} → 抛 INVALID_MATH_SEMESTER。
      4. 非 math（english/chinese 或默认 english）的 semester = None/空串 → 允许，
         cleaned 结果中**不**输出 semester/publisher/cover_url/isbn 的空值（留作可选，
         契约为 "仅数学生效"，英语侧为空不写入）。

    返回：cleaned dict（仅包含合法的已填键，可选字段空值不强制写入）。
    """
    cleaned: dict = {}

    # 1) 必填基本字段
    if title is not None:
        cleaned["title"] = title
    if grade is not None:
        cleaned["grade"] = grade
    if level is not None:
        cleaned["level"] = level
    if textbook_id is not None:
        cleaned["textbook_id"] = textbook_id
    if chapters is not None:
        cleaned["chapters"] = chapters

    # 2) subject_type 默认 + 合法性
    st_norm = subject_type.strip() if isinstance(subject_type, str) else None
    if not st_norm:
        st_norm = _DEFAULT_SUBJECT_TYPE
    if st_norm not in _VALID_SUBJECT_TYPES_SET:
        raise _TextbookPayloadError(
            _ERR_INVALID_SUBJECT_TYPE,
            f"subject_type 取值仅限 {sorted(_VALID_SUBJECT_TYPES_SET)}，实际={subject_type!r}",
        )
    cleaned["subject_type"] = st_norm

    # 3) semester / publisher / cover_url / isbn
    sem_norm = semester.strip() if isinstance(semester, str) else None
    if st_norm == _SUBJECT_TYPE_MATH:
        if not sem_norm:
            raise _TextbookPayloadError(
                _ERR_MATH_SEMESTER_REQUIRED,
                "subject_type='math' 时必须指定 semester(up/down)",
            )
        if sem_norm not in _VALID_MATH_SEMESTERS_SET:
            raise _TextbookPayloadError(
                _ERR_INVALID_MATH_SEMESTER,
                f"semester 取值仅限 {sorted(_VALID_MATH_SEMESTERS_SET)}，实际={semester!r}",
            )
        cleaned["semester"] = sem_norm
        # math 的可选元数据（只要 truthy 就写入，否则 DB 层保持字段少，兼容英语侧查询时不输出 None）
        if publisher:
            cleaned["publisher"] = publisher
        if cover_url:
            cleaned["cover_url"] = cover_url
        if isbn:
            cleaned["isbn"] = isbn
    # 非 math：semester/publisher/cover_url/isbn 不写入 cleaned，保持英语侧纯净
    # （未来如需语文学科的独立字段，再走新 SOP 扩展，这里不预留）

    return cleaned


# ---------------------------------------------------------------------------
# 查询辅助(全部指向新表)
# ---------------------------------------------------------------------------


async def get_chapters(db, textbook_id: str, limit: int = 1000) -> list[dict]:
    """按教材查章节, 按 order 升序。"""
    result = await db.query(
        collection=CHAPTER,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def get_lessons_by_textbook(db, textbook_id: str, limit: int = 5000) -> list[dict]:
    """按教材查全部课（无章教材: lesson 直接挂 book 下, chapter_id 为空）, 按 order 升序。"""
    result = await db.query(
        collection=LESSON,
        where={"textbook_id": textbook_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def get_sentences_by_lesson(db, lesson_id: str, limit: int = 1000) -> list[dict]:
    """按课查句子, 按 order 升序。"""
    result = await db.query(
        collection=SENTENCE_V2,
        where={"lesson_id": lesson_id},
        order=[{"field": "order", "direction": "asc"}],
        limit=limit,
    )
    return result.get("records", [])


async def query_all_pages(
    db,
    *,
    collection: str,
    where: dict | None = None,
    order: list[dict] | None = None,
    select: dict | None = None,
    page_size: int = 1000,
) -> list[dict]:
    """分页拉取集合全部匹配文档（规避单次 limit 上限）。

    供批量 $in 查询与全量学习数据拉取复用，避免 N+1 逐条查询。
    """
    records: list[dict] = []
    offset = 0
    while True:
        page = await db.query(
            collection=collection,
            where=where,
            order=order,
            select=select,
            offset=offset,
            limit=page_size,
        )
        recs = page.get("records", [])
        records.extend(recs)
        if len(recs) < page_size:
            break
        offset += page_size
    return records


async def get_lessons_by_chapter_ids(
    db,
    chapter_ids: list[str],
    page_size: int = 1000,
) -> list[dict]:
    """按多个章节批量查课（$in，每批 200 防 $in 数组过大），按 order 升序。

    替代逐章 get_lessons 的 N+1 查询。
    """
    if not chapter_ids:
        return []
    records: list[dict] = []
    for i in range(0, len(chapter_ids), 200):
        records.extend(await query_all_pages(
            db,
            collection=LESSON,
            where={"chapter_id": {"$in": chapter_ids[i:i + 200]}},
            order=[{"field": "order", "direction": "asc"}],
            page_size=page_size,
        ))
    return records


async def get_sentences_by_lesson_ids(
    db,
    lesson_ids: list[str],
    page_size: int = 1000,
) -> list[dict]:
    """按多个课批量查句子（$in，每批 200 防 $in 数组过大），按 order 升序。

    替代逐课 get_sentences_by_lesson 的 N+1 查询。
    """
    if not lesson_ids:
        return []
    records: list[dict] = []
    for i in range(0, len(lesson_ids), 200):
        records.extend(await query_all_pages(
            db,
            collection=SENTENCE_V2,
            where={"lesson_id": {"$in": lesson_ids[i:i + 200]}},
            order=[{"field": "order", "direction": "asc"}],
            page_size=page_size,
        ))
    return records


async def get_sentences_by_ids(
    db,
    sentence_ids: list[str],
    page_size: int = 1000,
) -> list[dict]:
    """按多个句子 ID 批量查句子（$in，每批 200 防 $in 数组过大）。

    供学者级调度接口（review-plan / weakness-plan）跨课/跨教材一次性
    加载候选句子内容（text / translation / lesson_id），避免逐句 N+1。
    """
    if not sentence_ids:
        return []
    records: list[dict] = []
    for i in range(0, len(sentence_ids), 200):
        records.extend(await query_all_pages(
            db,
            collection=SENTENCE_V2,
            where={"sentence_id": {"$in": sentence_ids[i:i + 200]}},
            page_size=page_size,
        ))
    return records


async def get_textbook_v2(db, textbook_id: str) -> dict | None:
    """按主键查教材 v2, 不存在返回 None。"""
    result = await db.query(collection=TEXTBOOK_V2, where={"_id": textbook_id}, limit=1)
    records = result.get("records", [])
    return records[0] if records else None


# ---------------------------------------------------------------------------
# 纯函数: 文档构建(可单测, 不触网)
# ---------------------------------------------------------------------------


def group_units_into_chapters(
    units: list[dict],
    units_per_chapter: int = DEFAULT_UNITS_PER_CHAPTER,
) -> list[dict]:
    """把 units 按顺序分成若干章。

    Returns:
        [{"chapter_index": 1, "units": [unit, ...]}, ...]
        units_per_chapter <= 0 时全部归入第 1 章。
    """
    if not units:
        return []
    size = units_per_chapter if units_per_chapter and units_per_chapter > 0 else len(units)
    groups = []
    for start in range(0, len(units), size):
        groups.append({
            "chapter_index": len(groups) + 1,
            "units": units[start:start + size],
        })
    return groups


def build_textbook_v2_doc(
    textbook_id: str,
    title: str,
    grade: str = "",
    level: str = "",
    chapter_count: int = 0,
    lesson_count: int = 0,
    sentence_count: int = 0,
    now: int | None = None,
    # 2026-08-20 SOP G0.1 扩展：多学科元数据（契约 §4.1）
    subject_type: str | None = None,
    semester: str | None = None,
    publisher: str | None = None,
    cover_url: str | None = None,
    isbn: str | None = None,
) -> dict:
    """旧 textbook → textbook_v2(全量复制 + version=1 + 冗余计数)。

    2026-08-20 G0.1 扩展：
      - `subject_type` 缺省 = english（契约 §4.1），存量调用零修改自动写入 english；
      - `semester/publisher/cover_url/isbn` 仅当 subject_type='math' 或 调用方显式传
        非 None 时写入（英语侧默认不写入，DB 无多余字段）。
    """
    now = now or int(time.time())
    # subject_type 默认 english
    st_norm = subject_type.strip() if isinstance(subject_type, str) else None
    if not st_norm:
        st_norm = _DEFAULT_SUBJECT_TYPE
    if st_norm not in _VALID_SUBJECT_TYPES_SET:
        # build_doc 是内部辅助函数：非法值不给 routes 层抛错的机会，
        # 写入侧的合法性应由 validate_textbook_payload 前置把关。
        # 这里做一个兜底：非法值 → 注入 english，避免写入脏数据。
        st_norm = _DEFAULT_SUBJECT_TYPE

    doc = {
        "_id": textbook_id,
        "textbook_id": textbook_id,
        "title": title,
        "grade": grade,
        "level": level,
        "version": 1,
        "chapter_count": chapter_count,
        "lesson_count": lesson_count,
        "sentence_count": sentence_count,
        "created_at": now,
        "updated_at": now,
        "subject_type": st_norm,
    }
    # math 的可选元数据（仅 math 时写入）
    if st_norm == _SUBJECT_TYPE_MATH:
        if semester:
            doc["semester"] = semester
        if publisher:
            doc["publisher"] = publisher
        if cover_url:
            doc["cover_url"] = cover_url
        if isbn:
            doc["isbn"] = isbn
    return doc


def build_chapter_doc(
    chapter_id: str,
    textbook_id: str,
    order: int,
    title: str,
    lesson_count: int,
    now: int | None = None,
) -> dict:
    now = now or int(time.time())
    return {
        "_id": chapter_id,
        "chapter_id": chapter_id,
        "textbook_id": textbook_id,
        "order": order,
        "title": title,
        "lesson_count": lesson_count,
        "created_at": now,
    }


def build_lesson_doc(
    lesson_id: str,
    chapter_id: str,
    textbook_id: str,
    order: int,
    title: str,
    sentence_count: int,
    now: int | None = None,
) -> dict:
    now = now or int(time.time())
    return {
        "_id": lesson_id,
        "lesson_id": lesson_id,
        "chapter_id": chapter_id,
        "textbook_id": textbook_id,
        "order": order,
        "title": title,
        "sentence_count": sentence_count,
        "created_at": now,
        "updated_at": now,
    }


def build_sentence_v2_doc(
    sentence_doc: dict,
    chapter_id: str,
    lesson_id: str,
    textbook_id: str,
    now: int | None = None,
) -> dict:
    """sentence → sentence_v2(全量复制 + 回填 chapter_id / lesson_id / textbook_id)。"""
    now = now or int(time.time())
    return {
        "_id": sentence_doc["sentence_id"],
        "sentence_id": sentence_doc["sentence_id"],
        "textbook_id": textbook_id,
        "chapter_id": chapter_id,
        "lesson_id": lesson_id,
        "order": sentence_doc.get("index", sentence_doc.get("order", 1)),
        "text": sentence_doc.get("text", ""),
        "translation": sentence_doc.get("translation", ""),
        "audio_url": sentence_doc.get("audio_url", ""),
        "knowledge_point_ids": sentence_doc.get("knowledge_point_ids", []),
        "created_at": now,
        "updated_at": now,
    }


# ---------------------------------------------------------------------------
# 双写辅助: 将一次构建的结果写入新表
# ---------------------------------------------------------------------------


async def write_content_v2(
    db,
    *,
    textbook_id: str,
    textbook_title: str,
    grade: str = "",
    level: str = "",
    units: list[dict],
    now: int | None = None,
    units_per_chapter: int = DEFAULT_UNITS_PER_CHAPTER,
    chapterless: bool = False,
) -> dict:
    """将构建的内容双写进新表(textbook_v2 + chapter + lesson + sentence_v2)。

    Args:
        textbook_id: 教材 ID; 为空时不写 textbook_v2(视觉识别等无教材场景)。
        units: 每个单元含 lesson_id / lesson_title / sentences(句子 doc 列表)。
               句子 doc 需含 sentence_id / index / text / translation 等字段。
        units_per_chapter: 每章包含的课数; 仅 chapterless=False 时生效。
        chapterless: True 时不创建 chapter, lesson 直接挂在 book 下(chapter_id 为空)。

    Returns:
        {"chapter_count": n, "lesson_count": n, "sentence_count": n}
    """
    now = now or int(time.time())

    # 分批构建时 order 不与已有记录冲突: 以已有 lesson 数量为偏移
    existing_lessons = await db.query(
        collection=LESSON, where={"textbook_id": textbook_id}, select={"_id": 1},
    )
    lesson_offset = len(existing_lessons.get("records", []))

    # 2. 构建 chapter / lesson / sentence_v2 文档
    chapter_docs: list[dict] = []
    lesson_docs: list[dict] = []
    sentence_docs: list[dict] = []

    def _append_lesson(u: dict, chapter_id: str) -> None:
        lesson_id = u["lesson_id"]
        unit_sentences = u.get("sentences", [])
        lesson_docs.append(build_lesson_doc(
            lesson_id,
            chapter_id,
            textbook_id,
            lesson_offset + len(lesson_docs) + 1,
            u.get("lesson_title", f"Lesson {len(lesson_docs) + 1}"),
            len(unit_sentences),
            now,
        ))
        for s in unit_sentences:
            sentence_docs.append(build_sentence_v2_doc(
                s, chapter_id, lesson_id, textbook_id, now,
            ))

    if chapterless:
        # 无章教材: Book → Lesson → Sentence
        for u in units:
            _append_lesson(u, "")
    else:
        # 有章教材: Book → Chapter → Lesson → Sentence
        existing_chapters = await db.query(
            collection=CHAPTER, where={"textbook_id": textbook_id}, select={"_id": 1},
        )
        chapter_offset = len(existing_chapters.get("records", []))

        groups = group_units_into_chapters(units, units_per_chapter)
        for g in groups:
            chapter_id = f"chapter_{uuid.uuid4().hex[:16]}"
            chapter_docs.append(build_chapter_doc(
                chapter_id,
                textbook_id,
                chapter_offset + g["chapter_index"],
                f"Chapter {chapter_offset + g['chapter_index']}",
                len(g["units"]),
                now,
            ))
            for u in g["units"]:
                _append_lesson(u, chapter_id)

    # 3. 写 textbook_v2(幂等 upsert, 计数累加)
    tb_doc = build_textbook_v2_doc(
        textbook_id, textbook_title, grade=grade, level=level,
        chapter_count=len(chapter_docs),
        lesson_count=len(lesson_docs),
        sentence_count=len(sentence_docs),
        now=now,
    )
    if textbook_id:
        existing_tb = await get_textbook_v2(db, textbook_id)
        if existing_tb:
            await db.update(
                collection=TEXTBOOK_V2,
                where={"_id": textbook_id},
                data={"$set": {
                    "chapter_count": int(existing_tb.get("chapter_count", 0) or 0) + len(chapter_docs),
                    "lesson_count": int(existing_tb.get("lesson_count", 0) or 0) + len(lesson_docs),
                    "sentence_count": int(existing_tb.get("sentence_count", 0) or 0) + len(sentence_docs),
                    "updated_at": now,
                }},
                multi=False,
            )
        else:
            await db.insert(collection=TEXTBOOK_V2, data=tb_doc)

    # 4. 写 chapter / lesson / sentence_v2
    if chapter_docs:
        await db.insert(collection=CHAPTER, data=chapter_docs)
    if lesson_docs:
        await db.insert(collection=LESSON, data=lesson_docs)
    if sentence_docs:
        await db.insert(collection=SENTENCE_V2, data=sentence_docs)

    logger.info(
        f"[models_content] 新表写入完成: textbook_v2={textbook_id}, "
        f"chapters={len(chapter_docs)}, lessons={len(lesson_docs)}, "
        f"sentences={len(sentence_docs)}"
    )
    return {
        "chapter_count": len(chapter_docs),
        "lesson_count": len(lesson_docs),
        "sentence_count": len(sentence_docs),
    }
