"""教材构建接口 — POST /build/sentence 和 /build/sentence-fixed"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from services.build_sentence import build_textbook_sentences
from services.build_sentence_fixed import build_textbook_fixed
from services.build_nce import build_nce_book, NCE_BOOKS
from services.dependencies import get_db
from services.models_content import (
    LESSON,
    SENTENCE_V2,
    TEXTBOOK_V2,
    build_sentence_v2_doc,
    write_content_v2,
)

logger = logging.getLogger("scholar-admin.routes.build")
router = APIRouter(tags=["构建"])


# ---------------------------------------------------------------------------
# 公共写库逻辑
# ---------------------------------------------------------------------------


async def _write_to_db(
    content: dict,
    textbook_name_fallback: str,
    textbook_id: str | None = None,
) -> dict:
    """将生成的 content 写入新表 textbook_v2 / chapter / lesson / sentence_v2

    Args:
        content: 模型输出的结构化内容 {textbook_info, units}
        textbook_name_fallback: 教材名回退值（从请求参数来的）
        textbook_id: 可选，指定已有的 textbook_v2 记录复用（不创建新记录）

    Returns:
        写入结果摘要 dict，供接口返回
    """
    db = get_db()
    now = int(time.time())
    info = content.get("textbook_info", {})

    # 1. 获取或创建 textbook_v2 记录（未指定则生成新 ID）
    if not textbook_id:
        textbook_id = f"tb_{uuid.uuid4().hex[:16]}"
        logger.info(f"[build] textbook_v2 新建 ID: {textbook_id}")
    else:
        logger.info(f"[build] textbook_v2 复用: {textbook_id}")

    # 2. 组装新表载荷(lesson_id / lesson_title / sentences → chapter/lesson/sentence_v2)
    units = content.get("units", [])
    created_units = []
    v2_units: list[dict] = []

    for u in units:
        unit_index = u.get("unit_index", 1)
        lesson_id = f"lesson_{uuid.uuid4().hex[:16]}"
        sentences = u.get("sentences", [])

        sentence_docs: list[dict] = []
        for i, s in enumerate(sentences):
            sentence_docs.append({
                "sentence_id": f"sent_{uuid.uuid4().hex[:16]}",
                "index": i + 1,
                "text": s.get("text", ""),
                "translation": s.get("translation", ""),
            })

        created_units.append({
            "unit_index": unit_index,
            "lesson_id": lesson_id,
            "lesson_title": u.get("unit_title", ""),
            "sentence_count": len(sentences),
        })
        v2_units.append({
            "lesson_id": lesson_id,
            "lesson_title": u.get("unit_title", ""),
            "sentences": sentence_docs,
        })

    total_sentences = sum(u["sentence_count"] for u in created_units)

    # 3. 写新表(textbook_v2 / chapter / lesson / sentence_v2)
    v2_stats = await write_content_v2(
        db,
        textbook_id=textbook_id,
        textbook_title=info.get("name", textbook_name_fallback),
        grade=info.get("grade", ""),
        level=info.get("semester", ""),
        units=v2_units,
        now=now,
    )
    logger.info(
        f"[build] 完成: textbook={textbook_id}, "
        f"lessons={len(created_units)}, sentences={total_sentences}, "
        f"v2={v2_stats}"
    )

    return {
        "textbook_id": textbook_id,
        "textbook_name": info.get("name", textbook_name_fallback),
        "unit_count": len(created_units),
        "total_sentences": total_sentences,
        "units": created_units,
        "v2": v2_stats,
    }


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------


@router.post("/build/sentence")
async def build_sentence(data: dict):
    """一键生成教材语句 — 指定教材名称，火山模型自动生成所有单元和核心语句

    请求体：
    {
      "textbook_name": "四年级英语上册 秋天 广州版"
    }
    """
    textbook_name = data.get("textbook_name", "").strip()
    if not textbook_name:
        raise HTTPException(status_code=400, detail="缺少参数 textbook_name")

    try:
        gen_result = await build_textbook_sentences(textbook_name)
        if not gen_result["success"] or gen_result["content"] is None:
            raise HTTPException(
                status_code=500,
                detail=gen_result.get("error", "生成失败"),
            )

        write_result = await _write_to_db(gen_result["content"], textbook_name)
        return {"success": True, "data": write_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[build] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


@router.post("/build/sentence-fixed")
async def build_sentence_fixed(data: dict):
    """一键生成固定教材语句 — 教科版广州专用 2024 新版，按年级生成

    请求体：
    {
      "grade": "4"   // 可选，默认 "4"
    }

    年级对照：
    - "3"：三年级上册 9 单元（Letters in Our Life / ... / Review A Music Show）
    - "3b"：三年级下册 9 单元（Get Up / ... / Review Road Helper Day）
    - "4"：四年级上册 8 单元（Come on in / Help yourself / ... / Joy in the Air）
    - "4b"：四年级下册 9 单元（The School Garden / ... / Review A Happy Trip）
    - "5"：五年级上册 8 单元（Learn Words in Chunks / It's for everybody / ... / Let's Go Camping!）
    """
    grade = str(data.get("grade", "4")).strip()
    if grade not in ("3", "3b", "4", "4b", "5"):
        raise HTTPException(status_code=400, detail="grade 仅支持 3、3b、4、4b 或 5")

    from services.build_sentence_fixed import TEXTBOOK_CONFIGS

    cfg = TEXTBOOK_CONFIGS[grade]
    try:
        gen_result = await build_textbook_fixed(grade)
        if not gen_result["success"] or gen_result["content"] is None:
            raise HTTPException(
                status_code=500,
                detail=gen_result.get("error", "生成失败"),
            )

        write_result = await _write_to_db(gen_result["content"], cfg["name"])
        return {"success": True, "data": write_result}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[build-fixed] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


# ---------------------------------------------------------------------------
# 新概念英语 NCE 端点
# ---------------------------------------------------------------------------


@router.post("/build/nce")
async def build_nce(data: dict):
    """导入新概念英语教材原文 — 忠实还原每课正文所有英文语句

    请求体：
    {
      "book": "1",           // 册数：1/2/3/4（必填）
      "start_lesson": 1,     // 起始课号，默认 1
      "end_lesson": 20       // 结束课号，默认该册总课数
    }

    教材对照：
    - "1"：新概念英语第一册 英语初阶（144 课，情景对话为主）
    - "2"：新概念英语第二册 实践与进步（96 课，短篇故事）
    - "3"：新概念英语第三册 培养技能（60 课，中篇文章）
    - "4"：新概念英语第四册 流利英语（48 课，长篇文章）

    注意：单次请求课数过多可能超出模型输出上限，建议每次 10-30 课分批调用。
    """
    book = str(data.get("book", "")).strip()
    if book not in ("1", "2", "3", "4"):
        raise HTTPException(status_code=400, detail="book 仅支持 1、2、3、4")

    cfg = NCE_BOOKS[book]
    start_lesson = int(data.get("start_lesson", 1))
    end_lesson = int(data.get("end_lesson", cfg["total_lessons"]))

    if start_lesson < 1:
        start_lesson = 1
    if end_lesson > cfg["total_lessons"]:
        end_lesson = cfg["total_lessons"]
    if start_lesson > end_lesson:
        raise HTTPException(status_code=400, detail="start_lesson 不能大于 end_lesson")

    try:
        gen_result = await build_nce_book(book, start_lesson, end_lesson)
        if not gen_result["success"] or gen_result["content"] is None:
            raise HTTPException(
                status_code=500,
                detail=gen_result.get("error", "复现失败"),
            )

        # 同一册书复用已有的 textbook_v2 记录
        db = get_db()
        existing = await db.query(
            collection=TEXTBOOK_V2,
            where={"title": cfg["name"]},
            limit=1,
        )
        existing_textbook_id: str | None = None
        if existing.get("records"):
            existing_textbook_id = existing["records"][0]["_id"]
            logger.info(
                f"[build-nce] 复用已有 textbook: {existing_textbook_id}"
            )

        write_result = await _write_to_db(
            gen_result["content"], cfg["name"],
            textbook_id=existing_textbook_id,
        )
        return {
            "success": True,
            "data": write_result,
            "reused_textbook": existing_textbook_id is not None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[build-nce] 异常: {e}")
        raise HTTPException(status_code=500, detail=f"构建失败: {str(e)}")


# ---------------------------------------------------------------------------
# NCE 修复端点 — 补齐缺失 sentence + 补齐 tracking
# ---------------------------------------------------------------------------


@router.post("/build/nce/repair")
async def repair_nce_sentences(data: dict):
    """修复 NCE 教材缺失语句 — 补齐没有 sentence_v2 记录的空 lesson

    请求体：
    {
      "textbook_id": "tb_xxx"   // NCE 教材 ID（必填）
    }

    流程：
    1. 找到该教材下没有 sentence_v2 记录的 lesson（空 lesson）
    2. 从 lesson 标题中提取课号，调用 LLM 复现原文语句
    3. 将新语句写入 sentence_v2 表、更新 lesson 计数
    """
    import re as _re

    textbook_id = str(data.get("textbook_id", "")).strip()
    if not textbook_id:
        raise HTTPException(status_code=400, detail="缺少参数 textbook_id")

    try:
        db = get_db()
        now = int(time.time())

        # ── 1. 验证教材存在（textbook_v2）──
        tb_result = await db.query(
            collection=TEXTBOOK_V2, where={"_id": textbook_id}, limit=1,
        )
        if not tb_result.get("records"):
            raise HTTPException(status_code=404, detail="教材不存在")

        textbook = tb_result["records"][0]
        textbook_title = textbook.get("title", "")
        logger.info(
            f"[repair] 教材: {textbook_title}  (_id={textbook_id})"
        )

        # ── 2. 确认是新概念教材并找到册数 ──
        book_num: str | None = None
        for num, cfg in NCE_BOOKS.items():
            if cfg["name"] in textbook_title:
                book_num = num
                break
        if not book_num:
            raise HTTPException(
                status_code=400,
                detail=f"教材「{textbook_title}」非新概念教材，无法使用 NCE 修复",
            )

        # ── 3. 查询该教材下所有 lesson ──
        lessons_result = await db.query(
            collection=LESSON, where={"textbook_id": textbook_id},
        )
        all_lessons = lessons_result.get("records", [])
        if not all_lessons:
            return {"success": True, "message": "该教材下没有 lesson 记录", "repaired": 0}

        all_lesson_ids = [l.get("lesson_id", "") for l in all_lessons if l.get("lesson_id")]
        all_sents_result = await db.query(
            collection=SENTENCE_V2,
            where={"lesson_id": {"$in": all_lesson_ids}},
        )
        all_sentences = all_sents_result.get("records", [])
        logger.info(
            f"[repair] textbook 下有 {len(all_lessons)} lesson, "
            f"已查到 {len(all_sentences)} 条 sentence_v2"
        )

        # 按 lesson_id 分组统计 sentence_v2 数量
        lesson_sent_count: dict[str, int] = {}
        for s in all_sentences:
            lid = s.get("lesson_id", "")
            if lid:
                lesson_sent_count[lid] = lesson_sent_count.get(lid, 0) + 1

        # ── 4. 找出没有 sentence_v2 的 lesson + 提取课号 ──
        empty_lessons: list[dict] = []
        lesson_to_lesson: dict[int, dict] = {}

        for lesson in all_lessons:
            lesson_id = lesson.get("lesson_id", "")
            if lesson_sent_count.get(lesson_id, 0) == 0:
                empty_lessons.append(lesson)
                title = lesson.get("title", "")
                m = _re.search(r"Lesson\s+(\d+)", title)
                if m:
                    lesson_num = int(m.group(1))
                    lesson_to_lesson[lesson_num] = lesson

        if not empty_lessons:
            return {
                "success": True,
                "message": "所有 lesson 已有语句记录，无需修复",
                "repaired": 0,
                "new_sentences": 0,
            }

        logger.info(
            f"[repair] 发现 {len(empty_lessons)} 个空 lesson，"
            f"共 {len(lesson_to_lesson)} 个可识别课号的 lesson"
        )

        # ── 5. 对课号分组（连续课号合并为一批减少 LLM 调用） ──
        lesson_nums = sorted(lesson_to_lesson.keys())
        batches: list[tuple[int, int]] = []
        bs = be = lesson_nums[0]
        for ln in lesson_nums[1:]:
            if ln == be + 1:
                be = ln
            else:
                batches.append((bs, be))
                bs = be = ln
        batches.append((bs, be))
        logger.info(f"[repair] 课号分组: {batches}")

        # ── 6. 逐批调用 LLM 复现 ──
        all_new_sentences: dict[str, list[dict]] = {}  # lesson_id → sentence dicts
        empty_llm_lessons: list[str] = []  # LLM 返回了 0 句的 lesson（如纯练习课）

        for b_start, b_end in batches:
            logger.info(f"[repair] 生成课号 {b_start}-{b_end} ...")
            gen = await build_nce_book(book_num, b_start, b_end)
            if not gen["success"] or gen["content"] is None:
                logger.error(
                    f"[repair] 课号 {b_start}-{b_end} 生成失败: "
                    f"{gen.get('error')}"
                )
                continue

            for u in gen["content"].get("units", []):
                u_idx = u.get("unit_index", 1)
                ln = b_start + u_idx - 1
                if ln not in lesson_to_lesson:
                    logger.warning(
                        f"[repair] 课号 {ln}（u_idx={u_idx}）不在可识别 lesson 中，跳过"
                    )
                    continue

                lesson_record = lesson_to_lesson[ln]
                lesson_id = lesson_record["lesson_id"]
                title = lesson_record.get("title", f"Lesson {ln}")
                sentences = u.get("sentences", [])

                if not sentences:
                    logger.info(f"[repair] {title} LLM 返回 0 条语句（可能为纯练习课），跳过")
                    empty_llm_lessons.append(lesson_id)
                    continue

                sent_dicts: list[dict] = []
                for i, s in enumerate(sentences):
                    sent_id = f"sent_{uuid.uuid4().hex[:16]}"
                    sent_dicts.append({
                        "sentence_id": sent_id,
                        "index": i + 1,
                        "text": s.get("text", ""),
                        "translation": s.get("translation", ""),
                    })

                all_new_sentences[lesson_id] = sent_dicts
                logger.info(
                    f"[repair] {title} 生成 {len(sent_dicts)} 条语句"
                )

        if empty_llm_lessons:
            logger.info(
                f"[repair] {len(empty_llm_lessons)} 个 lesson LLM 返回 0 句（纯练习课），跳过写入"
            )

        if not all_new_sentences:
            return {
                "success": True,
                "message": (
                    f"{len(empty_lessons)} 个空 lesson 中，"
                    f"{len(empty_llm_lessons)} 个为纯练习课（无需句子），"
                    f"{len(empty_lessons) - len(empty_llm_lessons)} 个生成失败"
                ),
                "data": {
                    "textbook_id": textbook_id,
                    "textbook_title": textbook_title,
                    "empty_lessons_found": len(empty_lessons),
                    "llm_empty_lessons": len(empty_llm_lessons),
                    "units_repaired": 0,
                    "units_failed": len(empty_lessons) - len(empty_llm_lessons),
                    "new_sentences": 0,
                },
            }

        # ── 7. 写入 sentence_v2 + 更新 lesson 计数 ──
        repaired_lesson_ids: list[str] = []
        failed_lesson_ids: list[str] = []
        total_new_sents = 0
        lesson_records_by_id = {l.get("lesson_id"): l for l in all_lessons}

        for lesson_id, sent_dicts in all_new_sentences.items():
            lesson_record = lesson_records_by_id.get(lesson_id, {})
            lesson_title = lesson_record.get("title", "")

            # 7a. 逐条插入 sentence_v2（build_sentence_v2_doc 回填 chapter/lesson/textbook）
            success_sents = 0
            for s in sent_dicts:
                doc = build_sentence_v2_doc(
                    s,
                    chapter_id=lesson_record.get("chapter_id", ""),
                    lesson_id=lesson_id,
                    textbook_id=textbook_id,
                    now=now,
                )
                try:
                    await db.insert(collection=SENTENCE_V2, data=doc)
                    success_sents += 1
                except Exception as e:
                    logger.error(
                        f"[repair] {lesson_title} sentence "
                        f"{s.get('sentence_id')} 插入失败: {e}"
                    )

            # 7b. 验证写入
            try:
                verify = await db.query(
                    collection=SENTENCE_V2,
                    where={"lesson_id": lesson_id},
                    select={"sentence_id": 1},
                )
                actual_count = len(verify.get("records", []))
            except Exception as e:
                logger.error(f"[repair] {lesson_title} 验证查询失败: {e}")
                actual_count = -1

            if actual_count >= len(sent_dicts):
                logger.info(
                    f"[repair] {lesson_title} 写入成功: "
                    f"{success_sents}/{len(sent_dicts)} 条语句, "
                    f"验证实际 {actual_count} 条"
                )

                # 7c. 更新 lesson 的 sentence_count
                try:
                    await db.update(
                        collection=LESSON,
                        where={"lesson_id": lesson_id},
                        data={"$set": {
                            "sentence_count": actual_count,
                            "updated_at": now,
                        }},
                        multi=False,
                    )
                except Exception as e:
                    logger.error(f"[repair] {lesson_title} 更新 lesson 失败: {e}")

                repaired_lesson_ids.append(lesson_id)
                total_new_sents += success_sents
            else:
                logger.error(
                    f"[repair] {lesson_title} 写入异常: "
                    f"期望 {len(sent_dicts)} 条, 实际 {actual_count} 条"
                )
                failed_lesson_ids.append(lesson_id)

        logger.info(
            f"[repair] 写入完成: {len(repaired_lesson_ids)} lesson 已修复, "
            f"{total_new_sents} 条新语句, "
            f"{len(failed_lesson_ids)} lesson 失败"
        )

        # ── 返回结果 ──
        # failed 包括 LLM 返回空（纯练习课）+ 写入失败
        not_repaired = len(empty_lessons) - len(repaired_lesson_ids)
        return {
            "success": True,
            "message": (
                f"修复完成: {len(empty_lessons)} 个空 lesson 中成功修复 "
                f"{len(repaired_lesson_ids)} 个"
                + (
                    f"（{len(empty_llm_lessons)} 个纯练习课无需句子, "
                    f"{not_repaired - len(empty_llm_lessons)} 个写入失败）"
                    if not_repaired else ""
                )
            ),
            "data": {
                "textbook_id": textbook_id,
                "textbook_title": textbook_title,
                "empty_lessons_found": len(empty_lessons),
                "units_repaired": len(repaired_lesson_ids),
                "units_llm_no_sentences": len(empty_llm_lessons),
                "units_failed": len(failed_lesson_ids),
                "new_sentences": total_new_sents,
                "repaired_lesson_ids": repaired_lesson_ids,
                "failed_lesson_ids": failed_lesson_ids,
                "llm_empty_lesson_ids": empty_llm_lessons,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[repair] 修复异常: {e}")
        raise HTTPException(status_code=500, detail=f"修复失败: {str(e)}")
