"""火山引擎 AI 图片识别接口"""

from __future__ import annotations

import base64
import json as json_lib
import logging
import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import VOLCANO_IMAGE_FORMATS
from services.dependencies import ImageUrlRequest, RecognizeBase64Request, get_db, get_vision
from services.models_content import write_content_v2

logger = logging.getLogger("scholar-admin.routes.vision")
router = APIRouter(tags=["AI 识别"])


# ==================== 存储逻辑 ====================


async def _store_recognition_result(
    db,
    result: dict,
    image_source: str,
    text_book_id: str | None = None,
) -> dict:
    """将识别结果存入 unit / paragraph / sentence 三张表"""
    now = int(time.time())
    unit_id = f"unit_{uuid.uuid4().hex[:16]}"
    paragraph_id = f"para_{uuid.uuid4().hex[:16]}"

    sentences = result.get("sentences", [])
    sentence_ids = [f"sent_{uuid.uuid4().hex[:16]}" for _ in sentences]

    # 1. 写入 unit
    unit_doc = {
        "unit_id": unit_id,
        "title": result.get("title", ""),
        "material_type": result.get("material_type", "other"),
        "language": result.get("language", "en"),
        "summary": result.get("summary", ""),
        "image_source": image_source,
        "total_sentences": len(sentences),
        "paragraph_count": 1,
        "text_book_id": text_book_id or "",
        "created_at": now,
        "updated_at": now,
    }
    await db.insert(collection="unit", data=unit_doc)

    # 2. 写入 paragraph
    paragraph_doc = {
        "paragraph_id": paragraph_id,
        "unit_id": unit_id,
        "index": 1,
        "sentence_ids": sentence_ids,
        "sentence_count": len(sentences),
        "created_at": now,
    }
    await db.insert(collection="paragraph", data=paragraph_doc)

    # 3. 逐条写入 sentence(旧表照旧)
    sentence_docs: list[dict] = []
    for i, s in enumerate(sentences):
        sentence_doc = {
            "sentence_id": sentence_ids[i],
            "unit_id": unit_id,
            "paragraph_id": paragraph_id,
            "index": s.get("index", i + 1),
            "text": s.get("text", ""),
            "translation": s.get("translation", ""),
            "level": s.get("level", ""),
            "keywords": s.get("keywords", []),
            "text_book_id": text_book_id or "",
            "created_at": now,
        }
        sentence_docs.append(sentence_doc)
        await db.insert(collection="sentence", data=sentence_doc)

    # 4. 双写新表(chapter / lesson / sentence_v2), 无教材时不写 textbook_v2
    textbook_id = text_book_id or ""
    v2_stats = await write_content_v2(
        db,
        textbook_id=textbook_id,
        textbook_title=result.get("title", ""),
        units=[{
            "unit_id": unit_id,
            "unit_title": result.get("title", ""),
            "sentences": sentence_docs,
        }],
        now=now,
        units_per_chapter=1,
    )

    logger.info(
        f"[存储] unit={unit_id}, sentences={len(sentences)}, "
        f"text_book_id={text_book_id}, image_source={image_source}, "
        f"v2={v2_stats}"
    )

    return {
        "unit_id": unit_id,
        "paragraph_id": paragraph_id,
        "sentence_count": len(sentences),
        "v2": v2_stats,
    }


# ==================== 文件上传识别 ====================


@router.post("/vision/recognize")
async def recognize_image(
    file: UploadFile = File(...),
    text_book_id: str | None = Form(None, alias="textbookId"),
):
    """识别英文教材图片，提取语句并返回结构化 JSON

    参数：
    - file: 图片文件（必填）
    - text_book_id: 关联的教材 ID（选填，表单字段名 textbookId）
    """
    _validate_image_ext(file.filename or "")

    try:
        contents = await file.read()
        service = get_vision()
        db = get_db()

        result = service.recognize(image_bytes=contents)
        store_info = await _store_recognition_result(db, result, "upload", text_book_id)

        return {
            "success": True,
            "data": result,
            "store": store_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ==================== URL 图片识别 ====================


@router.post("/vision/recognize-url")
async def recognize_image_url(body: ImageUrlRequest):
    """通过 URL 识别英文教材图片"""
    if not body.url:
        raise HTTPException(status_code=400, detail="缺少 url 参数")

    try:
        service = get_vision()
        db = get_db()

        result = service.recognize(image_url=body.url)
        store_info = await _store_recognition_result(db, result, "url")

        return {
            "success": True,
            "data": result,
            "store": store_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ==================== Base64 图片识别 ====================


@router.post("/vision/recognize-base64")
async def recognize_image_base64(body: RecognizeBase64Request):
    """通过 Base64 字符串识别英文教材图片"""
    if not body.base64:
        raise HTTPException(status_code=400, detail="缺少 base64 数据")

    try:
        raw = base64.b64decode(body.base64)
        service = get_vision()
        db = get_db()

        buf = raw  # bytes
        result = service.recognize(image_bytes=buf)

        store_info = await _store_recognition_result(db, result, "base64")

        return {
            "success": True,
            "data": result,
            "store": store_info,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")


# ==================== 工具 ====================


def _validate_image_ext(filename: str) -> None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in VOLCANO_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的图片格式: .{ext}，仅支持 {', '.join(VOLCANO_IMAGE_FORMATS)}",
        )
