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
    textbook_id: str | None = None,
) -> dict:
    """将识别结果存入新表 chapter / lesson / sentence_v2（无教材关联时不写 textbook_v2）"""
    now = int(time.time())
    lesson_id = f"lesson_{uuid.uuid4().hex[:16]}"

    sentences = result.get("sentences", [])

    sentence_docs: list[dict] = []
    for i, s in enumerate(sentences):
        sentence_docs.append({
            "sentence_id": f"sent_{uuid.uuid4().hex[:16]}",
            "index": s.get("index", i + 1),
            "text": s.get("text", ""),
            "translation": s.get("translation", ""),
            "level": s.get("level", ""),
            "keywords": s.get("keywords", []),
        })

    v2_stats = await write_content_v2(
        db,
        textbook_id=textbook_id or "",
        textbook_title=result.get("title", ""),
        units=[{
            "lesson_id": lesson_id,
            "lesson_title": result.get("title", ""),
            "sentences": sentence_docs,
        }],
        now=now,
        units_per_chapter=1,
    )

    logger.info(
        f"[存储] lesson={lesson_id}, sentences={len(sentences)}, "
        f"textbook_id={textbook_id or ''}, image_source={image_source}, "
        f"v2={v2_stats}"
    )

    return {
        "lesson_id": lesson_id,
        "sentence_count": len(sentences),
        "v2": v2_stats,
    }


# ==================== 文件上传识别 ====================


@router.post("/vision/recognize")
async def recognize_image(
    file: UploadFile = File(...),
    textbook_id: str | None = Form(None, alias="textbookId"),
):
    """识别英文教材图片，提取语句并返回结构化 JSON

    参数：
    - file: 图片文件（必填）
    - textbook_id: 关联的教材 ID（选填，表单字段名 textbookId）
    """
    _validate_image_ext(file.filename or "")

    try:
        contents = await file.read()
        service = get_vision()
        db = get_db()

        result = service.recognize(image_bytes=contents)
        store_info = await _store_recognition_result(db, result, "upload", textbook_id)

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
