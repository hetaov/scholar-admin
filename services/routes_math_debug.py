"""数学错题识别 Admin 调试干跑路由（api-contract §3.15）

路由前缀 `/math/scan/debug`，仅一个 `POST /recognize`。
仅在 `MATH_SCAN_DEBUG_ENABLED=1` 时由 main.py 条件 include；
鉴权走 `require_debug_access`（env 开关 + token 双重门控，B02）。

**不落库**：handler 调 `recognize_error_scan_dry_run`（B04~B06），不写任何集合。
**真实计费**：OCR + Judge 为真实付费调用。

设计文档：docs_v1/AI错题/数学AI错题识别-设计文档.md §4.4
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from services.dependencies import get_db
from services.infra.auth import require_debug_access
from services.math.error_scan_debug import recognize_error_scan_dry_run
from services.math.error_scanner import (
    ImageTooLargeError,
    ImageValidationError,
    JudgeNotConfiguredError,
    JudgeResponseError,
    ScanUploadError,
)
from services.math.ocr import OcrError
from services.routes.math import _scan_upload_error_to_http

logger = logging.getLogger("scholar-admin.routes.math_debug")

router = APIRouter(prefix="/math/scan/debug", tags=["math-scan-debug"])


@router.post("/recognize", dependencies=[Depends(require_debug_access)])
async def math_scan_debug_recognize(
    image: UploadFile = File(..., description="错题图片（multipart 文件）"),
    textbook_id: str = Form("", description="教材 id（空=全量候选）"),
    scholar_id: str = Form("scholar_debug_01", description="学者 id（干跑用，默认占位）"),
    confidence_threshold: float = Form(0.6, description="置信度阈值（默认 0.6）"),
    compare_all_candidates: bool = Form(False, description="是否对照全量候选（双倍 Judge 消耗）"),
    include_ocr_text: bool = Form(True, description="是否在 debug.ocr 中返回完整 OCR 文本"),
    db=Depends(get_db),
):
    """POST 干跑识别（api-contract §3.15）

    链路：校验图片 → OCR（真实付费）→ 加载候选集（只读）→ Judge（真实付费）
    → 组装 items[] + decisions[] + debug 全段。**不落库、不传云存储、不写审计**。

    出参：{success, data: {scan_id, status, items[], debug{...}}}
    """
    image_bytes = await image.read()
    filename = image.filename or "upload.jpg"
    try:
        data = await recognize_error_scan_dry_run(
            db,
            image_bytes=image_bytes,
            filename=filename,
            textbook_id=textbook_id,
            scholar_id=scholar_id,
            confidence_threshold=confidence_threshold,
            compare_all_candidates=compare_all_candidates,
            include_ocr_text=include_ocr_text,
        )
        return {"success": True, "data": data}
    except (ImageValidationError, ImageTooLargeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OcrError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except (JudgeNotConfiguredError, JudgeResponseError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ScanUploadError as e:
        raise _scan_upload_error_to_http(e)
