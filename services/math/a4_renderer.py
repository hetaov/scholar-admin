"""A4 练习纸渲染管线（F3.2）

契约：
- ADR-0010 决策 4：HTML/CSS + Headless Chrome → PDF（可同时产出 PNG 预览）；
  渲染失败降级为纯文本模板（A-14）
- data-model-contract.md §4.12.6 sheet_render_job（status / artifacts / error_code / retries）
- api-contract.md §3.10：练习纸产物 file_refs{pdf,png,preview} + 家长核对二维码 qrcode_ref
- ADR-0010 A-13：二维码 ≥20×20mm，内容 {sheet_id, signature, expires_at}

说明：
- 渲染引擎：playwright（Headless Chromium）。依赖**延迟导入**：未安装 / Chromium 缺失时
  渲染任务置 failed（error_code=dependency_missing），不影响主链路（生成接口正常返回）。
- 状态机：queued → rendering → success / failed / degraded；retries 上限 1 次，
  失败先降级为纯文本模板重试，仍失败才置 failed（A-14）。
- 产物：RENDER_OUTPUT_DIR/{sheet_id}/sheet.pdf + sheet.png + preview.png，
  file_refs 填 RENDER_STATIC_URL_PREFIX 前缀的 URL；渲染成功后回写 practice_sheet。
- 二维码：HMAC-SHA256(sheet_id + expires_at) 签名（密钥 SHEET_QR_SECRET），
  qrcode_ref = {qr_url, signature, expires_at}；二维码图片嵌入 PDF 页脚（扫码对答案）。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import io
import logging
import os
import time
from pathlib import Path

from config import (
    RENDER_OUTPUT_DIR,
    RENDER_STATIC_URL_PREFIX,
    RENDER_TIMEOUT_SECONDS,
    SHEET_QR_SCAN_PAGE,
    SHEET_QR_SECRET,
    SHEET_QR_TTL_SECONDS,
)
from services.database import (
    PRACTICE_SHEET_COLLECTION,
    SHEET_RENDER_JOB_COLLECTION,
)

logger = logging.getLogger("scholar-admin.math.a4_renderer")

# ---------------------------------------------------------------------------
# 常量（契约 §4.12.6）
# ---------------------------------------------------------------------------

# sheet_render_job.status 状态机（queued 由 F3.1 practice_sheet 写入）
RENDER_JOB_QUEUED = "queued"
RENDER_JOB_RENDERING = "rendering"
RENDER_JOB_SUCCESS = "success"
RENDER_JOB_FAILED = "failed"
RENDER_JOB_DEGRADED = "degraded"

# 错误码（契约 §4.12.6 error_code：重试上限 / 字体缺失 / 超时）
ERROR_DEPENDENCY_MISSING = "dependency_missing"
ERROR_FONT_MISSING = "font_missing"
ERROR_TIMEOUT = "timeout"
ERROR_RENDER_FAILED = "render_failed"
ERROR_RETRY_EXCEEDED = "retry_exceeded"

# 重试上限（契约：retries 上限 1 次，超限降级）
RENDER_RETRY_LIMIT = 1

# A4 版式（mm）
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
PAGE_MARGIN_MM = 12

# 产物文件名
PDF_FILENAME = "sheet.pdf"
PNG_FILENAME = "sheet.png"
PREVIEW_FILENAME = "preview.png"

# 页脚二维码尺寸（ADR-0010 A-13：≥20×20mm）
QR_SIZE_MM = 20


# ---------------------------------------------------------------------------
# 业务异常
# ---------------------------------------------------------------------------


class RenderError(Exception):
    """渲染管线错误基类"""


class RendererUnavailableError(RenderError):
    """playwright / Chromium 不可用（未安装或启动失败）"""


class RenderJobNotFoundError(RenderError):
    """sheet_render_job 不存在"""


class SheetNotFoundError(RenderError):
    """practice_sheet 不存在"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _html_escape(text: str) -> str:
    """题干/答案转义：防 HTML 注入，保留数学符号（Unicode 直接渲染）"""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _qrcode_ref(sheet_id: str) -> dict:
    """家长核对二维码引用（ADR-0010 A-13：含签名与有效期）

    signature = HMAC-SHA256(SHEET_QR_SECRET, f"{sheet_id}:{expires_at}")
    密钥未配置时降级：signature 为空串、qr_url 为空，二维码区域显示"未配置"。
    """
    now = _now_ms()
    expires_at = now + SHEET_QR_TTL_SECONDS * 1000
    signature = ""
    if SHEET_QR_SECRET:
        signature = hmac.new(
            SHEET_QR_SECRET.encode("utf-8"),
            f"{sheet_id}:{expires_at}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    qr_url = (
        f"{SHEET_QR_SCAN_PAGE}?sheet_id={sheet_id}"
        f"&expires_at={expires_at}&signature={signature}"
        if signature
        else ""
    )
    return {"qr_url": qr_url, "signature": signature, "expires_at": expires_at}


def _qrcode_data_uri(text: str) -> str:
    """qrcode 库生成二维码 PNG → data URI（延迟导入，未安装抛 RendererUnavailableError）"""
    try:
        import qrcode  # noqa: WPS433
    except ImportError as e:
        raise RendererUnavailableError("qrcode 未安装，无法生成二维码") from e
    try:
        qr = qrcode.QRCode(border=1)
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:  # noqa: BLE001
        raise RendererUnavailableError(f"二维码生成失败: {e}") from e


def _difficulty_band(difficulty) -> str:
    """奥数扩展题难度档文案（与 F3.1 _band_to_difficulty 反向映射）"""
    try:
        d = int(difficulty)
    except (TypeError, ValueError):
        return ""
    return {3: "入门", 4: "普及", 5: "竞赛"}.get(d, "")


# ---------------------------------------------------------------------------
# A4 HTML 模板
# ---------------------------------------------------------------------------


def build_sheet_html(sheet: dict, qr_data_uri: str = "") -> str:
    """practice_sheet 落库文档 → A4 版式 HTML

    版式（任务卡 F3.2）：
    - 页头：标题（模板 + 学生信息 + 日期）
    - 题面区：题号 + 题干 + 奥数题标注难度档 + 作答区留白
    - 末页答案页（防背题：答案独立一页，供家长核对）
    - 页脚：页码 + 扫码对答案二维码（≥20mm）
    """
    items = sheet.get("items") or []
    scholar_id = sheet.get("scholar_id") or ""
    generated_at = sheet.get("generated_at") or sheet.get("created_at") or 0
    date_str = time.strftime("%Y-%m-%d", time.localtime(generated_at / 1000))
    tpl = (sheet.get("template_ref") or {}).get("template_type") or "standard"
    tpl_name = {"standard": "标准版式", "practice": "巩固练习版式"}.get(tpl, tpl)

    rows: list[str] = []
    for idx, it in enumerate(items, start=1):
        question = _html_escape(it.get("question") or "")
        target_error = it.get("target_error") or ""
        difficulty = it.get("difficulty")
        band = _difficulty_band(difficulty)
        badge = ""
        if band:
            badge = f'<span class="badge ext">奥数·{band}</span>'
        elif target_error:
            badge = f'<span class="badge err">错因:{_html_escape(target_error)}</span>'
        rows.append(
            '<div class="q">'
            f'<div class="q-head"><span class="q-no">{idx}.</span>'
            f'<span class="q-body">{question}</span>{badge}</div>'
            f'<div class="answer-area"></div>'
            "</div>"
        )

    # 末页答案页（防背题：答案独立成页，家长扫码核对）
    answers: list[str] = []
    for idx, it in enumerate(items, start=1):
        ans = _html_escape(it.get("answer") or "")
        hint = _html_escape(it.get("hint_card") or "")
        hint_html = f'<div class="hint">提示:{hint}</div>' if hint else ""
        answers.append(
            f'<div class="ans"><span class="q-no">{idx}.</span>{ans}{hint_html}</div>'
        )

    qr_html = ""
    if qr_data_uri:
        qr_html = (
            f'<div class="qr-wrap"><img class="qr" src="{qr_data_uri}" '
            f'alt="扫码核对答案"/><div class="qr-tip">扫码核对答案</div></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<style>
@page {{ size: A4; margin: {PAGE_MARGIN_MM}mm; }}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; color: #222; font-size: 14px; line-height: 1.7; }}
.header {{ border-bottom: 2px solid #222; padding-bottom: 6px; margin-bottom: 10px; }}
.header .title {{ font-size: 22px; font-weight: 700; text-align: center; letter-spacing: 2px; }}
.header .meta {{ font-size: 12px; color: #555; margin-top: 6px; display: flex; justify-content: space-between; }}
.q {{ margin-bottom: 14px; page-break-inside: avoid; }}
.q-head {{ display: flex; align-items: baseline; }}
.q-no {{ font-weight: 700; margin-right: 6px; }}
.q-body {{ flex: 1; }}
.badge {{ display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 8px; margin-left: 8px; vertical-align: middle; }}
.badge.ext {{ background: #fdecea; color: #c0392b; border: 1px solid #e8b4ad; }}
.badge.err {{ background: #fef9e7; color: #b7950b; border: 1px solid #f3e2a9; }}
.answer-area {{ height: 26mm; margin-top: 6px; border-bottom: 1px dashed #bbb; }}
.answers-page {{ page-break-before: always; }}
.answers-page .title {{ font-size: 18px; font-weight: 700; text-align: center; margin-bottom: 10px; }}
.ans {{ margin-bottom: 8px; page-break-inside: avoid; }}
.hint {{ font-size: 12px; color: #888; margin-left: 20px; }}
.footer {{ margin-top: 14px; display: flex; align-items: center; justify-content: space-between; font-size: 12px; color: #666; }}
.qr-wrap {{ text-align: center; }}
.qr {{ width: {QR_SIZE_MM}mm; height: {QR_SIZE_MM}mm; }}
.qr-tip {{ font-size: 11px; color: #888; }}
</style>
</head>
<body>
<div class="header">
  <div class="title">数学练习纸</div>
  <div class="meta">
    <span>学生:{_html_escape(scholar_id)}</span>
    <span>日期:{date_str}</span>
    <span>版式:{tpl_name}</span>
  </div>
</div>
{''.join(rows)}
<div class="answers-page">
  <div class="title">参考答案（家长核对）</div>
  {''.join(answers)}
</div>
<div class="footer">
  <span>家长可扫码核对答案与解析</span>
  {qr_html}
</div>
</body>
</html>"""


def build_degraded_html(sheet: dict) -> str:
    """降级纯文本模板（ADR-0010 A-14：渲染失败时保证仍可打印）"""
    items = sheet.get("items") or []
    scholar_id = sheet.get("scholar_id") or ""
    rows = "\n".join(
        f"{idx}. {it.get('question') or ''}" for idx, it in enumerate(items, start=1)
    )
    answers = "\n".join(
        f"{idx}. {it.get('answer') or ''}" for idx, it in enumerate(items, start=1)
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<style>
@page {{ size: A4; margin: 12mm; }}
body {{ font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; font-size: 13px; line-height: 1.8; color: #222; }}
h2 {{ text-align: center; }}
.meta {{ color: #555; font-size: 12px; }}
.answer-block {{ page-break-before: always; }}
pre {{ white-space: pre-wrap; font-family: inherit; }}
</style>
</head>
<body>
<h2>数学练习纸（纯文本版式）</h2>
<div class="meta">学生:{_html_escape(scholar_id)}</div>
<pre>{rows}</pre>
<div class="answer-block">
<h2>参考答案（家长核对）</h2>
<pre>{answers}</pre>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# playwright 渲染（延迟导入）
# ---------------------------------------------------------------------------


def _playwright_available() -> bool:
    """检查 playwright + Chromium 是否可用（不抛异常）"""
    try:
        import playwright  # noqa: F401, WPS433
    except ImportError:
        return False
    return True


async def _render_artifacts(html: str, sheet_dir: Path) -> None:
    """Headless Chromium 渲染 HTML → sheet.pdf + sheet.png + preview.png

    依赖延迟导入：未安装 playwright 抛 RendererUnavailableError（调用方降级为 failed）。
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise RendererUnavailableError("playwright 未安装，无法渲染练习纸") from e

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--force-color-profile=srgb",
                ]
            )
            page = await browser.new_page(
                viewport={"width": 794, "height": 1123}  # A4 @96dpi 近似
            )
            await page.set_content(html, wait_until="load")
            pdf_path = str(sheet_dir / PDF_FILENAME)
            await page.pdf(
                path=pdf_path,
                format="A4",
                print_background=True,
                margin={"top": f"{PAGE_MARGIN_MM}mm", "bottom": f"{PAGE_MARGIN_MM}mm",
                        "left": f"{PAGE_MARGIN_MM}mm", "right": f"{PAGE_MARGIN_MM}mm"},
            )
            png_path = str(sheet_dir / PNG_FILENAME)
            await page.screenshot(path=png_path, full_page=True)
            preview_path = str(sheet_dir / PREVIEW_FILENAME)
            await page.screenshot(path=preview_path, full_page=False)
            await browser.close()
    except Exception as e:  # noqa: BLE001
        raise RenderError(f"Headless Chromium 渲染失败: {e}") from e


# ---------------------------------------------------------------------------
# 渲染任务状态机
# ---------------------------------------------------------------------------


async def renderSheetJob(db, sheet_id: str) -> dict:
    """消费单个 sheet_render_job：queued → rendering → success / failed / degraded

    流程：
    1. 查询 sheet_render_job（按 sheet_id，最新一条）
    2. 查询 practice_sheet；构建 HTML（含二维码）
    3. playwright 渲染 → 产物写入 RENDER_OUTPUT_DIR/{sheet_id}/
    4. 回写 job（success + artifacts + retries）与 sheet（file_refs + qrcode_ref）
    5. 失败：retries+1；≤RENDER_RETRY_LIMIT 时降级纯文本模板重试（degraded），
       超过上限或降级失败 → failed（error_code 区分依赖缺失/字体缺失/超时/重试超限）

    Args:
        db: CloudBaseNoSQLClient
        sheet_id: practice_sheet.sheet_id

    Returns:
        job 文档（含最终 status / artifacts / error_code）
    """
    jobs = await db.query(
        SHEET_RENDER_JOB_COLLECTION,
        where={"sheet_id": sheet_id},
        order=[{"field": "created_at", "direction": "desc"}],
        limit=1,
    )
    job = (jobs.get("records") or [None])[0]
    if not job:
        raise RenderJobNotFoundError(f"渲染任务不存在: sheet_id={sheet_id}")

    # 已成功 / 渲染中：幂等直接返回
    if job.get("status") in (RENDER_JOB_SUCCESS, RENDER_JOB_RENDERING):
        return job

    sheets = await db.query(
        PRACTICE_SHEET_COLLECTION,
        where={"sheet_id": sheet_id},
        limit=1,
    )
    sheet = (sheets.get("records") or [None])[0]
    if not sheet:
        raise SheetNotFoundError(f"练习纸不存在: sheet_id={sheet_id}")

    retries = int(job.get("retries") or 0)
    now = _now_ms()

    # 1. 标记 rendering
    await db.update(
        SHEET_RENDER_JOB_COLLECTION,
        {"sheet_id": sheet_id},
        {
            "$set": {
                "status": RENDER_JOB_RENDERING,
                "error_code": "",
                "updated_at": now,
            }
        },
    )

    sheet_dir = Path(RENDER_OUTPUT_DIR) / sheet_id
    sheet_dir.mkdir(parents=True, exist_ok=True)

    # 二维码引用（渲染前生成；密钥未配置时 qr_url 为空）
    qr_ref = _qrcode_ref(sheet_id)

    try:
        # 2. 完整版式渲染（含二维码，二维码缺失不阻塞：降级为文字提示）
        qr_uri = ""
        try:
            qr_uri = _qrcode_data_uri(qr_ref["qr_url"] or sheet_id)
        except RendererUnavailableError as e:
            logger.warning("[renderSheetJob] 二维码生成降级: %s", e)
        html = build_sheet_html(sheet, qr_data_uri=qr_uri)
        await asyncio.wait_for(
            _render_artifacts(html, sheet_dir),
            timeout=RENDER_TIMEOUT_SECONDS,
        )
    except (asyncio.TimeoutError, RenderError) as e:
        # 3. 失败：重试一次（降级纯文本模板）
        retries += 1
        if retries <= RENDER_RETRY_LIMIT:
            logger.warning(
                "[renderSheetJob] 渲染失败，降级纯文本模板重试（retries=%d）: %s",
                retries,
                e,
            )
            try:
                degraded_html = build_degraded_html(sheet)
                await asyncio.wait_for(
                    _render_artifacts(degraded_html, sheet_dir),
                    timeout=RENDER_TIMEOUT_SECONDS,
                )
                return await _finalize_job(
                    db, sheet_id, RENDER_JOB_DEGRADED, qr_ref, retries,
                    error_code=_classify_error(e),
                )
            except Exception as de:  # noqa: BLE001
                retries += 1
                logger.error("[renderSheetJob] 降级渲染也失败: %s", de)
                return await _finalize_job(
                    db, sheet_id, RENDER_JOB_FAILED, qr_ref, retries,
                    error_code=_classify_error(de),
                )
        # 超限（不应发生，防御）
        return await _finalize_job(
            db, sheet_id, RENDER_JOB_FAILED, qr_ref, retries,
            error_code=ERROR_RETRY_EXCEEDED,
        )
    except Exception as e:  # noqa: BLE001
        # 非渲染类异常（如 DB 问题）直接失败
        logger.error("[renderSheetJob] 未预期异常: %s", e)
        return await _finalize_job(
            db, sheet_id, RENDER_JOB_FAILED, qr_ref, retries,
            error_code=ERROR_RENDER_FAILED,
        )

    # 4. 成功
    return await _finalize_job(
        db, sheet_id, RENDER_JOB_SUCCESS, qr_ref, retries,
    )


def _classify_error(exc: Exception) -> str:
    """异常 → 契约错误码（依赖缺失 / 字体缺失 / 超时 / 渲染失败）"""
    if isinstance(exc, RendererUnavailableError):
        return ERROR_DEPENDENCY_MISSING
    if isinstance(exc, asyncio.TimeoutError):
        return ERROR_TIMEOUT
    msg = str(exc)
    if "font" in msg.lower() or "glyph" in msg.lower():
        return ERROR_FONT_MISSING
    return ERROR_RENDER_FAILED


async def _finalize_job(
    db,
    sheet_id: str,
    status: str,
    qr_ref: dict,
    retries: int,
    error_code: str = "",
) -> dict:
    """回写 sheet_render_job（status/artifacts/error_code/retries）+ practice_sheet（file_refs/qrcode_ref）"""
    now = _now_ms()
    base_url = f"{RENDER_STATIC_URL_PREFIX}/{sheet_id}"
    artifacts = {
        "pdf": f"{base_url}/{PDF_FILENAME}",
        "png": f"{base_url}/{PNG_FILENAME}",
        "preview": f"{base_url}/{PREVIEW_FILENAME}",
    }
    # success / degraded 才回写产物引用；failed 清空
    file_refs = (
        {
            "pdf": artifacts["pdf"],
            "png": artifacts["png"],
            "preview": artifacts["preview"],
        }
        if status in (RENDER_JOB_SUCCESS, RENDER_JOB_DEGRADED)
        else {"pdf": "", "png": "", "preview": ""}
    )

    job_update: dict = {
        "status": status,
        "retries": retries,
        "error_code": error_code,
        "artifacts": artifacts if status in (RENDER_JOB_SUCCESS, RENDER_JOB_DEGRADED) else {},
        "updated_at": now,
    }
    await db.update(
        SHEET_RENDER_JOB_COLLECTION,
        {"sheet_id": sheet_id},
        {"$set": job_update},
    )
    await db.update(
        PRACTICE_SHEET_COLLECTION,
        {"sheet_id": sheet_id},
        {
            "$set": {
                "file_refs": file_refs,
                "qrcode_ref": qr_ref,
                "updated_at": now,
            }
        },
    )
    return {
        "sheet_id": sheet_id,
        "job_id": sheet_id,
        "status": status,
        "artifacts": job_update["artifacts"],
        "error_code": error_code,
        "retries": retries,
        "file_refs": file_refs,
        "qrcode_ref": qr_ref,
    }


async def renderPendingJobs(db, limit: int = 20) -> list[dict]:
    """批量消费 queued 渲染任务（供定时脚本 / 启动兜底调用）

    Args:
        db: CloudBaseNoSQLClient
        limit: 单次最多处理任务数（默认 20）

    Returns:
        每任务 renderSheetJob 返回的 job 摘要列表
    """
    jobs = await db.query(
        SHEET_RENDER_JOB_COLLECTION,
        where={"status": RENDER_JOB_QUEUED},
        order=[{"field": "created_at", "direction": "asc"}],
        limit=limit,
    )
    results: list[dict] = []
    for job in (jobs.get("records") or []):
        sheet_id = job.get("sheet_id") or ""
        if not sheet_id:
            continue
        try:
            results.append(await renderSheetJob(db, sheet_id))
        except RenderJobNotFoundError as e:
            logger.warning("[renderPendingJobs] %s", e)
        except Exception as e:  # noqa: BLE001
            logger.error("[renderPendingJobs] 任务失败 sheet_id=%s: %s", sheet_id, e)
    return results
