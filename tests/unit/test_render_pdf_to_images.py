"""PDF → sheet.png / preview.png（2026-10-23 修复：线上产物 404 根因）

被测对象：services/math/a4_renderer.py::_render_pdf_to_images

背景（线上实测）：WeasyPrint ≥60 移除 `HTML.write_png`，原实现吞掉异常 →
`sheet.png` / `preview.png` 从不生成（静态产物 404），但 `file_refs` 仍照常
拼出 URL，前端「保存预览图」必然失败。

覆盖用例：
  - pypdfium2 可用 → 生成 sheet.png + preview.png（预览按 1:1.4 裁剪顶部）
  - pypdfium2 不可用 → 回落 PyMuPDF(fitz)
  - 两者均不可用 → 抛 RuntimeError（由调用方降级为「仅 PDF」并告警，不再静默）
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from services.math import a4_renderer as ar


@pytest.fixture
def pdf_path(tmp_path) -> str:
    p = tmp_path / "sheet.pdf"
    p.write_bytes(b"%PDF-1.4 fake")
    return str(p)


def _install_fake_pypdfium2(monkeypatch, size=(800, 1200)):
    """注入假 pypdfium2：render().to_pil() 返回一张纯白图"""

    class _Bmp:
        def to_pil(self):
            return Image.new("RGB", size, "white")

    class _Page:
        def render(self, scale=1):
            return _Bmp()

    class _Doc(list):
        def __init__(self, path):
            super().__init__([_Page()])

    mod = types.ModuleType("pypdfium2")
    mod.PdfDocument = _Doc
    monkeypatch.setitem(sys.modules, "pypdfium2", mod)


def test_renders_png_and_preview(monkeypatch, tmp_path, pdf_path):
    _install_fake_pypdfium2(monkeypatch)
    png = str(tmp_path / "sheet.png")
    preview = str(tmp_path / "preview.png")

    ar._render_pdf_to_images(pdf_path, png, preview)

    assert Path(png).exists() and Path(png).stat().st_size > 0
    assert Path(preview).exists()
    with Image.open(preview) as img:
        w, h = img.size
    assert (w, h) == (800, int(800 * 1.4))  # 顶部裁剪，A4 预览比例 1:1.4


def test_falls_back_to_pymupdf(monkeypatch, tmp_path, pdf_path):
    """pypdfium2 缺失 → 回落 PyMuPDF"""
    monkeypatch.setitem(sys.modules, "pypdfium2", None)  # import 触发 ImportError

    class _Pix:
        def tobytes(self, fmt="png"):
            buf = io.BytesIO()
            Image.new("RGB", (600, 900), "white").save(buf, "PNG")
            return buf.getvalue()

    class _Page:
        def get_pixmap(self, dpi=144):
            return _Pix()

    class _Doc(list):
        def __init__(self, path):
            super().__init__([_Page()])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    fitz = types.ModuleType("fitz")
    fitz.open = lambda path: _Doc(path)
    monkeypatch.setitem(sys.modules, "fitz", fitz)

    png = str(tmp_path / "sheet.png")
    preview = str(tmp_path / "preview.png")

    ar._render_pdf_to_images(pdf_path, png, preview)

    assert Path(png).exists()
    assert Path(preview).exists()


def test_raises_when_no_renderer(monkeypatch, tmp_path, pdf_path):
    """两者均不可用 → 抛错（调用方降级为仅 PDF，不再静默吞掉）"""
    monkeypatch.setitem(sys.modules, "pypdfium2", None)
    monkeypatch.setitem(sys.modules, "fitz", None)

    with pytest.raises(RuntimeError, match="无可用的 PDF→PNG 渲染器"):
        ar._render_pdf_to_images(
            pdf_path, str(tmp_path / "sheet.png"), str(tmp_path / "preview.png")
        )
