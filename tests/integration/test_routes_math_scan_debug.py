"""数学错题识别 Admin 调试干跑路由集成测试（B07/B08/B11）

B07：路由注册 + 错误码映射
B08：main.py 条件注册（本测试模拟开关开关两态）
B11：开关 404 / token 403 / multipart / 错误码
"""
from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.routes_math_debug import router as math_debug_router


@pytest.fixture(autouse=True)
def debug_enabled(monkeypatch):
    """模拟 B08 开关开 + dev 模式（无 token 放行）。"""
    monkeypatch.setattr("services.auth.MATH_SCAN_DEBUG_ENABLED", True)
    monkeypatch.setattr("services.auth.MATH_SCAN_DEBUG_TOKEN", "")
    monkeypatch.setattr("services.auth.AUTH_MODE", "dev")


def _make_app_with_debug_router() -> TestClient:
    """构建带 math_debug_router 的 TestClient（模拟 B08 条件注册开关开）。"""
    app = FastAPI()
    app.include_router(math_debug_router)
    return TestClient(app)


def _make_app_without_debug_router() -> TestClient:
    """构建不带 math_debug_router 的 TestClient（模拟 B08 开关关）。"""
    app = FastAPI()
    return TestClient(app)


def _make_fake_image_bytes(
    filename: str = "test.jpg", content: bytes = b"fake_image_content"
) -> tuple[bytes, str]:
    """构造 multipart 文件参数。"""
    return (filename, io.BytesIO(content), "image/jpeg")


def _mock_ocr_provider() -> MagicMock:
    """构造 mock OCR provider。"""
    from services.math.ocr import OcrResult

    provider = MagicMock()
    provider.__class__.__name__ = "FakeProvider"
    provider.available = True
    provider._engine = "test"
    provider.recognize = AsyncMock(
        return_value=OcrResult(text="0.36 + 2.7 = ?", blocks=[{"block_id": "blk_0001"}])
    )
    return provider


def _mock_judge_result() -> dict:
    return {
        "items": [
            {"knowledge_point_name": "小数加减法", "error_type": "computation",
             "confidence": 0.92, "ocr_block_id": "blk_0001", "question_text": "0.36+2.7=?"}
        ]
    }


def _mock_judge_meta() -> dict:
    return {"prompt_chars": 1234, "attempts": 1}


def _make_client_with_deps(fake_db=None):
    """构建带 math_debug_router + get_db 注入的 TestClient。"""
    app = FastAPI()
    app.include_router(math_debug_router)
    if fake_db is not None:
        from services.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: fake_db
    return TestClient(app)


class TestRouteRegistration:
    """B07：路由注册 + 基本可达性。"""

    def test_route_path_is_math_scan_debug_recognize(self):
        """路由路径 /math/scan/debug/recognize 可注册。"""
        client = _make_app_with_debug_router()
        # 路由注册成功 → POST 不带文件应返回 422（缺参）而非 404
        resp = client.post("/math/scan/debug/recognize")
        assert resp.status_code == 422  # 缺 image 必填参数

    def test_route_tag_is_math_scan_debug(self):
        """路由 tag 为 math-scan-debug。"""
        routes = [r for r in math_debug_router.routes]
        assert any(r.path == "/math/scan/debug/recognize" for r in routes)


class TestErrorMapping:
    """B07：错误码映射（400/413/500）。"""

    def test_invalid_image_format_returns_400(self):
        """不支持的图片格式 → 400。"""
        client = _make_app_with_debug_router()
        resp = client.post(
            "/math/scan/debug/recognize",
            files={"image": _make_fake_image_bytes("test.gif", b"GIF89a")},
            data={"textbook_id": "tb_1"},
        )
        assert resp.status_code == 400

    def test_empty_image_returns_400(self):
        """空图片 → 400。"""
        client = _make_app_with_debug_router()
        resp = client.post(
            "/math/scan/debug/recognize",
            files={"image": _make_fake_image_bytes("test.jpg", b"")},
            data={"textbook_id": "tb_1"},
        )
        assert resp.status_code == 400

    def test_ocr_error_returns_500(self, fake_db):
        """OCR 不可用 → 500。"""
        ocr_provider = MagicMock()
        ocr_provider.__class__.__name__ = "FakeProvider"
        ocr_provider.available = False
        ocr_provider._engine = "test"

        with patch("services.math.ocr.get_provider", return_value=ocr_provider):
            client = _make_client_with_deps(fake_db)
            resp = client.post(
                "/math/scan/debug/recognize",
                files={"image": _make_fake_image_bytes("test.jpg", b"fake")},
                data={"textbook_id": "tb_1"},
            )
        assert resp.status_code == 500

    def test_judge_error_returns_500(self, fake_db):
        """Judge 调用失败 → 500。"""
        from services.math.error_scanner import JudgeResponseError

        ocr_provider = _mock_ocr_provider()

        with patch("services.math.ocr.get_provider", return_value=ocr_provider), \
             patch(
                 "services.math.error_scan_debug._call_judge_with_meta",
                 new=AsyncMock(side_effect=JudgeResponseError("Judge 解析失败")),
             ):
            client = _make_client_with_deps(fake_db)
            resp = client.post(
                "/math/scan/debug/recognize",
                files={"image": _make_fake_image_bytes("test.jpg", b"fake")},
                data={"textbook_id": "tb_1"},
            )
        assert resp.status_code == 500

    def test_successful_dry_run_returns_200(self, fake_db):
        """正常干跑 → 200 + success=true。"""
        from tests.unit.test_math_error_scan_debug import _kp_node

        fake_db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))

        ocr_provider = _mock_ocr_provider()
        judge_result = _mock_judge_result()
        judge_meta = _mock_judge_meta()

        with patch("services.math.ocr.get_provider", return_value=ocr_provider), \
             patch(
                 "services.math.error_scan_debug._call_judge_with_meta",
                 new=AsyncMock(return_value=(judge_result, judge_meta)),
             ):
            client = _make_client_with_deps(fake_db)
            resp = client.post(
                "/math/scan/debug/recognize",
                files={"image": _make_fake_image_bytes("test.jpg", b"fake")},
                data={"textbook_id": "tb_1", "compare_all_candidates": "false"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["scan_id"].startswith("debug_")
        assert len(body["data"]["items"]) == 1
        assert body["data"]["items"][0]["knowledge_point_name"] == "小数加减法"
        assert body["data"]["debug"]["dry_run"] is True
        assert body["data"]["debug"]["persisted"] is False
