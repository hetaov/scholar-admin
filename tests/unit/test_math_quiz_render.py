"""单元测试：数学同类型题打印稿渲染 services/math/a4_renderer.renderQuizJob（B12 · 设计文档 D7）

对应任务：docs_v1/数学/同类型题-原子任务清单.md B12。

覆盖：
- **成功**：`math_quiz.status='rendered'` / `render_status='success'` / `file_refs.pdf` 非空 /
  `rendered_at` 已写；产物目录 `RENDER_OUTPUT_DIR/{quiz_id}`。
- **降级重试 1 次**：首次渲染失败 → 纯文本模板重试 → `render_status='degraded'` 且仍 rendered。
- **依赖缺失失败**：两次均失败 → `status='failed'`（`error_code='dependency_missing'`）、
  `file_refs` 清空、**不抛裸异常**。
- **重复渲染**：已 rendered 的 quiz 再调用 → **零渲染**，返回既有 `file_refs`。

红线：
- **R2** 不触碰 `practice_sheet` / `sheet_render_job`（无 job 集合、无既有链路回写）。
- **D7 无二维码**：落库文档不含 `qrcode_ref` / `qr_url`。
"""
from __future__ import annotations

import asyncio

import pytest

import services.math.a4_renderer as ar
from services.math.a4_renderer import RENDER_JOB_FAILED, QuizNotFoundError
from services.math.quiz import getQuizById
from tests.fakes.fake_db import FakeDB

QUIZ_ID = "qz_1"

QUIZ = {
    "quiz_id": QUIZ_ID,
    "scholar_id": "s1",
    "created_at": 1760000000000,
    "status": "generated",
    "sources": [
        {"source_index": 0, "question_text": "0.36 + 2.7 = ?", "knowledge_point_name": "小数加减法"},
        {"source_index": 2, "question_text": "一个长方形长 5cm ...", "knowledge_point_name": "长方形面积"},
    ],
    "items": [
        {
            "quiz_item_id": "qi_1",
            "source_index": 0,
            "question": "1.25 + 3.8 = ?",
            "answer": "5.05",
            "answer_type": "number",
            "variant_level": "L1",
            "difficulty": 2,
            "hint": "小数点对齐",
        },
        {
            "quiz_item_id": "qi_2",
            "source_index": 2,
            "question": "一个长方形长 8cm，宽 6cm，面积是多少？",
            "answer": "48",
            "answer_type": "number",
            "variant_level": "L2",
            "difficulty": 3,
            "hint": "",
        },
    ],
}

SENSITIVE_COLLECTIONS = ["practice_sheet", "sheet_render_job"]


def _run(coro):
    return asyncio.run(coro)


def _seed_quiz(**overrides) -> FakeDB:
    db = FakeDB()
    db.add("math_quiz", {**QUIZ, **overrides})
    return db


def _saved(db: FakeDB) -> dict:
    res = _run(db.query("math_quiz", where={"quiz_id": QUIZ_ID}, limit=1))
    return (res.get("records") or [{}])[0]


class _RenderSpy:
    """替换 `_render_artifacts`：记录调用并按需制造失败（不真实依赖 WeasyPrint）。"""

    def __init__(self, fail_times: int = 0, exc: Exception | None = None):
        self.fail_times = fail_times
        self.exc = exc or ar.RendererUnavailableError("weasyprint 未安装")
        self.calls: list[str] = []
        self.dirs: list[str] = []

    async def __call__(self, html: str, target_dir):
        if len(self.calls) < self.fail_times:
            self.calls.append(html)
            raise self.exc
        self.calls.append(html)
        self.dirs.append(str(target_dir))
        (target_dir / ar.PDF_FILENAME).write_bytes(b"%PDF-1.4 fake")
        return None


@pytest.fixture
def render_env(tmp_path, monkeypatch):
    """隔离产物目录与静态前缀，返回 (spy 安装函数, tmp_path)。"""
    monkeypatch.setattr(ar, "RENDER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ar, "RENDER_STATIC_URL_PREFIX", "/static")

    def install(fail_times: int = 0, exc: Exception | None = None) -> _RenderSpy:
        spy = _RenderSpy(fail_times=fail_times, exc=exc)
        monkeypatch.setattr(ar, "_render_artifacts", spy)
        return spy

    return install, tmp_path


class TestQuizRender:
    def test_render_success(self, render_env):
        install, tmp_path = render_env
        spy = install()
        db = _seed_quiz()

        res = _run(ar.renderQuizJob(db, QUIZ_ID))
        saved = _saved(db)

        assert res["status"] == "rendered"
        assert saved["status"] == "rendered"
        assert saved["render_status"] == "success"
        assert saved["render_message"] == ""  # 成功无提示
        assert saved["file_refs"]["pdf"] == f"/static/{QUIZ_ID}/{ar.PDF_FILENAME}"
        assert saved["file_refs"]["png"] == f"/static/{QUIZ_ID}/{ar.PNG_FILENAME}"
        assert saved["file_refs"]["preview"] == f"/static/{QUIZ_ID}/{ar.PREVIEW_FILENAME}"
        assert saved["rendered_at"]
        # 产物目录 = RENDER_OUTPUT_DIR/{quiz_id}，且真的写了东西进去
        assert spy.dirs == [str(tmp_path / QUIZ_ID)]
        assert (tmp_path / QUIZ_ID / ar.PDF_FILENAME).exists()
        # D7：无二维码
        assert "qrcode_ref" not in saved and "qr_url" not in saved
        # R2：不触碰既有练习纸链路
        assert _run(_counts(db)) == [0, 0]
        # HTML 侧：来源角标 + 统计条 + 无二维码（与 B10 契约一致）
        assert "来自原题 1" in spy.calls[0] and "共 2 题" in spy.calls[0]
        assert "<img" not in spy.calls[0]

    def test_degraded_retry_then_success(self, render_env):
        install, _ = render_env
        spy = install(fail_times=1)
        db = _seed_quiz()

        res = _run(ar.renderQuizJob(db, QUIZ_ID))
        saved = _saved(db)

        assert len(spy.calls) == 2  # 首次失败 + 降级重试 1 次
        assert res["render_status"] == "degraded"
        assert saved["render_status"] == "degraded"
        assert res["status"] == saved["status"] == "rendered"
        assert saved["file_refs"]["pdf"]  # 降级产物仍可用
        assert res["error_code"] == ar.ERROR_DEPENDENCY_MISSING
        # 降级也要让用户知道「版式已降级」
        assert saved["render_message"] == ar.RENDER_DEGRADED_MESSAGE
        # 降级走的是纯文本模板（无来源角标 CSS/结构）
        assert "来自原题" not in spy.calls[1]

    def test_dependency_missing_failed(self, render_env):
        install, _ = render_env
        spy = install(fail_times=99)
        db = _seed_quiz()

        res = _run(ar.renderQuizJob(db, QUIZ_ID))  # 不抛裸异常
        saved = _saved(db)

        assert len(spy.calls) == 2  # 首次 + 降级重试 1 次（RENDER_RETRY_LIMIT=1）
        assert res["status"] == saved["status"] == "failed"
        assert saved["render_status"] == RENDER_JOB_FAILED
        assert res["error_code"] == ar.ERROR_DEPENDENCY_MISSING
        assert saved["file_refs"] == {"pdf": "", "png": "", "preview": ""}
        assert saved.get("rendered_at") is None
        # 失败原因对用户可见（落库 render_message + 接口 31 出参透出）
        assert saved["render_message"] == ar.RENDER_ERROR_MESSAGES[ar.ERROR_DEPENDENCY_MISSING]
        assert res["render_message"] == saved["render_message"]
        assert _run(getQuizById(db, quiz_id=QUIZ_ID))["render_message"] == saved["render_message"]

    def test_duplicate_render_is_idempotent(self, render_env):
        install, _ = render_env
        spy = install()
        db = _seed_quiz()

        first = _run(ar.renderQuizJob(db, QUIZ_ID))
        second = _run(ar.renderQuizJob(db, QUIZ_ID))

        assert len(spy.calls) == 1  # 第二次零渲染
        assert second["status"] == first["status"] == "rendered"
        assert second["file_refs"]["pdf"] == first["file_refs"]["pdf"]

    def test_quiz_not_found(self, render_env):
        install, _ = render_env
        install()
        with pytest.raises(QuizNotFoundError):
            _run(ar.renderQuizJob(FakeDB(), "qz_nope"))


async def _counts(db: FakeDB) -> list[int]:
    return [await db.count(c) for c in SENSITIVE_COLLECTIONS]
