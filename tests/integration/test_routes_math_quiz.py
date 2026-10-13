"""数学同类型题出题 4 接口集成测试（契约 api-contract.md §3.10 接口 28~31 · SOP ④）

对应任务：docs_v1/数学/同类型题-原子任务清单.md **B15**。

遵循 `tests/conftest.py` 约定：
- 客户端统一走 `make_client` 工厂（`FakeDB` 双通道注入）；
- 外部调用由 `tests/integration/conftest.py` 的 `no_external_calls` autouse 兜底，
  本文件再用 monkeypatch 替换 quiz 链路的 3 个外部出口（Judge / 出题 LLM / 渲染器），
  保证**不触网、不落真实产物**；
- 造数用本文件 `_seed_*` 局部工厂（`tests/fakes/seed_factory` 无 math quiz 造数）。

核心断言：
- 4 条路由 happy path；
- 28 / 29 / 30 各产生 **1 条** `audit_log`（B13 生效）；
- **R3** 出参 `items[]` 恒无 `answer` / `hint`；**R2** 不触碰 `practice_sheet` / `sheet_render_job`。
"""
from __future__ import annotations

import asyncio
import time

import pytest

import services.math.a4_renderer as ar
import services.math.quiz as quiz_mod
import services.math.quiz_recognizer as recog
import services.routes.math as routes_math
from services.routes_math import router as math_router

SCAN_ID = "su_quiz_001"
QUIZ_ID = "qz_quiz_001"
SCHOLAR_ID = "s_quiz_001"

# R2 红线集合：本页链路全程不得触碰
LEGACY_COLLECTIONS = ("practice_sheet", "sheet_render_job")


# ---------------------------------------------------------------------------
# 造数（局部工厂：本文件专用，seed_factory 无 math quiz 造数）
# ---------------------------------------------------------------------------


def _seed_scan(fake_db, *, ocr_status: str = "success", **overrides) -> dict:
    doc = {
        "scan_id": SCAN_ID,
        "scholar_id": SCHOLAR_ID,
        "ocr_status": ocr_status,
        "ocr_text": "1. 0.36 + 2.7 = ?\n2. 一个长方形长 5cm，宽 3cm，面积是多少？",
        "classify_status": "pending",
    }
    doc.update(overrides)
    fake_db.add("math_scan_upload", doc)
    return doc


def _seed_knowledge_point_candidates(fake_db) -> None:
    """知识点候选集（`ai_summary.status=success` 的节点才进候选集）。"""
    fake_db.add(
        "curriculum_node",
        {
            "node_id": "n_quiz_001",
            "code": "5-3-1-kp2",
            "title": "小数加减法",
            "grade": "五年级",
            "semester": "上",
            "textbook_id": "tb_quiz_001",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [
                    {"name": "小数加减法", "summary": "小数点对齐，低位补零"},
                    {"name": "长方形面积", "summary": "长 × 宽"},
                ],
            },
        },
    )


def _seed_quiz(fake_db, *, status: str = "generated", **overrides) -> dict:
    doc = {
        "quiz_id": QUIZ_ID,
        "scholar_id": SCHOLAR_ID,
        "status": status,
        "render_status": "",
        "source_count": 1,
        "count_per_source": 2,
        "sources": [
            {
                "source_index": 0,
                "question_text": "0.36 + 2.7 = ?",
                "knowledge_point_name": "小数加减法",
                "node_code": "",
            }
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
            }
        ],
        "file_refs": {"pdf": "", "png": "", "preview": ""},
        "warnings": [],
        "signature": "sig_test",
        "created_at": 1760000000000,
    }
    doc.update(overrides)
    fake_db.add("math_quiz", doc)
    return doc


# ---------------------------------------------------------------------------
# 外部出口替身
# ---------------------------------------------------------------------------


@pytest.fixture
def judge_result(monkeypatch):
    """接口 28 的 Judge 出口：返回切分 + 知识点定位结果（不触网）。"""

    async def _fake_judge(ocr_text: str, candidates: list) -> dict:
        return {
            "items": [
                {
                    "question_text": "0.36 + 2.7 = ?",
                    "knowledge_point_name": "小数加减法",
                    "node_code": "5-3-1-kp2",
                    "confidence": 0.88,
                },
                {
                    "question_text": "一个长方形长 5cm，宽 3cm，面积是多少？",
                    "knowledge_point_name": "长方形面积",
                    "node_code": "5-3-1-kp5",
                    "confidence": 0.72,
                },
            ]
        }

    monkeypatch.setattr(recog, "_call_quiz_recognize_judge", _fake_judge)
    return _fake_judge


@pytest.fixture
def quiz_llm(monkeypatch):
    """接口 29 的出题 LLM 出口：按 count 返回「变式」题（不触网）。"""
    calls: list[dict] = []

    async def _fake_llm(
        *,
        source_question: str,
        knowledge_point_name: str = "",
        summary: str = "",
        count: int = 2,
        extended: bool = False,
        extended_band: str = "",
    ) -> list[dict]:
        calls.append({"count": count, "extended": extended})
        return [
            {
                "question": f"把长改成 {i}{i} 分米、宽 {i} 分米的长方形面积是多少？",
                "answer": str(i * 11),
                "answer_type": "number",
                "difficulty": 2,
                "hint": "",
            }
            for i in range(count)
        ]

    monkeypatch.setattr(quiz_mod, "_generate_quiz_questions", _fake_llm)
    monkeypatch.setattr(quiz_mod, "LLM_SUMMARY_MODEL", "mock-model")
    return calls


@pytest.fixture
def render_env(monkeypatch, tmp_path):
    """接口 30 的渲染出口：不落真实 PDF，产物目录隔离到 tmp_path。"""
    monkeypatch.setattr(ar, "RENDER_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(ar, "RENDER_STATIC_URL_PREFIX", "/static")
    return tmp_path


def _slow_render(delay: float = 1.0):
    """慢渲染替身：用于验证接口 30 **不阻塞等待**（返回快于渲染耗时）。"""

    async def _fake(html: str, target_dir) -> None:
        await asyncio.sleep(delay)
        (target_dir / ar.PDF_FILENAME).write_bytes(b"%PDF-1.4 fake")

    return _fake


def _audit_actions(fake_db) -> list[str]:
    return [a.get("action") for a in fake_db.all("audit_log")]


async def _legacy_counts(fake_db) -> list[int]:
    return [await fake_db.count(c) for c in LEGACY_COLLECTIONS]


# ---------------------------------------------------------------------------
# 接口 28 · POST /math/scan/quiz-recognize
# ---------------------------------------------------------------------------


class TestQuizRecognizeRoute:
    def test_recognize_returns_items_and_writes_audit(self, make_client, fake_db, judge_result):
        """happy path：返回切分后的题目列表（零写库）+ 写 1 条 quiz_recognize 审计"""
        _seed_scan(fake_db)
        _seed_knowledge_point_candidates(fake_db)

        client = make_client(math_router)
        res = client.post(
            "/math/scan/quiz-recognize",
            json={"scholar_id": SCHOLAR_ID, "scan_id": SCAN_ID},
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]

        assert data["scan_id"] == SCAN_ID
        assert data["status"] == "success"
        assert len(data["items"]) == 2
        # 候选集命中 → 知识点名 / node_code 直接可取（供前端回显与接口 29 回传）
        assert data["items"][0]["knowledge_point_name"] == "小数加减法"
        assert data["items"][1]["knowledge_point_name"] == "长方形面积"
        assert data["items"][0]["question_text"] == "0.36 + 2.7 = ?"
        # 零写库（R1）：不写 error_record，也不回写 math_scan_upload
        assert asyncio.run(fake_db.count("error_record")) == 0
        assert fake_db.all("math_scan_upload")[0]["classify_status"] == "pending"
        # B13 生效：1 条 quiz_recognize 审计
        assert _audit_actions(fake_db) == ["quiz_recognize"]
        # R2：不触碰既有练习纸链路
        assert asyncio.run(_legacy_counts(fake_db)) == [0, 0]

    def test_recognize_missing_scan_id_returns_400(self, make_client, fake_db, judge_result):
        """缺 scan_id → 400（非 Pydantic 422）"""
        client = make_client(math_router)
        res = client.post(
            "/math/scan/quiz-recognize", json={"scholar_id": SCHOLAR_ID, "scan_id": ""}
        )
        assert res.status_code == 400, res.text

    def test_recognize_ocr_not_ready_returns_404(self, make_client, fake_db, judge_result):
        """OCR 未就绪 → 404（与接口 14 同口径）"""
        _seed_scan(fake_db, ocr_status="pending")
        client = make_client(math_router)
        res = client.post(
            "/math/scan/quiz-recognize", json={"scholar_id": SCHOLAR_ID, "scan_id": SCAN_ID}
        )
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# 接口 29 · POST /math/quiz/generate
# ---------------------------------------------------------------------------


class TestQuizGenerateRoute:
    def test_generate_returns_public_items_and_writes_audit(
        self, make_client, fake_db, quiz_llm
    ):
        """happy path：落库 math_quiz + 出参净化（R3）+ 写 1 条 quiz_generate 审计"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={
                "scholar_id": SCHOLAR_ID,
                "scan_id": SCAN_ID,
                "source_questions": [
                    {
                        "source_index": 0,
                        "question_text": "0.36 + 2.7 = ?",
                        "knowledge_point_name": "小数加减法",
                    }
                ],
                "count_per_source": 2,
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]

        assert data["status"] == "generated"
        assert data["source_count"] == 1
        assert data["count_per_source"] == 2
        assert len(data["items"]) == 2
        # R3：answer / hint 落库不出参
        assert all("answer" not in it and "hint" not in it for it in data["items"])
        saved = fake_db.all("math_quiz")[0]
        assert saved["items"][0]["answer"]  # 落库保留（供渲染 / 日后判题）
        assert saved["source_scan_id"] == SCAN_ID  # 溯源
        # B13 生效：1 条 quiz_generate 审计，context 含契约字段
        audits = fake_db.all("audit_log")
        assert [a["action"] for a in audits] == ["quiz_generate"]
        assert audits[0]["context"]["source_count"] == 1
        assert audits[0]["context"]["generated_count"] == 2
        # D9：只对数组内源题出题（1 道源题 → 1 次 LLM 调用）
        assert len(quiz_llm) == 1
        # R2：不触碰既有练习纸链路
        assert asyncio.run(_legacy_counts(fake_db)) == [0, 0]

    def test_generate_empty_source_questions_returns_400(self, make_client, fake_db, quiz_llm):
        """source_questions 为空 → 400（服务层口径，非 Pydantic 422）"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate", json={"scholar_id": SCHOLAR_ID, "source_questions": []}
        )
        assert res.status_code == 400, res.text
        assert len(quiz_llm) == 0  # 零 LLM 调用


# ---------------------------------------------------------------------------
# 接口 30 · POST /math/quiz/{quiz_id}/render
# ---------------------------------------------------------------------------


class TestQuizRenderRoute:
    def test_render_returns_rendering_without_blocking(
        self, make_client, fake_db, render_env, monkeypatch
    ):
        """happy path：`asyncio.create_task` 异步触发 → **立即返回 rendering**（不阻塞）"""
        monkeypatch.setattr(ar, "_render_artifacts", _slow_render(delay=1.0))
        _seed_quiz(fake_db)

        client = make_client(math_router)
        started = time.monotonic()
        res = client.post(f"/math/quiz/{QUIZ_ID}/render", json={"scholar_id": SCHOLAR_ID})
        elapsed = time.monotonic() - started

        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["quiz_id"] == QUIZ_ID
        assert data["status"] == "rendering"  # 契约：立即返回，前端轮询接口 31
        assert data["file_refs"] == {"pdf": "", "png": "", "preview": ""}
        assert elapsed < 0.9, f"渲染阻塞了主流程（{elapsed:.2f}s）"
        # B13 生效：1 条 quiz_render 审计
        assert _audit_actions(fake_db) == ["quiz_render"]

    def test_render_already_rendered_returns_existing_file_refs(
        self, make_client, fake_db, render_env
    ):
        """已渲染过 → 幂等返回既有 file_refs（不重复触发、不重复写审计）"""
        _seed_quiz(
            fake_db,
            status="rendered",
            render_status="success",
            file_refs={"pdf": f"/static/{QUIZ_ID}/sheet.pdf", "png": "", "preview": ""},
        )
        client = make_client(math_router)
        res = client.post(f"/math/quiz/{QUIZ_ID}/render", json={"scholar_id": SCHOLAR_ID})

        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "rendered"
        assert data["file_refs"]["pdf"] == f"/static/{QUIZ_ID}/sheet.pdf"
        assert data["render_message"] == ""
        assert _audit_actions(fake_db) == []  # 幂等分支不写审计

    def test_render_failed_status_returns_409_with_reason(self, make_client, fake_db, render_env):
        """渲染失败态再点打印 → 409，且 detail 带**用户可见原因**"""
        _seed_quiz(
            fake_db,
            status="failed",
            render_status="failed",
            render_message="打印稿渲染依赖未安装（WeasyPrint），请联系管理员或稍后重试",
        )
        client = make_client(math_router)
        res = client.post(f"/math/quiz/{QUIZ_ID}/render", json={"scholar_id": SCHOLAR_ID})

        assert res.status_code == 409, res.text
        assert "依赖未安装" in res.json()["detail"]

    def test_render_quiz_not_found_returns_404(self, make_client, fake_db, render_env):
        client = make_client(math_router)
        res = client.post("/math/quiz/qz_nope/render", json={"scholar_id": SCHOLAR_ID})
        assert res.status_code == 404, res.text

    def test_render_task_wrapper_swallows_exceptions(self, fake_db, monkeypatch):
        """后台任务包装器吞掉所有异常 —— 结果由 math_quiz 状态承载（接口 31 轮询）"""

        async def _boom(db, quiz_id: str) -> dict:
            raise RuntimeError("渲染炸了")

        monkeypatch.setattr(routes_math, "renderQuizJob", _boom)
        asyncio.run(routes_math._quiz_render_task(fake_db, QUIZ_ID))  # 不抛出


# ---------------------------------------------------------------------------
# 接口 31 · GET /math/quiz/{quiz_id}
# ---------------------------------------------------------------------------


class TestQuizDetailRoute:
    def test_detail_returns_public_quiz_with_file_refs(self, make_client, fake_db):
        """happy path：轮询渲染状态与产物，出参无 answer / hint（R3）"""
        _seed_quiz(
            fake_db,
            status="rendered",
            render_status="success",
            file_refs={
                "pdf": f"/static/{QUIZ_ID}/sheet.pdf",
                "png": f"/static/{QUIZ_ID}/sheet.png",
                "preview": f"/static/{QUIZ_ID}/preview.png",
            },
        )
        client = make_client(math_router)
        res = client.get(f"/math/quiz/{QUIZ_ID}")

        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["quiz_id"] == QUIZ_ID
        assert data["status"] == "rendered"
        assert data["render_status"] == "success"
        assert data["file_refs"]["pdf"] == f"/static/{QUIZ_ID}/sheet.pdf"
        assert all("answer" not in it and "hint" not in it for it in data["items"])

    def test_detail_not_found_returns_404(self, make_client, fake_db):
        client = make_client(math_router)
        res = client.get("/math/quiz/qz_nope")
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# 边界与错误码（B16 · 设计文档 §8.1）
# ---------------------------------------------------------------------------


def _one_source(**overrides) -> dict:
    src = {"source_index": 0, "question_text": "0.36 + 2.7 = ?"}
    src.update(overrides)
    return src


class TestQuizBoundaryAndErrorCodes:
    """B16：契约 §4.2~§4.5 的错误码矩阵 —— **全部非法入参走 400，不是 Pydantic 422**"""

    @pytest.mark.parametrize(
        "count_per_source",
        [0, 4, -1, 1.5, "2", True, None],
        ids=["0-下溢", "4-上溢", "负数", "小数", "字符串", "bool", "None"],
    )
    def test_generate_invalid_count_per_source_returns_400(
        self, make_client, fake_db, quiz_llm, count_per_source
    ):
        """`count_per_source` 非 1~3 整数 → 400 且零 LLM 调用"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={
                "scholar_id": SCHOLAR_ID,
                "source_questions": [_one_source()],
                "count_per_source": count_per_source,
            },
        )
        assert res.status_code == 400, res.text
        assert len(quiz_llm) == 0

    def test_generate_source_count_over_limit_returns_400(self, make_client, fake_db, quiz_llm):
        """`source_questions` 长度 > QUIZ_MAX_SOURCE_COUNT(5) → 400"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={
                "scholar_id": SCHOLAR_ID,
                "source_questions": [_one_source(source_index=i) for i in range(6)],
            },
        )
        assert res.status_code == 400, res.text
        assert "最多" in res.json()["detail"]
        assert len(quiz_llm) == 0

    def test_generate_duplicate_source_index_returns_400(self, make_client, fake_db, quiz_llm):
        """`source_index` 重复 → 400"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={
                "scholar_id": SCHOLAR_ID,
                "source_questions": [_one_source(), _one_source(source_index=0)],
            },
        )
        assert res.status_code == 400, res.text
        assert "重复" in res.json()["detail"]

    def test_generate_missing_question_text_returns_400(self, make_client, fake_db, quiz_llm):
        """`question_text` 为空串 → 400（契约：空串 → 400）"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={"scholar_id": SCHOLAR_ID, "source_questions": [_one_source(question_text="")]},
        )
        assert res.status_code == 400, res.text

    def test_generate_missing_scholar_id_returns_400(self, make_client, fake_db, quiz_llm):
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate", json={"scholar_id": "", "source_questions": [_one_source()]}
        )
        assert res.status_code == 400, res.text

    def test_generate_node_code_not_found_returns_404(self, make_client, fake_db, quiz_llm):
        """`node_code` 不存在 → 404（在出题前拦截，零 LLM 调用）"""
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={
                "scholar_id": SCHOLAR_ID,
                "source_questions": [_one_source(node_code="no_such_node")],
            },
        )
        assert res.status_code == 404, res.text
        assert len(quiz_llm) == 0

    def test_generate_all_sources_failed_returns_500(
        self, make_client, fake_db, quiz_llm, monkeypatch
    ):
        """全部源题出题失败 → 500，且落 1 条 failed 审计"""

        async def _boom(**kwargs) -> list[dict]:
            raise quiz_mod.LLMResponseError("LLM 出题失败")

        monkeypatch.setattr(quiz_mod, "_generate_quiz_questions", _boom)
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={"scholar_id": SCHOLAR_ID, "source_questions": [_one_source()]},
        )
        assert res.status_code == 500, res.text
        audits = fake_db.all("audit_log")
        assert [a["action"] for a in audits] == ["quiz_generate"]
        assert audits[0]["result"] == "failed"

    def test_generate_llm_not_configured_returns_400(self, make_client, fake_db, monkeypatch):
        """LLM_SUMMARY_MODEL 未配置 → 400（非 500，与既有练习纸链路同口径）

        本用例**不装** `quiz_llm` 替身 —— 走的正是真实 `_generate_quiz_questions`
        的「模型未配置」分支（该分支在调用 LLM 之前短路，故仍不触网）。
        """
        monkeypatch.setattr(quiz_mod, "LLM_SUMMARY_MODEL", "")
        client = make_client(math_router)
        res = client.post(
            "/math/quiz/generate",
            json={"scholar_id": SCHOLAR_ID, "source_questions": [_one_source()]},
        )
        assert res.status_code == 400, res.text

    def test_recognize_scan_not_found_returns_404(self, make_client, fake_db, judge_result):
        """scan_id 不存在 → 404"""
        client = make_client(math_router)
        res = client.post(
            "/math/scan/quiz-recognize",
            json={"scholar_id": SCHOLAR_ID, "scan_id": "su_no_such"},
        )
        assert res.status_code == 404, res.text

    def test_recognize_invalid_photo_context_returns_400(
        self, make_client, fake_db, judge_result
    ):
        """`photo_context` 非法 JSON → 400（契约 §4.2）"""
        _seed_scan(fake_db)
        client = make_client(math_router)
        res = client.post(
            "/math/scan/quiz-recognize",
            json={
                "scholar_id": SCHOLAR_ID,
                "scan_id": SCAN_ID,
                "photo_context": "{不是JSON",
            },
        )
        assert res.status_code == 400, res.text
        assert "photo_context" in res.json()["detail"]

    def test_render_rendering_status_is_idempotent(self, make_client, fake_db, render_env):
        """渲染中重复点打印 → 幂等返回 rendering + 既有 file_refs（不重复触发）"""
        _seed_quiz(fake_db, status="rendering", render_status="rendering")
        client = make_client(math_router)
        res = client.post(f"/math/quiz/{QUIZ_ID}/render", json={"scholar_id": SCHOLAR_ID})

        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "rendering"
        assert res.json()["data"]["render_status"] == "rendering"
        assert _audit_actions(fake_db) == []  # 幂等分支不写审计

    def test_render_missing_scholar_id_returns_400(self, make_client, fake_db, render_env):
        _seed_quiz(fake_db)
        client = make_client(math_router)
        res = client.post(f"/math/quiz/{QUIZ_ID}/render", json={"scholar_id": ""})
        assert res.status_code == 400, res.text
