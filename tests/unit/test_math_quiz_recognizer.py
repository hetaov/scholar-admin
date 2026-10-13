"""单元测试：数学同类型题拍照识别 services/math/quiz_recognizer（B05 · 红线 R1 零写库）

对应任务：docs_v1/数学/同类型题-原子任务清单.md B05；伪代码见设计文档 §8.2。

覆盖：
- **零写库（红线 R1）**：`error_record` / `curriculum_node` / `textbook` 三集合计数前后不变；
  写操作探针显示除 `audit_log` 外**没有任何写调用**；
  `math_scan_upload.classify_status` / `classify_result` 前后不变（不回写归类状态）。
  （`math_scan_upload` 本身不计入零写断言 —— 其 1 条由 `POST /math/scan/upload`
  负责创建，属预期行为；本模块对它是只读。）
- 出参 `items[]` 每项含 5 字段；无 `question_text` 的项被丢弃；全空 → `status="empty"`。
- OCR 未就绪 → `OcrNotReadyError`（路由层 404）；scan 不存在 → `ScanNotFoundError`（404）。
- 未命中候选 → `knowledge_point_name` / `node_code` 为空串，**不新建节点、不报错**。
- Judge 失败 → 写 `failed` 审计后原样抛出。
"""
from __future__ import annotations

import asyncio

import pytest

import services.math.quiz_recognizer as qr
from services.math.error_scanner import JudgeResponseError, OcrNotReadyError, ScanNotFoundError
from tests.fakes.fake_db import FakeDB

SENSITIVE_COLLECTIONS = ["error_record", "curriculum_node", "textbook"]

CANDIDATES = [
    {
        "node_id": "n1",
        "node_code": "5-3-1-kp2",
        "kp_name": "小数加减法",
        "grade": "五年级",
        "semester": "上",
        "textbook_id": "tb_1",
        "title": "小数加减法",
        "unit_title": "第3单元 小数加减法",
        "lesson_title": "小数加减法",
    },
    {
        "node_id": "n2",
        "node_code": "5-3-2-kp1",
        "kp_name": "长方形面积",
        "grade": "五年级",
        "semester": "上",
        "textbook_id": "tb_1",
        "title": "长方形的面积",
        "unit_title": "第6单元 多边形的面积",
        "lesson_title": "长方形的面积",
    },
]

PUBLIC_ITEM_KEYS = (
    "block_index",
    "question_text",
    "knowledge_point_name",
    "node_code",
    "confidence",
)


def _run(coro):
    return asyncio.run(coro)


def _judge_items():
    """Judge 输出：3 条有效题 + 1 条无题干项 + 1 条候选集外知识点项。"""
    return [
        {
            "knowledge_point_name": "小数加减法",
            "node_code": "5-3-1-kp2",
            "question_text": "0.36 + 2.7 = ?",
            "confidence": 0.88,
        },
        {
            "knowledge_point_name": "长方形面积",
            "node_code": "5-3-2-kp1",
            "question_text": "一个长方形长 5cm，宽 3cm，面积是多少？",
            "confidence": 0.75,
        },
        {
            "knowledge_point_name": "卷头占位",
            "node_code": "",
            "question_text": "   ",
            "confidence": 0.9,
        },
        {
            "knowledge_point_name": "三位数乘两位数",
            "node_code": "x-other",
            "question_text": "988 x 76 = ?",
            "confidence": 0.7,
        },
    ]


def _scan(scan_id: str = "su_1", ocr_status: str = "success") -> dict:
    return {
        "scan_id": scan_id,
        "scholar_id": "s1",
        "ocr_status": ocr_status,
        "ocr_text": "1. 0.36 + 2.7 = ?\n2. 一个长方形长 5cm\n3. 988 x 76 = ?",
        "ocr_blocks": [],
        "classify_status": "pending",
        "classify_result": None,
    }


def _patch(monkeypatch, *, judge_items=None, candidates=None):
    """注入假候选集与假 Judge，并记录 Judge 实际收到的候选集。"""
    captured: dict = {}

    async def fake_load_candidates(db, scholar_id):
        captured["scholar_id"] = scholar_id
        if candidates is not None:
            return candidates
        return list(CANDIDATES)

    async def fake_judge(ocr_text, cands):
        captured["candidates"] = list(cands)
        captured["ocr_text"] = ocr_text
        result = {"items": list(judge_items if judge_items is not None else _judge_items())}
        # 与真实 Judge 调用同口径：先过结构校验（归一化空白 / 数值）
        qr._validate_quiz_recognize_result(result)
        return result

    monkeypatch.setattr(qr, "_load_knowledge_point_candidates", fake_load_candidates)
    monkeypatch.setattr(qr, "_call_quiz_recognize_judge", fake_judge)
    return captured


def _db(scan: dict | None = None) -> FakeDB:
    db = FakeDB()
    db.add("math_scan_upload", scan if scan is not None else _scan())
    return db


class TestHappyPath:
    """出参结构：5 字段齐全 / 丢弃无题干项 / 全空 empty / photo_context 收敛候选。"""

    def test_items_have_five_public_fields(self, monkeypatch):
        _patch(monkeypatch)
        data = _run(qr.recognize_quiz_scan(_db(), scan_id="su_1", scholar_id="s1"))
        assert data["status"] == "success"
        assert len(data["items"]) == 3  # 无题干项被丢弃
        for item in data["items"]:
            assert set(item) == set(PUBLIC_ITEM_KEYS)
        assert data["items"][0]["question_text"] == "0.36 + 2.7 = ?"
        assert data["items"][0]["knowledge_point_name"] == "小数加减法"
        assert data["items"][0]["node_code"] == "5-3-1-kp2"

    def test_drops_items_without_question_text(self, monkeypatch):
        _patch(
            monkeypatch,
            judge_items=[
                {"knowledge_point_name": "小数加减法", "node_code": "5-3-1-kp2", "question_text": "", "confidence": 0.9},
                {"knowledge_point_name": "小数加减法", "node_code": "5-3-1-kp2", "question_text": "1.2 + 3.4 = ?", "confidence": 0.8},
            ],
        )
        data = _run(qr.recognize_quiz_scan(_db(), scan_id="su_1", scholar_id="s1"))
        assert len(data["items"]) == 1
        # block_index 取 Judge 原始下标（此处为第 2 项 → 1）
        assert data["items"][0]["block_index"] == 1

    def test_all_items_empty_status(self, monkeypatch):
        _patch(
            monkeypatch,
            judge_items=[
                {"knowledge_point_name": "小数加减法", "node_code": "5-3-1-kp2", "question_text": "", "confidence": 0.9},
            ],
        )
        data = _run(qr.recognize_quiz_scan(_db(), scan_id="su_1", scholar_id="s1"))
        assert data["status"] == "empty"
        assert data["items"] == []

    def test_unmatched_kp_blanked_without_error(self, monkeypatch):
        """未命中候选 → 空知识点（不建节点、不报错）。"""
        _patch(monkeypatch)
        db = _db()
        data = _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        unmatched = [it for it in data["items"] if it["question_text"] == "988 x 76 = ?"]
        assert len(unmatched) == 1
        assert unmatched[0]["knowledge_point_name"] == ""
        assert unmatched[0]["node_code"] == ""
        assert _run(db.count("curriculum_node")) == 0

    def test_photo_context_narrows_candidates(self, monkeypatch):
        captured = _patch(monkeypatch)
        _run(
            qr.recognize_quiz_scan(
                _db(),
                scan_id="su_1",
                scholar_id="s1",
                photo_context='{"textbook_id": "tb_1", "unit_title": "第6单元 多边形的面积"}',
            )
        )
        names = [c["kp_name"] for c in captured["candidates"]]
        assert names == ["长方形面积"]

    def test_photo_context_invalid_json_falls_back_to_all(self, monkeypatch):
        captured = _patch(monkeypatch)
        _run(
            qr.recognize_quiz_scan(
                _db(), scan_id="su_1", scholar_id="s1", photo_context="{not-json"
            )
        )
        assert len(captured["candidates"]) == len(CANDIDATES)


class TestZeroWrite:
    """红线 R1：整个识别链路零写业务集合。"""

    def test_sensitive_collections_unchanged(self, monkeypatch):
        _patch(monkeypatch)
        db = _db()
        before = {c: _run(db.count(c)) for c in SENSITIVE_COLLECTIONS}
        _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        after = {c: _run(db.count(c)) for c in SENSITIVE_COLLECTIONS}
        assert before == after

    def test_no_write_except_audit(self, monkeypatch):
        _patch(monkeypatch)
        db = _db()
        db.reset_write_log()
        _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        # 除审计写外，不允许任何 insert / update / delete
        assert db.write_log == [("insert", "audit_log")]

    def test_classify_status_not_rewritten(self, monkeypatch):
        _patch(monkeypatch)
        db = _db()
        before = db.snapshot("math_scan_upload")
        _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        after = db.snapshot("math_scan_upload")
        assert after == before
        assert after[0]["classify_status"] == "pending"
        assert after[0]["classify_result"] is None


class TestErrors:
    """异常分支：404（scan 不存在 / OCR 未就绪）+ Judge 失败审计。"""

    def test_scan_not_found_raises(self, monkeypatch):
        _patch(monkeypatch)
        with pytest.raises(ScanNotFoundError):
            _run(qr.recognize_quiz_scan(_db(), scan_id="su_nope", scholar_id="s1"))

    def test_ocr_not_ready_raises(self, monkeypatch):
        _patch(monkeypatch)
        db = _db(_scan(ocr_status="processing"))
        with pytest.raises(OcrNotReadyError):
            _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        # OCR 阻塞时不触发 Judge / 不写业务集合
        assert _run(db.count("error_record")) == 0

    def test_missing_scan_id_raises(self, monkeypatch):
        _patch(monkeypatch)
        with pytest.raises(ScanNotFoundError):
            _run(qr.recognize_quiz_scan(_db(), scan_id="", scholar_id="s1"))

    def test_judge_failure_writes_failed_audit(self, monkeypatch):
        async def fake_load_candidates(db, scholar_id):
            return list(CANDIDATES)

        async def boom(ocr_text, cands):
            raise JudgeResponseError("boom")

        monkeypatch.setattr(qr, "_load_knowledge_point_candidates", fake_load_candidates)
        monkeypatch.setattr(qr, "_call_quiz_recognize_judge", boom)
        db = _db()
        db.reset_write_log()
        with pytest.raises(JudgeResponseError):
            _run(qr.recognize_quiz_scan(db, scan_id="su_1", scholar_id="s1"))
        assert db.write_log == [("insert", "audit_log")]
        log = db.all("audit_log")[0]
        assert log["action"] == "quiz_recognize"
        assert log["result"] == "failed"
        assert log["object_ref"] == "su_1"
        # Judge 失败也不回写归类状态
        assert db.snapshot("math_scan_upload")[0]["classify_status"] == "pending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
