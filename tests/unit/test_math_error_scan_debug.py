"""数学错题识别 Admin 调试干跑（error_scan_debug.py）单元测试

B03：_load_knowledge_point_candidates_by_textbook 候选集只读加载
B04：干跑主干 _validate_image → OCR → Judge → 组装 items[]
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from services.math.error_scan_debug import (
    _assemble_dry_run_items,
    _gen_debug_scan_id,
    _load_knowledge_point_candidates_by_textbook,
    _run_ocr_for_debug,
    recognize_error_scan_dry_run,
)
from services.math.ocr import OcrError, OcrResult
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _kp_node(
    node_id: str = "n1",
    code: str = "5-3-1-kp2",
    kp_name: str = "小数加减法",
    textbook_id: str = "tb_1",
    grade: str = "五年级",
    semester: str = "up",
    title: str = "小数乘整数",
    unit_title: str = "第3单元 小数乘法",
    lesson_title: str = "小数乘整数",
    status: str = "success",
) -> dict:
    """构造一个带 ai_summary.knowledge_points 的 curriculum_node 文档。"""
    return {
        "node_id": node_id,
        "code": code,
        "textbook_id": textbook_id,
        "grade": grade,
        "semester": semester,
        "title": title,
        "unit_title": unit_title,
        "lesson_title": lesson_title,
        "ai_summary": {"status": status, "knowledge_points": [{"name": kp_name}]},
    }


# ---------------- B03: _load_knowledge_point_candidates_by_textbook ----------------


class TestLoadKpCandidatesByTextbook:
    """候选集只读加载（B03）。

    覆盖：
    - 按 textbook_id 精确加载
    - textbook_id 为空 → 全量候选
    - textbook_id 无匹配 → 回退全量
    - 候选项字段结构正确
    - 不读 scholar_book 集合
    - debug 段 source/count/prompt_included/truncated 正确
    """

    def test_textbook_id_filter(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["source"] == "textbook_id"
        assert result["count"] == 1
        assert result["candidates"][0]["kp_name"] == "小数加减法"
        assert result["candidates"][0]["textbook_id"] == "tb_1"

    def test_empty_textbook_id_loads_all(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, ""))
        assert result["source"] == "all"
        assert result["count"] == 2

    def test_textbook_id_no_match_fallback_all(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_nonexistent"))
        assert result["source"] == "all"
        assert result["count"] == 1

    def test_candidate_fields_complete(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node())
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        c = result["candidates"][0]
        assert c["node_id"] == "n1"
        assert c["node_code"] == "5-3-1-kp2"
        assert c["kp_name"] == "小数加减法"
        assert c["grade"] == "五年级"
        assert c["semester"] == "up"
        assert c["textbook_id"] == "tb_1"
        assert c["title"] == "小数乘整数"
        assert c["unit_title"] == "第3单元 小数乘法"
        assert c["lesson_title"] == "小数乘整数"

    def test_skips_nodes_without_success_summary(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(status="pending"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 0

    def test_skips_kp_with_empty_name(self):
        node = _kp_node()
        node["ai_summary"]["knowledge_points"] = [{"name": ""}, {"name": "  "}]
        db = FakeDB()
        db.add("curriculum_node", node)
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 0

    def test_truncated_when_exceeds_limit(self, monkeypatch):
        monkeypatch.setattr("config.LLM_JUDGE_CANDIDATE_LIMIT", 2)
        db = FakeDB()
        for i in range(5):
            db.add(
                "curriculum_node",
                _kp_node(node_id=f"n{i}", code=f"c{i}", kp_name=f"kp{i}"),
            )
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 5
        assert result["prompt_included"] == 2
        assert result["truncated"] is True

    def test_no_truncation_within_limit(self, monkeypatch):
        monkeypatch.setattr("config.LLM_JUDGE_CANDIDATE_LIMIT", 40)
        db = FakeDB()
        db.add("curriculum_node", _kp_node())
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["prompt_included"] == 1
        assert result["truncated"] is False

    def test_does_not_read_scholar_book(self):
        """干跑不读 scholar_book 集合（与生产路径隔离）。

        若干跑读了 scholar_book，结果会被指向 tb_2，但实际传入 textbook_id=tb_1
        仍应只返回 tb_1 的候选——证明不走 scholar_book 路径。
        """
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        db.add("scholar_book", {"scholar_id": "s1", "textbook_id": "tb_2"})
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1", scholar_id="s1"))
        assert result["count"] == 1
        assert result["candidates"][0]["textbook_id"] == "tb_1"


# ---------------- B04: 干跑主干 ----------------


class TestGenDebugScanId:
    """_gen_debug_scan_id 生成 debug_ 前缀的 scan_id。"""

    def test_prefix_is_debug(self):
        sid = _gen_debug_scan_id()
        assert sid.startswith("debug_")

    def test_unique(self):
        ids = {_gen_debug_scan_id() for _ in range(10)}
        assert len(ids) == 10


class TestRunOcrForDebug:
    """_run_ocr_for_debug 调用真实 provider（mock 测试）。"""

    def test_provider_unavailable_returns_error_info(self):
        provider = MagicMock()
        provider.__class__.__name__ = "FakeProvider"
        provider.available = False
        provider._engine = "test_engine"
        with patch("services.math.ocr.get_provider", return_value=provider):
            result = _run(_run_ocr_for_debug(b"fake"))
        assert result["available"] is False
        assert result["error"] == "OCR 未配置凭据"
        assert result["text"] == ""

    def test_provider_available_returns_ocr_result(self):
        provider = MagicMock()
        provider.__class__.__name__ = "FakeProvider"
        provider.available = True
        provider._engine = "general_accurate"
        provider.recognize = AsyncMock(
            return_value=OcrResult(text="0.36 + 2.7 = ?", blocks=[{"block_id": "blk_0001"}])
        )
        with patch("services.math.ocr.get_provider", return_value=provider):
            result = _run(_run_ocr_for_debug(b"fake_image"))
        assert result["available"] is True
        assert result["text"] == "0.36 + 2.7 = ?"
        assert result["blocks_count"] == 1
        assert result["text_length"] == 14
        assert result["ocr_ms"] >= 0

    def test_ocr_error_propagates(self):
        provider = MagicMock()
        provider.__class__.__name__ = "FakeProvider"
        provider.available = True
        provider._engine = "test"
        provider.recognize = AsyncMock(side_effect=OcrError("OCR 调用失败"))
        with patch("services.math.ocr.get_provider", return_value=provider):
            try:
                _run(_run_ocr_for_debug(b"fake"))
                assert False, "应抛 OcrError"
            except OcrError as e:
                assert "干跑 OCR 失败" in str(e)


class TestAssembleDryRunItems:
    """_assemble_dry_run_items 组装干跑 items[]。"""

    def test_exact_match_item(self):
        candidates = [
            {"node_id": "n1", "node_code": "5-3-1-kp2", "kp_name": "小数加减法",
             "textbook_id": "tb_1", "grade": "五年级", "semester": "up",
             "title": "小数乘整数", "unit_title": "第3单元", "lesson_title": "小数乘整数"}
        ]
        judge_result = {
            "items": [
                {"knowledge_point_name": "小数加减法", "error_type": "computation",
                 "confidence": 0.92, "ocr_block_id": "blk_0001", "question_text": "0.36+2.7=?"}
            ]
        }
        items = _run(_assemble_dry_run_items(judge_result, candidates, "tb_1", 0.6))
        assert len(items) == 1
        assert items[0]["knowledge_point_name"] == "小数加减法"
        assert items[0]["error_record_id"] == ""  # 干跑不落库
        assert items[0]["confidence"] == 0.92
        assert "original_kp_name" not in items[0]  # 非 renamed
        assert "new_kp_name" not in items[0]  # 非 extra_ai

    def test_renamed_item_has_original_kp_name(self):
        candidates = [
            {"node_id": "n1", "node_code": "c1", "kp_name": "小数加减法",
             "textbook_id": "tb_1", "grade": "五年级", "semester": "up",
             "title": "", "unit_title": "", "lesson_title": ""}
        ]
        judge_result = {
            "items": [
                {"knowledge_point_name": "小数的加减运算", "error_type": "computation",
                 "confidence": 0.85, "ocr_block_id": "blk_0001", "question_text": "0.36+2.7=?"}
            ]
        }
        items = _run(_assemble_dry_run_items(judge_result, candidates, "tb_1", 0.6))
        assert len(items) == 1
        assert items[0]["knowledge_point_name"] == "小数加减法"  # 改名为候选标准名
        assert items[0]["original_kp_name"] == "小数的加减运算"

    def test_extra_ai_item_has_new_kp_name(self):
        """无候选匹配且高置信 → extra_ai 预演（不调 _ensure_extra_ai_node）。"""
        candidates = []  # 空候选集
        judge_result = {
            "items": [
                {"knowledge_point_name": "全新的知识点", "error_type": "concept",
                 "confidence": 0.88, "ocr_block_id": "blk_0001", "question_text": "某题"}
            ]
        }
        items = _run(_assemble_dry_run_items(judge_result, candidates, "tb_1", 0.6))
        assert len(items) == 1
        assert items[0]["new_kp_name"] == "全新的知识点"
        assert items[0]["knowledge_point_name"] == "全新的知识点"

    def test_low_confidence_no_match(self):
        """低置信无匹配 → 无 original_kp_name / new_kp_name。"""
        candidates = [
            {"node_id": "n1", "node_code": "c1", "kp_name": "小数加减法",
             "textbook_id": "tb_1", "grade": "", "semester": "",
             "title": "", "unit_title": "", "lesson_title": ""}
        ]
        judge_result = {
            "items": [
                {"knowledge_point_name": "完全不同的知识点", "error_type": "concept",
                 "confidence": 0.3, "ocr_block_id": "blk_0001", "question_text": "?"}
            ]
        }
        items = _run(_assemble_dry_run_items(judge_result, candidates, "tb_1", 0.6))
        assert len(items) == 1
        assert "original_kp_name" not in items[0]
        assert "new_kp_name" not in items[0]

    def test_empty_judge_items(self):
        items = _run(_assemble_dry_run_items({}, [], "tb_1", 0.6))
        assert items == []


class TestRecognizeErrorScanDryRun:
    """B04 干跑主干入口集成测试（mock OCR + Judge）。"""

    def test_dry_run_returns_items_and_debug(self, monkeypatch):
        """端到端干跑：mock OCR + Judge，验证 items[] + debug 结构。"""
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))

        # mock OCR provider
        ocr_provider = MagicMock()
        ocr_provider.__class__.__name__ = "FakeProvider"
        ocr_provider.available = True
        ocr_provider._engine = "test"
        ocr_provider.recognize = AsyncMock(
            return_value=OcrResult(text="0.36 + 2.7 = ?", blocks=[{"block_id": "blk_0001"}])
        )

        # mock Judge
        judge_result = {
            "items": [
                {"knowledge_point_name": "小数加减法", "error_type": "computation",
                 "confidence": 0.92, "ocr_block_id": "blk_0001", "question_text": "0.36+2.7=?"}
            ]
        }

        with patch("services.math.ocr.get_provider", return_value=ocr_provider), \
             patch("services.math.error_scan_debug._call_classify_judge", new=AsyncMock(return_value=judge_result)):
            result = _run(recognize_error_scan_dry_run(
                db,
                image_bytes=b"fake_image_bytes",
                filename="test.jpg",
                textbook_id="tb_1",
            ))

        assert result["scan_id"].startswith("debug_")
        assert result["status"] == "success"
        assert len(result["items"]) == 1
        assert result["items"][0]["knowledge_point_name"] == "小数加减法"
        assert result["items"][0]["error_record_id"] == ""

        debug = result["debug"]
        assert debug["dry_run"] is True
        assert debug["persisted"] is False
        assert debug["request"]["textbook_id"] == "tb_1"
        assert debug["request"]["image_ext"] == "jpg"
        assert debug["timings"]["ocr_ms"] >= 0
        assert debug["timings"]["judge_ms"] >= 0
        assert debug["candidates"]["source"] == "textbook_id"
        assert debug["candidates"]["count"] == 1
        assert debug["ocr"]["available"] is True
        assert debug["ocr"]["text"] == "0.36 + 2.7 = ?"
        assert debug["judge"]["model"] is not None

    def test_dry_run_invalid_image_raises(self):
        """图片格式不合法 → ImageValidationError。"""
        from services.math.error_scanner import ImageValidationError

        db = FakeDB()
        try:
            _run(recognize_error_scan_dry_run(
                db,
                image_bytes=b"fake",
                filename="test.gif",  # 不支持的格式
                textbook_id="tb_1",
            ))
            assert False, "应抛 ImageValidationError"
        except ImageValidationError:
            pass

    def test_dry_run_ocr_unavailable_raises(self):
        """OCR 不可用 → OcrError。"""
        from services.math.ocr import OcrError

        db = FakeDB()
        ocr_provider = MagicMock()
        ocr_provider.__class__.__name__ = "FakeProvider"
        ocr_provider.available = False
        ocr_provider._engine = "test"

        with patch("services.math.ocr.get_provider", return_value=ocr_provider):
            try:
                _run(recognize_error_scan_dry_run(
                    db,
                    image_bytes=b"fake",
                    filename="test.jpg",
                    textbook_id="tb_1",
                ))
                assert False, "应抛 OcrError"
            except OcrError:
                pass

    def test_dry_run_no_db_writes(self, monkeypatch):
        """红线：干跑后 FakeDB 任何集合都不应有写操作。"""
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))

        # 记录初始数据快照
        initial_data = {k: [dict(r) for r in v] for k, v in db._data.items()}

        ocr_provider = MagicMock()
        ocr_provider.__class__.__name__ = "FakeProvider"
        ocr_provider.available = True
        ocr_provider._engine = "test"
        ocr_provider.recognize = AsyncMock(
            return_value=OcrResult(text="test", blocks=[])
        )

        judge_result = {"items": []}

        with patch("services.math.ocr.get_provider", return_value=ocr_provider), \
             patch("services.math.error_scan_debug._call_classify_judge", new=AsyncMock(return_value=judge_result)):
            _run(recognize_error_scan_dry_run(
                db,
                image_bytes=b"fake",
                filename="test.jpg",
                textbook_id="tb_1",
            ))

        # 验证：无新集合、无新文档、无修改
        for coll, rows in initial_data.items():
            assert len(db._data.get(coll, [])) == len(rows), f"集合 {coll} 文档数变化"
        # 无新增集合
        new_collections = set(db._data.keys()) - set(initial_data.keys())
        assert not new_collections, f"干跑新增了集合: {new_collections}"
