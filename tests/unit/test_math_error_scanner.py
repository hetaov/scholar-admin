"""单元测试：错题扫描归类 services/math/error_scanner（F4.3 + F4.4）

F4.3 覆盖：
- 前置校验：scan_id 不存在 → ScanNotFoundError；OCR 未完成 → OcrNotReadyError
- 幂等：classify_status ∈ {success, needs_review} 且非 force_reclassify
  → 直接返回已有结果（Judge 不被调用）
- 高置信 + 知识点匹配 → 写 error_record + classify_status=success
- 低置信 / 知识点未匹配 → 不写 error_record + classify_status=needs_review
- Judge 未配置 → JudgeNotConfiguredError + classify_status=failed + 失败审计
- Judge 响应解析失败 → JudgeResponseError + classify_status=failed + 失败审计
- force_reclassify=true → 忽略已有结果，重新调用 Judge

F4.4 覆盖：
- 前置校验：scan_id 不存在 → ScanNotFoundError；items 为空 → ScanCorrectValidationError
- 已有 error_record_id → 更新归类（classify_method=manual_corrected）
- 无 error_record_id → 新建 error_record（classify_method=manual_corrected, source=manual_corrected）
- 修正后 classify_status 回写 success（契约五态无 corrected）
- 写审计 scan_correct
- error_type 非法 → ScanCorrectValidationError
"""
from __future__ import annotations

import pytest

from services.audit import (
    AUDIT_ACTION_SCAN_CLASSIFY,
    AUDIT_ACTION_SCAN_CORRECT,
    AUDIT_LOG_COLLECTION,
)
from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)
from services.math import error_scanner
from services.math.error_scanner import (
    CLASSIFY_METHOD_MANUAL_CORRECTED,
    CLASSIFY_STATUS_CLASSIFYING,
    CLASSIFY_STATUS_FAILED,
    CLASSIFY_STATUS_NEEDS_REVIEW,
    CLASSIFY_STATUS_PENDING,
    CLASSIFY_STATUS_SUCCESS,
    OCR_STATUS_PENDING,
    OCR_STATUS_SUCCESS,
    SOURCE_MANUAL_CORRECTED,
    JudgeNotConfiguredError,
    JudgeResponseError,
    OcrNotReadyError,
    ScanCorrectValidationError,
    ScanNotFoundError,
    classify_scan_upload,
    correct_scan_classify,
)
from tests.fakes.fake_db import FakeDB

SCAN_ID = "scan_test_001"
SCHOLAR_ID = "scholar_test_001"


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------


def _seed_scan(
    fake_db: FakeDB,
    *,
    scan_id: str = SCAN_ID,
    ocr_status: str = OCR_STATUS_SUCCESS,
    classify_status: str = CLASSIFY_STATUS_PENDING,
    classify_result: list | None = None,
    ocr_text: str = "1. 计算 3+5=___\n2. 下列哪个是质数？",
    **overrides,
) -> dict:
    doc = {
        "scan_id": scan_id,
        "scholar_id": SCHOLAR_ID,
        "image_url": "https://example.com/scan.jpg",
        "image_file_id": "cloud://test/scan.jpg",
        "image_hash": "abc123",
        "ocr_status": ocr_status,
        "ocr_text": ocr_text,
        "ocr_blocks": [
            {"block_id": "blk_0001", "text": "1. 计算 3+5=___"},
            {"block_id": "blk_0002", "text": "2. 下列哪个是质数？"},
        ],
        "classify_status": classify_status,
        "classify_result": classify_result or [],
        "note": "",
        "created_at": 1700000000000,
        "completed_at": None,
        "audit_log_id": "",
    }
    doc.update(overrides)
    fake_db.add(MATH_SCAN_UPLOAD_COLLECTION, doc)
    return doc


def _seed_knowledge_nodes(fake_db: FakeDB) -> None:
    """种子 curriculum_node（含 ai_summary.knowledge_points）"""
    fake_db.add(
        CURRICULUM_NODE_COLLECTION,
        {
            "node_id": "n1",
            "code": "u1",
            "grade": "3",
            "textbook_id": "tb1",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [
                    {"name": "加法运算"},
                    {"name": "质数与合数"},
                ],
            },
        },
    )


def _judge_result_high_confidence() -> dict:
    """高置信 Judge 输出（知识点匹配 + confidence >= 0.6）"""
    return {
        "items": [
            {
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            },
            {
                "knowledge_point_name": "质数与合数",
                "error_type": "concept",
                "confidence": 0.85,
                "ocr_block_id": "blk_0002",
            },
        ]
    }


def _judge_result_low_confidence() -> dict:
    """低置信 Judge 输出（confidence < 0.6 或知识点无法定位）"""
    return {
        "items": [
            {
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            },
            {
                "knowledge_point_name": "未知知识点",
                "error_type": "concept",
                "confidence": 0.3,
                "ocr_block_id": "blk_0002",
            },
        ]
    }


# ---------------------------------------------------------------------------
# 前置校验
# ---------------------------------------------------------------------------


class TestClassifyPreChecks:
    @pytest.mark.asyncio
    async def test_scan_not_found_raises(self):
        """scan_id 不存在 → ScanNotFoundError"""
        db = FakeDB()
        with pytest.raises(ScanNotFoundError):
            await classify_scan_upload(db, scan_id="no_such_scan", actor="u1")

    @pytest.mark.asyncio
    async def test_ocr_not_ready_raises(self):
        """OCR 未完成（status=pending）→ OcrNotReadyError"""
        db = FakeDB()
        _seed_scan(db, ocr_status=OCR_STATUS_PENDING)
        with pytest.raises(OcrNotReadyError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


class TestClassifyIdempotency:
    @pytest.mark.asyncio
    async def test_already_success_returns_existing(self, monkeypatch):
        """classify_status=success 且非 force → 直接返回，Judge 不调用"""
        db = FakeDB()
        existing_items = [
            {
                "error_record_id": "er_existing",
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            }
        ]
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_SUCCESS,
            classify_result=existing_items,
        )
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert len(calls) == 0  # Judge 未被调用
        assert result["scan_id"] == SCAN_ID
        assert result["status"] == "success"
        assert result["items"] == existing_items

    @pytest.mark.asyncio
    async def test_already_needs_review_returns_existing(self, monkeypatch):
        """classify_status=needs_review 且非 force → 直接返回"""
        db = FakeDB()
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_NEEDS_REVIEW,
            classify_result=[],
        )
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert len(calls) == 0
        assert result["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_force_reclassify_re_invokes_judge(self, monkeypatch):
        """force_reclassify=true → 忽略已有结果，重新调用 Judge"""
        db = FakeDB()
        _seed_scan(
            db,
            classify_status=CLASSIFY_STATUS_SUCCESS,
            classify_result=[],
        )
        _seed_knowledge_nodes(db)
        calls: list = []

        async def fake_judge(ocr_text, candidates):
            calls.append(1)
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(
            db, scan_id=SCAN_ID, force_reclassify=True, actor="u1"
        )
        assert len(calls) == 1
        assert result["status"] == "success"


# ---------------------------------------------------------------------------
# 置信度门控
# ---------------------------------------------------------------------------


class TestClassifyConfidenceGating:
    @pytest.mark.asyncio
    async def test_high_confidence_writes_error_record(self, monkeypatch):
        """高置信 + 知识点匹配 → 写 error_record + classify_status=success"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "success"
        assert len(result["items"]) == 2
        # 每道题都写了 error_record
        for item in result["items"]:
            assert item["error_record_id"]

        # error_record 落库 2 条
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        assert records[0]["classify_method"] == "auto_scan"
        assert records[0]["source"] == "auto_scan"
        assert records[0]["scan_upload_id"] == SCAN_ID

        # scan 记录 classify_status 更新为 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS

    @pytest.mark.asyncio
    async def test_low_confidence_skips_error_record(self, monkeypatch):
        """低置信 / 知识点未匹配 → 不写 error_record + classify_status=needs_review"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_low_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "needs_review"
        assert len(result["items"]) == 2
        # 第一道高置信 → 写了 error_record；第二道低置信 → 未写
        assert result["items"][0]["error_record_id"]
        assert not result["items"][1]["error_record_id"]

        # error_record 仅落库 1 条（高置信项）
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_NEEDS_REVIEW

    @pytest.mark.asyncio
    async def test_empty_items_results_in_needs_review(self, monkeypatch):
        """Judge 返回空 items → needs_review（无 error_record 写入）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return {"items": []}

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        result = await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")
        assert result["status"] == "needs_review"
        assert result["items"] == []
        assert len(db.all(ERROR_RECORD_COLLECTION)) == 0


# ---------------------------------------------------------------------------
# Judge 异常处理
# ---------------------------------------------------------------------------


class TestClassifyJudgeErrors:
    @pytest.mark.asyncio
    async def test_judge_not_configured_sets_failed(self, monkeypatch):
        """LLM_JUDGE_MODEL 未配置 → JudgeNotConfiguredError + classify_status=failed"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            raise JudgeNotConfiguredError("not configured")

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        with pytest.raises(JudgeNotConfiguredError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_FAILED

        # 失败审计落库
        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        assert any(
            log.get("action") == AUDIT_ACTION_SCAN_CLASSIFY
            and log.get("result") == "failed"
            for log in audit_logs
        )

    @pytest.mark.asyncio
    async def test_judge_response_error_sets_failed(self, monkeypatch):
        """Judge 响应解析失败 → JudgeResponseError + classify_status=failed"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            raise JudgeResponseError("bad json")

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        with pytest.raises(JudgeResponseError):
            await classify_scan_upload(db, scan_id=SCAN_ID, actor="u1")

        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_FAILED


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


class TestClassifyAudit:
    @pytest.mark.asyncio
    async def test_success_writes_audit(self, monkeypatch):
        """归类成功 → 写 scan_classify 审计（result=success）"""
        db = FakeDB()
        _seed_scan(db)
        _seed_knowledge_nodes(db)

        async def fake_judge(ocr_text, candidates):
            return _judge_result_high_confidence()

        monkeypatch.setattr(error_scanner, "_call_classify_judge", fake_judge)

        await classify_scan_upload(db, scan_id=SCAN_ID, actor="actor_001")

        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        classify_audits = [
            log for log in audit_logs if log.get("action") == AUDIT_ACTION_SCAN_CLASSIFY
        ]
        assert len(classify_audits) == 1
        assert classify_audits[0]["actor"] == "actor_001"
        assert classify_audits[0]["object_ref"] == SCAN_ID


# ===========================================================================
# F4.4 人工修正归类（correct_scan_classify）
# ===========================================================================


class TestCorrectPreChecks:
    @pytest.mark.asyncio
    async def test_scan_not_found_raises(self):
        """scan_id 不存在 → ScanNotFoundError"""
        db = FakeDB()
        with pytest.raises(ScanNotFoundError):
            await correct_scan_classify(
                db,
                scan_id="no_such_scan",
                items=[{"knowledge_point_name": "加法运算", "error_type": "computation"}],
                actor="u1",
            )

    @pytest.mark.asyncio
    async def test_empty_items_raises(self):
        """items 为空 → ScanCorrectValidationError"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db, scan_id=SCAN_ID, items=[], actor="u1"
            )

    @pytest.mark.asyncio
    async def test_invalid_error_type_raises(self):
        """error_type 非四分类 → ScanCorrectValidationError"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db,
                scan_id=SCAN_ID,
                items=[{"knowledge_point_name": "加法运算", "error_type": "unknown"}],
                actor="u1",
            )

    @pytest.mark.asyncio
    async def test_new_item_without_kp_or_error_raises(self):
        """无 error_record_id 且无 knowledge_point_name/error_type → 校验失败"""
        db = FakeDB()
        _seed_scan(db)
        with pytest.raises(ScanCorrectValidationError):
            await correct_scan_classify(
                db,
                scan_id=SCAN_ID,
                items=[{"raw_text_corrected": "some text"}],
                actor="u1",
            )


class TestCorrectUpdateExisting:
    @pytest.mark.asyncio
    async def test_update_existing_error_record(self):
        """已有 error_record_id → 更新归类（classify_method=manual_corrected）"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)
        # 预置一条 auto_scan 归类的 error_record
        db.add(
            ERROR_RECORD_COLLECTION,
            {
                "record_id": "er_existing_001",
                "scholar_id": SCHOLAR_ID,
                "node_code": "",
                "primary_error": "computation",
                "classify_method": "auto_scan",
                "source": "auto_scan",
                "scan_upload_id": SCAN_ID,
                "confidence": 0.3,
            },
        )

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "error_record_id": "er_existing_001",
                    "knowledge_point_name": "加法运算",
                    "error_type": "concept",
                }
            ],
            actor="actor_001",
        )

        assert result["scan_id"] == SCAN_ID
        assert len(result["corrected"]) == 1
        assert result["corrected"][0]["error_record_id"] == "er_existing_001"
        assert result["corrected"][0]["knowledge_point_name"] == "加法运算"
        assert result["corrected"][0]["error_type"] == "concept"

        # error_record 已更新
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED
        assert records[0]["primary_error"] == "concept"
        assert records[0]["node_code"] == "u1"  # 匹配候选集 node_code
        assert "corrected_at" in records[0]

        # scan classify_status 回写 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS


class TestCorrectCreateNew:
    @pytest.mark.asyncio
    async def test_create_new_error_record(self):
        """无 error_record_id → 新建 error_record（source=manual_corrected）"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "knowledge_point_name": "质数与合数",
                    "error_type": "reading",
                    "raw_text_corrected": "修正后的题干",
                }
            ],
            actor="actor_001",
        )

        assert len(result["corrected"]) == 1
        new_id = result["corrected"][0]["error_record_id"]
        assert new_id  # 新生成 record_id
        assert result["corrected"][0]["knowledge_point_name"] == "质数与合数"
        assert result["corrected"][0]["error_type"] == "reading"

        # error_record 落库
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["record_id"] == new_id
        assert records[0]["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED
        assert records[0]["source"] == SOURCE_MANUAL_CORRECTED
        assert records[0]["scan_upload_id"] == SCAN_ID
        assert records[0]["primary_error"] == "reading"
        assert records[0]["node_code"] == "u1"
        assert records[0]["confidence"] == 1.0
        assert records[0]["raw_text_corrected"] == "修正后的题干"

        # scan classify_status 回写 success
        scans = db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == CLASSIFY_STATUS_SUCCESS


class TestCorrectMixed:
    @pytest.mark.asyncio
    async def test_update_and_create_in_same_batch(self):
        """同一批修正：已有 record 更新 + 新建并存"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)
        db.add(
            ERROR_RECORD_COLLECTION,
            {
                "record_id": "er_old_001",
                "scholar_id": SCHOLAR_ID,
                "node_code": "",
                "primary_error": "computation",
                "classify_method": "auto_scan",
                "source": "auto_scan",
                "scan_upload_id": SCAN_ID,
            },
        )

        result = await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {
                    "error_record_id": "er_old_001",
                    "knowledge_point_name": "加法运算",
                    "error_type": "concept",
                },
                {
                    "knowledge_point_name": "质数与合数",
                    "error_type": "method",
                },
            ],
            actor="actor_001",
        )

        assert len(result["corrected"]) == 2
        # 第一项更新
        assert result["corrected"][0]["error_record_id"] == "er_old_001"
        # 第二项新建
        assert result["corrected"][1]["error_record_id"] != "er_old_001"

        # error_record 共 2 条
        records = db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        # 全部 classify_method=manual_corrected
        assert all(r["classify_method"] == CLASSIFY_METHOD_MANUAL_CORRECTED for r in records)


class TestCorrectAudit:
    @pytest.mark.asyncio
    async def test_correct_writes_audit(self):
        """修正成功 → 写 scan_correct 审计"""
        db = FakeDB()
        _seed_scan(db, classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed_knowledge_nodes(db)

        await correct_scan_classify(
            db,
            scan_id=SCAN_ID,
            items=[
                {"knowledge_point_name": "加法运算", "error_type": "computation"}
            ],
            actor="actor_001",
        )

        audit_logs = db.all(AUDIT_LOG_COLLECTION)
        correct_audits = [
            log for log in audit_logs if log.get("action") == AUDIT_ACTION_SCAN_CORRECT
        ]
        assert len(correct_audits) == 1
        assert correct_audits[0]["actor"] == "actor_001"
        assert correct_audits[0]["object_ref"] == SCAN_ID
        assert correct_audits[0]["context"]["items_count"] == 1
        assert correct_audits[0]["context"]["created_count"] == 1
