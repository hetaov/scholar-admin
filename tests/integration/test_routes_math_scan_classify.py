"""F4.3 扫描归类接口验收测试（契约 api-contract.md §3.10 POST /math/scan/classify）

验收场景：
- 400 缺参（scan_id 为空）
- 404 scan_id 不存在
- 404 OCR 未完成
- 200 高置信归类成功 → 写 error_record + status=success
- 200 低置信 → needs_review（不写 error_record）
- 200 幂等命中（已归类返回既有结果，Judge 不调用）
"""
from __future__ import annotations

import pytest

from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)
from services.routes_math import router as math_router

SCAN_ID = "scan_int_001"
SCHOLAR_ID = "scholar_int_001"


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------


def _seed_scan(
    fake_db,
    *,
    scan_id: str = SCAN_ID,
    ocr_status: str = "success",
    classify_status: str = "pending",
    classify_result: list | None = None,
) -> dict:
    doc = {
        "scan_id": scan_id,
        "scholar_id": SCHOLAR_ID,
        "image_url": "https://example.com/scan.jpg",
        "image_file_id": "cloud://test/scan.jpg",
        "image_hash": "abc123",
        "ocr_status": ocr_status,
        "ocr_text": "1. 计算 3+5=___\n2. 下列哪个是质数？",
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
    fake_db.add(MATH_SCAN_UPLOAD_COLLECTION, doc)
    return doc


def _seed_knowledge_nodes(fake_db) -> None:
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


def _high_conf_judge_result() -> dict:
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


def _low_conf_judge_result() -> dict:
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
# 错误码验收
# ---------------------------------------------------------------------------


class TestScanClassifyErrorCodes:
    def test_400_when_scan_id_missing(self, make_client, fake_db):
        """缺参 scan_id 为空 → 400"""
        client = make_client(math_router)
        res = client.post("/math/scan/classify", json={"scan_id": ""})
        assert res.status_code == 400, res.text
        assert "scan_id" in res.json()["detail"]

    def test_404_when_scan_not_found(self, make_client, fake_db):
        """scan_id 不存在 → 404"""
        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": "no_such_scan"}
        )
        assert res.status_code == 404, res.text

    def test_404_when_ocr_not_ready(self, make_client, fake_db):
        """OCR 未完成（pending）→ 404"""
        _seed_scan(fake_db, ocr_status="pending")
        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# 归类成功
# ---------------------------------------------------------------------------


class TestScanClassifySuccess:
    def test_high_confidence_returns_success(
        self, make_client, fake_db, monkeypatch
    ):
        """高置信 + 知识点匹配 → 200 status=success + error_record 落库"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        async def fake_judge(ocr_text, candidates):
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["scan_id"] == SCAN_ID
        assert data["status"] == "success"
        assert len(data["items"]) == 2
        # 每道题都写了 error_record
        for item in data["items"]:
            assert item["error_record_id"]
            assert item["knowledge_point_name"]
            assert item["error_type"] in (
                "concept",
                "method",
                "computation",
                "reading",
            )

        # error_record 落库 2 条
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 2
        assert records[0]["classify_method"] == "auto_scan"
        assert records[0]["source"] == "auto_scan"

    def test_low_confidence_returns_needs_review(
        self, make_client, fake_db, monkeypatch
    ):
        """低置信 / 知识点未匹配 → 200 status=needs_review（不写 error_record）"""
        _seed_scan(fake_db)
        _seed_knowledge_nodes(fake_db)

        async def fake_judge(ocr_text, candidates):
            return _low_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "needs_review"
        # 第一道高置信写了 error_record；第二道低置信未写
        assert data["items"][0]["error_record_id"]
        assert not data["items"][1]["error_record_id"]
        # error_record 仅落库 1 条
        assert len(fake_db.all(ERROR_RECORD_COLLECTION)) == 1


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


class TestScanClassifyIdempotency:
    def test_already_classified_returns_existing(
        self, make_client, fake_db, monkeypatch
    ):
        """已归类（status=success）非 force → 返回既有结果，Judge 不调用"""
        existing_items = [
            {
                "error_record_id": "er_existing_001",
                "knowledge_point_name": "加法运算",
                "error_type": "computation",
                "confidence": 0.9,
                "ocr_block_id": "blk_0001",
            }
        ]
        _seed_scan(
            fake_db,
            classify_status="success",
            classify_result=existing_items,
        )

        call_count: list = []

        async def fake_judge(ocr_text, candidates):
            call_count.append(1)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify", json={"scan_id": SCAN_ID}
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["status"] == "success"
        assert data["items"] == existing_items
        assert len(call_count) == 0  # Judge 未被调用

    def test_force_reclassify_re_invokes_judge(
        self, make_client, fake_db, monkeypatch
    ):
        """force_reclassify=true → 忽略已有结果，重新调用 Judge"""
        _seed_scan(
            fake_db,
            classify_status="success",
            classify_result=[],
        )
        _seed_knowledge_nodes(fake_db)

        call_count: list = []

        async def fake_judge(ocr_text, candidates):
            call_count.append(1)
            return _high_conf_judge_result()

        monkeypatch.setattr(
            "services.math.error_scanner._call_classify_judge", fake_judge
        )

        client = make_client(math_router)
        res = client.post(
            "/math/scan/classify",
            json={"scan_id": SCAN_ID, "force_reclassify": True},
        )
        assert res.status_code == 200, res.text
        assert res.json()["data"]["status"] == "success"
        assert len(call_count) == 1  # Judge 被重新调用
