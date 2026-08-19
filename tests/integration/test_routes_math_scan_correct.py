"""F4.4 人工修正归类接口验收测试（契约 api-contract.md §3.10 POST /math/scan/{scan_id}/correct）

验收场景：
- 400 items 为空
- 400 error_type 非法
- 404 scan_id 不存在
- 200 更新已有 error_record（classify_method=manual_corrected）
- 200 新建 error_record（source=manual_corrected）
- 200 混合修正（更新 + 新建）
- 200 修正后 classify_status 回写 success + 审计落库
"""
from __future__ import annotations

from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)
from services.routes_math import router as math_router

SCAN_ID = "scan_correct_001"
SCHOLAR_ID = "scholar_correct_001"


# ---------------------------------------------------------------------------
# 种子数据
# ---------------------------------------------------------------------------


def _seed_scan(
    fake_db,
    *,
    scan_id: str = SCAN_ID,
    classify_status: str = "needs_review",
) -> dict:
    doc = {
        "scan_id": scan_id,
        "scholar_id": SCHOLAR_ID,
        "image_url": "https://example.com/scan.jpg",
        "image_file_id": "cloud://test/scan.jpg",
        "image_hash": "abc123",
        "ocr_status": "success",
        "ocr_text": "1. 计算 3+5=___",
        "ocr_blocks": [{"block_id": "blk_0001", "text": "1. 计算 3+5=___"}],
        "classify_status": classify_status,
        "classify_result": [],
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


def _seed_error_record(fake_db, record_id="er_auto_001") -> dict:
    doc = {
        "record_id": record_id,
        "scholar_id": SCHOLAR_ID,
        "attempt_ref": "",
        "node_code": "",
        "primary_error": "computation",
        "secondary_error": None,
        "stuck_step": None,
        "occurrence": 1,
        "created_at": 1700000000000,
        "scan_upload_id": SCAN_ID,
        "classify_method": "auto_scan",
        "ocr_block_id": "blk_0001",
        "source": "auto_scan",
        "confidence": 0.3,
    }
    fake_db.add(ERROR_RECORD_COLLECTION, doc)
    return doc


# ---------------------------------------------------------------------------
# 错误码验收
# ---------------------------------------------------------------------------


class TestScanCorrectErrorCodes:
    def test_400_when_items_empty(self, make_client, fake_db):
        """items 为空 → 400"""
        _seed_scan(fake_db)
        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct", json={"items": []}
        )
        assert res.status_code == 400, res.text
        assert "items" in res.json()["detail"]

    def test_400_when_error_type_invalid(self, make_client, fake_db):
        """error_type 非四分类 → 400"""
        _seed_scan(fake_db)
        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct",
            json={
                "items": [
                    {
                        "knowledge_point_name": "加法运算",
                        "error_type": "unknown_type",
                    }
                ]
            },
        )
        assert res.status_code == 400, res.text

    def test_404_when_scan_not_found(self, make_client, fake_db):
        """scan_id 不存在 → 404"""
        client = make_client(math_router)
        res = client.post(
            "/math/scan/no_such_scan/correct",
            json={
                "items": [
                    {"knowledge_point_name": "加法运算", "error_type": "computation"}
                ]
            },
        )
        assert res.status_code == 404, res.text


# ---------------------------------------------------------------------------
# 修正成功
# ---------------------------------------------------------------------------


class TestScanCorrectSuccess:
    def test_update_existing_error_record(self, make_client, fake_db):
        """已有 error_record_id → 200 更新归类（classify_method=manual_corrected）"""
        _seed_scan(fake_db, classify_status="needs_review")
        _seed_knowledge_nodes(fake_db)
        _seed_error_record(fake_db, "er_auto_001")

        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct",
            json={
                "items": [
                    {
                        "error_record_id": "er_auto_001",
                        "knowledge_point_name": "加法运算",
                        "error_type": "concept",
                    }
                ]
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["scan_id"] == SCAN_ID
        assert len(data["corrected"]) == 1
        assert data["corrected"][0]["error_record_id"] == "er_auto_001"
        assert data["corrected"][0]["knowledge_point_name"] == "加法运算"
        assert data["corrected"][0]["error_type"] == "concept"

        # error_record 已更新
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["classify_method"] == "manual_corrected"
        assert records[0]["primary_error"] == "concept"

        # classify_status 回写 success
        scans = fake_db.all(MATH_SCAN_UPLOAD_COLLECTION)
        assert scans[0]["classify_status"] == "success"

    def test_create_new_error_record(self, make_client, fake_db):
        """无 error_record_id → 200 新建（source=manual_corrected）"""
        _seed_scan(fake_db, classify_status="needs_review")
        _seed_knowledge_nodes(fake_db)

        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct",
            json={
                "items": [
                    {
                        "knowledge_point_name": "质数与合数",
                        "error_type": "reading",
                        "raw_text_corrected": "修正题干",
                    }
                ]
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert len(data["corrected"]) == 1
        new_id = data["corrected"][0]["error_record_id"]
        assert new_id

        # error_record 落库
        records = fake_db.all(ERROR_RECORD_COLLECTION)
        assert len(records) == 1
        assert records[0]["record_id"] == new_id
        assert records[0]["classify_method"] == "manual_corrected"
        assert records[0]["source"] == "manual_corrected"
        assert records[0]["raw_text_corrected"] == "修正题干"

    def test_mixed_update_and_create(self, make_client, fake_db):
        """混合修正：更新 + 新建"""
        _seed_scan(fake_db, classify_status="needs_review")
        _seed_knowledge_nodes(fake_db)
        _seed_error_record(fake_db, "er_old_001")

        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct",
            json={
                "items": [
                    {
                        "error_record_id": "er_old_001",
                        "knowledge_point_name": "加法运算",
                        "error_type": "concept",
                    },
                    {
                        "knowledge_point_name": "质数与合数",
                        "error_type": "method",
                    },
                ]
            },
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert len(data["corrected"]) == 2
        # 第一项更新
        assert data["corrected"][0]["error_record_id"] == "er_old_001"
        # 第二项新建
        assert data["corrected"][1]["error_record_id"]
        assert data["corrected"][1]["error_record_id"] != "er_old_001"

        # error_record 共 2 条
        assert len(fake_db.all(ERROR_RECORD_COLLECTION)) == 2

    def test_correct_writes_audit(self, make_client, fake_db):
        """修正成功 → 审计 scan_correct 落库"""
        _seed_scan(fake_db, classify_status="needs_review")
        _seed_knowledge_nodes(fake_db)

        client = make_client(math_router)
        res = client.post(
            f"/math/scan/{SCAN_ID}/correct",
            json={
                "items": [
                    {"knowledge_point_name": "加法运算", "error_type": "computation"}
                ]
            },
        )
        assert res.status_code == 200, res.text

        audit_logs = fake_db.all("audit_log")
        correct_audits = [
            log for log in audit_logs if log.get("action") == "scan_correct"
        ]
        assert len(correct_audits) == 1
        assert correct_audits[0]["object_ref"] == SCAN_ID
