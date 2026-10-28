"""GET /math/scan/pending 接口验收测试（F4.5 / P-61 · 契约 api-contract.md §3.10）

为什么要有这个接口：D6 裁决（错题双入口）接线时发现，前端扫完只弹 toast 告知
「N 张待人工确认」，而服务层没有「拉历史待确认」的查询能力 —— 跳走或重进页面就
查不到，人工归类只能「当场做完」，做不到「以后再补」。

验收场景：
- 200 出参 { success, data: { total, items[] } }，只含 needs_review
- 400 缺 scholar_id
- 按 scholar_id 隔离（别人的待确认不串号）
"""
from __future__ import annotations

from services.database import MATH_SCAN_UPLOAD_COLLECTION
from services.routes_math import router as math_router

SCHOLAR_ID = "scholar_pending_001"
OTHER_SCHOLAR_ID = "scholar_pending_002"


def _seed(fake_db, scan_id, *, scholar_id=SCHOLAR_ID, classify_status="needs_review", created_at=1700000000000):
    fake_db.add(
        MATH_SCAN_UPLOAD_COLLECTION,
        {
            "scan_id": scan_id,
            "scholar_id": scholar_id,
            "image_url": "https://example.com/scan.jpg",
            "classify_status": classify_status,
            "classify_result": [
                {
                    "error_record_id": "",
                    "knowledge_point_name": "",
                    "error_type": "",
                    "confidence": 0.4,
                    "ocr_block_id": "blk_0001",
                }
            ],
            "created_at": created_at,
        },
    )


class TestScanPending:
    def test_200_shape(self, make_client, fake_db):
        """出参 { success, data:{ total, items[] } }，字段齐备"""
        _seed(fake_db, "s1")
        client = make_client(math_router)

        res = client.get("/math/scan/pending", params={"scholar_id": SCHOLAR_ID})

        assert res.status_code == 200
        body = res.json()
        assert body["success"] is True
        assert body["data"]["total"] == 1
        item = body["data"]["items"][0]
        assert item["scan_id"] == "s1"
        assert item["status"] == "needs_review"
        assert item["items"][0]["ocr_block_id"] == "blk_0001"

    def test_200_only_needs_review(self, make_client, fake_db):
        """success / failed 不算人工待办"""
        _seed(fake_db, "s_review", classify_status="needs_review")
        _seed(fake_db, "s_success", classify_status="success")
        _seed(fake_db, "s_failed", classify_status="failed")
        client = make_client(math_router)

        res = client.get("/math/scan/pending", params={"scholar_id": SCHOLAR_ID})

        assert [i["scan_id"] for i in res.json()["data"]["items"]] == ["s_review"]

    def test_400_without_scholar(self, make_client, fake_db):
        """缺 scholar_id → 400（不让全表扫描）"""
        _seed(fake_db, "s1")
        client = make_client(math_router)

        res = client.get("/math/scan/pending")

        assert res.status_code == 400

    def test_scoped_by_scholar(self, make_client, fake_db):
        """别人的待确认不串号"""
        _seed(fake_db, "s_mine")
        _seed(fake_db, "s_other", scholar_id=OTHER_SCHOLAR_ID)
        client = make_client(math_router)

        res = client.get("/math/scan/pending", params={"scholar_id": SCHOLAR_ID})

        assert [i["scan_id"] for i in res.json()["data"]["items"]] == ["s_mine"]
