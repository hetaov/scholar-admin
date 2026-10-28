"""单元测试：待人工确认列表 services/math/error_scanner.list_pending_scans（F4.5 / P-61）

背景（D6 裁决接线时暴露的缺口）：链路原本只有 upload / classify / correct 三个
动作，前端扫完才知道有几张待确认，跳走或重进页面就再也查不到 —— 人工归类只能
「当场做完」，做不到「以后再补」。本接口把「历史待确认」变成可查询。

覆盖：
- 只返回 needs_review（success / failed / classifying / pending 都不算人工待办）
- 按 scholar_id 隔离（别人的待确认不串号）
- items 取落库的 classify_result；缺失字段回退空数组/0
- total 与 items 长度一致
- limit 夹取（<=0 → 1；>100 → 100）
- 未传 scholar_id → MissingScholarError
"""
from __future__ import annotations

import pytest

from services.database import MATH_SCAN_UPLOAD_COLLECTION
from services.math.error_scanner import (
    CLASSIFY_STATUS_CLASSIFYING,
    CLASSIFY_STATUS_FAILED,
    CLASSIFY_STATUS_NEEDS_REVIEW,
    CLASSIFY_STATUS_PENDING,
    CLASSIFY_STATUS_SUCCESS,
    MissingScholarError,
    list_pending_scans,
)
from tests.fakes.fake_db import FakeDB

SCHOLAR_A = "scholar_a"
SCHOLAR_B = "scholar_b"


def _seed(
    db: FakeDB,
    scan_id: str,
    *,
    scholar_id: str = SCHOLAR_A,
    classify_status: str = CLASSIFY_STATUS_NEEDS_REVIEW,
    classify_result: list | None = None,
    created_at: int = 1700000000000,
    image_url: str = "https://example.com/scan.jpg",
) -> None:
    db.add(
        MATH_SCAN_UPLOAD_COLLECTION,
        {
            "scan_id": scan_id,
            "scholar_id": scholar_id,
            "image_url": image_url,
            "classify_status": classify_status,
            "classify_result": classify_result or [],
            "created_at": created_at,
        },
    )


class TestPendingFiltering:
    @pytest.mark.asyncio
    async def test_only_needs_review_returned(self):
        """五态里只有 needs_review 属于人工待办"""
        db = FakeDB()
        _seed(db, "s_review", classify_status=CLASSIFY_STATUS_NEEDS_REVIEW)
        _seed(db, "s_success", classify_status=CLASSIFY_STATUS_SUCCESS)
        _seed(db, "s_failed", classify_status=CLASSIFY_STATUS_FAILED)
        _seed(db, "s_pending", classify_status=CLASSIFY_STATUS_PENDING)
        _seed(db, "s_classifying", classify_status=CLASSIFY_STATUS_CLASSIFYING)

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        assert [i["scan_id"] for i in data["items"]] == ["s_review"]
        assert data["total"] == 1

    @pytest.mark.asyncio
    async def test_scoped_by_scholar(self):
        """别人的待确认不串号"""
        db = FakeDB()
        _seed(db, "s_a", scholar_id=SCHOLAR_A)
        _seed(db, "s_b", scholar_id=SCHOLAR_B)

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        assert [i["scan_id"] for i in data["items"]] == ["s_a"]

    @pytest.mark.asyncio
    async def test_sorted_by_created_at_desc(self):
        """最新扫的排前面（人工先处理近的）"""
        db = FakeDB()
        _seed(db, "old", created_at=1700000000000)
        _seed(db, "new", created_at=1800000000000)

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        assert [i["scan_id"] for i in data["items"]] == ["new", "old"]


class TestPendingPayload:
    @pytest.mark.asyncio
    async def test_items_from_classify_result(self):
        """items = 落库的 classify_result（归类产出，前端拿到即可渲染修正表单）"""
        db = FakeDB()
        items = [
            {
                "error_record_id": "",
                "knowledge_point_name": "",
                "error_type": "",
                "confidence": 0.4,
                "ocr_block_id": "blk_0001",
            }
        ]
        _seed(db, "s1", classify_result=items)

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        assert data["items"][0]["items"] == items
        assert data["items"][0]["status"] == CLASSIFY_STATUS_NEEDS_REVIEW
        assert data["items"][0]["image_url"] == "https://example.com/scan.jpg"
        assert data["items"][0]["created_at"] == 1700000000000

    @pytest.mark.asyncio
    async def test_missing_fields_fallback(self):
        """历史记录缺字段 → 回退空数组/0，不抛异常（存量数据兼容）"""
        db = FakeDB()
        db.add(
            MATH_SCAN_UPLOAD_COLLECTION,
            {"scan_id": "s_bare", "scholar_id": SCHOLAR_A, "classify_status": CLASSIFY_STATUS_NEEDS_REVIEW},
        )

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        item = data["items"][0]
        assert item["items"] == []
        assert item["image_url"] == ""
        assert item["created_at"] == 0

    @pytest.mark.asyncio
    async def test_total_matches(self):
        """total 与返回条数一致（前端据此决定横幅是否显示）"""
        db = FakeDB()
        for i in range(3):
            _seed(db, f"s{i}", created_at=1700000000000 + i)

        data = await list_pending_scans(db, scholar_id=SCHOLAR_A)

        assert data["total"] == 3
        assert len(data["items"]) == 3


class TestPendingGuards:
    @pytest.mark.asyncio
    async def test_missing_scholar_raises(self):
        """未传 scholar_id → MissingScholarError（不让全表扫描）"""
        db = FakeDB()
        _seed(db, "s1")
        with pytest.raises(MissingScholarError):
            await list_pending_scans(db, scholar_id="")

    @pytest.mark.asyncio
    async def test_limit_clamped(self):
        """limit 越界夹取：<=0 → 1；>100 → 100（防一次拖全表）"""
        db = FakeDB()
        for i in range(5):
            _seed(db, f"s{i}", created_at=1700000000000 + i)

        assert len((await list_pending_scans(db, scholar_id=SCHOLAR_A, limit=0))["items"]) == 1
        assert len((await list_pending_scans(db, scholar_id=SCHOLAR_A, limit=999))["items"]) == 5
