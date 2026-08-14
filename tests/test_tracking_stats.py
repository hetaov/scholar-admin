"""POST /tracking/stats 学习进度统计接口单元测试

覆盖：
1. 统计服务纯函数（services/tracking_stats.py）
2. 接口参数校验与响应结构（通过 FastAPI TestClient + 假 DB）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# 保证以项目根目录为包根路径
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.routes_tracking import router as tracking_router  # noqa: E402
from services.tracking_stats import (  # noqa: E402
    compute_tracking_stats,
    format_duration,
    is_learned,
    merge_records,
    parse_time_spent,
    sentence_progress,
)


# ===========================================================================
# 测试数据
# ===========================================================================

SENTENCES = [
    {"sentence_id": "sent_1", "unit_id": "unit_a", "index": 1, "text": "Hello", "text_book_id": "tb_1"},
    {"sentence_id": "sent_2", "unit_id": "unit_a", "index": 2, "text": "World", "text_book_id": "tb_1"},
    {"sentence_id": "sent_3", "unit_id": "unit_b", "index": 1, "text": "Goodbye", "text_book_id": "tb_1"},
    {"sentence_id": "sent_4", "unit_id": "unit_b", "index": 2, "text": "Friend", "text_book_id": "tb_1"},
]

UNITS = [
    {"unit_id": "unit_a", "title": "Unit A", "text_book_id": "tb_1"},
    {"unit_id": "unit_b", "title": "Unit B", "text_book_id": "tb_1"},
]


class FakeDB:
    """模拟 CloudBaseNoSQLClient.query，仅支持 sentence / unit 集合按 text_book_id 过滤分页"""

    def __init__(self, sentences=None, units=None):
        self.sentences = list(sentences or [])
        self.units = list(units or [])

    async def query(
        self,
        collection: str,
        where: dict | None = None,
        order: list | None = None,
        offset: int = 0,
        limit: int = 100,
        select: dict | None = None,
    ) -> dict:
        pool = self.sentences if collection == "sentence" else self.units if collection == "unit" else []
        text_book_id = (where or {}).get("text_book_id")
        rows = [r for r in pool if not text_book_id or r.get("text_book_id") == text_book_id]
        if order:
            for spec in order:
                field = spec.get("field", "")
                reverse = spec.get("direction", "asc") == "desc"
                rows.sort(key=lambda r, f=field: str(r.get(f) or ""), reverse=reverse)
        page = rows[offset : offset + limit]
        return {"records": page, "total": len(page), "offset": offset, "limit": limit}


@pytest.fixture()
def app():
    _app = FastAPI()
    _app.include_router(tracking_router)
    return _app


@pytest.fixture()
def client(app, monkeypatch):
    monkeypatch.setattr(
        "services.routes_tracking.get_db",
        lambda: FakeDB(sentences=SENTENCES, units=UNITS),
    )
    return TestClient(app)


# ===========================================================================
# 工具函数测试
# ===========================================================================


class TestParseTimeSpent:
    def test_valid_values(self):
        assert parse_time_spent(120) == 120.0
        assert parse_time_spent(0) == 0.0
        assert parse_time_spent(1.5) == 1.5
        assert parse_time_spent("90") == 90.0
        assert parse_time_spent("12.5") == 12.5

    def test_invalid_values(self):
        assert parse_time_spent(None) == 0.0
        assert parse_time_spent("") == 0.0
        assert parse_time_spent("abc") == 0.0
        assert parse_time_spent(-5) == 0.0
        assert parse_time_spent(float("nan")) == 0.0
        assert parse_time_spent(float("inf")) == 0.0

    def test_custom_default(self):
        assert parse_time_spent(None, default=7.0) == 7.0


class TestIsLearned:
    def test_status_keywords_en(self):
        assert is_learned({"status": "learned"}) is True
        assert is_learned({"status": "mastered"}) is True
        assert is_learned({"status": "completed"}) is True

    def test_status_keywords_zh(self):
        assert is_learned({"status": "已学"}) is True
        assert is_learned({"status": "已掌握"}) is True
        assert is_learned({"status": "未学"}) is False

    def test_unknown_status_falls_back(self):
        # 未知状态没有 score/mastery → 视为未学
        assert is_learned({"status": "learning"}) is False
        assert is_learned({"status": "unknown_xyz"}) is False

    def test_score_threshold(self):
        assert is_learned({"score": 90}) is True
        assert is_learned({"score": 60}) is True
        assert is_learned({"score": 59.9}) is False

    def test_mastery_threshold(self):
        assert is_learned({"mastery": 0.8}) is True
        assert is_learned({"mastery": 0.5}) is False

    def test_empty_record(self):
        assert is_learned({}) is False


class TestSentenceProgress:
    def test_score_based(self):
        assert sentence_progress({"score": 90}) == pytest.approx(0.9)
        assert sentence_progress({"score": 0}) == 0.0
        assert sentence_progress({"score": 150}) == 1.0  # 封顶 1

    def test_mastery_based(self):
        assert sentence_progress({"mastery": 0.8}) == pytest.approx(0.8)
        assert sentence_progress({"mastery": 1.2}) == 1.0

    def test_status_based(self):
        assert sentence_progress({"status": "learned"}) == 1.0
        assert sentence_progress({"status": "未学"}) == 0.0
        assert sentence_progress({}) == 0.0

    def test_score_takes_priority(self):
        # score 存在时优先于 mastery
        assert sentence_progress({"score": 80, "mastery": 0.3}) == pytest.approx(0.8)


class TestMergeRecords:
    def test_sum_time_and_max_score(self):
        merged = merge_records(
            [
                {"sentence_id": "sent_1", "time_spent": 60, "score": 50},
                {"sentence_id": "sent_1", "time_spent": 30, "score": 90},
            ]
        )
        item = merged["sent_1"]
        assert item["time_spent"] == 90.0
        assert item["score"] == 90.0

    def test_learned_status_wins(self):
        merged = merge_records(
            [
                {"sentence_id": "sent_1", "status": "learning"},
                {"sentence_id": "sent_1", "status": "learned"},
            ]
        )
        assert merged["sent_1"]["status"] == "learned"

    def test_dirty_data_ignored(self):
        merged = merge_records(
            [
                {"time_spent": 60},  # 缺少 sentence_id
                "not-a-dict",        # 非 dict
                None,                # None
                {"sentence_id": "sent_2", "time_spent": 10},
            ]
        )
        assert set(merged.keys()) == {"sent_2"}

    def test_empty_list(self):
        assert merge_records([]) == {}


class TestFormatDuration:
    def test_seconds_only(self):
        assert format_duration(45) == "45秒"

    def test_minutes(self):
        assert format_duration(3725) == "1小时2分5秒"
        assert format_duration(150) == "2分30秒"

    def test_zero(self):
        assert format_duration(0) == "0秒"


# ===========================================================================
# 统计主函数测试
# ===========================================================================


class TestComputeTrackingStats:
    def test_empty_records_zero_progress(self):
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=[],
            sentences=SENTENCES,
            units=UNITS,
        )
        summary = stats["summary"]
        assert summary["total_time_spent"] == 0.0
        assert summary["learned_sentence_count"] == 0
        assert summary["total_sentence_count"] == 4
        assert summary["textbook_progress"] == 0.0
        assert summary["unit_count"] == 2
        # 未学习时各单元仍返回进度条目（进度为 0）
        assert len(stats["units"]) == 2
        assert all(u["progress"] == 0.0 for u in stats["units"])
        assert len(stats["sentences"]) == 4
        assert all(not s["learned"] for s in stats["sentences"])

    def test_full_stats(self):
        record_list = [
            {"sentence_id": "sent_1", "time_spent": 120, "status": "learned", "score": 90},
            {"sentence_id": "sent_2", "time_spent": 60, "score": 50},   # 不及格 → 未学
            {"sentence_id": "sent_3", "time_spent": 30, "mastery": 0.9},
            {"sentence_id": "sent_4", "time_spent": 10, "status": "learning"},
        ]
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=record_list,
            sentences=SENTENCES,
            units=UNITS,
        )
        summary = stats["summary"]
        assert summary["total_time_spent"] == 220.0
        assert summary["total_time_spent_display"] == "3分40秒"
        assert summary["learned_sentence_count"] == 2  # sent_1, sent_3
        assert summary["total_sentence_count"] == 4
        assert summary["textbook_progress"] == pytest.approx(0.5)
        assert summary["record_count"] == 4
        assert summary["matched_record_count"] == 4

        # unit 级
        units = {u["unit_id"]: u for u in stats["units"]}
        assert units["unit_a"]["total_sentence_count"] == 2
        assert units["unit_a"]["learned_sentence_count"] == 1
        assert units["unit_a"]["progress"] == pytest.approx(0.5)
        assert units["unit_a"]["time_spent"] == 180.0
        assert units["unit_b"]["learned_sentence_count"] == 1
        assert units["unit_b"]["progress"] == pytest.approx(0.5)

        # sentence 级
        sent_map = {s["sentence_id"]: s for s in stats["sentences"]}
        assert sent_map["sent_1"]["learned"] is True
        assert sent_map["sent_1"]["progress"] == pytest.approx(0.9)
        assert sent_map["sent_1"]["time_spent"] == 120.0
        assert sent_map["sent_2"]["learned"] is False
        assert sent_map["sent_2"]["progress"] == pytest.approx(0.5)
        assert sent_map["sent_3"]["learned"] is True
        assert sent_map["sent_4"]["learned"] is False

    def test_unknown_sentence_ignored(self):
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=[
                {"sentence_id": "sent_999", "time_spent": 999, "status": "learned"},
                {"sentence_id": "sent_1", "time_spent": 10, "status": "learned"},
            ],
            sentences=SENTENCES,
            units=UNITS,
        )
        summary = stats["summary"]
        assert summary["total_time_spent"] == 10.0  # sent_999 不计入
        assert summary["learned_sentence_count"] == 1
        assert summary["matched_record_count"] == 1

    def test_duplicate_records_merged(self):
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=[
                {"sentence_id": "sent_1", "time_spent": 60, "score": 40},
                {"sentence_id": "sent_1", "time_spent": 60, "score": 80, "status": "learned"},
            ],
            sentences=SENTENCES,
            units=UNITS,
        )
        summary = stats["summary"]
        sent_1 = next(s for s in stats["sentences"] if s["sentence_id"] == "sent_1")
        assert summary["total_time_spent"] == 120.0  # 时长累加
        assert sent_1["score"] == 80.0               # 取最大分
        assert sent_1["learned"] is True
        assert sent_1["progress"] == pytest.approx(0.8)

    def test_no_units_provided(self):
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=[],
            sentences=SENTENCES,
            units=[],
        )
        # unit 粒度由句子派生，units 参数仅用于补充标题
        assert stats["summary"]["unit_count"] == 2
        assert len(stats["units"]) == 2
        assert all(u["unit_title"] == "" for u in stats["units"])
        assert stats["summary"]["avg_unit_progress"] == 0.0

    def test_sentence_order_by_unit_and_index(self):
        stats = compute_tracking_stats(
            scholar_id="scholar_1",
            text_book_id="tb_1",
            record_list=[],
            sentences=[
                {"sentence_id": "sent_b2", "unit_id": "unit_b", "index": 2, "text": "b2"},
                {"sentence_id": "sent_a1", "unit_id": "unit_a", "index": 1, "text": "a1"},
            ],
            units=UNITS,
        )
        ids = [s["sentence_id"] for s in stats["sentences"]]
        assert ids == ["sent_a1", "sent_b2"]


# ===========================================================================
# 接口测试
# ===========================================================================


class TestTrackingStatsEndpoint:
    def test_success(self, client):
        resp = client.post(
            "/tracking/stats",
            json={
                "scholar_id": "scholar_1",
                "text_book_id": "tb_1",
                "record_list": [
                    {"sentence_id": "sent_1", "time_spent": 120, "status": "learned"},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["scholar_id"] == "scholar_1"
        assert data["text_book_id"] == "tb_1"
        assert data["summary"]["total_time_spent"] == 120.0
        assert data["summary"]["learned_sentence_count"] == 1
        assert data["summary"]["textbook_progress"] == pytest.approx(0.25)
        assert len(data["units"]) == 2
        assert len(data["sentences"]) == 4

    def test_missing_scholar_id(self, client):
        resp = client.post(
            "/tracking/stats",
            json={"text_book_id": "tb_1", "record_list": []},
        )
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]

    def test_missing_textbook_id(self, client):
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "record_list": []},
        )
        assert resp.status_code == 400
        assert "text_book_id" in resp.json()["detail"]

    def test_record_list_not_list(self, client):
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "text_book_id": "tb_1", "record_list": "oops"},
        )
        assert resp.status_code == 400
        assert "record_list" in resp.json()["detail"]

    def test_empty_record_list(self, client):
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "text_book_id": "tb_1", "record_list": []},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["summary"]["total_time_spent"] == 0.0
        assert data["summary"]["textbook_progress"] == 0.0

    def test_db_error_returns_500(self, app, monkeypatch):
        class BoomDB:
            async def query(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr("services.routes_tracking.get_db", lambda: BoomDB())
        resp = TestClient(app).post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "text_book_id": "tb_1", "record_list": []},
        )
        assert resp.status_code == 500
        assert "统计失败" in resp.json()["detail"]
