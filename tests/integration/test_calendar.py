"""集成测试:f6 学习日历接口 — GET /tracking/{scholar_id}/calendar

覆盖(契约 §3.6):
- 按天聚合:同一天多条 attempt 计数,日期升序
- 连续打卡:从今天起往前连续 ≥1 条 attempt 的天数
- 今天无记录 → streak_days=0(不回溯)
- 中断即停(昨天无记录,只算今天)
- 窗口限定:days 控制统计范围(默认 30,上限 90)
- 空态:无记录 → heatmap:[], streak_days:0,不报错
- 参数校验:days 越界 → 400
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.routes_tracking import router as tracking_router


def _ts(days_ago: int, hour: int = 10) -> int:
    """今天往前 days_ago 天、hour 时(UTC)的时间戳,测试不依赖系统日期。"""
    day = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp())


def _attempt(sid: str, ts: int, scholar: str = "scholar_1") -> dict:
    return {"attempt_id": sid, "scholar_id": scholar, "created_at": ts}


def _client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_tracking.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(tracking_router)
    return TestClient(app)


class TestStudyCalendar:
    """GET /tracking/{scholar_id}/calendar 学习日历热力图"""

    def test_heatmap_aggregation(self, monkeypatch, fake_db):
        """按天聚合:同天多条 attempt 合并计数,日期升序。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(0, 9)))
        fake_db.add("study_attempt", _attempt("a2", _ts(0, 20)))
        fake_db.add("study_attempt", _attempt("a3", _ts(1)))
        fake_db.add("study_attempt", _attempt("a4", _ts(2)))
        fake_db.add("study_attempt", _attempt("a5", _ts(2)))
        fake_db.add("study_attempt", _attempt("a6", _ts(2)))
        fake_db.add("study_attempt", _attempt("a7", _ts(4)))
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["streak_days"] == 3  # 今天+昨天+前天连续
        dates = [h["date"] for h in data["heatmap"]]
        assert dates == sorted(dates)  # 升序
        by_date = {h["date"]: h["attempt_count"] for h in data["heatmap"]}
        # 2天前:3,1天前:1,今天:2,4天前:1
        assert by_date[_ts_day_str(2)] == 3
        assert by_date[_ts_day_str(1)] == 1
        assert by_date[_ts_day_str(0)] == 2
        assert by_date[_ts_day_str(4)] == 1

    def test_streak_consecutive(self, monkeypatch, fake_db):
        """连续打卡:今天/昨天/前天各 1 条 → 3 天。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(0)))
        fake_db.add("study_attempt", _attempt("a2", _ts(1)))
        fake_db.add("study_attempt", _attempt("a3", _ts(2)))
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        assert resp.json()["data"]["streak_days"] == 3

    def test_streak_zero_when_today_missing(self, monkeypatch, fake_db):
        """今天无记录 → streak=0,即使昨天有(不回溯)。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(1)))
        fake_db.add("study_attempt", _attempt("a2", _ts(2)))
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["streak_days"] == 0
        assert len(data["heatmap"]) == 2  # 昨天/前天的记录仍在热力图中

    def test_streak_stops_at_gap(self, monkeypatch, fake_db):
        """昨天无记录 → 只算今天(中断即停)。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(0)))
        fake_db.add("study_attempt", _attempt("a2", _ts(2)))
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        assert resp.json()["data"]["streak_days"] == 1

    def test_window_days_limit(self, monkeypatch, fake_db):
        """days=3 → 窗口仅最近 3 天,更早记录不统计。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(0)))
        fake_db.add("study_attempt", _attempt("a2", _ts(4)))  # 4天前,窗口外
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar?days=3")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["streak_days"] == 1
        assert len(data["heatmap"]) == 1
        assert data["heatmap"][0]["date"] == _ts_day_str(0)

    def test_default_days_30(self, monkeypatch, fake_db):
        """默认窗口 30 天:含 29 天前,不含 30 天前。"""
        fake_db.add("study_attempt", _attempt("a1", _ts(29)))
        fake_db.add("study_attempt", _attempt("a2", _ts(30)))  # 窗口外
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["heatmap"]) == 1
        assert data["heatmap"][0]["date"] == _ts_day_str(29)
        assert data["streak_days"] == 0

    def test_empty_no_attempts(self, monkeypatch, fake_db):
        """无任何记录 → success:true, heatmap:[], streak_days:0(不报错)。"""
        client = _client(monkeypatch, fake_db)
        resp = client.get("/tracking/scholar_1/calendar")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["streak_days"] == 0
        assert data["heatmap"] == []

    def test_days_invalid_400(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        for bad in ("0", "91", "-1"):
            resp = client.get(f"/tracking/scholar_1/calendar?days={bad}")
            assert resp.status_code == 400, f"days={bad} 应返回 400"


def _ts_day_str(days_ago: int) -> str:
    day = datetime.now(timezone.utc).date() - timedelta(days=days_ago)
    return day.strftime("%Y-%m-%d")
