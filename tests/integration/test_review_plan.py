"""集成测试:f4 复习队列接口 — POST /tracking/review-plan

覆盖:
- 到期过滤:next_review_at ≤ 当日 23:59:59 且 status ≠ mastered(契约 §3.6)
- 乐观 pick_state:同句多能力取 progress 最高者(与掌握度聚合同源)
- 派生字段与 §3.1 sentences 接口同构(content/translation/lesson_id/skills/
  weakest_skill/status/review_count/next_review_at)
- 排序:next_review_at 升序 + 同到期日 mastery_score 升序(薄弱优先)
- 空态:无到期 → success:true, total:0
- 参数校验:缺 scholar_id / date 非法 → 400
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.routes_tracking import router as tracking_router


def _client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_tracking.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(tracking_router)
    return TestClient(app)


def _seed_content(fake_db):
    """tb_1: 1 章 2 课 4 句(sentence_v2 新集合),含跨课候选。"""
    fake_db.add("chapter", {"chapter_id": "c1", "textbook_id": "tb_1", "title": "Ch1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l1", "chapter_id": "c1", "title": "L1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l2", "chapter_id": "c1", "title": "L2", "order": 2})
    for sid, lid, order in [("s1", "l1", 1), ("s2", "l1", 2), ("s3", "l2", 3), ("s4", "l2", 4)]:
        fake_db.add("sentence_v2", {
            "sentence_id": sid, "lesson_id": lid, "chapter_id": "c1",
            "textbook_id": "tb_1", "text": f"Text {sid}", "translation": f"译{sid}",
            "order": order,
        })


def _ts(day: int, hour: int) -> int:
    """2026-08-day hour:00 UTC → 秒级时间戳(测试固定日期,避开"今天"漂移)。"""
    return int(datetime(2026, 8, day, hour, 0, 0, tzinfo=timezone.utc).timestamp())


class TestReviewPlan:
    """POST /tracking/review-plan 复习队列"""

    def test_due_queue_filter_and_fields(self, monkeypatch, fake_db):
        """到期过滤 + 派生字段:mastered 排除、无记录排除、内容/技能字段同构。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(16, 10),
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(16, 12),
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s3", "skill_code": "translation",
            "status": "mastered", "mastery_score": 95, "attempt_count": 3,
            "next_review_at": _ts(16, 11),
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/review-plan",
            json={"scholar_id": "scholar_1", "date": "2026-08-16"},  # 显式传固定日期,避免"今日"漂移
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["date"] == "2026-08-16"
        assert data["total"] == 2
        queue = data["review_queue"]
        # 排序:next_review_at 升序(s1 10:00 在 s2 12:00 前);s3 mastered 被排除,s4 无记录
        assert [q["sentence_id"] for q in queue] == ["s1", "s2"]
        s1 = queue[0]
        assert s1["content"] == "Text s1"
        assert s1["translation"] == "译s1"
        assert s1["lesson_id"] == "l1"
        assert s1["skills"] == {"translation": 2}  # status_to_int(learned)
        assert s1["weakest_skill"] == "translation"
        assert s1["status"] == 2
        assert s1["review_count"] == 2
        assert s1["next_review_at"].startswith("2026-08-16T10:00:00")

    def test_pick_state_highest_progress(self, monkeypatch, fake_db):
        """乐观 pick_state:同句多能力取 progress 最高者(listening 90 > translation 80)。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(16, 10),
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "listening",
            "status": "learning", "mastery_score": 90, "attempt_count": 5,
            "next_review_at": _ts(16, 11),
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/review-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["review_queue"]
        assert len(queue) == 1
        s1 = queue[0]
        assert s1["sentence_id"] == "s1"
        # picked = listening(0.9):review_count / next_review_at 取其值,status 取其状态
        assert s1["review_count"] == 5
        assert s1["next_review_at"].startswith("2026-08-16T11:00:00")
        assert s1["status"] == 1  # learning
        # skills 全量列出(同句多能力),weakest 取 status 数字最小者
        assert s1["skills"] == {"translation": 2, "listening": 1}
        assert s1["weakest_skill"] == "listening"

    def test_sort_by_next_then_mastery(self, monkeypatch, fake_db):
        """同到期时间按 mastery_score 升序(薄弱优先)。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(16, 10),
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
            "status": "learned", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(16, 10),
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/review-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["review_queue"]
        assert [q["sentence_id"] for q in queue] == ["s2", "s1"]  # 40 分薄弱在前

    def test_specific_date_and_future_excluded(self, monkeypatch, fake_db):
        """指定 date;未来到期(next_review_at > 当日 23:59:59)不进入队列。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learned", "mastery_score": 80, "attempt_count": 2,
            "next_review_at": _ts(15, 10),  # 8/15 到期
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
            "next_review_at": _ts(17, 8),  # 8/17 未到期
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/review-plan", json={"scholar_id": "scholar_1", "date": "2026-08-16"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["date"] == "2026-08-16"
        assert data["total"] == 1
        assert data["review_queue"][0]["sentence_id"] == "s1"

    def test_empty_queue_no_due(self, monkeypatch, fake_db):
        """无到期记录 → success:true, total:0,空队列(不报错)。"""
        _seed_content(fake_db)
        # 仅 mastered + 无 next_review_at 的记录
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "mastered", "mastery_score": 95, "attempt_count": 3,
            "next_review_at": _ts(16, 10),
        })
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 1,
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/review-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 0
        assert data["review_queue"] == []

    def test_missing_content_graceful(self, monkeypatch, fake_db):
        """候选句子在内容集合缺失 → content/translation 空串仍入队(容错不死)。"""
        # 不 seed 内容,只 seed 状态(句子内容表里没有 s1)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learning", "mastery_score": 50, "attempt_count": 1,
            "next_review_at": _ts(16, 10),
        })
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/review-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["review_queue"]
        assert len(queue) == 1
        assert queue[0]["sentence_id"] == "s1"
        assert queue[0]["content"] == ""
        assert queue[0]["translation"] == ""

    def test_missing_scholar_id_400(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/review-plan", json={})
        assert resp.status_code == 400

    def test_invalid_date_400(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/review-plan", json={"scholar_id": "scholar_1", "date": "2026/08/16"},
        )
        assert resp.status_code == 400
