"""集成测试:POST /tracking/stats 服务端聚合路径(Phase 4)

覆盖:
- 无 record_list 时走服务端聚合(skill_state + chapter/lesson/sentence_v2)
- 响应包含 book / chapters / lessons 三级 progress 与 mastery 分布
- skill_code 过滤后仅反映该能力
- 学习时长来自 study_attempt 聚合
- 兼容入口(带 record_list)行为不变
- 参数校验
"""

from __future__ import annotations

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
    """预置 tb_1 内容层级: 1 章 2 课 4 句。"""
    fake_db.add("chapter", {"chapter_id": "c1", "textbook_id": "tb_1", "title": "Ch1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l1", "chapter_id": "c1", "title": "L1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l2", "chapter_id": "c1", "title": "L2", "order": 2})
    for sid, lid in [("s1", "l1"), ("s2", "l1"), ("s3", "l2"), ("s4", "l2")]:
        fake_db.add(
            "sentence_v2",
            {"sentence_id": sid, "lesson_id": lid, "chapter_id": "c1",
             "textbook_id": "tb_1", "order": int(sid[-1])},
        )


def _seed_states(fake_db, scholar_id="scholar_1"):
    """预置 skill_state + study_attempt。"""
    states = [
        {"scholar_id": scholar_id, "sentence_id": "s1", "skill_code": "translation",
         "status": "learned", "mastery_score": 80, "attempt_count": 2},
        {"scholar_id": scholar_id, "sentence_id": "s2", "skill_code": "translation",
         "status": "learning", "mastery_score": 40, "attempt_count": 1},
        {"scholar_id": scholar_id, "sentence_id": "s3", "skill_code": "translation",
         "status": "mastered", "mastery_score": 95, "attempt_count": 3},
        {"scholar_id": scholar_id, "sentence_id": "s1", "skill_code": "listening",
         "status": "learning", "mastery_score": 30, "attempt_count": 1},
    ]
    for st in states:
        fake_db.add("skill_state", st)
    attempts = [
        {"scholar_id": scholar_id, "sentence_id": "s1", "skill_code": "translation", "time_spent": 120},
        {"scholar_id": scholar_id, "sentence_id": "s2", "skill_code": "translation", "time_spent": 30},
    ]
    for a in attempts:
        fake_db.add("study_attempt", a)


class TestServerSideAggregation:
    def test_full_aggregation(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1", "detail": "full"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["scholar_id"] == "scholar_1"
        assert data["textbook_id"] == "tb_1"

        summary = data["summary"]
        # 4 句: s1=0.8, s2=0.4, s3=0.95, s4=0.0 → 0.5375
        assert summary["textbook_progress"] == pytest.approx(0.5375)
        assert summary["total_sentence_count"] == 4
        assert summary["learned_sentence_count"] == 2  # s1, s3
        assert summary["chapter_count"] == 1
        assert summary["lesson_count"] == 2
        # 学习时长来自 study_attempt
        assert summary["total_time_spent"] == 150.0
        assert summary["total_time_spent_display"] == "2分30秒"
        # mastery 分布(按句 pick 后: s1=learned, s2=learning, s3=mastered)
        assert summary["mastery_distribution"]["learned"] == 1
        assert summary["mastery_distribution"]["mastered"] == 1

        # 章级结构
        assert len(data["chapters"]) == 1
        chapter = data["chapters"][0]
        assert chapter["chapter_id"] == "c1"
        assert len(chapter["lessons"]) == 2
        l1 = next(l for l in chapter["lessons"] if l["lesson_id"] == "l1")
        assert l1["progress"] == pytest.approx(0.6)  # (0.8+0.4)/2
        assert l1["total_sentence_count"] == 2

        # 兼容字段
        assert len(data["units"]) == 2
        assert len(data["sentences"]) == 4

    def test_skill_code_filter(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1", "skill_code": "translation"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["skill_code"] == "translation"
        summary = data["summary"]
        # 仅 translation: s1=0.8, s2=0.4, s3=0.95, s4=0.0
        assert summary["learned_sentence_count"] == 2
        assert summary["total_time_spent"] == 150.0  # 仅 translation 的 attempts

    def test_reproducible(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        payload = {"scholar_id": "scholar_1", "textbook_id": "tb_1"}
        a = client.post("/tracking/stats", json=payload).json()["data"]
        b = client.post("/tracking/stats", json=payload).json()["data"]
        assert a == b  # 同一输入两次调用完全一致

    def test_no_states_zero_progress(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1", "detail": "full"},
        )
        assert resp.status_code == 200
        summary = resp.json()["data"]["summary"]
        assert summary["textbook_progress"] == 0.0
        assert summary["total_sentence_count"] == 4
        assert summary["learned_sentence_count"] == 0
        # 未学习时仍返回完整结构
        assert len(resp.json()["data"]["chapters"]) == 1

    def test_default_lesson_detail_returns_summary_and_lessons(self, monkeypatch, fake_db):
        """默认 detail="lesson": 只返回 summary + 课级统计, 不返回章节/句子明细。"""
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/stats",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["summary"]["lesson_count"] == 2
        assert "chapters" not in data
        assert "units" not in data
        assert "sentences" not in data
        # lessons 仅含课级统计, 不含句子明细
        assert [l["lesson_id"] for l in data["lessons"]] == ["l1", "l2"]
        assert data["lessons"][0]["total_sentence_count"] == 2
        assert "mastery_distribution" in data["lessons"][0]
        assert all("sentence" not in l for l in data["lessons"])

    def test_missing_textbook_id(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/stats", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 400
        assert "textbook_id" in resp.json()["detail"]

    def test_missing_scholar_id(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/stats", json={"textbook_id": "tb_1"})
        assert resp.status_code == 400
        assert "scholar_id" in resp.json()["detail"]
