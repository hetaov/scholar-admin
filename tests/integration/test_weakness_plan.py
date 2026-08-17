"""集成测试:f5 补漏清单接口 — POST /tracking/weakness-plan

覆盖(契约 §3.6):
- 候选过滤:低分(mastery_score < 60)+ 未学(status = not_started),
  mastered/高分/无记录 排除
- 乐观 pick_state:同句多能力取 progress 最高者(高分能力"救活"句子)
- 范围限定:textbook_id / lesson_id 仅返回范围内候选
- 排序:weakest_skill 升序 + 同最弱能力按 chapter_id / order 章节顺序
- 派生字段与 review-plan 同构(content/translation/lesson_id/chapter_id/
  skills/weakest_skill/mastery_score/status/review_count)
- 空态:无候选 → success:true, total:0
- 参数校验:缺 scholar_id / lesson_id 缺 textbook_id → 400
- 内容缺失容错:content/translation 空串仍入队
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
    """tb_1: c1: l1(s1,s2), l2(s3,s4);tb_2: c2: l3(s5,s6)。"""
    fake_db.add("chapter", {"chapter_id": "c1", "textbook_id": "tb_1", "title": "Ch1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l1", "chapter_id": "c1", "textbook_id": "tb_1", "title": "L1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l2", "chapter_id": "c1", "textbook_id": "tb_1", "title": "L2", "order": 2})
    fake_db.add("chapter", {"chapter_id": "c2", "textbook_id": "tb_2", "title": "Ch2", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l3", "chapter_id": "c2", "textbook_id": "tb_2", "title": "L3", "order": 1})
    for sid, lid, order, tb in [
        ("s1", "l1", 1, "tb_1"), ("s2", "l1", 2, "tb_1"),
        ("s3", "l2", 3, "tb_1"), ("s4", "l2", 4, "tb_1"),
        ("s5", "l3", 5, "tb_2"), ("s6", "l3", 6, "tb_2"),
    ]:
        fake_db.add("sentence_v2", {
            "sentence_id": sid, "lesson_id": lid, "chapter_id": "c1" if tb == "tb_1" else "c2",
            "textbook_id": tb, "text": f"Text {sid}", "translation": f"译{sid}",
            "order": order,
        })


def _state(sid: str, skill: str, status: str, score: int, attempts: int = 1) -> dict:
    return {
        "scholar_id": "scholar_1", "sentence_id": sid, "skill_code": skill,
        "status": status, "mastery_score": score, "attempt_count": attempts,
    }


class TestWeaknessPlan:
    """POST /tracking/weakness-plan 补漏清单"""

    def test_low_score_and_not_started_filter(self, monkeypatch, fake_db):
        """候选过滤:低分(<60)与未学(not_started)入选,高分/mastered/无记录排除。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learned", 80))    # ≥60 排除
        fake_db.add("skill_state", _state("s2", "translation", "learning", 40))   # 低分入选
        fake_db.add("skill_state", _state("s3", "translation", "not_started", 0))  # 未学入选
        fake_db.add("skill_state", _state("s4", "translation", "mastered", 95))   # mastered 排除
        fake_db.add("skill_state", _state("s5", "translation", "learning", 60))   # =60 不入选
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["total"] == 2
        queue = data["weakness_queue"]
        # 排序:weakest_skill 同 translation,按章节 order:s2(order2) 在 s3(order3) 前
        assert [q["sentence_id"] for q in queue] == ["s2", "s3"]
        s2 = queue[0]
        assert s2["content"] == "Text s2"
        assert s2["translation"] == "译s2"
        assert s2["lesson_id"] == "l1"
        assert s2["chapter_id"] == "c1"
        assert s2["skills"] == {"translation": 1}  # status_to_int(learning)
        assert s2["weakest_skill"] == "translation"
        assert s2["mastery_score"] == 40
        assert s2["status"] == 1
        assert s2["review_count"] == 1
        s3 = queue[1]
        assert s3["status"] == 0  # status_to_int(not_started)
        assert s3["mastery_score"] == 0

    def test_pick_state_highest_progress(self, monkeypatch, fake_db):
        """乐观 pick_state:同句多能力取 progress 最高(listening 90 救活 translation 40)。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learned", 80, attempts=2))
        fake_db.add("skill_state", _state("s1", "listening", "learning", 90, attempts=5))
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 0  # picked=90 ≥ 60,不入选
        assert data["weakness_queue"] == []

    def test_weakest_skill_pick_is_used_for_score(self, monkeypatch, fake_db):
        """多能力句子:最高分 < 60 → 入选,且输出 picked(progress 最高)的字段值。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learned", 50, attempts=3))
        fake_db.add("skill_state", _state("s1", "listening", "learning", 40, attempts=1))
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["weakness_queue"]
        assert len(queue) == 1
        s1 = queue[0]
        assert s1["sentence_id"] == "s1"
        assert s1["mastery_score"] == 50  # picked=translation(0.5)
        assert s1["status"] == 2
        assert s1["review_count"] == 3
        assert s1["skills"] == {"translation": 2, "listening": 1}
        assert s1["weakest_skill"] == "listening"

    def test_sort_by_weakest_then_chapter(self, monkeypatch, fake_db):
        """排序:weakest_skill 升序(listening < translation),同最弱按章节 order。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learning", 40))
        fake_db.add("skill_state", _state("s2", "listening", "learning", 30))
        fake_db.add("skill_state", _state("s3", "translation", "not_started", 0))
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["weakness_queue"]
        # listening(s2) < translation(s1, s3);s1 与 s3 同最弱,按 order:s1(2) < s3(3)
        assert [q["sentence_id"] for q in queue] == ["s2", "s1", "s3"]
        assert [q["weakest_skill"] for q in queue] == ["listening", "translation", "translation"]

    def test_scope_textbook(self, monkeypatch, fake_db):
        """限定教材:textbook_id=tb_1 → 仅返回 tb_1 内候选。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s2", "translation", "learning", 40))  # tb_1
        fake_db.add("skill_state", _state("s5", "translation", "learning", 30))  # tb_2
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/weakness-plan",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1"},
        )
        assert resp.status_code == 200
        queue = resp.json()["data"]["weakness_queue"]
        assert [q["sentence_id"] for q in queue] == ["s2"]

    def test_scope_lesson(self, monkeypatch, fake_db):
        """限定课时:textbook_id + lesson_id → 仅返回该课候选。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learning", 40))  # tb_1/l1
        fake_db.add("skill_state", _state("s3", "translation", "learning", 30))  # tb_1/l2
        fake_db.add("skill_state", _state("s5", "translation", "learning", 20))  # tb_2/l3
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/weakness-plan",
            json={"scholar_id": "scholar_1", "textbook_id": "tb_1", "lesson_id": "l1"},
        )
        assert resp.status_code == 200
        queue = resp.json()["data"]["weakness_queue"]
        assert [q["sentence_id"] for q in queue] == ["s1"]

    def test_empty_queue(self, monkeypatch, fake_db):
        """全部 mastered/高分/无记录 → success:true, total:0,空队列(不报错)。"""
        _seed_content(fake_db)
        fake_db.add("skill_state", _state("s1", "translation", "learned", 80))
        fake_db.add("skill_state", _state("s2", "translation", "mastered", 95))
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["total"] == 0
        assert data["weakness_queue"] == []

    def test_missing_content_graceful(self, monkeypatch, fake_db):
        """候选句子在内容集合缺失 → content/translation 空串仍入队(容错不死)。"""
        # 不 seed 内容,只 seed 状态
        fake_db.add("skill_state", _state("s1", "translation", "learning", 50))
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={"scholar_id": "scholar_1"})
        assert resp.status_code == 200
        queue = resp.json()["data"]["weakness_queue"]
        assert len(queue) == 1
        assert queue[0]["sentence_id"] == "s1"
        assert queue[0]["content"] == ""
        assert queue[0]["translation"] == ""

    def test_missing_scholar_id_400(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/tracking/weakness-plan", json={})
        assert resp.status_code == 400

    def test_lesson_without_textbook_400(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post(
            "/tracking/weakness-plan",
            json={"scholar_id": "scholar_1", "lesson_id": "l1"},
        )
        assert resp.status_code == 400
