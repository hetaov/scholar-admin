"""集成测试:查询接口拆分(Phase 6) — 接口 2 / 接口 3

覆盖:
- 接口 2 GET /scholar/{scholar_id}/textbooks/{textbook_id}/lessons
  教材详情(lesson 列表 + summary): summary.mastery / lessons[].progress
  (overall_percent / mastery / skills / status_distribution)
- 接口 3 GET /tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences
  章节句子明细: summary + sentences[](status/skills/weakest_skill/review_count/next_review_at)
- 与聚合路径口径一致(乐观聚合 / 4 级档位加权掌握度)
- 无学习记录 / lesson 不存在
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
    """tb_1: 1 章 2 课 4 句(sentence_v2 新集合)。"""
    fake_db.add("chapter", {"chapter_id": "c1", "textbook_id": "tb_1", "title": "Ch1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l1", "chapter_id": "c1", "title": "L1", "order": 1})
    fake_db.add("lesson", {"lesson_id": "l2", "chapter_id": "c1", "title": "L2", "order": 2})
    for sid, lid in [("s1", "l1"), ("s2", "l1"), ("s3", "l2"), ("s4", "l2")]:
        fake_db.add("sentence_v2", {
            "sentence_id": sid, "lesson_id": lid, "chapter_id": "c1",
            "textbook_id": "tb_1", "text": f"Text {sid}", "translation": f"译{sid}",
            "order": int(sid[-1]),
        })


def _seed_states(fake_db, scholar_id="scholar_1"):
    """与 POST /tracking/stats 聚合测试一致的状态 + next_review_at。"""
    states = [
        {"scholar_id": scholar_id, "sentence_id": "s1", "skill_code": "translation",
         "status": "learned", "mastery_score": 80, "attempt_count": 2,
         "next_review_at": 1784282400},
        {"scholar_id": scholar_id, "sentence_id": "s2", "skill_code": "translation",
         "status": "learning", "mastery_score": 40, "attempt_count": 1},
        {"scholar_id": scholar_id, "sentence_id": "s3", "skill_code": "translation",
         "status": "mastered", "mastery_score": 95, "attempt_count": 3},
        {"scholar_id": scholar_id, "sentence_id": "s1", "skill_code": "listening",
         "status": "learning", "mastery_score": 30, "attempt_count": 1},
    ]
    for st in states:
        fake_db.add("skill_state", st)


class TestGetTextbookLessons:
    """接口 2: GET /scholar/{scholar_id}/textbooks/{textbook_id}/lessons"""

    def test_full_aggregation(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        data = resp.json()["data"]

        # summary: 与聚合路径一致
        summary = data["summary"]
        assert summary["textbook_progress"] == pytest.approx(0.5375)  # (0.8+0.4+0.95+0)/4
        assert summary["total_sentence_count"] == 4
        assert summary["learned_sentence_count"] == 2  # s1, s3
        # mastery: 档位加权 (learning=1, learned=2, mastered=3) /3: s2=1, s1=2, s3=3, s4=0 → 6/12
        assert summary["mastery"] == pytest.approx(0.5)

        # lessons 列表(2 课, 按序)
        assert [l["lesson_id"] for l in data["lessons"]] == ["l1", "l2"]
        l1 = data["lessons"][0]
        assert l1["lesson_title"] == "L1"
        prog = l1["progress"]
        assert prog["overall_percent"] == 60  # (0.8+0.4)/2 = 0.6 → 60
        # l1: s1=learned, s2=learning → (2+1)/(3*2)=0.5
        assert prog["mastery"] == pytest.approx(0.5)
        assert prog["status_distribution"] == [0, 1, 1, 0, 0, 0]
        # skills: 各能力独立聚合
        assert prog["skills"]["translation"] == pytest.approx(0.5)  # s1=learned, s2=learning
        # listening: l1 共 2 句, 仅 s1 有记录(learning) → 1/(3*2)
        assert prog["skills"]["listening"] == pytest.approx(0.1667, abs=1e-4)

    def test_no_states(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["summary"]["textbook_progress"] == 0.0
        assert data["summary"]["mastery"] == 0.0
        assert data["summary"]["learned_sentence_count"] == 0
        l1 = data["lessons"][0]
        assert l1["progress"]["overall_percent"] == 0
        assert l1["progress"]["mastery"] == 0.0
        assert l1["progress"]["skills"] == {}
        assert l1["progress"]["status_distribution"] == [0, 0, 0, 0, 0, 0]


class TestGetLessonSentences:
    """接口 3: GET /tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences"""

    def test_sentence_detail_and_summary(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["lesson_id"] == "l1"
        assert data["lesson_title"] == "L1"

        # 仅返回该 lesson 的 2 句
        assert [s["sentence_id"] for s in data["sentences"]] == ["s1", "s2"]
        s1 = data["sentences"][0]
        assert s1["content"] == "Text s1"
        assert s1["translation"] == "译s1"
        # 乐观聚合 pick: translation(learned, 80) > listening(learning, 30) → learned=2
        assert s1["status"] == 2
        assert s1["skills"] == {"translation": 2, "listening": 1}
        assert s1["weakest_skill"] == "listening"
        assert s1["review_count"] == 2
        assert s1["next_review_at"] is not None  # int 时间戳 → ISO
        s2 = data["sentences"][1]
        assert s2["status"] == 1  # learning
        assert s2["skills"] == {"translation": 1}
        assert s2["weakest_skill"] == "translation"
        assert s2["review_count"] == 1
        assert s2["next_review_at"] is None

        # summary: lesson 粒度
        summary = data["summary"]
        assert summary["total_sentence_count"] == 2
        assert summary["learned_sentence_count"] == 1  # s1
        assert summary["mastery"] == pytest.approx(0.5)  # (learning=1 + learned=2)/(3*2)
        assert summary["skills"]["translation"] == pytest.approx(0.5)  # s1=learned, s2=learning
        # listening: 该课 2 句仅 s1 有记录(learning) → 1/(3*2)
        assert summary["skills"]["listening"] == pytest.approx(0.1667, abs=1e-4)

    def test_no_states(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["sentences"]) == 2
        s1 = data["sentences"][0]
        assert s1["status"] == 0
        assert s1["skills"] == {}
        assert s1["weakest_skill"] is None
        assert s1["review_count"] == 0
        assert s1["next_review_at"] is None
        assert data["summary"]["mastery"] == 0.0
        assert data["summary"]["learned_sentence_count"] == 0
        assert data["summary"]["skills"] == {}

    def test_lesson_not_found(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        client = _client(monkeypatch, fake_db)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/nope/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 404


class TestQueryCountOptimization:
    """性能回归：接口 2 查询次数固定（内容 3 次 + states 1 次 + attempts 1 次 = 5 次）。

    防退化点：
    - 内容层级必须批量 $in（chapters / lessons / sentences 各 1 次），
      不允许逐章 get_lessons / 逐课 get_sentences_by_lesson 的 N+1；
    - base + 4 能力聚合必须内存内过滤复用同一份数据，不允许重复触库。
    优化前本场景（1 章 2 课）需 5×(1+1+1+2+1) = 30 次，优化后恒为 5 次。
    """

    def test_query_count_is_constant(self, monkeypatch, fake_db):
        _seed_content(fake_db)
        _seed_states(fake_db)

        calls = {"n": 0}
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls["n"] += 1
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        client = _client(monkeypatch, fake_db)

        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        assert calls["n"] == 5  # chapters + lessons + sentences + states + attempts

    def test_query_count_scales_with_lessons_only_via_batch(self, monkeypatch, fake_db):
        """扩大教材规模（3 章 6 课 12 句）查询次数仍为 5，验证批量 $in 生效。"""
        for i in range(1, 4):
            fake_db.add("chapter", {
                "chapter_id": f"c{i}", "textbook_id": "tb_1", "title": f"Ch{i}", "order": i,
            })
        for i in range(1, 7):
            ch = f"c{(i - 1) // 2 + 1}"
            fake_db.add("lesson", {
                "lesson_id": f"l{i}", "chapter_id": ch, "title": f"L{i}", "order": i,
            })
        for i in range(1, 13):
            fake_db.add("sentence_v2", {
                "sentence_id": f"s{i}", "lesson_id": f"l{(i - 1) // 2 + 1}",
                "chapter_id": f"c{(i - 1) // 4 + 1}", "textbook_id": "tb_1",
                "text": f"Text s{i}", "translation": f"译s{i}", "order": i,
            })

        calls = {"n": 0}
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls["n"] += 1
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        client = _client(monkeypatch, fake_db)

        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["lessons"]) == 6
        assert calls["n"] == 5
