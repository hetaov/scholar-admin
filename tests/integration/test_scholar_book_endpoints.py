"""集成测试:学者×教材关联接口(Phase 5)

覆盖:
- GET /scholar/{scholar_id}/books — 我的教材列表(含进度),空列表/有记录
- PUT /scholar/{scholar_id}/books/{textbook_id}/position — 更新断点:
  首次加入 / 断点更新 / 重复加入幂等(同一 学者×教材 仅一条)
- 断点更新后重新获取列表能取回 current_lesson_id(验收标准)
- 会话结算回写:end 后 scholar_book 的 last_studied_at 与 total_time_spent 更新
- 参数校验(缺参 / 至少一个字段)
"""

from __future__ import annotations

import pytest

from services.models_scholar_book import SCHOLAR_BOOK
from services.routes_state import router as state_router
from services.routes_tracking import router as tracking_router
from tests.fakes.seed_factory import seed_content


class TestPutPosition:
    def test_first_join_creates_record(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["scholar_id"] == "s1"
        assert data["textbook_id"] == "tb_1"
        assert data["current_chapter_id"] == "c1"
        assert data["current_lesson_id"] == "l1"
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_update_position_overwrites(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c2", "current_lesson_id": "l2"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["current_chapter_id"] == "c2"
        assert resp.json()["data"]["current_lesson_id"] == "l2"
        # 幂等: 同一 学者×教材 只有一条
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_repeated_join_idempotent(self, make_client, fake_db):
        client = make_client(tracking_router)
        for _ in range(3):
            resp = client.put(
                "/scholar/s1/books/tb_1/position",
                json={"current_lesson_id": "l1"},
            )
            assert resp.status_code == 200
        assert len(fake_db.all(SCHOLAR_BOOK)) == 1

    def test_only_last_studied_at_allowed(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.put(
            "/scholar/s1/books/tb_1/position",
            json={"last_studied_at": 1234567890},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["last_studied_at"] == 1234567890

    def test_empty_body_rejected(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.put("/scholar/s1/books/tb_1/position", json={})
        assert resp.status_code == 400
        assert "current_chapter_id" in resp.json()["detail"]


class TestGetBooks:
    def test_empty_list(self, make_client, fake_db):
        client = make_client(tracking_router)
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["scholar_id"] == "s1"
        assert data["books"] == []

    def test_list_with_progress(self, make_client, fake_db):
        seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s1", "s2"), include_text=False)
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s1", "skill_code": "translation",
             "status": "learned", "mastery_score": 80, "attempt_count": 2},
        )
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert len(books) == 1
        book = books[0]
        assert book["textbook_id"] == "tb_1"
        assert book["current_lesson_id"] == "l1"
        summary = book["summary"]
        assert summary["total_sentence_count"] == 2
        assert summary["learned_sentence_count"] == 1
        # f3 累计学习次数：s1 translation(learned) attempt=2
        assert summary["total_attempt_count"] == 2
        assert summary["textbook_progress"] == pytest.approx(0.4)  # (0.8 + 0.0) / 2
        # 综合掌握度：mastery_ratio 口径（s1 learned → 2/(3×2)，分母含未学 s2）
        assert summary["mastery"] == pytest.approx(0.3333)
        # 分能力掌握度：仅该能力有记录时输出
        assert summary["skills"]["translation"] == pytest.approx(0.3333)
        assert "conversation" not in summary["skills"]
        assert "listening" not in summary["skills"]
        assert "speaking" not in summary["skills"]
        assert "reading" not in summary["skills"]

    def test_summary_mastery_and_skills_weighted(self, make_client, fake_db):
        """多状态加权：综合掌握度含未学分母；分能力只统计该能力状态。"""
        seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s1", "s2"), include_text=False)
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s1", "skill_code": "translation",
             "status": "learned", "mastery_score": 80, "attempt_count": 2},
        )
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s1", "skill_code": "conversation",
             "status": "mastered", "mastery_score": 90, "attempt_count": 3},
        )
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s2", "skill_code": "translation",
             "status": "learning", "mastery_score": 40, "attempt_count": 1},
        )
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        summary = resp.json()["data"]["books"][0]["summary"]
        # 加权 = 1×learning + 2×learned + 3×mastered = 1+2+3 = 6；
        # 分母取 max(2 句, 3 记录) = 3 → 6/(3×3) ≈ 0.6667
        assert summary["mastery"] == pytest.approx(0.6667)
        # f3 乐观 pick_state：s1 取 progress 最高者 conversation(mastered)=3，不计 translation=2
        assert summary["total_attempt_count"] == 4  # s1=3 + s2=1
        # translation：learned + learning → 2+1 = 3，分母 max(2, 2) = 2 → 3/(3×2) = 0.5
        assert summary["skills"]["translation"] == pytest.approx(0.5)
        # conversation：mastered → 3，分母 max(2, 1) = 2 → 3/(3×2) = 0.5
        assert summary["skills"]["conversation"] == pytest.approx(0.5)
        assert "listening" not in summary["skills"]

    def test_breakpoint_retrieved_after_update(self, make_client, fake_db):
        """验收标准: 断点更新后重新获取列表能取回 current_lesson_id。"""
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_lesson_id": "l5"},
        )
        resp = client.get("/scholar/s1/books")
        book = resp.json()["data"]["books"][0]
        assert book["current_chapter_id"] == "c1"
        assert book["current_lesson_id"] == "l5"

    def test_attempt_count_zero_without_learning(self, make_client, fake_db):
        """f3 空态：加入教材但无学习记录 → total_attempt_count 为 0。"""
        seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s1", "s2"), include_text=False)
        client = make_client(tracking_router)
        client.put(
            "/scholar/s1/books/tb_1/position",
            json={"current_chapter_id": "c1", "current_lesson_id": "l1"},
        )
        resp = client.get("/scholar/s1/books")
        summary = resp.json()["data"]["books"][0]["summary"]
        assert summary["learned_sentence_count"] == 0
        assert summary["total_attempt_count"] == 0

    def test_multiple_books_isolated(self, make_client, fake_db):
        client = make_client(tracking_router)
        client.put("/scholar/s1/books/tb_1/position", json={"current_lesson_id": "l1"})
        client.put("/scholar/s1/books/tb_2/position", json={"current_lesson_id": "l2"})
        resp = client.get("/scholar/s1/books")
        books = resp.json()["data"]["books"]
        assert {b["textbook_id"] for b in books} == {"tb_1", "tb_2"}
        # 每本教材独立断点
        by_id = {b["textbook_id"]: b["current_lesson_id"] for b in books}
        assert by_id == {"tb_1": "l1", "tb_2": "l2"}


class TestBooksQueryCount:
    """性能回归：教材列表接口学习数据只拉一次、内容按书批量加载。

    防退化点：
    - skill_state / study_attempt 是学者级数据，必须全量仅查询一次后在内存内
      按教材句子集合过滤，不允许每本书重复拉取（原实现每本走一次
      独立聚合，各查 1 次 states + 1 次 attempts）；
    - 书名必须批量 $in（textbook_v2 一次取回，Phase 6 已移除旧表回退），
      不允许逐本查询。
    查询次数公式：1(books) + 3×N(内容) + 1(书名) + 1(states) + 1(attempts)。
    优化前本场景（2 本教材）需 1 + 2×(2 书名 + 5 聚合) = 15 次，优化后为 10 次。
    """

    def test_learning_data_fetched_once(self, make_client, fake_db):
        # 两本教材内容：tb_1 复用 _seed_content；tb_2 手动预置
        seed_content(fake_db, lesson_ids=("l1",), sentence_ids=("s1", "s2"), include_text=False)
        fake_db.add("chapter", {"chapter_id": "c2", "textbook_id": "tb_2", "title": "Ch2", "order": 1})
        fake_db.add("lesson", {"lesson_id": "l2", "chapter_id": "c2", "title": "L2", "order": 1})
        for i, sid in enumerate(("s3", "s4"), 1):
            fake_db.add(
                "sentence_v2",
                {"sentence_id": sid, "lesson_id": "l2", "chapter_id": "c2",
                 "textbook_id": "tb_2", "order": i},
            )
        # 仅 tb_1 有学习记录
        fake_db.add(
            "skill_state",
            {"scholar_id": "s1", "sentence_id": "s1", "skill_code": "translation",
             "status": "learned", "mastery_score": 80, "attempt_count": 2},
        )

        client = make_client(tracking_router)
        client.put("/scholar/s1/books/tb_1/position", json={"current_lesson_id": "l1"})
        client.put("/scholar/s1/books/tb_2/position", json={"current_lesson_id": "l2"})

        calls: list[str] = []
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls.append(kwargs.get("collection", args[0] if args else "?"))
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        resp = client.get("/scholar/s1/books")
        assert resp.status_code == 200
        books = resp.json()["data"]["books"]
        assert {b["textbook_id"] for b in books} == {"tb_1", "tb_2"}

        # 学者级学习数据只查一次
        assert calls.count("skill_state") == 1
        assert calls.count("study_attempt") == 1
        # 总查询 = 1(books) + 3×2(内容) + 1(书名: 批量 $in) + 1 + 1 = 10
        assert len(calls) == 10
        # 每本教材的 summary 独立：只有 tb_1 有 1 句 learned
        by_id = {b["textbook_id"]: b["summary"] for b in books}
        assert by_id["tb_1"]["total_sentence_count"] == 2
        assert by_id["tb_1"]["learned_sentence_count"] == 1
        assert by_id["tb_2"]["total_sentence_count"] == 2
        assert by_id["tb_2"]["learned_sentence_count"] == 0


class TestSessionSettlementWriteback:
    @pytest.mark.asyncio
    async def test_end_session_writeback_time_and_stamp(self, make_client, fake_db):
        """会话结算回写: scholar_book 的 last_studied_at 与 total_time_spent 被更新。"""
        client = make_client(state_router)
        session = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1", "textbook_id": "tb_1"},
        ).json()["data"]
        # 模拟已学习 30 秒,保证 duration_sec 非零,验证 total_time_spent 回写
        await fake_db.update(
            "study_session",
            where={"_id": session["session_id"]},
            data={"$set": {"started_at": 1}},
        )
        client.post("/tracking/session/end", json={"session_id": session["session_id"]})

        books = fake_db.all(SCHOLAR_BOOK)
        assert len(books) == 1
        book = books[0]
        assert book["scholar_id"] == "s1"
        assert book["textbook_id"] == "tb_1"
        assert book["last_studied_at"] is not None
        assert book["total_time_spent"] > 0

    @pytest.mark.asyncio
    async def test_end_session_accumulates_over_sessions(self, make_client, fake_db):
        client = make_client(state_router)
        for _ in range(2):
            session = client.post(
                "/tracking/session/start",
                json={"scholar_id": "s1", "textbook_id": "tb_1"},
            ).json()["data"]
            # 模拟已学习 30 秒,再结算,保证 duration_sec 非零
            await fake_db.update(
                "study_session",
                where={"_id": session["session_id"]},
                data={"$set": {"started_at": 1}},
            )
            client.post("/tracking/session/end", json={"session_id": session["session_id"]})

        books = fake_db.all(SCHOLAR_BOOK)
        assert len(books) == 1  # 幂等: 多次会话仍只有一条
        # 两次会话各自结算, total_time_spent 为两次 duration 之和(均 > 0)
        assert books[0]["total_time_spent"] > 0

    @pytest.mark.asyncio
    async def test_end_session_without_textbook_no_book(self, make_client, fake_db):
        client = make_client(state_router)
        session = client.post(
            "/tracking/session/start",
            json={"scholar_id": "s1"},
        ).json()["data"]
        client.post("/tracking/session/end", json={"session_id": session["session_id"]})
        assert fake_db.all(SCHOLAR_BOOK) == []
