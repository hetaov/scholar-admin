"""单元测试:学习事件模型(Phase 3) — services.events

覆盖:
- 归一化/推断纯函数(infer_attempt_type / normalize_attempt_type / normalize_attempt_status)
- 文档构建(build_attempt_doc / build_session_doc)
- 写入辅助(record_attempt / start_session / end_session / count_session_attempts)
- 核心原则:study_attempt 只增不改;study_session 结算回填
"""

from __future__ import annotations

import pytest

from services.events import (
    SESSION_STATUS_ACTIVE,
    SESSION_STATUS_ENDED,
    STUDY_ATTEMPT,
    STUDY_SESSION,
    build_attempt_doc,
    build_session_doc,
    end_session,
    infer_attempt_type,
    normalize_attempt_status,
    normalize_attempt_type,
    record_attempt,
    start_session,
)
from tests.fakes.fake_db import FakeDB


class TestInfer:
    def test_known_skill_codes(self):
        assert infer_attempt_type("translation") == "translate"
        assert infer_attempt_type("listening") == "listen"
        assert infer_attempt_type("speaking") == "speak"
        assert infer_attempt_type("reading") == "read"

    def test_unknown_skill_falls_back_to_quiz(self):
        assert infer_attempt_type("grammar") == "quiz"
        assert infer_attempt_type(None) == "quiz"
        assert infer_attempt_type("") == "quiz"

    def test_normalize_attempt_type(self):
        assert normalize_attempt_type("READ") == "read"
        assert normalize_attempt_type("  quiz ") == "quiz"
        assert normalize_attempt_type(None) == "quiz"
        assert normalize_attempt_type("bogus") == "quiz"

    def test_normalize_attempt_status(self):
        assert normalize_attempt_status("CORRECT") == "correct"
        assert normalize_attempt_status("abandoned") == "abandoned"
        assert normalize_attempt_status(None) == "completed"
        assert normalize_attempt_status("bogus") == "completed"
        assert normalize_attempt_status("已学") == "completed"


class TestBuildAttemptDoc:
    def test_full_doc(self):
        doc = build_attempt_doc(
            scholar_id="s1",
            sentence_id="sent_1",
            skill_code="translation",
            attempt_type="translate",
            status="correct",
            score=90,
            mastery=0.9,
            time_spent=120,
            lesson_id="unit_1",
            session_id="ses_1",
            attempt_id="att_fixed",
            now=1000,
        )
        assert doc["_id"] == "att_fixed"
        assert doc["attempt_id"] == "att_fixed"
        assert doc["scholar_id"] == "s1"
        assert doc["sentence_id"] == "sent_1"
        assert doc["skill_code"] == "translation"
        assert doc["attempt_type"] == "translate"
        assert doc["status"] == "correct"
        assert doc["score"] == 90
        assert doc["mastery"] == 0.9
        assert doc["time_spent"] == 120
        assert doc["lesson_id"] == "unit_1"
        assert doc["session_id"] == "ses_1"
        assert doc["created_at"] == 1000

    def test_type_inferred_and_auto_ids(self):
        doc = build_attempt_doc(scholar_id="s1", sentence_id="sent_1", skill_code="listening")
        assert doc["attempt_type"] == "listen"
        assert doc["attempt_id"]
        assert doc["attempt_id"].startswith("att_")

    def test_missing_fields_defaults(self):
        doc = build_attempt_doc(scholar_id="s1", sentence_id="sent_1", skill_code="x")
        assert doc["attempt_type"] == "quiz"
        assert doc["status"] == "completed"
        assert doc["time_spent"] is None
        assert doc["session_id"] is None


class TestBuildSessionDoc:
    def test_start_state(self):
        doc = build_session_doc(scholar_id="s1", textbook_id="tb_1", session_id="ses_1", now=1000)
        assert doc["session_id"] == "ses_1"
        assert doc["scholar_id"] == "s1"
        assert doc["textbook_id"] == "tb_1"
        assert doc["status"] == SESSION_STATUS_ACTIVE
        assert doc["started_at"] == 1000
        assert doc["ended_at"] is None
        assert doc["duration_sec"] == 0
        assert doc["attempt_count"] == 0


class TestRecordAttempt:
    @pytest.mark.asyncio
    async def test_inserts_append_only(self):
        db = FakeDB()
        doc = await record_attempt(
            db, scholar_id="s1", sentence_id="sent_1", skill_code="translation", now=1000
        )
        assert doc["attempt_id"]
        # 只插入不修改
        docs = db.all(STUDY_ATTEMPT)
        assert len(docs) == 1
        assert docs[0]["sentence_id"] == "sent_1"
        assert docs[0]["attempt_type"] == "translate"


class TestSessionFlow:
    @pytest.mark.asyncio
    async def test_start_then_end(self):
        db = FakeDB()
        session = await start_session(db, scholar_id="s1", textbook_id="tb_1", now=1000)
        assert session["status"] == SESSION_STATUS_ACTIVE

        await record_attempt(
            db,
            scholar_id="s1",
            sentence_id="sent_1",
            skill_code="translation",
            session_id=session["session_id"],
            now=1100,
        )
        await record_attempt(
            db,
            scholar_id="s1",
            sentence_id="sent_2",
            skill_code="listening",
            session_id=session["session_id"],
            now=1200,
        )
        ended = await end_session(db, session_id=session["session_id"], ended_at=1300)
        assert ended["status"] == SESSION_STATUS_ENDED
        assert ended["ended_at"] == 1300
        assert ended["duration_sec"] == 1300 - 1000
        assert ended["attempt_count"] == 2

    @pytest.mark.asyncio
    async def test_end_missing_session_returns_none(self):
        db = FakeDB()
        assert await end_session(db, session_id="ses_missing") is None

    @pytest.mark.asyncio
    async def test_count_session_attempts(self):
        db = FakeDB()
        session = await start_session(db, scholar_id="s1")
        await record_attempt(
            db,
            scholar_id="s1",
            sentence_id="sent_1",
            skill_code="translation",
            session_id=session["session_id"],
        )
        await record_attempt(
            db,
            scholar_id="s1",
            sentence_id="sent_2",
            skill_code="translation",
            session_id=session["session_id"],
        )
        assert await db.count(collection=STUDY_ATTEMPT, where={"session_id": session["session_id"]}) == 2

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self):
        db = FakeDB()
        ses_a = await start_session(db, scholar_id="s1")
        ses_b = await start_session(db, scholar_id="s1")
        await record_attempt(db, scholar_id="s1", sentence_id="s1", skill_code="x", session_id=ses_a["session_id"])
        await record_attempt(db, scholar_id="s1", sentence_id="s2", skill_code="x", session_id=ses_a["session_id"])
        await record_attempt(db, scholar_id="s1", sentence_id="s3", skill_code="x", session_id=ses_b["session_id"])
        assert len(db.all(STUDY_SESSION)) == 2
        ended_a = await end_session(db, session_id=ses_a["session_id"], ended_at=2000)
        ended_b = await end_session(db, session_id=ses_b["session_id"], ended_at=2000)
        assert ended_a["attempt_count"] == 2
        assert ended_b["attempt_count"] == 1
