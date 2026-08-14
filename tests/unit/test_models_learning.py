"""services/models_learning.py 单元测试(Phase 2 能力模型)

覆盖:
- 状态归一化(中文 → 英文枚举)
- 复合键 / 分数换算
- 滚动调度公式(基础间隔、掌握度加权)
- skill_state 文档构建
- upsert 幂等性 / attempt_count 累加 / 复合键冲突
- skill 种子幂等预置
"""

from __future__ import annotations

import asyncio

import pytest

from services.models_learning import (
    DEFAULT_SKILL_CODE,
    SKILL,
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_LEARNING,
    STATUS_MASTERED,
    STATUS_NOT_STARTED,
    STATUS_REVIEW_DUE,
    build_skill_state_doc,
    compute_next_review_at,
    derive_progress,
    derive_status,
    get_default_skills,
    normalize_status,
    review_interval_seconds,
    seed_skills,
    skill_state_id,
    to_mastery_score,
    upsert_skill_state,
)
from tests.fakes.fake_db import FakeDB

NOW = 1_700_000_000
DAY = 86400


# ---------------------------------------------------------------------------
# 状态归一化
# ---------------------------------------------------------------------------


class TestNormalizeStatus:
    def test_chinese_status_map(self):
        assert normalize_status("已学") == STATUS_LEARNED
        assert normalize_status("已学会") == STATUS_LEARNED
        assert normalize_status("已学完") == STATUS_LEARNED
        assert normalize_status("已完成") == STATUS_LEARNED
        assert normalize_status("已掌握") == STATUS_MASTERED
        assert normalize_status("掌握") == STATUS_MASTERED
        assert normalize_status("学习中") == STATUS_LEARNING
        assert normalize_status("未学") == STATUS_NOT_STARTED

    def test_english_status_aliases(self):
        assert normalize_status("learned") == STATUS_LEARNED
        assert normalize_status("completed") == STATUS_LEARNED
        assert normalize_status("mastered") == STATUS_MASTERED
        assert normalize_status("review_due") == STATUS_REVIEW_DUE

    def test_unknown_falls_back_to_learning(self):
        assert normalize_status("whatever_unknown") == STATUS_LEARNING
        assert normalize_status(None) == STATUS_LEARNING
        assert normalize_status("") == STATUS_LEARNING


# ---------------------------------------------------------------------------
# 复合键 / 分数换算
# ---------------------------------------------------------------------------


class TestKeyAndScore:
    def test_skill_state_id(self):
        assert skill_state_id("s1", "sent_1", "translation") == "s1_sent_1_translation"

    def test_to_mastery_score_prefers_score(self):
        assert to_mastery_score(90, 0.5) == 90.0

    def test_to_mastery_score_from_mastery(self):
        assert to_mastery_score(None, 0.8) == pytest.approx(80.0)

    def test_to_mastery_score_clamps(self):
        assert to_mastery_score(150, None) == 100.0
        assert to_mastery_score(-10, None) == 0.0
        assert to_mastery_score(90, None) == 90.0

    def test_to_mastery_score_invalid_returns_none(self):
        assert to_mastery_score(None, None) is None
        assert to_mastery_score("abc", None) is None


# ---------------------------------------------------------------------------
# 滚动调度公式
# ---------------------------------------------------------------------------


class TestReviewInterval:
    def test_base_intervals_increase_with_attempt(self):
        days = [review_interval_seconds(n, None) // DAY for n in (1, 2, 3, 4, 5)]
        assert days == [1, 3, 7, 14, 30]

    def test_interval_caps_after_max(self):
        assert review_interval_seconds(99, None) // DAY == 30

    def test_high_mastery_lengthens_interval(self):
        assert review_interval_seconds(1, 90) == int(1 * 1.5 * DAY)

    def test_low_mastery_shortens_interval(self):
        assert review_interval_seconds(1, 50) == int(max(1.0, 1 / 2) * DAY)
        assert review_interval_seconds(3, 50) == int(max(1.0, 7 / 2) * DAY)

    def test_compute_next_review_at(self):
        assert compute_next_review_at(NOW, 1, None) == NOW + 1 * DAY


# ---------------------------------------------------------------------------
# 状态 / 进度推导
# ---------------------------------------------------------------------------


class TestDerive:
    def test_explicit_status_wins(self):
        assert derive_status("已掌握", 30, True) == STATUS_MASTERED

    def test_high_mastery_implies_mastered(self):
        assert derive_status(None, 90, True) == STATUS_MASTERED

    def test_low_mastery_implies_review_due(self):
        assert derive_status(None, 50, True) == STATUS_REVIEW_DUE

    def test_mid_mastery_implies_learning(self):
        assert derive_status(None, 70, True) == STATUS_LEARNING

    def test_no_mastery_falls_back_learning(self):
        assert derive_status(None, None, False) == STATUS_LEARNING

    def test_progress_from_mastery(self):
        assert derive_progress(STATUS_LEARNING, 80) == pytest.approx(0.8)
        assert derive_progress(STATUS_LEARNING, None) == 0.5
        assert derive_progress(STATUS_NOT_STARTED, None) == 0.0
        assert derive_progress(STATUS_MASTERED, None) == 1.0


# ---------------------------------------------------------------------------
# 文档构建
# ---------------------------------------------------------------------------


class TestBuildDoc:
    def test_doc_fields(self):
        doc = build_skill_state_doc(
            scholar_id="s1",
            sentence_id="sent_1",
            skill_code="translation",
            lesson_id="unit_a",
            status="已学",
            mastery_score=90,
            attempt_count=1,
            last_studied_at=NOW,
            now=NOW,
        )
        assert doc["_id"] == "s1_sent_1_translation"
        assert doc["state_id"] == "s1_sent_1_translation"
        assert doc["scholar_id"] == "s1"
        assert doc["sentence_id"] == "sent_1"
        assert doc["lesson_id"] == "unit_a"
        assert doc["skill_code"] == "translation"
        assert doc["status"] == STATUS_LEARNED
        assert doc["mastery_score"] == 90
        assert doc["attempt_count"] == 1
        assert doc["last_studied_at"] == NOW
        assert doc["next_review_at"] == NOW + int(1 * 1.5 * DAY)  # 高分间隔 ×1.5


# ---------------------------------------------------------------------------
# upsert 幂等性 / attempt_count 累加 / 复合键冲突
# ---------------------------------------------------------------------------


class TestUpsertSkillState:
    def _db(self) -> FakeDB:
        return FakeDB()

    def test_create_new(self):
        db = self._db()
        doc = asyncio.run(
            upsert_skill_state(
                db,
                scholar_id="s1",
                sentence_id="sent_1",
                skill_code="translation",
                status="已学",
                score=85,
                now=NOW,
            )
        )
        assert doc["attempt_count"] == 1
        assert doc["status"] == STATUS_LEARNED
        assert len(db.all(SKILL_STATE)) == 1

    def test_repeat_accumulates_attempt_count(self):
        db = self._db()
        for i in range(1, 4):
            doc = asyncio.run(
                upsert_skill_state(
                    db,
                    scholar_id="s1",
                    sentence_id="sent_1",
                    skill_code="translation",
                    status="已学",
                    score=80,
                    now=NOW + i * 1000,
                )
            )
        assert doc["attempt_count"] == 3
        assert len(db.all(SKILL_STATE)) == 1  # 同复合键只一条

    def test_last_studied_at_refreshes(self):
        db = self._db()
        asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", now=NOW
            )
        )
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", now=NOW + 5000
            )
        )
        assert doc["last_studied_at"] == NOW + 5000
        assert doc["attempt_count"] == 2

    def test_different_skill_code_separate_records(self):
        db = self._db()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", skill_code="translation", now=NOW)
        )
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", skill_code="listening", now=NOW)
        )
        assert len(db.all(SKILL_STATE)) == 2  # 复合键含 skill_code, 不冲突

    def test_different_sentence_separate_records(self):
        db = self._db()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", now=NOW)
        )
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_2", now=NOW)
        )
        assert len(db.all(SKILL_STATE)) == 2

    def test_mastery_without_status_derives_status(self):
        db = self._db()
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", mastery=0.9, now=NOW
            )
        )
        assert doc["status"] == STATUS_MASTERED
        assert doc["mastery_score"] == pytest.approx(90.0)

    def test_lesson_id_backfill_on_update(self):
        db = self._db()
        asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", lesson_id="unit_a", now=NOW
            )
        )
        doc = asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", now=NOW + 10)
        )
        assert doc["lesson_id"] == "unit_a"  # 更新时保留旧 lesson_id


# ---------------------------------------------------------------------------
# skill 种子幂等预置
# ---------------------------------------------------------------------------


class TestSeedSkills:
    def test_seed_creates_all(self):
        db = FakeDB()
        stats = asyncio.run(seed_skills(db))
        assert stats["created"] == 4
        assert stats["skipped"] == 0
        codes = {s["skill_code"] for s in db.all(SKILL)}
        assert codes == {"translation", "listening", "speaking", "reading"}

    def test_seed_idempotent(self):
        db = FakeDB()
        asyncio.run(seed_skills(db))
        stats2 = asyncio.run(seed_skills(db))
        assert stats2["created"] == 0
        assert stats2["skipped"] == 4
        assert len(db.all(SKILL)) == 4

    def test_default_skills_is_copy(self):
        seeds = get_default_skills()
        seeds[0]["name"] = "mutated"
        assert get_default_skills()[0]["name"] != "mutated"


class TestConstants:
    def test_default_skill_code(self):
        assert DEFAULT_SKILL_CODE == "translation"
