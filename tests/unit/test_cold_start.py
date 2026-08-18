"""services/cold_start.py 单元测试（S2.4 冷启动落地，设计文档 §5.6）

覆盖:
- §5.6.1 先验默认值（mastery / confidence / stability / difficulty）
- §5.6.2 证据稀疏打折系数（< MIN_EVIDENCE 线性打折，≥ 阈值满权重）
- §5.6.3 标准引导序列
- §5.6.5 "cold_start" 标记 + has_skill_history 判定
"""

from __future__ import annotations

import asyncio

import pytest

from config import COLD_START_DIFFICULTY, COLD_START_MASTERY, MIN_EVIDENCE
from services.cold_start import (
    COLD_START_SEQUENCE,
    cold_start_flag,
    cold_start_prior,
    has_skill_history,
    is_sparse,
    sparse_evidence_weight,
)
from tests.fakes.fake_db import FakeDB


# ---------------------------------------------------------------------------
# §5.6.1 先验默认值
# ---------------------------------------------------------------------------


class TestColdStartPrior:
    def test_prior_defaults(self):
        prior = cold_start_prior()
        assert prior["mastery"] == COLD_START_MASTERY
        assert prior["confidence"] == 0.0
        assert prior["stability"] == 0.0
        assert prior["difficulty"] == COLD_START_DIFFICULTY

    def test_prior_accepts_difficulty_override(self):
        prior = cold_start_prior(difficulty=2)
        assert prior["difficulty"] == 2

    def test_prior_ignores_zero_difficulty(self):
        # difficulty=0 视为未提供（用先验默认），防最低档被误写成 0
        prior = cold_start_prior(difficulty=0)
        assert prior["difficulty"] == COLD_START_DIFFICULTY

    def test_prior_is_deterministic(self):
        assert cold_start_prior() == cold_start_prior()


# ---------------------------------------------------------------------------
# §5.6.2 证据稀疏打折系数
# ---------------------------------------------------------------------------


class TestSparseEvidenceWeight:
    def test_zero_attempts_has_no_weight(self):
        assert sparse_evidence_weight(0) == 0.0

    def test_first_attempt_one_third(self):
        assert sparse_evidence_weight(1) == pytest.approx(1 / MIN_EVIDENCE, abs=1e-3)

    def test_second_attempt_two_thirds(self):
        assert sparse_evidence_weight(2) == pytest.approx(2 / MIN_EVIDENCE, abs=1e-3)

    def test_at_threshold_full_weight(self):
        assert sparse_evidence_weight(MIN_EVIDENCE) == 1.0

    def test_beyond_threshold_full_weight(self):
        assert sparse_evidence_weight(MIN_EVIDENCE + 5) == 1.0


class TestIsSparse:
    def test_sparse_below_threshold(self):
        assert is_sparse(MIN_EVIDENCE - 1) is True
        assert is_sparse(1) is True

    def test_not_sparse_at_threshold(self):
        assert is_sparse(MIN_EVIDENCE) is False
        assert is_sparse(MIN_EVIDENCE + 1) is False


# ---------------------------------------------------------------------------
# §5.6.3 标准引导序列
# ---------------------------------------------------------------------------


class TestColdStartSequence:
    def test_sequence_defined(self):
        assert len(COLD_START_SEQUENCE) == 4
        assert COLD_START_SEQUENCE == ("content", "shadowing", "translation", "listening")


# ---------------------------------------------------------------------------
# §5.6.5 "cold_start" 标记 / 历史判定
# ---------------------------------------------------------------------------


class TestColdStartFlag:
    def test_no_history_means_cold(self):
        assert cold_start_flag(has_history=False) is True

    def test_with_history_not_cold(self):
        assert cold_start_flag(has_history=True) is False


class TestHasSkillHistory:
    def _db(self) -> FakeDB:
        return FakeDB()

    def _insert_state(self, db, scholar_id: str, skill_code: str = "translation") -> None:
        asyncio.run(
            db.insert(
                collection="skill_state",
                data={
                    "_id": f"{scholar_id}_sent_1_{skill_code}",
                    "scholar_id": scholar_id,
                    "sentence_id": "sent_1",
                    "skill_code": skill_code,
                    "status": "learning",
                },
            )
        )

    def test_no_records_is_cold(self):
        assert asyncio.run(has_skill_history(self._db(), scholar_id="s1")) is False

    def test_with_any_record_has_history(self):
        db = self._db()
        self._insert_state(db, "s1")
        assert asyncio.run(has_skill_history(db, scholar_id="s1")) is True

    def test_other_scholar_not_counted(self):
        db = self._db()
        self._insert_state(db, "s2")
        assert asyncio.run(has_skill_history(db, scholar_id="s1")) is False

    def test_skill_filter_narrowing(self):
        db = self._db()
        self._insert_state(db, "s1", skill_code="listening")
        assert asyncio.run(has_skill_history(db, scholar_id="s1", skill_code="translation")) is False
        assert asyncio.run(has_skill_history(db, scholar_id="s1", skill_code="listening")) is True

    def test_skill_filter_without_match(self):
        db = self._db()
        self._insert_state(db, "s1", skill_code="translation")
        assert asyncio.run(has_skill_history(db, scholar_id="s1", skill_code="speaking")) is False
