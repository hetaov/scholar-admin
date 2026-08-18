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
    ACTIVITY_SKILL_WEIGHT,
    ACTIVITY_SKILL_WEIGHT_SEEDS,
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
    get_activity_skill_weights,
    get_default_skills,
    normalize_status,
    review_interval_seconds,
    seed_activity_skill_weights,
    seed_skills,
    skill_state_id,
    to_mastery_score,
    update_confidence,
    update_difficulty,
    update_stability,
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

    # --- S2.4 冷启动：证据稀疏打折（§5.6.2） ---

    def test_sparse_discount_blends_on_second_update(self):
        """第 2 次更新（attempt=2 < MIN_EVIDENCE=3）：增量只按 2/3 权重贡献。"""
        db = self._db()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", score=60, now=NOW)
        )
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", score=90,
                sparse_discount=True, now=NOW + 1000,
            )
        )
        # 60 + (90-60) * 2/3 = 80
        assert doc["mastery_score"] == pytest.approx(80.0)
        assert doc["attempt_count"] == 2

    def test_sparse_discount_full_weight_at_threshold(self):
        """第 3 次更新（attempt=3 = MIN_EVIDENCE）：满权重覆盖。"""
        db = self._db()
        for i, score in enumerate((60, 60, 90), start=1):
            doc = asyncio.run(
                upsert_skill_state(
                    db, scholar_id="s1", sentence_id="sent_1", score=score,
                    sparse_discount=True, now=NOW + i * 1000,
                )
            )
        assert doc["attempt_count"] == 3
        assert doc["mastery_score"] == pytest.approx(90.0)

    def test_sparse_discount_off_preserves_old_behavior(self):
        """默认关闭：旧行为不变，直接采用最新分数（不因稀疏打折）。"""
        db = self._db()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", score=60, now=NOW)
        )
        doc = asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", score=90, now=NOW + 1000)
        )
        assert doc["mastery_score"] == pytest.approx(90.0)

    def test_sparse_discount_first_update_full_blend_from_zero(self):
        """第 2 次更新且旧分缺失（None → 0）：折扣后 = 90 * 2/3 = 60。"""
        db = self._db()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", now=NOW)
        )
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", score=90,
                sparse_discount=True, now=NOW + 1000,
            )
        )
        assert doc["mastery_score"] == pytest.approx(60.0)


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


# ---------------------------------------------------------------------------
# S3.1 P1：confidence / stability / difficulty 更新策略（§5.6.2，契约 §4.11.4）
# ---------------------------------------------------------------------------


class TestUpdateConfidence:
    def test_first_attempt_evidence_discount(self):
        """attempt=1：EWMA=新值，×证据系数(1/3)。"""
        assert update_confidence(None, 0.9, 1) == pytest.approx(0.9 / 3, abs=1e-4)

    def test_second_attempt_blends(self):
        """attempt=2：alpha=1/2，ewma=(0.3+0.9)/2=0.6，×2/3=0.4。"""
        assert update_confidence(0.3, 0.9, 2) == pytest.approx(0.4, abs=1e-4)

    def test_full_evidence_after_threshold(self):
        """attempt=3（=MIN_EVIDENCE）：证据系数=1，EWMA=1/3 均值近似。"""
        val = update_confidence(0.4, 0.9, 3)
        # ewma = 0.4*(2/3) + 0.9*(1/3) = 0.5667
        assert val == pytest.approx(round(0.4 * 2 / 3 + 0.9 / 3, 4), abs=1e-4)

    def test_clamps_to_unit(self):
        assert update_confidence(0.9, 1.5, 10) <= 1.0
        assert update_confidence(None, -1.0, 10) >= 0.0


class TestUpdateStability:
    def test_first_outcome_records_direction_no_change(self):
        stab, last, streak = update_stability(0.0, "success", None, 0)
        assert stab == 0.0
        assert last == "success"
        assert streak == 1

    def test_two_same_directions_raises(self):
        s1, last1, streak1 = update_stability(0.0, "success", None, 0)
        s2, last2, streak2 = update_stability(s1, "success", last1, streak1)
        assert streak2 == 2
        assert s2 == pytest.approx(0.2)

    def test_opposite_direction_decays_and_resets(self):
        s1, last1, streak1 = update_stability(0.5, "success", "success", 2)
        assert s1 == pytest.approx(0.7)
        s2, last2, streak2 = update_stability(s1, "fail", last1, streak1)
        assert s2 == pytest.approx(0.6)
        assert streak2 == 1
        assert last2 == "fail"

    def test_stability_never_negative(self):
        s, _, _ = update_stability(0.05, "fail", "success", 3)
        assert s >= 0.0


class TestUpdateDifficulty:
    def test_defaults_to_min_when_missing(self):
        assert update_difficulty(None, None) == 1

    def test_takes_new_level(self):
        assert update_difficulty(2, 3) == 3

    def test_clamps(self):
        assert update_difficulty(1, 0) == 1
        assert update_difficulty(5, 99) == 5


class TestBuildDocP1:
    def test_doc_optional_p1_fields(self):
        doc = build_skill_state_doc(
            scholar_id="s1",
            sentence_id="sent_1",
            skill_code="translation",
            mastery_score=80,
            now=NOW,
            confidence=0.8,
            stability=0.2,
            difficulty=2,
        )
        assert doc["confidence"] == pytest.approx(0.8)
        assert doc["stability"] == pytest.approx(0.2)
        assert doc["difficulty"] == 2

    def test_doc_omits_fields_when_none(self):
        doc = build_skill_state_doc(
            scholar_id="s1", sentence_id="sent_1", skill_code="translation", now=NOW
        )
        assert "confidence" not in doc
        assert "stability" not in doc
        assert "difficulty" not in doc


class TestUpsertSkillStateP1:
    def test_upsert_with_confidence_and_outcome(self):
        db = FakeDB()
        doc = asyncio.run(
            upsert_skill_state(
                db,
                scholar_id="s1",
                sentence_id="sent_1",
                score=85,
                now=NOW,
                confidence=0.9,
                outcome="success",
                difficulty=2,
            )
        )
        assert doc["confidence"] == pytest.approx(0.9 / 3, abs=1e-4)  # attempt=1 证据打折
        assert doc["difficulty"] == 2
        assert doc["last_outcome"] == "success"
        assert doc["stable_streak"] == 1
        assert "stability" not in doc  # 首次不写稳定性

    def test_upsert_stability_raises_after_two_success(self):
        db = FakeDB()
        for i in range(1, 3):
            doc = asyncio.run(
                upsert_skill_state(
                    db,
                    scholar_id="s1",
                    sentence_id="sent_1",
                    score=80,
                    sparse_discount=True,
                    now=NOW + i * 1000,
                    confidence=0.8,
                    outcome="success",
                    difficulty=2,
                )
            )
        assert doc["stability"] == pytest.approx(0.2)
        assert doc["stable_streak"] == 2

    def test_upsert_weight_downgrades_delta(self):
        """门控降权 weight=0.5：增量打折（旧 60 → 新 90，增量 30×0.5=15 → 75）。"""
        db = FakeDB()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", score=60, now=NOW)
        )
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", score=90,
                weight=0.5, now=NOW + 1000,
            )
        )
        assert doc["mastery_score"] == pytest.approx(75.0)

    def test_upsert_weight_stacks_with_sparse_discount(self):
        """weight=0.5 × sparse(2/3)：60 + 30×0.5×2/3 = 70。"""
        db = FakeDB()
        asyncio.run(
            upsert_skill_state(db, scholar_id="s1", sentence_id="sent_1", score=60, now=NOW)
        )
        doc = asyncio.run(
            upsert_skill_state(
                db, scholar_id="s1", sentence_id="sent_1", score=90,
                weight=0.5, sparse_discount=True, now=NOW + 1000,
            )
        )
        assert doc["mastery_score"] == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# S3.1 P1：Activity → Skill 权重配置（契约 §4.11.5，草稿 §二十四）
# ---------------------------------------------------------------------------


class TestActivitySkillWeights:
    def test_seed_creates_all(self):
        db = FakeDB()
        stats = asyncio.run(seed_activity_skill_weights(db))
        assert stats["created"] == 4
        assert stats["skipped"] == 0
        assert len(db.all(ACTIVITY_SKILL_WEIGHT)) == 4

    def test_seed_idempotent(self):
        db = FakeDB()
        asyncio.run(seed_activity_skill_weights(db))
        stats2 = asyncio.run(seed_activity_skill_weights(db))
        assert stats2["created"] == 0
        assert stats2["skipped"] == 4

    def test_get_returns_seeded(self):
        db = FakeDB()
        asyncio.run(seed_activity_skill_weights(db))
        weights = asyncio.run(get_activity_skill_weights(db, "SHADOWING"))
        assert weights["Pronunciation"] == pytest.approx(0.45)

    def test_get_falls_back_to_seed_default(self):
        db = FakeDB()  # 未预置
        weights = asyncio.run(get_activity_skill_weights(db, "CONVERSATION"))
        assert weights == ACTIVITY_SKILL_WEIGHT_SEEDS["CONVERSATION"]

    def test_get_unknown_activity_returns_empty(self):
        db = FakeDB()
        assert asyncio.run(get_activity_skill_weights(db, "NOPE")) == {}
