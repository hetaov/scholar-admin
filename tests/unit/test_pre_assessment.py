"""services/pre_assessment.py 单元测试（S3.2 前置评估，设计文档 §6.2 / §9-5）

覆盖:
- §6.2 Gate 建议：mastery < 0.3 → content（先看内容）；< 0.5 → training（先训练）；
  否则 pass（建议层不阻断）；无数据 → pass（先验不触发推荐）
- §6.2 难度档位：0.5~0.7 → 1；0.7~0.85 → 2；> 0.85 → 3；参考历史难度档位；均无 → 1
- 草稿 §二十四/§二十五 Activity 推荐：弱项驱动排序；无弱项 → 回退标准引导序列
- §5.6 冷启动回退：无历史 → 先验默认 + 引导序列，不报错不拒绝
- §5.6.3 证据稀疏保护：总尝试数 < MIN_EVIDENCE → Activity 推荐回退引导序列
"""
from __future__ import annotations

import asyncio

import pytest

from config import MIN_EVIDENCE
from services.cold_start import COLD_START_SEQUENCE
from services.models_conversation import DEFAULT_SKILL_CODE
from services.models_learning import (
    ACTIVITY_SKILL_WEIGHT_SEEDS,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
)
from services.pre_assessment import (
    GATE_CONTENT,
    GATE_PASS,
    GATE_TRAINING,
    aggregate_mastery,
    gate_suggestion,
    pre_assess,
    recommend_activities,
    suggest_difficulty,
)
from tests.fakes.fake_db import FakeDB


# ---------------------------------------------------------------------------
# §6.2 Gate 建议
# ---------------------------------------------------------------------------


class TestGateSuggestion:
    def test_no_data_means_pass(self):
        # 无可用数据 → 不触发推荐（§5.6.1 先验默认）
        assert gate_suggestion(None) == GATE_PASS

    def test_unknown_skill_suggests_content(self):
        assert gate_suggestion(0.2) == GATE_CONTENT

    def test_boundary_content_training(self):
        # mastery == 0.3 不属于"完全陌生"，进入训练建议档
        assert gate_suggestion(0.3) == GATE_TRAINING

    def test_weak_suggests_training(self):
        assert gate_suggestion(0.4) == GATE_TRAINING

    def test_boundary_training_pass(self):
        # mastery == 0.5 不再建议先训练（建议层：可进入自由会话）
        assert gate_suggestion(0.5) == GATE_PASS

    def test_strong_pass(self):
        assert gate_suggestion(0.85) == GATE_PASS


# ---------------------------------------------------------------------------
# §6.2 难度档位
# ---------------------------------------------------------------------------


class TestSuggestDifficulty:
    def test_low_mastery_lowest_band(self):
        assert suggest_difficulty(0.2) == 1
        assert suggest_difficulty(0.5) == 1

    def test_mid_low_mastery_band_1(self):
        assert suggest_difficulty(0.6) == 1

    def test_boundary_band_2(self):
        assert suggest_difficulty(0.7) == 2

    def test_mid_mastery_band_2(self):
        assert suggest_difficulty(0.75) == 2

    def test_boundary_band_3(self):
        assert suggest_difficulty(0.85) == 3

    def test_high_mastery_band_3(self):
        assert suggest_difficulty(0.9) == 3

    def test_history_difficulty_fallback(self):
        # 无 mastery → 参考历史难度档位（§6.2）
        assert suggest_difficulty(None, history_difficulty=3.4) == 3

    def test_history_difficulty_clamped(self):
        assert suggest_difficulty(None, history_difficulty=9.0) == DIFFICULTY_MAX
        assert suggest_difficulty(None, history_difficulty=0.4) == DIFFICULTY_MIN

    def test_no_data_default_1(self):
        assert suggest_difficulty(None) == 1


# ---------------------------------------------------------------------------
# 草稿 §二十四/§二十五 Activity 推荐（弱项驱动）
# ---------------------------------------------------------------------------


class TestRecommendActivities:
    def test_no_weakness_falls_back_to_guide(self):
        assert recommend_activities({}) == list(COLD_START_SEQUENCE)

    def test_weak_translation_prioritizes_translation(self):
        # translation 最弱（mastery 0.2）→ TRANSLATION（Recall/Usage/Grammar 权重最高）排最前
        result = recommend_activities({"translation": 0.2, "listening": 0.9})
        assert result[0] == "translation"
        assert result == ["translation", "conversation", "dictation", "shadowing"]

    def test_weak_listening_prioritizes_dictation(self):
        # listening 最弱 → DICTATION（Listening 权重 0.60 最高）排最前
        result = recommend_activities({"translation": 0.9, "listening": 0.2})
        assert result[0] == "dictation"
        assert result == ["dictation", "conversation", "shadowing", "translation"]

    def test_weak_speaking_prioritizes_shadowing(self):
        # speaking 最弱 → SHADOWING（Pronunciation/Fluency/Speaking 权重最高）排最前
        result = recommend_activities({"speaking": 0.2})
        assert result[0] == "shadowing"
        assert result == ["shadowing", "conversation", "dictation", "translation"]

    def test_unknown_skill_does_not_crash(self):
        # reading 无对应受控技能（匹配 0）→ 不报错，仍返回 4 个活动（确定性排序）
        result = recommend_activities({"reading": 0.1})
        assert len(result) == 4
        assert set(result) == {a.lower() for a in ACTIVITY_SKILL_WEIGHT_SEEDS}

    def test_all_strong_skills_stable_order(self):
        # 全部高 mastery → 弱项得分趋近 0，仍返回 4 个活动（确定性排序，不报错）
        result = recommend_activities(
            {"translation": 0.95, "listening": 0.95, "speaking": 0.95}
        )
        assert len(result) == 4
        assert set(result) == {a.lower() for a in ACTIVITY_SKILL_WEIGHT_SEEDS}


# ---------------------------------------------------------------------------
# skill_state 聚合
# ---------------------------------------------------------------------------


class TestAggregateMastery:
    def test_average_per_skill(self):
        assert aggregate_mastery(
            [
                {"skill_code": "translation", "mastery_score": 60},
                {"skill_code": "translation", "mastery_score": 80},
                {"skill_code": "listening", "mastery_score": 100},
            ]
        ) == {"translation": 0.7, "listening": 1.0}

    def test_records_without_mastery_skipped(self):
        # §5.6.5 存量/仅 status 记录：无有效 mastery → 不参与聚合
        assert aggregate_mastery([{"skill_code": "translation", "status": "learning"}]) == {}

    def test_score_and_mastery_normalized(self):
        assert aggregate_mastery(
            [
                {"skill_code": "translation", "score": 60},
                {"skill_code": "translation", "mastery": 0.9},
            ]
        ) == {"translation": 0.75}

    def test_default_skill_when_missing(self):
        assert aggregate_mastery([{"mastery_score": 50}]) == {DEFAULT_SKILL_CODE: 0.5}


# ---------------------------------------------------------------------------
# 主入口 pre_assess（冷启动回退 + 有历史计算）
# ---------------------------------------------------------------------------


def _add_state(
    db: FakeDB,
    scholar_id: str,
    *,
    sentence_id: str = "sent_1",
    skill_code: str = "translation",
    mastery_score: float | None = None,
    difficulty: int | None = None,
    attempt_count: int | None = None,
    status: str = "learning",
) -> None:
    doc: dict = {
        "scholar_id": scholar_id,
        "sentence_id": sentence_id,
        "skill_code": skill_code,
        "status": status,
    }
    if mastery_score is not None:
        doc["mastery_score"] = mastery_score
    if difficulty is not None:
        doc["difficulty"] = difficulty
    if attempt_count is not None:
        doc["attempt_count"] = attempt_count
    db.add("skill_state", doc)


def _run(db: FakeDB, **kw) -> dict:
    return asyncio.run(pre_assess(db, **kw))


class TestPreAssessColdStart:
    def test_no_history_falls_back(self):
        # §5.6/§6.2 冷启动约束：无历史 → 先验默认，不报错不拒绝
        result = _run(FakeDB(), scholar_id="s1")
        assert result["has_history"] is False
        assert result["gate_suggestion"] == GATE_PASS
        assert result["difficulty"] == 1
        assert result["activity_recommendation"] == list(COLD_START_SEQUENCE)
        assert result["mastery"] is None
        assert result["evidence_sparse"] is True

    def test_history_without_mastery(self):
        # 有记录但无有效 mastery（存量/仅 status）：不触发推荐，难度回退默认 1
        db = FakeDB()
        _add_state(db, "s1")
        result = _run(db, scholar_id="s1")
        assert result["has_history"] is True
        assert result["gate_suggestion"] == GATE_PASS
        assert result["difficulty"] == 1
        assert result["activity_recommendation"] == list(COLD_START_SEQUENCE)
        assert result["mastery"] is None


class TestPreAssessWithHistory:
    def _db(self, **state) -> FakeDB:
        db = FakeDB()
        _add_state(db, "s1", **state)
        return db

    def test_sparse_evidence_falls_back_to_guide(self):
        # §5.6.3：attempt=1 < MIN_EVIDENCE → 弱项不可信，Activity 回退引导序列
        db = self._db(mastery_score=20, attempt_count=1)
        result = _run(db, scholar_id="s1")
        assert result["mastery"] == pytest.approx(0.2)
        assert result["gate_suggestion"] == GATE_CONTENT
        assert result["evidence_sparse"] is True
        assert result["activity_recommendation"] == list(COLD_START_SEQUENCE)

    def test_sufficient_evidence_weakness_driven(self):
        # attempt=3 ≥ MIN_EVIDENCE → 弱项驱动推荐（translation 弱 → translation 排最前）
        db = self._db(mastery_score=20, attempt_count=MIN_EVIDENCE)
        result = _run(db, scholar_id="s1")
        assert result["evidence_sparse"] is False
        assert result["activity_recommendation"][0] == "translation"

    def test_training_gate_at_low_mastery(self):
        db = self._db(mastery_score=40, attempt_count=MIN_EVIDENCE)
        result = _run(db, scholar_id="s1")
        assert result["gate_suggestion"] == GATE_TRAINING
        assert result["difficulty"] == 1  # 0.4 < 0.7 → 最低档

    def test_pass_gate_and_band_2(self):
        db = self._db(mastery_score=75, attempt_count=MIN_EVIDENCE)
        result = _run(db, scholar_id="s1")
        assert result["gate_suggestion"] == GATE_PASS
        assert result["difficulty"] == 2  # 0.75 ∈ [0.7, 0.85)

    def test_high_mastery_band_3(self):
        db = self._db(mastery_score=90, attempt_count=MIN_EVIDENCE)
        result = _run(db, scholar_id="s1")
        assert result["gate_suggestion"] == GATE_PASS
        assert result["difficulty"] == 3  # 0.9 ≥ 0.85

    def test_multi_skill_average(self):
        db = FakeDB()
        _add_state(db, "s1", mastery_score=20, attempt_count=3)
        _add_state(db, "s1", skill_code="listening", mastery_score=80, attempt_count=3)
        result = _run(db, scholar_id="s1")
        assert result["mastery"] == pytest.approx(0.5)
        assert result["gate_suggestion"] == GATE_PASS  # 0.5 → 可进入
        assert result["difficulty"] == 1  # 0.5 < 0.7

    def test_history_difficulty_fallback_without_mastery(self):
        # 无 mastery 但有历史难度档位 → 参考历史（§6.2）
        db = self._db(difficulty=4)
        result = _run(db, scholar_id="s1")
        assert result["mastery"] is None
        assert result["difficulty"] == 4

    def test_sentence_id_filter(self):
        # 绑定目标句时仅聚合该句的 skill_state（§6.2 数据来源）
        db = FakeDB()
        _add_state(db, "s1", sentence_id="sent_1", mastery_score=90, attempt_count=3)
        _add_state(db, "s1", sentence_id="sent_2", mastery_score=20, attempt_count=3)
        result = _run(db, scholar_id="s1", sentence_id="sent_2")
        assert result["mastery"] == pytest.approx(0.2)
        assert result["difficulty"] == 1
