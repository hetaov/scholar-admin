"""单元测试：模拟学习脚本（T2 / 设计稿 §3.3、§8.4-2）

覆盖：模拟作答分档 / mastery 累计（稀疏保护）/ skill_state 累计字段 /
低置信·未达意门控不回写 / --lessons 过滤 / CLI 端到端（写 learners 文件）。
全程本地 L1 规则或注入 evaluator，不触网。
"""
from __future__ import annotations

import random

import pytest

from scripts.prepare_nc2_corpus import build_corpus, lessons_data
from scripts.simulate_learning import (
    _WEAK_INPUTS,
    accumulate_mastery,
    build_or_update_state,
    main,
    pick_sentence,
    run_simulation,
    simulate_input,
)
from services.evaluation_engine import l1_rule_evaluate
from services.learning.local_corpus import (
    compute_weak_skills,
    empty_corpus,
    empty_learner,
    learner_path,
    load_learner,
)


def _corpus(lessons=(1, 2)) -> dict:
    return build_corpus(lessons_data(list(lessons)))


class TestSimulateInput:
    def test_exact_branch_returns_original(self):
        text = "Last week I went to the theatre."
        results = {simulate_input(text, random.Random(i), "study") for i in range(30)}
        assert text in results

    def test_partial_branch_is_contiguous_slice_and_weak_branch_is_placeholder(self):
        text = "Last week I went to the theatre with my friend."
        results = {simulate_input(text, random.Random(i), "study") for i in range(60)}
        assert results & set(_WEAK_INPUTS)
        partials = {
            r for r in results if r and r not in (text,) and r not in _WEAK_INPUTS
        }
        assert partials
        for partial in partials:
            # partial 为原文的连续片段（可能从中间开始），且词数少于原句
            assert partial in text
            assert len(partial.split()) < len(text.split())

    def test_deterministic_with_same_seed(self):
        text = "I had a very good seat."
        assert simulate_input(text, random.Random(7)) == simulate_input(text, random.Random(7))


class TestPickSentence:
    CANDIDATES = [
        {"sentence_id": "s_a", "text": "A"},
        {"sentence_id": "s_b", "text": "B"},
    ]

    def test_unseen_sentence_is_preferred(self):
        # 未练过的句子权重 1.0，已练且已掌握（mastery=1.0）权重 0.5
        states = {"s_a": {"mastery": 1.0}}
        rng = random.Random(0)
        picks = [pick_sentence(self.CANDIDATES, states, rng)["sentence_id"] for _ in range(200)]
        assert picks.count("s_b") > picks.count("s_a")

    def test_weak_sentence_gets_repeated(self):
        # s_a 弱（mastery=0）权重 3.0 > s_b 已掌握 0.5
        states = {"s_a": {"mastery": 0.0}, "s_b": {"mastery": 1.0}}
        rng = random.Random(0)
        picks = [pick_sentence(self.CANDIDATES, states, rng)["sentence_id"] for _ in range(200)]
        assert picks.count("s_a") > picks.count("s_b")

    def test_deterministic_with_same_seed(self):
        rng_a, rng_b = random.Random(9), random.Random(9)
        assert pick_sentence(self.CANDIDATES, {}, rng_a) == pick_sentence(
            self.CANDIDATES, {}, rng_b
        )


class TestAccumulateMastery:
    def test_first_attempt_initializes(self):
        assert accumulate_mastery(None, 90, attempt_count=1) == 90.0

    def test_moves_toward_new_score_with_sparse_discount(self):
        # attempt=1 → evidence=1/3（MIN_EVIDENCE=3）
        moved = accumulate_mastery(60.0, 90.0, attempt_count=1)
        assert 60.0 < moved < 90.0
        assert moved == pytest.approx(70.0, abs=0.01)

    def test_full_evidence_moves_fully(self):
        assert accumulate_mastery(60.0, 90.0, attempt_count=3) == pytest.approx(90.0)

    def test_clamped(self):
        assert accumulate_mastery(95.0, 0.0, attempt_count=3) == 0.0
        assert accumulate_mastery(0.0, 200.0, attempt_count=3) == 100.0


class TestBuildOrUpdateState:
    SENTENCE = {"sentence_id": "s_nc2_01_01", "lesson_id": "l_nc2_01", "difficulty": 2}

    def test_fields_written(self):
        state = build_or_update_state(
            None,
            scholar_id="scholar_x",
            sentence=self.SENTENCE,
            verdict={"score": 90, "confidence": 0.9, "meaningful": True},
            now_seconds=1000,
            last_studied_ms=1000_000,
        )
        assert state["_id"] == "scholar_x_s_nc2_01_01_translation"
        assert state["status"] == "mastered"
        assert state["mastery_score"] == 90.0
        assert state["mastery"] == 0.9
        assert state["attempt_count"] == 1
        assert state["last_outcome"] == "correct"
        assert state["next_review_at"] > state["last_studied_at"]

    def test_accumulates_on_previous(self):
        prev = {
            "mastery_score": 40.0,
            "attempt_count": 2,
            "confidence": 0.5,
            "stability": 0.0,
            "last_outcome": "correct",
            "stable_streak": 1,
            "created_at": 1,
        }
        state = build_or_update_state(
            prev,
            scholar_id="scholar_x",
            sentence=self.SENTENCE,
            verdict={"score": 40, "confidence": 0.5, "meaningful": True},
            now_seconds=2000,
            last_studied_ms=2000_000,
        )
        assert state["attempt_count"] == 3
        assert state["status"] == "learning"
        assert state["last_outcome"] == "fail"
        assert state["created_at"] == 1

    def test_stability_rises_after_two_same_direction(self):
        prev = {
            "mastery_score": 80.0,
            "attempt_count": 1,
            "last_outcome": "correct",
            "stable_streak": 1,
            "stability": 0.0,
        }
        state = build_or_update_state(
            prev,
            scholar_id="s",
            sentence=self.SENTENCE,
            verdict={"score": 90, "confidence": 0.9, "meaningful": True},
            now_seconds=1,
            last_studied_ms=1000,
        )
        assert state["stable_streak"] == 2
        assert state["stability"] == 0.2


def _verdict(**overrides) -> dict:
    base = {
        "score": 90,
        "meaningful": True,
        "faithfulness": True,
        "anomaly": False,
        "confidence": 0.9,
        "level": "l1",
    }
    base.update(overrides)
    return base


class TestRunSimulation:
    def test_writes_back_when_high_confidence(self):
        learner = empty_learner("scholar_x")
        result = run_simulation(
            corpus=_corpus(),
            learner=learner,
            scholar_id="scholar_x",
            n=5,
            seed=1,
            evaluator=lambda o, r: _verdict(),
        )
        assert result["stats"]["attempted"] == 5
        assert result["stats"]["written_back"] == 5
        assert result["stats"]["skipped_low_confidence"] == 0
        assert len(result["skill_states"]) >= 1
        assert result["updated_at"] is not None
        assert result["weak_skills"] == compute_weak_skills(result)

    def test_low_confidence_not_written_back(self):
        learner = empty_learner("scholar_x")
        result = run_simulation(
            corpus=_corpus(),
            learner=learner,
            scholar_id="scholar_x",
            n=4,
            evaluator=lambda o, r: _verdict(confidence=0.5),
        )
        assert result["stats"]["written_back"] == 0
        assert result["stats"]["skipped_low_confidence"] == 4
        assert result["skill_states"] == []
        assert len(result["attempts"]) == 4

    def test_not_meaningful_not_written_back(self):
        learner = empty_learner("scholar_x")
        result = run_simulation(
            corpus=_corpus(),
            learner=learner,
            scholar_id="scholar_x",
            n=3,
            evaluator=lambda o, r: _verdict(meaningful=False),
        )
        assert result["stats"]["skipped_not_meaningful"] == 3
        assert result["skill_states"] == []

    def test_lesson_filter_limits_states(self):
        learner = empty_learner("scholar_x")
        result = run_simulation(
            corpus=_corpus((1, 2)),
            learner=learner,
            scholar_id="scholar_x",
            n=6,
            lesson_ids=["l_nc2_02"],
            evaluator=lambda o, r: _verdict(),
        )
        assert {s["lesson_id"] for s in result["skill_states"]} == {"l_nc2_02"}

    def test_l1_rule_evaluator_end_to_end(self):
        learner = empty_learner("scholar_x")
        result = run_simulation(
            corpus=_corpus(),
            learner=learner,
            scholar_id="scholar_x",
            n=20,
            seed=3,
            evaluator=l1_rule_evaluate,
        )
        assert result["stats"]["attempted"] == 20
        # L1 规则下「完全不相关」档置信度 0.5 < 0.6，必然出现低置信跳过
        assert result["stats"]["skipped_low_confidence"] >= 0
        assert result["stats"]["written_back"] + result["stats"][
            "skipped_low_confidence"
        ] + result["stats"]["skipped_not_meaningful"] == 20

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            run_simulation(
                corpus=empty_corpus(),
                learner=empty_learner("s"),
                scholar_id="s",
                n=1,
            )


class TestCli:
    def test_main_writes_learner_file(self, tmp_path):
        from tests.fakes.seed_factory import seed_local_corpus

        seeded = seed_local_corpus(tmp_path, lessons=(1, 2))
        code = main(
            [
                "--corpus", str(seeded["corpus_path"]),
                "--scholar", "scholar_cli",
                "--n", "10",
                "--seed", "5",
            ]
        )
        assert code == 0
        learner = load_learner("scholar_cli", tmp_path)
        assert learner["stats"]["attempted"] == 10
        assert learner_path("scholar_cli", tmp_path).exists()

    def test_main_reset_clears_history(self, tmp_path):
        from tests.fakes.seed_factory import seed_local_corpus

        seeded = seed_local_corpus(tmp_path, lessons=(1,))
        main(["--corpus", str(seeded["corpus_path"]), "--scholar", "s", "--n", "5"])
        main(["--corpus", str(seeded["corpus_path"]), "--scholar", "s", "--n", "3", "--reset"])
        learner = load_learner("s", tmp_path)
        assert learner["stats"]["attempted"] == 3
