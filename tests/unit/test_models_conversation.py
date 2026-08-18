"""单元测试：Conversation 数据层 + L1 轻量状态机（设计文档 §4.2，附录 B-1）

覆盖：
- 状态机：达意成功重置 / 连续失败 hint → rephrase → 降档 / 最低档转训练建议
- 会话文档构建（active 态默认档位、冷启动先验默认 difficulty=1）
- 轮次文档构建（证据快照不可变）
- 会话小结统计（达意率/平均分/平均置信度）
"""
from __future__ import annotations

import pytest

from services.models_conversation import (
    DEFAULT_DIFFICULTY,
    SESSION_STAGE_ACTIVE,
    TURN_STAGE_ANSWER,
    TURN_STAGE_DOWNGRADE,
    TURN_STAGE_HINT,
    TURN_STAGE_REPHRASE,
    build_session_doc,
    build_session_summary,
    build_turn_doc,
    next_turn_stage,
    session_gate,
)


class TestNextTurnStage:
    def test_success_resets(self):
        r = next_turn_stage(
            consecutive_failures=2, difficulty=1, meaningful=True, low_confidence=False
        )
        assert r["stage"] == TURN_STAGE_ANSWER
        assert r["reset_failures"] is True
        assert r["difficulty"] == 1

    def test_low_confidence_not_meaningful(self):
        # 低置信即使 meaningful=True 也不视为成功（§9-2 门控语义）→ 计为失败
        r = next_turn_stage(
            consecutive_failures=1, difficulty=1, meaningful=True, low_confidence=True
        )
        assert r["stage"] == TURN_STAGE_HINT  # 累计 1 次失败进入 hint
        assert r["hint"] is True

    def test_first_failure_hint(self):
        r = next_turn_stage(
            consecutive_failures=1, difficulty=1, meaningful=False, low_confidence=False
        )
        assert r["stage"] == TURN_STAGE_HINT
        assert r["hint"] is True
        assert r["reset_failures"] is False

    def test_second_failure_rephrase(self):
        r = next_turn_stage(
            consecutive_failures=2, difficulty=1, meaningful=False, low_confidence=False
        )
        assert r["stage"] == TURN_STAGE_REPHRASE
        assert r["rephrased"] is True

    def test_third_failure_downgrade(self):
        r = next_turn_stage(
            consecutive_failures=3, difficulty=2, meaningful=False, low_confidence=False
        )
        assert r["stage"] == TURN_STAGE_DOWNGRADE
        assert r["difficulty"] == 1  # 2 → 1 降档
        assert r["reset_failures"] is True

    def test_min_difficulty_suggestion(self):
        # 已在最低档且累计 3 次失败 → 不再降档，输出转训练建议（P0 不终止会话）
        r = next_turn_stage(
            consecutive_failures=3, difficulty=1, meaningful=False, low_confidence=False
        )
        assert r["stage"] == TURN_STAGE_DOWNGRADE
        assert r["difficulty"] == 1
        assert r["suggestion"] is not None


class TestSessionDoc:
    def test_default_cold_start(self):
        doc = build_session_doc(scholar_id="u1")
        assert doc["difficulty"] == DEFAULT_DIFFICULTY == 1  # 冷启动先验默认（§5.6.1）
        assert doc["stage"] == SESSION_STAGE_ACTIVE
        assert doc["ended_at"] is None
        assert doc["skill_updates"] == []
        assert doc["review_schedule"] == []

    def test_sentence_ids_bound(self):
        doc = build_session_doc(scholar_id="u1", sentence_ids=["s1", "s2"])
        assert doc["sentence_ids"] == ["s1", "s2"]


class TestTurnDoc:
    def test_evidence_snapshot_immutable(self):
        doc = build_turn_doc(
            session_id="cvs_1",
            sentence_id="s1",
            original_text="It is a watch.",
            translation="这是一块手表。",
            utterance="it is a watch",
            reply="Good job!",
            stage=TURN_STAGE_ANSWER,
        )
        assert doc["original_text"] == "It is a watch."
        assert doc["translation"] == "这是一块手表。"
        assert doc["utterance"] == "it is a watch"
        assert doc["stage"] == TURN_STAGE_ANSWER
        assert doc["hint"] is None


class TestSessionSummary:
    def test_empty(self):
        s = build_session_summary([])
        assert s["total_turns"] == 0
        assert s["meaningful_rate"] == 0.0

    def test_mixed(self):
        turns = [
            {"eval_verdict": {"meaningful": True, "score": 90, "confidence": 0.9}},
            {"eval_verdict": {"meaningful": False, "score": 40, "confidence": 0.5}},
        ]
        s = build_session_summary(turns)
        assert s["total_turns"] == 2
        assert s["meaningful_rate"] == 0.5
        assert s["avg_score"] == 65
        assert round(s["avg_confidence"], 2) == 0.7

    def test_p1_faithfulness_and_anomaly_rates(self):
        turns = [
            {
                "eval_verdict": {
                    "meaningful": True,
                    "score": 90,
                    "confidence": 0.9,
                    "faithfulness": True,
                    "anomaly": False,
                }
            },
            {
                "eval_verdict": {
                    "meaningful": False,
                    "score": 40,
                    "confidence": 0.5,
                    "faithfulness": False,
                    "anomaly": True,
                }
            },
        ]
        s = build_session_summary(turns)
        # 忠实率分母 = 达意轮数（1 达意 1 忠实 → 1.0）；异常率分母 = 全部轮
        assert s["faithfulness_rate"] == 1.0
        assert s["anomaly_rate"] == 0.5

    def test_p1_faithfulness_no_meaningful_turns_is_neutral(self):
        turns = [
            {
                "eval_verdict": {
                    "meaningful": False,
                    "score": 40,
                    "confidence": 0.5,
                    "faithfulness": False,
                    "anomaly": False,
                }
            }
        ]
        s = build_session_summary(turns)
        assert s["faithfulness_rate"] == 1.0  # 无达意轮视为无偏差（不沉淀另有门控）


# ---------------------------------------------------------------------------
# S3.1 P1：会话级门控（§9-2）
# ---------------------------------------------------------------------------


def _verdict_turns(verdicts: list[dict]) -> list[dict]:
    turns = []
    for v in verdicts:
        item = dict(v)
        item.setdefault("meaningful", True)  # 忠实率分母 = 达意轮
        turns.append({"eval_verdict": item})
    return turns


class TestSessionGate:
    def test_all_clear(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.9, "faithfulness": True, "anomaly": False},
                {"confidence": 0.8, "faithfulness": True, "anomaly": False},
            ]
        )
        g = session_gate(turns)
        assert g["downgrade_factor"] == 1.0
        assert g["ai_content_bias"] is False
        assert g["alert"] is False

    def test_consecutive_two_low_conf_downgrades(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.9, "faithfulness": True, "anomaly": False},
                {"confidence": 0.3, "faithfulness": True, "anomaly": False},
                {"confidence": 0.4, "faithfulness": True, "anomaly": False},
            ]
        )
        g = session_gate(turns)
        assert g["consecutive_low_conf"] == 2
        assert g["downgrade_factor"] == 0.5

    def test_interleaved_low_conf_no_downgrade(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.3, "faithfulness": True, "anomaly": False},
                {"confidence": 0.9, "faithfulness": True, "anomaly": False},
                {"confidence": 0.4, "faithfulness": True, "anomaly": False},
            ]
        )
        g = session_gate(turns)
        assert g["consecutive_low_conf"] == 1
        assert g["downgrade_factor"] == 1.0

    def test_cold_start_exempts_downgrade(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.9, "faithfulness": True, "anomaly": False},
                {"confidence": 0.3, "faithfulness": True, "anomaly": False},
                {"confidence": 0.4, "faithfulness": True, "anomaly": False},
            ]
        )
        g = session_gate(turns, is_cold=True)
        assert g["consecutive_low_conf"] == 2
        assert g["downgrade_factor"] == 1.0  # 冷启动期豁免（§9-2）

    def test_low_faithfulness_marks_ai_content_bias(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.9, "faithfulness": True, "anomaly": False},
                {"confidence": 0.8, "faithfulness": False, "anomaly": False},
                {"confidence": 0.7, "faithfulness": False, "anomaly": False},
            ]
        )
        g = session_gate(turns)
        assert g["faithfulness_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert g["ai_content_bias"] is True

    def test_anomaly_rate_alert(self):
        turns = _verdict_turns(
            [
                {"confidence": 0.9, "faithfulness": True, "anomaly": True},
                {"confidence": 0.8, "faithfulness": True, "anomaly": True},
                {"confidence": 0.7, "faithfulness": True, "anomaly": False},
            ]
        )
        g = session_gate(turns)
        assert g["anomaly_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert g["alert"] is True

    def test_empty_turns(self):
        g = session_gate([])
        assert g["downgrade_factor"] == 1.0
        assert g["ai_content_bias"] is False
        assert g["alert"] is False
