"""单元测试：Training 受控任务数据层（设计文档 §3.2/§3.3，附录 B-2）

覆盖：
- 任务文档构建（mode='training' 并入 learning_attempt，不建 training_session）
- item 构建（三种活动类型的 stimulus/prompt）
- 逐题判定（正确 / 低置信 / 空作答 / 不达意）
- 结果合并与任务汇总（overall / error_type / attempt_status）
- 弱项驱动排序（含该 skill 状态最弱优先；无状态冷启动排末尾）
"""
from __future__ import annotations

from services.models_training import (
    apply_results_to_task,
    build_item,
    build_training_task,
    evaluate_item,
    sort_sentences_by_weakness,
    summarize_results,
)


def _stub_judge(score: int, meaningful: bool, confidence: float, anomaly: bool = False):
    def judge(original: str, response: str):
        return {
            "score": score,
            "meaningful": meaningful,
            "confidence": confidence,
            "anomaly": anomaly,
        }
    return judge


class TestBuildTask:
    def test_mode_training_in_learning_attempt(self):
        task = build_training_task(
            scholar_id="u1", skill_code="translation", difficulty=1, items=[]
        )
        assert task["mode"] == "training"
        assert task["attempt_status"] == "pending"
        assert task["task_id"].startswith("trn_")

    def test_items(self):
        item = build_item(0, {"sentence_id": "s1", "text": "It is a watch.", "translation": "这是一块手表。"}, "translation")
        assert item["item_id"] == "it_1"
        assert item["prompt"] == "翻译为英文：这是一块手表。"


class TestEvaluateItem:
    def test_correct_high_confidence(self):
        item = {"item_id": "it_1", "content": "It is a watch.", "sentence_id": "s1"}
        r = evaluate_item(item, "it is a watch", _stub_judge(90, True, 0.9))
        assert r["correct"] is True
        assert r["score"] == 90

    def test_low_confidence_not_correct(self):
        item = {"item_id": "it_1", "content": "It is a watch.", "sentence_id": "s1"}
        r = evaluate_item(item, "it is a watch", _stub_judge(90, True, 0.5))
        assert r["correct"] is False
        assert "无法可靠判定" in r["feedback"]

    def test_anomaly_empty_response(self):
        item = {"item_id": "it_1", "content": "It is a watch.", "sentence_id": "s1"}
        r = evaluate_item(item, "", _stub_judge(0, False, 0.9, anomaly=True))
        assert r["correct"] is False
        assert "未检测到有效作答" in r["feedback"]

    def test_not_meaningful_fail(self):
        item = {"item_id": "it_1", "content": "It is a watch.", "sentence_id": "s1"}
        r = evaluate_item(item, "banana apple", _stub_judge(15, False, 0.5))
        assert r["correct"] is False


class TestSummarizeAndApply:
    def test_summarize(self):
        s = summarize_results([
            {"correct": True, "score": 90},
            {"correct": False, "score": 40},
            {"correct": True, "score": 80},
        ])
        assert s == {"correct": 2, "total": 3, "avg_score": 70}

    def test_apply_results(self):
        task = build_training_task(
            scholar_id="u1",
            skill_code="translation",
            difficulty=1,
            items=[
                {"item_id": "it_1", "sentence_id": "s1", "content": "A", "prompt": "P"},
                {"item_id": "it_2", "sentence_id": "s2", "content": "B", "prompt": "P"},
            ],
        )
        updated = apply_results_to_task(task, [
            {"item_id": "it_1", "correct": True, "feedback": "ok", "score": 90, "confidence": 0.9},
            {"item_id": "it_2", "correct": False, "feedback": "no", "score": 40, "confidence": 0.5},
        ])
        assert updated["attempt_status"] == "failed"
        assert updated["error_type"] == "comprehension"
        assert updated["overall"]["correct"] == 1
        assert updated["items"][0]["score"] == 90


class TestSortByWeakness:
    def test_weakest_first(self):
        sentences = [
            {"sentence_id": "s1", "order": 1, "_states": [{"skill_code": "translation", "mastery_score": 90, "status": "mastered"}]},
            {"sentence_id": "s2", "order": 2, "_states": [{"skill_code": "translation", "mastery_score": 30, "status": "learning"}]},
            {"sentence_id": "s3", "order": 3, "_states": [{"skill_code": "translation", "mastery_score": 0, "status": "not_started"}]},
        ]
        ordered = sort_sentences_by_weakness(sentences, "translation")
        assert [s["sentence_id"] for s in ordered] == ["s3", "s2", "s1"]

    def test_cold_start_no_state_last(self):
        sentences = [
            {"sentence_id": "s1", "order": 1, "_states": [{"skill_code": "translation", "mastery_score": 60, "status": "learning"}]},
            {"sentence_id": "s2", "order": 2, "_states": []},
        ]
        ordered = sort_sentences_by_weakness(sentences, "translation")
        # 有状态者优先；无状态（冷启动）排末尾但不剔除（§9-9 不阻断）
        assert ordered[0]["sentence_id"] == "s1"
        assert ordered[1]["sentence_id"] == "s2"
