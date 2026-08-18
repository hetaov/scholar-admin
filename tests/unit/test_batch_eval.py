"""L3 批量评估服务单测（S4.1-⑦）：抽样确定性/边界、RAGAS 输入映射、周报聚合、
异常率阈值边界（0.1）、样本 <100 不告警、周报幂等落库。

纯函数不触网；ragas / langchain 延迟导入，未安装不影响本测试。
"""
from __future__ import annotations

import datetime as _dt

import pytest

from services import batch_eval
from services.batch_eval import (
    RAGAS_METRICS,
    build_weekly_report,
    fetch_recent_evaluations,
    sample_evaluations,
    save_weekly_report,
    _iso_week_range,
    _safe_float,
    _to_ragas_sample,
)


def _eval(record_id: str, anomaly: bool = False, snapshot: dict | None = None):
    """构造最小 evaluation 记录（对齐契约 §4.11.2 落库结构）。"""
    return {
        "_id": record_id,
        "attempt_ref": f"learning_attempt:{record_id}",
        "level": "attempt",
        "type": "text",
        "score": 80,
        "confidence": 0.8,
        "verdict": {"meaningful": True, "faithfulness": 1.0, "anomaly": anomaly},
        "raw": {
            "evidence_snapshot": snapshot
            or {"original": "目标表达", "response": "用户输出"}
        },
        "created_at": 1700000000000,
    }


def _metric_row(**values):
    return {name: values.get(name) for name in RAGAS_METRICS}


# ---------------------------------------------------------------- 抽样（S4.1-③）

class TestSampleEvaluations:
    def test_same_seed_same_output(self):
        records = [_eval(f"e{i}") for i in range(20)]
        first = sample_evaluations(records, 0.1, seed=42)
        second = sample_evaluations(records, 0.1, seed=42)
        assert [r["_id"] for r in first] == [r["_id"] for r in second]

    def test_order_independent(self):
        records = [_eval(f"e{i}") for i in range(20)]
        shuffled = list(reversed(records))
        a = sample_evaluations(records, 0.5, seed=7)
        b = sample_evaluations(shuffled, 0.5, seed=7)
        assert [r["_id"] for r in a] == [r["_id"] for r in b]

    def test_rate_zero_returns_empty(self):
        records = [_eval(f"e{i}") for i in range(10)]
        assert sample_evaluations(records, 0.0, seed=42) == []
        assert sample_evaluations(records, -0.1, seed=42) == []

    def test_rate_at_least_one_returns_all(self):
        records = [_eval(f"e{i}") for i in range(5)]
        sampled = sample_evaluations(records, 1.0, seed=42)
        assert len(sampled) == 5
        assert {r["_id"] for r in sampled} == {f"e{i}" for i in range(5)}

    def test_count_rounds(self):
        # 10 * 0.1 = 1；15 * 0.1 = 1.5 → round = 2
        assert len(sample_evaluations([_eval(f"e{i}") for i in range(10)], 0.1, 42)) == 1
        assert len(sample_evaluations([_eval(f"e{i}") for i in range(15)], 0.1, 42)) == 2

    def test_empty_records(self):
        assert sample_evaluations([], 0.1, seed=42) == []


# ---------------------------------------------------------------- RAGAS 输入映射（S4.1-④）

class TestToRagasSample:
    def test_basic_mapping(self):
        record = _eval("e1", snapshot={"original": "It is a watch.", "response": "它是一块手表。"})
        mapped = _to_ragas_sample(record)
        assert mapped["user_input"] == "It is a watch."
        assert mapped["response"] == "它是一块手表。"
        assert mapped["retrieved_contexts"] == ["It is a watch."]

    def test_reference_falls_back_to_original(self):
        record = _eval("e1", snapshot={"original": "It is a watch.", "response": "它是一块手表。"})
        assert _to_ragas_sample(record)["reference"] == "It is a watch."

    def test_reference_translation_preferred(self):
        record = _eval(
            "e1",
            snapshot={
                "original": "It is a watch.",
                "response": "它是一块手表。",
                "reference_translation": "这是一块手表。",
            },
        )
        assert _to_ragas_sample(record)["reference"] == "这是一块手表。"

    def test_empty_snapshot_graceful(self):
        record = {"_id": "e1", "raw": {"evidence_snapshot": {}}}
        mapped = _to_ragas_sample(record)
        assert mapped["user_input"] == ""
        assert mapped["response"] == ""
        assert mapped["retrieved_contexts"] == []
        assert mapped["reference"] == ""


class TestSafeFloat:
    def test_nan_to_none(self):
        assert _safe_float(float("nan")) is None

    def test_none_and_bad_value(self):
        assert _safe_float(None) is None
        assert _safe_float("oops") is None

    def test_number_passthrough(self):
        assert _safe_float(0.85) == 0.85
        assert _safe_float("0.9") == 0.9


# ---------------------------------------------------------------- 周报聚合与告警（S4.1-⑤）

class TestBuildWeeklyReport:
    def test_metric_aggregation_skips_none(self):
        samples = [_eval("e1"), _eval("e2"), _eval("e3")]
        metrics = [
            _metric_row(faithfulness=0.8, answer_relevancy=0.9),
            _metric_row(faithfulness=0.6, answer_relevancy=None),
            _metric_row(faithfulness=0.7, answer_relevancy=None),
        ]
        report = build_weekly_report(samples, metrics)
        assert report["metrics"]["faithfulness"] == {
            "mean": 0.7,
            "sample_count": 3,
        }
        assert report["metrics"]["answer_relevancy"] == {
            "mean": 0.9,
            "sample_count": 1,
        }
        assert report["sample_count"] == 3

    def test_anomaly_rate(self):
        samples = [_eval("e1", anomaly=True), _eval("e2"), _eval("e3", anomaly=True)]
        report = build_weekly_report(samples, [_metric_row()] * 3)
        assert report["anomaly_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_alert_requires_strictly_greater_than_threshold(self):
        # 恰好等于 0.1 → 不告警（严格大于）
        samples = [_eval(f"e{i}", anomaly=(i == 0)) for i in range(10)]
        metrics = [_metric_row() for _ in samples]
        report = build_weekly_report(
            samples, metrics, alert_rate=0.1, min_samples=1
        )
        assert report["anomaly_rate"] == pytest.approx(0.1)
        assert report["alert"] is False

    def test_alert_above_threshold(self):
        # 0.2 > 0.1 → 告警
        samples = [_eval(f"e{i}", anomaly=(i < 2)) for i in range(10)]
        metrics = [_metric_row() for _ in samples]
        report = build_weekly_report(
            samples, metrics, alert_rate=0.1, min_samples=1
        )
        assert report["anomaly_rate"] == pytest.approx(0.2)
        assert report["alert"] is True

    def test_alert_disabled_below_min_samples(self):
        # §5.6.4 冷启动：样本 < min_samples 不告警（即使异常率高）
        samples = [_eval(f"e{i}", anomaly=True) for i in range(10)]
        metrics = [_metric_row() for _ in samples]
        report = build_weekly_report(
            samples, metrics, alert_rate=0.1, min_samples=100
        )
        assert report["anomaly_rate"] == pytest.approx(1.0)
        assert report["alert"] is False

    def test_empty_samples(self):
        report = build_weekly_report([], [])
        assert report["sample_count"] == 0
        assert report["anomaly_rate"] == 0.0
        assert report["alert"] is False
        assert report["metrics"]["faithfulness"] == {"mean": 0.0, "sample_count": 0}

    def test_week_range_default_and_override(self):
        report = build_weekly_report([], [], week_range="2026-W33")
        assert report["week_range"] == "2026-W33"
        default = build_weekly_report([], [])
        assert default["week_range"] == _iso_week_range()


class TestIsoWeekRange:
    def test_known_week(self):
        assert _iso_week_range(_dt.date(2026, 8, 18)) == "2026-W34"

    def test_monday(self):
        assert _iso_week_range(_dt.date(2026, 8, 17)) == "2026-W34"


# ---------------------------------------------------------------- 近一周拉取 / 周报落库（S4.1-③⑥）

class TestFetchAndSave:
    @pytest.mark.asyncio
    async def test_fetch_recent_filters_by_created_at(self, fake_db):
        fake_db.add(
            "evaluation",
            _eval("e1", snapshot={"original": "a", "response": "b"}),
        )
        fake_db.add(
            "evaluation",
            {
                **_eval("e2", snapshot={"original": "a", "response": "b"}),
                "created_at": 1699999999999,  # 早于 since_ms
            },
        )
        fake_db.add(
            "evaluation",
            {
                **_eval("e3", snapshot={"original": "a", "response": "b"}),
                "created_at": 1700000001000,
            },
        )
        records = await fetch_recent_evaluations(fake_db, since_ms=1700000000000)
        assert [r["_id"] for r in records] == ["e1", "e3"]

    @pytest.mark.asyncio
    async def test_save_weekly_report_upsert_by_week_range(self, fake_db):
        doc = build_weekly_report(
            [_eval("e1")], [_metric_row(faithfulness=0.8)], week_range="2026-W34"
        )
        first_id = await save_weekly_report(fake_db, doc)
        assert first_id is not None

        # 同周重跑 → 更新而非新增（幂等）
        updated = build_weekly_report(
            [_eval("e1"), _eval("e2")],
            [_metric_row(faithfulness=0.9), _metric_row(faithfulness=0.7)],
            week_range="2026-W34",
        )
        second_id = await save_weekly_report(fake_db, updated)
        assert second_id == first_id

        rows = fake_db.all("evaluation_weekly_report")
        assert len(rows) == 1
        assert rows[0]["sample_count"] == 2
        assert rows[0]["metrics"]["faithfulness"]["mean"] == pytest.approx(0.8)
