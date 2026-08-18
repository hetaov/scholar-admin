"""L3 周报全链路集成测试（S4.1-⑧）：fake_db + fake RAGAS 评估（monkeypatch
_evaluate_with_ragas）走通 抽样→指标→周报落库→告警/不告警两态 + 冷启动不告警。

- 不依赖 ragas / langchain 安装（评估函数被替换），离线可跑。
- fake LLM 通过 monkeypatch 的评估函数体现（透传 llm 参数在
  test_run_ragas_batch_passes_llm 中验证）。
"""
from __future__ import annotations

import time

import pytest

from services import batch_eval
from services.batch_eval import RAGAS_METRICS

NOW_MS = int(time.time() * 1000)
DAY_MS = 24 * 3600 * 1000


def _eval(record_id: str, anomaly: bool = False, days_ago: int = 0):
    return {
        "_id": record_id,
        "attempt_ref": f"learning_attempt:{record_id}",
        "level": "attempt",
        "type": "text",
        "score": 80,
        "confidence": 0.8,
        "verdict": {"meaningful": True, "faithfulness": 1.0, "anomaly": anomaly},
        "raw": {
            "evidence_snapshot": {
                "original": f"It is a watch. {record_id}",
                "response": f"它是一块手表。{record_id}",
            }
        },
        "created_at": NOW_MS - days_ago * DAY_MS,
    }


async def _run_pipeline(db, *, days=7, rate=0.1, seed=42, alert_rate=None,
                        min_samples=None):
    """模拟 scripts/batch_eval_weekly.py 的核心流程（不含 argparse/打印）。"""
    since_ms = batch_eval.recent_since_ms(days)
    records = await batch_eval.fetch_recent_evaluations(db, since_ms)
    samples = batch_eval.sample_evaluations(records, rate, seed)
    if not samples:
        # 与 scripts/batch_eval_weekly.py 一致：无样本不生成周报、不落库
        return {"records": records, "samples": [], "report": None, "doc_id": None}
    metrics = await batch_eval.run_ragas_batch(samples, llm="fake-llm")
    report = batch_eval.build_weekly_report(
        samples,
        metrics,
        sample_rate=rate,
        alert_rate=alert_rate,
        min_samples=min_samples,
    )
    doc_id = await batch_eval.save_weekly_report(db, report)
    return {"records": records, "samples": samples, "report": report, "doc_id": doc_id}


def _fake_evaluate(value: float = 0.85):
    async def _inner(samples, llm):
        assert llm == "fake-llm" or llm is None
        return [
            {name: value for name in RAGAS_METRICS} for _ in samples
        ]

    return _inner


class TestWeeklyPipeline:
    @pytest.mark.asyncio
    async def test_alert_on_high_anomaly(self, fake_db, monkeypatch):
        # 120 条近一周 evaluation，其中 30 条 anomaly（25% > 10% 阈值）
        for i in range(120):
            fake_db.add("evaluation", _eval(f"e{i}", anomaly=(i % 4 == 0), days_ago=i % 7))
        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", _fake_evaluate(0.85))

        result = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )

        assert len(result["samples"]) == 12  # 120 * 0.1
        report = result["report"]
        # 总体异常 25%（30/120）；抽样为随机无放回，比例有波动，只需显著 > 10% 触发告警
        assert report["anomaly_rate"] > 0.1
        assert report["alert"] is True
        assert report["metrics"]["faithfulness"]["mean"] == pytest.approx(0.85)
        assert report["metrics"]["faithfulness"]["sample_count"] == 12

        # 落库校验
        rows = fake_db.all("evaluation_weekly_report")
        assert len(rows) == 1
        assert rows[0]["week_range"] == report["week_range"]
        assert rows[0]["alert"] is True

    @pytest.mark.asyncio
    async def test_no_alert_on_normal_quality(self, fake_db, monkeypatch):
        for i in range(120):
            fake_db.add("evaluation", _eval(f"e{i}", anomaly=False, days_ago=i % 7))
        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", _fake_evaluate(0.9))

        result = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )

        assert result["report"]["anomaly_rate"] == 0.0
        assert result["report"]["alert"] is False

    @pytest.mark.asyncio
    async def test_cold_start_no_alert_below_min_samples(self, fake_db, monkeypatch):
        # §5.6.4 冷启动：样本 5 < min_samples 10，即使异常率 60% 也不告警
        for i in range(50):
            fake_db.add("evaluation", _eval(f"e{i}", anomaly=(i % 5 < 3), days_ago=i % 7))
        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", _fake_evaluate(0.5))

        result = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )

        assert len(result["samples"]) == 5
        # 抽样比例有波动，但样本数 5 < min_samples 10，无论异常率多高都不告警（§5.6.4）
        assert result["report"]["anomaly_rate"] > 0.1
        assert result["report"]["sample_count"] == 5
        assert result["report"]["alert"] is False

    @pytest.mark.asyncio
    async def test_rerun_same_week_upserts(self, fake_db, monkeypatch):
        for i in range(60):
            fake_db.add("evaluation", _eval(f"e{i}", anomaly=(i % 5 == 0), days_ago=i % 7))
        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", _fake_evaluate(0.8))

        first = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )
        second = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )

        assert first["doc_id"] == second["doc_id"]
        assert len(fake_db.all("evaluation_weekly_report")) == 1  # 幂等，无新增

    @pytest.mark.asyncio
    async def test_no_samples_skips_report(self, fake_db, monkeypatch):
        # 无近一周数据 → 空抽样 → 不落库（脚本打印提示）
        for i in range(5):
            fake_db.add("evaluation", _eval(f"e{i}", days_ago=30))
        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", _fake_evaluate(0.8))

        result = await _run_pipeline(
            fake_db, days=7, rate=0.1, seed=42, alert_rate=0.1, min_samples=10
        )
        assert result["samples"] == []
        assert result["report"] is None  # 无样本不生成周报
        assert fake_db.all("evaluation_weekly_report") == []


class TestRunRagasBatch:
    @pytest.mark.asyncio
    async def test_passes_llm_and_returns_rows(self, monkeypatch):
        captured = {}

        async def fake_evaluate(samples, llm):
            captured["llm"] = llm
            captured["count"] = len(samples)
            return [{name: 0.9 for name in RAGAS_METRICS} for _ in samples]

        monkeypatch.setattr(batch_eval, "_evaluate_with_ragas", fake_evaluate)
        samples = [_eval("e1"), _eval("e2")]
        rows = await batch_eval.run_ragas_batch(samples, llm="my-fake-llm")

        assert captured == {"llm": "my-fake-llm", "count": 2}
        assert len(rows) == 2
        assert rows[0]["faithfulness"] == 0.9

    @pytest.mark.asyncio
    async def test_empty_samples(self):
        assert await batch_eval.run_ragas_batch([]) == []
