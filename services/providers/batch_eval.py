"""L3 批量评估服务（S4.1）：近一周 evaluation 抽样 → RAGAS 四指标 → 周报聚合与告警。

设计文档 §9-6（周报 + 异常率 >10% 告警）/ 附录 B-3（10% 抽样）/ §5.6.4（冷启动
样本门槛）；数据契约 data-model-contract §4.11.7（evaluation_weekly_report）。
L3 为离线批处理：无新增 HTTP 接口（api-contract §3 注明），由
scripts/batch_eval_weekly.py 驱动。

设计约定：
- ragas / langchain_openai 延迟导入：未安装时仅 run_ragas_batch 不可用，
  抽样 / 周报构建 / 落库等其余能力不受影响（测试可完全离线运行）。
- LLM 可注入（run_ragas_batch(llm=...)），默认火山 OpenAI 兼容接口
  （VOLCANO_BASE_URL + LLM_JUDGE_MODEL）；测试注入 fake。
- 抽样确定性：seed 固定 + 输入先按 _id 排序 → 同 seed 同输入同输出。
"""
from __future__ import annotations

import datetime as _dt
import logging
import math
import random
import time
from statistics import fmean
from typing import Any, Optional

import config
from services.database import CloudBaseNoSQLClient
from services.evaluation_engine import EVALUATION_COLLECTION

logger = logging.getLogger("scholar-admin.batch_eval")

# 周报集合（契约 §4.11.7）
EVAL_REPORT_COLLECTION = "evaluation_weekly_report"

# RAGAS 四指标（与设计文档 §9-6 / 任务 S4.1-④ 对齐）
RAGAS_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)

# 抽样默认 seed（固定可复现，脚本可用 --seed 覆盖）
DEFAULT_SAMPLE_SEED = 42


# ---------------------------------------------------------------- 抽样（S4.1-③）

async def fetch_recent_evaluations(
    db: CloudBaseNoSQLClient, since_ms: int
) -> list[dict]:
    """查询 created_at >= since_ms 的 evaluation 记录（升序，契约 §4.11.2）。

    Args:
        db: 数据库客户端（真实 CloudBase 或 FakeDB）。
        since_ms: 起始时间戳（ms），如近一周。
    """
    result = await db.query(
        EVALUATION_COLLECTION,
        where={"created_at": {"$gte": since_ms}},
        order=[{"field": "created_at", "direction": "asc"}],
        limit=1000,
    )
    return result.get("records") or []


def sample_evaluations(
    records: list[dict],
    rate: float = config.EVAL_BATCH_SAMPLE_RATE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[dict]:
    """确定性抽样：同 seed 同输入 → 同输出（附录 B-3）。

    实现：先按 _id 稳定排序（消除输入顺序影响），再用固定 seed 的 RNG 无放回
    抽样。边界：rate<=0 → 空；count>=len → 全量；rate>=1 → 全量。
    """
    if rate <= 0 or not records:
        return []
    count = int(round(len(records) * rate))
    if count >= len(records):
        return list(records)
    ordered = sorted(records, key=lambda r: str(r.get("_id", "")))
    rng = random.Random(seed)
    return rng.sample(ordered, count)


# ---------------------------------------------------------------- RAGAS 指标（S4.1-④）

def _to_ragas_sample(record: dict) -> dict:
    """evaluation 记录 → RAGAS SingleTurnSample 输入映射（纯函数，可单测）。

    映射（执行计划 S4.1-④）：
    - question（目标表达）     = raw.evidence_snapshot.original
    - answer（用户输出）       = raw.evidence_snapshot.response
    - contexts（原文上下文）   = [original]（学习场景原句即上下文，无检索链路）
    - ground_truth（参考译文） = raw.evidence_snapshot.reference_translation
                               （落库含参考译文时优先），否则兜底 original
    """
    snapshot = ((record.get("raw") or {}).get("evidence_snapshot")) or {}
    original = (snapshot.get("original") or "").strip()
    response = (snapshot.get("response") or "").strip()
    reference = (
        (snapshot.get("reference_translation") or "").strip() or original
    )
    return {
        "user_input": original,
        "response": response,
        "retrieved_contexts": [original] if original else [],
        "reference": reference,
    }


def _safe_float(value: Any) -> Optional[float]:
    """RAGAS 分数安全转 float：NaN / 非数值 → None（失败行不参与聚合）。"""
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(num) else num


def _default_llm() -> Any:
    """默认 LLM：火山 OpenAI 兼容 ChatOpenAI（延迟导入 langchain_openai）。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=config.VOLCANO_BASE_URL,
        api_key=config.VOLCANO_API_KEY,
        model=config.LLM_JUDGE_MODEL,
        temperature=0,
    )


async def run_ragas_batch(
    samples: list[dict],
    llm: Any = None,
) -> list[dict]:
    """RAGAS 四指标批量评估（llm 可注入，默认 Volcano）。

    Args:
        samples: evaluation 记录列表（抽样结果）。
        llm: RAGAS LLM 客户端；None → 默认火山（LLM_JUDGE_MODEL）。
             测试注入 fake LLM（透传给 _evaluate_with_ragas）。

    Returns:
        与 samples 等长的 [{metric: float|None}]，失败行（NaN/异常）记 None。
    """
    if not samples:
        return []
    return await _evaluate_with_ragas(samples, llm)


async def _evaluate_with_ragas(
    samples: list[dict],
    llm: Any,
) -> list[dict]:
    """真实 RAGAS 计算（延迟导入 ragas：未安装时仅此不可用，不影响其余模块）。

    独立函数便于集成测试 monkeypatch 全链路离线运行（见
    tests/integration/test_batch_eval_weekly.py）。
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    try:
        # ragas 0.4.x：collections 路径无 DeprecationWarning
        from ragas.metrics.collections import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:  # pragma: no cover - 旧版回退
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

    dataset = EvaluationDataset(
        samples=[SingleTurnSample(**_to_ragas_sample(s)) for s in samples]
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm or _default_llm(),
        raise_exceptions=False,
    )
    if hasattr(result, "__await__"):
        result = await result

    scores = getattr(result, "scores", None) or []
    rows: list[dict] = []
    for i, sample in enumerate(samples):
        row = scores[i] if i < len(scores) else {}
        rows.append(
            {
                name: _safe_float(row.get(name) if isinstance(row, dict) else None)
                for name in RAGAS_METRICS
            }
        )
    return rows


# ---------------------------------------------------------------- 周报聚合与告警（S4.1-⑤）

def _iso_week_range(day: Optional[_dt.date] = None) -> str:
    """ISO 周键（如 2026-W33），周报幂等键（契约 §4.11.7）。"""
    iso = (day or _dt.date.today()).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def build_weekly_report(
    samples: list[dict],
    metrics: list[dict],
    week_range: Optional[str] = None,
    sample_rate: Optional[float] = None,
    alert_rate: Optional[float] = None,
    min_samples: Optional[int] = None,
) -> dict:
    """纯函数：样本 + 指标 → 周报文档（契约 §4.11.7）。

    - metrics 聚合：四指标均值 + 有效样本数（跳过 None/NaN）
    - anomaly_rate：verdict.anomaly=true 占比
    - alert：anomaly_rate > alert_rate 且 sample_count >= min_samples（§5.6.4 冷启动）
    - 可选参数默认取 config（保证生产默认一致，测试可显式传入）
    """
    week_range = week_range or _iso_week_range()
    sample_rate = (
        config.EVAL_BATCH_SAMPLE_RATE if sample_rate is None else sample_rate
    )
    alert_rate = config.EVAL_BATCH_ALERT_RATE if alert_rate is None else alert_rate
    min_samples = (
        config.EVAL_BATCH_MIN_SAMPLES if min_samples is None else min_samples
    )

    sample_count = len(samples)
    anomaly_count = sum(
        1 for s in samples if bool(((s.get("verdict") or {}).get("anomaly")))
    )
    anomaly_rate = anomaly_count / sample_count if sample_count else 0.0

    metric_report: dict[str, dict] = {}
    for name in RAGAS_METRICS:
        values = [
            m[name] for m in metrics if isinstance(m.get(name), (int, float))
        ]
        metric_report[name] = {
            "mean": round(fmean(values), 4) if values else 0.0,
            "sample_count": len(values),
        }

    return {
        "week_range": week_range,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "metrics": metric_report,
        "anomaly_rate": round(anomaly_rate, 4),
        "alert": bool(
            anomaly_rate > alert_rate and sample_count >= min_samples
        ),
    }


# ---------------------------------------------------------------- 周报落库（S4.1-⑥）

async def save_weekly_report(
    db: CloudBaseNoSQLClient, doc: dict
) -> Optional[str]:
    """周报落库：week_range 幂等 upsert（契约 §4.11.7）；返回 _id。"""
    week_range = doc["week_range"]
    existing = await db.query(
        EVAL_REPORT_COLLECTION, where={"week_range": week_range}, limit=1
    )
    records = existing.get("records") or []
    now_ms = int(time.time() * 1000)
    doc = {**doc, "updated_at": now_ms}

    if records:
        target_id = records[0]["_id"]
        await db.update(
            EVAL_REPORT_COLLECTION,
            where={"_id": target_id},
            data={"$set": {**doc, "_id": target_id}},
            multi=False,
        )
        return target_id

    doc = {**doc, "created_at": now_ms}
    inserted = await db.insert(EVAL_REPORT_COLLECTION, doc)
    return (inserted.get("ids") or [None])[0]


# ---------------------------------------------------------------- 通用工具

def recent_since_ms(days: int) -> int:
    """近 N 天起始时间戳（ms，UTC）。"""
    now = _dt.datetime.now(_dt.timezone.utc)
    return int((now - _dt.timedelta(days=days)).timestamp() * 1000)
