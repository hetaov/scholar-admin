"""L3 周报批处理脚本（S4.1-⑥）：近一周 evaluation 抽样 → RAGAS 四指标 → 周报落库 + 告警。

用法示例（服务进程 .venv 中运行）：
    .venv/bin/python scripts/batch_eval_weekly.py                      # 近 7 天、10% 抽样、seed=42
    .venv/bin/python scripts/batch_eval_weekly.py --since-days 30 --sample-rate 0.2 --seed 7
    .venv/bin/python scripts/batch_eval_weekly.py --dry-run            # 只汇总不落库

依赖：ragas / langchain-openai（requirements.txt S4.1 段）；未安装时脚本在
run_ragas_batch 处报错并给出提示（详见 会话训练评估重构执行计划.md §5 S4.1 登记）。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 允许直接以脚本方式运行（scripts/ 下无包结构）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services import batch_eval  # noqa: E402
from services.dependencies import get_db  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger("batch_eval_weekly")


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _print_summary(report: dict) -> None:
    print("\n========== L3 周报汇总（evaluation_weekly_report） ==========")
    print(f"week_range     : {report['week_range']}")
    print(f"sample_rate    : {report['sample_rate']}")
    print(f"sample_count   : {report['sample_count']}")
    print(f"anomaly_rate   : {report['anomaly_rate']}")
    print("metrics (RAGAS):")
    for name, agg in report["metrics"].items():
        print(f"  - {name:<18} mean={_fmt(agg['mean'])}  sample_count={agg['sample_count']}")
    print(f"alert          : {'⚠ ALERT' if report['alert'] else 'OK'}")
    print("=============================================================")


async def run(args: argparse.Namespace) -> int:
    db = get_db()
    since_ms = batch_eval.recent_since_ms(args.since_days)

    logger.info(
        "拉取近 %d 天 evaluation（since_ms=%d, sample_rate=%s, seed=%s, dry_run=%s）",
        args.since_days,
        since_ms,
        args.sample_rate,
        args.seed,
        args.dry_run,
    )
    records = await batch_eval.fetch_recent_evaluations(db, since_ms)
    samples = batch_eval.sample_evaluations(records, args.sample_rate, args.seed)
    logger.info("近一周 evaluation=%d 条，抽样=%d 条", len(records), len(samples))

    if not samples:
        print("无抽样样本，本次不生成周报。")
        return 0

    metrics = await batch_eval.run_ragas_batch(samples)
    report = batch_eval.build_weekly_report(
        samples,
        metrics,
        sample_rate=args.sample_rate,
    )
    _print_summary(report)

    if args.dry_run:
        print("dry-run：未落库。")
        return 0

    doc_id = await batch_eval.save_weekly_report(db, report)
    logger.info("周报已落库 _id=%s（week_range=%s 幂等覆盖）", doc_id, report["week_range"])
    if report["alert"]:
        print(f"\n⚠ ALERT：异常率 {report['anomaly_rate']} > 阈值 "
              f"{batch_eval.config.EVAL_BATCH_ALERT_RATE}，请关注本周评估质量！")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="L3 周报批处理：近一周 evaluation 抽样 → RAGAS 四指标 → 周报落库 + 告警"
    )
    parser.add_argument(
        "--since-days", type=int, default=7, help="统计近 N 天（默认 7）"
    )
    parser.add_argument(
        "--sample-rate", type=float, default=0.1, help="抽样率（默认 0.1）"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="抽样随机种子（默认 42，固定可复现）"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只汇总打印，不落库"
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
