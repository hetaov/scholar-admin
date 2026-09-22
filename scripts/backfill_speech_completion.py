#!/usr/bin/env python3
"""存量 speech_evaluation.parsed.completion 量纲回填脚本（2026-09-22 缺陷修复配套）。

背景：
- `PronCompletion` 的官方量纲是 **0~1**（同 `PronFluency`；F1-3 定标实测：干净音频 = `1.0`），
  但 `normalize_soe_result` 此前**只对 fluency 做了 ×100，completion 原值透传**
  ⇒ 存量 `parsed.completion` 全部是 0~1 旧口径（干净跟读 ≈ 1.0，前端显示「完整度 1」）。
- 代码已修（新写入为 0~100）；本脚本一次性回填存量。

范围（严格，勿扩大）：
- **仅改 `parsed.completion` 一个字段**；`accuracy`/`fluency`/`suggested_score`/`words` 原样保留。
- **不改 `raw`**（原始 JSON 存档是定标/复核依据，必须保持原样）。
- 期望值**复用 `normalize_soe_result(raw)`**（与写侧同源），不写第二套换算公式。
- 仅当 `expected > stored` 才回填（本次缺陷是「漏乘 100」，方向单一）；
  `expected < stored` 视为异常，**不动**并单独计数，留人工判断。
- 幂等：回填后 `stored == expected` → 重跑 0 变更。

用法：
  python scripts/backfill_speech_completion.py                # dry-run（默认，只打印计划）
  python scripts/backfill_speech_completion.py --limit 5       # 金丝雀：只看前 5 条计划
  python scripts/backfill_speech_completion.py --commit        # 真实写入
  python scripts/backfill_speech_completion.py --commit --limit 100  # 小批灰度写入

退出码：0 = 执行完成；1 = 运行异常。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("backfill_speech_completion")

from services.infra.database import CloudBaseNoSQLClient  # noqa: E402
from services.providers.speech_eval import (  # noqa: E402
    SPEECH_EVALUATION_COLLECTION,
    normalize_soe_result,
)

PAGE = 100

# 计划条目：(doc_id, stored_completion, new_completion)
PlanItem = tuple[str, float | None, float]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="回填存量 speech_evaluation.parsed.completion：0~1 旧口径 → ×100 的 0~100 口径"
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="真实写入 DB（默认 dry-run，只打印计划）",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="只处理前 N 条待回填记录（金丝雀/灰度）；缺省=全部",
    )
    return p.parse_args()


def _as_float(value: Any) -> float | None:
    """宽松数值转换；None/非数 → None（区别于真实的 0.0）。"""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_raw_dict(raw: Any) -> dict | None:
    """`raw` 归一为 dict：正常是 dict；存量若被存成 JSON 文本则解析；否则不可判定。"""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def build_plan(
    records: Iterable[dict],
    *,
    limit: int | None = None,
) -> tuple[list[PlanItem], int, int, int]:
    """纯函数：生成 (doc_id → 新 completion) 回填计划。

    规则：
    - `raw` 不可归一出 dict，或 `parsed` 非 dict → skipped（不可判定，不猜）。
    - `expected = normalize_soe_result(raw)["completion"]`（**与写侧同源**）。
    - `expected == stored` → already（新口径 / 本就为 0）→ 幂等。
    - `expected > stored` → 进计划（本次缺陷是漏乘 100，方向单一）。
    - `expected < stored` → 异常，**不进计划**（不动），单独计数待人工判断。

    返回 (to_fill, already, skipped, anomalies)：
    - to_fill: list[(doc_id, stored_completion, new_completion)]
    - skipped: raw/parsed 不可判定
    - anomalies: expected < stored（未被本缺陷解释的形态）
    """
    to_fill: list[PlanItem] = []
    already = 0
    skipped = 0
    anomalies = 0

    for doc in records:
        doc_id = doc.get("_id")
        parsed = doc.get("parsed")
        raw = _as_raw_dict(doc.get("raw"))
        if not doc_id or not isinstance(parsed, dict) or raw is None:
            skipped += 1
            continue

        stored = _as_float(parsed.get("completion"))
        expected = float(normalize_soe_result(raw)["completion"])

        if stored is not None and expected == stored:
            already += 1
            continue
        if expected > (stored if stored is not None else -1.0):
            to_fill.append((str(doc_id), stored, expected))
            continue
        anomalies += 1

    if limit is not None and limit >= 0:
        to_fill = to_fill[:limit]
    return to_fill, already, skipped, anomalies


async def fetch_all(db: CloudBaseNoSQLClient) -> list[dict]:
    """分页拉取 speech_evaluation 全量（只需 _id / parsed / raw）。"""
    all_records: list[dict] = []
    offset = 0
    select = {"_id": 1, "parsed": 1, "raw": 1}
    while True:
        res = await db.query(
            SPEECH_EVALUATION_COLLECTION, offset=offset, limit=PAGE, select=select
        )
        records = res.get("records") or []
        all_records.extend(records)
        if len(records) < PAGE:
            break
        offset += PAGE
    logger.info(f"[fetch] {SPEECH_EVALUATION_COLLECTION} 共 {len(all_records)} 条")
    return all_records


async def run(args: argparse.Namespace) -> int:
    db = CloudBaseNoSQLClient()
    records = await fetch_all(db)

    to_fill, already, skipped, anomalies = build_plan(records, limit=args.limit)
    logger.info(
        f"[plan] 总数={len(records)} 已是新口径={already} 可回填={len(to_fill)} "
        f"raw/parsed 不可判定={skipped} 异常(expected<stored，未动)={anomalies}"
    )
    if to_fill:
        doc_id, stored, new = to_fill[0]
        logger.info(f"[plan] 示例：_id={doc_id} completion {stored} → {new}")
    if args.limit is not None:
        logger.info(f"[plan] --limit={args.limit}（金丝雀）")

    if not args.commit:
        logger.info("[dry-run] 未写入。确认后加 --commit 执行。")
        return 0

    by_id = {str(r.get("_id")): r for r in records}
    done = 0
    for doc_id, _stored, new_completion in to_fill:
        stored_parsed = by_id.get(doc_id, {}).get("parsed") or {}
        # 只替换 completion 一个字段，其余 parsed 字段原样保留（严格范围）；
        # 不写 updated_at —— data-model-contract §4.9 未定义该字段，避免为一次性回填扩 schema。
        # 已回填行可由「parsed.completion == raw.PronCompletion × 100」识别（契约 §4.9 同款表述）。
        new_parsed = {**stored_parsed, "completion": new_completion}
        await db.update(
            SPEECH_EVALUATION_COLLECTION,
            where={"_id": doc_id},
            data={"$set": {"parsed": new_parsed}},
        )
        done += 1
    logger.info(f"[commit] 已回填 {done} 条 speech_evaluation.parsed.completion（0~1 → 0~100）")
    return 0


def main() -> None:
    args = parse_args()
    try:
        code = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - 顶层兜底
        logger.error(f"运行异常: {exc!r}", exc_info=True)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
