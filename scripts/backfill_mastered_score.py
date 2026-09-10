#!/usr/bin/env python3
"""存量 skill_state.mastery_score（消灭语义保底）回填脚本（短板消灭战 D5 · P1）。

背景：
- 消灭写侧历史上只写 status='mastered'、不带 score；后端 upsert 无分时保留旧低分，
  导致「已掌握」句的 mastery_score 长期偏低（avgMastery / 难度档 / 复习间隔失真）。
- P0 已让新写入正确（评估路径带真实分 / 自评由后端 MASTERED_SCORE_FLOOR 兜底）；
  本脚本一次性回填存量「status='mastered' 且 mastery_score 缺失或 < FLOOR」的句。

范围（严格，勿扩大）：
- 仅 status='mastered'；不动 learning / learned（不变量 I3）。
- 仅 mastery_score is None or < MASTERED_SCORE_FLOOR（默认 80）。
- 幂等：重跑时这些句已 ≥ FLOOR → 0 变更。
- 重算 progress / next_review_at 复用 derive_progress / compute_next_review_at，
  不写第二套公式（与写侧同源）。

用法：
  python scripts/backfill_mastered_score.py                          # dry-run（默认）
  python scripts/backfill_mastered_score.py --commit                 # 真实写入
  python scripts/backfill_mastered_score.py --scholar s1 --scholar s2  # 按学者灰度（Q4）

退出码：0 = 执行完成；1 = 运行异常。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("backfill_mastered_score")

from services.database import CloudBaseNoSQLClient  # noqa: E402
from services.models_learning import (  # noqa: E402
    MASTERED_SCORE_FLOOR,
    SKILL_STATE,
    STATUS_MASTERED,
    compute_next_review_at,
    derive_progress,
    normalize_status,
)

PAGE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="回填存量 skill_state：status=mastered 且 mastery_score<FLOOR → FLOOR"
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="真实写入 DB（默认 dry-run，只打印计划）",
    )
    p.add_argument(
        "--scholar",
        action="append",
        default=None,
        metavar="SCHOLAR_ID",
        help="仅回填指定学者（可重复，灰度用）；缺省=全部",
    )
    return p.parse_args()


def build_plan(
    records: Iterable[dict],
    *,
    floor: float = MASTERED_SCORE_FLOOR,
    scholars: Iterable[str] | None = None,
    now: int | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], int, int]:
    """纯函数：生成 (state_id → $set 字段) 回填计划。

    规则：
    - 仅 status 归一后 == 'mastered' 的句；scholars 非空时再按 scholar_id 过滤（灰度）。
    - 仅 mastery_score 缺失或 < floor 的句进计划；已达标的计入 already。
    - 无法定位主键（_id/state_id 均缺）计入 skipped。

    返回 (to_fill, already, skipped)：
    - to_fill: list[(state_id, {"mastery_score","progress","next_review_at","updated_at"})]
      —— 值即 `$set` 内容，与写侧同源，避免第二套公式。
    """
    ts = int(now if now is not None else time.time())
    allowed = set(scholars) if scholars else None
    to_fill: list[tuple[str, dict[str, Any]]] = []
    already = 0
    skipped = 0

    for doc in records:
        if normalize_status(doc.get("status")) != STATUS_MASTERED:
            continue
        if allowed is not None and doc.get("scholar_id") not in allowed:
            continue

        state_id = doc.get("state_id") or doc.get("_id")
        if not state_id:
            skipped += 1
            continue

        raw = doc.get("mastery_score")
        try:
            cur = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            cur = None
        if cur is not None and cur >= floor:
            already += 1
            continue

        new_score = max(cur or 0.0, float(floor))
        last_studied_at = doc.get("last_studied_at") or ts
        attempt_count = doc.get("attempt_count") or 1
        to_fill.append(
            (
                state_id,
                {
                    "mastery_score": new_score,
                    "progress": derive_progress(STATUS_MASTERED, new_score),
                    "next_review_at": compute_next_review_at(
                        last_studied_at, attempt_count, new_score
                    ),
                    "updated_at": ts,
                },
            )
        )
    return to_fill, already, skipped


async def fetch_mastered(db: CloudBaseNoSQLClient) -> list[dict]:
    """分页拉取 status='mastered' 全量（find limit 上限按页循环）。"""
    all_records: list[dict] = []
    offset = 0
    where = {"status": STATUS_MASTERED}
    select = {
        "state_id": 1,
        "scholar_id": 1,
        "status": 1,
        "mastery_score": 1,
        "last_studied_at": 1,
        "attempt_count": 1,
    }
    while True:
        res = await db.query(
            SKILL_STATE, where=where, offset=offset, limit=PAGE, select=select
        )
        records = res.get("records") or []
        all_records.extend(records)
        if len(records) < PAGE:
            break
        offset += PAGE
    logger.info(f"[fetch] skill_state(status=mastered) 共 {len(all_records)} 条")
    return all_records


async def run(args: argparse.Namespace) -> int:
    db = CloudBaseNoSQLClient()
    records = await fetch_mastered(db)

    to_fill, already, skipped = build_plan(records, scholars=args.scholar)
    logger.info(
        f"[plan] mastered 总数={len(records)} 已达标={already} "
        f"可回填={len(to_fill)} 缺主键跳过={skipped} "
        f"FLOOR={MASTERED_SCORE_FLOOR}"
    )
    if args.scholar:
        logger.info(f"[plan] 灰度学者={args.scholar}")
    if to_fill:
        sid, fields = to_fill[0]
        logger.info(
            f"[plan] 示例：state_id={sid} → mastery_score={fields['mastery_score']}"
        )

    if not args.commit:
        logger.info("[dry-run] 未写入。确认后加 --commit 执行。")
        return 0

    done = 0
    for state_id, fields in to_fill:
        await db.update(SKILL_STATE, where={"state_id": state_id}, data={"$set": fields})
        done += 1
    logger.info(f"[commit] 已回填 {done} 条 skill_state.mastery_score≥{MASTERED_SCORE_FLOOR}")
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
