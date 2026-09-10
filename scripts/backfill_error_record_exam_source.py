#!/usr/bin/env python3
"""存量 error_record 双源标记回填脚本（M13 双源知识锚 · Phase 2-b）。

背景：
- M13（方案 §3.2 / O3=A）将 EXTRA_AI 虚拟教材机制升级为「真题/课外归集库」：
  真题/图谱外（EXTRA_AI 锚定，textbook_id='EXTRA_AI'）error_record 语义上即
  「真题锚」，读取层按数据事实缺省识别为 kp_source='exam_paper'。
- 本脚本把该派生事实物化为 error_record.kp_source='exam_paper'，供后续
  直接按字段查询/聚合（不依赖 textbook_id 推导）。
- **不移动任何记录**（链锚点 textbook_id/unit_title/lesson_title 零改动）：
  存量 EXTRA_AI 卡（「课外补充 · AI 归类」独立卡 + 未归链兜底）历史视图照常查询。
- exam_backlink_to（真题题簇 → 教材点软回链）由 M13 新版 classify/correct
  新链路落库补齐；存量记录不回填（需 force_reclassify 才可重算），本脚本不做。

用法：
  python scripts/backfill_error_record_exam_source.py           # dry-run（默认）
  python scripts/backfill_error_record_exam_source.py --commit  # 真实写入

退出码：0 = 执行完成；1 = 运行异常。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("backfill_error_record_exam_source")

from services.database import (  # noqa: E402
    CloudBaseNoSQLClient,
    ERROR_RECORD_COLLECTION,
)
from services.math.error_scanner import (  # noqa: E402
    EXTRA_AI_TEXTBOOK_ID,
    KP_SOURCE_EXAM_PAPER,
)

PAGE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="回填存量 EXTRA_AI（真题/图谱外）error_record.kp_source='exam_paper'"
    )
    p.add_argument(
        "--commit",
        action="store_true",
        help="真实写入 DB（默认 dry-run，只打印计划）",
    )
    return p.parse_args()


def build_plan(records: list[dict]) -> tuple[list[tuple[str, str]], int]:
    """纯函数：对缺 kp_source 的 EXTRA_AI 记录生成 (record_id, kp_source) 回填计划。

    - 只看 textbook_id == EXTRA_AI 且 kp_source 缺失/非 exam_paper 的记录；
    - 其余（正式教材链 / 已标记）不进计划（存量零改动）。
    """
    to_fill: list[tuple[str, str]] = []
    already: int = 0
    for rec in records:
        rid = (rec.get("record_id") or "").strip()
        if not rid:
            continue
        if (rec.get("textbook_id") or "") != EXTRA_AI_TEXTBOOK_ID:
            continue
        if rec.get("kp_source") == KP_SOURCE_EXAM_PAPER:
            already += 1
            continue
        to_fill.append((rid, KP_SOURCE_EXAM_PAPER))
    return to_fill, already


async def fetch_all(db: CloudBaseNoSQLClient, collection: str, select: dict) -> list[dict]:
    """分页拉全量（find limit 上限按页循环）。"""
    all_records: list[dict] = []
    offset = 0
    while True:
        res = await db.query(
            collection, where={}, offset=offset, limit=PAGE, select=select
        )
        records = res.get("records") or []
        all_records.extend(records)
        if len(records) < PAGE:
            break
        offset += PAGE
    logger.info(f"[fetch] {collection} 共 {len(all_records)} 条")
    return all_records


async def run(args: argparse.Namespace) -> int:
    db = CloudBaseNoSQLClient()
    records = await fetch_all(
        db,
        ERROR_RECORD_COLLECTION,
        {"record_id": 1, "textbook_id": 1, "kp_source": 1},
    )

    to_fill, already = build_plan(records)
    logger.info(
        f"[plan] error_record 总数={len(records)} EXTRA_AI 已标记={already} "
        f"可回填={len(to_fill)}（链锚点零改动，EXTRA_AI 卡历史视图不受影响）"
    )
    if to_fill:
        logger.info(f"[plan] 示例：record_id={to_fill[0][0]} → kp_source='{to_fill[0][1]}'")

    if not args.commit:
        logger.info("[dry-run] 未写入。确认后加 --commit 执行。")
        return 0

    done = 0
    for rid, kp_source in to_fill:
        await db.update(
            ERROR_RECORD_COLLECTION,
            where={"record_id": rid},
            data={"$set": {"kp_source": kp_source}},
        )
        done += 1
    logger.info(f"[commit] 已回填 {done} 条 error_record.kp_source='{KP_SOURCE_EXAM_PAPER}'")
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
