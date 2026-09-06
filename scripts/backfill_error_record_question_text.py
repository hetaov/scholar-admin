#!/usr/bin/env python3
"""存量 error_record.question_text（题干）回填脚本。

背景：
- question_text 落库由 2026-09-05（commit 86c0c9e，B1）引入；此前
  （F4 早期版本）写入的 error_record 均无 question_text 字段。
- 新版 classify 落库时 math_scan_upload.classify_result 逐题带
  question_text + error_record_id；correct 也会把题干回填进 classify_result。
- 旧版 manual correct 只把题干存在 error_record.raw_text_corrected（legacy 字段）。

本脚本对「缺 question_text 字段」的 error_record 按两路取源回填：
  A) 其来源 scan 的 classify_result 对应项 question_text（新链路留底）；
  B) 记录自身 legacy raw_text_corrected（旧版人工修正留底）。
两路皆无（老 auto 记录，OCR 前未留题干）→ 跳过并计数，
需 force_reclassify / 重新拍照才可补题干。

用法：
  python scripts/backfill_error_record_question_text.py           # dry-run（默认）
  python scripts/backfill_error_record_question_text.py --commit  # 真实写入

退出码：0 = 执行完成；1 = 运行异常（含无法回填的跳过项，见输出汇总）。
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
logger = logging.getLogger("backfill_error_record_question_text")

from services.database import (  # noqa: E402
    CloudBaseNoSQLClient,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)

PAGE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="回填存量 error_record.question_text（题干）")
    p.add_argument(
        "--commit",
        action="store_true",
        help="真实写入 DB（默认 dry-run，只打印计划）",
    )
    return p.parse_args()


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


def build_scan_source(scans: list[dict]) -> dict[str, str]:
    """从 math_scan_upload.classify_result 建 error_record_id → question_text 映射。

    classify_result 项结构（新版 classify / correct 回填后）：
      {error_record_id, knowledge_point_name, error_type, question_text, ...}
    """
    mapping: dict[str, str] = {}
    for scan in scans:
        for item in scan.get("classify_result") or []:
            rid = (item.get("error_record_id") or "").strip()
            qt = (item.get("question_text") or "").strip()
            if rid and qt:
                mapping[rid] = qt
    logger.info(f"[source-A] classify_result 带题干映射 {len(mapping)} 条 record_id")
    return mapping


def pick_src(
    rec: dict, scan_source: dict[str, str]
) -> str:
    """回填源：A) 来源 scan 的 classify_result 题干；B) 自身 legacy raw_text_corrected。"""
    rid = (rec.get("record_id") or "").strip()
    src = scan_source.get(rid, "")
    if not src:
        src = (rec.get("raw_text_corrected") or "").strip()
    return src


async def run(args: argparse.Namespace) -> int:
    db = CloudBaseNoSQLClient()

    scans = await fetch_all(
        db, MATH_SCAN_UPLOAD_COLLECTION, {"scan_id": 1, "classify_result": 1}
    )
    scan_source = build_scan_source(scans)

    records = await fetch_all(
        db,
        ERROR_RECORD_COLLECTION,
        {
            "record_id": 1,
            "scan_upload_id": 1,
            "question_text": 1,
            "raw_text_corrected": 1,
        },
    )

    to_fill: list[tuple[str, str]] = []  # (record_id, 题干)
    skipped: list[str] = []
    already: int = 0
    for rec in records:
        rid = (rec.get("record_id") or "").strip()
        if not rid:
            continue
        if (rec.get("question_text") or "").strip():
            already += 1
            continue
        src = pick_src(rec, scan_source)
        if src:
            to_fill.append((rid, src))
        else:
            skipped.append(rid)

    logger.info(
        f"[plan] error_record 总数={len(records)} 已有题干={already} "
        f"可回填={len(to_fill)} 无法回填(需重判/重拍)={len(skipped)}"
    )

    if to_fill:
        sample = to_fill[0][1][:50]
        logger.info(
            f"[plan] 示例：record_id={to_fill[0][0]} 题干前50字={sample!r}"
        )
    if skipped:
        logger.warning(
            f"[plan] {len(skipped)} 条无题干且无回填源（旧 auto 记录）——"
            f"需 force_reclassify 或重新拍照。例：{skipped[:3]}"
        )

    if not args.commit:
        logger.info("[dry-run] 未写入。确认后加 --commit 执行。")
        return 0

    done = 0
    for rid, qt in to_fill:
        await db.update(
            ERROR_RECORD_COLLECTION,
            where={"record_id": rid},
            data={"$set": {"question_text": qt}},
        )
        done += 1
    logger.info(f"[commit] 已回填 {done} 条 error_record.question_text")

    return 0 if not skipped else 1


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
