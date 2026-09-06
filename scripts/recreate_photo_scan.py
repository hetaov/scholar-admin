#!/usr/bin/env python3
"""一次性 helper：删除 question_text 落库前的旧拍照 scan 链路并重建（幂等，安全跳过已修复数据）。

背景：
- error_record.question_text 落库由 2026-09-05 16:11（commit 86c0c9e，B1）引入；
  此前旧 Judge/旧写库产生的链路 doc 的 classify_result 与 error_record
  均无题干留底，原样重跑链脚本会因「upload image_hash 幂等 + classify
  终态幂等」而空跑（见 scholar-skill/tmp_media 的两次产物与本次会话结论）。
- 本脚本针对学者测试数据中两条旧链路（同图 ct_0260905102639_15_35.png）：
    scan_1788577070114_43537a0d  (10:58, needs_review, 12 条 error_record)
    scan_1788588396308_0b02b182  (14:07, success,      11 条 error_record)
  流程：摸底(dry-run，只读) → --commit 校验删除 scan+关联 error_record →
        重建（--run-rebuild 自动调用今天的 math_wrong_photo_chain_verify.py）。

重要防护（幂等）：
- 现状核对（2026-09-05 21:22）：
    scan_1788577070114 已不在 DB；
    scan_1788588396308 关联 11 条 error_record 已重建为带 question_text 的新记录。
  因此当前 dry-run 删除计划 = 0。若目标 scan 的关联记录已全部带题干
  （state=fresh），脚本一律跳过不删，避免误伤已修复数据；
  仅 state=stale（存在缺题干记录）才纳入删除计划。
  本脚本同时可作为任何缺题干旧链路的通用核查工具（改 TARGET_SCANS 即用）。

用法：
  python scripts/recreate_photo_scan.py                          # dry-run 摸底
  python scripts/recreate_photo_scan.py --commit                 # 删除后打印重建命令
  python scripts/recreate_photo_scan.py --commit --run-rebuild   # 删除后自动跑今天的链脚本

前置：
- TENCENTCLOUD_SECRETID / TENCENTCLOUD_SECRETKEY 环境变量（同其它 DB 脚本）；
- --run-rebuild 还需本地后端 127.0.0.1:8080 在跑（chain_verify phase0 会自检）。

退出码：0 = 完成；1 = 运行异常或校验失败（dry-run 不删除任何数据）。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("recreate_photo_scan")

from services.database import (  # noqa: E402
    CloudBaseNoSQLClient,
    ERROR_RECORD_COLLECTION,
    MATH_SCAN_UPLOAD_COLLECTION,
)

# 目标：09-05 question_text 落库前的两条旧链路（学者测试数据）
TARGET_SCANS = [
    {"scan_id": "scan_1788577070114_43537a0d",
     "label": "10:58 needs_review（12 条 error_record）"},
    {"scan_id": "scan_1788588396308_0b02b182",
     "label": "14:07 success（11 条 error_record）"},
]
DEFAULT_SCHOLAR_ID = "6d758f346a6daee000859c332ed11089"
DEFAULT_TEXTBOOK_ID = "tb_math_五年级_up_70963119"
DEFAULT_IMAGE = "scholar-skill/tmp_media/ct_0260905102639_15_35.png"  # 相对仓库根
CHAIN_SCRIPT = "scripts/math_wrong_photo_chain_verify.py"
DEFAULT_OUTPUT = "/tmp/scan_result_v2.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="删除旧 schema 的拍照 scan 链路（scan+error_record）并重建"
    )
    p.add_argument("--commit", action="store_true",
                   help="真实删除（默认 dry-run，只摸底不删）")
    p.add_argument("--run-rebuild", action="store_true",
                   help="删除后自动运行今天的链脚本重建（需本地后端在跑）")
    p.add_argument("--image", default=DEFAULT_IMAGE, help="错题照片路径（重建用）")
    p.add_argument("--scholar-id", default=DEFAULT_SCHOLAR_ID)
    p.add_argument("--textbook-id", default=DEFAULT_TEXTBOOK_ID)
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="重建产物 JSON 路径")
    return p.parse_args()


def _fmt_ms(ms: Any) -> str:
    """毫秒时间戳 → 可读时间（容错：None/字符串原样）。"""
    try:
        return datetime.fromtimestamp(int(ms) / 1000).strftime("%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return str(ms or "")


async def fetch_scan(db: CloudBaseNoSQLClient, scan_id: str) -> dict | None:
    res = await db.query(
        MATH_SCAN_UPLOAD_COLLECTION,
        where={"scan_id": scan_id},
        limit=1,
    )
    records = res.get("records") or []
    return records[0] if records else None


async def fetch_records(db: CloudBaseNoSQLClient, scan_id: str) -> list[dict]:
    res = await db.query(
        ERROR_RECORD_COLLECTION,
        where={"scan_upload_id": scan_id},
        limit=200,
        select={"record_id": 1, "question_text": 1, "created_at": 1},
    )
    return res.get("records") or []


async def inspect_target(
    db: CloudBaseNoSQLClient, t: dict
) -> dict:
    """判定单个目标的现状：absent（已删）/ fresh（已带题干，勿删）/ stale（旧 schema，需删）。"""
    scan_id = t["scan_id"]
    scan = await fetch_scan(db, scan_id)
    if not scan:
        return {"target": t, "scan": None, "records": [], "state": "absent"}
    records = await fetch_records(db, scan_id)
    no_qt = [r for r in records if not (r.get("question_text") or "").strip()]
    # 已带题干 = 该链路已用新代码重建，删除会误伤 → 防护跳过
    state = "fresh" if records and not no_qt else "stale"
    return {
        "target": t, "scan": scan, "records": records,
        "state": state, "no_qt": len(no_qt),
    }


async def scan_dry(db: CloudBaseNoSQLClient) -> tuple[int, int]:
    """摸底：目标 scan 现状 + 关联 error_record + 同 image_hash 其它 scan。"""
    total_scan = total_records = 0
    for t in TARGET_SCANS:
        info = await inspect_target(db, t)
        scan = info["scan"]
        records = info["records"]
        if info["state"] == "absent":
            logger.warning("[plan] %s 在 DB 中不存在（可能已删）——%s",
                           t["scan_id"], t["label"])
            continue
        no_qt = info["no_qt"]
        if info["state"] == "fresh":
            logger.info(
                "[plan] scan=%s %s 已带题干（新 schema），跳过不删 | classify=%s "
                "created=%s | error_record=%d（缺 question_text=0）",
                t["scan_id"], t["label"], scan.get("classify_status"),
                _fmt_ms(scan.get("created_at")), len(records),
            )
            continue
        total_scan += 1
        total_records += len(records)
        logger.info(
            "[plan] scan=%s %s | classify=%s created=%s image_hash=%s | "
            "关联 error_record=%d（缺 question_text=%d）",
            t["scan_id"], t["label"], scan.get("classify_status"),
            _fmt_ms(scan.get("created_at")), str(scan.get("image_hash"))[:12],
            len(records), no_qt,
        )
        for r in records[:3]:
            logger.info("        record=%s created=%s", r.get("record_id"),
                        _fmt_ms(r.get("created_at")))
        if len(records) > 3:
            logger.info("        … 共 %d 条（其余省略）", len(records))

        # 同图其它 scan：确认删除目标范围不会被漏掉
        image_hash = scan.get("image_hash") or ""
        if image_hash:
            res = await db.query(
                MATH_SCAN_UPLOAD_COLLECTION,
                where={"image_hash": image_hash},
                limit=50,
                select={"scan_id": 1, "classify_status": 1, "created_at": 1},
            )
            same = [s for s in (res.get("records") or [])
                    if s.get("scan_id") != t["scan_id"]]
            if same:
                logger.warning(
                    "[plan] 同 image_hash 下还有 %d 条 scan（不在删除清单）: %s",
                    len(same),
                    [f"{s.get('scan_id')}({s.get('classify_status')})" for s in same],
                )
    logger.info("[plan] 合计将删除 scan=%d 条、error_record=%d 条",
                total_scan, total_records)
    return total_scan, total_records


async def run_delete(db: CloudBaseNoSQLClient) -> tuple[int, int]:
    """删除 state=stale 的目标 scan + 关联 error_record（先子后父，跳过已带题干者）。"""
    total_scan = total_records = 0
    for t in TARGET_SCANS:
        info = await inspect_target(db, t)
        scan_id = t["scan_id"]
        if info["state"] == "absent":
            logger.warning("[commit] scan=%s 已不存在，跳过", scan_id)
            continue
        if info["state"] == "fresh":
            logger.info("[commit] scan=%s 记录已带题干（新 schema），跳过不删",
                        scan_id)
            continue
        d_records = await db.delete(
            ERROR_RECORD_COLLECTION, {"scan_upload_id": scan_id}, multi=True
        )
        d_scan = await db.delete(
            MATH_SCAN_UPLOAD_COLLECTION, {"scan_id": scan_id}, multi=True
        )
        n_rec = int(d_records.get("deleted_count", 0) or 0)
        n_scan = int(d_scan.get("deleted_count", 0) or 0)
        total_records += n_rec
        total_scan += n_scan
        logger.info("[commit] scan=%s 已删 scan=%d error_record=%d（%s）",
                    scan_id, n_scan, n_rec, t["label"])
        if n_scan == 0 or n_rec == 0:
            logger.warning("[commit] scan=%s 删除数异常：scan=%d records=%d",
                           scan_id, n_scan, n_rec)
    logger.info("[commit] 删除完成：scan=%d error_record=%d", total_scan, total_records)
    return total_scan, total_records


def rebuild_cmd(args: argparse.Namespace) -> list[str]:
    image = Path(args.image)
    if not image.is_absolute():
        image = HERE / args.image
    return [
        sys.executable, CHAIN_SCRIPT,
        "--image", str(image),
        "--scholar-id", args.scholar_id,
        "--textbook-id", args.textbook_id,
        "--output", args.output,
    ]


async def run(args: argparse.Namespace) -> int:
    db = CloudBaseNoSQLClient()

    if not args.commit:
        await scan_dry(db)
        logger.info("[dry-run] 未删除任何数据。确认计划后加 --commit 执行删除；"
                    "再加 --run-rebuild 自动重建。")
        return 0

    await scan_dry(db)
    await run_delete(db)

    cmd = rebuild_cmd(args)
    logger.info("[rebuild] 已删除。重新生成题干链路（下次执行或 --run-rebuild）：")
    logger.info("    cd %s", HERE)
    logger.info("    %s", " \\\n    ".join(cmd))

    if args.run_rebuild:
        logger.info("[rebuild] 自动执行：%s", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(HERE), check=False)
        if proc.returncode != 0:
            logger.error("[rebuild] 链脚本退出码=%d —— 见上方 FAIL 明细（后端未起？）",
                         proc.returncode)
            return 1
        logger.info("[rebuild] 链脚本全部通过。建议到 error-stats / 云 DB 抽查新记录已带 question_text")
    return 0


def main() -> None:
    args = parse_args()
    if args.run_rebuild and not args.commit:
        logger.error("--run-rebuild 需要先 --commit（先删除旧链路再重建）")
        sys.exit(1)
    try:
        code = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - 顶层兜底
        logger.error("运行异常: %r", exc, exc_info=True)
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
