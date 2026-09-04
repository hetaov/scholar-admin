"""C 区（MIXED）强幽灵指纹清理修复脚本：默认 dry-run 出清单，--apply 删 skill_state

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§10.5 结论 2-C + 2026-09-04 细扫复盘）：
  - C 区 = MIXED 桶（幽灵与真实证据共存）：950208…806 × 新概念（真实活跃学者，L1/L2 mastered、
    L3/L4/L50 真实访问、3 sessions 2817s）与 6d758f…089 × 三上广州版（L2 2 mastered）。
  - 禁止整书/整课删除；仅清理「强幽灵指纹」行 = 同批 created（≥5 行 / 跨度 ≤1h / gap ≤900s）
    + attempt_count=1 + mastery=null + 课无任何强证据 + **created 在脚本日 2026-08-14**。
  - 脚本日门槛排除真实活跃期内形成的同形小簇（细扫发现新概念 L4 08-18 新建 speaking/
    listening/conversation 5 行即属此类，判 off-script-date 保留）。
  - 其余幽灵行默认保留待产品确认（本脚本不降级、不删 scholar_book）。

判据单源：行级分类复用 scan_ghost_c.analyze_c（与细扫脚本同一实现，含课级证据、
簇批判定与脚本日门槛），本脚本仅负责「取 fingerprint 行 → 汇总 → 可选删除」。
study_attempt / study_session 为 append-only 事件日志，一律不动。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载）：
  python scripts/repair_ghost_c.py                          # dry-run 清单
  python scripts/repair_ghost_c.py --json /tmp/c_apply_list.json
  python scripts/repair_ghost_c.py --apply                  # 删 fingerprint 行（幂等）

退出码：0 = 成功（dry-run 或 apply）；仅 DB/参数错误 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_ghost_c as cscan  # noqa: E402  行级分类单源（细扫脚本同实现）
from services.dependencies import get_db  # noqa: E402
from services.models.learning import SKILL_STATE  # noqa: E402

_DEL_CHUNK = 100


def _fmt_scholar(sid: str) -> str:
    return sid[:12]


async def plan_cleanup(db, *, min_session_sec: int, result_mastery_threshold: float,
                       batch_gap_sec: int, script_date: str) -> dict:
    """实时拉取 + 行级分类，产出 C 区指纹删除计划（只读，与细扫同判据）。"""
    plan = await cscan.analyze_c(
        db,
        min_session_sec=min_session_sec,
        result_mastery_threshold=result_mastery_threshold,
        batch_gap_sec=batch_gap_sec,
        script_date=script_date,
    )
    fp = [r for r in plan["rows"] if r["reason"] == cscan.R_FINGERPRINT]
    fp_by_book: Counter = Counter()
    for r in fp:
        fp_by_book[(r["scholar_id"], r["lesson_label"].split(" ")[0])] += 1
    return {
        "meta": plan["meta"],
        "fingerprint": fp,
        "reason_totals": plan["reason_totals"],
        "fp_by_book": dict(fp_by_book),
    }


async def apply_deletes(db, fp_rows: list[dict]) -> dict:
    """按 _id 分块删除指纹行。返回删除统计。"""
    ids = sorted({(r.get("_id") or r.get("state_id") or "") for r in fp_rows if
                  (r.get("_id") or r.get("state_id"))})
    total = 0
    report: list[dict] = []
    for start in range(0, len(ids), _DEL_CHUNK):
        chunk = ids[start:start + _DEL_CHUNK]
        res = await db.delete(SKILL_STATE, where={"_id": {"$in": chunk}}, multi=True)
        n = res.get("deleted_count", 0)
        total += n
        report.append({"chunk_start": start, "ids": len(chunk), "deleted": n})
    return {"total": total, "ids": len(ids), "chunks": report}


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="C 区（MIXED）强幽灵指纹清理（默认 dry-run 出清单）"
    )
    parser.add_argument("--min-session-sec", type=int, default=cscan.scan.DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float,
                        default=cscan.scan.DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--batch-gap-sec", type=int, default=cscan.DEFAULT_BATCH_GAP_SEC)
    parser.add_argument("--script-date", default=cscan.DEFAULT_SCRIPT_DATE)
    parser.add_argument("--json", default="", help="可选：输出删除清单 JSON（含行级明细）")
    parser.add_argument("--apply", action="store_true",
                        help="写库：删除 fingerprint 行；缺省仅 dry-run")
    args = parser.parse_args()

    db = get_db()
    plan = await plan_cleanup(
        db,
        min_session_sec=args.min_session_sec,
        result_mastery_threshold=args.result_mastery_threshold,
        batch_gap_sec=args.batch_gap_sec,
        script_date=args.script_date,
    )
    fp = plan["fingerprint"]
    rt = plan["reason_totals"]
    mode = "WRITE" if args.apply else "DRY-RUN"
    out: list[str] = []
    P = out.append

    P("=" * 96)
    P(f"C 区（MIXED）强幽灵指纹清理（{mode}）| 判据同 scan_ghost_c.analyze_c")
    P(f"指纹 = 同批 created(≥5行/≤1h/gap≤{args.batch_gap_sec}s) + attempt=1 + mastery=null "
      f"+ 课无强证据 + created 在 {args.script_date}")
    P("=" * 96)
    P(f"\n[候选] fingerprint 行 {len(fp)}")
    by_key: Counter = Counter()
    for r in fp:
        by_key[(r["scholar_id"][:12], r["lesson_label"].split(" ")[0])] += 1
    last_key = None
    for r in sorted(fp, key=lambda x: (x["scholar_id"], x["lesson_label"], x["_id"] or "")):
        k = (r["scholar_id"][:12], r["lesson_label"].split(" ")[0])
        if k != last_key:
            P(f"  -- {k[0]} {k[1]}（{by_key[k]} 行）--")
            last_key = k
    if not fp:
        P("  无（无当前 fingerprint 行 → 已幂等归零）")

    keep_n = sum(v for k, v in rt.items() if k != cscan.R_FINGERPRINT)
    P(f"\n[保留] 其余幽灵 {keep_n} 行: "
      + "  ".join(f"{k} {v}" for k, v in sorted(rt.items()) if k != cscan.R_FINGERPRINT))
    P("  （默认保留待产品确认；如需降级 not_started 另议，本脚本不改状态）")
    P("\n[合计] 删除 fingerprint skill_state "
      + f"{len(fp)} 行；study_attempt/study_session/scholar_book 均不动")
    if not args.apply:
        P("dry-run 未写库；过目清单后加 --apply 执行（脚本实时重判，幂等）")

    if args.json:
        payload = {
            "meta": plan["meta"],
            "reason_totals": rt,
            "fingerprint": fp,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        P(f"\n删除清单 JSON 已输出: {args.json}")

    if args.apply:
        r1 = await apply_deletes(db, fp)
        P(f"\n[WRITE] skill_state 删除 {r1['total']}/{len(fp)} 行"
          + ("（期望全删；不足说明部分行已被并发清理）"
             if r1["total"] < len(fp) else ""))
        for c in r1["chunks"]:
            if c["deleted"] != c["ids"]:
                P(f"  ! chunk@{c['chunk_start']} 期望 {c['ids']} 实际删 {c['deleted']}")

    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main())
