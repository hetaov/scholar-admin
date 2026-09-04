"""C 区残留幽灵行清理 + MIXED 书锚点归一化（默认 dry-run 出清单）

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§10.5 结论 2-C + 2026-09-04 复盘）：
  - C 区 MIXED 两本中，1441 行「强幽灵指纹」（课无强证据）已由 repair_ghost_c.py --apply 清理。
  - 剩 23 行原为保守保留（课内有强证据 → lesson-has-evidence / 真实复学 → row-has-revisit /
    非批簇 / 非脚本日）。逐行 attempt 佐证后分两类：
      * 纯幽灵残留 8 行 = 课内其它行有真实证据，但本行 created 在脚本日 2026-08-14、
        updated==created（自写入从未再动）、att=1、mastery=null、该 scholar 对该句
        零 translate attempt → 幽灵底座，可安全删除（真实学习从未发生）。
      * 其余 15 行均有真实 attempt/复学支撑 → 保留。
  - scholar_book：6d758f…089 × 三上广州版 current_chapter_id=L2（真实活跃课，09-02
    mastered+attempts 佐证），但 current_lesson_id 为空（幽灵脚本顺手写形态）→ 归一化补齐
    lesson 级锚；该书记录本身有真实证据，**不删除**。

判据单源：行级/课级分类复用 scan_ghost_c.analyze_c（与 C 细扫同一实现）；
残留幽灵判定 = 在前者保留行上叠加「从未再动 + 零 translate attempt」双重护栏。
study_attempt / study_session 为 append-only 事件日志，一律不动。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载）：
  python scripts/repair_ghost_c2.py                              # dry-run 清单
  python scripts/repair_ghost_c2.py --json /tmp/c2_list.json
  python scripts/repair_ghost_c2.py --apply                       # 删残留幽灵行
  python scripts/repair_ghost_c2.py --apply --apply-book          # 一并归一化 book 锚点

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

import scan_ghost_c as cscan  # noqa: E402  行级分类单源（C 细扫/修复同实现）
import scan_ghost_distribution as scan  # noqa: E402
from services.dependencies import get_db  # noqa: E402
from services.models.content import query_all_pages  # noqa: E402
from services.models.events import STUDY_ATTEMPT  # noqa: E402
from services.models.learning import SKILL_STATE  # noqa: E402
from services.models.scholar_book import SCHOLAR_BOOK  # noqa: E402

_DEL_CHUNK = 100
_TRANSLATE_TYPES = {"translate", "translation"}


def _fmt_scholar(sid: str) -> str:
    return sid[:12]


def _is_ghost_row(r: dict, translate_sents: set) -> bool:
    """残留幽灵判定：learning + att=1 + mastery=null + created 脚本日 +
    updated==created（从未再动）+ 该句零 translate attempt。"""
    if r.get("status") not in cscan.GHOST_CANDIDATE_STATUSES:
        return False
    if scan._num(r.get("attempt_count")) != 1:
        return False
    if r.get("mastery_score") is not None:
        return False
    if r.get("updated_delta_sec") != 0:  # created 后被动过 → 真实痕迹
        return False
    cd = r.get("created_at") or ""
    if not cd.startswith("2026-08-14"):
        return False
    if (r.get("scholar_id"), r.get("sentence_id")) in translate_sents:
        return False
    return True


def _attempt_ctx(r: dict, attempts: list[dict]) -> str:
    sid, sent = r.get("scholar_id"), r.get("sentence_id")
    c: Counter = Counter()
    for a in attempts:
        if a.get("scholar_id") == sid and a.get("sentence_id") == sent:
            c[str(a.get("attempt_type") or "-")] += 1
    return " ".join(f"{k}={v}" for k, v in sorted(c.items())) if c else "无"


async def plan_c2(db, *, min_session_sec: int, result_mastery_threshold: float,
                  batch_gap_sec: int, script_date: str) -> dict:
    """实时拉取 + 行级分类 + 残留幽灵判定 + book 锚点归一化计划（只读）。"""
    plan = await cscan.analyze_c(
        db,
        min_session_sec=min_session_sec,
        result_mastery_threshold=result_mastery_threshold,
        batch_gap_sec=batch_gap_sec,
        script_date=script_date,
    )
    attempts = await query_all_pages(db, collection=STUDY_ATTEMPT, where={})
    translate_sents = {
        (a.get("scholar_id"), a.get("sentence_id"))
        for a in attempts if (a.get("attempt_type") or "") in _TRANSLATE_TYPES
    }

    del_rows: list[dict] = []
    keep_rows: list[dict] = []
    for r in plan["rows"]:
        item = dict(r)
        item["attempts_on_sentence"] = _attempt_ctx(r, attempts)
        if _is_ghost_row(r, translate_sents):
            del_rows.append(item)
        else:
            keep_rows.append(item)

    # book 锚点归一化：仅补「半锚」——current_lesson_id 为空、但 current_chapter_id
    # 已指向本桶真实课（chapter 级锚由幽灵顺手写但恰为真实活跃课）→ 补齐 lesson=chapter。
    # 已存在的 lesson 锚一律不动（真实断点已表达）；chapter 无效/缺省的留待 L4
    # repair_book_anchor.py 按证据整体重算，本脚本不移动既有锚点。
    book_fixes: list[dict] = []
    for b in plan["buckets"]:
        sb = b.get("scholar_book")
        if not sb:
            continue
        cur_ch, cur_les = sb.get("current_chapter_id"), sb.get("current_lesson_id")
        if cur_les:
            continue  # lesson 锚已存在 → 不动
        bucket_lesson_ids = {l["lesson_id"] for l in plan["lessons"]
                             if l["scholar_id"] == b["scholar_id"]
                             and l["textbook_id"] == b["textbook_id"]
                             and l["lesson_id"]}
        if cur_ch not in bucket_lesson_ids:
            continue  # chapter 锚无效（非本桶课）→ 不猜，留待 L4 重算
        lid_label = next(
            ((f"L{l.get('order')}" if l.get("order") is not None else "L-")
             + (f" {l.get('title') or ''}" if l.get("title") else ""))
            for l in plan["lessons"]
            if l["lesson_id"] == cur_ch
        )
        book_fixes.append({
            "scholar_id": b["scholar_id"],
            "textbook_id": b["textbook_id"],
            "textbook_title": b.get("textbook_title"),
            "book_id": sb.get("_id"),
            "before": {"current_chapter_id": cur_ch, "current_lesson_id": cur_les},
            "after": {"current_chapter_id": cur_ch, "current_lesson_id": cur_ch},
            "target_lesson": cur_ch,
            "target_label": lid_label,
        })

    return {
        "meta": plan["meta"],
        "del_rows": del_rows,
        "keep_rows": keep_rows,
        "book_fixes": book_fixes,
        "reason_counts_del": dict(Counter(r["reason"] for r in del_rows)),
        "reason_counts_keep": dict(Counter(r["reason"] for r in keep_rows)),
    }


async def apply_deletes(db, rows: list[dict]) -> dict:
    ids = sorted({r.get("_id") or r.get("state_id") for r in rows if
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


async def apply_book_fixes(db, fixes: list[dict]) -> dict:
    import time
    now = int(time.time())
    detail: list[dict] = []
    for f in fixes:
        res = await db.update(
            collection=SCHOLAR_BOOK,
            where={"_id": f["book_id"]},
            data={"$set": {
                **f["after"],
                "updated_at": now,
            }},
            multi=False,
        )
        detail.append({**f, "matched": res.get("matched_count", 0)
                       if isinstance(res, dict) else 0})
    return {"total": len(fixes), "detail": detail}


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="C 区残留幽灵行清理 + MIXED book 锚点归一化（默认 dry-run）"
    )
    parser.add_argument("--min-session-sec", type=int, default=scan.DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float,
                        default=scan.DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--batch-gap-sec", type=int, default=cscan.DEFAULT_BATCH_GAP_SEC)
    parser.add_argument("--script-date", default=cscan.DEFAULT_SCRIPT_DATE)
    parser.add_argument("--json", default="", help="可选：输出清单 JSON")
    parser.add_argument("--apply", action="store_true",
                        help="写库：删除残留幽灵行；缺省仅 dry-run")
    parser.add_argument("--apply-book", action="store_true",
                        help="（需 --apply）一并归一化 MIXED book 的 lesson 级锚点")
    args = parser.parse_args()

    if args.apply_book and not args.apply:
        parser.error("--apply-book 需配合 --apply 使用")

    db = get_db()
    plan = await plan_c2(
        db,
        min_session_sec=args.min_session_sec,
        result_mastery_threshold=args.result_mastery_threshold,
        batch_gap_sec=args.batch_gap_sec,
        script_date=args.script_date,
    )
    del_rows = plan["del_rows"]
    keep_rows = plan["keep_rows"]
    fixes = plan["book_fixes"]
    mode = "WRITE" if args.apply else "DRY-RUN"
    out: list[str] = []
    P = out.append

    P("=" * 100)
    P(f"C 区残留幽灵行清理 + MIXED book 锚点归一化（{mode}）")
    P(f"残留幽灵 = learning + att=1 + mastery=null + created {args.script_date} "
      f"+ updated==created(未再动) + 该句零 translate attempt")
    P("=" * 100)

    P(f"\n[删除候选] 残留幽灵行 {len(del_rows)}"
      + (f"（原 reason: " + "  ".join(f"{k} {v}" for k, v in sorted(plan['reason_counts_del'].items()))
         + "）" if del_rows else ""))
    if del_rows:
        P("  " + f"{'scholar':<13}{'lesson':<22}{'skill':<14}{'att':<3}{'created':<18}"
          f"{'sentence(34)':<36}该句 attempts")
        for r in sorted(del_rows, key=lambda x: (x["scholar_id"], x["lesson_label"], x["_id"] or "")):
            P(f"  {_fmt_scholar(r['scholar_id']):<13}"
              f"{str(r['lesson_label'])[:20]:<22}"
              f"{str(r.get('skill_code'))[:12]:<14}"
              f"{str(r.get('attempt_count')):<3}"
              f"{str(r.get('created_at')):<18}"
              f"{str(r.get('sentence_text') or r.get('sentence_id'))[:34]:<36}"
              f"{r.get('attempts_on_sentence')}")
    else:
        P("  无（无残留幽灵 → 幂等归零）")

    P(f"\n[保留] 其余真实行 {len(keep_rows)}"
      + (f"（" + "  ".join(f"{k} {v}" for k, v in sorted(plan['reason_counts_keep'].items())) + "）"
         if keep_rows else ""))
    for r in sorted(keep_rows, key=lambda x: (x["scholar_id"], x["lesson_label"], x["_id"] or "")):
        P(f"  {_fmt_scholar(r['scholar_id']):<13}{str(r['lesson_label'])[:20]:<22}"
          f"{str(r.get('skill_code'))[:12]:<14}att={r.get('attempt_count')} "
          f"{str(r.get('created_at')):<18}{str(r.get('sentence_text') or '')[:34]}")

    P(f"\n[book 锚点归一化] {len(fixes)} 本")
    for f in fixes:
        P(f"  {_fmt_scholar(f['scholar_id'])} × {str(f.get('textbook_title'))[:20]} "
          f"{f['textbook_id'][:14]}")
        P(f"    before: chapter={str(f['before']['current_chapter_id'])[:16] or '-'} "
          f"lesson={str(f['before']['current_lesson_id'])[:16] or '-'}")
        P(f"    after : chapter={str(f['after']['current_chapter_id'])[:16] or '-'} "
          f"lesson={str(f['after']['current_lesson_id'])[:16] or '-'}  "
          f"(证据课 {f['target_label']})")
    if not fixes:
        P("  无（lesson 锚已存在或书内无 result 证据课）")

    P(f"\n[合计] 删除 skill_state {len(del_rows)} 行；book 锚点归一化 {len(fixes)} 本；"
      f"study_attempt/study_session 不动")
    if not args.apply:
        P("dry-run 未写库；过目后加 --apply 执行（--apply-book 一并归一化锚点）")

    if args.json:
        payload = {
            "meta": plan["meta"],
            "del_rows": del_rows,
            "keep_rows": keep_rows,
            "book_fixes": fixes,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        P(f"\n清单 JSON 已输出: {args.json}")

    if args.apply:
        r1 = await apply_deletes(db, del_rows)
        P(f"\n[WRITE] skill_state 删除 {r1['total']}/{len(del_rows)} 行"
          + ("（不足说明部分行已被并发清理）" if r1["total"] < len(del_rows) else ""))
        for c in r1["chunks"]:
            if c["deleted"] != c["ids"]:
                P(f"  ! chunk@{c['chunk_start']} 期望 {c['ids']} 实际删 {c['deleted']}")
        if args.apply_book:
            r2 = await apply_book_fixes(db, fixes)
            P(f"[WRITE] book 锚点归一化 {r2['total']}/{len(fixes)} 本")
            for d in r2["detail"]:
                if d.get("matched") != 1:
                    P(f"  ! {d['book_id']} 期望 matched=1 实际 {d.get('matched')}")
        else:
            P(f"[SKIP] 未改 book 锚点（{len(fixes)} 本待归一化）；加 --apply-book 执行")

    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main())
