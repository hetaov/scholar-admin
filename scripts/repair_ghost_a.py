"""A 区幽灵批量清理修复脚本：GHOST_FULL 整书 + 孤儿幽灵行（默认 dry-run 出清单）

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§10.5 修复分区 A/B）：
  - A 区：全量扫描判为 GHOST_FULL 的（学者×教材）整书（193 行 / 3 本 / 2 学者，
    全部为 8-14 凌晨脚本性批量写入的 learning 行、无任何强证据、scholar_book learning 无锚）
    → 删除该桶全部 skill_state 行，scholar_book 记录一并删除/重置（冷启动）。
  - B 区（并入 A 处置批次）：孤儿 skill_state 行（引用的 lesson/sentence 已不在当前目录），
    仅删「幽灵候选 + 无带分 attempt」子集（76 learning），保留非幽灵行（e2e learned 等）。

判据同源：复用 scan_ghost_distribution.collect_all / classify_book / resolve_textbook，
不重复实现任何幽灵判据。

安全设计：
  - 默认 dry-run：只拉数据、分类、出删除清单（文本 + --json 明细），不写库。
  - --apply：先重新 collect_all 实时判定（非读旧 JSON），仅删除当前仍判为
    GHOST_FULL 桶的 skill_state 行 + 孤儿幽灵行；天然幂等（删后重跑清单归零）。
  - --apply-books：额外删除 A 区桶对应 scholar_book（_id = {scholar}_{textbook}）。
    缺省不删——先由 dry-run 人工确认 scholar_book 无真实时长/锚点后再执行。
  - study_attempt / study_session 为 append-only 事件日志，本脚本一律不动。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载）：
  python scripts/repair_ghost_a.py                      # dry-run 清单
  python scripts/repair_ghost_a.py --json /tmp/a_list.json
  python scripts/repair_ghost_a.py --apply              # 删 skill_state（A 桶 + 孤儿幽灵）
  python scripts/repair_ghost_a.py --apply --apply-books   # 一并删 A 区 scholar_book

退出码：0 = 成功（dry-run 或 apply 均完成）；参数/DB 错误 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_ghost_distribution as scan  # noqa: E402  判据/聚合单源
from services.dependencies import get_db  # noqa: E402
from services.models.learning import (  # noqa: E402
    SKILL_STATE,
    normalize_status,
)
from services.models.scholar_book import (  # noqa: E402
    SCHOLAR_BOOK,
    scholar_book_id,
)

GHOST_CANDIDATE_STATUSES = scan.GHOST_CANDIDATE_STATUSES

_DEL_CHUNK = 100  # delete 按 _id $in 分块，避免单命令过大


def _ts(v_raw) -> str:
    return scan._ts(v_raw)


def _fmt_scholar(sid: str) -> str:
    return sid[:12]


def attempt_has_score(a: dict) -> bool:
    """带分 attempt（强证据）：score / mastery 任一非空。"""
    return a.get("score") is not None or a.get("mastery") is not None


def summarize_rows(rows: list[dict]) -> dict:
    """行清单的状态/时间摘要（用于报告）。"""
    by_status: Counter = Counter()
    span_min, span_max = None, None
    for r in rows:
        by_status[normalize_status(r.get("status"))] += 1
        v = scan._sec(r.get("created_at") or r.get("updated_at"))
        if v > 0:
            span_min = v if span_min is None else min(span_min, v)
            span_max = v if span_max is None else max(span_max, v)
    return {
        "rows": len(rows),
        "by_status": dict(by_status),
        "span_sec": (span_max - span_min if span_min is not None and span_max is not None else None),
        "span_hms": (scan._span_hms(span_max - span_min)
                     if span_min is not None and span_max is not None else "-"),
    }


def build_row_item(r: dict, *, reason: str) -> dict:
    """行清单条目（JSON 用，含删除所需主键与判据信息）。"""
    return {
        "_id": r.get("_id"),
        "state_id": r.get("state_id") or r.get("_id"),
        "scholar_id": r.get("scholar_id"),
        "lesson_id": r.get("lesson_id"),
        "sentence_id": r.get("sentence_id"),
        "skill_code": r.get("skill_code"),
        "status": normalize_status(r.get("status")),
        "attempt_count": r.get("attempt_count"),
        "mastery_score": r.get("mastery_score"),
        "progress": r.get("progress"),
        "created_at": _ts(r.get("created_at")),
        "updated_at": _ts(r.get("updated_at")),
        "reason": reason,
    }


async def plan_cleanup(db, *, min_session_sec: int, result_mastery_threshold: float) -> dict:
    """实时拉取 + 分类，产出 A 区（GHOST_FULL 桶行 + 孤儿幽灵行）清理计划（只读）。"""
    data = await scan.collect_all(
        db,
        min_session_sec=min_session_sec,
        result_mastery_threshold=result_mastery_threshold,
    )
    states = data["states"]
    attempts = data["attempts"]
    lesson_map = data["lesson_map"]
    sentence_map = data["sentence_map"]
    buckets = data["buckets"]
    orphan_rows = data["orphan_rows"]
    book_by_scholar_tb = data["book_by_scholar_tb"]
    title_by_tid = data["title_by_tid"]

    # skill_state 行 → (学者, 教材) 归桶（仅用于选 A 区行；孤儿单独）
    rows_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in states:
        sid = s.get("scholar_id")
        if not sid:
            continue
        tb = scan.resolve_textbook(s, lesson_map, sentence_map)
        if tb == "__orphan__":
            continue
        rows_by_key[(sid, tb)].append(s)

    # A 区：当前仍判 GHOST_FULL 的桶（含桶级尝试/会话残留说明）
    ghost_buckets = [
        c for c in buckets.values()
        if scan.classify_book(c) == "GHOST_FULL"
    ]
    a_rows: list[dict] = []
    bucket_meta: list[dict] = []
    for c in sorted(ghost_buckets, key=lambda x: (-x["ghost_rows"], x["scholar_id"])):
        key = (c["scholar_id"], c["textbook_id"])
        rows = sorted(rows_by_key.get(key, []), key=lambda r: r.get("_id") or "")
        a_rows.extend(build_row_item(r, reason="GHOST_FULL 整书无强证据（8-14 批量幽灵）") for r in rows)
        book = book_by_scholar_tb.get(key)
        bucket_meta.append({
            "scholar_id": c["scholar_id"],
            "textbook_id": c["textbook_id"],
            "textbook_title": title_by_tid.get(c["textbook_id"], c["textbook_id"]),
            "rows": [r["_id"] for r in rows],
            "summary": summarize_rows(rows),
            "attempt_rows_no_score": c["attempt_rows"] - c["scored_attempts"],
            "session_dur_sec": c["session_dur_sec"],
            "scholar_book": book,
            # delete 条件：无 lesson 级断点 + 零真实时长（chapter 级锚为幽灵脚本顺手写入，
            # 不构成真实使用证据；total_time_spent 为唯一真实时长口径）。其余 → review 人工。
            "scholar_book_action": (
                "delete" if (book and not book.get("current_lesson_id")
                             and not (book.get("total_time_spent") or 0))
                else ("review" if book else "none")
            ),
        })

    # B 区（孤儿幽灵行）：幽灵候选 + 其 sentence 无带分 attempt
    scored_sents = {
        (a.get("scholar_id"), a.get("sentence_id"))
        for a in attempts if attempt_has_score(a)
    }
    orphan_ghost: list[dict] = []
    orphan_keep: list[dict] = []
    for o in orphan_rows:
        sid = o.get("scholar_id")
        sent = o.get("sentence_id")
        item = build_row_item(
            o,
            reason="孤儿幽灵（sentence 不在目录 + 幽灵候选 + 无带分 attempt）"
            if ((sid, sent) not in scored_sents)
            else "孤儿但有带分 attempt → 保留复核",
        )
        if normalize_status(o.get("status")) in GHOST_CANDIDATE_STATUSES \
                and (sid, sent) not in scored_sents:
            orphan_ghost.append(item)
        else:
            orphan_keep.append(item)

    by_scholar: Counter = Counter(i["scholar_id"] for i in orphan_ghost)
    return {
        "data": data,
        "ghost_buckets": bucket_meta,
        "a_rows": a_rows,
        "orphan_ghost": orphan_ghost,
        "orphan_keep": orphan_keep,
        "orphan_by_scholar": dict(by_scholar),
        "totals": {
            "a_state_rows": len(a_rows),
            "ghost_bucket_books": len(bucket_meta),
            "books_to_delete": sum(1 for b in bucket_meta if b["scholar_book_action"] == "delete"),
            "books_to_review": sum(1 for b in bucket_meta if b["scholar_book_action"] == "review"),
            "orphan_ghost": len(orphan_ghost),
            "orphan_keep": len(orphan_keep),
            "delete_state_rows": len(a_rows) + len(orphan_ghost),
        },
    }


def _book_action_line(meta: dict) -> str:
    b = meta["scholar_book"]
    act = meta["scholar_book_action"]
    if not b:
        return "  scholar_book: (不存在)"
    spent = b.get("total_time_spent") or 0
    return (
        f"  scholar_book: {act:<6} status={b.get('status') or '-'} "
        f"time_spent={spent}s last={_ts(b.get('last_studied_at'))} "
        f"started={_ts(b.get('started_at'))} "
        f"anchor_lesson={str(b.get('current_lesson_id'))[:16] or '-'} "
        f"anchor_chapter={str(b.get('current_chapter_id'))[:16] or '-'}"
    )


async def apply_deletes(db, plan: dict) -> dict:
    """执行 skill_state 删除（A 区桶行 + 孤儿幽灵行）。返回删除统计。"""
    ids = [i["_id"] for i in plan["a_rows"] + plan["orphan_ghost"] if i.get("_id")]
    # 防御：重复/空 id
    ids = sorted({i for i in ids if i})
    total = 0
    report: list[dict] = []
    for start in range(0, len(ids), _DEL_CHUNK):
        chunk = ids[start:start + _DEL_CHUNK]
        res = await db.delete(SKILL_STATE, where={"_id": {"$in": chunk}}, multi=True)
        n = res.get("deleted_count", 0)
        total += n
        report.append({"chunk_start": start, "ids": len(chunk), "deleted": n})
    return {"total": total, "chunks": report}


async def apply_book_deletes(db, plan: dict) -> dict:
    """删除 scholar_book_action == delete 的 A 区记录。"""
    total = 0
    detail: list[dict] = []
    for meta in plan["ghost_buckets"]:
        if meta["scholar_book_action"] != "delete":
            continue
        _id = scholar_book_id(meta["scholar_id"], meta["textbook_id"])
        res = await db.delete(SCHOLAR_BOOK, where={"_id": _id}, multi=False)
        n = res.get("deleted_count", 0)
        total += n
        detail.append({"_id": _id, "deleted": n})
    return {"total": total, "detail": detail}


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="A 区幽灵批量清理（GHOST_FULL 整书 + 孤儿幽灵行），默认 dry-run"
    )
    parser.add_argument("--min-session-sec", type=int, default=scan.DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float,
                        default=scan.DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--json", default="", help="可选：输出清理清单 JSON")
    parser.add_argument("--apply", action="store_true",
                        help="写库：删除 skill_state（A 区桶行 + 孤儿幽灵行）；缺省仅 dry-run")
    parser.add_argument("--apply-books", action="store_true",
                        help="（需 --apply）一并删除 A 区 scholar_book 记录；缺省不删")
    args = parser.parse_args()

    if args.apply_books and not args.apply:
        parser.error("--apply-books 需配合 --apply 使用")

    db = get_db()
    plan = await plan_cleanup(
        db,
        min_session_sec=args.min_session_sec,
        result_mastery_threshold=args.result_mastery_threshold,
    )
    gb = plan["ghost_buckets"]
    og = plan["orphan_ghost"]
    ok = plan["orphan_keep"]
    t = plan["totals"]
    mode = "WRITE" if args.apply else "DRY-RUN"
    out: list[str] = []
    P = out.append

    P("=" * 96)
    P(f"A 区幽灵批量清理计划（{mode}）| 判据同 scan_ghost_distribution.collect_all")
    P("=" * 96)
    P(f"\n[A1] GHOST_FULL 整书（skill_state 删除 {t['a_state_rows']} 行 / "
      f"{t['ghost_bucket_books']} 本）")
    for meta in gb:
        s = meta["summary"]
        P(f"  {_fmt_scholar(meta['scholar_id']):<14} | {meta['textbook_title'][:24]:<24} "
          f"{meta['textbook_id'][:14]:<14} | 行 {s['rows']} "
          + " ".join(f"{k}={v}" for k, v in sorted(s['by_status'].items()))
          + f" | 跨度 {s['span_hms']}"
          + f" | 无分attempt {meta['attempt_rows_no_score']} | session {meta['session_dur_sec']}s")
        P(_book_action_line(meta))
    if not gb:
        P("  无（无当前 GHOST_FULL 书）")

    P(f"\n[A2] 孤儿幽灵行（skill_state 删除 {len(og)} 行 / "
      + "，".join(f"{_fmt_scholar(k)} {v}" for k, v in sorted(plan['orphan_by_scholar'].items())) + "）")
    if og:
        for i in sorted(og, key=lambda x: (x["scholar_id"], x["_id"] or "")):
            P(f"  {_fmt_scholar(i['scholar_id']):<14} {str(i['_id'] or '')[:52]:<52} "
              f"{i['status']:<10} att={i['attempt_count']} {_ts(i['created_at'])}")
    if ok:
        P(f"  保留（非幽灵 / 有带分 attempt，不动）{len(ok)} 行：")
        for i in sorted(ok, key=lambda x: (x["scholar_id"], x["_id"] or "")):
            P(f"    {_fmt_scholar(i['scholar_id']):<14} {str(i['_id'] or '')[:52]:<52} "
              f"{i['status']:<10} {i['reason']}")

    P("\n[合计] " + f"删除 skill_state {t['delete_state_rows']} 行；"
      f"scholar_book delete {t['books_to_delete']} / review {t['books_to_review']}")
    P("（study_attempt / study_session 为 append-only 事件日志，本脚本不动）")
    if not args.apply:
        P("dry-run 未写库；确认后加 --apply（skill_state），--apply-books（scholar_book）")

    if args.json:
        payload = {
            "meta": {
                "mode": mode,
                "min_session_sec": args.min_session_sec,
                "result_mastery_threshold": args.result_mastery_threshold,
                "n_state": len(plan["data"]["states"]),
                "n_attempt": len(plan["data"]["attempts"]),
                "n_session": len(plan["data"]["sessions"]),
                "n_book": len(plan["data"]["books"]),
            },
            "totals": t,
            "buckets": [
                {
                    "scholar_id": m["scholar_id"],
                    "textbook_id": m["textbook_id"],
                    "textbook_title": m["textbook_title"],
                    "summary": m["summary"],
                    "attempt_rows_no_score": m["attempt_rows_no_score"],
                    "session_dur_sec": m["session_dur_sec"],
                    "scholar_book": m["scholar_book"],
                    "scholar_book_action": m["scholar_book_action"],
                }
                for m in gb
            ],
            "a_rows": plan["a_rows"],
            "orphan_ghost": og,
            "orphan_keep": ok,
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        P(f"\n清单 JSON 已输出: {args.json}")

    if args.apply:
        r1 = await apply_deletes(db, plan)
        P(f"\n[WRITE] skill_state 删除 {r1['total']}/{t['delete_state_rows']} 行"
          + ("（期望全删；不足说明部分行已被并发清理）"
             if r1["total"] < t["delete_state_rows"] else ""))
        for c in r1["chunks"]:
            if c["deleted"] != c["ids"]:
                P(f"  ! chunk@{c['chunk_start']} 期望 {c['ids']} 实际删 {c['deleted']}")
        if args.apply_books:
            r2 = await apply_book_deletes(db, plan)
            P(f"[WRITE] scholar_book 删除 {r2['total']}/{t['books_to_delete']} 条")
            for d in r2["detail"]:
                if d["deleted"] != 1:
                    P(f"  ! {d['_id']} 期望删 1 实际 {d['deleted']}")
        else:
            P(f"[SKIP] 未删 scholar_book（{t['books_to_delete']} 条可删 / "
              f"{t['books_to_review']} 条需复核）；加 --apply-books 执行")

    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main())
