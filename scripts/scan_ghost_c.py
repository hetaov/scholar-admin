"""C 区（MIXED 幽灵行）行级细扫：只读，出「强幽灵指纹」删除候选清单供人工过目

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§10.5 结论 2-C）：
  - C 区 = MIXED 桶（幽灵候选与真实证据共存，不能整书清），当前 2 本：
      950208…806 × 新概念英语第一册 tb_db3e2209b3cc4e（1395 learning + 50 mastered，
        真实活跃学者，R:50 S:3）
      6d758f…089 × 三年级上 广州版 tb_ed12c746d8aa45（69 learning + 2 mastered，R:2）
  - §10.3 已确认历史 skip 与真实低分在事件层不可区分 → 禁止整书/整课删除。
  - 仅清理「强幽灵指纹」子集 = 同批 created + attempt_count=1 + mastery=null +
    该课无任何强证据行（强证据 = 课内结果性状态行 / mastery≥阈值行 / 该课带分 attempt）。
  - 其余幽灵行默认保留/降级待产品确认（本脚本**不写库**，纯清单输出）。

判据同源：复用 scan_ghost_distribution.collect_all / resolve_textbook / classify_book，
不重复实现任何幽灵判据。study_session 仅教材级粒度 → 作为书级佐证展示，不参与课级指纹。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载；纯只读）：
  python scripts/scan_ghost_c.py                           # 文本汇总
  python scripts/scan_ghost_c.py --json /tmp/c_list.json   # 行级明细（人工过目/备份）
  python scripts/scan_ghost_c.py --full                    # 无幽灵的课也列出
  python scripts/scan_ghost_c.py --batch-gap-sec 300       # created 簇判批间隔（默认 900s）

退出码：0（扫描完成）；仅 DB 连接/参数错误 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scan_ghost_distribution as scan  # noqa: E402  判据/聚合单源
from services.dependencies import get_db  # noqa: E402
from services.models.content import (  # noqa: E402
    LESSON,
    SENTENCE_V2,
)
from services.models.learning import (  # noqa: E402
    normalize_status,
)

GHOST_CANDIDATE_STATUSES = scan.GHOST_CANDIDATE_STATUSES
RESULT_STATUSES = scan.RESULT_STATUSES
BATCH_MIN_ROWS = scan.BATCH_MIN_ROWS
BATCH_MAX_SPAN_SEC = scan.BATCH_MAX_SPAN_SEC

DEFAULT_BATCH_GAP_SEC = 900  # created 簇内相邻行最大间隔（脚本批量逐行毫秒级；真实会话间隔更久）
# 幽灵批量脚本写入日（§10.1/§10.5：8-14 凌晨一次多学者×多教材的脚本性写入）。
# 指纹候选要求 created 落在该日，排除真实活跃期内形成的同形小簇（如新概念 L4 08-18 新建键）。
DEFAULT_SCRIPT_DATE = "2026-08-14"

# 幽灵行去向（reason）分类
R_FINGERPRINT = "fingerprint"            # 强幽灵指纹（脚本日批簇）→ 删除候选（人工过目）
R_OFF_SCRIPT_DATE = "off-script-date"    # 同批簇形但非脚本日（真实活跃期新建）→ 保留
R_LESSON_EVIDENCE = "lesson-has-evidence"  # 课内有强证据行 → 保留
R_ROW_REVISIT = "row-has-revisit"        # 行自身被重复学习/带分 → 保留
R_NON_BATCH = "non-batch"                # 非批簇单行 → 保留待查
R_NO_LESSON = "no-lesson"                # 无法归属课（句在目录但课缺失）→ 保留复核


def _date(v_raw) -> str:
    """时间戳 → UTC 日期（秒/毫秒自适应）；无效返回空串。"""
    v = scan._sec(v_raw)
    if v <= 0:
        return ""
    return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d")


def _fmt_scholar(sid: str) -> str:
    return sid[:12]


def resolve_lesson(row: dict, lesson_map: dict, sentence_map: dict) -> str | None:
    """行归属课 id（与 scan.resolve_textbook 同源优先级：lesson_id → sentence.lesson_id）。"""
    lid = row.get("lesson_id")
    if lid and lid in lesson_map:
        return lid
    sid = row.get("sentence_id")
    if sid:
        doc = sentence_map.get(sid)
        if doc:
            l2 = doc.get("lesson_id")
            if l2 and l2 in lesson_map:
                return l2
    return None


def _row_ts(r: dict) -> float:
    return scan._sec(r.get("created_at") or r.get("updated_at"))


def cluster_ghost_rows(rows: list[dict], gap_sec: int) -> list[dict]:
    """按 created 时间把幽灵行切成簇；返回簇元数据 + 行→簇归属映射。

    相邻行间隔 > gap_sec 切开；无时间戳的行单独成簇（不判批）。
    """
    rows = sorted(rows, key=_row_ts)
    clusters: list[list[dict]] = []
    cur: list[dict] = []
    prev: float | None = None
    for r in rows:
        v = _row_ts(r)
        if v <= 0:
            if cur:
                clusters.append(cur)
                cur = []
            clusters.append([r])
            prev = None
            continue
        if prev is not None and v - prev > gap_sec:
            clusters.append(cur)
            cur = []
        cur.append(r)
        prev = v
    if cur:
        clusters.append(cur)

    out: list[dict] = []
    by_id: dict[int, dict] = {}
    for ci, cl in enumerate(clusters):
        times = [scan._sec(r.get("created_at") or r.get("updated_at")) for r in cl]
        times = [t for t in times if t > 0]
        span = (max(times) - min(times)) if times else 0
        batch = (len(cl) >= BATCH_MIN_ROWS and times
                 and span <= BATCH_MAX_SPAN_SEC)
        meta = {
            "cluster_id": ci,
            "size": len(cl),
            "span_sec": int(span),
            "batch": bool(batch),
        }
        out.append(meta)
        for r in cl:
            by_id[id(r)] = meta
    return {"clusters": out, "by_id": by_id}


def _attempt_score_ctx(a: dict) -> bool:
    return a.get("score") is not None or a.get("mastery") is not None


def _truncate(s: str, n: int = 60) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


async def analyze_c(db, *, min_session_sec: int, result_mastery_threshold: float,
                    batch_gap_sec: int, script_date: str) -> dict:
    """实时拉取 + MIXED 桶行级细扫，产出清单（只读）。"""
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
    title_by_tid = data["title_by_tid"]
    book_by_scholar_tb = data["book_by_scholar_tb"]

    # 1) 行 → (学者×教材) 归桶（孤儿已由 collect_all 单列，不进桶）
    rows_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for s in states:
        sid = s.get("scholar_id")
        if not sid:
            continue
        tb = scan.resolve_textbook(s, lesson_map, sentence_map)
        if tb == "__orphan__":
            continue
        rows_by_key[(sid, tb)].append(s)

    # 2) 取 MIXED 桶
    mixed = [c for c in buckets.values() if scan.classify_book(c) == "MIXED"]
    mixed_keys = {(c["scholar_id"], c["textbook_id"]) for c in mixed}

    # 3) 桶内 lesson / sentence 元数据（order/title / text 前 60 字）
    bucket_lessons: dict[str, dict] = {}
    bucket_sents: dict[str, str] = {}
    for key in mixed_keys:
        for r in rows_by_key.get(key, []):
            lid = resolve_lesson(r, lesson_map, sentence_map)
            if lid:
                bucket_lessons[lid] = None
            sid = r.get("sentence_id")
            if sid:
                bucket_sents[sid] = ""
    lesson_meta_raw = await scan.query_by_ids(
        db, LESSON, "lesson_id", list(bucket_lessons),
        select={"lesson_id": 1, "textbook_id": 1, "order": 1, "title": 1},
    )
    lesson_meta: dict[str, dict] = {}
    for doc in lesson_meta_raw:
        lid = doc.get("lesson_id") or doc.get("_id")
        lesson_meta[lid] = {"order": doc.get("order"), "title": doc.get("title") or lid}
    for lid in list(bucket_lessons):  # 目录里查不到的课 → 用空标题占位
        if lid not in lesson_meta:
            lesson_meta[lid] = {"order": None, "title": lid}
    sent_raw = await scan.query_by_ids(
        db, SENTENCE_V2, "sentence_id", list(bucket_sents),
        select={"sentence_id": 1, "text": 1},
    )
    sent_text: dict[str, str] = {}
    for doc in sent_raw:
        sid = doc.get("sentence_id") or doc.get("_id")
        if sid:
            sent_text[sid] = _truncate(doc.get("text"), 80)

    # 4) 逐桶细扫
    buckets_out: list[dict] = []
    all_rows_out: list[dict] = []
    reason_totals: Counter = Counter()
    lesson_lines: list[dict] = []

    for c in sorted(mixed, key=lambda x: (-x["ghost_rows"], x["scholar_id"])):
        key = (c["scholar_id"], c["textbook_id"])
        sid, tid = key
        rows = rows_by_key.get(key, [])
        # 行 → 课
        rows_by_lesson: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            lid = resolve_lesson(r, lesson_map, sentence_map) or ""
            rows_by_lesson[lid].append(r)
        # 该学者 attempt → 课（仅统计本桶课）
        att_by_lesson: dict[str, Counter] = defaultdict(lambda: Counter(scored=0, noscore=0))
        for a in attempts:
            if a.get("scholar_id") != sid:
                continue
            lid = resolve_lesson(a, lesson_map, sentence_map)
            if not lid or lid not in rows_by_lesson:
                continue
            k = "scored" if _attempt_score_ctx(a) else "noscore"
            att_by_lesson[lid][k] += 1

        # 幽灵行簇（桶级）
        ghost_rows = [r for r in rows
                      if normalize_status(r.get("status")) in GHOST_CANDIDATE_STATUSES]
        cluster_res = cluster_ghost_rows(ghost_rows, batch_gap_sec)
        cluster_by_id = cluster_res["by_id"]

        # 课级证据 + 行分类
        reason_counter: Counter = Counter()
        rows_items: list[dict] = []
        for lid, lrows in sorted(
            rows_by_lesson.items(),
            key=lambda kv: (
                (lesson_meta.get(kv[0]) or {}).get("order") is None,
                (lesson_meta.get(kv[0]) or {}).get("order") or 0,
                kv[0],
            ),
        ):
            meta = lesson_meta.get(lid) or {"order": None, "title": lid or "(无课)"}
            by_status: Counter = Counter(normalize_status(r.get("status")) for r in lrows)
            result_rows = sum(1 for r in lrows
                              if normalize_status(r.get("status")) in RESULT_STATUSES)
            mastery_hit = sum(1 for r in lrows
                              if scan._num(r.get("mastery_score")) >= result_mastery_threshold)
            att_c = att_by_lesson.get(lid, Counter(scored=0, noscore=0))
            strong = bool(result_rows or mastery_hit or att_c.get("scored", 0))
            ghost_in_lesson = [r for r in lrows
                               if normalize_status(r.get("status")) in GHOST_CANDIDATE_STATUSES]

            l_finger = 0
            for r in ghost_in_lesson:
                st = normalize_status(r.get("status"))
                cl = cluster_by_id.get(id(r))
                att_n = scan._num(r.get("attempt_count"))
                _cd = _date(r.get("created_at"))
                _created_sec = scan._sec(r.get("created_at") or 0)
                _updated_sec = scan._sec(r.get("updated_at") or 0)
                _delta = int(max(0, _updated_sec - _created_sec)) if _created_sec > 0 else 0
                item = {
                    "_id": r.get("_id"),
                    "state_id": r.get("state_id") or r.get("_id"),
                    "scholar_id": sid,
                    "lesson_id": lid or None,
                    "lesson_label": (
                        (f"L{meta['order']}" if meta["order"] is not None else "L-")
                        + (f" {meta['title']}" if meta["title"] else "")
                    ),
                    "sentence_id": r.get("sentence_id"),
                    "sentence_text": sent_text.get(r.get("sentence_id"), "") or None,
                    "skill_code": r.get("skill_code"),
                    "status": st,
                    "attempt_count": r.get("attempt_count"),
                    "mastery_score": r.get("mastery_score"),
                    "progress": r.get("progress"),
                    "created_at": scan._ts(r.get("created_at")),
                    "updated_at": scan._ts(r.get("updated_at")),
                    "updated_delta_sec": _delta,
                    "lesson_evidence": {
                        "result_rows": result_rows,
                        "mastery_hit": mastery_hit,
                        "scored_attempts": att_c.get("scored", 0),
                        "noscore_attempts": att_c.get("noscore", 0),
                    },
                    "cluster": cl,
                }
                if not lid:
                    reason = R_NO_LESSON
                elif strong:
                    reason = R_LESSON_EVIDENCE
                elif att_n != 1 or r.get("mastery_score") is not None:
                    reason = R_ROW_REVISIT
                elif cl and cl["batch"]:
                    if _cd == script_date:
                        reason = R_FINGERPRINT
                        l_finger += 1
                    else:
                        reason = R_OFF_SCRIPT_DATE
                else:
                    reason = R_NON_BATCH
                item["reason"] = reason
                reason_counter[reason] += 1
                rows_items.append(item)

            lesson_lines.append({
                "scholar_id": sid, "textbook_id": tid, "lesson_id": lid or None,
                "order": meta.get("order"), "title": meta.get("title"),
                "state_rows": len(lrows), "by_status": dict(by_status),
                "ghost_rows": len(ghost_in_lesson),
                "fingerprint_rows": l_finger,
                "lesson_evidence": {
                    "result_rows": result_rows,
                    "mastery_hit": mastery_hit,
                    "scored_attempts": att_c.get("scored", 0),
                    "noscore_attempts": att_c.get("noscore", 0),
                    "strong": strong,
                },
            })

        buckets_out.append({
            "scholar_id": sid,
            "textbook_id": tid,
            "textbook_title": title_by_tid.get(tid, tid),
            "scholar_book": book_by_scholar_tb.get(key) or None,
            "state_rows": c["state_rows"],
            "state_by_status": dict(c["state_by_status"]),
            "ghost_rows": c["ghost_rows"],
            "ghost_attempt1_no_mastery": c["ghost_attempt1_no_mastery"],
            "result_rows": c["result_rows"],
            "mastery_hit": c["mastery_hit"],
            "scored_attempts": c["scored_attempts"],
            "attempt_rows": c["attempt_rows"],
            "long_sessions": c["long_sessions"],
            "session_dur_sec": c["session_dur_sec"],
            "clusters": cluster_res["clusters"],
            "reason_counts": dict(reason_counter),
        })
        all_rows_out.extend(rows_items)
        reason_totals.update(reason_counter)

    return {
        "meta": {
            "n_state": len(states),
            "n_attempt": len(attempts),
            "mixed_books": len(buckets_out),
            "min_session_sec": min_session_sec,
            "result_mastery_threshold": result_mastery_threshold,
            "batch_gap_sec": batch_gap_sec,
            "script_date": script_date,
        },
        "buckets": buckets_out,
        "lessons": lesson_lines,
        "rows": all_rows_out,
        "reason_totals": dict(reason_totals),
        "fingerprint_rows": [r for r in all_rows_out if r["reason"] == R_FINGERPRINT],
    }


def _book_line(b: dict) -> str:
    sb = b.get("scholar_book")
    if not sb:
        return "  scholar_book: (不存在)"
    return (
        f"  scholar_book: status={sb.get('status') or '-'} "
        f"total_time_spent={sb.get('total_time_spent') or 0}s "
        f"last={scan._ts(sb.get('last_studied_at'))} started={scan._ts(sb.get('started_at'))} "
        f"anchor_lesson={str(sb.get('current_lesson_id'))[:14] or '-'} "
        f"anchor_chapter={str(sb.get('current_chapter_id'))[:14] or '-'}"
    )


def _lesson_display(l: dict) -> str:
    order = l.get("order")
    label = f"L{order}" if order is not None else "L-"
    title = _truncate(l.get("title"), 18)
    st = " ".join(f"{k}={v}" for k, v in sorted(l["by_status"].items()))
    ev = l["lesson_evidence"]
    ev_s = (f"证据 R{ev['result_rows']} M{ev['mastery_hit']} "
            f"A{ev['scored_attempts']} att0分{ev['noscore_attempts']}")
    return (f"    {label:<4} {title:<20} | state {l['state_rows']:>3}({st}) "
            f"ghost {l['ghost_rows']:>3} | 指纹 {l['fingerprint_rows']:>3} | {ev_s}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="C 区（MIXED）幽灵行级细扫（只读，出人工过目清单）")
    parser.add_argument("--min-session-sec", type=int, default=scan.DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float,
                        default=scan.DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--batch-gap-sec", type=int, default=DEFAULT_BATCH_GAP_SEC,
                        help="created 簇判批相邻行最大间隔（秒）")
    parser.add_argument("--script-date", default=DEFAULT_SCRIPT_DATE,
                        help="幽灵批量脚本写入日（指纹候选 created 须为该日；默认 2026-08-14）")
    parser.add_argument("--json", default="", help="可选：输出行级清单 JSON")
    parser.add_argument("--full", action="store_true", help="无幽灵的课也列出")
    args = parser.parse_args()

    db = get_db()
    plan = await analyze_c(
        db,
        min_session_sec=args.min_session_sec,
        result_mastery_threshold=args.result_mastery_threshold,
        batch_gap_sec=args.batch_gap_sec,
        script_date=args.script_date,
    )
    rt = plan["reason_totals"]
    out: list[str] = []
    P = out.append

    P("=" * 96)
    P("C 区（MIXED）幽灵行级细扫（DRY-RUN，只读）| 判据同 scan_ghost_distribution.collect_all")
    P(f"强幽灵指纹 = 同批 created(≥{BATCH_MIN_ROWS}行/跨度≤{scan._span_hms(BATCH_MAX_SPAN_SEC)},"
      f" gap≤{args.batch_gap_sec}s) + attempt=1 + mastery=null + 课无强证据"
      f" + created 在脚本日 {args.script_date}")
    P("=" * 96)
    n_fp = len(plan["fingerprint_rows"])
    P(f"\n[0] 结论速览")
    P(f"  MIXED 书 {plan['meta']['mixed_books']} | 幽灵行 {sum(b['ghost_rows'] for b in plan['buckets'])}")
    P(f"  指纹候选（删除候选，人工过目）: {n_fp}")
    P(f"  其余幽灵去向: "
      + "  ".join(f"{k} {v}" for k, v in sorted(rt.items()) if k != R_FINGERPRINT))

    P("\n[1] 书级上下文")
    for b in plan["buckets"]:
        P(f"  {_fmt_scholar(b['scholar_id']):<14} | {b['textbook_title'][:22]:<22} "
          f"{b['textbook_id'][:14]:<14} | 行 {b['state_rows']} ghost {b['ghost_rows']}"
          + " ".join(f" {k}={v}" for k, v in sorted(b["state_by_status"].items())))
        P(f"    书级证据 R:{b['result_rows']} M:{b['mastery_hit']} "
          f"A带分:{b['scored_attempts']} A总:{b['attempt_rows']} S:{b['long_sessions']}"
          f"({b['session_dur_sec']}s) | 批簇 {len(b['clusters'])} 个"
          + ("（无 ≥5行/≤1h 批簇）" if not any(cl["batch"] for cl in b["clusters"]) else ""))
        P(_book_line(b))
        rc = b["reason_counts"]
        P(f"    行级去向: " + "  ".join(f"{k} {v}" for k, v in sorted(rc.items())))

    P(f"\n[2] 课级明细（幽灵行 > 0；指纹候选所在课）")
    for b in plan["buckets"]:
        P(f"  -- {_fmt_scholar(b['scholar_id'])} × {b['textbook_title'][:20]} "
          f"{b['textbook_id'][:14]} --")
        shown = 0
        for l in plan["lessons"]:
            if l["scholar_id"] != b["scholar_id"] or l["textbook_id"] != b["textbook_id"]:
                continue
            if l["ghost_rows"] == 0 and not args.full:
                continue
            P(_lesson_display(l))
            shown += 1
        if shown == 0:
            P("    （无幽灵行课）")

    if plan["fingerprint_rows"]:
        P(f"\n[3] 指纹候选行明细（{n_fp} 行；全量见 --json 清单，此处展示前 120 行）")
        P("  " + f"{'state_id':<44} | {'lesson':<26} | {'sentence 文本前40':<42} | created")
        for r in sorted(plan["fingerprint_rows"],
                        key=lambda x: (x["lesson_label"], x["_id"] or ""))[:120]:
            P(f"  {_truncate(str(r['_id'] or ''), 42):<44} | "
              f"{_truncate(r['lesson_label'], 24):<26} | "
              f"{_truncate(r.get('sentence_text'), 40):<42} | {r['created_at']}")
        if n_fp > 120:
            P(f"  …（其余 {n_fp - 120} 行略，见清单 JSON）")

    P(f"\n[合计] 幽灵行 {sum(b['ghost_rows'] for b in plan['buckets'])} | "
      f"指纹候选 {n_fp} | 其余 "
      + "  ".join(f"{k} {v}" for k, v in sorted(rt.items()) if k != R_FINGERPRINT))
    P("dry-run 未写库。指纹子集删除前请人工过目 --json 行级清单；"
      "其余幽灵默认保留/降级待产品确认。")

    if args.json:
        payload = {
            "meta": plan["meta"],
            "reason_totals": rt,
            "buckets": plan["buckets"],
            "lessons": plan["lessons"],
            "rows": plan["rows"],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        P(f"\n行级清单 JSON 已输出: {args.json}")

    print("\n".join(out))


if __name__ == "__main__":
    asyncio.run(main())
