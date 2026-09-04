"""全量幽灵分布扫描：所有 skill_state 的 learning/review_due 无证据行，按学者 × 教材聚合（只读）

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§10.4 建议）：
  - 阶段 0 个案（scholar 950208…319 × tb_5aefa2dee7e34546）确认为 8-14 凌晨幽灵批量：
    64 行全 learning、无 attempt/session/评估佐证、scholar_book 无锚点。
  - 幽灵批量疑似覆盖多本教材、可能不止一个学者中招 → 全量扫描评估修复面，
    决定 repair 脚本范围。

判据口径（与 diagnose_book_evidence.py / 方案 §5.5.1 一致）：
  - 幽灵候选状态：normalize_status(status) ∈ {learning, review_due}
  - 强证据（判定书级「真实」）：同（学者×教材）内任一
      * skill_state 结果性状态行（learned/mastered）
      * mastery_score ≥ result-mastery-threshold 的行
      * 归属该教材的 study_attempt 带 score / mastery
      * 归属该教材的 study_session duration_sec ≥ min-session-sec
  - 书级分类：
      GHOST_FULL = 有幽灵候选行 且 无任何强证据（建议整体清理）
      MIXED      = 有幽灵候选行 且 存在强证据（需行级甄别，不能整书清）
      REAL       = 无幽灵候选行（至少一行非幽灵）
      EMPTY      = 仅有目录外/孤儿归属的行（单独桶）
  - 行归属教材：lesson_id → lesson.textbook_id；未命中则 sentence_id → sentence_v2.textbook_id；
    两者皆无 → __orphan__（引用了当前目录中不存在的内容，单独统计）。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载；纯只读）：
  python scripts/scan_ghost_distribution.py
  python scripts/scan_ghost_distribution.py --full          # 打印全部学者×教材行（含 REAL）
  python scripts/scan_ghost_distribution.py --json /tmp/ghost_scan.json

退出码：0（扫描本身成功）；仅 DB 连接/参数错误 → 1。
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

from services.dependencies import get_db  # noqa: E402
from services.models.content import (  # noqa: E402
    LESSON,
    SENTENCE_V2,
    TEXTBOOK_V2,
    get_textbook_v2,
    query_all_pages,
)
from services.models.events import (  # noqa: E402
    STUDY_ATTEMPT,
    STUDY_SESSION,
)
from services.models.learning import (  # noqa: E402
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_MASTERED,
    STATUS_LEARNING,
    STATUS_REVIEW_DUE,
    normalize_status,
)
from services.models.scholar_book import (  # noqa: E402
    SCHOLAR_BOOK,
)

# ---------------------------------------------------------------------------
# 判据常量（命令行可覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MIN_SESSION_SEC = 60
DEFAULT_RESULT_MASTERY_THRESHOLD = 60  # mastery_score（0-100 分制）

RESULT_STATUSES = {STATUS_LEARNED, STATUS_MASTERED}
GHOST_CANDIDATE_STATUSES = {STATUS_LEARNING, STATUS_REVIEW_DUE}

# 批量写入指纹：幽灵行 ≥ BATCH_MIN_ROWS 且 created_at 跨度 ≤ BATCH_MAX_SPAN_SEC
BATCH_MIN_ROWS = 5
BATCH_MAX_SPAN_SEC = 3600


def _num(v, default=0):
    try:
        if v is None:
            return default
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return default


def _sec(v) -> float:
    """时间戳统一转秒（>=1e11 视为毫秒）。"""
    n = _num(v, 0)
    return n / 1000.0 if n >= 100_000_000_000 else n


def _ts(v_raw) -> str:
    """时间戳 → UTC 字符串（秒/毫秒自适应）。"""
    try:
        v = _sec(v_raw)
        if v <= 0:
            return "-"
        return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def _span_hms(sec: float) -> str:
    sec = int(sec)
    if sec < 0:
        return "-"
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


async def query_by_ids(db, collection: str, id_field: str, ids: list[str],
                       select: dict | None = None, chunk: int = 100) -> list[dict]:
    """按 id 列表分块 $in 查询（无 id 输入返回空）。"""
    out: list[dict] = []
    uniq = sorted({i for i in ids if i})
    for i in range(0, len(uniq), chunk):
        res = await db.query(
            collection,
            where={id_field: {"$in": uniq[i:i + chunk]}},
            limit=chunk,
            select=select,
        )
        out.extend(res.get("records", []))
    return out


def resolve_textbook(row: dict, lesson_map: dict[str, dict],
                     sentence_map: dict[str, dict]) -> str:
    """行归属教材 id；失败返回 '__orphan__'。

    优先级：行内 lesson_id → lesson 文档；否则 sentence_id → sentence_v2 文档；
    都没有 → '__orphan__'（引用的内容已不在当前目录）。
    """
    lid = row.get("lesson_id")
    if lid:
        doc = lesson_map.get(lid)
        if doc and doc.get("textbook_id"):
            return doc["textbook_id"]
    sid = row.get("sentence_id")
    if sid:
        doc = sentence_map.get(sid)
        if doc:
            if doc.get("textbook_id"):
                return doc["textbook_id"]
            if doc.get("lesson_id"):
                ldoc = lesson_map.get(doc["lesson_id"])
                if ldoc and ldoc.get("textbook_id"):
                    return ldoc["textbook_id"]
    return "__orphan__"


def classify_book(stats: dict) -> str:
    """按强证据给（学者×教材）分档。"""
    if stats["ghost_rows"]:
        if (stats["result_rows"] or stats["mastery_hit"] or stats["scored_attempts"]
                or stats["long_sessions"]):
            return "MIXED"
        return "GHOST_FULL"
    if stats["state_rows"] or stats["attempt_rows"]:
        return "REAL"
    return "EMPTY"


async def collect_all(
    db,
    *,
    min_session_sec: int = DEFAULT_MIN_SESSION_SEC,
    result_mastery_threshold: float = DEFAULT_RESULT_MASTERY_THRESHOLD,
) -> dict:
    """一次性拉取全部原始记录 + 目录映射 +（学者×教材）桶聚合（只读）。

    **扫描报告（main）与修复脚本共用本函数，保证幽灵判据单源。**
    返回结构：
    - states / attempts / sessions / books：全量原始行
    - lesson_map / sentence_map / title_by_tid：目录映射
    - buckets：{(scholar_id, textbook_id): cell}（cell 含 ghost_rows 等聚合）
    - orphan_rows：归属失败（引用内容不在当前目录）的 skill_state 行
    - book_by_scholar_tb：{(scholar_id, textbook_id): scholar_book 文档}
    """
    # ── 1) 原始记录 ──────────────────────────────────────────────
    states = await query_all_pages(db, collection=SKILL_STATE, where={})
    attempts = await query_all_pages(db, collection=STUDY_ATTEMPT, where={})
    sessions = await query_all_pages(db, collection=STUDY_SESSION, where={})
    books = await query_all_pages(db, collection=SCHOLAR_BOOK, where={})

    # ── 2) 目录映射 ──────────────────────────────────────────────
    lesson_ids = [s.get("lesson_id") for s in states] + [a.get("lesson_id") for a in attempts]
    sentence_ids = [s.get("sentence_id") for s in states] + [a.get("sentence_id") for a in attempts]

    lesson_map: dict[str, dict] = {}
    for doc in await query_by_ids(db, LESSON, "lesson_id", lesson_ids,
                                  select={"lesson_id": 1, "textbook_id": 1}):
        lesson_map[doc.get("lesson_id") or doc.get("_id")] = doc

    sentence_map: dict[str, dict] = {}
    for doc in await query_by_ids(db, SENTENCE_V2, "sentence_id", sentence_ids,
                                  select={"sentence_id": 1, "lesson_id": 1, "textbook_id": 1}):
        sentence_map[doc.get("sentence_id") or doc.get("_id")] = doc
    # sentence_v2 引用的 lesson 也可能不在 lesson_ids 里，补查一次
    extra_lessons = [d.get("lesson_id") for d in sentence_map.values() if d.get("lesson_id")]
    for doc in await query_by_ids(db, LESSON, "lesson_id", extra_lessons,
                                  select={"lesson_id": 1, "textbook_id": 1}):
        lesson_map.setdefault(doc.get("lesson_id") or doc.get("_id"), doc)

    # 教材标题
    tids = {b.get("textbook_id") for b in books if b.get("textbook_id")}
    for lm in lesson_map.values():
        if lm.get("textbook_id"):
            tids.add(lm["textbook_id"])
    for sm in sentence_map.values():
        if sm.get("textbook_id"):
            tids.add(sm["textbook_id"])
    title_by_tid: dict[str, str] = {}
    for tid in sorted(t for t in tids if t):
        try:
            doc = await get_textbook_v2(db, tid)
            title_by_tid[tid] = (doc or {}).get("title") or tid
        except Exception:
            title_by_tid[tid] = tid

    # ── 3) 行归属 + 桶聚合 ───────────────────────────────────────
    # 桶 key: (scholar_id, textbook_id)
    buckets: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "scholar_id": "", "textbook_id": "",
        "state_rows": 0, "state_by_status": Counter(),
        "ghost_rows": 0, "ghost_attempt1_no_mastery": 0,
        "result_rows": 0, "mastery_hit": 0,
        "created_min": None, "created_max": None,
        "attempt_rows": 0, "scored_attempts": 0,
        "long_sessions": 0, "session_dur_sec": 0,
    })
    orphan_rows = []  # 未归属（引用内容不在目录）的 skill_state 行

    for s in states:
        scholar = s.get("scholar_id")
        if not scholar:
            continue
        tb = resolve_textbook(s, lesson_map, sentence_map)
        st = normalize_status(s.get("status"))
        if tb == "__orphan__":
            orphan_rows.append(s)
            continue
        key = (scholar, tb)
        cell = buckets[key]
        cell["scholar_id"] = scholar
        cell["textbook_id"] = tb
        cell["state_rows"] += 1
        cell["state_by_status"][st] += 1
        if st in GHOST_CANDIDATE_STATUSES:
            cell["ghost_rows"] += 1
            if _num(s.get("attempt_count")) == 1 and s.get("mastery_score") is None \
                    and _num(s.get("progress")) == 0.5:
                cell["ghost_attempt1_no_mastery"] += 1
        if st in RESULT_STATUSES:
            cell["result_rows"] += 1
        if _num(s.get("mastery_score")) >= result_mastery_threshold:
            cell["mastery_hit"] += 1
        for ts_key, f in (("created_min", min), ("created_max", max)):
            v = _sec(s.get("created_at") or s.get("updated_at"))
            if v > 0:
                cell[ts_key] = f(cell[ts_key], v) if cell[ts_key] is not None else v

    # attempt → 教材（仅计入证据：带分）
    for a in attempts:
        scholar = a.get("scholar_id")
        if not scholar:
            continue
        tb = resolve_textbook(a, lesson_map, sentence_map)
        if tb == "__orphan__":
            continue
        cell = buckets[(scholar, tb)]
        cell["scholar_id"] = scholar
        cell["textbook_id"] = tb
        cell["attempt_rows"] += 1
        if a.get("score") is not None or a.get("mastery") is not None:
            cell["scored_attempts"] += 1

    # session → 教材（弱信号：时长）
    for s in sessions:
        scholar = s.get("scholar_id")
        tb = s.get("textbook_id")
        if not scholar or not tb:
            continue
        key = (scholar, tb)
        cell = buckets[key]
        cell["scholar_id"] = scholar
        cell["textbook_id"] = tb
        dur = _num(s.get("duration_sec"))
        cell["session_dur_sec"] += dur
        if dur >= min_session_sec:
            cell["long_sessions"] += 1

    # scholar_book 锚点信息（展示 / 修复用）
    book_by_scholar_tb: dict[tuple[str, str], dict] = {}
    for b in books:
        sid, tid = b.get("scholar_id"), b.get("textbook_id")
        if sid and tid:
            book_by_scholar_tb[(sid, tid)] = b

    return {
        "states": states,
        "attempts": attempts,
        "sessions": sessions,
        "books": books,
        "lesson_map": lesson_map,
        "sentence_map": sentence_map,
        "title_by_tid": title_by_tid,
        "buckets": buckets,
        "orphan_rows": orphan_rows,
        "book_by_scholar_tb": book_by_scholar_tb,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="全量幽灵分布扫描（只读）")
    parser.add_argument("--min-session-sec", type=int, default=DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float, default=DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--json", default="", help="可选：输出完整聚合 JSON")
    parser.add_argument("--full", action="store_true", help="打印全部（学者×教材）行，而非仅幽灵相关")
    args = parser.parse_args()

    db = get_db()
    data = await collect_all(
        db,
        min_session_sec=args.min_session_sec,
        result_mastery_threshold=args.result_mastery_threshold,
    )
    states = data["states"]
    attempts = data["attempts"]
    sessions = data["sessions"]
    books = data["books"]
    buckets = data["buckets"]
    orphan_rows = data["orphan_rows"]
    title_by_tid = data["title_by_tid"]
    book_by_scholar_tb = data["book_by_scholar_tb"]

    n_state = len(states)
    n_attempt = len(attempts)
    n_session = len(sessions)
    n_book = len(books)
    n_scholars = len({s.get("scholar_id") for s in states} | {a.get("scholar_id") for a in attempts})
    print(f"数据集: skill_state={n_state} study_attempt={n_attempt} "
          f"study_session={n_session} scholar_book={n_book} 涉及学者={n_scholars}")

    # 孤儿行聚合（引用内容不在目录，无法归属教材）
    orphan_scholars: dict[str, Counter] = defaultdict(Counter)
    orphan_span: dict[str, list] = defaultdict(list)
    for o in orphan_rows:
        sid = o.get("scholar_id") or "-"
        orphan_scholars[sid][normalize_status(o.get("status"))] += 1
        v = _sec(o.get("created_at"))
        if v > 0:
            orphan_span[sid].append(v)

    # ── 4) 报告 ─────────────────────────────────────────────────
    out: list[str] = []
    P = out.append

    def cls_of(cell: dict) -> str:
        return classify_book(cell)

    rows_sorted = sorted(
        buckets.values(),
        key=lambda c: (-c["ghost_rows"], c["scholar_id"], c["textbook_id"]),
    )

    # 汇总
    cat_count: Counter[str] = Counter()
    cat_scholars: dict[str, set] = defaultdict(set)
    cat_ghost_rows: Counter[str] = Counter()
    cat_state_rows: Counter[str] = Counter()
    for c in rows_sorted:
        k = cls_of(c)
        cat_count[k] += 1
        cat_scholars[k].add(c["scholar_id"])
        cat_ghost_rows[k] += c["ghost_rows"]
        cat_state_rows[k] += c["state_rows"]

    P("=" * 96)
    P("全量幽灵分布扫描：skill_state learning/review_due 无证据行 × (学者×教材)")
    P(f"判据: 幽灵候选 ∈ {{{','.join(sorted(GHOST_CANDIDATE_STATUSES))}}} | 强证据: "
      f"结果性状态行 / mastery≥{args.result_mastery_threshold} / 带分 attempt / "
      f"session≥{args.min_session_sec}s")
    P("=" * 96)
    P(f"\n[1] 汇总（按书分类）")
    P(f"  GHOST_FULL  书 {cat_count.get('GHOST_FULL', 0)}（学者 {len(cat_scholars['GHOST_FULL'])}）"
      f" | 幽灵行 {cat_ghost_rows['GHOST_FULL']}（该档 state 行 {cat_state_rows['GHOST_FULL']}）→ 建议整书清理")
    P(f"  MIXED       书 {cat_count.get('MIXED', 0)}（学者 {len(cat_scholars['MIXED'])}）"
      f" | 幽灵行 {cat_ghost_rows['MIXED']} → 行级甄别，不能整书清")
    P(f"  REAL        书 {cat_count.get('REAL', 0)}（学者 {len(cat_scholars['REAL'])}）"
      f" | 幽灵行 {cat_ghost_rows['REAL']}")
    total_ghost = sum(c["ghost_rows"] for c in rows_sorted)
    total_state = sum(c["state_rows"] for c in rows_sorted)
    P(f"  幽灵行合计 {total_ghost} / 全部 state 行 {total_state}（书内可归属）")
    P(f"  孤儿行（引用内容不在当前目录，无法归属教材）: {len(orphan_rows)}"
      + (f"（学者 {len({o.get('scholar_id') for o in orphan_rows})}）" if orphan_rows else ""))

    def _fmt_scholar(sid: str) -> str:
        return sid[:12]

    def _bucket_line(cell: dict, tag: str = "") -> str:
        cls = cls_of(cell)
        tid = cell["textbook_id"]
        title = title_by_tid.get(tid, tid)
        b = book_by_scholar_tb.get((cell["scholar_id"], tid))
        anchor = "no-book"
        if b:
            cl = b.get("current_lesson_id")
            anchor = f"{b.get('status') or '-'}" + (f"/L:{cl[:16]}" if cl else "/无锚")
        span = None
        if cell["created_min"] is not None and cell["created_max"] is not None:
            span = cell["created_max"] - cell["created_min"]
        batch = (cell["ghost_rows"] >= BATCH_MIN_ROWS and span is not None
                 and span <= BATCH_MAX_SPAN_SEC)
        status_str = " ".join(f"{k}={v}" for k, v in sorted(cell["state_by_status"].items()))
        s = (
            f"{_fmt_scholar(cell['scholar_id']):<14} | {title[:22]:<22} {tid[:16]:<16} | "
            f"sb:{anchor:<14} | state:{cell['state_rows']:>3}({status_str}) "
            f"ghost:{cell['ghost_rows']:>3}"
        )
        if cls in ("GHOST_FULL", "MIXED"):
            s += f"[1-att-no-mastery:{cell['ghost_attempt1_no_mastery']}]"
        s += f" | 证据 R:{cell['result_rows']} M:{cell['mastery_hit']} A:{cell['scored_attempts']} S:{cell['long_sessions']}"
        if span is not None:
            s += f" | 跨度 {_span_hms(span)}" + (" <BATCH!>" if batch else "")
        s += f" | {cls}"
        if tag:
            s += f" {tag}"
        return s

    # GHOST_FULL 明细
    P("\n[2] GHOST_FULL 明细（无任何强证据的整书幽灵，降序）")
    gf = [c for c in rows_sorted if cls_of(c) == "GHOST_FULL"]
    if not gf:
        P("  无")
    else:
        P("  scholar         | 教材                            | 锚点           | state 行(状态)             ghost | 证据 R/M/A/S | 时间跨度")
        for c in gf:
            P("  " + _bucket_line(c))

    # MIXED 明细
    P("\n[3] MIXED 明细（幽灵与真实证据共存，行级甄别）")
    mx = [c for c in rows_sorted if cls_of(c) == "MIXED"]
    if not mx:
        P("  无")
    else:
        for c in mx:
            P("  " + _bucket_line(c))

    # 孤儿行明细
    if orphan_rows:
        P("\n[4] 孤儿行（无法归属教材的 skill_state）")
        for sid, cnt in sorted(orphan_scholars.items()):
            span_s = ""
            if orphan_span[sid] and len(orphan_span[sid]) >= 2:
                span_s = f" 跨度 {_span_hms(max(orphan_span[sid]) - min(orphan_span[sid]))}"
            P(f"  {_fmt_scholar(sid):<14} 行 {sum(cnt.values()):>3} "
              + " ".join(f"{k}={v}" for k, v in sorted(cnt.items())) + span_s)

    if args.full:
        P("\n[5] 全部（学者×教材）行")
        for c in rows_sorted:
            P("  " + _bucket_line(c))
        # REAL 目录（无幽灵行）
        real_books = [c for c in rows_sorted if cls_of(c) == "REAL"]
        if real_books:
            P("  -- REAL（无幽灵候选行）--")
            for c in real_books:
                P("  " + _bucket_line(c))

    print("\n".join(out))

    if args.json:
        payload = {
            "meta": {
                "n_state": n_state, "n_attempt": n_attempt,
                "n_session": n_session, "n_book": n_book, "n_scholars": n_scholars,
                "lesson_map": len(data["lesson_map"]), "sentence_map": len(data["sentence_map"]),
                "orphan_rows": len(orphan_rows),
                "min_session_sec": args.min_session_sec,
                "result_mastery_threshold": args.result_mastery_threshold,
            },
            "summary": {
                k: {
                    "books": cat_count.get(k, 0),
                    "scholars": len(cat_scholars[k]),
                    "ghost_rows": cat_ghost_rows[k],
                    "state_rows": cat_state_rows[k],
                }
                for k in ("GHOST_FULL", "MIXED", "REAL")
            },
            "buckets": [
                {
                    "scholar_id": c["scholar_id"],
                    "textbook_id": c["textbook_id"],
                    "textbook_title": title_by_tid.get(c["textbook_id"], ""),
                    "state_rows": c["state_rows"],
                    "state_by_status": dict(c["state_by_status"]),
                    "ghost_rows": c["ghost_rows"],
                    "ghost_attempt1_no_mastery": c["ghost_attempt1_no_mastery"],
                    "result_rows": c["result_rows"],
                    "mastery_hit": c["mastery_hit"],
                    "scored_attempts": c["scored_attempts"],
                    "long_sessions": c["long_sessions"],
                    "session_dur_sec": c["session_dur_sec"],
                    "created_min": c["created_min"],
                    "created_max": c["created_max"],
                    "span_sec": (c["created_max"] - c["created_min"]
                                 if c["created_min"] is not None and c["created_max"] is not None else None),
                    "classify": cls_of(c),
                    "scholar_book": book_by_scholar_tb.get((c["scholar_id"], c["textbook_id"])) or None,
                }
                for c in rows_sorted
            ],
            "orphan_scholars": [
                {
                    "scholar_id": sid,
                    "rows": sum(cnt.values()),
                    "by_status": dict(cnt),
                }
                for sid, cnt in sorted(orphan_scholars.items())
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        print(f"\n完整聚合 JSON 已输出: {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
