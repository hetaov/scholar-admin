"""阶段 0 诊断脚本：目标 scholar × textbook 的「学习证据 / 幽灵分布 / 锚点」盘点（只读）

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§7 阶段 0）：
  - 目标个案 scholar_id=950208346a743b7a04d4a5d40fbfe319，
    textbook_id=tb_5aefa2dee7e34546（目录序 Unit1..8，Review 无课号排末）。
  - 现象：产品语义应视为「新开始」，但沉浸式页冷启动命中的首个任务是第 8 课。
  - 需核实：stored 锚点位置、各课 skill_state / study_attempt / study_session
    证据分布、learning 幽灵假设、evidence 阈值校正。

判据口径（与重构方案 §5.5.1 对齐，参数可用 -- 覆盖）：
  - 结果性状态（强证据）：skill_state.status ∈ {learned, mastered}（中文经
    normalize_status 归一）或 mastery_score ≥ result-mastery-threshold。
  - 带分作答（强证据）：study_attempt.score / mastery 任一非空。
  - 持续时长（弱信号，book 级）：study_session.duration_sec ≥ min-session-sec。
  - 已知限制：历史 skip（前端带 status=learning、无分）与真实低分作答在后端
    normalize_attempt_status 统一回落 completed，事件表不可区分 —— 诊断把
    「仅 learning 行且无任何强证据」的课标为 GHOST-LIKE，输出供人工核对。

用法（scholar-admin 项目根目录，需 CloudBase 凭据，.env 自动加载；纯只读）：
  python scripts/diagnose_book_evidence.py --scholar-id 950208346a743b7a04d4a5d40fbfe319 --textbook-id tb_5aefa2dee7e34546
  python scripts/diagnose_book_evidence.py --scholar-id S --textbook-id T --json /tmp/diag.json

退出码：0（即使发现幽灵，诊断本身成功）；仅参数缺失/DB 连接失败 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.models.content import (  # noqa: E402
    CHAPTER,
    LESSON,
    SENTENCE_V2,
    TEXTBOOK_V2,
    get_chapters,
    get_lessons_by_textbook,
    get_textbook_v2,
    query_all_pages,
)
from services.models.events import (  # noqa: E402
    STUDY_ATTEMPT,
    STUDY_SESSION,
    normalize_attempt_status,
)
from services.models.learning import (  # noqa: E402
    SKILL_STATE,
    STATUS_LEARNED,
    STATUS_LEARNING,
    STATUS_MASTERED,
    STATUS_NOT_STARTED,
    STATUS_REVIEW_DUE,
    normalize_status,
)
from services.models.scholar_book import (  # noqa: E402
    SCHOLAR_BOOK,
    scholar_book_id,
)

# ---------------------------------------------------------------------------
# 判据常量（命令行可覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MIN_SESSION_SEC = 60
DEFAULT_RESULT_MASTERY_THRESHOLD = 60  # mastery_score（0-100 分制）

RESULT_STATUSES = {STATUS_LEARNED, STATUS_MASTERED}
GHOST_CANDIDATE_STATUSES = {STATUS_LEARNING, STATUS_REVIEW_DUE}


def _ts(v_raw) -> str:
    """时间戳 → UTC 字符串。自动识别秒/毫秒（>=1e11 视为毫秒）；空/无效返回 '-'。"""
    try:
        v = _num(v_raw, 0)
        if v <= 0:
            return "-"
        if v >= 100_000_000_000:
            v = v / 1000
        return datetime.fromtimestamp(v, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return "-"


def _num(v, default=0):
    try:
        if v is None:
            return default
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 数据拉取
# ---------------------------------------------------------------------------


async def load_lesson_sentence_map(db, lessons: list[dict]) -> dict[str, str]:
    """教材内 sentence_id → lesson_id 映射（供缺 lesson_id 的历史行回填归属）。"""
    mapping: dict[str, str] = {}
    for lesson in lessons:
        lid = lesson.get("lesson_id") or lesson.get("_id")
        sents = await query_all_pages(
            db, collection=SENTENCE_V2, where={"lesson_id": lid}, page_size=1000
        )
        for s in sents:
            sid = s.get("sentence_id") or s.get("_id")
            if sid:
                mapping[sid] = lid
    return mapping


def assign_lesson(row: dict, lesson_ids: set[str], sent2lesson: dict[str, str]) -> str:
    """把 skill_state / study_attempt 行归属到课。

    优先级：行内 lesson_id ∈ 目标教材课集合 → 用它；否则 sentence_id 反查；
    都失败返回 OUTSIDE（不属于该教材，可能来自其它教材/其它学科）。
    """
    lid = row.get("lesson_id")
    if lid and lid in lesson_ids:
        return lid
    sid = row.get("sentence_id")
    if sid:
        hit = sent2lesson.get(sid)
        if hit:
            return hit
    return "__outside__"


async def collect_evidence(db, scholar_id: str, textbook_id: str) -> dict:
    """拉取全部原始记录并按课聚合（只读）。"""
    # 1) 教材与锚点
    book = None
    book_res = await db.query(
        collection=SCHOLAR_BOOK,
        where={"_id": scholar_book_id(scholar_id, textbook_id)},
        limit=1,
    )
    if book_res.get("records"):
        book = book_res["records"][0]

    textbook = await get_textbook_v2(db, textbook_id)
    chapters = await get_chapters(db, textbook_id)
    lessons = await get_lessons_by_textbook(db, textbook_id)

    # 2) 学者全量学习记录
    states = await query_all_pages(db, collection=SKILL_STATE, where={"scholar_id": scholar_id})
    attempts = await query_all_pages(db, collection=STUDY_ATTEMPT, where={"scholar_id": scholar_id})
    sessions = await query_all_pages(db, collection=STUDY_SESSION, where={"scholar_id": scholar_id})
    all_books = await query_all_pages(db, collection=SCHOLAR_BOOK, where={"scholar_id": scholar_id})

    # 3) 句→课映射（仅目标教材）
    sent2lesson = await load_lesson_sentence_map(db, lessons)
    lesson_ids = {l.get("lesson_id") or l.get("_id") for l in lessons if l.get("lesson_id") or l.get("_id")}

    return {
        "book": book,
        "textbook": textbook,
        "chapters": chapters,
        "lessons": lessons,
        "states": states,
        "attempts": attempts,
        "sessions": sessions,
        "all_books": all_books,
        "sent2lesson": sent2lesson,
        "lesson_ids": lesson_ids,
    }


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def aggregate(db, ev: dict, *, min_session_sec: int, result_mastery: float) -> dict:
    lesson_ids = ev["lesson_ids"]
    sent2lesson = ev["sent2lesson"]

    # 行级证据标记
    for st in ev["states"]:
        st["_lesson"] = assign_lesson(st, lesson_ids, sent2lesson)
        st["_norm_status"] = normalize_status(st.get("status"))
    for at in ev["attempts"]:
        at["_lesson"] = assign_lesson(at, lesson_ids, sent2lesson)
        at["_norm_attempt"] = normalize_attempt_status(at.get("status"))

    # 每课聚合
    per_lesson: dict[str, dict] = {}
    for lesson in ev["lessons"]:
        lid = lesson.get("lesson_id") or lesson.get("_id")
        per_lesson[lid] = {
            "lesson_id": lid,
            "chapter_id": lesson.get("chapter_id"),
            "order": lesson.get("order"),
            "title": lesson.get("title", ""),
            "sentence_count": _num(lesson.get("sentence_count"), 0),
            "state_total": 0,
            "state_by_status": Counter(),
            "state_result_rows": 0,
            "state_mastery_hit": 0,
            "state_attempt_count": 0,
            "state_min_ts": None,
            "state_max_ts": None,
            "attempt_total": 0,
            "attempt_by_status": Counter(),
            "attempt_with_score": 0,
            "attempt_with_mastery": 0,
            "attempt_time_spent": 0,
            "sentences_seen": set(),
        }

    # 行落课
    for st in ev["states"]:
        key = st["_lesson"]
        if key == "__outside__":
            continue
        cell = per_lesson[key]
        cell["state_total"] += 1
        cell["state_by_status"][st["_norm_status"]] += 1
        if st["_norm_status"] in RESULT_STATUSES:
            cell["state_result_rows"] += 1
        ms = _num(st.get("mastery_score"))
        if ms and ms >= result_mastery:
            cell["state_mastery_hit"] += 1
        cell["state_attempt_count"] += _num(st.get("attempt_count"))
        ts = st.get("last_studied_at") or st.get("created_at")
        if ts:
            tsn = _num(ts)
            if cell["state_min_ts"] is None or tsn < _num(cell["state_min_ts"]):
                cell["state_min_ts"] = ts
            if cell["state_max_ts"] is None or tsn > _num(cell["state_max_ts"]):
                cell["state_max_ts"] = ts

    for at in ev["attempts"]:
        key = at["_lesson"]
        if key == "__outside__":
            continue
        cell = per_lesson[key]
        cell["attempt_total"] += 1
        cell["attempt_by_status"][at["_norm_attempt"]] += 1
        if at.get("score") is not None:
            cell["attempt_with_score"] += 1
        if at.get("mastery") is not None:
            cell["attempt_with_mastery"] += 1
        cell["attempt_time_spent"] += _num(at.get("time_spent"))

    # 会话（book 级弱信号）
    _tid = (ev["textbook"] or {}).get("_id") or ""
    session_for_book = [s for s in ev["sessions"] if (s.get("textbook_id") or "") == _tid]
    session_dur_book = sum(_num(s.get("duration_sec")) for s in session_for_book)
    session_long_book = sum(
        1 for s in session_for_book if _num(s.get("duration_sec")) >= min_session_sec
    )

    return {
        "book": ev["book"],
        "textbook": ev["textbook"],
        "chapters": ev["chapters"],
        "lessons": ev["lessons"],
        "all_books": ev["all_books"],
        "raw_states": ev["states"],
        "raw_attempts": ev["attempts"],
        "per_lesson": per_lesson,
        "outside": {
            "state_rows": [s for s in ev["states"] if s["_lesson"] == "__outside__"],
            "attempt_rows": [a for a in ev["attempts"] if a["_lesson"] == "__outside__"],
        },
        "session": {
            "total": len(ev["sessions"]),
            "for_book": len(session_for_book),
            "duration_sec_book": session_dur_book,
            "long_count_book": session_long_book,
        },
    }


# ---------------------------------------------------------------------------
# 判定档位（输出 lesson 级结论）
# ---------------------------------------------------------------------------


def classify_lesson(cell: dict) -> dict:
    """按强证据给课分档（输出 reason + 建议动作）。"""
    if cell["state_result_rows"] > 0 or cell["state_mastery_hit"] > 0:
        return {"level": "REAL_RESULT", "note": "有结果性状态（learned/mastered 或高 mastery）"}
    if cell["attempt_with_score"] > 0 or cell["attempt_with_mastery"] > 0:
        return {"level": "REAL_ATTEMPT", "note": "无结果行，但有带分作答事件"}
    ghost_statuses = GHOST_CANDIDATE_STATUSES & set(cell["state_by_status"])
    if ghost_statuses:
        # 有 learning/review_due 行，但没有任何强证据 → 幽灵候选（skip/占位产生）
        if cell["attempt_total"] == 0:
            note = "仅 skill_state 无任何 attempt（纯幽灵候选）"
        else:
            note = "有 attempt 但全无分（历史 skip 与真实低分不可区分，待人工复核）"
        return {"level": "GHOST_LIKE", "note": note}
    if cell["state_total"] == 0 and cell["attempt_total"] == 0:
        return {"level": "NO_DATA", "note": "无任何学习记录"}
    return {"level": "NOISE", "note": "存在非目标状态行（not_started/其它），无有效证据"}


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def render_report(scholar_id: str, textbook_id: str, agg: dict, *, result_mastery: float) -> str:
    """渲染人类可读诊断报告。"""
    book = agg["book"]
    lessons = agg["lessons"]
    per = agg["per_lesson"]
    textbook = agg["textbook"] or {}
    out: list[str] = []
    P = out.append

    P("=" * 78)
    P(f"阶段 0 证据诊断：scholar_id={scholar_id}  textbook_id={textbook_id}")
    P(f"教材：{textbook.get('title') or textbook_id}")
    P(f"判据：结果性状态 ∈ {{{','.join(sorted(RESULT_STATUSES))}}} | "
      f"mastery_score ≥ {result_mastery} | 会话时长阈值秒（book 级弱信号）")
    P("=" * 78)

    # 1) stored 锚点
    P("\n[1] scholar_book（stored 锚点）")
    if not book:
        P("  未找到 scholar_book 记录（可能从未建过锚点）")
    else:
        lesson_by_id = {l.get("lesson_id") or l.get("_id"): l for l in lessons}
        cl = book.get("current_lesson_id")
        cc = book.get("current_chapter_id")
        cl_doc = lesson_by_id.get(cl)
        P(f"  _id                : {book.get('_id')}")
        P(f"  status             : {book.get('status')}")
        P(f"  current_lesson_id  : {cl or '-'}"
          + (f"  → 《{cl_doc.get('title')}》(order={cl_doc.get('order')})" if cl_doc else "  ⚠ 不在该教材目录内"))
        P(f"  current_chapter_id : {cc or '-'}")
        P(f"  current_group_id   : {book.get('current_group_id') or '-'}")
        P(f"  last_studied_at    : {_ts(book.get('last_studied_at'))}")
        P(f"  total_time_spent   : {_num(book.get('total_time_spent'))}s  "
          f"(created_at={_ts(book.get('created_at'))})")

    # 2) 目录序
    P("\n[2] 教材目录序（chapter/lesson order ASC）")
    ch_by_id = {c.get("chapter_id"): c for c in agg["chapters"]}
    cur_ch, cur_no = None, 0
    for lesson in lessons:
        cid = lesson.get("chapter_id")
        if cid and cid != cur_ch:
            cur_ch = cid
            ch = ch_by_id.get(cid, {})
            cur_no += 1
            P(f"  ── [{cur_no}] 章 {ch.get('chapter_id')} {ch.get('title') or '(未命名)'}")
        lid = lesson.get("lesson_id") or lesson.get("_id")
        P(f"     L{_num(lesson.get('order')):>2}  {lesson.get('title')}  "
          f"[{lid}] sentences={_num(lesson.get('sentence_count'))}")

    # 3) 全量概览
    P("\n[3] 全量记录概览（该学者全集合）")
    all_state_rows = sum(c["state_total"] for c in per.values())
    all_attempt_rows = sum(c["attempt_total"] for c in per.values())
    P(f"  教材内 skill_state 行 = {all_state_rows}；study_attempt 行 = {all_attempt_rows}；"
      f"教材外（其它教材/学科）state {len(agg['outside']['state_rows'])} 行、"
      f"attempt {len(agg['outside']['attempt_rows'])} 行")
    sess = agg["session"]
    P(f"  study_session：全量 {sess['total']}；属本教材 {sess['for_book']}，"
      f"时长合计 {sess['duration_sec_book']}s，≥阈值 {sess['long_count_book']} 个")

    # 3.5) 学者全部教材关联
    P("\n[3.5] 该学者全部 scholar_book（多教材视图）")
    if not agg["all_books"]:
        P("  无任何 scholar_book 记录")
    else:
        for b in agg["all_books"]:
            P(
                f"  {b.get('textbook_id')}  status={b.get('status')}  "
                f"current_lesson={b.get('current_lesson_id') or '-'}  "
                f"chapter={b.get('current_chapter_id') or '-'}  "
                f"total_time={_num(b.get('total_time_spent'))}s  "
                f"last_studied={_ts(b.get('last_studied_at'))}"
            )

    # 4) 每课证据分布
    P("\n[4] 每课证据分布")
    header = (
        f"  {'lesson':<12}{'课':<4}{'order':>5} | "
        f"{'state':>5}{'learned/mstr':>13}{'ghost-like':>11} | "
        f"{'attempt':>7}{'带分':>5}{'带mastery':>10} | 判定"
    )
    P(header)
    P("  " + "-" * (len(header) - 1))
    order_rank = {l.get("lesson_id") or l.get("_id"): l.get("order") for l in lessons}
    for lid, cell in sorted(per.items(), key=lambda kv: (_num(kv[1]["order"]) or 0, kv[0])):
        cls = classify_lesson(cell)
        statuses = cell["state_by_status"]
        n_ghost_status = sum(v for k, v in statuses.items() if k in GHOST_CANDIDATE_STATUSES)
        attempts = cell["attempt_by_status"]
        title = (cell["title"] or "")[:10]
        P(
            f"  {title:<12}{str(order_rank.get(lid, '')):<4}{_num(cell['order']):>5} | "
            f"{cell['state_total']:>5}{cell['state_result_rows']:>13}{n_ghost_status:>11} | "
            f"{cell['attempt_total']:>7}{cell['attempt_with_score']:>5}{cell['attempt_with_mastery']:>10} | "
            f"{cls['level']:<13} {cls['note']}"
        )
        if cell["state_total"]:
            P(
                f"          状态分层: "
                + " ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
                + f"  时间 {_ts(cell['state_min_ts'])} ~ {_ts(cell['state_max_ts'])}"
            )
        if cell["attempt_total"]:
            P(
                f"          attempt 分层: "
                + " ".join(f"{k}={v}" for k, v in sorted(attempts.items()))
                + f"  time_spent={cell['attempt_time_spent']}s"
            )

    # 4.5) 行级小样（每课最多 2 行，核查字段/同批性）
    P("\n[4.5] skill_state 行级小样（前 2 行/课）")
    for lid, cell in sorted(per.items(), key=lambda kv: (_num(kv[1]["order"]) or 0, kv[0])):
        rows = [s for s in agg["raw_states"] if s.get("_lesson") == lid]
        P(f"  《{cell['title']}》(order={cell['order']}) 共 {len(rows)} 行：")
        for s in rows[:2]:
            P(
                f"    {str(s.get('sentence_id'))[:28]:<30} "
                f"skill={s.get('skill_code')} status={s.get('status')} "
                f"att={s.get('attempt_count')} mastery={s.get('mastery_score')} "
                f"progress={s.get('progress')} lesson_id={s.get('lesson_id')} "
                f"created={_ts(s.get('created_at'))} updated={_ts(s.get('updated_at'))} "
                f"last={_ts(s.get('last_studied_at'))}"
            )

    # 4.6) 教材外记录（不属于目标教材，可能是其它教材/旧模型残留）
    P("\n[4.6] 教材外记录（state/attempt 未归属到本教材任何课）")
    out_st = agg["outside"]["state_rows"]
    out_at = agg["outside"]["attempt_rows"]
    if not out_st and not out_at:
        P("  无")
    if out_st:
        by_lesson: dict[str, Counter] = {}
        for s in out_st:
            by_lesson.setdefault(str(s.get("lesson_id") or "no-lesson_id"), Counter())[
                normalize_status(s.get("status"))
            ] += 1
        P(f"  教材外 skill_state {len(out_st)} 行（按 lesson_id × status）：")
        for lid_key, cnt in sorted(by_lesson.items()):
            P(f"    {str(lid_key)[:40]:<42} " + " ".join(f"{k}={v}" for k, v in sorted(cnt.items())))
    if out_at:
        P(f"  教材外 study_attempt {len(out_at)} 行：")
        for a in out_at:
            P(
                f"    {str(a.get('sentence_id'))[:24]:<26} lesson={a.get('lesson_id')} "
                f"skill={a.get('skill_code')} type={a.get('attempt_type')} "
                f"status={a.get('status')} score={a.get('score')} mastery={a.get('mastery')} "
                f"session={a.get('session_id')} ts={_ts(a.get('created_at'))}"
            )

    # 5) 幽灵核验清单
    P("\n[5] 幽灵核验（learning/review_due 行但无强证据的课）")
    ghost_found = False
    for lid, cell in sorted(per.items(), key=lambda kv: (_num(kv[1]["order"]) or 0, kv[0])):
        cls = classify_lesson(cell)
        n_ghost = sum(v for k, v in cell["state_by_status"].items() if k in GHOST_CANDIDATE_STATUSES)
        if cls["level"] == "GHOST_LIKE" and n_ghost:
            ghost_found = True
            P(
                f"  ⚠ 《{cell['title']}》(order={cell['order']}) ghost-like 状态行 {n_ghost} 条："
                f"{cls['note']}"
            )
    if not ghost_found:
        P("  未发现纯幽灵 learning 课（GHOST_LIKE=0）")

    # 结果行无 attempt 佐证（人工复核清单）
    P("\n[6] learned/mastered 行无任何 attempt 佐证（不自动清理，人工复核）")
    found_no_attempt = False
    for lid, cell in sorted(per.items(), key=lambda kv: (_num(kv[1]["order"]) or 0, kv[0])):
        if cell["state_result_rows"] and cell["attempt_total"] == 0:
            found_no_attempt = True
            P(f"  ※ 《{cell['title']}》(order={cell['order']}) 结果行 {cell['state_result_rows']} 条，无 attempt")
    if not found_no_attempt:
        P("  无（所有结果行均有 attempt 佐证或带分）")

    # 7) 结论
    P("\n[7] 结论速览")
    if book:
        cl = book.get("current_lesson_id")
        cl_doc = next((l for l in lessons if (l.get("lesson_id") or l.get("_id")) == cl), None)
        if cl_doc and cl in per:
            cls = classify_lesson(per[cl])
            P(f"  stored 锚点课 = 《{cl_doc.get('title')}》(order={cl_doc.get('order')}) → {cls['level']}")
        else:
            P(f"  stored 锚点课 {cl} 不在目录内 → 孤儿锚点")
        n_real = sum(1 for c in per.values() if classify_lesson(c)["level"].startswith("REAL"))
        n_ghost = sum(1 for c in per.values() if classify_lesson(c)["level"] == "GHOST_LIKE")
        P(f"  真实证据课 {n_real} / {len(per)}；幽灵候选课 {n_ghost} / {len(per)}")
    return "\n".join(out)


async def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 0 证据诊断（只读）")
    parser.add_argument("--scholar-id", required=True)
    parser.add_argument("--textbook-id", required=True)
    parser.add_argument("--min-session-sec", type=int, default=DEFAULT_MIN_SESSION_SEC)
    parser.add_argument("--result-mastery-threshold", type=float, default=DEFAULT_RESULT_MASTERY_THRESHOLD)
    parser.add_argument("--json", default="", help="可选：输出原始聚合 JSON 到文件")
    args = parser.parse_args()

    db = get_db()
    ev = await collect_evidence(db, args.scholar_id, args.textbook_id)
    if not ev["lessons"]:
        print(f"未找到教材 {args.textbook_id} 的 lesson 记录（检查 textbook_id）")
        sys.exit(1)

    agg = aggregate(
        db, ev,
        min_session_sec=args.min_session_sec,
        result_mastery=args.result_mastery_threshold,
    )
    print(render_report(
        args.scholar_id, args.textbook_id, agg,
        result_mastery=args.result_mastery_threshold,
    ))

    if args.json:
        payload = {
            "scholar_id": args.scholar_id,
            "textbook_id": args.textbook_id,
            "scholar_book": agg["book"],
            "lessons": [
                {
                    "lesson_id": c["lesson_id"],
                    "chapter_id": c["chapter_id"],
                    "order": c["order"],
                    "title": c["title"],
                    "sentence_count": c["sentence_count"],
                    "state_by_status": dict(c["state_by_status"]),
                    "state_total": c["state_total"],
                    "state_result_rows": c["state_result_rows"],
                    "state_mastery_hit": c["state_mastery_hit"],
                    "attempt_by_status": dict(c["attempt_by_status"]),
                    "attempt_total": c["attempt_total"],
                    "attempt_with_score": c["attempt_with_score"],
                    "attempt_with_mastery": c["attempt_with_mastery"],
                    "classify": classify_lesson(c)["level"],
                }
                for c in agg["per_lesson"].values()
            ],
            "session": agg["session"],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        print(f"\n原始聚合 JSON 已输出: {args.json}")


if __name__ == "__main__":
    asyncio.run(main())
