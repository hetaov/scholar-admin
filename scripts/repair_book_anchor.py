"""§7 阶段 4：存量锚点证据化重算脚本 repair_book_anchor.py（默认 dry-run 出清单）

背景（docs_v1《沉浸式锚点与learning写入证据化重构方案》§5.5.4 + §7 阶段 4 + §10.8）：
  A/B/C 区幽灵行清理（§10.6/§10.7：skill_state 1957 → 239）后 skill_state 仅剩真实
  证据行 → 锚点无需再按百分比猜真伪。本脚本对现存 scholar_book（A 区已删 3 条
  GHOST_FULL 书后剩 12 条）做存量锚点校准与全量验证：
  - 证据标注（§5.5.1 判据，但行源 = 清理后保留的 skill_state 真实行 + 带分 attempt）：
    课级标签 none / started / done；
  - 决策复用 services/learning/anchor.resolve_book_anchor（§10.8 阶段 1 纯函数，
    与读侧 §5.3 简单规则同源口径：零证据→第 1 课 / 存量锚点真实→信任采纳 /
    孤立尾部(仅存量脏锚钉尾)→回退 / 无锚点→最后真实进行中课 / 全通→完成态或下一课）；
  - 写回契约：current_chapter_id / current_lesson_id 同写课 id、current_group_id 置 null
    （与小程序 _enterNextLesson、repair_ghost_c2.py 归一化同一契约）。

课级标签标注口径（对齐 anchor.py docstring 的 tags 语义 + 读侧 aggregate_progress
progress=100 同义）：
  none    = 课无任何 skill_state 行且无带分 attempt（无真实证据）
  started = 有真实证据但未通关：无结果态行 / 课内句子未全覆盖结果态 /
            内容句子目录缺失（保守不判通关）/ 行无法归属课句子 / 仅带分 attempt
  done    = 课已通关：课内每句（sentence_v2）都有结果态行（normalize_status ∈
            {learned, mastered} 或 mastery_score ≥ 阈值），且全部行可归属课内句子

注意：study_session 仅教材级粒度 → 只在书头展示佐证，不参与课级标签；
本脚本只校准 scholar_book 锚点，不删任何 skill_state（幽灵已由 A/B/C 区清理）。

用法（scholar-admin 根目录，CloudBase 凭据由 .env 自动加载）：
  python scripts/repair_book_anchor.py --scholar-id S --textbook-id T     # 单本 dry-run
  python scripts/repair_book_anchor.py --all                              # 全部现存 scholar_book dry-run
  python scripts/repair_book_anchor.py --scholar-id S --textbook-id T --apply
  python scripts/repair_book_anchor.py --all --apply
  python scripts/repair_book_anchor.py --all --json /tmp/ba.json          # 清单 JSON

退出码：0 = 成功（dry-run 或 apply）；仅 DB/参数错误 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from repair_lesson_order import plan_book_order  # noqa: E402  目录序（课序口径单源）
from services.dependencies import get_db  # noqa: E402
from services.learning.anchor import (  # noqa: E402  证据化锚点解析器（阶段 1）
    TAG_DONE,
    TAG_NONE,
    TAG_STARTED,
    lesson_key,
    resolve_book_anchor,
)
from services.models.content import (  # noqa: E402
    SENTENCE_V2,
    TEXTBOOK_V2,
    get_chapters,
    get_lessons_by_textbook,
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
    normalize_status,
)
from services.models.scholar_book import (  # noqa: E402
    SCHOLAR_BOOK,
    get_scholar_book,
    scholar_book_id,
)

RESULT_STATUSES = {STATUS_LEARNED, STATUS_MASTERED}


def _num(v, default=0):
    try:
        if v is None:
            return default
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return default


def _fmt_scholar(sid: str) -> str:
    return sid[:12]


def _is_result_row(row: dict, result_mastery_threshold: float) -> bool:
    """行级结果态判定（§5.5.1 强证据）：结果性状态 或 mastery_score ≥ 阈值。"""
    if normalize_status(row.get("status")) in RESULT_STATUSES:
        return True
    return _num(row.get("mastery_score")) >= result_mastery_threshold


# ---------------------------------------------------------------------------
# 纯函数：课级标签 / 决策 diff / 排序（可单测，不触 DB）
# ---------------------------------------------------------------------------


def sort_lessons(lessons: list[dict], chapters: list[dict]) -> list[dict]:
    """目录序课列表（锚点决策口径，anchor.py 同源）。

    DB lesson.order 已是 1..N 唯一递增（repair_lesson_order 已对全库执行）→ 保持
    order ASC 返回序；否则（未修复教材/旧数据）回退按标题课号重排
    （repair_lesson_order.plan_book_order，孤儿课排最后、无课号排桶末）。
    """
    if not lessons:
        return []
    orders = [_num(l.get("order")) for l in lessons]
    if (len(set(orders)) == len(orders)
            and sorted(orders) == list(range(1, len(lessons) + 1))):
        return list(lessons)
    return plan_book_order(chapters, lessons)


def assign_row(row: dict, lesson_ids: set, sent2lesson: dict) -> str | None:
    """行（skill_state / study_attempt）归属课 id；失败返回 None（不属于本教材目录）。

    优先级与 scan.resolve_textbook / diagnose_book_evidence 同源：
    行内 lesson_id ∈ 本教材课集合 → 用之；否则 sentence_id 反查；都失败 → None。
    """
    lid = row.get("lesson_id")
    if lid and lid in lesson_ids:
        return lid
    sid = row.get("sentence_id")
    if sid:
        hit = sent2lesson.get(sid)
        if hit:
            return hit
    return None


def compute_lesson_tags(
    lessons_sorted: list[dict],
    *,
    rows_by_lesson: dict[str, list[dict]],
    scored_attempts_by_lesson: dict[str, int],
    sentences_by_lesson: dict[str, list[str]],
    result_mastery_threshold: float = 60,
) -> tuple[dict[str, str], dict[str, str]]:
    """课级证据标签（anchor.py tags 语义：none/started/done）。

    rows_by_lesson：课内已归属的 skill_state 行（幽灵清理后即真实证据）；
    scored_attempts_by_lesson：课内带分 study_attempt 数（无 skill_state 时的兜底真实信号）；
    sentences_by_lesson：课内句子 id 列表（sentence_v2 内容结构，判 done 的分母）。

    返回 (tags, detail)；tags key = lesson_id，值 ∈ {none, started, done}。
    """
    tags: dict[str, str] = {}
    detail: dict[str, str] = {}
    for lesson in lessons_sorted:
        lid = lesson_key(lesson)
        rows = rows_by_lesson.get(lid) or []
        scored = int(scored_attempts_by_lesson.get(lid) or 0)
        sids = list(sentences_by_lesson.get(lid) or [])

        if not rows and scored == 0:
            tags[lid] = TAG_NONE
            detail[lid] = "无 skill_state 行、无带分 attempt → 无真实证据"
            continue
        if not rows:
            tags[lid] = TAG_STARTED
            detail[lid] = f"无 skill_state 行但带分 attempt {scored} → 真实已开始（不判通关）"
            continue
        if not sids:
            tags[lid] = TAG_STARTED
            detail[lid] = f"skill_state {len(rows)} 行但课句子目录缺失 → 保守不判通关"
            continue

        sid_set = set(sids)
        rows_in_sids = [r for r in rows if r.get("sentence_id") in sid_set]
        unassignable = len(rows) - len(rows_in_sids)
        result_sids = {
            r.get("sentence_id")
            for r in rows_in_sids
            if _is_result_row(r, result_mastery_threshold)
        }
        all_sent_result = all(sid in result_sids for sid in sids)
        if unassignable == 0 and all_sent_result:
            tags[lid] = TAG_DONE
            detail[lid] = f"课内 {len(sids)} 句全部达结果态（skill_state {len(rows)} 行）→ 通关"
        else:
            extra = f"，另有 {unassignable} 行无法归属课句子" if unassignable else ""
            tags[lid] = TAG_STARTED
            detail[lid] = (
                f"句 {len(result_sids)}/{len(sids)} 达结果态{extra}（skill_state {len(rows)} 行）"
                f" → 真实进行中"
            )
    return tags, detail


def plan_single_book(book: dict, lessons_sorted: list[dict], tags: dict[str, str]) -> dict:
    """存量锚点 vs 证据重算的 diff 计划（纯函数）。

    返回：{stored_lesson, stored_group, resolved_lesson, resolved_group, order, title,
          reason, note, book_finished, changed}。
    changed = 存量锚点课 != 重算课，或存量 group 级锚非空（需归一置 null）。
    """
    stored_lesson = (
        (book or {}).get("current_lesson_id")
        or (book or {}).get("current_chapter_id")
        or None
    )
    resolved = resolve_book_anchor(lessons_sorted=lessons_sorted, tags=tags, stored=book)
    after_lesson = resolved["lesson_id"]
    # 归一化写回契约：chapter/lesson 同写课 id、group 置 null（_enterNextLesson/C2 同一契约）。
    # changed = 存量三字段未完全归一 —— 覆盖「锚点课不一致 / 半锚(lesson 空 chapter 有) /
    # group 非空 / chapter 缺失或指向他值」全部需写回形态。
    want = {
        "current_chapter_id": after_lesson,
        "current_lesson_id": after_lesson,
        "current_group_id": None,
    }
    cur = {
        "current_chapter_id": (book or {}).get("current_chapter_id"),
        "current_lesson_id": (book or {}).get("current_lesson_id"),
        "current_group_id": (book or {}).get("current_group_id"),
    }
    changed = cur != want
    return {
        "stored_lesson": stored_lesson,
        "resolved_lesson": after_lesson,
        "order": resolved.get("order"),
        "title": resolved.get("title", ""),
        "reason": resolved["reason"],
        "note": resolved["note"],
        "book_finished": bool(resolved.get("book_finished")),
        "changed": bool(changed),
    }


# ---------------------------------------------------------------------------
# 只读聚合 / 主流程
# ---------------------------------------------------------------------------


async def _load_sentences_by_lesson(db, lesson_ids: list[str]) -> dict[str, list[str]]:
    """本教材全部句子按课分组（sentence_id 列表）。"""
    out: dict[str, list[str]] = {lid: [] for lid in lesson_ids}
    uniq = sorted({lid for lid in lesson_ids if lid})
    for i in range(0, len(uniq), 100):
        res = await db.query(
            SENTENCE_V2,
            where={"lesson_id": {"$in": uniq[i:i + 100]}},
            select={"sentence_id": 1, "lesson_id": 1},
            limit=100,
        )
        for s in res.get("records", []):
            sid = s.get("sentence_id") or s.get("_id")
            lid = s.get("lesson_id")
            if sid and lid in out:
                out[lid].append(sid)
    return out


async def repair_one_book(
    db,
    scholar_id: str,
    textbook_id: str,
    *,
    result_mastery_threshold: float,
) -> dict | None:
    """处理单本（scholar × textbook）的锚点重算计划（只读）；scholar_book 不存在返回 None。"""
    book = await get_scholar_book(db, scholar_id=scholar_id, textbook_id=textbook_id)
    if book is None:
        return None  # 无 scholar_book 记录（如 A 区已删的 GHOST_FULL 书）→ 冷启动自动重建，不处理

    textbook_title = textbook_id
    try:
        tb = await db.query(TEXTBOOK_V2, where={"_id": textbook_id}, limit=1)
        recs = tb.get("records", []) if isinstance(tb, dict) else []
        if recs:
            textbook_title = recs[0].get("title") or textbook_id
    except Exception:
        pass

    chapters = await get_chapters(db, textbook_id)
    lessons_raw = await get_lessons_by_textbook(db, textbook_id)
    if not lessons_raw:
        # 教材目录无课（内容缺失/下线）：解析器无法决策 → 不猜存量锚点，跳过并报告人工介入
        return {
            "skipped": True,
            "scholar_id": scholar_id,
            "textbook_id": textbook_id,
            "textbook_title": textbook_title,
            "skipped_reason": "教材无 lesson 目录（内容缺失/下线），存量锚点不动、待人工复核",
        }
    lessons_sorted = sort_lessons(lessons_raw, chapters)
    lesson_ids = [lesson_key(l) for l in lessons_sorted]
    lesson_id_set = set(lesson_ids)

    # 句子目录（done 判定分母 + sentence→lesson 反查）
    sentences_by_lesson = await _load_sentences_by_lesson(db, lesson_ids)
    sent2lesson: dict[str, str] = {}
    for lid, sids in sentences_by_lesson.items():
        for sid in sids:
            sent2lesson[sid] = lid

    # 该学者全量学习行 → 归属本教材目录课
    states = await query_all_pages(db, collection=SKILL_STATE, where={"scholar_id": scholar_id})
    attempts = await query_all_pages(db, collection=STUDY_ATTEMPT, where={"scholar_id": scholar_id})

    rows_by_lesson: dict[str, list[dict]] = {lid: [] for lid in lesson_ids}
    scored_attempts_by_lesson: dict[str, int] = {lid: 0 for lid in lesson_ids}
    for st in states:
        key = assign_row(st, lesson_id_set, sent2lesson)
        if key:
            rows_by_lesson.setdefault(key, []).append(st)
    for a in attempts:
        key = assign_row(a, lesson_id_set, sent2lesson)
        if key and (a.get("score") is not None or a.get("mastery") is not None):
            scored_attempts_by_lesson[key] = scored_attempts_by_lesson.get(key, 0) + 1

    tags, tag_detail = compute_lesson_tags(
        lessons_sorted,
        rows_by_lesson=rows_by_lesson,
        scored_attempts_by_lesson=scored_attempts_by_lesson,
        sentences_by_lesson=sentences_by_lesson,
        result_mastery_threshold=result_mastery_threshold,
    )

    plan = plan_single_book(book, lessons_sorted, tags)

    # 书级会话佐证（展示用，不参与课级标签）
    sessions = await query_all_pages(
        db,
        collection=STUDY_SESSION,
        where={"scholar_id": scholar_id, "textbook_id": textbook_id},
    )
    long_sessions = sum(1 for s in sessions if _num(s.get("duration_sec")) >= 60)

    return {
        "skipped": False,
        "scholar_id": scholar_id,
        "textbook_id": textbook_id,
        "textbook_title": textbook_title,
        "book": book,
        "state_rows": sum(len(v) for v in rows_by_lesson.values()),
        "attempt_rows": sum(v for v in scored_attempts_by_lesson.values()),
        "long_sessions": long_sessions,
        "lessons_sorted": lessons_sorted,
        "tags": tags,
        "tag_detail": tag_detail,
        "plan": plan,
    }


def _lesson_label(lesson: dict) -> str:
    order = lesson.get("order")
    title = str(lesson.get("title") or "")[:26]
    prefix = f"L{_num(order)}" if order is not None and order != 0 else "L-"
    return f"{prefix} {title}"


async def render_report(
    items: list[dict],
    *,
    skipped: list[dict],
    no_book_count: int,
    apply: bool,
) -> str:
    out: list[str] = []
    P = out.append
    mode = "WRITE" if apply else "DRY-RUN"
    P("=" * 100)
    P(f"存量锚点证据化重算 repair_book_anchor.py（{mode}，复用 anchor.py 解析器）")
    P("=" * 100)
    changed_items: list[dict] = []
    for it in items:
        plan = it["plan"]
        P(f"\n[book] {_fmt_scholar(it['scholar_id'])} × {str(it['textbook_id'])[:20]} "
          f"《{str(it['textbook_title'])[:30]}》"
          f"  status={it['book'].get('status')}  state={it['state_rows']} "
          f"带分attempt={it['attempt_rows']} session≥60s={it['long_sessions']}")
        P(f"  stored : chapter={str(it['book'].get('current_chapter_id'))[:18] or '-'}  "
          f"lesson={str(it['book'].get('current_lesson_id'))[:18] or '-'}  "
          f"group={str(it['book'].get('current_group_id'))[:12] or '-'}")
        tag_cells = []
        for lesson in it["lessons_sorted"]:
            lid = lesson_key(lesson)
            tag_cells.append(f"{it['tags'].get(lid, 'none')}({_lesson_label(lesson).split(' ')[0][:6]})")
        P(f"  tags  : " + "  ".join(tag_cells[:14]) + (" …" if len(tag_cells) > 14 else ""))
        res = plan["resolved_lesson"]
        res_label = next(
            (_lesson_label(l) for l in it["lessons_sorted"] if lesson_key(l) == res), res
        )
        P(f"  resolved: {res_label}  reason={plan['reason']}"
          + ("  book_finished" if plan["book_finished"] else ""))
        P(f"    {plan['note']}")
        if plan["changed"]:
            changed_items.append(it)
            P(f"  → 变化：stored={str(plan['stored_lesson'])[:22] or '∅'} "
              f"→ resolved={res}")
        else:
            P(f"  → 零变化（存量锚点与证据重算一致，幂等）")
        for lesson in it["lessons_sorted"]:
            lid = lesson_key(lesson)
            if it["tags"].get(lid) != TAG_NONE:
                P(f"      {_lesson_label(lesson)} [{it['tags'][lid]}] {it['tag_detail'][lid]}")

    if skipped:
        P(f"\n[跳过] {len(skipped)} 本（教材目录无课，存量锚点不动、待人工复核）")
        for s in skipped:
            P(f"  {_fmt_scholar(s['scholar_id'])} × {str(s['textbook_id'])[:20]} "
              f"《{str(s['textbook_title'])[:30]}》  {s['skipped_reason']}")

    P(f"\n[合计] 现存 scholar_book {len(items) + len(skipped) + no_book_count} 条："
      f"已校准 {len(items)} 本、需调整 {len(changed_items)} 本、"
      f"跳过 {len(skipped)} 本、无 scholar_book 记录 {no_book_count} 条")
    if changed_items:
        P("  变更明细（reason 分布: "
          + "  ".join(f"{k}={v}" for k, v in sorted(
              Counter(i["plan"]["reason"] for i in changed_items).items())) + "）")
        for it in changed_items:
            plan = it["plan"]
            P(f"    {_fmt_scholar(it['scholar_id'])} × {str(it['textbook_id'])[:20]}  "
              f"stored={str(plan['stored_lesson'])[:16] or '∅'} → "
              f"{str(plan['resolved_lesson'])[:16]}  reason={plan['reason']}")
    if not apply:
        P("dry-run 未写库；过目后加 --apply 执行")
    return "\n".join(out)


async def apply_book_anchors(db, items: list[dict]) -> dict:
    """--apply：写回锚点（chapter/lesson 同写课 id、group 置 null）。"""
    now = int(time.time())
    total = 0
    detail: list[dict] = []
    for it in items:
        plan = it["plan"]
        if not plan["changed"]:
            continue
        res = await db.update(
            collection=SCHOLAR_BOOK,
            where={"_id": it["book"]["_id"]},
            data={"$set": {
                "current_chapter_id": plan["resolved_lesson"],
                "current_lesson_id": plan["resolved_lesson"],
                "current_group_id": None,
                "updated_at": now,
            }},
            multi=False,
        )
        matched = res.get("matched_count", 0) if isinstance(res, dict) else 0
        total += 1 if matched else 0
        detail.append({
            "book_id": it["book"]["_id"],
            "stored": plan["stored_lesson"],
            "resolved": plan["resolved_lesson"],
            "reason": plan["reason"],
            "matched": matched,
        })
    return {"total": total, "detail": detail}


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="存量锚点证据化重算（默认 dry-run；决策复用 services/learning/anchor.py）"
    )
    parser.add_argument("--scholar-id", default="", help="限定学者（需配 --textbook-id）")
    parser.add_argument("--textbook-id", default="", help="限定教材")
    parser.add_argument("--result-mastery-threshold", type=float, default=60,
                        help="mastery_score 结果态阈值（默认 60）")
    parser.add_argument("--all", action="store_true", help="处理全部现存 scholar_book（缺省行为）")
    parser.add_argument("--apply", action="store_true", help="写库；缺省仅 dry-run")
    parser.add_argument("--json", default="", help="可选：输出清单 JSON")
    args = parser.parse_args()

    if args.scholar_id and not args.textbook_id:
        parser.error("--scholar-id 需配合 --textbook-id 使用")
    if args.textbook_id and not args.scholar_id:
        parser.error("--textbook-id 需配合 --scholar-id 使用")

    db = get_db()
    if args.scholar_id:
        targets = [{"scholar_id": args.scholar_id, "textbook_id": args.textbook_id}]
    else:
        books = await query_all_pages(db, collection=SCHOLAR_BOOK, where={})
        targets = [
            {"scholar_id": b.get("scholar_id"), "textbook_id": b.get("textbook_id")}
            for b in books if b.get("scholar_id") and b.get("textbook_id")
        ]

    if not targets:
        print("未找到任何 scholar_book 记录")
        return

    items: list[dict] = []
    skipped: list[dict] = []
    no_book_count = 0
    for t in targets:
        item = await repair_one_book(
            db,
            t["scholar_id"],
            t["textbook_id"],
            result_mastery_threshold=args.result_mastery_threshold,
        )
        if item is None:
            no_book_count += 1
        elif item.get("skipped"):
            skipped.append(item)
        else:
            items.append(item)

    report = await render_report(
        items, skipped=skipped, no_book_count=no_book_count, apply=args.apply
    )
    print(report)

    if args.json:
        payload = {
            "meta": {
                "mode": "WRITE" if args.apply else "DRY-RUN",
                "result_mastery_threshold": args.result_mastery_threshold,
                "targets": len(targets),
                "processed": len(items),
                "changed": sum(1 for i in items if i["plan"]["changed"]),
                "skipped": len(skipped),
                "no_scholar_book": no_book_count,
            },
            "books": [
                {
                    "scholar_id": i["scholar_id"],
                    "textbook_id": i["textbook_id"],
                    "textbook_title": i["textbook_title"],
                    "stored": {
                        "lesson": i["book"].get("current_lesson_id"),
                        "chapter": i["book"].get("current_chapter_id"),
                        "group": i["book"].get("current_group_id"),
                    },
                    "resolved": {
                        "lesson": i["plan"]["resolved_lesson"],
                        "group": None,
                        "reason": i["plan"]["reason"],
                        "note": i["plan"]["note"],
                        "book_finished": i["plan"]["book_finished"],
                    },
                    "changed": i["plan"]["changed"],
                    "lessons": [
                        {
                            "lesson_id": lesson_key(l),
                            "order": l.get("order"),
                            "title": l.get("title", ""),
                            "tag": i["tags"][lesson_key(l)],
                            "detail": i["tag_detail"][lesson_key(l)],
                        }
                        for l in i["lessons_sorted"]
                    ],
                }
                for i in items
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
        print(f"\n清单 JSON 已输出: {args.json}")

    if args.apply:
        r = await apply_book_anchors(db, items)
        print(f"\n[WRITE] 锚点回写 {r['total']}/{sum(1 for i in items if i['plan']['changed'])} 本")
        for d in r["detail"]:
            if d["matched"] != 1:
                print(f"  ! {d['book_id']} 期望 matched=1 实际 {d['matched']}")


if __name__ == "__main__":
    asyncio.run(main())
