"""一次性修复脚本：按标题课号重写 lesson.order（教材内唯一递增 → DB 层 order ASC 统一）

背景（2026-09-03，统一修复「换教材即复现首个任务=第 8 课」问题）：
  旧导入管道对每本教材的所有 lesson 写死 order=1（快照证实例：tb_5aefa2dee7e34546
  的 8 个 lesson 全部 "order": 1），导致 DB order ASC 查询 / 接口透传都无法还原
  目录顺序。本脚本把每本教材的 lesson 按「章节目录序 + 标题课号」重写为 1..N 唯一递增：
  - 有章教材：章按 chapter.order 升序，章内课按标题课号升序（孤儿课排最后）；
  - 无章教材（lesson 直挂 book）：全书课按标题课号升序；
  - 跨章分配全局 order = 1..N：任何 book 级 order ASC 查询（get_lessons_by_textbook）
    得到的即为目录顺序；前端 fetchBookDetail 透传 order 后可单字段还原。
  标题课号解析与小程序前端 parseLessonNumber 同一口径：
    "Lesson 4" / "第 4 课" / "UNIT 3" / "Module 1 Unit 2"(取 Module) / "章节2" / "1. xxx" / "8、标题"
  标题解析不到课号的课（如 "Review Let's Go Camping!"）保持原相对顺序排同桶末尾，
  并计入 unresolved 报告（供人工复核）。

用法（scholar-admin 项目根目录，需 CloudBase 凭据，可重复执行）：
    python scripts/repair_lesson_order.py                          # 干跑：仅输出修复计划
    python scripts/repair_lesson_order.py --apply                  # 写库
    python scripts/repair_lesson_order.py --textbook-id tb_xxx     # 只评估指定教材
    python scripts/repair_lesson_order.py --textbook-id tb_xxx --apply
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.models.content import (  # noqa: E402
    LESSON,
    TEXTBOOK_V2,
    get_chapters,
    get_lessons_by_textbook,
    query_all_pages,
)

# 与小程序前端 parseLessonNumber 对齐（services/task/immersive.js）：
# 关键词+数字："Lesson 4" / "第 4 课" / "UNIT 3" / "Module 1 Unit 2" / "章节2"
_KEYWORD_NO_RE = re.compile(r"(?:Lesson|第|章节|Module|UNIT|Unit)\s*(\d+)", re.IGNORECASE)
# 阿拉伯数字开头 + 分隔符："1. xxx" / "8、标题" / "12：标题"
_LEADING_NO_RE = re.compile(r"^\s*(\d{1,3})\s*[.、．:：\-—]\s*")


def parse_lesson_no(title) -> int | None:
    """从课程标题解析课号；无课号返回 None（与前端返回 0 等价）。"""
    text = str(title or "")
    m = _KEYWORD_NO_RE.search(text)
    if m:
        return int(m.group(1))
    m = _LEADING_NO_RE.match(text)
    if m:
        return int(m.group(1))
    return None


def _order_val(doc) -> int:
    """读取 order（无效/缺失按 0，与 models.content 各 getter 一致）。"""
    v = doc.get("order")
    return v if isinstance(v, int) and not isinstance(v, bool) and v > 0 else 0


def plan_book_order(chapters: list[dict], lessons: list[dict]) -> list[dict]:
    """计算一本书修复后的目录序 lesson 列表（不触库，纯函数可单测）。

    排序 key（逐级稳定）：
      (孤儿标记, 章序 rank, 课号缺失标记, 课号, 原数组下标)
    - 孤儿课（chapter_id 不在章表）排最后；
    - 章序按 chapter.order 升序（order 缺失/重复时按 DB 返回序兜底，即原下标）；
    - 章内按标题课号升序，解析不到的课保持原相对顺序排章内末尾；
    - 同课号保持原相对顺序（Python sort 稳定）。
    """
    if not lessons:
        return []

    # 章序 rank：chapters 已按 order ASC 返回；防御性重排一次
    chapters_sorted = sorted(
        enumerate(chapters),
        key=lambda item: (_order_val(item[1]) or 0, item[0]),
    )
    rank_of = {c.get("chapter_id"): rank for rank, (_, c) in enumerate(chapters_sorted)}

    def sort_key(item):
        lesson, index = item
        cid = lesson.get("chapter_id")
        no = parse_lesson_no(lesson.get("title"))
        no_flag = 1 if no is None else 0
        if chapters:
            # 有章教材：先按章序（孤儿课排最后），章内按标题课号（无课号保持原序排章末）
            orphan = 0 if cid in rank_of else 1
            return (orphan, rank_of.get(cid, 0), no_flag, no or 0, index)
        # 无章教材：全部按标题课号排（无课号保持原序排末尾）
        return (0, 0, no_flag, no or 0, index)

    ordered = [lesson for lesson, _ in sorted(
        ((l, i) for i, l in enumerate(lessons)),
        key=sort_key,
    )]
    return ordered


def build_plan(chapters: list[dict], lessons: list[dict]) -> list[dict]:
    """生成逐课更新计划：[{lesson_id, title, chapter_id, old_order, new_order}]。"""
    ordered = plan_book_order(chapters, lessons)
    plan = []
    for new_order, lesson in enumerate(ordered, start=1):
        lid = lesson.get("lesson_id") or lesson.get("_id") or ""
        plan.append({
            "lesson_id": lid,
            "chapter_id": lesson.get("chapter_id"),
            "title": lesson.get("title", ""),
            "old_order": _order_val(lesson),
            "new_order": new_order,
        })
    return plan


async def repair_book(db, textbook: dict, *, apply: bool) -> dict:
    """处理单本教材，返回报告 dict。"""
    tid = textbook.get("_id")
    title = textbook.get("title", "")
    chapters = await get_chapters(db, tid)
    lessons = await get_lessons_by_textbook(db, tid)
    report = {
        "textbook_id": tid,
        "textbook_title": title,
        "chapters": len(chapters),
        "lessons": len(lessons),
        "changed": 0,
        "unresolved": [],
        "applied": apply,
    }
    if not lessons:
        return report

    plan = build_plan(chapters, lessons)
    updates = [p for p in plan if p["old_order"] != p["new_order"]]
    report["changed"] = len(updates)
    report["unresolved"] = [
        {"lesson_id": p["lesson_id"], "title": p["title"]}
        for p in plan
        if parse_lesson_no(p["title"]) is None
    ]

    mode = "WRITE" if apply else "DRY-RUN"
    print(f"\n[{mode}] {tid} 《{title}》 chapters={len(chapters)} lessons={len(lessons)}")
    for p in plan:
        flag = "" if p["old_order"] == p["new_order"] else "  <- change"
        print(f"    order {p['old_order']:>3} -> {p['new_order']:>3}  {p['title']}{flag}")

    if apply:
        now = int(time.time())
        for p in updates:
            await db.update(
                LESSON,
                where={"_id": p["lesson_id"]},
                data={"$set": {"order": p["new_order"], "updated_at": now}},
                multi=False,
            )
        print(f"    已写库 {len(updates)} 条（order + updated_at）")
    else:
        print(f"    需更新 {len(updates)} 条（加 --apply 写库）")
    return report


async def main() -> None:
    parser = argparse.ArgumentParser(description="按标题课号重写 lesson.order（教材内唯一递增）")
    parser.add_argument(
        "--textbook-id", default="",
        help="只处理指定教材（缺省处理全部 textbook_v2）",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="写库；缺省仅干跑输出修复计划",
    )
    args = parser.parse_args()

    db = get_db()
    where = {"_id": args.textbook_id} if args.textbook_id else {}
    textbooks = await query_all_pages(db, collection=TEXTBOOK_V2, where=where)
    if not textbooks:
        print(f"未找到教材: {args.textbook_id or '（空集合）'}")
        return

    total_changed = 0
    for tb in textbooks:
        report = await repair_book(db, tb, apply=args.apply)
        total_changed += report["changed"]
        if report["unresolved"]:
            print(
                f"  ! {len(report['unresolved'])} 课标题无课号（保持原序，请人工复核）："
                + ", ".join(u["title"] for u in report["unresolved"])
            )

    print(
        f"\n完成：教材 {len(textbooks)} 本，共需更新 {total_changed} 条 "
        + ("（已写库）" if args.apply else "（dry-run，未写库；确认后加 --apply 执行）")
    )


if __name__ == "__main__":
    asyncio.run(main())
