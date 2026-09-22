"""清理 NCE Book 2 lesson 重复记录 — 按 title 去重 + 重排 order

逻辑:
1. 查 textbook_id='tb_e6bce2b577554d02' 下所有 lesson(带 limit)
2. 按 title 分组;每组取"关联 sentence_v2 数 > 0"的 lesson 中 sentence 数最多的保留,
   其余 lesson + 关联 sentence_v2 全删
3. 全 sentence_v2 = 0 的 lesson(空 lesson)整体删除,不保留
4. 重排 order:按 title 中的 Lesson 编号(1~96)更新 order 字段

用法:
    python scripts/cleanup_nce2_lessons.py --dry-run    # 只预览,不删
    python scripts/cleanup_nce2_lessons.py              # 真删
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

# 让脚本能从项目根目录导入 services 包
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.infra.dependencies import get_db
from services.models.content import LESSON, SENTENCE_V2, TEXTBOOK_V2

logger = logging.getLogger("cleanup_nce2")

TB_ID = "tb_e6bce2b577554d02"

LESSON_NO_RE = re.compile(r"Lesson\s+(\d+)", re.IGNORECASE)


async def fetch_lessons(db) -> list[dict]:
    """拉所有 lesson,带 limit。"""
    res = await db.query(
        collection=LESSON, where={"textbook_id": TB_ID}, limit=200
    )
    return res.get("records", [])


async def count_sentences(db, lesson_id: str) -> int:
    res = await db.query(
        collection=SENTENCE_V2,
        where={"lesson_id": lesson_id},
        select={"_id": 1},
        limit=2000,
    )
    return len(res.get("records", []))


async def fetch_all_sentence_ids(db, lesson_id: str) -> list[str]:
    res = await db.query(
        collection=SENTENCE_V2,
        where={"lesson_id": lesson_id},
        select={"_id": 1},
        limit=2000,
    )
    return [r["_id"] for r in res.get("records", [])]


async def delete_lesson_and_sentences(db, lesson_id: str, dry_run: bool) -> tuple[int, int]:
    """删 lesson + 关联 sentence_v2,返回 (删 lesson 数, 删 sentence 数)。"""
    sent_ids = await fetch_all_sentence_ids(db, lesson_id)
    if dry_run:
        return 1, len(sent_ids)
    # 删 sentence
    if sent_ids:
        await db.delete(collection=SENTENCE_V2, where={"lesson_id": lesson_id}, multi=True)
    # 删 lesson
    await db.delete(collection=LESSON, where={"_id": lesson_id}, multi=False)
    return 1, len(sent_ids)


def lesson_no_from_title(title: str) -> int | None:
    m = LESSON_NO_RE.search(title or "")
    return int(m.group(1)) if m else None


async def main(dry_run: bool):
    db = get_db()

    # 1. 拉 textbook(确认存在)
    tb = await db.query(
        collection=TEXTBOOK_V2, where={"_id": TB_ID}, limit=5
    )
    if not tb.get("records"):
        print(f"ERROR: textbook_v2 _id={TB_ID} 不存在")
        return 1
    print(f"textbook: {tb['records'][0].get('title')} ({TB_ID})")

    # 2. 拉所有 lesson
    lessons = await fetch_lessons(db)
    print(f"\n现有 lesson 记录: {len(lessons)} 条")

    # 3. 统计每条 lesson 关联的 sentence 数
    print("统计每条 lesson 的 sentence_v2 数量...")
    lesson_with_count = []
    for i, les in enumerate(lessons, 1):
        cnt = await count_sentences(db, les["_id"])
        lesson_with_count.append({**les, "_sentence_count": cnt})
        if i % 20 == 0:
            print(f"  进度 {i}/{len(lessons)}")

    # 4. 按 title 分组
    groups: dict[str, list[dict]] = defaultdict(list)
    for les in lesson_with_count:
        groups[les.get("title", "")].append(les)

    # 5. 决定要删的 lesson
    to_delete: list[dict] = []
    keep: list[dict] = []
    for title, items in groups.items():
        if len(items) == 1:
            # 单条,但 sentence=0 也要删(用户要求)
            if items[0]["_sentence_count"] == 0:
                to_delete.append(items[0])
                print(
                    f"  [删空] {title}: sentence=0,删除"
                )
            else:
                keep.append(items[0])
        else:
            # 多条:按 sentence 数降序,sentence>0 的保留第一条,其余删
            sorted_items = sorted(
                items, key=lambda x: -x["_sentence_count"]
            )
            # 全为 0:全删
            if sorted_items[0]["_sentence_count"] == 0:
                for it in sorted_items:
                    to_delete.append(it)
                print(
                    f"  [全删] {title}: {len(items)} 条,sentence 全 0"
                )
                continue
            # 保留 sentence 最多的一条(并列时取 _id 字典序最小,稳定)
            kept = sorted_items[0]
            keep.append(kept)
            for it in sorted_items[1:]:
                to_delete.append(it)
            print(
                f"  [去重] {title}: 保留 sentence={kept['_sentence_count']}"
                f"({_id_short(kept['_id'])}),删 {len(sorted_items) - 1} 条"
            )

    print(f"\n待删 lesson: {len(to_delete)} 条")
    print(f"保留 lesson: {len(keep)} 条")
    total_sents_to_del = sum(it["_sentence_count"] for it in to_delete)
    print(f"待删 sentence_v2: {total_sents_to_del} 条")

    if dry_run:
        print("\n[dry-run] 不执行删除。详细删除列表:")
        for it in to_delete:
            print(
                f"  - lesson _id={it['_id']} title={it.get('title')} "
                f"sentence={it['_sentence_count']} order={it.get('order')}"
            )
        # 也打印保留列表,核对覆盖 1-96
        keep_nos = sorted(
            n for n in (lesson_no_from_title(k.get("title", "")) for k in keep) if n
        )
        print(f"\n保留 lesson 的 Lesson 编号 ({len(keep_nos)} 条):")
        missing = set(range(1, 97)) - set(keep_nos)
        if missing:
            print(f"  ⚠ 缺失 Lesson: {sorted(missing)}")
        else:
            print(f"  ✓ Lesson 1-96 全覆盖")
        print(f"  编号列表: {keep_nos}")
        return 0

    # 6. 执行删除
    print("\n开始删除...")
    total_deleted_lessons = 0
    total_deleted_sents = 0
    for i, it in enumerate(to_delete, 1):
        d_les, d_sent = await delete_lesson_and_sentences(db, it["_id"], dry_run=False)
        total_deleted_lessons += d_les
        total_deleted_sents += d_sent
        if i % 5 == 0 or i == len(to_delete):
            print(f"  删除进度 {i}/{len(to_delete)}")

    print(
        f"\n删除完成:lesson {total_deleted_lessons} 条,"
        f"sentence_v2 {total_deleted_sents} 条"
    )

    # 7. 重排 order:按 Lesson 编号升序
    print("\n重排 order...")
    keep_sorted = sorted(
        keep, key=lambda x: (lesson_no_from_title(x.get("title", "")) or 9999, x["_id"])
    )
    for new_order, les in enumerate(keep_sorted, 1):
        old_order = les.get("order")
        if old_order == new_order:
            continue
        await db.update(
            collection=LESSON,
            where={"_id": les["_id"]},
            data={"$set": {"order": new_order}},
            multi=False,
        )
        # 只打印变化了的
        print(
            f"  {les.get('title')}: order {old_order} → {new_order}"
        )

    print(f"\n✓ 清理完成,最终 lesson 数: {len(keep_sorted)}")
    return 0


def _id_short(_id: str) -> str:
    return _id[-8:] if _id else "?"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="清理 NCE2 lesson 重复")
    ap.add_argument(
        "--dry-run", action="store_true", help="只预览,不删除"
    )
    ap.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    sys_rc = asyncio.run(main(args.dry_run))
    sys.exit(sys_rc)
