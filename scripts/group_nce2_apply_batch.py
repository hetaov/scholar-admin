"""NCE Book 2 句子分组分批 apply 脚本 — 调 LLM 分组后写 sentence_group + 回写 sentence_v2.group_id

用法::

    # 默认 NCE Book 2,每批 5 课,批间 5s
    python scripts/group_nce2_apply_batch.py

    # 自定义批次大小 + 间隔
    python scripts/group_nce2_apply_batch.py --batch-size 5 --sleep 5

    # 只跑指定 lesson order 范围(用于补跑失败批次)
    python scripts/group_nce2_apply_batch.py --start 21 --end 40

    # dry-run(只 plan 不 apply,看分组质量)
    python scripts/group_nce2_apply_batch.py --dry-run

断点文件:data/nc2_group_progress.json
- completed: 已成功 apply 的 lesson order 列表
- failed: 失败 lesson 列表(含 error / attempts)
- 中断(Ctrl+C)再次启动会自动跳过 completed 中的 lesson
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.dependencies import get_db  # noqa: E402
from services.models.content import LESSON, SENTENCE_V2  # noqa: E402
from scripts.group_lesson_sentences import (  # noqa: E402
    _is_ungrouped,
    collapse_by_text,
    split_windows,
    plan_window,
    apply_groups,
    entry_sentence_ids,
)

logger = logging.getLogger("nce2_group_batch")

TB_ID = "tb_e6bce2b577554d02"
PROGRESS_FILE = Path(__file__).resolve().parents[1] / "data" / "nc2_group_progress.json"


# ---------------------------------------------------------------------------
# 进度文件
# ---------------------------------------------------------------------------


def load_progress() -> dict:
    if not PROGRESS_FILE.exists():
        return {"book": TB_ID, "completed": [], "failed": []}
    try:
        return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"进度文件损坏,重置: {e}")
        return {"book": TB_ID, "completed": [], "failed": []}


def save_progress(p: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(
        json.dumps(p, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def run(
    batch_size: int,
    sleep_between: float,
    start_override: int | None,
    end_override: int | None,
    dry_run: bool,
) -> int:
    db = get_db()
    progress = load_progress()
    completed_set = set(progress.get("completed", []))

    # 拉全 lesson 按 order 升序
    res = await db.query(
        collection=LESSON, where={"textbook_id": TB_ID}, limit=200
    )
    lessons_sorted = sorted(
        res.get("records", []), key=lambda x: x.get("order", 0)
    )

    # 过滤范围
    if start_override is not None:
        lessons_sorted = [
            l for l in lessons_sorted
            if l.get("order", 0) >= start_override
            and (end_override is None or l.get("order", 0) <= end_override)
        ]

    # 跳过已完成(非 override 模式下)
    if start_override is None:
        lessons_sorted = [
            l for l in lessons_sorted if l.get("order") not in completed_set
        ]

    total = len(lessons_sorted)
    print(
        f"=== NCE Book 2 共 {total} 课待处理 "
        f"(已完成 {len(completed_set)} 课,本次跑 {total} 课,batch={batch_size},sleep={sleep_between}s) ==="
    )
    if not total:
        print("✓ 全部 lesson 已完成")
        return 0

    mode = "DRY-RUN" if dry_run else "APPLY"
    print(f"模式: {mode}\n")

    # 切批
    batches = [
        lessons_sorted[i : i + batch_size]
        for i in range(0, total, batch_size)
    ]

    success_count = 0
    fail_count = 0
    total_groups_written = 0
    total_sentences_updated = 0

    for batch_idx, batch in enumerate(batches, 1):
        first_order = batch[0].get("order")
        last_order = batch[-1].get("order")
        print(f"--- 批次 {batch_idx}/{len(batches)}: L{first_order}-{last_order} ---")

        # 收集每课 plan_results
        plan_results: list[dict] = []
        batch_failed_lessons: list[int] = []

        for les in batch:
            lid = les["_id"]
            order = les.get("order")
            title = (les.get("title") or "")[:30]

            # 拉句子
            sents_res = await db.query(
                collection=SENTENCE_V2, where={"lesson_id": lid}, limit=200
            )
            all_sents = sents_res.get("records", [])
            ungrouped = [s for s in all_sents if _is_ungrouped(s)]
            unique = collapse_by_text(ungrouped)

            if not unique:
                print(f"  L{order} {title}: 无未分组句,跳过")
                # 仍记为完成(没东西可做)
                progress.setdefault("completed", []).append(order)
                completed_set.add(order)
                save_progress(progress)
                success_count += 1
                continue

            # 切窗口 + plan
            windows = split_windows(unique, 40)
            ok_total = True
            groups_total = 0
            kept_total = 0
            err = None

            for w_no, w_sents in enumerate(windows):
                task = {
                    "textbook_id": TB_ID,
                    "textbook_title": "新概念英语第二册",
                    "lesson_id": lid,
                    "chapter_id": les.get("chapter_id", ""),
                    "lesson_order": order,
                    "lesson_title": les.get("title", ""),
                    "window_no": w_no,
                    "sentences": w_sents,
                    # plan_window 返回时会合并这些字段
                }
                r = await plan_window(task)
                if not r["ok"]:
                    ok_total = False
                    err = r.get("error")
                    break
                # 合并 task + plan 结果,供 apply_groups 使用
                pr = dict(task)
                pr.update(r)
                plan_results.append(pr)
                groups_total += len(r["groups"])
                kept_total += len(r["kept_ungrouped"])

            if not ok_total:
                print(f"  L{order} {title}: FAIL 句={len(all_sents)} err={err[:60] if err else ''}")
                progress.setdefault("failed", []).append({
                    "order": order,
                    "title": les.get("title"),
                    "error": (err or "")[:200],
                    "ts": int(time.time()),
                    "attempts": 1,
                })
                save_progress(progress)
                fail_count += 1
                batch_failed_lessons.append(order)
                continue

            # apply_groups(可选;dry-run 不写)
            apply_stats = {
                "groups_written": 0,
                "sentences_updated": 0,
                "skipped_existing": 0,
                "errors": 0,
            }
            if not dry_run and plan_results:
                # 只对本课的 plan_results apply(避免跨 lesson 串)
                lesson_plan_results = [
                    pr for pr in plan_results if pr.get("lesson_id") == lid
                ]
                r = await apply_groups(db, lesson_plan_results)
                apply_stats = r["stats"]

            progress.setdefault("completed", []).append(order)
            completed_set.add(order)
            save_progress(progress)
            success_count += 1
            total_groups_written += apply_stats["groups_written"]
            total_sentences_updated += apply_stats["sentences_updated"]
            mark = "OK " if not dry_run else "DRY"
            print(
                f"  L{order} {title}: {mark} 句={len(all_sents)} "
                f"组={groups_total} 保留={kept_total}"
                + (f" 写入组={apply_stats['groups_written']} 句={apply_stats['sentences_updated']}"
                   if not dry_run else "")
            )

        # 批间 sleep
        if batch_idx < len(batches) and sleep_between > 0:
            print(f"  ... sleep {sleep_between}s ...\n")
            await asyncio.sleep(sleep_between)

    # 总结
    print("\n=== 总结 ===")
    print(f"成功 {success_count} 课 / 失败 {fail_count} 课")
    if not dry_run:
        print(f"累计写入 sentence_group {total_groups_written} 个")
        print(f"累计回写 sentence_v2.group_id {total_sentences_updated} 条")
    print(f"断点文件: {PROGRESS_FILE}")

    if progress.get("failed"):
        print(f"\n失败 lesson 明细(共 {len(progress['failed'])} 课):")
        for f in progress["failed"]:
            print(f"  L{f['order']} {f.get('title', '')[:30]}: {f.get('error', '')[:80]}")

    return 0 if fail_count == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="NCE Book 2 句子分组分批 apply")
    ap.add_argument("--batch-size", type=int, default=5, help="每批课数(默认 5)")
    ap.add_argument("--sleep", type=float, default=5.0, help="批间 sleep 秒(默认 5)")
    ap.add_argument("--start", type=int, default=None, help="指定起始 lesson order")
    ap.add_argument("--end", type=int, default=None, help="指定结束 lesson order")
    ap.add_argument("--dry-run", action="store_true", help="只 plan 不 apply")
    ap.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sys.exit(asyncio.run(run(
        batch_size=args.batch_size,
        sleep_between=args.sleep,
        start_override=args.start,
        end_override=args.end,
        dry_run=args.dry_run,
    )))


if __name__ == "__main__":
    main()
