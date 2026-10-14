"""临时探针：为 O01 收尾 / O02 跑量取现状快照。

输出四块：
1. sentence_v2 各教材句子数；(this+| slim)
2. english_extension_point 已抽取记录（幂等缓存现状）；
3. extension_task 近期任务（含 kind / status / 耗时）；
4. extension_review / extension_review_log 学习者校对现状。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from services.dependencies import get_db  # noqa: E402


async def _count_by(db, collection: str, key: str) -> None:
    for tid in [
        "tb_e6bce2b577554d02",
        "tb_db3e2209b3cc4e9e",
        "EXTRA_AI",
        "tb_5aefa2dee7e34546",
    ]:
        res = await db.query(collection, where={key: tid}, limit=1)
        n = len(res.get("records", []))
        print(f"  {collection}[{key}={tid}] -> 首屏 {n} 条")


async def main() -> None:
    db = get_db()

    print("=== 1) sentence_v2 教材分布 ===")
    await _count_by(db, "sentence_v2", "textbook_id")

    print("\n=== 2) english_extension_point 现状 ===")
    res = await db.query(config.EXTENSION_POINT_COLLECTION, where={}, limit=50)
    pts = res.get("records", [])
    print("首屏记录数:", len(pts))
    for p in pts[:20]:
        meta = p.get("meta", {}) or {}
        print(
            f"  - {p.get('sentence_id')} source={p.get('source')} "
            f"points={len(p.get('points', []) or [])} pv={meta.get('prompt_version')} "
            f"attempts={meta.get('attempts')} fb={meta.get('fallback_reason')} "
            f"model={meta.get('model')}"
        )

    print("\n=== 3) extension_task 近期任务 ===")
    res = await db.query(config.EXTENSION_TASK_COLLECTION, where={}, limit=50)
    tasks = res.get("records", [])
    print("首屏记录数:", len(tasks))
    kinds: dict[str, int] = {}
    for t in tasks:
        k = f"{t.get('kind')}/{t.get('status')}"
        kinds[k] = kinds.get(k, 0) + 1
    print("  kind/status 分布:", kinds)
    for t in tasks[:10]:
        created, updated = t.get("created_at", 0), t.get("updated_at", 0)
        print(
            f"  - {t.get('task_id')} {t.get('kind')} {t.get('status')} "
            f"{(updated - created) / 1000:.1f}s sid={t.get('sentence_id')}"
        )

    print("\n=== 4) 学习者校对现状 ===")
    for coll in (config.EXTENSION_REVIEW_COLLECTION, config.EXTENSION_REVIEW_LOG_COLLECTION):
        res = await db.query(coll, where={}, limit=20)
        recs = res.get("records", [])
        print(f"  {coll}: 首屏 {len(recs)} 条")
        for r in recs[:5]:
            print(
                f"    - {r.get('scholar_id')} / {r.get('sentence_id')} "
                f"removed={len(r.get('removed_texts', []) or [])} "
                f"added={len(r.get('added', []) or [])} at={r.get('updated_at') or r.get('at')}"
            )


if __name__ == "__main__":
    asyncio.run(main())
