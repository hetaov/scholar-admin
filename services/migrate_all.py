"""统一存量迁移脚本 —— 重构前旧表 → 重构后新表（自动建表 + 备份快照 + 可回退）

用法:
    python -m services.migrate_all check
    python -m services.migrate_all create-tables
    python -m services.migrate_all migrate [--backup-dir DIR] [--batch N] [--local]
    python -m services.migrate_all rollback <backup_dir> [--drop-created-tables]
    任一子命令均支持 --dry-run 预览将要执行的操作。

迁移映射（旧表只读，新表写入）:
    textbook                → textbook_v2
    unit                    → chapter / lesson
    sentence                → sentence_v2
    learning_mastery_tracking → skill_state
    (种子数据, 无旧表)      → skill

回退机制:
    1. migrate 前: 对每个目标新表导出全量快照 snapshot_<table>.json,
       并记录本次创建(迁移前不存在)的集合。
    2. migrate 后: diff 迁移前后各表 _id 集合, 本次新建文档的 _id 写入 changelog_<table>.json。
    3. rollback : 按 changelog 删除新建文档 → 按 snapshot 覆盖恢复迁移前已有文档
                  (可选 --drop-created-tables 删除"本次创建"的空集合)。
    旧表全程只读, 回退不会触碰旧数据。

注意:
    - 迁移期间建议暂停写入服务, 避免并发写入被误判为"本次新建"。
    - 真实环境默认连 CloudBase (services.dependencies.get_db), 需先配置腾讯云密钥。
    - --local 使用内存 FakeDB, 便于本地联调与测试。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from services.dependencies import get_db
from services.events import STUDY_ATTEMPT, STUDY_SESSION
from services.migrate_content_v2 import run_migration
from services.migrate_learning import migrate_tracking_to_skill_state
from services.migrate_scholar_book import migrate_task_to_scholar_book
from services.models_content import CHAPTER, LESSON, SENTENCE_V2, TEXTBOOK_V2
from services.models_learning import DEFAULT_SKILL_CODE, SKILL, SKILL_STATE, seed_skills
from services.models_scholar_book import SCHOLAR_BOOK

logger = logging.getLogger("scholar-admin.migrate_all")

# 目标新表：(集合名, 数据来源说明)
TARGET_TABLES: list[tuple[str, str]] = [
    (TEXTBOOK_V2, "旧表 textbook"),
    (CHAPTER, "旧表 unit(分组生成)"),
    (LESSON, "旧表 unit"),
    (SENTENCE_V2, "旧表 sentence"),
    (SKILL, "种子数据(无旧表)"),
    (SKILL_STATE, "旧表 learning_mastery_tracking"),
    (STUDY_ATTEMPT, "运行时表(无旧数据)"),
    (STUDY_SESSION, "运行时表(无旧数据)"),
    (SCHOLAR_BOOK, "旧表 task"),
]

DEFAULT_BACKUP_ROOT = Path("migrations")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _env_name(db) -> str:
    return str(getattr(db, "env_id", "local"))


async def _fetch_all(db, collection: str, batch_size: int = 100) -> list[dict]:
    """分页读取集合全量文档。"""
    records: list[dict] = []
    offset = 0
    while True:
        page = await db.query(collection=collection, where={}, offset=offset, limit=batch_size)
        recs = page.get("records", [])
        if not recs:
            break
        records.extend(recs)
        if len(recs) < batch_size:
            break
        offset += len(recs)
    return records


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _doc_ids(records: list[dict]) -> set[str]:
    return {str(r.get("_id", "")) for r in records}


# ---------------------------------------------------------------------------
# 1. 检查
# ---------------------------------------------------------------------------


async def check_schema(db) -> dict:
    """只读检查：列出旧表 / 新表存在状态。"""
    collections = await db.list_collections()
    names = {c.get("TableName") for c in collections}

    legacy_tables = {
        "textbook": "内容-教材(旧)",
        "unit": "内容-单元(旧)",
        "sentence": "内容-句子(旧)",
        "learning_mastery_tracking": "学习-掌握度(旧)",
        "task": "学者-教材关联(旧)",
    }
    print("=== 旧表(重构前) ===")
    legacy_status: dict[str, bool] = {}
    for table, desc in legacy_tables.items():
        exists = table in names
        legacy_status[table] = exists
        print(f"  [{'✔' if exists else '✘'}] {table:<28} {desc}")

    print("\n=== 新表(重构后) ===")
    target_status: dict[str, bool] = {}
    for table, desc in TARGET_TABLES:
        exists = table in names
        target_status[table] = exists
        print(f"  [{'✔' if exists else '✘'}] {table:<28} {desc}")

    return {"legacy": legacy_status, "targets": target_status}


async def check_progress(db, batch_size: int = 100) -> dict[str, dict[str, int]]:
    """只读检查迁移进度：对比新旧表键集合，统计每张表已迁移 / 待迁移数据量。

    幂等判断: 各迁移按唯一键判重(见各 migrate_* 模块), 已迁移的键重跑会跳过,
    因此"待迁移"= 旧表键 − 新表键, 重跑 migrate 只会补这些缺失数据, 不会重复。
    """

    def _tracking_key(r: dict[str, Any]) -> str:
        s, t = r.get("scholar_id"), r.get("sentence_id")
        return f"{s}_{t}_{DEFAULT_SKILL_CODE}" if s and t else ""

    def _task_key(r: dict[str, Any]) -> str:
        s, t = r.get("scholar_id"), r.get("text_book_id")
        return f"{s}_{t}" if s and t else ""

    # (旧表, 新表, 说明, 旧键提取, 新键提取, 旧表 select, 新表 select)
    specs = [
        (
            "textbook", TEXTBOOK_V2, "内容-教材",
            lambda r: str(r.get("_id", "")), lambda r: str(r.get("_id", "")),
            {"_id": 1}, {"_id": 1},
        ),
        (
            "unit", LESSON, "内容-课",
            lambda r: str(r.get("unit_id", "")), lambda r: str(r.get("lesson_id", "")),
            {"unit_id": 1}, {"lesson_id": 1},
        ),
        (
            "sentence", SENTENCE_V2, "内容-句",
            lambda r: str(r.get("sentence_id", "")), lambda r: str(r.get("sentence_id", "")),
            {"sentence_id": 1}, {"sentence_id": 1},
        ),
        (
            "learning_mastery_tracking", SKILL_STATE, "学习-能力状态",
            _tracking_key, lambda r: str(r.get("_id", "")),
            {"scholar_id": 1, "sentence_id": 1}, {"_id": 1},
        ),
        (
            "task", SCHOLAR_BOOK, "学者-教材",
            _task_key, lambda r: str(r.get("_id", "")),
            {"scholar_id": 1, "text_book_id": 1}, {"_id": 1},
        ),
    ]

    async def _key_set(collection: str, key_fn, select: dict[str, int]) -> set[str]:
        """分页读取某表指定字段的键集合(只读)。表不存在时返回空集。"""
        if not await db.check_collection(collection):
            return set()
        keys: set[str] = set()
        offset = 0
        while True:
            page = await db.query(
                collection=collection, where={}, offset=offset, limit=batch_size, select=select
            )
            recs = page.get("records", [])
            if not recs:
                break
            for r in recs:
                k = key_fn(r)
                if k:
                    keys.add(k)
            offset += len(recs)
            if len(recs) < batch_size:
                break
        return keys

    print("\n=== 迁移进度(已迁移 → 待迁移) ===")
    progress: dict[str, dict[str, int]] = {}
    for old, new, desc, old_fn, new_fn, old_sel, new_sel in specs:
        old_keys = await _key_set(old, old_fn, old_sel)
        new_keys = await _key_set(new, new_fn, new_sel)
        migrated = len(old_keys & new_keys)
        pending = len(old_keys - new_keys)
        progress[f"{old}→{new}"] = {
            "old_total": len(old_keys),
            "migrated": migrated,
            "pending": pending,
        }
        mark = "✔" if pending == 0 else "⚠"
        print(f"  [{mark}] {old:<26} {len(old_keys):>5}条 → {new:<16} 已迁移 {migrated:>5} / 待迁移 {pending:>5}  {desc}")

    # chapter 由 unit 分组生成, 无一一对应, 仅展示当前数量
    chapter_count = 0
    if await db.check_collection(CHAPTER):
        chapter_count = len(await _key_set(CHAPTER, lambda r: str(r.get("_id", "")), {"_id": 1}))
    print(f"  [·] {'unit(分组)':<26}      → {CHAPTER:<16} 当前 {chapter_count} 章 (分组生成, 无一一对应)")
    progress["unit→chapter"] = {"chapter_count": chapter_count}
    return progress


# ---------------------------------------------------------------------------
# 2. 建表
# ---------------------------------------------------------------------------


async def ensure_tables(db, *, dry_run: bool = False) -> list[str]:
    """创建缺失的目标新表，返回本次创建的集合名列表。"""
    created: list[str] = []
    for table, desc in TARGET_TABLES:
        exists = await db.check_collection(table)
        if exists:
            continue
        if dry_run:
            print(f"[dry-run] 将创建集合: {table} ({desc})")
        else:
            await db.create_collection(table)
            print(f"[create] 集合已创建: {table} ({desc})")
        created.append(table)
    if not created:
        print("[create] 所有目标新表均已存在, 无需创建。")
    return created


# ---------------------------------------------------------------------------
# 3. 备份快照
# ---------------------------------------------------------------------------


async def backup_snapshots(db, backup_dir: Path, *, dry_run: bool = False) -> dict:
    """导出每个目标新表的迁移前全量快照，返回 {表名: 文档数}。"""
    counts: dict[str, int] = {}
    if dry_run:
        for table, desc in TARGET_TABLES:
            print(f"[dry-run] 将备份快照: {table} ({desc})")
        return counts
    backup_dir.mkdir(parents=True, exist_ok=True)
    for table, _desc in TARGET_TABLES:
        records = await _fetch_all(db, table)
        _dump_json(backup_dir / f"snapshot_{table}.json", records)
        counts[table] = len(records)
        print(f"[backup] snapshot_{table}.json: {len(records)} 条")
    return counts


# ---------------------------------------------------------------------------
# 4. 迁移
# ---------------------------------------------------------------------------


async def run_full_migration(
    db,
    *,
    backup_dir: Path,
    batch_size: int = 100,
    dry_run: bool = False,
) -> dict:
    """建表 → 备份快照 → 迁移 → diff changelog → 写 manifest。"""
    # 1) 记录迁移前已存在的集合, 并创建缺失表
    pre_existing = set()
    for table, _desc in TARGET_TABLES:
        if await db.check_collection(table):
            pre_existing.add(table)
    print(f"[migrate] 迁移前已存在的集合: {sorted(pre_existing) or '(空)'}")
    created_tables = await ensure_tables(db, dry_run=dry_run)

    # 2) 备份快照(迁移前状态)
    snapshot_counts = await backup_snapshots(db, backup_dir, dry_run=dry_run)
    if dry_run:
        print("[dry-run] 将执行迁移: 内容(textbook/unit/sentence → v2 表) + skill 种子 + 学习(→ skill_state) + 学者教材(task → scholar_book)")
        print("[dry-run] 将生成 changelog 并写入 manifest.json, 本次结束, 未做任何写入。")
        return {"dry_run": True}

    # 3) 执行迁移(旧表只读, 新表写入)
    before_ids: dict[str, set[str]] = {}
    for table, _desc in TARGET_TABLES:
        before_ids[table] = _doc_ids(await _fetch_all(db, table))

    stats: dict[str, Any] = {}
    stats["content"] = await run_migration(db, batch_size)
    stats["skill_seed"] = await seed_skills(db)
    stats["learning"] = await migrate_tracking_to_skill_state(db, batch_size=batch_size)
    stats["scholar_book"] = await migrate_task_to_scholar_book(db, batch_size=batch_size)

    # 4) diff changelog: 本次新建文档的 _id
    changelog_counts: dict[str, int] = {}
    for table, _desc in TARGET_TABLES:
        after_ids = _doc_ids(await _fetch_all(db, table))
        created_ids = sorted(after_ids - before_ids[table])
        if created_ids:
            _dump_json(backup_dir / f"changelog_{table}.json", {"created_ids": created_ids})
        changelog_counts[table] = len(created_ids)
        print(f"[changelog] {table}: 本次新建 {len(created_ids)} 条")

    # 5) manifest
    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "env_id": _env_name(db),
        "tables": [t for t, _d in TARGET_TABLES],
        "created_tables": created_tables,
        "snapshot_counts": snapshot_counts,
        "changelog_counts": changelog_counts,
        "migration_stats": stats,
    }
    _dump_json(backup_dir / "manifest.json", manifest)
    print(f"[manifest] 已写入 {backup_dir}/manifest.json")
    return {"snapshot_counts": snapshot_counts, "changelog_counts": changelog_counts, "stats": stats}


# ---------------------------------------------------------------------------
# 5. 回退
# ---------------------------------------------------------------------------


async def rollback(
    db,
    backup_dir: Path,
    *,
    drop_created_tables: bool = False,
    dry_run: bool = False,
) -> dict:
    """按备份目录回退：删除本次新建文档 → 按快照恢复迁移前文档 → (可选)删本次创建的集合。"""
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"未找到 {manifest_path}, 无法回退")

    manifest = _load_json(manifest_path)
    tables: list[str] = manifest.get("tables", [t for t, _d in TARGET_TABLES])
    created_tables: set[str] = set(manifest.get("created_tables", []))
    deleted_total = 0
    restored_total = 0
    dropped_tables: list[str] = []

    for table in tables:
        # a) 删除本次新建的文档
        changelog_path = backup_dir / f"changelog_{table}.json"
        created_ids: list[str] = []
        if changelog_path.exists():
            created_ids = _load_json(changelog_path).get("created_ids", [])
        if created_ids:
            if dry_run:
                print(f"[dry-run] 将删除 {table} 中本次新建的 {len(created_ids)} 条文档")
            else:
                # 分批删除, 避免单次 $in 过大
                step = 200
                for i in range(0, len(created_ids), step):
                    await db.delete(
                        collection=table,
                        where={"_id": {"$in": created_ids[i : i + step]}},
                        multi=True,
                    )
                deleted_total += len(created_ids)
            print(f"[rollback] {table}: 删除本次新建 {len(created_ids)} 条")

        # b) 恢复迁移前快照(覆盖字段, 防御迁移逻辑意外改动)
        snapshot_path = backup_dir / f"snapshot_{table}.json"
        snapshot: list[dict] = _load_json(snapshot_path) if snapshot_path.exists() else []
        if snapshot:
            if dry_run:
                print(f"[dry-run] 将恢复 {table} 迁移前已有文档 {len(snapshot)} 条(覆盖字段)")
            else:
                for doc in snapshot:
                    fields = {k: v for k, v in doc.items() if k != "_id"}
                    if not fields:
                        continue
                    await db.update(
                        collection=table,
                        where={"_id": doc["_id"]},
                        data={"$set": fields},
                        upsert=True,
                        multi=False,
                    )
                restored_total += len(snapshot)
            print(f"[rollback] {table}: 恢复快照 {len(snapshot)} 条")

        # c) 可选: 删除本次创建的集合
        if drop_created_tables and table in created_tables:
            if dry_run:
                print(f"[dry-run] 将删除本次创建的集合: {table}")
            else:
                await db.delete_collection(table)
                dropped_tables.append(table)
            print(f"[rollback] 集合已删除: {table}")

    summary = {
        "deleted_created": deleted_total,
        "restored_snapshot": restored_total,
        "dropped_tables": dropped_tables,
    }
    print(f"\n[rollback] 完成: 删除新建 {deleted_total} 条, 恢复快照 {restored_total} 条, 删除集合 {dropped_tables or '无'}")
    return summary


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="重构前数据 → 新表统一迁移脚本(自动建表 + 备份 + 回退)"
    )
    parser.add_argument(
        "action",
        choices=["check", "create-tables", "migrate", "rollback"],
        help="操作: check=检查, create-tables=创建缺失新表, migrate=备份+迁移, rollback=回退",
    )
    parser.add_argument("backup_dir", nargs="?", help="回退用的备份目录(仅 rollback 需要)")
    parser.add_argument("--backup-dir", dest="backup_dir_opt", help="迁移时指定备份目录")
    parser.add_argument("--batch", type=int, default=100, help="批量大小, 默认 100")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的操作, 不写入")
    parser.add_argument("--local", action="store_true", help="使用内存 FakeDB(本地联调/测试), 默认连 CloudBase")
    parser.add_argument("--drop-created-tables", action="store_true", help="回退时删除本次创建的集合")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    db = None
    if args.local:
        from tests.fakes.fake_db import FakeDB

        db = FakeDB()
        print("[env] 使用内存 FakeDB(本地模式)")
    else:
        db = get_db()

    t0 = time.time()

    if args.action == "check":
        await check_schema(db)
        await check_progress(db, batch_size=args.batch)

    elif args.action == "create-tables":
        await ensure_tables(db, dry_run=args.dry_run)

    elif args.action == "migrate":
        backup_dir = Path(args.backup_dir_opt) if args.backup_dir_opt else (
            DEFAULT_BACKUP_ROOT / f"backup_{_now_tag()}"
        )
        if not args.dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
        print(f"[migrate] 备份目录: {backup_dir.resolve()}")
        await run_full_migration(db, backup_dir=backup_dir, batch_size=args.batch, dry_run=args.dry_run)

    elif args.action == "rollback":
        backup_dir = args.backup_dir or args.backup_dir_opt
        if not backup_dir:
            parser.error("rollback 需要提供备份目录(位置参数或 --backup-dir)")
        await rollback(
            db,
            Path(backup_dir),
            drop_created_tables=args.drop_created_tables,
            dry_run=args.dry_run,
        )

    print(f"\n[完成] 耗时 {time.time() - t0:.2f}s")
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
