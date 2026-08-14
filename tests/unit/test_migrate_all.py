"""services/migrate_all.py 统一迁移脚本测试 —— 建表 / 备份快照 / 迁移 / 回退 / 幂等

覆盖:
- ensure_tables: 新表缺失时自动创建, 已存在跳过
- run_full_migration: 旧表 → 新表迁移, 生成 snapshot / changelog / manifest, 旧表只读
- rollback: 删除本次新建文档 + 恢复快照, 旧表不受影响
- 幂等: 重复迁移不产生重复数据
- dry-run: 只读不写
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from services.migrate_all import (
    TARGET_TABLES,
    check_progress,
    ensure_tables,
    rollback,
    run_full_migration,
)
from tests.fakes.fake_db import FakeDB

TARGET_NAMES = [t for t, _d in TARGET_TABLES]

# 旧表种子数据(重构前)
TEXTBOOK_SEED = [
    {"_id": "tb_1", "title": "NCE Book 1", "grade": "1", "semester": "Book 1"},
]
UNIT_SEED = [
    {"unit_id": "unit_a", "title": "Unit A", "text_book_id": "tb_1", "unit_index": 1, "total_sentences": 2},
    {"unit_id": "unit_b", "title": "Unit B", "text_book_id": "tb_1", "unit_index": 2, "total_sentences": 2},
]
SENTENCE_SEED = [
    {"sentence_id": "sent_1", "unit_id": "unit_a", "index": 1, "text": "Hello", "text_book_id": "tb_1"},
    {"sentence_id": "sent_2", "unit_id": "unit_a", "index": 2, "text": "World", "text_book_id": "tb_1"},
    {"sentence_id": "sent_3", "unit_id": "unit_b", "index": 1, "text": "Goodbye", "text_book_id": "tb_1"},
    {"sentence_id": "sent_4", "unit_id": "unit_b", "index": 2, "text": "Friend", "text_book_id": "tb_1"},
]
TRACKING_SEED = [
    {"scholar_id": "s1", "sentence_id": "sent_1", "score": 80, "mastery": 0.8, "study_count": 3},
]
TASK_SEED = [
    {"scholar_id": "s1", "text_book_id": "tb_1", "created_at": 1700000000},
    {"scholar_id": "s2", "text_book_id": "tb_1", "created_at": 1700000000000},
]


def _make_db(extra: dict[str, list[dict]] | None = None) -> FakeDB:
    seed = {
        "textbook": TEXTBOOK_SEED,
        "unit": UNIT_SEED,
        "sentence": SENTENCE_SEED,
        "learning_mastery_tracking": TRACKING_SEED,
        "task": TASK_SEED,
    }
    seed.update(extra or {})
    return FakeDB(seed=seed)


def _run(coro):
    return asyncio.run(coro)


def _load(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestEnsureTables:
    def test_creates_all_missing_tables(self):
        db = _make_db()  # 只有旧表
        created = _run(ensure_tables(db))
        assert set(created) == set(TARGET_NAMES)
        for name in TARGET_NAMES:
            assert _run(db.check_collection(name)) is True

    def test_skips_existing_tables(self):
        db = _make_db()
        _run(ensure_tables(db))
        created = _run(ensure_tables(db))
        assert created == []

    def test_dry_run_does_not_create(self):
        db = _make_db()
        created = _run(ensure_tables(db, dry_run=True))
        assert set(created) == set(TARGET_NAMES)
        for name in TARGET_NAMES:
            assert _run(db.check_collection(name)) is False


class TestRunFullMigration:
    def test_migrates_legacy_to_new_tables(self, tmp_path: Path):
        db = _make_db()
        result = _run(run_full_migration(db, backup_dir=tmp_path))

        assert db.all("textbook_v2") and len(db.all("textbook_v2")) == 1
        assert db.all("chapter") and len(db.all("chapter")) == 1
        assert len(db.all("lesson")) == 2
        assert len(db.all("sentence_v2")) == 4
        assert len(db.all("skill_state")) == 1
        assert len(db.all("skill")) == 4  # 种子数据
        assert len(db.all("scholar_book")) == 2  # task → scholar_book

        # 备份产物
        for name in TARGET_NAMES:
            assert (tmp_path / f"snapshot_{name}.json").exists()
        assert (tmp_path / "changelog_skill_state.json").exists()
        assert (tmp_path / "changelog_textbook_v2.json").exists()
        assert (tmp_path / "manifest.json").exists()

        manifest = _load(tmp_path / "manifest.json")
        assert set(manifest["tables"]) == set(TARGET_NAMES)
        assert set(manifest["created_tables"]) == set(TARGET_NAMES)  # 迁移前全部不存在

    def test_legacy_tables_readonly(self, tmp_path: Path):
        db = _make_db()
        before = {
            c: db.all(c)
            for c in ("textbook", "unit", "sentence", "learning_mastery_tracking", "task")
        }
        _run(run_full_migration(db, backup_dir=tmp_path))
        for c, rows in before.items():
            assert db.all(c) == rows

    def test_idempotent_re_run(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path))
        new_counts_1 = {c: len(db.all(c)) for c in TARGET_NAMES}

        backup2 = tmp_path / "run2"
        result2 = _run(run_full_migration(db, backup_dir=backup2))
        for name in TARGET_NAMES:
            assert (backup2 / f"changelog_{name}.json").exists() is False or _load(
                backup2 / f"changelog_{name}.json"
            ).get("created_ids") == []
            assert len(db.all(name)) == new_counts_1[name]

    def test_migrate_dry_run_writes_nothing(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path, dry_run=True))
        # 新表未创建、无备份产物、旧表未动
        for name in TARGET_NAMES:
            assert _run(db.check_collection(name)) is False
        assert list(tmp_path.glob("*")) == []
        assert db.all("textbook") == TEXTBOOK_SEED


class TestRollback:
    def test_rollback_removes_created_records(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path))

        summary = _run(rollback(db, tmp_path))
        assert summary["deleted_created"] > 0
        assert summary["restored_snapshot"] == 0  # 迁移前新表为空

        # 新表全部恢复迁移前(空)状态
        for name in TARGET_NAMES:
            assert db.all(name) == []
        # 旧表原样
        assert db.all("textbook") == TEXTBOOK_SEED
        assert db.all("learning_mastery_tracking") == TRACKING_SEED

    def test_rollback_keeps_preexisting_rows(self, tmp_path: Path):
        # 迁移前 textbook_v2 已存在 1 条(之前部分迁移), 回退后应保留且字段不变
        preexisting = {"_id": "tb_0", "title": "Old", "version": 1}
        db = _make_db(extra={"textbook_v2": [preexisting]})

        _run(run_full_migration(db, backup_dir=tmp_path))
        assert len(db.all("textbook_v2")) == 2  # tb_0 + tb_1

        _run(rollback(db, tmp_path))
        rows = db.all("textbook_v2")
        assert len(rows) == 1
        assert rows[0]["_id"] == "tb_0"
        assert rows[0]["title"] == "Old"
        assert rows[0]["version"] == 1

    def test_rollback_drop_created_tables(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path))
        _run(rollback(db, tmp_path, drop_created_tables=True))
        for name in TARGET_NAMES:
            assert _run(db.check_collection(name)) is False

    def test_rollback_requires_manifest(self, tmp_path: Path):
        db = _make_db()
        with pytest.raises(FileNotFoundError):
            _run(rollback(db, tmp_path))

    def test_rollback_dry_run_writes_nothing(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path))
        migrated = {c: len(db.all(c)) for c in TARGET_NAMES}
        _run(rollback(db, tmp_path, dry_run=True))
        for name in TARGET_NAMES:
            assert len(db.all(name)) == migrated[name]


class TestScholarBookMigration:
    """task → scholar_book 迁移：字段映射 / 幂等 / 无效记录跳过。"""

    def _migrate(self, db, tmp_path: Path):
        return _run(run_full_migration(db, backup_dir=tmp_path))

    def test_maps_task_fields(self, tmp_path: Path):
        db = _make_db()
        self._migrate(db, tmp_path)

        books = {r["scholar_id"]: r for r in db.all("scholar_book")}
        assert len(books) == 2

        b1 = books["s1"]
        assert b1["_id"] == "s1_tb_1"
        assert b1["scholar_book_id"] == "s1_tb_1"
        assert b1["textbook_id"] == "tb_1"
        assert b1["status"] == "learning"
        assert b1["total_time_spent"] == 0
        # task.created_at 秒 → 毫秒
        assert b1["started_at"] == 1700000000000
        assert b1["last_studied_at"] == 1700000000000

        b2 = books["s2"]
        # task.created_at 已是毫秒, 原样保留
        assert b2["started_at"] == 1700000000000

    def test_idempotent_re_run(self, tmp_path: Path):
        db = _make_db()
        self._migrate(db, tmp_path)
        first = db.all("scholar_book")

        backup2 = tmp_path / "run2"
        _run(run_full_migration(db, backup_dir=backup2))
        assert db.all("scholar_book") == first

    def test_skips_invalid_records(self, tmp_path: Path):
        db = _make_db(
            extra={
                "task": TASK_SEED
                + [
                    {"scholar_id": "", "text_book_id": "tb_1"},  # 缺 scholar_id
                    {"scholar_id": "s3", "text_book_id": None},  # 缺 textbook_id
                    {"scholar_id": "s3", "text_book_id": "tb_1", "created_at": 1700000000},
                ]
            }
        )
        self._migrate(db, tmp_path)

        books = db.all("scholar_book")
        assert {r["scholar_id"] for r in books} == {"s1", "s2", "s3"}
        assert len(books) == 3  # 2 条有效 + 2 条无效跳过


class TestOrphanIdempotent:
    """孤儿 unit(text_book_id 为空) → orphan chapter 的幂等性: 重跑不重复创建空壳章。"""

    ORPHAN_UNIT = {
        "unit_id": "unit_orphan", "title": "Orphan Unit", "text_book_id": "",
        "unit_index": 1, "total_sentences": 1,
    }
    ORPHAN_SENTENCE = {
        "sentence_id": "sent_orphan", "unit_id": "unit_orphan", "index": 1,
        "text": "Lone", "text_book_id": "",
    }

    def _make(self) -> FakeDB:
        return _make_db(
            extra={"unit": UNIT_SEED + [self.ORPHAN_UNIT], "sentence": SENTENCE_SEED + [self.ORPHAN_SENTENCE]}
        )

    def _orphan_chapters(self, db) -> list[dict]:
        return [c for c in db.all("chapter") if not c.get("textbook_id")]

    def test_re_run_reuses_orphan_chapter(self, tmp_path: Path):
        db = self._make()
        _run(run_full_migration(db, backup_dir=tmp_path))
        first = self._orphan_chapters(db)
        assert len(first) == 1

        backup2 = tmp_path / "run2"
        _run(run_full_migration(db, backup_dir=backup2))
        second = self._orphan_chapters(db)
        assert len(second) == 1  # 未重复创建空壳章
        assert second[0]["_id"] == first[0]["_id"]  # 复用同一 orphan chapter

    def test_orphan_lesson_sentence_not_duplicated(self, tmp_path: Path):
        db = self._make()
        _run(run_full_migration(db, backup_dir=tmp_path))
        lessons_1 = len(db.all("lesson"))
        sentences_1 = len(db.all("sentence_v2"))

        backup2 = tmp_path / "run2"
        _run(run_full_migration(db, backup_dir=backup2))
        assert len(db.all("lesson")) == lessons_1
        assert len(db.all("sentence_v2")) == sentences_1


class TestCheckProgress:
    """check_progress: 只读 diff 报告, 反映已迁移 / 待迁移数据量。"""

    def test_before_migration_all_pending(self):
        db = _make_db()  # 只有旧表
        progress = _run(check_progress(db))
        assert progress["textbook→textbook_v2"]["pending"] == 1
        assert progress["unit→lesson"]["pending"] == 2
        assert progress["sentence→sentence_v2"]["pending"] == 4
        assert progress["learning_mastery_tracking→skill_state"]["pending"] == 1
        assert progress["task→scholar_book"]["pending"] == 2

    def test_after_migration_all_done(self, tmp_path: Path):
        db = _make_db()
        _run(run_full_migration(db, backup_dir=tmp_path))
        progress = _run(check_progress(db))
        for key in (
            "textbook→textbook_v2",
            "unit→lesson",
            "sentence→sentence_v2",
            "learning_mastery_tracking→skill_state",
            "task→scholar_book",
        ):
            assert progress[key]["pending"] == 0
            assert progress[key]["migrated"] == progress[key]["old_total"]

    def test_partial_migration_reports_pending(self, tmp_path: Path):
        # 模拟"迁移到一半": 只迁移 task → scholar_book, 其余未动
        db = _make_db()
        db.add("scholar_book", {
            "_id": "s1_tb_1", "scholar_book_id": "s1_tb_1", "scholar_id": "s1",
            "textbook_id": "tb_1", "status": "learning", "total_time_spent": 0,
        })
        progress = _run(check_progress(db))
        assert progress["task→scholar_book"]["migrated"] == 1
        assert progress["task→scholar_book"]["pending"] == 1  # s2_tb_1 未迁移
        assert progress["sentence→sentence_v2"]["pending"] == 4  # 内容未迁移
