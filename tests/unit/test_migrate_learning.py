"""services/migrate_learning.py 迁移逻辑单元测试(经 FakeDB)

覆盖 Phase 2 存量迁移:
- learning_mastery_tracking → skill_state(全量复制 + 字段映射)
- 幂等: 重复执行不产生重复数据
- 安全: 旧表只读, 绝不修改旧表数据
"""

from __future__ import annotations

import asyncio

from services.migrate_learning import migrate_tracking_to_skill_state
from services.models_learning import (
    STATUS_LEARNED,
    STATUS_LEARNING,
    STATUS_MASTERED,
    STATUS_REVIEW_DUE,
    skill_state_id,
)
from tests.fakes.fake_db import FakeDB

TRACKING_SEED = [
    {
        "scholar_id": "s1",
        "sentence_id": "sent_1",
        "status": "已学",
        "score": 85,
        "study_count": 3,
        "last_study_time": 1_700_000_000,
    },
    {
        "scholar_id": "s1",
        "sentence_id": "sent_2",
        "status": "learning",
        "mastery": 0.5,
        "study_count": 2,
        "last_study_time": 1_700_100_000,
    },
    {
        "scholar_id": "s2",
        "sentence_id": "sent_1",
        "status": "已掌握",
        "score": 90,
    },
]
SENTENCE_SEED = [
    {"sentence_id": "sent_1", "unit_id": "unit_a", "index": 1, "text": "Hello", "text_book_id": "tb_1"},
    {"sentence_id": "sent_2", "unit_id": "unit_b", "index": 1, "text": "World", "text_book_id": "tb_1"},
]


def _make_db() -> FakeDB:
    return FakeDB(seed={
        "learning_mastery_tracking": TRACKING_SEED,
        "sentence": SENTENCE_SEED,
    })


class TestMigrateTrackingToSkillState:
    def test_full_copy_with_field_mapping(self):
        db = _make_db()
        stats = asyncio.run(migrate_tracking_to_skill_state(db))
        assert stats["processed"] == 3
        assert stats["created"] == 3
        assert stats["skipped"] == 0
        states = db.all("skill_state")
        assert len(states) == 3

    def test_chinese_status_normalized(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        by_key = {s["_id"]: s for s in db.all("skill_state")}
        learned = by_key[skill_state_id("s1", "sent_1", "translation")]
        assert learned["status"] == STATUS_LEARNED
        assert learned["mastery_score"] == 85.0
        assert learned["attempt_count"] == 3  # study_count → attempt_count
        assert learned["last_studied_at"] == 1_700_000_000
        mastered = by_key[skill_state_id("s2", "sent_1", "translation")]
        assert mastered["status"] == STATUS_MASTERED

    def test_lesson_id_backfilled_from_sentence(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        by_key = {s["_id"]: s for s in db.all("skill_state")}
        # 旧 unit_id → 新 lesson_id
        assert by_key[skill_state_id("s1", "sent_1", "translation")]["lesson_id"] == "unit_a"
        assert by_key[skill_state_id("s1", "sent_2", "translation")]["lesson_id"] == "unit_b"

    def test_missing_meta_falls_back(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        by_key = {s["_id"]: s for s in db.all("skill_state")}
        s2_s1 = by_key[skill_state_id("s2", "sent_1", "translation")]
        assert s2_s1["attempt_count"] == 1  # 缺 study_count → 1
        assert s2_s1["mastery_score"] == 90.0

    def test_low_mastery_derives_review_due(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        by_key = {s["_id"]: s for s in db.all("skill_state")}
        s1_s2 = by_key[skill_state_id("s1", "sent_2", "translation")]
        # mastery 0.5 → 50 分 < 60 → review_due
        assert s1_s2["status"] == STATUS_REVIEW_DUE
        assert s1_s2["mastery_score"] == 50.0

    def test_idempotent(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        stats2 = asyncio.run(migrate_tracking_to_skill_state(db))
        assert stats2["created"] == 0
        assert stats2["skipped"] == 3
        assert len(db.all("skill_state")) == 3

    def test_old_table_not_modified(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db))
        assert db.all("learning_mastery_tracking") == TRACKING_SEED  # 旧表原样保留


class TestMigrateCustomSkillCode:
    def test_custom_skill_code(self):
        db = _make_db()
        asyncio.run(migrate_tracking_to_skill_state(db, skill_code="listening"))
        states = db.all("skill_state")
        assert len(states) == 3
        assert {s["skill_code"] for s in states} == {"listening"}
        assert states[0]["_id"].endswith("_listening")
