"""M3 G1.1 单元测试 — 组视图服务 getLessonSentenceGroups（service-contract §8.5）

覆盖：
- lesson 不存在 → LessonNotFoundError（404 LESSON_NOT_FOUND）
- 无 group → 读兼容层逐句构造临时组 `legacy_{lesson_id}_{sentence_id}`
  （组标题 = text 前 20 字，type = stand_alone，返回结构逐字一致）
- 有 group → 按组组织（order_in_lesson 排序 / 组内句子按 sentence_ids 顺序）
- 组内句子 status / skills / weakest_skill / review_count / next_review_at
  与 /sentences 接口口径一致（乐观 pick_state）
- is_canonical / canonical_sentence_id（null / 自身 = canonical）
- summary（mastery / skills / learned_sentence_count / total_sentence_count / group_count）
"""
from __future__ import annotations

import pytest

from services.english import LessonNotFoundError
from services.english.group_view import getLessonSentenceGroups
from tests.fakes.fake_db import FakeDB
from tests.fakes.seed_factory import seed_content, seed_skill_states

STATES = [
    {"scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
     "status": "learned", "mastery_score": 80, "attempt_count": 2,
     "next_review_at": 1784282400},
    {"scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
     "status": "learning", "mastery_score": 40, "attempt_count": 1},
    {"scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "listening",
     "status": "learning", "mastery_score": 30, "attempt_count": 1},
]


def _run(db, **kw):
    import asyncio
    return asyncio.run(getLessonSentenceGroups(
        db,
        textbook_id=kw.get("textbook_id", "tb_1"),
        lesson_id=kw.get("lesson_id", "l1"),
        scholar_id=kw.get("scholar_id", "scholar_1"),
    ))


class TestLessonNotFound:
    def test_missing_lesson_raises_lesson_not_found(self):
        db = FakeDB()
        with pytest.raises(LessonNotFoundError) as exc:
            _run(db, lesson_id="l_ghost")
        assert "LESSON_NOT_FOUND" in str(exc.value)

    def test_lesson_of_other_textbook_not_found(self):
        db = FakeDB()
        seed_content(db)  # l1 属于 tb_1
        with pytest.raises(LessonNotFoundError):
            _run(db, textbook_id="tb_other", lesson_id="l1")


class TestLegacyTempGroups:
    """无任何 sentence_group → 读兼容层逐句构造临时组，返回结构逐字一致。"""

    def test_no_groups_builds_legacy_per_sentence(self):
        db = FakeDB()
        seed_content(db)
        data = _run(db, lesson_id="l1")

        assert data["lesson_id"] == "l1"
        assert data["lesson_title"] == "L1"
        groups = data["groups"]
        # l1 有 s1/s2 两句 → 2 个临时组
        assert [g["group_id"] for g in groups] == [
            "legacy_l1_s1", "legacy_l1_s2",
        ]
        g0 = groups[0]
        assert g0["group_type"] == "stand_alone"
        assert g0["order_in_lesson"] == 0
        assert g0["group_title"] == "Text s1"[:20]  # text 前 20 字
        assert len(g0["sentences"]) == 1
        assert g0["sentences"][0]["sentence_id"] == "s1"

    def test_legacy_structure_matches_group_structure(self):
        """临时组与真实组的字段键完全一致（调用方零改动）。"""
        db = FakeDB()
        seed_content(db)
        data = _run(db, lesson_id="l1")
        for g in data["groups"]:
            assert {"group_id", "group_title", "group_type", "order_in_lesson", "sentences"} <= set(g)
            for s in g["sentences"]:
                assert {
                    "sentence_id", "content", "translation", "status", "skills",
                    "weakest_skill", "review_count", "next_review_at",
                    "is_canonical", "canonical_sentence_id",
                } <= set(s)


class TestRealGroups:
    def test_groups_organized_by_order_in_lesson(self):
        db = FakeDB()
        seed_content(db)
        # 打乱插入顺序，验证按 order_in_lesson 升序
        db.add("sentence_group", {
            "_id": "grp_2", "group_id": "grp_2", "lesson_id": "l1",
            "title": "G2", "type": "vocab_family", "sentence_ids": ["s2"],
            "order_in_lesson": 1,
        })
        db.add("sentence_group", {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "l1",
            "title": "Dialogue 1", "type": "dialogue_pair",
            "sentence_ids": ["s1"], "order_in_lesson": 0,
        })
        data = _run(db, lesson_id="l1")

        assert [g["group_id"] for g in data["groups"]] == ["grp_1", "grp_2"]
        g1 = data["groups"][0]
        assert g1["group_title"] == "Dialogue 1"
        assert g1["group_type"] == "dialogue_pair"
        assert g1["order_in_lesson"] == 0
        assert [s["sentence_id"] for s in g1["sentences"]] == ["s1"]
        assert data["summary"]["group_count"] == 2

    def test_members_follow_sentence_ids_order(self):
        """组内句子按 sentence_ids 顺序（dialogue_pair 顺序不可乱）。"""
        db = FakeDB()
        seed_content(db)
        db.add("sentence_group", {
            "_id": "grp_d", "group_id": "grp_d", "lesson_id": "l1",
            "title": "D", "type": "dialogue_pair",
            "sentence_ids": ["s2", "s1"],  # 逆序
            "order_in_lesson": 0,
        })
        data = _run(db, lesson_id="l1")
        assert [s["sentence_id"] for s in data["groups"][0]["sentences"]] == ["s2", "s1"]

    def test_other_lesson_groups_ignored(self):
        db = FakeDB()
        seed_content(db)
        db.add("sentence_group", {
            "_id": "grp_l2", "group_id": "grp_l2", "lesson_id": "l2",
            "title": "L2G", "type": "stand_alone", "sentence_ids": ["s3"],
            "order_in_lesson": 0,
        })
        data = _run(db, lesson_id="l1")
        # l1 无分组 → 走 legacy
        assert data["groups"][0]["group_id"] == "legacy_l1_s1"
        assert data["summary"]["group_count"] == 2


class TestStatusAggregation:
    def test_sentence_status_matches_sentences_interface(self):
        db = FakeDB()
        seed_content(db)
        seed_skill_states(db, STATES)
        data = _run(db, lesson_id="l1")

        entries = {s["sentence_id"]: s for g in data["groups"] for s in g["sentences"]}
        s1 = entries["s1"]
        # 乐观聚合：s1 translation=learned(2)，listening=learning(1) → 取最高 learned
        assert s1["status"] == 2  # learned
        assert s1["review_count"] == 2
        assert s1["skills"]["translation"] == 2
        assert s1["skills"]["listening"] == 1
        assert s1["next_review_at"] is not None
        s2 = entries["s2"]
        assert s2["status"] == 1  # learning
        assert s2["review_count"] == 1

    def test_summary_learned_and_mastery(self):
        db = FakeDB()
        seed_content(db)
        seed_skill_states(db, STATES)
        data = _run(db, lesson_id="l1")

        summary = data["summary"]
        assert summary["total_sentence_count"] == 2
        assert summary["learned_sentence_count"] == 1  # 仅 s1
        assert summary["group_count"] == 2  # legacy
        # mastery = (learned=2 + learning=1) / (3*2) = 0.5
        assert summary["mastery"] == pytest.approx(0.5)
        assert summary["skills"]["translation"] == pytest.approx(0.5)


class TestCanonicalFields:
    def test_null_canonical_is_canonical_true(self):
        db = FakeDB()
        seed_content(db)
        data = _run(db, lesson_id="l1")
        s = data["groups"][0]["sentences"][0]
        assert s["is_canonical"] is True
        assert s["canonical_sentence_id"] is None

    def test_self_reference_is_canonical_true(self):
        db = FakeDB()
        seed_content(db)
        db.add("sentence_group", {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "l1",
            "title": "G", "type": "stand_alone", "sentence_ids": ["s1"],
            "order_in_lesson": 0,
        })
        # canonical_sentence_id = 自身 → canonical
        fake_db_add_sentence_canonical(db, "s1", "s1")
        data = _run(db, lesson_id="l1")
        s = data["groups"][0]["sentences"][0]
        assert s["is_canonical"] is True
        assert s["canonical_sentence_id"] == "s1"

    def test_other_reference_is_not_canonical(self):
        db = FakeDB()
        seed_content(db)
        db.add("sentence_group", {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "l1",
            "title": "G", "type": "stand_alone", "sentence_ids": ["s2"],
            "order_in_lesson": 0,
        })
        fake_db_add_sentence_canonical(db, "s2", "s1")  # 重复句，canonical 指向 s1
        data = _run(db, lesson_id="l1")
        s = data["groups"][0]["sentences"][0]
        assert s["is_canonical"] is False
        assert s["canonical_sentence_id"] == "s1"


def fake_db_add_sentence_canonical(db: FakeDB, sid: str, csid: str) -> None:
    """给 FakeDB 中某句追加 canonical_sentence_id（经公开 API 全量重写）。"""
    rows = db.all("sentence_v2")
    db.clear("sentence_v2")
    for r in rows:
        if r.get("sentence_id") == sid:
            r["canonical_sentence_id"] = csid
        db.add("sentence_v2", r)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
