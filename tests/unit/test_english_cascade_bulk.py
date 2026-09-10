"""批量级联清理单测：_cascade_delete_sentences（E-API-12 去重执行阶段）

覆盖：
- 4 张状态表 `$in` 分块物理删（含 >500 待删分块冒烟）
- conversation_turn 批量标记 deleted_sentence_ref（不物理删）
- sentence_group 组内引用摘除（不删组）
- sentence_semantic_key registry：duplicate 摘除 / canonical 删除提升 / 簇空删表
- 空输入幂等（全 0 汇总，零写入）
"""
from __future__ import annotations

import asyncio

from services.english.sentence_management import _cascade_delete_sentences
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _add_sentence(
    db: FakeDB,
    *,
    sentence_id: str,
    text: str,
    created_at: int,
    semantic_key: str | None = None,
    canonical_sentence_id: str | None = None,
) -> None:
    doc = {
        "sentence_id": sentence_id,
        "text": text,
        "textbook_id": "tb_bulk",
        "lesson_id": "ls_1",
        "chapter_id": "ch_1",
        "created_at": created_at,
    }
    if semantic_key is not None:
        doc["semantic_key"] = semantic_key
    if canonical_sentence_id is not None:
        doc["canonical_sentence_id"] = canonical_sentence_id
    db.add("sentence_v2", doc)


def _add_registry(
    db: FakeDB, *, key: str, canonical: str, duplicates: list[str]
) -> None:
    db.add(
        "sentence_semantic_key",
        {
            "_id": key,
            "semantic_key": key,
            "canonical_sentence_id": canonical,
            "duplicate_sentence_ids": list(duplicates),
            "text_hash": key,
            "created_at": 1000,
            "updated_at": 1000,
        },
    )


class TestBulkCascade:
    def test_removes_related_groups_turns_and_registry_duplicates(self):
        """典型去重形态：canonical 保留，dup 批量删除 → 状态表删、组摘引用、
        turn 标记、registry duplicate 摘除；canonical 自己的关联记录不受影响。"""
        db = FakeDB()
        _add_sentence(
            db, sentence_id="s1", text="Hello!", created_at=100,
            semantic_key="h1", canonical_sentence_id="s1",
        )
        _add_sentence(
            db, sentence_id="s2", text="hello", created_at=200, semantic_key="h1"
        )
        _add_sentence(
            db, sentence_id="s3", text="HELLO", created_at=300, semantic_key="h1"
        )
        _add_sentence(db, sentence_id="s4", text="Unique.", created_at=400)
        _add_registry(db, key="h1", canonical="s1", duplicates=["s2", "s3"])

        # canonical s1 的关联记录必须保留
        db.add("learning_attempt", {"sentence_id": "s1", "scholar_id": "u0"})
        db.add("study_attempt", {"sentence_id": "s2", "scholar_id": "u1"})
        db.add("study_attempt", {"sentence_id": "s2", "scholar_id": "u2"})
        db.add("study_attempt", {"sentence_id": "s3", "scholar_id": "u3"})
        db.add("skill_state", {"sentence_id": "s2", "scholar_id": "u1"})

        db.add("sentence_group", {"group_id": "g1", "sentence_ids": ["s1", "s2", "s3"]})
        db.add("sentence_group", {"group_id": "g2", "sentence_ids": ["s2"]})
        db.add("sentence_group", {"group_id": "g3", "sentence_ids": ["s4"]})

        db.add("conversation_turn", {"turn_id": "t1", "utterance": "ref s2 出现", "reply": ""})
        db.add("conversation_turn", {"turn_id": "t2", "utterance": "", "reply": "ref s3 出现"})
        db.add(
            "conversation_turn",
            {"turn_id": "t3", "utterance": "ref s1", "reply": "", "deleted_sentence_ref": True},
        )
        db.add("conversation_turn", {"turn_id": "t4", "utterance": "无关", "reply": ""})

        deleted = _run(
            _cascade_delete_sentences(db, sentence_ids=["s2", "s3"])
        )

        # 汇总计数
        assert deleted["study_attempt"] == 3
        assert deleted["skill_state"] == 1
        assert deleted["speech_evaluation"] == 0
        assert deleted["learning_attempt"] == 0
        assert deleted["audio_asset"] == 0
        assert deleted["conversation_turn_marked"] == 2
        assert deleted["sentence_group_refs_removed"] == 2
        assert deleted["semantic_registry_refs_removed"] == 1

        # 状态表：canonical s1 的 learning_attempt 保留
        assert db.all("study_attempt") == []
        assert db.all("skill_state") == []
        assert {r["sentence_id"] for r in db.all("learning_attempt")} == {"s1"}

        # 组：摘引用不删组
        groups = {g["group_id"]: g["sentence_ids"] for g in db.all("sentence_group")}
        assert groups == {"g1": ["s1"], "g2": [], "g3": ["s4"]}

        # turn：命中标记、幂等已标记保持、无关不动
        turns = {t["turn_id"]: t.get("deleted_sentence_ref") for t in db.all("conversation_turn")}
        assert turns == {"t1": True, "t2": True, "t3": True, "t4": None}

        # registry：dup 摘除，canonical 保持
        reg = db.all("sentence_semantic_key")[0]
        assert reg["canonical_sentence_id"] == "s1"
        assert reg["duplicate_sentence_ids"] == []

    def test_canonical_deleted_promotes_earliest_duplicate(self):
        """canonical 被删（跨教材去重等场景）：剩余 dup 中 created_at 最早者
        提升为新 canonical，写回 registry + sentence_v2；canonical 不入 dup 列表。"""
        db = FakeDB()
        _add_sentence(
            db, sentence_id="s1", text="Same!", created_at=100,
            semantic_key="h2", canonical_sentence_id="s1",
        )
        _add_sentence(
            db, sentence_id="s2", text="same", created_at=50, semantic_key="h2"
        )
        _add_sentence(
            db, sentence_id="s3", text="SAME", created_at=200, semantic_key="h2"
        )
        _add_registry(db, key="h2", canonical="s1", duplicates=["s2", "s3"])

        deleted = _run(_cascade_delete_sentences(db, sentence_ids=["s1"]))

        assert deleted["semantic_registry_refs_removed"] == 1
        reg = db.all("sentence_semantic_key")[0]
        assert reg["canonical_sentence_id"] == "s2"
        assert reg["duplicate_sentence_ids"] == ["s3"]
        # sentence_v2 写回：新 canonical 自指
        s2 = next(r for r in db.all("sentence_v2") if r["sentence_id"] == "s2")
        assert s2["canonical_sentence_id"] == "s2"

    def test_canonical_deleted_cluster_emptied_deletes_registry(self):
        """canonical 被删且簇内无剩余 dup → 删 registry。"""
        db = FakeDB()
        _add_sentence(
            db, sentence_id="s1", text="Only!", created_at=100,
            semantic_key="h3", canonical_sentence_id="s1",
        )
        _add_registry(db, key="h3", canonical="s1", duplicates=[])

        deleted = _run(_cascade_delete_sentences(db, sentence_ids=["s1"]))

        assert deleted["semantic_registry_refs_removed"] == 1
        assert db.all("sentence_semantic_key") == []

    def test_sentence_without_semantic_key_skips_registry(self):
        """存量句无 semantic_key（未走 M3/M5）：跳过 registry 维护，其余级联照常。"""
        db = FakeDB()
        _add_sentence(db, sentence_id="s1", text="Legacy.", created_at=100)
        db.add("study_attempt", {"sentence_id": "s1", "scholar_id": "u1"})

        deleted = _run(_cascade_delete_sentences(db, sentence_ids=["s1"]))

        assert deleted["study_attempt"] == 1
        assert deleted["semantic_registry_refs_removed"] == 0
        assert db.all("sentence_semantic_key") == []

    def test_empty_input_is_noop(self):
        db = FakeDB()
        _add_sentence(db, sentence_id="s1", text="Hello!", created_at=100)
        deleted = _run(_cascade_delete_sentences(db, sentence_ids=[]))
        assert all(v == 0 for v in deleted.values())
        assert len(db.all("sentence_v2")) == 1

    def test_chunked_delete_over_500(self):
        """>500 待删句验证 $in 分块物理删路径（500/块）。"""
        db = FakeDB()
        sids = [f"s_{i:04d}" for i in range(520)]
        for sid in sids:
            db.add("study_attempt", {"sentence_id": sid, "scholar_id": "u1"})
        # 仅前 20 句有 skill_state，验证部分命中
        for sid in sids[:20]:
            db.add("skill_state", {"sentence_id": sid, "scholar_id": "u1"})

        deleted = _run(_cascade_delete_sentences(db, sentence_ids=sids))

        assert deleted["study_attempt"] == 520
        assert deleted["skill_state"] == 20
        assert db.all("study_attempt") == []
