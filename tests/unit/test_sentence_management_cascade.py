"""M5 单元测试 — 编辑/删除的 registry 级联维护（data-model-contract §4.15）

覆盖 M5 验收标准中改句/删句对 sentence_semantic_key 的级联行为：

  编辑（edit_english_sentence，text 变更）：
  - 重复句改文本 → 清 semantic_key/canonical + 从 registry duplicate_sentence_ids 移除
  - canonical 改文本 → 提升剩余 created_at 最早者为新 canonical（写回其 sentence_v2）+ 脱离原簇
  - 簇仅剩 canonical 时改文本 → 删除 registry（簇空删表）
  - 非 text 字段编辑 / text 未变化 → registry 零触碰

  删除（delete_english_sentence）：
  - 删重复句 → 从 registry 移除（semantic_registry_refs_removed=1）
  - 删 canonical → 提升剩余最早者（registry + 新 canonical 的 sentence_v2 自指）
  - 删簇内最后成员 → 删除 registry
  - 无 semantic_key → 零操作（幂等）
"""
from __future__ import annotations

import asyncio

from services.english.sentence_management import (
    delete_english_sentence,
    edit_english_sentence,
)
from services.models_content import compute_text_hash
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _seed_sentence(db: FakeDB, *, sentence_id: str, text: str, created_at: int, **extra) -> None:
    doc = {"sentence_id": sentence_id, "text": text, "created_at": created_at}
    doc.update(extra)
    db.add("sentence_v2", doc)


def _seed_registry(db: FakeDB, *, key: str, canonical: str, dups: list[str]) -> None:
    db.add("sentence_semantic_key", {
        "_id": key,
        "semantic_key": key,
        "canonical_sentence_id": canonical,
        "duplicate_sentence_ids": list(dups),
        "text_hash": key,
        "created_at": 1,
        "updated_at": 1,
    })


def _registry(db: FakeDB) -> dict:
    return {r["semantic_key"]: r for r in db.all("sentence_semantic_key")}


def _sentence(db: FakeDB, sentence_id: str) -> dict:
    return next(r for r in db.all("sentence_v2") if r["sentence_id"] == sentence_id)


# ===========================================================================
# 编辑：text 变更 → 脱离原语义簇（M5 Lazy dedup 口径，新簇下次上报补齐）
# ===========================================================================


class TestEditUnlinksRegistry:
    def test_duplicate_edit_text_removes_from_registry_and_clears_fields(self):
        """重复句改文本 → 从 registry 移除 + 清 semantic_key/canonical_sentence_id。"""
        db = FakeDB()
        key = compute_text_hash("Hello!")
        _seed_sentence(db, sentence_id="s1", text="Hello!", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_sentence(db, sentence_id="s2", text="hello", created_at=200,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=["s2"])
        _run(edit_english_sentence(db, sentence_id="s2", text="Goodbye!"))
        s2 = _sentence(db, "s2")
        assert s2["semantic_key"] is None
        assert s2["canonical_sentence_id"] is None
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "s1"
        assert reg["duplicate_sentence_ids"] == []  # s2 已脱离

    def test_canonical_edit_text_promotes_earliest_remaining(self):
        """canonical 改文本 → 提升剩余 created_at 最早者为新 canonical（registry + sentence_v2 自指）。"""
        db = FakeDB()
        key = compute_text_hash("Bye!")
        _seed_sentence(db, sentence_id="s1", text="Bye!", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_sentence(db, sentence_id="s2", text="bye", created_at=300)
        _seed_sentence(db, sentence_id="s3", text="BYE", created_at=200)
        _seed_registry(db, key=key, canonical="s1", dups=["s2", "s3"])
        _run(edit_english_sentence(db, sentence_id="s1", text="See you!"))
        # s1 脱离原簇
        s1 = _sentence(db, "s1")
        assert s1["semantic_key"] is None
        assert s1["canonical_sentence_id"] is None
        # registry：s3（created_at=200）被提升，且不再出现在 duplicate 列表
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "s3"
        assert reg["duplicate_sentence_ids"] == ["s2"]
        # 新 canonical 的 sentence_v2 写回自指
        assert _sentence(db, "s3")["canonical_sentence_id"] == "s3"

    def test_canonical_edit_text_last_member_deletes_registry(self):
        """簇仅剩 canonical 时改文本 → 删除 registry（簇空删表）。"""
        db = FakeDB()
        key = compute_text_hash("Hi")
        _seed_sentence(db, sentence_id="s1", text="Hi", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=[])
        _run(edit_english_sentence(db, sentence_id="s1", text="Hey!"))
        assert _registry(db) == {}  # 簇空 → registry 删除
        s1 = _sentence(db, "s1")
        assert s1["semantic_key"] is None
        assert s1["canonical_sentence_id"] is None

    def test_edit_non_text_fields_keep_registry(self):
        """只改 translation/knowledge_point_ids → registry 与语义字段零触碰。"""
        db = FakeDB()
        key = compute_text_hash("OK")
        _seed_sentence(db, sentence_id="s1", text="OK", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=[])
        _run(edit_english_sentence(db, sentence_id="s1", translation="好的"))
        s1 = _sentence(db, "s1")
        assert s1["semantic_key"] == key
        assert s1["canonical_sentence_id"] == "s1"
        assert _registry(db)[key]["canonical_sentence_id"] == "s1"

    def test_edit_same_text_no_unlink(self):
        """text 传原值（未变化）→ 不触发脱离，registry 不变。"""
        db = FakeDB()
        key = compute_text_hash("Same!")
        _seed_sentence(db, sentence_id="s1", text="Same!", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=[])
        _run(edit_english_sentence(db, sentence_id="s1", text="Same!"))
        assert _sentence(db, "s1")["semantic_key"] == key
        assert _registry(db)[key]["canonical_sentence_id"] == "s1"


# ===========================================================================
# 删除：脱离语义簇 + canonical 提升 / 簇空删表（data-model §4.15）
# ===========================================================================


class TestDeleteUnlinksRegistry:
    def test_delete_duplicate_removes_from_registry(self):
        """删重复句 → 从 registry duplicate_sentence_ids 移除 + 计数 1。"""
        db = FakeDB()
        key = compute_text_hash("Morning!")
        _seed_sentence(db, sentence_id="sa", text="Morning!", created_at=100,
                       semantic_key=key, canonical_sentence_id="sa")
        _seed_sentence(db, sentence_id="sb", text="morning", created_at=200,
                       semantic_key=key, canonical_sentence_id="sa")
        _seed_sentence(db, sentence_id="sc", text="MORNING", created_at=300,
                       semantic_key=key, canonical_sentence_id="sa")
        _seed_registry(db, key=key, canonical="sa", dups=["sb", "sc"])
        result = _run(delete_english_sentence(
            db, sentence_id="sb", confirm_text="morning"))
        assert result["deleted"]["semantic_registry_refs_removed"] == 1
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "sa"
        assert reg["duplicate_sentence_ids"] == ["sc"]

    def test_delete_canonical_promotes_earliest_remaining(self):
        """删 canonical → 提升剩余 created_at 最早者为新 canonical（写回 sentence_v2 自指）。"""
        db = FakeDB()
        key = compute_text_hash("Good night")
        _seed_sentence(db, sentence_id="s1", text="Good night", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_sentence(db, sentence_id="s2", text="good night", created_at=400)
        _seed_sentence(db, sentence_id="s3", text="GOOD NIGHT", created_at=200)
        _seed_registry(db, key=key, canonical="s1", dups=["s2", "s3"])
        result = _run(delete_english_sentence(
            db, sentence_id="s1", confirm_text="Good night"))
        assert result["deleted"]["semantic_registry_refs_removed"] == 1
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "s3"  # 最早剩余者
        assert reg["duplicate_sentence_ids"] == ["s2"]  # s3 提升后移出 dup 列表
        assert _sentence(db, "s3")["canonical_sentence_id"] == "s3"

    def test_delete_last_member_deletes_registry(self):
        """删簇内最后成员（canonical，无 dup）→ 删除 registry + 计数 1。"""
        db = FakeDB()
        key = compute_text_hash("Alone")
        _seed_sentence(db, sentence_id="s1", text="Alone", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=[])
        result = _run(delete_english_sentence(
            db, sentence_id="s1", confirm_text="Alone"))
        assert result["deleted"]["semantic_registry_refs_removed"] == 1
        assert _registry(db) == {}

    def test_delete_without_semantic_key_zero_op(self):
        """无 semantic_key 的句子删除 → registry 零触碰（幂等）。"""
        db = FakeDB()
        key = compute_text_hash("Keep")
        _seed_sentence(db, sentence_id="s1", text="Keep", created_at=100,
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=[])
        _seed_sentence(db, sentence_id="s_plain", text="Plain", created_at=200)
        result = _run(delete_english_sentence(
            db, sentence_id="s_plain", confirm_text="Plain"))
        assert result["deleted"]["semantic_registry_refs_removed"] == 0
        assert _registry(db)[key]["canonical_sentence_id"] == "s1"  # 不受影响

    def test_delete_duplicates_recursive_removes_cluster(self):
        """delete_duplicates=true：canonical + 全部重复句删除 → registry 一并删除。"""
        db = FakeDB()
        key = compute_text_hash("Dup!")
        _seed_sentence(db, sentence_id="s1", text="Dup!", created_at=100,
                       textbook_id="tb_x", lesson_id="ls_x",
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_sentence(db, sentence_id="s2", text="dup", created_at=200,
                       textbook_id="tb_x", lesson_id="ls_x",
                       semantic_key=key, canonical_sentence_id="s1")
        _seed_registry(db, key=key, canonical="s1", dups=["s2"])
        result = _run(delete_english_sentence(
            db, sentence_id="s1", confirm_text="Dup!", delete_duplicates=True))
        assert result["duplicates_deleted"] == 1
        assert _registry(db) == {}  # 簇随所有成员删除而删除
        assert result["deleted"]["semantic_registry_refs_removed"] >= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
