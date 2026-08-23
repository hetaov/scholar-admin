"""M3 G1.2 + M5 单元测试 — ensureSentenceSemanticKey（Lazy dedup + sentence_semantic_key 落表）

覆盖 service-contract §8.5 + api-contract §3.2 API-G3 + data-model §4.15：

  - 句子不存在 → SentenceNotFoundError（404 SENTENCE_NOT_FOUND）
  - 已有 semantic_key → 零写 sentence_v2（幂等，不触发 update；M5 registry 缺失时回填）
  - 无重复 → 自指 canonical（semantic_key = compute_text_hash(text)）
  - 有重复 → canonical = 同 hash 组 created_at 最早者（M3 口径）
  - 组内已有 canonical（自指 / 指向他句）→ 指向之（优先于 created_at 规则）
  - text 入参覆盖（可选）
  - 空文本 → 不写语义键，返回现状
  - 仅写 sentence_v2 + sentence_semantic_key，不触碰 skill_state 写入键
  - M5 registry：建簇 / 登记重复 / 幂等追加 / M3 存量回填 / canonical 后报不误登记
"""
from __future__ import annotations

import asyncio

import pytest

from services.english import SentenceNotFoundError
from services.english.sentence_management import ensureSentenceSemanticKey
from services.models_content import compute_text_hash
from tests.fakes.fake_db import FakeDB


def asyncio_run(coro):
    return asyncio.run(coro)


def _seed_sentence(db: FakeDB, *, sentence_id: str, text: str, created_at: int, **extra) -> None:
    doc = {"sentence_id": sentence_id, "text": text, "created_at": created_at}
    doc.update(extra)
    db.add("sentence_v2", doc)


class TestSentenceNotFound:
    def test_missing_sentence_raises(self):
        db = FakeDB()
        with pytest.raises(SentenceNotFoundError):
            asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s_none"))

    def test_empty_sentence_id_raises(self):
        db = FakeDB()
        with pytest.raises(SentenceNotFoundError):
            asyncio_run(ensureSentenceSemanticKey(db, sentence_id=""))


class TestAlreadyBackfilled:
    def test_returns_existing_without_write(self):
        db = FakeDB()
        _seed_sentence(
            db, sentence_id="s1", text="Hello!", created_at=1, updated_at=1,
            semantic_key="sk_existing", canonical_sentence_id="s_other",
        )
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result == {
            "sentence_id": "s1",
            "semantic_key": "sk_existing",
            "canonical_sentence_id": "s_other",
        }
        # 零写：updated_at 未被刷新
        stored = db.all("sentence_v2")[0]
        assert stored["updated_at"] == 1


class TestSelfCanonical:
    def test_single_sentence_backfills_self_canonical(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Hello!", created_at=1000)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        key = compute_text_hash("Hello!")
        assert result == {
            "sentence_id": "s1",
            "semantic_key": key,
            "canonical_sentence_id": "s1",  # 自指 = canonical
        }
        stored = db.all("sentence_v2")[0]
        assert stored["semantic_key"] == key
        assert stored["canonical_sentence_id"] == "s1"


class TestDuplicateCanonicalSelection:
    def test_points_to_earliest_created_at(self):
        db = FakeDB()
        # 同归一文本（标点/大小写差异），s1 更早 → canonical = s1
        _seed_sentence(db, sentence_id="s1", text="Hello!", created_at=1000)
        _seed_sentence(db, sentence_id="s2", text="hello", created_at=2000)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))
        assert result["semantic_key"] == compute_text_hash("hello")
        assert result["canonical_sentence_id"] == "s1"
        by_id = {r["sentence_id"]: r for r in db.all("sentence_v2")}
        assert by_id["s2"]["canonical_sentence_id"] == "s1"
        # s1 未上报 → 未被触碰（惰性，冷门句零成本）
        assert by_id["s1"].get("semantic_key") is None

    def test_earliest_self_becomes_canonical(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Hi", created_at=500)
        _seed_sentence(db, sentence_id="s2", text="Hi!", created_at=1500)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result["canonical_sentence_id"] == "s1"  # 自身最早 → 自指

    def test_existing_canonical_wins_over_created_at(self):
        db = FakeDB()
        # s3 虽非最早，但已是 canonical（自指）→ 优先指向之
        _seed_sentence(db, sentence_id="s1", text="OK", created_at=100)
        _seed_sentence(
            db, sentence_id="s3", text="ok!", created_at=300,
            semantic_key=compute_text_hash("ok!"), canonical_sentence_id="s3",
        )
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result["canonical_sentence_id"] == "s3"

    def test_peer_pointing_to_other_member_resolves(self):
        db = FakeDB()
        _seed_sentence(
            db, sentence_id="s5", text="Bye", created_at=100,
            semantic_key=compute_text_hash("bye"), canonical_sentence_id="s5",
        )
        _seed_sentence(
            db, sentence_id="s6", text="bye!", created_at=50,
            semantic_key=compute_text_hash("bye!"), canonical_sentence_id="s5",
        )
        # s7 上报：组内 s6 指向 s5 → 解析到 s5（即使 s6 created_at 更早）
        _seed_sentence(db, sentence_id="s7", text="BYE", created_at=10)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s7"))
        assert result["canonical_sentence_id"] == "s5"


class TestTextOverride:
    def test_text_param_used_for_hash(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="old text", created_at=1)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1", text="New Text"))
        assert result["semantic_key"] == compute_text_hash("New Text")


class TestEmptyText:
    def test_empty_text_does_not_write(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="", created_at=1, updated_at=1)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result["semantic_key"] is None
        assert result["canonical_sentence_id"] is None
        stored = db.all("sentence_v2")[0]
        assert stored.get("semantic_key") is None
        assert stored["updated_at"] == 1  # 未写


class TestSkillStateUntouched:
    def test_does_not_touch_skill_state_write_key(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Hello", created_at=1)
        db.add("skill_state", {
            "_id": "u1_s1_translation",
            "scholar_id": "u1",
            "sentence_id": "s1",
            "skill_code": "translation",
            "status": "learned",
        })
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        states = db.all("skill_state")
        assert len(states) == 1
        assert states[0]["sentence_id"] == "s1"  # 写入键零变化


# ===========================================================================
# M5 — sentence_semantic_key 落表（data-model §4.15，registry 成为权威源）
# ===========================================================================


def _registry(db):
    rows = db.all("sentence_semantic_key")
    return {r["semantic_key"]: r for r in rows}


class TestRegistryCreatedOnBackfill:
    def test_single_sentence_creates_self_canonical_registry(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Hello!", created_at=1000)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        key = compute_text_hash("Hello!")
        reg = _registry(db)[key]
        assert reg["_id"] == key
        assert reg["canonical_sentence_id"] == "s1"
        assert reg["duplicate_sentence_ids"] == []
        assert reg["text_hash"] == key
        # sentence_v2 与 registry 保持一致
        stored = db.all("sentence_v2")[0]
        assert stored["semantic_key"] == key
        assert stored["canonical_sentence_id"] == "s1"

    def test_duplicate_creates_registry_with_canonical_and_dups(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Good morning!", created_at=1000)
        _seed_sentence(db, sentence_id="s2", text="good morning", created_at=2000)
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))
        key = compute_text_hash("good morning")
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "s1"  # 最早者
        assert reg["duplicate_sentence_ids"] == ["s2"]
        assert result["canonical_sentence_id"] == "s1"

    def test_existing_canonical_seed_registry_keeps_it(self):
        """M3 存量：s3 已是 canonical（自指）→ registry canonical = s3（不因 created_at 更晚而变更）。"""
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="OK", created_at=100)
        _seed_sentence(
            db, sentence_id="s3", text="ok!", created_at=300,
            semantic_key=compute_text_hash("ok!"), canonical_sentence_id="s3",
        )
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        reg = _registry(db)[compute_text_hash("ok!")]
        assert reg["canonical_sentence_id"] == "s3"
        assert result["canonical_sentence_id"] == "s3"


class TestRegistryExistingJoins:
    def test_new_duplicate_appends_to_registry(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Bye", created_at=100)
        _seed_sentence(db, sentence_id="s2", text="bye!", created_at=200)
        # 先报 s1 → 建簇 canonical=s1
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        # 再报 s2 → 命中 registry，指向 s1 并登记重复
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))
        assert result["canonical_sentence_id"] == "s1"
        reg = _registry(db)[compute_text_hash("bye!")]
        assert reg["duplicate_sentence_ids"] == ["s2"]
        by_id = {r["sentence_id"]: r for r in db.all("sentence_v2")}
        assert by_id["s2"]["canonical_sentence_id"] == "s1"

    def test_re_report_does_not_duplicate_registry_entry(self):
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Hi", created_at=100)
        _seed_sentence(db, sentence_id="s2", text="hi", created_at=200)
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))
        # 重复上报 s2 → 幂等：duplicate_sentence_ids 不追加重复项
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))
        reg = _registry(db)[compute_text_hash("hi")]
        assert reg["duplicate_sentence_ids"] == ["s2"]

    def test_canonical_late_report_not_marked_duplicate(self):
        """canonical 句后上报 → 自指 canonical，不进入 duplicate_sentence_ids。"""
        db = FakeDB()
        _seed_sentence(db, sentence_id="s1", text="Nice", created_at=100)
        _seed_sentence(db, sentence_id="s2", text="nice", created_at=200)
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s2"))  # 建簇 canonical=s1
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result["canonical_sentence_id"] == "s1"
        reg = _registry(db)[compute_text_hash("nice")]
        assert reg["canonical_sentence_id"] == "s1"
        assert reg["duplicate_sentence_ids"] == ["s2"]
        by_id = {r["sentence_id"]: r for r in db.all("sentence_v2")}
        assert by_id["s1"]["canonical_sentence_id"] == "s1"  # 自指


class TestRegistryBackfillM3Legacy:
    def test_sentence_with_semantic_key_backfills_registry(self):
        """M3 存量（M3 不落库本表）：已带 semantic_key 的句子 → 回填 registry（不改 sentence_v2）。"""
        db = FakeDB()
        key = compute_text_hash("Legacy!")
        _seed_sentence(
            db, sentence_id="s1", text="Legacy!", created_at=1000, updated_at=1000,
            semantic_key=key, canonical_sentence_id="s1",
        )
        result = asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))
        assert result["semantic_key"] == key
        # registry 回填
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "s1"
        # sentence_v2 零写：updated_at 未被刷新
        stored = db.all("sentence_v2")[0]
        assert stored["updated_at"] == 1000

    def test_legacy_duplicate_backfills_full_duplicate_list(self):
        """M3 存量簇：canonical 已有 semantic_key，重复句字段也在 → 回填完整 duplicate 列表。"""
        db = FakeDB()
        key = compute_text_hash("Morning")
        _seed_sentence(
            db, sentence_id="sa", text="Morning", created_at=100,
            semantic_key=key, canonical_sentence_id="sa",
        )
        _seed_sentence(
            db, sentence_id="sb", text="morning!", created_at=200,
            semantic_key=key, canonical_sentence_id="sa",
        )
        # 上报 sb（已有 semantic_key）→ 回填 registry 时应包含 sb
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="sb"))
        reg = _registry(db)[key]
        assert reg["canonical_sentence_id"] == "sa"
        assert set(reg["duplicate_sentence_ids"]) == {"sb"}

    def test_already_existing_registry_not_duplicated(self):
        db = FakeDB()
        key = compute_text_hash("Ok!")
        _seed_sentence(
            db, sentence_id="s1", text="Ok!", created_at=1,
            semantic_key=key, canonical_sentence_id="s1",
        )
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))  # 回填
        assert len(_registry(db)) == 1
        asyncio_run(ensureSentenceSemanticKey(db, sentence_id="s1"))  # 再报
        assert len(_registry(db)) == 1  # 幂等


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
