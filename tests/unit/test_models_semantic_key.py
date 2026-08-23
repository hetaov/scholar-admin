"""M5 单元测试 — sentence_semantic_key 集合模型（data-model-contract §4.15，SOP ④ DM-G2）

覆盖：
- build_sentence_semantic_key_doc：字段逐字对齐契约 §4.15（_id = semantic_key）
- normalize_sentence_semantic_key_doc：可选字段缺省注入（读兼容层）
- get_sentence_semantic_key / get_sentence_semantic_key_by_text_hash：经 FakeDB 查询
"""
from __future__ import annotations

import asyncio

from services.models_content import (
    build_sentence_semantic_key_doc,
    get_sentence_semantic_key,
    get_sentence_semantic_key_by_text_hash,
    normalize_sentence_semantic_key_doc,
)
from tests.fakes.fake_db import FakeDB


class TestBuildSentenceSemanticKeyDoc:
    def test_builds_contract_aligned_doc(self):
        doc = build_sentence_semantic_key_doc(
            semantic_key="sk_abc",
            canonical_sentence_id="s1",
            duplicate_sentence_ids=["s2", "s3"],
            now=1000,
        )
        assert doc["_id"] == "sk_abc"
        assert doc["semantic_key"] == "sk_abc"
        assert doc["canonical_sentence_id"] == "s1"
        assert doc["duplicate_sentence_ids"] == ["s2", "s3"]
        assert doc["text_hash"] == "sk_abc"  # L1 同源缺省
        assert doc["similarity_score"] is None  # 仅 L2 写入
        assert doc["build_version"] is None     # 仅 M4 写入
        assert doc["created_at"] == 1000
        assert doc["updated_at"] == 1000

    def test_text_hash_explicit_overrides(self):
        doc = build_sentence_semantic_key_doc(
            semantic_key="sk_embed", canonical_sentence_id="s1",
            text_hash="sk_l1", now=1,
        )
        assert doc["text_hash"] == "sk_l1"  # L2 聚类键与 L1 hash 分离

    def test_duplicates_default_empty_and_copied(self):
        doc = build_sentence_semantic_key_doc(
            semantic_key="sk", canonical_sentence_id="s1", now=1,
        )
        assert doc["duplicate_sentence_ids"] == []

    def test_mutating_input_list_not_alias(self):
        dup = ["s2"]
        doc = build_sentence_semantic_key_doc(
            semantic_key="sk", canonical_sentence_id="s1", duplicate_sentence_ids=dup, now=1,
        )
        dup.append("s3")
        assert doc["duplicate_sentence_ids"] == ["s2"]


class TestNormalizeSentenceSemanticKeyDoc:
    def test_injects_defaults_for_missing_optional_fields(self):
        doc = normalize_sentence_semantic_key_doc({
            "_id": "sk",
            "semantic_key": "sk",
            "canonical_sentence_id": "s1",
            "created_at": 1,
        })
        assert doc["duplicate_sentence_ids"] == []
        assert doc["text_hash"] == "sk"  # 缺省 = semantic_key
        assert doc["similarity_score"] is None
        assert doc["build_version"] is None

    def test_keeps_explicit_values(self):
        doc = normalize_sentence_semantic_key_doc({
            "semantic_key": "sk",
            "canonical_sentence_id": "s1",
            "duplicate_sentence_ids": ["s2"],
            "text_hash": "th",
            "similarity_score": 0.92,
            "build_version": "b1",
        })
        assert doc["duplicate_sentence_ids"] == ["s2"]
        assert doc["text_hash"] == "th"
        assert doc["similarity_score"] == 0.92
        assert doc["build_version"] == "b1"

    def test_does_not_mutate_input(self):
        original = {"semantic_key": "sk", "canonical_sentence_id": "s1"}
        normalize_sentence_semantic_key_doc(original)
        assert "duplicate_sentence_ids" not in original


class TestGetSentenceSemanticKey:
    def test_by_pk(self):
        db = FakeDB()
        db.add("sentence_semantic_key", {
            "_id": "sk_1", "semantic_key": "sk_1", "canonical_sentence_id": "s1",
            "duplicate_sentence_ids": ["s2"], "created_at": 1,
        })
        doc = asyncio.run(get_sentence_semantic_key(db, "sk_1"))
        assert doc is not None
        assert doc["canonical_sentence_id"] == "s1"
        assert doc["duplicate_sentence_ids"] == ["s2"]

    def test_by_pk_missing_returns_none(self):
        db = FakeDB()
        assert asyncio.run(get_sentence_semantic_key(db, "sk_nope")) is None

    def test_by_pk_empty_returns_none(self):
        db = FakeDB()
        assert asyncio.run(get_sentence_semantic_key(db, "")) is None

    def test_by_text_hash(self):
        db = FakeDB()
        db.add("sentence_semantic_key", {
            "_id": "sk_1", "semantic_key": "sk_1", "text_hash": "th_1",
            "canonical_sentence_id": "s1", "created_at": 1,
        })
        doc = asyncio.run(get_sentence_semantic_key_by_text_hash(db, "th_1"))
        assert doc is not None
        assert doc["canonical_sentence_id"] == "s1"

    def test_by_text_hash_missing_returns_none(self):
        db = FakeDB()
        assert asyncio.run(get_sentence_semantic_key_by_text_hash(db, "th_nope")) is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
