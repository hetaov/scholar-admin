"""E0.1 单元测试 — sentence_v2 text_hash 字段扩展 + 惰性计算 getter 兼容

覆盖 SOP E0.1 验收标准 5 条：
  1. normalize_sentence_text：中英文标点去除 / 大小写统一 / 空白压缩
  2. compute_text_hash：相同 normalize → 相同 hash；不同 text → 不同 hash；空 text → ''
  3. normalize_sentence_doc getter 兼容：缺字段 / None / 空串 → 惰性计算注入；有值保留；返回副本不突变输入
  4. FakeDB query 注入：sentence_v2 集合的 GET 即使读原始缺字段也输出 text_hash（与生产 database.py 同逻辑）
  5. TDD 铁律：RED → GREEN → REGRESSION 零破坏

契约依据：data-model-contract.md §4.3 DM-1（text_hash 字段 + normalize 规则 + 惰性计算兼容层）
"""
from __future__ import annotations

import asyncio
import hashlib

import pytest


# ===========================================================================
# 1. normalize_sentence_text — 归一化规则（去标点 + 大小写 + 压缩空白）
# ===========================================================================


class TestNormalizeSentenceText:
    """normalize_sentence_text：trim + lower + 去标点(！？。，、!?,.) + 压缩空白"""

    def test_imports(self):
        from services.models_content import normalize_sentence_text  # noqa: F401

    def test_half_width_punct_removed(self):
        """半角标点 ! ?, . 被移除。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("Hello!") == "hello"
        assert normalize_sentence_text("How are you?") == "how are you"
        assert normalize_sentence_text("Yes, no, maybe.") == "yes no maybe"

    def test_full_width_punct_removed(self):
        """全角标点 ！？。，、 被移除。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("你好。") == "你好"
        assert normalize_sentence_text("真的？") == "真的"
        assert normalize_sentence_text("你好，世界！") == "你好世界"

    def test_lowercase(self):
        """大小写统一为小写。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("HELLO") == "hello"
        assert normalize_sentence_text("Hello World") == "hello world"
        assert normalize_sentence_text("AbCdEf") == "abcdef"

    def test_whitespace_compressed(self):
        """连续空白压缩为单空格 + strip。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("  hello  ") == "hello"
        assert normalize_sentence_text("hello   world") == "hello world"
        assert normalize_sentence_text("\thello\tworld\n") == "hello world"

    def test_combined_punct_and_case_and_whitespace(self):
        """组合：标点 + 大小写 + 空白同时处理。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("  Hello!  How are you?  ") == "hello how are you"

    def test_empty_text_returns_empty(self):
        """空字符串 / None → 返回 ''。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("") == ""
        assert normalize_sentence_text(None) == ""

    def test_text_without_punct_unchanged_except_case(self):
        """无标点文本仅做大小写 + 空白归一。"""
        from services.models_content import normalize_sentence_text

        assert normalize_sentence_text("Good Morning") == "good morning"


# ===========================================================================
# 2. compute_text_hash — sha256 指纹计算
# ===========================================================================


class TestComputeTextHash:
    """compute_text_hash：normalize 后 sha256，64 字符 hex"""

    def test_imports(self):
        from services.models_content import compute_text_hash  # noqa: F401

    def test_same_text_same_hash(self):
        """相同文本 → 相同 hash。"""
        from services.models_content import compute_text_hash

        h1 = compute_text_hash("Hello!")
        h2 = compute_text_hash("Hello!")
        assert h1 == h2

    def test_equivalent_after_normalize_same_hash(self):
        """normalize 后等价的文本 → 相同 hash（核心：L1 重复检测依据）。"""
        from services.models_content import compute_text_hash

        # 标点差异 → 同 hash
        assert compute_text_hash("Hello!") == compute_text_hash("Hello")
        # 大小写差异 → 同 hash
        assert compute_text_hash("HELLO") == compute_text_hash("hello")
        # 空白差异 → 同 hash
        assert compute_text_hash("hello   world") == compute_text_hash("hello world")
        # 组合差异 → 同 hash
        assert compute_text_hash("  Hello!  ") == compute_text_hash("hello")

    def test_different_text_different_hash(self):
        """不同文本 → 不同 hash。"""
        from services.models_content import compute_text_hash

        assert compute_text_hash("Hello!") != compute_text_hash("Goodbye!")
        assert compute_text_hash("apple") != compute_text_hash("apply")

    def test_hash_is_64_char_hex(self):
        """sha256 → 64 字符 hex 串。"""
        from services.models_content import compute_text_hash

        h = compute_text_hash("Hello!")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_matches_manual_sha256(self):
        """与手动 sha256(normalize(text)) 结果一致。"""
        from services.models_content import compute_text_hash, normalize_sentence_text

        text = "Hello! How are you?"
        expected = hashlib.sha256(normalize_sentence_text(text).encode("utf-8")).hexdigest()
        assert compute_text_hash(text) == expected

    def test_empty_text_returns_empty_string(self):
        """空文本 → 返回 ''（不计算 hash，避免空字符串误判重复）。"""
        from services.models_content import compute_text_hash

        assert compute_text_hash("") == ""
        assert compute_text_hash(None) == ""


# ===========================================================================
# 3. normalize_sentence_doc — getter 兼容层（惰性计算注入，不写回 DB）
# ===========================================================================


class TestNormalizeSentenceDocGetterCompat:
    """normalize_sentence_doc：读侧 getter，对 sentence_v2 单条记录惰性注入 text_hash"""

    def test_imports(self):
        from services.models_content import normalize_sentence_doc  # noqa: F401

    def test_missing_text_hash_computes_from_text(self):
        """存量记录无 text_hash 字段 → 按 text 惰性计算注入。"""
        from services.models_content import normalize_sentence_doc, compute_text_hash

        legacy = {
            "sentence_id": "s_001",
            "text": "Hello!",
            "translation": "你好！",
        }
        out = normalize_sentence_doc(legacy)
        assert out["text_hash"] == compute_text_hash("Hello!")

    def test_text_hash_none_computes_from_text(self):
        """text_hash=None → 惰性计算注入。"""
        from services.models_content import normalize_sentence_doc, compute_text_hash

        doc = {"sentence_id": "s_002", "text": "Good morning", "text_hash": None}
        out = normalize_sentence_doc(doc)
        assert out["text_hash"] == compute_text_hash("Good morning")

    def test_text_hash_empty_string_computes_from_text(self):
        """text_hash='' → 惰性计算注入。"""
        from services.models_content import normalize_sentence_doc, compute_text_hash

        doc = {"sentence_id": "s_003", "text": "How are you?", "text_hash": ""}
        out = normalize_sentence_doc(doc)
        assert out["text_hash"] == compute_text_hash("How are you?")

    def test_text_hash_present_keeps_original_value(self):
        """text_hash 有值 → 保留原值不覆盖。"""
        from services.models_content import normalize_sentence_doc

        doc = {"sentence_id": "s_004", "text": "Hello!", "text_hash": "precomputed_hash_value"}
        out = normalize_sentence_doc(doc)
        assert out["text_hash"] == "precomputed_hash_value"

    def test_text_missing_returns_empty_text_hash(self):
        """text 缺失 → text_hash=''（不计算 hash）。"""
        from services.models_content import normalize_sentence_doc

        doc = {"sentence_id": "s_005"}  # 无 text 字段
        out = normalize_sentence_doc(doc)
        assert out["text_hash"] == ""

    def test_text_empty_returns_empty_text_hash(self):
        """text='' → text_hash=''。"""
        from services.models_content import normalize_sentence_doc

        doc = {"sentence_id": "s_006", "text": ""}
        out = normalize_sentence_doc(doc)
        assert out["text_hash"] == ""

    def test_returns_copy_does_not_mutate_input(self):
        """返回副本，不修改传入对象（避免副作用污染调用方原始对象）。"""
        from services.models_content import normalize_sentence_doc

        legacy = {"sentence_id": "s_007", "text": "Hello!"}
        legacy_keys_before = set(legacy.keys())
        out = normalize_sentence_doc(legacy)
        assert out is not legacy
        assert set(legacy.keys()) == legacy_keys_before
        assert "text_hash" not in legacy

    def test_preserves_other_fields(self):
        """getter 不破坏既有字段。"""
        from services.models_content import normalize_sentence_doc

        doc = {
            "sentence_id": "s_008",
            "text": "Hello!",
            "translation": "你好！",
            "audio_url": "https://example.com/a.mp3",
            "knowledge_point_ids": ["kp_1", "kp_2"],
            "textbook_id": "tb_1",
            "lesson_id": "ls_1",
            "chapter_id": "ch_1",
        }
        out = normalize_sentence_doc(doc)
        for k, v in doc.items():
            assert out[k] == v
        assert "text_hash" in out


# ===========================================================================
# 4. FakeDB query 注入 — sentence_v2 集合 GET 输出 text_hash
# ===========================================================================


class TestFakeDBQueryInjectsTextHash:
    """FakeDB query 与生产 database.py 同步：sentence_v2 集合的 GET 注入 text_hash"""

    def test_fakedb_query_sentence_v2_injects_text_hash(self):
        """FakeDB.query(sentence_v2) 返回的 records 含 text_hash（即使原始无字段）。"""
        from tests.fakes.fake_db import FakeDB
        from services.models_content import compute_text_hash

        db = FakeDB()
        asyncio.run(db.insert(
            collection="sentence_v2",
            data={
                "_id": "s_101",
                "sentence_id": "s_101",
                "text": "Hello World!",
                "translation": "你好世界",
            },
        ))
        result = asyncio.run(db.query(collection="sentence_v2", where={"_id": "s_101"}))
        records = result["records"]
        assert len(records) == 1
        rec = records[0]
        assert rec["text_hash"] == compute_text_hash("Hello World!")

    def test_fakedb_query_sentence_v2_preserves_existing_text_hash(self):
        """FakeDB.query(sentence_v2) 保留已有 text_hash 不覆盖。"""
        from tests.fakes.fake_db import FakeDB

        db = FakeDB()
        asyncio.run(db.insert(
            collection="sentence_v2",
            data={
                "_id": "s_102",
                "sentence_id": "s_102",
                "text": "Hello",
                "text_hash": "preset_abc",
            },
        ))
        result = asyncio.run(db.query(collection="sentence_v2", where={"_id": "s_102"}))
        rec = result["records"][0]
        assert rec["text_hash"] == "preset_abc"

    def test_fakedb_query_other_collections_not_affected(self):
        """非 sentence_v2 集合不受 text_hash 注入影响（如 textbook_v2 走 subject_type 注入）。"""
        from tests.fakes.fake_db import FakeDB

        db = FakeDB()
        asyncio.run(db.insert(
            collection="lesson",
            data={"_id": "ls_1", "lesson_id": "ls_1", "title": "Lesson 1"},
        ))
        result = asyncio.run(db.query(collection="lesson", where={"_id": "ls_1"}))
        rec = result["records"][0]
        assert "text_hash" not in rec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
