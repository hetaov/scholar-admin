"""M3 G0.1 单元测试 — sentence_group 集合模型（services.models.content）

覆盖 data-model-contract §4.14 + service-contract §8.5：

  - VALID_SENTENCE_GROUP_TYPES / VALID_SENTENCE_ROLE_IN_GROUP 枚举
  - build_sentence_group_id：`grp_{textbook_id}_{lesson_id}_{short_hash}` 主键生成
  - build_sentence_group_doc：字段逐字对齐 / type 非法抛 ValueError / 纯函数无副作用
  - normalize_sentence_group_doc：可选字段缺省 None 注入（零迁移 getter）
  - get_sentence_group / get_sentence_groups_by_lesson（经 FakeDB）
  - normalize_sentence_doc M3 4 字段缺省 None 注入（读兼容层）

约定：type 枚举与 role_in_group 枚举相互独立（type = 组类型，role = 句内角色）。
"""
from __future__ import annotations

import pytest

from services.models_content import (
    SENTENCE_GROUP,
    VALID_SENTENCE_GROUP_TYPES,
    VALID_SENTENCE_ROLE_IN_GROUP,
    build_sentence_group_doc,
    build_sentence_group_id,
    get_sentence_group,
    get_sentence_groups_by_lesson,
    normalize_sentence_doc,
    normalize_sentence_group_doc,
)
from tests.fakes.fake_db import FakeDB


# ---------------------------------------------------------------------------
# 枚举（契约 §4.14 type + §4.3 role_in_group）
# ---------------------------------------------------------------------------


class TestEnums:
    def test_group_type_enum(self):
        """type 枚举：dialogue_pair / grammar_family / vocab_family / stand_alone。"""
        assert VALID_SENTENCE_GROUP_TYPES == frozenset({
            "dialogue_pair",
            "grammar_family",
            "vocab_family",
            "stand_alone",
        })

    def test_role_in_group_enum(self):
        """role_in_group 枚举：question / answer_A / answer_B / statement（与 type 相互独立）。"""
        assert VALID_SENTENCE_ROLE_IN_GROUP == frozenset({
            "question",
            "answer_A",
            "answer_B",
            "statement",
        })


# ---------------------------------------------------------------------------
# build_sentence_group_id
# ---------------------------------------------------------------------------


class TestBuildSentenceGroupId:
    def test_format_grp_textbook_lesson_short_hash(self):
        gid = build_sentence_group_id("tb_1", "lesson_3", now=1000)
        prefix = "grp_tb_1_lesson_3_"
        assert gid.startswith(prefix)
        suffix = gid[len(prefix):]
        assert len(suffix) == 8  # sha256[:8]
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_deterministic_for_same_seed(self):
        assert build_sentence_group_id("tb_1", "lesson_3", now=1000) == \
            build_sentence_group_id("tb_1", "lesson_3", now=1000)

    def test_differs_across_lessons(self):
        assert build_sentence_group_id("tb_1", "lesson_3", now=1000) != \
            build_sentence_group_id("tb_1", "lesson_4", now=1000)

    def test_differs_across_time(self):
        assert build_sentence_group_id("tb_1", "lesson_3", now=1000) != \
            build_sentence_group_id("tb_1", "lesson_3", now=1001)


# ---------------------------------------------------------------------------
# build_sentence_group_doc（纯函数）
# ---------------------------------------------------------------------------


class TestBuildSentenceGroupDoc:
    def test_full_doc_fields(self):
        doc = build_sentence_group_doc(
            group_id="grp_1",
            textbook_id="tb_1",
            lesson_id="lesson_3",
            title="Dialogue 1",
            type_="dialogue_pair",
            sentence_ids=["s1", "s2"],
            order_in_lesson=1,
            chapter_id="ch_1",
            difficulty=2,
            focus_grammar="be going to",
            focus_vocab=["travel"],
            build_version="m3",
            now=1000,
        )
        assert doc["_id"] == "grp_1"
        assert doc["group_id"] == "grp_1"
        assert doc["textbook_id"] == "tb_1"
        assert doc["chapter_id"] == "ch_1"
        assert doc["lesson_id"] == "lesson_3"
        assert doc["title"] == "Dialogue 1"
        assert doc["type"] == "dialogue_pair"
        assert doc["sentence_ids"] == ["s1", "s2"]
        assert doc["order_in_lesson"] == 1
        assert doc["difficulty"] == 2
        assert doc["focus_grammar"] == "be going to"
        assert doc["focus_vocab"] == ["travel"]
        assert doc["build_version"] == "m3"
        assert doc["created_at"] == 1000
        assert doc["updated_at"] == 1000

    def test_optional_fields_default(self):
        doc = build_sentence_group_doc(
            group_id="grp_2",
            textbook_id="tb_1",
            lesson_id="lesson_3",
            title="G",
            type_="stand_alone",
            sentence_ids=["s3"],
            order_in_lesson=2,
            now=1000,
        )
        assert doc["chapter_id"] == ""  # 缺省空串
        assert doc["difficulty"] is None
        assert doc["focus_grammar"] is None
        assert doc["focus_vocab"] is None
        assert doc["build_version"] is None

    def test_sentence_ids_copied_not_aliased(self):
        ids = ["s1", "s2"]
        doc = build_sentence_group_doc(
            group_id="grp_3",
            textbook_id="tb_1",
            lesson_id="lesson_3",
            title="G",
            type_="vocab_family",
            sentence_ids=ids,
            order_in_lesson=3,
            now=1000,
        )
        ids.append("s3")  # 外部修改不应影响 doc
        assert doc["sentence_ids"] == ["s1", "s2"]

    @pytest.mark.parametrize("bad_type", ["", "dialogue", "PAIR", None, "unknown"])
    def test_invalid_type_raises_value_error(self, bad_type):
        with pytest.raises(ValueError) as exc:
            build_sentence_group_doc(
                group_id="grp_bad",
                textbook_id="tb_1",
                lesson_id="lesson_3",
                title="G",
                type_=bad_type,
                sentence_ids=["s1"],
                order_in_lesson=1,
                now=1000,
            )
        assert "INVALID_SENTENCE_GROUP_TYPE" in str(exc.value)

    def test_valid_types_all_accepted(self):
        for t in sorted(VALID_SENTENCE_GROUP_TYPES):
            doc = build_sentence_group_doc(
                group_id=f"grp_{t}",
                textbook_id="tb_1",
                lesson_id="lesson_3",
                title="G",
                type_=t,
                sentence_ids=["s1"],
                order_in_lesson=1,
                now=1000,
            )
            assert doc["type"] == t


# ---------------------------------------------------------------------------
# normalize_sentence_group_doc（getter 兼容层）
# ---------------------------------------------------------------------------


class TestNormalizeSentenceGroupDoc:
    def test_injects_none_for_missing_optional_fields(self):
        doc = normalize_sentence_group_doc({
            "_id": "grp_1",
            "group_id": "grp_1",
            "lesson_id": "lesson_3",
            "title": "G",
            "type": "dialogue_pair",
            "sentence_ids": ["s1"],
            "order_in_lesson": 1,
        })
        assert doc["difficulty"] is None
        assert doc["focus_grammar"] is None
        assert doc["focus_vocab"] is None
        assert doc["build_version"] is None

    def test_keeps_explicit_values(self):
        doc = normalize_sentence_group_doc({
            "difficulty": 2,
            "focus_grammar": "be",
            "focus_vocab": ["a"],
            "build_version": "m3",
        })
        assert doc["difficulty"] == 2
        assert doc["focus_grammar"] == "be"
        assert doc["focus_vocab"] == ["a"]
        assert doc["build_version"] == "m3"

    def test_does_not_mutate_input(self):
        original = {"_id": "grp_1", "group_id": "grp_1"}
        normalize_sentence_group_doc(original)
        assert "difficulty" not in original


# ---------------------------------------------------------------------------
# 读侧查询（经 FakeDB）
# ---------------------------------------------------------------------------


class TestGetSentenceGroup:
    def test_missing_returns_none(self):
        db = FakeDB()
        assert asyncio_run(get_sentence_group(db, "grp_missing")) is None

    def test_found_returns_normalized_doc(self):
        db = FakeDB()
        db.add(SENTENCE_GROUP, {
            "_id": "grp_1",
            "group_id": "grp_1",
            "lesson_id": "lesson_3",
            "title": "G",
            "type": "dialogue_pair",
            "sentence_ids": ["s1"],
            "order_in_lesson": 1,
        })
        doc = asyncio_run(get_sentence_group(db, "grp_1"))
        assert doc["group_id"] == "grp_1"
        assert doc["type"] == "dialogue_pair"
        assert doc["difficulty"] is None  # 经 normalize 注入


class TestGetSentenceGroupsByLesson:
    def test_returns_groups_ordered_by_order_in_lesson(self):
        db = FakeDB()
        db.add(SENTENCE_GROUP, {
            "_id": "grp_2", "group_id": "grp_2", "lesson_id": "lesson_3",
            "title": "G2", "type": "stand_alone", "sentence_ids": ["s2"],
            "order_in_lesson": 2,
        })
        db.add(SENTENCE_GROUP, {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "lesson_3",
            "title": "G1", "type": "dialogue_pair", "sentence_ids": ["s1"],
            "order_in_lesson": 1,
        })
        db.add(SENTENCE_GROUP, {
            "_id": "grp_other", "group_id": "grp_other", "lesson_id": "lesson_9",
            "title": "GX", "type": "stand_alone", "sentence_ids": ["s9"],
            "order_in_lesson": 1,
        })
        groups = asyncio_run(get_sentence_groups_by_lesson(db, "lesson_3"))
        assert [g["group_id"] for g in groups] == ["grp_1", "grp_2"]
        assert all(g["difficulty"] is None for g in groups)  # 经 normalize 注入

    def test_empty_lesson_returns_empty_list(self):
        db = FakeDB()
        assert asyncio_run(get_sentence_groups_by_lesson(db, "lesson_none")) == []


# ---------------------------------------------------------------------------
# normalize_sentence_doc — M3 4 字段缺省注入（读兼容层，零迁移）
# ---------------------------------------------------------------------------


class TestNormalizeSentenceDocM3Fields:
    def test_injects_none_for_missing_m3_fields(self):
        doc = normalize_sentence_doc({"sentence_id": "s1", "text": "Hello"})
        assert doc["group_id"] is None
        assert doc["semantic_key"] is None
        assert doc["canonical_sentence_id"] is None
        assert doc["role_in_group"] is None
        # text_hash 照常惰性计算（不回归 E0.1）
        assert doc["text_hash"]

    def test_keeps_explicit_m3_values(self):
        doc = normalize_sentence_doc({
            "sentence_id": "s1",
            "text": "Hello",
            "group_id": "grp_1",
            "semantic_key": "sk_1",
            "canonical_sentence_id": "s0",
            "role_in_group": "question",
        })
        assert doc["group_id"] == "grp_1"
        assert doc["semantic_key"] == "sk_1"
        assert doc["canonical_sentence_id"] == "s0"
        assert doc["role_in_group"] == "question"

    def test_does_not_mutate_input(self):
        original = {"sentence_id": "s1", "text": "Hello"}
        normalize_sentence_doc(original)
        assert "group_id" not in original
        assert "role_in_group" not in original


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
