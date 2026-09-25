"""单元测试:课文断点持久化模型 — services.models_lesson_resume（详情页重构 v1 · T1）

覆盖（契约 §7 T1 验收）:
- build_resume_point_doc 构造正确（_id 拼接 / updated_at / None 字段处理）
- upsert 幂等（同 openid+lesson_id 多次 upsert 只产生一条记录，最新值生效）
- get 空返回（无记录返回 None）
- clean（删除后 get 返回 None；返回 True；不存在时返回 False，不抛错）
- 非法 tab 抛 ValueError
- 非法 microflow_node 抛 ValueError
"""

from __future__ import annotations

import pytest

from services.models_lesson_resume import (
    LESSON_RESUME_POINT,
    RESUME_TAB_CONTENT,
    RESUME_TAB_CONVERSATION,
    RESUME_TAB_MASTERY,
    RESUME_TAB_TRAINING,
    VALID_MICROFLOW_NODES,
    VALID_RESUME_TABS,
    build_resume_point_doc,
    clean_resume_point,
    get_resume_point,
    lesson_resume_point_id,
    upsert_resume_point,
)
from tests.fakes.fake_db import FakeDB


class TestConstants:
    def test_collection_name(self):
        assert LESSON_RESUME_POINT == "lesson_resume_point"

    def test_tab_enums(self):
        assert RESUME_TAB_CONTENT == "content"
        assert RESUME_TAB_MASTERY == "mastery"
        assert RESUME_TAB_TRAINING == "training"
        assert RESUME_TAB_CONVERSATION == "conversation"
        assert VALID_RESUME_TABS == {
            "content",
            "mastery",
            "training",
            "conversation",
        }

    def test_microflow_node_enums(self):
        assert VALID_MICROFLOW_NODES == {
            "listen",
            "shadowing",
            "recall",
            "compare",
            "ai_followup",
        }


class TestResumePointId:
    def test_composite_key(self):
        assert lesson_resume_point_id("o1", "l1") == "o1_l1"

    def test_distinguishes_order(self):
        assert lesson_resume_point_id("o1", "l2") != lesson_resume_point_id("o2", "l1")


class TestBuildDoc:
    def test_full_doc(self):
        doc = build_resume_point_doc(
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_TRAINING,
            sentence_id="s_99",
            microflow_node="shadowing",
            now=2000,
        )
        assert doc["_id"] == "o1_l1"
        assert doc["openid"] == "o1"
        assert doc["lesson_id"] == "l1"
        assert doc["tab"] == "training"
        assert doc["sentence_id"] == "s_99"
        assert doc["microflow_node"] == "shadowing"
        assert doc["updated_at"] == 2000
        assert doc["created_at"] == 2000

    def test_none_fields_default_to_none(self):
        """掌握 Tab 无句、未进微流程无节点：sentence_id / microflow_node 缺省 None。"""
        doc = build_resume_point_doc(
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_MASTERY,
            now=2000,
        )
        assert doc["sentence_id"] is None
        assert doc["microflow_node"] is None
        assert doc["tab"] == "mastery"

    def test_now_defaults_to_current_time(self):
        """now 缺省时用当前时间（秒级时间戳，非 None）。"""
        doc = build_resume_point_doc(
            openid="o1", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        assert isinstance(doc["updated_at"], int)
        assert doc["updated_at"] > 0
        assert doc["created_at"] == doc["updated_at"]


class TestUpsertIdempotent:
    @pytest.mark.asyncio
    async def test_first_insert(self, fake_db):
        doc = await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONTENT,
            sentence_id="s_1",
        )
        assert doc["_id"] == "o1_l1"
        assert doc["tab"] == "content"
        assert doc["sentence_id"] == "s_1"
        assert isinstance(doc["updated_at"], int)
        assert len(fake_db.all(LESSON_RESUME_POINT)) == 1

    @pytest.mark.asyncio
    async def test_repeated_upsert_is_idempotent(self, fake_db):
        """同 openid+lesson_id 多次 upsert 只产生一条记录，最新值生效。"""
        await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONTENT,
            sentence_id="s_1",
            microflow_node="listen",
        )
        await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_TRAINING,
            sentence_id="s_2",
            microflow_node="shadowing",
        )
        records = fake_db.all(LESSON_RESUME_POINT)
        assert len(records) == 1
        assert records[0]["tab"] == "training"
        assert records[0]["sentence_id"] == "s_2"
        assert records[0]["microflow_node"] == "shadowing"

    @pytest.mark.asyncio
    async def test_update_overwrites_none_fields(self, fake_db):
        """切到掌握 Tab（无 sentence）应覆盖原 sentence_id 为 None（断点只记最后位置）。"""
        await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONTENT,
            sentence_id="s_1",
            microflow_node="listen",
        )
        await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_MASTERY,
        )
        records = fake_db.all(LESSON_RESUME_POINT)
        assert len(records) == 1
        assert records[0]["tab"] == "mastery"
        assert records[0]["sentence_id"] is None
        assert records[0]["microflow_node"] is None

    @pytest.mark.asyncio
    async def test_isolated_by_user_lesson(self, fake_db):
        """不同用户/课程各自独立成记录。"""
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        await upsert_resume_point(
            fake_db, openid="o2", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l2", tab=RESUME_TAB_CONTENT
        )
        assert len(fake_db.all(LESSON_RESUME_POINT)) == 3

    @pytest.mark.asyncio
    async def test_returns_latest_doc(self, fake_db):
        """upsert 返回值是最新文档（updated_at 刷新；created_at 不变）。"""
        import services.models.lesson_resume as _mod

        first = await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONTENT,
        )
        first_created = first["created_at"]
        # 第二次 upsert：替换模块级 time 引用为稍后时间戳，
        # 验证 update 路径刷新 updated_at 且不破坏 created_at
        orig_time = _mod.time

        class _FakeTime:
            @staticmethod
            def time():
                return first_created + 1000

        _mod.time = _FakeTime
        try:
            second = await upsert_resume_point(
                fake_db,
                openid="o1",
                lesson_id="l1",
                tab=RESUME_TAB_TRAINING,
            )
        finally:
            _mod.time = orig_time
        assert second["tab"] == "training"
        assert second["updated_at"] == first_created + 1000
        assert second["created_at"] == first_created


class TestGetResumePoint:
    @pytest.mark.asyncio
    async def test_get_empty_returns_none(self, fake_db):
        """无记录返回 None。"""
        assert await get_resume_point(fake_db, openid="o1", lesson_id="l1") is None

    @pytest.mark.asyncio
    async def test_get_returns_record(self, fake_db):
        await upsert_resume_point(
            fake_db,
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONVERSATION,
            sentence_id="s_5",
            microflow_node="ai_followup",
        )
        doc = await get_resume_point(fake_db, openid="o1", lesson_id="l1")
        assert doc is not None
        assert doc["_id"] == "o1_l1"
        assert doc["tab"] == "conversation"
        assert doc["sentence_id"] == "s_5"
        assert doc["microflow_node"] == "ai_followup"

    @pytest.mark.asyncio
    async def test_get_isolated_by_user(self, fake_db):
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        # 另一用户同课应查不到
        assert await get_resume_point(fake_db, openid="o2", lesson_id="l1") is None


class TestCleanResumePoint:
    @pytest.mark.asyncio
    async def test_clean_existing_returns_true(self, fake_db):
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        ok = await clean_resume_point(fake_db, openid="o1", lesson_id="l1")
        assert ok is True
        # 删除后 get 返回 None
        assert await get_resume_point(fake_db, openid="o1", lesson_id="l1") is None
        assert fake_db.all(LESSON_RESUME_POINT) == []

    @pytest.mark.asyncio
    async def test_clean_nonexistent_returns_false(self, fake_db):
        """不存在时返回 False，不抛错。"""
        ok = await clean_resume_point(fake_db, openid="o1", lesson_id="l1")
        assert ok is False

    @pytest.mark.asyncio
    async def test_clean_only_targets_specified_record(self, fake_db):
        """只删指定 openid+lesson_id，不影响其他断点。"""
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l1", tab=RESUME_TAB_CONTENT
        )
        await upsert_resume_point(
            fake_db, openid="o1", lesson_id="l2", tab=RESUME_TAB_CONTENT
        )
        ok = await clean_resume_point(fake_db, openid="o1", lesson_id="l1")
        assert ok is True
        # l2 仍在
        assert await get_resume_point(fake_db, openid="o1", lesson_id="l2") is not None
        assert len(fake_db.all(LESSON_RESUME_POINT)) == 1


class TestValidation:
    def test_build_invalid_tab_raises(self):
        with pytest.raises(ValueError):
            build_resume_point_doc(
                openid="o1", lesson_id="l1", tab="unknown_tab"
            )

    def test_build_empty_tab_raises(self):
        with pytest.raises(ValueError):
            build_resume_point_doc(openid="o1", lesson_id="l1", tab="")

    def test_build_invalid_microflow_node_raises(self):
        with pytest.raises(ValueError):
            build_resume_point_doc(
                openid="o1",
                lesson_id="l1",
                tab=RESUME_TAB_CONTENT,
                microflow_node="sing_along",
            )

    def test_build_none_microflow_node_ok(self):
        """microflow_node=None 合法（未进微流程）。"""
        doc = build_resume_point_doc(
            openid="o1",
            lesson_id="l1",
            tab=RESUME_TAB_CONTENT,
            microflow_node=None,
        )
        assert doc["microflow_node"] is None

    def test_build_all_valid_microflow_nodes(self):
        """每个合法 microflow_node 都能构建。"""
        for node in VALID_MICROFLOW_NODES:
            doc = build_resume_point_doc(
                openid="o1",
                lesson_id="l1",
                tab=RESUME_TAB_TRAINING,
                microflow_node=node,
            )
            assert doc["microflow_node"] == node

    def test_build_all_valid_tabs(self):
        """每个合法 tab 都能构建。"""
        for tab in VALID_RESUME_TABS:
            doc = build_resume_point_doc(
                openid="o1", lesson_id="l1", tab=tab
            )
            assert doc["tab"] == tab

    @pytest.mark.asyncio
    async def test_upsert_invalid_tab_raises(self, fake_db):
        with pytest.raises(ValueError):
            await upsert_resume_point(
                fake_db, openid="o1", lesson_id="l1", tab="nope"
            )

    @pytest.mark.asyncio
    async def test_upsert_invalid_microflow_node_raises(self, fake_db):
        with pytest.raises(ValueError):
            await upsert_resume_point(
                fake_db,
                openid="o1",
                lesson_id="l1",
                tab=RESUME_TAB_CONTENT,
                microflow_node="read_aloud",
            )

    @pytest.mark.asyncio
    async def test_upsert_invalid_tab_does_not_write(self, fake_db):
        """非法 tab 抛错时不应落库。"""
        with pytest.raises(ValueError):
            await upsert_resume_point(
                fake_db, openid="o1", lesson_id="l1", tab="bad"
            )
        assert fake_db.all(LESSON_RESUME_POINT) == []
