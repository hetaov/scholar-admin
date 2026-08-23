"""M3 G1.3 单元测试 — 语句分组服务层（services.english.sentence_group）

覆盖 service-contract §8.5 + api-contract §3.11 E-API-8~E-API-11 的 4 个服务函数：
1. listSentenceGroups     分组列表（order 升序 + 组内句子详情 + 未分组计数 + 分页）
2. createSentenceGroup    新建分组（title/type 校验、跨 lesson 校验、order 自增、成员写回 + 审计）
3. editSentenceGroup      编辑分组（全可选、sentence_ids 全量替换、成员增删 + 审计）
4. deleteSentenceGroup    删除分组（confirm 二次确认、成员回退未分组 + 审计）

统一断言模式：业务异常（TextbookNotFoundError / LessonNotFoundError / GroupNotFoundError /
SentencePayloadError / ConfirmTextMismatchError）按契约 §8.2 抛错；审计动作 = 必审 21~23。
"""
from __future__ import annotations

import asyncio

import pytest

from services.audit import (
    AUDIT_ACTION_CREATE_SENTENCE_GROUP,
    AUDIT_ACTION_DELETE_SENTENCE_GROUP,
    AUDIT_ACTION_EDIT_SENTENCE_GROUP,
)
from services.english import (
    ConfirmTextMismatchError,
    GroupNotFoundError,
    LessonNotFoundError,
    SentencePayloadError,
    TextbookNotFoundError,
)
from services.english.sentence_group import (
    createSentenceGroup,
    deleteSentenceGroup,
    editSentenceGroup,
    listSentenceGroups,
)
from tests.fakes.fake_db import FakeDB


# ===========================================================================
# Test 辅助函数
# ===========================================================================

TB_ID = "tb_eng_pep4"
LS_ID = "ls_001"
CH_ID = "ch_001"
SID_1 = "s_001"
SID_2 = "s_002"
SID_3 = "s_003"
GID = "grp_tb_eng_pep4_ls_001_aabbccdd"


def _run(coro):
    """同步包装：服务函数均为 async，单测用 asyncio.run 直跑（与 test_group_view 同款）。"""
    return asyncio.run(coro)


def _seed_textbook(fake_db, **overrides) -> dict:
    """预置 1 条英语教材（textbook_v2，含 chapters → lessons 结构）。"""
    doc = {
        "textbook_id": TB_ID,
        "title": "PEP 四年级上册",
        "grade": "四年级",
        "level": None,
        "subject_type": "english",
        "chapters": [
            {
                "chapter_id": CH_ID,
                "title": "Unit 1 Hello",
                "lessons": [{"lesson_id": LS_ID, "title": "Let's talk"}],
            }
        ],
        "created_at": 1750000000000,
        "updated_at": 1750000000000,
    }
    doc.update(overrides)
    fake_db.add("textbook_v2", doc)
    return doc


def _seed_sentences(fake_db):
    """3 条独立语句（s_001 / s_002 / s_003），全部属于 ls_001。"""
    for i, (sid, text, translation) in enumerate(
        [
            (SID_1, "Hello!", "你好！"),
            (SID_2, "Good morning!", "早上好！"),
            (SID_3, "How are you?", "你好吗？"),
        ]
    ):
        fake_db.add(
            "sentence_v2",
            {
                "sentence_id": sid,
                "text": text,
                "translation": translation,
                "audio_url": "",
                "knowledge_point_ids": [],
                "textbook_id": TB_ID,
                "lesson_id": LS_ID,
                "chapter_id": CH_ID,
                "created_at": 1750000000000 + i,
                "updated_at": 1750000000000 + i,
            },
        )


def _seed_group(
    fake_db,
    *,
    group_id: str = GID,
    title: str = "对话组",
    type_: str = "dialogue_pair",
    sentence_ids: list[str] | None = None,
    order_in_lesson: int = 0,
    textbook_id: str = TB_ID,
    lesson_id: str = LS_ID,
) -> dict:
    """预置 1 条 sentence_group 文档。"""
    doc = {
        "_id": group_id,
        "group_id": group_id,
        "textbook_id": textbook_id,
        "chapter_id": CH_ID,
        "lesson_id": lesson_id,
        "title": title,
        "type": type_,
        "sentence_ids": list(sentence_ids or []),
        "order_in_lesson": order_in_lesson,
        "created_at": 1750000000000,
        "updated_at": 1750000000000,
    }
    fake_db.add("sentence_group", doc)
    return doc


def _set_sentence_groups(fake_db, mapping: dict[str, str | None]) -> None:
    """批量写 sentence_v2.group_id（经全量重写，模拟服务层写回后的存量状态）。"""
    rows = fake_db.all("sentence_v2")
    fake_db.clear("sentence_v2")
    for r in rows:
        r["group_id"] = mapping.get(r["sentence_id"])
        fake_db.add("sentence_v2", r)


def _audit_actions(fake_db) -> list[str]:
    return [row["action"] for row in fake_db.all("audit_log")]


def _sentence_by_id(fake_db) -> dict:
    return {r["sentence_id"]: r for r in fake_db.all("sentence_v2")}


# ===========================================================================
# 1. listSentenceGroups — 分组列表（E-API-8）
# ===========================================================================


class TestListSentenceGroups:
    def test_list_groups_ordered_with_members_and_ungrouped(self):
        """按 order_in_lesson 升序 + 组内句子详情 + 未分组句子计数。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, group_id=GID, sentence_ids=[SID_1, SID_2], order_in_lesson=1)
        _seed_group(
            db,
            group_id="grp_x_2",
            title="语法组",
            type_="grammar_family",
            sentence_ids=[SID_3],
            order_in_lesson=0,
        )
        data = _run(listSentenceGroups(db, textbook_id=TB_ID, lesson_id=LS_ID))

        assert data["lesson_id"] == LS_ID
        assert data["lesson_title"] == "Let's talk"
        assert data["total"] == 2
        # 种子句未写 group_id → 3 条都算未分组
        assert data["ungrouped_sentences"] == 3
        # order 升序：grp_x_2(order=0) 在前，GID(order=1) 在后
        assert [g["group_id"] for g in data["groups"]] == ["grp_x_2", GID]

        g2 = data["groups"][1]
        assert g2["title"] == "对话组"
        assert g2["type"] == "dialogue_pair"
        assert g2["sentence_ids"] == [SID_1, SID_2]
        assert g2["sentence_count"] == 2
        members = {m["sentence_id"]: m for m in g2["sentences"]}
        assert members[SID_1]["text"] == "Hello!"
        assert members[SID_1]["translation"] == "你好！"
        assert members[SID_1]["is_canonical"] is True
        assert members[SID_2]["text"] == "Good morning!"

    def test_list_ungrouped_counts_excludes_grouped_sentences(self):
        """已入组句子（sentence_v2.group_id 已写回）不再计入 ungrouped_sentences。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1], order_in_lesson=0)
        _set_sentence_groups(db, {SID_1: GID, SID_2: None, SID_3: None})

        data = _run(listSentenceGroups(db, textbook_id=TB_ID, lesson_id=LS_ID))
        assert data["ungrouped_sentences"] == 2
        assert data["groups"][0]["sentence_count"] == 1

    def test_list_pagination(self):
        """内存分页：page/page_size 生效，total 不变。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        for i in range(3):
            _seed_group(db, group_id=f"grp_pg_{i}", order_in_lesson=i)
        data = _run(
            listSentenceGroups(db, textbook_id=TB_ID, lesson_id=LS_ID, page=1, page_size=2)
        )
        assert data["total"] == 3
        assert len(data["groups"]) == 2
        data2 = _run(
            listSentenceGroups(db, textbook_id=TB_ID, lesson_id=LS_ID, page=2, page_size=2)
        )
        assert len(data2["groups"]) == 1

    def test_list_textbook_not_found(self):
        """教材不存在 → TextbookNotFoundError（404 TEXTBOOK_NOT_FOUND）。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(TextbookNotFoundError):
            _run(listSentenceGroups(db, textbook_id="tb_xxx", lesson_id=LS_ID))

    def test_list_lesson_not_found(self):
        """lesson 不存在 → LessonNotFoundError（404 LESSON_NOT_FOUND）。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(LessonNotFoundError):
            _run(listSentenceGroups(db, textbook_id=TB_ID, lesson_id="ls_999"))


# ===========================================================================
# 2. createSentenceGroup — 新建分组（E-API-9）
# ===========================================================================


class TestCreateSentenceGroup:
    def test_create_with_members_writes_group_id_and_roles(self):
        """建组成功：group_id 生成 + title strip + 成员 group_id/role 写回 + 审计。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        data = _run(
            createSentenceGroup(
                db,
                textbook_id=TB_ID,
                lesson_id=LS_ID,
                title="  对话组  ",
                type="dialogue_pair",
                sentence_ids=[SID_1, SID_2, SID_3],
                editor_id="admin_1",
            )
        )
        assert data["group_id"].startswith("grp_")
        assert data["title"] == "对话组"  # strip
        assert data["type"] == "dialogue_pair"
        assert data["order_in_lesson"] == 0  # 无既有组 → 0
        assert data["sentence_ids"] == [SID_1, SID_2, SID_3]

        rows = db.all("sentence_group")
        assert len(rows) == 1
        g = rows[0]
        assert g["group_id"] == data["group_id"]
        assert g["lesson_id"] == LS_ID
        assert g["textbook_id"] == TB_ID

        # 成员写回 + dialogue_pair 角色推断：question / answer_A / statement
        s = _sentence_by_id(db)
        assert s[SID_1]["group_id"] == data["group_id"]
        assert s[SID_1]["role_in_group"] == "question"
        assert s[SID_2]["group_id"] == data["group_id"]
        assert s[SID_2]["role_in_group"] == "answer_A"
        assert s[SID_3]["group_id"] == data["group_id"]
        assert s[SID_3]["role_in_group"] == "statement"

        # 审计：必审 21
        assert AUDIT_ACTION_CREATE_SENTENCE_GROUP in _audit_actions(db)

    def test_create_empty_group_without_sentences(self):
        """sentence_ids 缺省 → 空组（可先建组后填句）。"""
        db = FakeDB()
        _seed_textbook(db)
        data = _run(
            createSentenceGroup(
                db,
                textbook_id=TB_ID,
                lesson_id=LS_ID,
                title="待填空组",
                type="stand_alone",
                editor_id="admin_1",
            )
        )
        assert data["sentence_ids"] == []
        assert len(db.all("sentence_group")) == 1
        assert AUDIT_ACTION_CREATE_SENTENCE_GROUP in _audit_actions(db)

    def test_create_order_in_lesson_auto_increment(self):
        """order_in_lesson 缺省 = 当前最大 + 1。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_group(db, group_id="grp_a", order_in_lesson=2)
        _seed_group(db, group_id="grp_b", order_in_lesson=5)
        data = _run(
            createSentenceGroup(
                db,
                textbook_id=TB_ID,
                lesson_id=LS_ID,
                title="新组",
                type="vocab_family",
            )
        )
        assert data["order_in_lesson"] == 6

    def test_create_empty_title_raises(self):
        """title 空白 → SentencePayloadError（400）。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(SentencePayloadError) as exc:
            _run(
                createSentenceGroup(
                    db,
                    textbook_id=TB_ID,
                    lesson_id=LS_ID,
                    title="   ",
                    type="stand_alone",
                )
            )
        assert "title" in str(exc.value)
        assert db.all("sentence_group") == []

    def test_create_invalid_type_raises(self):
        """type 非法 → SentencePayloadError（400），不落库。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(SentencePayloadError) as exc:
            _run(
                createSentenceGroup(
                    db,
                    textbook_id=TB_ID,
                    lesson_id=LS_ID,
                    title="非法",
                    type="dialogue",
                )
            )
        assert "type" in str(exc.value)
        assert db.all("sentence_group") == []

    def test_create_cross_lesson_sentence_raises(self):
        """sentence_ids 含不属于该 lesson 的句 → SentencePayloadError（400）。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        # 另一 lesson 的句子（s_009 不存在于 ls_001）
        db.add(
            "sentence_v2",
            {
                "sentence_id": "s_009",
                "text": "Other",
                "translation": "别课",
                "textbook_id": TB_ID,
                "lesson_id": "ls_999",
                "chapter_id": CH_ID,
            },
        )
        with pytest.raises(SentencePayloadError) as exc:
            _run(
                createSentenceGroup(
                    db,
                    textbook_id=TB_ID,
                    lesson_id=LS_ID,
                    title="跨课组",
                    type="stand_alone",
                    sentence_ids=[SID_1, "s_009"],
                )
            )
        assert "不属于该 lesson" in str(exc.value)
        assert db.all("sentence_group") == []

    def test_create_textbook_not_found(self):
        """教材不存在 → TextbookNotFoundError。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(TextbookNotFoundError):
            _run(
                createSentenceGroup(
                    db,
                    textbook_id="tb_xxx",
                    lesson_id=LS_ID,
                    title="G",
                    type="stand_alone",
                )
            )

    def test_create_lesson_not_found(self):
        """lesson 不存在 → LessonNotFoundError。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(LessonNotFoundError):
            _run(
                createSentenceGroup(
                    db,
                    textbook_id=TB_ID,
                    lesson_id="ls_999",
                    title="G",
                    type="stand_alone",
                )
            )


# ===========================================================================
# 3. editSentenceGroup — 编辑分组（E-API-10）
# ===========================================================================


class TestEditSentenceGroup:
    def test_edit_title_and_type_only(self):
        """只改 title/type：组字段更新 + 审计 edit_sentence_group。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1], order_in_lesson=0)
        data = _run(
            editSentenceGroup(
                db,
                group_id=GID,
                title="新标题",
                type="grammar_family",
                editor_id="admin_2",
            )
        )
        assert data["title"] == "新标题"
        assert data["type"] == "grammar_family"
        assert data["sentence_ids"] == [SID_1]

        rows = db.all("sentence_group")
        g = rows[0]
        assert g["title"] == "新标题"
        assert g["type"] == "grammar_family"
        assert AUDIT_ACTION_EDIT_SENTENCE_GROUP in _audit_actions(db)

    def test_edit_full_replace_members(self):
        """sentence_ids 全量替换：旧成员回退未分组 → 新成员写回 + 重推角色。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1], order_in_lesson=0)
        _set_sentence_groups(db, {SID_1: GID, SID_2: None, SID_3: None})

        data = _run(
            editSentenceGroup(
                db,
                group_id=GID,
                sentence_ids=[SID_2, SID_3],
                editor_id="admin_2",
            )
        )
        assert data["sentence_ids"] == [SID_2, SID_3]

        # 旧成员 s_001 回退
        s = _sentence_by_id(db)
        assert s[SID_1]["group_id"] is None
        assert s[SID_1]["role_in_group"] is None
        # 新成员写回 + dialogue_pair 角色重推
        assert s[SID_2]["group_id"] == GID
        assert s[SID_2]["role_in_group"] == "question"
        assert s[SID_3]["group_id"] == GID
        assert s[SID_3]["role_in_group"] == "answer_A"
        assert AUDIT_ACTION_EDIT_SENTENCE_GROUP in _audit_actions(db)

    def test_edit_group_not_found(self):
        """group 不存在 → GroupNotFoundError（404 GROUP_NOT_FOUND）。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(GroupNotFoundError):
            _run(editSentenceGroup(db, group_id="grp_ghost", title="G"))

    def test_edit_empty_title_raises(self):
        """title 置空 → SentencePayloadError（400）。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_group(db)
        with pytest.raises(SentencePayloadError):
            _run(editSentenceGroup(db, group_id=GID, title="   "))

    def test_edit_invalid_type_raises(self):
        """type 置非法 → SentencePayloadError（400）。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_group(db)
        with pytest.raises(SentencePayloadError):
            _run(editSentenceGroup(db, group_id=GID, type="dialogue"))

    def test_edit_cross_lesson_member_raises_atomic(self):
        """新成员跨 lesson → SentencePayloadError（400），旧成员不被清。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1], order_in_lesson=0)
        _set_sentence_groups(db, {SID_1: GID, SID_2: None, SID_3: None})

        with pytest.raises(SentencePayloadError):
            _run(
                editSentenceGroup(
                    db,
                    group_id=GID,
                    sentence_ids=[SID_1, "s_999"],
                )
            )
        # 组不变、旧成员未清
        rows = db.all("sentence_group")
        assert rows[0]["sentence_ids"] == [SID_1]
        assert _sentence_by_id(db)[SID_1]["group_id"] == GID
        # 未写审计
        assert _audit_actions(db) == []


# ===========================================================================
# 4. deleteSentenceGroup — 删除分组（E-API-11）
# ===========================================================================


class TestDeleteSentenceGroup:
    def test_delete_releases_members_and_audit(self):
        """confirm 匹配：组删除 + 成员 group_id 置 null（不物理删句）+ 审计。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1, SID_2], order_in_lesson=0)
        _set_sentence_groups(db, {SID_1: GID, SID_2: GID, SID_3: None})

        data = _run(
            deleteSentenceGroup(
                db,
                group_id=GID,
                confirm_sentence_count=2,
                editor_id="admin_3",
            )
        )
        assert data["group_id"] == GID
        assert data["released_sentences"] == 2

        # 组已删、句仍在、group_id 回退未分组
        assert db.all("sentence_group") == []
        s = _sentence_by_id(db)
        assert len(s) == 3
        assert s[SID_1]["group_id"] is None
        assert s[SID_2]["group_id"] is None
        assert AUDIT_ACTION_DELETE_SENTENCE_GROUP in _audit_actions(db)

    def test_delete_confirm_mismatch_raises(self):
        """confirm_sentence_count 不匹配 → ConfirmTextMismatchError（400），不删组。"""
        db = FakeDB()
        _seed_textbook(db)
        _seed_sentences(db)
        _seed_group(db, sentence_ids=[SID_1], order_in_lesson=0)
        with pytest.raises(ConfirmTextMismatchError) as exc:
            _run(deleteSentenceGroup(db, group_id=GID, confirm_sentence_count=99))
        assert "不匹配" in str(exc.value)
        assert len(db.all("sentence_group")) == 1
        assert _audit_actions(db) == []

    def test_delete_group_not_found(self):
        """group 不存在 → GroupNotFoundError（404 GROUP_NOT_FOUND）。"""
        db = FakeDB()
        _seed_textbook(db)
        with pytest.raises(GroupNotFoundError):
            _run(deleteSentenceGroup(db, group_id="grp_ghost", confirm_sentence_count=0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
