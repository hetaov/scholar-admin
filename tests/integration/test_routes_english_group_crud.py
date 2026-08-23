"""M3 G1.3 验收集成测试：英语语句分组 CRUD 4 接口 + DM-G7 级联

覆盖契约 api-contract.md §3.11 E-API-8~E-API-11（service-contract §8.5）：
8.  GET    /english/textbook/{textbook_id}/lessons/{lesson_id}/groups  分组列表（order 升序 + 组内详情 + 未分组计数）
9.  POST   /english/textbook/{textbook_id}/lessons/{lesson_id}/groups  新建分组（成员写回 + 审计 create_sentence_group）
10. PUT    /english/group/{group_id}                                   编辑分组（sentence_ids 全量替换 + 审计 edit_sentence_group）
11. DELETE /english/group/{group_id}                                   删除分组（confirm 二次确认 + 成员回退未分组 + 审计 delete_sentence_group）

级联（DM-G7，data-model-contract §4.3.1）：删句 → 从所属组 sentence_ids[] 移除（**不删组**）。

统一断言模式：成功 → 200 + success:true + data:{...}；失败：400/404 对应契约错误码；
写审计 action = 必审 21~23 之一。
"""
from __future__ import annotations

from services.routes_english import router as english_router


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


def _seed_group(fake_db, sentence_ids: list[str] | None = None, **overrides) -> dict:
    """预置 1 条 sentence_group。"""
    doc = {
        "_id": GID,
        "group_id": GID,
        "textbook_id": TB_ID,
        "chapter_id": CH_ID,
        "lesson_id": LS_ID,
        "title": "对话组",
        "type": "dialogue_pair",
        "sentence_ids": list(sentence_ids or []),
        "order_in_lesson": 0,
        "created_at": 1750000000000,
        "updated_at": 1750000000000,
    }
    doc.update(overrides)
    fake_db.add("sentence_group", doc)
    return doc


def _audit_actions(fake_db) -> list[str]:
    return [row["action"] for row in fake_db.all("audit_log")]


# ===========================================================================
# 8. GET /english/textbook/{tid}/lessons/{lid}/groups — 分组列表（E-API-8）
# ===========================================================================


class TestGroupList:
    def test_list_groups_with_members_and_ungrouped(self, make_client, fake_db):
        """返回分组（order 升序）+ 组内句子详情 + 未分组计数 + lesson_title。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1, SID_2])
        _seed_group(
            fake_db,
            sentence_ids=[SID_3],
            group_id="grp_b",
            title="语法组",
            type="grammar_family",
            order_in_lesson=1,
        )
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB_ID}/lessons/{LS_ID}/groups")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["lesson_id"] == LS_ID
        assert data["lesson_title"] == "Let's talk"
        assert data["total"] == 2
        # 种子句未写 group_id → 3 条未分组
        assert data["ungrouped_sentences"] == 3
        # order 升序：grp_...aabbccdd(order=0) 在前，grp_b(order=1) 在后
        assert [g["group_id"] for g in data["groups"]] == [GID, "grp_b"]

        g0 = data["groups"][0]
        assert g0["title"] == "对话组"
        assert g0["type"] == "dialogue_pair"
        assert g0["sentence_count"] == 2
        members = {m["sentence_id"]: m for m in g0["sentences"]}
        assert members[SID_1]["text"] == "Hello!"
        assert members[SID_1]["translation"] == "你好！"
        assert members[SID_1]["is_canonical"] is True

    def test_list_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/tb_xxx/lessons/{LS_ID}/groups")
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]

    def test_list_lesson_not_found_404(self, make_client, fake_db):
        """lesson 不存在 → 404 LESSON_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB_ID}/lessons/ls_999/groups")
        assert resp.status_code == 404, resp.text
        assert "LESSON_NOT_FOUND" in resp.json()["detail"]


# ===========================================================================
# 9. POST /english/textbook/{tid}/lessons/{lid}/groups — 新建分组（E-API-9）
# ===========================================================================


class TestGroupCreate:
    def test_create_group_success_and_audit(self, make_client, fake_db):
        """建组成功：group_id 生成 + 成员写回 role + 审计 create_sentence_group。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/groups",
            json={
                "title": "问候对话",
                "type": "dialogue_pair",
                "sentence_ids": [SID_1, SID_2, SID_3],
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["group_id"].startswith("grp_")
        assert data["title"] == "问候对话"
        assert data["order_in_lesson"] == 0
        assert data["sentence_ids"] == [SID_1, SID_2, SID_3]
        # 落库可查
        rows = fake_db.all("sentence_group")
        assert len(rows) == 1
        g = rows[0]
        assert g["lesson_id"] == LS_ID
        assert g["textbook_id"] == TB_ID
        # 成员写回 + role 推断
        by_id = {r["sentence_id"]: r for r in fake_db.all("sentence_v2")}
        assert by_id[SID_1]["group_id"] == data["group_id"]
        assert by_id[SID_1]["role_in_group"] == "question"
        assert by_id[SID_2]["role_in_group"] == "answer_A"
        assert by_id[SID_3]["role_in_group"] == "statement"
        # 审计：必审 21
        assert "create_sentence_group" in _audit_actions(fake_db)

    def test_create_cross_lesson_sentence_400(self, make_client, fake_db):
        """sentence_ids 含不属于该 lesson 的句 → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/groups",
            json={"title": "跨课组", "type": "stand_alone", "sentence_ids": [SID_1, "s_999"]},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]
        assert fake_db.all("sentence_group") == []

    def test_create_invalid_type_400(self, make_client, fake_db):
        """type 非法 → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/groups",
            json={"title": "非法", "type": "dialogue"},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]

    def test_create_empty_title_400(self, make_client, fake_db):
        """title 空白 → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/groups",
            json={"title": "  ", "type": "stand_alone"},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]
        assert fake_db.all("sentence_group") == []

    def test_create_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/tb_xxx/lessons/{LS_ID}/groups",
            json={"title": "G", "type": "stand_alone"},
        )
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]


# ===========================================================================
# 10. PUT /english/group/{group_id} — 编辑分组（E-API-10）
# ===========================================================================


class TestGroupEdit:
    def test_edit_full_replace_members_and_audit(self, make_client, fake_db):
        """sentence_ids 全量替换：旧成员回退未分组 + 新成员写回 + 审计 edit_sentence_group。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1])
        # 模拟服务层已写回的存量：s_001 已入组
        rows = fake_db.all("sentence_v2")
        fake_db.clear("sentence_v2")
        for r in rows:
            r["group_id"] = GID if r["sentence_id"] == SID_1 else None
            fake_db.add("sentence_v2", r)

        client = make_client(english_router)
        resp = client.put(
            f"/english/group/{GID}",
            json={"title": "新标题", "sentence_ids": [SID_2, SID_3]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["title"] == "新标题"
        assert data["sentence_ids"] == [SID_2, SID_3]

        # 旧成员 s_001 回退未分组，新成员写回 + role 重推
        by_id = {r["sentence_id"]: r for r in fake_db.all("sentence_v2")}
        assert by_id[SID_1]["group_id"] is None
        assert by_id[SID_2]["group_id"] == GID
        assert by_id[SID_2]["role_in_group"] == "question"
        assert by_id[SID_3]["group_id"] == GID
        assert by_id[SID_3]["role_in_group"] == "answer_A"
        # 审计：必审 22
        assert "edit_sentence_group" in _audit_actions(fake_db)

    def test_edit_group_not_found_404(self, make_client, fake_db):
        """group 不存在 → 404 GROUP_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.put("/english/group/grp_ghost", json={"title": "G"})
        assert resp.status_code == 404, resp.text
        assert "GROUP_NOT_FOUND" in resp.json()["detail"]

    def test_edit_cross_lesson_member_400(self, make_client, fake_db):
        """新成员跨 lesson → 400 SENTENCE_PAYLOAD_INVALID，组不变。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1])
        client = make_client(english_router)
        resp = client.put(
            f"/english/group/{GID}",
            json={"sentence_ids": [SID_1, "s_999"]},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]
        rows = fake_db.all("sentence_group")
        assert rows[0]["sentence_ids"] == [SID_1]
        assert "edit_sentence_group" not in _audit_actions(fake_db)


# ===========================================================================
# 11. DELETE /english/group/{group_id} — 删除分组（E-API-11）
# ===========================================================================


class TestGroupDelete:
    def test_delete_group_releases_members_and_audit(self, make_client, fake_db):
        """confirm 匹配：组删除 + 成员回退未分组（不物理删句）+ 审计 delete_sentence_group。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1, SID_2])
        rows = fake_db.all("sentence_v2")
        fake_db.clear("sentence_v2")
        for r in rows:
            r["group_id"] = GID if r["sentence_id"] in (SID_1, SID_2) else None
            fake_db.add("sentence_v2", r)

        client = make_client(english_router)
        resp = client.request(
            "DELETE",
            f"/english/group/{GID}",
            json={"confirm_sentence_count": 2},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["group_id"] == GID
        assert data["released_sentences"] == 2

        # 组已删、句仍在、group_id 回退
        assert fake_db.all("sentence_group") == []
        by_id = {r["sentence_id"]: r for r in fake_db.all("sentence_v2")}
        assert len(by_id) == 3
        assert by_id[SID_1]["group_id"] is None
        assert by_id[SID_2]["group_id"] is None
        # 审计：必审 23
        assert "delete_sentence_group" in _audit_actions(fake_db)

    def test_delete_confirm_mismatch_400(self, make_client, fake_db):
        """confirm_sentence_count 不匹配 → 400 CONFIRM_TEXT_MISMATCH，不删组。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1])
        client = make_client(english_router)
        resp = client.request(
            "DELETE",
            f"/english/group/{GID}",
            json={"confirm_sentence_count": 99},
        )
        assert resp.status_code == 400, resp.text
        assert "CONFIRM_TEXT_MISMATCH" in resp.json()["detail"]
        assert len(fake_db.all("sentence_group")) == 1
        assert "delete_sentence_group" not in _audit_actions(fake_db)

    def test_delete_group_not_found_404(self, make_client, fake_db):
        """group 不存在 → 404 GROUP_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.request("DELETE", "/english/group/grp_ghost", json={"confirm_sentence_count": 0})
        assert resp.status_code == 404, resp.text
        assert "GROUP_NOT_FOUND" in resp.json()["detail"]


# ===========================================================================
# DM-G7 级联：删句 → 从所属组 sentence_ids[] 移除（不删组）
# ===========================================================================


class TestSentenceDeleteCascadesGroupRef:
    def test_delete_sentence_removes_from_group_members(self, make_client, fake_db):
        """删除 s_001 后：组仍在、sentence_ids 不再含 s_001、deleted 计数 = 1。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_group(fake_db, sentence_ids=[SID_1, SID_2])
        rows = fake_db.all("sentence_v2")
        fake_db.clear("sentence_v2")
        for r in rows:
            r["group_id"] = GID if r["sentence_id"] in (SID_1, SID_2) else None
            fake_db.add("sentence_v2", r)

        client = make_client(english_router)
        resp = client.request(
            "DELETE",
            f"/english/sentence/{SID_1}",
            json={"confirm_text": "Hello!"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["deleted"]["sentence_group_refs_removed"] == 1

        # 组不删，仅移除引用；s_002 保留
        groups = fake_db.all("sentence_group")
        assert len(groups) == 1
        assert groups[0]["group_id"] == GID
        assert groups[0]["sentence_ids"] == [SID_2]
