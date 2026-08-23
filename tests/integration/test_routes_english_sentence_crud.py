"""E1.1 验收集成测试：英语语句 CRUD 4 接口

覆盖 SOP §5 E1.1 规格来源（契约 api-contract.md §3.11 E-API-4~E-API-7）：
1. GET    /english/textbook/{textbook_id}/lessons/{lesson_id}/sentences  语句列表（含重复标记 + 关联数据计数）
2. POST   /english/textbook/{textbook_id}/lessons/{lesson_id}/sentences  新增语句（批量，hash 去重）
3. PUT    /english/sentence/{sentence_id}                                编辑语句（字段白名单 + text_hash 重算）
4. DELETE /english/sentence/{sentence_id}                                删除语句 + 级联清理 6 表 + 二次确认

统一断言模式：成功 → 200 + success:true + data:{...}；失败：400/404 对应契约错误码；
写审计 action = E0.2 新增 3 类之一（create_english_sentences / edit_english_sentence / delete_english_sentence）。
"""
from __future__ import annotations

import pytest

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
    """3 条语句：s_001 / s_002 同 hash（重复对），s_003 独立。"""
    fake_db.add(
        "sentence_v2",
        {
            "sentence_id": SID_1,
            "text": "Hello!",
            "translation": "你好！",
            "audio_url": "https://a/hello.mp3",
            "knowledge_point_ids": ["kp_001"],
            "textbook_id": TB_ID,
            "lesson_id": LS_ID,
            "chapter_id": CH_ID,
            "created_at": 1750000000000,
            "updated_at": 1750000000000,
        },
    )
    # 同 hash：标点差异归一后相同（"hello。"）
    fake_db.add(
        "sentence_v2",
        {
            "sentence_id": SID_2,
            "text": "hello。",
            "translation": "你好！",
            "audio_url": "",
            "knowledge_point_ids": [],
            "textbook_id": TB_ID,
            "lesson_id": LS_ID,
            "chapter_id": CH_ID,
            "created_at": 1750000000001,
            "updated_at": 1750000000001,
        },
    )
    fake_db.add(
        "sentence_v2",
        {
            "sentence_id": SID_3,
            "text": "Good morning!",
            "translation": "早上好！",
            "audio_url": "",
            "knowledge_point_ids": [],
            "textbook_id": TB_ID,
            "lesson_id": LS_ID,
            "chapter_id": CH_ID,
            "created_at": 1750000000002,
            "updated_at": 1750000000002,
        },
    )


def _seed_related_data(fake_db):
    """s_001 关联 5 表数据 + 1 条 conversation_turn 引用 s_001。"""
    fake_db.add("study_attempt", {"sentence_id": SID_1, "scholar_id": "sch_1"})
    fake_db.add("study_attempt", {"sentence_id": SID_1, "scholar_id": "sch_2"})
    fake_db.add("skill_state", {"sentence_id": SID_1, "skill_code": "translation"})
    fake_db.add(
        "speech_evaluation",
        {"sentence_id": SID_1, "scholar_id": "sch_1"},
    )
    fake_db.add("learning_attempt", {"sentence_id": SID_1})
    fake_db.add("audio_asset", {"sentence_id": SID_1, "asset_id": "as_1"})
    fake_db.add(
        "conversation_turn",
        {"turn_id": "t_1", "utterance": f"请翻译：{SID_1}", "reply": "Hello!"},
    )
    fake_db.add(
        "conversation_turn",
        {"turn_id": "t_2", "utterance": "无关内容", "reply": "Good morning!"},
    )


def _audit_actions(fake_db) -> list[str]:
    return [row["action"] for row in fake_db.all("audit_log")]


# ===========================================================================
# 1. GET /english/textbook/{tid}/lessons/{lid}/sentences — 语句列表
# ===========================================================================


class TestSentenceList:
    def test_list_returns_sentences_with_related_counts(self, make_client, fake_db):
        """列表返回语句元数据 + 关联数据计数 + duplicate 统计 + lesson_title。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_related_data(fake_db)
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["lesson_id"] == LS_ID
        assert data["lesson_title"] == "Let's talk"
        assert data["total"] == 3
        assert data["page"] == 1
        assert data["page_size"] == 50
        by_id = {s["sentence_id"]: s for s in data["sentences"]}
        s1 = by_id[SID_1]
        assert s1["text"] == "Hello!"
        assert s1["translation"] == "你好！"
        assert s1["text_hash"]  # getter 注入
        # 重复统计：s_001 与 s_002 同 hash
        assert s1["duplicate_count"] == 1
        assert SID_2 in s1["duplicate_sentence_ids"]
        # 关联数据计数
        assert s1["related_data"]["study_attempt_count"] == 2
        assert s1["related_data"]["skill_state_count"] == 1
        assert s1["related_data"]["speech_evaluation_count"] == 1
        assert s1["related_data"]["learning_attempt_count"] == 1
        assert s1["related_data"]["audio_asset_count"] == 1
        # s_003 独立句：无重复、无关联数据
        s3 = by_id[SID_3]
        assert s3["duplicate_count"] == 0
        assert s3["duplicate_sentence_ids"] == []
        assert s3["related_data"]["study_attempt_count"] == 0

    def test_list_keyword_filter(self, make_client, fake_db):
        """keyword 对 text 模糊匹配（大小写不敏感）。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.get(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            params={"keyword": "hello"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        ids = sorted(s["sentence_id"] for s in data["sentences"])
        assert ids == [SID_1, SID_2], f"hello 应匹配 Hello! 与 hello。，实得 {ids}"

    def test_list_duplicate_only(self, make_client, fake_db):
        """duplicate_only=true 只返回重复句。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.get(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            params={"duplicate_only": "true"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        ids = sorted(s["sentence_id"] for s in data["sentences"])
        assert ids == [SID_1, SID_2]

    def test_list_pagination(self, make_client, fake_db):
        """分页：page_size=2 时第 1 页 2 条、第 2 页 1 条，total 不变。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        r1 = client.get(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            params={"page": 1, "page_size": 2},
        )
        d1 = r1.json()["data"]
        assert d1["total"] == 3
        assert len(d1["sentences"]) == 2
        r2 = client.get(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            params={"page": 2, "page_size": 2},
        )
        d2 = r2.json()["data"]
        assert len(d2["sentences"]) == 1

    def test_list_lesson_not_found_404(self, make_client, fake_db):
        """lesson 不存在 → 404 LESSON_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB_ID}/lessons/ls_999/sentences")
        assert resp.status_code == 404, resp.text
        assert "LESSON_NOT_FOUND" in resp.json()["detail"]

    def test_list_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/tb_xxx/lessons/{LS_ID}/sentences")
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]


# ===========================================================================
# 2. POST /english/textbook/{tid}/lessons/{lid}/sentences — 新增语句
# ===========================================================================


class TestSentenceCreate:
    def test_create_batch_success_and_audit(self, make_client, fake_db):
        """批量新增成功：返回 created/skipped/sentences + 审计 create_english_sentences。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            json={
                "sentences": [
                    {
                        "text": "Good morning!",
                        "translation": "早上好！",
                        "audio_url": "https://a/gm.mp3",
                        "knowledge_point_ids": ["kp_002"],
                    },
                    {"text": "How are you?", "translation": "你好吗？"},
                ]
            },
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["created"] == 2
        assert data["skipped_duplicates"] == 0
        assert len(data["sentences"]) == 2
        assert all(s["sentence_id"].startswith("s_") for s in data["sentences"])
        assert all(s["text_hash"] for s in data["sentences"])
        # 审计落库
        assert "create_english_sentences" in _audit_actions(fake_db)
        # 落库可查
        rows = fake_db.all("sentence_v2")
        assert len(rows) == 2

    def test_create_skips_duplicate_in_lesson(self, make_client, fake_db):
        """同 lesson 内 text_hash 重复 → 跳过 + skipped_duplicates 计数。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)  # 已含 "Hello!"
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            json={"sentences": [{"text": "hello", "translation": "你好"}]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["created"] == 0
        assert data["skipped_duplicates"] == 1

    def test_create_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/tb_xxx/lessons/{LS_ID}/sentences",
            json={"sentences": [{"text": "Hi", "translation": "嗨"}]},
        )
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]

    def test_create_lesson_not_found_404(self, make_client, fake_db):
        """lesson 不存在 → 404 LESSON_NOT_FOUND。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/ls_999/sentences",
            json={"sentences": [{"text": "Hi", "translation": "嗨"}]},
        )
        assert resp.status_code == 404, resp.text
        assert "LESSON_NOT_FOUND" in resp.json()["detail"]

    def test_create_empty_text_400(self, make_client, fake_db):
        """text 为空 → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/lessons/{LS_ID}/sentences",
            json={"sentences": [{"text": "   ", "translation": "空白"}]},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]
        # 不落库
        assert fake_db.all("sentence_v2") == []


# ===========================================================================
# 3. PUT /english/sentence/{sentence_id} — 编辑语句
# ===========================================================================


class TestSentenceEdit:
    def test_edit_text_recomputes_hash_and_audit(self, make_client, fake_db):
        """改 text → 重算 text_hash + 审计 edit_english_sentence。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.put(
            f"/english/sentence/{SID_1}",
            json={"text": "Hello! How are you?"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["sentence_id"] == SID_1
        assert data["text"] == "Hello! How are you?"
        assert data["text_hash"] != ""  # 新 hash
        # 落库回读
        rows = fake_db.all("sentence_v2")
        row = next(r for r in rows if r["sentence_id"] == SID_1)
        assert row["text"] == "Hello! How are you?"
        assert row["text_hash"] == data["text_hash"]
        # 审计
        assert "edit_english_sentence" in _audit_actions(fake_db)

    def test_edit_optional_fields_only(self, make_client, fake_db):
        """只改 translation/knowledge_point_ids，text_hash 不变。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.put(
            f"/english/sentence/{SID_1}",
            json={"translation": "你好世界！", "knowledge_point_ids": ["kp_x"]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["translation"] == "你好世界！"
        assert data["knowledge_point_ids"] == ["kp_x"]
        assert data["text"] == "Hello!"  # text 不变

    def test_edit_sentence_not_found_404(self, make_client, fake_db):
        """sentence 不存在 → 404 SENTENCE_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.put("/english/sentence/s_999", json={"text": "Hi"})
        assert resp.status_code == 404, resp.text
        assert "SENTENCE_NOT_FOUND" in resp.json()["detail"]

    def test_edit_empty_text_400(self, make_client, fake_db):
        """text 置空 → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.put(f"/english/sentence/{SID_1}", json={"text": ""})
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]


# ===========================================================================
# 4. DELETE /english/sentence/{sentence_id} — 删除 + 级联清理
# ===========================================================================


class TestSentenceDelete:
    def test_delete_cascades_6_tables_and_audit(self, make_client, fake_db):
        """confirm 匹配删除：5 表物理删 + conversation_turn 标记 + 审计 delete_english_sentence。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_related_data(fake_db)
        client = make_client(english_router)
        resp = client.request("DELETE", f"/english/sentence/{SID_1}",
            json={"confirm_text": "Hello!", "delete_audio_asset": True},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["sentence_id"] == SID_1
        deleted = data["deleted"]
        assert deleted["study_attempt"] == 2
        assert deleted["skill_state"] == 1
        assert deleted["speech_evaluation"] == 1
        assert deleted["learning_attempt"] == 1
        assert deleted["audio_asset"] == 1  # delete_audio_asset=true
        assert deleted["conversation_turn_marked"] == 1  # t_1 引用 s_001
        assert deleted["sentence_v2"] == 1
        assert data["duplicates_deleted"] == 0
        # sentence_v2 已删
        assert all(r["sentence_id"] != SID_1 for r in fake_db.all("sentence_v2"))
        # 关联表已删
        assert all(r["sentence_id"] != SID_1 for r in fake_db.all("study_attempt"))
        assert all(r["sentence_id"] != SID_1 for r in fake_db.all("skill_state"))
        assert all(r["sentence_id"] != SID_1 for r in fake_db.all("speech_evaluation"))
        assert all(r["sentence_id"] != SID_1 for r in fake_db.all("learning_attempt"))
        assert fake_db.all("audio_asset") == []
        # conversation_turn 标记不删
        turns = fake_db.all("conversation_turn")
        t1 = next(t for t in turns if t["turn_id"] == "t_1")
        assert t1.get("deleted_sentence_ref") is True
        t2 = next(t for t in turns if t["turn_id"] == "t_2")
        assert not t2.get("deleted_sentence_ref")
        # 审计
        assert "delete_english_sentence" in _audit_actions(fake_db)

    def test_delete_audio_asset_default_false(self, make_client, fake_db):
        """delete_audio_asset 缺省 false → audio_asset 保留。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_related_data(fake_db)
        client = make_client(english_router)
        resp = client.request("DELETE", f"/english/sentence/{SID_1}", json={"confirm_text": "Hello!"}
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["deleted"]["audio_asset"] == 0
        assert len(fake_db.all("audio_asset")) == 1

    def test_delete_confirm_mismatch_400(self, make_client, fake_db):
        """confirm_text 不匹配 → 400 CONFIRM_TEXT_MISMATCH，不删除。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.request("DELETE", f"/english/sentence/{SID_1}", json={"confirm_text": "Wrong text"}
        )
        assert resp.status_code == 400, resp.text
        assert "CONFIRM_TEXT_MISMATCH" in resp.json()["detail"]
        assert len(fake_db.all("sentence_v2")) == 3

    def test_delete_sentence_not_found_404(self, make_client, fake_db):
        """sentence 不存在 → 404 SENTENCE_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.request("DELETE", "/english/sentence/s_999", json={"confirm_text": "Hi"}
        )
        assert resp.status_code == 404, resp.text
        assert "SENTENCE_NOT_FOUND" in resp.json()["detail"]

    def test_delete_with_duplicates_recursive(self, make_client, fake_db):
        """delete_duplicates=true → 同 text_hash 其他句（s_002）递归级联删除。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        _seed_related_data(fake_db)
        client = make_client(english_router)
        resp = client.request("DELETE", f"/english/sentence/{SID_1}",
            json={"confirm_text": "Hello!", "delete_duplicates": True},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["duplicates_deleted"] == 1
        remaining = [r["sentence_id"] for r in fake_db.all("sentence_v2")]
        assert SID_1 not in remaining
        assert SID_2 not in remaining
        assert SID_3 in remaining
