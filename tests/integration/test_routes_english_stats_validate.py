"""E1.2 验收集成测试：英语统计/树/校验 3 接口

覆盖 SOP §5 E1.2 规格来源（契约 api-contract.md §3.11 E-API-1~E-API-3）：
1. GET  /english/textbook/stats                  教材统计概览（章/课/句/重复数 + 校验状态）
2. GET  /english/textbook/{tid}/chapters         章节课时树（lesson 语句数 + 重复数 + 校验状态）
3. POST /english/textbook/{tid}/validate-sentences  语句归属校验（5 类异常 + summary + 缓存 TTL 1h）

统一断言模式：成功 → 200 + success:true + data:{...}；失败：400/404 对应契约错误码；
validation_status 从校验缓存读取，缺省 pending（service-contract §8.4）。
"""
from __future__ import annotations

import pytest

from services.routes_english import router as english_router
from services.english.validation import clear_validation_cache

TB_ID = "tb_eng_pep4"
TB2_ID = "tb_eng_pep5"
CH_1 = "ch_001"
CH_2 = "ch_002"
LS_1 = "ls_001"
LS_2 = "ls_002"
LS_3 = "ls_003"


@pytest.fixture(autouse=True)
def _fresh_validation_cache():
    """每个测试独立校验缓存（模块级 TTLCache 会跨测试残留）。"""
    clear_validation_cache()
    yield
    clear_validation_cache()


def _seed_textbook(fake_db, *, textbook_id=TB_ID, grade="四年级", **overrides) -> dict:
    doc = {
        "textbook_id": textbook_id,
        "title": "PEP 四年级上册" if textbook_id == TB_ID else "PEP 五年级上册",
        "grade": grade,
        "level": None,
        "subject_type": "english",
        "chapters": [
            {
                "chapter_id": CH_1,
                "title": "Unit 1 Hello",
                "lessons": [
                    {"lesson_id": LS_1, "title": "Let's talk"},
                    {"lesson_id": LS_2, "title": "Let's learn"},
                ],
            },
            {
                "chapter_id": CH_2,
                "title": "Unit 2 Colours",
                "lessons": [{"lesson_id": LS_3, "title": "Let's check"}],
            },
        ],
        "created_at": 1750000000000,
        "updated_at": 1750000000000,
    }
    doc.update(overrides)
    fake_db.add("textbook_v2", doc)
    return doc


def _seed_english_textbooks(fake_db) -> None:
    """两本英语教材：tb_eng_pep4（四年级）+ tb_eng_pep5（五年级）。"""
    _seed_textbook(fake_db)
    _seed_textbook(
        fake_db,
        textbook_id=TB2_ID,
        grade="五年级",
        chapters=[
            {
                "chapter_id": "ch_101",
                "title": "Unit A",
                "lessons": [{"lesson_id": "ls_101", "title": "Story time"}],
            }
        ],
    )


def _sentence(fake_db, *, sid, text, translation, textbook_id, lesson_id, chapter_id):
    """通用语句种子。"""
    fake_db.add(
        "sentence_v2",
        {
            "sentence_id": sid,
            "text": text,
            "translation": translation,
            "audio_url": "",
            "knowledge_point_ids": [],
            "textbook_id": textbook_id,
            "lesson_id": lesson_id,
            "chapter_id": chapter_id,
            "created_at": 1750000000000,
            "updated_at": 1750000000000,
        },
    )


def _seed_sentences(fake_db) -> None:
    """tb_eng_pep4 下 7 句 + tb_eng_pep5 下 1 句（跨教材重复）。

    tb_eng_pep4（7 句）：
      ls_001/ch_001：s_001 Hello! / s_002 hello。（同 hash 重复对）/ s_003 Good morning!
      ls_002/ch_001：s_004 How are you? / s_007 See you!（translation 空 → empty_content）
      orphan：      s_005 Who is that?（lesson_id=ls_999 不存在，chapter_id=ch_002）
      mismatch：    s_006 Nice to meet you!（lesson_id=ls_001 但 chapter_id=ch_002）
    tb_eng_pep5：  s_008 Hello!（与 s_001/s_002 同 hash → cross_textbook_duplicate）
    """
    # tb_eng_pep4
    _sentence(fake_db, sid="s_001", text="Hello!", translation="你好！",
              textbook_id=TB_ID, lesson_id=LS_1, chapter_id=CH_1)
    _sentence(fake_db, sid="s_002", text="hello。", translation="你好！",
              textbook_id=TB_ID, lesson_id=LS_1, chapter_id=CH_1)
    _sentence(fake_db, sid="s_003", text="Good morning!", translation="早上好！",
              textbook_id=TB_ID, lesson_id=LS_1, chapter_id=CH_1)
    _sentence(fake_db, sid="s_004", text="How are you?", translation="你好吗？",
              textbook_id=TB_ID, lesson_id=LS_2, chapter_id=CH_1)
    _sentence(fake_db, sid="s_005", text="Who is that?", translation="那是谁？",
              textbook_id=TB_ID, lesson_id="ls_999", chapter_id=CH_2)
    _sentence(fake_db, sid="s_006", text="Nice to meet you!", translation="见到你很高兴！",
              textbook_id=TB_ID, lesson_id=LS_1, chapter_id=CH_2)
    _sentence(fake_db, sid="s_007", text="See you!", translation="",
              textbook_id=TB_ID, lesson_id=LS_2, chapter_id=CH_1)
    # tb_eng_pep5（跨教材重复）
    _sentence(fake_db, sid="s_008", text="Hello!", translation="你好！",
              textbook_id=TB2_ID, lesson_id="ls_101", chapter_id="ch_101")


def _seed_math_textbook(fake_db) -> None:
    """一本数学教材（subject_type=math，不应出现在英语 stats）。"""
    fake_db.add(
        "textbook_v2",
        {
            "textbook_id": "tb_math_4a",
            "title": "数学四年级上册",
            "grade": "四年级",
            "level": None,
            "subject_type": "math",
            "chapters": [{"chapter_id": "mch_1", "title": "第一章", "lessons": []}],
            "created_at": 1750000000000,
            "updated_at": 1750000000000,
        },
    )


# ===========================================================================
# 1. GET /english/textbook/stats — 教材统计概览（E-API-1）
# ===========================================================================


class TestTextbookStats:
    def test_stats_aggregates_counts(self, make_client, fake_db):
        """聚合统计：chapter/lesson/sentence/duplicate 计数 + validation_status=pending。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.get("/english/textbook/stats")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        by_id = {t["textbook_id"]: t for t in body["data"]["textbooks"]}
        assert set(by_id) == {TB_ID, TB2_ID}
        t1 = by_id[TB_ID]
        assert t1["title"] == "PEP 四年级上册"
        assert t1["grade"] == "四年级"
        assert t1["level"] is None
        assert t1["chapter_count"] == 2
        assert t1["lesson_count"] == 3
        assert t1["sentence_count"] == 7
        # 重复数 = 冗余实例数（s_001/s_002 同 hash → 1）
        assert t1["duplicate_count"] == 1
        assert t1["validation_status"] == "pending"
        assert t1["updated_at"] == 1750000000000
        t2 = by_id[TB2_ID]
        assert t2["sentence_count"] == 1
        assert t2["duplicate_count"] == 0

    def test_stats_excludes_other_subjects(self, make_client, fake_db):
        """非英语教材（subject_type=math）不出现在英语 stats。"""
        _seed_english_textbooks(fake_db)
        _seed_math_textbook(fake_db)
        client = make_client(english_router)
        resp = client.get("/english/textbook/stats")
        assert resp.status_code == 200, resp.text
        ids = [t["textbook_id"] for t in resp.json()["data"]["textbooks"]]
        assert TB_ID in ids
        assert TB2_ID in ids
        assert "tb_math_4a" not in ids

    def test_stats_grade_filter(self, make_client, fake_db):
        """grade 过滤：四年级 → 仅 tb_eng_pep4。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.get("/english/textbook/stats", params={"grade": "四年级"})
        assert resp.status_code == 200, resp.text
        ids = [t["textbook_id"] for t in resp.json()["data"]["textbooks"]]
        assert ids == [TB_ID]

    def test_stats_validation_status_from_cache(self, make_client, fake_db):
        """校验（scope=full）后 stats 的 validation_status 从缓存读取为 error。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        rv = client.post(f"/english/textbook/{TB_ID}/validate-sentences", json={})
        assert rv.status_code == 200, rv.text
        assert rv.json()["data"]["summary"]["error_count"] > 0
        rs = client.get("/english/textbook/stats")
        by_id = {t["textbook_id"]: t for t in rs.json()["data"]["textbooks"]}
        assert by_id[TB_ID]["validation_status"] == "error"


# ===========================================================================
# 2. GET /english/textbook/{tid}/chapters — 章节课时树（E-API-2）
# ===========================================================================


class TestChapterTree:
    def test_chapter_tree_counts(self, make_client, fake_db):
        """章节树：chapter→lesson 结构 + 每 lesson 语句数/重复数 + orphan 不归属任何 lesson。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB_ID}/chapters")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["textbook_id"] == TB_ID
        assert data["title"] == "PEP 四年级上册"
        chapters = {c["chapter_id"]: c for c in data["chapters"]}
        assert set(chapters) == {CH_1, CH_2}
        lessons = {l["lesson_id"]: l for l in chapters[CH_1]["lessons"]}
        assert set(lessons) == {LS_1, LS_2}
        # ls_001：s_001/s_002/s_003/s_006 4 句（orphan s_005 不归属）
        assert lessons[LS_1]["sentence_count"] == 4
        assert lessons[LS_1]["duplicate_count"] == 1  # s_001/s_002
        # ls_002：s_004/s_007 2 句
        assert lessons[LS_2]["sentence_count"] == 2
        assert lessons[LS_2]["duplicate_count"] == 0
        # ch_002/ls_003：0 句
        ch2_lessons = chapters[CH_2]["lessons"]
        assert ch2_lessons[0]["lesson_id"] == LS_3
        assert ch2_lessons[0]["sentence_count"] == 0
        # 校验状态缺省 pending
        assert lessons[LS_1]["validation_status"] == "pending"

    def test_chapter_tree_validation_status_from_cache(self, make_client, fake_db):
        """校验某 lesson 后，章节树该 lesson 显示 warning（error=0 但 warning>0）。"""
        _seed_textbook(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        rv = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "lesson", "lesson_id": LS_1},
        )
        assert rv.status_code == 200, rv.text
        summary = rv.json()["data"]["summary"]
        assert summary["error_count"] == 0
        assert summary["warning_count"] > 0
        resp = client.get(f"/english/textbook/{TB_ID}/chapters")
        data = resp.json()["data"]
        ch1 = next(c for c in data["chapters"] if c["chapter_id"] == CH_1)
        ls1 = next(l for l in ch1["lessons"] if l["lesson_id"] == LS_1)
        assert ls1["validation_status"] == "warning"

    def test_chapter_tree_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.get("/english/textbook/tb_xxx/chapters")
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]

    def test_chapter_tree_direct_lessons_without_chapters(self, make_client, fake_db):
        """无章节教材（textbook.lessons[]）：聚合为「未分章」伪章节，lesson 仍可见。"""
        _seed_textbook(
            fake_db,
            textbook_id=TB2_ID,
            grade="五年级",
            chapters=[],
            lessons=[{"lesson_id": "ls_101", "title": "Story time"}],
        )
        _sentence(fake_db, sid="s_101", text="Hello!", translation="你好！",
                  textbook_id=TB2_ID, lesson_id="ls_101", chapter_id="")
        client = make_client(english_router)
        resp = client.get(f"/english/textbook/{TB2_ID}/chapters")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert len(data["chapters"]) == 1
        ch = data["chapters"][0]
        assert ch["chapter_id"] == ""
        assert ch["lessons"][0]["lesson_id"] == "ls_101"
        assert ch["lessons"][0]["sentence_count"] == 1


# ===========================================================================
# 3. POST /english/textbook/{tid}/validate-sentences — 语句归属校验（E-API-3）
# ===========================================================================


class TestValidateSentences:
    def test_validate_full_all_checks(self, make_client, fake_db):
        """scope=full 全量校验：5 类异常全部命中 + summary 分级计数。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(f"/english/textbook/{TB_ID}/validate-sentences", json={})
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["textbook_id"] == TB_ID
        assert data["total_sentences"] == 7
        issues = data["issues"]
        # orphan_lesson：s_005（lesson_id=ls_999 不存在）
        assert len(issues["orphan_lesson"]) == 1
        assert issues["orphan_lesson"][0]["sentence_id"] == "s_005"
        assert issues["orphan_lesson"][0]["lesson_id"] == "ls_999"
        # chapter_mismatch：s_006（lesson 属 ch_001 但 chapter_id=ch_002）
        assert len(issues["chapter_mismatch"]) == 1
        cm = issues["chapter_mismatch"][0]
        assert cm["sentence_id"] == "s_006"
        assert cm["sentence_chapter_id"] == CH_2
        assert cm["lesson_chapter_id"] == CH_1
        # duplicate_in_textbook：s_001/s_002 同 hash
        assert len(issues["duplicate_in_textbook"]) == 1
        dup = issues["duplicate_in_textbook"][0]
        assert dup["count"] == 2
        assert set(dup["sentence_ids"]) == {"s_001", "s_002"}
        # cross_textbook_duplicate：Hello 在 tb_eng_pep5 也存在
        assert len(issues["cross_textbook_duplicate"]) == 1
        cross = issues["cross_textbook_duplicate"][0]
        assert cross["count"] == 2
        assert set(cross["in_textbooks"]) == {TB_ID, TB2_ID}
        # empty_content：s_007 translation 空
        assert len(issues["empty_content"]) == 1
        ec = issues["empty_content"][0]
        assert ec["sentence_id"] == "s_007"
        assert ec["field"] == "translation"
        # summary：error=orphan+empty=2；warning=chapter_mismatch+dup=2；info=cross=1
        summary = data["summary"]
        assert summary["total_issues"] == 5
        assert summary["error_count"] == 2
        assert summary["warning_count"] == 2
        assert summary["info_count"] == 1

    def test_validate_chapter_scope(self, make_client, fake_db):
        """scope=chapter 只校验指定 chapter（按 sentence.chapter_id）的句子。

        s_006 的 chapter_id=ch_002（其 lesson ls_001 错挂 ch_001），
        因此不在 ch_001 范围；chapter_mismatch 只在 full 校验中暴露。
        """
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "chapter", "chapter_id": CH_1},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        # ch_001 下 5 句（s_001/s_002/s_003/s_004/s_007）
        assert data["total_sentences"] == 5
        issues = data["issues"]
        assert issues["orphan_lesson"] == []  # s_005 在 ch_002
        assert issues["chapter_mismatch"] == []  # s_006 在 ch_002
        assert len(issues["duplicate_in_textbook"]) == 1
        assert len(issues["cross_textbook_duplicate"]) == 1
        assert len(issues["empty_content"]) == 1
        summary = data["summary"]
        assert summary["total_issues"] == 3
        assert summary["error_count"] == 1  # empty_content
        assert summary["warning_count"] == 1  # duplicate_in_textbook
        assert summary["info_count"] == 1  # cross_textbook_duplicate

    def test_validate_lesson_scope(self, make_client, fake_db):
        """scope=lesson 只校验指定 lesson 的句子。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "lesson", "lesson_id": LS_1},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["total_sentences"] == 4  # s_001/s_002/s_003/s_006
        issues = data["issues"]
        assert issues["orphan_lesson"] == []
        assert len(issues["chapter_mismatch"]) == 1  # s_006
        assert len(issues["duplicate_in_textbook"]) == 1
        assert len(issues["cross_textbook_duplicate"]) == 1
        assert issues["empty_content"] == []
        summary = data["summary"]
        assert summary["error_count"] == 0
        assert summary["warning_count"] == 2
        assert summary["info_count"] == 1

    def test_validate_check_types_subset(self, make_client, fake_db):
        """check_types 白名单：只执行指定维度。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"check_types": ["orphan_lesson", "empty_content"]},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        issues = data["issues"]
        assert len(issues["orphan_lesson"]) == 1
        assert len(issues["empty_content"]) == 1
        assert issues["chapter_mismatch"] == []
        assert issues["duplicate_in_textbook"] == []
        assert issues["cross_textbook_duplicate"] == []
        summary = data["summary"]
        assert summary["total_issues"] == 2
        assert summary["error_count"] == 2
        assert summary["warning_count"] == 0
        assert summary["info_count"] == 0

    def test_validate_missing_chapter_id_400(self, make_client, fake_db):
        """scope=chapter 缺 chapter_id → 400 SENTENCE_PAYLOAD_INVALID。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "chapter"},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]

    def test_validate_missing_lesson_id_400(self, make_client, fake_db):
        """scope=lesson 缺 lesson_id → 400。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "lesson"},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]

    def test_validate_invalid_scope_400(self, make_client, fake_db):
        """非法 scope → 400。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"scope": "bogus"},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]

    def test_validate_invalid_check_type_400(self, make_client, fake_db):
        """非法 check_types → 400。"""
        _seed_textbook(fake_db)
        client = make_client(english_router)
        resp = client.post(
            f"/english/textbook/{TB_ID}/validate-sentences",
            json={"check_types": ["bogus"]},
        )
        assert resp.status_code == 400, resp.text
        assert "SENTENCE_PAYLOAD_INVALID" in resp.json()["detail"]

    def test_validate_textbook_not_found_404(self, make_client, fake_db):
        """教材不存在 → 404 TEXTBOOK_NOT_FOUND。"""
        client = make_client(english_router)
        resp = client.post("/english/textbook/tb_xxx/validate-sentences", json={})
        assert resp.status_code == 404, resp.text
        assert "TEXTBOOK_NOT_FOUND" in resp.json()["detail"]

    def test_validate_result_cached(self, make_client, fake_db):
        """校验结果缓存：DB 变更后重复调用返回缓存结果（不重算）。"""
        _seed_english_textbooks(fake_db)
        _seed_sentences(fake_db)
        client = make_client(english_router)
        r1 = client.post(f"/english/textbook/{TB_ID}/validate-sentences", json={})
        assert r1.status_code == 200, r1.text
        assert r1.json()["data"]["total_sentences"] == 7
        # 删除一条语句后再次校验：命中缓存仍返回 7（未重算）
        fake_db.delete("sentence_v2", where={"sentence_id": "s_007"})
        r2 = client.post(f"/english/textbook/{TB_ID}/validate-sentences", json={})
        assert r2.status_code == 200, r2.text
        assert r2.json()["data"]["total_sentences"] == 7
