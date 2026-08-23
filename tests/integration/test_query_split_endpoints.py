"""集成测试:查询接口拆分(Phase 6) — 接口 2 / 接口 3

覆盖:
- 接口 2 GET /scholar/{scholar_id}/textbooks/{textbook_id}/lessons
  教材详情(lesson 列表 + summary): summary.mastery / lessons[].progress
  (overall_percent / mastery / skills / status_distribution)
- 接口 3 GET /tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences
  章节句子明细: summary + sentences[](status/skills/weakest_skill/review_count/next_review_at)
- 与聚合路径口径一致(乐观聚合 / 4 级档位加权掌握度)
- 无学习记录 / lesson 不存在
"""

from __future__ import annotations

import pytest

from services.routes_tracking import router as tracking_router
from tests.fakes.seed_factory import seed_content, seed_skill_states

# 与聚合测试一致的状态 + next_review_at（skill_state 数据工厂用例）
SCHOLAR_STATES = [
    {"scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
     "status": "learned", "mastery_score": 80, "attempt_count": 2,
     "next_review_at": 1784282400},
    {"scholar_id": "scholar_1", "sentence_id": "s2", "skill_code": "translation",
     "status": "learning", "mastery_score": 40, "attempt_count": 1},
    {"scholar_id": "scholar_1", "sentence_id": "s3", "skill_code": "translation",
     "status": "mastered", "mastery_score": 95, "attempt_count": 3},
    {"scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "listening",
     "status": "learning", "mastery_score": 30, "attempt_count": 1},
]


class TestGetTextbookLessons:
    """接口 2: GET /scholar/{scholar_id}/textbooks/{textbook_id}/lessons"""

    def test_full_aggregation(self, make_client, fake_db):
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        client = make_client(tracking_router)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        data = resp.json()["data"]

        # summary: 与聚合路径一致
        summary = data["summary"]
        assert summary["textbook_progress"] == pytest.approx(0.5375)  # (0.8+0.4+0.95+0)/4
        assert summary["total_sentence_count"] == 4
        assert summary["learned_sentence_count"] == 2  # s1, s3
        # mastery: 档位加权 (learning=1, learned=2, mastered=3) /3: s2=1, s1=2, s3=3, s4=0 → 6/12
        assert summary["mastery"] == pytest.approx(0.5)
        # f3 累计学习次数：乐观 pick_state 求和（s1 取 translation=2，不计 listening=1）
        assert summary["total_attempt_count"] == 6  # s1=2 + s2=1 + s3=3
        assert summary["avg_attempt_count"] == 3.0  # 6 / learned(2)

        # lessons 列表(2 课, 按序)
        assert [l["lesson_id"] for l in data["lessons"]] == ["l1", "l2"]
        l1 = data["lessons"][0]
        assert l1["lesson_title"] == "L1"
        prog = l1["progress"]
        assert prog["overall_percent"] == 60  # (0.8+0.4)/2 = 0.6 → 60
        # l1: s1=learned, s2=learning → (2+1)/(3*2)=0.5
        assert prog["mastery"] == pytest.approx(0.5)
        assert prog["status_distribution"] == [0, 1, 1, 0, 0, 0]
        # skills: 各能力独立聚合
        assert prog["skills"]["translation"] == pytest.approx(0.5)  # s1=learned, s2=learning
        # listening: l1 共 2 句, 仅 s1 有记录(learning) → 1/(3*2)
        assert prog["skills"]["listening"] == pytest.approx(0.1667, abs=1e-4)

    def test_lesson_skills_include_conversation(self, make_client, fake_db):
        """每课 progress.skills 纳入对话能力：与句子级 skills 口径一致（概览不缺对话）。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1",
            "skill_code": "conversation", "status": "learned",
            "mastery_score": 70, "attempt_count": 1,
        })
        client = make_client(tracking_router)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        l1 = resp.json()["data"]["lessons"][0]
        # l1 共 2 句，仅 s1 有 conversation(learned) → 2/(3*2)
        assert l1["progress"]["skills"]["conversation"] == pytest.approx(0.3333, abs=1e-4)

    def test_no_states(self, make_client, fake_db):
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["summary"]["textbook_progress"] == 0.0
        assert data["summary"]["mastery"] == 0.0
        assert data["summary"]["learned_sentence_count"] == 0
        # f3 空态：无任何学习记录 → 累计次数为 0，avg 分母为 0 → 0.0
        assert data["summary"]["total_attempt_count"] == 0
        assert data["summary"]["avg_attempt_count"] == 0.0
        l1 = data["lessons"][0]
        assert l1["progress"]["overall_percent"] == 0
        assert l1["progress"]["mastery"] == 0.0
        assert l1["progress"]["skills"] == {}
        assert l1["progress"]["status_distribution"] == [0, 0, 0, 0, 0, 0]

    def test_avg_attempt_count_zero_when_none_learned(self, make_client, fake_db):
        """f3 分母边界：有学习次数但无 learned/mastered → avg 恒 0.0（不除零）。"""
        seed_content(fake_db)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1", "skill_code": "translation",
            "status": "learning", "mastery_score": 40, "attempt_count": 5,
        })
        client = make_client(tracking_router)
        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        summary = resp.json()["data"]["summary"]
        assert summary["learned_sentence_count"] == 0
        assert summary["total_attempt_count"] == 5
        assert summary["avg_attempt_count"] == 0.0


class TestGetLessonSentences:
    """接口 3: GET /tracking/textbooks/{textbook_id}/lessons/{lesson_id}/sentences"""

    def test_sentence_detail_and_summary(self, make_client, fake_db):
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["lesson_id"] == "l1"
        assert data["lesson_title"] == "L1"

        # 仅返回该 lesson 的 2 句
        assert [s["sentence_id"] for s in data["sentences"]] == ["s1", "s2"]
        s1 = data["sentences"][0]
        assert s1["content"] == "Text s1"
        assert s1["translation"] == "译s1"
        # 乐观聚合 pick: translation(learned, 80) > listening(learning, 30) → learned=2
        assert s1["status"] == 2
        assert s1["skills"] == {"translation": 2, "listening": 1}
        assert s1["weakest_skill"] == "listening"
        assert s1["review_count"] == 2
        assert s1["next_review_at"] is not None  # int 时间戳 → ISO
        s2 = data["sentences"][1]
        assert s2["status"] == 1  # learning
        assert s2["skills"] == {"translation": 1}
        assert s2["weakest_skill"] == "translation"
        assert s2["review_count"] == 1
        assert s2["next_review_at"] is None

        # summary: lesson 粒度
        summary = data["summary"]
        assert summary["total_sentence_count"] == 2
        assert summary["learned_sentence_count"] == 1  # s1
        assert summary["mastery"] == pytest.approx(0.5)  # (learning=1 + learned=2)/(3*2)
        assert summary["skills"]["translation"] == pytest.approx(0.5)  # s1=learned, s2=learning
        # listening: 该课 2 句仅 s1 有记录(learning) → 1/(3*2)
        assert summary["skills"]["listening"] == pytest.approx(0.1667, abs=1e-4)

    def test_summary_skills_include_conversation(self, make_client, fake_db):
        """概览 summary.skills 纳入对话能力：与句子级 skills 口径一致（概览不缺对话）。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        fake_db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": "s1",
            "skill_code": "conversation", "status": "learned",
            "mastery_score": 70, "attempt_count": 1,
        })
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        # 句子级 skills：conversation 全量输出（learned → 2）
        s1 = data["sentences"][0]
        assert s1["skills"]["conversation"] == 2
        # 概览 summary.skills：对话能力纳入聚合（s1 learned → 2/(3*2)）
        assert data["summary"]["skills"]["conversation"] == pytest.approx(0.3333, abs=1e-4)
        # s1 仍为 learned，其余统计不受影响
        assert s1["status"] == 2
        assert data["summary"]["learned_sentence_count"] == 1

    def test_no_states(self, make_client, fake_db):
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data["sentences"]) == 2
        s1 = data["sentences"][0]
        assert s1["status"] == 0
        assert s1["skills"] == {}
        assert s1["weakest_skill"] is None
        assert s1["review_count"] == 0
        assert s1["next_review_at"] is None
        assert data["summary"]["mastery"] == 0.0
        assert data["summary"]["learned_sentence_count"] == 0
        assert data["summary"]["skills"] == {}

    def test_lesson_not_found(self, make_client, fake_db):
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/nope/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 404

    def test_states_query_scoped_to_lesson(self, make_client, fake_db):
        """性能回归：states 按本课句子 $in 过滤，查询固定 5 次，与学者总状态量解耦。

        防退化点：skill_state 必须按 sentence_id $in 查询（仅本课句子），
        不允许全量分页拉取该学者全部状态（1100+ 条无关状态会触发 2 次分页）。
        优化后章节明细查询恒为 5 次：chapter + lesson + sentence_v2 +
        sentence_group（M3 G1.1 组元数据）+ skill_state。
        """
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        # 大量无关状态（其它句子/教材），验证不会触发全量分页拉取
        for i in range(1100):
            fake_db.add("skill_state", {
                "scholar_id": "scholar_1",
                "sentence_id": f"other_{i}",
                "skill_code": "reading",
                "status": "learned",
                "mastery_score": 50,
                "attempt_count": 1,
            })

        calls: list[dict] = []
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls.append({"collection": kwargs.get("collection"), "where": kwargs.get("where")})
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["lesson_id"] == "l1"
        # 输出不受无关状态影响：仍只返回本课 2 句、s1 learned
        assert [s["sentence_id"] for s in data["sentences"]] == ["s1", "s2"]
        assert data["summary"]["total_sentence_count"] == 2
        assert data["summary"]["learned_sentence_count"] == 1

        state_calls = [c for c in calls if c["collection"] == "skill_state"]
        assert len(state_calls) == 1  # 一课 2 句 → $in 一次查回
        assert state_calls[0]["where"]["scholar_id"] == "scholar_1"
        assert set(state_calls[0]["where"]["sentence_id"]["$in"]) == {"s1", "s2"}
        # 总查询固定：chapter + lesson + sentence_v2 + sentence_group(M3) + skill_state = 5
        assert len(calls) == 5
        assert len([c for c in calls if c["collection"] == "sentence_group"]) == 1


class TestQueryCountOptimization:
    """性能回归：接口 2 查询次数固定（内容 3 次 + states 1 次 + attempts 1 次 = 5 次）。

    防退化点：
    - 内容层级必须批量 $in（chapters / lessons / sentences 各 1 次），
      不允许逐章 get_lessons / 逐课 get_sentences_by_lesson 的 N+1；
    - base + 4 能力聚合必须内存内过滤复用同一份数据，不允许重复触库。
    优化前本场景（1 章 2 课）需 5×(1+1+1+2+1) = 30 次，优化后恒为 5 次。
    """

    def test_query_count_is_constant(self, make_client, fake_db):
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)

        calls = {"n": 0}
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls["n"] += 1
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        client = make_client(tracking_router)

        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        assert calls["n"] == 5  # chapters + lessons + sentences + states + attempts

    def test_query_count_scales_with_lessons_only_via_batch(self, make_client, fake_db):
        """扩大教材规模（3 章 6 课 12 句）查询次数仍为 5，验证批量 $in 生效。"""
        for i in range(1, 4):
            fake_db.add("chapter", {
                "chapter_id": f"c{i}", "textbook_id": "tb_1", "title": f"Ch{i}", "order": i,
            })
        for i in range(1, 7):
            ch = f"c{(i - 1) // 2 + 1}"
            fake_db.add("lesson", {
                "lesson_id": f"l{i}", "chapter_id": ch, "title": f"L{i}", "order": i,
            })
        for i in range(1, 13):
            fake_db.add("sentence_v2", {
                "sentence_id": f"s{i}", "lesson_id": f"l{(i - 1) // 2 + 1}",
                "chapter_id": f"c{(i - 1) // 4 + 1}", "textbook_id": "tb_1",
                "text": f"Text s{i}", "translation": f"译s{i}", "order": i,
            })

        calls = {"n": 0}
        orig_query = fake_db.query

        async def counting_query(*args, **kwargs):
            calls["n"] += 1
            return await orig_query(*args, **kwargs)

        fake_db.query = counting_query
        client = make_client(tracking_router)

        resp = client.get("/scholar/scholar_1/textbooks/tb_1/lessons")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["lessons"]) == 6
        assert calls["n"] == 5


class TestGetLessonGroups:
    """M3 G1.1: GET /tracking/textbooks/{textbook_id}/lessons/{lesson_id}/groups — 组视图（读兼容层）"""

    def test_legacy_groups_when_no_groups(self, make_client, fake_db):
        """无任何 sentence_group → 逐句构造临时组 legacy_{lesson_id}_{sentence_id}，结构逐字一致。"""
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/groups",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["lesson_id"] == "l1"
        assert data["lesson_title"] == "L1"
        groups = data["groups"]
        assert [g["group_id"] for g in groups] == ["legacy_l1_s1", "legacy_l1_s2"]
        g0 = groups[0]
        assert g0["group_type"] == "stand_alone"
        assert g0["order_in_lesson"] == 0
        assert g0["group_title"] == "Text s1"
        assert len(g0["sentences"]) == 1
        assert g0["sentences"][0]["content"] == "Text s1"
        # 临时组与真实组字段键一致
        for g in groups:
            assert {"group_id", "group_title", "group_type", "order_in_lesson", "sentences"} <= set(g)
        # summary 含 group_count
        assert data["summary"]["group_count"] == 2
        assert data["summary"]["total_sentence_count"] == 2

    def test_real_groups_organized(self, make_client, fake_db):
        """有 sentence_group → 按组组织返回（含组元数据 + 组内句子）。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        fake_db.add("sentence_group", {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "l1",
            "title": "Yes/No 应答组", "type": "dialogue_pair",
            "sentence_ids": ["s2", "s1"], "order_in_lesson": 0,
        })
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/groups",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        groups = data["groups"]
        assert len(groups) == 1
        g = groups[0]
        assert g["group_id"] == "grp_1"
        assert g["group_title"] == "Yes/No 应答组"
        assert g["group_type"] == "dialogue_pair"
        assert [s["sentence_id"] for s in g["sentences"]] == ["s2", "s1"]  # 按 sentence_ids 顺序
        assert data["summary"]["group_count"] == 1

    def test_status_matches_sentences_interface(self, make_client, fake_db):
        """组内句子 status/skills 与 /sentences 逐字一致（乐观聚合）。"""
        seed_content(fake_db)
        seed_skill_states(fake_db, SCHOLAR_STATES)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/groups",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        entries = {s["sentence_id"]: s for g in data["groups"] for s in g["sentences"]}
        s1 = entries["s1"]
        assert s1["status"] == 2  # learned（乐观）
        assert s1["skills"] == {"translation": 2, "listening": 1}
        assert s1["weakest_skill"] == "listening"
        assert s1["review_count"] == 2
        assert s1["next_review_at"] is not None
        assert s1["is_canonical"] is True  # 未去重 → canonical
        assert s1["canonical_sentence_id"] is None
        # summary 与 /sentences 口径一致 + group_count
        summary = data["summary"]
        assert summary["learned_sentence_count"] == 1
        assert summary["mastery"] == pytest.approx(0.5)

    def test_missing_scholar_id_400(self, make_client, fake_db):
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get("/tracking/textbooks/tb_1/lessons/l1/groups")
        assert resp.status_code == 400

    def test_lesson_not_found_404(self, make_client, fake_db):
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/nope/groups",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 404


class TestGetLessonSentencesM3Fields:
    """M3 G1.1: /sentences 追加 5 可选字段（group_id/group_title/group_type/is_canonical/canonical_sentence_id）"""

    def test_5_fields_null_without_groups(self, make_client, fake_db):
        """未分组 → 5 字段全 null（纯新增，不破坏旧调用方）。"""
        seed_content(fake_db)
        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        s1 = resp.json()["data"]["sentences"][0]
        assert s1["group_id"] is None
        assert s1["group_title"] is None
        assert s1["group_type"] is None
        assert s1["is_canonical"] is None
        assert s1["canonical_sentence_id"] is None
        # 既有字段不受影响
        assert s1["status"] == 0
        assert s1["content"] == "Text s1"

    def test_5_fields_populated_with_groups(self, make_client, fake_db):
        """已分组 → group_* 来自 sentence_group，is_canonical 按 canonical_sentence_id 判定。"""
        seed_content(fake_db)
        fake_db.add("sentence_group", {
            "_id": "grp_1", "group_id": "grp_1", "lesson_id": "l1",
            "title": "Yes/No 应答组", "type": "dialogue_pair",
            "sentence_ids": ["s1", "s2"], "order_in_lesson": 0,
        })
        # s2 为重复句（canonical 指向 s1）
        rows = fake_db.all("sentence_v2")
        fake_db.clear("sentence_v2")
        for r in rows:
            if r.get("sentence_id") == "s2":
                r["canonical_sentence_id"] = "s1"
            if r.get("sentence_id") == "s1":
                r["group_id"] = "grp_1"
            if r.get("sentence_id") == "s2":
                r["group_id"] = "grp_1"
            fake_db.add("sentence_v2", r)

        client = make_client(tracking_router)
        resp = client.get(
            "/tracking/textbooks/tb_1/lessons/l1/sentences",
            params={"scholar_id": "scholar_1"},
        )
        assert resp.status_code == 200
        sentences = resp.json()["data"]["sentences"]
        by_id = {s["sentence_id"]: s for s in sentences}

        s1 = by_id["s1"]
        assert s1["group_id"] == "grp_1"
        assert s1["group_title"] == "Yes/No 应答组"
        assert s1["group_type"] == "dialogue_pair"
        assert s1["is_canonical"] is True  # canonical 为 null/自身
        assert s1["canonical_sentence_id"] is None

        s2 = by_id["s2"]
        assert s2["group_id"] == "grp_1"
        assert s2["group_title"] == "Yes/No 应答组"
        assert s2["is_canonical"] is False  # 重复句
        assert s2["canonical_sentence_id"] == "s1"
