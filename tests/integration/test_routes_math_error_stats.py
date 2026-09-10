"""F4.6 学者错题列表接口验收测试（契约 api-contract.md §3.10 接口 25）

验收场景：
- GET /math/error-stats?scholar_id=xxx：
  - 无 scholar_id → 400；
  - 空学者 → { items: [], total: 0 }；
  - 多条记录 → 按 created_at 倒序，字段映射 record_id→error_record_id、
    primary_error→error_type，透传 knowledge_point_name/source/created_at；
  - knowledge_point_name 过滤（可选）；
  - total 不受 limit 截断影响。
- M13 双源透传（接口 25 出参增 kp_source / exam_backlink_to，data-model §4.12.9(b)）：
  - 缺省按数据事实兜底：EXTRA_AI（真题/图谱外）→ exam_paper；正式教材链/无链 → textbook；
  - 显式落库字段原样透传（kp_source / exam_backlink_to）；无回链 → exam_backlink_to=None。
"""
from __future__ import annotations

from services.routes_math import router as math_router


def _seed_error_record(fake_db, *, record_id, scholar_id="s1", kp="进位加法",
                       error_type="computation", source="auto_scan",
                       created_at=1000, node_code="", **overrides) -> dict:
    doc = {
        "record_id": record_id,
        "scholar_id": scholar_id,
        "attempt_ref": "",
        "node_code": node_code or f"code_{record_id}",
        "knowledge_point_name": kp,
        "primary_error": error_type,
        "source": source,
        "created_at": created_at,
    }
    doc.update(overrides)
    fake_db.add("error_record", doc)
    return doc


class TestMathErrorStats:
    def test_missing_scholar_id_returns_400(self, make_client, fake_db):
        client = make_client(math_router)
        res = client.get("/math/error-stats")
        assert res.status_code == 400, res.text
        assert "scholar_id" in res.json()["detail"]

    def test_empty_scholar_returns_empty_list(self, make_client, fake_db):
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s_ghost")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["items"] == []
        assert data["total"] == 0

    def test_returns_records_sorted_desc_with_mapping(self, make_client, fake_db):
        _seed_error_record(fake_db, record_id="er_1", kp="进位加法",
                           error_type="computation", created_at=1000)
        _seed_error_record(fake_db, record_id="er_2", kp="分数除法",
                           error_type="concept", created_at=3000)
        _seed_error_record(fake_db, record_id="er_3", kp="小数乘法",
                           error_type="method", created_at=2000)

        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["total"] == 3
        ids = [it["error_record_id"] for it in data["items"]]
        assert ids == ["er_2", "er_3", "er_1"]  # created_at 倒序
        first = data["items"][0]
        assert first["knowledge_point_name"] == "分数除法"
        assert first["error_type"] == "concept"
        assert first["source"] == "auto_scan"
        assert first["created_at"] == 3000

    def test_filters_by_knowledge_point_name(self, make_client, fake_db):
        _seed_error_record(fake_db, record_id="er_1", kp="进位加法")
        _seed_error_record(fake_db, record_id="er_2", kp="分数除法")

        client = make_client(math_router)
        res = client.get(
            "/math/error-stats?scholar_id=s1&knowledge_point_name=%E5%88%86%E6%95%B0%E9%99%A4%E6%B3%95"
        )
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["total"] == 1
        assert data["items"][0]["error_record_id"] == "er_2"

    def test_total_ignores_limit_truncation(self, make_client, fake_db):
        # 5 条记录，limit=2：items 只回 2 条，但 total=5
        for i in range(5):
            _seed_error_record(fake_db, record_id=f"er_{i}",
                               created_at=(5 - i) * 1000)

        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1&limit=2")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_legacy_record_without_kp_name_falls_back_to_node_code(
        self, make_client, fake_db
    ):
        # 存量记录无 knowledge_point_name 字段 → 回退 node_code
        _seed_error_record(fake_db, record_id="er_old", kp="", node_code="nc_001")

        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["knowledge_point_name"] == "nc_001"

    def test_b2_passthrough_fields_echoed(self, make_client, fake_db):
        # B1/B1.5 已随 error_record 落库题干/链锚点/巩固证据 → B2 逐字透传
        _seed_error_record(
            fake_db,
            record_id="er_full",
            kp="分数除法",
            error_type="method",
            question_text="把 3/4 米平均分成 3 份，每份是多少米？",
            occurrence=2,
            textbook_id="TB-A",
            grade="五年级",
            semester="up",
            unit_title="第3单元 分数",
            lesson_title="课时3 分数除法",
            node_title="分数除法",
            drill_stats={"drill_count": 1, "pass_count": 1},
            last_drill_result={"correct": True, "at": 5000},
        )

        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["question_text"] == "把 3/4 米平均分成 3 份，每份是多少米？"
        assert item["occurrence"] == 2
        assert item["textbook_id"] == "TB-A"
        assert item["grade"] == "五年级"
        assert item["semester"] == "up"
        assert item["unit_title"] == "第3单元 分数"
        assert item["lesson_title"] == "课时3 分数除法"
        assert item["node_title"] == "分数除法"
        assert item["node_code"] == "code_er_full"
        assert item["drill_stats"] == {"drill_count": 1, "pass_count": 1}
        assert item["last_drill_result"] == {"correct": True, "at": 5000}

    def test_b2_defaults_for_legacy_records(self, make_client, fake_db):
        # 存量记录无 B1 扩展字段 → 空串/0/{} 兜底（前端 normalize 零改动）
        _seed_error_record(fake_db, record_id="er_legacy")

        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["question_text"] == ""
        assert item["occurrence"] == 0
        assert item["textbook_id"] == ""
        assert item["grade"] == ""
        assert item["semester"] == ""
        assert item["unit_title"] == ""
        assert item["lesson_title"] == ""
        assert item["node_title"] == ""
        assert item["node_code"] == "code_er_legacy"
        assert item["drill_stats"] == {}
        assert item["last_drill_result"] == {}
        # M11：无正式教材链 → skill_progress 缺省空数组（存量兼容）
        assert res.json()["data"]["skill_progress"] == []


class TestMathErrorStatsSkillProgress:
    """M11 ②：error-stats 出参新增 skill_progress（学者维度技能条统计）"""

    def test_skill_progress_empty_when_no_unit_summary(self, make_client, fake_db):
        """学者有错题但教材单元尚无 display_groups 总结 → 缺省空数组"""
        _seed_error_record(
            fake_db, record_id="er_u1",
            textbook_id="TB-3A", unit_title="第1单元 万以内的加法和减法",
            kp="整数加法",
        )
        # 教材链上存在 unit 节点，但 ai_summary 是老总结（无 display_groups）
        fake_db.add("curriculum_node", {
            "node_id": "unit_a",
            "node_type": "unit",
            "textbook_id": "TB-3A",
            "unit_title": "第1单元 万以内的加法和减法",
            "ai_summary": {"status": "success", "knowledge_points": [],
                           "extended_points": [], "idempotency_key": "k_old"},
        })
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["total"] == 1
        assert data["skill_progress"] == []

    def test_skill_progress_computed_from_unit_display_groups(self, make_client, fake_db):
        """display_groups 已生成 → 按 kp 名映射聚合，weakness 综合信号正确"""
        _seed_error_record(
            fake_db, record_id="er_1",
            textbook_id="TB-3A", unit_title="第1单元 万以内的加法和减法",
            kp="整数加法", error_type="computation", occurrence=2,
        )
        _seed_error_record(
            fake_db, record_id="er_2",
            textbook_id="TB-3A", unit_title="第1单元 万以内的加法和减法",
            kp="生活应用", error_type="method",
        )
        fake_db.add("curriculum_node", {
            "node_id": "unit_a",
            "node_type": "unit",
            "textbook_id": "TB-3A",
            "unit_title": "第1单元 万以内的加法和减法",
            "ai_summary": {
                "status": "success",
                "knowledge_points": [],
                "extended_points": [],
                "display_groups": [
                    {"display_group": "进位加法",
                     "kp_names": ["整数加法", "整数减法"],
                     "kp_ids": ["kp_jf", "kp_jt"], "count": 2},
                    {"display_group": "应用与建模",
                     "kp_names": ["生活应用"],
                     "kp_ids": ["kp_yy"], "count": 1},
                    {"display_group": "逻辑推理",
                     "kp_names": ["图形推理"],
                     "kp_ids": ["kp_tl"], "count": 1},
                ],
            },
        })
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["total"] == 2
        progress = data["skill_progress"]
        # 逻辑推理无错误证据 → 不入列；保 display_groups 顺序
        assert [it["display_group"] for it in progress] == ["进位加法", "应用与建模"]
        first = progress[0]
        assert first["kp_ids"] == ["kp_jf", "kp_jt"]
        assert first["mastery_avg"] == 0.5
        assert first["weakness_signal"] is True   # 重复错 occ=2
        second = progress[1]
        assert second["kp_ids"] == ["kp_yy"]
        assert second["mastery_avg"] == 0.5
        assert second["weakness_signal"] is False  # 单条首错非薄弱


class TestMathErrorStatsExamSource:
    """M13 双源透传：error-stats 出参 kp_source / exam_backlink_to
    （data-model §4.12.9(b)；缺省按数据事实兜底，存量记录零回填可识别）"""

    def test_legacy_extra_ai_record_defaults_to_exam_paper(self, make_client, fake_db):
        """存量 EXTRA_AI 记录（无 kp_source 字段）→ 读取层兜底 exam_paper；
        无回链 → exam_backlink_to=None"""
        _seed_error_record(
            fake_db, record_id="er_extra_legacy",
            kp="鸡兔同笼", textbook_id="EXTRA_AI",
        )
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["kp_source"] == "exam_paper"
        assert item["exam_backlink_to"] is None

    def test_formal_and_unlinked_records_default_to_textbook(self, make_client, fake_db):
        """正式教材链 / 无链记录（无 kp_source 字段）→ 兜底 textbook（教材锚）"""
        _seed_error_record(
            fake_db, record_id="er_formal",
            kp="进位加法", textbook_id="TB-3A",
        )
        _seed_error_record(fake_db, record_id="er_unlinked", kp="分数除法")
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        items = {it["error_record_id"]: it for it in res.json()["data"]["items"]}
        assert items["er_formal"]["kp_source"] == "textbook"
        assert items["er_unlinked"]["kp_source"] == "textbook"
        assert items["er_formal"]["exam_backlink_to"] is None
        assert items["er_unlinked"]["exam_backlink_to"] is None

    def test_explicit_fields_echoed_verbatim(self, make_client, fake_db):
        """M13 新版落库字段（kp_source='exam_paper' + exam_backlink_to）→ 原样透传"""
        backlink = {
            "node_code": "ua1",
            "kp_name": "小数乘整数",
            "textbook_id": "tb-multi",
            "unit_title": "第3单元 小数乘法",
            "lesson_title": "课时1 小数乘整数",
        }
        _seed_error_record(
            fake_db, record_id="er_exam_new",
            kp="鸡兔同笼", textbook_id="EXTRA_AI",
            kp_source="exam_paper", exam_backlink_to=backlink,
        )
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["kp_source"] == "exam_paper"
        assert item["exam_backlink_to"] == backlink

    def test_explicit_textbook_source_not_overridden(self, make_client, fake_db):
        """防御：正式记录显式 kp_source='textbook' 不被推导覆盖（显式字段优先）"""
        _seed_error_record(
            fake_db, record_id="er_tb_explicit",
            kp="加法运算", textbook_id="TB-3A", kp_source="textbook",
        )
        client = make_client(math_router)
        res = client.get("/math/error-stats?scholar_id=s1")
        assert res.status_code == 200, res.text
        item = res.json()["data"]["items"][0]
        assert item["kp_source"] == "textbook"
        assert item["exam_backlink_to"] is None
