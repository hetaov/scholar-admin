"""F4.6 学者错题列表接口验收测试（契约 api-contract.md §3.10 接口 25）

验收场景：
- GET /math/error-stats?scholar_id=xxx：
  - 无 scholar_id → 400；
  - 空学者 → { items: [], total: 0 }；
  - 多条记录 → 按 created_at 倒序，字段映射 record_id→error_record_id、
    primary_error→error_type，透传 knowledge_point_name/source/created_at；
  - knowledge_point_name 过滤（可选）；
  - total 不受 limit 截断影响。
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
