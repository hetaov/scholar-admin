"""F3.3 练习纸详情与选题接口验收测试（契约 api-contract.md §3.10）

验收场景：
- GET /math/practice-sheet/{sheet_id}：
  - 已渲染完成 → 200 返回 _to_public_sheet 结构（含 file_refs.pdf）+ render_status=success；
  - 渲染中（file_refs 为空）→ render_status=rendering，轮询语义成立；
  - 渲染失败 → render_status=failed，前端可据此提示重试；
  - 练习纸不存在 → 404。
- GET /math/knowledge-summary/list：
  - 仅返回含 ai_summary 的节点摘要（knowledge_points/extended_points 清单），
    不含 ai_summary 的节点被过滤；透出字段不含完整 ai_summary 文本。
"""
from __future__ import annotations

import pytest

from services.routes_math import router as math_router

SHEET_ID = "sheet_f3_test_001"


def _seed_sheet(fake_db, *, status="generated", file_refs=None, **overrides) -> dict:
    doc = {
        "sheet_id": SHEET_ID,
        "status": status,
        "source": "ai_knowledge",
        "template_ref": {"template_id": "standard", "version": 1},
        "nodes": [{"node_id": "n_unit_1", "title": "有理数"}],
        "primary_errors": [],
        "difficulty_bands": [],
        "items": [
            {
                "item_id": "q1",
                "question": "填空：-3 的相反数是___",
                "node_code": "n_unit_1",
                "target_error": "",
                "variant_level": 1,
                "difficulty": "基础",
                "source_kp": "有理数的定义",
            }
        ],
        "qrcode_ref": None,
        "file_refs": file_refs,
    }
    doc.update(overrides)
    fake_db.add("practice_sheet", doc)
    return doc


def _seed_render_job(fake_db, *, status="success", **overrides) -> dict:
    doc = {
        "job_id": "job_f3_test_001",
        "sheet_id": SHEET_ID,
        "status": status,
        "created_at": "2026-08-19T10:00:00.000Z",
    }
    doc.update(overrides)
    fake_db.add("sheet_render_job", doc)
    return doc


def _seed_summary_node(fake_db, **overrides) -> dict:
    doc = {
        "node_id": "n_unit_1",
        "code": "u1",
        "title": "有理数",
        "grade": "七年级",
        "semester": "上",
        "unit_title": "有理数",
        "lesson_title": "",
        "textbook_id": "tb_math_7a",
        "description_version": 1,
        "ai_summary": {
            "knowledge_points": [
                {
                    "name": "有理数的定义",
                    "summary": "正数与负数统称有理数",
                    "ability_dimensions": ["arithmetic"],
                    "source_node_id": "n_unit_1",
                    "source_lesson_id": "",
                }
            ],
            "extended_points": [
                {
                    "name": "数轴上的有理数",
                    "summary": "在数轴上表示有理数",
                    "difficulty_band": "入门",
                    "related_knowledge_name": "有理数的定义",
                    "source_lesson_id": "",
                }
            ],
        },
    }
    doc.update(overrides)
    fake_db.add("curriculum_node", doc)
    return doc


class TestGetPracticeSheetDetail:
    def test_detail_returns_public_sheet_with_render_status(
        self, make_client, fake_db
    ):
        """已渲染完成：GET 返回公开结构 + file_refs.pdf + render_status=success"""
        _seed_sheet(
            fake_db,
            file_refs={
                "pdf": "https://static.example.com/sheet_f3_test_001/sheet.pdf",
                "png": "https://static.example.com/sheet_f3_test_001/preview.png",
            },
        )
        _seed_render_job(fake_db, status="success")

        client = make_client(math_router)
        res = client.get(f"/math/practice-sheet/{SHEET_ID}")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["sheet_id"] == SHEET_ID
        assert data["status"] == "generated"
        assert data["file_refs"]["pdf"].endswith(".pdf")
        assert data["render_status"] == "success"
        # 契约：出参不含 answer / hint_card
        assert "answer" not in data["items"][0]

    def test_detail_rendering_keeps_empty_file_refs(self, make_client, fake_db):
        """渲染中：file_refs 为空 + render_status=rendering（轮询等待语义）"""
        _seed_sheet(fake_db, file_refs=None)
        _seed_render_job(fake_db, status="rendering")

        client = make_client(math_router)
        res = client.get(f"/math/practice-sheet/{SHEET_ID}")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert data["file_refs"] is None
        assert data["render_status"] == "rendering"

    def test_detail_failed_render_reports_failed(self, make_client, fake_db):
        """渲染失败：render_status=failed，前端可提示重试"""
        _seed_sheet(fake_db, file_refs=None)
        _seed_render_job(fake_db, status="failed")

        client = make_client(math_router)
        res = client.get(f"/math/practice-sheet/{SHEET_ID}")
        assert res.status_code == 200, res.text
        assert res.json()["data"]["render_status"] == "failed"

    def test_detail_not_found_returns_404(self, make_client, fake_db):
        """练习纸不存在 → 404"""
        client = make_client(math_router)
        res = client.get("/math/practice-sheet/no_such_sheet")
        assert res.status_code == 404, res.text


class TestListKnowledgeSummaries:
    def test_list_returns_only_summarized_nodes(self, make_client, fake_db):
        """仅返回含 ai_summary 的节点，透出知识点/拓展点清单"""
        _seed_summary_node(fake_db)
        fake_db.add(
            "curriculum_node",
            {
                "node_id": "n_unit_2",
                "code": "u2",
                "title": "整式",
                "grade": "七年级",
                "semester": "上",
                "unit_title": "整式",
                "lesson_title": "",
                "textbook_id": "tb_math_7a",
                "description_version": 1,
                # 无 ai_summary：应被过滤
            },
        )

        client = make_client(math_router)
        res = client.get("/math/knowledge-summary/list")
        assert res.status_code == 200, res.text
        data = res.json()["data"]
        assert len(data) == 1
        node = data[0]
        assert node["node_id"] == "n_unit_1"
        assert node["knowledge_points"][0]["name"] == "有理数的定义"
        assert node["extended_points"][0]["name"] == "数轴上的有理数"
        # 知识点条目保留展示字段（name/summary/ability_dimensions），
        # 但节点顶层不透出完整 ai_summary（避免传输描述等大文本）
        assert node["knowledge_points"][0]["summary"] == "正数与负数统称有理数"
        assert "ai_summary" not in node

    def test_list_empty_when_no_summaries(self, make_client, fake_db):
        """无任何已总结节点 → 空数组"""
        client = make_client(math_router)
        res = client.get("/math/knowledge-summary/list")
        assert res.status_code == 200, res.text
        assert res.json()["data"] == []

    def test_list_route_not_shadowed_by_dynamic_route(self, make_client, fake_db):
        """静态路由 /list 不被 /{curriculum_node_id} 吞掉：返回列表而非 404"""
        client = make_client(math_router)
        res = client.get("/math/knowledge-summary/list")
        assert res.status_code == 200, res.text
        assert isinstance(res.json()["data"], list)
