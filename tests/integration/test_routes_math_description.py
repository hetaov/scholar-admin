"""F2.1 教材描述四接口验收测试

验收标准（任务清单 F2.1 + 契约 api-contract.md §3.10）：
- POST 保存描述 → GET 返回含 version/history；审计 edit_description
- POST draft → 返回草稿且不污染 description；审计 draft_description
- POST adopt → description 更新、source=ai_adopted；审计 adopt_description
- 节点不存在 → 404；描述结构非法 → 400；非描述节点 GET → description=null
"""
from __future__ import annotations

import pytest

from services.routes_math import router as math_router

NODE_ID = "n_unit_1"


def _seed_node(fake_db, **overrides) -> dict:
    doc = {
        "node_id": NODE_ID,
        "node_type": "unit",
        "title": "有理数",
        "code": "U1",
        "grade": "七年级",
        "semester": "上",
        "unit_title": "有理数",
        "lesson_title": "",
        "content_type": "text",
    }
    doc.update(overrides)
    fake_db.add("curriculum_node", doc)
    return doc


def _description() -> dict:
    return {
        "summary": "认识有理数，理解数轴与相反数的概念。",
        "key_points": ["正负数与零的区分", "数轴三要素"],
        "typical_examples": [{"ref": "P5 例1", "note": "数轴上标点"}],
        "prerequisites": ["整数与分数"],
        "teaching_tips": ["用温度计类比数轴"],
    }


def _audit_actions(fake_db) -> list[str]:
    return [row["action"] for row in fake_db.all("audit_log")]


class TestGetDescription:
    def test_returns_null_for_unsupported_node_type(self, make_client, fake_db):
        _seed_node(fake_db, node_type="grade")
        client = make_client(math_router)
        resp = client.get(f"/math/curriculum-node/{NODE_ID}/description")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["node_type"] == "grade"
        assert body["data"]["description"] is None
        assert body["data"]["description_version"] == 0

    def test_returns_empty_for_node_without_description(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)
        resp = client.get(f"/math/curriculum-node/{NODE_ID}/description")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"] is None
        assert data["description_history"] == []

    def test_404_when_node_missing(self, make_client, fake_db):
        client = make_client(math_router)
        resp = client.get("/math/curriculum-node/n_missing/description")
        assert resp.status_code == 404


class TestSaveDescription:
    def test_save_then_get_contains_version_and_history(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)

        # 首次保存 → version=1
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": _description()},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description_version"] == 1
        assert data["description_source"] == "manual"

        # 再次保存 → version=2，history 含 v1 快照
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": {**_description(), "summary": "v2 描述"}},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["description_version"] == 2

        # GET → 当前 v2 + history 含 v1
        resp = client.get(f"/math/curriculum-node/{NODE_ID}/description")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"]["summary"] == "v2 描述"
        assert data["description_version"] == 2
        assert data["description_source"] == "manual"
        assert len(data["description_history"]) == 1
        assert data["description_history"][0]["version"] == 1
        assert (
            data["description_history"][0]["snapshot"]["summary"]
            == "认识有理数，理解数轴与相反数的概念。"
        )

        # 审计：人工编辑记录
        assert "edit_description" in _audit_actions(fake_db)

    def test_400_on_invalid_structure(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": {"summary": 123}},
        )
        assert resp.status_code == 400

    def test_400_on_summary_too_long(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": {"summary": "长" * 801}},
        )
        assert resp.status_code == 400

    def test_400_on_unsupported_node_type(self, make_client, fake_db):
        _seed_node(fake_db, node_type="grade")
        client = make_client(math_router)
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": _description()},
        )
        assert resp.status_code == 400

    def test_404_when_node_missing(self, make_client, fake_db):
        client = make_client(math_router)
        resp = client.post(
            "/math/curriculum-node/n_missing/description",
            json={"description": _description()},
        )
        assert resp.status_code == 404


class TestGenerateDraft:
    def test_returns_draft_without_polluting_description(self, make_client, fake_db, monkeypatch):
        _seed_node(fake_db)

        async def _fake_llm(node):
            return {
                "summary": "AI 生成的描述草稿",
                "key_points": ["AI 要点1", "AI 要点2"],
                "typical_examples": [{"ref": "P6", "note": "AI 注"}],
                "prerequisites": ["AI 先修"],
                "teaching_tips": ["AI 提示"],
            }

        monkeypatch.setattr(
            "services.math.curriculum_description._call_summary_llm", _fake_llm
        )
        client = make_client(math_router)

        resp = client.post(f"/math/curriculum-node/{NODE_ID}/description/draft", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["draft"]["summary"] == "AI 生成的描述草稿"
        assert data["model"]
        assert data["source_versions"][0]["description_version"] == 0

        # 草稿不污染正式 description
        resp = client.get(f"/math/curriculum-node/{NODE_ID}/description")
        data = resp.json()["data"]
        assert data["description"] is None
        assert data["description_version"] == 0

        # 审计：AI 草稿生成记录
        assert "draft_description" in _audit_actions(fake_db)

    def test_400_when_model_not_configured(self, make_client, fake_db, monkeypatch):
        _seed_node(fake_db)
        monkeypatch.setattr(
            "services.math.curriculum_description.LLM_SUMMARY_MODEL", ""
        )
        client = make_client(math_router)
        resp = client.post(f"/math/curriculum-node/{NODE_ID}/description/draft", json={})
        assert resp.status_code == 400

    def test_404_when_node_missing(self, make_client, fake_db):
        client = make_client(math_router)
        resp = client.post(
            "/math/curriculum-node/n_missing/description/draft", json={}
        )
        assert resp.status_code == 404


class TestAdoptDraft:
    def test_adopt_updates_description_source_ai_adopted(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)

        # 先人工保存 v1
        client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": _description()},
        )

        # 采纳 AI 草稿 → v2，source=ai_adopted
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description/adopt",
            json={"description": {**_description(), "summary": "AI 采纳后的描述"}},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description_version"] == 2
        assert data["description_source"] == "ai_adopted"

        # GET → description 更新、source=ai_adopted、history 含 v1
        resp = client.get(f"/math/curriculum-node/{NODE_ID}/description")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["description"]["summary"] == "AI 采纳后的描述"
        assert data["description_source"] == "ai_adopted"
        assert data["description_version"] == 2
        assert len(data["description_history"]) == 1
        assert data["description_history"][0]["source"] == "manual"

        # 审计：草稿采纳记录
        assert "adopt_description" in _audit_actions(fake_db)

    def test_400_on_invalid_structure(self, make_client, fake_db):
        _seed_node(fake_db)
        client = make_client(math_router)
        resp = client.post(
            f"/math/curriculum-node/{NODE_ID}/description/adopt",
            json={"description": {"summary": 123}},
        )
        assert resp.status_code == 400

    def test_404_when_node_missing(self, make_client, fake_db):
        client = make_client(math_router)
        resp = client.post(
            "/math/curriculum-node/n_missing/description/adopt",
            json={"description": _description()},
        )
        assert resp.status_code == 404


class TestAuditRecords:
    def test_audit_log_contains_all_three_actions(self, make_client, fake_db, monkeypatch):
        _seed_node(fake_db)

        async def _fake_llm(node):
            return {
                "summary": "AI 生成的描述草稿",
                "key_points": ["AI 要点1"],
                "typical_examples": [],
                "prerequisites": [],
                "teaching_tips": [],
            }

        monkeypatch.setattr(
            "services.math.curriculum_description._call_summary_llm", _fake_llm
        )
        client = make_client(math_router)

        client.post(
            f"/math/curriculum-node/{NODE_ID}/description",
            json={"description": _description()},
        )
        client.post(f"/math/curriculum-node/{NODE_ID}/description/draft", json={})
        client.post(
            f"/math/curriculum-node/{NODE_ID}/description/adopt",
            json={"description": _description()},
        )

        actions = _audit_actions(fake_db)
        assert actions.count("edit_description") == 1
        assert actions.count("draft_description") == 1
        assert actions.count("adopt_description") == 1
