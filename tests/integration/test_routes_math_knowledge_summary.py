"""F1.2 知识总结路由与审计验收测试（契约 api-contract.md §3.10）

验收标准（任务清单 F1.2）：
- POST /math/knowledge-summary/generate 生成成功 → GET /math/knowledge-summary/{id} 可取回完整总结；
- 无描述节点 POST → 400 业务错误（缺 curriculum_node_id → 422）；
- audit_log 落 action=generate_knowledge_summary（成功/失败均写）；
- 幂等命中不重复调用 LLM。

说明：F1 路由挂载于 services.routes_math.math_router（main.py 付费组），
本测试用 make_client 直接注入 FakeDB，绕过付费鉴权只验接口链路。
"""
from __future__ import annotations

import pytest

from services.math.knowledge_summary import LLMResponseError
from services.routes_math import router as math_router

NODE_ID = "n_unit_1"
AUDIT_ACTION = "generate_knowledge_summary"


def _seed_node(fake_db, **overrides) -> dict:
    doc = {
        "node_id": NODE_ID,
        "node_type": "unit",
        "title": "有理数",
        "textbook_id": "tb_math_7a",
        "grade": "七年级",
        "semester": "上",
        "unit_id": "U1",
        "unit_title": "有理数",
        "lesson_id": "",
        "lesson_title": "",
        "description_version": 1,
        "description": {
            "summary": "认识有理数，理解数轴与相反数的概念。",
            "key_points": ["正负数与零的区分", "数轴三要素"],
            "typical_examples": [],
            "prerequisites": [],
            "teaching_tips": [],
        },
    }
    doc.update(overrides)
    fake_db.add("curriculum_node", doc)
    return doc


def _fake_summary_result() -> dict:
    return {
        "knowledge_points": [
            {
                "name": "有理数的定义",
                "summary": "正数与负数统称有理数",
                "ability_dimensions": ["arithmetic"],
                "source_node_id": NODE_ID,
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
    }


def _audit_rows(fake_db) -> list[dict]:
    return [row for row in fake_db.all("audit_log")]


class TestGenerateKnowledgeSummary:
    def test_generate_then_get_returns_summary_and_audit(
        self, make_client, fake_db, monkeypatch
    ):
        """POST 生成成功 → GET 可取回完整总结；audit_log 落成功记录"""
        _seed_node(fake_db)
        monkeypatch.setattr(
            "services.math.knowledge_summary.LLM_SUMMARY_MODEL", "test-summary-model"
        )

        async def _fake_llm(node, **kwargs):
            return _fake_summary_result()

        monkeypatch.setattr(
            "services.math.knowledge_summary._call_summary_llm", _fake_llm
        )
        client = make_client(math_router)

        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["summary_id"] == NODE_ID
        assert data["status"] == "success"
        assert data["idempotency_key"]
        assert data["knowledge_points"][0]["name"] == "有理数的定义"
        assert data["extended_points"][0]["difficulty_band"] == "入门"
        assert data["generated_at"] > 0

        rows = _audit_rows(fake_db)
        assert any(
            r["action"] == AUDIT_ACTION
            and r["object_ref"] == NODE_ID
            and r["result"] == "success"
            for r in rows
        ), "成功生成必须写 generate_knowledge_summary 成功审计"

        # GET 取回完整总结
        resp = client.get(f"/math/knowledge-summary/{NODE_ID}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["curriculum_node_id"] == NODE_ID
        assert data["status"] == "success"
        assert data["model"] == "test-summary-model"
        assert data["knowledge_points"][0]["name"] == "有理数的定义"
        assert data["extended_points"][0]["name"] == "数轴上的有理数"

    def test_400_when_node_has_no_description(self, make_client, fake_db):
        """无描述不总结 → 400 业务错误（任务卡验收）"""
        _seed_node(fake_db, description=None, description_version=0)
        client = make_client(math_router)
        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp.status_code == 400
        assert "无描述" in resp.json()["detail"]

    def test_400_when_node_type_unsupported(self, make_client, fake_db):
        """节点类型不要求总结 → 400"""
        _seed_node(fake_db, node_type="grade")
        client = make_client(math_router)
        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp.status_code == 400

    def test_404_when_node_missing(self, make_client, fake_db):
        """节点不存在 → 404"""
        client = make_client(math_router)
        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": "n_missing"},
        )
        assert resp.status_code == 404

    def test_422_when_curriculum_node_id_missing(self, make_client, fake_db):
        """缺 curriculum_node_id → 422（Pydantic 校验，任务卡验收）"""
        client = make_client(math_router)
        resp = client.post("/math/knowledge-summary/generate", json={})
        assert resp.status_code == 422

    def test_500_when_llm_fails_and_writes_failed_audit(
        self, make_client, fake_db, monkeypatch
    ):
        """LLM 调用失败 → 500，写回 failed 状态 + failed 审计"""
        _seed_node(fake_db)
        monkeypatch.setattr(
            "services.math.knowledge_summary.LLM_SUMMARY_MODEL", "test-summary-model"
        )

        async def _bad_llm(node, **kwargs):
            raise LLMResponseError("llm boom")

        monkeypatch.setattr(
            "services.math.knowledge_summary._call_summary_llm", _bad_llm
        )
        client = make_client(math_router)

        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp.status_code == 500
        rows = _audit_rows(fake_db)
        assert any(
            r["action"] == AUDIT_ACTION
            and r["object_ref"] == NODE_ID
            and r["result"] == "failed"
            for r in rows
        ), "生成失败必须写 generate_knowledge_summary 失败审计"

        # GET 可看到 failed 状态
        resp = client.get(f"/math/knowledge-summary/{NODE_ID}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "failed"

    def test_idempotent_hit_skips_llm(self, make_client, fake_db, monkeypatch):
        """同参数二次 POST 幂等命中，LLM 仅调用 1 次"""
        _seed_node(fake_db)
        monkeypatch.setattr(
            "services.math.knowledge_summary.LLM_SUMMARY_MODEL", "test-summary-model"
        )
        calls = {"n": 0}

        async def _fake_llm(node, **kwargs):
            calls["n"] += 1
            return _fake_summary_result()

        monkeypatch.setattr(
            "services.math.knowledge_summary._call_summary_llm", _fake_llm
        )
        client = make_client(math_router)

        resp1 = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp1.status_code == 200
        resp2 = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID},
        )
        assert resp2.status_code == 200
        assert calls["n"] == 1, "幂等命中不得重复调用 LLM"
        assert (
            resp2.json()["data"]["idempotency_key"]
            == resp1.json()["data"]["idempotency_key"]
        )


class TestGetKnowledgeSummary:
    def test_not_generated_returns_200_with_status(self, make_client, fake_db):
        """未生成 → 200 + status=not_generated（不报错）"""
        _seed_node(fake_db)
        client = make_client(math_router)
        resp = client.get(f"/math/knowledge-summary/{NODE_ID}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["curriculum_node_id"] == NODE_ID
        assert data["status"] == "not_generated"
        assert data["knowledge_points"] == []

    def test_404_when_node_missing(self, make_client, fake_db):
        """节点不存在 → 404"""
        client = make_client(math_router)
        resp = client.get("/math/knowledge-summary/n_missing")
        assert resp.status_code == 404

    @pytest.mark.parametrize(
        "extra",
        [
            {"include_extended_points": False},
            {"force_regenerate": True},
        ],
    )
    def test_generate_with_optional_params(
        self, make_client, fake_db, monkeypatch, extra
    ):
        """可选参数 force_regenerate / include_extended_points 均可提交"""
        _seed_node(fake_db)
        monkeypatch.setattr(
            "services.math.knowledge_summary.LLM_SUMMARY_MODEL", "test-summary-model"
        )

        async def _fake_llm(node, **kwargs):
            return _fake_summary_result()

        monkeypatch.setattr(
            "services.math.knowledge_summary._call_summary_llm", _fake_llm
        )
        client = make_client(math_router)
        resp = client.post(
            "/math/knowledge-summary/generate",
            json={"curriculum_node_id": NODE_ID, **extra},
        )
        assert resp.status_code == 200
