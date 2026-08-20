"""G1.2 验收集成测试：批量总结 3 接口
覆盖 SOP §5 G1.2 规格来源（契约 §3.10 API-3：batch-generate / batch-status / manual-edit 3 接口）：
1. POST /math/knowledge-summary/batch-generate   批量总结调度（scope 3 选 + node_ids）
2. GET  /math/knowledge-summary/batch-status     批任务状态（running/done/failed + progress + items）
3. POST /math/curriculum-node/{id}/manual-edit-summary   人工修正知识点（manual_edited* 3 字段）

统一断言模式：成功 → 200 + success:true + data:{...}；失败 → 404/400 对应契约错误码；
写审计 action = G0.2 新增 2 类：generate_knowledge_summary（批完成 1 次）+ manual_edit_summary（修正 1 次）。
"""
from __future__ import annotations

import pytest

from services.math import knowledge_summary as ks_mod
from services.math import (
    SUBJECT_TYPE_MATH,
    SUMMARY_STATUS_SUCCESS,
)
from services.routes_math import router as math_router


TB_ID_01 = "tb_math_g7_up"


def _seed_textbook(fake_db, **overrides) -> dict:
    doc = {
        "textbook_id": TB_ID_01,
        "title": "人教版数学七年级上册",
        "subject_type": SUBJECT_TYPE_MATH,
        "grade": "七年级",
        "semester": "up",
        "publisher": "人民教育出版社",
        "chapters": [],
        "created_at": 1_750_000_000_000,
        "updated_at": 1_750_000_000_000,
    }
    doc.update(overrides)
    fake_db.add("textbook_v2", doc)
    return doc


def _seed_nodes_batch_fixture(fake_db):
    """5 节点夹具（exact 10 断言场景：5 success+blocked+has_desc）。

    n1: status=success, has_desc → not_generated_only: skip; force: include
    n2: status=success, has_desc → not_generated_only: skip; force: include
    n3: status=<none>,  has_desc → 生成 success
    n4: status=<none>,  has_desc → 生成 success
    n5: status=<none>,  NO desc  →  blocked（NoDescriptionError → failed/blocked）
    """
    rows = [
        {"node_id": "n1", "node_type": "lesson", "code": "L1", "title": "课1",
         "description": {"summary": "描述1"},
         "ai_summary": {"status": SUMMARY_STATUS_SUCCESS, "knowledge_points": [{"point_id": "p1"}]}},
        {"node_id": "n2", "node_type": "lesson", "code": "L2", "title": "课2",
         "description": {"summary": "描述2"},
         "ai_summary": {"status": SUMMARY_STATUS_SUCCESS, "knowledge_points": [{"point_id": "p2"}]}},
        {"node_id": "n3", "node_type": "lesson", "code": "L3", "title": "课3",
         "description": {"summary": "描述3"}},
        {"node_id": "n4", "node_type": "lesson", "code": "L4", "title": "课4",
         "description": {"summary": "描述4"}},
        {"node_id": "n5", "node_type": "lesson", "code": "L5", "title": "课5",
         "description": None},
    ]
    base = {"textbook_id": TB_ID_01, "grade": "七年级", "semester": "up", "parent_code": "U7-1"}
    for r in rows:
        fake_db.add("curriculum_node", dict(base, **r))
    return rows


def _audit_actions(fake_db):
    return [r.get("action") for r in fake_db.all("audit_log")]


# ===========================================================================
# A) batch-generate 3 tests
# ===========================================================================

class TestBatchGenerate:
    def _install_sync_mode(self, monkeypatch):
        """同步模式：调度直接 await 执行，不创建后台 task（测试确定性）。"""
        monkeypatch.setattr(ks_mod, "_BATCH_RUN_SYNC", True)

    def _install_mock_generator(self, monkeypatch, *, no_desc_nodes=("n5",)):
        """mock generateKnowledgeSummary：node in no_desc_nodes → raise NoDescriptionError；否则返回 success。"""
        from services.math.knowledge_summary import NoDescriptionError

        async def _mock(db, *, curriculum_node_id, force_regenerate=False, include_extended_points=True):
            if curriculum_node_id in no_desc_nodes:
                raise NoDescriptionError(f"node {curriculum_node_id} 无描述")
            return {
                "summary_id": curriculum_node_id,
                "status": SUMMARY_STATUS_SUCCESS,
                "idempotency_key": "mock_key",
                "knowledge_points": [{"point_id": f"kp_{curriculum_node_id}"}],
                "generated_at": 1_000_000_002_000,
            }

        monkeypatch.setattr(ks_mod, "generateKnowledgeSummary", _mock)

    def test_batch_generate_scope_not_generated_only_returns_stats_2success_1blocked(self, make_client, fake_db, monkeypatch):
        """验收 A-1: not_generated_only → 跳过 2 个已 success(n1,n2) + 有描述生成 2 个(n3,n4)+ 1 blocked(n5) → stats success=2 blocked=1。"""
        _seed_textbook(fake_db)
        _seed_nodes_batch_fixture(fake_db)
        self._install_sync_mode(monkeypatch)
        self._install_mock_generator(monkeypatch, no_desc_nodes=("n5",))
        client = make_client(math_router)
        resp = client.post("/math/knowledge-summary/batch-generate", json={
            "textbook_id": TB_ID_01, "scope": "not_generated_only",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert "job_id" in data and data["job_id"].startswith("batch_")
        # 同步模式：status 直接 = done（生产异步会是 running）
        assert data["status"] == "done", f"同步模式下批任务应已完成，收到 {data!r}"
        stats = data["stats"]
        assert stats["success"] == 2, f"n3/n4 生成，收到 stats={stats!r}"
        assert stats["blocked_no_desc"] == 1, f"n5 无描述，收到 stats={stats!r}"
        assert stats["skipped_existing"] == 2, f"n1/n2 已 success 被 skip，收到 stats={stats!r}"
        assert stats["failed"] == 0
        assert stats["total"] == 5
        # 审计：批完成后写 1 条 generate_knowledge_summary，context 有 scope/textbook_id
        acts = _audit_actions(fake_db)
        assert "generate_knowledge_summary" in acts
        audit_log = next(r for r in fake_db.all("audit_log") if r["action"] == "generate_knowledge_summary")
        ctx = audit_log.get("context") or {}
        assert ctx.get("scope") == "not_generated_only"
        assert ctx.get("textbook_id") == TB_ID_01
        assert ctx.get("batch_job_id") == data["job_id"]

    def test_batch_generate_scope_force_increments_success_to_4_and_reset_existing(self, make_client, fake_db, monkeypatch):
        """验收 A-2: force → 即使 n1/n2 已有 success 也重生成 → success=4(n1..n4) + blocked=1(n5)。"""
        _seed_textbook(fake_db)
        _seed_nodes_batch_fixture(fake_db)
        self._install_sync_mode(monkeypatch)
        self._install_mock_generator(monkeypatch, no_desc_nodes=("n5",))
        client = make_client(math_router)
        resp = client.post("/math/knowledge-summary/batch-generate", json={
            "textbook_id": TB_ID_01, "scope": "force",
        })
        assert resp.status_code == 200, resp.text
        stats = resp.json()["data"]["stats"]
        assert stats["success"] == 4, f"force 下 n1~n4 都重生成，收到 {stats!r}"
        assert stats["skipped_existing"] == 0
        assert stats["blocked_no_desc"] == 1
        assert stats["total"] == 5

    def test_batch_generate_missing_textbook_returns_404_textbook_not_found(self, make_client, fake_db, monkeypatch):
        """验收 A-3: textbook 不存在 → 404 + code=TEXTBOOK_NOT_FOUND。"""
        self._install_sync_mode(monkeypatch)
        self._install_mock_generator(monkeypatch)
        client = make_client(math_router)
        resp = client.post("/math/knowledge-summary/batch-generate", json={
            "textbook_id": "DOES_NOT_EXIST", "scope": "all",
        })
        assert resp.status_code == 404, resp.text
        err = resp.json()["detail"]
        assert err["code"] == "TEXTBOOK_NOT_FOUND", f"期望 404 code，收到 {err!r}"

    def test_batch_generate_invalid_scope_returns_400(self, make_client, fake_db, monkeypatch):
        """验收 A-4: scope=bad_value → Pydantic 校验 422/400 或业务 400。"""
        _seed_textbook(fake_db)
        self._install_sync_mode(monkeypatch)
        self._install_mock_generator(monkeypatch)
        client = make_client(math_router)
        resp = client.post("/math/knowledge-summary/batch-generate", json={
            "textbook_id": TB_ID_01, "scope": "invalid_scope",
        })
        # Pydantic Literal 校验失败是 422；若我们自定义业务异常则是 400；两者任一视为合理
        assert resp.status_code in (400, 422), resp.text


# ===========================================================================
# B) batch-status 2 tests
# ===========================================================================

class TestBatchStatus:
    def test_batch_status_job_not_found_returns_404(self, make_client, fake_db):
        """验收 B-1: job_id 不存在 → 404 code=BATCH_JOB_NOT_FOUND。"""
        client = make_client(math_router)
        resp = client.get("/math/knowledge-summary/batch-status", params={"job_id": "batch_NOT_EXIST"})
        assert resp.status_code == 404, resp.text
        err = resp.json()["detail"]
        assert err["code"] == "BATCH_JOB_NOT_FOUND", f"收到 {err!r}"

    def test_batch_status_done_contains_progress_100_and_items_with_statuses(self, make_client, fake_db, monkeypatch):
        """验收 B-2: generate 完成 → GET status → progress_pct=100 + items 5 条各含 node_id/status/error?。"""
        _seed_textbook(fake_db)
        _seed_nodes_batch_fixture(fake_db)
        from services.math.knowledge_summary import NoDescriptionError
        monkeypatch.setattr(ks_mod, "_BATCH_RUN_SYNC", True)

        async def _mock(db, *, curriculum_node_id, force_regenerate=False, include_extended_points=True):
            if curriculum_node_id == "n5":
                raise NoDescriptionError("n5 no desc")
            return {"summary_id": curriculum_node_id, "status": SUMMARY_STATUS_SUCCESS,
                    "idempotency_key": "k", "knowledge_points": [], "generated_at": 1}

        monkeypatch.setattr(ks_mod, "generateKnowledgeSummary", _mock)
        client = make_client(math_router)
        job_id = client.post("/math/knowledge-summary/batch-generate", json={
            "textbook_id": TB_ID_01, "scope": "force"}).json()["data"]["job_id"]
        resp = client.get("/math/knowledge-summary/batch-status", params={"job_id": job_id})
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["status"] == "done"
        assert body["progress_pct"] == 100
        assert body["total"] == 5
        assert len(body["items"]) == 5
        by_id = {it["node_id"]: it for it in body["items"]}
        assert by_id["n1"]["status"] == "success"
        assert by_id["n2"]["status"] == "success"
        assert by_id["n3"]["status"] == "success"
        assert by_id["n4"]["status"] == "success"
        assert by_id["n5"]["status"] == "blocked_no_desc"
        assert "error" in by_id["n5"] and by_id["n5"]["error"]


# ===========================================================================
# C) manual-edit-summary 3 tests
# ===========================================================================

class TestManualEditSummary:
    def test_manual_edit_writes_knowledge_points_and_manual_edited_3_fields(self, make_client, fake_db):
        """验收 C-1: 修正知识点 → manual_edited=true + manual_edited_at + manual_edited_by + changed_fields_count。"""
        _seed_nodes_4 = [
            {"node_id": "m1", "node_type": "lesson", "code": "M1", "title": "课M1",
             "textbook_id": TB_ID_01, "grade": "七年级", "semester": "up",
             "description": {"summary": "M1描述"},
             "ai_summary": {"status": SUMMARY_STATUS_SUCCESS,
                            "knowledge_points": [{"point_id": "old_p1", "title": "旧P1"}],
                            "extended_points": [{"point_id": "ex_old"}]}},
        ]
        for r in _seed_nodes_4:
            fake_db.add("curriculum_node", r)
        client = make_client(math_router)
        resp = client.post("/math/curriculum-node/m1/manual-edit-summary", json={
            "knowledge_points": [{"point_id": "p1_new", "title": "新P1"}],
            "extended_points": None,
            "overwrite_ai": False,
            "editor_id": "editor_001",
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()["data"]
        assert body["node_id"] == "m1"
        assert body["manual_edited"] is True
        assert body["manual_edited_by"] == "editor_001"
        assert body["manual_edited_at"] > 0
        assert body["changed_points_count"] == 1
        # 落库验证：ai_summary.manual_edited=true 且 manual_* 三字段在 ai_summary 下（契约 DM-3）
        stored = next(r for r in fake_db.all("curriculum_node") if r["node_id"] == "m1")
        ai = stored["ai_summary"] or {}
        assert ai.get("manual_edited") is True
        assert ai.get("manual_edited_by") == "editor_001"
        assert ai.get("manual_edited_at") > 0
        assert ai["knowledge_points"][0]["point_id"] == "p1_new"
        # extended_points=None → 保留原有
        assert ai["extended_points"] == [{"point_id": "ex_old"}]
        # 审计：manual_edit_summary action + changed_points_count context
        acts = _audit_actions(fake_db)
        assert "manual_edit_summary" in acts
        audit = next(r for r in fake_db.all("audit_log") if r["action"] == "manual_edit_summary")
        assert (audit.get("context") or {}).get("changed_points_count") == 1

    def test_manual_edit_missing_node_returns_404_node_not_found(self, make_client, fake_db):
        """验收 C-2: 修正不存在节点 → 404 code=NODE_NOT_FOUND。"""
        client = make_client(math_router)
        resp = client.post("/math/curriculum-node/DOES_NOT_EXIST/manual-edit-summary", json={
            "knowledge_points": [{"point_id": "p"}],
            "editor_id": "e1",
        })
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"]["code"] == "NODE_NOT_FOUND"

    def test_manual_edit_default_editor_id_falls_back_to_request_openid(self, make_client, fake_db):
        """验收 C-3: payload 未传 editor_id → 使用 actor=openid(anonymous) 为 manual_edited_by。"""
        fake_db.add("curriculum_node", {
            "node_id": "m2", "node_type": "lesson", "code": "M2", "title": "课M2",
            "textbook_id": TB_ID_01, "grade": "七年级", "semester": "up",
            "description": {"summary": "M2描述"},
            "ai_summary": {"status": SUMMARY_STATUS_SUCCESS,
                           "knowledge_points": [], "extended_points": []},
        })
        client = make_client(math_router)
        resp = client.post("/math/curriculum-node/m2/manual-edit-summary", json={
            "knowledge_points": [{"point_id": "pA"}],
            # 未传 editor_id → default anonymous
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["manual_edited_by"] == "anonymous"
