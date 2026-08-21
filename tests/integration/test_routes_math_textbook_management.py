"""G1.1 验收集成测试：数学教材管理 6 接口

覆盖 SOP §5 G1.1 规格来源（契约 §3.10 API-3：G-API-4/5/6 + CRUD 3 = 6 接口）：
1. GET    /math/textbook                 列表（默认 math，支持 grade/semester/keyword 过滤）
2. POST   /math/textbook                 新增
3. PUT    /math/textbook/{textbook_id}   更新（改 subject_type 需二次确认 confirm_title）
4. GET    /math/textbook/overview        教材概览（6 个 node_stats 字段聚合）
5. POST   /math/textbook/import-nodes    批量导入 curriculum_node（幂等 code，on_duplicate）
6. DELETE /math/textbook/{textbook_id}   清理（仅清 description + ai_summary，返回 cleared）

统一断言模式：成功 → 200 + success:true + data:{...}；失败：400/404/422 对应契约错误码；写审计 action = G0.2 新增 5 类之一。
"""
from __future__ import annotations

import pytest

from services.routes_math import router as math_router


# ===========================================================================
# Test 辅助函数
# ===========================================================================

TB_ID_01 = "tb_math_g7_up"  # 数学 七年级上册


def _seed_textbook(fake_db, **overrides) -> dict:
    """预置 1 条数学教材记录（textbook_v2）。"""
    doc = {
        "textbook_id": TB_ID_01,
        "title": "人教版数学七年级上册",
        "subject_type": "math",
        "grade": "七年级",
        "semester": "up",
        "publisher": "人民教育出版社",
        "cover_url": "",
        "isbn": "978-7-107-00000-1",
        "chapters": [],
        "created_at": 1750000000000,
        "updated_at": 1750000000000,
    }
    doc.update(overrides)
    fake_db.add("textbook_v2", doc)
    return doc


def _seed_three_textbooks(fake_db):
    """3 条数学教材 + 1 条英语，用于 list 过滤测试。"""
    fake_db.add("textbook_v2", {"textbook_id": "t1", "title": "人教七上", "subject_type": "math", "grade": "七年级", "semester": "up"})
    fake_db.add("textbook_v2", {"textbook_id": "t2", "title": "北师大七下", "subject_type": "math", "grade": "七年级", "semester": "down"})
    fake_db.add("textbook_v2", {"textbook_id": "t3", "title": "人教八上", "subject_type": "math", "grade": "八年级", "semester": "up"})
    fake_db.add("textbook_v2", {"textbook_id": "te1", "title": "牛津英语 7A", "subject_type": "english", "grade": "七年级", "semester": "up"})


def _audit_actions(fake_db) -> list[str]:
    return [row["action"] for row in fake_db.all("audit_log")]


# ===========================================================================
# 1. GET /math/textbook — 列表
# ===========================================================================


class TestTextbookList:
    def test_default_only_returns_math_subject_type(self, make_client, fake_db):
        """默认 subject_type=math（不返回英语教材）。"""
        _seed_three_textbooks(fake_db)
        client = make_client(math_router)
        resp = client.get("/math/textbook")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        items = body["data"]["items"]
        ids = sorted([r["textbook_id"] for r in items])
        assert ids == ["t1", "t2", "t3"], f"英语教材 te1 不应出现在数学管理列表里，实得 ids={ids}"
        assert all(r["subject_type"] == "math" for r in items)
        assert body["data"]["total"] == 3

    def test_filter_grade_and_semester(self, make_client, fake_db):
        """grade + semester 组合过滤生效。"""
        _seed_three_textbooks(fake_db)
        client = make_client(math_router)
        resp = client.get("/math/textbook?grade=七年级&semester=down")
        assert resp.status_code == 200, resp.text
        items = resp.json()["data"]["items"]
        assert [r["textbook_id"] for r in items] == ["t2"]

    def test_filter_keyword_matches_title(self, make_client, fake_db):
        """keyword 在 title 中模糊匹配（人教 vs 北师大区分）。"""
        _seed_three_textbooks(fake_db)
        client = make_client(math_router)
        resp = client.get("/math/textbook?keyword=北师大")
        assert resp.status_code == 200
        items = resp.json()["data"]["items"]
        assert [r["textbook_id"] for r in items] == ["t2"]

    def test_empty_when_no_textbooks(self, make_client, fake_db):
        """空集合返回 items=[] total=0（不抛错）。"""
        client = make_client(math_router)
        resp = client.get("/math/textbook")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["items"] == []
        assert body["total"] == 0


# ===========================================================================
# 2. POST /math/textbook — 新增
# ===========================================================================


class TestTextbookCreate:
    BASE_PAYLOAD = {
        "title": "苏科版数学八年级下册",
        "grade": "八年级",
        "semester": "down",
        "subject_type": "math",
        "publisher": "江苏凤凰科学技术出版社",
        "cover_url": "",
        "isbn": "978-7-5537-0000-1",
        "chapters": [],
    }

    def test_create_success_returns_doc_with_textbook_id_and_audit(self, make_client, fake_db):
        """成功：写入 textbbook_v2 + 落审计 create_math_textbook + 返回 HTTP 200 + textbook_id 非空。"""
        client = make_client(math_router)
        resp = client.post("/math/textbook", json=self.BASE_PAYLOAD)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["textbook_id"], "新建必须返回 textbook_id"
        assert data["title"] == self.BASE_PAYLOAD["title"]
        assert data["subject_type"] == "math"
        # DB 写入
        rows = fake_db.all("textbook_v2")
        assert len(rows) == 1
        assert rows[0]["textbook_id"] == data["textbook_id"]
        # 审计
        assert "create_math_textbook" in _audit_actions(fake_db)

    def test_create_defaults_subject_type_to_math_when_omitted(self, make_client, fake_db):
        """不传 subject_type → 默认 math（零误创建英语）。"""
        payload = {k: v for k, v in self.BASE_PAYLOAD.items() if k != "subject_type"}
        client = make_client(math_router)
        resp = client.post("/math/textbook", json=payload)
        assert resp.status_code == 200
        assert resp.json()["data"]["subject_type"] == "math"

    def test_create_missing_semester_returns_400_with_error_code(self, make_client, fake_db):
        """数学缺 semester → HTTP 400 + 错误码 MATH_TEXTBOOK_SEMESTER_REQUIRED。"""
        payload = {k: v for k, v in self.BASE_PAYLOAD.items() if k != "semester"}
        client = make_client(math_router)
        resp = client.post("/math/textbook", json=payload)
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        # 错误码格式：{code, message} 或字符串（只要含约定码即可）
        flat = " ".join(str(v) for v in (detail if isinstance(detail, dict) else {"msg": detail}).values())
        assert "MATH_TEXTBOOK_SEMESTER_REQUIRED" in flat, f"实得 detail={detail!r}"

    def test_create_invalid_semester_returns_400(self, make_client, fake_db):
        """semester 非法值 spring → INVALID_MATH_SEMESTER 400。"""
        payload = dict(self.BASE_PAYLOAD, semester="spring")
        client = make_client(math_router)
        resp = client.post("/math/textbook", json=payload)
        assert resp.status_code == 400
        flat = str(resp.json()["detail"])
        assert "INVALID_MATH_SEMESTER" in flat

    def test_create_rejects_english_subject_type_with_400(self, make_client, fake_db):
        """在 /math/textbook 接口里显式传 subject_type=english → 拒绝（英语走既有 /textbook）。"""
        payload = dict(self.BASE_PAYLOAD, subject_type="english")
        client = make_client(math_router)
        resp = client.post("/math/textbook", json=payload)
        assert resp.status_code == 400, resp.text


# ===========================================================================
# 3. PUT /math/textbook/{id} — 更新（改 subject_type 需二次确认）
# ===========================================================================


class TestTextbookUpdate:
    def test_update_title_and_publisher_success_audit(self, make_client, fake_db):
        """普通字段更新成功 + 审计 update_math_textbook + context 有 changed_fields。"""
        _seed_textbook(fake_db)
        client = make_client(math_router)
        resp = client.put(f"/math/textbook/{TB_ID_01}", json={
            "title": "人教版数学七年级上册（2024 修订）",
            "publisher": "人教社",
            "confirm_title": "人教版数学七年级上册（2024 修订）",  # 变更标题的二次确认
        })
        assert resp.status_code == 200, resp.text
        assert resp.json()["success"] is True
        rows = fake_db.all("textbook_v2")
        assert rows[0]["title"] == "人教版数学七年级上册（2024 修订）"
        assert rows[0]["publisher"] == "人教社"
        assert "update_math_textbook" in _audit_actions(fake_db)

    def test_update_missing_404(self, make_client, fake_db):
        """不存在 ID → 404。"""
        client = make_client(math_router)
        resp = client.put("/math/textbook/tb_not_found", json={"title": "X", "confirm_title": "X"})
        assert resp.status_code == 404

    def test_update_attempt_change_subject_type_without_confirm_returns_400(self, make_client, fake_db):
        """尝试从 math 改成 english（或其他跨学科改 subject_type）需要二次确认；
        未确认 → 400 + 二次确认错误码。"""
        _seed_textbook(fake_db)
        client = make_client(math_router)
        # 仅传 subject_type=english，confirm_title 为空字符串 → 触发二次确认拒绝
        resp = client.put(f"/math/textbook/{TB_ID_01}", json={
            "subject_type": "english",
            "confirm_title": "",
        })
        assert resp.status_code == 400, resp.text
        flat = str(resp.json()["detail"])
        # 契约期望 SECONDARY_CONFIRMATION_REQUIRED / SUBJECT_TYPE_CHANGE_CONFIRM_REQUIRED 任一
        assert "CONFIRM" in flat.upper(), f"二次确认失败需带 CONFIRM* 错误码，实得 {flat!r}"


# ===========================================================================
# 4. GET /math/textbook/overview — 概览（6 node_stats 字段聚合）
# ===========================================================================


class TestTextbookOverview:
    def _seed(self, fake_db):
        _seed_textbook(fake_db)
        # 3 个 unit + 2 个 lesson + 4 个 knowledge_point，部分描述/总结已完成
        base = {"textbook_id": TB_ID_01, "grade": "七年级", "semester": "up"}
        fake_db.add("curriculum_node", dict(base, node_id="u1", node_type="unit", title="U1", description="d", ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="u2", node_type="unit", title="U2", description=None, ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="u3", node_type="unit", title="U3", description="d", ai_summary={"status": "success"}))
        fake_db.add("curriculum_node", dict(base, node_id="l1", node_type="lesson", title="L1", description="d", ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="l2", node_type="lesson", title="L2", description=None, ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="k1", node_type="knowledge_point", title="K1", description="d", ai_summary={"status": "success", "manual_edited": False}))
        fake_db.add("curriculum_node", dict(base, node_id="k2", node_type="knowledge_point", title="K2", description="d", ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="k3", node_type="knowledge_point", title="K3", description="d", ai_summary={"status": "success", "manual_edited": True}))
        fake_db.add("curriculum_node", dict(base, node_id="k4", node_type="knowledge_point", title="K4", description=None, ai_summary=None, needs_review=True))

    def test_overview_6_stats_fields_and_textbook_basic(self, make_client, fake_db):
        """G-API-4：返回 6 个 stats 字段（unit/lesson/kp/described/summarized/needs_review_count） + 教材基础信息。"""
        self._seed(fake_db)
        client = make_client(math_router)
        resp = client.get(f"/math/textbook/overview?textbook_id={TB_ID_01}")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["textbook_id"] == TB_ID_01
        stats = data["node_stats"]
        assert stats["unit_count"] == 3
        assert stats["lesson_count"] == 2
        assert stats["kp_count"] == 4
        # described 计数：node_id ∈ {u1,u3,l1,k1,k2,k3} description=真值 → 6（k4 没有）
        assert stats["described_count"] == 6
        # summarized = ai_summary.status=success（u3, k1, k3）→ 3
        assert stats["summarized_count"] == 3
        # needs_review_count = True（k4）→ 1
        assert stats["needs_review_count"] == 1


# ===========================================================================
# 5. POST /math/textbook/import-nodes — 批量导入 curriculum_node
# ===========================================================================


class TestImportNodes:
    BASE_NODES = [
        {"node_id": "u1", "node_type": "unit", "code": "U7-1", "title": "U1 有理数", "parent_code": None, "grade": "七年级", "semester": "up"},
        {"node_id": "u2", "node_type": "unit", "code": "U7-2", "title": "U2 整式加减", "parent_code": None, "grade": "七年级", "semester": "up"},
        {"node_id": "l1", "node_type": "lesson", "code": "L7-1", "title": "L1 正数与负数", "parent_code": "U7-1", "grade": "七年级", "semester": "up"},
    ]

    def test_import_inserts_rows_and_audit_import_math_nodes(self, make_client, fake_db):
        """首次导入：nodes 全部落库 curriculum_node + 审计 import_math_nodes + 返回 inserted=3。"""
        _seed_textbook(fake_db, textbook_id=TB_ID_01, title="七年级上册数学(导入用)", grade="七年级", semester="up")
        client = make_client(math_router)
        resp = client.post("/math/textbook/import-nodes", json={
            "textbook_id": TB_ID_01,
            "nodes": self.BASE_NODES,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        stats = body["data"]["stats"]
        assert stats["inserted"] == 3
        assert stats["skipped"] == 0
        assert stats["updated"] == 0
        assert stats["errors"] == []
        assert len(fake_db.all("curriculum_node")) == 3
        assert fake_db.all("curriculum_node")[0]["textbook_id"] == TB_ID_01
        assert "import_math_nodes" in _audit_actions(fake_db)

    def test_import_on_duplicate_skip_default_skips_existing_code(self, make_client, fake_db):
        """同 code 重复导入，默认 skip → skipped=N，不覆盖原值。"""
        _seed_textbook(fake_db, textbook_id=TB_ID_01, title="七年级上册数学(skip)", grade="七年级", semester="up")
        client = make_client(math_router)
        first = client.post("/math/textbook/import-nodes", json={
            "textbook_id": TB_ID_01,
            "nodes": self.BASE_NODES,
        })
        assert first.status_code == 200, first.text
        # 再次导入：l1 改标题，on_duplicate=skip（默认）→ skipped=1, updated=0
        payload_dup = [dict(self.BASE_NODES[2], title="L1 正数和负数 NEW")]
        resp = client.post("/math/textbook/import-nodes", json={
            "textbook_id": TB_ID_01,
            "nodes": payload_dup,
            "on_duplicate": "skip",
        })
        assert resp.status_code == 200, resp.text
        stats = resp.json()["data"]["stats"]
        assert stats["inserted"] == 0
        assert stats["skipped"] == 1
        assert stats["updated"] == 0
        rows = fake_db.all("curriculum_node")
        l1 = next(r for r in rows if r["code"] == "L7-1")
        assert l1["title"] == "L1 正数与负数", "skip 模式下不应覆盖"

    def test_import_on_duplicate_update_updates_title(self, make_client, fake_db):
        """on_duplicate=update 覆盖 code 相同文档 → updated=1。"""
        _seed_textbook(fake_db, textbook_id=TB_ID_01, title="七年级上册数学(update)", grade="七年级", semester="up")
        client = make_client(math_router)
        first = client.post("/math/textbook/import-nodes", json={"textbook_id": TB_ID_01, "nodes": self.BASE_NODES})
        assert first.status_code == 200, first.text
        new_nodes = [dict(self.BASE_NODES[0], title="U1 有理数【更新版】")]
        resp = client.post("/math/textbook/import-nodes", json={
            "textbook_id": TB_ID_01, "nodes": new_nodes, "on_duplicate": "update"})
        assert resp.status_code == 200, resp.text
        stats = resp.json()["data"]["stats"]
        assert stats["updated"] == 1
        assert stats["inserted"] == 0
        u1 = next(r for r in fake_db.all("curriculum_node") if r["code"] == "U7-1")
        assert u1["title"] == "U1 有理数【更新版】"


# ===========================================================================
# 6. DELETE /math/textbook/{id} — 清理（仅清 description+ai_summary，不删节点）
# ===========================================================================


class TestTextbookDeleteCleanup:
    def _seed(self, fake_db):
        _seed_textbook(fake_db)
        base = {"textbook_id": TB_ID_01, "grade": "七年级", "semester": "up"}
        fake_db.add("curriculum_node", dict(base, node_id="u1", node_type="unit", title="U1",
                                            description={"summary": "旧内容"},
                                            ai_summary={"status": "success", "version": 1}))
        fake_db.add("curriculum_node", dict(base, node_id="k1", node_type="knowledge_point", title="K1",
                                            description={"summary": "kp 描述"},
                                            ai_summary=None))
        fake_db.add("curriculum_node", dict(base, node_id="u2", node_type="unit", title="U2",
                                            description=None,
                                            ai_summary=None))  # 无内容，不计 cleared

    def test_delete_only_clears_description_and_summary_not_structural_fields(self, make_client, fake_db):
        """G-API-6：高危，入参 confirm_textbook_title=当前标题匹配；仅清 description/ai_summary，保留 title/node_type/code 等结构字段。返回 cleared_count 合计。"""
        self._seed(fake_db)
        client = make_client(math_router)
        resp = client.request(
            "DELETE", f"/math/textbook/{TB_ID_01}",
            json={"confirm_textbook_title": "人教版数学七年级上册"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["success"] is True
        # 3 节点：u1（清 2 项）+ k1（清 1 项）= cleared=3；u2 两字段本来为空 → 不计
        assert body["data"]["cleared_count"] == 3
        nodes = {r["node_id"]: r for r in fake_db.all("curriculum_node")}
        # 结构字段保留
        assert nodes["u1"]["title"] == "U1"
        assert nodes["u1"]["node_type"] == "unit"
        # description / ai_summary 被清空
        assert nodes["u1"]["description"] is None
        assert nodes["u1"]["ai_summary"] is None
        assert nodes["k1"]["description"] is None
        # 审计
        assert "delete_math_textbook" in _audit_actions(fake_db)

    def test_delete_mismatch_confirm_title_returns_400(self, make_client, fake_db):
        """二次确认标题不匹配 → 400，不执行清理。"""
        self._seed(fake_db)
        client = make_client(math_router)
        resp = client.request(
            "DELETE", f"/math/textbook/{TB_ID_01}",
            json={"confirm_textbook_title": "错的标题"},
        )
        assert resp.status_code == 400, resp.text
        # 未清
        u1 = next(r for r in fake_db.all("curriculum_node") if r["node_id"] == "u1")
        assert u1["description"] is not None

    def test_delete_missing_textbook_404(self, make_client, fake_db):
        """不存在 ID → 404。"""
        client = make_client(math_router)
        resp = client.request(
            "DELETE", "/math/textbook/tb_not_exist",
            json={"confirm_textbook_title": "X"},
        )
        assert resp.status_code == 404
