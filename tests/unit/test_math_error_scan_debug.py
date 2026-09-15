"""数学错题识别 Admin 调试干跑（error_scan_debug.py）单元测试

B03：_load_knowledge_point_candidates_by_textbook 候选集只读加载
"""
from __future__ import annotations

import asyncio

from services.math.error_scan_debug import _load_knowledge_point_candidates_by_textbook
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _kp_node(
    node_id: str = "n1",
    code: str = "5-3-1-kp2",
    kp_name: str = "小数加减法",
    textbook_id: str = "tb_1",
    grade: str = "五年级",
    semester: str = "up",
    title: str = "小数乘整数",
    unit_title: str = "第3单元 小数乘法",
    lesson_title: str = "小数乘整数",
    status: str = "success",
) -> dict:
    """构造一个带 ai_summary.knowledge_points 的 curriculum_node 文档。"""
    return {
        "node_id": node_id,
        "code": code,
        "textbook_id": textbook_id,
        "grade": grade,
        "semester": semester,
        "title": title,
        "unit_title": unit_title,
        "lesson_title": lesson_title,
        "ai_summary": {"status": status, "knowledge_points": [{"name": kp_name}]},
    }


# ---------------- B03: _load_knowledge_point_candidates_by_textbook ----------------


class TestLoadKpCandidatesByTextbook:
    """候选集只读加载（B03）。

    覆盖：
    - 按 textbook_id 精确加载
    - textbook_id 为空 → 全量候选
    - textbook_id 无匹配 → 回退全量
    - 候选项字段结构正确
    - 不读 scholar_book 集合
    - debug 段 source/count/prompt_included/truncated 正确
    """

    def test_textbook_id_filter(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["source"] == "textbook_id"
        assert result["count"] == 1
        assert result["candidates"][0]["kp_name"] == "小数加减法"
        assert result["candidates"][0]["textbook_id"] == "tb_1"

    def test_empty_textbook_id_loads_all(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, ""))
        assert result["source"] == "all"
        assert result["count"] == 2

    def test_textbook_id_no_match_fallback_all(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_nonexistent"))
        assert result["source"] == "all"
        assert result["count"] == 1

    def test_candidate_fields_complete(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node())
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        c = result["candidates"][0]
        assert c["node_id"] == "n1"
        assert c["node_code"] == "5-3-1-kp2"
        assert c["kp_name"] == "小数加减法"
        assert c["grade"] == "五年级"
        assert c["semester"] == "up"
        assert c["textbook_id"] == "tb_1"
        assert c["title"] == "小数乘整数"
        assert c["unit_title"] == "第3单元 小数乘法"
        assert c["lesson_title"] == "小数乘整数"

    def test_skips_nodes_without_success_summary(self):
        db = FakeDB()
        db.add("curriculum_node", _kp_node(status="pending"))
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 0

    def test_skips_kp_with_empty_name(self):
        node = _kp_node()
        node["ai_summary"]["knowledge_points"] = [{"name": ""}, {"name": "  "}]
        db = FakeDB()
        db.add("curriculum_node", node)
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 0

    def test_truncated_when_exceeds_limit(self, monkeypatch):
        monkeypatch.setattr("config.LLM_JUDGE_CANDIDATE_LIMIT", 2)
        db = FakeDB()
        for i in range(5):
            db.add(
                "curriculum_node",
                _kp_node(node_id=f"n{i}", code=f"c{i}", kp_name=f"kp{i}"),
            )
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["count"] == 5
        assert result["prompt_included"] == 2
        assert result["truncated"] is True

    def test_no_truncation_within_limit(self, monkeypatch):
        monkeypatch.setattr("config.LLM_JUDGE_CANDIDATE_LIMIT", 40)
        db = FakeDB()
        db.add("curriculum_node", _kp_node())
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1"))
        assert result["prompt_included"] == 1
        assert result["truncated"] is False

    def test_does_not_read_scholar_book(self):
        """干跑不读 scholar_book 集合（与生产路径隔离）。

        若干跑读了 scholar_book，结果会被指向 tb_2，但实际传入 textbook_id=tb_1
        仍应只返回 tb_1 的候选——证明不走 scholar_book 路径。
        """
        db = FakeDB()
        db.add("curriculum_node", _kp_node(textbook_id="tb_1", kp_name="小数加减法"))
        db.add("curriculum_node", _kp_node(node_id="n2", textbook_id="tb_2", kp_name="分数"))
        db.add("scholar_book", {"scholar_id": "s1", "textbook_id": "tb_2"})
        result = _run(_load_knowledge_point_candidates_by_textbook(db, "tb_1", scholar_id="s1"))
        assert result["count"] == 1
        assert result["candidates"][0]["textbook_id"] == "tb_1"
