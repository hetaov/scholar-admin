"""单元测试：数学同类型题出参净化 services/math/quiz._to_public_quiz（B03 · 红线 R3）

覆盖：
- `items[]` **不含** `answer` / `hint` 键（答案与提示落库不出参）；
- 保留字段齐全：`quiz_item_id` / `source_index` / `question` / `answer_type` / `variant_level` / `difficulty`；
- 顶层保留 `quiz_id` / `status` / `sources[]` / `source_count` / `count_per_source` / `file_refs` / `warnings[]` / `render_status`；
- **原 quiz 对象未被就地修改**（净化返回新对象，避免污染落库数据）。
"""
from __future__ import annotations

import pytest

from services.math.quiz import _to_public_quiz

KEPT_ITEM_KEYS = (
    "quiz_item_id",
    "source_index",
    "question",
    "answer_type",
    "variant_level",
    "difficulty",
)


def _quiz() -> dict:
    return {
        "quiz_id": "qz_1",
        "scholar_id": "s1",
        "source_scan_id": "su_1",
        "sources": [
            {
                "source_index": 0,
                "question_text": "0.36 + 2.7 = ?",
                "knowledge_point_name": "小数加减法",
                "node_code": "5-3-1-kp2",
            }
        ],
        "source_count": 1,
        "count_per_source": 2,
        "items": [
            {
                "quiz_item_id": "qi_1",
                "source_index": 0,
                "question": "2.7 + 0.36 = ?",
                "answer": "3.06",
                "answer_type": "number",
                "variant_level": "L1",
                "difficulty": 2,
                "hint": "小数点对齐，低位补零",
            },
            {
                "quiz_item_id": "qi_2",
                "source_index": 0,
                "question": "1.25 + 3.8 = ?",
                "answer": "5.05",
                "answer_type": "number",
                "variant_level": "L2",
                "difficulty": 3,
                "hint": "",
            },
        ],
        "status": "generated",
        "file_refs": {"pdf": "/static/sheets/qz_1/sheet.pdf", "png": "", "preview": ""},
        "render_status": "",
        "warnings": [],
        "signature": "sig_1",
        "created_at": 1000,
    }


def test_items_exclude_answer_and_hint():
    public = _to_public_quiz(_quiz())
    assert len(public["items"]) == 2
    for item in public["items"]:
        assert "answer" not in item
        assert "hint" not in item


def test_items_keep_public_fields():
    public = _to_public_quiz(_quiz())
    first = public["items"][0]
    for key in KEPT_ITEM_KEYS:
        assert key in first
    assert first["quiz_item_id"] == "qi_1"
    assert first["source_index"] == 0
    assert first["question"] == "2.7 + 0.36 = ?"
    assert first["answer_type"] == "number"
    assert first["variant_level"] == "L1"
    assert first["difficulty"] == 2


def test_top_level_fields():
    quiz = _quiz()
    public = _to_public_quiz(quiz)
    assert public["quiz_id"] == "qz_1"
    assert public["status"] == "generated"
    assert public["source_count"] == 1
    assert public["count_per_source"] == 2
    assert public["sources"] == quiz["sources"]
    assert public["file_refs"] == quiz["file_refs"]
    assert public["warnings"] == []
    assert public["render_status"] == ""
    # 内部字段不外泄
    assert "signature" not in public
    assert "audit_log_id" not in public


def test_source_quiz_not_mutated():
    quiz = _quiz()
    snapshot = {
        "items": [dict(it) for it in quiz["items"]],
        "sources": [dict(s) for s in quiz["sources"]],
    }
    _to_public_quiz(quiz)
    # 原对象逐字段不变（answer / hint 仍在落库文档中）
    assert quiz["items"] == snapshot["items"]
    assert quiz["sources"] == snapshot["sources"]
    assert quiz["items"][0]["answer"] == "3.06"
    assert quiz["items"][0]["hint"] == "小数点对齐，低位补零"


def test_empty_items_and_defaults():
    public = _to_public_quiz({"quiz_id": "qz_2"})
    assert public["items"] == []
    assert public["sources"] == []
    assert public["warnings"] == []
    assert public["source_count"] == 0
    assert public["count_per_source"] == 0
    assert public["file_refs"] == {"pdf": "", "png": "", "preview": ""}
    assert public["render_status"] == ""


def test_render_status_passthrough():
    quiz = _quiz()
    quiz["render_status"] = "success"
    assert _to_public_quiz(quiz)["render_status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
