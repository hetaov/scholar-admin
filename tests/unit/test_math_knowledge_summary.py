"""单元测试：AI 知识总结 services/math/knowledge_summary（F1.1-1 / F1.1-2 / F1.1-3）

覆盖：
- F1.1-1 骨架：summary_idempotency_key、异常层级、__init__ 追加导出、empty_ai_summary
- F1.1-2 核心逻辑（mock LLM）：
  - 无描述节点 → NoDescriptionError 且 ai_summary 未写入
  - 有描述节点 → 写回结构化 knowledge_points + 三档 extended_points，
    写回 model 证明 LLM_SUMMARY_MODEL 生效
  - 同参数第二次调用幂等命中（LLM 仅调用 1 次）
  - JSON 解析失败重试 1 次仍失败 → status=failed 写回 + failed 审计 + 明确错误
- F1.1-3 getKnowledgeSummary：
  - 节点不存在 → NodeNotFoundError
  - 未生成 → {status: "not_generated"}（不报错）
  - 已生成 → 返回完整 ai_summary
"""
from __future__ import annotations

import pytest

from services.audit import AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY, AUDIT_LOG_COLLECTION
from services.database import CURRICULUM_NODE_COLLECTION
from services.math import (
    ABILITY_DIMENSIONS,
    DESCRIPTION_NODE_TYPES,
    DESCRIPTION_SOURCE_MANUAL,
    EXTENDED_DIFFICULTY_BANDS,
    SUMMARY_NODE_TYPES,
    SUMMARY_STATUS_FAILED,
    SUMMARY_STATUS_NOT_GENERATED,
    SUMMARY_STATUS_SUCCESS,
    description_idempotency_key,
    knowledge_summary,
)
from services.math.knowledge_summary import (
    KnowledgeSummaryError,
    LLMResponseError,
    NoDescriptionError,
    NodeNotFoundError,
    empty_ai_summary,
    summary_idempotency_key,
)
from tests.fakes.fake_db import FakeDB


def _key(**overrides):
    base = dict(
        textbook_id="tb1",
        grade="3",
        semester="上",
        unit_id="u1",
        lesson_id="l1",
        description_version=2,
        model="summary-model-v1",
    )
    base.update(overrides)
    return summary_idempotency_key(**base)


# ---------------------------------------------------------------------------
# summary_idempotency_key
# ---------------------------------------------------------------------------


def test_summary_idempotency_key_same_input_same_key():
    assert _key() == _key()
    assert len(_key()) == 64  # sha256 hexdigest


def test_summary_idempotency_key_changes_on_description_version():
    assert _key(description_version=2) != _key(description_version=3)


def test_summary_idempotency_key_changes_on_model():
    assert _key(model="summary-model-v1") != _key(model="summary-model-v2")


# ---------------------------------------------------------------------------
# 异常层级
# ---------------------------------------------------------------------------


def test_exceptions_instantiable_with_chinese_message():
    cases = [
        (NodeNotFoundError, "curriculum_node 不存在: n1"),
        (NoDescriptionError, "节点无描述，不总结: n1"),
        (LLMResponseError, "LLM 响应解析失败: invalid json"),
    ]
    for cls, message in cases:
        err = cls(message)
        assert isinstance(err, KnowledgeSummaryError)
        assert str(err) == message


# ---------------------------------------------------------------------------
# __init__ 追加导出不破坏既有导入
# ---------------------------------------------------------------------------


def test_math_package_exports_f1_constants_and_keeps_f2():
    # F1 新增导出
    assert SUMMARY_STATUS_SUCCESS == "success"
    assert SUMMARY_STATUS_FAILED == "failed"
    assert SUMMARY_NODE_TYPES == ("unit", "lesson", "knowledge_point")
    assert set(ABILITY_DIMENSIONS) == {
        "arithmetic",
        "computation",
        "modeling",
        "reasoning",
    }
    assert EXTENDED_DIFFICULTY_BANDS == ("入门", "普及", "竞赛")
    # 既有 F2 导出未被破坏
    assert DESCRIPTION_NODE_TYPES == SUMMARY_NODE_TYPES
    assert DESCRIPTION_SOURCE_MANUAL == "manual"
    assert description_idempotency_key("n1", 1) == "n1:v1"


# ---------------------------------------------------------------------------
# empty_ai_summary
# ---------------------------------------------------------------------------


def test_empty_ai_summary_structure():
    summary = empty_ai_summary(model="m1", idempotency_key="k1")
    assert summary["status"] == "pending"
    assert summary["generated_at"] == 0
    assert summary["model"] == "m1"
    assert summary["knowledge_points"] == []
    assert summary["extended_points"] == []
    assert summary["idempotency_key"] == "k1"


# ---------------------------------------------------------------------------
# generateKnowledgeSummary（F1.1-2 核心逻辑）
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _summary_model(monkeypatch):
    """固定 LLM_SUMMARY_MODEL，验证写回 model 字段（契约 §4.12.8(b)）"""
    monkeypatch.setattr(knowledge_summary, "LLM_SUMMARY_MODEL", "summary-model-test")


def _node(**overrides):
    """构造带描述的 curriculum_node 种子（契约 §4.12.1 字段命名兼容 unit_id/lesson_id）"""
    base = {
        "_id": "n1",
        "node_id": "n1",
        "node_type": "lesson",
        "textbook_id": "tb1",
        "title": "万以内的加法和减法",
        "grade": "3",
        "semester": "上",
        "unit_id": "u1",
        "unit_title": "万以内的加法和减法",
        "lesson_id": "l1",
        "lesson_title": "笔算加法",
        "description_version": 2,
        "description": {"目标": "掌握三位数加减法笔算方法"},
    }
    base.update(overrides)
    return base


def _llm_result():
    """合法结构化输出：knowledge_points + 三档 extended_points"""
    return {
        "knowledge_points": [
            {
                "name": "整数加减法",
                "summary": "掌握三位数加减法的笔算方法",
                "ability_dimensions": ["arithmetic", "computation"],
                "source_node_id": "kp1",
                "source_lesson_id": "l1",
            }
        ],
        "extended_points": [
            {
                "name": "凑整巧算",
                "summary": "加减法凑整",
                "difficulty_band": "入门",
                "related_knowledge_name": "整数加减法",
                "source_lesson_id": "l1",
            },
            {
                "name": "速算技巧",
                "summary": "常见速算",
                "difficulty_band": "普及",
                "related_knowledge_name": "整数加减法",
                "source_lesson_id": "l1",
            },
            {
                "name": "竞赛拓展",
                "summary": "竞赛题型",
                "difficulty_band": "竞赛",
                "related_knowledge_name": "整数加减法",
                "source_lesson_id": "l1",
            },
        ],
    }


@pytest.mark.asyncio
async def test_generate_summary_success_writes_structured_ai_summary(monkeypatch):
    """有描述节点 → 写回结构化 knowledge_points + 三档 extended_points，model 生效"""
    db = FakeDB()
    db.add(CURRICULUM_NODE_COLLECTION, _node())
    calls: list[str] = []

    async def fake_llm(node, *, include_extended_points):
        calls.append(node["node_id"])
        return _llm_result()

    monkeypatch.setattr(knowledge_summary, "_call_summary_llm", fake_llm)

    result = await knowledge_summary.generateKnowledgeSummary(
        db, curriculum_node_id="n1"
    )

    assert result["summary_id"] == "n1"
    assert result["status"] == SUMMARY_STATUS_SUCCESS
    assert len(result["idempotency_key"]) == 64
    assert result["knowledge_points"][0]["name"] == "整数加减法"
    assert {ep["difficulty_band"] for ep in result["extended_points"]} == set(
        EXTENDED_DIFFICULTY_BANDS
    )
    assert result["generated_at"] > 0
    assert calls == ["n1"]

    ai_summary = db.all(CURRICULUM_NODE_COLLECTION)[0]["ai_summary"]
    assert ai_summary["status"] == SUMMARY_STATUS_SUCCESS
    assert ai_summary["model"] == "summary-model-test"  # LLM_SUMMARY_MODEL 生效
    assert ai_summary["idempotency_key"] == result["idempotency_key"]
    assert ai_summary["knowledge_points"][0]["ability_dimensions"] == [
        "arithmetic",
        "computation",
    ]
    assert len(ai_summary["extended_points"]) == 3

    logs = db.all(AUDIT_LOG_COLLECTION)
    assert len(logs) == 1
    assert logs[0]["action"] == AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY
    assert logs[0]["object_ref"] == "n1"
    assert logs[0]["result"] == "success"
    assert logs[0]["context"]["model"] == "summary-model-test"
    assert logs[0]["context"]["idempotency_key"] == result["idempotency_key"]


@pytest.mark.asyncio
async def test_generate_summary_no_description_raises_without_write(monkeypatch):
    """无描述节点 → NoDescriptionError 且 ai_summary 未写入（"无描述不总结"）"""
    db = FakeDB()
    db.add(CURRICULUM_NODE_COLLECTION, _node(description=None))

    async def fake_llm(node, *, include_extended_points):
        raise AssertionError("不应调用 LLM")

    monkeypatch.setattr(knowledge_summary, "_call_summary_llm", fake_llm)

    with pytest.raises(NoDescriptionError, match="无描述"):
        await knowledge_summary.generateKnowledgeSummary(db, curriculum_node_id="n1")

    node = db.all(CURRICULUM_NODE_COLLECTION)[0]
    assert "ai_summary" not in node
    assert db.all(AUDIT_LOG_COLLECTION) == []


@pytest.mark.asyncio
async def test_generate_summary_idempotent_hit_calls_llm_once(monkeypatch):
    """同参数第二次调用幂等命中 → LLM 仅调用 1 次，直接返回已有结果"""
    db = FakeDB()
    db.add(CURRICULUM_NODE_COLLECTION, _node())
    calls: list[str] = []

    async def fake_llm(node, *, include_extended_points):
        calls.append(node["node_id"])
        return _llm_result()

    monkeypatch.setattr(knowledge_summary, "_call_summary_llm", fake_llm)

    first = await knowledge_summary.generateKnowledgeSummary(
        db, curriculum_node_id="n1"
    )
    second = await knowledge_summary.generateKnowledgeSummary(
        db, curriculum_node_id="n1"
    )

    assert calls == ["n1"]  # 幂等命中，第二次未调用 LLM
    assert second["idempotency_key"] == first["idempotency_key"]
    assert second["status"] == SUMMARY_STATUS_SUCCESS
    assert second["knowledge_points"] == first["knowledge_points"]
    assert second["generated_at"] == first["generated_at"]


@pytest.mark.asyncio
async def test_generate_summary_json_failure_retries_once_then_failed(monkeypatch):
    """JSON 解析失败重试 1 次仍失败 → status=failed 写回 + failed 审计 + 明确错误"""
    db = FakeDB()
    db.add(CURRICULUM_NODE_COLLECTION, _node())
    attempts: list[str] = []

    def fake_chat_sync(client, model, prompt):
        attempts.append(model)
        return "这不是合法 JSON"

    monkeypatch.setattr(knowledge_summary, "_call_chat_sync", fake_chat_sync)
    monkeypatch.setattr(knowledge_summary, "_get_llm_client", lambda: object())

    with pytest.raises(LLMResponseError, match="AI 知识总结生成失败"):
        await knowledge_summary.generateKnowledgeSummary(db, curriculum_node_id="n1")

    assert attempts == ["summary-model-test", "summary-model-test"]  # 首次 + 重试 1 次
    ai_summary = db.all(CURRICULUM_NODE_COLLECTION)[0]["ai_summary"]
    assert ai_summary["status"] == SUMMARY_STATUS_FAILED
    assert ai_summary["model"] == "summary-model-test"
    assert ai_summary["knowledge_points"] == []
    assert ai_summary["extended_points"] == []
    assert ai_summary["idempotency_key"]

    logs = db.all(AUDIT_LOG_COLLECTION)
    assert len(logs) == 1
    assert logs[0]["action"] == AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY
    assert logs[0]["result"] == "failed"


# ---------------------------------------------------------------------------
# getKnowledgeSummary（F1.1-3）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_summary_node_not_found_raises():
    """节点不存在 → NodeNotFoundError（路由层映射 404）"""
    db = FakeDB()
    with pytest.raises(NodeNotFoundError, match="不存在"):
        await knowledge_summary.getKnowledgeSummary(db, curriculum_node_id="n1")


@pytest.mark.asyncio
async def test_get_summary_not_generated_returns_not_generated():
    """节点存在但 ai_summary 未生成 → {status: "not_generated"}（不报错）"""
    db = FakeDB()
    db.add(CURRICULUM_NODE_COLLECTION, _node())  # 有描述但从未生成
    result = await knowledge_summary.getKnowledgeSummary(db, curriculum_node_id="n1")
    assert result["curriculum_node_id"] == "n1"
    assert result["status"] == SUMMARY_STATUS_NOT_GENERATED
    assert result["generated_at"] == 0
    assert result["model"] == ""
    assert result["knowledge_points"] == []
    assert result["extended_points"] == []
    assert result["idempotency_key"] == ""


@pytest.mark.asyncio
async def test_get_summary_generated_returns_full_ai_summary():
    """已生成 → 返回完整 ai_summary（curriculum_node_id + ai_summary 字段）"""
    db = FakeDB()
    node = _node()
    node["ai_summary"] = {
        "status": SUMMARY_STATUS_SUCCESS,
        "generated_at": 12345,
        "model": "summary-model-test",
        "knowledge_points": [{"name": "整数加减法"}],
        "extended_points": [{"name": "凑整巧算", "difficulty_band": "入门"}],
        "idempotency_key": "k1",
    }
    db.add(CURRICULUM_NODE_COLLECTION, node)
    result = await knowledge_summary.getKnowledgeSummary(db, curriculum_node_id="n1")
    assert result["curriculum_node_id"] == "n1"
    assert result["status"] == SUMMARY_STATUS_SUCCESS
    assert result["generated_at"] == 12345
    assert result["model"] == "summary-model-test"
    assert result["knowledge_points"][0]["name"] == "整数加减法"
    assert result["extended_points"][0]["difficulty_band"] == "入门"
    assert result["idempotency_key"] == "k1"
