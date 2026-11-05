"""描述出题异步任务验收（POST /math/quiz/generate 描述分支 · 方案 §7.2）

背景：推理型模型单次出题 30~120s，超 callContainer 15s 上限（真机 timeout 实测）
⇒ 描述分支改为「骨架文档 + 后台任务」，前端轮询接口 31（status: generating →
generated / failed）。

覆盖场景：
- 派发即返回 {quiz_id, status: generating}，骨架已落库（轮询可查）
- 后台任务完成 → status=generated + items 回写（R3：不出参 answer）
- 后台任务产出 failed → status=failed + error_message（前端可直接展示）
- 后台任务抛异常 → status=failed（不向上抛）
- 入参非法（题量越界）→ **同步 400**（不能等后台任务才暴露）
- intent 与 source_questions 互斥 → 400
"""
from __future__ import annotations

import asyncio

import pytest

from services.math.quiz import QUIZ_STATUS_FAILED, QUIZ_STATUS_GENERATED, QUIZ_STATUS_GENERATING
from services.routes_math import router as math_router


SCHOLAR_ID = "scholar_desc_001"


@pytest.fixture()
def spawn_capture(monkeypatch):
    """截获后台任务参数（TestClient 的 loop 不等后台任务，测试内手动驱动）。"""
    captured: dict = {}

    def _capture(db, **kwargs):
        captured["db"] = db
        captured.update(kwargs)

    monkeypatch.setattr("services.routes.math._spawn_desc_quiz_task", _capture)
    return captured


def _post(client, **overrides):
    payload = {
        "scholar_id": SCHOLAR_ID,
        "intent": {"knowledge_points": ["分数的简单加减"], "question_count": 2},
        **overrides,
    }
    return client.post("/math/quiz/generate", json=payload)


def test_dispatch_returns_generating_with_skeleton_doc(make_client, fake_db, spawn_capture):
    client = make_client(math_router)
    res = _post(client)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["success"] is True
    data = body["data"]
    assert data["status"] == QUIZ_STATUS_GENERATING
    quiz_id = data["quiz_id"]
    assert quiz_id
    # 骨架已落库，轮询接口 31 可查（且此刻还无题目）
    got = client.get(f"/math/quiz/{quiz_id}").json()["data"]
    assert got["status"] == QUIZ_STATUS_GENERATING
    assert got["items"] == []
    # 后台任务参数完整（含 count 校验后题量）
    assert spawn_capture["quiz_id"] == quiz_id
    assert spawn_capture["intent"]["question_count"] == 2


def test_task_completion_writes_back_items(make_client, fake_db, spawn_capture, monkeypatch):
    client = make_client(math_router)
    quiz_id = _post(client).json()["data"]["quiz_id"]

    async def _fake_generate(db, **kwargs):  # noqa: ANN001
        assert kwargs["persist"] is False
        assert kwargs["quiz_id"] == quiz_id
        return {
            "quiz_id": quiz_id,
            "scholar_id": SCHOLAR_ID,
            "items": [
                {"quiz_item_id": "qi1", "question": "3/7+2/7=?", "answer": "5/7"},
            ],
            "warnings": [],
            "graph_version": 1,
            "intake_meta": {},
            "count_per_source": 2,
        }

    monkeypatch.setattr(
        "services.routes.math.generateQuizFromDescription", _fake_generate
    )
    asyncio.run(_task(spawn_capture))

    got = client.get(f"/math/quiz/{quiz_id}").json()["data"]
    assert got["status"] == QUIZ_STATUS_GENERATED
    assert got["items"][0]["question"] == "3/7+2/7=?"
    assert "answer" not in got["items"][0]  # R3
    assert got["graph_version"] == 1


def test_task_failed_result_writes_error_message(make_client, fake_db, spawn_capture, monkeypatch):
    client = make_client(math_router)
    quiz_id = _post(client).json()["data"]["quiz_id"]

    async def _fake_generate(db, **kwargs):  # noqa: ANN001
        return {
            "quiz_id": "",
            "status": QUIZ_STATUS_FAILED,
            "items": [],
            "warnings": ["DESC_BASIC_ITEM_DROPPED[V2_CHECK]: 1"],
            "error_message": "未生成有效题目，请补充描述后重试",
            "graph_version": 0,
        }

    monkeypatch.setattr(
        "services.routes.math.generateQuizFromDescription", _fake_generate
    )
    asyncio.run(_task(spawn_capture))

    got = client.get(f"/math/quiz/{quiz_id}").json()["data"]
    assert got["status"] == QUIZ_STATUS_FAILED
    assert "未生成有效题目" in got["error_message"]


def test_task_exception_marks_failed(make_client, fake_db, spawn_capture, monkeypatch):
    client = make_client(math_router)
    quiz_id = _post(client).json()["data"]["quiz_id"]

    async def _boom(db, **kwargs):  # noqa: ANN001
        raise RuntimeError("图节点炸了")

    monkeypatch.setattr("services.routes.math.generateQuizFromDescription", _boom)
    asyncio.run(_task(spawn_capture))  # 不抛出

    got = client.get(f"/math/quiz/{quiz_id}").json()["data"]
    assert got["status"] == QUIZ_STATUS_FAILED
    assert "图节点炸了" in got["error_message"]


def test_invalid_count_sync_400(make_client, fake_db, spawn_capture):
    client = make_client(math_router)
    for bad in (21, -1):
        res = _post(client, intent={"knowledge_points": ["分数"], "question_count": bad})
        assert res.status_code == 400, f"count={bad} 应同步 400"
    assert spawn_capture == {}  # 非法入参不派发任务


def test_count_zero_falls_back_to_default(make_client, fake_db, spawn_capture):
    """0 按「未提供」处理回落默认 5（与既有单步路径 `_normalize_count` 同口径）"""
    client = make_client(math_router)
    res = _post(client, intent={"knowledge_points": ["分数"], "question_count": 0})
    assert res.status_code == 200
    assert spawn_capture["intent"]["question_count"] == 0  # 原样透传，任务内归一


def test_intent_and_sources_conflict_400(make_client, fake_db, spawn_capture):
    client = make_client(math_router)
    res = _post(
        client,
        source_questions=[{"source_index": 0, "question_text": "0.36+2.7=?"}],
    )
    assert res.status_code == 400
    assert "不可同时提供" in res.json()["detail"]


def _task(captured: dict):
    """按截获参数驱动后台任务（与 _spawn_desc_quiz_task 等价）"""
    from services.routes.math import _desc_quiz_generate_task

    kwargs = {k: v for k, v in captured.items() if k != "db"}
    return _desc_quiz_generate_task(captured["db"], **kwargs)
