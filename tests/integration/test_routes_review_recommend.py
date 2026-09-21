"""v4 R4 第二段（AI 排序与理由）路由集成测试

覆盖：
1. **门控**：开关关 → 真实 app（main.app）不注册路由 → 404；开关开 → 路由可用
2. 入参校验：缺 scholar_id / 空 candidates → 200 + `success=false` + `INVALID_INPUT`
3. 提交返回 task_id（毫秒级，pending/processing）
4. 轮询：任务成功 → `recommendations` 的 `group_id` ⊆ 候选集；不存在 → `TASK_NOT_FOUND`
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.learning.review_recommend_task import (
    create_review_recommend_task,
    run_review_recommend_task,
)
from services.routes_review_recommend import router as review_recommend_router

from main import app as real_app


def _run(coro):
    return asyncio.run(coro)


CANDIDATES = [
    {"group_id": "g1", "group_title": "第 3 组", "group_status": 1,
     "group_mastery": 0.4, "due_sentence_count": 2, "sentence_count": 3},
    {"group_id": "g2", "group_title": "第 4 组", "group_status": 2,
     "group_mastery": 0.6, "due_sentence_count": 0, "sentence_count": 2},
]


def test_gate_off_real_app_returns_404():
    """REVIEW_RECOMMEND_ENABLED 默认 0 → main.py 不注册 → 404（不是 403，避免信息泄露）。"""
    client = TestClient(real_app)
    resp = client.post("/ai/review-recommend", json={"scholar_id": "s1", "candidates": CANDIDATES})
    assert resp.status_code == 404


def test_gate_off_without_router_returns_404():
    """模拟开关关：不带路由的 app → 404。"""
    client = TestClient(FastAPI())
    assert client.get("/ai/review-recommend/task/rr_x").status_code == 404


@pytest.fixture()
def client(make_client):
    """开关开：带 review_recommend_router（FakeDB 注入）。"""
    return make_client(review_recommend_router)


def test_submit_invalid_input(client):
    r1 = client.post("/ai/review-recommend", json={"candidates": CANDIDATES})
    assert r1.status_code == 200
    assert r1.json()["success"] is False and r1.json()["code"] == "INVALID_INPUT"

    r2 = client.post("/ai/review-recommend", json={"scholar_id": "s1", "candidates": []})
    assert r2.json()["success"] is False and r2.json()["code"] == "INVALID_INPUT"

    # 候选缺 group_id → 视为空
    r3 = client.post("/ai/review-recommend", json={"scholar_id": "s1", "candidates": [{"x": 1}]})
    assert r3.json()["code"] == "INVALID_INPUT"


def test_submit_returns_task_id(client):
    resp = client.post("/ai/review-recommend", json={"scholar_id": "s1", "candidates": CANDIDATES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["data"]["task_id"].startswith("rr_")
    # 后台任务可能已在跑 → pending 或 processing 均合法
    assert body["data"]["status"] in ("pending", "processing")


def test_poll_success_recommendations_subset_of_candidates(client, fake_db, monkeypatch):
    """确定性轮询：建任务 → 打桩 LLM → 显式跑执行器 → 轮询到 success 且 ⊆ 候选集。"""
    content = json.dumps({"recommendations": [
        {"group_id": "g2", "priority": 1, "reason_text": "最早学过"},
        {"group_id": "g1", "priority": 2, "reason_text": "到期 2 句"},
    ]})
    monkeypatch.setattr(
        "services.providers.review_recommend._call_review_recommend_llm",
        lambda *a, **k: content,
    )
    task = _run(create_review_recommend_task(fake_db, scholar_id="s1", candidates=CANDIDATES))
    _run(run_review_recommend_task(task["task_id"]))

    resp = client.get(f"/ai/review-recommend/task/{task['task_id']}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "success"
    recs = data["result"]["recommendations"]
    assert {r["group_id"] for r in recs} <= {c["group_id"] for c in CANDIDATES}
    assert [r["group_id"] for r in recs] == ["g2", "g1"]  # 模型排序被保留
    assert data["result"]["strategy"] == "ai_rank"


def test_poll_failed_task_reports_error(client, fake_db, monkeypatch):
    """LLM 输出越界 → failed（客户端据 status 回退规则排序）。"""
    monkeypatch.setattr(
        "services.providers.review_recommend._call_review_recommend_llm",
        lambda *a, **k: json.dumps({"recommendations": [
            {"group_id": "gX", "priority": 1, "reason_text": "自造组"},
        ]}),
    )
    task = _run(create_review_recommend_task(fake_db, scholar_id="s1", candidates=CANDIDATES))
    _run(run_review_recommend_task(task["task_id"]))
    data = client.get(f"/ai/review-recommend/task/{task['task_id']}").json()["data"]
    assert data["status"] == "failed"
    assert data["error"]["error_code"] == "INVALID_OUTPUT"


def test_poll_missing_task(client):
    resp = client.get("/ai/review-recommend/task/rr_not_exist")
    assert resp.status_code == 200
    assert resp.json()["success"] is False and resp.json()["code"] == "TASK_NOT_FOUND"
