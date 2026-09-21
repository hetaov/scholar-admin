"""v4 R4 第二段（AI 排序与理由）单测

覆盖：
1. `parse_review_recommend_output`：**强校验**（越界/重复/缺项 → None；reason 容错与截断）
2. `review_recommend_task` 任务状态流转（create/claim/finish/get/cleanup）
3. `run_review_recommend_task` 后台执行器（成功 / 输出非法 / LLM 不可用 / 空候选 / 已被抢占）
4. `call_review_recommend_llm` 超时强取消（LLM_TIMEOUT）
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from services.learning import review_recommend_task as rrt
from services.providers.review_recommend import (
    ERR_INVALID_OUTPUT,
    ERR_LLM_TIMEOUT,
    ERR_LLM_UNAVAILABLE,
    ReviewRecommendError,
    build_review_recommend_messages,
    call_review_recommend_llm,
    parse_review_recommend_output,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


CANDIDATES = [
    {
        "group_id": "g1", "group_title": "第 3 组", "group_status": 1,
        "group_mastery": 0.4, "due_sentence_count": 2, "sentence_count": 3,
    },
    {
        "group_id": "g2", "group_title": "第 4 组", "group_status": 2,
        "group_mastery": 0.6, "due_sentence_count": 0, "sentence_count": 2,
    },
]
CANDIDATE_IDS = ["g1", "g2"]


# ---------------------------------------------------------------------------
# 1. 解析与强校验（A2/D7：不得增删候选）
# ---------------------------------------------------------------------------


def test_parse_full_coverage_keeps_model_order():
    content = json.dumps({
        "recommendations": [
            {"group_id": "g2", "priority": 1, "reason_text": "最早学过，先复盘"},
            {"group_id": "g1", "priority": 2, "reason_text": "到期 2 句"},
        ]
    })
    out = parse_review_recommend_output(content, CANDIDATE_IDS)
    assert [r["group_id"] for r in out] == ["g2", "g1"]  # 顺序 = 模型排序
    assert out[0]["priority"] == 1
    assert out[1]["reason_text"] == "到期 2 句"


def test_parse_bare_list_and_code_fence_tolerated():
    bare = json.dumps([
        {"group_id": "g1", "priority": 1, "reason_text": "a"},
        {"group_id": "g2", "priority": 2, "reason_text": "b"},
    ])
    assert parse_review_recommend_output(bare, CANDIDATE_IDS) is not None
    fenced = "```json\n" + bare + "\n```"
    assert parse_review_recommend_output(fenced, CANDIDATE_IDS) is not None


def test_parse_out_of_scope_group_id_is_invalid():
    """越界 group_id（模型自造组）→ 非法（A2）。"""
    content = json.dumps({"recommendations": [
        {"group_id": "g1", "priority": 1, "reason_text": "a"},
        {"group_id": "gX", "priority": 2, "reason_text": "b"},
    ]})
    assert parse_review_recommend_output(content, CANDIDATE_IDS) is None


def test_parse_duplicate_group_id_is_invalid():
    content = json.dumps({"recommendations": [
        {"group_id": "g1", "priority": 1, "reason_text": "a"},
        {"group_id": "g1", "priority": 2, "reason_text": "b"},
    ]})
    assert parse_review_recommend_output(content, CANDIDATE_IDS) is None


def test_parse_missing_candidate_is_invalid():
    """缺项 = 删除候选 → 非法（A2）。"""
    content = json.dumps({"recommendations": [
        {"group_id": "g1", "priority": 1, "reason_text": "a"},
    ]})
    assert parse_review_recommend_output(content, CANDIDATE_IDS) is None


def test_parse_reason_tolerance_and_truncation():
    long_reason = "很" * 200
    content = json.dumps({"recommendations": [
        {"group_id": "g1", "priority": 1},  # 缺 reason_text → 容错为空串
        {"group_id": "g2", "priority": "bad", "reason_text": long_reason},  # priority 非法 + 超长
    ]})
    out = parse_review_recommend_output(content, CANDIDATE_IDS)
    assert out is not None
    assert out[0]["reason_text"] == ""
    assert len(out[1]["reason_text"]) == 120
    assert out[1]["priority"] == 2  # 非法 priority → 按顺序补


def test_parse_non_json_and_empty_are_none():
    assert parse_review_recommend_output("not-json", CANDIDATE_IDS) is None
    assert parse_review_recommend_output("", CANDIDATE_IDS) is None
    assert parse_review_recommend_output(None, CANDIDATE_IDS) is None
    assert parse_review_recommend_output(json.dumps({"recommendations": []}), CANDIDATE_IDS) is None


def test_build_messages_contains_candidates_and_forbids_adding():
    msgs = build_review_recommend_messages(CANDIDATES)
    assert msgs[0]["role"] == "system"
    assert "不得新增" in msgs[0]["content"]
    assert "g1" in msgs[1]["content"] and "g2" in msgs[1]["content"]


# ---------------------------------------------------------------------------
# 2. 任务状态流转
# ---------------------------------------------------------------------------


def test_task_lifecycle():
    db = FakeDB()
    task = _run(rrt.create_review_recommend_task(db, scholar_id="s1", candidates=CANDIDATES))
    assert task["status"] == "pending"
    assert task["task_id"].startswith("rr_")
    assert task["expires_at"] > task["created_at"]

    assert _run(rrt.claim_task(db, task["task_id"])) is True
    assert _run(rrt.claim_task(db, task["task_id"])) is False  # 已被抢占

    _run(rrt.finish_task(db, task["task_id"], result={"recommendations": [], "strategy": "ai_rank"}))
    got = _run(rrt.get_task(db, task["task_id"]))
    assert got["status"] == "success"
    assert got["result"]["strategy"] == "ai_rank"
    assert got["error"] is None

    # failed 分支：result 置 null
    t2 = _run(rrt.create_review_recommend_task(db, scholar_id="s1", candidates=CANDIDATES))
    _run(rrt.claim_task(db, t2["task_id"]))
    _run(rrt.finish_task(db, t2["task_id"], error={"error_code": "TEST"}))
    got2 = _run(rrt.get_task(db, t2["task_id"]))
    assert got2["status"] == "failed" and got2["result"] is None and got2["error"] == {"error_code": "TEST"}


def test_cleanup_expired():
    db = FakeDB()
    task = _run(rrt.create_review_recommend_task(db, scholar_id="s1", candidates=CANDIDATES))
    # 未过期 → 不删
    assert _run(rrt.cleanup_expired(db)) == 0
    # 过期 → 删除
    assert _run(rrt.cleanup_expired(db, now_ms=task["expires_at"] + 1)) == 1
    assert _run(rrt.get_task(db, task["task_id"])) is None


def test_get_task_missing_returns_none():
    assert _run(rrt.get_task(FakeDB(), "rr_missing")) is None


# ---------------------------------------------------------------------------
# 3. 后台执行器
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_db(monkeypatch):
    db = FakeDB()
    monkeypatch.setattr(rrt, "get_db", lambda: db)
    return db


def _executor_env(db, monkeypatch, llm_result):
    """建任务并打桩 LLM（stub 可为 callable / 字符串）。"""
    task = _run(rrt.create_review_recommend_task(db, scholar_id="s1", candidates=CANDIDATES))
    if callable(llm_result):
        monkeypatch.setattr(
            "services.providers.review_recommend._call_review_recommend_llm", llm_result
        )
    else:
        monkeypatch.setattr(
            "services.providers.review_recommend._call_review_recommend_llm",
            lambda *a, **k: llm_result,
        )
    return task


def test_executor_success(patched_db, monkeypatch):
    content = json.dumps({"recommendations": [
        {"group_id": "g1", "priority": 1, "reason_text": "先复习"},
        {"group_id": "g2", "priority": 2, "reason_text": "其次"},
    ]})
    task = _executor_env(patched_db, monkeypatch, content)
    _run(rrt.run_review_recommend_task(task["task_id"]))
    got = _run(rrt.get_task(patched_db, task["task_id"]))
    assert got["status"] == "success"
    recs = got["result"]["recommendations"]
    assert {r["group_id"] for r in recs} <= set(CANDIDATE_IDS)  # ⊆ 候选集（A2）
    assert got["result"]["strategy"] == "ai_rank"


def test_executor_invalid_output_fails(patched_db, monkeypatch):
    content = json.dumps({"recommendations": [
        {"group_id": "gX", "priority": 1, "reason_text": "自造组"},
        {"group_id": "g1", "priority": 2, "reason_text": "a"},
    ]})
    task = _executor_env(patched_db, monkeypatch, content)
    _run(rrt.run_review_recommend_task(task["task_id"]))
    got = _run(rrt.get_task(patched_db, task["task_id"]))
    assert got["status"] == "failed"
    assert got["error"]["error_code"] == ERR_INVALID_OUTPUT
    assert got["error"]["failure_stage"] == "parse"


def test_executor_llm_unavailable_fails(patched_db, monkeypatch):
    task = _executor_env(patched_db, monkeypatch, None)
    _run(rrt.run_review_recommend_task(task["task_id"]))
    got = _run(rrt.get_task(patched_db, task["task_id"]))
    assert got["status"] == "failed"
    assert got["error"]["error_code"] == ERR_LLM_UNAVAILABLE


def test_executor_empty_candidates_fails(patched_db, monkeypatch):
    task = _run(rrt.create_review_recommend_task(patched_db, scholar_id="s1", candidates=[]))
    _run(rrt.run_review_recommend_task(task["task_id"]))
    got = _run(rrt.get_task(patched_db, task["task_id"]))
    assert got["status"] == "failed"
    assert got["error"]["error_code"] == ERR_INVALID_OUTPUT


def test_executor_skips_if_already_claimed(patched_db, monkeypatch):
    """已被抢占（processing）→ 直接返回，不覆盖状态。"""
    task = _run(rrt.create_review_recommend_task(patched_db, scholar_id="s1", candidates=CANDIDATES))
    assert _run(rrt.claim_task(patched_db, task["task_id"])) is True
    _run(rrt.run_review_recommend_task(task["task_id"]))
    got = _run(rrt.get_task(patched_db, task["task_id"]))
    assert got["status"] == "processing"  # 未被改成终态


# ---------------------------------------------------------------------------
# 4. LLM 包装超时强取消
# ---------------------------------------------------------------------------


def test_llm_timeout_raises(monkeypatch):
    def slow_llm(*a, **k):
        time.sleep(0.2)  # 远超 0.01s 上限

    monkeypatch.setattr(
        "services.providers.review_recommend._call_review_recommend_llm", slow_llm
    )
    with pytest.raises(ReviewRecommendError) as ei:
        _run(call_review_recommend_llm([{"role": "user", "content": "x"}], timeout_seconds=0.01))
    assert ei.value.error_code == ERR_LLM_TIMEOUT
    assert ei.value.failure_stage == "llm"
