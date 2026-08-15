"""单元测试：对话异步任务模型 services/dialogue_task

覆盖（对应 execution-guide Phase 1 验收）：
- create_task      ：生成唯一 dt_ 前缀 task_id、初始 pending、字段完整、expires_at = created_at + TTL
- claim_task       ：抢占互斥（第二次 False）；非 pending 状态不可抢占
- finish_task      ：success 写回 result/is_question；failed 写回 error 且 result 置空
- get_task         ：命中返回文档 / 未命中返回 None
- cleanup_expired  ：只删过期任务，未过期保留
"""
from __future__ import annotations

import asyncio
import time

from services.dialogue_task import (
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SUCCESS,
    TASK_TTL_MS,
    claim_task,
    cleanup_expired,
    create_task,
    finish_task,
    get_task,
    recover_stale_tasks,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# create_task
# ---------------------------------------------------------------------------


def test_create_task_fields_and_unique_id():
    db = FakeDB()
    doc1 = _run(create_task(db, scholar_id="s1", sentence="Hello"))
    doc2 = _run(create_task(db, scholar_id="s1", sentence="Hello"))

    assert doc1["task_id"].startswith("dt_")
    assert doc1["task_id"] != doc2["task_id"]  # 唯一性
    assert doc1["status"] == STATUS_PENDING
    assert doc1["scholar_id"] == "s1"
    assert doc1["sentence"] == "Hello"
    assert doc1["result"] is None
    assert doc1["is_question"] is None
    assert doc1["error"] is None
    assert doc1["expires_at"] - doc1["created_at"] == TASK_TTL_MS

    # 两条均已落库
    stored = db.all("dialogue_task")
    assert len(stored) == 2
    assert {d["task_id"] for d in stored} == {doc1["task_id"], doc2["task_id"]}


# ---------------------------------------------------------------------------
# claim_task
# ---------------------------------------------------------------------------


def test_claim_task_mutual_exclusion():
    db = FakeDB()
    doc = _run(create_task(db, scholar_id="s1", sentence="Hi"))

    assert _run(claim_task(db, doc["task_id"])) is True
    # 第二次抢占失败（已 processing）
    assert _run(claim_task(db, doc["task_id"])) is False

    stored = db.all("dialogue_task")[0]
    assert stored["status"] == STATUS_PROCESSING


def test_claim_task_rejects_non_pending():
    db = FakeDB()
    doc = _run(create_task(db, scholar_id="s1", sentence="Hi"))
    _run(claim_task(db, doc["task_id"]))  # → processing

    assert _run(claim_task(db, doc["task_id"])) is False
    assert db.all("dialogue_task")[0]["status"] == STATUS_PROCESSING


# ---------------------------------------------------------------------------
# finish_task
# ---------------------------------------------------------------------------


def test_finish_task_success_writes_result():
    db = FakeDB()
    doc = _run(create_task(db, scholar_id="s1", sentence="Hi"))
    result = {
        "type": "qa",
        "statement": "Hi",
        "question": "Hello?",
        "source": "matched",
        "matched_text": "Hi",
    }
    _run(claim_task(db, doc["task_id"]))
    _run(finish_task(db, doc["task_id"], result=result, is_question=False))

    stored = db.all("dialogue_task")[0]
    assert stored["status"] == STATUS_SUCCESS
    assert stored["result"] == result
    assert stored["is_question"] is False
    assert stored["error"] is None


def test_finish_task_failed_writes_error_and_clears_result():
    db = FakeDB()
    doc = _run(create_task(db, scholar_id="s1", sentence="Hi"))
    _run(claim_task(db, doc["task_id"]))
    _run(finish_task(db, doc["task_id"], error="LLM 调用超时"))

    stored = db.all("dialogue_task")[0]
    assert stored["status"] == STATUS_FAILED
    assert stored["error"] == "LLM 调用超时"
    assert stored["result"] is None


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


def test_get_task_hit_and_miss():
    db = FakeDB()
    doc = _run(create_task(db, scholar_id="s1", sentence="Hi"))

    got = _run(get_task(db, doc["task_id"]))
    assert got is not None
    assert got["task_id"] == doc["task_id"]
    assert got["status"] == STATUS_PENDING

    assert _run(get_task(db, "dt_not_exists")) is None


# ---------------------------------------------------------------------------
# cleanup_expired
# ---------------------------------------------------------------------------


def test_cleanup_expired_only_deletes_expired():
    db = FakeDB()
    now = int(time.time() * 1000)

    db.add(
        "dialogue_task",
        {
            "task_id": "dt_expired",
            "scholar_id": "s1",
            "sentence": "Hi",
            "status": "pending",
            "result": None,
            "is_question": None,
            "error": None,
            "created_at": now - 100_000,
            "updated_at": now - 100_000,
            "expires_at": now - 1000,  # 已过期
        },
    )
    db.add(
        "dialogue_task",
        {
            "task_id": "dt_alive",
            "scholar_id": "s2",
            "sentence": "Bye",
            "status": "success",
            "result": {"type": "qa"},
            "is_question": False,
            "error": None,
            "created_at": now - 1000,
            "updated_at": now - 1000,
            "expires_at": now + TASK_TTL_MS,  # 未过期
        },
    )

    deleted = _run(cleanup_expired(db, now_ms=now))
    assert deleted == 1

    remaining = db.all("dialogue_task")
    assert len(remaining) == 1
    assert remaining[0]["task_id"] == "dt_alive"


# ---------------------------------------------------------------------------
# recover_stale_tasks
# ---------------------------------------------------------------------------


def _seed(db, task_id: str, status: str, updated_at: int, **extra) -> dict:
    doc = {
        "task_id": task_id,
        "scholar_id": "s1",
        "sentence": "Hi",
        "status": status,
        "result": None,
        "is_question": None,
        "error": None,
        "created_at": updated_at,
        "updated_at": updated_at,
        "expires_at": updated_at + TASK_TTL_MS,
    }
    doc.update(extra)
    db.add("dialogue_task", doc)
    return doc


def test_recover_stale_tasks_only_touches_stale_processing():
    db = FakeDB()
    now = int(time.time() * 1000)
    timeout_s = 120

    stale = _seed(db, "dt_stale", STATUS_PROCESSING, now - 130_000)
    fresh = _seed(db, "dt_fresh", STATUS_PROCESSING, now - 1_000)
    pending = _seed(db, "dt_pending", STATUS_PENDING, now - 130_000)

    recovered = _run(recover_stale_tasks(db, timeout_s=timeout_s))
    assert recovered == 1

    by_id = {d["task_id"]: d for d in db.all("dialogue_task")}
    assert by_id["dt_stale"]["status"] == STATUS_FAILED
    assert by_id["dt_stale"]["error"] == "执行超时"
    # 未超时的 processing 与 pending 均不受影响
    assert by_id["dt_fresh"]["status"] == STATUS_PROCESSING
    assert by_id["dt_fresh"]["error"] is None
    assert by_id["dt_pending"]["status"] == STATUS_PENDING
