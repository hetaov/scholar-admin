"""单元测试：英文语句扩展任务层 services/learning/extension_task.py

逐条对齐 test_translation_task.py：
- create_extension_task：ex_ 前缀 task_id、pending、kind、TTL、error 5 字段
- claim_task：抢占互斥
- finish_task：success/failed 写回
- get_task：命中/未命中
- cleanup_expired：只删过期
- recover_stale_tasks / recover_task_if_stale：卡死自愈
"""
from __future__ import annotations

import asyncio
import time

from services.learning.extension_task import (
    COLLECTION,
    KIND_EVALUATE,
    KIND_EXTRACT,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SUCCESS,
    TASK_TTL_MS,
    build_task_id,
    claim_task,
    cleanup_expired,
    create_extension_task,
    finish_task,
    get_task,
    recover_stale_tasks,
    recover_task_if_stale,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def test_build_task_id_prefix():
    assert build_task_id().startswith("ex_")


def test_create_extension_task_fields():
    db = FakeDB()
    doc = _run(
        create_extension_task(
            db,
            kind=KIND_EXTRACT,
            sentence_id="s1",
            textbook_id="tb1",
            lesson_id="l1",
        )
    )
    assert doc["task_id"].startswith("ex_")
    assert doc["kind"] == KIND_EXTRACT
    assert doc["status"] == STATUS_PENDING
    assert doc["result"] is None
    assert doc["error"] is None
    assert doc["expires_at"] - doc["created_at"] == TASK_TTL_MS
    assert doc["audio_base64"] is None  # 不落库
    assert doc["sentence_id"] == "s1"


def test_create_evaluate_task_has_selected_ids():
    db = FakeDB()
    doc = _run(
        create_extension_task(
            db,
            kind=KIND_EVALUATE,
            task_type="l1_fill",
            selected_ids=["s1#0", "s1#1"],
            points_snapshot=[{"id": "s1#0"}],
        )
    )
    assert doc["kind"] == KIND_EVALUATE
    assert doc["task_type"] == "l1_fill"
    assert doc["selected_ids"] == ["s1#0", "s1#1"]
    assert doc["points_snapshot"] == [{"id": "s1#0"}]


def test_claim_task_mutex():
    db = FakeDB()
    doc = _run(create_extension_task(db, kind=KIND_EXTRACT))
    assert _run(claim_task(db, doc["task_id"])) is True
    assert _run(claim_task(db, doc["task_id"])) is False  # 已被抢占


def test_finish_task_success():
    db = FakeDB()
    doc = _run(create_extension_task(db, kind=KIND_EXTRACT))
    _run(finish_task(db, doc["task_id"], result={"points": []}))
    got = _run(get_task(db, doc["task_id"]))
    assert got["status"] == STATUS_SUCCESS
    assert got["result"] == {"points": []}
    assert got["error"] is None


def test_finish_task_failed_with_error_fields():
    db = FakeDB()
    doc = _run(create_extension_task(db, kind=KIND_EXTRACT))
    error = {
        "error_code": "LLM_TIMEOUT",
        "error_detail": "超时",
        "failure_stage": "llm",
        "llm_timeout_seconds": 120,
        "raw": None,
    }
    _run(finish_task(db, doc["task_id"], error=error))
    got = _run(get_task(db, doc["task_id"]))
    assert got["status"] == STATUS_FAILED
    assert got["error"] == error
    assert got["result"] is None


def test_get_task_miss_returns_none():
    db = FakeDB()
    assert _run(get_task(db, "ex_nonexistent")) is None


def test_cleanup_expired_only_deletes_expired():
    db = FakeDB()
    fresh = _run(create_extension_task(db, kind=KIND_EXTRACT))
    # 手动插一条过期任务
    expired_doc = {
        "task_id": "ex_expired",
        "kind": KIND_EXTRACT,
        "status": STATUS_PENDING,
        "expires_at": 1,
        "created_at": 0,
        "updated_at": 0,
    }
    _run(db.insert(COLLECTION, expired_doc))
    count = _run(cleanup_expired(db))
    assert count == 1
    assert _run(get_task(db, "ex_expired")) is None
    assert _run(get_task(db, fresh["task_id"])) is not None


def test_recover_stale_tasks():
    db = FakeDB()
    doc = _run(create_extension_task(db, kind=KIND_EXTRACT))
    # 置为 processing 且超时
    db._data[COLLECTION][0]["status"] = STATUS_PROCESSING
    db._data[COLLECTION][0]["updated_at"] = 0
    count = _run(recover_stale_tasks(db, timeout_s=1))
    assert count == 1
    got = _run(get_task(db, doc["task_id"]))
    assert got["status"] == STATUS_FAILED
    assert got["error"]["error_code"] == "LLM_TIMEOUT"


def test_recover_task_if_stale_noop_when_not_processing():
    db = FakeDB()
    doc = _run(create_extension_task(db, kind=KIND_EXTRACT))
    assert _run(recover_task_if_stale(db, doc)) is False


# ---------------------------------------------------------------------------
# 开关路由：EXTENSION_USE_LANGGRAPH 控制抽取路径
# ---------------------------------------------------------------------------


def test_extract_routes_to_graph_when_switch_on(monkeypatch):
    """EXTENSION_USE_LANGGRAPH=1 → 调用 run_extract_pipeline_via_graph。"""
    import config
    import services.english.extension as ext_mod
    from services.learning import extension_task as et

    monkeypatch.setattr(config, "EXTENSION_USE_LANGGRAPH", 1)
    called = {"via_graph": False, "legacy": False}
    db = FakeDB()
    monkeypatch.setattr(et, "get_db", lambda: db)

    async def fake_via_graph(db, **k):
        called["via_graph"] = True
        return {"points": [], "meta": {}, "cached": False}

    async def fake_legacy(db, **k):
        called["legacy"] = True
        return {"points": [], "meta": {}, "cached": False}

    # run_extension_task 内部用 `from services.english.extension import ...`，
    # 故需 patch 源模块而非 et 命名空间。
    monkeypatch.setattr(ext_mod, "run_extract_pipeline_via_graph", fake_via_graph)
    monkeypatch.setattr(ext_mod, "run_extract_pipeline", fake_legacy)

    task = _run(create_extension_task(db, kind=KIND_EXTRACT))
    _run(et.run_extension_task(
        task["task_id"], kind=KIND_EXTRACT,
        sentence_id="s1", original="I take part in the meeting.",
    ))
    assert called["via_graph"] is True
    assert called["legacy"] is False


def test_extract_routes_to_legacy_when_switch_off(monkeypatch):
    """EXTENSION_USE_LANGGRAPH=0 → 调用 run_extract_pipeline（一期行为不变）。"""
    import config
    import services.english.extension as ext_mod
    from services.learning import extension_task as et

    monkeypatch.setattr(config, "EXTENSION_USE_LANGGRAPH", 0)
    called = {"via_graph": False, "legacy": False}
    db = FakeDB()
    monkeypatch.setattr(et, "get_db", lambda: db)

    async def fake_via_graph(db, **k):
        called["via_graph"] = True
        return {"points": [], "meta": {}, "cached": False}

    async def fake_legacy(db, **k):
        called["legacy"] = True
        return {"points": [], "meta": {}, "cached": False}

    monkeypatch.setattr(ext_mod, "run_extract_pipeline_via_graph", fake_via_graph)
    monkeypatch.setattr(ext_mod, "run_extract_pipeline", fake_legacy)

    task = _run(create_extension_task(db, kind=KIND_EXTRACT))
    _run(et.run_extension_task(
        task["task_id"], kind=KIND_EXTRACT,
        sentence_id="s1", original="I take part in the meeting.",
    ))
    assert called["via_graph"] is False
    assert called["legacy"] is True
