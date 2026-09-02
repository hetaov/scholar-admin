"""单元测试：翻译异步任务模型 services/translation_task（ADR-0022 决策 A）

覆盖（对应 docs_v1 §11 步骤 1 / §12 测试要点）：
- create_translation_task ：生成唯一 tr_ 前缀 task_id、初始 pending、字段完整、
  expires_at = created_at + TTL；audio_base64 不落库（None 占位）
- claim_task              ：抢占互斥（第二次 False）；非 pending 状态不可抢占
- finish_task             ：success 写回 result；failed 写回 error 对象（五字段）且 result 置空
- get_task                ：命中返回文档 / 未命中返回 None
- cleanup_expired         ：只删过期任务，未过期保留
- recover_stale_tasks     ：只恢复超时 processing（error=LLM_TIMEOUT 对象）
- recover_task_if_stale   ：查询热路径定点自愈
"""
from __future__ import annotations

import asyncio
import time

from services.translation_task import (
    COLLECTION,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SUCCESS,
    TASK_TTL_MS,
    claim_task,
    cleanup_expired,
    create_translation_task,
    finish_task,
    get_task,
    recover_stale_tasks,
    recover_task_if_stale,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# create_translation_task
# ---------------------------------------------------------------------------


def test_create_translation_task_fields_and_unique_id():
    db = FakeDB()
    doc1 = _run(
        create_translation_task(
            db,
            original_text="It is a watch.",
            input_mode="text",
            mode="ec",
            user_input="它是一块手表。",
        )
    )
    doc2 = _run(
        create_translation_task(
            db,
            original_text="这是一块手表。",
            input_mode="voice",
            mode="ce",
            scholar_id="s1",
            sentence_id="sent_1",
            audio_base64="ZmFrZS1tcDM=",
        )
    )

    assert doc1["task_id"].startswith("tr_")
    assert doc1["task_id"] != doc2["task_id"]  # 唯一性
    assert doc1["status"] == STATUS_PENDING
    assert doc1["input_mode"] == "text"
    assert doc1["mode"] == "ec"
    assert doc1["user_input"] == "它是一块手表。"
    assert doc1["result"] is None
    assert doc1["error"] is None
    assert doc1["expires_at"] - doc1["created_at"] == TASK_TTL_MS
    assert doc1["asr_engine"] == "16k_en"  # 2026-09-02：默认英语引擎（ce/存量任务语义不变）

    # 语音路径：audio_base64 不落库（None 占位），scholar_id/sentence_id 透传
    assert doc2["audio_base64"] is None
    assert doc2["voice_format"] == "mp3"
    assert doc2["input_mode"] == "voice"
    assert doc2["scholar_id"] == "s1"
    assert doc2["sentence_id"] == "sent_1"
    assert doc2["asr_engine"] == "16k_en"

    # 两条均已落库
    stored = db.all(COLLECTION)
    assert len(stored) == 2
    assert {d["task_id"] for d in stored} == {doc1["task_id"], doc2["task_id"]}


def test_create_translation_task_records_zh_asr_engine():
    """英译中语音作答任务记录中文引擎 16k_zh（POST /eval/translate/v2/zh 提交语义）。"""
    db = FakeDB()
    doc = _run(
        create_translation_task(
            db,
            original_text="It is a watch.",
            input_mode="voice",
            mode="ec",
            audio_base64="ZmFrZS1tcDM=",
            asr_engine="16k_zh",
        )
    )
    assert doc["asr_engine"] == "16k_zh"
    stored = db.all(COLLECTION)
    assert stored[0]["asr_engine"] == "16k_zh"


# ---------------------------------------------------------------------------
# claim_task
# ---------------------------------------------------------------------------


def test_claim_task_mutual_exclusion():
    db = FakeDB()
    doc = _run(create_translation_task(db, original_text="Hi", input_mode="text", mode="ec"))

    assert _run(claim_task(db, doc["task_id"])) is True
    # 第二次抢占失败（已 processing）
    assert _run(claim_task(db, doc["task_id"])) is False

    stored = db.all(COLLECTION)[0]
    assert stored["status"] == STATUS_PROCESSING


def test_claim_task_rejects_non_pending():
    db = FakeDB()
    doc = _run(create_translation_task(db, original_text="Hi", input_mode="text", mode="ec"))
    _run(claim_task(db, doc["task_id"]))  # → processing

    assert _run(claim_task(db, doc["task_id"])) is False
    assert db.all(COLLECTION)[0]["status"] == STATUS_PROCESSING


# ---------------------------------------------------------------------------
# finish_task
# ---------------------------------------------------------------------------


def test_finish_task_success_writes_result():
    db = FakeDB()
    doc = _run(create_translation_task(db, original_text="Hi", input_mode="text", mode="ec"))
    result = {
        "transcription": "嗨",
        "status": 5,
        "feedback": "完全正确",
        "confidence": 0.9,
        "raw_model_output": '{"status": 5}',
    }
    _run(claim_task(db, doc["task_id"]))
    _run(finish_task(db, doc["task_id"], result=result))

    stored = db.all(COLLECTION)[0]
    assert stored["status"] == STATUS_SUCCESS
    assert stored["result"] == result
    assert stored["error"] is None


def test_finish_task_failed_writes_error_object_and_clears_result():
    db = FakeDB()
    doc = _run(create_translation_task(db, original_text="Hi", input_mode="text", mode="ec"))
    _run(claim_task(db, doc["task_id"]))
    error = {
        "error_code": "LLM_TIMEOUT",
        "error_detail": "LLM 调用超过 300s 未返回",
        "failure_stage": "llm",
        "llm_timeout_seconds": 300,
        "raw": None,
    }
    _run(finish_task(db, doc["task_id"], error=error))

    stored = db.all(COLLECTION)[0]
    assert stored["status"] == STATUS_FAILED
    assert stored["error"] == error
    assert stored["result"] is None


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


def test_get_task_hit_and_miss():
    db = FakeDB()
    doc = _run(create_translation_task(db, original_text="Hi", input_mode="text", mode="ec"))

    got = _run(get_task(db, doc["task_id"]))
    assert got is not None
    assert got["task_id"] == doc["task_id"]
    assert got["status"] == STATUS_PENDING

    assert _run(get_task(db, "tr_not_exists")) is None


# ---------------------------------------------------------------------------
# cleanup_expired
# ---------------------------------------------------------------------------


def _seed(db, task_id: str, status: str, updated_at: int, **extra) -> dict:
    doc = {
        "task_id": task_id,
        "scholar_id": None,
        "sentence_id": None,
        "original_text": "Hi",
        "user_input": "嗨",
        "audio_base64": None,
        "voice_format": "mp3",
        "input_mode": "text",
        "mode": "ec",
        "status": status,
        "result": None,
        "error": None,
        "created_at": updated_at,
        "updated_at": updated_at,
        "expires_at": updated_at + TASK_TTL_MS,
    }
    doc.update(extra)
    db.add(COLLECTION, doc)
    return doc


def test_cleanup_expired_only_deletes_expired():
    db = FakeDB()
    now = int(time.time() * 1000)

    _seed(db, "tr_expired", STATUS_PENDING, now - 100_000, expires_at=now - 1000)  # 已过期
    _seed(db, "tr_alive", STATUS_SUCCESS, now - 1000, expires_at=now + TASK_TTL_MS)  # 未过期

    deleted = _run(cleanup_expired(db, now_ms=now))
    assert deleted == 1

    remaining = db.all(COLLECTION)
    assert len(remaining) == 1
    assert remaining[0]["task_id"] == "tr_alive"


# ---------------------------------------------------------------------------
# recover_stale_tasks / recover_task_if_stale
# ---------------------------------------------------------------------------


def test_recover_stale_tasks_only_touches_stale_processing():
    db = FakeDB()
    now = int(time.time() * 1000)
    timeout_s = 120

    _seed(db, "tr_stale", STATUS_PROCESSING, now - 130_000)
    _seed(db, "tr_fresh", STATUS_PROCESSING, now - 1_000)
    _seed(db, "tr_pending", STATUS_PENDING, now - 130_000)

    recovered = _run(recover_stale_tasks(db, timeout_s=timeout_s))
    assert recovered == 1

    by_id = {d["task_id"]: d for d in db.all(COLLECTION)}
    assert by_id["tr_stale"]["status"] == STATUS_FAILED
    assert by_id["tr_stale"]["error"]["error_code"] == "LLM_TIMEOUT"
    assert by_id["tr_stale"]["error"]["error_detail"] == "执行超时"
    # 未超时的 processing 与 pending 均不受影响
    assert by_id["tr_fresh"]["status"] == STATUS_PROCESSING
    assert by_id["tr_fresh"]["error"] is None
    assert by_id["tr_pending"]["status"] == STATUS_PENDING


def test_recover_task_if_stale_revives_stale_processing_only():
    db = FakeDB()
    now = int(time.time() * 1000)
    stale = _seed(db, "tr_stale", STATUS_PROCESSING, now - 130_000)
    fresh = _seed(db, "tr_fresh", STATUS_PROCESSING, now - 1_000)

    # 卡死任务被恢复为 failed
    assert _run(recover_task_if_stale(db, stale)) is True
    by_id = {d["task_id"]: d for d in db.all(COLLECTION)}
    assert by_id["tr_stale"]["status"] == STATUS_FAILED
    assert by_id["tr_stale"]["error"]["error_code"] == "LLM_TIMEOUT"
    # 未超时的 processing 即使传入文档也不被误伤
    assert _run(recover_task_if_stale(db, fresh)) is False
    assert by_id["tr_fresh"]["status"] == STATUS_PROCESSING
    assert by_id["tr_fresh"]["error"] is None


def test_recover_task_if_stale_skips_non_processing():
    db = FakeDB()
    now = int(time.time() * 1000)
    pending = _seed(db, "tr_pending", STATUS_PENDING, now - 130_000)
    success = _seed(db, "tr_done", STATUS_SUCCESS, now - 130_000)

    assert _run(recover_task_if_stale(db, pending)) is False
    assert _run(recover_task_if_stale(db, success)) is False
    by_id = {d["task_id"]: d for d in db.all(COLLECTION)}
    assert by_id["tr_pending"]["status"] == STATUS_PENDING
    assert by_id["tr_done"]["status"] == STATUS_SUCCESS
