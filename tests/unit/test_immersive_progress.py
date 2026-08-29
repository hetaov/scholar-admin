"""单元测试：沉浸式五步进度持久化服务层 services/learning/immersive_progress

覆盖（对应契约 docs_v2/03-change/proposals/2026-08-29-沉浸式五步进度持久化后端接口.md）：
- validate_progress_payload ：version 匹配 / 主键存在性 / 字段类型 / payload JSON 与 5KB 上限
- get_progress                ：单查命中 / 未命中返回 None
- save_progress               ：新建 / 覆盖（last-write-wins，created_at 保留、updated_at 刷新）
- clear_progress              ：删除成功 / 无记录幂等（deleted=False）
"""
from __future__ import annotations

import asyncio
import json

import pytest

from services.learning.immersive_progress import (
    COLLECTION,
    MAX_PAYLOAD_BYTES,
    PROGRESS_VERSION,
    clear_progress,
    get_progress,
    save_progress,
    validate_progress_payload,
)
from tests.fakes.fake_db import FakeDB


def _run(coro):
    return asyncio.run(coro)


def _mk_body(overrides: dict | None = None) -> dict:
    """合法 PUT 入参 fixture（与前端 serializeSkillProgress 输出对齐）。"""
    body = {
        "version": PROGRESS_VERSION,
        "scholar_id": "scholar_1",
        "textbook_id": "tb_1",
        "group_id": "g_1",
        "sentence_id": "sent_1",
        "challenge_active": True,
        "saved_at": 1724918400000,
        "payload": {
            "skill_flow": {"group_id": "g_1", "current_index": 1, "steps": [], "mastered": False},
            "timeline": [{"code": "ec_translation", "status": "pass"}],
            "challenge_input": "草稿",
            "listening": None,
        },
    }
    if overrides:
        body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# validate_progress_payload
# ---------------------------------------------------------------------------


def test_validate_ok_normalizes_fields():
    fields = validate_progress_payload(_mk_body())
    assert fields["version"] == PROGRESS_VERSION
    assert fields["scholar_id"] == "scholar_1"
    assert fields["sentence_id"] == "sent_1"
    assert fields["challenge_active"] is True
    assert fields["saved_at"] == 1724918400000
    assert fields["payload"]["skill_flow"]["current_index"] == 1
    # payload 原样（不解析内部结构）
    assert fields["payload"] == _mk_body()["payload"]


def test_validate_version_mismatch_rejected():
    with pytest.raises(ValueError, match="version"):
        validate_progress_payload(_mk_body({"version": 0}))
    with pytest.raises(ValueError, match="version"):
        validate_progress_payload(_mk_body({"version": 999}))
    body = _mk_body()
    del body["version"]
    with pytest.raises(ValueError, match="version"):
        validate_progress_payload(body)


@pytest.mark.parametrize("key", ["scholar_id", "textbook_id", "group_id", "sentence_id"])
def test_validate_missing_primary_keys_rejected(key):
    with pytest.raises(ValueError, match=key):
        validate_progress_payload(_mk_body({key: ""}))
    body = _mk_body()
    del body[key]
    with pytest.raises(ValueError, match=key):
        validate_progress_payload(body)


def test_validate_non_bool_challenge_active_rejected():
    with pytest.raises(ValueError, match="challenge_active"):
        validate_progress_payload(_mk_body({"challenge_active": "true"}))
    with pytest.raises(ValueError, match="challenge_active"):
        validate_progress_payload(_mk_body({"challenge_active": 1}))  # int 不可（True 才是 bool）


def test_validate_saved_at_type_rejected():
    with pytest.raises(ValueError, match="saved_at"):
        validate_progress_payload(_mk_body({"saved_at": "now"}))
    with pytest.raises(ValueError, match="saved_at"):
        validate_progress_payload(_mk_body({"saved_at": True}))


def test_validate_payload_not_object_rejected():
    with pytest.raises(ValueError, match="payload"):
        validate_progress_payload(_mk_body({"payload": "raw"}))
    with pytest.raises(ValueError, match="payload"):
        validate_progress_payload(_mk_body({"payload": []}))


def test_validate_payload_over_size_rejected():
    big = {"data": "x" * (MAX_PAYLOAD_BYTES + 100)}
    with pytest.raises(ValueError, match="上限"):
        validate_progress_payload(_mk_body({"payload": big}))


def test_validate_payload_non_json_value_rejected():
    with pytest.raises(ValueError, match="JSON"):
        validate_progress_payload(_mk_body({"payload": {"bad": set()}}))


def test_validate_payload_compact_json_sizing():
    # 恰好小于上限的合法 payload 应通过（5KB 以紧凑 JSON 字节计）
    payload = {"data": "y" * (MAX_PAYLOAD_BYTES - 200)}
    fields = validate_progress_payload(_mk_body({"payload": payload}))
    size = len(json.dumps(fields["payload"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    assert size <= MAX_PAYLOAD_BYTES


# ---------------------------------------------------------------------------
# get_progress
# ---------------------------------------------------------------------------


def test_get_progress_hit_and_miss():
    db = FakeDB()
    db.add(COLLECTION, _mk_body())
    got = _run(get_progress(db, "scholar_1", "tb_1", "g_1"))
    assert got is not None
    assert got["sentence_id"] == "sent_1"
    assert got["payload"]["timeline"][0]["status"] == "pass"
    assert _run(get_progress(db, "scholar_1", "tb_1", "g_2")) is None


# ---------------------------------------------------------------------------
# save_progress
# ---------------------------------------------------------------------------


def test_save_progress_creates_new_doc():
    db = FakeDB()
    doc = _run(save_progress(db, _mk_body()))
    assert doc["sentence_id"] == "sent_1"
    assert doc["challenge_active"] is True
    assert doc["created_at"] == doc["updated_at"]
    stored = db.all(COLLECTION)
    assert len(stored) == 1
    assert stored[0]["scholar_id"] == "scholar_1"


def test_save_progress_overwrites_last_write_wins():
    db = FakeDB()
    _run(save_progress(db, _mk_body()))
    first_created_at = db.all(COLLECTION)[0]["created_at"]

    doc2 = _run(save_progress(db, _mk_body({
        "sentence_id": "sent_2",
        "challenge_active": False,
        "saved_at": 1724918401000,
        "payload": {"timeline": [], "skill_flow": None},
    })))
    stored = db.all(COLLECTION)
    assert len(stored) == 1  # 复合键 upsert，不新增行
    assert stored[0]["sentence_id"] == "sent_2"
    assert stored[0]["challenge_active"] is False
    assert stored[0]["payload"]["timeline"] == []
    assert stored[0]["created_at"] == first_created_at  # 覆盖保留首插时间
    assert stored[0]["updated_at"] >= first_created_at
    assert doc2["sentence_id"] == "sent_2"


def test_save_progress_validation_error_aborts():
    db = FakeDB()
    with pytest.raises(ValueError):
        _run(save_progress(db, _mk_body({"version": 0})))
    assert db.all(COLLECTION) == []  # 校验失败不落库


# ---------------------------------------------------------------------------
# clear_progress
# ---------------------------------------------------------------------------


def test_clear_progress_deletes_existing():
    db = FakeDB()
    db.add(COLLECTION, _mk_body())
    assert _run(clear_progress(db, "scholar_1", "tb_1", "g_1")) is True
    assert db.all(COLLECTION) == []


def test_clear_progress_idempotent_when_missing():
    db = FakeDB()
    assert _run(clear_progress(db, "scholar_1", "tb_1", "g_1")) is False
    assert _run(clear_progress(db, "scholar_1", "tb_1", "g_1")) is False
