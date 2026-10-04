"""单元测试：抽取流水线 run_extract_pipeline 的分层行为与缓存 / 失败语义

- 分层：规则候选 → LLM 只回 candidate_id → 映射校验 → 配额/重叠 → audit → rank → id
- 缓存：同 `point_key`（sentence_id + content_hash + prompt_version + model）命中即复用
- force_refresh：跳过缓存重抽，**原地覆盖**同 `point_key` 记录，不新增行
- R8：LLM 全失败 → 规则兜底；兜底也关 → failed + fallback_reason

2026-10-03（修订 2）：原 R6「同句存在 source=manual 时优先且不覆盖」随全局人工校对层
下线而删除 —— 本集合只存 AI 产出（`llm` / `rule`），学习者判断独立落 `extension_review`。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import config
from services.english.extension import run_extract_pipeline
from services.english.extension_candidates import generate_candidates

ORIGINAL = "I will take part in the discussion."


def _run(coro):
    return asyncio.run(coro)


def _llm_reply(candidate_text: str, **kw) -> str:
    """按候选文本找到其 candidate_id，构造一个合法的 LLM 回复。"""
    cid = next(c["candidate_id"] for c in generate_candidates(ORIGINAL) if c["text"] == candidate_text)
    item = {
        "candidate_id": cid,
        "type": "phrase",
        "meaning_zh": "参加",
        "register": "neutral",
        "example_en": f"We {candidate_text} the meeting.",
        "reason": "固定搭配",
        "collocations": [f"{candidate_text} the meeting"],
        "confidence": 0.9,
    }
    item.update(kw)
    return json.dumps({"points": [item]})


def _patch_llm(monkeypatch, replies):
    """依次返回 replies；用完后抛异常模拟调用失败。"""
    calls = {"n": 0}

    async def _fake(messages, timeout_seconds=None):
        i = calls["n"]
        calls["n"] += 1
        if i >= len(replies):
            raise RuntimeError("llm exhausted")
        return replies[i]

    monkeypatch.setattr("services.english.extension.call_extension_llm", _fake)
    return calls


def _add_ai_record(fake_db, sentence_id, points, source="llm", updated_at=1000):
    """插入一条 AI 抽取记录，point_key 与 run_extract_pipeline 一致（可被缓存命中）。"""
    from services.english.extension import extension_idempotency_key
    from services.models.content import compute_text_hash

    content_hash = compute_text_hash(ORIGINAL)
    fake_db.add(
        config.EXTENSION_POINT_COLLECTION,
        {
            "sentence_id": sentence_id,
            "point_key": extension_idempotency_key(
                sentence_id=sentence_id,
                content_hash=content_hash,
                prompt_version=config.EXTENSION_PROMPT_VERSION,
                model=config.EXTENSION_LLM_MODEL,
            ),
            "content_hash": content_hash,
            "source": source,
            "status": "success",
            "points": points,
            "updated_at": updated_at,
        },
    )


def _records(fake_db):
    return fake_db.all(config.EXTENSION_POINT_COLLECTION)


# ---------------------------------------------------------------------------
# 分层流水线
# ---------------------------------------------------------------------------


def test_pipeline_extracts_from_candidates_and_ranks(fake_db, monkeypatch):
    _patch_llm(monkeypatch, [_llm_reply("take part in")])
    result = _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))

    assert result["cached"] is False
    assert result["meta"]["source"] == "llm"
    assert result["meta"]["prompt_version"] == config.EXTENSION_PROMPT_VERSION
    pts = result["points"]
    assert pts and pts[0]["text"] == "take part in"
    # 后端按候选定位 text/span，不信任模型
    assert ORIGINAL[pts[0]["span"][0] : pts[0]["span"][1]] == "take part in"
    assert pts[0]["id"] == "s1#0"
    assert "confidence" in pts[0] and "risk" in pts[0]


def test_pipeline_cache_hit_returns_cached(fake_db, monkeypatch):
    calls = _patch_llm(monkeypatch, [_llm_reply("take part in")])
    _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))
    first = calls["n"]

    second = _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))
    assert second["cached"] is True
    assert calls["n"] == first  # 未再次调用 LLM


def test_pipeline_llm_failure_falls_back_to_rule(fake_db, monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_RULE_FALLBACK_ENABLED", 1)
    _patch_llm(monkeypatch, [])  # 每次调用都抛异常

    result = _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))
    assert result["meta"]["source"] == "rule"
    assert result["points"], "规则兜底应产出 word"
    assert all(p["type"] == "word" for p in result["points"])


def test_pipeline_failed_when_no_points_and_no_fallback(fake_db, monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_RULE_FALLBACK_ENABLED", 0)
    _patch_llm(monkeypatch, [json.dumps({"points": []})])

    result = _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))
    assert result["points"] == []
    assert result["meta"]["fallback_reason"] == "NO_POINTS"
    rec = _records(fake_db)[0]
    assert rec["status"] == "failed"


# ---------------------------------------------------------------------------
# 缓存命中与 force_refresh（2026-10-03 修订 2 后取代原 R6「人工校对优先」）
# ---------------------------------------------------------------------------


def test_cached_ai_record_wins_and_is_not_overwritten(fake_db, monkeypatch):
    cached_points = [{"id": "s1#0", "type": "word", "text": "kept", "meaning_zh": "x"}]
    _add_ai_record(fake_db, "s1", cached_points)
    calls = _patch_llm(monkeypatch, [_llm_reply("take part in")])

    result = _run(run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL))

    assert result["cached"] is True
    assert result["meta"]["source"] == "llm"
    assert result["points"] == cached_points
    assert calls["n"] == 0, "缓存命中时不应调用 LLM"

    # 同句仍只有一条记录（未新增行）
    recs = _records(fake_db)
    assert len(recs) == 1 and recs[0]["source"] == "llm"


def test_force_refresh_success_overwrites_in_place(fake_db, monkeypatch):
    _add_ai_record(fake_db, "s1", [{"id": "s1#0", "type": "word", "text": "kept", "meaning_zh": "x"}])
    _patch_llm(monkeypatch, [_llm_reply("take part in")])

    result = _run(
        run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL, force_refresh=True)
    )

    assert result["cached"] is False
    assert result["meta"]["source"] == "llm"
    recs = _records(fake_db)
    assert len(recs) == 1, "force_refresh 应原地覆盖（同 point_key），不新增记录"
    assert recs[0]["status"] == "success"
    assert recs[0]["points"][0]["text"] == "take part in"


def test_force_refresh_failure_lands_failed_status_in_place(fake_db, monkeypatch):
    _add_ai_record(fake_db, "s1", [{"id": "s1#0", "type": "word", "text": "kept", "meaning_zh": "x"}])
    monkeypatch.setattr(config, "EXTENSION_RULE_FALLBACK_ENABLED", 0)
    _patch_llm(monkeypatch, [])  # 全失败

    result = _run(
        run_extract_pipeline(fake_db, sentence_id="s1", original=ORIGINAL, force_refresh=True)
    )

    assert result["points"] == []
    recs = _records(fake_db)
    # R8：失败不静默 —— 原地改为 failed 并带 fallback_reason，不新增行、不留半张表
    assert len(recs) == 1
    assert recs[0]["status"] == "failed"
    assert recs[0]["points"] == []
    assert recs[0]["fallback_reason"]
