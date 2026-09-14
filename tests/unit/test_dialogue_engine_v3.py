"""单元测试：会话 v3 引擎适配层（设计稿 §11.3 / §11.6）。

不触网不连库（注入 fake generator + FakeDB checkpointer），验证：
- 素材规整：new 组全保留、review 句 clamp 到 [RECALL_MIN, RECALL_MAX]、召回不足留痕
  `recall_insufficient`（§11.3 / §8.4 #4）；
- 图节点纯函数：load_context / build_prompt / generate_reply / persist；
- 端到端 `generate_session_reply`：出口 4 字段对齐 v2；checkpoint 落独立集合
  `ai_session_v3_checkpoint`（与会话历史集合严格隔离，§11.6）；
- 错误映射：批量面 DialogueGenError → v2 会话 SessionGenError（§11.4，不降级不静默）；
- 断点续写：`resume=True` 复用已完成节点。
"""
from __future__ import annotations

import asyncio
import json

from config import SESSION_V3_CHECKPOINT_COLLECTION
from services.learning import dialogue_engine
from services.learning.dialogue_engine import (
    NOTE_RECALL_INSUFFICIENT,
    STAGE_CONTEXT_LOADED,
    STAGE_PERSISTED,
    generate_session_reply,
    get_checkpointer,
    latest_checkpoint_id,
    list_session_checkpoints,
    node_build_prompt,
    node_load_context,
    node_persist,
    normalize_session_materials,
    session_checkpoint_exists,
)
from services.providers.dialogue_gen import (
    ERR_LLM_TIMEOUT,
    ERR_LLM_UNAVAILABLE,
    DialogueGenError,
)
from services.providers.session_gen import (
    ERR_EVAL_UNAVAILABLE,
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT as SG_ERR_LLM_TIMEOUT,
    SessionGenError,
)

SESSION_ID = "s_engine"


def _run(coro):
    return asyncio.run(coro)


def _materials(**overrides) -> list[dict]:
    materials = [
        {
            "kind": "new",
            "sentences": [
                {"sentence_id": "sid_new_1", "content": "I'd like to check in."},
                {"sentence_id": "sid_new_2", "content": "Could I have a window seat?"},
            ],
        },
        {
            "kind": "review",
            "sentences": [
                {"sentence_id": f"sid_rev_{i}", "content": f"Review sentence {i}."}
                for i in range(1, 9)
            ],
        },
    ]
    materials[0].update(overrides.get("new_group", {}))
    return materials


def _context(**overrides) -> dict:
    ctx = {
        "mode": "start",
        "scenario": {"scene": "At the airport check-in counter.", "goal": "Check in"},
        "roles": {
            "ai_role": {"name": "Airport Staff", "style": "friendly"},
            "learner_role": {"name": "Passenger"},
        },
        "materials": _materials(),
        "history": [],
        "user_input": None,
        "assisted": False,
    }
    ctx.update(overrides)
    return ctx


def _llm_json(
    content_type: str = "dialogue",
    *,
    ai_text: str = "Good morning! Where are you flying today?",
    targets: list[str] | None = None,
    hint: dict | None = None,
) -> str:
    return json.dumps(
        {
            "content_type": content_type,
            "ai_text": ai_text,
            "hint": hint,
            "suggested_targets": targets or [],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# 素材规整（召回 clamp [2, 6]）
# ---------------------------------------------------------------------------


class TestNormalizeMaterials:
    def test_new_kept_and_review_clamped_to_max(self):
        materials, required, recalled, notes = normalize_session_materials(_materials())
        # new 组全保留（2 句必用）
        assert required == ["sid_new_1", "sid_new_2"]
        # review 8 句 → clamp 到 RECALL_MAX=6
        assert len(recalled) == 6
        assert recalled == [f"sid_rev_{i}" for i in range(1, 7)]
        # 达标 → 无降级留痕
        assert notes == []
        # 规整后 materials = new 组 + 单条 review 组（≤6）
        assert materials[0]["kind"] == "new"
        assert materials[-1]["kind"] == "review"
        assert len(materials[-1]["sentences"]) == 6

    def test_review_below_min_notes_insufficient(self):
        materials = [
            {
                "kind": "new",
                "sentences": [{"sentence_id": "sid_new_1", "content": "I'd like to check in."}],
            },
            {"kind": "review", "sentences": [{"sentence_id": "sid_rev_1", "content": "One."}]},
        ]
        _, _, recalled, notes = normalize_session_materials(materials)
        assert recalled == ["sid_rev_1"]
        assert notes == [NOTE_RECALL_INSUFFICIENT]

    def test_no_review_notes_insufficient(self):
        materials = [
            {"kind": "new", "sentences": [{"sentence_id": "sid_new_1", "content": "Hi."}]}
        ]
        normalized, _, recalled, notes = normalize_session_materials(materials)
        assert recalled == []
        assert notes == [NOTE_RECALL_INSUFFICIENT]
        # 无 review → 不追加 review 组
        assert all(g["kind"] == "new" for g in normalized)

    def test_missing_kind_defaults_to_new(self):
        materials = [{"sentences": [{"sentence_id": "sid_x", "content": "Hello."}]}]
        _, required, recalled, notes = normalize_session_materials(materials)
        assert required == ["sid_x"]
        assert recalled == []
        assert notes == [NOTE_RECALL_INSUFFICIENT]

    def test_skips_blank_sentences(self):
        materials = [
            {
                "kind": "new",
                "sentences": [
                    {"sentence_id": "", "content": "no id"},
                    {"sentence_id": "sid_ok", "content": "ok"},
                    {"sentence_id": "sid_blank", "content": "   "},
                ],
            }
        ]
        _, required, _, _ = normalize_session_materials(materials)
        assert required == ["sid_ok"]


# ---------------------------------------------------------------------------
# 图节点纯函数
# ---------------------------------------------------------------------------


class TestNodes:
    def test_load_context_sets_stage_and_materials(self):
        out = node_load_context(
            {"context": _context(), "preferred_type": "dialogue", "session_id": SESSION_ID}
        )
        assert out["stage"] == STAGE_CONTEXT_LOADED
        assert out["content_type"] == "dialogue"
        assert out["required_ids"] == ["sid_new_1", "sid_new_2"]
        assert len(out["recall_ids"]) == 6
        assert out["context_notes"] == []

    def test_build_prompt_uses_normalized_materials(self):
        loaded = node_load_context({"context": _context(), "preferred_type": "auto"})
        out = node_build_prompt({**loaded, "context": _context()})
        assert len(out["messages"]) == 2
        assert out["messages"][0]["role"] == "system"
        # 规整后的 review 句（clamp 到 6）应进入 prompt，第 7/8 句不出现
        user_msg = out["messages"][1]["content"]
        assert "sid_rev_6" in user_msg
        assert "sid_rev_7" not in user_msg

    def test_persist_builds_v2_aligned_result(self):
        parsed = {
            "content_type": "fill",
            "ai_text": "Nice to meet you.",
            "hint": {"levels": ["a", "b", "c"], "max_level": 3},
            "suggested_targets": ["sid_new_1"],
        }
        out = node_persist({"parsed": parsed, "session_id": SESSION_ID, "context_notes": []})
        assert out["stage"] == STAGE_PERSISTED
        assert out["result"] == {
            "content_type": "fill",
            "ai_text": "Nice to meet you.",
            "hint": {"levels": ["a", "b", "c"], "max_level": 3},
            "suggested_targets": ["sid_new_1"],
        }

    def test_persist_defaults_targets_empty(self):
        out = node_persist({"parsed": {"content_type": "dialogue", "ai_text": "Hi"}})
        assert out["result"]["suggested_targets"] == []
        assert out["result"]["hint"] is None


# ---------------------------------------------------------------------------
# 端到端：generate_session_reply + checkpoint
# ---------------------------------------------------------------------------


class TestGenerateReply:
    def test_end_to_end_writes_dedicated_checkpoints(self, fake_db):
        async def gen(messages):
            return _llm_json(targets=["sid_new_1"])

        result = _run(
            generate_session_reply(
                db=fake_db,
                session_id=SESSION_ID,
                context=_context(),
                generator=gen,
            )
        )
        assert result["content_type"] == "dialogue"
        assert result["ai_text"].startswith("Good morning!")
        assert result["suggested_targets"] == ["sid_new_1"]
        # hint 缺省 → None（本轮无需作答）
        assert result["hint"] is None

        # checkpoint 落独立集合 ai_session_v3_checkpoint，且与会话图集合隔离
        docs = fake_db.all(SESSION_V3_CHECKPOINT_COLLECTION)
        assert docs
        assert fake_db.all("conversation_graph_checkpoint") == []
        assert fake_db.all("ai_session_v3_checkpoint")  # 与 SESSION_V3_CHECKPOINT_COLLECTION 同名

    def test_checkpoint_timeline_and_exists(self, fake_db):
        async def gen(messages):
            return _llm_json()

        _run(
            generate_session_reply(
                db=fake_db,
                session_id=SESSION_ID,
                context=_context(),
                generator=gen,
            )
        )
        assert _run(session_checkpoint_exists(fake_db, SESSION_ID)) is True
        assert _run(session_checkpoint_exists(fake_db, "s_missing")) is False

        latest = _run(latest_checkpoint_id(fake_db, SESSION_ID))
        assert latest
        assert _run(session_checkpoint_exists(fake_db, SESSION_ID, latest)) is True

        items = _run(list_session_checkpoints(fake_db, SESSION_ID))
        # 旧→新：含 4 个节点阶段 + 入口 start
        stages = [i["stage"] for i in items]
        assert stages[0] == "start"
        assert STAGE_CONTEXT_LOADED in stages
        assert stages[-1] == STAGE_PERSISTED
        ts = [i["ts"] for i in items]
        assert ts == sorted(ts)

    def test_parse_error_raises_session_parse_error(self, fake_db):
        async def gen(messages):
            return "not a json at all"

        try:
            _run(
                generate_session_reply(
                    db=fake_db,
                    session_id=SESSION_ID,
                    context=_context(),
                    generator=gen,
                )
            )
        except SessionGenError as e:
            assert e.error_code == ERR_LLM_PARSE_ERROR
        else:  # pragma: no cover
            raise AssertionError("应当抛出 LLM_PARSE_ERROR")


# ---------------------------------------------------------------------------
# 错误映射（§11.4 语义对齐，不降级不静默）
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def _expect(self, gen_fn, code):
        try:
            _run(
                generate_session_reply(
                    db=None,
                    session_id=SESSION_ID,
                    context=_context(),
                    generator=gen_fn,
                )
            )
        except SessionGenError as e:
            assert e.error_code == code
        else:  # pragma: no cover
            raise AssertionError(f"应当抛出 {code}")

    def test_empty_llm_maps_to_eval_unavailable(self):
        async def gen(messages):
            return None

        self._expect(gen, ERR_EVAL_UNAVAILABLE)

    def test_dialogue_timeout_maps_to_session_timeout(self):
        async def gen(messages):
            raise DialogueGenError(ERR_LLM_TIMEOUT, "llm", "timeout")

        self._expect(gen, SG_ERR_LLM_TIMEOUT)

    def test_dialogue_unavailable_maps_to_eval_unavailable(self):
        async def gen(messages):
            raise DialogueGenError(ERR_LLM_UNAVAILABLE, "llm", "unavailable")

        self._expect(gen, ERR_EVAL_UNAVAILABLE)

    def test_unknown_dialogue_error_maps_to_network_error(self):
        async def gen(messages):
            raise DialogueGenError("SOMETHING_ELSE", "llm", "boom")

        self._expect(gen, "NETWORK_ERROR")


# ---------------------------------------------------------------------------
# 断点续写
# ---------------------------------------------------------------------------


class TestResume:
    def test_resume_reuses_checkpoint(self, fake_db):
        saver = get_checkpointer(fake_db)
        gen_calls = {"n": 0}

        async def gen(messages):
            gen_calls["n"] += 1
            return _llm_json(ai_text="First reply.")

        first = _run(
            generate_session_reply(
                db=fake_db,
                session_id=SESSION_ID,
                context=_context(),
                generator=gen,
                checkpointer=saver,
            )
        )
        assert first["ai_text"] == "First reply."
        assert gen_calls["n"] == 1

        # resume=True：线程已有 checkpoint → 复用已完成节点，不再调用 generator
        second = _run(
            generate_session_reply(
                db=fake_db,
                session_id=SESSION_ID,
                context=_context(),
                generator=gen,
                checkpointer=saver,
                resume=True,
            )
        )
        assert second["ai_text"] == "First reply."
        assert gen_calls["n"] == 1

    def test_resume_without_checkpoint_runs_fresh(self, fake_db):
        saver = get_checkpointer(fake_db)
        gen_calls = {"n": 0}

        async def gen(messages):
            gen_calls["n"] += 1
            return _llm_json(ai_text=f"Reply {gen_calls['n']}.")

        out = _run(
            generate_session_reply(
                db=fake_db,
                session_id="s_fresh",
                context=_context(),
                generator=gen,
                checkpointer=saver,
                resume=True,
            )
        )
        assert out["ai_text"] == "Reply 1."
        assert gen_calls["n"] == 1
