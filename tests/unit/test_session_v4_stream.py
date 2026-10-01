"""定向测试：AI 会话 v4 伪流式（S6-A/B/C，contract §3.17 / 调研报告 §5.2）

覆盖链路（不触真实火山）：
- `extract_ai_text_prefix`：未闭合 JSON 容错提取 `ai_text` 前缀 / 静默降级；
- `_default_stream_generator`：SSE 逐帧累积 → 增量回调递增 → 返回完整 raw（同同步生成器语义）；
- `invoke_dialogue_llm(on_delta=)`：未注入 generator 时走流式；注入 generator 时不流式（v3 路径不变）；
- `run_session_task`：增量落库递增 + `partial_seq` 单调 + 终态清 `partial_text` + `timings` 完整。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from services.learning import session_task_v4
from services.learning.session_task_v4 import run_session_task
from services.providers import dialogue_gen
from tests.fakes.seed_factory import seed_ai_session_v4

SCHOLAR = "scholar_1"


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# fake SSE（替换 httpx.AsyncClient，避免真实外呼）
# ---------------------------------------------------------------------------


def sse(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]})


class _FakeResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b"upstream error"


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResponse):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *args):
        return False


class _FakeAsyncClient:
    resp: _FakeResponse

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method, url, **kwargs):
        return _FakeStreamCtx(type(self).resp)


@pytest.fixture(autouse=True)
def _volcano_credentials(monkeypatch):
    monkeypatch.setattr(dialogue_gen, "VOLCANO_API_KEY", "test-key")
    monkeypatch.setattr(dialogue_gen, "VOLCANO_CHAT_MODEL", "test-model")


@pytest.fixture(autouse=True)
def _reset_partial_state():
    session_task_v4._partial_state.clear()
    yield
    session_task_v4._partial_state.clear()


@pytest.fixture(autouse=True)
def _fake_get_db(monkeypatch, fake_db):
    """执行器内部 `get_db()` 指向 FakeDB（执行器链路断言用）。"""
    monkeypatch.setattr(session_task_v4, "get_db", lambda: fake_db)


# ---------------------------------------------------------------------------
# S6-A：增量提取（未闭合 JSON 容错）
# ---------------------------------------------------------------------------


class TestExtractAiTextPrefix:
    def test_closed_json_returns_full_value(self):
        raw = '{"ai_text": "Hello there!", "hint": null}'
        assert dialogue_gen.extract_ai_text_prefix(raw) == "Hello there!"

    def test_unclosed_json_returns_prefix(self):
        assert dialogue_gen.extract_ai_text_prefix('{"ai_text": "Hello the') == "Hello the"

    def test_missing_key_returns_none(self):
        assert dialogue_gen.extract_ai_text_prefix('{"content_type": "dia') is None

    def test_empty_buffer_returns_none(self):
        assert dialogue_gen.extract_ai_text_prefix("") is None

    def test_dangling_escape_dropped(self):
        # 尾部悬空转义符：等下一个 chunk 补齐，不输出半个转义序列
        assert dialogue_gen.extract_ai_text_prefix('{"ai_text": "Hello\\') == "Hello"

    def test_escaped_chars_decoded(self):
        assert dialogue_gen.extract_ai_text_prefix('{"ai_text": "a\\nb\\"c"') == 'a\nb"c'


# ---------------------------------------------------------------------------
# S6-A：流式生成器
# ---------------------------------------------------------------------------


class TestDefaultStreamGenerator:
    def test_deltas_incremental_and_raw_complete(self, monkeypatch):
        lines = [
            sse('{"ai_text": "Hello'),
            sse(" there, welcome"),
            sse(' aboard!"}'),
            "data: [DONE]",
        ]

        class Client(_FakeAsyncClient):
            resp = _FakeResponse(lines)

        monkeypatch.setattr(httpx, "AsyncClient", Client)

        deltas: list[str] = []

        async def on_delta(piece: str) -> None:
            deltas.append(piece)

        raw = _run(dialogue_gen._default_stream_generator([{"role": "user", "content": "hi"}], on_delta))

        assert deltas == ["Hello", " there, welcome", " aboard!"]
        assert raw == '{"ai_text": "Hello there, welcome aboard!"}'

    def test_async_callback_is_awaited(self, monkeypatch):
        class Client(_FakeAsyncClient):
            resp = _FakeResponse([sse('{"ai_text": "Hi'), sse(' there"}]'), "data: [DONE]"])

        monkeypatch.setattr(httpx, "AsyncClient", Client)

        calls: list[str] = []

        async def on_delta(piece: str) -> None:
            await asyncio.sleep(0)  # 异步回调：必须被 await，否则不落库
            calls.append(piece)

        _run(dialogue_gen._default_stream_generator([], on_delta))
        assert calls == ["Hi", " there"]

    def test_silent_degrade_when_key_absent(self, monkeypatch):
        class Client(_FakeAsyncClient):
            resp = _FakeResponse([sse('{"content_type": "dia'), sse('logue"}'), "data: [DONE]"])

        monkeypatch.setattr(httpx, "AsyncClient", Client)

        deltas: list[str] = []
        raw = _run(dialogue_gen._default_stream_generator([], deltas.append))

        assert deltas == []  # 提取失败 → 静默降级，不回调
        assert raw == '{"content_type": "dialogue"}'

    def test_non_200_returns_none(self, monkeypatch):
        class Client(_FakeAsyncClient):
            resp = _FakeResponse([sse("x")], status_code=500)

        monkeypatch.setattr(httpx, "AsyncClient", Client)
        assert _run(dialogue_gen._default_stream_generator([], lambda _x: None)) is None

    def test_missing_credentials_returns_none(self, monkeypatch):
        monkeypatch.setattr(dialogue_gen, "VOLCANO_API_KEY", "")
        assert _run(dialogue_gen._default_stream_generator([], lambda _x: None)) is None

    def test_upstream_exception_returns_none(self, monkeypatch):
        class BoomClient(_FakeAsyncClient):
            def stream(self, method, url, **kwargs):
                raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
        assert _run(dialogue_gen._default_stream_generator([], lambda _x: None)) is None


class TestInvokeDialogueLlmOnDelta:
    def test_on_delta_switches_to_stream_generator(self, monkeypatch):
        class Client(_FakeAsyncClient):
            resp = _FakeResponse([sse('{"ai_text": "Yo"}'), "data: [DONE]"])

        monkeypatch.setattr(httpx, "AsyncClient", Client)

        deltas: list[str] = []
        raw = asyncio.run(
            dialogue_gen.invoke_dialogue_llm(
                [{"role": "user", "content": "hi"}], on_delta=deltas.append
            )
        )
        assert raw == '{"ai_text": "Yo"}'
        assert deltas == ["Yo"]

    def test_injected_generator_wins_no_stream(self, monkeypatch):
        """注入 generator（单测 fake）时不流式：on_delta 不被调用，返回语义不变。"""

        async def fake_gen(messages):
            return '{"ai_text": "fake"}'

        deltas: list[str] = []
        raw = asyncio.run(
            dialogue_gen.invoke_dialogue_llm(
                [], generator=fake_gen, on_delta=deltas.append
            )
        )
        assert raw == '{"ai_text": "fake"}'
        assert deltas == []

    def test_no_on_delta_keeps_sync_default(self, monkeypatch):
        async def fake_default(messages):
            return "sync-raw"

        monkeypatch.setattr(dialogue_gen, "_default_generator", fake_default)
        assert asyncio.run(dialogue_gen.invoke_dialogue_llm([])) == "sync-raw"


# ---------------------------------------------------------------------------
# S6-C：执行器接线（增量递增 / 终态清 partial / timings）
# ---------------------------------------------------------------------------


def _create_task(fake_db, task_id: str, stream: bool = True) -> dict:
    return _run(
        session_task_v4.create_session_task(
            fake_db,
            task_id=task_id,
            scholar_id=SCHOLAR,
            session_id="s_test",
            mode="turn",
            preferred_type="auto",
            context={
                "mode": "turn",
                "materials": [],
                "scenario": {"scene": "airport", "title": "t", "scene_id": "a"},
                "roles": {},
                "history": [],
                "user_input": "hi",
                "assisted": False,
            },
            stream=stream,
        )
    )


class TestRunSessionTaskStream:
    def test_partial_incremental_then_cleared(self, monkeypatch, fake_db):
        monkeypatch.setattr(session_task_v4, "SESSION_V4_STREAM_ENABLED", 1)
        monkeypatch.setattr(session_task_v4, "SESSION_V4_PARTIAL_THROTTLE_MS", 0)
        seed_ai_session_v4(fake_db)

        CANNED = {
            "content_type": "dialogue",
            "ai_text": "Hello there, welcome aboard!",
            "hint": None,
            "suggested_targets": [],
        }
        snapshots: list[tuple[str | None, int]] = []

        async def fake_generate(**kwargs):
            on_delta = kwargs.get("on_delta")
            assert on_delta is not None, "stream=true + 开关开启时必须注入增量回调"
            for piece in ("Hello ", "there, ", "welcome ", "aboard!"):
                await on_delta(piece)
                task = [t for t in fake_db.all("ai_session_v4_task") if t["task_id"] == "st_stream"][0]
                snapshots.append((task.get("partial_text"), task.get("partial_seq") or 0))
            return CANNED

        monkeypatch.setattr(
            "services.learning.dialogue_engine.generate_session_reply", fake_generate
        )

        _create_task(fake_db, "st_stream", stream=True)
        _run(run_session_task("st_stream"))

        # 增量递增：每段都是上一段的前缀扩展
        texts = [s[0] for s in snapshots]
        assert texts == [
            "Hello ",
            "Hello there, ",
            "Hello there, welcome ",
            "Hello there, welcome aboard!",
        ]
        # partial_seq 单调递增
        seqs = [s[1] for s in snapshots]
        assert seqs == sorted(seqs) and seqs[-1] > seqs[0]

        # 终态：清增量脏字段，result 结构与 v3 一致
        stored = [t for t in fake_db.all("ai_session_v4_task") if t["task_id"] == "st_stream"][0]
        assert stored["status"] == "success"
        assert stored["partial_text"] is None
        assert stored["result"]["ai_text"] == CANNED["ai_text"]
        assert set(stored["result"].keys()) == {
            "session_id",
            "content_type",
            "ai_text",
            "hint",
            "suggested_targets",
        }

        # timings 完整（含伪流式首帧埋点）
        timings = stored["timings"]
        for key in ("submitted_at", "claimed_at", "llm_total_ms", "total_ms"):
            assert key in timings
        assert timings["partial_first_ms"] >= 0
        assert timings["partial_first_ms"] <= timings["total_ms"]

    def test_stream_off_no_partial_and_no_callback(self, monkeypatch, fake_db):
        """stream=false（或开关关闭）→ 不注入回调、全程不写 partial（路径 ≡ v3）。"""
        seed_ai_session_v4(fake_db)

        seen: dict[str, Any] = {}

        async def fake_generate(**kwargs):
            seen["on_delta"] = kwargs.get("on_delta")
            return {
                "content_type": "dialogue",
                "ai_text": "no stream",
                "hint": None,
                "suggested_targets": [],
            }

        monkeypatch.setattr(
            "services.learning.dialogue_engine.generate_session_reply", fake_generate
        )

        _create_task(fake_db, "st_plain", stream=False)
        _run(run_session_task("st_plain"))

        assert seen["on_delta"] is None
        stored = [t for t in fake_db.all("ai_session_v4_task") if t["task_id"] == "st_plain"][0]
        assert stored["status"] == "success"
        assert stored["partial_text"] is None
        assert stored["partial_seq"] == 0
        assert "partial_first_ms" not in (stored["timings"] or {})

    def test_stream_on_but_service_switch_off(self, monkeypatch, fake_db):
        """请求 stream=true 但服务级开关关闭 → 仍不产增量（双开关门控）。"""
        monkeypatch.setattr(session_task_v4, "SESSION_V4_STREAM_ENABLED", 0)
        seed_ai_session_v4(fake_db)

        seen: dict[str, Any] = {}

        async def fake_generate(**kwargs):
            seen["on_delta"] = kwargs.get("on_delta")
            return {
                "content_type": "dialogue",
                "ai_text": "switch off",
                "hint": None,
                "suggested_targets": [],
            }

        monkeypatch.setattr(
            "services.learning.dialogue_engine.generate_session_reply", fake_generate
        )

        _create_task(fake_db, "st_off", stream=True)
        _run(run_session_task("st_off"))

        assert seen["on_delta"] is None
        stored = [t for t in fake_db.all("ai_session_v4_task") if t["task_id"] == "st_off"][0]
        assert stored["partial_text"] is None

    def test_failure_clears_partial_and_keeps_error(self, monkeypatch, fake_db):
        """生成失败：终态 failed，partial 脏字段清空，error 保留（不降级、不静默）。"""
        monkeypatch.setattr(session_task_v4, "SESSION_V4_STREAM_ENABLED", 1)
        monkeypatch.setattr(session_task_v4, "SESSION_V4_PARTIAL_THROTTLE_MS", 0)
        seed_ai_session_v4(fake_db)

        async def fake_generate(**kwargs):
            await kwargs["on_delta"]("partial before failure")
            raise RuntimeError("connection reset")

        monkeypatch.setattr(
            "services.learning.dialogue_engine.generate_session_reply", fake_generate
        )

        _create_task(fake_db, "st_fail")
        _run(run_session_task("st_fail"))

        stored = [t for t in fake_db.all("ai_session_v4_task") if t["task_id"] == "st_fail"][0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "NETWORK_ERROR"
        assert stored["partial_text"] is None
        # 失败不污染 history
        sess = [s for s in fake_db.all("ai_session_v4") if s["session_id"] == "s_test"][0]
        assert sess["pending_task"] is None
