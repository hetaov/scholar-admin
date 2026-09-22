"""services/build/llm.py 单测 — 三元组解析 + 客户端缓存 + 调用参数路由。

不触真实 API,monkeypatch 替换 OpenAI 类拦截 chat.completions.create 调用。
"""
from __future__ import annotations

import pytest

from services.build import llm as build_llm


# ---------------------------------------------------------------------------
# Fake OpenAI 客户端:记录 init 参数 + 拦截 chat.completions.create 调用
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content="FAKE_LLM_REPLY"):
        self.content = content


class _FakeChoice:
    def __init__(self, content="FAKE_LLM_REPLY"):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content="FAKE_LLM_REPLY"):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    """拦截 chat.completions.create 调用,记录所有 kwargs。"""

    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeChatNamespace:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    """伪 OpenAI 客户端;init 时记录 api_key/base_url/timeout。"""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self.chat = _FakeChatNamespace()


# ---------------------------------------------------------------------------
# _resolve_provider_config
# ---------------------------------------------------------------------------


def test_resolve_volcano_default(monkeypatch):
    """LLM_PROVIDER=volcano → 返回火山三元组。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "https://volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-123")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://ds")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "ds-model")

    k, b, m = build_llm._resolve_provider_config()
    assert (k, b, m) == ("vk", "https://volcano", "ep-123")


def test_resolve_deepseek_when_enabled(monkeypatch):
    """LLM_PROVIDER=deepseek → 返回 deepseek 三元组。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "https://volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-123")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")

    k, b, m = build_llm._resolve_provider_config()
    assert (k, b, m) == ("dk", "https://api.deepseek.com/v1", "deepseek-chat")


@pytest.mark.parametrize("provider", ["", "unknown", "VOLCANO", "hunyuan", "  deepseek  "])
def test_resolve_falls_back_to_volcano_on_non_deepseek(monkeypatch, provider):
    """留空、未设、未知值、大小写不符、带空格 → 都按 volcano(保守回落)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", provider)
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://ds")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "ds-model")

    assert build_llm._resolve_provider_config() == ("vk", "u", "m")


# ---------------------------------------------------------------------------
# call_llm 调用参数路由
# ---------------------------------------------------------------------------


def test_call_llm_routes_to_deepseek(monkeypatch):
    """LLM_PROVIDER=deepseek 时,call_llm 应使用 deepseek 凭据 + model。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    out = build_llm.call_llm("hi", system_prompt="sys", temperature=0.1, max_tokens=100)

    assert out == "FAKE_LLM_REPLY"
    # 单例按 (api_key, base_url) 缓存
    assert len(build_llm._clients) == 1
    fake_client = next(iter(build_llm._clients.values()))
    # 客户端 init 用 deepseek 凭据
    assert fake_client.init_kwargs["api_key"] == "dk"
    assert fake_client.init_kwargs["base_url"] == "https://api.deepseek.com/v1"
    # create() 参数路由到 deepseek model + 用户的 temperature/max_tokens/messages
    create_kwargs = fake_client.chat.completions.calls[0]
    assert create_kwargs["model"] == "deepseek-chat"
    assert create_kwargs["temperature"] == 0.1
    assert create_kwargs["max_tokens"] == 100
    assert create_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_call_llm_routes_to_volcano(monkeypatch):
    """LLM_PROVIDER=volcano 时,call_llm 应使用火山凭据 + model。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "https://volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-123")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    build_llm.call_llm("hello", temperature=0.5, max_tokens=2048)

    fake_client = next(iter(build_llm._clients.values()))
    assert fake_client.init_kwargs["api_key"] == "vk"
    assert fake_client.init_kwargs["base_url"] == "https://volcano"
    create_kwargs = fake_client.chat.completions.calls[0]
    assert create_kwargs["model"] == "ep-123"
    assert create_kwargs["temperature"] == 0.5
    assert create_kwargs["max_tokens"] == 2048


def test_call_llm_skips_empty_system_prompt(monkeypatch):
    """system_prompt 为空时,messages 只有 user 角色。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    build_llm.call_llm("only user")
    fake_client = next(iter(build_llm._clients.values()))
    assert fake_client.chat.completions.calls[0]["messages"] == [
        {"role": "user", "content": "only user"},
    ]


def test_call_llm_strips_trailing_whitespace_in_response(monkeypatch):
    """call_llm 应 strip 返回内容(与 call_volcano 行为一致)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")

    # 让 FakeClient 返回带空白的 content
    class _StripClient(_FakeClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.chat.completions = _FakeCompletions()
            # 覆盖 create 返回带前后空白的 content
            original_create = self.chat.completions.create

            def _create(**kwargs):
                self.chat.completions.calls.append(kwargs)
                return _FakeResponse("  hi there  ")

            self.chat.completions.create = _create

    monkeypatch.setattr(build_llm, "OpenAI", _StripClient)
    build_llm.reset_clients_for_test()

    out = build_llm.call_llm("anything")
    assert out == "hi there"


def test_call_llm_raises_on_missing_api_key(monkeypatch):
    """deepseek provider 未配 API Key 时应抛 RuntimeError(不静默失败)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    with pytest.raises(RuntimeError, match="deepseek"):
        build_llm.call_llm("hi")


def test_call_llm_raises_on_missing_model(monkeypatch):
    """deepseek provider 未配 CHAT_MODEL 时应抛 RuntimeError。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    with pytest.raises(RuntimeError, match="deepseek"):
        build_llm.call_llm("hi")


def test_client_cached_per_provider(monkeypatch):
    """同一 provider 多次调用只创建一个 client(单例缓存)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    build_llm.call_llm("a")
    build_llm.call_llm("b")
    build_llm.call_llm("c")

    assert len(build_llm._clients) == 1
    fake_client = next(iter(build_llm._clients.values()))
    assert len(fake_client.chat.completions.calls) == 3


def test_client_cached_per_distinct_credentials(monkeypatch):
    """不同 (api_key, base_url) 应创建独立 client。"""
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    # 第一次:volcano
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk1")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u1")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m1")
    build_llm.call_llm("a")

    # 第二次:deepseek(api_key/base_url 不同)
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://ds")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "ds-model")
    build_llm.call_llm("b")

    assert len(build_llm._clients) == 2


def test_reset_clients_for_test_clears_cache(monkeypatch):
    """reset_clients_for_test 应清空客户端缓存。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    build_llm.call_llm("a")
    assert len(build_llm._clients) == 1

    build_llm.reset_clients_for_test()
    assert len(build_llm._clients) == 0


def test_timeout_passed_to_client(monkeypatch):
    """timeout 参数应透传到 OpenAI 客户端 init。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "m")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    build_llm.call_llm("a", timeout=120.0)
    fake_client = next(iter(build_llm._clients.values()))
    assert fake_client.init_kwargs["timeout"] == 120.0


# ---------------------------------------------------------------------------
# acall_llm:异步入口(to_thread 包装同步 _call_llm_with_messages)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acall_llm_returns_content(monkeypatch):
    """acall_llm 应返回 LLM content;messages 原样传给底层 create。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-1")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    out = await build_llm.acall_llm(msgs, temperature=0.0)
    assert out == "FAKE_LLM_REPLY"

    fake_client = next(iter(build_llm._clients.values()))
    call_kwargs = fake_client.chat.completions.calls[0]
    assert call_kwargs["model"] == "ep-1"
    assert call_kwargs["messages"] == msgs
    assert call_kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_acall_llm_uses_deepseek_when_enabled(monkeypatch):
    """LLM_PROVIDER=deepseek 时 acall_llm 走 deepseek 三元组。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "dk")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "https://volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-1")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    msgs = [{"role": "user", "content": "hi"}]
    await build_llm.acall_llm(msgs)
    fake_client = next(iter(build_llm._clients.values()))
    # 应使用 deepseek 凭据(api_key=dk, base_url=deepseek)
    assert fake_client.init_args == ()
    assert fake_client.init_kwargs["api_key"] == "dk"
    assert fake_client.init_kwargs["base_url"] == "https://api.deepseek.com/v1"
    call_kwargs = fake_client.chat.completions.calls[0]
    assert call_kwargs["model"] == "deepseek-chat"


@pytest.mark.asyncio
async def test_acall_llm_missing_creds_raises(monkeypatch):
    """凭据缺失 → RuntimeError(provider=deepseek 但未配 DEEPSEEK_API_KEY)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(build_llm, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(build_llm, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setattr(build_llm, "DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    with pytest.raises(RuntimeError, match="缺少 api_key 或 model 配置"):
        await build_llm.acall_llm([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_acall_llm_propagates_create_exception(monkeypatch):
    """底层 create 抛异常时,acall_llm 应向上抛(供上层重试)。"""

    class _BoomCompletions:
        def create(self, **kwargs):
            raise RuntimeError("upstream 500")

    class _BoomClient:
        def __init__(self, *args, **kwargs):
            self.chat = type(
                "ChatNS", (), {"completions": _BoomCompletions()}
            )()

    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-1")
    monkeypatch.setattr(build_llm, "OpenAI", _BoomClient)
    build_llm.reset_clients_for_test()

    with pytest.raises(RuntimeError, match="upstream 500"):
        await build_llm.acall_llm([{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_acall_llm_empty_content_returns_empty_string(monkeypatch):
    """LLM 返回空 content 时,acall_llm 返回空串(不抛错,由调用方判定)。"""

    class _EmptyResponse:
        choices = [type("C", (), {"message": type("M", (), {"content": None})()})()]

    class _EmptyCompletions:
        def create(self, **kwargs):
            return _EmptyResponse()

    class _EmptyClient:
        def __init__(self, *args, **kwargs):
            self.chat = type("ChatNS", (), {"completions": _EmptyCompletions()})()

    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-1")
    monkeypatch.setattr(build_llm, "OpenAI", _EmptyClient)
    build_llm.reset_clients_for_test()

    out = await build_llm.acall_llm([{"role": "user", "content": "x"}])
    assert out == ""


@pytest.mark.asyncio
async def test_acall_llm_uses_client_cache(monkeypatch):
    """acall_llm 多次调用应复用同一 OpenAI 客户端单例(按 key 缓存)。"""
    monkeypatch.setattr(build_llm, "LLM_PROVIDER", "volcano")
    monkeypatch.setattr(build_llm, "VOLCANO_API_KEY", "vk")
    monkeypatch.setattr(build_llm, "VOLCANO_BASE_URL", "u")
    monkeypatch.setattr(build_llm, "VOLCANO_CHAT_MODEL", "ep-1")
    monkeypatch.setattr(build_llm, "OpenAI", _FakeClient)
    build_llm.reset_clients_for_test()

    msgs = [{"role": "user", "content": "x"}]
    await build_llm.acall_llm(msgs)
    await build_llm.acall_llm(msgs)
    await build_llm.acall_llm(msgs)
    # 3 次调用只用 1 个客户端
    assert len(build_llm._clients) == 1
    fake_client = next(iter(build_llm._clients.values()))
    assert len(fake_client.chat.completions.calls) == 3
