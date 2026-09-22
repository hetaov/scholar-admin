"""build 模块内部 LLM 抽象 — 只服务 services.build.build_nce

按 LLM_PROVIDER 切换火山 / deepseek;默认 volcano 保持向后兼容。
不依赖 services.build_sentence.call_volcano,避免对其它脚本的耦合。

约束:
- 只服务 build_nce 链路;其它脚本(build_sentence / build_sentence_fixed 等)
  仍走各自的 call_volcano,不受本模块影响。
- LLM_PROVIDER 留空 / 未设 / 非 "deepseek" 都按 volcano 处理(保守回落,
  保证不配置新环境变量时行为与改动前完全一致)。
"""
from __future__ import annotations

import logging

from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_CHAT_MODEL,
    LLM_PROVIDER,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_CHAT_MODEL,
)

logger = logging.getLogger("scholar-admin.build.llm")

# 单例客户端按 (api_key, base_url) 缓存,避免每次 new OpenAI
_clients: dict[tuple[str, str], OpenAI] = {}


def _resolve_provider_config() -> tuple[str, str, str]:
    """返回 (api_key, base_url, model)。

    LLM_PROVIDER == "deepseek"(精确匹配)→ 返回 deepseek 三元组;
    其它任何值(包括留空、未设、"volcano"、未知值)→ 返回火山三元组,
    保证默认行为与改动前完全一致。
    """
    if LLM_PROVIDER == "deepseek":
        return (DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_CHAT_MODEL)
    return (VOLCANO_API_KEY, VOLCANO_BASE_URL, VOLCANO_CHAT_MODEL)


def _get_client(api_key: str, base_url: str, timeout: float = 60.0) -> OpenAI:
    """按 (api_key, base_url) 缓存 OpenAI 客户端单例。"""
    key = (api_key, base_url)
    c = _clients.get(key)
    if c is None:
        c = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        _clients[key] = c
    return c


def _call_llm_with_messages(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout: float = 60.0,
) -> str:
    """统一 chat 调用底层(接收已组装 messages,复用单例客户端缓存)。

    call_llm(同步,prompt + system_prompt)与 acall_llm(异步,messages)
    都委托到这里,避免实现重复。
    """
    api_key, base_url, model = _resolve_provider_config()
    if not (api_key and model):
        raise RuntimeError(
            f"[build.llm] provider={LLM_PROVIDER!r} 缺少 api_key 或 model 配置"
        )
    client = _get_client(api_key, base_url, timeout=timeout)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def call_llm(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout: float = 60.0,
) -> str:
    """统一 chat 调用入口(build_nce 链路用)。

    签名是 build_sentence.call_volcano 的子集(NCE 链路只用这几个参数)。
    调用失败抛 Exception,由 build_nce.generate_content 的 try/except 兜底,
    与原 call_volcano 失败行为一致(异常向上冒泡,被外层捕获)。
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return _call_llm_with_messages(
        messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout
    )


async def acall_llm(
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 8192,
    timeout: float = 60.0,
) -> str:
    """异步入口(适合 await 调用方,如 scripts/ 脚本)。

    内部用 asyncio.to_thread 包装同步 _call_llm_with_messages,复用客户端
    单例缓存,避免引入 AsyncOpenAI 的 event-loop 隔离复杂度。调用失败抛
    Exception,由调用方(plan_window 等)的 try/except 兜底,行为与
    原 _call_hunyuan 失败一致。
    """
    import asyncio

    return await asyncio.to_thread(
        _call_llm_with_messages,
        messages,
        temperature,
        max_tokens,
        timeout,
    )


def reset_clients_for_test() -> None:
    """供单测清空客户端缓存,避免跨用例污染。"""
    _clients.clear()
