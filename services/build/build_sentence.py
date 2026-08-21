"""教材语句生成服务 — 基于 LangGraph + 火山模型自动生成教材单元和语句

工作流：
1. generate_content — 调用火山模型，按单元输出所有需要掌握的英文语句
2. 返回结构化 JSON 供外部路由层写入数据库
"""

from __future__ import annotations

import json as json_lib
import logging
from typing import TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, END

from config import (
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
    VOLCANO_CHAT_MODEL,
)

logger = logging.getLogger("scholar-admin.build-sentence")

# ---------------------------------------------------------------------------
# 客户端
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL)
    return _client


def call_volcano(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> str:
    client = _get_client()
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=VOLCANO_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content or ""
    return content.strip()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class BuildState(TypedDict):
    textbook_name: str       # 教材名称（如"四年级英语上册 秋天 广州版"）
    generated_json: str | None  # 模型输出的 JSON 原文
    content: dict | None     # 解析后的结构化内容 {units: [...], textbook_info: {...}}
    error: str | None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """你是一位小学英语教材编辑专家，非常熟悉广州版（教科版）小学英语教材。
你的任务是根据教材名称，生成该教材每个单元所有需要掌握的英语核心句型/语句。

要求：
1. 输出必须是合法的 JSON，不要包含 markdown 代码块标记
2. 按单元组织，每个单元 5-8 句核心语句
3. 每句话包含：英文原文(text)、中文翻译(translation)
4. 广州版四年级上册通常有 6 个单元（Module），外加一个复习单元
5. 语句难度要符合四年级小学生水平（CEFR A1-A2）
6. 不输出图片中可能存在的语句，而是输出该教材该单元实际会出现的核心句型"""


def _build_user_prompt(textbook_name: str) -> str:
    return f"""请为教材"{textbook_name}"生成每个单元的核心语句。

输出 JSON 结构如下：
{{
  "textbook_info": {{
    "name": "教材全称",
    "grade": "四年级",
    "semester": "上册",
    "edition": "广州版"
  }},
  "units": [
    {{
      "unit_index": 1,
      "unit_title": "Module 1: ...",
      "topic": "本单元主题（中文）",
      "sentences": [
        {{"text": "英文语句", "translation": "中文翻译"}},
        ...
      ]
    }},
    ...
  ]
}}

请直接输出 JSON，不要带任何 markdown 标记。"""


# ---------------------------------------------------------------------------
# JSON 修复 — 处理 LLM 输出的常见格式问题
# ---------------------------------------------------------------------------


def _repair_json(raw: str) -> str:
    """修复模型输出 JSON 的常见语法问题：尾部逗号、不完整等"""
    import re

    raw = raw.strip()

    # 1. 去掉行内注释 // ... （单行）
    raw = re.sub(r'//[^\n]*', '', raw)

    # 2. 移除尾部逗号：,"key" 或 ,"value" 前面不应有孤立逗号
    raw = re.sub(r',(\s*[}\]])', r'\1', raw)

    # 3. 在对象键的 }" 之间补充逗号：}"  -> },"  （中间夹着换行/空格）
    raw = re.sub(r'}(\s*\n\s*)"', r'},\n"', raw)

    # 4. 在 ]" 之间补充逗号（少见）
    raw = re.sub(r'](\s*\n\s*)"', r'],\n"', raw)

    # 5. 移除可能的多余逗号在末尾
    raw = re.sub(r',\s*$', '', raw)

    return raw


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def generate_content(state: BuildState) -> BuildState:
    """调用火山模型生成教材结构化内容"""
    logger.info(f"[build] 开始生成教材内容: {state['textbook_name']}")

    try:
        raw = call_volcano(
            prompt=_build_user_prompt(state["textbook_name"]),
            system_prompt=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=8192,
        )
        # 清理可能的 markdown 标记
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            # 去掉第一行 ```json 和最后一行 ```
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        content = json_lib.loads(_repair_json(raw))
        state["generated_json"] = raw
        state["content"] = content
        state["error"] = None
        logger.info(
            f"[build] 生成成功: {content['textbook_info']['name']}, "
            f"共 {len(content['units'])} 个单元"
        )
    except json_lib.JSONDecodeError as e:
        logger.error(f"[build] JSON 解析失败: {e}, raw={raw[:500]}")
        state["error"] = f"模型输出非 JSON: {e}"
        state["content"] = None
    except Exception as e:
        logger.error(f"[build] 生成失败: {e}")
        state["error"] = str(e)
        state["content"] = None

    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    graph = StateGraph(BuildState)

    graph.add_node("generate_content", generate_content)
    graph.set_entry_point("generate_content")
    graph.add_edge("generate_content", END)

    return graph


_app = build_graph().compile()


async def build_textbook_sentences(textbook_name: str) -> dict:
    """执行教材语句生成工作流

    Returns:
        {"success": bool, "content": dict | None, "error": str | None}
        其中 content 包含 textbook_info + units 结构化数据
    """
    state: BuildState = {
        "textbook_name": textbook_name,
        "generated_json": None,
        "content": None,
        "error": None,
    }
    result = _app.invoke(state)

    if result["error"]:
        return {"success": False, "content": None, "error": result["error"]}

    return {"success": True, "content": result["content"], "error": None}
