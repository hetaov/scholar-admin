"""对话匹配服务 — 基于 LangGraph + 火山方舟模型的问答匹配工作流

工作流：
1. classify_input — 判断输入句是陈述句还是疑问句
2. match_from_learned — 从已学语句中寻找匹配的问答对
3. check_natural_qa — 判断问答对是否符合日常用语习惯
4. generate — 不符合时，针对输入生成合适的问句/答句
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

logger = logging.getLogger("scholar-admin.dialogue")

# ---------------------------------------------------------------------------
# 火山方舟对话模型客户端（单例）
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """获取火山方舟 OpenAI 兼容客户端"""
    global _client
    if _client is None:
        _client = OpenAI(api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL)
    return _client


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class DialogueState(TypedDict):
    """LangGraph 工作流状态"""

    input_sentence: str
    scholar_id: str
    learned_sentences: list[dict]  # [{"text": str, "translation": str}, ...]
    is_question: bool
    matched_text: str | None
    is_natural: bool
    generated_text: str | None
    final_pair: dict | None
    error: str | None


# ---------------------------------------------------------------------------
# callVolcano — 调用火山方舟对话模型
# ---------------------------------------------------------------------------


def call_volcano(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """通过火山方舟 OpenAI 兼容接口调用对话模型"""
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


def _parse_json_response(text: str) -> dict:
    """从模型回复中提取 JSON（兼容 markdown ```json 包裹）"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉首行 ```json 和末行 ```
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json_lib.loads(text)


# ---------------------------------------------------------------------------
# LangGraph 节点
# ---------------------------------------------------------------------------


async def node_classify_input(state: DialogueState) -> dict:
    """节点 1：判断输入句是陈述句还是疑问句"""
    input_sentence = state["input_sentence"]

    prompt = f"""请判断以下英文句子是"陈述句(statement)"还是"疑问句(question)"。

句子："{input_sentence}"

请只返回一个 JSON 对象，不要有任何其他文字：
{{"type": "statement"}} 或 {{"type": "question"}}"""

    try:
        reply = call_volcano(prompt)
        result = _parse_json_response(reply)
        is_question = result.get("type") == "question"
        logger.info(f"[classify] 输入句类型: {'疑问句' if is_question else '陈述句'}")
        return {"is_question": is_question}
    except Exception as e:
        logger.error(f"[classify] 分类失败: {e}")
        # 兜底：按末尾问号判断
        stripped = input_sentence.strip()
        is_question = stripped.endswith("?")
        return {"is_question": is_question}


async def node_match_from_learned(state: DialogueState) -> dict:
    """节点 2：从已学语句中匹配最合适的问答对"""
    input_sentence = state["input_sentence"]
    is_question = state["is_question"]
    learned = state["learned_sentences"]

    if not learned:
        logger.warning("[match] 已学语句列表为空，跳过匹配")
        return {"matched_text": None}

    # 构建已学语句列表（仅保留 text，最多 50 条）
    sentence_texts = [s.get("text", "") for s in learned[:50]]
    sentences_joined = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sentence_texts) if t)

    if not sentences_joined:
        return {"matched_text": None}

    if is_question:
        instruction = (
            "输入是一个英文疑问句，请从以下已学语句中找到**最合适的一句陈述句作为回答**。"
        )
    else:
        instruction = (
            "输入是一个英文陈述句，请从以下已学语句中找到**最合适的一句可以作为该陈述句的提问**"
            "（应是一个以 what、where、when、who、how、why、is、are、do、does、did、can 等开头的特殊疑问句或一般疑问句）。"
        )

    prompt = f"""{instruction}

输入句子："{input_sentence}"

已学语句列表：
{sentences_joined}

请只返回一个 JSON 对象，不要有任何其他文字。如果找不到合适的匹配，返回 null：
{{"matched_index": <数字>}}  或  {{"matched_index": null}}"""

    try:
        reply = call_volcano(prompt)
        result = _parse_json_response(reply)
        idx = result.get("matched_index")
        if idx and 1 <= idx <= len(sentence_texts):
            matched = sentence_texts[idx - 1]
            logger.info(f"[match] 匹配到第 {idx} 句: {matched[:50]}...")
            return {"matched_text": matched}
        logger.info("[match] 未找到合适匹配")
        return {"matched_text": None}
    except Exception as e:
        logger.error(f"[match] 匹配失败: {e}")
        return {"matched_text": None}


async def node_check_natural_qa(state: DialogueState) -> dict:
    """节点 3：判断问答对是否符合日常用语问答习惯"""
    input_sentence = state["input_sentence"]
    matched_text = state["matched_text"]
    is_question = state["is_question"]

    if not matched_text:
        logger.info("[natural] 无匹配句，跳过自然度检查")
        return {"is_natural": False}

    if is_question:
        question, answer = input_sentence, matched_text
    else:
        answer, question = input_sentence, matched_text

    prompt = f"""请判断以下英文问答对是否符合日常英语对话习惯（自然、合理、匹配）。

问句："{question}"
答句："{answer}"

请只返回一个 JSON 对象，不要有任何其他文字：
{{"natural": true, "reason": "简短理由"}} 或 {{"natural": false, "reason": "简短理由"}}"""

    try:
        reply = call_volcano(prompt)
        result = _parse_json_response(reply)
        is_natural = result.get("natural", False)
        logger.info(f"[natural] 结果={is_natural}, 理由={result.get('reason', '')}")
        return {"is_natural": is_natural}
    except Exception as e:
        logger.error(f"[natural] 判断失败: {e}")
        # 兜底：有匹配就认为自然
        return {"is_natural": True}


async def node_generate(state: DialogueState) -> dict:
    """节点 4：针对输入句生成合适的问句或答句"""
    input_sentence = state["input_sentence"]
    is_question = state["is_question"]

    if is_question:
        prompt = f"""请为以下英文疑问句生成一个简单、自然的英文回答（1-2 句即可）。

疑问句："{input_sentence}"

请只返回一个 JSON 对象，不要有任何其他文字：
{{"text": "你的回答"}}"""
    else:
        prompt = f"""请为以下英文陈述句生成一个合适的一般疑问句或特殊疑问句（以 what、where、when、who、how、why、is、are、do、does、did、can 等开头）。

陈述句："{input_sentence}"

请只返回一个 JSON 对象，不要有任何其他文字：
{{"text": "生成的疑问句"}}"""

    try:
        reply = call_volcano(prompt)
        result = _parse_json_response(reply)
        generated = result.get("text", "")
        logger.info(f"[generate] 生成: {generated[:80]}...")
        return {"generated_text": generated}
    except Exception as e:
        logger.error(f"[generate] 生成失败: {e}")
        return {"generated_text": ""}


async def node_format_output(state: DialogueState) -> dict:
    """节点 5：格式化最终输出"""
    is_question = state["is_question"]
    input_sentence = state["input_sentence"]
    is_natural = state["is_natural"]
    matched_text = state["matched_text"]
    generated_text = state["generated_text"]

    if is_natural and matched_text:
        # 自然匹配 — 直接返回原文对
        if is_question:
            pair = {"type": "qa", "question": input_sentence, "answer": matched_text, "source": "matched"}
        else:
            pair = {"type": "qa", "statement": input_sentence, "question": matched_text, "source": "matched"}
    else:
        # 不自然 — 使用生成的
        if is_question:
            pair = {"type": "qa", "question": input_sentence, "answer": generated_text or "", "source": "generated"}
        else:
            pair = {"type": "qa", "statement": input_sentence, "question": generated_text or "", "source": "generated"}

    logger.info(f"[format] 最终结果: source={pair['source']}")
    return {"final_pair": pair}


# ---------------------------------------------------------------------------
# 条件边
# ---------------------------------------------------------------------------


def router_after_match(state: DialogueState) -> str:
    """匹配后：有匹配句 → 检查自然度，无匹配 → 直接生成"""
    if state.get("matched_text"):
        return "check_natural_qa"
    return "generate"


def router_after_check(state: DialogueState) -> str:
    """自然度检查后：自然 → 输出，不自然 → 生成"""
    if state.get("is_natural"):
        return "format_output"
    return "generate"


# ---------------------------------------------------------------------------
# 构建 Graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """构建 LangGraph 工作流"""
    workflow = StateGraph(DialogueState)

    # 注册节点
    workflow.add_node("classify_input", node_classify_input)
    workflow.add_node("match_from_learned", node_match_from_learned)
    workflow.add_node("check_natural_qa", node_check_natural_qa)
    workflow.add_node("generate", node_generate)
    workflow.add_node("format_output", node_format_output)

    # 入口
    workflow.set_entry_point("classify_input")

    # 分类 → 匹配
    workflow.add_edge("classify_input", "match_from_learned")

    # 匹配 → 自然度检查 / 直接生成
    workflow.add_conditional_edges(
        "match_from_learned",
        router_after_match,
        {
            "check_natural_qa": "check_natural_qa",
            "generate": "generate",
        },
    )

    # 自然度检查 → 输出 / 生成
    workflow.add_conditional_edges(
        "check_natural_qa",
        router_after_check,
        {
            "format_output": "format_output",
            "generate": "generate",
        },
    )

    # 生成 → 输出
    workflow.add_edge("generate", "format_output")

    # 输出 → 结束
    workflow.add_edge("format_output", END)

    return workflow


# ---------------------------------------------------------------------------
# 服务入口
# ---------------------------------------------------------------------------


async def match_dialogue(
    input_sentence: str,
    scholar_id: str,
    learned_sentences: list[dict],
) -> dict:
    """执行对话匹配工作流

    若已学语句列表为空，则不传 learn_sentences，接口侧会从数据库查询。

    Args:
        input_sentence: 用户输入的英文句子
        scholar_id: 学者 ID
        learned_sentences: 已学语句列表 [{"text": "...", "translation": "..."}, ...]

    Returns:
        {"success": True, "data": {...}} 或 {"success": False, "error": "..."}
    """
    initial_state: DialogueState = {
        "input_sentence": input_sentence.strip(),
        "scholar_id": scholar_id,
        "learned_sentences": learned_sentences,
        "is_question": False,
        "matched_text": None,
        "is_natural": False,
        "generated_text": None,
        "final_pair": None,
        "error": None,
    }

    workflow = build_graph()
    graph = workflow.compile()

    try:
        final_state = await graph.ainvoke(initial_state)  # type: ignore[arg-type]

        if isinstance(final_state, dict) and "error" in final_state and final_state["error"]:
            return {"success": False, "error": final_state["error"]}

        result = (
            final_state["final_pair"]
            if isinstance(final_state, dict)
            else None
        )

        return {
            "success": True,
            "data": result or {},
            "is_question": final_state.get("is_question") if isinstance(final_state, dict) else False,
        }
    except Exception as e:
        logger.error(f"[dialogue] 工作流异常: {e}")
        return {"success": False, "error": str(e)}
