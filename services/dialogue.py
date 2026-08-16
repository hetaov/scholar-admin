"""对话匹配服务 — 基于 LangGraph + 火山方舟模型的问答匹配工作流

工作流：
1. classify_and_match — 单次调用完成：判断输入句类型 + 从已学语句匹配问答对 + 判断是否符合日常用语习惯
2. generate — 无匹配或不符合习惯时，针对输入生成合适的问句/答句
"""
from __future__ import annotations

import asyncio
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
from services.models_content import query_all_pages
from services.models_learning import SKILL_STATE, STATUS_NOT_STARTED

logger = logging.getLogger("scholar-admin.dialogue")

# ---------------------------------------------------------------------------
# 火山方舟对话模型客户端（单例）
# ---------------------------------------------------------------------------

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """获取火山方舟 OpenAI 兼容客户端（单次调用最长 60 秒，避免无限挂起）"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=VOLCANO_API_KEY,
            base_url=VOLCANO_BASE_URL,
            timeout=60.0,
        )
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


def _recall_candidates(
    input_sentence: str,
    learned: list[dict],
    top_n: int = 10,
) -> list[dict]:
    """轻量相似度召回：从已学语句中选出最相似的 top_n 句，压缩 LLM 输入规模。

    已学语句可能上百句，全部塞进提示词会导致模型推理极慢（实测单次 30s+）。
    先用字符相似度召回候选，再交给模型选择，可显著降低单次调用耗时。
    """
    from difflib import SequenceMatcher

    def _score(text: str) -> float:
        a, b = input_sentence.strip().lower(), text.strip().lower()
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    ranked = sorted(
        (s for s in learned if s.get("text")),
        key=lambda s: _score(s["text"]),
        reverse=True,
    )
    return ranked[:top_n]


async def node_classify_and_match(state: DialogueState) -> dict:
    """节点 1：合并 分类 + 匹配 + 自然度判断 为单次 LLM 调用

    原工作流需要 3 次串行调用（classify → match → check_natural），实测耗时
    30~60 秒，超过小程序 callContainer 的 15 秒超时上限。合并后匹配成功场景
    仅需 1 次调用，配合相似度召回压缩输入，可把整条链路控制在 15 秒内。
    """
    input_sentence = state["input_sentence"]

    if not state.get("learned_sentences"):
        logger.warning("[match] 已学语句列表为空，跳过匹配")
        return {
            "is_question": input_sentence.strip().endswith("?"),
            "matched_text": None,
            "is_natural": False,
        }

    # 1. 相似度召回 top 10 候选，压缩 LLM 输入规模
    candidates = _recall_candidates(input_sentence, state["learned_sentences"], top_n=10)
    sentence_texts = [s.get("text", "") for s in candidates if s.get("text")]
    sentences_joined = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(sentence_texts))

    if not sentences_joined:
        return {
            "is_question": input_sentence.strip().endswith("?"),
            "matched_text": None,
            "is_natural": False,
        }

    prompt = f"""请完成以下三项任务，只返回一个 JSON 对象（不要有任何其他文字）：

1. 判断输入英文句是"陈述句(statement)"还是"疑问句(question)"。
2. 从下方已学语句中选出最合适的一句作为配对句：若输入是疑问句，找一句合适的陈述句作为回答；若输入是陈述句，找一句合适的问句（以 what/where/when/who/how/why/is/are/do/does/did/can 等开头）作为对该陈述句的提问。
3. 判断该配对是否符合日常英语交流习惯（自然、合理）。

输入句子："{input_sentence}"

已学语句（候选）：
{sentences_joined}

返回格式：
{{"type": "question"|"statement", "matched_index": <数字，找不到则填 null>, "natural": true|false, "reason": "简短理由"}}"""

    try:
        # 同步 LLM 调用放进线程池，避免阻塞事件循环（后台异步任务与查询接口共用同循环）
        reply = await asyncio.to_thread(call_volcano, prompt)
        result = _parse_json_response(reply)
        is_question = result.get("type") == "question"
        idx = result.get("matched_index")
        matched_text = None
        if idx and 1 <= idx <= len(sentence_texts):
            matched_text = sentence_texts[idx - 1]
            logger.info(f"[match] 匹配到第 {idx} 句: {matched_text[:50]}...")
        else:
            logger.info("[match] 未找到合适匹配")
        is_natural = bool(result.get("natural", False)) if matched_text else False
        logger.info(f"[classify] 输入句类型: {'疑问句' if is_question else '陈述句'}")
        logger.info(f"[natural] 结果={is_natural}, 理由={result.get('reason', '')}")
        return {
            "is_question": is_question,
            "matched_text": matched_text,
            "is_natural": is_natural,
        }
    except Exception as e:
        logger.error(f"[classify/match] 分类、匹配或自然度判断失败: {e}")
        # 兜底：按末尾问号判断类型，走生成
        stripped = input_sentence.strip()
        return {
            "is_question": stripped.endswith("?"),
            "matched_text": None,
            "is_natural": False,
        }


async def node_generate(state: DialogueState) -> dict:
    """节点 2：针对输入句生成合适的问句或答句"""
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
        reply = await asyncio.to_thread(call_volcano, prompt)
        result = _parse_json_response(reply)
        generated = result.get("text", "")
        logger.info(f"[generate] 生成: {generated[:80]}...")
        return {"generated_text": generated}
    except Exception as e:
        logger.error(f"[generate] 生成失败: {e}")
        return {"generated_text": ""}


async def node_format_output(state: DialogueState) -> dict:
    """节点 3：格式化最终输出"""
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
    """分类匹配后：有匹配句且自然 → 直接输出；否则 → 生成"""
    if state.get("matched_text") and state.get("is_natural"):
        return "format_output"
    return "generate"


# ---------------------------------------------------------------------------
# 构建 Graph
# ---------------------------------------------------------------------------


def build_graph() -> StateGraph:
    """构建 LangGraph 工作流"""
    workflow = StateGraph(DialogueState)

    # 注册节点
    workflow.add_node("classify_and_match", node_classify_and_match)
    workflow.add_node("generate", node_generate)
    workflow.add_node("format_output", node_format_output)

    # 入口
    workflow.set_entry_point("classify_and_match")

    # 分类匹配 → 直接输出 / 生成
    workflow.add_conditional_edges(
        "classify_and_match",
        router_after_match,
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


async def load_learned_sentences(db, scholar_id: str) -> list[dict]:
    """加载该学者全部已学语句（文本 + 翻译）

    从 `skill_state`（Phase 2 能力模型）取 sentence_id（去重、排除未开始），
    再到 `sentence_v2` 集合分批查询（$in 上限 100 条/批）。

    Returns:
        [{"text": str, "translation": str}, ...]；无已学语句时返回 []
        （旧接口的"该学者暂无已学语句"由调用方依据空列表判断）
    """
    state_records = await query_all_pages(
        db,
        collection=SKILL_STATE,
        where={
            "scholar_id": scholar_id,
            "status": {"$ne": STATUS_NOT_STARTED},
        },
        select={"sentence_id": 1},
    )
    if not state_records:
        return []

    sentence_ids = list(
        {r.get("sentence_id") for r in state_records if r.get("sentence_id")}
    )
    if not sentence_ids:
        return []

    learned_sentences: list[dict] = []
    for i in range(0, len(sentence_ids), 100):
        batch = sentence_ids[i : i + 100]
        sentence_result = await db.query(
            collection="sentence_v2",
            where={"sentence_id": {"$in": batch}},
            limit=100,
        )
        for rec in sentence_result.get("records", []):
            learned_sentences.append(
                {
                    "text": rec.get("text", ""),
                    "translation": rec.get("translation", ""),
                }
            )

    logger.info(f"[match] scholar={scholar_id}, 已学={len(sentence_ids)} 句")
    return learned_sentences


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
