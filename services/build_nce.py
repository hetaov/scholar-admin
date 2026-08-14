"""新概念英语教材原文导入服务 — 忠实还原 Lesson 正文所有英文语句

与 build_sentence_fixed 的核心区别：
- 不生成核心语句，而是复现原文所有语句
- 不做总结筛选，完全使用原文内容
- 每课(lesson)对应一个 unit，原文句子逐句存入 sentence 表
"""
from __future__ import annotations

import json as json_lib
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from services.build_sentence import call_volcano, _repair_json

logger = logging.getLogger("scholar-admin.build-nce")

# ---------------------------------------------------------------------------
# 新概念英语 4 册教材配置
# ---------------------------------------------------------------------------

NCE_BOOKS: dict[str, dict] = {
    "1": {
        "name": "新概念英语第一册",
        "name_en": "New Concept English Book 1: First Things First",
        "subtitle": "英语初阶",
        "total_lessons": 144,
        "lesson_type": "对话为主，每课为一段简短的情景对话",
        "edition": "新概念英语（New Concept English）",
        "author": "L.G. Alexander",
        "hint": (
            "第一册为英语初阶，课文以情景对话为主。"
            "Lesson 1-2 等双数以练习为主，通常无新增对话正文，"
            "这些课根据实际是否有对话内容输出；如仅含练习可输出0句或少量示例句。"
        ),
    },
    "2": {
        "name": "新概念英语第二册",
        "name_en": "New Concept English Book 2: Practice and Progress",
        "subtitle": "实践与进步",
        "total_lessons": 96,
        "lesson_type": "每课为一篇约150词的短篇故事或叙述文",
        "edition": "新概念英语（New Concept English）",
        "author": "L.G. Alexander",
        "hint": (
            "第二册为实践与进步，每课一篇独立短文，围绕一个语法点展开。"
            "每课正文均为完整英文段落，逐句拆分输出。"
        ),
    },
    "3": {
        "name": "新概念英语第三册",
        "name_en": "New Concept English Book 3: Developing Skills",
        "subtitle": "培养技能",
        "total_lessons": 60,
        "lesson_type": "每课为一篇约250-300词的中篇故事或论述文",
        "edition": "新概念英语（New Concept English）",
        "author": "L.G. Alexander",
        "hint": (
            "第三册为培养技能，每课为完整的中篇英文文章。"
            "文章结构清晰，逐句拆分输出正文所有语句。"
        ),
    },
    "4": {
        "name": "新概念英语第四册",
        "name_en": "New Concept English Book 4: Fluency in English",
        "subtitle": "流利英语",
        "total_lessons": 48,
        "lesson_type": "每课为一篇约350-500词的长篇英文文章，涉及多学科领域",
        "edition": "新概念英语（New Concept English）",
        "author": "L.G. Alexander",
        "hint": (
            "第四册为流利英语，每课为长篇英文原文（非简化），"
            "涵盖科学、文学、哲学等多领域。逐句拆分输出正文所有语句。"
        ),
    },
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class BuildNCEState(TypedDict):
    book: str
    start_lesson: int
    end_lesson: int
    generated_json: str | None
    content: dict | None
    error: str | None


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------


def _build_system_prompt(cfg: dict, start: int, end: int) -> str:
    return (
        f"你是一位新概念英语（New Concept English）教材专家，"
        f"拥有全套新概念英语教材的完整内容记忆。\n"
        "\n"
        f"教材：《{cfg['name']}》（{cfg['name_en']}）\n"
        f"副标题：{cfg['subtitle']}\n"
        f"作者：{cfg['author']}\n"
        f"全书共 {cfg['total_lessons']} 课\n"
        f"课文形式：{cfg['lesson_type']}\n"
        "\n"
        f"你的任务是输出第 {start} 课到第 {end} 课的全部英文原文语句。\n"
        "\n"
        "核心原则 — 必须严格遵循：\n"
        "1. **忠实原文**：只输出教材原文中的英文语句，不得修改、精简、改写或总结\n"
        "2. **完整输出**：每课正文(Dialogue 或 Passage)中的所有英文语句，一句不漏\n"
        "3. **原文原样**：保持原文的拼写、标点、大小写完全一致\n"
        "4. **官方译文**：中文翻译使用新概念英语官方/标准译文\n"
        f"5. {cfg['hint']}\n"
        "\n"
        "输出格式要求：\n"
        "1. 输出必须是合法的 JSON，不要包含 markdown 代码块标记（不要 ```json 包裹）\n"
        "2. 每课对应一个 unit，unit_index 从 1 开始递增\n"
        f"3. unit_index=1 对应 Lesson {start}，unit_index=2 对应 Lesson {start+1}，以此类推\n"
        "4. unit_title 使用原课标题格式，如 \"Lesson 1: A Private Conversation\"\n"
        "5. topic 为该课主题的一句话中文概括\n"
        "6. sentences 数组包含该课正文中每一句英文及其翻译\n"
        "7. 对话课中，说话人切换用独立 sentence 表示（每句话独立一条）\n"
    )


def _build_user_prompt(cfg: dict, start: int, end: int) -> str:
    lesson_count = end - start + 1
    return (
        f"请输出《{cfg['name']}》第 {start}-{end} 课（共 {lesson_count} 课）的原文全部英文语句。\n"
        "\n"
        f"教材：{cfg['name']} / {cfg['name_en']}\n"
        f"版本：{cfg['edition']}\n"
        f"课文形式：{cfg['lesson_type']}\n"
        "\n"
        "输出 JSON 结构如下：\n"
        "{\n"
        f'  "textbook_info": {{\n'
        f'    "name": "{cfg["name"]}",\n'
        f'    "name_en": "{cfg["name_en"]}",\n'
        f'    "edition": "{cfg["edition"]}",\n'
        f'    "author": "{cfg["author"]}"\n'
        "  },\n"
        '  "units": [\n'
        "    {\n"
        '      "unit_index": 1,\n'
        f'      "unit_title": "Lesson {start}: [原课英文标题]",\n'
        '      "topic": "本课主题（中文，一句话）",\n'
        '      "sentences": [\n'
        '        {"text": "教材英文原文语句", "translation": "中文翻译"},\n'
        "        ...\n"
        "      ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "关键要求：\n"
        "- 逐句原文照录，保持原样，不增不减不改\n"
        "- 每课正文全部语句完整输出，一句不漏\n"
        "- 如某课为纯练习课、无正文对话/文章，可为空 sentences 数组\n"
        "- 请直接输出 JSON，不要带任何 markdown 标记"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def generate_content(state: BuildNCEState) -> BuildNCEState:
    """调用火山模型复现 NCE 原文内容"""
    book = state["book"]
    start = state["start_lesson"]
    end = state["end_lesson"]
    cfg = NCE_BOOKS[book]

    logger.info(
        f"[build-nce] 开始复现: {cfg['name']} "
        f"第 {start}-{end} 课 (book={book})"
    )

    raw = ""
    try:
        raw = call_volcano(
            prompt=_build_user_prompt(cfg, start, end),
            system_prompt=_build_system_prompt(cfg, start, end),
            temperature=0.1,       # 极低温度，减少创造性
            max_tokens=16384,       # 足够输出大量原文
        )
        # 清理 markdown 代码块
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        # 修复常见 JSON 语法问题后再解析
        raw = _repair_json(raw)
        content = json_lib.loads(raw)

        state["generated_json"] = raw
        state["content"] = content
        state["error"] = None
        unit_count = len(content.get("units", []))
        total_sents = sum(
            len(u.get("sentences", [])) for u in content.get("units", [])
        )
        logger.info(
            f"[build-nce] 复现成功: {unit_count} 课, 共 {total_sents} 句"
        )
    except json_lib.JSONDecodeError as e:
        logger.error(f"[build-nce] JSON 解析失败: {e}, raw={raw[:500]}")
        state["error"] = f"模型输出非 JSON: {e}"
        state["content"] = None
    except Exception as e:
        logger.error(f"[build-nce] 复现失败: {e}")
        state["error"] = str(e)
        state["content"] = None

    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    graph = StateGraph(BuildNCEState)
    graph.add_node("generate_content", generate_content)
    graph.set_entry_point("generate_content")
    graph.add_edge("generate_content", END)
    return graph


_app = _build_graph().compile()


async def build_nce_book(
    book: str,
    start_lesson: int = 1,
    end_lesson: int | None = None,
) -> dict:
    """执行新概念英语原文复现工作流

    Args:
        book: 册数，可选 "1"、"2"、"3"、"4"
        start_lesson: 起始课号，默认 1
        end_lesson: 结束课号，默认取该册总课数

    Returns:
        {"success": bool, "content": dict | None, "error": str | None}
        content 中包含 textbook_info + units（每课为一个 unit）
    """
    cfg = NCE_BOOKS[book]
    if end_lesson is None:
        end_lesson = cfg["total_lessons"]
    if start_lesson < 1:
        start_lesson = 1
    if end_lesson > cfg["total_lessons"]:
        end_lesson = cfg["total_lessons"]

    state: BuildNCEState = {
        "book": book,
        "start_lesson": start_lesson,
        "end_lesson": end_lesson,
        "generated_json": None,
        "content": None,
        "error": None,
    }
    result = _app.invoke(state)

    if result["error"]:
        return {"success": False, "content": None, "error": result["error"]}

    return {"success": True, "content": result["content"], "error": None}
