"""教材语句生成服务（固定教材版）— 教科版广州专用 2024 新版

工作流：
1. generate_content — 调用火山模型，按指定年级的单元目录输出核心语句
2. 返回结构化 JSON 供外部路由层写入数据库
"""
from __future__ import annotations

import json as json_lib
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from services.build_sentence import call_volcano, _repair_json

logger = logging.getLogger("scholar-admin.build-fixed")

# ---------------------------------------------------------------------------
# 教材配置注册表 — 统一管理各年级固定信息
# ---------------------------------------------------------------------------

TEXTBOOK_CONFIGS: dict[str, dict] = {
    "3": {
        "name": "三年级英语上册 广州版",
        "grade": "三年级",
        "semester": "上册",
        "edition": "教科版（教育科学出版社，2024新版，广州专用）",
        "unit_hints": (
            "注意单元间的递进关系："
            "Unit1-2 字母与语言基础认知 → Unit3-4 身份与操作/键盘 → "
            "Unit5-6 密码与绘画/指令 → Unit7-8 倾听与运动/指令 → "
            "Review 综合音乐表演复习"
        ),
        "units": [
            "Unit 1 Letters in Our Life",
            "Unit 2 English and Chinese",
            "Unit 3 Who's Who?",
            "Unit 4 Colour a Keyboard",
            "Unit 5 Set a Password",
            "Unit 6 I Can Draw",
            "Unit 7 Be a Good Listener",
            "Unit 8 It's Time to Exercise!",
            "Review A Music Show",
        ],
    },
    "4": {
        "name": "四年级英语上册 秋天 广州版",
        "grade": "四年级",
        "semester": "上册",
        "edition": "教科版（广州专用，2024新版）",
        "unit_hints": (
            "注意单元间的递进关系："
            "Unit1-2 打招呼/礼貌用语 → Unit3-4 厨房/家务 → "
            "Unit5-6 家庭成员 → Unit7-8 家庭时光与节日"
        ),
        "units": [
            "Unit 1 Come on in",
            "Unit 2 Help yourself",
            "Unit 3 I'm the Chef Today",
            "Unit 4 Help out in the Kitchen",
            "Unit 5 She Helps Me a Lot",
            "Unit 6 I Love My Family",
            "Unit 7 Family Time",
            "Unit 8 Joy in the Air",
        ],
    },
    "3b": {
        "name": "三年级英语下册 广州版",
        "grade": "三年级",
        "semester": "下册",
        "edition": "教科版（教育科学出版社，2024新版，广州专用）",
        "unit_hints": (
            "注意单元间的递进关系："
            "Unit1-3 日常作息与时间管理 → Unit4 邀请与社交活动 → "
            "Unit5-6 教室与值日/责任 → Unit7-8 规则与礼仪 → "
            "Review 综合道路安全主题复习"
        ),
        "units": [
            "Unit 1 Get Up",
            "Unit 2 What a Day!",
            "Unit 3 Plan My Day",
            "Unit 4 Come and Join Us!",
            "Unit 5 Our Classroom, Our Call",
            "Unit 6 I'm on Duty!",
            "Unit 7 School Rules",
            "Unit 8 A Polite Way",
            "Review Road Helper Day",
        ],
    },
    "4b": {
        "name": "四年级英语下册 广州版",
        "grade": "四年级",
        "semester": "下册",
        "edition": "教科版（教育科学出版社，2024新版，广州专用）",
        "unit_hints": (
            "注意单元间的递进关系："
            "Unit1-2 校园与动物/忙碌 → Unit3 身体成长与比较级 → "
            "Unit4 动手实践与叙事 → Unit5 天气与自然 → "
            "Unit6-7 整理物品与通讯 → Unit8 居住地与社区 → "
            "Review 综合旅行主题复习"
        ),
        "units": [
            "Unit 1 The School Garden",
            "Unit 2 Busy as a Bee",
            "Unit 3 Grow Taller",
            "Unit 4 Let's Do It!",
            "Unit 5 How's the Weather?",
            "Unit 6 Tidy Up My Closet",
            "Unit 7 Let's Make a Call!",
            "Unit 8 The Best Place to Live",
            "Review A Happy Trip",
        ],
    },
    "5": {
        "name": "五年级英语上册 广州版",
        "grade": "五年级",
        "semester": "上册",
        "edition": "教科版（教育科学出版社，2024新版，广州通用）",
        "unit_hints": (
            "注意单元间的递进关系："
            "Unit1 词汇学习策略 → Unit2 日常生活 → Unit3 友谊 → "
            "Unit4 天气与自然灾害 → Unit5 太空与科学 → "
            "Unit6 帮助他人 → Unit7 动手实践(手工) → Review 综合复习与露营"
        ),
        "units": [
            "Unit 1 Learn Words in Chunks",
            "Unit 2 It's for everybody",
            "Unit 3 Best Friends Forever",
            "Unit 4 Here Comes a Typhoon",
            "Unit 5 Space Travel",
            "Unit 6 Helping Out",
            "Unit 7 Make a Bird Feeder",
            "Review Let's Go Camping!",
        ],
    },
}

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class BuildFixedState(TypedDict):
    grade: str
    generated_json: str | None
    content: dict | None
    error: str | None


# ---------------------------------------------------------------------------
# Prompt 构建
# ---------------------------------------------------------------------------


def _build_system_prompt(cfg: dict) -> str:
    unit_list = cfg["units"]
    return (
        "你是一位小学英语教材编辑专家，非常熟悉教科版（广州专用）小学英语教材。\n"
        "\n"
        "教材版本确认：\n"
        f"这套是 {cfg['edition']} 小学英语{cfg['grade']}{cfg['semester']}"
        "（教育科学出版社出版，广州地区主流小学英语教材）。\n"
        "\n"
        "完整单元目录：\n"
        f"{chr(10).join(unit_list)}\n"
        "\n"
        f"你的任务是为以上 {len(unit_list)} 个单元分别生成该单元所有需要掌握的英语核心句型/语句。\n"
        "\n"
        "要求：\n"
        "1. 输出必须是合法的 JSON，不要包含 markdown 代码块标记（不要 ```json 包裹）\n"
        "2. 严格按上述单元输出，unit_index 从 1 递增，unit_title 使用上述完整单元名\n"
        "3. 每个单元 6-10 句核心语句（对话常用句 + 重点句型）；Review 类单元可适当增加到 8-12 句\n"
        "4. 每句话包含：英文原文(text)、中文翻译(translation)\n"
        f"5. 语句难度要符合{cfg['grade']}小学生水平（CEFR A1-A2），语言地道自然\n"
        f"6. {cfg['unit_hints']}"
    )


def _build_user_prompt(cfg: dict) -> str:
    units_lines = "\n".join(f"  {i+1}. {u}" for i, u in enumerate(cfg["units"]))
    unit_title_example = cfg["units"][0]
    return (
        "请为以下教材生成每个单元的核心语句。\n"
        "\n"
        "教材信息：\n"
        f"- 名称：{cfg['name']}\n"
        f"- 版本：{cfg['edition']}\n"
        f"- 年级学期：{cfg['grade']}{cfg['semester']}\n"
        "\n"
        "单元列表：\n"
        f"{units_lines}\n"
        "\n"
        "输出 JSON 结构如下：\n"
        "{\n"
        f'  "textbook_info": {{\n'
        f'    "name": "{cfg["name"]}",\n'
        f'    "grade": "{cfg["grade"]}",\n'
        f'    "semester": "{cfg["semester"]}",\n'
        f'    "edition": "{cfg["edition"]}"\n'
        "  },\n"
        '  "units": [\n'
        "    {\n"
        '      "unit_index": 1,\n'
        f'      "unit_title": "{unit_title_example}",\n'
        '      "topic": "本单元主题（中文，一句话概括）",\n'
        '      "sentences": [\n'
        '        {"text": "英文语句", "translation": "中文翻译"},\n'
        "        ...\n"
        "      ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "\n"
        "请直接输出 JSON，不要带任何 markdown 标记。"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def generate_content(state: BuildFixedState) -> BuildFixedState:
    """调用火山模型生成教材结构化内容"""
    grade = state["grade"]
    cfg = TEXTBOOK_CONFIGS[grade]
    logger.info(f"[build-fixed] 开始生成固定教材内容: {cfg['name']} (grade={grade})")

    raw = ""
    try:
        raw = call_volcano(
            prompt=_build_user_prompt(cfg),
            system_prompt=_build_system_prompt(cfg),
            temperature=0.3,
            max_tokens=8192,
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
        logger.info(
            f"[build-fixed] 生成成功: 共 {len(content['units'])} 个单元"
        )
    except json_lib.JSONDecodeError as e:
        logger.error(f"[build-fixed] JSON 解析失败: {e}, raw={raw[:500]}")
        state["error"] = f"模型输出非 JSON: {e}"
        state["content"] = None
    except Exception as e:
        logger.error(f"[build-fixed] 生成失败: {e}")
        state["error"] = str(e)
        state["content"] = None

    return state


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    graph = StateGraph(BuildFixedState)
    graph.add_node("generate_content", generate_content)
    graph.set_entry_point("generate_content")
    graph.add_edge("generate_content", END)
    return graph


_app = _build_graph().compile()


async def build_textbook_fixed(grade: str) -> dict:
    """执行固定教材语句生成工作流

    Args:
        grade: 年级，可选 "3"、"3b"、"4"、"4b"、"5"

    Returns:
        {"success": bool, "content": dict | None, "error": str | None}
        content 中包含 textbook_info + units
    """
    state: BuildFixedState = {
        "grade": grade,
        "generated_json": None,
        "content": None,
        "error": None,
    }
    result = _app.invoke(state)

    if result["error"]:
        return {"success": False, "content": None, "error": result["error"]}

    return {"success": True, "content": result["content"], "error": None}
