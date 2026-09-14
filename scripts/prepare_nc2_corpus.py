"""数据准备脚本：新概念英语 2 前 20 篇 → 本地文件语料（设计稿 §3.1 / §3.2）

产出 `data/nc2/corpus.json`（教材层级：book / chapter / lesson / sentence / group），
字段对齐线上 `textbook_v2 / chapter / lesson / sentence_v2`，供本地生成 / 页面 / 测试共用。
**不写真实库、不触网**（零侵入，§9-7）。

用法（项目根目录）::

    # ① 离线兜底（内置 20 篇占位文本，链路即可跑通）
    python scripts/prepare_nc2_corpus.py

    # ② 指定原文（txt/目录），解析课号 → 句子
    python scripts/prepare_nc2_corpus.py --source ../nc2_lessons.txt --lessons 1-20

    # ③ 只准备部分课 / 自定义输出目录
    python scripts/prepare_nc2_corpus.py --lessons 1-5 --out data/nc2

原文格式约定（--source）：以 `Lesson 12 ...` 或行首数字开头的课号行分段，段内按句切分；
解析不足时按课号回落到内置占位文本（不直接失败，保证链路可跑）。
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.learning.local_corpus import (  # noqa: E402
    BOOK_ID,
    CHAPTER_ID,
    CHAPTER_TITLE,
    GENERATED_DIRNAME,
    LEARNERS_DIRNAME,
    build_ids,
    corpus_path,
    now_ms,
    resolve_data_dir,
    save_corpus,
)

logger = logging.getLogger("scholar-admin.prepare_nc2_corpus")

# ---------------------------------------------------------------------------
# 内置占位语料（§3.2 离线兜底：本地无原文时仍能产出 20 篇结构）
# ---------------------------------------------------------------------------

SEED_LESSONS: list[tuple[int, str, list[tuple[str, str]]]] = [
    (1, "A private conversation", [
        ("Last week I went to the theatre.", "上周我去看了戏。"),
        ("I had a very good seat.", "我的座位很好。"),
    ]),
    (2, "Breakfast or lunch?", [
        ("It was Sunday.", "那天是星期天。"),
        ("I never get up early on Sundays.", "星期天我从不早起。"),
    ]),
    (3, "Please send me a card", [
        ("Postcards always spoil my holidays.", "明信片总是把我的假期搞糟。"),
        ("Last summer I went to Italy.", "去年夏天我去了意大利。"),
    ]),
    (4, "An exciting trip", [
        ("I have just received a letter from my brother, Tim.", "我刚收到弟弟蒂姆的一封信。"),
        ("He is in Australia.", "他在澳大利亚。"),
    ]),
    (5, "No wrong numbers", [
        ("Mr. James Scott has a garage in Silbury.", "詹姆斯·斯科特先生在锡尔伯里有一个汽车修理部。"),
        ("He has just bought another garage in Pinhurst.", "他刚在平赫斯特又买了一个修理部。"),
    ]),
    (6, "Percy Buttons", [
        ("I have just moved to a house in Bridge Street.", "我刚搬到布里奇街的一所房子。"),
        ("Yesterday a beggar knocked at my door.", "昨天一个乞丐敲响了我的门。"),
    ]),
    (7, "Too late", [
        ("The plane was late.", "飞机晚点了。"),
        ("Detectives were waiting at the airport all morning.", "侦探们在机场等了一上午。"),
    ]),
    (8, "The best and the worst", [
        ("Joe Sanders has the most beautiful garden in our town.", "乔·桑德斯拥有我们镇上最漂亮的花园。"),
        ("Nearly everybody enters for the garden competition each year.", "几乎每个人都参加一年一度的花园比赛。"),
    ]),
    (9, "A cold welcome", [
        ("On Wednesday evening, we went to the Town Hall.", "星期三晚上，我们去了市政厅。"),
        ("It was the last day of the year.", "那是一年的最后一天。"),
    ]),
    (10, "Not for jazz", [
        ("We have an old musical instrument.", "我们有一件古老的乐器。"),
        ("It is called a clavichord.", "它叫古钢琴。"),
    ]),
    (11, "One good turn deserves another", [
        ("I was having dinner at a restaurant when Tony Steele came in.", "我正要在饭店吃饭，托尼·斯蒂尔走了进来。"),
        ("Tony worked in a lawyer's office years ago.", "托尼多年前在一家律师事务所工作。"),
    ]),
    (12, "Goodbye and good luck", [
        ("Our neighbour, Captain Charles Alison, will sail from Portsmouth tomorrow.", "我们的邻居查尔斯·艾利森船长明天将从朴茨茅斯启航。"),
        ("We shall meet him at the harbour early in the morning.", "我们明天一早将在港口为他送行。"),
    ]),
    (13, "The Greenwood Boys", [
        ("The Greenwood Boys are a group of pop singers.", "绿林少年是一个流行歌曲演唱组。"),
        ("At present, they are visiting all parts of the country.", "目前他们正在全国各地巡回演出。"),
    ]),
    (14, "Do you speak English?", [
        ("I had an amusing experience last year.", "去年我有过一次有趣的经历。"),
        ("After I had left a small village in the south of France, I drove on to the next town.", "离开法国南部的一个小村庄后，我继续驾车前往下一个城镇。"),
    ]),
    (15, "Good news", [
        ("The secretary told me that Mr. Harmsworth would see me.", "秘书告诉我哈姆斯沃思先生要见我。"),
        ("I felt very nervous when I went into his office.", "走进他办公室时我感到非常紧张。"),
    ]),
    (16, "A polite request", [
        ("If you park your car in the wrong place, a traffic policeman will soon find it.", "如果你把车停错地方，交警很快就会找上你。"),
        ("You will be very lucky if he lets you go without a ticket.", "如果他不开罚单就放你走，那你就太幸运了。"),
    ]),
    (17, "Always young", [
        ("My aunt Jennifer is an actress.", "我姑妈珍妮弗是个演员。"),
        ("She must be at least thirty-five years old.", "她至少有三十五岁了。"),
    ]),
    (18, "He often does this!", [
        ("I had left my bag in the restaurant.", "我把包落在了饭店里。"),
        ("The waiter asked me if I had enjoyed my meal.", "服务员问我是否用得满意。"),
    ]),
    (19, "Sold out", [
        ("The play may begin at any moment.", "戏随时可能开演。"),
        ("I hurried to the ticket office.", "我匆匆赶往售票处。"),
    ]),
    (20, "One man in a boat", [
        ("Fishing is my favourite sport.", "钓鱼是我最喜欢的运动。"),
        ("I often fish for hours without catching anything.", "我常常一钓几个小时却一无所获。"),
    ]),
]

_SEED_BY_NO = {no: (title, sentences) for no, title, sentences in SEED_LESSONS}


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def parse_lessons_arg(spec: str) -> list[int]:
    """解析课号表达式：`1-20` / `1,3,5` / `1-5,8` → 升序去重列表。"""
    numbers: list[int] = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, _, end = chunk.partition("-")
            start, end = int(start), int(end)
            if start > end:
                start, end = end, start
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(chunk))
    ordered = sorted({n for n in numbers if n > 0})
    if not ordered:
        raise ValueError(f"--lessons 非法：{spec!r}（示例：1-20 / 1,3,5）")
    return ordered


# ---------------------------------------------------------------------------
# 原文解析（best-effort）
# ---------------------------------------------------------------------------

_LESSON_HEADER_RE = re.compile(
    r"^\s*(?:Lesson\s+)?(\d{1,3})\b[\s.:\-–—]*(.*)$", re.IGNORECASE
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_SENTENCES_PER_LESSON = 12


def parse_source_text(text: str) -> dict[int, dict]:
    """从原文文本尽力解析 `{课号: {title, sentences: [text]}}`；解析不出返回空 dict。"""
    blocks: dict[int, list[str]] = {}
    titles: dict[int, str] = {}
    current: int | None = None
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LESSON_HEADER_RE.match(line)
        # 仅当行首明确是课号（或 Lesson N）时才视为分段头，避免把正文行误判
        if match and (line.lower().startswith("lesson") or match.group(2) == ""):
            current = int(match.group(1))
            blocks.setdefault(current, [])
            title = match.group(2).strip()
            if title:
                titles[current] = title
            continue
        if current is not None:
            blocks[current].append(line)

    parsed: dict[int, dict] = {}
    for no, lines in blocks.items():
        joined = " ".join(lines).strip()
        sentences = [
            s.strip()
            for s in _SENTENCE_SPLIT_RE.split(joined)
            if len(s.strip().split()) >= 3
        ]
        if not sentences:
            continue
        parsed[no] = {
            "title": titles.get(no) or _SEED_BY_NO.get(no, ("", []))[0],
            "sentences": sentences[:_MAX_SENTENCES_PER_LESSON],
            "translations": [],
        }
    return parsed


def load_source(source: str | Path) -> dict[int, dict]:
    """读取 `--source`（单文件或目录，取目录下 *.txt 合并）。"""
    path = Path(source)
    if path.is_dir():
        text = "\n".join(
            p.read_text(encoding="utf-8", errors="ignore")
            for p in sorted(path.glob("*.txt"))
        )
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_source_text(text)


# ---------------------------------------------------------------------------
# 组装 lessons_data / corpus
# ---------------------------------------------------------------------------


def lessons_data(numbers: list[int], parsed: dict[int, dict] | None = None) -> list[dict]:
    """组装课数据；`parsed`（原文解析结果）优先，缺失课号回落内置占位文本。"""
    parsed = parsed or {}
    result: list[dict] = []
    for no in numbers:
        hit = parsed.get(no)
        if hit and hit.get("sentences"):
            title = hit.get("title") or _SEED_BY_NO.get(no, (f"Lesson {no}", []))[0]
            translations = hit.get("translations") or []
            result.append(
                {
                    "no": no,
                    "title": title or f"Lesson {no}",
                    "source": "source",
                    "sentences": [
                        {
                            "text": text,
                            "translation": translations[i] if i < len(translations) else "",
                        }
                        for i, text in enumerate(hit["sentences"])
                    ],
                }
            )
            continue
        title, sentences = _SEED_BY_NO.get(no, (f"Practice {no}", []))
        if not sentences:
            logger.warning("课 %d 无内置占位文本 → 生成结构占位句", no)
            sentences = [
                (f"This is the first placeholder sentence of lesson {no}.", ""),
                (f"This is the second placeholder sentence of lesson {no}.", ""),
            ]
        result.append(
            {
                "no": no,
                "title": title,
                "source": "seed",
                "sentences": [{"text": t, "translation": tr} for t, tr in sentences],
            }
        )
    return result


def build_corpus(lessons: list[dict]) -> dict:
    """把课数据展开为 §3.1 结构（课 → 句子/分组，字段对齐线上集合）。"""
    ids = {lesson["no"]: build_ids(lesson["no"]) for lesson in lessons}
    chapters = [
        {
            "chapter_id": CHAPTER_ID,
            "textbook_id": BOOK_ID,
            "title": CHAPTER_TITLE,
            "order": 1,
        }
    ]
    lesson_docs: list[dict] = []
    sentence_docs: list[dict] = []
    group_docs: list[dict] = []

    for order, lesson in enumerate(lessons, start=1):
        no = lesson["no"]
        bucket = ids[no]
        lesson_id = bucket["lesson_id"]
        lesson_title = str(lesson["title"]).strip()
        if not lesson_title.lower().startswith("lesson"):
            lesson_title = f"Lesson {no} {lesson_title}".strip()
        lesson_docs.append(
            {
                "lesson_id": lesson_id,
                "chapter_id": CHAPTER_ID,
                "textbook_id": BOOK_ID,
                "title": lesson_title,
                "order": order,
            }
        )
        sentence_ids: list[str] = []
        for idx, sentence in enumerate(lesson["sentences"], start=1):
            sid = bucket["sentence_id"](idx)
            sentence_ids.append(sid)
            sentence_docs.append(
                {
                    "sentence_id": sid,
                    "lesson_id": lesson_id,
                    "chapter_id": CHAPTER_ID,
                    "textbook_id": BOOK_ID,
                    "text": sentence["text"],
                    "translation": sentence.get("translation") or "",
                    "order": idx,
                    "difficulty": 2,
                }
            )
        if sentence_ids:
            group_docs.append(
                {
                    "group_id": bucket["group_id"],
                    "lesson_id": lesson_id,
                    "group_label": f"L{no} {lesson['title']}".strip(),
                    "sentence_ids": sentence_ids,
                }
            )

    return {
        "book": {"textbook_id": BOOK_ID, "title": "新概念英语 2", "level": "B1"},
        "chapters": chapters,
        "lessons": lesson_docs,
        "sentences": sentence_docs,
        "groups": group_docs,
        "meta": {
            "lesson_count": len(lesson_docs),
            "sentence_count": len(sentence_docs),
            "group_count": len(group_docs),
            "generated_at": now_ms(),
            "spec": "§3.1 local corpus (file-only, zero-intrusion)",
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="准备本地 NC2 文件语料（设计稿 §3.1/§3.2）")
    parser.add_argument("--source", default=None, help="新概念2 原文文件或目录（缺省用内置占位文本）")
    parser.add_argument("--lessons", default="1-20", help="课号范围，如 1-20 / 1,3,5")
    parser.add_argument("--out", default=None, help="输出目录（缺省 DIALOGUE_CORPUS_DIR=data/nc2）")
    parser.add_argument("--quiet", action="store_true", help="仅打印结果路径")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    numbers = parse_lessons_arg(args.lessons)

    parsed: dict[int, dict] = {}
    if args.source:
        parsed = load_source(args.source)
        if parsed:
            logger.info("原文解析到 %d 课（%s）", len(parsed), args.source)
        else:
            logger.warning("原文未解析出课号段落 → 全部回落内置占位文本：%s", args.source)

    corpus = build_corpus(lessons_data(numbers, parsed))
    target = corpus_path(args.out)
    save_corpus(corpus, target)

    data_dir = resolve_data_dir(args.out)
    (data_dir / LEARNERS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (data_dir / GENERATED_DIRNAME).mkdir(parents=True, exist_ok=True)

    meta = corpus["meta"]
    if args.quiet:
        print(target)
    else:
        print(
            f"✅ 本地语料已生成：{target}\n"
            f"   课 {meta['lesson_count']} / 句 {meta['sentence_count']} / 任务组 {meta['group_count']}"
        )
        print(
            "   下一步：python scripts/simulate_learning.py --scholar scholar_debug_01 --n 40"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
