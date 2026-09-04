"""存量教材未分组语句的 LLM 语义分组脚本（默认 dry-run 出计划，--apply 写库）

背景（scholar-skill/docs_v1《英语句子分组与重复处理重构建议》§3.3 Phase D +
data-model-contract §4.14 sentence_group）：sentence_group 是学习/复习的最小编排
单位。存量教材（广州版 5 本 + 新概念第一册）大部分句子 group_id 为空 —— 学习时
回退 legacy 单句兼容层。本脚本对每个 lesson 的**未分组句子**用混元（hy3）判断
可否组成语义组，并在 lesson 内建组（sentence_group 文档 + sentence_v2 回写
group_id / role_in_group）。

分组规则（需求约束 + 设计稿）：
- **只在 lesson 内分组**，不跨课、不新增/改写句子；
- **课内同文本折叠**：新概念课内同一句被复制多遍（探针：29/50 课有重复，最甚
  L21 仅 7 个唯一文本 × 重复 10 遍）—— 先按归一化文本折叠成「唯一句」参与
  分组决策，折叠句同组同 role 同归属（不落库去重，M5 canonical 另行处理）；
- 组大小 **2~6 个唯一句**（语义原子化，不强行凑组）：无法与任何句组成有意义组
  的唯一句保持未分组（skip，可人工后续处理），避免 1 句硬组；
- 组 type ∈ dialogue_pair(对话问答) / grammar_family(同句型) / vocab_family(同词汇)；
- 语句特别多的课按唯一句切成 `--window`（默认 40）一个**任务**，每任务独立调
  LLM 控制上下文；窗口间不跨组；
- 组内顺序 = 语句在 lesson 内的 order 顺序；role_in_group 推断与
  services/english/sentence_group.py._infer_role_in_group 同口径；
- order_in_lesson 接续该 lesson 已有组（人工/先前建组）之后递增。

LLM 调用复用 ai_session_eval.py 的混元网关范式：AsyncOpenAI + HUNYUAN_* 配置，
不用 response_format=json_object（hy 系列实测可能空 content），prompt 约束 +
容错解析（去 markdown fence / 首尾花括号提取）。

用法（scholar-admin 根目录，.env 自动加载 CloudBase + 混元凭据）：
  python scripts/group_lesson_sentences.py --textbook-id tb_xxx           # 单教材 dry-run（调 LLM 出计划）
  python scripts/group_lesson_sentences.py --textbook-id tb_xxx --apply   # 按计划写库
  python scripts/group_lesson_sentences.py --all                          # 全部英语教材 dry-run
  python scripts/group_lesson_sentences.py --all --apply
  python scripts/group_lesson_sentences.py --all --json /tmp/group_plan.json   # 计划落盘（复跑免 LLM）
  python scripts/group_lesson_sentences.py --plan-file /tmp/group_plan.json --apply  # 直接按盘计划写库
  python scripts/group_lesson_sentences.py --textbook-id tb_xxx --lesson-id l1  # 精确到课
  python scripts/group_lesson_sentences.py --all --concurrency 3 --window 40

退出码：0 = 成功；参数/DB/LLM 全败 → 1。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    HUNYUAN_BASE_URL,
    HUNYUAN_EVAL_MODEL,
    HUNYUAN_SECRET_KEY,
    HUNYUAN_TIMEOUT_SECONDS,
)
from services.dependencies import get_db  # noqa: E402
from services.models.content import (  # noqa: E402
    LESSON,
    SENTENCE_GROUP,
    SENTENCE_V2,
    TEXTBOOK_V2,
    build_sentence_group_doc,
    build_sentence_group_id,
    compute_text_hash,
    get_lessons_by_textbook,
    get_sentences_by_lesson,
    query_all_pages,
)

# 组大小约束（用户需求：原则上 2-6，语义原子化）
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 6
# 单任务（一次 LLM 调用）容纳的句子上限：控制上下文
DEFAULT_WINDOW = 40
DEFAULT_CONCURRENCY = 3

# 本脚本可产生的组类型（stand_alone 单句组不产生 —— 不成组句子保持未分组）
GROUPABLE_TYPES = frozenset({"dialogue_pair", "grammar_family", "vocab_family"})

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


# ===========================================================================
# 纯函数：窗口切分 / 解析 / 校验 / 角色推断（可单测，不触网）
# ===========================================================================


def _is_ungrouped(sentence: dict) -> bool:
    """未分组判据：group_id 空（None / 缺失 / 空串），且非语义重复句（M5 未上线，防御）。"""
    if sentence.get("group_id"):
        return False
    csid = sentence.get("canonical_sentence_id")
    if csid and csid != sentence.get("sentence_id"):
        return False  # duplicate 句不参与建组
    return True


def collapse_by_text(sentences: list[dict]) -> list[dict]:
    """课内同文本折叠：归一化文本相同的句子并为一个「唯一句」条目（保序）。

    新概念等教材课内同句被复制多遍（同一 sentence 出现在多个练习位）——分组决策
    以唯一句为原子，避免 LLM 把 5 个相同文本凑成无意义组；折叠句写回时同组
    同 role（组归属随代表句），**不写 canonical_sentence_id（M5 去重另办）**。

    Returns: 按 order 升序的条目列表；每条 = 代表句(dict) + sentence_ids(折叠句
    id 保序) + dup_count。
    """
    buckets: dict[str, dict] = {}
    for s in sentences:
        text = str(s.get("text") or "").strip()
        key = compute_text_hash(text) if text else f"__raw_{s.get('sentence_id') or id(s)}"
        b = buckets.get(key)
        if b is None:
            b = {"entry": None, "order": None, "sentence_ids": []}
            buckets[key] = b
        if b["entry"] is None:
            b["entry"] = s
        o = s.get("order")
        if o is not None and (b["order"] is None or o < b["order"]):
            b["order"] = o
        sid = s.get("sentence_id")
        if sid and sid not in b["sentence_ids"]:
            b["sentence_ids"].append(sid)
    out: list[dict] = []
    for b in buckets.values():
        entry = dict(b["entry"])
        entry["sentence_ids"] = b["sentence_ids"]
        entry["order"] = b["order"]
        entry["dup_count"] = len(b["sentence_ids"])
        out.append(entry)
    out.sort(key=lambda e: (e.get("order") is None, e.get("order") or 0))
    return out


def entry_sentence_ids(entry: dict) -> list[str]:
    """取条目实际句 id 列表（纯函数）：折叠条目=sentence_ids；普通句=[sentence_id]。

    折叠后的唯一句条目带 `sentence_ids`（含代表句自身 + 全部折叠句）；普通未折叠
    句子只有 sentence_id。两组场景统一出口，供 kept/展开/报告复用。
    """
    sids = entry.get("sentence_ids")
    if sids:
        return [s for s in sids if s]
    sid = entry.get("sentence_id")
    return [sid] if sid else []


def compute_kept_sentence_ids(sentences: list[dict], groups: list[dict]) -> list[str]:
    """窗口内未进组的实际句子 id（按 entry 展开，保序；组 sentence_ids 已含折叠句）。"""
    used = {sid for g in groups for sid in g.get("sentence_ids", [])}
    return [
        sid
        for s in sentences
        for sid in entry_sentence_ids(s)
        if sid not in used
    ]


def split_windows(sentences: list[dict], window: int) -> list[list[dict]]:
    """把已按 order 升序的（唯一句）切成 ≤ window 的连续窗口（窗口间不跨组）。"""
    if not sentences:
        return []
    window = max(int(window or DEFAULT_WINDOW), 1)
    return [sentences[i : i + window] for i in range(0, len(sentences), window)]


def parse_llm_content(content: str) -> dict:
    """解析 hy3 返回 JSON（容错：去 markdown fence / 首尾花括号提取）。

    Returns: {"ok": bool, "obj": dict | None, "error": str | None}
    """
    text = (content or "").strip()
    if not text:
        return {"ok": False, "obj": None, "error": "LLM 响应 content 为空"}
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return {"ok": False, "obj": None, "error": "未找到 JSON 对象"}
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError) as e:
        return {"ok": False, "obj": None, "error": f"JSON 解析失败: {e}"}
    if not isinstance(obj, dict):
        return {"ok": False, "obj": None, "error": "JSON 顶层不是对象"}
    return {"ok": True, "obj": obj, "error": None}


def _as_int(v) -> int | None:
    """把 LLM 输出序号（int/str/float）转 int，失败返回 None。"""
    if v is None:
        return None
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def normalize_window_groups(obj: dict, window_sentences: list[dict]) -> dict:
    """校验 + 归一化一个窗口的 LLM 分组建议（纯函数）。

    规则：
    - 组大小 2~6（<2 或 >6 → 拒绝该组记 invalid；不截断不硬并）；
    - indices 必须 ∈ [1..N]（窗口内 1-based）；每句至多进一组（冲突 → 该组 invalid）；
    - type ∈ GROUPABLE_TYPES；title 缺失用首句文本前 20 字兜底；
    - 组内顺序按 indices 原序（LLM 输出顺序=学习顺序，不允许乱序打散）。

    Returns: {"groups": [...], "invalid": [...], "used": set[int]}
      每条 group: {type, title, reason, indices: [int...], sentence_ids: [str...]}
    """
    groups: list[dict] = []
    invalid: list[dict] = []
    used: set[int] = set()
    n = len(window_sentences)

    raw_groups = obj.get("groups")
    if raw_groups is None:
        raw_groups = []
    if not isinstance(raw_groups, list):
        invalid.append({"error": "groups 不是数组", "raw": raw_groups})
        raw_groups = []

    for i, g in enumerate(raw_groups):
        if not isinstance(g, dict):
            invalid.append({"error": f"组 {i + 1} 不是对象"})
            continue
        indices_raw = g.get("indices")
        if indices_raw is None:
            indices_raw = g.get("sentence_indices")
        indices: list[int] = []
        for v in (indices_raw or []):
            iv = _as_int(v)
            if iv is not None and 1 <= iv <= n and iv not in indices:
                indices.append(iv)
            elif iv is not None and iv not in indices:
                indices.append(iv)  # 保留越界项供校验报错
        type_ = str(g.get("type") or "").strip()
        title = str(g.get("title") or "").strip()
        reason = str(g.get("reason") or "").strip()

        def _reject(msg: str) -> None:
            invalid.append({
                "error": msg,
                "group_index": i + 1,
                "indices": indices,
                "type": type_,
                "title": title,
            })

        if type_ not in GROUPABLE_TYPES:
            _reject(f"type={type_!r} 不在 {sorted(GROUPABLE_TYPES)}")
            continue
        if not indices:
            _reject("indices 为空")
            continue
        if any(iv < 1 or iv > n for iv in indices):
            _reject(f"indices 越界（窗口 {n} 句）")
            continue
        if len(indices) < MIN_GROUP_SIZE or len(indices) > MAX_GROUP_SIZE:
            _reject(f"组大小 {len(indices)} 不在 [{MIN_GROUP_SIZE},{MAX_GROUP_SIZE}]")
            continue
        if used & set(indices):
            _reject(f"句子与已用组冲突: {sorted(used & set(indices))}")
            continue
        # 组内顺序保持窗口原序（学习顺序 = lesson order 序）
        ordered = sorted(indices)
        used.update(ordered)
        if not title:
            first_entry = window_sentences[ordered[0] - 1]
            first_text = str(first_entry.get("text") or "").strip()
            title = (first_text[:20] or f"组{i + 1}").strip()
        # sentence_ids 展开折叠句（重复句同组同 role；role 按唯一句位次定）
        sentence_ids: list[str] = []
        for idx in ordered:
            sentence_ids.extend(entry_sentence_ids(window_sentences[idx - 1]))
        groups.append({
            "type": type_,
            "title": title,
            "reason": reason,
            "indices": ordered,
            "sentence_ids": sentence_ids,
        })
    return {"groups": groups, "invalid": invalid, "used": used}


def infer_role_in_group(group_type: str, index: int) -> str:
    """句内角色推断（与 services/english/sentence_group.py._infer_role_in_group 同口径）。

    dialogue_pair：首句 question、次句 answer_A、其余 statement；其余组型全 statement。
    """
    if group_type == "dialogue_pair":
        if index == 0:
            return "question"
        if index == 1:
            return "answer_A"
    return "statement"


# ===========================================================================
# 数据拉取（只读）
# ===========================================================================


async def list_textbooks(db, textbook_id: str = "") -> list[dict]:
    """枚举待处理教材：--textbook-id 单本，否则全部 subject_type=english 教材。"""
    if textbook_id:
        tb = (await db.query(
            collection=TEXTBOOK_V2, where={"_id": textbook_id}, limit=1
        )).get("records", [])
        if not tb:
            # 兼容 textbook_id 字段存储
            tb = (await db.query(
                collection=TEXTBOOK_V2,
                where={"textbook_id": textbook_id}, limit=1,
            )).get("records", [])
        if not tb:
            print(f"[ERROR] 教材不存在: {textbook_id}")
            sys.exit(1)
        return tb
    return await query_all_pages(db, collection=TEXTBOOK_V2, where={"subject_type": "english"})


async def collect_lesson_tasks(
    db,
    textbooks: list[dict],
    *,
    lesson_id: str = "",
    window: int = DEFAULT_WINDOW,
) -> list[dict]:
    """对每本教材每课收集「未分组唯一句窗口」任务。

    新概念等课内同文本复制多遍：先 _is_ungrouped 过滤，再 collapse_by_text 按
    归一化文本折叠成唯一句（代表句 + sentence_ids 展开全量 + dup_count），LLM
    只对唯一句做分组决策，窗口按唯一句切分（窗口间不跨组）。

    task = {textbook_id, textbook_title, lesson_id, chapter_id, lesson_order,
            lesson_title, window_no, sentences}   # sentences=唯一句条目
    """
    tasks: list[dict] = []
    for tb in textbooks:
        tid = tb.get("textbook_id") or tb.get("_id")
        tb_title = tb.get("title") or tid
        lessons = await get_lessons_by_textbook(db, tid)
        if lesson_id:
            lessons = [l for l in lessons if (l.get("lesson_id")) == lesson_id]
        for lesson in lessons:
            lid = lesson.get("lesson_id")
            sents = await get_sentences_by_lesson(db, lid, limit=2000)
            unique_rows = collapse_by_text([s for s in sents if _is_ungrouped(s)])
            if not unique_rows:
                continue
            for w_no, w_sents in enumerate(split_windows(unique_rows, window)):
                tasks.append({
                    "textbook_id": tid,
                    "textbook_title": tb_title,
                    "lesson_id": lid,
                    "chapter_id": lesson.get("chapter_id") or "",
                    "lesson_order": lesson.get("order"),
                    "lesson_title": lesson.get("title") or lid,
                    "window_no": w_no,
                    "sentences": w_sents,
                })
    return tasks


# ===========================================================================
# 混元（hy3）分组调用
# ===========================================================================

_SYSTEM_PROMPT = """你是小学英语教材内容编排助手。你会收到一个课时(lesson)内的全部待分组英文语句（含中文翻译与序号）。请按语义相关性把它们分成若干「语句组」，供学生按组学习。

判定可同组的依据（满足其一即可）：
1. 对话问答配对：一句提问 + 对应回答；或一段连贯的微型对话（问-答-问-答）
2. 同场景/同话题连贯：同一语境下相互衔接、合起来才完整的语句
3. 同语法句型练习族：结构相同的句型或替换练习（如 Is this ...? Yes, it is. / No, it is not.）
4. 同词汇主题族：围绕同一主题词/词汇的句子（如颜色、数字、食物、动物）

硬性规则：
- 每组 2~6 句，优先 2~4 句；组内句子合并后应是一个语义原子（语义原子化），不要为了凑数把不相关的句子并在一起
- 每句只能属于一个组；组内顺序保持清单中的先后顺序（学习顺序）
- 只在给出的清单内分组：严禁跨课/跨清单、严禁改写或新增任何语句
- 无法与任何语句组成有意义组的语句 → 放进 skip_indices，不要强行凑组
- type 只能是 dialogue_pair（对话问答）/ grammar_family（同句型练习）/ vocab_family（同词汇主题）

只输出一个 JSON 对象，不要输出任何解释、注释或 markdown：
{"groups": [{"indices": [1, 3], "type": "dialogue_pair", "title": "简短中文组标题", "reason": "一句话中文说明为何同组"}], "skip_indices": [2, 5]}"""

_USER_TEMPLATE = """教材：{textbook_title}
课时：{lesson_title}
待分组语句共 {n} 句（序号 1~{n}，按课内顺序）：

{sentences}"""


def build_window_prompt(task: dict) -> list[dict]:
    """构建单个窗口任务的 messages（纯函数）。"""
    lines = []
    for i, s in enumerate(task["sentences"], start=1):
        text = str(s.get("text") or "").strip()
        trans = str(s.get("translation") or "").strip()
        lines.append(f'{i}. text: "{text}"' + (f' / translation: "{trans}"' if trans else ""))
    user = _USER_TEMPLATE.format(
        textbook_title=task["textbook_title"],
        lesson_title=task["lesson_title"],
        n=len(task["sentences"]),
        sentences="\n".join(lines),
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def _call_hunyuan(messages: list[dict]) -> str:
    """调用混元 hy3（OpenAI 兼容网关）。失败/空 content 抛异常（上层重试）。"""
    if not (HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL and HUNYUAN_BASE_URL):
        raise RuntimeError("混元凭据未配置（HUNYUAN_SECRET_KEY / HUNYUAN_EVAL_MODEL / HUNYUAN_BASE_URL）")
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=HUNYUAN_SECRET_KEY,
        base_url=HUNYUAN_BASE_URL,
        timeout=HUNYUAN_TIMEOUT_SECONDS,
        max_retries=0,
    )
    resp = await client.chat.completions.create(
        model=HUNYUAN_EVAL_MODEL,
        messages=messages,
        temperature=0.0,
    )
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError(
            f"LLM 响应 content 为空（model={HUNYUAN_EVAL_MODEL}，"
            f"base_url={HUNYUAN_BASE_URL} 须为 OpenAI 兼容 /chat/completions 网关）"
        )
    return content


async def plan_window(
    task: dict,
    *,
    retries: int = 2,
) -> dict:
    """对一个窗口任务调 LLM 分组。

    Returns: {ok, groups, invalid, kept_ungrouped, error}
    """
    messages = build_window_prompt(task)
    last_err = ""
    for attempt in range(retries + 1):
        try:
            content = await _call_hunyuan(messages)
            parsed = parse_llm_content(content)
            if not parsed["ok"]:
                last_err = parsed["error"] or "解析失败"
                raise ValueError(last_err)
            norm = normalize_window_groups(parsed["obj"], task["sentences"])
            kept = compute_kept_sentence_ids(task["sentences"], norm["groups"])
            return {
                "ok": True,
                "groups": norm["groups"],
                "invalid": norm["invalid"],
                "kept_ungrouped": kept,
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(1.0 * (attempt + 1))
    return {
        "ok": False,
        "groups": [],
        "invalid": [],
        "kept_ungrouped": compute_kept_sentence_ids(task["sentences"], []),
        "error": last_err,
    }


async def plan_all(
    tasks: list[dict],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict]:
    """并发规划全部窗口任务；单任务失败不中断整体（error 入结果）。"""
    sem = asyncio.Semaphore(max(int(concurrency), 1))

    async def _one(task: dict) -> dict:
        async with sem:
            result = await plan_window(task)
            out = dict(task)
            out.update(result)
            return out

    return [await _one(t) for t in tasks]


# ===========================================================================
# apply：写 sentence_group + sentence_v2 回写
# ===========================================================================


async def _lesson_base_order(db, lesson_id: str) -> int:
    """该 lesson 现有组最大 order_in_lesson + 1（新组接续其后）。"""
    existing = await query_all_pages(
        db, collection=SENTENCE_GROUP, where={"lesson_id": lesson_id},
        select={"order_in_lesson": 1},
    )
    return max((int(g.get("order_in_lesson") or -1) for g in existing), default=-1) + 1


async def _all_sentences_ungrouped(db, sentence_ids: list[str]) -> bool:
    """apply 幂等防御：组内句子必须当前仍未分组（否则说明已建过，跳过）。"""
    if not sentence_ids:
        return False
    docs = await query_all_pages(
        db, collection=SENTENCE_V2,
        where={"sentence_id": {"$in": sentence_ids}},
        select={"sentence_id": 1, "group_id": 1},
    )
    return all(not d.get("group_id") for d in docs)


async def apply_groups(db, plan_results: list[dict]) -> dict:
    """按规划写库：insert sentence_group + sentence_v2 回写 group_id/role_in_group。

    group 排序：同一 lesson 内先窗口序（=句子 order 序），再组内顺序；
    order_in_lesson 接续该 lesson 已有组。
    """
    now_ms = int(time.time() * 1000)
    stats = {"groups_written": 0, "sentences_updated": 0, "skipped_existing": 0, "errors": 0}
    detail: list[dict] = []

    # lesson 内组序：窗口按句子 order 起点排序后顺序分配 order
    # 已按 lesson 收集（plan_results 已按窗口序）
    by_lesson: dict[str, list[dict]] = defaultdict(list)
    for pr in plan_results:
        if pr.get("ok") and pr["groups"]:
            by_lesson[pr["lesson_id"]].append(pr)

    for lesson_id, prs in by_lesson.items():
        base_order = await _lesson_base_order(db, lesson_id)
        order_cursor = base_order
        for pr in prs:
            textbook_id = pr["textbook_id"]
            chapter_id = pr.get("chapter_id") or ""
            window_sents = pr["sentences"]
            for g_idx, g in enumerate(pr["groups"]):
                # role 按「组内唯一句位次」推断，折叠句（同文本重复）继承代表句 role；
                # 展开顺序 = 唯一句窗口序 → 组内展开 id 保序
                sid_role: list[tuple[str, str]] = []
                for pos, idx in enumerate(g["indices"]):
                    if not (1 <= idx <= len(window_sents)):
                        continue
                    role = infer_role_in_group(g["type"], pos)
                    sid_role.extend((sid, role) for sid in entry_sentence_ids(window_sents[idx - 1]))
                sids = [sid for sid, _ in sid_role]
                if not sids:
                    continue
                if not await _all_sentences_ungrouped(db, sids):
                    stats["skipped_existing"] += 1
                    detail.append({
                        "lesson_id": lesson_id,
                        "title": g["title"],
                        "action": "skipped_existing",
                    })
                    continue
                try:
                    group_id = build_sentence_group_id(
                        textbook_id, lesson_id, now=now_ms + stats["groups_written"]
                    )
                    doc = build_sentence_group_doc(
                        group_id=group_id,
                        textbook_id=textbook_id,
                        lesson_id=lesson_id,
                        title=g["title"],
                        type_=g["type"],
                        sentence_ids=sids,
                        order_in_lesson=order_cursor,
                        chapter_id=chapter_id,
                        build_version=f"llm_hy3_{int(now_ms / 1000)}",
                        now=now_ms,
                    )
                    await db.insert(collection=SENTENCE_GROUP, data=doc)
                    for sid, role in sid_role:
                        await db.update(
                            collection=SENTENCE_V2,
                            where={"sentence_id": sid},
                            data={"$set": {
                                "group_id": group_id,
                                "role_in_group": role,
                                "updated_at": now_ms,
                            }},
                            multi=False,
                        )
                    order_cursor += 1
                    stats["groups_written"] += 1
                    stats["sentences_updated"] += len(sids)
                    detail.append({
                        "lesson_id": lesson_id,
                        "group_id": group_id,
                        "title": g["title"],
                        "type": g["type"],
                        "sentence_count": len(sids),
                        "order_in_lesson": order_cursor - 1,
                        "action": "written",
                    })
                except Exception as e:  # noqa: BLE001
                    stats["errors"] += 1
                    detail.append({
                        "lesson_id": lesson_id,
                        "title": g["title"],
                        "action": "error",
                        "error": f"{type(e).__name__}: {e}",
                    })
    return {"stats": stats, "detail": detail}


# ===========================================================================
# 报告
# ===========================================================================


def _fmt_sid(sid) -> str:
    return str(sid or "")[-8:]


def render_report(
    plan_results: list[dict],
    *,
    textbooks: list[dict],
    apply: bool,
) -> str:
    out: list[str] = []
    P = out.append
    mode = "WRITE" if apply else "DRY-RUN"
    P("=" * 100)
    P(f"存量教材未分组语句 LLM 语义分组 group_lesson_sentences.py（{mode}，hy3）")
    P("=" * 100)

    by_book: dict[str, list[dict]] = defaultdict(list)
    for pr in plan_results:
        by_book[pr["textbook_id"]].append(pr)

    total_groups = 0
    total_kept = 0
    failed_tasks = []
    invalid_groups = []
    type_counter: Counter = Counter()
    for tid, prs in by_book.items():
        title = prs[0]["textbook_title"] if prs else tid
        P(f"\n[book] 《{title}》 {tid}  （任务 {len(prs)} 个）")
        # 窗口任务按课/窗口排
        keyed = sorted(
            prs,
            key=lambda x: (
                (x.get("lesson_order") is None),
                x.get("lesson_order") or 0,
                x.get("window_no") or 0,
            ),
        )
        for pr in keyed:
            if not pr.get("ok"):
                failed_tasks.append(pr)
                P(f"  [FAIL] L{pr.get('lesson_order')} {str(pr['lesson_title'])[:20]} "
                  f"窗口{pr.get('window_no')}  {pr.get('error')}")
                continue
            groups = pr["groups"]
            kept = pr.get("kept_ungrouped") or []
            total_groups += len(groups)
            total_kept += len(kept)
            type_counter.update(g["type"] for g in groups)
            uniq = len(pr["sentences"])
            total = sum(len(entry_sentence_ids(s)) for s in pr["sentences"])
            P(f"  L{pr.get('lesson_order')} {str(pr['lesson_title'])[:22]}  "
              f"窗口{pr.get('window_no')} 句子{total}"
              + (f"（唯一句{uniq}）" if uniq != total else "")
              + f" → 成组 {len(groups)} 组 / 保留未分组 {len(kept)} 句")
            for g in groups:
                g_sids = set(g["sentence_ids"])
                head = " | ".join(
                    str(s.get("text") or "")[:34].replace("\n", " ")
                    for s in pr["sentences"]
                    if set(entry_sentence_ids(s)) & g_sids
                )
                P(f"      + [{g['type']}] {g['title']}（{len(g['sentence_ids'])}句）"
                  + (f"：{head}" if head else ""))
            for inv in pr.get("invalid") or []:
                invalid_groups.append(inv)
                P(f"      ! 非法组建议: {inv.get('error')}  indices={inv.get('indices')}")

    if invalid_groups:
        P(f"\n[非法组建议] {len(invalid_groups)} 条（LLM 越界/冲突/超大小，已拒绝不写）")
    if failed_tasks:
        P(f"\n[失败任务] {len(failed_tasks)} 个（未产生组，可重跑；见上 FAIL 行）")

    P(f"\n[合计] 教材 {len(textbooks)} 本 / 任务 {len(plan_results)} 个："
      f"计划成组 {total_groups} 组"
      + (f"（type: " + "  ".join(f"{k}={v}" for k, v in sorted(type_counter.items())) + "）" if type_counter else "")
      + f" / 保留未分组 {total_kept} 句")
    if not apply:
        P("dry-run 未写库；过目后加 --apply 执行（或 --json 落盘计划后 --plan-file --apply 复用，免二次 LLM）")
    return "\n".join(out)


# ===========================================================================
# main
# ===========================================================================


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="存量教材未分组语句 LLM 语义分组（hy3；默认 dry-run 出计划，--apply 写库）"
    )
    parser.add_argument("--textbook-id", default="", help="限定教材（单本 dry-run/apply）")
    parser.add_argument("--lesson-id", default="", help="限定课（配合 --textbook-id）")
    parser.add_argument("--all", action="store_true", help="全部英语教材")
    parser.add_argument("--apply", action="store_true", help="写库；缺省仅 dry-run")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"单任务语句数上限（默认 {DEFAULT_WINDOW}；超出的课自动切窗口任务）")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"并发 LLM 任务数（默认 {DEFAULT_CONCURRENCY}）")
    parser.add_argument("--json", default="", help="可选：计划落盘 JSON（供 --plan-file 复用免二次 LLM）")
    parser.add_argument("--plan-file", default="",
                        help="读入先前 --json 落盘的计划，跳过 LLM 直接 dry-run/apply")
    args = parser.parse_args()

    if not args.textbook_id and not args.all and not args.plan_file:
        parser.error("请指定 --textbook-id / --all / --plan-file 之一")
    if args.textbook_id and args.all:
        parser.error("--textbook-id 与 --all 互斥")
    if args.lesson_id and not args.textbook_id:
        parser.error("--lesson-id 需配合 --textbook-id")

    db = get_db()

    if args.plan_file:
        payload = json.loads(Path(args.plan_file).read_text(encoding="utf-8"))
        plan_results = payload["tasks"]
        textbooks = payload.get("textbooks", [])
        print(f"[plan-file] 载入计划 {args.plan_file}：任务 {len(plan_results)} 个（跳过 LLM）")
    else:
        textbooks = await list_textbooks(db, args.textbook_id)
        tasks = await collect_lesson_tasks(
            db, textbooks, lesson_id=args.lesson_id, window=args.window
        )
        if not tasks:
            print("未找到任何未分组语句（全部已分组或教材无句子）")
            return
        print(f"收集任务 {len(tasks)} 个（教材 {len(textbooks)} 本），开始 hy3 分组…")
        plan_results = await plan_all(tasks, concurrency=args.concurrency)
        textbooks_payload = [{
            "textbook_id": t.get("textbook_id") or t.get("_id"),
            "title": t.get("title") or "",
        } for t in textbooks]

    report = render_report(plan_results, textbooks=textbooks, apply=args.apply)
    print(report)

    if args.json and not args.plan_file:
        Path(args.json).write_text(
            json.dumps({
                "textbooks": textbooks_payload,
                "tasks": plan_results,
            }, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n计划 JSON 已输出: {args.json}（可 --plan-file 复用，免二次 LLM）")

    if args.apply:
        r = await apply_groups(db, plan_results)
        st = r["stats"]
        print(f"\n[WRITE] 建组 {st['groups_written']} 个 / 句子回写 {st['sentences_updated']} 条 / "
              f"已存在跳过 {st['skipped_existing']} / 错误 {st['errors']}")
        for d in r["detail"]:
            if d.get("action") in ("written",):
                print(f"  + {str(d['lesson_id'])[-8:]} {d['title']} "
                      f"[{d['type']}] {d['sentence_count']}句 order={d['order_in_lesson']} {d['group_id']}")
            elif d.get("action") == "error":
                print(f"  ! {d.get('lesson_id')} {d.get('title')} 写入失败: {d.get('error')}")


if __name__ == "__main__":
    asyncio.run(main())
