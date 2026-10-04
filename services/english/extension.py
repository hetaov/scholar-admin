"""英文语句扩展 — 抽取核心（校验 / 配额 / 去重 / 规则兜底 / L1 判分）

契约：data-model-contract.md §4.24；api-contract.md §3.18。
设计：第一期设计 §3.2.3 / §3.5 / §3.6 / §3.7。

抽取为**分层流水线**（策略：docs_v1/扩展/重点词汇短语抽取策略.md §6）：
    文本 → 规则/词表生成候选（extension_candidates）→ LLM 语境筛选与释义
         → 候选映射校验（map_candidate_points）→ 配额/重叠 → 审核（audit_points）
         → 排序（rank_points）→ 跨句去重 → L1 题面 → 落库

核心函数：
- extension_idempotency_key(sentence_id, content_hash, prompt_version, model) → sha256
- validate_and_normalize_points(raw, original) → (valid, discarded)  # schema + span（自由抽取口径）
- map_candidate_points(raw_items, candidates, original) → (valid, discarded)  # 候选口径：模型只回 candidate_id
- apply_quota_and_overlap(points) → points                        # 配额 + 重叠优先级
- audit_points(points) → points                                   # 敏感词与质量审核（标 risk，不删）
- rank_points(points) → points                                    # 按学习价值排序（仅展示序）
- mark_cross_sentence_excluding(points, seen_texts) → points      # 跨句去重标 excluding
- rule_fallback_points(original) → points                         # 停用词 + 实词（只产 word）
- build_l1_items(points, original, candidates) → points[].l1       # L1 挖空题面 + 4 选项
- grade_l1_fill(point, user_answer) → {score, passed, ...}        # 确定性判分
- run_extract_pipeline(db, ...) → result                          # E1 后台执行器
- run_evaluate_pipeline(db, ...) → result                         # E2 后台执行器

红线：
- R1：不写句子 mastery status（本模块零调用 SKILL_STATE 接口）
- R6：非 force_refresh 不覆盖 source=manual（同 sentence_id 存在 manual 即跳过）
- R7：span 切出字符 ≠ text → 丢该条（不是整句失败）
- R8：失败链 重试1次 → 规则兜底 → failed+fallback_reason
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
import time
from typing import Any

import config
from services.english.extension_candidates import (
    RISKY_TERMS,
    STOPWORDS as _STOPWORDS,
    generate_candidates,
)
from services.models.content import compute_text_hash, normalize_sentence_text
from services.providers.extension_llm import (
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT,
    ERR_PROVIDER_UNAVAILABLE,
    ExtensionError,
    STAGE_PARSE,
    build_extension_messages,
    call_extension_llm,
    extract_json,
)

logger = logging.getLogger("scholar-admin.extension")

# ===========================================================================
# 常量
# ===========================================================================

POINT_TYPES = ("word", "phrase", "slang", "idiom")
TYPE_PRIORITY = {"idiom": 4, "slang": 3, "phrase": 2, "word": 1}
# 配额（v2.md §2.2）
QUOTA = {"word": 3, "phrase": 2, "slang": 1, "idiom": 1}
TOTAL_QUOTA = config.EXTENSION_MAX_POINTS_PER_SENTENCE  # 5

# ---- L1 填空选项（零 LLM、零抖动；见 build_l1_items）------------------------
# 选项总数：1 个正解 + 最多 3 个干扰项
L1_OPTIONS_MAX = 4
# 生成题面所需的最小干扰项数：低于此值 → 该题 l1=None（不出题）。
# 只有正解、零干扰项的单选项题等于送分，不如不出。
L1_DISTRACTORS_MIN = 1

REGISTER_VALUES = ("neutral", "formal", "informal")
CEFR_VALUES = ("A1", "A2", "B1", "B2", "C1", "C2")
# 慎用表达风险等级（""=无风险）
RISK_VALUES = ("", "offensive", "vulgar", "regional", "dated")

# 停用词表见 services/english/extension_candidates.py::STOPWORDS（本模块 _STOPWORDS 即其别名，
# 供规则兜底抽取实词用）。

# 英文标点（归一化用）
_PUNCT_RE = re.compile(r"[^\w\s'-]", re.UNICODE)

# 抽取最大尝试次数（= 首次 + 最多 1 次重试）
MAX_EXTRACT_ATTEMPTS = 2

# 值得重试的错误码：只有「供应商不可用」这类**瞬时故障**重试才有意义。
# 解析失败（LLM_PARSE_ERROR / 候选校验全废）是模型输出问题，同 prompt 重试大概率得到
# 同样的坏结果，却要多付一次完整 LLM 耗时——在慢模型上直接翻倍（R8 失速主因之一）。
RETRYABLE_ERROR_CODES = frozenset({ERR_PROVIDER_UNAVAILABLE})


# ===========================================================================
# 幂等键（§3.7）
# ===========================================================================


def extension_idempotency_key(
    *,
    sentence_id: str,
    content_hash: str,
    prompt_version: str,
    model: str,
) -> str:
    """抽取幂等键：sha256(sentence_id|content_hash|prompt_version|model)。

    升 prompt_version / model / 内容变更 → 自然失效缓存。
    """
    raw = f"{sentence_id}|{content_hash}|{prompt_version}|{model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ===========================================================================
# Schema + span 校验（R7：坏字段丢该条，不整句失败）
# ===========================================================================


def _validate_point(p: dict, original: str, index: int) -> dict | None:
    """校验单条语言点，返回规范化后的 point 或 None（丢弃）。

    丢弃条件（任一命中即丢该条）：
    - type 不在枚举
    - span 非法 / 切出字符 ≠ text
    - 必填字段缺失 / 超长
    - example_en 不含 text
    """
    if not isinstance(p, dict):
        return None

    ptype = p.get("type")
    if ptype not in POINT_TYPES:
        return None

    text = p.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()

    # span 解析：优先 spans（多区间），回退 span（单区间）
    spans_raw = p.get("spans")
    if isinstance(spans_raw, list) and spans_raw:
        spans: list[list[int]] = []
        ok = True
        for s in spans_raw:
            if not isinstance(s, (list, tuple)) or len(s) != 2:
                ok = False
                break
            try:
                spans.append([int(s[0]), int(s[1])])
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            return None
    else:
        span = p.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            return None
        try:
            spans = [[int(span[0]), int(span[1])]]
        except (TypeError, ValueError):
            return None

    # 区间合法性：升序、不重叠、在原文范围内
    prev_end = -1
    for s in spans:
        start, end = s[0], s[1]
        if not (0 <= start < end <= len(original)):
            return None
        if start <= prev_end:
            return None
        prev_end = end
    # R7：所有区间切出字符拼接后必须等于 text。
    # 非连续短语的 text 含空格（如 "turn off"），但 spans 只覆盖词不覆盖中间插入词，
    # 故两侧去空白后比对；连续短语去空白后仍相等，无副作用。
    joined = "".join(original[s[0]:s[1]] for s in spans)
    if re.sub(r"\s+", "", joined) != re.sub(r"\s+", "", text):
        return None

    discontinuous = len(spans) > 1

    meaning_zh = p.get("meaning_zh", "")
    if not isinstance(meaning_zh, str) or not meaning_zh.strip():
        return None
    meaning_zh = meaning_zh.strip()[:30]

    register = p.get("register", "neutral")
    if register not in REGISTER_VALUES:
        register = "neutral"
    # slang 强制 informal
    if ptype == "slang":
        register = "informal"

    literal_zh = p.get("literal_zh", "")
    if not isinstance(literal_zh, str):
        literal_zh = ""
    literal_zh = literal_zh.strip()
    # slang 必填 literal_zh
    if ptype == "slang" and not literal_zh:
        return None

    example_en = p.get("example_en", "")
    example_zh = p.get("example_zh", "")
    if not isinstance(example_en, str) or not example_en.strip():
        return None
    example_en = example_en.strip()
    # example_en 必须包含 text（大小写不敏感）
    if text.lower() not in example_en.lower():
        return None
    if not isinstance(example_zh, str):
        example_zh = ""

    pos = p.get("pos", "")
    if not isinstance(pos, str):
        pos = ""
    pos = pos.strip()

    collocations = p.get("collocations", [])
    if not isinstance(collocations, list):
        collocations = []
    collocations = [
        str(c)[:60] for c in collocations if isinstance(c, str) and c.strip()
    ]
    # phrase 必填 collocations ≥1
    if ptype == "phrase" and len(collocations) < 1:
        return None

    synonyms = p.get("synonyms", [])
    if not isinstance(synonyms, list):
        synonyms = []
    synonyms = [str(s) for s in synonyms if isinstance(s, str) and s.strip()]

    confusable = p.get("confusable", [])
    if not isinstance(confusable, list):
        confusable = []
    confusable = [str(c) for c in confusable if isinstance(c, str) and c.strip()]

    cefr = p.get("cefr", "")
    if cefr not in CEFR_VALUES:
        cefr = ""

    reason = p.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        return None
    reason = reason.strip()[:40]

    # confidence：非法 / 缺失 → 0.0，夹到 [0, 1]（仅作排序 tie-break，不参与配额）
    try:
        confidence = float(p.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    # risk：不在枚举 → ""（标记不删除）
    risk = p.get("risk", "")
    if not isinstance(risk, str) or risk not in RISK_VALUES:
        risk = ""

    return {
        "id": "",  # 由调用方按 sentence_id#index 重算
        "type": ptype,
        "text": text,
        "span": spans[0],            # 兼容旧消费方：取首区间
        "spans": spans,              # 多区间（连续点为单区间）
        "discontinuous": discontinuous,
        "pos": pos,
        "meaning_zh": meaning_zh,
        "literal_zh": literal_zh,
        "register": register,
        "example_en": example_en,
        "example_zh": example_zh.strip(),
        "collocations": collocations,
        "synonyms": synonyms,
        "confusable": confusable,
        "cefr": cefr,
        "reason": reason,
        "confidence": confidence,
        "risk": risk,
        "excluding": False,
    }


def map_candidate_points(
    raw_items: list[dict], candidates: list[dict], original: str
) -> tuple[list[dict], int]:
    """候选口径映射：把 LLM 输出按 `candidate_id` 映射回候选自带的 text/span，再走 schema 校验。

    **不信任模型给出的 text/span**（模型只回 `candidate_id`），故 R7 天然成立。
    丢弃条件（任一命中即丢该条，丢点不丢句）：
    - `candidate_id` 缺失 / 未知 / 重复
    - 字段校验不过（`_validate_point` 返回 None）

    Returns:
        (valid_points, discarded_count)；`valid` 为空时由调用方判 `LLM_PARSE_ERROR`。
    """
    by_id = {c["candidate_id"]: c for c in candidates}
    valid: list[dict] = []
    discarded = 0
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            discarded += 1
            continue
        cid = item.get("candidate_id")
        if not isinstance(cid, str) or cid not in by_id or cid in seen:
            discarded += 1
            continue
        seen.add(cid)
        cand = by_id[cid]
        merged = dict(item)
        merged["text"] = cand["text"]
        merged["span"] = list(cand["span"])
        merged["spans"] = [list(s) for s in cand.get("spans", [cand["span"]])]
        norm = _validate_point(merged, original, len(valid))
        if norm is None:
            discarded += 1
            continue
        valid.append(norm)
    return valid, discarded


def validate_and_normalize_points(
    raw_points: list[dict], original: str
) -> tuple[list[dict], int]:
    """批量校验 + 规范化。返回 (valid_points, discarded_count)。

    坏字段丢该条不整句失败（R7）。id 暂空，由后续按顺序重算。
    """
    valid: list[dict] = []
    discarded = 0
    for p in raw_points:
        norm = _validate_point(p, original, len(valid))
        if norm is None:
            discarded += 1
            continue
        valid.append(norm)
    return valid, discarded


# ===========================================================================
# 配额 + 重叠优先级（§3.5）
# ===========================================================================


def _spans_overlap(a: list[int], b: list[int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _point_spans_overlap(a: dict, b: dict) -> bool:
    """两点是否重叠：任一组区间相交即重叠（支持多 span）。"""
    a_spans = a.get("spans") or [a["span"]]
    b_spans = b.get("spans") or [b["span"]]
    for sa in a_spans:
        for sb in b_spans:
            if _spans_overlap(sa, sb):
                return True
    return False


def apply_quota_and_overlap(points: list[dict]) -> list[dict]:
    """应用配额与 span 重叠优先级。

    - 重叠：idiom > slang > phrase > word，保留高优先级，剔除低优先级；
      多 span 点任一组区间相交即视为重叠。
    - 配额：word≤3 / phrase≤2 / slang≤1 / idiom≤1，合计 ≤5；
      超限按「固定度高的先留、其余按原序截断」。
    """
    # 1. 先按优先级排序（高→低），原序作为次级排序键
    indexed = list(enumerate(points))
    indexed.sort(key=lambda x: (-TYPE_PRIORITY[x[1]["type"]], x[0]))

    # 2. 重叠剔除：遍历排序后的，保留不与已保留重叠的
    kept: list[dict] = []
    for _, p in indexed:
        if any(_point_spans_overlap(p, k) for k in kept):
            continue
        kept.append(p)

    # 3. 配额截断
    count_by_type = {"word": 0, "phrase": 0, "slang": 0, "idiom": 0}
    result: list[dict] = []
    for p in kept:
        t = p["type"]
        if count_by_type[t] >= QUOTA[t]:
            continue
        if len(result) >= TOTAL_QUOTA:
            break
        count_by_type[t] += 1
        result.append(p)

    # 4. 恢复原序
    result.sort(key=lambda p: points.index(p) if p in points else 0)
    return result


# ===========================================================================
# 敏感词与质量审核（策略 §4：慎用表达标 risk）
# ===========================================================================


def _normalize_term(text: str) -> str:
    """词表比对用的归一化：小写 + 弯撇号归一 + 压缩空白。"""
    return " ".join(text.strip().lower().replace("’", "'").split())


def audit_points(points: list[dict]) -> list[dict]:
    """敏感词与质量审核：为词表中标注的高危表达补 `risk`；**只标记不删除**。

    消费者：一期＝实验页风险角标；二期门禁＝面向学生的界面须过滤 `risk != ""`
    （不得静默上屏）。此处不删点，便于实验页观察命中情况。
    """
    for p in points:
        forced = RISKY_TERMS.get(_normalize_term(p["text"]))
        if forced and not p.get("risk"):
            p["risk"] = forced
    return points


# ===========================================================================
# 按学习价值排序（策略 §4；仅展示序，不参与配额）
# ===========================================================================


def rank_points(points: list[dict]) -> list[dict]:
    """按学习价值排序：固定度（idiom > slang > phrase > word）降序 → confidence 降序。

    稳定排序（同键保持入参相对顺序）。**仅决定展示顺序**；配额与重叠剔除在
    `apply_quota_and_overlap` 中已完成，不依赖本函数的顺序。
    """
    return sorted(
        points,
        key=lambda p: (-TYPE_PRIORITY.get(p["type"], 0), -float(p.get("confidence", 0.0))),
    )


# ===========================================================================
# 跨句去重（标 excluding:true，不删除）
# ===========================================================================


def mark_cross_sentence_excluding(
    points: list[dict], seen_normalized_texts: set[str]
) -> list[dict]:
    """标记跨句重复的语言点 excluding=true（不删除，供实验页观察）。

    seen_normalized_texts：此前句子已出现的归一化 text 集合。
    """
    for p in points:
        norm = normalize_sentence_text(p["text"])
        if norm in seen_normalized_texts:
            p["excluding"] = True
        else:
            seen_normalized_texts.add(norm)
            p["excluding"] = False
    return points


# ===========================================================================
# 规则兜底（B08：只产 word）
# ===========================================================================


def rule_fallback_points(original: str) -> list[dict]:
    """停用词 + 实词抽取（只产 word 类型）。

    当 LLM 失败时启用（EXTENSION_RULE_FALLBACK_ENABLED=1）。
    抽取原句中首个非停用词、长度 ≥3 的单词作为 word 类型语言点。
    """
    if not config.EXTENSION_RULE_FALLBACK_ENABLED:
        return []

    # 去除标点，按空白切词
    cleaned = _PUNCT_RE.sub(" ", original)
    words = [w for w in cleaned.split() if w]
    fallback: list[dict] = []
    for w in words:
        lw = w.lower().strip("'")
        if lw in _STOPWORDS:
            continue
        if len(lw) < 3:
            continue
        # 在原句中定位 span
        idx = original.lower().find(lw)
        if idx < 0:
            continue
        fallback.append(
            {
                "id": "",
                "type": "word",
                "text": original[idx : idx + len(lw)],
                "span": [idx, idx + len(lw)],
                "spans": [[idx, idx + len(lw)]],
                "discontinuous": False,
                "pos": "",
                "meaning_zh": "（规则兜底，需人工补充释义）",
                "literal_zh": "",
                "register": "neutral",
                "example_en": original,
                "example_zh": "",
                "collocations": [],
                "synonyms": [],
                "confusable": [],
                "cefr": "",
                "reason": "规则兜底抽取",
                "confidence": 0.0,
                "risk": "",
                "excluding": False,
            }
        )
        if len(fallback) >= QUOTA["word"]:
            break
    return fallback


# ===========================================================================
# L1 题面与判分（B12：零 LLM、零抖动）
# ===========================================================================


def _l1_distractor_pool(
    points: list[dict], candidates: list[dict] | None
) -> list[dict]:
    """候选层干扰项池：未被选为语言点的候选（L1 干扰项的兜底来源）。

    只保留未入选的候选（归一化比对），保持原句顺序，便于确定性取用。
    """
    chosen = {_normalize_answer(p["text"]) for p in points}
    pool: list[dict] = []
    for idx, c in enumerate(candidates or []):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if _normalize_answer(text) in chosen:
            continue
        span = c.get("span")
        pool.append(
            {
                "text": text,
                "hint_type": c.get("hint_type", ""),
                "span": list(span) if span else None,
                "idx": idx,
            }
        )
    return pool


def _l1_pick_distractors(
    text: str,
    ptype: str,
    pool: list[dict],
    limit: int,
    answer_spans: list | None = None,
) -> list[str]:
    """从候选池挑 ≤limit 条干扰项：同 hint_type 优先 → 长度接近优先 → 原句顺序。

    排除两类「送分 / 纠缠」候选：
    - 与正解互为子串（take part ↔ take part in）；
    - 与正解区间重叠（正解 take part in [7,19)，候选 will take part [2,15)）
      —— 挖空处塞进一个含答案片段的选项，语义上不成立。
    """
    key = _normalize_answer(text)
    ranked = sorted(
        pool,
        key=lambda x: (
            0 if x["hint_type"] == ptype else 1,
            abs(len(x["text"]) - len(text)),
            x["idx"],
        ),
    )
    out: list[str] = []
    for item in ranked:
        cand = item["text"]
        n = _normalize_answer(cand)
        if n == key or (key and (key in n or n in key)):
            continue
        cspan = item.get("span")
        if cspan and answer_spans and any(_spans_overlap(cspan, s) for s in answer_spans):
            continue
        out.append(cand)
        if len(out) >= limit:
            break
    return out


def build_l1_items(
    points: list[dict],
    original: str = "",
    candidates: list[dict] | None = None,
) -> list[dict]:
    """为每个语言点生成 L1 挖空题面 + 4 选项（不含答案键）。

    题面：把原句中 text 的 spans 替换为 ____；
    选项：正确答案 + 干扰项，**确定性打乱**（正解不再恒在首位）。

    2026-10-03 修订：
    - **原句显式化**：`original` 提升为入参，原隐式契约 `p["_original"]` 仅作兼容
      回退（打 debug 提示改造）。两者皆缺 → `logger.warning` 且该点 `l1=None`，
      不再静默丢题。
    - **干扰项三级来源**：① 同句其它语言点 text → ② LLM `confusable`
      → ③ 候选层未入选候选。**不再补 `（选项N）` 占位**；
      真实干扰项 < `L1_DISTRACTORS_MIN` → `l1=None`；
      凑不满 3 个 → `options_incomplete=true`，由前端提示。
    - **确定性打乱**：seed = `id|text|stem`，同一句每次抽取结果一致（可复现），
      但正解不再恒在首位。判分走 `grade_l1_fill`（只比 `p["text"]`），
      与选项顺序无关，改动安全。

    出参 `l1 = { stem, options, options_incomplete }`；无法生成 → `l1 = None`。
    """
    pool = _l1_distractor_pool(points, candidates)
    skipped = 0
    for p in points:
        text = p["text"]
        spans = p.get("spans") or [p["span"]]

        # --- 原句：显式入参优先，回退隐式 _original（兼容旧调用，打日志提示改造）---
        src = original or p.get("_original", "")
        if not src:
            logger.warning(
                "[extension] build_l1_items 缺少原句（original 入参为空且 point 无 "
                "_original），跳过 L1 题面生成：id=%r text=%r —— 调用方应显式传 original",
                p.get("id", ""),
                text,
            )
            p["l1"] = None
            skipped += 1
            continue
        if not original:
            logger.debug(
                "[extension] build_l1_items 走 _original 回退（已废弃的隐式契约，"
                "请改为显式传 original）：text=%r",
                text,
            )

        # 题面：原句按 spans（多区间）挖空为 ____
        stem = original_with_blank(src, spans)
        if not stem:
            logger.warning(
                "[extension] L1 题面生成失败（span 与原句不匹配）：id=%r text=%r spans=%r",
                p.get("id", ""),
                text,
                spans,
            )
            p["l1"] = None
            skipped += 1
            continue

        # --- 干扰项：① 同句其它点 ② confusable ③ 候选池 ---
        key = _normalize_answer(text)
        seen = {key}
        distractors: list[str] = []

        def _take(c) -> None:
            c = c.strip() if isinstance(c, str) else ""
            if not c:
                return
            n = _normalize_answer(c)
            # 与已选重复，或与正解互为子串（送分）→ 跳过
            if n in seen or (key and (key in n or n in key)):
                return
            seen.add(n)
            distractors.append(c)

        for q in points:
            if q is not p:
                _take(q["text"])
        for c in p.get("confusable") or []:
            _take(c)
        if len(distractors) < L1_OPTIONS_MAX - 1:
            for c in _l1_pick_distractors(
                text,
                p.get("type", ""),
                pool,
                L1_OPTIONS_MAX - 1 - len(distractors),
                answer_spans=spans,
            ):
                _take(c)

        if len(distractors) < L1_DISTRACTORS_MIN:
            logger.warning(
                "[extension] L1 干扰项不足（<%d 条），不生成题面：id=%r text=%r "
                "（可检查 confusable 保底与候选池密度）",
                L1_DISTRACTORS_MIN,
                p.get("id", ""),
                text,
            )
            p["l1"] = None
            skipped += 1
            continue

        options = [text] + distractors[: L1_OPTIONS_MAX - 1]
        # 确定性打乱：同一次抽取可复现；判分不看 options，安全
        random.Random(f"{p.get('id', '')}|{text}|{stem}").shuffle(options)
        p["l1"] = {
            "stem": stem,
            "options": options,
            "options_incomplete": len(options) < L1_OPTIONS_MAX,
        }

    if skipped:
        logger.warning(
            "[extension] build_l1_items：%d/%d 个点未生成 L1 题面（逐条原因见上方日志）",
            skipped,
            len(points),
        )
    return points


def original_with_blank(original: str, spans) -> str:
    """按 spans 把原句挖空为 ____。

    spans 可为单区间 [s,e] 或多区间 [[s1,e1],[s2,e2],...]。
    多区间：从右往左替换，避免偏移错乱。挖空后 text 位置被 ____ 替代。
    """
    if not original or not spans:
        return ""
    # 统一为多区间列表
    if spans and isinstance(spans[0], int):
        span_list = [list(spans)]
    else:
        span_list = [list(s) for s in spans]
    # 合法性校验
    for s in span_list:
        if not (0 <= s[0] < s[1] <= len(original)):
            return ""
    # 从右往左挖空，避免左侧替换影响右侧偏移
    result = original
    for s in sorted(span_list, key=lambda x: x[0], reverse=True):
        result = result[: s[0]] + "____" + result[s[1]:]
    return result


def _normalize_answer(s: str) -> str:
    """答案归一化：去空白 + 小写 + 去标点。"""
    return normalize_sentence_text(s)


def grade_l1_fill(point: dict, user_answer: str) -> dict:
    """L1 确定性判分：normalize 后相等即命中。

    零 LLM、零抖动。passed = 归一化答案 == 归一化正确答案。
    """
    correct = _normalize_answer(point["text"])
    given = _normalize_answer(user_answer or "")
    passed = correct == given
    return {
        "score": 1 if passed else 0,
        "passed": passed,
        "correct_ids": [point["id"]] if passed else [],
        "wrong_ids": [] if passed else [point["id"]],
        "answer_key": point["text"],
    }


# ===========================================================================
# L2 rubric 判分（B13：4 维 0-2，总分 ≥6 通过）
# ===========================================================================


def grade_l2_rubric(points: list[dict], user_answer: str) -> dict:
    """L2 造句 rubric 评测（B13：4 维 0-2，总分 ≥6 通过）。

    must_use_hit 由后端独立判定：所选语言点 text 是否出现在作答中（大小写不敏感）。
    实际 LLM 4 维打分由 _call_l2_rubric_llm 完成。

    2026-10-03（B34 / 契约 §3.18 红线⑥ 与 D2）：`must_use_hit` **只针对
    `origin="ai"` 的点** —— 学习者自建点（`origin="mine"`）无后端答案键、不进判分，
    混入时既不计数也不参与命中判定。未带 `origin` 的点按 `ai` 处理（向后兼容）。
    """
    ai_points = [p for p in (points or []) if p.get("origin", "ai") != "mine"]
    used_texts = [
        p["text"] for p in ai_points if p["text"].lower() in (user_answer or "").lower()
    ]
    must_use_hit = len(used_texts) == len(ai_points) and len(ai_points) > 0
    return {
        "must_use_hit": must_use_hit,
        "used_points": used_texts,
    }


# ===========================================================================
# L2 rubric LLM 调用（B13）
# ===========================================================================

L2_RUBRIC_SYSTEM_PROMPT = """你是英语造句评测专家。请根据「目标语言点」与「学习者造句」，从 4 个维度评分（每维 0-2 分）：

1. 目标点使用（target_usage）：是否正确使用了所选语言点（词义 / 搭配 / 语法形式）
2. 语法（grammar）：句子语法是否正确
3. 语义贴合（semantics）：句子是否表达了合理、通顺的意思
4. 语域与搭配（register_collocation）：语域是否恰当、搭配是否自然

总分 ≥ 6 为通过（passed）。

严格输出 JSON：
{
  "scores": {"target_usage": 0-2, "grammar": 0-2, "semantics": 0-2, "register_collocation": 0-2},
  "total": 0-8,
  "passed": true/false,
  "errors": ["具体错误描述"],
  "suggestion": "改进建议",
  "model_sentence": "示范句（正确使用目标语言点）"
}
"""


def _build_l2_rubric_messages(points: list[dict], user_answer: str) -> list[dict]:
    target_desc = "; ".join(
        f"{p['text']}({p['type']}): {p.get('meaning_zh', '')}" for p in points
    )
    user = f"目标语言点：{target_desc}\n学习者造句：{user_answer}\n请评分。"
    return [
        {"role": "system", "content": L2_RUBRIC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def _call_l2_rubric_llm(points: list[dict], user_answer: str) -> dict:
    """调用 LLM 做 L2 rubric 4 维评测，返回解析后的 dict（失败抛 ExtensionError）。"""
    from services.providers.extension_llm import call_extension_llm, extract_json, ExtensionError

    messages = _build_l2_rubric_messages(points, user_answer)
    content = await call_extension_llm(messages)
    parsed = extract_json(content)
    if not parsed:
        raise ExtensionError("LLM_PARSE_ERROR", "parse", "rubric 输出解析失败")
    return parsed


# ===========================================================================
# E1 抽取流水线（后台执行器）
# ===========================================================================


async def get_cached_point(
    db, *, sentence_id: str, original: str
) -> dict | None:
    """查询缓存命中的抽取结果（E1 快路径）。

    返回 run_extract_pipeline 格式的 {points, meta, cached:true} 或 None。
    force_refresh 由调用方决定是否跳过缓存。
    """
    content_hash = compute_text_hash(original)

    # 2026-10-03（修订 2）：`source=manual` 全局人工校对层已下线，学习者判断改由
    # 读侧 merge_review_points 覆盖合并，AI 缓存只按 point_key 命中。
    point_key = extension_idempotency_key(
        sentence_id=sentence_id,
        content_hash=content_hash,
        prompt_version=config.EXTENSION_PROMPT_VERSION,
        model=config.EXTENSION_LLM_MODEL,
    )
    cached = await _find_cached_point(db, point_key, sentence_id)
    if not cached:
        return None
    return _point_to_result(cached, sentence_id, content_hash)


async def run_extract_pipeline(
    db,
    *,
    sentence_id: str,
    textbook_id: str | None = None,
    lesson_id: str | None = None,
    original: str,
    translation: str = "",
    scholar_id: str | None = None,
    force_refresh: bool = False,
    enable_fallback: bool = True,
) -> dict:
    """E1 后台执行（分层抽取）：

    生成候选 → LLM 对候选逐条判定（只回 candidate_id）→ 候选映射校验 →
    配额/重叠 → 敏感词审核 → 按价值排序 → 重算 id → L1 题面 → 持久化。

    失败链（R8）：LLM 异常 → 重试 1 次 → 规则兜底 → 仍失败则 failed+fallback_reason。
    越界/重复的 `candidate_id` **丢该条**（与 R7 一致），仅当有效点归零时判 `LLM_PARSE_ERROR`。
    """
    content_hash = compute_text_hash(original)
    point_key = extension_idempotency_key(
        sentence_id=sentence_id,
        content_hash=content_hash,
        prompt_version=config.EXTENSION_PROMPT_VERSION,
        model=config.EXTENSION_LLM_MODEL,
    )

    if not force_refresh:
        # 缓存查询（同 point_key，即同 prompt_version + model + 内容）
        cached = await _find_cached_point(db, point_key, sentence_id)
        if cached:
            return _point_to_result(cached, sentence_id, content_hash)

    # 句子过短 → 直接返回空
    if len(original.split()) < 4:
        return _empty_result(sentence_id, content_hash, source="rule", reason="SENTENCE_TOO_SHORT")

    # 候选召回（规则/词表层；LLM 只能从中挑，见 prompt v2）
    candidates = generate_candidates(original)

    # 连续抽取（LLM 对候选逐条判定；仅瞬时故障重试 1 次 + 规则兜底）
    points, attempts, fallback_reason, source = await _run_contiguous_extract(
        original, translation, candidates, enable_fallback=enable_fallback
    )

    if not points:
        # 最终仍无结果
        await _persist_point(
            db,
            point_key=point_key,
            sentence_id=sentence_id,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            content_hash=content_hash,
            status="failed",
            source=source,
            points=[],
            attempts=attempts,
            fallback_reason=fallback_reason or "NO_POINTS",
            scholar_id=scholar_id,
            force_refresh=force_refresh,
        )
        return _empty_result(
            sentence_id, content_hash, source=source, reason=fallback_reason or "NO_POINTS"
        )

    # 配额 + 重叠
    points = apply_quota_and_overlap(points)
    # 敏感词与质量审核（标 risk，不删除）
    points = audit_points(points)
    # 按学习价值排序（仅展示序）
    points = rank_points(points)
    # 重算 id（必须在 rank 之后：id 与展示顺序一致）
    for i, p in enumerate(points):
        p["id"] = f"{sentence_id}#{i}"
    # L1 题面（显式传 original 与候选池；不再依赖 p["_original"] 隐式契约）
    points = build_l1_items(points, original=original, candidates=candidates)

    count_by_type = _count_by_type(points)

    # 持久化
    await _persist_point(
        db,
        point_key=point_key,
        sentence_id=sentence_id,
        textbook_id=textbook_id,
        lesson_id=lesson_id,
        content_hash=content_hash,
        status="success",
        source=source,
        points=points,
        attempts=attempts,
        fallback_reason=fallback_reason,
        scholar_id=scholar_id,
        force_refresh=force_refresh,
    )

    return {
        "points": points,
        "meta": {
            "sentence_id": sentence_id,
            "source": source,
            "prompt_version": config.EXTENSION_PROMPT_VERSION,
            "model": config.EXTENSION_LLM_MODEL,
            "content_hash": content_hash,
            "count_by_type": count_by_type,
            "attempts": attempts,
            "fallback_reason": fallback_reason,
        },
        "cached": False,
    }


async def _run_contiguous_extract(
    original: str,
    translation: str,
    candidates: list[dict],
    enable_fallback: bool = True,
) -> tuple[list[dict], int, str, str]:
    """连续短语抽取（候选层 + LLM 筛选 + 规则兜底）。

    供 run_extract_pipeline 与 run_extract_pipeline_via_graph 复用。

    重试策略（2026-10-03 收敛，原「任何失败都重试 1 次」）：
    - LLM_TIMEOUT：**不重试**，直接进兜底（重试只是再等一个完整 timeout）；
    - LLM_PARSE_ERROR / 候选校验全废：**不重试**（模型输出问题，重试同 prompt 同结果，
      却要多付一次完整调用耗时——慢模型上直接翻倍）；
    - PROVIDER_UNAVAILABLE（凭据缺失 / 非 200 / 空返回）：瞬时故障，**重试 1 次**。

    Args:
        enable_fallback: 本次调用是否允许规则兜底（请求级开关，与全局
            `config.EXTENSION_RULE_FALLBACK_ENABLED` 为「与」关系）。

    Returns: (points, attempts, fallback_reason, source)
    """
    attempts = 0
    points: list[dict] = []
    fallback_reason = ""
    source = "llm"
    messages = build_extension_messages(original, translation, candidates)
    for attempt in range(MAX_EXTRACT_ATTEMPTS):
        attempts += 1
        has_next = attempt + 1 < MAX_EXTRACT_ATTEMPTS
        try:
            content = await call_extension_llm(messages)
            parsed = extract_json(content)
            raw_items = (parsed or {}).get("points", []) if parsed else []
            if not isinstance(raw_items, list):
                raise ExtensionError(ERR_LLM_PARSE_ERROR, STAGE_PARSE, "points 不是数组")
            valid, discarded = map_candidate_points(raw_items, candidates, original)
            if raw_items and not valid:
                raise ExtensionError(ERR_LLM_PARSE_ERROR, STAGE_PARSE, "全部语言点校验失败")
            points = valid
            break
        except ExtensionError as e:
            if e.error_code == ERR_LLM_TIMEOUT:
                fallback_reason = ERR_LLM_TIMEOUT
                break
            if e.error_code in RETRYABLE_ERROR_CODES and has_next:
                logger.warning(
                    f"[extension] LLM 抽取瞬时失败，重试 1 次：{e.error_code} - {e}"
                )
                continue
            fallback_reason = e.error_code
            break
        except Exception as e:  # noqa: BLE001
            if has_next:
                logger.warning(f"[extension] LLM 抽取异常，重试 1 次：{e}")
                continue
            fallback_reason = ERR_PROVIDER_UNAVAILABLE
            logger.error(f"[extension] LLM 抽取失败: {e}", exc_info=True)

    if not points and enable_fallback and config.EXTENSION_RULE_FALLBACK_ENABLED:
        points = rule_fallback_points(original)
        if points:
            source = "rule"
            if not fallback_reason:
                fallback_reason = "RULE_FALLBACK"
    return points, attempts, fallback_reason, source


async def run_extract_pipeline_via_graph(
    db,
    *,
    sentence_id: str,
    textbook_id: str | None = None,
    lesson_id: str | None = None,
    original: str,
    translation: str = "",
    scholar_id: str | None = None,
    force_refresh: bool = False,
    enable_fallback: bool = True,
) -> dict:
    """E1 后台执行（LangGraph 路径）：连续抽取 + 非连续短语图并行 → merge。

    与 run_extract_pipeline 的差异：
    - 额外并行执行 extension_graph 发现非连续短语（turn the light off → turn off）；
    - 两路结果 merge 后统一走配额/重叠/审核/排序/L1；
    - 图失败仅丢非连续结果，不影响连续结果（降级）。

    缓存/人工校对/过短/持久化逻辑与 run_extract_pipeline 一致。
    """
    content_hash = compute_text_hash(original)
    point_key = extension_idempotency_key(
        sentence_id=sentence_id,
        content_hash=content_hash,
        prompt_version=config.EXTENSION_PROMPT_VERSION,
        model=config.EXTENSION_LLM_MODEL,
    )

    if not force_refresh:
        cached = await _find_cached_point(db, point_key, sentence_id)
        if cached:
            return _point_to_result(cached, sentence_id, content_hash)

    if len(original.split()) < 4:
        return _empty_result(
            sentence_id, content_hash, source="rule", reason="SENTENCE_TOO_SHORT"
        )

    candidates = generate_candidates(original)

    # 连续抽取 与 非连续图 并行
    # 延迟导入：非连续短语图仅在 EXTENSION_USE_LANGGRAPH=1 时启用（默认关），
    # 关闭时主链路不加载 langgraph，该依赖故障不影响一期连续抽取。
    from services.english.extension_graph import run_extension_graph

    contiguous_task = _run_contiguous_extract(
        original, translation, candidates, enable_fallback=enable_fallback
    )
    graph_task = run_extension_graph(
        original=original, translation=translation, sentence_id=sentence_id
    )
    (contig_points, attempts, fallback_reason, source), graph_result = await asyncio.gather(
        contiguous_task, graph_task
    )

    # 非连续点走一遍 _validate_point 统一字段规范化（图产出已基本合规，但过一遍护栏）
    non_contig_points: list[dict] = []
    if graph_result.get("points"):
        for p in graph_result["points"]:
            norm = _validate_point(p, original, len(non_contig_points))
            if norm is not None:
                non_contig_points.append(norm)

    # merge：连续在前，非连续在后（重叠时连续优先级由 apply_quota_and_overlap 按 type 处理）
    points = contig_points + non_contig_points

    if not points:
        await _persist_point(
            db,
            point_key=point_key,
            sentence_id=sentence_id,
            textbook_id=textbook_id,
            lesson_id=lesson_id,
            content_hash=content_hash,
            status="failed",
            source=source,
            points=[],
            attempts=attempts,
            fallback_reason=fallback_reason or "NO_POINTS",
            scholar_id=scholar_id,
            force_refresh=force_refresh,
        )
        return _empty_result(
            sentence_id,
            content_hash,
            source=source,
            reason=fallback_reason or "NO_POINTS",
        )

    points = apply_quota_and_overlap(points)
    points = audit_points(points)
    points = rank_points(points)
    for i, p in enumerate(points):
        p["id"] = f"{sentence_id}#{i}"
    # L1 题面（显式传 original 与候选池；不再依赖 p["_original"] 隐式契约）
    points = build_l1_items(points, original=original, candidates=candidates)

    count_by_type = _count_by_type(points)

    await _persist_point(
        db,
        point_key=point_key,
        sentence_id=sentence_id,
        textbook_id=textbook_id,
        lesson_id=lesson_id,
        content_hash=content_hash,
        status="success",
        source=source,
        points=points,
        attempts=attempts,
        fallback_reason=fallback_reason,
        scholar_id=scholar_id,
        force_refresh=force_refresh,
    )

    return {
        "points": points,
        "meta": {
            "sentence_id": sentence_id,
            "source": source,
            "prompt_version": config.EXTENSION_PROMPT_VERSION,
            "model": config.EXTENSION_LLM_MODEL,
            "content_hash": content_hash,
            "count_by_type": count_by_type,
            "attempts": attempts,
            "fallback_reason": fallback_reason,
        },
        "cached": False,
    }


def _count_by_type(points: list[dict]) -> dict:
    counts = {"word": 0, "phrase": 0, "slang": 0, "idiom": 0}
    for p in points:
        if not p.get("excluding"):
            counts[p["type"]] = counts.get(p["type"], 0) + 1
    return counts


def _empty_result(sentence_id: str, content_hash: str, source: str, reason: str) -> dict:
    return {
        "points": [],
        "meta": {
            "sentence_id": sentence_id,
            "source": source,
            "prompt_version": config.EXTENSION_PROMPT_VERSION,
            "model": config.EXTENSION_LLM_MODEL,
            "content_hash": content_hash,
            "count_by_type": {"word": 0, "phrase": 0, "slang": 0, "idiom": 0},
            "attempts": 0,
            "fallback_reason": reason,
        },
        "cached": False,
    }


async def _find_cached_point(db, point_key: str, sentence_id: str) -> dict | None:
    res = await db.query(
        config.EXTENSION_POINT_COLLECTION,
        where={"point_key": point_key, "sentence_id": sentence_id},
        limit=1,
    )
    records = res.get("records", [])
    return records[0] if records else None


async def _query_points_by_sentence(db, sentence_id: str) -> list[dict]:
    """取该句全部抽取记录，按 updated_at 降序（最新在前）。

    注意：同一 `sentence_id` 下可能并存多条记录（不同 `point_key`：换 prompt_version 后
    重抽、force_refresh、人工校对各一条），故**不能**用 `limit=1` 随意取一条。
    """
    res = await db.query(
        config.EXTENSION_POINT_COLLECTION,
        where={"sentence_id": sentence_id},
        order=[{"field": "updated_at", "direction": "desc"}],
        limit=50,
    )
    return res.get("records", [])


async def resolve_effective_point(db, sentence_id: str) -> dict | None:
    """取该句「生效」的抽取记录：`updated_at` 最新的一条。

    供 E2（评测取题）使用，避免同句多记录下的取错行。

    2026-10-03（修订 2）：原「优先 `source=manual`」随全局人工校对层下线而移除，
    `find_manual_point` 一并删除 —— 本集合只存 AI 产出（`llm` / `rule`）。
    """
    records = await _query_points_by_sentence(db, sentence_id)
    if not records:
        return None
    return records[0]  # 已按 updated_at 降序


def _point_to_result(cached: dict, sentence_id: str, content_hash: str) -> dict:
    """把集合记录整形为 E1 / 任务的 `{points, meta, cached}` 出参。"""
    return {
        "points": cached.get("points", []),
        "meta": {
            "sentence_id": sentence_id,
            "source": cached.get("source", "llm"),
            "prompt_version": cached.get("prompt_version"),
            "model": cached.get("model"),
            "content_hash": content_hash,
            "count_by_type": cached.get("count_by_type", {}),
            "attempts": cached.get("attempts", 1),
            "fallback_reason": cached.get("fallback_reason", ""),
        },
        "cached": True,
    }


async def _persist_point(
    db,
    *,
    point_key: str,
    sentence_id: str,
    textbook_id: str | None,
    lesson_id: str | None,
    content_hash: str,
    status: str,
    source: str,
    points: list[dict],
    attempts: int,
    fallback_reason: str,
    scholar_id: str | None,
    force_refresh: bool = False,
) -> None:
    """写入 english_extension_point（**只存 AI 产出**）。

    2026-10-03（修订 2）：原 R6（同句存在 `source=manual` 时不覆盖）随全局人工校对层
    下线而删除 —— 学习者判断独立落 `extension_review`，AI 层任何重抽都碰不到它。
    同 `point_key` 命中则原地更新（避免同句多行并存）。
    """
    existing = await _find_cached_point(db, point_key, sentence_id)
    now = int(time.time() * 1000)
    doc = {
        "point_key": point_key,
        "sentence_id": sentence_id,
        "textbook_id": textbook_id,
        "lesson_id": lesson_id,
        "content_hash": content_hash,
        "status": status,
        "source": source,
        "points": points,
        "count_by_type": _count_by_type(points),
        "model": config.EXTENSION_LLM_MODEL,
        "prompt_version": config.EXTENSION_PROMPT_VERSION,
        "attempts": attempts,
        "fallback_reason": fallback_reason,
        "scholar_id": scholar_id,
        "updated_at": now,
    }
    # 同 point_key 命中则原地更新（避免同句多行并存）
    target = existing
    if target is not None:
        doc["created_at"] = target.get("created_at", now)
        await db.update(
            config.EXTENSION_POINT_COLLECTION,
            where={"_id": target["_id"]},
            data={"$set": doc},
            multi=False,
        )
    else:
        doc["created_at"] = now
        await db.insert(config.EXTENSION_POINT_COLLECTION, doc)


# ===========================================================================
# E2 评测流水线（后台执行器）
# ===========================================================================


async def run_evaluate_pipeline(
    db,
    *,
    task_type: str,
    selected_ids: list[str],
    points_snapshot: list[dict],
    input_mode: str,
    user_input: str,
    audio_base64: str | None = None,
    voice_format: str = "mp3",
    scholar_id: str | None = None,
) -> dict:
    """E2 后台执行：L1 确定性判分 / L2 rubric 异步。

    - l1_fill：后端确定性比对，不调 LLM，同请求即 success；
    - l2_sentence：LLM rubric（B13 实现），异步 pending → success。
    """
    selected = [p for p in points_snapshot if p.get("id") in selected_ids]

    if task_type == "l1_fill":
        # 确定性判分
        results = []
        all_passed = True
        for p in selected:
            g = grade_l1_fill(p, user_input)
            results.append(g)
            if not g["passed"]:
                all_passed = False
        return {
            "task_type": "l1_fill",
            "score": sum(r["score"] for r in results),
            "passed": all_passed,
            "correct_ids": [cid for r in results for cid in r["correct_ids"]],
            "wrong_ids": [wid for r in results for wid in r["wrong_ids"]],
            "answer_key": selected[0]["text"] if selected else "",
        }

    elif task_type == "l2_sentence":
        # L2 rubric：LLM 4 维评测 + 后端独立 must_use_hit 判定
        must_info = grade_l2_rubric(selected, user_input)
        try:
            rubric = await _call_l2_rubric_llm(selected, user_input)
        except ExtensionError as e:
            return {
                "task_type": "l2_sentence",
                "must_use_hit": must_info["must_use_hit"],
                "used_points": must_info["used_points"],
                "score": 0,
                "passed": False,
                "errors": [f"评测失败：{e.error_code}"],
                "suggestion": "",
                "model_sentence": "",
            }
        scores = rubric.get("scores", {})
        total = rubric.get("total", sum(scores.values()))
        return {
            "task_type": "l2_sentence",
            "must_use_hit": must_info["must_use_hit"],
            "used_points": must_info["used_points"],
            "score": total,
            "passed": rubric.get("passed", total >= 6),
            "errors": rubric.get("errors", []),
            "suggestion": rubric.get("suggestion", ""),
            "model_sentence": rubric.get("model_sentence", ""),
            "rubric_scores": scores,
        }

    return {"score": 0, "passed": False, "errors": ["unknown task_type"]}
