"""英文语句扩展 —— 候选召回层（分层抽取第一段：规则/词典生成候选）

策略：`docs_v1/扩展/重点词汇短语抽取策略.md` §1 / §2 / §6
设计：第一期设计 §3.4；契约：api-contract.md §3.18（抽取语义）

流水线定位：
    文本 → **本模块（规则/词典生成候选）** → LLM 语境筛选与释义 → 质量审核 → 去重·排序·输出

设计要点：
- **无第三方 NLP 依赖**：正则切词 + 内置小型人工词表；召回主力是广覆盖的 2–5 gram，
  词表只做高精度加成。故「LLM 只能从候选中挑」不会把召回卡死。
- **`text` 一律取 `original[start:end]` 切片，绝不 re-join token**：否则 `well-known`、
  `Hello, world` 这类与原文不符，会触发下游 R7（span 切出字符必须 == text）而丢点。
- **缩写展开仅作 prompt 上下文提示**（`gonna → going to`），不参与返回、不参与去重键——
  避免 `gonna` 与真实 `going to` 在跨句去重里串味。
- **已知召回损失（明确登记）**：非连续动词短语（`turn the light off` / `pick it up`）无法
  构成单一 span，本层抓不到；本期接受，靠实验页 E4 人工校对 diff 观察缺口。
- 非 ASCII 字母（带重音等）不覆盖：一期教材为 ASCII，登记为已知边界。
"""
from __future__ import annotations

import re

import config

# ===========================================================================
# 词表（人工小型种子表；可扩充，扩充后无需改动调用方）
# ===========================================================================

# 口语缩写 → 书面展开（仅 prompt 提示用）
ABBREVIATIONS = {
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "kinda": "kind of",
    "sorta": "sort of",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "ain't": "am not / is not / are not",
    "cuz": "because",
    "'cause": "because",
    "'em": "them",
    "outta": "out of",
    "hafta": "have to",
    "oughta": "ought to",
    "coulda": "could have",
    "shoulda": "should have",
    "woulda": "would have",
    "musta": "must have",
}

# 固定短语 / 动词短语 / 常见搭配（hint_type=phrase）
PHRASE_SEED = {
    "figure out", "give up", "show up", "put off", "take off", "turn on", "turn off",
    "look after", "look for", "look forward to", "take part in", "take care of",
    "come up with", "run out of", "get along with", "get rid of", "make up for",
    "carry on", "set up", "break down", "bring up", "call off", "check out",
    "on the same page", "in the end", "at least", "in fact", "for instance",
    "as well as", "instead of", "according to", "in charge of", "in front of",
}

# 惯用套语 / 固定表达（hint_type=idiom）
IDIOM_SEED = {
    "as a matter of fact", "by the way", "on the other hand", "in other words",
    "for the time being", "at the end of the day", "up to you", "it's up to you",
    "no wonder", "take it easy", "piece of cake", "once in a while",
    "make sense", "make up one's mind", "as soon as possible", "in advance",
    "hit the books", "spill the beans", "under the weather", "break the ice",
    "call it a day", "get the hang of",
}

# 俚语 / 口语（hint_type=slang）
SLANG_SEED = {
    "hit me up", "no biggie", "low-key", "lowkey", "hang out", "chill out",
    "what's up", "catch up", "kick back", "screw up", "freak out", "zone out",
    "grab a bite", "crash at", "dude", "bail",
}

# 短语虚词（n-gram 尾词命中 → 动词短语加分，裁剪时优先保留）
PHRASAL_PARTICLES = {
    "up", "down", "out", "off", "on", "in", "over", "away", "back",
    "along", "around", "through", "into", "with", "for", "at", "to",
}

# 慎用表达 → 风险等级（audit_points 用；枚举 ""｜offensive｜vulgar｜regional｜dated）
RISKY_TERMS = {
    "damn": "offensive",
    "hell": "offensive",
    "crap": "vulgar",
    "bloody": "regional",
}

# 停用词表（超高频功能词，不产 word 候选）。
# 末尾 s/t/d/ll/re/ve/m 为历史遗留的缩写碎片（切词不拆缩写后基本失效），保留以零 diff。
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "have", "has", "had", "will", "would",
    "shall", "should", "can", "could", "may", "might", "must", "to", "of",
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "not", "so",
    "yet", "both", "either", "neither", "each", "every", "all", "any",
    "few", "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "because", "as", "until",
    "while", "if", "when", "where", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "i", "me", "my", "we", "us", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "there", "here", "s", "t", "d", "ll", "re", "ve",
    "m", "o", "y",
}

# 不规则缩写（`n't` 规则取不到词头的例外）
_IRREGULAR_CONTRACTIONS = {"can't": "can", "won't": "will", "shan't": "shall", "ain't": "be"}

# 同区间冲突时的类型优先级（lexicon > n-gram > word）
_HINT_PRIORITY = {"idiom": 3, "slang": 3, "phrase": 2, "word": 1}

# 切词：字母串，可含撇号/弯撇号/连字符（不拆缩写，故 don't / it's / well-known 各为一个 token）
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")

# 桶配额（保证 word 候选不被 n-gram 挤光）
_LEXICON_BUCKET = 15
_WORD_BUCKET = 10
_NGRAM_MIN, _NGRAM_MAX = 2, 5
_MIN_WORD_LEN = 3


# ===========================================================================
# 预处理
# ===========================================================================


def tokenize_with_spans(text: str) -> list[tuple[str, int, int]]:
    """切词并保留字符偏移，返回 [(token, start, end)]。

    不拆缩写（`don't` / `it's`）、不拆连字符词（`well-known`）；兼容弯撇号 `’`。
    """
    if not text:
        return []
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def expand_abbreviations(text: str) -> list[tuple[str, str]]:
    """找出句中出现的口语缩写，返回 [(原形, 书面展开)]。仅作 prompt 提示。"""
    if not text:
        return []
    return [
        (term, expansion)
        for term, expansion in ABBREVIATIONS.items()
        if next(_find_term_spans(text, term), None) is not None
    ]


def _term_pattern(term: str) -> str:
    """把词表条目编译为带词边界、容忍空白/撇号变体的正则。"""
    body = re.escape(term).replace(r"\ ", r"\s+").replace("'", "['’]")
    return r"(?<![A-Za-z])" + body + r"(?![A-Za-z])"


def _find_term_spans(text: str, term: str):
    """在 text 中定位词表条目（大小写不敏感），逐个产出 (start, end)。"""
    for m in re.finditer(_term_pattern(term), text, re.IGNORECASE):
        yield m.start(), m.end()


def _is_stopword(token: str) -> bool:
    """停用词判定：小写 + 弯撇号归一；缩写取其词头（`don't` → `do`、`it's` → `it`）。"""
    t = token.lower().replace("’", "'")
    if t in STOPWORDS:
        return True
    if t in _IRREGULAR_CONTRACTIONS:
        t = _IRREGULAR_CONTRACTIONS[t]
    elif t.endswith("n't"):
        t = t[:-3]
    elif "'" in t:
        t = t.split("'", 1)[0]
    return t in STOPWORDS


# ===========================================================================
# 候选生成
# ===========================================================================


def generate_candidates(original: str) -> list[dict]:
    """规则/词典召回候选，返回
    `[{candidate_id, text, span, hint_type, source}]`（按原句位置排序，id 从 c1 递增）。

    - `word`：实词（非停用词、长度 ≥3）
    - `phrase`：2–5 gram（跨实词窗口，丢弃首尾皆停用词者）
    - `slang` / `idiom` / `phrase`：词表命中（高精度加成）
    """
    if not original or not original.strip():
        return []
    tokens = tokenize_with_spans(original)
    if not tokens:
        return []

    picked: dict[tuple[int, int], dict] = {}

    def _put(start: int, end: int, hint: str, source: str, size: int = 1, particle: bool = False):
        key = (start, end)
        cur = picked.get(key)
        if cur is not None and _HINT_PRIORITY[hint] <= _HINT_PRIORITY[cur["hint_type"]]:
            return
        picked[key] = {
            "text": original[start:end],  # 切原文，绝不 re-join token
            "span": [start, end],
            "hint_type": hint,
            "source": source,
            "_size": size,
            "_particle": particle,
        }

    # 1) 词表命中（高精度）
    lexicon_keys: list[tuple[int, int]] = []
    for terms, hint in ((SLANG_SEED, "slang"), (IDIOM_SEED, "idiom"), (PHRASE_SEED, "phrase")):
        for term in terms:
            for start, end in _find_term_spans(original, term):
                _put(start, end, hint, "lexicon", size=term.count(" ") + 1)
                lexicon_keys.append((start, end))

    # 2) 实词 → word
    word_keys: list[tuple[int, int]] = []
    for token, start, end in tokens:
        if len(token) < _MIN_WORD_LEN or _is_stopword(token):
            continue
        _put(start, end, "word", "token")
        word_keys.append((start, end))

    # 3) 2–5 gram → phrase
    ngram_keys: list[tuple[int, int]] = []
    n = len(tokens)
    for size in range(_NGRAM_MIN, _NGRAM_MAX + 1):
        for i in range(0, n - size + 1):
            window = tokens[i : i + size]
            if _is_stopword(window[0][0]) and _is_stopword(window[-1][0]):
                continue
            if all(_is_stopword(tok) for tok, _, _ in window):
                continue
            start, end = window[0][1], window[-1][2]
            particle = window[-1][0].lower() in PHRASAL_PARTICLES
            _put(start, end, "phrase", "ngram", size=size, particle=particle)
            ngram_keys.append((start, end))

    # 裁剪：词表桶 + 实词桶 + n-gram 桶（保证 word 不被挤光），再整体截到上限
    ngram_keys = sorted(
        set(ngram_keys),
        key=lambda k: (picked[k]["_particle"], picked[k]["_size"], -k[0]),
        reverse=True,
    )
    ordered = _dedup_keys(
        _dedup_keys(lexicon_keys)[:_LEXICON_BUCKET]
        + _dedup_keys(word_keys)[:_WORD_BUCKET]
        + ngram_keys
    )[: config.EXTENSION_CANDIDATE_MAX]

    # 展示序：按原句位置，同起点先长后短（短语在构成它的词之前）
    ordered.sort(key=lambda k: (k[0], -k[1]))

    return [
        {
            "candidate_id": f"c{idx}",
            "text": picked[key]["text"],
            "span": picked[key]["span"],
            "spans": [list(picked[key]["span"])],  # 连续候选统一包为单区间 spans，与非连续点同构
            "hint_type": picked[key]["hint_type"],
            "source": picked[key]["source"],
        }
        for idx, key in enumerate(ordered, start=1)
    ]


def _dedup_keys(keys: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """保序去重。"""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out
