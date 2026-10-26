"""ASR 转写后处理（`smart_punctuate`）单测。

锁住四条语义：
 1. 空白归一 + 首字母大写 + 句末补句号；
 2. 已有句末标点（. ! ?）**不重复追加**；
 3. 空串 / 纯空白 / None **原样返回**（不无中生有，避免下游把空白误判成有内容）；
 4. 刻意**不**做 ITN 与句内逗号 —— 只保留 ASR 裸文本里本来就有的词。
"""
import pytest

from services.infra.text_normalize import smart_punctuate


def test_collapses_whitespace_and_capitalizes_first_letter():
    assert smart_punctuate("  hello   world ") == "Hello world."


def test_appends_missing_terminal_punctuation():
    assert smart_punctuate("i go to school") == "I go to school."


@pytest.mark.parametrize("raw", ["I like it.", "I like it!", "I like it?"])
def test_keeps_existing_terminal_punctuation(raw):
    assert smart_punctuate(raw) == raw


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_returns_falsy_or_blank_input_unchanged(raw):
    assert smart_punctuate(raw) == raw


def test_single_word_gets_capitalized_and_punctuated():
    assert smart_punctuate("yes") == "Yes."


def test_does_not_touch_inner_words_or_digits():
    # 刻意不做 ITN：数字保持 ASR 原样，不在 greet("two") 上做文章
    assert smart_punctuate("i have 2 books") == "I have 2 books."
