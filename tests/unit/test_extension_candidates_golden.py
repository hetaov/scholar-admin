"""召回金标测试：候选集**必含**指定语言点

分层抽取把「选什么」交给 LLM，但「能不能选到」取决于本层召回。
若召回漏掉了这些教科书高频点，LLM 再准也无从选择 —— 故用金标锁定召回下限。
（D3 采用「LLM 只能从候选中挑」，本测试即该决策的安全网。）
"""
from __future__ import annotations

import pytest

from services.english.extension_candidates import generate_candidates

# (句子, 期望候选**必含**的文本（小写比对）)
GOLDEN: list[tuple[str, set[str]]] = [
    ("I will take part in the discussion.", {"take part in"}),
    ("I'm gonna figure it out later.", {"gonna", "figure it out"}),
    ("Let's hit the books tonight.", {"hit the books"}),
    ("No biggie, we can hang out later.", {"no biggie", "hang out"}),
    ("As a matter of fact, I disagree.", {"as a matter of fact"}),
    ("She is under the weather today.", {"under the weather"}),
    ("Please look forward to the trip.", {"look forward to"}),
    ("It is a well-known fact.", {"well-known", "fact"}),
    ("We should come up with a better plan.", {"come up with"}),
    ("Take care of yourself, buddy.", {"take care of"}),
]


@pytest.mark.parametrize("sentence,expected", GOLDEN, ids=[g[0][:32] for g in GOLDEN])
def test_golden_candidates_are_recalled(sentence, expected):
    got = {c["text"].lower() for c in generate_candidates(sentence)}
    missing = expected - got
    assert not missing, f"{sentence!r} 召回缺少 {missing}；实际 {sorted(got)}"
