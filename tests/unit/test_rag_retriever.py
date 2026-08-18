"""单元测试: P2-2 RAG Retriever — 跨课课程知识召回（向量版）

覆盖（对应 会话训练评估重构执行计划.md §3 S4.3 后续方向① P2-2 / ADR-0017）：
- cosine_similarity：相同 / 正交 / 空向量 / 维度不一致
- NoopCurriculumRetriever：降级恒返回 []
- FakeCurriculumRetriever：排除课 / top_k / 确定性分数
- VolcanoEmbeddingRetriever：
  - 候选集 = skill_state 已学句（排除当前课）→ sentence_v2 → 向量 → 余弦 top-K
  - 惰性缓存：命中 text_hash+model 复用不重复调 embedding；缺失增量生成并 upsert
  - 降级：embedding API 失败 / 无候选 / query 为空 → []
- get_curriculum_retriever 工厂：未配置模型 → Noop；配置 → Volcano
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from services import rag_retriever
from services.rag_retriever import (
    FakeCurriculumRetriever,
    NoopCurriculumRetriever,
    VolcanoEmbeddingRetriever,
    cosine_similarity,
    get_curriculum_retriever,
)
from tests.fakes.fake_db import FakeDB
from tests.fakes.seed_factory import seed_content


class FakeEmbeddingClient:
    """OpenAI 兼容 embedding 客户端（fake）：确定性向量 + 调用统计。

    OpenAI 客户端用法为 client.embeddings.create(model=..., input=...)，
    故暴露 embeddings 属性；create 按文本字符和生成稳定伪向量。
    """

    def __init__(self, *, dim: int = 3, fail: bool = False):
        self.calls: list[list[str]] = []
        self.fail = fail
        self.dim = dim

    @property
    def embeddings(self):
        return self

    def create(self, *, model: str, input):
        if self.fail:
            raise RuntimeError("embedding api down")
        texts = [input] if isinstance(input, str) else list(input)
        self.calls.append(texts)
        data = []
        for text in texts:
            seed = sum(ord(c) for c in text)
            vec = [
                float(((seed >> shift) & 0x3F) / 64.0)
                for shift in range(0, self.dim * 6, 6)
            ]
            data.append(SimpleNamespace(embedding=vec))
        return SimpleNamespace(data=data)


def _seed_scholar(db) -> None:
    """内容层级 3 课 6 句 + scholar_1 已学句（s1..s5，最近在 l3）。"""
    seed_content(
        db,
        textbook_id="tb_1",
        chapter_id="c1",
        lesson_ids=("l1", "l2", "l3"),
        sentence_ids=("s1", "s2", "s3", "s4", "s5", "s6"),
    )
    for sid, lid in [("s1", "l1"), ("s2", "l1"), ("s3", "l2"), ("s4", "l2"), ("s5", "l3")]:
        db.add("skill_state", {
            "scholar_id": "scholar_1", "sentence_id": sid, "lesson_id": lid,
            "skill_code": "translation", "attempt_count": 1, "mastery_score": 50,
            "status": "learning", "updated_at": 1000 + int(sid[1:]),
        })


def _retriever(**overrides) -> VolcanoEmbeddingRetriever:
    defaults = {"model": "ep-m", "client": FakeEmbeddingClient(), "top_k": 5}
    defaults.update(overrides)
    return VolcanoEmbeddingRetriever(**defaults)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_zero_norm(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_empty_or_dim_mismatch(self):
        assert cosine_similarity([], [1.0, 0.0]) == 0.0
        assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_similar_non_identical(self):
        a = [1.0, 1.0, 0.0]
        b = [1.0, 1.0, 1.0]
        sim = cosine_similarity(a, b)
        assert 0.0 < sim < 1.0


# ---------------------------------------------------------------------------
# 降级实现
# ---------------------------------------------------------------------------


class TestNoopCurriculumRetriever:
    def test_always_empty(self):
        retriever = NoopCurriculumRetriever()
        hits = asyncio.run(retriever.retrieve(
            None, scholar_id="s", query="q", exclude_lesson_ids=["l1"],
        ))
        assert hits == []


class TestFakeCurriculumRetriever:
    def test_exclude_lesson_and_top_k(self):
        retriever = FakeCurriculumRetriever(sentences=[
            {"sentence_id": "sX", "text": "A", "lesson_id": "l2"},
            {"sentence_id": "sY", "text": "B", "lesson_id": "l1"},
            {"sentence_id": "sZ", "text": "C", "lesson_id": "l2"},
        ])
        hits = asyncio.run(retriever.retrieve(
            None, scholar_id="s", query="q", top_k=1, exclude_lesson_ids=["l1"],
        ))
        assert [h["sentence_id"] for h in hits] == ["sX"]
        assert hits[0]["score"] == 1.0

    def test_deterministic_scores(self):
        retriever = FakeCurriculumRetriever(sentences=[
            {"sentence_id": "s1", "text": "A", "lesson_id": "l1"},
            {"sentence_id": "s2", "text": "B", "lesson_id": "l2"},
        ])
        hits = asyncio.run(retriever.retrieve(
            None, scholar_id="s", query="q", exclude_lesson_ids=[],
        ))
        assert [h["score"] for h in hits] == [1.0, 0.9]


# ---------------------------------------------------------------------------
# VolcanoEmbeddingRetriever
# ---------------------------------------------------------------------------


class TestVolcanoEmbeddingRetriever:
    def test_retrieve_cross_lesson(self):
        """候选 = 学者已学句（排除当前课）→ 余弦 top-K，score 降序。"""
        db = FakeDB()
        _seed_scholar(db)
        retriever = _retriever(top_k=3)
        hits = asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="Text s5", exclude_lesson_ids=["l3"],
        ))
        # 候选：l1/l2 的 s1/s2/s3/s4；排除 l3 的 s5
        assert hits
        assert all(h["lesson_id"] != "l3" for h in hits)
        assert len(hits) <= 3
        scores = [h["score"] for h in hits]
        assert scores == sorted(scores, reverse=True)
        # 快照含文本（供 LearningContext sentences 使用）
        assert all(h.get("text") for h in hits)
        assert all(h.get("sentence_id") for h in hits)

    def test_cache_reuse_avoids_re_embedding(self):
        """惰性缓存：首轮候选句+query 各生成 1 次；次轮仅 query 生成。"""
        db = FakeDB()
        _seed_scholar(db)
        client = FakeEmbeddingClient()
        retriever = VolcanoEmbeddingRetriever(model="ep-m", client=client, top_k=5)
        asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="q", exclude_lesson_ids=["l1"],
        ))
        first = sum(len(b) for b in client.calls)
        # 候选 = l2/l3 的 s3/s4/s5（3 句）+ query（1 次）
        assert first == 4
        asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="q", exclude_lesson_ids=["l1"],
        ))
        second = sum(len(b) for b in client.calls) - first
        assert second == 1  # 仅 query，候选句命中缓存

    def test_cache_invalidated_on_model_change(self):
        """model 变更 → 旧缓存失效，重新生成。"""
        db = FakeDB()
        _seed_scholar(db)
        client = FakeEmbeddingClient()
        r1 = VolcanoEmbeddingRetriever(model="ep-m1", client=client, top_k=5)
        asyncio.run(r1.retrieve(db, scholar_id="scholar_1", query="q", exclude_lesson_ids=["l1"]))
        calls_before = sum(len(b) for b in client.calls)
        r2 = VolcanoEmbeddingRetriever(model="ep-m2", client=client, top_k=5)
        asyncio.run(r2.retrieve(db, scholar_id="scholar_1", query="q", exclude_lesson_ids=["l1"]))
        calls_after = sum(len(b) for b in client.calls)
        assert calls_after > calls_before  # 候选句重新 embedding

    def test_embedding_failure_degrades_to_empty(self):
        db = FakeDB()
        _seed_scholar(db)
        retriever = VolcanoEmbeddingRetriever(
            model="ep-m", client=FakeEmbeddingClient(fail=True), top_k=3,
        )
        hits = asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="q", exclude_lesson_ids=["l1"],
        ))
        assert hits == []

    def test_no_history_empty(self):
        """无 skill_state → 无候选 → []。"""
        db = FakeDB()
        retriever = _retriever()
        hits = asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="q", exclude_lesson_ids=[],
        ))
        assert hits == []

    def test_empty_query_empty(self):
        db = FakeDB()
        _seed_scholar(db)
        retriever = _retriever()
        hits = asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="  ", exclude_lesson_ids=[],
        ))
        assert hits == []

    def test_all_excluded_empty(self):
        db = FakeDB()
        _seed_scholar(db)
        retriever = _retriever()
        hits = asyncio.run(retriever.retrieve(
            db, scholar_id="scholar_1", query="q",
            exclude_lesson_ids=["l1", "l2", "l3"],
        ))
        assert hits == []


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


class TestGetCurriculumRetriever:
    def test_disabled_noop(self, monkeypatch):
        monkeypatch.setattr(rag_retriever, "RAG_EMBEDDING_MODEL", "")
        monkeypatch.setattr(rag_retriever, "RAG_RETRIEVER_ENABLED", True)
        assert isinstance(get_curriculum_retriever(), NoopCurriculumRetriever)

    def test_switch_off_noop(self, monkeypatch):
        monkeypatch.setattr(rag_retriever, "RAG_EMBEDDING_MODEL", "ep-m")
        monkeypatch.setattr(rag_retriever, "RAG_RETRIEVER_ENABLED", False)
        assert isinstance(get_curriculum_retriever(), NoopCurriculumRetriever)

    def test_configured_volcano(self, monkeypatch):
        monkeypatch.setattr(rag_retriever, "RAG_EMBEDDING_MODEL", "ep-m")
        monkeypatch.setattr(rag_retriever, "RAG_RETRIEVER_ENABLED", True)
        assert isinstance(get_curriculum_retriever(), VolcanoEmbeddingRetriever)
