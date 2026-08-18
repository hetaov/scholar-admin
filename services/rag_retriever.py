"""P2-2 RAG Retriever — 跨课课程知识召回（LearningContextBuilder 向量版）

设计文档 §2.1 依赖裁剪 P2-2 / §4.1-§4.2（Required + Optional knowledge）/
草稿 §十八（RAG = Curriculum Context）：
AI Training Context → Retriever → Relevant Curriculum Knowledge → LLM

职责：
- 输入：学者 ID + 检索 query（weakSkills + 当前课场景）+ 排除课（当前课）
- 输出：跨课相关句子快照（sentence_v2 文档 + score），按相似度降序
- 数据源：skill_state（学者已学句）→ sentence_v2.text → embedding（惰性缓存 sentence_embedding）
- 可替换基础设施（设计文档 §2.2 Provider 化）：CurriculumRetriever（ABC），
  默认 VolcanoEmbeddingRetriever（方舟 embeddings + NoSQL 缓存 + 内存余弦），
  测试注入 FakeCurriculumRetriever；后续可切 LlamaIndex / VectorDB / CloudBase 知识库。

降级策略（不阻断主链路）：
- RAG_RETRIEVER_ENABLED=0 或 RAG_EMBEDDING_MODEL 未配置 → get_curriculum_retriever()
  返回 NoopCurriculumRetriever（retrieve 返回 []，等同一期非向量版）
- embedding API 调用失败 → 捕获记日志返回 []，不抛错
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod

from config import (
    RAG_EMBED_BATCH_SIZE,
    RAG_EMBEDDING_COLLECTION,
    RAG_EMBEDDING_MODEL,
    RAG_RETRIEVER_ENABLED,
    RAG_TOP_K,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
from services.models_content import get_sentences_by_ids
from services.models_learning import get_skill_states

logger = logging.getLogger("scholar-admin.rag")

_SENTENCE_ID_FIELD = "sentence_id"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度（零依赖）。空向量或维度不一致返回 0.0。"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _text_hash(text: str) -> str:
    """文本 SHA-256 摘要（缓存失效判定）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


class CurriculumRetriever(ABC):
    """跨课课程知识检索器（可替换基础设施，设计文档 §2.2）。"""

    @abstractmethod
    async def retrieve(
        self,
        db,
        *,
        scholar_id: str,
        query: str,
        top_k: int | None = None,
        exclude_lesson_ids: list[str] | None = None,
    ) -> list[dict]:
        """按语义相关度召回跨课句子快照。

        Args:
            db: 数据库客户端（CloudBaseNoSQLClient / FakeDB）。
            scholar_id: 学者 ID（候选集 = 该学者已学句）。
            query: 检索 query（由 weakSkills + 当前课场景拼接）。
            top_k: 召回条数（默认 RAG_TOP_K）。
            exclude_lesson_ids: 排除的课（当前课），实现"跨课"。

        Returns:
            句子快照列表（sentence_v2 文档字段 + score），按相似度降序。
            任何失败 / 降级返回 []（不阻断主链路）。
        """
        raise NotImplementedError


class NoopCurriculumRetriever(CurriculumRetriever):
    """降级实现：不检索，恒返回 []（等同一期非向量版）。"""

    async def retrieve(
        self, db, *, scholar_id, query, top_k=None, exclude_lesson_ids=None
    ):
        return []


class FakeCurriculumRetriever(CurriculumRetriever):
    """确定性实现（测试用）：按候选句顺序返回 top_k 条，附带递减分数。

    用于离线测试 LearningContextBuilder 向量版接线，不触网、不依赖配置。
    """

    def __init__(self, sentences: list[dict] | None = None, top_k: int = 5):
        self.sentences = sentences or []
        self.top_k = top_k

    async def retrieve(
        self, db, *, scholar_id, query, top_k=None, exclude_lesson_ids=None
    ):
        k = top_k or self.top_k
        excluded = set(exclude_lesson_ids or [])
        hits = [s for s in self.sentences if s.get("lesson_id") not in excluded]
        return [
            dict(s, score=round(1.0 - i * 0.1, 4))
            for i, s in enumerate(hits[:k])
        ]


class VolcanoEmbeddingRetriever(CurriculumRetriever):
    """默认实现：方舟 embeddings + sentence_embedding 缓存 + 内存余弦 top-K。

    - 候选集 = 该学者 skill_state 覆盖的句子（排除当前课），天然"跨课"；
    - embedding 惰性索引：命中缓存（text_hash + model 一致）复用，缺失批量生成并 upsert；
    - 失败降级：embedding API 异常 → 返回 []（不抛错，不阻断 planner）。
    """

    def __init__(
        self,
        *,
        model: str = RAG_EMBEDDING_MODEL,
        collection: str = RAG_EMBEDDING_COLLECTION,
        top_k: int = RAG_TOP_K,
        embed_batch_size: int = RAG_EMBED_BATCH_SIZE,
        client=None,  # 可注入（测试）；None → 惰性创建方舟客户端
    ):
        self.model = model
        self.collection = collection
        self.top_k = top_k
        self.embed_batch_size = embed_batch_size
        self._client = client
        self._logger = logger

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=VOLCANO_API_KEY, base_url=VOLCANO_BASE_URL, timeout=60.0
            )
        return self._client

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        """批量 embedding（分批）；失败返回 None（降级空召回）。"""
        if not texts:
            return []
        try:
            client = self._get_client()
            vectors: list[list[float]] = []
            for i in range(0, len(texts), self.embed_batch_size):
                batch = texts[i : i + self.embed_batch_size]
                resp = await asyncio.to_thread(
                    client.embeddings.create, model=self.model, input=batch
                )
                vectors.extend([item.embedding for item in resp.data])
            return vectors
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("RAG embedding 失败（降级空召回）: %s", exc)
            return None

    # ---------------- 缓存 ----------------

    async def _load_cache(self, db, sentence_ids: list[str]) -> dict[str, dict]:
        """按 sentence_id 批量读缓存，返回 {sentence_id: 缓存文档}。"""
        hits: dict[str, dict] = {}
        for i in range(0, len(sentence_ids), 200):
            result = await db.query(
                collection=self.collection,
                where={_SENTENCE_ID_FIELD: {"$in": sentence_ids[i : i + 200]}},
                limit=200,
            )
            for doc in result.get("records") or []:
                sid = doc.get(_SENTENCE_ID_FIELD)
                if sid:
                    hits[sid] = doc
        return hits

    async def _upsert_cache(self, db, docs: list[dict]) -> None:
        """逐条 upsert（按 sentence_id 幂等）。"""
        for doc in docs:
            await db.update(
                collection=self.collection,
                where={_SENTENCE_ID_FIELD: doc[_SENTENCE_ID_FIELD]},
                data={"$set": doc},
                upsert=True,
                multi=False,
            )

    async def _build_vectors(self, db, sentences: list[dict]) -> list[list[float]] | None:
        """为句子构建向量：缓存命中复用，缺失批量生成并 upsert。失败返回 None。

        返回顺序与 sentences 一致（未命中条目被替换为真实向量）。
        """
        ids = [s.get(_SENTENCE_ID_FIELD) for s in sentences]
        cache = await self._load_cache(db, ids)

        vectors: list[list[float] | None] = []
        missing: list[tuple[int, str]] = []  # (sent_index, text)
        for idx, s in enumerate(sentences):
            sid = s.get(_SENTENCE_ID_FIELD)
            text = s.get("text", "")
            cached = cache.get(sid)
            if (
                cached
                and cached.get("text_hash") == _text_hash(text)
                and cached.get("model") == self.model
                and cached.get("embedding")
            ):
                vectors.append(cached["embedding"])
            else:
                vectors.append(None)
                missing.append((idx, text))

        if missing:
            texts = [item[1] for item in missing]
            new_vectors = await self._embed(texts)
            if new_vectors is None:
                return None  # 整体降级空召回
            now = _now_ms()
            upsert_docs: list[dict] = []
            for (idx, text), vec in zip(missing, new_vectors):
                vectors[idx] = vec
                upsert_docs.append(
                    {
                        _SENTENCE_ID_FIELD: sentences[idx].get(_SENTENCE_ID_FIELD),
                        "text_hash": _text_hash(text),
                        "embedding": vec,
                        "model": self.model,
                        "dim": len(vec),
                        "updated_at": now,
                    }
                )
            try:
                await self._upsert_cache(db, upsert_docs)
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "RAG embedding 缓存写入失败（不影响本次召回）: %s", exc
                )

        return [v for v in vectors if v is not None]

    # ---------------- 检索 ----------------

    async def retrieve(
        self,
        db,
        *,
        scholar_id: str,
        query: str,
        top_k: int | None = None,
        exclude_lesson_ids: list[str] | None = None,
    ) -> list[dict]:
        top_k = top_k or self.top_k
        if top_k <= 0 or not query.strip():
            return []
        try:
            # 1. 候选句：学者已学句（排除当前课 → 跨课）
            states = (
                (await get_skill_states(db, scholar_id=scholar_id)).get("records") or []
            )
            excluded = set(exclude_lesson_ids or [])
            candidate_ids: list[str] = []
            seen: set[str] = set()
            for r in states:
                sid = r.get("sentence_id")
                lesson_id = r.get("lesson_id")
                if not sid or sid in seen or lesson_id in excluded:
                    continue
                seen.add(sid)
                candidate_ids.append(sid)
            if not candidate_ids:
                return []

            sentences = await get_sentences_by_ids(db, candidate_ids)
            sentences = [s for s in sentences if s.get("text")]
            if not sentences:
                return []

            # 2. 句子向量（惰性缓存）
            vectors = await self._build_vectors(db, sentences)
            if vectors is None:
                return []

            # 3. query 向量
            query_vectors = await self._embed([query])
            if not query_vectors:
                return []
            qv = query_vectors[0]

            # 4. 余弦 top-K（降序，负分截断）
            scored = [
                (cosine_similarity(qv, vec), sent)
                for sent, vec in zip(sentences, vectors)
            ]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            hits: list[dict] = []
            for score, sent in scored[:top_k]:
                if score <= 0:
                    break
                hit = dict(sent)
                hit["score"] = round(score, 4)
                hits.append(hit)
            return hits
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("RAG retrieve 失败（降级空召回）: %s", exc)
            return []


def get_curriculum_retriever() -> CurriculumRetriever:
    """按配置返回 retriever（可替换基础设施，Provider 化）。

    - RAG_RETRIEVER_ENABLED=0 或 RAG_EMBEDDING_MODEL 为空 → NoopCurriculumRetriever
      （book/lesson/sentences 占位，planner 回退 S4.3 非向量版）
    - 否则 → VolcanoEmbeddingRetriever（方舟 embeddings + 惰性缓存 + 内存余弦）
    """
    if not RAG_RETRIEVER_ENABLED or not RAG_EMBEDDING_MODEL:
        return NoopCurriculumRetriever()
    return VolcanoEmbeddingRetriever()
