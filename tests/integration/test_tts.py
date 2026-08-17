"""集成测试:TTS 语音合成(F3/3.3) — POST /audio/tts

被测链路:FastAPI TestClient + Fake TTSProvider + FakeDB,覆盖:
- 正常合成:校验 → 缓存 miss → Provider 合成 → 落库 audio_asset(text_hash 唯一键)→ from_cache=false
- 文本键控缓存:二次请求命中 → from_cache=true,不再调 Provider,ref_count +1
- 参数校验:缺失 / 空白 / 超 200 字符 → INVALID_TEXT
- 降级:Provider 不可用 / 合成失败 / 响应缺 Audio → TTS_UNAVAILABLE
- 容错:落库异常不阻塞返回(对齐 /eval/speech)
- 契约:业务失败 200 + success=false + code,与 /eval/translate 对齐
"""
from __future__ import annotations

import base64
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.dependencies import get_db
from services.routes_tts import router as tts_router
from services.tts import (
    SEMANTIC_ENGINE_RESERVED,
    SEMANTIC_SEARCH_MAX_TEXT_CHARS,
    SEMANTIC_SEARCH_MAX_TOP_K,
    TTS_CODEC,
    TTS_MAX_TEXT_CHARS,
    TTS_SAMPLE_RATE,
    TTS_VOICE_TYPE,
    get_tts_provider,
    hash_text,
    normalize_text,
)

FAKE_AUDIO = base64.b64encode(b"fake-mp3-audio-bytes").decode()
RAW_TTS_OK = {
    "Audio": FAKE_AUDIO,
    "RequestId": "req-tts-001",
    "SessionId": "sess-abc-001",
}
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog"


class FakeTtsProvider:
    """可用且返回固定合成结果的 Provider（记录调用次数与收到的文本）。"""

    available = True

    def __init__(self, raw=None):
        self.raw = raw if raw is not None else RAW_TTS_OK
        self.call_count = 0
        self.called_texts: list[str] = []

    def synthesize(self, text):
        self.call_count += 1
        self.called_texts.append(text)
        return self.raw


class UnavailableTtsProvider:
    """凭据未配置的 Provider。"""

    available = False

    def synthesize(self, text):
        return None


class FailTtsProvider:
    """可用但合成失败（返回 None）。"""

    available = True

    def synthesize(self, text):
        return None


def _client(fake_db, provider=None) -> TestClient:
    """构建 TestClient：注入 FakeDB 与 Fake TTSProvider（get_db / get_tts_provider
    均为 Depends 在定义期捕获的函数对象，必须走 dependency_overrides 才能生效）。"""
    if provider is None:
        provider = FakeTtsProvider()
    app = FastAPI()
    app.include_router(tts_router)
    app.dependency_overrides[get_db] = lambda: fake_db
    app.dependency_overrides[get_tts_provider] = lambda: provider
    return TestClient(app)


class TestTtsSuccess:
    def test_normal_synthesis_and_persist(self, fake_db):
        """正常链路：校验 → 缓存 miss → Provider 合成 → 落库 audio_asset。"""
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["code"] == "OK"

        data = body["data"]
        assert data["audio_base64"] == FAKE_AUDIO
        assert data["codec"] == TTS_CODEC
        assert data["sample_rate"] == TTS_SAMPLE_RATE
        assert data["from_cache"] is False

        # 落库字段（data-model-contract §4.10）
        rows = fake_db.all("audio_asset")
        assert len(rows) == 1
        doc = rows[0]
        assert doc["text_hash"] == hash_text(DEFAULT_TEXT)
        assert doc["text"] == normalize_text(DEFAULT_TEXT)
        assert doc["audio_base64"] == FAKE_AUDIO
        assert doc["codec"] == TTS_CODEC
        assert doc["sample_rate"] == TTS_SAMPLE_RATE
        assert doc["voice"] == str(TTS_VOICE_TYPE)
        assert doc["tts_request_id"] == "req-tts-001"
        assert doc["ref_count"] == 0
        assert isinstance(doc["created_at"], int) and doc["created_at"] > 0

    def test_sentence_id_persisted(self, fake_db):
        """sentence_id 透传落库（便于运营对账）。"""
        client = _client(fake_db)
        resp = client.post(
            "/audio/tts",
            json={"text": DEFAULT_TEXT, "sentence_id": "sent_0001"},
        )
        assert resp.json()["success"] is True
        doc = fake_db.all("audio_asset")[0]
        assert doc["sentence_id"] == "sent_0001"

    def test_provider_receives_normalized_text(self, fake_db):
        """Provider 收到的 text 应为 normalize 后（trim + 合并连续空白）。"""
        provider = FakeTtsProvider()
        client = _client(fake_db, provider=provider)
        resp = client.post("/audio/tts", json={"text": "  The   quick  brown fox  "})
        assert resp.json()["success"] is True
        assert provider.called_texts == [normalize_text("  The   quick  brown fox  ")]


class TestTtsCache:
    def test_cache_hit_second_request(self, fake_db):
        """二次请求命中 text_hash 缓存：from_cache=true、不再调 Provider、ref_count +1。"""
        provider = FakeTtsProvider()
        client = _client(fake_db, provider=provider)

        first = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        assert first.json()["data"]["from_cache"] is False
        assert provider.call_count == 1

        second = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = second.json()
        assert body["success"] is True
        assert body["data"]["from_cache"] is True
        assert body["data"]["audio_base64"] == FAKE_AUDIO
        assert provider.call_count == 1  # 缓存命中,不再调 TTS

        doc = fake_db.all("audio_asset")[0]
        assert doc["ref_count"] == 1  # 命中 +1
        assert len(fake_db.all("audio_asset")) == 1  # 不重复落库

    def test_cache_hit_on_normalized_variant(self, fake_db):
        """空白变体文本归一后命中同一 text_hash 缓存（trim + 合并空白）。"""
        client = _client(fake_db)
        client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        resp = client.post("/audio/tts", json={"text": f"  {DEFAULT_TEXT.replace(' ', '   ')}  "})
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["from_cache"] is True
        assert len(fake_db.all("audio_asset")) == 1

    def test_ref_count_update_failure_does_not_block_cache(self, fake_db, monkeypatch):
        """命中时 ref_count 更新失败仅记日志，缓存仍正常返回。"""
        client = _client(fake_db)
        client.post("/audio/tts", json={"text": DEFAULT_TEXT})

        async def _boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(fake_db, "update", _boom)
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = resp.json()
        assert resp.status_code == 200
        assert body["success"] is True
        assert body["data"]["from_cache"] is True
        assert body["data"]["audio_base64"] == FAKE_AUDIO


class TestTtsValidation:
    def test_empty_text(self, fake_db):
        """text 为空串 → INVALID_TEXT。"""
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={"text": ""})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_TEXT"
        assert fake_db.all("audio_asset") == []

    def test_blank_text(self, fake_db):
        """text 为纯空白 → INVALID_TEXT。"""
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={"text": "   "})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_TEXT"
        assert fake_db.all("audio_asset") == []

    def test_text_too_long(self, fake_db):
        """text 超 200 字符 → INVALID_TEXT。"""
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={"text": "a" * (TTS_MAX_TEXT_CHARS + 1)})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_TEXT"
        assert fake_db.all("audio_asset") == []

    def test_missing_text_422(self, fake_db):
        """缺 text 字段 → pydantic 必填校验 422（契约外的技术校验）。"""
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={})
        assert resp.status_code == 422


class TestTtsFallback:
    def test_provider_unavailable(self, fake_db):
        """Provider 无凭据 → TTS_UNAVAILABLE（前端可回退看文字模式）。"""
        client = _client(fake_db, provider=UnavailableTtsProvider())
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TTS_UNAVAILABLE"
        assert fake_db.all("audio_asset") == []

    def test_synthesis_failure(self, fake_db):
        """Provider 可用但合成失败（返回 None）→ TTS_UNAVAILABLE。"""
        client = _client(fake_db, provider=FailTtsProvider())
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TTS_UNAVAILABLE"
        assert fake_db.all("audio_asset") == []

    def test_response_missing_audio(self, fake_db):
        """Provider 返回原始响应但缺 Audio 字段 → TTS_UNAVAILABLE。"""
        client = _client(fake_db, provider=FakeTtsProvider(raw={"RequestId": "req-x"}))
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TTS_UNAVAILABLE"
        assert fake_db.all("audio_asset") == []

    def test_persist_failure_does_not_block_result(self, fake_db, monkeypatch):
        """落库异常仅记日志，不阻塞合成结果返回（对齐 /eval/speech）。"""
        client = _client(fake_db)

        async def _boom(*args, **kwargs):
            raise RuntimeError("db down")

        monkeypatch.setattr(fake_db, "insert", _boom)
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        body = resp.json()
        assert resp.status_code == 200
        assert body["success"] is True  # 合成成功，落库失败不降级
        assert body["data"]["from_cache"] is False
        assert body["data"]["audio_base64"] == FAKE_AUDIO

    def test_created_at_uses_ms_timestamp(self, fake_db):
        """落库 created_at 为毫秒时间戳（与既有 data-model 口径一致）。"""
        before = int(time.time() * 1000)
        client = _client(fake_db)
        client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        after = int(time.time() * 1000)
        doc = fake_db.all("audio_asset")[0]
        assert before <= doc["created_at"] <= after


class TestTtsSemanticSearchReserved:
    """F3/3.4 语义召回复用预留：接口占位，本步不实现召回（RAG 接入为后续非目标预留位）。"""

    def test_reserved_returns_empty_hits(self, fake_db):
        """预留接口：success=true + engine=reserved + hits=[]（对齐契约出参形态）。"""
        client = _client(fake_db)
        resp = client.post("/audio/assets/search", json={"text": "a brown fox"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["code"] == "OK"
        assert body["data"]["engine"] == SEMANTIC_ENGINE_RESERVED
        assert body["data"]["hits"] == []
        assert fake_db.all("audio_asset") == []  # 只读占位，不落库

    def test_invalid_text(self, fake_db):
        """text 空 / 纯空白 → INVALID_INPUT（200 + success=false）；缺字段走 pydantic 422。"""
        client = _client(fake_db)
        # 缺 text：pydantic 必填校验 422（对齐 /audio/tts 的 test_missing_text_422）
        assert client.post("/audio/assets/search", json={}).status_code == 422
        for payload in ({"text": ""}, {"text": "   "}):
            body = client.post("/audio/assets/search", json=payload).json()
            assert body["success"] is False
            assert body["code"] == "INVALID_INPUT"

    def test_text_too_long(self, fake_db):
        """text 超 500 字符 → INVALID_INPUT。"""
        client = _client(fake_db)
        resp = client.post(
            "/audio/assets/search",
            json={"text": "a" * (SEMANTIC_SEARCH_MAX_TEXT_CHARS + 1)},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_top_k_out_of_range(self, fake_db):
        """top_k 越界（0 / 21）→ INVALID_INPUT（契约：业务失败 200 + success=false，非 422）。"""
        client = _client(fake_db)
        for top_k in (0, SEMANTIC_SEARCH_MAX_TOP_K + 1):
            resp = client.post(
                "/audio/assets/search", json={"text": "fox", "top_k": top_k}
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["success"] is False
            assert body["code"] == "INVALID_INPUT"

    def test_tts_miss_invokes_semantic_lookup(self, fake_db, monkeypatch):
        """/audio/tts 缓存 miss 时调用 semantic_lookup 预留位（F3/3.4 联动），不改变合成行为。"""
        calls = []

        def _fake_lookup(text, db=None, top_k=1):
            calls.append(text)
            return None

        monkeypatch.setattr("services.routes_tts.semantic_lookup", _fake_lookup)
        client = _client(fake_db)
        resp = client.post("/audio/tts", json={"text": DEFAULT_TEXT})
        assert resp.json()["success"] is True
        assert calls == [normalize_text(DEFAULT_TEXT)]  # 预留调用点已接好
        assert fake_db.all("audio_asset")  # 占位返回 None → 正常合成落库
