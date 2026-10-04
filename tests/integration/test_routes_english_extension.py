"""集成测试：英文语句扩展接口（E1-E3；旧 E4 已下线）

被测链路：FastAPI TestClient + FakeDB，不触真实 LLM（no_external_calls 已屏蔽）
- E1 POST /english/extension/points       缓存快路径 / 未命中入队
- E2 POST /english/extension/evaluate     l1_fill 同步 / l2_sentence 异步
- E3 GET  /english/extension/task/{id}    状态轮询 / 过期 404
- 开关关闭 → EXTENSION_DISABLED

2026-10-03（修订 2）：旧 E4 `PUT /points/{sid}` 人工校对覆盖随 `source=manual` 全局
校对层下线而删除，相关用例一并移除；同句多记录的取行口径由「优先 manual」改为
「`updated_at` 最新」。学习者校对路径（E4' / E6）的用例见 B35。

错误响应结构：{"success": false, "code": "...", "message": "..."}
成功响应结构：{"success": true, "data": {...}}
"""
from __future__ import annotations

import config
from services.models.content import compute_text_hash
from services.routes.extension import router as ext_router


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _enable(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_ENABLED", 1)


async def _noop_task(*a, **k):
    """异步空操作，替代 run_extension_task。"""
    return None


def _seed_review(fake_db, scholar_id, sentence_id, original=None, removed=None, added=None):
    """预置学习者的 overlay 补丁（`extension_review`）。"""
    from services.models.content import compute_text_hash

    fake_db.add(
        config.EXTENSION_REVIEW_COLLECTION,
        {
            "scholar_id": scholar_id,
            "sentence_id": sentence_id,
            "content_hash": compute_text_hash(original or "I take part in the discussion."),
            "removed_texts": removed or [],
            "added": added or [],
            "note": "",
            "created_at": 1,
            "updated_at": 1,
        },
    )


def _seed_cached_point(fake_db, sentence_id, points, original="I take part in the discussion."):
    from services.english.extension import extension_idempotency_key
    from services.models.content import compute_text_hash

    content_hash = compute_text_hash(original)
    point_key = extension_idempotency_key(
        sentence_id=sentence_id,
        content_hash=content_hash,
        prompt_version=config.EXTENSION_PROMPT_VERSION,
        model=config.EXTENSION_LLM_MODEL,
    )
    fake_db.add(
        config.EXTENSION_POINT_COLLECTION,
        {
            "sentence_id": sentence_id,
            "point_key": point_key,
            "content_hash": content_hash,
            "status": "success",
            "source": "llm",
            "points": points,
            "prompt_version": config.EXTENSION_PROMPT_VERSION,
            "model": config.EXTENSION_LLM_MODEL,
            "count_by_type": {},
            "attempts": 1,
            "fallback_reason": "",
        },
    )


# ---------------------------------------------------------------------------
# E1 抽取
# ---------------------------------------------------------------------------


class TestExtractPoints:
    def test_cache_hit_returns_cached_no_task(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"}]
        _seed_cached_point(fake_db, "s1", pts)
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={"sentence_id": "s1", "original": "I take part in the discussion."},
        )
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["cached"] is True
        assert body["data"]["task_id"] is None
        assert body["data"]["points"] == pts

    def test_miss_returns_pending_with_task_id(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={"sentence_id": "s2", "original": "I take part in the discussion."},
        )
        body = resp.json()
        assert body["data"]["cached"] is False
        assert body["data"]["status"] == "pending"
        assert body["data"]["task_id"].startswith("ex_")

    def test_without_scholar_id_review_is_null(self, make_client, fake_db, monkeypatch):
        """不传 scholar_id → 纯 AI 结果，`review` 为 null（教研观察视角）。"""
        _enable(monkeypatch)
        pts = [
            {"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"},
            {"id": "s1#1", "type": "phrase", "text": "take part in", "meaning_zh": "参加"},
        ]
        _seed_cached_point(fake_db, "s1", pts)
        _seed_review(fake_db, "sch_1", "s1", removed=["discussion"])
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={"sentence_id": "s1", "original": "I take part in the discussion."},
        )
        data = resp.json()["data"]
        assert data["review"] is None
        assert [p["text"] for p in data["points"]] == ["discussion", "take part in"]

    def test_with_scholar_id_merges_overlay(self, make_client, fake_db, monkeypatch):
        """传 scholar_id → 读侧合并：剔除 removed、追加 mine，`review` 出摘要。"""
        _enable(monkeypatch)
        pts = [
            {"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"},
            {"id": "s1#1", "type": "phrase", "text": "take part in", "meaning_zh": "参加"},
        ]
        _seed_cached_point(fake_db, "s1", pts)
        _seed_review(
            fake_db,
            "sch_1",
            "s1",
            removed=["discussion"],
            added=[{"id": "mine_0", "type": "word", "text": "debate", "meaning_zh": "辩论"}],
        )
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={
                "sentence_id": "s1",
                "original": "I take part in the discussion.",
                "scholar_id": "sch_1",
            },
        )
        data = resp.json()["data"]
        assert [p["text"] for p in data["points"]] == ["take part in", "debate"]
        assert [p["origin"] for p in data["points"]] == ["ai", "mine"]
        assert data["points"][1]["l1"] is None
        assert data["review"] == {
            "has_review": True,
            "added_count": 1,
            "removed_count": 1,
            "stale": False,
            "updated_at": 1,
        }
        # 合并不得写回 AI 集合
        assert fake_db.all(config.EXTENSION_POINT_COLLECTION)[0]["points"] == pts

    def test_review_disabled_returns_pure_ai(self, make_client, fake_db, monkeypatch):
        """校对开关关 → E1 仍是纯 AI 结果（不是 EXTENSION_DISABLED，那是 E4'/E6 的语义）。"""
        _enable(monkeypatch)
        monkeypatch.setattr(config, "EXTENSION_REVIEW_ENABLED", 0)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"}]
        _seed_cached_point(fake_db, "s1", pts)
        _seed_review(fake_db, "sch_1", "s1", removed=["discussion"])
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={
                "sentence_id": "s1",
                "original": "I take part in the discussion.",
                "scholar_id": "sch_1",
            },
        )
        data = resp.json()["data"]
        assert data["review"] is None
        assert [p["text"] for p in data["points"]] == ["discussion"]

    def test_sentence_too_short(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)
        resp = client.post(
            "/english/extension/points",
            json={"sentence_id": "s3", "original": "Hi there"},
        )
        assert resp.json()["code"] == "SENTENCE_TOO_SHORT"


# ---------------------------------------------------------------------------
# E2 评测
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_l1_fill_success(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"}]
        _seed_cached_point(fake_db, "s1", pts)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#0"],
                "task_type": "l1_fill",
                "user_input": "discussion",
            },
        )
        body = resp.json()
        assert body["data"]["status"] == "success"
        assert body["data"]["result"]["passed"] is True
        assert body["data"]["result"]["score"] == 1

    def test_l1_fill_wrong_answer(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"}]
        _seed_cached_point(fake_db, "s1", pts)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#0"],
                "task_type": "l1_fill",
                "user_input": "meeting",
            },
        )
        assert resp.json()["data"]["result"]["passed"] is False

    def test_l1_selected_ids_not_in_list(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion"}]
        _seed_cached_point(fake_db, "s1", pts)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#999"],
                "task_type": "l1_fill",
                "user_input": "x",
            },
        )
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_l2_sentence_returns_pending(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        # 提交时落 points_snapshot：先有抽取记录，否则 INVALID_INPUT
        _seed_cached_point(
            fake_db, "s1", [{"id": "s1#0", "type": "phrase", "text": "take part in"}]
        )
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#0"],
                "task_type": "l2_sentence",
                "user_input": "I take part in the discussion.",
            },
        )
        assert resp.json()["data"]["status"] == "pending"
        assert resp.json()["data"]["task_id"].startswith("ex_")

    def test_l2_selected_ids_not_in_list(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        _seed_cached_point(
            fake_db, "s1", [{"id": "s1#0", "type": "phrase", "text": "take part in"}]
        )
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#999"],
                "task_type": "l2_sentence",
                "user_input": "I take part in the discussion.",
            },
        )
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_my_own_point_is_rejected(self, make_client, fake_db, monkeypatch):
        """B34：学习者自建点（origin=mine / id 前缀 mine_）进判分 → INVALID_INPUT，不静默过滤。"""
        _enable(monkeypatch)
        _seed_cached_point(
            fake_db, "s1", [{"id": "s1#0", "type": "word", "text": "discussion"}]
        )
        client = make_client(ext_router)

        for task_type in ("l1_fill", "l2_sentence"):
            resp = client.post(
                "/english/extension/evaluate",
                json={
                    "sentence_id": "s1",
                    "selected_ids": ["s1#0", "mine_0"],
                    "task_type": task_type,
                    "user_input": "discussion",
                },
            )
            body = resp.json()
            assert body["success"] is False
            assert body["code"] == "INVALID_INPUT"
            assert "自建语言点" in body["message"]


# ---------------------------------------------------------------------------
# E3 任务查询
# ---------------------------------------------------------------------------


class TestGetTask:
    def test_pending_task(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        import asyncio
        from services.learning.extension_task import create_extension_task

        async def _create():
            return await create_extension_task(fake_db, kind="extract", sentence_id="s1")

        doc = asyncio.run(_create())
        client = make_client(ext_router)

        resp = client.get(f"/english/extension/task/{doc['task_id']}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "pending"

    def test_expired_task_404(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        fake_db.add(
            config.EXTENSION_TASK_COLLECTION,
            {
                "task_id": "ex_expired",
                "kind": "extract",
                "status": "pending",
                "expires_at": 1,
                "created_at": 0,
                "updated_at": 0,
            },
        )
        client = make_client(ext_router)
        resp = client.get("/english/extension/task/ex_expired")
        assert resp.status_code == 404

    def test_nonexistent_task_404(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        client = make_client(ext_router)
        resp = client.get("/english/extension/task/ex_notfound")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 开关
# ---------------------------------------------------------------------------


class TestExtensionDisabled:
    def test_all_endpoints_return_disabled(self, make_client, fake_db, monkeypatch):
        monkeypatch.setattr(config, "EXTENSION_ENABLED", 0)
        client = make_client(ext_router)

        r1 = client.post(
            "/english/extension/points",
            json={"sentence_id": "s1", "original": "hello world foo bar"},
        )
        assert r1.json()["code"] == "EXTENSION_DISABLED"

        r2 = client.post(
            "/english/extension/evaluate",
            json={"sentence_id": "s1", "selected_ids": ["x"], "task_type": "l1_fill"},
        )
        assert r2.json()["code"] == "EXTENSION_DISABLED"

        r3 = client.get("/english/extension/task/ex_any")
        assert r3.json()["code"] == "EXTENSION_DISABLED"

        # 旧 E4（PUT /points/{sid}）已随 source=manual 下线删除：端点不存在 → 404
        r4 = client.put("/english/extension/points/s1", json={"points": []})
        assert r4.status_code == 404


# ---------------------------------------------------------------------------
# E4' / E6 学习者校对（修订 2 · B35）
# ---------------------------------------------------------------------------


class TestMyReview:
    def test_put_and_get_review_roundtrip(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        pts = [{"id": "s1#0", "type": "word", "text": "discussion", "meaning_zh": "讨论"}]
        _seed_cached_point(fake_db, "s1", pts)
        client = make_client(ext_router)

        put = client.put(
            "/english/extension/review/s1",
            json={
                "scholar_id": "sch_1",
                "content_hash": compute_text_hash("I take part in the discussion."),
                "removed_texts": ["discussion"],
                "added": [{"type": "word", "text": "debate", "meaning_zh": "辩论"}],
            },
        )
        body = put.json()
        assert body["success"] is True
        assert body["data"]["status"] == "success"
        assert body["data"]["changes"] == {"added": 1, "removed": 1, "restored": 0}
        assert [p["text"] for p in body["data"]["points"]] == ["debate"]
        assert [p["origin"] for p in body["data"]["points"]] == ["mine"]

        got = client.get("/english/extension/review/s1?scholar_id=sch_1")
        data = got.json()["data"]
        assert data["review"]["removed_texts"] == ["discussion"]
        assert data["review"]["stale"] is False
        assert {(h["action"], h["target"]) for h in data["history"]} == {
            ("remove", "discussion"),
            ("add", "debate"),
        }
        # 全程不写 AI 集合
        assert fake_db.all(config.EXTENSION_POINT_COLLECTION)[0]["points"] == pts

    def test_put_review_rejects_empty_scholar_id(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        client = make_client(ext_router)

        resp = client.put(
            "/english/extension/review/s1",
            json={"scholar_id": "", "content_hash": "h1", "removed_texts": [], "added": []},
        )
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_get_review_rejects_empty_scholar_id(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        client = make_client(ext_router)

        resp = client.get("/english/extension/review/s1?scholar_id=")
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_review_disabled_returns_extension_disabled(self, make_client, fake_db, monkeypatch):
        """校对开关独立于总开关：关 → E4' / E6 报 EXTENSION_DISABLED。"""
        _enable(monkeypatch)
        monkeypatch.setattr(config, "EXTENSION_REVIEW_ENABLED", 0)
        client = make_client(ext_router)

        r1 = client.put(
            "/english/extension/review/s1",
            json={"scholar_id": "sch_1", "content_hash": "h1"},
        )
        assert r1.json()["code"] == "EXTENSION_DISABLED"

        r2 = client.get("/english/extension/review/s1?scholar_id=sch_1")
        assert r2.json()["code"] == "EXTENSION_DISABLED"


# ---------------------------------------------------------------------------
# 同句多记录下的取行正确性
# （换 prompt_version / 换 model 重抽会让同一 sentence_id 并存多条记录，
#   故 E2 必须按「`updated_at` 最新」取行，不能 limit=1 随意取一条。
#   2026-10-03 修订 2：原「优先 source=manual」随全局人工校对层下线移除）
# ---------------------------------------------------------------------------


def _add_point_record(fake_db, sentence_id, point_key, source, points, updated_at):
    fake_db.add(
        config.EXTENSION_POINT_COLLECTION,
        {
            "sentence_id": sentence_id,
            "point_key": point_key,
            "source": source,
            "status": "success",
            "points": points,
            "updated_at": updated_at,
        },
    )


class TestEffectiveRecordResolution:
    def test_evaluate_uses_latest_record_by_updated_at(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        _add_point_record(
            fake_db, "s1", "k_old", "llm",
            [{"id": "s1#0", "type": "word", "text": "WRONG", "meaning_zh": "x"}], 1000,
        )
        _add_point_record(
            fake_db, "s1", "k_new", "llm",
            [{"id": "s1#0", "type": "word", "text": "take part in", "meaning_zh": "x"}], 2000,
        )
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#0"],
                "task_type": "l1_fill",
                "user_input": "take part in",
            },
        )
        # 取行按 updated_at 最新（k_new / 2000）→ 命中；若误取旧行（k_old / 1000）则为 False
        assert resp.json()["data"]["result"]["passed"] is True

    def test_evaluate_does_not_pick_stale_record(self, make_client, fake_db, monkeypatch):
        """旧行的答案不得被判为命中：证明取行不是 limit=1 随意取一条。"""
        _enable(monkeypatch)
        _add_point_record(
            fake_db, "s1", "k_old", "llm",
            [{"id": "s1#0", "type": "word", "text": "WRONG", "meaning_zh": "x"}], 1000,
        )
        _add_point_record(
            fake_db, "s1", "k_new", "rule",
            [{"id": "s1#0", "type": "word", "text": "take part in", "meaning_zh": "x"}], 2000,
        )
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/evaluate",
            json={
                "sentence_id": "s1",
                "selected_ids": ["s1#0"],
                "task_type": "l1_fill",
                "user_input": "WRONG",
            },
        )
        assert resp.json()["data"]["result"]["passed"] is False

    def test_e1_cache_path_hits_ai_record_by_point_key(self, make_client, fake_db, monkeypatch):
        _enable(monkeypatch)
        monkeypatch.setattr("services.routes.extension.run_extension_task", _noop_task)
        _seed_cached_point(
            fake_db, "s1", [{"id": "s1#0", "type": "word", "text": "kept", "meaning_zh": "x"}]
        )
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/points",
            json={"sentence_id": "s1", "original": "I take part in the discussion."},
        )
        data = resp.json()["data"]
        assert data["cached"] is True
        assert data["task_id"] is None
        assert data["meta"]["source"] == "llm"
        assert data["points"][0]["text"] == "kept"
