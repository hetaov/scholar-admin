"""集成测试：评估证据接口（契约 api-contract §3.9）— POST /evaluation/{ref}/evaluate、GET /evaluation/{id}

被测链路：FastAPI TestClient + FakeDB，覆盖：
- 文本评估：learning_attempt:<id> → eval_verdict 落 evaluation 集合
- 语音评估：speech:<id> → 基于 SOE-N parsed 的 L1 评估
- 幂等：重复触发同 ref 直接命中已有评估；force=true 重算更新
- 错误：ref 非法 / 记录不存在 / 查询不存在
- 原则：证据只读不写；低置信不回写 SkillState（状态回写由会话/训练路由门控）
"""
from __future__ import annotations

from services.routes_evaluation import router as evaluation_router
from tests.fakes.seed_factory import seed_attempt, seed_speech


class TestEvaluationTrigger:
    def test_text_attempt(self, make_client, fake_db):
        client = make_client(evaluation_router)
        attempt_id = seed_attempt(fake_db)
        resp = client.post(f"/evaluation/learning_attempt:{attempt_id}/evaluate", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["type"] == "text"
        assert data["attempt_ref"] == f"learning_attempt:{attempt_id}"
        assert data["verdict"]["meaningful"] is True
        assert data["score"] == 100
        assert 0 <= data["confidence"] <= 1

    def test_speech_evaluation(self, make_client, fake_db):
        client = make_client(evaluation_router)
        speech_id = seed_speech(fake_db)
        resp = client.post(f"/evaluation/speech:{speech_id}/evaluate", json={})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["type"] == "speech"
        assert data["score"] == 88  # suggested_score 优先
        assert data["verdict"]["anomaly"] is False
        assert data["verdict"]["meaningful"] is True

    def test_idempotent_no_force(self, make_client, fake_db):
        client = make_client(evaluation_router)
        attempt_id = seed_attempt(fake_db)
        ref = f"learning_attempt:{attempt_id}"
        r1 = client.post(f"/evaluation/{ref}/evaluate", json={}).json()["data"]
        r2 = client.post(f"/evaluation/{ref}/evaluate", json={}).json()["data"]
        assert r1["_id"] == r2["_id"]  # 命中同一条，幂等
        assert len(fake_db.all("evaluation")) == 1

    def test_force_recompute_updates(self, make_client, fake_db):
        client = make_client(evaluation_router)
        attempt_id = seed_attempt(fake_db)
        ref = f"learning_attempt:{attempt_id}"
        client.post(f"/evaluation/{ref}/evaluate", json={})
        r2 = client.post(
            f"/evaluation/{ref}/evaluate", json={"force": True}
        ).json()["data"]
        assert len(fake_db.all("evaluation")) == 1  # 不新增，仅更新
        assert r2["updated_at"] >= r2["created_at"]

    def test_invalid_ref(self, make_client, fake_db):
        client = make_client(evaluation_router)
        resp = client.post("/evaluation/bad-ref/evaluate", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_missing_record(self, make_client, fake_db):
        client = make_client(evaluation_router)
        resp = client.post("/evaluation/learning_attempt:missing/evaluate", json={})
        assert resp.status_code == 200
        assert resp.json()["code"] == "NOT_FOUND"


class TestEvaluationGet:
    def test_get_existing(self, make_client, fake_db):
        client = make_client(evaluation_router)
        attempt_id = seed_attempt(fake_db)
        created = client.post(
            f"/evaluation/learning_attempt:{attempt_id}/evaluate", json={}
        ).json()["data"]
        resp = client.get(f"/evaluation/{created['_id']}")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["_id"] == created["_id"]
        assert data["attempt_ref"] == f"learning_attempt:{attempt_id}"

    def test_get_missing(self, make_client, fake_db):
        client = make_client(evaluation_router)
        resp = client.get("/evaluation/nope")
        assert resp.status_code == 200
        assert resp.json()["code"] == "NOT_FOUND"

    def test_get_empty_id(self, make_client, fake_db):
        client = make_client(evaluation_router)
        resp = client.get("/evaluation/%20")
        assert resp.status_code == 200
        assert resp.json()["code"] == "INVALID_INPUT"
