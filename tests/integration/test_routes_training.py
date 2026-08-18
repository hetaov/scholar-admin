"""集成测试：Training 受控任务接口（契约 api-contract §3.9）— exercise / evaluate

被测链路：FastAPI TestClient + FakeDB，覆盖：
- POST /training/exercise：弱项驱动选句生成受控任务（learning_attempt mode='training'）
- POST /training/evaluate：逐题判定 + 即时反馈 + 门控 SkillState 回写 + attempt 事件
- 错误：缺参 / 任务不存在 / 重复提交
- 原则：弱项驱动（§9-4）；低置信不回写（§9-2）；并入 learning_attempt（附录 B-2）
"""
from __future__ import annotations

import pytest

from services.cold_start import COLD_START_SEQUENCE
from services.routes_training import router as training_router


def _seed_state(fake_db, scholar: str = "u1") -> None:
    # 弱项：s2 分最低 → 应优先选中
    fake_db.add("skill_state", {
        "_id": f"{scholar}_s1_translation",
        "scholar_id": scholar,
        "sentence_id": "s1",
        "skill_code": "translation",
        "mastery_score": 90,
        "status": "mastered",
        "attempt_count": 5,
    })
    fake_db.add("skill_state", {
        "_id": f"{scholar}_s2_translation",
        "scholar_id": scholar,
        "sentence_id": "s2",
        "skill_code": "translation",
        "mastery_score": 30,
        "status": "learning",
        "attempt_count": 2,
    })
    fake_db.add("sentence_v2", {
        "_id": "sv_s1",
        "sentence_id": "s1",
        "text": "It is a watch.",
        "translation": "这是一块手表。",
        "lesson_id": "l1",
        "chapter_id": "c1",
        "order": 1,
    })
    fake_db.add("sentence_v2", {
        "_id": "sv_s2",
        "sentence_id": "s2",
        "text": "The weather is nice.",
        "translation": "天气很好。",
        "lesson_id": "l1",
        "chapter_id": "c1",
        "order": 2,
    })


class TestExercise:
    def test_weakness_driven_generation(self, make_client, fake_db):
        _seed_state(fake_db)
        client = make_client(training_router)
        resp = client.post(
            "/training/exercise",
            json={"scholar_id": "u1", "skill_code": "translation"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["task_id"].startswith("trn_")
        assert data["difficulty"] == 1
        # 弱项驱动：mastery=30 的 s2 应入选
        items = data["items"]
        assert items and items[0]["sentence_id"] == "s2"
        assert items[0]["prompt"] == "翻译为英文：天气很好。"
        # 已有历史 → 非冷启动（§5.6.5）
        assert data["cold_start"] is False
        # 并入 learning_attempt（附录 B-2）
        stored = fake_db.all("learning_attempt")
        assert len(stored) == 1
        assert stored[0]["mode"] == "training"
        assert stored[0]["attempt_status"] == "pending"

    def test_cold_start_no_state(self, make_client, fake_db):
        # 无任何 skill_state（冷启动 §5.6）：不报错不阻断（§9-9），任务体为空
        client = make_client(training_router)
        resp = client.post(
            "/training/exercise",
            json={"scholar_id": "newbie", "skill_code": "listening"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["items"] == []
        assert data["cold_start"] is True  # §5.6.5 冷启动标记
        assert data["prior_defaults"]["mastery"] == pytest.approx(0.35)
        assert data["prior_defaults"]["difficulty"] == 1
        assert resp.json()["success"] is True

    def test_missing_params(self, make_client, fake_db):
        client = make_client(training_router)
        resp = client.post("/training/exercise", json={})
        assert resp.json()["success"] is False
        assert resp.json()["code"] == "INVALID_INPUT"


class TestRecommend:
    """GET /training/recommend（S3.3 训练 Tab 推荐）：冷启动 / 弱项驱动 / 错误契约。"""

    def test_cold_start_returns_guide_sequence(self, make_client, fake_db):
        # 无历史（冷启动 §5.6）：strategy=cold_start，activities 回退标准引导序列，不报错
        client = make_client(training_router)
        resp = client.get("/training/recommend", params={"scholar_id": "newbie"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["strategy"] == "cold_start"
        assert data["has_history"] is False
        assert data["evidence_sparse"] is True
        assert data["activities"] == list(COLD_START_SEQUENCE)
        assert data["mastery"] is None
        assert data["skill_states"] == []

    def test_weakness_driven_with_sufficient_evidence(self, make_client, fake_db):
        # 有历史且证据充足（总尝试 5+2=7 ≥ MIN_EVIDENCE）：strategy=weakness，
        # 弱项驱动 Activities，能力状态按 skill_code 聚合
        _seed_state(fake_db)
        client = make_client(training_router)
        resp = client.get("/training/recommend", params={"scholar_id": "u1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["strategy"] == "weakness"
        assert data["has_history"] is True
        assert data["evidence_sparse"] is False
        # 弱项驱动 → 推荐 4 个活动（非空且不重复）
        assert data["activities"]
        assert len(set(data["activities"])) == len(data["activities"])
        # 能力状态：translation 聚合（mastery=(0.9+0.3)/2=0.6，总尝试 7）
        states = {s["skill_code"]: s for s in data["skill_states"]}
        assert "translation" in states
        assert states["translation"]["mastery"] == pytest.approx(0.6, abs=1e-4)
        assert states["translation"]["attempt_count"] == 7

    def test_sparse_evidence_falls_back(self, make_client, fake_db):
        # 有历史但证据稀疏（总尝试 1 < MIN_EVIDENCE）：回退引导序列 + cold_start 策略
        fake_db.add("skill_state", {
            "_id": "u3_s9_translation",
            "scholar_id": "u3",
            "sentence_id": "s9",
            "skill_code": "translation",
            "mastery_score": 30,
            "status": "learning",
            "attempt_count": 1,
        })
        client = make_client(training_router)
        resp = client.get("/training/recommend", params={"scholar_id": "u3"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["strategy"] == "cold_start"
        assert data["has_history"] is True
        assert data["evidence_sparse"] is True
        assert data["activities"] == list(COLD_START_SEQUENCE)

    def test_missing_scholar_id(self, make_client, fake_db):
        client = make_client(training_router)
        resp = client.get("/training/recommend")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"


class TestEvaluate:
    def test_evaluate_flow_with_gate(self, make_client, fake_db):
        _seed_state(fake_db)
        client = make_client(training_router)
        task_id = client.post(
            "/training/exercise",
            json={"scholar_id": "u1", "skill_code": "translation"},
        ).json()["data"]["task_id"]

        # 弱项驱动选出 2 句（s2 最弱 + s1），提交全部作答
        task = fake_db.all("learning_attempt")[0]
        assert [it["sentence_id"] for it in task["items"]] == ["s2", "s1"]
        resp = client.post(
            "/training/evaluate",
            json={
                "task_id": task_id,
                "answers": [
                    {"item_id": "it_1", "response": "the weather is nice"},
                    {"item_id": "it_2", "response": "it is a watch"},
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["overall"]["total"] == 2
        assert data["overall"]["correct"] == 2
        r = data["results"][0]
        assert r["item_id"] == "it_1"
        assert r["correct"] is True
        assert r["feedback"]  # 即时反馈（§9-4）

        # 落库更新：learning_attempt 回写判定
        stored = fake_db.all("learning_attempt")[0]
        assert stored["attempt_status"] == "completed"
        assert stored["overall"]["correct"] == 2
        # 门控回写：correct 且高置信 → skill_state attempt_count 增加
        s2_state = [
            s for s in fake_db.all("skill_state")
            if s["sentence_id"] == "s2" and s["skill_code"] == "translation"
        ][0]
        assert s2_state["attempt_count"] == 3

    def test_missing_task(self, make_client, fake_db):
        client = make_client(training_router)
        resp = client.post(
            "/training/evaluate",
            json={"task_id": "trn_nope", "answers": []},
        )
        assert resp.json()["code"] == "NOT_FOUND"

    def test_duplicate_evaluate_conflict(self, make_client, fake_db):
        _seed_state(fake_db)
        client = make_client(training_router)
        task_id = client.post(
            "/training/exercise",
            json={"scholar_id": "u1", "skill_code": "translation"},
        ).json()["data"]["task_id"]
        payload = {
            "task_id": task_id,
            "answers": [{"item_id": "it_1", "response": "the weather is nice"}],
        }
        client.post("/training/evaluate", json=payload)
        resp2 = client.post("/training/evaluate", json=payload)
        assert resp2.json()["success"] is False
        assert resp2.json()["code"] == "CONFLICT"
