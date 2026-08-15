"""集成测试：异步对话匹配任务接口（Phase 2）

被测链路（FastAPI TestClient + FakeDB，不触真实火山）：
- POST /match/dialogue/task          提交任务 → 毫秒级返回 {taskId, status: pending}
- GET  /match/dialogue/task/{task_id} 查询任务状态透传（pending/success/failed）

要点：
- monkeypatch get_db → FakeDB，提交接口同时触发后台执行器
- 执行器 monkeypatch 为记录型 stub，避免真实异步时序影响断言
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.dialogue_task import run_dialogue_task
from services.routes_dialogue import router as dialogue_router

TASK_TTL_MS = 24 * 60 * 60 * 1000


def _run(coro):
    import asyncio

    return asyncio.run(coro)


def _client(monkeypatch, fake_db) -> TestClient:
    monkeypatch.setattr("services.routes_dialogue.get_db", lambda: fake_db)
    monkeypatch.setattr("services.dialogue_task.get_db", lambda: fake_db)
    app = FastAPI()
    app.include_router(dialogue_router)
    return TestClient(app)


def _seed_task(fake_db, **overrides) -> dict:
    now = int(time.time() * 1000)
    doc = {
        "task_id": "dt_test",
        "scholar_id": "s1",
        "sentence": "Hello",
        "status": "pending",
        "result": None,
        "is_question": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "expires_at": now + TASK_TTL_MS,
    }
    doc.update(overrides)
    fake_db.add("dialogue_task", doc)
    return doc


class TestCreateDialogueTask:
    """POST /match/dialogue/task"""

    def test_ok_returns_task_id_pending(self, monkeypatch, fake_db):
        called = {}

        async def fake_run(task_id, scholar_id, sentence):
            called["task_id"] = task_id
            called["scholar_id"] = scholar_id
            called["sentence"] = sentence

        monkeypatch.setattr(
            "services.routes_dialogue.run_dialogue_task", fake_run
        )
        client = _client(monkeypatch, fake_db)

        resp = client.post(
            "/match/dialogue/task",
            json={"scholarId": "s1", "sentence": "Hello"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["taskId"].startswith("dt_")
        assert data["status"] == "pending"

        # 后台执行器已调度，参数正确
        assert called["scholar_id"] == "s1"
        assert called["sentence"] == "Hello"
        assert called["task_id"] == data["taskId"]

        # 任务已落库且为 pending
        stored = fake_db.all("dialogue_task")
        assert len(stored) == 1
        assert stored[0]["status"] == "pending"

    def test_missing_scholar_id(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/match/dialogue/task", json={"sentence": "Hello"})
        assert resp.status_code == 400
        assert "scholarId" in resp.json()["detail"]
        assert fake_db.all("dialogue_task") == []

    def test_missing_sentence(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.post("/match/dialogue/task", json={"scholarId": "s1"})
        assert resp.status_code == 400
        assert "sentence" in resp.json()["detail"]
        assert fake_db.all("dialogue_task") == []


class TestGetDialogueTask:
    """GET /match/dialogue/task/{task_id}"""

    def test_ok_pending(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        _seed_task(fake_db)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["taskId"] == "dt_test"
        assert data["status"] == "pending"
        assert data["result"] is None
        assert data["is_question"] is None
        assert data["error"] is None

    def test_ok_success_with_result(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        result = {
            "type": "qa",
            "statement": "Hi",
            "question": "Hello?",
            "source": "matched",
            "matched_text": "Hi",
        }
        _seed_task(fake_db, status="success", result=result, is_question=False)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        assert data["result"] == result
        assert data["is_question"] is False

    def test_ok_failed_with_error(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        _seed_task(fake_db, status="failed", error="LLM 调用超时")
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["error"] == "LLM 调用超时"
        assert data["result"] is None

    def test_not_found(self, monkeypatch, fake_db):
        client = _client(monkeypatch, fake_db)
        resp = client.get("/match/dialogue/task/dt_missing")
        assert resp.status_code == 404
        assert "任务不存在" in resp.json()["detail"]

    def test_expired_treated_as_not_found(self, monkeypatch, fake_db):
        """过期任务在惰性清理时被删除 → 查询按不存在处理（404）"""
        client = _client(monkeypatch, fake_db)
        _seed_task(fake_db, expires_at=int(time.time() * 1000) - 1000)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 404
        assert fake_db.all("dialogue_task") == []

    def test_query_triggers_lazy_recovery_and_cleanup(self, monkeypatch, fake_db):
        """GET 惰性巡检：卡死 processing 置 failed，过期任务被清理"""
        client = _client(monkeypatch, fake_db)
        now = int(time.time() * 1000)
        _seed_task(fake_db, task_id="dt_ok")
        _seed_task(
            fake_db,
            task_id="dt_stale",
            status="processing",
            updated_at=now - 130_000,
            expires_at=now + TASK_TTL_MS,
        )
        _seed_task(
            fake_db,
            task_id="dt_expired",
            status="success",
            expires_at=now - 1000,
        )

        resp = client.get("/match/dialogue/task/dt_ok")
        assert resp.status_code == 200

        by_id = {d["task_id"]: d for d in fake_db.all("dialogue_task")}
        assert by_id["dt_stale"]["status"] == "failed"
        assert by_id["dt_stale"]["error"] == "执行超时"
        assert "dt_expired" not in by_id
        assert by_id["dt_ok"]["status"] == "pending"


class TestRunDialogueTask:
    """后台执行器 run_dialogue_task（mock 已学语句与 match_dialogue，不触真实火山）"""

    RESULT = {
        "type": "qa",
        "statement": "I like apples.",
        "question": "Do you like apples?",
        "source": "matched",
    }

    def _patch_worker(self, monkeypatch, fake_db, learned, match_result):
        monkeypatch.setattr("services.dialogue_task.get_db", lambda: fake_db)

        async def fake_learn(db, scholar_id):
            return learned

        async def fake_match(**kwargs):
            if isinstance(match_result, Exception):
                raise match_result
            return match_result

        monkeypatch.setattr(
            "services.dialogue_task.load_learned_sentences", fake_learn
        )
        monkeypatch.setattr("services.dialogue_task.match_dialogue", fake_match)

    def test_success_writes_result(self, monkeypatch, fake_db):
        self._patch_worker(
            monkeypatch,
            fake_db,
            learned=[{"text": "I like apples.", "translation": "我喜欢苹果。"}],
            match_result={
                "success": True,
                "data": self.RESULT,
                "is_question": False,
            },
        )
        task = _seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "success"
        assert stored["result"] == self.RESULT
        assert stored["is_question"] is False
        assert stored["error"] is None

    def test_no_learned_sentences_fails(self, monkeypatch, fake_db):
        self._patch_worker(monkeypatch, fake_db, learned=[], match_result=None)
        task = _seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"] == "该学者暂无已学语句"
        assert stored["result"] is None

    def test_match_business_failure(self, monkeypatch, fake_db):
        self._patch_worker(
            monkeypatch,
            fake_db,
            learned=[{"text": "I like apples.", "translation": "我喜欢苹果。"}],
            match_result={
                "success": False,
                "error": "模型输出无法解析",
            },
        )
        task = _seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"] == "模型输出无法解析"
        assert stored["result"] is None

    def test_match_raises_marks_failed(self, monkeypatch, fake_db):
        self._patch_worker(
            monkeypatch,
            fake_db,
            learned=[{"text": "I like apples.", "translation": "我喜欢苹果。"}],
            match_result=RuntimeError("LLM 超时"),
        )
        task = _seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert "对话匹配失败" in stored["error"]
        assert "LLM 超时" in stored["error"]
        assert stored["result"] is None

    def test_claim_failed_skips_execution(self, monkeypatch, fake_db):
        """任务已被抢占（processing）→ 执行器不调用匹配，状态保持不变"""
        calls = {"learn": 0, "match": 0}

        async def fake_learn(db, scholar_id):
            calls["learn"] += 1
            return [{"text": "x", "translation": "y"}]

        async def fake_match(**kwargs):
            calls["match"] += 1
            return {"success": True, "data": {}, "is_question": False}

        monkeypatch.setattr("services.dialogue_task.get_db", lambda: fake_db)
        monkeypatch.setattr(
            "services.dialogue_task.load_learned_sentences", fake_learn
        )
        monkeypatch.setattr("services.dialogue_task.match_dialogue", fake_match)

        task = _seed_task(fake_db, status="processing")
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        assert calls["learn"] == 0
        assert calls["match"] == 0
        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "processing"
