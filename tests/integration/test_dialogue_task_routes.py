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

from services.dialogue_task import run_dialogue_task
from services.routes_dialogue import router as dialogue_router
from tests.fakes.seed_factory import seed_task

TASK_TTL_MS = 24 * 60 * 60 * 1000


def _run(coro):
    import asyncio

    return asyncio.run(coro)


class TestCreateDialogueTask:
    """POST /match/dialogue/task"""

    def test_ok_returns_task_id_pending(self, make_client, monkeypatch, fake_db):
        called = {}

        async def fake_run(
            task_id, scholar_id, sentence, scenario=None, session_id=None
        ):
            called["task_id"] = task_id
            called["scholar_id"] = scholar_id
            called["sentence"] = sentence
            called["scenario"] = scenario
            called["session_id"] = session_id

        monkeypatch.setattr(
            "services.routes_dialogue.run_dialogue_task", fake_run
        )
        client = make_client(dialogue_router)

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

    def test_missing_scholar_id(self, make_client, fake_db):
        client = make_client(dialogue_router)
        resp = client.post("/match/dialogue/task", json={"sentence": "Hello"})
        assert resp.status_code == 400
        assert "scholarId" in resp.json()["detail"]
        assert fake_db.all("dialogue_task") == []

    def test_missing_sentence(self, make_client, fake_db):
        client = make_client(dialogue_router)
        resp = client.post("/match/dialogue/task", json={"scholarId": "s1"})
        assert resp.status_code == 400
        assert "sentence" in resp.json()["detail"]
        assert fake_db.all("dialogue_task") == []


class TestGetDialogueTask:
    """GET /match/dialogue/task/{task_id}"""

    def test_ok_pending(self, make_client, fake_db):
        client = make_client(dialogue_router)
        seed_task(fake_db)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["taskId"] == "dt_test"
        assert data["status"] == "pending"
        assert data["result"] is None
        assert data["is_question"] is None
        assert data["error"] is None

    def test_ok_success_with_result(self, make_client, fake_db):
        client = make_client(dialogue_router)
        result = {
            "type": "qa",
            "statement": "Hi",
            "question": "Hello?",
            "source": "matched",
            "matched_text": "Hi",
        }
        seed_task(fake_db, status="success", result=result, is_question=False)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "success"
        assert data["result"] == result
        assert data["is_question"] is False

    def test_ok_failed_with_error(self, make_client, fake_db):
        client = make_client(dialogue_router)
        seed_task(fake_db, status="failed", error="LLM 调用超时")
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["error"] == "LLM 调用超时"
        assert data["result"] is None

    def test_not_found(self, make_client, fake_db):
        client = make_client(dialogue_router)
        resp = client.get("/match/dialogue/task/dt_missing")
        assert resp.status_code == 404
        assert "任务不存在" in resp.json()["detail"]

    def test_expired_returns_404_doc_kept(self, make_client, fake_db):
        """过期任务查询按不存在处理（404）；物理清理已移出查询热路径，文档保留"""
        client = make_client(dialogue_router)
        seed_task(fake_db, expires_at=int(time.time() * 1000) - 1000)
        resp = client.get("/match/dialogue/task/dt_test")
        assert resp.status_code == 404
        assert "已过期" in resp.json()["detail"]
        # TTL 清理由提交接口概率巡检执行，查询不再全集合 delete
        assert len(fake_db.all("dialogue_task")) == 1

    def test_query_revives_stale_processing_task(self, make_client, fake_db):
        """GET 定点自愈：被查询的卡死 processing 任务 → failed"""
        client = make_client(dialogue_router)
        now = int(time.time() * 1000)
        seed_task(
            fake_db,
            task_id="dt_stale",
            status="processing",
            updated_at=now - 130_000,
            expires_at=now + TASK_TTL_MS,
        )
        resp = client.get("/match/dialogue/task/dt_stale")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "failed"
        assert data["error"] == "执行超时"
        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"] == "执行超时"

    def test_query_fresh_processing_unaffected(self, make_client, fake_db):
        """查询热路径只做定点恢复：其他卡死任务不被误伤，正常任务不受影响"""
        client = make_client(dialogue_router)
        now = int(time.time() * 1000)
        seed_task(fake_db, task_id="dt_ok")
        seed_task(
            fake_db,
            task_id="dt_stale",
            status="processing",
            updated_at=now - 130_000,
            expires_at=now + TASK_TTL_MS,
        )
        resp = client.get("/match/dialogue/task/dt_ok")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "pending"

        by_id = {d["task_id"]: d for d in fake_db.all("dialogue_task")}
        assert by_id["dt_stale"]["status"] == "processing"
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

    def test_success_writes_result(self, make_client, monkeypatch, fake_db):
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
        task = seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "success"
        assert stored["result"] == self.RESULT
        assert stored["is_question"] is False
        assert stored["error"] is None

    def test_no_learned_sentences_fails(self, make_client, monkeypatch, fake_db):
        self._patch_worker(monkeypatch, fake_db, learned=[], match_result=None)
        task = seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"] == "该学者暂无已学语句"
        assert stored["result"] is None

    def test_match_business_failure(self, make_client, monkeypatch, fake_db):
        self._patch_worker(
            monkeypatch,
            fake_db,
            learned=[{"text": "I like apples.", "translation": "我喜欢苹果。"}],
            match_result={
                "success": False,
                "error": "模型输出无法解析",
            },
        )
        task = seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert stored["error"] == "模型输出无法解析"
        assert stored["result"] is None

    def test_match_raises_marks_failed(self, make_client, monkeypatch, fake_db):
        self._patch_worker(
            monkeypatch,
            fake_db,
            learned=[{"text": "I like apples.", "translation": "我喜欢苹果。"}],
            match_result=RuntimeError("LLM 超时"),
        )
        task = seed_task(fake_db)
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "failed"
        assert "对话匹配失败" in stored["error"]
        assert "LLM 超时" in stored["error"]
        assert stored["result"] is None

    def test_claim_failed_skips_execution(self, make_client, monkeypatch, fake_db):
        """任务已被抢占（processing）→ 执行器不调用匹配，状态保持不变"""
        calls = {"learn": 0, "match": 0}

        async def fake_learn(db, scholar_id):
            calls["learn"] += 1
            return [{"text": "x", "translation": "y"}]

        async def fake_match(**kwargs):
            calls["match"] += 1
            return {"success": True, "data": {}, "is_question": False}

        monkeypatch.setattr(
            "services.dialogue_task.load_learned_sentences", fake_learn
        )
        monkeypatch.setattr("services.dialogue_task.match_dialogue", fake_match)

        task = seed_task(fake_db, status="processing")
        _run(run_dialogue_task(task["task_id"], "s1", "I like apples."))

        assert calls["learn"] == 0
        assert calls["match"] == 0
        stored = fake_db.all("dialogue_task")[0]
        assert stored["status"] == "processing"
