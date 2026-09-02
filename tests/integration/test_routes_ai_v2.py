"""集成测试：沉浸式 AI 会话 v2 异步接口（proposal 2026-09-02 / api-contract §3.12 / §4.18-4.19）

被测链路（FastAPI TestClient + FakeDB，不触真实火山）：
- POST /ai/session/v2           提交会话生成任务（mode=start / mode=turn）
  → 毫秒级返回 { task_id, status: pending, session_id }，生成由后台执行；
  → 业务失败 HTTP 200 + success=false + code（INVALID_INPUT / SESSION_NOT_FOUND /
     TURN_IN_PROGRESS / TYPE_NOT_SUPPORTED）
- GET  /ai/session/v2/task/{id} 查询任务状态（pending/processing/success/failed
  + 404 缺失/过期 + 卡死定点自愈并释放会话在途位）
- run_session_task             后台执行器（真实驱动：claim → 生成 → 回写 history/
  释放在途位 → finish；失败不污染 history）

要点（同 test_routes_eval_v2 / test_dialogue_task_routes）：
- 提交接口用 stub 替换后台执行器（services.routes_ai.run_session_task），
  避免异步时序影响断言；
- 执行器全链路用 asyncio.run 直接驱动（patch generate_session_reply 替身）。
"""
from __future__ import annotations

import asyncio
import time

from config import SESSION_LLM_TIMEOUT_SECONDS
from services.learning.session_state import get_session
from services.learning.session_task import run_session_task
from services.providers.session_gen import ERR_EVAL_UNAVAILABLE, SessionGenError
from services.routes_ai import router as ai_router
from tests.fakes.seed_factory import seed_ai_session, seed_ai_session_task

SCHOLAR = "scholar_1"


def _run(coro):
    return asyncio.run(coro)


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# 入参造数
# ---------------------------------------------------------------------------


def start_payload(**overrides) -> dict:
    """合法的 mode=start 提交（2 新句 + 1 复习句）。"""
    payload = {
        "scholar_id": SCHOLAR,
        "mode": "start",
        "scenario": {
            "scene_id": "airport",
            "title": "机场值机",
            "scene": "Learner is at the airport check-in counter.",
            "goal": "Check in for a flight",
            "constraints": "Stay in role, no grammar lectures.",
        },
        "roles": {
            "ai_role": {"name": "Airport Staff", "identity": "Clerk at counter", "style": "friendly"},
            "learner_role": {"name": "Passenger", "identity": "Traveller"},
        },
        "groups": [
            {
                "kind": "new",
                "sentences": [
                    {"sentence_id": "sid_new_1", "content": "I'd like to check in for my flight."},
                    {"sentence_id": "sid_new_2", "content": "Could I have a window seat?"},
                ],
            },
            {
                "kind": "review",
                "sentences": [
                    {"sentence_id": "sid_rev_1", "content": "May I see your passport, please?"}
                ],
            },
        ],
        "preferred_type": "auto",
        "user_input": None,
        "assisted": False,
    }
    payload.update(overrides)
    return payload


def turn_payload(session_id: str = "s_test", **overrides) -> dict:
    """合法的 mode=turn 提交（user_input + 可带 assisted）。"""
    payload = {
        "scholar_id": SCHOLAR,
        "mode": "turn",
        "session_id": session_id,
        "user_input": "I would like to check in.",
        "assisted": False,
        "preferred_type": "auto",
    }
    payload.update(overrides)
    return payload


async def _noop_run(task_id: str, **kwargs) -> None:
    """记录调度而不真正执行（确定性断言用；须为 async 以满足 create_task）。"""


class TestSubmitStart:
    """POST /ai/session/v2 — mode=start"""

    def test_ok_returns_task_session_and_creates_docs(
        self, make_client, monkeypatch, fake_db
    ):
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id
            called.update(kwargs)

        monkeypatch.setattr("services.routes_ai.run_session_task", fake_run)
        client = make_client(ai_router)

        resp = client.post("/ai/session/v2", json=start_payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["task_id"].startswith("st_")
        assert data["status"] == "pending"
        assert data["session_id"].startswith("s_")

        # 后台执行器已调度，参数正确
        assert called["task_id"] == data["task_id"]

        # 任务已落库 pending + context 自包含快照
        tasks = fake_db.all("ai_session_task")
        assert len(tasks) == 1
        task = tasks[0]
        assert task["status"] == "pending"
        assert task["mode"] == "start"
        assert task["session_id"] == data["session_id"]
        ctx = task["context"]
        assert ctx["mode"] == "start"
        assert ctx["materials"] == start_payload()["groups"]
        assert ctx["history"] == []
        assert ctx["user_input"] is None

        # 会话态已创建，start 创建即占位 pending_task=本任务
        sessions = fake_db.all("ai_session")
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == data["session_id"]
        assert sessions[0]["pending_task"] == data["task_id"]
        assert sessions[0]["history"] == []

    def test_invalid_scenario_missing_scene(self, make_client, monkeypatch):
        client = make_client(ai_router)
        payload = start_payload(scenario={"title": "no scene"})
        resp = client.post("/ai/session/v2", json=payload)
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_missing_groups(self, make_client, monkeypatch):
        client = make_client(ai_router)
        payload = start_payload(groups=None)
        resp = client.post("/ai/session/v2", json=payload)
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_unsupported_preferred_type(self, make_client, monkeypatch):
        client = make_client(ai_router)
        resp = client.post(
            "/ai/session/v2", json=start_payload(preferred_type="retell")
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TYPE_NOT_SUPPORTED"

    def test_illegal_mode(self, make_client, monkeypatch):
        client = make_client(ai_router)
        resp = client.post("/ai/session/v2", json=start_payload(mode="restart"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_empty_scholar_id(self, make_client, monkeypatch):
        client = make_client(ai_router)
        resp = client.post("/ai/session/v2", json=start_payload(scholar_id="  "))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"


class TestSubmitTurn:
    """POST /ai/session/v2 — mode=turn"""

    def test_turn_requires_session_id(self, make_client, monkeypatch, fake_db):
        client = make_client(ai_router)
        resp = client.post(
            "/ai/session/v2",
            json={"scholar_id": SCHOLAR, "mode": "turn", "user_input": "hello"},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SESSION_NOT_FOUND"

    def test_turn_session_not_found(self, make_client, monkeypatch, fake_db):
        client = make_client(ai_router)
        resp = client.post(
            "/ai/session/v2", json=turn_payload(session_id="s_missing")
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SESSION_NOT_FOUND"

    def test_turn_wrong_scholar(self, make_client, monkeypatch, fake_db):
        seed_ai_session(fake_db)
        client = make_client(ai_router)
        resp = client.post(
            "/ai/session/v2", json=turn_payload(scholar_id="scholar_other")
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "SESSION_NOT_FOUND"

    def test_turn_session_busy(self, make_client, monkeypatch, fake_db):
        seed_ai_session(fake_db, pending_task="st_other")
        client = make_client(ai_router)
        resp = client.post("/ai/session/v2", json=turn_payload())
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TURN_IN_PROGRESS"

    def test_turn_ok_claims_slot_and_snapshots_context(
        self, make_client, monkeypatch, fake_db
    ):
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id

        monkeypatch.setattr("services.routes_ai.run_session_task", fake_run)
        seed_session = seed_ai_session(fake_db)
        client = make_client(ai_router)

        resp = client.post(
            "/ai/session/v2", json=turn_payload(assisted=True)
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "pending"
        assert called["task_id"] == data["task_id"]

        # 任务 context 自包含：会话快照 + user_input/assisted
        tasks = fake_db.all("ai_session_task")
        assert len(tasks) == 1
        ctx = tasks[0]["context"]
        assert ctx["mode"] == "turn"
        assert ctx["user_input"] == "I would like to check in."
        assert ctx["assisted"] is True
        # 素材/场景/角色取自会话态快照（与种子一致）
        assert ctx["materials"] == seed_session["materials"]
        assert ctx["scenario"] == seed_session["scenario"]
        assert ctx["roles"] == seed_session["roles"]

        # 会话在途位已被抢占为本次任务
        sess = _run(get_session(fake_db, "s_test"))
        assert sess["pending_task"] == data["task_id"]


class TestQueryTask:
    """GET /ai/session/v2/task/{task_id}"""

    def test_missing_task_404(self, make_client, monkeypatch, fake_db):
        client = make_client(ai_router)
        resp = client.get("/ai/session/v2/task/st_unknown")
        assert resp.status_code == 404

    def test_pending_query_ok(self, make_client, monkeypatch, fake_db):
        seed_ai_session_task(fake_db, task_id="st_pending")
        client = make_client(ai_router)
        resp = client.get("/ai/session/v2/task/st_pending")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "pending"
        assert body["data"]["result"] is None

    def test_success_query_returns_result(self, make_client, monkeypatch, fake_db):
        result = {
            "session_id": "s_test",
            "content_type": "dialogue",
            "ai_text": "Good morning! Where are you flying today?",
            "hint": None,
            "suggested_targets": [],
        }
        seed_ai_session_task(
            fake_db,
            task_id="st_done",
            status="success",
            result=result,
        )
        client = make_client(ai_router)
        resp = client.get("/ai/session/v2/task/st_done")
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "success"
        assert body["data"]["result"]["ai_text"] == result["ai_text"]
        assert body["data"]["error"] is None

    def test_failed_query_returns_error_display(self, make_client, monkeypatch, fake_db):
        seed_ai_session_task(
            fake_db,
            task_id="st_fail",
            status="failed",
            error={
                "error_code": "LLM_PARSE_ERROR",
                "error_detail": "模型输出解析失败: {bad",
                "failure_stage": "parse",
            },
        )
        client = make_client(ai_router)
        resp = client.get("/ai/session/v2/task/st_fail")
        body = resp.json()
        assert body["data"]["status"] == "failed"
        # 接口层只展示可读 error_detail，完整 error 对象留在任务文档
        assert body["data"]["error"] == "模型输出解析失败: {bad"
        assert body["data"]["result"] is None

    def test_expired_task_404(self, make_client, monkeypatch, fake_db):
        seed_ai_session_task(
            fake_db,
            task_id="st_expired",
            status="success",
            result={},
            expires_at=_now_ms() - 1000,
        )
        client = make_client(ai_router)
        resp = client.get("/ai/session/v2/task/st_expired")
        assert resp.status_code == 404

    def test_stale_processing_recovered_and_slot_released(
        self, make_client, monkeypatch, fake_db
    ):
        task_id = "st_stale"
        seed_ai_session(fake_db, pending_task=task_id)
        seed_ai_session_task(
            fake_db,
            task_id=task_id,
            status="processing",
            updated_at=_now_ms() - (SESSION_LLM_TIMEOUT_SECONDS + 5) * 1000,
        )
        client = make_client(ai_router)
        resp = client.get(f"/ai/session/v2/task/{task_id}")
        assert resp.status_code == 200
        body = resp.json()
        # 卡死自愈：定点恢复 → failed + LLM_TIMEOUT
        assert body["data"]["status"] == "failed"
        assert body["data"]["error"] == "执行超时"
        # 同步释放会话在途位
        sess = _run(get_session(fake_db, "s_test"))
        assert sess["pending_task"] is None


class TestWorkerFlow:
    """run_session_task 后台执行器真实链路（patch generate_session_reply）。"""

    def test_start_worker_success_writes_single_ai_head(
        self, make_client, monkeypatch, fake_db
    ):
        """start 开场产出回写为 history 首条单 ai 记录（不入 user 条，§4.19 回写口径）。"""
        CANNED = {
            "content_type": "dialogue",
            "ai_text": "Welcome to the airport! May I see your passport?",
            "hint": None,
            "suggested_targets": [],
        }

        async def fake_generate(**kwargs):
            return CANNED

        monkeypatch.setattr(
            "services.providers.session_gen.generate_session_reply", fake_generate
        )
        client = make_client(ai_router)
        monkeypatch.setattr("services.routes_ai.run_session_task", _noop_run)
        resp = client.post("/ai/session/v2", json=start_payload())
        task_id = resp.json()["data"]["task_id"]
        session_id = resp.json()["data"]["session_id"]

        _run(run_session_task(task_id))

        stored = [t for t in fake_db.all("ai_session_task") if t["task_id"] == task_id][0]
        assert stored["status"] == "success"

        sess = _run(get_session(fake_db, session_id))
        assert sess["pending_task"] is None
        assert len(sess["history"]) == 1
        head = sess["history"][0]
        assert head["role"] == "ai"
        assert head["text"] == CANNED["ai_text"]
        assert head["content_type"] == "dialogue"
        assert "user" not in head
        assert sess["assisted_count"] == 0

    def test_turn_worker_success_writes_history_and_releases(
        self, make_client, monkeypatch, fake_db
    ):
        CANNED = {
            "content_type": "dialogue",
            "ai_text": "Good morning! May I see your passport?",
            "hint": {
                "levels": ["词义提示L1", "句式骨架L2", "对照引导L3"],
                "max_level": 3,
            },
            "suggested_targets": ["sid_rev_1"],
        }

        async def fake_generate(**kwargs):
            return CANNED

        monkeypatch.setattr(
            "services.providers.session_gen.generate_session_reply", fake_generate
        )
        # 会话带 1 条开场 ai 历史；turn 提交（占位）后手动驱动执行器
        seed_ai_session(
            fake_db,
            history=[
                {
                    "role": "ai",
                    "text": "Welcome to the airport!",
                    "content_type": "dialogue",
                    "created_at": _now_ms() - 1000,
                }
            ],
        )
        client = make_client(ai_router)
        monkeypatch.setattr("services.routes_ai.run_session_task", _noop_run)
        resp = client.post("/ai/session/v2", json=turn_payload(assisted=True))
        task_id = resp.json()["data"]["task_id"]

        _run(run_session_task(task_id))

        # 任务 success + result
        stored = [t for t in fake_db.all("ai_session_task") if t["task_id"] == task_id][0]
        assert stored["status"] == "success"
        assert stored["result"]["ai_text"] == CANNED["ai_text"]
        assert stored["error"] is None

        # 会话历史回写 [user(assisted), ai]，在途位已释放
        sess = _run(get_session(fake_db, "s_test"))
        assert sess["pending_task"] is None
        roles = [h["role"] for h in sess["history"]]
        assert roles == ["ai", "user", "ai"]
        user_entry = sess["history"][-2]
        assert user_entry["role"] == "user"
        assert user_entry["assisted"] is True
        assert user_entry["text"] == "I would like to check in."
        assert sess["history"][-1]["content_type"] == "dialogue"
        assert sess["assisted_count"] == 1

        # GET 查询一致
        resp = client.get(f"/ai/session/v2/task/{task_id}")
        assert resp.json()["data"]["status"] == "success"

    def test_worker_failure_no_history_and_release(
        self, make_client, monkeypatch, fake_db
    ):
        async def fake_generate(**kwargs):
            raise SessionGenError(
                ERR_EVAL_UNAVAILABLE, "llm", "LLM 调用失败（模型不可用或返回空）"
            )

        monkeypatch.setattr(
            "services.providers.session_gen.generate_session_reply", fake_generate
        )
        seed_ai_session(fake_db)
        client = make_client(ai_router)
        monkeypatch.setattr("services.routes_ai.run_session_task", _noop_run)
        resp = client.post("/ai/session/v2", json=turn_payload())
        task_id = resp.json()["data"]["task_id"]

        _run(run_session_task(task_id))

        stored = [t for t in fake_db.all("ai_session_task") if t["task_id"] == task_id][0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "EVAL_UNAVAILABLE"
        assert stored["error"]["failure_stage"] == "llm"

        # 失败不污染 history（仍为空），在途位已释放
        sess = _run(get_session(fake_db, "s_test"))
        assert sess["pending_task"] is None
        assert sess["history"] == []
        assert sess["assisted_count"] == 0

    def test_worker_generic_exception_maps_network_error(
        self, make_client, monkeypatch, fake_db
    ):
        async def fake_generate(**kwargs):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(
            "services.providers.session_gen.generate_session_reply", fake_generate
        )
        seed_ai_session(fake_db)
        client = make_client(ai_router)
        monkeypatch.setattr("services.routes_ai.run_session_task", _noop_run)
        resp = client.post("/ai/session/v2", json=turn_payload())
        task_id = resp.json()["data"]["task_id"]

        _run(run_session_task(task_id))

        stored = [t for t in fake_db.all("ai_session_task") if t["task_id"] == task_id][0]
        assert stored["status"] == "failed"
        assert stored["error"]["error_code"] == "NETWORK_ERROR"
        assert "connection reset" in stored["error"]["error_detail"]
        sess = _run(get_session(fake_db, "s_test"))
        assert sess["pending_task"] is None
