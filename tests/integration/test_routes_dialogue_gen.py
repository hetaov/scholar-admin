"""集成测试：AI 对话生成域 v1（设计稿 §5 接口 / §6 数据模型 / §8 本地调测）

被测链路（FastAPI TestClient + FakeDB，不触真实火山）：
- POST /ai/dialogue/v1/generate      提交批量生成任务（毫秒级返回，生成后台执行）
  → 业务失败 HTTP 200 + success=false + code（INVALID_INPUT / TASK_GROUP_REQUIRED /
     TYPE_NOT_SUPPORTED / DIALOGUE_GEN_DISABLED）
- GET  /ai/dialogue/v1/task/{id}     查询（pending/processing/success/failed + 404 缺失/过期
     + 卡死定点自愈）
- run_dialogue_gen_task              后台执行器（claim → 直连生成 → finish；失败置 failed 不静默）

要点（同 test_routes_ai_v2）：
- 提交接口用 stub 替换后台执行器（services.routes.dialogue_gen.run_dialogue_gen_task）；
- 执行器全链路用 asyncio.run 直接驱动（patch dialogue_gen.generate_dialogue 替身）。
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest

from config import DIALOGUE_LLM_TIMEOUT_SECONDS
from services.learning.dialogue_gen_graph import (
    get_checkpointer,
    latest_checkpoint_id,
    run_dialogue_gen_graph,
)
from services.learning.dialogue_gen_task import get_task, run_dialogue_gen_task
from services.providers.dialogue_gen import (
    ERR_LLM_UNAVAILABLE,
    STAGE_LLM,
    DialogueGenError,
)
from services.routes_dialogue_gen import router as dialogue_gen_router
from tests.fakes.seed_factory import (
    dialogue_gen_v2_flow_gen,
    seed_dialogue_gen_task,
    seed_local_corpus,
)

SCHOLAR = "scholar_1"


def _run(coro):
    return asyncio.run(coro)


def _now_ms() -> int:
    return int(time.time() * 1000)


@pytest.fixture(autouse=True)
def _enable_dialogue_gen(monkeypatch):
    """默认开启总开关（关闭场景的用例内再显式置 0）。"""
    monkeypatch.setattr("services.routes.dialogue_gen.DIALOGUE_GEN_ENABLED", 1)


# ---------------------------------------------------------------------------
# 入参造数
# ---------------------------------------------------------------------------


def payload(**overrides) -> dict:
    """合法的批量生成提交（任务组 2 句 + A/B/C 三角色）。"""
    body = {
        "scholar_id": SCHOLAR,
        "task_group": {
            "lesson_id": "l_nc2_01",
            "group_id": "g_nc2_01_a",
            "group_label": "L1 叙述组",
            "sentences": [
                {"sentence_id": "s_nc2_01_01", "content": "Last week I went to the theatre."},
                {"sentence_id": "s_nc2_01_02", "content": "I had a very good seat."},
            ],
        },
        "scenario": {"background": "两个同学在讨论上周末的活动", "goal": "自然用出任务组句子"},
        "roles": [
            {"code": "A", "name": "Tom", "identity": "student"},
            {"code": "B", "name": "Lily", "identity": "classmate"},
            {"code": "C", "name": "Teacher", "identity": "teacher"},
        ],
        "recall": {"enabled": True, "top_k": 4},
        "metrics": {"enabled": True, "weak_skills": ["past_tense"]},
        "prompt_lang": "zh",
        "preferred_type": "auto",
    }
    body.update(overrides)
    return body


async def _noop_run(task_id: str, **kwargs) -> None:
    """记录调度而不真正执行（确定性断言用；须为 async 以满足 create_task）。"""


def _graph_context() -> dict:
    """生成图上下文（关闭召回，避免触达假库/检索器）。"""
    body = payload()
    return {
        "task_group": body["task_group"],
        "scenario": body["scenario"],
        "roles": body["roles"],
        "recall": {"enabled": False, "top_k": 4},
        "metrics": body["metrics"],
        "prompt_lang": body["prompt_lang"],
        "preferred_type": body["preferred_type"],
    }


def _dialogue_json() -> str:
    """覆盖任务组 2 句的合法对话输出（T4 续写造数用）。"""
    return json.dumps(
        {
            "content_type": "dialogue",
            "sub_type": None,
            "background_intro": "背景",
            "turns": [
                {
                    "speaker": "A",
                    "text": "Last week I went to the theatre.",
                    "target_sentence_id": "s_nc2_01_01",
                },
                {
                    "speaker": "B",
                    "text": "I had a very good seat.",
                    "target_sentence_id": "s_nc2_01_02",
                },
            ],
            "prompts": [],
            "used_sentence_ids": ["s_nc2_01_01", "s_nc2_01_02"],
            "recalled_sentence_ids": [],
            "difficulty": 2,
            "notes": None,
        },
        ensure_ascii=False,
    )


async def _seed_graph_checkpoints(fake_db, task_id: str) -> None:
    """真实跑一次生成图，向 ai_dialogue_checkpoint 落 checkpoint（路由断言用）。"""

    async def gen(messages):
        return _dialogue_json()

    gen = dialogue_gen_v2_flow_gen(gen)

    await run_dialogue_gen_graph(
        db=fake_db,
        task_id=task_id,
        scholar_id=SCHOLAR,
        context=_graph_context(),
        generator=gen,
        checkpointer=get_checkpointer(fake_db),
    )


class TestSubmit:
    """POST /ai/dialogue/v1/generate"""

    def test_ok_returns_pending_and_creates_doc(
        self, make_client, monkeypatch, fake_db
    ):
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id

        monkeypatch.setattr(
            "services.routes.dialogue_gen.run_dialogue_gen_task", fake_run
        )
        client = make_client(dialogue_gen_router)

        resp = client.post("/ai/dialogue/v1/generate", json=payload())
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["task_id"].startswith("dg_")
        assert data["status"] == "pending"
        assert data["result"] is None
        assert data["error"] is None
        assert data["resumable"] is False
        assert called["task_id"] == data["task_id"]

        tasks = fake_db.all("ai_dialogue_task")
        assert len(tasks) == 1
        task = tasks[0]
        assert task["status"] == "pending"
        assert task["retry_count"] == 0
        assert task["checkpoint_id"] is None
        ctx = task["context"]
        assert ctx["task_group"] == payload()["task_group"]
        assert ctx["preferred_type"] == "auto"
        assert ctx["prompt_lang"] == "zh"
        # 不写任何 ai_session* 集合（两面隔离，§8.4 第 16 条）
        assert fake_db.all("ai_session") == []
        assert fake_db.all("ai_session_task") == []

    def test_disabled_gate(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr("services.routes.dialogue_gen.DIALOGUE_GEN_ENABLED", 0)
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload())
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "DIALOGUE_GEN_DISABLED"
        assert fake_db.all("ai_dialogue_task") == []

    def test_missing_task_group(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload(task_group=None))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TASK_GROUP_REQUIRED"

    def test_empty_sentences(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        tg = payload()["task_group"]
        tg = {**tg, "sentences": []}
        resp = client.post("/ai/dialogue/v1/generate", json=payload(task_group=tg))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TASK_GROUP_REQUIRED"

    def test_sentence_content_too_long(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        tg = payload()["task_group"]
        tg = {**tg, "sentences": [{"sentence_id": "s1", "content": "x" * 501}]}
        resp = client.post("/ai/dialogue/v1/generate", json=payload(task_group=tg))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_unsupported_preferred_type(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload(preferred_type="retell"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TYPE_NOT_SUPPORTED"

    def test_illegal_preferred_type(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload(preferred_type="xxx"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_empty_scholar_id(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload(scholar_id="  "))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_illegal_prompt_lang(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/generate", json=payload(prompt_lang="fr"))
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_recall_top_k_clamped_into_context(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr("services.routes.dialogue_gen.run_dialogue_gen_task", _noop_run)
        client = make_client(dialogue_gen_router)
        resp = client.post(
            "/ai/dialogue/v1/generate", json=payload(recall={"enabled": True, "top_k": 99})
        )
        assert resp.json()["success"] is True
        ctx = fake_db.all("ai_dialogue_task")[0]["context"]
        assert ctx["recall"]["top_k"] == 6  # clamp 到 [2,6]


class TestSubmitLegacyKeysRejected:
    """v2：场景候选机制已移除，generation / test_scenario 传参应报 INVALID_INPUT。"""

    def test_generation_key_rejected(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post(
            "/ai/dialogue/v1/generate",
            json=payload(generation={"scene_mode": "auto", "candidate_count": 3}),
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"
        assert "generation" in body["message"]
        assert fake_db.all("ai_dialogue_task") == []

    def test_test_scenario_key_rejected(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post(
            "/ai/dialogue/v1/generate",
            json=payload(
                test_scenario={
                    "enabled": True,
                    "category": "shopping",
                    "description": "周末在商场买礼物",
                }
            ),
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"
        assert "test_scenario" in body["message"]
        assert fake_db.all("ai_dialogue_task") == []

    def test_request_without_legacy_keys_has_no_scene_keys_in_context(
        self, make_client, monkeypatch, fake_db
    ):
        monkeypatch.setattr("services.routes.dialogue_gen.run_dialogue_gen_task", _noop_run)
        client = make_client(dialogue_gen_router)
        assert client.post("/ai/dialogue/v1/generate", json=payload()).json()["success"] is True
        ctx = fake_db.all("ai_dialogue_task")[0]["context"]
        for removed in ("generation", "test_scenario"):
            assert removed not in ctx


class TestQuery:
    """GET /ai/dialogue/v1/task/{task_id}"""

    def test_missing_task_404(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        assert client.get("/ai/dialogue/v1/task/dg_unknown").status_code == 404

    def test_pending_query_ok(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_pending")
        client = make_client(dialogue_gen_router)
        body = client.get("/ai/dialogue/v1/task/dg_pending").json()
        assert body["success"] is True
        assert body["data"]["status"] == "pending"
        assert body["data"]["result"] is None

    def test_success_query_returns_result(self, make_client, monkeypatch, fake_db):
        result = {
            "session_id": "s_1",
            "content_type": "dialogue",
            "turns": [{"speaker": "A", "text": "hi", "target_sentence_id": None}],
            "coverage": {"required_total": 1, "required_used": 1, "ratio": 1.0},
        }
        seed_dialogue_gen_task(fake_db, task_id="dg_done", status="success", result=result)
        client = make_client(dialogue_gen_router)
        body = client.get("/ai/dialogue/v1/task/dg_done").json()
        assert body["data"]["status"] == "success"
        assert body["data"]["result"]["session_id"] == "s_1"
        assert body["data"]["error"] is None

    def test_failed_query_returns_error_detail(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_fail",
            status="failed",
            error={
                "error_code": "LLM_PARSE_ERROR",
                "error_detail": "模型输出解析失败: {bad",
                "failure_stage": "parse",
            },
        )
        client = make_client(dialogue_gen_router)
        body = client.get("/ai/dialogue/v1/task/dg_fail").json()
        assert body["data"]["status"] == "failed"
        assert body["data"]["error"] == "模型输出解析失败: {bad"
        assert body["data"]["result"] is None

    def test_expired_task_404(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(
            fake_db, task_id="dg_expired", status="success", result={},
            expires_at=_now_ms() - 1000,
        )
        client = make_client(dialogue_gen_router)
        assert client.get("/ai/dialogue/v1/task/dg_expired").status_code == 404

    def test_stale_processing_recovered(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_stale",
            status="processing",
            updated_at=_now_ms() - (DIALOGUE_LLM_TIMEOUT_SECONDS + 10) * 1000,
        )
        client = make_client(dialogue_gen_router)
        body = client.get("/ai/dialogue/v1/task/dg_stale").json()
        assert body["data"]["status"] == "failed"
        assert body["data"]["error"] == "执行超时"


class TestRunTask:
    """run_dialogue_gen_task — 后台执行器（真实驱动）"""

    def test_success_writes_result(self, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ok")
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)

        async def fake_gen(**kwargs):
            assert kwargs["preferred_type"] == "auto"
            assert kwargs["context"]["task_group"]["group_id"] == "g_nc2_01_a"
            return {"session_id": "s_ok", "content_type": "dialogue"}

        monkeypatch.setattr("services.providers.dialogue_gen.generate_dialogue", fake_gen)
        _run(run_dialogue_gen_task("dg_ok"))

        task = _run(get_task(fake_db, "dg_ok"))
        assert task["status"] == "success"
        assert task["result"]["session_id"] == "s_ok"
        assert task["error"] is None

    def test_business_error_sets_failed(self, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_bad")
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)

        async def failing_gen(**kwargs):
            raise DialogueGenError(ERR_LLM_UNAVAILABLE, STAGE_LLM, "模型不可用")

        monkeypatch.setattr("services.providers.dialogue_gen.generate_dialogue", failing_gen)
        _run(run_dialogue_gen_task("dg_bad"))

        task = _run(get_task(fake_db, "dg_bad"))
        assert task["status"] == "failed"
        assert task["error"]["error_code"] == "LLM_UNAVAILABLE"
        assert task["result"] is None

    def test_unexpected_error_maps_to_network_error(self, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_boom")
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)

        async def boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("services.providers.dialogue_gen.generate_dialogue", boom)
        _run(run_dialogue_gen_task("dg_boom"))

        task = _run(get_task(fake_db, "dg_boom"))
        assert task["status"] == "failed"
        assert task["error"]["error_code"] == "NETWORK_ERROR"

    def test_graph_enabled_delegates_to_graph(self, monkeypatch, fake_db):
        """DIALOGUE_GEN_GRAPH_ENABLED=1 → 执行器改走 T3 生成图（传入任务快照）。"""
        seed_dialogue_gen_task(fake_db, task_id="dg_graph")
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)
        monkeypatch.setattr(
            "services.learning.dialogue_gen_task.DIALOGUE_GEN_GRAPH_ENABLED", True
        )
        captured = {}

        async def fake_graph(**kwargs):
            captured.update(kwargs)
            return {
                "session_id": "s_graph",
                "content_type": "non_dialogue",
                "coverage": {"required_total": 1, "required_used": 1, "ratio": 1.0},
                "retry_count": 2,
                "notes": "fallback_non_dialogue",
            }

        monkeypatch.setattr(
            "services.learning.dialogue_gen_graph.run_dialogue_gen_graph", fake_graph
        )
        _run(run_dialogue_gen_task("dg_graph"))

        task = _run(get_task(fake_db, "dg_graph"))
        assert task["status"] == "success"
        assert task["result"]["session_id"] == "s_graph"
        assert task["result"]["notes"] == "fallback_non_dialogue"
        assert captured["task_id"] == "dg_graph"
        assert captured["scholar_id"] == "scholar_1"
        assert captured["db"] is fake_db
        assert captured["preferred_type"] == "auto"

    def test_graph_disabled_uses_direct_path(self, monkeypatch, fake_db):
        """DIALOGUE_GEN_GRAPH_ENABLED=0 → 仍走 T1 单函数直连（逐断言兼容）。"""
        seed_dialogue_gen_task(fake_db, task_id="dg_direct")
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)
        monkeypatch.setattr(
            "services.learning.dialogue_gen_task.DIALOGUE_GEN_GRAPH_ENABLED", False
        )

        async def fake_gen(**kwargs):
            return {"session_id": "s_direct", "content_type": "dialogue"}

        async def graph_should_not_run(**kwargs):  # pragma: no cover
            raise AssertionError("关闭图开关时不应走图")

        monkeypatch.setattr("services.providers.dialogue_gen.generate_dialogue", fake_gen)
        monkeypatch.setattr(
            "services.learning.dialogue_gen_graph.run_dialogue_gen_graph",
            graph_should_not_run,
        )
        _run(run_dialogue_gen_task("dg_direct"))

        task = _run(get_task(fake_db, "dg_direct"))
        assert task["status"] == "success"
        assert task["result"]["session_id"] == "s_direct"

    def test_claim_skip_when_not_pending(self, monkeypatch, fake_db):
        """已完成任务再跑不会重复执行/覆盖结果。"""
        seed_dialogue_gen_task(
            fake_db, task_id="dg_done2", status="success", result={"session_id": "s_keep"}
        )
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)

        async def should_not_run(**kwargs):  # pragma: no cover
            raise AssertionError("不应执行")

        monkeypatch.setattr("services.providers.dialogue_gen.generate_dialogue", should_not_run)
        _run(run_dialogue_gen_task("dg_done2"))

        task = _run(get_task(fake_db, "dg_done2"))
        assert task["status"] == "success"
        assert task["result"]["session_id"] == "s_keep"

    def test_resume_claims_failed_and_refreshes_cursor(self, monkeypatch, fake_db):
        """T4：resume=True → 原子抢占 failed（清 error）→ 续跑 → 回写新 checkpoint_id。"""
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_resume_run",
            status="failed",
            checkpoint_id="cp_old",
            error={"error_code": "LLM_UNAVAILABLE", "error_detail": "旧错误"},
        )
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)
        monkeypatch.setattr(
            "services.learning.dialogue_gen_task.DIALOGUE_GEN_GRAPH_ENABLED", True
        )
        captured = {}

        async def fake_graph(**kwargs):
            captured.update(kwargs)
            return {
                "session_id": "s_resume",
                "content_type": "dialogue",
                "retry_count": 1,
                "checkpoint": {"checkpoint_id": "cp_new"},
            }

        monkeypatch.setattr(
            "services.learning.dialogue_gen_graph.run_dialogue_gen_graph", fake_graph
        )
        _run(
            run_dialogue_gen_task(
                "dg_resume_run", resume=True, from_checkpoint_id="cp_old"
            )
        )

        task = _run(get_task(fake_db, "dg_resume_run"))
        assert task["status"] == "success"
        assert task["error"] is None
        assert task["checkpoint_id"] == "cp_new"
        assert task["retry_count"] == 1
        assert captured["resume"] is True
        assert captured["from_checkpoint_id"] == "cp_old"

    def test_resume_skips_when_not_failed(self, monkeypatch, fake_db):
        """resume=True 但任务非 failed → 不抢占、不执行（并发安全）。"""
        seed_dialogue_gen_task(fake_db, task_id="dg_r_skip", status="success", result={})
        monkeypatch.setattr("services.learning.dialogue_gen_task.get_db", lambda: fake_db)

        async def should_not_run(**kwargs):  # pragma: no cover
            raise AssertionError("非 failed 任务不应续跑")

        monkeypatch.setattr(
            "services.learning.dialogue_gen_graph.run_dialogue_gen_graph", should_not_run
        )
        _run(run_dialogue_gen_task("dg_r_skip", resume=True))

        task = _run(get_task(fake_db, "dg_r_skip"))
        assert task["status"] == "success"


class TestCorpus:
    """GET /ai/dialogue/v1/corpus（T2 本地语料读取，免 db、不触网）"""

    def test_ok_returns_task_groups_and_learner(self, make_client, monkeypatch, tmp_path):
        seeded = seed_local_corpus(tmp_path, lessons=(1, 2))
        monkeypatch.setattr(
            "services.routes.dialogue_gen.corpus_path", lambda: seeded["corpus_path"]
        )
        client = make_client(dialogue_gen_router)

        resp = client.get("/ai/dialogue/v1/corpus")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["corpus_available"] is True
        assert data["book"]["textbook_id"] == "tb_nc2"
        assert len(data["lessons"]) == 2
        assert len(data["task_groups"]) == 2

        first_group = data["task_groups"][0]
        assert first_group["lesson_id"] == "l_nc2_01"
        assert first_group["group_id"] == "g_nc2_01_a"
        assert first_group["sentences"][0] == {
            "sentence_id": "s_nc2_01_01",
            "content": "Last week I went to the theatre.",
        }
        assert len(first_group["sentences"]) == 2
        # 指标卡真实 metrics：来自本地 learner 文件（weak = mastery < 0.6）
        assert data["scholar_id"] == "scholar_debug_01"
        assert data["learner"]["available"] is True
        assert data["learner"]["skill_state_count"] == 1
        assert data["learner"]["weak_skills"] == ["l_nc2_01"]

    def test_missing_corpus_degrades_without_error(self, make_client, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "services.routes.dialogue_gen.corpus_path",
            lambda: tmp_path / "nope" / "corpus.json",
        )
        client = make_client(dialogue_gen_router)

        resp = client.get("/ai/dialogue/v1/corpus")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["corpus_available"] is False
        assert body["data"]["task_groups"] == []
        assert body["data"]["lessons"] == []
        assert body["data"]["learner"]["available"] is False

    def test_scholar_id_param_selects_learner(self, make_client, monkeypatch, tmp_path):
        seeded = seed_local_corpus(tmp_path, lessons=(1,), scholar_id="scholar_other")
        monkeypatch.setattr(
            "services.routes.dialogue_gen.corpus_path", lambda: seeded["corpus_path"]
        )
        client = make_client(dialogue_gen_router)

        resp = client.get("/ai/dialogue/v1/corpus", params={"scholar_id": "scholar_other"})
        assert resp.json()["data"]["learner"]["available"] is True

        resp2 = client.get("/ai/dialogue/v1/corpus", params={"scholar_id": "scholar_missing"})
        assert resp2.json()["data"]["learner"]["available"] is False

    def test_disabled_returns_business_code(self, make_client, monkeypatch):
        monkeypatch.setattr("services.routes.dialogue_gen.DIALOGUE_GEN_ENABLED", 0)
        client = make_client(dialogue_gen_router)

        resp = client.get("/ai/dialogue/v1/corpus")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "DIALOGUE_GEN_DISABLED"


class TestCheckpoints:
    """GET /ai/dialogue/v1/task/{task_id}/checkpoints（T4 checkpoint 轨迹）"""

    def test_ok_returns_timeline(self, make_client, monkeypatch, fake_db):
        _run(_seed_graph_checkpoints(fake_db, "dg_tl"))
        seed_dialogue_gen_task(fake_db, task_id="dg_tl", status="failed")
        client = make_client(dialogue_gen_router)

        resp = client.get("/ai/dialogue/v1/task/dg_tl/checkpoints")
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        cps = body["data"]["checkpoints"]
        assert len(cps) >= 5
        assert set(cps[-1]) == {"checkpoint_id", "stage", "retry_count", "ts"}
        assert cps[0]["stage"] == "start"
        assert cps[-1]["stage"] == "persisted"
        assert [c["ts"] for c in cps] == sorted(c["ts"] for c in cps)

    def test_missing_task_404(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.get("/ai/dialogue/v1/task/dg_unknown/checkpoints")
        assert resp.status_code == 404

    def test_expired_task_404(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(
            fake_db, task_id="dg_tl_exp", status="failed", expires_at=_now_ms() - 1000
        )
        client = make_client(dialogue_gen_router)
        resp = client.get("/ai/dialogue/v1/task/dg_tl_exp/checkpoints")
        assert resp.status_code == 404

    def test_no_checkpoints_returns_empty(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_tl_empty", status="failed")
        client = make_client(dialogue_gen_router)
        body = client.get("/ai/dialogue/v1/task/dg_tl_empty/checkpoints").json()
        assert body["success"] is True
        assert body["data"]["checkpoints"] == []


class TestResume:
    """POST /ai/dialogue/v1/task/{task_id}/resume（T4 断点续写）"""

    def test_resume_schedules_with_latest_checkpoint(
        self, make_client, monkeypatch, fake_db
    ):
        _run(_seed_graph_checkpoints(fake_db, "dg_res"))
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_res",
            status="failed",
            checkpoint_id="cp_stale",
            error={"error_code": "LLM_UNAVAILABLE", "error_detail": "旧错误"},
        )
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id
            called.update(kwargs)

        monkeypatch.setattr(
            "services.routes.dialogue_gen.run_dialogue_gen_task", fake_run
        )
        client = make_client(dialogue_gen_router)

        resp = client.post("/ai/dialogue/v1/task/dg_res/resume", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["task_id"] == "dg_res"
        assert body["data"]["status"] == "processing"
        assert body["data"]["resumable"] is False
        assert called["task_id"] == "dg_res"
        assert called["resume"] is True
        assert called["from_checkpoint_id"] is None  # 缺省取最新 checkpoint

    def test_resume_with_explicit_checkpoint_id(self, make_client, monkeypatch, fake_db):
        _run(_seed_graph_checkpoints(fake_db, "dg_res2"))
        seed_dialogue_gen_task(fake_db, task_id="dg_res2", status="failed")
        cp_id = _run(latest_checkpoint_id(fake_db, "dg_res2"))
        called = {}

        async def fake_run(task_id, **kwargs):
            called.update(kwargs)

        monkeypatch.setattr(
            "services.routes.dialogue_gen.run_dialogue_gen_task", fake_run
        )
        client = make_client(dialogue_gen_router)

        resp = client.post(
            "/ai/dialogue/v1/task/dg_res2/resume",
            json={"from_checkpoint_id": cp_id},
        )
        assert resp.json()["success"] is True
        assert called["from_checkpoint_id"] == cp_id

    def test_success_task_not_resumable(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_s_done", status="success", result={})
        client = make_client(dialogue_gen_router)

        resp = client.post("/ai/dialogue/v1/task/dg_s_done/resume", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "TASK_NOT_RESUMABLE"

    def test_failed_without_checkpoint_not_found(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_nocp", status="failed")
        client = make_client(dialogue_gen_router)

        resp = client.post("/ai/dialogue/v1/task/dg_nocp/resume", json={})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "CHECKPOINT_NOT_FOUND"

    def test_explicit_missing_checkpoint_not_found(self, make_client, monkeypatch, fake_db):
        _run(_seed_graph_checkpoints(fake_db, "dg_cp_ok"))
        seed_dialogue_gen_task(fake_db, task_id="dg_cp_ok", status="failed")
        client = make_client(dialogue_gen_router)

        resp = client.post(
            "/ai/dialogue/v1/task/dg_cp_ok/resume",
            json={"from_checkpoint_id": "cp_does_not_exist"},
        )
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "CHECKPOINT_NOT_FOUND"

    def test_missing_task_404(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/task/dg_unknown/resume", json={})
        assert resp.status_code == 404

    def test_stale_processing_recovered_then_resumable(
        self, make_client, monkeypatch, fake_db
    ):
        _run(_seed_graph_checkpoints(fake_db, "dg_stale_res"))
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_stale_res",
            status="processing",
            updated_at=_now_ms() - (DIALOGUE_LLM_TIMEOUT_SECONDS + 10) * 1000,
        )
        called = {}

        async def fake_run(task_id, **kwargs):
            called["task_id"] = task_id

        monkeypatch.setattr(
            "services.routes.dialogue_gen.run_dialogue_gen_task", fake_run
        )
        client = make_client(dialogue_gen_router)

        resp = client.post("/ai/dialogue/v1/task/dg_stale_res/resume", json={})
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["status"] == "processing"
        assert called["task_id"] == "dg_stale_res"

    def test_disabled_gate(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr("services.routes.dialogue_gen.DIALOGUE_GEN_ENABLED", 0)
        client = make_client(dialogue_gen_router)
        resp = client.post("/ai/dialogue/v1/task/dg_any/resume", json={})
        body = resp.json()
        assert body["success"] is False
        assert body["code"] == "DIALOGUE_GEN_DISABLED"


class TestUserInput:
    """POST /ai/dialogue/v1/task/{task_id}/user-input（T5 用户作答评测）"""

    def test_ok_evaluates_and_writes_back(self, make_client, monkeypatch, fake_db):
        """参考句取自结果最后一个目标句；评测回显并回写任务 user_inputs（零侵入）。"""
        seed_dialogue_gen_task(
            fake_db,
            task_id="dg_ui",
            status="success",
            result={
                "turns": [
                    {"speaker": "A", "text": "hi", "target_sentence_id": None},
                    {
                        "speaker": "B",
                        "text": "Last week I went to the theatre.",
                        "target_sentence_id": "s_nc2_01_01",
                    },
                ]
            },
        )
        client = make_client(dialogue_gen_router)

        resp = client.post(
            "/ai/dialogue/v1/task/dg_ui/user-input",
            json={"text": "Last week I went to the theatre."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        data = body["data"]
        assert data["target_sentence_id"] == "s_nc2_01_01"
        assert data["reference"] == "Last week I went to the theatre."
        assert data["score"] == 100  # 与参考完全一致 → L1 满分（集成测试屏蔽 Judge）
        assert data["meaningful"] is True
        assert data["anomaly"] is False
        assert data["level"] == "l1"
        assert data["judge_model"] is None
        assert data["low_confidence"] is False
        assert data["written_back"] is True

        task = fake_db.all("ai_dialogue_task")[0]
        assert len(task["user_inputs"]) == 1
        assert task["user_inputs"][0]["score"] == 100
        # 零侵入：不写 skill_state / evaluation 证据（§9-7）
        assert fake_db.all("skill_state") == []
        assert fake_db.all("evaluation") == []

    def test_explicit_target_resolves_reference(self, make_client, monkeypatch, fake_db):
        context = {
            "task_group": {
                "sentences": [
                    {"sentence_id": "s1", "content": "Last week I went to the theatre."},
                    {"sentence_id": "s2", "content": "I had a very good seat."},
                ]
            }
        }
        seed_dialogue_gen_task(fake_db, task_id="dg_ui2", context=context)
        client = make_client(dialogue_gen_router)

        body = client.post(
            "/ai/dialogue/v1/task/dg_ui2/user-input",
            json={"text": "I had a very good seat.", "target_sentence_id": "s2"},
        ).json()
        assert body["success"] is True
        assert body["data"]["target_sentence_id"] == "s2"
        assert body["data"]["reference"] == "I had a very good seat."
        assert body["data"]["score"] == 100

    def test_explicit_target_not_in_group_invalid(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ui3")
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_ui3/user-input",
            json={"text": "hello", "target_sentence_id": "s_fake"},
        ).json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_no_sentences_returns_task_group_required(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(
            fake_db, task_id="dg_ui_no_tg", context={"task_group": {"sentences": []}}
        )
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_ui_no_tg/user-input", json={"text": "hello"}
        ).json()
        assert body["success"] is False
        assert body["code"] == "TASK_GROUP_REQUIRED"

    def test_empty_text_invalid(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ui_empty")
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_ui_empty/user-input", json={"text": "   "}
        ).json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"
        assert not fake_db.all("ai_dialogue_task")[0].get("user_inputs")

    def test_text_too_long_invalid(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ui_long")
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_ui_long/user-input", json={"text": "x" * 2001}
        ).json()
        assert body["success"] is False
        assert body["code"] == "INVALID_INPUT"

    def test_low_confidence_flagged(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ui_low")
        monkeypatch.setattr(
            "services.routes.dialogue_gen.evaluate_text",
            lambda original, response: {
                "score": 40,
                "meaningful": False,
                "faithfulness": False,
                "anomaly": False,
                "confidence": 0.5,
                "level": "l1",
                "judge_model": None,
            },
        )
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_ui_low/user-input", json={"text": "blah"}
        ).json()
        assert body["success"] is True
        assert body["data"]["low_confidence"] is True
        assert body["data"]["confidence"] == 0.5

    def test_missing_task_404(self, make_client, monkeypatch, fake_db):
        client = make_client(dialogue_gen_router)
        resp = client.post(
            "/ai/dialogue/v1/task/dg_unknown/user-input", json={"text": "hello"}
        )
        assert resp.status_code == 404

    def test_expired_task_404(self, make_client, monkeypatch, fake_db):
        seed_dialogue_gen_task(fake_db, task_id="dg_ui_exp", expires_at=_now_ms() - 1000)
        client = make_client(dialogue_gen_router)
        resp = client.post(
            "/ai/dialogue/v1/task/dg_ui_exp/user-input", json={"text": "hello"}
        )
        assert resp.status_code == 404

    def test_disabled_gate(self, make_client, monkeypatch, fake_db):
        monkeypatch.setattr("services.routes.dialogue_gen.DIALOGUE_GEN_ENABLED", 0)
        client = make_client(dialogue_gen_router)
        body = client.post(
            "/ai/dialogue/v1/task/dg_any/user-input", json={"text": "hello"}
        ).json()
        assert body["success"] is False
        assert body["code"] == "DIALOGUE_GEN_DISABLED"
