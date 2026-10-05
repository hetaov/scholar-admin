"""集成测试：语言点造句「多轮」E7 / E8 / E9 + E3 轮询全链路

契约：api-contract.md §3.18（E7 / E8 / E9）
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §2（状态机）/ §3.2（端点）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B10）

与 `tests/unit/test_extension_round*.py` 的分工：
- unit：直接调 `run_round_start` / `run_round_turn`，覆盖纯函数与红线；
- 本文件：**经 HTTP 层**（TestClient + E3 轮询）驱动状态机 4 路径，验证路由→任务→
  执行器→落库→轮询的端到端形态。

策略（仿 e2e/test_translation_flow_v2.py）：把路由的 `run_extension_task` 替换为
「记录型 stub」，HTTP 请求返回 pending 后，**同步驱动真实执行器** `run_extension_task`，
再经 E3 轮询取结果。LLM（出题 / 判分）打桩，不触网（conftest no_external_calls 已兜底）。
"""
from __future__ import annotations

import asyncio
import json

import pytest

import config
from services.english import extension_round as er
from services.learning.extension_task import run_extension_task
from services.providers.extension_llm import ExtensionError
from services.routes.extension import router as ext_router
from tests.fakes.fake_db import FakeDB

# 造数与桩复用红线文件（同一套 fixtures，避免两套数据各写一遍）
from tests.unit.test_extension_round import (  # noqa: E402
    ANSWER_OK,
    ORIGINAL,
    POINTS,
    PROMPTS,
    SCHOLAR,
    SENTENCE,
    TRANSLATION,
    _RUBRIC_FAIL,
    _RUBRIC_PASS,
)


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 桩与辅助
# ===========================================================================


def _enable(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    # E3 轮询走 EXTENSION_ENABLED（总开关），不关则 E3 也报 disabled
    monkeypatch.setattr(config, "EXTENSION_ENABLED", 1)


def _install_prompt_stub(monkeypatch, prompts=None):
    items = list(prompts or PROMPTS)
    calls: dict = {"count": 0}

    async def _fake(messages, timeout_seconds=None, thinking_disabled=None):  # noqa: ANN001
        calls["count"] += 1
        idx = min(calls["count"] - 1, len(items) - 1)
        return json.dumps(items[idx], ensure_ascii=False)

    monkeypatch.setattr(er, "call_round_prompt_llm", _fake)
    return calls


def _install_judge_stub(monkeypatch, results=None):
    items = list(results or [_RUBRIC_PASS])
    calls: dict = {"count": 0}

    async def _fake(points, user_answer, round_context=None):  # noqa: ANN001
        calls["count"] += 1
        idx = min(calls["count"] - 1, len(items) - 1)
        return items[idx]

    monkeypatch.setattr(er, "_call_l2_rubric_llm", _fake)
    return calls


def _recording_stub():
    """把路由的 `run_extension_task` 换成记录型 stub：只记参数，不执行。"""
    captured: dict = {}

    async def _fake(task_id, **kwargs):
        captured["task_id"] = task_id
        captured["kwargs"] = kwargs

    return captured, _fake


def _drive_real_task(captured: dict, fake_db: FakeDB):
    """同步驱动真实 `run_extension_task`（参数来自路由透传的 captured）。"""
    kwargs = captured["kwargs"]
    _run(run_extension_task(captured["task_id"], **kwargs))
    # 任务执行器内部调 get_db()，须指向当前 fake_db
    # （integration conftest 的 fake_db_auto_inject 已覆盖 services.* 模块）


def _poll(client, task_id: str) -> dict:
    """E3 轮询：取任务结果。"""
    resp = client.get(f"/english/extension/task/{task_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    return body["data"]


# ===========================================================================
# 状态机 4 路径（E7 → E3 → E8 → E3 → E9）
# ===========================================================================


class TestStateMachinePaths:
    def test_continue_path(self, make_client, fake_db, monkeypatch):
        """路径 1：未达标且 k<N → continue，下发第 k+1 轮（不含 reference_en，D5）。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)  # start + continue 各出一次
        _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7 开会话
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
                "max_turns": 3,
            },
        )
        body = resp.json()
        assert body["success"] is True
        start_task_id = body["data"]["task_id"]

        # 驱动 start 执行器
        _drive_real_task(captured, fake_db)

        # E3 轮询 start 结果
        data = _poll(client, start_task_id)
        assert data["status"] == "success"
        start_result = data["result"]
        round_id = start_result["round_id"]
        assert start_result["turn_index"] == 1
        assert "reference_en" not in start_result["turn"]  # D5 防泄题

        # E8 提交第 1 轮作答
        captured2, stub2 = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub2)
        resp = client.post(
            "/english/extension/round/turn",
            json={
                "round_id": round_id,
                "scholar_id": SCHOLAR,
                "user_input": ANSWER_OK,
                "client_turn_index": 1,
            },
        )
        turn_task_id = resp.json()["data"]["task_id"]

        # 驱动 turn 执行器
        _drive_real_task(captured2, fake_db)

        # E3 轮询 turn 结果 → continue
        data = _poll(client, turn_task_id)
        assert data["status"] == "success"
        turn_result = data["result"]
        assert turn_result["action"] == "continue"
        assert turn_result["next"] is not None
        assert turn_result["next"]["turn_index"] == 2
        assert "reference_en" not in turn_result["next"]  # D5

        # E9 会话详情：只读，turns[] 含 reference_en（复盘用）
        resp = client.get(
            f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}"
        )
        detail = resp.json()["data"]
        assert detail["status"] == "active"
        assert detail["turn_index"] == 2
        assert detail["turns"][0]["reference_en"] == PROMPTS[0]["reference_en"]
        assert detail["turns"][1]["reference_en"] == PROMPTS[1]["reference_en"]

    def test_finish_passed_path(self, make_client, fake_db, monkeypatch):
        """路径 2：达标 → finish_passed，立即收尾，不再出题。"""
        _enable(monkeypatch)
        prompt_calls = _install_prompt_stub(monkeypatch)  # 只出 1 次（start）
        _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
                "max_turns": 3,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # E8 → 达标
        captured2, stub2 = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub2)
        resp = client.post(
            "/english/extension/round/turn",
            json={"round_id": round_id, "scholar_id": SCHOLAR, "user_input": ANSWER_OK},
        )
        turn_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured2, fake_db)

        data = _poll(client, turn_task_id)
        assert data["status"] == "success"
        turn_result = data["result"]
        assert turn_result["action"] == "finish_passed"
        assert turn_result["next"] is None  # 达标不再出题
        assert turn_result["progress"]["passed"] is True
        assert prompt_calls["count"] == 1  # 达标后不调出题 LLM

        # E9：状态已收尾为 passed
        resp = client.get(
            f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}"
        )
        detail = resp.json()["data"]
        assert detail["status"] == "passed"
        assert detail["summary"]["passed"] is True

    def test_finish_max_path(self, make_client, fake_db, monkeypatch):
        """路径 3：k=N 仍未达标 → finish_max（exhausted），不再出题。"""
        _enable(monkeypatch)
        prompt_calls = _install_prompt_stub(monkeypatch)  # start + 2 次 continue = 3 次
        _install_judge_stub(monkeypatch, [_RUBRIC_FAIL, _RUBRIC_FAIL, _RUBRIC_FAIL])
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7（max_turns=3）
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
                "max_turns": 3,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # E8 × 3：第 1、2 轮 continue，第 3 轮 finish_max
        last_action = None
        for i in range(3):
            cap, st = _recording_stub()
            monkeypatch.setattr("services.routes.extension.run_extension_task", st)
            resp = client.post(
                "/english/extension/round/turn",
                json={
                    "round_id": round_id,
                    "scholar_id": SCHOLAR,
                    "user_input": ANSWER_OK,
                    "client_turn_index": i + 1,
                },
            )
            tid = resp.json()["data"]["task_id"]
            _drive_real_task(cap, fake_db)
            last_action = _poll(client, tid)["result"]["action"]

        assert last_action == "finish_max"
        assert prompt_calls["count"] == 3  # start + 两次 continue，第 3 轮后不出

        # E9：exhausted + summary
        resp = client.get(
            f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}"
        )
        detail = resp.json()["data"]
        assert detail["status"] == "exhausted"
        assert detail["summary"]["turns_used"] == 3
        assert detail["summary"]["passed"] is False

    def test_failed_path_judge_llm_error(self, make_client, fake_db, monkeypatch):
        """路径 4：判分 LLM 失败 → failed，turn_index 不前进（R13）。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)

        async def _judge_error(points, user_answer, round_context=None):  # noqa: ANN001
            raise ExtensionError("LLM_TIMEOUT", "llm", "判分超时")

        monkeypatch.setattr(er, "_call_l2_rubric_llm", _judge_error)

        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
                "max_turns": 3,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # E8 → 判分失败
        captured2, stub2 = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub2)
        resp = client.post(
            "/english/extension/round/turn",
            json={"round_id": round_id, "scholar_id": SCHOLAR, "user_input": ANSWER_OK},
        )
        turn_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured2, fake_db)

        data = _poll(client, turn_task_id)
        assert data["status"] == "failed"
        assert data["error"] is not None
        # E3 只暴露 error_detail（契约红线），错误码在服务端 error.error_code
        assert "判分超时" in data["error"]

        # E9：turn_index 仍为 1，未前进（R13：失败不消耗轮次）
        resp = client.get(
            f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}"
        )
        detail = resp.json()["data"]
        assert detail["turn_index"] == 1
        assert detail["turns"][0]["result"] is None
        assert detail["turns"][0]["user_input"] is None


# ===========================================================================
# R12：轮次数据仅本人可见（归属不符 → 404，与不存在同码）
# ===========================================================================


class TestOwnershipAndVisibility:
    def test_e8_other_scholar_returns_404(self, make_client, fake_db, monkeypatch):
        """R12：他人提交作答 → HTTP 404（不区分「不存在 / 归属不符」，防 round_id 枚举）。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7 建会话
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # E8：他人提交 → 404
        resp = client.post(
            "/english/extension/round/turn",
            json={
                "round_id": round_id,
                "scholar_id": "sch_other",
                "user_input": ANSWER_OK,
            },
        )
        assert resp.status_code == 404

    def test_e9_other_scholar_returns_404(self, make_client, fake_db, monkeypatch):
        """R12：他人查会话详情 → 404。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        resp = client.get(
            f"/english/extension/round/{round_id}?scholar_id=sch_other"
        )
        assert resp.status_code == 404

    def test_e9_nonexistent_round_returns_404(self, make_client, fake_db, monkeypatch):
        """R12：不存在的 round_id → 404（与归属不符同码）。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        resp = client.get(f"/english/extension/round/rnd_missing?scholar_id={SCHOLAR}")
        assert resp.status_code == 404


# ===========================================================================
# R13：禁静默降级（关开关 → EXTENSION_ROUND_DISABLED）
# ===========================================================================


class TestDisabledSwitch:
    def test_all_round_endpoints_disabled(self, make_client, fake_db, monkeypatch):
        """R13：关开关 → E7/E8/E9 统一 EXTENSION_ROUND_DISABLED，不回退单轮 L2。"""
        monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 0)
        monkeypatch.setattr(config, "EXTENSION_ENABLED", 1)
        client = make_client(ext_router)

        r7 = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
            },
        )
        assert r7.json()["code"] == "EXTENSION_ROUND_DISABLED"

        r8 = client.post(
            "/english/extension/round/turn",
            json={"round_id": "rnd_1", "scholar_id": SCHOLAR, "user_input": ANSWER_OK},
        )
        assert r8.json()["code"] == "EXTENSION_ROUND_DISABLED"

        r9 = client.get(f"/english/extension/round/rnd_1?scholar_id={SCHOLAR}")
        assert r9.json()["code"] == "EXTENSION_ROUND_DISABLED"

        # 不静默降级：零写入、不建任务
        assert fake_db.write_log == []


# ===========================================================================
# R14：入参校验（max_turns / user_input / selected_ids / scholar_id）
# ===========================================================================


class TestInputValidation:
    def test_e7_max_turns_over_hard_cap(self, make_client, fake_db, monkeypatch):
        """R14：max_turns 超硬上限 → INVALID_INPUT，不建任务。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "max_turns": config.EXTENSION_ROUND_MAX_TURNS_HARD + 1,
            },
        )
        assert resp.json()["code"] == "INVALID_INPUT"
        assert fake_db.write_log == []

    def test_e7_selected_ids_contains_mine_rejected(self, make_client, fake_db, monkeypatch):
        """自建点（mine_ 前缀）进多轮 → INVALID_INPUT。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0", "mine_1"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
            },
        )
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_e7_selected_ids_not_in_snapshot(self, make_client, fake_db, monkeypatch):
        """勾选项不在快照里 → INVALID_INPUT。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#9"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
            },
        )
        assert resp.json()["code"] == "INVALID_INPUT"

    def test_e8_user_input_too_long(self, make_client, fake_db, monkeypatch):
        """R14：user_input 超长 → INVALID_INPUT（进任务前拦截）。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        too_long = "x" * (config.EXTENSION_ROUND_MAX_INPUT_LEN + 1)
        resp = client.post(
            "/english/extension/round/turn",
            json={"round_id": "rnd_1", "scholar_id": SCHOLAR, "user_input": too_long},
        )
        assert resp.json()["code"] == "INVALID_INPUT"
        assert fake_db.write_log == []

    def test_e8_user_input_empty_rejected(self, make_client, fake_db, monkeypatch):
        """R14（配套）：空 / 纯空白 user_input → INVALID_INPUT。"""
        _enable(monkeypatch)
        client = make_client(ext_router)
        for bad in ("", "   "):
            resp = client.post(
                "/english/extension/round/turn",
                json={"round_id": "rnd_1", "scholar_id": SCHOLAR, "user_input": bad},
            )
            assert resp.json()["code"] == "INVALID_INPUT"

    def test_e8_closed_round_returns_round_closed(self, make_client, fake_db, monkeypatch):
        """终态会话提交 → ROUND_CLOSED。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        # E7 建会话
        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # 手动把会话置为终态
        from services.learning import extension_round as round_repo

        _run(
            round_repo.close_round(
                fake_db, round_id, status=round_repo.STATUS_PASSED, summary={}
            )
        )

        # E8 → ROUND_CLOSED
        resp = client.post(
            "/english/extension/round/turn",
            json={"round_id": round_id, "scholar_id": SCHOLAR, "user_input": ANSWER_OK},
        )
        assert resp.json()["code"] == "ROUND_CLOSED"

    def test_e8_client_turn_index_mismatch(self, make_client, fake_db, monkeypatch):
        """乐观并发：客户端轮次与服务端不一致 → ROUND_TURN_MISMATCH。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # 服务端当前第 1 轮，客户端传 9 → 不匹配
        resp = client.post(
            "/english/extension/round/turn",
            json={
                "round_id": round_id,
                "scholar_id": SCHOLAR,
                "user_input": ANSWER_OK,
                "client_turn_index": 9,
            },
        )
        assert resp.json()["code"] == "ROUND_TURN_MISMATCH"


# ===========================================================================
# E9 只读不推进状态机
# ===========================================================================


class TestE9ReadOnly:
    def test_e9_does_not_advance_state(self, make_client, fake_db, monkeypatch):
        """E9 是同步只读：连续调用不改变 turn_index / status。"""
        _enable(monkeypatch)
        _install_prompt_stub(monkeypatch)
        captured, stub = _recording_stub()
        monkeypatch.setattr("services.routes.extension.run_extension_task", stub)
        client = make_client(ext_router)

        resp = client.post(
            "/english/extension/round",
            json={
                "sentence_id": SENTENCE,
                "scholar_id": SCHOLAR,
                "selected_ids": ["sent_1#0"],
                "points_snapshot": POINTS,
                "original": ORIGINAL,
                "translation": TRANSLATION,
            },
        )
        start_task_id = resp.json()["data"]["task_id"]
        _drive_real_task(captured, fake_db)
        round_id = _poll(client, start_task_id)["result"]["round_id"]

        # 连续调两次 E9，结果一致
        r1 = client.get(f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}")
        r2 = client.get(f"/english/extension/round/{round_id}?scholar_id={SCHOLAR}")
        assert r1.json()["data"] == r2.json()["data"]
        assert r1.json()["data"]["turn_index"] == 1
