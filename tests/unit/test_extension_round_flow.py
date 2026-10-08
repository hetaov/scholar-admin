"""单元测试：语言点造句「多轮」E7 / E8 执行器四路径 + F9 语域解析 + 路由侧形态

契约：api-contract.md §3.18（E7 / E8 / E9）
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §3.4（决策矩阵）/ §3.5（F9）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B09）

与 `test_extension_round.py` 的分工：
- `test_extension_round.py`：红线 R10~R14 + 仓储状态机 + 纯函数；
- 本文件：**编排层** —— E8 的四条推进路径（continue / finish_passed /
  finish_max / failed）、F9 的三组语域解析、E7 / E9 的下发形态（防泄题 D5）。

路由侧只覆盖「**建任务之前就 return**」的分支（关开关 / 终态 / 轮次不匹配 /
404），不起 HTTP、不建后台任务；全链路（TestClient 跑 E7 → E3 轮询）留给 **B10**。
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

import config
from services.english import extension_round as er
from services.english.extension_round import (
    ACTION_CONTINUE,
    ACTION_FINISH_MAX,
    ACTION_FINISH_PASSED,
    DIFFICULTY_HARDER,
    DIFFICULTY_SAME,
    REGISTER_AUTO,
    run_round_start,
    run_round_turn,
)
from services.learning import extension_round as repo
from services.providers.extension_llm import ExtensionError
from services.routes.extension import SubmitRoundTurnRequest, fetch_round, submit_round_turn
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
    _install_judge_stub,
    _install_prompt_stub,
    _run,
    _start,
)


def _turn(db, round_id, monkeypatch, **kwargs):
    return _run(
        run_round_turn(
            db, round_id=round_id, scholar_id=SCHOLAR, user_input=ANSWER_OK, **kwargs
        )
    )


# ===========================================================================
# E7：开会话 + 第 1 轮情景（下发形态：不含 reference_en，D5）
# ===========================================================================


def test_e7_start_returns_first_turn_without_reference(monkeypatch):
    """E7：第 1 轮下发的 `turn` **不含 reference_en**（防泄题 D5）。"""
    db = FakeDB()
    out = _start(db, monkeypatch, max_turns=3)

    assert out["round_id"].startswith("rnd_")
    assert out["turn_index"] == 1
    assert out["max_turns"] == 3
    assert out["status"] == repo.STATUS_ACTIVE
    assert "reference_en" not in out["turn"]
    assert out["turn"]["prompt_zh"] == PROMPTS[0]["prompt_zh"]
    assert out["turn"]["must_use"] == ["take part in"]
    assert out["meta"]["prompt_version"] == config.EXTENSION_ROUND_PROMPT_VERSION
    assert out["meta"]["repeat_risk"] is False

    # 参考句落库了（判分锚点），只是不下发
    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["turns"][0]["reference_en"] == PROMPTS[0]["reference_en"]


def test_e7_start_selected_ids_out_of_snapshot(monkeypatch):
    """E7：勾选项不在快照里 → INVALID_INPUT（同 E2 口径），不建会话。"""
    db = FakeDB()
    _install_prompt_stub(monkeypatch)
    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_start(
                db,
                scholar_id=SCHOLAR,
                sentence_id=SENTENCE,
                selected_ids=["sent_1#9"],
                points_snapshot=POINTS,
                original=ORIGINAL,
            )
        )
    assert e.value.error_code == "INVALID_INPUT"
    assert db.all(config.EXTENSION_ROUND_COLLECTION) == []


# ===========================================================================
# E8 四路径
# ===========================================================================


def test_e8_continue_emits_next_turn(monkeypatch):
    """continue：出第 k+1 轮，`next` 不含 reference_en，used_prompts 累积。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out = _turn(db, started["round_id"], monkeypatch)

    assert out["action"] == ACTION_CONTINUE
    assert out["turn_index"] == 1
    assert out["result"]["score"] == 4
    assert out["progress"] == {
        "turn_index": 1,
        "max_turns": 3,
        "passed": False,
        "best_score": 4,
    }
    assert out["next"] is not None
    assert out["next"]["turn_index"] == 2
    assert "reference_en" not in out["next"]
    assert out["next"]["prompt_zh"] == PROMPTS[1]["prompt_zh"]

    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["turn_index"] == 2
    assert len(doc["used_prompts"]) == 2
    assert doc["turns"][0]["user_input"] == ANSWER_OK
    assert doc["turns"][1]["user_input"] is None  # 新一轮尚未作答


def test_e8_finish_passed_closes_without_further_prompt(monkeypatch):
    """finish_passed：立即收尾为 passed，写小结，**不再调用出题 LLM**。"""
    db = FakeDB()
    prompt_stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            translation=TRANSLATION,
            max_turns=3,
        )
    )
    assert prompt_stub["count"] == 1
    _install_judge_stub(monkeypatch, [_RUBRIC_PASS])

    out = _turn(db, started["round_id"], monkeypatch)
    assert out["action"] == ACTION_FINISH_PASSED
    assert out["next"] is None
    assert out["progress"]["passed"] is True
    assert prompt_stub["count"] == 1  # 达标后不再出题

    doc = _run(repo.get_round(db, started["round_id"], scholar_id=SCHOLAR))
    assert doc["status"] == repo.STATUS_PASSED
    assert doc["summary"]["passed"] is True
    assert doc["summary"]["turns_used"] == 1
    assert doc["summary"]["best_score"] == 8


def test_e8_finish_max_when_turns_exhausted(monkeypatch):
    """finish_max：k=N 仍未达标 → exhausted，不再出题。"""
    db = FakeDB()
    prompt_stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            max_turns=3,
        )
    )
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL, _RUBRIC_FAIL, _RUBRIC_FAIL])

    for expected in (ACTION_CONTINUE, ACTION_CONTINUE, ACTION_FINISH_MAX):
        out = _turn(db, started["round_id"], monkeypatch)
        assert out["action"] == expected

    assert prompt_stub["count"] == 3  # start + 两次 continue，第 3 轮后不再出
    doc = _run(repo.get_round(db, started["round_id"], scholar_id=SCHOLAR))
    assert doc["status"] == repo.STATUS_EXHAUSTED
    assert doc["summary"] == {
        "turns_used": 3,
        "passed": False,
        "best_score": 4,
        "top_errors": ["时态不一致", "搭配生硬"],
        # ★ ADR-0034（第五期 P2）：三轮示范句字符串全等 → 去重保序后只留 1 条
        # （旧断言的 3 条重复正是本期要修的缺陷行为，故刷新断言，非回归）。
        "model_sentences": ["He takes part in the discussion."],
    }


def test_e8_failed_skips_prompt_llm_and_keeps_turn(monkeypatch):
    """failed：判分 LLM 抛错 → 不写 result、不出下一轮、turn_index 不变。"""
    db = FakeDB()
    prompt_stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            max_turns=3,
        )
    )
    _install_judge_stub(monkeypatch, error=ExtensionError("LLM_TIMEOUT", "llm", "超时"))

    with pytest.raises(ExtensionError) as e:
        _turn(db, started["round_id"], monkeypatch)
    assert e.value.error_code == "LLM_TIMEOUT"
    assert prompt_stub["count"] == 1  # 判分失败 → 不触发出题

    doc = _run(repo.get_round(db, started["round_id"], scholar_id=SCHOLAR))
    assert doc["turn_index"] == 1
    assert doc["turns"][-1]["result"] is None
    assert doc["turns"][-1]["user_input"] is None


def test_e8_uses_snapshot_points_for_judge(monkeypatch):
    """判分吃会话内 `points_snapshot` 的勾选项（R11：不回查 AI 集合）。"""
    db = FakeDB()
    started = _start(db, monkeypatch, selected_ids=["sent_1#0", "sent_1#1"], max_turns=3)
    calls = _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    _run(
        run_round_turn(
            db,
            round_id=started["round_id"],
            scholar_id=SCHOLAR,
            user_input="I take part in the discussion.",
        )
    )
    # 两个勾选项都进了 rubric（判分只吃会话内 snapshot，不回查 AI 集合）
    assert [p["text"] for p in calls["points"][0]] == [
        POINTS[0]["text"],
        POINTS[1]["text"],
    ]
    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["selected_ids"] == ["sent_1#0", "sent_1#1"]


# ===========================================================================
# F9：第 k+1 轮的 difficulty / register 解析（会话内自取）
# ===========================================================================


def test_f9_explicit_register_is_passed_through(monkeypatch):
    """F9 组 1：会话 `register` 非 auto → 直接用该偏好。"""
    db = FakeDB()
    stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            max_turns=3,
            register="formal",
            difficulty=DIFFICULTY_SAME,
        )
    )
    assert "语域偏好：formal" in stub["messages"][0][-1]["content"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _turn(db, started["round_id"], monkeypatch)
    assert "语域偏好：formal" in stub["messages"][-1][-1]["content"]


def test_f9_auto_same_inherits_previous_register(monkeypatch):
    """F9 组 2：auto + same → 沿用上一轮语域（换场景但保持语域）。"""
    db = FakeDB()
    stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            max_turns=3,
            register=REGISTER_AUTO,
            difficulty=DIFFICULTY_SAME,
        )
    )
    # 第 1 轮无偏好行（auto 交给模型按情景定）
    assert "语域偏好" not in stub["messages"][0][-1]["content"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _turn(db, started["round_id"], monkeypatch)
    content = stub["messages"][-1][-1]["content"]
    assert "语域偏好：neutral" in content  # 沿用 PROMPTS[0].register
    assert "换一个场景，语域保持不变" in content


def test_f9_auto_harder_does_not_inherit_register(monkeypatch):
    """F9 组 3：auto + harder → **不沿用**（沿用会把「升一级语域」锁死）。"""
    db = FakeDB()
    stub = _install_prompt_stub(monkeypatch)
    started = _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            max_turns=3,
            register=REGISTER_AUTO,
            difficulty=DIFFICULTY_HARDER,
        )
    )
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _turn(db, started["round_id"], monkeypatch)

    content = stub["messages"][-1][-1]["content"]
    assert "升一级语域" in content
    assert "语域偏好" not in content  # 关键：不留用上一轮语域


def test_f9_resolve_round_settings_matrix():
    """F9 解析表（纯函数）：显式直达 / auto+same 沿用 / auto+harder 强制 auto。"""
    resolve = er._resolve_round_settings
    turns = [{"register": "neutral"}]

    assert resolve({"register": "formal", "difficulty": DIFFICULTY_SAME}, turns) == (
        DIFFICULTY_SAME,
        "formal",
    )
    assert resolve({"register": REGISTER_AUTO, "difficulty": DIFFICULTY_SAME}, turns) == (
        DIFFICULTY_SAME,
        "neutral",
    )
    assert resolve({"register": REGISTER_AUTO, "difficulty": DIFFICULTY_HARDER}, turns) == (
        DIFFICULTY_HARDER,
        REGISTER_AUTO,
    )
    # 老文档（补订前无字段）→ 回落 auto / same，零迁移
    assert resolve({}, []) == (DIFFICULTY_SAME, REGISTER_AUTO)


# ===========================================================================
# 归属 / 轮次 / 终态
# ===========================================================================


def test_e8_closed_round_raises_round_closed(monkeypatch):
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=1)
    round_id = started["round_id"]
    _run(repo.abandon_round(db, round_id, scholar_id=SCHOLAR))
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])

    with pytest.raises(ExtensionError) as e:
        _turn(db, round_id, monkeypatch)
    assert e.value.error_code == repo.ERR_ROUND_CLOSED


def test_e8_client_turn_index_mismatch(monkeypatch):
    """乐观并发：客户端轮次与服务端不一致 → ROUND_TURN_MISMATCH。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    with pytest.raises(ExtensionError) as e:
        _turn(db, started["round_id"], monkeypatch, client_turn_index=2)
    assert e.value.error_code == repo.ERR_ROUND_TURN_MISMATCH

    # 一致 → 放行
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out = _turn(db, started["round_id"], monkeypatch, client_turn_index=1)
    assert out["action"] == ACTION_CONTINUE


def test_e8_missing_round_is_round_not_found(monkeypatch):
    db = FakeDB()
    _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    with pytest.raises(ExtensionError) as e:
        _turn(db, "rnd_missing", monkeypatch)
    assert e.value.error_code == repo.ERR_ROUND_NOT_FOUND


# ===========================================================================
# 路由侧形态（只覆盖「建任务前 return」的分支；全链路留 B10）
# ===========================================================================


def test_e9_fetch_round_is_read_only_and_exposes_reference(monkeypatch):
    """E9：同步只读，不推进状态机；`turns[]` **含** reference_en（复盘用）。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    db.reset_write_log()

    out = _run(fetch_round(started["round_id"], scholar_id=SCHOLAR, db=db))
    assert db.write_log == []  # 只读
    assert out["success"] is True
    assert out["data"]["status"] == repo.STATUS_ACTIVE
    assert out["data"]["turn_index"] == 1
    assert out["data"]["turns"][0]["reference_en"] == PROMPTS[0]["reference_en"]


def test_e9_fetch_round_requires_scholar_id(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    out = _run(fetch_round("rnd_1", scholar_id="", db=db))
    assert out["success"] is False
    assert out["code"] == "INVALID_INPUT"


def test_route_turn_closed_round_returns_round_closed(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=1)
    _run(repo.close_round(db, started["round_id"], status=repo.STATUS_PASSED, summary={}))

    out = _run(
        submit_round_turn(
            SubmitRoundTurnRequest(
                round_id=started["round_id"], scholar_id=SCHOLAR, user_input=ANSWER_OK
            ),
            db=db,
        )
    )
    assert out["success"] is False
    assert out["code"] == "ROUND_CLOSED"
    assert db.write_calls["insert"] == 1  # 只有建会话那一次，未建任务


def test_route_turn_mismatch_returns_round_turn_mismatch(monkeypatch):
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    out = _run(
        submit_round_turn(
            SubmitRoundTurnRequest(
                round_id=started["round_id"],
                scholar_id=SCHOLAR,
                user_input=ANSWER_OK,
                client_turn_index=9,
            ),
            db=db,
        )
    )
    assert out["success"] is False
    assert out["code"] == "ROUND_TURN_MISMATCH"


def test_route_turn_not_found_raises_404(monkeypatch):
    """R12：归属不符 / 不存在 → HTTP 404（不区分，防 round_id 枚举）。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    db.reset_write_log()

    with pytest.raises(HTTPException) as e:
        _run(
            submit_round_turn(
                SubmitRoundTurnRequest(
                    round_id=started["round_id"], scholar_id="sch_other", user_input=ANSWER_OK
                ),
                db=db,
            )
        )
    assert e.value.status_code == 404

    with pytest.raises(HTTPException) as e2:
        _run(fetch_round("rnd_missing", scholar_id=SCHOLAR, db=db))
    assert e2.value.status_code == 404
    assert db.write_log == []


# ===========================================================================
# 第四期 Z07：「同题重提（改一次）同样消耗一轮」
# ===========================================================================


def _doc(db, round_id):
    return _run(repo.get_round(db, round_id, scholar_id=SCHOLAR))


def test_e8_same_question_retry_is_accepted_and_consumes_a_round(monkeypatch):
    """同题重提：客户端仍报**上一轮**下标 ⇒ 把那一轮题面重新下发为当前轮并判分。

    要点：① 不得 ROUND_TURN_MISMATCH；② `turn_index` / `progress.turn_index` **前进到当前轮**
    ⇒ 该次作答**同样消耗一轮**；③ 当前轮题面被改写为**上一轮那一道**（同题）；④ **不新增轮次**。
    """
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    rid = started["round_id"]
    prompt_1 = started["turn"]["prompt_zh"]

    # 第 1 轮未达标 → continue（服务端随即下发第 2 轮）
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out1 = _turn(db, rid, monkeypatch, client_turn_index=1)
    assert out1["action"] == ACTION_CONTINUE
    assert out1["turn_index"] == 1
    assert _doc(db, rid)["turn_index"] == 2  # 已下发第 2 轮

    # 「按反馈改一次（同题）」：客户端仍报 1（= 服务端 2 - 1）——此前必然 ROUND_TURN_MISMATCH
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out2 = _turn(db, rid, monkeypatch, client_turn_index=1)
    assert out2["action"] == ACTION_CONTINUE
    assert out2["turn_index"] == 2           # ← 该次作答落在第 2 轮（消耗一轮）
    assert out2["progress"]["turn_index"] == 2

    doc = _doc(db, rid)
    # 重提本身**不新增轮次**（改写第 2 轮题面）；`continue` 之后照常下发第 3 轮 ⇒ 共 3 条
    assert len(doc["turns"]) == 3
    assert doc["turns"][1]["prompt_zh"] == prompt_1  # 第 2 轮题面 = 第 1 轮那一道（同题）
    assert doc["turns"][1]["result"] is not None     # 判分写入当前轮（第 2 轮）
    assert doc["turns"][0]["result"] is not None     # 第 1 轮原判分保留（不被覆盖）
    assert doc["turn_index"] == 3                    # 已下发第 3 轮


def test_e8_same_question_retry_can_exhaust_the_budget(monkeypatch):
    """轮次预算按**作答次数**计：max_turns=2 时，第 1 轮的「改一次」就吃掉最后一个名额 →
    未达标即 `finish_max`（会话收口），而不是白给一次重试。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=2)
    rid = started["round_id"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    assert _turn(db, rid, monkeypatch, client_turn_index=1)["action"] == ACTION_CONTINUE

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out = _turn(db, rid, monkeypatch, client_turn_index=1)  # 同题重提 → 占用第 2 轮 = 上限
    assert out["action"] == ACTION_FINISH_MAX
    assert out["next"] is None
    assert _doc(db, rid)["status"] == repo.STATUS_EXHAUSTED


def test_e8_retry_pass_ends_session(monkeypatch):
    """同题重提**通过** → 正常 `finish_passed`（会话收口，不再出题）。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    rid = started["round_id"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    assert _turn(db, rid, monkeypatch, client_turn_index=1)["action"] == ACTION_CONTINUE

    _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    out = _turn(db, rid, monkeypatch, client_turn_index=1)
    assert out["action"] == ACTION_FINISH_PASSED
    assert out["next"] is None
    assert _doc(db, rid)["status"] == repo.STATUS_PASSED


def test_e8_retry_index_range_is_narrow(monkeypatch):
    """放宽**仅限** `cur` 与 `cur-1`：更早轮次（cur-2）与未来的轮次仍 ROUND_TURN_MISMATCH。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=4)
    rid = started["round_id"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _turn(db, rid, monkeypatch, client_turn_index=1)   # → 下发第 2 轮
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _turn(db, rid, monkeypatch, client_turn_index=2)   # → 下发第 3 轮（cur=3）

    for bad in (1, 4, 9):  # 1 = cur-2（非「上一轮」）；4/9 = 未来轮次
        with pytest.raises(ExtensionError) as e:
            _turn(db, rid, monkeypatch, client_turn_index=bad)
        assert e.value.error_code == repo.ERR_ROUND_TURN_MISMATCH


def test_e8_retry_not_allowed_on_first_turn(monkeypatch):
    """第 1 轮（cur=1）没有「上一轮」可言 ⇒ `client_turn_index=0` 仍 ROUND_TURN_MISMATCH。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    with pytest.raises(ExtensionError) as e:
        _turn(db, started["round_id"], monkeypatch, client_turn_index=0)
    assert e.value.error_code == repo.ERR_ROUND_TURN_MISMATCH


def test_route_turn_accepts_same_question_retry(monkeypatch):
    """路由层同样接受「同题重提」（`client_turn_index == cur-1`）。

    ⚠️ 两处都放宽才算通（路由的 `allowed` 集合 + `run_round_turn` 的重提分支）——缺一即失败，
    故本用例锁路由那一处（编排那处由上面的 E8 用例覆盖）。
    """
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    rid = started["round_id"]

    # 第 1 轮未达标 → 服务端下发第 2 轮
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    assert _turn(db, rid, monkeypatch, client_turn_index=1)["action"] == ACTION_CONTINUE

    # 同题重提：客户端仍报 1（= cur 2 - 1）→ 路由**不得**回 ROUND_TURN_MISMATCH
    out = _run(
        submit_round_turn(
            SubmitRoundTurnRequest(
                round_id=rid,
                scholar_id=SCHOLAR,
                user_input=ANSWER_OK,
                client_turn_index=1,
            ),
            db=db,
        )
    )
    assert out["success"] is True
    assert out.get("code") != "ROUND_TURN_MISMATCH"
