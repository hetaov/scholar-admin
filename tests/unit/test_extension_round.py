"""单元测试：语言点造句「多轮」—— 红线 R10~R14 + 仓储状态机 + 出题/判分/决策纯函数

契约：api-contract.md §3.18（E7 / E8 / E9）；data-model-contract §4.28
设计：docs_v1/扩展/第三期-语言点造句多轮-v1.md §3.3（出题规格）/ §3.4（判分与决策）
账本：docs_v1/扩展/第三期-语言点造句多轮-任务拆分与断点-v1.md（B09）

本文件把 B02~B08 的验证（此前散在 `/tmp/b0*_smoke.py`，**不入库、不可回归**）
固化进 pytest，并补齐 **R10~R14 五条红线断言**：

| 红线 | 内容 | 落点 |
|---|---|---|
| R10 | 不写 mastery status（只写 `extension_round`） | `test_r10_*` |
| R11 | 不回写 `english_extension_point` | `test_r11_*` |
| R12 | 轮次数据仅本人可见 | `test_r12_*` |
| R13 | 失败不消耗轮次 / 禁静默降级 | `test_r13_*` |
| R14 | 轮次与输入长度封顶 | `test_r14_*` |

> 「同文件原则」：一期 R1 / R9 明文要求护栏**彼此同文件**放置（防拦截器绕过），
> 三期 R10~R14 沿用同一原则 —— 五条红线集中在本文件；mastery 语义键常量
> **import 复用** `test_extension_evaluate._MASTERY_KEYS`，避免护栏语义分叉。

**本期零生产代码改动**：仅新增测试（B09 的定位是「把不可回归的冒烟变成可回归的断言」）。
"""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

import config
from services.english import extension_round as er
from services.english.extension_round import (
    ACTION_CONTINUE,
    ACTION_FAILED,
    ACTION_FINISH_MAX,
    ACTION_FINISH_PASSED,
    DIFFICULTY_HARDER,
    DIFFICULTY_SAME,
    REGISTER_AUTO,
    build_round_prompt_messages,
    build_summary,
    decide_next_action,
    generate_round_prompt,
    is_repeat_prompt,
    judge_round_turn,
    run_round_start,
    run_round_turn,
    validate_prompt_item,
)
from services.learning import extension_round as repo
from services.providers.extension_llm import ExtensionError
from services.routes.extension import (
    StartRoundRequest,
    SubmitRoundTurnRequest,
    fetch_round,
    start_round,
    submit_round_turn,
)
from tests.fakes.fake_db import FakeDB

# 护栏语义键与一期同源（R1 / R9 同款写法），避免两套护栏各自漂移
from tests.unit.test_extension_evaluate import _MASTERY_KEYS  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# ===========================================================================
# 造数与桩
# ===========================================================================

SCHOLAR = "sch_1"
OTHER = "sch_other"
SENTENCE = "sent_1"

POINTS = [
    {
        "id": "sent_1#0",
        "type": "phrase",
        "text": "take part in",
        "meaning_zh": "参加",
        "origin": "ai",
    },
    {
        "id": "sent_1#1",
        "type": "word",
        "text": "discussion",
        "meaning_zh": "讨论",
        "origin": "ai",
    },
]

ORIGINAL = "He took part in the discussion yesterday."
TRANSLATION = "他昨天参加了讨论。"

# 出题桩：三条不同场景的情景句（banned_check 要求中文句里无目标点英文原文）
PROMPTS = [
    {
        "prompt_zh": "会议拖得太久，他终于把话题转到了预算上。",
        "register": "neutral",
        "focus_hint": "",
        "scene_tag": ["work", "meeting"],
        "reference_en": "He finally took part in the budget talk.",
    },
    {
        "prompt_zh": "孩子们在操场上排好队，等着老师发新书。",
        "register": "neutral",
        "focus_hint": "注意 take part in 后直接接名词",
        "scene_tag": ["school", "kids"],
        "reference_en": "The kids take part in the morning reading.",
    },
    {
        "prompt_zh": "社区晚会上，几位老人也报名参加了合唱。",
        "register": "formal",
        "focus_hint": "注意主语与 take part in 的数的一致",
        "scene_tag": ["community", "family"],
        "reference_en": "They take part in the choir every weekend.",
    },
]

_RUBRIC_PASS = {
    "scores": {"target_usage": 2, "grammar": 2, "semantics": 2, "register_collocation": 2},
    "total": 8,
    "errors": [],
    "suggestion": "",
    "model_sentence": "He took part in the discussion.",
}
_RUBRIC_FAIL = {
    "scores": {"target_usage": 1, "grammar": 1, "semantics": 1, "register_collocation": 1},
    "total": 4,
    "errors": ["时态不一致", "搭配生硬"],
    "suggestion": "把 took 改成 takes",
    "model_sentence": "He takes part in the discussion.",
}

ANSWER_OK = "I take part in the discussion."
ANSWER_MISS = "I joined the meeting."


def _install_prompt_stub(monkeypatch, prompts=None, error: Exception | None = None):
    """替换出题 LLM：按调用次序返回 `prompts`（`error` 非空则抛出）。"""
    items = list(prompts or PROMPTS)
    calls: dict = {"count": 0, "messages": []}

    async def _fake(messages, timeout_seconds=None, thinking_disabled=None):  # noqa: ANN001
        calls["count"] += 1
        calls["messages"].append(messages)
        if error is not None:
            raise error
        idx = min(calls["count"] - 1, len(items) - 1)
        return json.dumps(items[idx], ensure_ascii=False)

    monkeypatch.setattr(er, "call_round_prompt_llm", _fake)
    return calls


def _install_judge_stub(monkeypatch, results=None, error: Exception | None = None):
    """替换判分 LLM：按调用次序返回 `results`（`error` 非空则抛出）。"""
    items = list(results or [_RUBRIC_PASS])
    calls: dict = {"count": 0, "contexts": [], "points": [], "answers": []}

    async def _fake(points, user_answer, round_context=None):  # noqa: ANN001
        calls["count"] += 1
        calls["contexts"].append(round_context)
        calls["points"].append(points)
        calls["answers"].append(user_answer)
        if error is not None:
            raise error
        idx = min(calls["count"] - 1, len(items) - 1)
        return items[idx]

    monkeypatch.setattr(er, "_call_l2_rubric_llm", _fake)
    return calls


def _start(db, monkeypatch, *, selected_ids=None, max_turns=None, **kwargs):
    """建会话（出题桩已装）→ 返回 E7 result。"""
    _install_prompt_stub(monkeypatch)
    return _run(
        run_round_start(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=selected_ids or ["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            translation=TRANSLATION,
            max_turns=max_turns,
            **kwargs,
        )
    )


def _code(exc_info) -> str:
    return exc_info.value.error_code


# ===========================================================================
# 红线护栏（R10 / R11）
# ===========================================================================

# 多轮链路**只允许**写 `extension_round`（红线 R10）
_ROUND_WRITE_WHITELIST = frozenset({config.EXTENSION_ROUND_COLLECTION})


def _scan_mastery(node) -> list[str]:
    """递归扫描 mastery 语义键（会话文档 + `turns[]` 内嵌结构都要查）。"""
    hits: list[str] = []
    if isinstance(node, dict):
        hits += [k for k in _MASTERY_KEYS if k in node]
        for value in node.values():
            hits += _scan_mastery(value)
    elif isinstance(node, list):
        for value in node:
            hits += _scan_mastery(value)
    return hits


def _assert_round_writes_only(db: FakeDB) -> None:
    """红线护栏：写集合 ⊆ {extension_round}；会话文档内无 mastery 语义键。"""
    written = {coll for _, coll in db.write_log}
    outside = written - _ROUND_WRITE_WHITELIST
    assert not outside, f"多轮链路写了非 extension_round 集合：{outside}"
    for doc in db.all(config.EXTENSION_ROUND_COLLECTION):
        hit = _scan_mastery(doc)
        assert not hit, f"会话文档出现 mastery 语义字段：{hit}"


# ===========================================================================
# R10：不写 mastery status（只写 extension_round）
# ===========================================================================


def test_r10_full_round_writes_only_round_collection(monkeypatch):
    """R10：start → 3 轮 turn → close 全链路，写集合**只有** `extension_round`。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    round_id = started["round_id"]

    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL, _RUBRIC_FAIL, _RUBRIC_FAIL])
    for _ in range(3):
        _run(
            run_round_turn(
                db,
                round_id=round_id,
                scholar_id=SCHOLAR,
                user_input=ANSWER_OK,
                client_turn_index=None,
            )
        )

    assert {coll for _, coll in db.write_log} == {config.EXTENSION_ROUND_COLLECTION}
    _assert_round_writes_only(db)

    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["status"] == repo.STATUS_EXHAUSTED  # 3 轮未达标 → exhausted
    # 不进既有学习状态集合：全库除 extension_round 外零文档
    assert db.all(config.EXTENSION_POINT_COLLECTION) == []
    assert db.all(config.EXTENSION_TASK_COLLECTION) == []


def test_r10_passed_round_has_no_mastery_keys(monkeypatch):
    """R10：达标收尾同样只写 `extension_round`，且文档内无 mastery 语义键。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    out = _run(
        run_round_turn(
            db,
            round_id=started["round_id"],
            scholar_id=SCHOLAR,
            user_input=ANSWER_OK,
        )
    )
    assert out["action"] == ACTION_FINISH_PASSED
    _assert_round_writes_only(db)
    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["status"] == repo.STATUS_PASSED
    assert doc["summary"]["passed"] is True


def test_r10_repo_layer_writes_only_round_collection():
    """R10：仓储层（不含 LLM）每次写操作都落在 `extension_round`。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            max_turns=3,
        )
    )
    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    _run(
        repo.append_turn_result(
            db, doc["round_id"], turn_index=1, result=_RUBRIC_PASS, user_input=ANSWER_OK
        )
    )
    _run(
        repo.close_round(
            db, doc["round_id"], status=repo.STATUS_PASSED, summary={"turns_used": 1}
        )
    )
    assert {coll for _, coll in db.write_log} == {config.EXTENSION_ROUND_COLLECTION}
    _assert_round_writes_only(db)


# ===========================================================================
# R11：不回写 english_extension_point
# ===========================================================================


def test_r11_no_write_back_to_extension_point(monkeypatch):
    """R11：整条多轮跑完，`english_extension_point` 文档零变化（零回写）。"""
    db = FakeDB()
    db.add(
        config.EXTENSION_POINT_COLLECTION,
        {
            "point_key": "pk_1",
            "sentence_id": SENTENCE,
            "content_hash": "h1",
            "status": "success",
            "source": "llm",
            "points": POINTS,
            "updated_at": 1,
        },
    )
    before = db.snapshot(config.EXTENSION_POINT_COLLECTION)

    started = _start(db, monkeypatch, max_turns=3)
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL, _RUBRIC_PASS])
    for _ in range(2):
        _run(
            run_round_turn(
                db,
                round_id=started["round_id"],
                scholar_id=SCHOLAR,
                user_input=ANSWER_OK,
            )
        )

    assert db.snapshot(config.EXTENSION_POINT_COLLECTION) == before
    assert config.EXTENSION_POINT_COLLECTION not in {c for _, c in db.write_log}


def test_r11_points_snapshot_stays_inside_round(monkeypatch):
    """R11：判分只吃会话内 `points_snapshot`，不回查也不回写 AI 集合。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=2)
    doc = db.all(config.EXTENSION_ROUND_COLLECTION)[0]
    assert doc["points_snapshot"] == POINTS
    # AI 集合自始至终不存在（未被读、更未被写）
    assert db.all(config.EXTENSION_POINT_COLLECTION) == []


# ===========================================================================
# R12：轮次数据仅本人可见
# ===========================================================================


def test_r12_get_round_other_scholar_is_none():
    """R12：归属不匹配 → None（与「不存在」同码，防 round_id 枚举读 turns[]）。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    assert _run(repo.get_round(db, doc["round_id"], scholar_id=SCHOLAR)) is not None
    assert _run(repo.get_round(db, doc["round_id"], scholar_id=OTHER)) is None
    assert _run(repo.get_round(db, "rnd_missing", scholar_id=SCHOLAR)) is None


def test_r12_run_round_turn_other_scholar_is_round_not_found(monkeypatch):
    """R12：他人提交作答 → ROUND_NOT_FOUND（与不存在同码，不泄露会话存在性）。"""
    db = FakeDB()
    started = _start(db, monkeypatch)
    _install_judge_stub(monkeypatch)
    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_turn(
                db,
                round_id=started["round_id"],
                scholar_id=OTHER,
                user_input=ANSWER_OK,
            )
        )
    assert _code(e) == repo.ERR_ROUND_NOT_FOUND


def test_r12_read_api_is_scholar_scoped_only():
    """R12：无班级 / 家长 / 教师维度入口 —— 读接口形参不出现非 scholar 维度。"""
    forbidden = {"class_id", "class_ids", "parent_id", "teacher_id", "group_id"}
    for name in ("get_round",):
        params = inspect.signature(getattr(repo, name)).parameters
        assert not [p for p in params if p in forbidden]
    # 模块导出的读函数里也不得有按其他维度查询的入口
    for name in dir(repo):
        if not name.startswith(("get_", "list_", "fetch_", "query_")):
            continue
        fn = getattr(repo, name)
        if not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        assert not [p for p in params if p in forbidden], (
            f"repo.{name} 暴露了非 scholar 维度入参：{sorted(set(params) & forbidden)}"
        )


def test_r12_abandon_round_ownership_mismatch():
    """R12：放弃会话也做归属校验 —— 他人 → ROUND_NOT_FOUND（不泄露存在性）。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    with pytest.raises(ExtensionError) as e:
        _run(repo.abandon_round(db, doc["round_id"], scholar_id=OTHER))
    assert _code(e) == repo.ERR_ROUND_NOT_FOUND


# ===========================================================================
# R13：失败不消耗轮次 / 禁静默降级
# ===========================================================================


def test_r13_disabled_switch_blocks_all_round_endpoints(monkeypatch):
    """R13①：关开关 → 三个端点统一 EXTENSION_ROUND_DISABLED，**不建任务、不回退单轮 L2**。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 0)
    db = FakeDB()

    out_start = _run(
        start_round(
            StartRoundRequest(
                sentence_id=SENTENCE,
                scholar_id=SCHOLAR,
                original=ORIGINAL,
                selected_ids=["sent_1#0"],
                points_snapshot=POINTS,
            ),
            db=db,
        )
    )
    out_turn = _run(
        submit_round_turn(
            SubmitRoundTurnRequest(
                round_id="rnd_1", scholar_id=SCHOLAR, user_input=ANSWER_OK
            ),
            db=db,
        )
    )
    out_fetch = _run(fetch_round("rnd_1", scholar_id=SCHOLAR, db=db))

    for out in (out_start, out_turn, out_fetch):
        assert out["success"] is False
        assert out["code"] == "EXTENSION_ROUND_DISABLED"
    # 不静默降级：既不建多轮会话，也不回退单轮 L2（零写入）
    assert db.write_log == []
    assert db.all(config.EXTENSION_ROUND_COLLECTION) == []


def test_r13_prompt_llm_failure_creates_no_round(monkeypatch):
    """R13②：出题 LLM 失败 → 透出错误 + **不建会话**（无 turn_index=0 孤儿）。"""
    db = FakeDB()
    _install_prompt_stub(
        monkeypatch, error=ExtensionError("LLM_TIMEOUT", "llm", "出题超时")
    )
    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_start(
                db,
                scholar_id=SCHOLAR,
                sentence_id=SENTENCE,
                selected_ids=["sent_1#0"],
                points_snapshot=POINTS,
                original=ORIGINAL,
            )
        )
    assert _code(e) == "LLM_TIMEOUT"
    assert db.all(config.EXTENSION_ROUND_COLLECTION) == []
    assert db.write_log == []


def test_r13_judge_llm_failure_does_not_consume_turn(monkeypatch):
    """R13③：判分 LLM 失败 → 不写 result、turn_index 不变、user_input 不落库。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    round_id = started["round_id"]
    _install_judge_stub(
        monkeypatch, error=ExtensionError("LLM_TIMEOUT", "llm", "判分超时")
    )

    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_turn(
                db, round_id=round_id, scholar_id=SCHOLAR, user_input=ANSWER_OK
            )
        )
    assert _code(e) == "LLM_TIMEOUT"

    doc = _run(repo.get_round(db, round_id, scholar_id=SCHOLAR))
    assert doc["turn_index"] == 1  # 未前进
    assert doc["turns"][-1]["result"] is None
    assert doc["turns"][-1]["user_input"] is None
    assert doc["status"] == repo.STATUS_ACTIVE  # 不误判终态


def test_r13_banned_check_hit_raises_and_consumes_nothing(monkeypatch):
    """R13④：banned_check 命中 → LLM_PARSE_ERROR，轮次不消耗（无孤儿会话）。"""
    db = FakeDB()
    leak = {
        "prompt_zh": "请把 take part in 这个短语用在句子里。",  # 出现目标点英文原文
        "register": "neutral",
        "focus_hint": "",
        "scene_tag": ["work"],
        "reference_en": "He took part in it.",
    }
    _install_prompt_stub(monkeypatch, [leak])
    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_start(
                db,
                scholar_id=SCHOLAR,
                sentence_id=SENTENCE,
                selected_ids=["sent_1#0"],
                points_snapshot=POINTS,
                original=ORIGINAL,
            )
        )
    assert _code(e) == "LLM_PARSE_ERROR"
    assert "take part in" in e.value.detail
    assert db.all(config.EXTENSION_ROUND_COLLECTION) == []


def test_r13_banned_check_hit_on_followup_turn_keeps_turn_index(monkeypatch):
    """R13④：第 2 轮出题命中 banned_check → 本轮作废，服务端仍停在第 1 轮。"""
    db = FakeDB()
    started = _start(db, monkeypatch, max_turns=3)
    round_id = started["round_id"]
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    _install_prompt_stub(
        monkeypatch,
        [
            {
                "prompt_zh": "试着用 take part in 造一个句子。",
                "register": "neutral",
                "focus_hint": "",
                "scene_tag": ["school"],
                "reference_en": "He took part in it.",
            },
        ],
    )
    with pytest.raises(ExtensionError) as e:
        _run(
            run_round_turn(
                db, round_id=round_id, scholar_id=SCHOLAR, user_input=ANSWER_OK
            )
        )
    assert _code(e) == "LLM_PARSE_ERROR"

    doc = _run(repo.get_round(db, round_id, scholar_id=SCHOLAR))
    # 第 1 轮判分已落库，但第 2 轮未下发 → turn_index 仍为 1
    assert doc["turn_index"] == 1
    assert len(doc["turns"]) == 1
    assert doc["turns"][0]["result"] is not None


# ===========================================================================
# R14：轮次与输入长度封顶（双侧：仓储 + 路由入参）
# ===========================================================================


def test_r14_create_round_max_turns_over_hard_cap():
    """R14：max_turns 超硬上限 → INVALID_INPUT（仓储量侧）。"""
    db = FakeDB()
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.create_round(
                db,
                scholar_id=SCHOLAR,
                sentence_id=SENTENCE,
                selected_ids=["sent_1#0"],
                points_snapshot=POINTS,
                max_turns=config.EXTENSION_ROUND_MAX_TURNS_HARD + 1,
            )
        )
    assert _code(e) == repo.ERR_INVALID_INPUT
    assert db.write_log == []


def test_r14_create_round_max_turns_at_hard_cap_is_allowed():
    """R14：max_turns == 硬上限放行；0 / 负数拒绝。"""
    db = FakeDB()
    ok = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            max_turns=config.EXTENSION_ROUND_MAX_TURNS_HARD,
        )
    )
    assert ok["max_turns"] == config.EXTENSION_ROUND_MAX_TURNS_HARD

    for bad in (0, -1):
        with pytest.raises(ExtensionError) as e:
            _run(
                repo.create_round(
                    db,
                    scholar_id=SCHOLAR,
                    sentence_id=SENTENCE,
                    selected_ids=["sent_1#0"],
                    points_snapshot=POINTS,
                    max_turns=bad,
                )
            )
        assert _code(e) == repo.ERR_INVALID_INPUT


def test_r14_append_turn_result_user_input_too_long():
    """R14：user_input 超过 EXTENSION_ROUND_MAX_INPUT_LEN → INVALID_INPUT。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    too_long = "x" * (config.EXTENSION_ROUND_MAX_INPUT_LEN + 1)
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.append_turn_result(
                db,
                doc["round_id"],
                turn_index=1,
                result=_RUBRIC_PASS,
                user_input=too_long,
            )
        )
    assert _code(e) == repo.ERR_INVALID_INPUT

    # 恰好等于上限放行
    _run(
        repo.append_turn_result(
            db,
            doc["round_id"],
            turn_index=1,
            result=_RUBRIC_PASS,
            user_input="x" * config.EXTENSION_ROUND_MAX_INPUT_LEN,
        )
    )
    got = _run(repo.get_round(db, doc["round_id"], scholar_id=SCHOLAR))
    assert len(got["turns"][0]["user_input"]) == config.EXTENSION_ROUND_MAX_INPUT_LEN


def test_r14_append_turn_beyond_max_turns_rejected():
    """R14：已到 max_turns 仍追加轮次 → INVALID_INPUT（文档体积封顶）。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            max_turns=2,
        )
    )
    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[1]))
    with pytest.raises(ExtensionError) as e:
        _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[2]))
    assert _code(e) == repo.ERR_INVALID_INPUT


def test_r14_route_max_turns_over_hard_cap(monkeypatch):
    """R14：路由侧 max_turns 越界 → INVALID_INPUT，且不建任务、不落库。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    out = _run(
        start_round(
            StartRoundRequest(
                sentence_id=SENTENCE,
                scholar_id=SCHOLAR,
                original=ORIGINAL,
                selected_ids=["sent_1#0"],
                points_snapshot=POINTS,
                max_turns=config.EXTENSION_ROUND_MAX_TURNS_HARD + 1,
            ),
            db=db,
        )
    )
    assert out["success"] is False
    assert out["code"] == "INVALID_INPUT"
    assert db.write_log == []


def test_r14_route_user_input_too_long(monkeypatch):
    """R14：路由侧 user_input 超长 → INVALID_INPUT（红线：进任务前拦截）。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    out = _run(
        submit_round_turn(
            SubmitRoundTurnRequest(
                round_id="rnd_1",
                scholar_id=SCHOLAR,
                user_input="x" * (config.EXTENSION_ROUND_MAX_INPUT_LEN + 1),
            ),
            db=db,
        )
    )
    assert out["success"] is False
    assert out["code"] == "INVALID_INPUT"
    assert db.write_log == []  # 未读库即拦截，更未建任务


def test_r14_route_user_input_empty_rejected(monkeypatch):
    """R14（配套）：空 / 纯空白 user_input 同样 INVALID_INPUT。"""
    monkeypatch.setattr(config, "EXTENSION_ROUND_ENABLED", 1)
    db = FakeDB()
    for bad in ("", "   "):
        out = _run(
            submit_round_turn(
                SubmitRoundTurnRequest(
                    round_id="rnd_1", scholar_id=SCHOLAR, user_input=bad
                ),
                db=db,
            )
        )
        assert out["code"] == "INVALID_INPUT"
    assert db.write_log == []


# ===========================================================================
# 仓储：建会话 / 读 / 状态机（固化 B02）
# ===========================================================================


def test_build_round_id_prefix():
    assert repo.build_round_id().startswith("rnd_")


def test_build_prompt_fingerprint_normalizes():
    """指纹归一化：空白 / 大小写 / 标签排序都不影响相等判定。"""
    a = repo.build_prompt_fingerprint(" 会议 很长 ", ["Work", "meeting"])
    b = repo.build_prompt_fingerprint("会议很长", ["meeting", "work"])
    assert a == b
    assert a.endswith("#meeting,work")


def test_create_round_defaults_and_f9_fields():
    """F9：register / difficulty 落库；缺省 auto / same。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            original=ORIGINAL,
            translation=TRANSLATION,
        )
    )
    assert doc["round_id"].startswith("rnd_")
    assert doc["turn_index"] == 0  # 第 1 轮由 append_turn 写入后前进
    assert doc["status"] == repo.STATUS_ACTIVE
    assert doc["max_turns"] == config.EXTENSION_ROUND_MAX_TURNS
    assert doc["register"] == REGISTER_AUTO
    assert doc["difficulty"] == DIFFICULTY_SAME
    assert doc["summary"] is None
    assert doc["expires_at"] - doc["created_at"] == config.EXTENSION_ROUND_TTL_HOURS * 3600 * 1000

    explicit = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            register="formal",
            difficulty=DIFFICULTY_HARDER,
        )
    )
    assert explicit["register"] == "formal"
    assert explicit["difficulty"] == DIFFICULTY_HARDER


def test_create_round_invalid_inputs():
    """入参校验：空 scholar_id / sentence_id / selected_ids、超量勾选 → INVALID_INPUT。"""
    db = FakeDB()
    base = dict(
        scholar_id=SCHOLAR,
        sentence_id=SENTENCE,
        selected_ids=["sent_1#0"],
        points_snapshot=POINTS,
    )
    cases = [
        {**base, "scholar_id": ""},
        {**base, "sentence_id": ""},
        {**base, "selected_ids": []},
        {
            **base,
            "selected_ids": ["sent_1#0"] * (config.EXTENSION_ROUND_MAX_SELECTED + 1),
        },
    ]
    for kwargs in cases:
        with pytest.raises(ExtensionError) as e:
            _run(repo.create_round(db, **kwargs))
        assert _code(e) == repo.ERR_INVALID_INPUT
    assert db.write_log == []


def test_get_round_expired_is_none():
    """TTL 过期 → None（路由侧映射 404，与不存在同码）。"""
    db = FakeDB()
    db.add(
        config.EXTENSION_ROUND_COLLECTION,
        {
            "round_id": "rnd_expired",
            "scholar_id": SCHOLAR,
            "status": repo.STATUS_ACTIVE,
            "expires_at": 1,
        },
    )
    assert _run(repo.get_round(db, "rnd_expired", scholar_id=SCHOLAR)) is None


def test_append_turn_server_side_index_and_cap():
    """turn_index 服务端按 len(turns)+1 计算，不信任入参。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            max_turns=3,
        )
    )
    first = _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    assert first["turn_index"] == 1
    assert first["turns"][0]["turn_index"] == 1
    assert first["turns"][0]["user_input"] is None
    assert first["turns"][0]["result"] is None
    assert first["used_prompts"] == [
        repo.build_prompt_fingerprint(PROMPTS[0]["prompt_zh"], PROMPTS[0]["scene_tag"])
    ]

    second = _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[1]))
    assert second["turn_index"] == 2
    assert len(second["used_prompts"]) == 2


def test_append_turn_result_mismatch():
    """轮次不符 → ROUND_TURN_MISMATCH（防重复 / 乱序提交）。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.append_turn_result(db, doc["round_id"], turn_index=1, result=_RUBRIC_PASS)
        )
    assert _code(e) == repo.ERR_ROUND_TURN_MISMATCH  # 尚无轮次

    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.append_turn_result(db, doc["round_id"], turn_index=2, result=_RUBRIC_PASS)
        )
    assert _code(e) == repo.ERR_ROUND_TURN_MISMATCH

    ok = _run(
        repo.append_turn_result(
            db, doc["round_id"], turn_index=1, result=_RUBRIC_PASS, user_input=ANSWER_OK
        )
    )
    assert ok["turns"][0]["result"] == _RUBRIC_PASS
    assert ok["turn_index"] == 1  # 写结果不前进轮次


def test_close_round_only_accepts_passed_or_exhausted():
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.close_round(
                db, doc["round_id"], status=repo.STATUS_ABANDONED, summary={}
            )
        )
    assert _code(e) == repo.ERR_INVALID_INPUT

    closed = _run(
        repo.close_round(
            db,
            doc["round_id"],
            status=repo.STATUS_PASSED,
            summary={"turns_used": 1, "passed": True},
        )
    )
    assert closed["status"] == repo.STATUS_PASSED
    assert closed["summary"]["passed"] is True


def test_state_machine_is_one_way():
    """状态机单向：终态后 append_turn / append_turn_result 一律 ROUND_CLOSED。"""
    db = FakeDB()
    doc = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
            max_turns=3,
        )
    )
    _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[0]))
    _run(
        repo.abandon_round(db, doc["round_id"], scholar_id=SCHOLAR, summary={"turns_used": 0})
    )

    with pytest.raises(ExtensionError) as e:
        _run(repo.append_turn(db, doc["round_id"], turn=PROMPTS[1]))
    assert _code(e) == repo.ERR_ROUND_CLOSED
    with pytest.raises(ExtensionError) as e:
        _run(
            repo.append_turn_result(db, doc["round_id"], turn_index=1, result=_RUBRIC_PASS)
        )
    assert _code(e) == repo.ERR_ROUND_CLOSED

    got = _run(repo.get_round(db, doc["round_id"], scholar_id=SCHOLAR))
    assert got["status"] == repo.STATUS_ABANDONED


def test_cleanup_expired_only_deletes_expired():
    db = FakeDB()
    fresh = _run(
        repo.create_round(
            db,
            scholar_id=SCHOLAR,
            sentence_id=SENTENCE,
            selected_ids=["sent_1#0"],
            points_snapshot=POINTS,
        )
    )
    db.add(
        config.EXTENSION_ROUND_COLLECTION,
        {"round_id": "rnd_old", "scholar_id": SCHOLAR, "expires_at": 1},
    )
    assert _run(repo.cleanup_expired(db)) == 1
    assert _run(repo.get_round(db, "rnd_old", scholar_id=SCHOLAR)) is None
    assert _run(repo.get_round(db, fresh["round_id"], scholar_id=SCHOLAR)) is not None


# ===========================================================================
# 出题：校验 / 去重（固化 B05 纯函数）
# ===========================================================================


def test_validate_prompt_item_phrase_substring_hit():
    """多词短语：子串命中即算（中文句里不得出现目标点英文原文）。"""
    r = validate_prompt_item(
        {"prompt_zh": "他终于把 take up 的话题转到了预算上。"}, ["take up"]
    )
    assert r["ok"] is False
    assert r["hits"] == ["take up"]
    assert "take up" in r["reason"]


def test_validate_prompt_item_word_boundary_avoids_false_hit():
    """单词按词边界命中：`art` 不误伤 `start`。"""
    r = validate_prompt_item({"prompt_zh": "Let's start the meeting now."}, ["art"])
    assert r["ok"] is True
    assert r["hits"] == []

    r2 = validate_prompt_item({"prompt_zh": "这是一幅 art 作品。"}, ["art"])
    assert r2["ok"] is False


def test_validate_prompt_item_reference_en_hit_is_observability_only():
    """`reference_en_hit` 是观测位：屈折形式判 False，但**不判失败**。"""
    ok = validate_prompt_item(
        {"prompt_zh": "会议拖得太久。", "reference_en": "He took part in the talk."},
        ["take part in"],
    )
    assert ok["ok"] is True
    assert ok["reference_en_hit"] is False  # took part in ≠ take part in

    hit = validate_prompt_item(
        {"prompt_zh": "会议拖得太久。", "reference_en": "He take part in the talk."},
        ["take part in"],
    )
    assert hit["ok"] is True
    assert hit["reference_en_hit"] is True


def test_is_repeat_prompt_fingerprint_equal():
    used = [repo.build_prompt_fingerprint("会议拖得太久。", ["work"])]
    assert is_repeat_prompt("会议拖得太久。", ["work"], used) is True


def test_is_repeat_prompt_text_similarity_over_threshold():
    """判据 2：文本相似度 ≥ 0.7（换标签也判重复）。"""
    used = [repo.build_prompt_fingerprint("会议拖得太久他终于把话题转到预算上", ["work"])]
    assert (
        is_repeat_prompt("会议拖得太久他终于把话题转到预算上", ["school"], used) is True
    )
    assert is_repeat_prompt("孩子们在操场上排好队等着老师发新书", ["school"], used) is False


def test_is_repeat_prompt_same_scene_lower_threshold():
    """判据 3：场景标签完全相同 + 相似度 ≥ 0.5 → 判重复（场景至少换一项）。"""
    # ratio = 2*10/(10+20) ≈ 0.667：低于文本阈值 0.7、高于场景阈值 0.5
    used = [repo.build_prompt_fingerprint("abcdefghij", ["work", "meeting"])]
    assert (
        is_repeat_prompt("abcdefghijklmnopqrst", ["meeting", "work"], used) is True
    )
    # 换一个场景标签 → 放行（场景 / 人物 / 动作至少换一项）
    assert (
        is_repeat_prompt("abcdefghijklmnopqrst", ["school", "kids"], used) is False
    )


def test_is_repeat_prompt_no_used_is_false():
    assert is_repeat_prompt("任何情景。", ["work"], []) is False
    assert is_repeat_prompt("", ["work"], ["x#work"]) is False


def test_generate_round_prompt_retry_once_then_flag_repeat_risk(monkeypatch):
    """去重兜底：重出 1 次后仍重复 → repeat_risk=True **放行不阻断**。"""
    stub = _install_prompt_stub(
        monkeypatch, [PROMPTS[0], PROMPTS[0]]
    )  # 两次都返回同一情景
    out = _run(
        generate_round_prompt(
            points=[POINTS[0]],
            used_prompts=[
                repo.build_prompt_fingerprint(PROMPTS[0]["prompt_zh"], ["work"])
            ],
        )
    )
    assert stub["count"] == 2  # 重出 1 次
    assert out["repeat_risk"] is True
    assert out["attempts"] == 2
    assert out["prompt_version"] == config.EXTENSION_ROUND_PROMPT_VERSION
    # 重出时 messages 带「必须换场景」的强化指令
    assert "【重出要求】" in stub["messages"][1][-1]["content"]


def test_generate_round_prompt_must_use_excludes_mine_points(monkeypatch):
    """自建点（origin=mine）不进 must_use，也不参与 banned_check。"""
    _install_prompt_stub(monkeypatch, [PROMPTS[0]])
    out = _run(
        generate_round_prompt(
            points=[POINTS[0], {"id": "mine_1", "text": "scarf", "origin": "mine"}]
        )
    )
    assert out["must_use"] == ["take part in"]
    assert out["banned_check"]["ok"] is True


def test_build_round_prompt_messages_first_turn_has_empty_focus_hint():
    """第 1 轮：情景贴近原句语境，focus_hint 恒为空串。"""
    messages = build_round_prompt_messages(
        points=[POINTS[0]], original=ORIGINAL, translation=TRANSLATION, turn_index=1
    )
    content = messages[-1]["content"]
    assert "第 1 轮" in content
    assert ORIGINAL in content
    assert "（无，这是第 1 轮）" in content
    assert messages[0]["role"] == "system"


def test_build_round_prompt_messages_carries_prev_errors_and_harder():
    """后续轮：带上轮作答与 errors，harder 时输出「升一级语域」。"""
    messages = build_round_prompt_messages(
        points=[POINTS[0]],
        used_prompts=["会议#work"],
        prev_turn={"user_input": ANSWER_OK, "errors": ["时态不一致"]},
        difficulty=DIFFICULTY_HARDER,
        turn_index=2,
    )
    content = messages[-1]["content"]
    assert ANSWER_OK in content
    assert "时态不一致" in content
    assert "升一级语域" in content


# ===========================================================================
# 判分 / 决策 / 小结（固化 B06）
# ===========================================================================


def test_judge_round_turn_passed_is_decided_by_backend(monkeypatch):
    """passed 由后端判定：total ≥ PASS_SCORE **且** must_use_hit。"""
    _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    out = _run(
        judge_round_turn(
            points=[POINTS[0]], user_input=ANSWER_OK, prompt_zh="会议拖得太久。"
        )
    )
    assert out["score"] == 8
    assert out["must_use_hit"] is True
    assert out["passed"] is True


def test_judge_round_turn_score_below_pass(monkeypatch):
    _install_judge_stub(monkeypatch, [_RUBRIC_FAIL])
    out = _run(judge_round_turn(points=[POINTS[0]], user_input=ANSWER_OK))
    assert out["score"] == 4
    assert out["passed"] is False


def test_judge_round_turn_must_use_miss_blocks_pass(monkeypatch):
    """高分但没用上目标点 → 不达标（不信 LLM 自述）。"""
    high = {**_RUBRIC_PASS, "total": 10, "passed": True}  # LLM 自述达标
    _install_judge_stub(monkeypatch, [high])
    out = _run(judge_round_turn(points=[POINTS[0]], user_input=ANSWER_MISS))
    assert out["must_use_hit"] is False
    assert out["passed"] is False


def test_judge_round_turn_model_sentence_falls_back_to_reference(monkeypatch):
    """D5：LLM 未给示范句 → 用 reference_en 兜底。"""
    _install_judge_stub(monkeypatch, [{**_RUBRIC_PASS, "model_sentence": ""}])
    out = _run(
        judge_round_turn(
            points=[POINTS[0]],
            user_input=ANSWER_OK,
            prompt_zh="会议拖得太久。",
            reference_en="He took part in the talk.",
        )
    )
    assert out["model_sentence"] == "He took part in the talk."


def test_judge_round_turn_passes_round_context(monkeypatch):
    """多轮判分带语义锚点（prompt_zh + reference_en）；E2 单轮不传。"""
    calls = _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    _run(
        judge_round_turn(
            points=[POINTS[0]],
            user_input=ANSWER_OK,
            prompt_zh="会议拖得太久。",
            reference_en="He took part in the talk.",
        )
    )
    assert calls["contexts"][0] == {
        "prompt_zh": "会议拖得太久。",
        "reference_en": "He took part in the talk.",
    }

    calls2 = _install_judge_stub(monkeypatch, [_RUBRIC_PASS])
    _run(judge_round_turn(points=[POINTS[0]], user_input=ANSWER_OK))
    assert calls2["contexts"][0] is None  # 无情景 → 走 E2 原分支


def test_decide_next_action_four_branches():
    cap = config.EXTENSION_ROUND_MAX_TURNS_HARD
    assert decide_next_action(
        turn_index=1, max_turns=cap, score=8, must_use_hit=True
    )["action"] == ACTION_FINISH_PASSED
    assert decide_next_action(
        turn_index=1, max_turns=cap, score=4, must_use_hit=True
    )["action"] == ACTION_CONTINUE
    assert decide_next_action(
        turn_index=cap, max_turns=cap, score=4, must_use_hit=True
    )["action"] == ACTION_FINISH_MAX
    assert decide_next_action(
        turn_index=1, max_turns=cap, score=8, must_use_hit=True, failed=True
    )["action"] == ACTION_FAILED
    # 高分但未命中目标点 → 不达标
    assert decide_next_action(
        turn_index=1, max_turns=cap, score=10, must_use_hit=False
    )["action"] == ACTION_CONTINUE


def test_build_summary_top_errors_and_fallback():
    turns = [
        {
            "reference_en": "He took part in the talk.",
            "result": {"score": 4, "errors": ["时态不一致", "搭配生硬"], "model_sentence": ""},
        },
        {
            "reference_en": "She takes part in it.",
            "result": {
                "score": 6,
                "errors": ["时态不一致", "语序"],
                "model_sentence": "She takes part in the meeting.",
            },
        },
        {"reference_en": "", "result": None},  # 未判分的作废轮：不计
    ]
    s = build_summary(turns)
    assert s["turns_used"] == 2
    assert s["best_score"] == 6
    assert s["passed"] is False  # 末轮 result 无 passed 字段 → 按 False
    assert s["top_errors"][0] == "时态不一致"  # 出现 2 次，排最前
    # 第 1 轮无示范句 → reference_en 兜底
    assert s["model_sentences"] == [
        "He took part in the talk.",
        "She takes part in the meeting.",
    ]


def test_build_summary_passed_from_terminal_status():
    turns = [{"reference_en": "", "result": {"score": 8, "errors": [], "model_sentence": "m1"}}]
    assert build_summary(turns, passed=True)["passed"] is True
    assert build_summary(turns, passed=False)["passed"] is False
    assert build_summary([]) == {
        "turns_used": 0,
        "passed": False,
        "best_score": 0,
        "top_errors": [],
        "model_sentences": [],
    }
