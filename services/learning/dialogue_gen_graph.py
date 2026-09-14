"""T3 LangGraph 对话生成图 — v2 调整稿（每句学习语句扇出评估·自动选优·背景与角色后置）。

设计稿：`docs_v1/AI会话/AI英语对话生成设计-调整稿（每句学习语句扇出评估·自动选优·背景与角色后置）.md`

图结构（v2 / §2.2）::

    START
      → load_context                  # 任务组必用句 + 召回 + 语言/形态
      → evaluate_per_sentence × N     # Send 扇出：每条学习语句独立 LLM 调用
          │   评估 viable + recalled_used + turns(草案) + naturalness
          │   写入 candidate_dialogues（reducer=operator.add）
      → select_best                   # 纯函数：natural_first / score_first
          ├─ 有 viable 候选 → summarize → build_prompt → generate_dialogue
          │                   → validate_coverage →(evaluate|refine|fallback)
          │                   → evaluate → persist → END
          └─ 无 viable 候选 → fallback_non_dialogue → evaluate → persist → END

设计要点：
- **扇出/扇入**：`add_conditional_edges("load_context", dispatch_per_sentence, [Send(...)])`
  触发 N 个并行 `evaluate_per_sentence`；`candidate_dialogues` 用
  `Annotated[list[dict], operator.add]` 累积；`Send` 由 LangGraph 自动 join 到 `select_best`；
- **角色不再预设**：`summarize` 节点对选中 turns 反推 `{background, roles}`（1 次 LLM）；
  `build_prompt` 使用 summarize 产出的角色与背景，未走 summarize 时回退 state.roles/scenario；
- **覆盖校验闭环**：未覆盖 → `refine_prompt` → `generate_dialogue`（仅重生成选中对话环节）；
  用尽 → `fallback_non_dialogue`（全部必用句进题面）；
- **`DIALOGUE_GEN_GRAPH_ENABLED=0`** 时任务执行器回退 T1 单函数直连（`generate_dialogue`）；
- **`DIALOGUE_GEN_PARALLEL_ENABLED=0`** 时 `dispatch_per_sentence` 退化为顺序调用
  （不走 Send，本地调试与控成本）；
- **T4 断点续写**：`NoSQLCheckpointSaver`，集合 `ai_dialogue_checkpoint`，
  `thread_id = dg_<task_id>`：每节点执行后自动落 checkpoint，`resume` 从最近
  （或指定）checkpoint 续跑。
"""
from __future__ import annotations

import asyncio
import functools
import logging
import operator
from typing import Annotated, Any, Awaitable, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from config import (
    DIALOGUE_CHECKPOINT_COLLECTION,
    DIALOGUE_GEN_CHECKPOINT_ENABLED,
    DIALOGUE_GEN_MAX_CONCURRENCY,
    DIALOGUE_GEN_MAX_RETRY,
    DIALOGUE_GEN_PARALLEL_ENABLED,
)
from services.models_content import get_sentences_by_ids
from services.models_learning import get_skill_states
from services.learning.conversation_graph import NoSQLCheckpointSaver
from services.learning.rag_retriever import get_curriculum_retriever
from services.providers.dialogue_gen import (
    DEFAULT_SELECT_STRATEGY,
    ERR_COVERAGE_FAILED,
    ERR_LLM_PARSE_ERROR,
    ERR_NETWORK_ERROR,
    RECALL_MAX,
    RECALL_MIN,
    SELECT_STRATEGIES,
    STAGE_COVERAGE,
    STAGE_LLM,
    STAGE_PARSE,
    SUPPORTED_PROMPT_LANGS,
    DialogueGenError,
    basic_metrics,
    build_dialogue_messages,
    build_non_dialogue_fallback,
    build_per_sentence_evaluate_messages,
    build_result,
    build_summarize_messages,
    choose_content_type,
    clamp_top_k,
    compute_coverage,
    compute_missing_ids,
    invoke_dialogue_llm,
    merge_notes,
    normalize_roles,
    parse_dialogue_output,
    parse_per_sentence_evaluation,
    parse_summary,
    select_best,
    task_group_sentence_ids,
)

logger = logging.getLogger("scholar-admin.dialogue_gen_graph")

# 条件边分支名（validate_coverage 之后）
BRANCH_EVALUATE = "evaluate"
BRANCH_REFINE = "refine"
BRANCH_FALLBACK = "fallback"

# 条件边分支名（select_best / summarize 之后）
BRANCH_SUMMARIZE = "summarize"
BRANCH_SELECT_FALLBACK = "select_fallback"
BRANCH_BUILD_PROMPT = "build_prompt"
BRANCH_SUMMARIZE_FALLBACK = "summarize_fallback"

# 图状态追加的 notes 标记（契约字段 notes 内逗号分隔留痕）
NOTE_RECALL_INSUFFICIENT = "recall_insufficient"
NOTE_COVERAGE_INCOMPLETE = "coverage_incomplete"
NOTE_RESUME_WITHOUT_CHECKPOINT = "resume_without_checkpoint"
NOTE_NO_VIABLE_CANDIDATE = "no_viable_candidate"
NOTE_SUMMARIZE_FAILED = "summarize_failed"

# checkpoint 时间线的语义阶段（随节点写入图状态，供 /checkpoints 展示）
STAGE_CONTEXT_LOADED = "context_loaded"
STAGE_EVALUATED_PER_SENTENCE = "evaluated_per_sentence"
STAGE_SELECTED = "selected"
STAGE_SUMMARIZED = "summarized"
STAGE_PROMPT_BUILT = "prompt_built"
STAGE_GENERATED = "generated"
STAGE_VALIDATED = "validated"
STAGE_REFINED = "refined"
STAGE_FALLBACK = "fallback"
STAGE_EVALUATED = "evaluated"
STAGE_PERSISTED = "persisted"
STAGE_START = "start"


class DialogueGenGraphState(TypedDict, total=False):
    """生成图状态（LangGraph channel values）。

    字段分三类：
    - 任务级：一次 invoke 固定（task_id / scholar_id / context / preferred_type）；
    - 节点中间产物：load/evaluate_per_sentence/select_best/summarize/build/generate/...
      逐节点填充；
    - 终态：persist 产出的 §4.6 `result`。
    """

    # 任务级
    task_id: str
    scholar_id: str
    context: dict
    preferred_type: str

    # checkpoint 时间线阶段（T4：随节点写入，供 /checkpoints 展示）
    stage: str

    # load_context
    required_ids: list[str]
    roles: list[dict]              # v2 仅作旧请求兼容；新流程默认由 summarize 产出
    prompt_lang: str
    content_type: str
    recalled: list[dict]
    recall_ids: list[str]
    context_notes: list[str]

    # evaluate_per_sentence（v2 扇出 / 扇入）
    # type: ignore 解释：LangGraph 状态使用 reducer，需配合 Annotated
    candidate_dialogues: Annotated[list[dict], operator.add]

    # select_best
    selected_index: int | None
    selected: dict
    selection: dict

    # summarize
    summary: dict | None           # {background, roles}

    # build_prompt / refine_prompt
    messages: list[dict]
    refine_missing: list[str]

    # generate_dialogue
    raw: str | None
    parsed: dict | None

    # validate_coverage
    missing_ids: list[str]
    coverage_ok: bool
    retry_count: int

    # fallback
    fallback: bool

    # evaluate
    coverage: dict
    metrics: dict

    # persist
    result: dict


# ---------------------------------------------------------------------------
# 节点：load_context（装载任务组 + 召回 + 角色/背景/语言/形态）
# ---------------------------------------------------------------------------


def _normalize_recall_hits(hits: Any) -> list[dict]:
    """召回结果规整为 `{sentence_id, content, lesson_id?}`（去重保序、剔除空内容）。"""
    result: list[dict] = []
    seen: set[str] = set()
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        sid = str(hit.get("sentence_id") or "").strip()
        content = str(hit.get("content") or hit.get("text") or "").strip()
        if not sid or not content or sid in seen:
            continue
        seen.add(sid)
        item = {"sentence_id": sid, "content": content}
        if hit.get("lesson_id"):
            item["lesson_id"] = hit["lesson_id"]
        result.append(item)
        if len(result) >= RECALL_MAX:
            break
    return result


async def _backfill_recall(
    *,
    db: Any,
    scholar_id: str,
    lesson_id: str | None,
    exclude_ids: set[str],
    existing: list[dict],
    need: int,
) -> list[dict]:
    """§4.2 召回不足时补入「任务组所在课的相邻已学句」（best-effort，失败不阻断主链路）。"""
    if db is None or not lesson_id or need <= 0:
        return existing
    try:
        states = (await get_skill_states(db, scholar_id=scholar_id)).get("records") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dialogue_gen_graph] 召回补入读 skill_state 失败: %s", exc)
        return existing
    candidate_ids: list[str] = []
    for state in states:
        sid = str(state.get("sentence_id") or "").strip()
        if (
            not sid
            or sid in exclude_ids
            or sid in candidate_ids
            or str(state.get("lesson_id") or "") != str(lesson_id)
        ):
            continue
        candidate_ids.append(sid)
    if not candidate_ids:
        return existing
    try:
        sentences = await get_sentences_by_ids(db, candidate_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dialogue_gen_graph] 召回补入读句子失败: %s", exc)
        return existing
    hits = list(existing)
    for sentence in sentences:
        if len(hits) >= need:
            break
        sid = str(sentence.get("sentence_id") or "").strip()
        content = str(sentence.get("text") or "").strip()
        if not sid or not content or sid in exclude_ids:
            continue
        hits.append(
            {
                "sentence_id": sid,
                "content": content,
                "lesson_id": sentence.get("lesson_id"),
            }
        )
        exclude_ids.add(sid)
    return hits


async def _load_recall(
    *,
    db: Any,
    retriever: Any,
    context: dict,
    task_group: dict,
    scholar_id: str,
    required_ids: list[str],
) -> tuple[list[dict], list[str]]:
    """装载召回学习语句（§4.2）：查询 = 背景 + weakSkills；候选 = 已学句（排除当前课）。

    Returns:
        (recalled, notes)：recalled 为 `{sentence_id, content}` 列表；notes 为降级留痕。
    """
    recall_cfg = context.get("recall") or {}
    if not recall_cfg.get("enabled"):
        return [], []

    top_k = clamp_top_k(recall_cfg.get("top_k"))
    engine = retriever or get_curriculum_retriever()
    background = str((context.get("scenario") or {}).get("background") or "").strip()
    weak_skills = [str(x) for x in (context.get("metrics") or {}).get("weak_skills") or []]
    query = " ".join(part for part in (background, " ".join(weak_skills)) if part).strip()
    lesson_id = str(task_group.get("lesson_id") or "").strip() or None
    exclude_lessons = [lesson_id] if lesson_id else []

    try:
        hits = await engine.retrieve(
            db,
            scholar_id=scholar_id,
            query=query,
            top_k=top_k,
            exclude_lesson_ids=exclude_lessons,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("[dialogue_gen_graph] RAG 召回失败（降级空召回）: %s", exc)
        hits = []

    recalled = _normalize_recall_hits(hits)
    notes: list[str] = []
    if len(recalled) < RECALL_MIN:
        exclude = set(required_ids) | {item["sentence_id"] for item in recalled}
        recalled = await _backfill_recall(
            db=db,
            scholar_id=scholar_id,
            lesson_id=lesson_id,
            exclude_ids=exclude,
            existing=recalled,
            need=RECALL_MIN,
        )
        if len(recalled) < RECALL_MIN:
            notes.append(NOTE_RECALL_INSUFFICIENT)
    return recalled, notes


async def node_load_context(
    state: DialogueGenGraphState, *, db: Any = None, retriever: Any = None
) -> dict:
    """装载上下文：任务组必用句 + 角色（旧请求兼容）+ 语言 + 形态 + RAG 召回。"""
    context = state.get("context") or {}
    task_group = context.get("task_group") or {}
    required_ids = task_group_sentence_ids(task_group)
    if not required_ids:
        raise DialogueGenError(
            ERR_COVERAGE_FAILED, STAGE_COVERAGE, "task_group 无可校验句子（sentences 为空）"
        )

    raw_lang = str(context.get("prompt_lang") or "zh").strip().lower()
    prompt_lang = raw_lang if raw_lang in SUPPORTED_PROMPT_LANGS else "zh"
    # v2：暂时不锁定形态（角色由 summarize 后置给出），统一按 dialogue 走，
    # 后续 validate_coverage 时若必要可降级（generation 路径沿用 T3）。
    roles = normalize_roles(context.get("roles"))
    content_type = choose_content_type(state.get("preferred_type") or "auto", roles)

    recalled, notes = await _load_recall(
        db=db,
        retriever=retriever,
        context=context,
        task_group=task_group,
        scholar_id=str(state.get("scholar_id") or ""),
        required_ids=required_ids,
    )
    return {
        "required_ids": required_ids,
        "roles": roles,
        "prompt_lang": prompt_lang,
        "content_type": content_type,
        "recalled": recalled,
        "recall_ids": [item["sentence_id"] for item in recalled],
        "context_notes": notes,
        "stage": STAGE_CONTEXT_LOADED,
    }


# ---------------------------------------------------------------------------
# 节点：evaluate_per_sentence（v2 扇出） + dispatch + select_best
# ---------------------------------------------------------------------------


def _look_up_sentence(context: dict, sentence_id: str) -> dict | None:
    """从任务组找单条 sentence（v2 evaluate_per_sentence 输入）。"""
    for s in (context.get("task_group") or {}).get("sentences") or []:
        if isinstance(s, dict) and str(s.get("sentence_id") or "").strip() == sentence_id:
            return {
                "sentence_id": sentence_id,
                "content": str(s.get("content") or "").strip(),
            }
    return None


def dispatch_per_sentence(
    state: DialogueGenGraphState,
    *,
    parallel_enabled: bool = DIALOGUE_GEN_PARALLEL_ENABLED,
    max_concurrency: int = DIALOGUE_GEN_MAX_CONCURRENCY,
) -> list[Send]:
    """Send 扇出路由（v2 调整稿 §2.2）。

    返回 `list[Send]` 触发并行 `evaluate_per_sentence × N`；N==0 的场景由
    `node_load_context` 抛 `COVERAGE_FAILED`（v2 设计稿同 T3），不会到达本函数。
    `parallel_enabled` 与 `max_concurrency` 仅用于节点内 `asyncio.Semaphore` 限流与调度，
    Send 顺序无关（LangGraph 在 superstep 内 join）。
    """
    required_ids = state.get("required_ids") or []
    return [
        Send(
            "evaluate_per_sentence",
            {**state, "_learning_sentence_id": sid, "_branch_index": i},
        )
        for i, sid in enumerate(required_ids)
    ]


async def node_evaluate_per_sentence(
    state: DialogueGenGraphState,
    *,
    generator: Any = None,
    timeout_seconds: int | None = None,
    max_concurrency: int = DIALOGUE_GEN_MAX_CONCURRENCY,
) -> dict:
    """单条学习语句评估（v2 设计稿 §3.1，LLM#N）。

    由 `dispatch_per_sentence` 用 `Send` 触发；并发出 N 个分支各自返回 `candidate_dialogues`
    的单元素列表，LangGraph 用 `Annotated[..., operator.add]` reducer 累积为整组候选。
    注意：本节点**不写** `stage`，由 join 节点 `select_best` 写
    （避免并行分支对单一 `stage` channel 写入冲突）。
    解析失败 → `LLM_PARSE_ERROR`（不静默降级，保证数据可信）。
    """
    # `Send` 会把分支标记的字段混入 state；其余字段复用上游 state
    learning_sentence_id = str(state.get("_learning_sentence_id") or "")
    required_ids = state.get("required_ids") or []
    recall_ids = state.get("recall_ids") or []
    recalled = state.get("recalled") or []
    context = state.get("context") or {}

    if not learning_sentence_id:
        return {"candidate_dialogues": []}

    learning_sentence = _look_up_sentence(context, learning_sentence_id)
    if learning_sentence is None:
        return {"candidate_dialogues": []}

    sem = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def _run_one() -> dict:
        async with sem:
            messages = build_per_sentence_evaluate_messages(
                context=context,
                learning_sentence=learning_sentence,
                recalled=recalled,
            )
            content = await invoke_dialogue_llm(
                messages, generator=generator, timeout_seconds=timeout_seconds
            )
            result = parse_per_sentence_evaluation(
                content,
                learning_sentence_id=learning_sentence_id,
                required_set=set(required_ids),
                recall_set=set(recall_ids),
            )
            if result is None:
                raise DialogueGenError(
                    ERR_LLM_PARSE_ERROR,
                    STAGE_PARSE,
                    f"evaluate_per_sentence 解析失败: {content[:200]}",
                )
            return result

    candidate = await _run_one()
    return {"candidate_dialogues": [candidate]}


def node_select_best(state: DialogueGenGraphState) -> dict:
    """选优（v2 设计稿 §2.5，纯函数）：naturalness 降序，turn_count 降序保稳。"""
    candidates = state.get("candidate_dialogues") or []
    strategy = DEFAULT_SELECT_STRATEGY  # v2 当前不做策略参数化
    selected, selection = select_best(candidates, strategy=strategy)
    if selected is None:
        return {
            "selected_index": None,
            "selected": {},
            "selection": selection,
            "stage": STAGE_SELECTED,
        }
    return {
        "selected_index": selection.get("selected_index"),
        "selected": selected,
        "selection": selection,
        "stage": STAGE_SELECTED,
    }


def route_after_select_best(state: DialogueGenGraphState) -> str:
    """选优后路由：有 viable 候选 → summarize；无候选 → fallback_non_dialogue（v2 §2.2）。"""
    if state.get("selected"):
        return BRANCH_SUMMARIZE
    return BRANCH_SELECT_FALLBACK


# ---------------------------------------------------------------------------
# 节点：summarize（1 次 LLM；选中 turns → {background, roles}）
# ---------------------------------------------------------------------------


async def node_summarize(
    state: DialogueGenGraphState,
    *,
    generator: Any = None,
    timeout_seconds: int | None = None,
) -> dict:
    """背景 + 角色后置总结（v2 设计稿 §3.2，LLM#1）。

    失败 / roles 空 → 返回 summary=None，让 route_after_summarize 走 fallback。
    """
    selected = state.get("selected") or {}
    if not selected:
        return {
            "summary": None,
            "stage": STAGE_SUMMARIZED,
        }
    context = state.get("context") or {}
    learning_sentence_id = str(selected.get("learning_sentence_id") or "").strip()
    learning_sentence = (
        _look_up_sentence(context, learning_sentence_id) if learning_sentence_id else None
    )

    messages = build_summarize_messages(
        selected=selected,
        learning_sentence=learning_sentence,
        recalled=state.get("recalled"),
        task_group=context.get("task_group"),
    )
    try:
        content = await invoke_dialogue_llm(
            messages, generator=generator, timeout_seconds=timeout_seconds
        )
        summary = parse_summary(content)
    except DialogueGenError:
        raise
    except Exception:  # noqa: BLE001
        summary = None

    if summary is None:
        return {
            "summary": None,
            "stage": STAGE_SUMMARIZED,
        }
    return {
        "summary": summary,
        "stage": STAGE_SUMMARIZED,
    }


def route_after_summarize(state: DialogueGenGraphState) -> str:
    """summarize 后路由：summary 成功 → build_prompt；失败 → fallback。"""
    if state.get("summary"):
        return BRANCH_BUILD_PROMPT
    return BRANCH_SUMMARIZE_FALLBACK


# ---------------------------------------------------------------------------
# 节点：build_prompt / refine_prompt（使用 summarize 产出的 roles/background）
# ---------------------------------------------------------------------------


def _scenario_with_summary(context: dict, summary: dict | None) -> dict:
    """将 `summary.background` 合并进 scenario（缺省保留原 scenario）。"""
    scenario = dict(context.get("scenario") or {})
    if isinstance(summary, dict) and summary.get("background"):
        scenario["background"] = summary["background"]
    return scenario


def _roles_with_summary(state: dict, summary: dict | None) -> list[dict]:
    """优先使用 summary.roles，缺省回退 state.roles（旧请求兼容）。"""
    if isinstance(summary, dict) and isinstance(summary.get("roles"), list):
        roles = summary["roles"]
        if roles:
            return roles
    return state.get("roles") or []


def _messages_for(
    state: DialogueGenGraphState, *, refine_missing: list[str] | None = None
) -> list[dict]:
    """组装 prompt messages，角色与背景来自 summarize 节点（v2 调整稿 §2.3）。"""
    context = dict(state.get("context") or {})
    summary = state.get("summary")
    context["scenario"] = _scenario_with_summary(context, summary)
    roles = _roles_with_summary(state, summary)
    return build_dialogue_messages(
        context,
        content_type=state.get("content_type") or "dialogue",
        roles=roles,
        prompt_lang=state.get("prompt_lang") or "zh",
        recalled=state.get("recalled"),
        refine_missing=refine_missing,
    )


def node_build_prompt(state: DialogueGenGraphState) -> dict:
    """首轮 Prompt 组装（基于 summarize 产出的角色与背景）。"""
    return {"messages": _messages_for(state), "stage": STAGE_PROMPT_BUILT}


def node_refine_prompt(state: DialogueGenGraphState) -> dict:
    """覆盖校验未过：把缺失必用句回灌 Prompt，并消耗一次重试预算。"""
    missing = [str(x) for x in (state.get("missing_ids") or [])]
    retry_count = int(state.get("retry_count") or 0) + 1
    return {
        "messages": _messages_for(state, refine_missing=missing),
        "refine_missing": missing,
        "retry_count": retry_count,
        "stage": STAGE_REFINED,
    }


# ---------------------------------------------------------------------------
# 节点：generate_dialogue（单次 LLM + 解析）
# ---------------------------------------------------------------------------


async def node_generate_dialogue(
    state: DialogueGenGraphState,
    *,
    generator: Any = None,
    timeout_seconds: int | None = None,
) -> dict:
    """单次 LLM 调用 + JSON 解析（注入 fake_generator；超时/空值语义见 invoke_dialogue_llm）。"""
    content = await invoke_dialogue_llm(
        state.get("messages") or [],
        generator=generator,
        timeout_seconds=timeout_seconds,
    )
    parsed = parse_dialogue_output(
        content,
        required_ids=state.get("required_ids") or [],
        role_codes=[r["code"] for r in (state.get("roles") or [])],
        prompt_lang=state.get("prompt_lang") or "zh",
        recall_ids=state.get("recall_ids") or [],
    )
    if parsed is None:
        raise DialogueGenError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, f"模型输出解析失败: {content[:200]}"
        )
    return {"raw": content, "parsed": parsed, "stage": STAGE_GENERATED}


# ---------------------------------------------------------------------------
# 节点：validate_coverage + 条件边
# ---------------------------------------------------------------------------


def node_validate_coverage(state: DialogueGenGraphState) -> dict:
    """覆盖校验：任务组必用句是否全部出现在 used_sentence_ids。"""
    parsed = state.get("parsed") or {}
    required_ids = state.get("required_ids") or []
    missing = compute_missing_ids(required_ids, parsed.get("used_sentence_ids") or [])
    return {
        "missing_ids": missing,
        "coverage_ok": len(missing) == 0,
        "retry_count": int(state.get("retry_count") or 0),
        "stage": STAGE_VALIDATED,
    }


def route_after_validate(
    state: DialogueGenGraphState, *, max_retry: int = DIALOGUE_GEN_MAX_RETRY
) -> str:
    """覆盖校验后路由：通过 → evaluate；未过且有余量 → refine；用尽 → fallback（降级）。"""
    if state.get("coverage_ok"):
        return BRANCH_EVALUATE
    if int(state.get("retry_count") or 0) < int(max_retry):
        return BRANCH_REFINE
    return BRANCH_FALLBACK


def node_fallback_non_dialogue(state: DialogueGenGraphState) -> dict:
    """覆盖重试用尽 / 无 viable 候选 → 规则降级为非对话（retell），全部必用句进题面保证可产出。

    按来源留痕归因（v2）：selected 为空 → `no_viable_candidate`；
    selected 非空但 summary 失败 → `summarize_failed`；覆盖重试用尽不额外归因。
    """
    context = state.get("context") or {}
    task_group = context.get("task_group") or {}
    summary = state.get("summary") or {}
    fallback_roles = summary.get("roles") if isinstance(summary, dict) else None
    parsed = build_non_dialogue_fallback(
        state.get("parsed"),
        required_sentences=task_group.get("sentences") or [],
        recalled_sentences=state.get("recalled") or [],
        prompt_lang=state.get("prompt_lang") or "zh",
        background_intro=summary.get("background") if isinstance(summary, dict) else None,
        roles=fallback_roles,
    )
    context_notes = list(state.get("context_notes") or [])
    selected = state.get("selected") or {}
    if not selected and not summary:
        context_notes.append(NOTE_NO_VIABLE_CANDIDATE)
    elif selected and not summary:
        context_notes.append(NOTE_SUMMARIZE_FAILED)
    return {
        "parsed": parsed,
        "fallback": True,
        "stage": STAGE_FALLBACK,
        "context_notes": context_notes,
    }


# ---------------------------------------------------------------------------
# 节点：evaluate（本地 L1 指标；L2 Judge 属后续任务）
# ---------------------------------------------------------------------------


async def node_evaluate(state: DialogueGenGraphState, *, evaluator: Any = None) -> dict:
    """评估生成内容 → 指标卡（缺省本地 L1 `basic_metrics`；可注入 evaluator 扩 L2）。"""
    parsed = state.get("parsed") or {}
    required_ids = state.get("required_ids") or []
    coverage = compute_coverage(required_ids, parsed.get("used_sentence_ids") or [])
    turn_count = len(parsed.get("turns") or [])
    metrics = basic_metrics(coverage, turn_count)
    if evaluator is not None:
        result = evaluator(parsed, coverage)
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            metrics = result
    return {"coverage": coverage, "metrics": metrics, "stage": STAGE_EVALUATED}


# ---------------------------------------------------------------------------
# 节点：persist（组装 §4.6 result，含 summary；v2 删除 scene_* / generation 写入）
# ---------------------------------------------------------------------------


def node_persist(state: DialogueGenGraphState) -> dict:
    """组装成功 result（v2 删除 `scene_candidates/selection/generation`，新增 `summary`）。

    `roles` 优先使用 summary.roles（v2 后置总结），缺省回退 state.roles（旧请求兼容）。
    `background_intro` 使用 summary.background（如果有）。
    """
    parsed = state.get("parsed") or {}
    required_ids = state.get("required_ids") or []
    summary = state.get("summary")
    notes = list(state.get("context_notes") or [])
    is_fallback = bool(state.get("fallback"))
    if is_fallback:
        notes.append("fallback_non_dialogue")
    # v2：无候选 / 总结失败的归因 note 由 `node_fallback_non_dialogue` 按
    # 来源写入 context_notes（fallback 节点能区分路由来源，persist 不再推断）。

    roles_for_result = _roles_with_summary(state, summary)
    background_intro = (
        summary.get("background") if isinstance(summary, dict) else parsed.get("background_intro")
    )

    # 生成 result 的 background_intro 字段保持契约（兼容旧客户端）
    parsed_for_result = dict(parsed)
    if isinstance(summary, dict) and summary.get("background"):
        parsed_for_result["background_intro"] = background_intro

    coverage = compute_coverage(required_ids, parsed.get("used_sentence_ids") or [])
    if coverage["required_used"] < coverage["required_total"]:
        notes.append(NOTE_COVERAGE_INCOMPLETE)

    result = build_result(
        parsed_for_result,
        required_ids=required_ids,
        roles=roles_for_result,
        retry_count=int(state.get("retry_count") or 0),
        metrics=state.get("metrics"),
        notes=notes,
        summary=summary,
    )
    logger.info(
        "[dialogue_gen_graph] persist → content_type=%s, coverage=%d/%d, retry=%d, notes=%s",
        result["content_type"],
        coverage["required_used"],
        coverage["required_total"],
        result["retry_count"],
        result["notes"],
    )
    return {"result": result, "stage": STAGE_PERSISTED}


# ---------------------------------------------------------------------------
# 图构建 / 编译
# ---------------------------------------------------------------------------


def build_dialogue_gen_graph(
    *,
    db: Any = None,
    retriever: Any = None,
    generator: Any = None,
    evaluator: Any = None,
    max_retry: int | None = None,
    timeout_seconds: int | None = None,
    parallel_enabled: bool = DIALOGUE_GEN_PARALLEL_ENABLED,
    max_concurrency: int | None = None,
) -> StateGraph:
    """构建 v2 生成图（依赖经 partial 注入，单测传 fake）。

    `DIALOGUE_GEN_GRAPH_ENABLED=0` 时任务执行器回退 T1 直连生成（行为不变）；
    `DIALOGUE_GEN_PARALLEL_ENABLED=0` 时扇出退化为顺序执行（仅用于本地调试 / 控成本）。
    """
    retry_budget = DIALOGUE_GEN_MAX_RETRY if max_retry is None else int(max_retry)
    concurrency = DIALOGUE_GEN_MAX_CONCURRENCY if max_concurrency is None else int(max_concurrency)
    parallel = bool(parallel_enabled)

    workflow = StateGraph(DialogueGenGraphState)
    workflow.add_node(
        "load_context", functools.partial(node_load_context, db=db, retriever=retriever)
    )
    workflow.add_node(
        "evaluate_per_sentence",
        functools.partial(
            node_evaluate_per_sentence,
            generator=generator,
            timeout_seconds=timeout_seconds,
            max_concurrency=concurrency,
        ),
    )
    workflow.add_node("select_best", node_select_best)
    workflow.add_node(
        "summarize",
        functools.partial(
            node_summarize, generator=generator, timeout_seconds=timeout_seconds
        ),
    )
    workflow.add_node("build_prompt", node_build_prompt)
    workflow.add_node(
        "generate_dialogue",
        functools.partial(
            node_generate_dialogue, generator=generator, timeout_seconds=timeout_seconds
        ),
    )
    workflow.add_node("validate_coverage", node_validate_coverage)
    workflow.add_node("refine_prompt", node_refine_prompt)
    workflow.add_node("fallback_non_dialogue", node_fallback_non_dialogue)
    workflow.add_node("evaluate", functools.partial(node_evaluate, evaluator=evaluator))
    workflow.add_node("persist", node_persist)

    workflow.add_edge(START, "load_context")
    workflow.add_conditional_edges(
        "load_context",
        functools.partial(
            dispatch_per_sentence,
            parallel_enabled=parallel,
            max_concurrency=concurrency,
        ),
        # LangGraph：用 list[Send] 时条件边函数返回 list 会被自动派发；
        # N==0 由 node_load_context 抛 COVERAGE_FAILED，不会到本函数。
        ["evaluate_per_sentence"],
    )
    workflow.add_edge("evaluate_per_sentence", "select_best")
    workflow.add_conditional_edges(
        "select_best",
        route_after_select_best,
        {
            BRANCH_SUMMARIZE: "summarize",
            BRANCH_SELECT_FALLBACK: "fallback_non_dialogue",
        },
    )
    workflow.add_conditional_edges(
        "summarize",
        route_after_summarize,
        {
            BRANCH_BUILD_PROMPT: "build_prompt",
            BRANCH_SUMMARIZE_FALLBACK: "fallback_non_dialogue",
        },
    )
    workflow.add_edge("build_prompt", "generate_dialogue")
    workflow.add_edge("generate_dialogue", "validate_coverage")
    workflow.add_conditional_edges(
        "validate_coverage",
        functools.partial(route_after_validate, max_retry=retry_budget),
        {
            BRANCH_EVALUATE: "evaluate",
            BRANCH_REFINE: "refine_prompt",
            BRANCH_FALLBACK: "fallback_non_dialogue",
        },
    )
    workflow.add_edge("refine_prompt", "generate_dialogue")
    workflow.add_edge("fallback_non_dialogue", "evaluate")
    workflow.add_edge("evaluate", "persist")
    workflow.add_edge("persist", END)
    return workflow


def get_compiled_graph(
    *,
    db: Any = None,
    checkpointer: Any = None,
    retriever: Any = None,
    generator: Any = None,
    evaluator: Any = None,
    max_retry: int | None = None,
    timeout_seconds: int | None = None,
    parallel_enabled: bool = DIALOGUE_GEN_PARALLEL_ENABLED,
    max_concurrency: int | None = None,
):
    """编译生成图；T4 传入 checkpointer 即开启断点续写（thread_id = task_id）。"""
    workflow = build_dialogue_gen_graph(
        db=db,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        max_retry=max_retry,
        timeout_seconds=timeout_seconds,
        parallel_enabled=parallel_enabled,
        max_concurrency=max_concurrency,
    )
    return workflow.compile(checkpointer=checkpointer)


def _thread_config(task_id: str, checkpoint_id: str | None = None) -> dict:
    """checkpoint 线程配置（T4 启用；thread_id = `dg_<task_id>`，与 0015 口径对齐）。"""
    cfg = {"thread_id": task_id, "checkpoint_ns": ""}
    if checkpoint_id:
        cfg["checkpoint_id"] = checkpoint_id
    return {"configurable": cfg}


def get_checkpointer(db: Any) -> NoSQLCheckpointSaver:
    """构造生成域 checkpointer（独立集合 `ai_dialogue_checkpoint`，§4.3.1）。"""
    return NoSQLCheckpointSaver(db, collection=DIALOGUE_CHECKPOINT_COLLECTION)


def _default_checkpointer(db: Any) -> NoSQLCheckpointSaver | None:
    """生产缺省 checkpointer：仅当断点续写开关开启且 db 可用时接入（否则回退 T3 无 checkpoint）。"""
    if db is None or not DIALOGUE_GEN_CHECKPOINT_ENABLED:
        return None
    return get_checkpointer(db)


async def checkpoint_exists(
    saver: NoSQLCheckpointSaver, task_id: str, checkpoint_id: str | None = None
) -> bool:
    """线程（或指定 checkpoint）是否已存在 checkpoint（续写前探测）。"""
    return await saver.aget_tuple(_thread_config(task_id, checkpoint_id)) is not None


async def dialogue_checkpoint_exists(
    db: Any, task_id: str, checkpoint_id: str | None = None
) -> bool:
    """路由层探测：任务线程是否有可续写的 checkpoint（缺省集合）。"""
    return await checkpoint_exists(get_checkpointer(db), task_id, checkpoint_id)


async def latest_checkpoint_id(db: Any, task_id: str) -> str | None:
    """取任务线程最近一次 checkpoint id（失败后刷新续写游标用）；无则 None。"""
    tuple_ = await get_checkpointer(db).aget_tuple(_thread_config(task_id))
    if tuple_ is None:
        return None
    return tuple_.config["configurable"].get("checkpoint_id")


async def list_dialogue_checkpoints(
    db: Any, task_id: str, *, limit: int = 50
) -> list[dict]:
    """任务 checkpoint 时间线（按 ts 升序；`/checkpoints` 接口消费）。

    每项 `{checkpoint_id, stage, retry_count, ts}`；stage 取自图状态 channel_values
    （节点随执行写入，见各 `node_*` 的 `stage` 返回）。
    """
    saver = get_checkpointer(db)
    items: list[dict] = []
    async for tuple_ in saver.alist(_thread_config(task_id), limit=limit):
        checkpoint = tuple_.checkpoint if isinstance(tuple_.checkpoint, dict) else {}
        values = checkpoint.get("channel_values")
        values = values if isinstance(values, dict) else {}
        items.append(
            {
                "checkpoint_id": tuple_.config["configurable"].get("checkpoint_id"),
                # 入口 checkpoint（无 stage）归一为 start，避免暴露 LangGraph 内部 source
                "stage": values.get("stage") or STAGE_START,
                "retry_count": int(values.get("retry_count") or 0),
                "ts": checkpoint.get("ts"),
            }
        )
    items.reverse()  # saver 按 ts 倒序（最新在前）→ 时间线改为旧→新
    return items


def _initial_state(
    *, task_id: str, scholar_id: str, context: dict, preferred_type: str
) -> dict:
    """构造图初始状态（任务自包含快照，§6.1）。"""
    return {
        "task_id": task_id,
        "scholar_id": scholar_id,
        "context": context,
        "preferred_type": preferred_type,
        "retry_count": 0,
    }


async def _annotate_checkpoint(
    graph: Any, config: dict, result: dict, *, task_id: str
) -> None:
    """把本次执行落点写回 result.checkpoint（§4.6；终态不可续写）。"""
    snapshot = await graph.aget_state(config)
    values = getattr(snapshot, "values", None) or {}
    result["checkpoint"] = {
        "thread_id": task_id,
        "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        "stage": values.get("stage") or STAGE_PERSISTED,
        "resumable": False,
    }


async def run_dialogue_gen_graph(
    *,
    db: Any,
    task_id: str,
    scholar_id: str,
    context: dict,
    preferred_type: str = "auto",
    retriever: Any = None,
    generator: Any = None,
    evaluator: Any = None,
    checkpointer: Any = None,
    timeout_seconds: int | None = None,
    max_retry: int | None = None,
    parallel_enabled: bool = DIALOGUE_GEN_PARALLEL_ENABLED,
    max_concurrency: int | None = None,
    resume: bool = False,
    from_checkpoint_id: str | None = None,
) -> dict:
    """高层入口：跑完整图并返回 §4.6 `result`（任务执行器据此写回）。

    依赖缺省走生产实现（默认 retriever / 火山方舟 generator / L1 metrics）；
    单测与集成经参数注入 fake（不触网）。

    Args:
        resume: True 时从最近（或 `from_checkpoint_id` 指定）checkpoint 续跑，
            复用已完成节点与 `retry_count`（§4.3.1）；无 checkpoint → 整任务重跑并记
            `notes='resume_without_checkpoint'`。
        from_checkpoint_id: 指定续跑起点（缺省取最新）。
        parallel_enabled: v2 Send 扇出开关（False → 顺序执行）。
        max_concurrency: v2 扇出 LLM 并发上限。
    """
    saver = checkpointer if checkpointer is not None else _default_checkpointer(db)
    graph = get_compiled_graph(
        db=db,
        checkpointer=saver,
        retriever=retriever,
        generator=generator,
        evaluator=evaluator,
        max_retry=max_retry,
        timeout_seconds=timeout_seconds,
        parallel_enabled=parallel_enabled,
        max_concurrency=max_concurrency,
    )

    resume_without_checkpoint = False
    if resume:
        resume_without_checkpoint = saver is None or not await checkpoint_exists(
            saver, task_id, from_checkpoint_id
        )

    if resume and not resume_without_checkpoint:
        # 从断点续跑：传 None 从 checkpoint 恢复，无需重放已完成节点
        config = _thread_config(task_id, from_checkpoint_id)
        final = await graph.ainvoke(None, config)
    else:
        state = _initial_state(
            task_id=task_id,
            scholar_id=scholar_id,
            context=context,
            preferred_type=preferred_type,
        )
        config = _thread_config(task_id) if saver is not None else None
        final = await graph.ainvoke(state, config) if config else await graph.ainvoke(state)

    result = dict(final).get("result")
    if not isinstance(result, dict):
        raise DialogueGenError(ERR_NETWORK_ERROR, STAGE_LLM, "生成图未产出 result")

    if saver is not None and config is not None:
        await _annotate_checkpoint(graph, config, result, task_id=task_id)
    if resume_without_checkpoint:
        result["notes"] = merge_notes(result.get("notes"), NOTE_RESUME_WITHOUT_CHECKPOINT)
    return result
