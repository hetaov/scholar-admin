"""沉浸式 AI 会话 v3 引擎适配层（设计稿 §11.3 / §11.6）

职责：把会话提交的 `context` 快照（`scenario / roles / groups / history / user_input / assisted`）
按 `mode` 分派到「同一生成核」——**复用批量面（/ai/dialogue/v1）的 Prompt 重构与 LangGraph
范式**，产出对齐 v2 `result` 的结构：`{ content_type, ai_text, hint, suggested_targets }`。

与 v2 `session_gen` 的差异（§11）：
- 召回口径（§11.3）：`kind=review` 素材句 **clamp 到 [2, 6]**（不足记 `recall_insufficient`，
  对齐批量面 §4.2 / §8.4 #4 的降级留痕）；
- 图编排（§11.6）：`load_context → build_prompt → generate_reply → persist`，每轮 invoke
  落一个 checkpoint 到 `ai_session_v3_checkpoint`（**thread_id = session_id**），用户中断后
  同一 `session_id` 可续聊；
- 生成核复用：Prompt 组装/解析仍走 `services.providers.session_gen`（保证 v2 出口契约逐字段一致），
  LLM 单次调用走 `services.providers.dialogue_gen.invoke_dialogue_llm`（超时/空值语义统一）；
- 错误码语义对齐 v2（§11.4）：`LLM_TIMEOUT / EVAL_UNAVAILABLE / LLM_PARSE_ERROR / NETWORK_ERROR`，
  **不降级、不静默**。

依赖注入（单测传 fake，禁真外呼）：`generator`（假 LLM）/ `checkpointer`（假/内存 checkpointer）。
"""
from __future__ import annotations

import functools
import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from config import SESSION_LLM_TIMEOUT_SECONDS, SESSION_V3_CHECKPOINT_COLLECTION
from services.learning.conversation_graph import NoSQLCheckpointSaver
from services.providers import session_gen
from services.providers.dialogue_gen import (
    ERR_LLM_PARSE_ERROR as _DG_ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT as _DG_ERR_LLM_TIMEOUT,
    ERR_LLM_UNAVAILABLE as _DG_ERR_LLM_UNAVAILABLE,
    RECALL_MAX,
    RECALL_MIN,
    DialogueGenError,
    invoke_dialogue_llm,
)
from services.providers.session_gen import (
    ERR_EVAL_UNAVAILABLE,
    ERR_LLM_PARSE_ERROR,
    ERR_LLM_TIMEOUT,
    ERR_NETWORK_ERROR,
    STAGE_LLM,
    STAGE_PARSE,
    SessionGenError,
)

logger = logging.getLogger("scholar-admin.dialogue_engine")

# 图节点阶段（随 checkpoint 落库，供 /checkpoints 展示 / 排障）
STAGE_CONTEXT_LOADED = "context_loaded"
STAGE_PROMPT_BUILT = "prompt_built"
STAGE_GENERATED = "generated"
STAGE_PERSISTED = "persisted"
STAGE_START = "start"

# 降级留痕（§8.4 #4 允许的召回不足降级）
NOTE_RECALL_INSUFFICIENT = "recall_insufficient"


class SessionDialogueState(TypedDict, total=False):
    """会话 v3 生成图状态（LangGraph channel values，经 checkpointer 持久化）。"""

    # 任务/会话级
    session_id: str
    context: dict
    preferred_type: str
    stage: str

    # load_context
    materials: list[dict]
    required_ids: list[str]
    recall_ids: list[str]
    content_type: str
    context_notes: list[str]

    # build_prompt
    messages: list[dict]

    # generate_reply
    raw: str | None
    parsed: dict | None

    # persist
    result: dict


# ---------------------------------------------------------------------------
# 素材规整（召回 clamp [2, 6]）
# ---------------------------------------------------------------------------


def _split_materials(materials: Any) -> tuple[list[dict], list[dict]]:
    """拆分素材为 (new 组, review 句)。

    `kind` 缺省按 `new` 计（对齐契约入参默认）；只保留含 sentence_id 与 content 的句子。
    """
    new_groups: list[dict] = []
    review_sentences: list[dict] = []
    for group in materials or []:
        if not isinstance(group, dict):
            continue
        if str(group.get("kind") or "new") == "review":
            for s in group.get("sentences") or []:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("sentence_id") or "").strip()
                content = str(s.get("content") or "").strip()
                if sid and content:
                    review_sentences.append({"sentence_id": sid, "content": content})
        else:
            new_groups.append(group)
    return new_groups, review_sentences


def normalize_session_materials(
    materials: Any,
) -> tuple[list[dict], list[str], list[str], list[str]]:
    """规整会话素材：new 组全保留，review 句 clamp 到 [RECALL_MIN, RECALL_MAX]。

    Returns:
        (materials, required_ids, recall_ids, notes)
        - materials: new 组 + （若有）单条 review 组（句数 ≤ RECALL_MAX）；
        - required_ids: new（必用）句 id 有序清单；
        - recall_ids: 实际纳入的 review 句 id（≤ RECALL_MAX）；
        - notes: 召回不足 → `["recall_insufficient"]`（§11.3 / §8.4 #4）。
    """
    new_groups, review_sentences = _split_materials(materials)
    recalled = review_sentences[:RECALL_MAX]
    notes: list[str] = []
    if len(recalled) < RECALL_MIN:
        notes.append(NOTE_RECALL_INSUFFICIENT)

    required_ids: list[str] = []
    for group in new_groups:
        for s in group.get("sentences") or []:
            if not isinstance(s, dict):
                continue
            sid = str(s.get("sentence_id") or "").strip()
            if sid and str(s.get("content") or "").strip() and sid not in required_ids:
                required_ids.append(sid)

    normalized = list(new_groups)
    if recalled:
        normalized.append({"kind": "review", "sentences": recalled})
    recall_ids = [s["sentence_id"] for s in recalled]
    return normalized, required_ids, recall_ids, notes


# ---------------------------------------------------------------------------
# 节点：load_context / build_prompt
# ---------------------------------------------------------------------------


def node_load_context(state: SessionDialogueState) -> dict:
    """装载上下文：素材规整（召回 clamp）+ 形态选择 + 必用句/召回句清单。"""
    context = state.get("context") or {}
    normalized, required_ids, recall_ids, notes = normalize_session_materials(
        context.get("materials") or []
    )
    content_type = session_gen.choose_content_type(
        normalized, state.get("preferred_type") or "auto"
    )
    return {
        "materials": normalized,
        "required_ids": required_ids,
        "recall_ids": recall_ids,
        "content_type": content_type,
        "context_notes": notes,
        "stage": STAGE_CONTEXT_LOADED,
    }


def node_build_prompt(state: SessionDialogueState) -> dict:
    """复用 v2 Prompt 组装（同一生成核），素材取规整后的会话快照。"""
    context = state.get("context") or {}
    context = {**context, "materials": state.get("materials") or context.get("materials") or []}
    messages = session_gen.build_session_messages(
        context, state.get("content_type") or "dialogue"
    )
    return {"messages": messages, "stage": STAGE_PROMPT_BUILT}


# ---------------------------------------------------------------------------
# 节点：generate_reply（LLM + 解析）
# ---------------------------------------------------------------------------


def _map_dialogue_error(e: DialogueGenError) -> SessionGenError:
    """批量面错误码 → v2 会话错误码（§11.4 语义对齐，不降级不静默）。"""
    code = str(getattr(e, "error_code", "") or "")
    detail = str(getattr(e, "detail", e) or "")
    if code == _DG_ERR_LLM_TIMEOUT:
        return SessionGenError(ERR_LLM_TIMEOUT, STAGE_LLM, detail)
    if code == _DG_ERR_LLM_UNAVAILABLE:
        return SessionGenError(ERR_EVAL_UNAVAILABLE, STAGE_LLM, detail)
    if code == _DG_ERR_LLM_PARSE_ERROR:
        return SessionGenError(ERR_LLM_PARSE_ERROR, STAGE_PARSE, detail)
    return SessionGenError(ERR_NETWORK_ERROR, STAGE_LLM, detail)


async def node_generate_reply(
    state: SessionDialogueState,
    *,
    generator: Any = None,
    timeout_seconds: int | None = None,
) -> dict:
    """单次 LLM 调用 + 解析（注入 fake generator；超时/空值语义见 invoke_dialogue_llm）。"""
    timeout = timeout_seconds or SESSION_LLM_TIMEOUT_SECONDS
    try:
        raw = await invoke_dialogue_llm(
            state.get("messages") or [],
            generator=generator,
            timeout_seconds=timeout,
        )
    except DialogueGenError as e:
        raise _map_dialogue_error(e) from e

    valid_ids = session_gen.collect_sentence_ids(state.get("materials") or [])
    parsed = session_gen.parse_session_output(raw, valid_ids)
    if parsed is None:
        raise SessionGenError(
            ERR_LLM_PARSE_ERROR, STAGE_PARSE, f"模型输出解析失败: {str(raw)[:200]}"
        )
    return {"raw": raw, "parsed": parsed, "stage": STAGE_GENERATED}


# ---------------------------------------------------------------------------
# 节点：persist（组装 v2 对齐的 result）
# ---------------------------------------------------------------------------


def node_persist(state: SessionDialogueState) -> dict:
    """组装会话回应（出口仅 4 字段，§11.4 result 结构对齐 v2）。"""
    parsed = state.get("parsed") or {}
    result = {
        "content_type": parsed.get("content_type"),
        "ai_text": parsed.get("ai_text"),
        "hint": parsed.get("hint"),
        "suggested_targets": parsed.get("suggested_targets") or [],
    }
    logger.info(
        "[dialogue_engine] persist → session_id=%s, content_type=%s, hints=%s, targets=%s, notes=%s",
        state.get("session_id"),
        result["content_type"],
        len((result["hint"] or {}).get("levels") or []) if isinstance(result["hint"], dict) else 0,
        result["suggested_targets"],
        state.get("context_notes"),
    )
    return {"result": result, "stage": STAGE_PERSISTED}


# ---------------------------------------------------------------------------
# 图构建 / 编译 / checkpointer
# ---------------------------------------------------------------------------


def build_session_graph(
    *,
    generator: Any = None,
    timeout_seconds: int | None = None,
) -> StateGraph:
    """构建会话 v3 生成图（依赖经 partial 注入，单测传 fake；复用批量面图范式）。"""
    workflow = StateGraph(SessionDialogueState)
    workflow.add_node("load_context", node_load_context)
    workflow.add_node("build_prompt", node_build_prompt)
    workflow.add_node(
        "generate_reply",
        functools.partial(
            node_generate_reply, generator=generator, timeout_seconds=timeout_seconds
        ),
    )
    workflow.add_node("persist", node_persist)

    workflow.add_edge(START, "load_context")
    workflow.add_edge("load_context", "build_prompt")
    workflow.add_edge("build_prompt", "generate_reply")
    workflow.add_edge("generate_reply", "persist")
    workflow.add_edge("persist", END)
    return workflow


def get_compiled_session_graph(
    *,
    checkpointer: Any = None,
    generator: Any = None,
    timeout_seconds: int | None = None,
):
    """编译会话图；传入 checkpointer 即开启断点续写（thread_id = session_id）。"""
    return build_session_graph(
        generator=generator, timeout_seconds=timeout_seconds
    ).compile(checkpointer=checkpointer)


def get_checkpointer(db: Any) -> NoSQLCheckpointSaver:
    """构造会话 v3 checkpointer（独立集合 `ai_session_v3_checkpoint`，§11.6）。"""
    return NoSQLCheckpointSaver(db, collection=SESSION_V3_CHECKPOINT_COLLECTION)


def _thread_config(session_id: str, checkpoint_id: str | None = None) -> dict:
    """checkpoint 线程配置（§11.6：thread_id = session_id，与 0015 口径对齐）。"""
    cfg = {"thread_id": session_id, "checkpoint_ns": ""}
    if checkpoint_id:
        cfg["checkpoint_id"] = checkpoint_id
    return {"configurable": cfg}


async def session_checkpoint_exists(
    db: Any, session_id: str, checkpoint_id: str | None = None
) -> bool:
    """线程（或指定 checkpoint）是否已存在 checkpoint（续聊前探测）。"""
    return (
        await get_checkpointer(db).aget_tuple(_thread_config(session_id, checkpoint_id))
        is not None
    )


async def latest_checkpoint_id(db: Any, session_id: str) -> str | None:
    """取会话线程最近一次 checkpoint id；无则 None。"""
    tuple_ = await get_checkpointer(db).aget_tuple(_thread_config(session_id))
    if tuple_ is None:
        return None
    return tuple_.config["configurable"].get("checkpoint_id")


async def list_session_checkpoints(
    db: Any, session_id: str, *, limit: int = 50
) -> list[dict]:
    """会话 checkpoint 时间线（旧→新）；每项 `{checkpoint_id, stage, ts}`。"""
    saver = get_checkpointer(db)
    items: list[dict] = []
    async for tuple_ in saver.alist(_thread_config(session_id), limit=limit):
        checkpoint = tuple_.checkpoint if isinstance(tuple_.checkpoint, dict) else {}
        values = checkpoint.get("channel_values")
        values = values if isinstance(values, dict) else {}
        items.append(
            {
                "checkpoint_id": tuple_.config["configurable"].get("checkpoint_id"),
                "stage": values.get("stage") or STAGE_START,
                "ts": checkpoint.get("ts"),
            }
        )
    items.reverse()
    return items


def _default_checkpointer(db: Any) -> NoSQLCheckpointSaver | None:
    """生产缺省 checkpointer：db 可用即接入（会话面多轮续聊，§11.6）。"""
    if db is None:
        return None
    return get_checkpointer(db)


# ---------------------------------------------------------------------------
# 对外生成入口
# ---------------------------------------------------------------------------


async def generate_session_reply(
    *,
    db: Any = None,
    session_id: str,
    context: dict,
    preferred_type: str = "auto",
    timeout_seconds: int | None = None,
    generator: Any = None,
    checkpointer: Any = None,
    resume: bool = False,
    from_checkpoint_id: str | None = None,
) -> dict:
    """会话 v3 生成（start/turn 同一入口，按 context.mode 分派 Prompt）。

    复用批量面生成核：LangGraph 图 + 单次 LLM 调用；每轮 invoke 落 checkpoint
    （thread_id = session_id，§11.6）。

    Args:
        db: 数据库客户端（构造缺省 checkpointer；传 checkpointer 时可为 None）；
        session_id: 会话 ID（checkpoint thread_id）；
        context: 任务 context 快照（mode/materials/scenario/roles/history/user_input/assisted）；
        preferred_type: auto / dialogue / fill；
        generator: 可注入假 LLM（单测；缺省走火山方舟）；
        checkpointer: 可注入 checkpointer（单测；缺省按 db 构造，db 为 None 则无 checkpoint）；
        resume: True 且线程已有 checkpoint → 从最近（或 from_checkpoint_id）续跑。

    Returns:
        `{ content_type, ai_text, hint, suggested_targets }`（v2 对齐，§11.4）

    Raises:
        SessionGenError: LLM_TIMEOUT / EVAL_UNAVAILABLE / LLM_PARSE_ERROR / NETWORK_ERROR
    """
    saver = checkpointer if checkpointer is not None else _default_checkpointer(db)
    graph = get_compiled_session_graph(
        checkpointer=saver, generator=generator, timeout_seconds=timeout_seconds
    )

    if saver is not None:
        config = _thread_config(session_id, from_checkpoint_id)
        has_checkpoint = await saver.aget_tuple(config) is not None
        if resume and has_checkpoint:
            final = await graph.ainvoke(None, config)
        else:
            state = {
                "session_id": session_id,
                "context": context,
                "preferred_type": preferred_type,
                "stage": STAGE_START,
            }
            final = await graph.ainvoke(state, config)
    else:
        state = {
            "session_id": session_id,
            "context": context,
            "preferred_type": preferred_type,
            "stage": STAGE_START,
        }
        final = await graph.ainvoke(state)

    result = dict(final).get("result")
    if not isinstance(result, dict):
        raise SessionGenError(ERR_NETWORK_ERROR, STAGE_LLM, "会话生成图未产出 result")
    return result
