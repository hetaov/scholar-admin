"""S4.2 L2 ConversationGraph：LangGraph StateGraph 会话图 + NoSQL checkpointer。

对应执行计划 S4.2（任务③~⑦）：
- ③ ConversationGraphState TypedDict（图状态定义）
- ④⑤ 六个图节点 + 条件边（加载/场景/评估/推进/回复）
- ⑥ build_conversation_graph + get_compiled_graph 工厂（checkpointer 接入）
- ⑦ NoSQLCheckpointSaver（conversation_graph_checkpoint 持久化，契约 data-model-contract §4.11.8）

路由层（routes_conversation.py）通过 start_scenario_graph / run_turn_graph 消费；
HTTP 契约不变；CONVERSATION_GRAPH_ENABLED=0 时路由层整体回退 L1 轻量状态机。
"""

from __future__ import annotations

import asyncio
import base64
import functools
import time
import uuid
from typing import Any, Iterator, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.graph import END, START, StateGraph

from config import CONVERSATION_CHECKPOINT_COLLECTION
from services.evaluation_engine import evaluate_text
from services.models_conversation import (
    DEFAULT_DIFFICULTY,
    LOW_CONFIDENCE_THRESHOLD,
    MAX_TURNS,
    TURN_STAGE_ANSWER,
    TURN_STAGE_DOWNGRADE,
    TURN_STAGE_HINT,
    TURN_STAGE_REPHRASE,
    next_turn_stage,
)
from services.models_content import get_sentences_by_ids

# 图内阶段（conversation_session.stage 在 L2 开启时取这些值，路由冲突判定仍用 active/ended）
GRAPH_STAGE_READY = "ready_for_utterance"  # 待用户输入
GRAPH_STAGE_AWAITING_EVAL = "awaiting_eval"  # 待评估（图内部瞬态）
GRAPH_STAGE_ENDED = "session_ended"  # 会话结束


# ---------------------------------------------------------------------------
# ③ 图状态定义
# ---------------------------------------------------------------------------


class ConversationGraphState(TypedDict, total=False):
    """L2 会话图状态（LangGraph channel values，经 checkpointer 持久化）。

    字段分三类：
    - 会话级：跨轮次持久（checkpoint 恢复）；
    - 轮级：每轮 invoke 覆盖（utterance 由路由传入，其余由节点产出）；
    - 图内部：节点中间产物（learner_state / lesson_knowledge 等）。
    """

    # 会话级
    session_id: str
    scholar_id: str
    scenario: str
    topic: str
    difficulty: int
    sentence_ids: list[str]
    translation: str | None
    cold_start: bool
    current_target: str  # 当前目标句（L1 语义对齐：未换句则保持）
    consecutive_failures: int
    turn_index: int
    stage: str  # 图阶段：ready_for_utterance / awaiting_eval / session_ended

    # 轮级
    utterance: str | None  # 用户本轮输入（路由 invoke 时注入）
    reply: str | None  # AI 回复
    turn_stage: str  # 轮级阶段：answer / hint / rephrase / downgrade
    hint: Any  # True | None（由 AI 回复承载提示文本，L1 对齐）
    rephrased: Any  # True | None
    suggestion: Any  # 转训练建议文案 | None
    verdict: dict | None  # evaluate_text 结果（含 meaningful/confidence/anomaly）
    meaningful: bool
    low_confidence: bool
    next_target: str  # 达成后引导的下一句（空串=保持当前目标句）

    # 图内部
    learner_state: dict | None
    lesson_knowledge: list[dict] | None
    scenario_desc: str | None
    opening_utterance: str | None
    session_ended: bool


# ---------------------------------------------------------------------------
# ④⑤ 图节点
# ---------------------------------------------------------------------------


def node_load_learner_state(state: ConversationGraphState) -> dict:
    """加载学习者状态（P0：会话级信息聚合；skill_state 深聚合留待 L3 联动）。"""
    return {
        "learner_state": {
            "cold_start": bool(state.get("cold_start")),
            "difficulty": int(state.get("difficulty") or DEFAULT_DIFFICULTY),
            "consecutive_failures": int(state.get("consecutive_failures") or 0),
            "turn_index": int(state.get("turn_index") or 0),
        }
    }


async def node_load_lesson_knowledge(
    state: ConversationGraphState, *, db: Any = None
) -> dict:
    """加载绑定句知识点（evidence 快照）；首轮补齐 current_target / translation。"""
    sentence_ids = state.get("sentence_ids") or []
    knowledge: list[dict] = []
    current_target = str(state.get("current_target") or "")
    translation = state.get("translation")
    if db is not None and sentence_ids:
        sentences = await get_sentences_by_ids(db, sentence_ids)
        for s in sentences:
            knowledge.append(
                {
                    "sentence_id": s.get("sentence_id"),
                    "text": s.get("text"),
                    "translation": s.get("translation"),
                    "difficulty": s.get("difficulty"),
                }
            )
        if not current_target and sentences:
            current_target = str(sentences[0].get("text") or "")
        if translation is None and sentences:
            translation = sentences[0].get("translation")
    return {
        "lesson_knowledge": knowledge,
        "current_target": current_target,
        "translation": translation,
    }


def node_generate_scenario(state: ConversationGraphState) -> dict:
    """首轮场景：生成场景引导语 + 开场白（规则生成不调 LLM，保证 scenario 接口稳定）。"""
    scenario = str(state.get("scenario") or "free_talk")
    topic = str(state.get("topic") or "daily conversation")
    target = str(state.get("current_target") or "")
    opening = f"Let's talk about {topic}! Try: {target}" if target else f"Let's talk about {topic}!"
    return {
        "scenario_desc": f"场景：{scenario}；话题：{topic}。",
        "opening_utterance": opening,
        "stage": GRAPH_STAGE_READY,
    }


async def node_evaluate_user_response(
    state: ConversationGraphState, *, evaluator: Any = None
) -> dict:
    """评估用户输出 vs 目标表达（复用 evaluation_engine.evaluate_text，可注入 fake）。"""
    fn = evaluator or evaluate_text
    original = str(state.get("current_target") or "")
    utterance = str(state.get("utterance") or "")
    result = fn(original, utterance)
    if asyncio.iscoroutine(result):
        result = await result
    verdict = result or {}
    confidence = float(verdict.get("confidence") or 0.0)
    meaningful = bool(verdict.get("meaningful"))
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    return {
        "verdict": verdict,
        "meaningful": meaningful,
        "low_confidence": low_confidence,
        "stage": GRAPH_STAGE_AWAITING_EVAL,
    }


def node_advance_stage(state: ConversationGraphState) -> dict:
    """状态机推进（复用 L1 next_turn_stage）：answer / hint / rephrase / downgrade。"""
    difficulty = int(state.get("difficulty") or DEFAULT_DIFFICULTY)
    meaningful = bool(state.get("meaningful"))
    low_confidence = bool(state.get("low_confidence"))
    consecutive_failures = int(state.get("consecutive_failures") or 0)
    failed_this_turn = not (meaningful and not low_confidence)
    failures_after = consecutive_failures + (1 if failed_this_turn else 0)
    step = next_turn_stage(
        consecutive_failures=failures_after,
        difficulty=difficulty,
        meaningful=meaningful,
        low_confidence=low_confidence,
    )
    new_failures = 0 if step["reset_failures"] else failures_after
    new_turn_index = int(state.get("turn_index") or 0) + 1
    return {
        "turn_stage": step["stage"],
        "hint": step.get("hint"),
        "rephrased": step.get("rephrased"),
        "suggestion": step.get("suggestion"),
        "difficulty": step["difficulty"],
        "consecutive_failures": new_failures,
        "turn_index": new_turn_index,
        "session_ended": new_turn_index >= MAX_TURNS,
    }


def _rule_reply(*, topic: str, target: str, utterance: str, stage: str, difficulty: int) -> dict:
    """规则兜底回复（LLM 不可用 / 未注入 reply_generator 时保证图可用）。"""
    if stage == TURN_STAGE_HINT:
        words = target.split()[:2]
        return {"reply": f"提示：以 {(' '.join(words))} 开头试试？", "next_target": ""}
    if stage == TURN_STAGE_REPHRASE:
        return {"reply": f"换个说法：{target}（意思：说得更简单些）", "next_target": ""}
    if stage == TURN_STAGE_DOWNGRADE:
        return {"reply": "我们换一个更简单的句子，慢慢来。", "next_target": "It is a book."}
    if target:
        return {"reply": f"不错，继续。试着说：{target}", "next_target": ""}
    return {"reply": "很好！我们进入下一句。", "next_target": "I like apples."}


async def node_generate_ai_utterance(
    state: ConversationGraphState, *, reply_generator: Any = None
) -> dict:
    """生成 AI 回复（复用路由层 _generate_reply 注入；失败/未注入回落规则兜底）。"""
    generator = reply_generator or _rule_reply
    result = generator(
        topic=str(state.get("topic") or "daily conversation"),
        target=str(state.get("current_target") or ""),
        utterance=str(state.get("utterance") or ""),
        stage=str(state.get("turn_stage") or TURN_STAGE_ANSWER),
        difficulty=int(state.get("difficulty") or DEFAULT_DIFFICULTY),
    )
    if asyncio.iscoroutine(result):
        result = await result
    result = result or {}
    reply = str(result.get("reply") or "")
    next_target = str(result.get("next_target") or "")
    target = str(state.get("current_target") or "")
    if not next_target and target:
        next_target = target  # 未换句 → 保持当前目标句
    session_ended = bool(state.get("session_ended"))
    return {
        "reply": reply,
        "next_target": next_target,
        "current_target": next_target or target,
        "stage": GRAPH_STAGE_ENDED if session_ended else GRAPH_STAGE_READY,
    }


# ---------------------------------------------------------------------------
# ⑤ 条件边
# ---------------------------------------------------------------------------


def router_entry(state: ConversationGraphState) -> str:
    """首轮（scenario 初始化，无用户输入）→ generate_scenario；有用户输入 → 评估路径。"""
    return "evaluate" if str(state.get("utterance") or "").strip() else "scenario"


def router_after_advance(state: ConversationGraphState) -> str:
    """推进后：会话结束 → END；否则生成 AI 回复。"""
    return "end" if state.get("session_ended") else "utterance"


# ---------------------------------------------------------------------------
# ⑥ 图构建 + checkpointer 接入工厂
# ---------------------------------------------------------------------------


def build_conversation_graph(
    *, db: Any = None, evaluator: Any = None, reply_generator: Any = None
) -> StateGraph:
    """构建 L2 ConversationGraph（依赖经 partial 注入，便于单测注入 fake）。"""
    workflow = StateGraph(ConversationGraphState)
    workflow.add_node("load_learner_state", node_load_learner_state)
    workflow.add_node(
        "load_lesson_knowledge",
        functools.partial(node_load_lesson_knowledge, db=db),
    )
    workflow.add_node("generate_scenario", node_generate_scenario)
    workflow.add_node(
        "evaluate_user_response",
        functools.partial(node_evaluate_user_response, evaluator=evaluator),
    )
    workflow.add_node("advance_stage", node_advance_stage)
    workflow.add_node(
        "generate_ai_utterance",
        functools.partial(node_generate_ai_utterance, reply_generator=reply_generator),
    )

    workflow.add_edge(START, "load_learner_state")
    workflow.add_edge("load_learner_state", "load_lesson_knowledge")
    workflow.add_conditional_edges(
        "load_lesson_knowledge",
        router_entry,
        {"scenario": "generate_scenario", "evaluate": "evaluate_user_response"},
    )
    workflow.add_edge("generate_scenario", END)
    workflow.add_edge("evaluate_user_response", "advance_stage")
    workflow.add_conditional_edges(
        "advance_stage",
        router_after_advance,
        {"utterance": "generate_ai_utterance", "end": END},
    )
    workflow.add_edge("generate_ai_utterance", END)
    return workflow


def get_compiled_graph(
    *,
    db: Any = None,
    checkpointer: BaseCheckpointSaver | None = None,
    evaluator: Any = None,
    reply_generator: Any = None,
):
    """工厂：compile StateGraph；传入 checkpointer 则开启断点续聊（thread_id=session_id）。"""
    workflow = build_conversation_graph(
        db=db, evaluator=evaluator, reply_generator=reply_generator
    )
    return workflow.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# ⑦ NoSQLCheckpointSaver（LangGraph checkpointer → CloudBase NoSQL）
# ---------------------------------------------------------------------------


def _thread_config(session_id: str, checkpoint_id: str | None = None) -> dict:
    cfg = {"thread_id": session_id, "checkpoint_ns": ""}
    if checkpoint_id:
        cfg["checkpoint_id"] = checkpoint_id
    return {"configurable": cfg}


def _b64_encode(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def _b64_decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


class NoSQLCheckpointSaver(BaseCheckpointSaver):
    """LangGraph checkpointer → CloudBase NoSQL（conversation_graph_checkpoint，§4.11.8）。

    - thread_id = conversation_session.session_id，每轮图执行落一个 checkpoint 文档；
    - checkpoint / metadata 经 self.serde.dumps_typed（msgpack bytes）后 base64 落库；
    - LangGraph 运行时走 async 方法（aput/aget_tuple/alist/aput_writes），sync 方法作桥接
      （asyncio.run，供 sync invoke 使用）。
    """

    def __init__(self, db: Any, collection: str | None = None):
        super().__init__()
        self.db = db
        self.collection = collection or CONVERSATION_CHECKPOINT_COLLECTION

    # ---------------- async 入口（LangGraph 运行时主路径） ----------------

    async def aput(self, config, checkpoint, metadata, new_versions) -> dict:
        return await self._aput(config, checkpoint, metadata)

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        return await self._aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):
        async for t in self._iter_checkpoints(config, filter=filter, before=before, limit=limit):
            yield t

    async def aput_writes(self, config, writes, task_id, task_path=""):
        # P0：不持久化 streaming 中间写入（未启用 stream mode=updates 的中间 writes）
        return None

    # ---------------- sync 桥接（供 sync invoke 使用） ----------------

    def put(self, config, checkpoint, metadata, new_versions):
        return asyncio.run(self._aput(config, checkpoint, metadata))

    def get_tuple(self, config):
        return asyncio.run(self._aget_tuple(config))

    def list(self, config, *, filter=None, before=None, limit=None):
        return iter(asyncio.run(self._collect_checkpoints(config, filter=filter, before=before, limit=limit)))

    def put_writes(self, config, writes, task_id, task_path=""):
        return None

    # ---------------- 异步实现 ----------------

    async def _aput(self, config, checkpoint, metadata) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = str(checkpoint.get("id") or uuid.uuid4().hex)
        ctype, cblob = self.serde.dumps_typed(checkpoint)
        mtype, mblob = self.serde.dumps_typed(metadata)
        _id = f"{thread_id}:{checkpoint_id}"
        doc = {
            "_id": _id,
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
            "checkpoint_type": ctype,
            "metadata_type": mtype,
            "checkpoint": _b64_encode(cblob),
            "metadata": _b64_encode(mblob),
            "ts": checkpoint.get("ts"),  # LangGraph 生成，唯一递增，作为最新选择排序键
            "created_at": int(time.time() * 1000),
        }
        existing = await self.db.query(
            self.collection, where={"_id": _id}, limit=1
        )
        if existing.get("records"):
            await self.db.update(
                self.collection,
                where={"_id": _id},
                data={"$set": doc},
                multi=False,
            )
        else:
            await self.db.insert(self.collection, doc)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def _deserialize_doc(self, doc: dict) -> tuple[dict, dict]:
        checkpoint = self.serde.loads_typed(
            (doc.get("checkpoint_type"), _b64_decode(doc["checkpoint"]))
        )
        metadata = self.serde.loads_typed(
            (doc.get("metadata_type"), _b64_decode(doc["metadata"]))
        )
        return checkpoint, metadata

    async def _aget_tuple(self, config) -> CheckpointTuple | None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")
        where: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id:
            where["checkpoint_id"] = checkpoint_id
        records = await self.db.query(
            self.collection,
            where=where,
            limit=1,
            order=[{"field": "ts", "direction": "desc"}],
        )
        if not records.get("records"):
            return None
        doc = records["records"][0]
        checkpoint, metadata = self._deserialize_doc(doc)
        parent_id = doc.get("parent_checkpoint_id")
        return CheckpointTuple(
            config=_thread_config(thread_id, doc["checkpoint_id"]),
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=_thread_config(thread_id, parent_id) if parent_id else None,
            pending_writes=None,
        )

    async def _iter_checkpoints(self, config, *, filter=None, before=None, limit=None):
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        where: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if before:
            before_id = before.get("configurable", {}).get("checkpoint_id")
            if before_id:
                where["checkpoint_id"] = {"$ne": before_id}
        max_limit = limit or 10
        records = await self.db.query(
            self.collection,
            where=where,
            limit=max_limit,
            order=[{"field": "ts", "direction": "desc"}],
        )
        for doc in records.get("records", []):
            checkpoint, metadata = self._deserialize_doc(doc)
            parent_id = doc.get("parent_checkpoint_id")
            yield CheckpointTuple(
                config=_thread_config(thread_id, doc["checkpoint_id"]),
                checkpoint=checkpoint,
                metadata=metadata,
                parent_config=_thread_config(thread_id, parent_id) if parent_id else None,
                pending_writes=None,
            )

    async def _collect_checkpoints(self, config, *, filter=None, before=None, limit=None) -> list[CheckpointTuple]:
        result: list[CheckpointTuple] = []
        async for t in self._iter_checkpoints(config, filter=filter, before=before, limit=limit):
            result.append(t)
        return result


# ---------------------------------------------------------------------------
# 高层入口（路由层消费；HTTP 契约不变）
# ---------------------------------------------------------------------------


def _initial_state(session: dict) -> dict:
    """从 conversation_session 文档构造图初始状态（scenario 首轮 invoke 用）。"""
    return {
        "session_id": session["session_id"],
        "scholar_id": session.get("scholar_id"),
        "scenario": session.get("scenario"),
        "topic": session.get("topic"),
        "difficulty": int(session.get("difficulty") or DEFAULT_DIFFICULTY),
        "sentence_ids": session.get("sentence_ids") or [],
        "translation": session.get("translation"),
        "cold_start": bool(session.get("cold_start")),
        "consecutive_failures": int(session.get("consecutive_failures") or 0),
        "turn_index": int(session.get("turn_index") or 0),
        "current_target": str(session.get("current_target") or ""),
        "stage": GRAPH_STAGE_READY,
    }


async def start_scenario_graph(
    db: Any, session: dict, *, evaluator: Any = None, reply_generator: Any = None
) -> dict:
    """创建会话后初始化图 checkpoint（generate_scenario 路径）。

    返回 {"graph_state": 图最终状态, "checkpoint_id": 最新 checkpoint id}。
    """
    saver = NoSQLCheckpointSaver(db)
    graph = get_compiled_graph(
        db=db,
        checkpointer=saver,
        evaluator=evaluator,
        reply_generator=reply_generator,
    )
    config = _thread_config(session["session_id"])
    final = await graph.ainvoke(_initial_state(session), config=config)
    snapshot = await graph.aget_state(config)
    checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
    return {"graph_state": dict(final), "checkpoint_id": checkpoint_id}


async def run_turn_graph(
    db: Any,
    session: dict,
    utterance: str,
    *,
    evaluator: Any = None,
    reply_generator: Any = None,
) -> dict:
    """每轮用户输入走图：checkpoint 恢复 → 评估 → 推进 → 生成回复。

    返回 {"reply", "graph_state", "checkpoint_id", "turn_stage", "verdict"}；
    graph_state 含 turn_index / consecutive_failures / difficulty 等会话续写字段。
    """
    saver = NoSQLCheckpointSaver(db)
    graph = get_compiled_graph(
        db=db,
        checkpointer=saver,
        evaluator=evaluator,
        reply_generator=reply_generator,
    )
    config = _thread_config(session["session_id"])
    final = await graph.ainvoke({"utterance": utterance}, config=config)
    snapshot = await graph.aget_state(config)
    checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
    return {
        "reply": str(final.get("reply") or ""),
        "graph_state": dict(final),
        "checkpoint_id": checkpoint_id,
        "turn_stage": str(final.get("turn_stage") or TURN_STAGE_ANSWER),
        "verdict": final.get("verdict") or {},
    }
