"""LangGraph 知识总结图编排（F1 升级 · 2026-08-21 SOP ⑤ K2+K3）

契约：
- service-contract.md §7.3 generateKnowledgeSummary（内部实现切换，签名不变）
- data-model-contract.md §4.12.8(b) ai_summary 新增 quality_score / evaluation_feedback / evaluation_model / graph_version
- ADR-0015（LangGraph + feature flag 回退先例）

图结构：
  START → load_node → [check_description]
                           │           │
                      (有描述)       (无描述)
                           │           └→ END (status=blocked)
                     generate_summary
                           │
                           ↓
                     evaluate_summary
                           │
                   ┌───────┼──────────┐
              (pass)    (retry<max)  (retry≥max)
                  │         │            │
                  ↓         └→ generate   ↓
             persist_summary  (retry++)  persist_summary (degraded)
                  │                        │
                  ↓                        ↓
                 END (success)            END (degraded)
"""
from __future__ import annotations

import json
import logging
import time
from functools import partial
from typing import Any, TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from config import (
    HUNYUAN_BASE_URL,
    HUNYUAN_EVAL_MAX_RETRIES,
    HUNYUAN_EVAL_MODEL,
    HUNYUAN_EVAL_PASS_THRESHOLD,
    HUNYUAN_SECRET_KEY,
    HUNYUAN_TIMEOUT_SECONDS,
    USE_LANGGRAPH_SUMMARY,
)
from services.audit import (
    AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
    AUDIT_RESULT_FAILED,
    write_audit,
)
from services.database import CURRICULUM_NODE_COLLECTION
from services.math import knowledge_summary as _ks

logger = logging.getLogger("scholar-admin.math.summary_graph")

# 图版本号（用于后续升级评估策略时的版本区分）
_GRAPH_VERSION = 1

# ai_summary 新字段默认值
QUALITY_SCORE_NOT_EVALUATED = -1.0


# ---------------------------------------------------------------------------
# 图状态定义（设计文档 §3.2.1）
# ---------------------------------------------------------------------------


class KnowledgeSummaryState(TypedDict, total=False):
    """LangGraph 知识总结图状态"""

    # 输入
    curriculum_node_id: str
    force_regenerate: bool
    include_extended_points: bool
    # 节点数据
    node: dict
    idempotency_key: str
    # 生成
    parsed_summary: dict
    # 评估
    quality_score: float
    evaluation_feedback: str
    retry_count: int
    # 输出
    final_summary: dict | None
    status: str
    error: str


# ---------------------------------------------------------------------------
# 混元评估 Prompt（K3 · 设计文档 §3.3.2）
# ---------------------------------------------------------------------------

_EVAL_SYSTEM_PROMPT = (
    "你是小学数学教研评审专家。请评估以下 AI 生成的知识总结质量。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_EVAL_USER_TEMPLATE = """请评估以下 AI 生成的知识总结质量。

教材节点描述：
{description}

AI 知识总结（JSON）：
{summary_json}

评估维度：
1. 结构完整性（20%）：knowledge_points 非空、每项含 name + summary + ability_dimensions
2. 知识点准确性（40%）：与教材描述匹配，无幻觉、无错误
3. 覆盖度（25%）：描述中的知识点是否被充分覆盖
4. 扩展点合理性（15%）：extended_points 难度梯度合理、与核心知识点关联

输出 JSON：
{{"score": 0.0~1.0, "feedback": "评估反馈（含改进建议）", "issues": ["问题1", "问题2"]}}"""


# ---------------------------------------------------------------------------
# 混元 API 调用（K3 · 设计文档 §3.3）
# ---------------------------------------------------------------------------


def _parse_eval_response(text: str) -> dict:
    """解析混元返回的 JSON（容错：去 markdown fence / 提取首尾花括号）"""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


async def _call_hunyuan_evaluate(
    *, description: str, summary: dict
) -> dict:
    """调混元模型评估知识总结质量

    返回 {"score": float, "feedback": str, "issues": list}
    混元不可用 / 返回非法 JSON / 超时时降级返回 {"score": -1.0, "feedback": "..."}
    """
    prompt = _EVAL_USER_TEMPLATE.format(
        description=description,
        summary_json=json.dumps(summary, ensure_ascii=False),
    )

    try:
        async with httpx.AsyncClient(timeout=HUNYUAN_TIMEOUT_SECONDS) as client:
            resp = await client.post(
                f"{HUNYUAN_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {HUNYUAN_SECRET_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HUNYUAN_EVAL_MODEL,
                    "messages": [
                        {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            result = _parse_eval_response(content)
            score = float(result.get("score", -1.0))
            # 钳位到 [0.0, 1.0]
            if score < 0:
                score = -1.0
            elif score > 1.0:
                score = 1.0
            return {
                "score": score,
                "feedback": result.get("feedback", ""),
                "issues": result.get("issues", []),
            }
    except Exception as e:
        logger.warning(f"混元评估失败，降级跳过: {type(e).__name__}: {e}")
        return {
            "score": QUALITY_SCORE_NOT_EVALUATED,
            "feedback": f"评估跳过: {e}",
            "issues": [],
        }


# ---------------------------------------------------------------------------
# 图节点函数（设计文档 §3.2.2）
# ---------------------------------------------------------------------------


async def _graph_load_node(db, state: KnowledgeSummaryState) -> dict:
    """节点 1：读取 curriculum_node + 校验 description + 幂等检查"""
    node_id = state["curriculum_node_id"]
    force_regenerate = state.get("force_regenerate", False)

    node = await _ks._get_node(db, node_id)
    node_type = node.get("node_type") or ""
    _ks._require_summary_node(node_type)

    if not node.get("description"):
        raise _ks.NoDescriptionError(f"节点无描述，不总结: {node_id}")

    if not _ks.LLM_SUMMARY_MODEL:
        raise _ks.LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法生成知识总结")

    # 幂等检查
    key = _ks.summary_idempotency_key(
        textbook_id=node.get("textbook_id") or "",
        grade=node.get("grade") or "",
        semester=node.get("semester") or "",
        unit_id=node.get("unit_id") or node.get("unit_no") or "",
        lesson_id=node.get("lesson_id") or node.get("lesson_no") or "",
        description_version=node.get("description_version") or 0,
        model=_ks.LLM_SUMMARY_MODEL,
    )

    existing = node.get("ai_summary") or {}
    if not force_regenerate and existing.get("idempotency_key") == key:
        # 幂等命中：直接返回已有结果
        return {
            "node": node,
            "idempotency_key": key,
            "final_summary": existing,
            "status": _ks.SUMMARY_STATUS_SUCCESS,
            "retry_count": 0,
        }

    return {
        "node": node,
        "idempotency_key": key,
        "retry_count": 0,
    }


async def _graph_generate_summary(db, state: KnowledgeSummaryState) -> dict:
    """节点 2：调火山 LLM_SUMMARY_MODEL 生成知识总结 JSON"""
    node = state["node"]
    node_id = state["curriculum_node_id"]
    key = state.get("idempotency_key", "")
    include_extended_points = state.get("include_extended_points", True)

    try:
        result = await _ks._call_summary_llm(
            node, include_extended_points=include_extended_points
        )
    except _ks.LLMResponseError as e:
        # LLM 失败：写回 status=failed + failed 审计（与原 F1 行为一致）
        await _persist_ai_summary_with_eval(
            db,
            node_id,
            status=_ks.SUMMARY_STATUS_FAILED,
            idempotency_key=key,
            model=_ks.LLM_SUMMARY_MODEL,
            knowledge_points=[],
            extended_points=[],
            quality_score=QUALITY_SCORE_NOT_EVALUATED,
            evaluation_feedback=f"LLM 生成失败: {e}",
            evaluation_model="",
            graph_version=_GRAPH_VERSION,
        )
        await write_audit(
            db,
            action=AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
            object_ref=node_id,
            result=AUDIT_RESULT_FAILED,
            context={
                "idempotency_key": key,
                "model": _ks.LLM_SUMMARY_MODEL,
            },
        )
        raise
    return {"parsed_summary": result}


async def _graph_evaluate_summary(state: KnowledgeSummaryState) -> dict:
    """节点 3：混元模型评估总结质量（降级不阻塞）"""
    node = state["node"]
    description_raw = node.get("description") or {}
    if isinstance(description_raw, dict):
        description_str = json.dumps(description_raw, ensure_ascii=False)
    else:
        description_str = str(description_raw)

    evaluation = await _call_hunyuan_evaluate(
        description=description_str,
        summary=state["parsed_summary"],
    )
    return {
        "quality_score": evaluation["score"],
        "evaluation_feedback": evaluation.get("feedback", ""),
    }


async def _graph_persist_summary(db, state: KnowledgeSummaryState) -> dict:
    """节点 4：写回 curriculum_node.ai_summary（含评估新字段）"""
    node_id = state["curriculum_node_id"]
    key = state.get("idempotency_key", "")
    parsed = state.get("parsed_summary", {})
    quality_score = state.get("quality_score", QUALITY_SCORE_NOT_EVALUATED)
    evaluation_feedback = state.get("evaluation_feedback", "")
    status = state.get("status") or _ks.SUMMARY_STATUS_SUCCESS

    # 评估模型名（评估跳过时为空）
    eval_model = HUNYUAN_EVAL_MODEL if quality_score >= 0 else ""

    # 使用扩展版持久化写入（含评估新字段）
    ai_summary = await _persist_ai_summary_with_eval(
        db,
        node_id,
        status=status,
        idempotency_key=key,
        model=_ks.LLM_SUMMARY_MODEL,
        knowledge_points=parsed.get("knowledge_points") or [],
        extended_points=parsed.get("extended_points") or [],
        quality_score=quality_score,
        evaluation_feedback=evaluation_feedback,
        evaluation_model=eval_model,
        graph_version=_GRAPH_VERSION,
    )

    # 审计日志
    await write_audit(
        db,
        action=AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
        object_ref=node_id,
        context={
            "node_type": state["node"].get("node_type", ""),
            "idempotency_key": key,
            "model": _ks.LLM_SUMMARY_MODEL,
            "quality_score": quality_score,
            "evaluation_model": eval_model,
            "graph_version": _GRAPH_VERSION,
        },
    )

    return {
        "final_summary": ai_summary,
        "status": status,
    }


# ---------------------------------------------------------------------------
# 扩展版 _persist_ai_summary（写入评估新字段）
# ---------------------------------------------------------------------------


async def _persist_ai_summary_with_eval(
    db,
    node_id: str,
    *,
    status: str,
    idempotency_key: str,
    model: str,
    knowledge_points: list,
    extended_points: list,
    quality_score: float = QUALITY_SCORE_NOT_EVALUATED,
    evaluation_feedback: str = "",
    evaluation_model: str = "",
    graph_version: int = _GRAPH_VERSION,
) -> dict:
    """写回 curriculum_node.ai_summary（契约 §4.12.8(b) 含评估新字段）"""
    now = int(time.time() * 1000)
    ai_summary = {
        "status": status,
        "generated_at": now,
        "model": model,
        "knowledge_points": knowledge_points,
        "extended_points": extended_points,
        "idempotency_key": idempotency_key,
        # 评估新字段（2026-08-21 新增）
        "quality_score": quality_score,
        "evaluation_feedback": evaluation_feedback,
        "evaluation_model": evaluation_model,
        "graph_version": graph_version,
    }
    result = await db.update(
        CURRICULUM_NODE_COLLECTION,
        where={"node_id": node_id},
        data={"$set": {"ai_summary": ai_summary, "updated_at": now}},
    )
    # 检查 matched_count：node_id 未匹配时回退按 _id 更新
    matched = result.get("matched_count", 0) if isinstance(result, dict) else 0
    if matched == 0:
        logger.warning(
            f"[_persist_ai_summary_with_eval] node_id={node_id} 未匹配，回退 _id 更新"
        )
        await db.update(
            CURRICULUM_NODE_COLLECTION,
            where={"_id": node_id},
            data={"$set": {"ai_summary": ai_summary, "updated_at": now}},
        )
    logger.info(
        f"[_persist_ai_summary_with_eval] node_id={node_id} status={status} "
        f"kp={len(knowledge_points)} ep={len(extended_points)} "
        f"score={quality_score} matched={matched}"
    )
    return ai_summary


# ---------------------------------------------------------------------------
# 条件边路由函数（设计文档 §2.3）
# ---------------------------------------------------------------------------


def _route_after_load(state: KnowledgeSummaryState) -> str:
    """load_node 后路由：有描述 → generate；已幂等命中 → END"""
    if state.get("status") == _ks.SUMMARY_STATUS_SUCCESS and state.get("final_summary"):
        return "end"
    return "generate"


def _route_after_evaluate(state: KnowledgeSummaryState) -> str:
    """evaluate_summary 后路由：pass → persist；retry → generate；degraded → persist"""
    score = state.get("quality_score", QUALITY_SCORE_NOT_EVALUATED)
    retry_count = state.get("retry_count", 0)

    # 评估跳过（混元不可用）→ 直接落盘
    if score < 0:
        return "persist"

    # 评估通过 → 落盘
    if score >= HUNYUAN_EVAL_PASS_THRESHOLD:
        return "persist"

    # 不达标且还有重试次数 → 回到 generate
    if retry_count < HUNYUAN_EVAL_MAX_RETRIES:
        return "regenerate"

    # 重试上限 → 降级落盘
    return "persist_degraded"


# ---------------------------------------------------------------------------
# 图构建与编译
# ---------------------------------------------------------------------------


def _build_summary_graph(db=None):
    """构建并编译 LangGraph 知识总结图

    db 参数通过 partial 绑定到需要数据库访问的节点函数（load_node / persist_summary），
    避免 LangGraph TypedDict 序列化丢失非声明字段。
    """
    graph = StateGraph(KnowledgeSummaryState)

    # 注册节点（db 通过 partial 绑定）
    graph.add_node("load_node", partial(_graph_load_node, db) if db else _graph_load_node)
    graph.add_node("generate_summary", partial(_graph_generate_summary, db) if db else _graph_generate_summary)
    graph.add_node("evaluate_summary", _graph_evaluate_summary)
    graph.add_node("persist_summary", partial(_graph_persist_summary, db) if db else _graph_persist_summary)

    # 入口边
    graph.add_edge(START, "load_node")

    # load_node 条件路由
    graph.add_conditional_edges(
        "load_node",
        _route_after_load,
        {
            "generate": "generate_summary",
            "end": END,
        },
    )

    # generate → evaluate
    graph.add_edge("generate_summary", "evaluate_summary")

    # evaluate 条件路由
    graph.add_conditional_edges(
        "evaluate_summary",
        _route_after_evaluate,
        {
            "persist": "persist_summary",
            "regenerate": "generate_summary",
            "persist_degraded": "persist_summary",
        },
    )

    # persist → END
    graph.add_edge("persist_summary", END)

    return graph.compile()


# 编译图（按 db 缓存）
_compiled_graphs: dict[int, Any] = {}


def _get_compiled_graph(db=None):
    """获取编译后的图（按 db 实例缓存）"""
    cache_key = id(db) if db is not None else 0
    if cache_key not in _compiled_graphs:
        _compiled_graphs[cache_key] = _build_summary_graph(db)
    return _compiled_graphs[cache_key]


# ---------------------------------------------------------------------------
# 顶层入口：_run_summary_graph（被 generateKnowledgeSummary 调用）
# ---------------------------------------------------------------------------


async def _run_summary_graph(
    db,
    *,
    curriculum_node_id: str,
    force_regenerate: bool = False,
    include_extended_points: bool = True,
) -> dict:
    """执行 LangGraph 知识总结图

    返回值结构与原 F1 generateKnowledgeSummary 完全一致：
    {summary_id, status, idempotency_key, knowledge_points, extended_points, generated_at}
    """
    # 节点函数通过 partial 绑定 db，不在 state 中传递
    initial_state: dict[str, Any] = {
        "curriculum_node_id": curriculum_node_id,
        "force_regenerate": force_regenerate,
        "include_extended_points": include_extended_points,
        "retry_count": 0,
        "status": "",
    }

    graph = _get_compiled_graph(db)

    # 失败处理：LLM 失败由 _graph_generate_summary 内部处理（写 failed + audit + re-raise）
    # 其他异常直接传播
    try:
        final_state = await graph.ainvoke(initial_state)
    except (_ks.NoDescriptionError, _ks.NodeTypeUnsupportedError, _ks.LLMResponseError):
        raise
    except Exception as e:
        logger.error(f"LangGraph 知识总结图执行失败: {type(e).__name__}: {e}")
        raise

    # 幂等命中（load_node 直接返回 final_summary）
    if final_state.get("status") == _ks.SUMMARY_STATUS_SUCCESS and final_state.get("final_summary"):
        existing = final_state["final_summary"]
        return {
            "summary_id": curriculum_node_id,
            "status": existing.get("status") or _ks.SUMMARY_STATUS_SUCCESS,
            "idempotency_key": existing.get("idempotency_key") or "",
            "knowledge_points": existing.get("knowledge_points") or [],
            "extended_points": existing.get("extended_points") or [],
            "generated_at": existing.get("generated_at") or 0,
        }

    # 正常完成
    ai_summary = final_state.get("final_summary") or {}
    # 根据 evaluate 路由结果确定最终 status
    score = final_state.get("quality_score", QUALITY_SCORE_NOT_EVALUATED)
    retry_count = final_state.get("retry_count", 0)
    if score < 0:
        final_status = _ks.SUMMARY_STATUS_SUCCESS  # 评估跳过，总结仍成功
    elif score >= HUNYUAN_EVAL_PASS_THRESHOLD:
        final_status = _ks.SUMMARY_STATUS_SUCCESS
    elif retry_count >= HUNYUAN_EVAL_MAX_RETRIES:
        final_status = _ks.SUMMARY_STATUS_DEGRADED
    else:
        final_status = _ks.SUMMARY_STATUS_SUCCESS

    return {
        "summary_id": curriculum_node_id,
        "status": ai_summary.get("status") or final_status,
        "idempotency_key": final_state.get("idempotency_key", ""),
        "knowledge_points": ai_summary.get("knowledge_points") or [],
        "extended_points": ai_summary.get("extended_points") or [],
        "generated_at": ai_summary.get("generated_at") or 0,
    }
