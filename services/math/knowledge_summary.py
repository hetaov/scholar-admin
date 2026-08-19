"""AI 知识总结服务（F1）

契约：
- service-contract.md §7.3 generateKnowledgeSummary / getKnowledgeSummary
- data-model-contract.md §4.12.8(b) ai_summary 结构与幂等键
- api-contract.md §3.10 POST /math/knowledge-summary/generate、GET /math/knowledge-summary/{curriculum_node_id}
- ADR-0019（数学教材描述与 AI 知识点总结）

说明：
- ai_summary 承载在 curriculum_node 上，不另建集合。
- 幂等：sha256(f"{textbook_id}|{grade}|{semester}|{unit_id}|{lesson_id}|{description_version}|{model}")。
- 无描述不总结：节点 description 为空 → NoDescriptionError（不写库）。
- 状态机：pending → generating → success/failed（幂等命中直接返回已有结果，不调用 LLM）。
- 查询：getKnowledgeSummary 未生成 → {status: "not_generated"}（不报错）。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from openai import OpenAI

from config import LLM_SUMMARY_MODEL, VOLCANO_API_KEY, VOLCANO_BASE_URL
from services.audit import (
    AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
    AUDIT_RESULT_FAILED,
    write_audit,
)
from services.database import CURRICULUM_NODE_COLLECTION

logger = logging.getLogger("scholar-admin.math.knowledge_summary")

# 单例 LLM 客户端（火山方舟 OpenAI 兼容）
_llm_client: OpenAI | None = None


def _get_llm_client() -> OpenAI:
    """获取总结模型客户端（单次调用最长 60 秒）"""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=VOLCANO_API_KEY,
            base_url=VOLCANO_BASE_URL,
            timeout=60.0,
        )
    return _llm_client


# ---------------------------------------------------------------------------
# 常量（契约 §4.12.8(b)）
# ---------------------------------------------------------------------------

# ai_summary.status 枚举
SUMMARY_STATUS_PENDING = "pending"
SUMMARY_STATUS_GENERATING = "generating"
SUMMARY_STATUS_SUCCESS = "success"
SUMMARY_STATUS_FAILED = "failed"
SUMMARY_STATUS_DEGRADED = "degraded"
# getKnowledgeSummary 未生成语义（契约 §7.3 / api-contract §3.10 GET，不报错）
SUMMARY_STATUS_NOT_GENERATED = "not_generated"

# F1 适用节点类型（与 F2 一致：仅 unit / lesson / knowledge_point 有描述可总结）
SUMMARY_NODE_TYPES = ("unit", "lesson", "knowledge_point")

# knowledge_points[].ability_dimensions 枚举（契约 §4.12.8(b)）
ABILITY_DIMENSIONS = ("arithmetic", "computation", "modeling", "reasoning")

# extended_points[].difficulty_band 三档（奥数扩展，契约 §4.12.8(b)）
EXTENDED_DIFFICULTY_BANDS = ("入门", "普及", "竞赛")


# ---------------------------------------------------------------------------
# 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class KnowledgeSummaryError(Exception):
    """AI 知识总结业务错误基类"""


class NodeNotFoundError(KnowledgeSummaryError):
    """curriculum_node 不存在"""


class NoDescriptionError(KnowledgeSummaryError):
    """节点无描述，不总结"""


class NodeTypeUnsupportedError(KnowledgeSummaryError):
    """节点类型不要求知识总结"""


class LLMNotConfiguredError(KnowledgeSummaryError):
    """LLM_SUMMARY_MODEL 未配置，无法生成知识总结"""


class LLMResponseError(KnowledgeSummaryError):
    """LLM 响应解析失败"""


# ---------------------------------------------------------------------------
# 节点读取
# ---------------------------------------------------------------------------


async def _get_node(db, node_id: str) -> dict:
    """按 node_id 读取 curriculum_node，不存在抛 NodeNotFoundError"""
    if not node_id:
        raise NodeNotFoundError("缺少 node_id")
    res = await db.query(
        CURRICULUM_NODE_COLLECTION,
        where={"node_id": node_id},
        limit=1,
    )
    records = res.get("records") or []
    if not records:
        # 兜底：node_id 即 _id 的存储形态
        res = await db.query(
            CURRICULUM_NODE_COLLECTION,
            where={"_id": node_id},
            limit=1,
        )
        records = res.get("records") or []
    if not records:
        raise NodeNotFoundError(f"curriculum_node 不存在: {node_id}")
    return records[0]


# ---------------------------------------------------------------------------
# 幂等键（契约 §4.12.8(b)）
# ---------------------------------------------------------------------------


def summary_idempotency_key(
    *,
    textbook_id: str,
    grade: str,
    semester: str,
    unit_id: str,
    lesson_id: str,
    description_version: int,
    model: str,
) -> str:
    """ai_summary.idempotency_key：sha256 六要素拼接

    同输入同模型二次调用幂等；任一要素（尤其 description_version / model）变化 → key 变化。
    """
    raw = (
        f"{textbook_id}|{grade}|{semester}|{unit_id}|{lesson_id}"
        f"|{description_version}|{model}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# ai_summary 空结构（契约 §4.12.8(b) 字段）
# ---------------------------------------------------------------------------


def empty_ai_summary(
    *,
    model: str = "",
    idempotency_key: str = "",
    status: str = SUMMARY_STATUS_PENDING,
) -> dict[str, Any]:
    """构造 ai_summary 空结构：status / generated_at / model / knowledge_points[] /
    extended_points[] / idempotency_key
    """
    return {
        "status": status,
        "generated_at": 0,
        "model": model,
        "knowledge_points": [],
        "extended_points": [],
        "idempotency_key": idempotency_key,
    }


def _require_summary_node(node_type: str) -> None:
    """F1 仅 unit / lesson / knowledge_point 三类节点可总结（契约 §4.12.8(b)）"""
    if node_type not in SUMMARY_NODE_TYPES:
        raise NodeTypeUnsupportedError(
            f"节点类型 {node_type!r} 不要求知识总结（仅 {list(SUMMARY_NODE_TYPES)} 生效）"
        )


# ---------------------------------------------------------------------------
# Prompt 模板（契约 §4.12.8(b)：knowledge_points + extended_points）
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = (
    "你是小学数学教材教研助手。根据给定教材节点信息，生成结构化的 AI 知识总结。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_SUMMARY_USER_TEMPLATE = """请为以下数学教材节点生成 AI 知识总结。

节点信息：
- 教材版本：{textbook_id}（{title}）
- 类型：{node_type}
- 年级：{grade}（{semester}）
- 单元：{unit_title}
- 课时：{lesson_title}
- 教材描述：{description}

要求：
1. knowledge_points：核心知识点清单，每项 {{"name": "知识点名", "summary": "一句话总结", "ability_dimensions": ["arithmetic|computation|modeling|reasoning"], "source_node_id": "来源节点ID", "source_lesson_id": "来源课时ID"}}。
2. extended_points：奥数扩展点清单（{include_extended_points}），每项 {{"name": "扩展点名", "summary": "说明", "difficulty_band": "入门|普及|竞赛", "related_knowledge_name": "关联知识点名", "source_lesson_id": "来源课时ID"}}。

输出 JSON 结构：
{{
  "knowledge_points": [...],
  "extended_points": [...]
}}"""


def _call_chat_sync(client: OpenAI, model: str, prompt: str) -> str:
    """同步 chat 调用（在线程池中执行）"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


def _validate_summary_result(result: dict, *, include_extended_points: bool) -> None:
    """校验 LLM 输出结构（契约 §4.12.8(b)），非法抛 LLMResponseError"""
    if not isinstance(result, dict):
        raise LLMResponseError("知识总结结果必须为 JSON 对象")
    kps = result.get("knowledge_points")
    if not isinstance(kps, list) or not kps:
        raise LLMResponseError("knowledge_points 必须为非空数组")
    for kp in kps:
        if not isinstance(kp, dict) or not kp.get("name"):
            raise LLMResponseError("knowledge_points 每项必须含 name")
        dims = kp.get("ability_dimensions") or []
        if not isinstance(dims, list) or any(d not in ABILITY_DIMENSIONS for d in dims):
            raise LLMResponseError("ability_dimensions 必须为合法枚举子集")
    if include_extended_points:
        eps = result.get("extended_points") or []
        if not isinstance(eps, list):
            raise LLMResponseError("extended_points 必须为数组")
        for ep in eps:
            if not isinstance(ep, dict) or not ep.get("name"):
                raise LLMResponseError("extended_points 每项必须含 name")
            if ep.get("difficulty_band") not in EXTENDED_DIFFICULTY_BANDS:
                raise LLMResponseError(
                    f"difficulty_band 必须为 {list(EXTENDED_DIFFICULTY_BANDS)} 之一"
                )
    else:
        result.setdefault("extended_points", [])


async def _call_summary_llm(node: dict, *, include_extended_points: bool) -> dict:
    """调用 LLM_SUMMARY_MODEL 生成知识总结（JSON 解析失败重试 1 次，仍失败抛 LLMResponseError）"""
    from services.dialogue import _parse_json_response

    prompt = _SUMMARY_USER_TEMPLATE.format(
        textbook_id=node.get("textbook_id") or "",
        title=node.get("title") or "",
        node_type=node.get("node_type") or "",
        grade=node.get("grade") or "",
        semester=node.get("semester") or "",
        unit_title=node.get("unit_title") or "",
        lesson_title=node.get("lesson_title") or "",
        description=json.dumps(node.get("description") or {}, ensure_ascii=False),
        include_extended_points="是" if include_extended_points else "否",
    )
    client = _get_llm_client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_chat_sync, client, LLM_SUMMARY_MODEL, prompt
            )
            result = _parse_json_response(response)
            _validate_summary_result(
                result, include_extended_points=include_extended_points
            )
            return result
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 LLMResponseError
            last_err = e
            logger.warning(
                f"知识总结生成失败（第 {attempt} 次）node={node.get('node_id')}: "
                f"{type(e).__name__}: {e}"
            )
    raise LLMResponseError(f"AI 知识总结生成失败: {last_err}")


async def _persist_ai_summary(
    db,
    node_id: str,
    *,
    status: str,
    idempotency_key: str,
    model: str,
    knowledge_points: list,
    extended_points: list,
) -> dict:
    """写回 curriculum_node.ai_summary（契约 §4.12.8(b) 字段）"""
    now = int(time.time() * 1000)
    ai_summary = {
        "status": status,
        "generated_at": now,
        "model": model,
        "knowledge_points": knowledge_points,
        "extended_points": extended_points,
        "idempotency_key": idempotency_key,
    }
    await db.update(
        CURRICULUM_NODE_COLLECTION,
        where={"node_id": node_id},
        data={"$set": {"ai_summary": ai_summary, "updated_at": now}},
    )
    return ai_summary


# ---------------------------------------------------------------------------
# F1 主入口：generateKnowledgeSummary（service-contract §7.3 / api-contract §3.10）
# ---------------------------------------------------------------------------


async def generateKnowledgeSummary(
    db,
    *,
    curriculum_node_id: str,
    force_regenerate: bool = False,
    include_extended_points: bool = True,
) -> dict:
    """生成 AI 知识总结（或幂等返回已有结果）

    前置校验：
    - 节点不存在 → NodeNotFoundError
    - 节点 description 为空 → NoDescriptionError（"无描述不总结"，不写库）
    - 节点类型不在 SUMMARY_NODE_TYPES → NodeTypeUnsupportedError
    - LLM_SUMMARY_MODEL 未配置 → LLMNotConfiguredError

    幂等：ai_summary.idempotency_key 相同 → 直接返回已有结果（不调用 LLM）。
    状态机：pending → generating → success / failed。
    """
    node = await _get_node(db, curriculum_node_id)
    node_type = node.get("node_type") or ""
    _require_summary_node(node_type)

    if not node.get("description"):
        raise NoDescriptionError(f"节点无描述，不总结: {curriculum_node_id}")

    if not LLM_SUMMARY_MODEL:
        raise LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法生成知识总结")

    key = summary_idempotency_key(
        textbook_id=node.get("textbook_id") or "",
        grade=node.get("grade") or "",
        semester=node.get("semester") or "",
        unit_id=node.get("unit_id") or node.get("unit_no") or "",
        lesson_id=node.get("lesson_id") or node.get("lesson_no") or "",
        description_version=node.get("description_version") or 0,
        model=LLM_SUMMARY_MODEL,
    )

    existing = node.get("ai_summary") or {}
    if not force_regenerate and existing.get("idempotency_key") == key:
        # 幂等命中：直接返回已有结果（不调用 LLM）
        return {
            "summary_id": curriculum_node_id,
            "status": existing.get("status") or SUMMARY_STATUS_PENDING,
            "idempotency_key": existing.get("idempotency_key") or key,
            "knowledge_points": existing.get("knowledge_points") or [],
            "extended_points": existing.get("extended_points") or [],
            "generated_at": existing.get("generated_at") or 0,
        }

    try:
        result = await _call_summary_llm(
            node, include_extended_points=include_extended_points
        )
    except LLMResponseError as e:
        # 解析失败重试后仍失败 → 写回 status=failed 并返回明确错误
        await _persist_ai_summary(
            db,
            curriculum_node_id,
            status=SUMMARY_STATUS_FAILED,
            idempotency_key=key,
            model=LLM_SUMMARY_MODEL,
            knowledge_points=[],
            extended_points=[],
        )
        await write_audit(
            db,
            action=AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
            object_ref=curriculum_node_id,
            result=AUDIT_RESULT_FAILED,
            context={
                "node_type": node_type,
                "idempotency_key": key,
                "model": LLM_SUMMARY_MODEL,
            },
        )
        raise

    ai_summary = await _persist_ai_summary(
        db,
        curriculum_node_id,
        status=SUMMARY_STATUS_SUCCESS,
        idempotency_key=key,
        model=LLM_SUMMARY_MODEL,
        knowledge_points=result.get("knowledge_points") or [],
        extended_points=result.get("extended_points") or [],
    )
    await write_audit(
        db,
        action=AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
        object_ref=curriculum_node_id,
        context={
            "node_type": node_type,
            "idempotency_key": key,
            "model": LLM_SUMMARY_MODEL,
            "force_regenerate": bool(force_regenerate),
            "include_extended_points": bool(include_extended_points),
        },
    )
    return {
        "summary_id": curriculum_node_id,
        "status": SUMMARY_STATUS_SUCCESS,
        "idempotency_key": key,
        "knowledge_points": ai_summary["knowledge_points"],
        "extended_points": ai_summary["extended_points"],
        "generated_at": ai_summary["generated_at"],
    }


async def getKnowledgeSummary(db, *, curriculum_node_id: str) -> dict:
    """取节点最新 AI 知识总结（service-contract §7.3 / api-contract §3.10 GET）

    - 节点不存在 → NodeNotFoundError（路由层映射 404）
    - ai_summary 未生成 → 返回 {status: "not_generated"}（不报错）
    - 已生成 → 返回完整 ai_summary（curriculum_node_id + ai_summary 字段展开）
    """
    node = await _get_node(db, curriculum_node_id)
    ai_summary = node.get("ai_summary")
    if not ai_summary:
        return {
            "curriculum_node_id": curriculum_node_id,
            "status": SUMMARY_STATUS_NOT_GENERATED,
            "generated_at": 0,
            "model": "",
            "knowledge_points": [],
            "extended_points": [],
            "idempotency_key": "",
        }
    return {
        "curriculum_node_id": curriculum_node_id,
        "status": ai_summary.get("status") or SUMMARY_STATUS_PENDING,
        "generated_at": ai_summary.get("generated_at") or 0,
        "model": ai_summary.get("model") or "",
        "knowledge_points": ai_summary.get("knowledge_points") or [],
        "extended_points": ai_summary.get("extended_points") or [],
        "idempotency_key": ai_summary.get("idempotency_key") or "",
    }
