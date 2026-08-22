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

from config import (
    LLM_DISABLE_THINKING,
    LLM_SUMMARY_MODEL,
    USE_LANGGRAPH_SUMMARY,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
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
    """同步 chat 调用（在线程池中执行）

    推理模型默认禁用 thinking（LLM_DISABLE_THINKING），否则真实规模下
    推理耗时 >120s 会触发超时（同 Judge，见 error_scanner._call_judge_sync）。
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }
    if LLM_DISABLE_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(**kwargs)
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
    result = await db.update(
        CURRICULUM_NODE_COLLECTION,
        where={"node_id": node_id},
        data={"$set": {"ai_summary": ai_summary, "updated_at": now}},
    )
    # 检查 matched_count：node_id 未匹配时回退按 _id 更新
    matched = result.get("matched_count", 0) if isinstance(result, dict) else 0
    if matched == 0:
        logger.warning(
            f"[_persist_ai_summary] node_id={node_id} 未匹配，回退 _id 更新"
        )
        await db.update(
            CURRICULUM_NODE_COLLECTION,
            where={"_id": node_id},
            data={"$set": {"ai_summary": ai_summary, "updated_at": now}},
        )
    logger.info(
        f"[_persist_ai_summary] node_id={node_id} status={status} "
        f"kp={len(knowledge_points)} ep={len(extended_points)} matched={matched}"
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
    # feature flag：USE_LANGGRAPH_SUMMARY=true → 走 LangGraph 图编排（2026-08-21 SOP ⑤ K4）
    if USE_LANGGRAPH_SUMMARY:
        from services.math.summary_graph import _run_summary_graph

        return await _run_summary_graph(
            db,
            curriculum_node_id=curriculum_node_id,
            force_regenerate=force_regenerate,
            include_extended_points=include_extended_points,
        )

    # 以下为原直接调用路径（USE_LANGGRAPH_SUMMARY=false 时回退）
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


# ---------------------------------------------------------------------------
# G1.2 批量总结 + 人工修正（SOP §5 G1.2 · service-contract §7.7）
# ---------------------------------------------------------------------------
#
# 三接口：
#   1) batchGenerateKnowledgeSummary   — 按 textbook_id (+scope+node_ids) 调度批量生成
#   2) getBatchSummaryStatus           — 查 job 进度与 items
#   3) manualEditKnowledgeSummary      — 人工修正 knowledge_points/extended_points + 打标 manual_edited*
#
# 实现取舍（MVP）：
#   - 批任务状态存内存 dict（BATCH_JOBS），保留 TTL = BATCH_JOB_TTL_SEC（1h）
#   - 调度：生产 asyncio.create_task（后台），测试模式同步 await（_BATCH_RUN_SYNC 开关）
#   - 幂等键：完全复用 F1 summary_idempotency_key（无任何扩展，SOP 强制）
#   - 审计：批完成 1 条 generate_knowledge_summary；人工修正 1 条 manual_edit_summary
#   - overwrite_ai：只打标，MVP 不实现「下次重生成覆盖人工项」精细逻辑（SOP："仅打标标记"）

import secrets as _secrets  # noqa: E402  — 模块尾导入，避免影响 F1 路径

# 调度开关（测试用 monkeypatch 设 True）
_BATCH_RUN_SYNC: bool = False

# 批任务内存状态
BATCH_JOBS: dict[str, dict] = {}
BATCH_JOB_TTL_SEC: int = 3600  # 1h


def _batch_job_stats(job: dict) -> dict:
    """从 job 聚合 stats（routes 返回字段统一入口）。"""
    return {
        "success": job.get("success", 0),
        "blocked_no_desc": job.get("blocked", 0),
        "skipped_existing": job.get("skipped", 0),
        "failed": job.get("failed", 0),
        "total": job.get("total", 0),
    }


def _cleanup_expired_jobs(now_sec: float | None = None) -> None:
    """懒清理 BATCH_JOB_TTL_SEC 外的过期任务（每次访问 batch-status 或 dispatch 前触发）。"""
    now = now_sec if now_sec is not None else time.time()
    expired = [jid for jid, j in BATCH_JOBS.items() if (now - (j.get("created_at") or 0)) > BATCH_JOB_TTL_SEC]
    for jid in expired:
        BATCH_JOBS.pop(jid, None)


def _filter_candidates_by_scope(nodes: list[dict], scope: str) -> tuple[list[dict], int]:
    """按 scope 过滤本次要生成的节点，返回 (to_run, skipped_count)。

    SOP scope 语义：
      - not_generated_only：跳过 ai_summary.status = success/generating 的节点
      - all：只要是 SUMMARY_NODE_TYPES 的节点都跑（幂等键命中自然返回已有结果，但仍会调用 F1 函数）
      - force：全部都跑（调用时 force_regenerate=True 重置）
    """
    # 仅 SUMMARY_NODE_TYPES（lesson/knowledge_point 等）需要总结；年级/学期/unit 跳过（计入 skipped）
    summary_nodes = [n for n in nodes if n.get("node_type") in SUMMARY_NODE_TYPES]
    non_summary_skipped = len(nodes) - len(summary_nodes)
    if scope == "force" or scope == "all":
        return summary_nodes, non_summary_skipped
    # scope == "not_generated_only"（默认）
    to_run = []
    skipped_count = non_summary_skipped
    for n in summary_nodes:
        s = (n.get("ai_summary") or {}).get("status") or ""
        if s in (SUMMARY_STATUS_SUCCESS, SUMMARY_STATUS_GENERATING):
            skipped_count += 1
        else:
            to_run.append(n)
    return to_run, skipped_count


async def batchGenerateKnowledgeSummary(
    db,
    *,
    textbook_id: str,
    scope: str = "not_generated_only",
    node_ids: list[str] | None = None,
    actor: str = "",
) -> dict:
    """批量生成 AI 知识总结（路由层：POST /math/knowledge-summary/batch-generate）。

    返回：
      {job_id, status: running|done, scope, total_nodes, stats: {success,blocked,skipped,failed,total}, items?: [...]}
      同步模式下带 items + stats 最终值；异步模式仅 initial。
    """
    from services.math import (  # noqa: WPS433  — 延迟 import 防 math ↔ summary 循环（math/__init__.py 反向导入本模块）
        VALID_BATCH_SCOPE_SET,
        ConfirmationMismatchError,
        ERR_INVALID_BATCH_SCOPE,
        SUBJECT_TYPE_MATH,
        SUMMARY_STATUS_GENERATING,
        TextbookNotFoundError,
    )
    from services.audit import AUDIT_ACTION_MANUAL_EDIT_SUMMARY  # noqa: WPS433  — F1 模块没导入 G0.2 新常量
    from services.database import TEXTBOOK_V2  # noqa: WPS433
    # 复用 G1.1 教材存在性校验（不存在或非 math → TextbookNotFoundError 404，语义完全一致）
    from services.math.textbook_management import (  # noqa: WPS433 — 延迟 import
        _get_math_textbook_or_404,
    )

    _cleanup_expired_jobs()
    if not isinstance(scope, str) or scope not in VALID_BATCH_SCOPE_SET:
        raise ConfirmationMismatchError(
            ERR_INVALID_BATCH_SCOPE,
            f"scope 非法：{scope!r}；仅允许 {sorted(VALID_BATCH_SCOPE_SET)!r}",
        )

    # ---- 1. 校验 textbook 存在且为 math（复用 G1.1，避免重复 subject_type/missing 判断） ---- #
    tb = await _get_math_textbook_or_404(db, textbook_id)
    grade = tb.get("grade") or ""
    semester = tb.get("semester") or ""

    # ---- 2. 查询 curriculum_node：按 textbook_id + 可选 node_ids ---- #
    where: dict = {"textbook_id": textbook_id}
    if node_ids:
        where["node_id"] = {"$in": list(node_ids)}
    nodes_resp = await db.query(CURRICULUM_NODE_COLLECTION, where=where, limit=1000)
    # query() 返回 {records, total, offset, limit}（FakeDB / 真实 database.py 均保持该契约）
    nodes = nodes_resp.get("records", []) if isinstance(nodes_resp, dict) else list(nodes_resp)

    # ---- 3. scope 过滤候选 ---- #
    candidates, skipped = _filter_candidates_by_scope(nodes, scope)
    total = len(nodes)  # SOP total = 本次范围节点数（与 total_nodes 字段一致）

    # ---- 4. 创建 job ---- #
    now_s = time.time()
    job_id = f"batch_{int(now_s * 1000)}_{_secrets.token_hex(4)}"
    job: dict = {
        "job_id": job_id,
        "textbook_id": textbook_id,
        "grade": grade,
        "semester": semester,
        "scope": scope,
        "actor": actor,
        "status": "running",
        "total": total,
        "done": 0,
        "success": 0,
        "blocked": 0,
        "skipped": skipped,
        "failed": 0,
        "items": [],
        "created_at": now_s,
        "finished_at": 0,
    }
    BATCH_JOBS[job_id] = job

    # 对 force scope：先把 candidates 的 ai_summary.status → generating（按 SOP 语义）
    if scope == "force":
        for n in candidates:
            nid = n.get("node_id")
            ai = dict(n.get("ai_summary") or {})
            ai["status"] = SUMMARY_STATUS_GENERATING
            # 不落库：_per_node summarizer 会在 force_regenerate=True 时处理并最终落库

    # ---- 5. 定义批执行体 ---- #
    async def _run() -> None:
        success_n = 0
        blocked_n = 0
        failed_n = 0
        items_out: list[dict] = []
        logger.info(
            f"[batch_summary] job={job_id} 开始执行, candidates={len(candidates)}, "
            f"scope={scope}, textbook_id={textbook_id}"
        )
        if not candidates:
            logger.warning(f"[batch_summary] job={job_id} 无候选节点, candidates=0")
        for node in candidates:
            nid = node.get("node_id") or ""
            item: dict = {"node_id": nid, "status": "pending"}
            logger.info(f"[batch_summary] job={job_id} 开始处理 node_id={nid}")
            try:
                await generateKnowledgeSummary(
                    db,
                    curriculum_node_id=nid,
                    force_regenerate=(scope == "force"),
                    include_extended_points=True,
                )
                item["status"] = "success"
                success_n += 1
                logger.info(f"[batch_summary] job={job_id} node_id={nid} 成功")
            except NoDescriptionError as exc:
                item["status"] = "blocked_no_desc"
                item["error"] = str(exc)
                blocked_n += 1
                logger.warning(f"[batch_summary] job={job_id} node_id={nid} 无描述: {exc}")
            except Exception as exc:  # noqa: BLE001
                item["status"] = "failed"
                item["error"] = f"{type(exc).__name__}: {exc}"
                failed_n += 1
                logger.error(f"[batch_summary] job={job_id} node_id={nid} 失败: {type(exc).__name__}: {exc}")
            items_out.append(item)
            job["done"] += 1
            job["success"] = success_n
            job["blocked"] = blocked_n
            job["failed"] = failed_n
            job["items"] = items_out
        job["status"] = "failed" if (failed_n > 0 and success_n == 0 and blocked_n == 0) else "done"
        job["finished_at"] = time.time()
        logger.info(
            f"[batch_summary] job={job_id} 完成, status={job['status']}, "
            f"success={success_n} blocked={blocked_n} failed={failed_n}"
        )
        # 批完成 → 审计 1 条 generate_knowledge_summary（F1 同 action）
        stats_ctx = _batch_job_stats(job)
        try:
            await write_audit(
                db,
                action=AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY,
                object_ref=textbook_id,
                actor=actor,
                context={
                    "batch_job_id": job_id,
                    "scope": scope,
                    "textbook_id": textbook_id,
                    "grade": grade,
                    "semester": semester,
                    **stats_ctx,
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("batch_summary audit write failed: %s", exc)

    # ---- 6. 调度：sync(测试) 同步 await；生产 create_task ---- #
    if _BATCH_RUN_SYNC:
        await _run()
    else:
        asyncio.create_task(_run(), name=f"ks-batch-{job_id}")  # noqa: RUF006

    payload: dict = {
        "job_id": job_id,
        "status": job["status"],
        "scope": scope,
        "total_nodes": total,
        "stats": _batch_job_stats(job),
    }
    if _BATCH_RUN_SYNC:
        payload["items"] = job["items"]
    return payload


async def getBatchSummaryStatus(
    *,
    job_id: str,
) -> dict:
    """查询批任务进度（路由层：GET /math/knowledge-summary/batch-status）。"""
    from services.math import BatchJobNotFoundError  # noqa: WPS433 — 延迟 import 防循环

    _cleanup_expired_jobs()
    job = BATCH_JOBS.get(job_id)
    if job is None:
        raise BatchJobNotFoundError(job_id)
    total = job["total"] or 0
    done = job.get("done", 0)
    progress_pct = 100 if total == 0 else int(done * 100 / total)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "total": total,
        "done": done,
        "success": job.get("success", 0),
        "blocked_no_desc": job.get("blocked", 0),
        "skipped_existing": job.get("skipped", 0),
        "failed": job.get("failed", 0),
        "progress_pct": progress_pct,
        "items": job.get("items", []),
        "created_at": int(job.get("created_at", 0) * 1000),
        "finished_at": int((job.get("finished_at") or 0) * 1000),
    }


async def manualEditKnowledgeSummary(
    db,
    *,
    curriculum_node_id: str,
    editor_id: str = "",
    knowledge_points: list | None = None,
    extended_points: list | None = None,
    overwrite_ai: bool = False,
    actor: str = "",
) -> dict:
    """人工修正 AI 知识总结（契约 DM-3 manual_edited* 3 字段 · 路由层：POST /math/curriculum-node/{id}/manual-edit-summary）。

    SOP 规则：
      - knowledge_points is not None → 覆盖 ai_summary.knowledge_points
      - extended_points  is not None → 覆盖 ai_summary.extended_points
      - 其余：manual_edited=true / manual_edited_at / manual_edited_by（缺省 editor_id=actor）
      - overwrite_ai：打标（MVP 不影响任何行为；下次 force 重生成时由 F/G 实现语义）
      - 没有任何变更 → 400 MANUAL_EDIT_NOOP
    """
    from services.math import (  # noqa: WPS433 — 延迟 import
        ConfirmationMismatchError,
        ERR_MANUAL_EDIT_NOOP,
    )
    from services.audit import AUDIT_ACTION_MANUAL_EDIT_SUMMARY  # noqa: WPS433

    # NodeNotFoundError 抛出自定义 -> 路由层映射 NODE_NOT_FOUND + 404（KnowledgeSummaryError 子类）
    node = await _get_node(db, curriculum_node_id)
    ai_summary: dict = dict(node.get("ai_summary") or {})

    effective_editor = editor_id or actor or "anonymous"
    changed_points_count = 0

    if knowledge_points is not None:
        kp_list = list(knowledge_points)
        ai_summary["knowledge_points"] = kp_list
        changed_points_count += len(kp_list)
    if extended_points is not None:
        ep_list = list(extended_points)
        ai_summary["extended_points"] = ep_list
        changed_points_count += len(ep_list)

    if knowledge_points is None and extended_points is None:
        # 没有任何入参要改 → 400（DM-3 语义：人工修正必带至少 1 个字段）
        raise ConfirmationMismatchError(
            ERR_MANUAL_EDIT_NOOP,
            "人工修正至少提供 knowledge_points 或 extended_points 其中一个字段（允许空列表显式清空）。",
        )

    # DM-3：manual_edited 3 字段统一写入 ai_summary 下（与 F1 单节点 ai_summary 同位置）
    now_ms = int(time.time() * 1000)
    ai_summary["manual_edited"] = True
    ai_summary["manual_edited_at"] = now_ms
    ai_summary["manual_edited_by"] = effective_editor
    # overwrite_ai：MVP 仅打标
    ai_summary["overwrite_ai"] = bool(overwrite_ai)

    await db.update(
        CURRICULUM_NODE_COLLECTION,
        where={"node_id": curriculum_node_id},
        data={"$set": {"ai_summary": ai_summary, "updated_at": now_ms}},
    )

    # 审计：manual_edit_summary
    try:
        await write_audit(
            db,
            action=AUDIT_ACTION_MANUAL_EDIT_SUMMARY,
            object_ref=curriculum_node_id,
            actor=actor or effective_editor,
            context={
                "textbook_id": node.get("textbook_id") or "",
                "grade": node.get("grade") or "",
                "semester": node.get("semester") or "",
                "changed_points_count": changed_points_count,
                "overwrite_ai": bool(overwrite_ai),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("manual_edit_summary audit write failed: %s", exc)

    return {
        "node_id": curriculum_node_id,
        "manual_edited": True,
        "manual_edited_at": now_ms,
        "manual_edited_by": effective_editor,
        "changed_points_count": changed_points_count,
        "overwrite_ai_flag": bool(overwrite_ai),
    }
