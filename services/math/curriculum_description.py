"""数学教材描述服务（F2）

契约：
- service-contract.md §7.2 manageTextbookDescription（四端点共用逻辑）
- api-contract.md §3.10 F2 四接口
- data-model-contract.md §4.12.1 / §4.12.8 curriculum_node.description 字段展开

说明：
- description 承载在 curriculum_node 上，不另建集合。
- 版本化：每次人工编辑（save_manual）或 AI 草稿采纳（adopt）→ description_version+1，
  上一个版本写入 description_history（保留最近 N=10 版本快照，超出滚动出队）。
- 幂等键：{node_id}:v{description_version}；写入按 node_id + description_version 复合键幂等。
- 审计：三类操作（edit_description / draft_description / adopt_description）写 audit_log。
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from openai import OpenAI

from config import LLM_SUMMARY_MODEL, VOLCANO_API_KEY, VOLCANO_BASE_URL
from services.audit import (
    AUDIT_ACTION_ADOPT_DESCRIPTION,
    AUDIT_ACTION_DRAFT_DESCRIPTION,
    AUDIT_ACTION_EDIT_DESCRIPTION,
    write_audit,
)
from services.database import CURRICULUM_NODE_COLLECTION
from services.math import (
    DESCRIPTION_HISTORY_LIMIT,
    DESCRIPTION_NODE_TYPES,
    DESCRIPTION_SOURCE_AI_ADOPTED,
    DESCRIPTION_SOURCE_MANUAL,
    description_idempotency_key,
)

logger = logging.getLogger("scholar-admin.math.curriculum_description")

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
# 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class DescriptionError(Exception):
    """描述业务错误基类"""


class NodeNotFoundError(DescriptionError):
    """curriculum_node 不存在"""


class NodeTypeUnsupportedError(DescriptionError):
    """节点类型不要求 description"""


class DescriptionValidationError(DescriptionError):
    """description 结构 / 长度不合法"""


class LLMNotConfiguredError(DescriptionError):
    """LLM_SUMMARY_MODEL 未配置"""


class LLMGenerationError(DescriptionError):
    """草稿生成失败"""


# ---------------------------------------------------------------------------
# 描述结构校验（契约 §4.12.8(a)）
# ---------------------------------------------------------------------------


def _validate_description(description: Any) -> None:
    """校验 description 结构：
    summary(≤800字) / key_points[](每项≤120字) / typical_examples[]{ref,note} /
    prerequisites[] / teaching_tips[]
    """
    if not isinstance(description, dict):
        raise DescriptionValidationError("description 必须为对象")

    summary = description.get("summary", "")
    if not isinstance(summary, str) or len(summary) > 800:
        raise DescriptionValidationError("summary 必须为字符串且不超过 800 字")

    for field in ("key_points", "prerequisites", "teaching_tips"):
        items = description.get(field, [])
        if not isinstance(items, list):
            raise DescriptionValidationError(f"{field} 必须为数组")
        for item in items:
            if not isinstance(item, str):
                raise DescriptionValidationError(f"{field} 每项必须为字符串")
            if len(item) > 120:
                raise DescriptionValidationError(f"{field} 每项不超过 120 字")

    examples = description.get("typical_examples", [])
    if not isinstance(examples, list):
        raise DescriptionValidationError("typical_examples 必须为数组")
    for item in examples:
        if not isinstance(item, dict):
            raise DescriptionValidationError("typical_examples 每项必须为对象 {ref, note}")
        if not item.get("ref"):
            raise DescriptionValidationError("typical_examples 每项必须包含 ref")
        note = item.get("note", "")
        if not isinstance(note, str) or len(note) > 200:
            raise DescriptionValidationError("typical_examples 每项 note 不超过 200 字")


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


def _require_description_node(node_type: str) -> None:
    """F2 仅 unit / lesson / knowledge_point 三类节点生效（契约 §4.12.8(a)）"""
    if node_type not in DESCRIPTION_NODE_TYPES:
        raise NodeTypeUnsupportedError(
            f"节点类型 {node_type!r} 不要求 description（仅 {list(DESCRIPTION_NODE_TYPES)} 生效）"
        )


def _apply_version(
    node: dict, content: dict, source: str, updated_by: str
) -> tuple[int, list[dict]]:
    """版本化：description_version+1，上一个版本写入 description_history（保留 N=10）

    Returns:
        (new_version, new_history)
    """
    prev_version = node.get("description_version") or 0
    new_version = prev_version + 1
    history: list[dict] = list(node.get("description_history") or [])

    prev_description = node.get("description")
    if prev_description is not None:
        history.append(
            {
                "version": prev_version,
                "source": node.get("description_source") or "",
                "updated_at": node.get("description_updated_at") or 0,
                "updated_by": node.get("description_updated_by") or "",
                "snapshot": prev_description,
            }
        )
    # 保留最近 N=10 个版本快照（新版本在前）
    history = history[-DESCRIPTION_HISTORY_LIMIT:]
    return new_version, history


async def _persist_description(
    db, node_id: str, *, content: dict, source: str, updated_by: str, prev_version: int
) -> dict:
    """写入 description（按 node_id + description_version 复合键幂等）"""
    now = int(time.time() * 1000)
    new_version = prev_version + 1
    node = await _get_node(db, node_id)
    _, history = _apply_version(node, content, source, updated_by)

    # 复合键幂等：非首次写入带 description_version 条件，防止并发重复推进版本；
    # 首次写入（节点尚无 description_version 字段）不带版本条件
    where: dict[str, Any] = {"node_id": node_id}
    if prev_version > 0:
        where["description_version"] = prev_version

    result = await db.update(
        CURRICULUM_NODE_COLLECTION,
        where=where,
        data={
            "$set": {
                "description": content,
                "description_version": new_version,
                "description_source": source,
                "description_history": history,
                "description_updated_at": now,
                "description_updated_by": updated_by,
                "updated_at": now,
            }
        },
    )
    return {"new_version": new_version, "history_size": len(history), "update": result}


# ---------------------------------------------------------------------------
# F2 四端点共用逻辑（service-contract §7.2）
# ---------------------------------------------------------------------------


async def get_description(db, node_type: str, node_id: str) -> dict:
    """GET 当前描述：description + description_version + description_history"""
    node = await _get_node(db, node_id)
    node_type = node.get("node_type") or node_type

    # 非描述节点返回 description=null（契约口径），仍返回节点元信息
    if node_type not in DESCRIPTION_NODE_TYPES:
        return {
            "node_id": node_id,
            "node_type": node_type,
            "description": None,
            "description_version": node.get("description_version") or 0,
            "description_source": node.get("description_source") or "",
            "description_history": [],
        }

    return {
        "node_id": node_id,
        "node_type": node_type,
        "description": node.get("description"),
        "description_version": node.get("description_version") or 0,
        "description_source": node.get("description_source") or "",
        "description_history": node.get("description_history") or [],
    }


async def save_description(
    db, node_type: str, node_id: str, content: dict, updated_by: str
) -> dict:
    """POST 保存人工描述：source=manual，版本化 + 历史，写审计 edit_description"""
    _validate_description(content)
    node = await _get_node(db, node_id)
    node_type = node.get("node_type") or node_type
    _require_description_node(node_type)

    prev_version = node.get("description_version") or 0
    await _persist_description(
        db,
        node_id,
        content=content,
        source=DESCRIPTION_SOURCE_MANUAL,
        updated_by=updated_by,
        prev_version=prev_version,
    )

    # 审计：人工编辑（必审，action=edit_description，actor=家长/教师账号 ID）
    await write_audit(
        db,
        action=AUDIT_ACTION_EDIT_DESCRIPTION,
        object_ref=node_id,
        actor=updated_by,
        context={
            "node_type": node_type,
            "idempotency_key": description_idempotency_key(node_id, prev_version + 1),
        },
    )
    return {
        "node_id": node_id,
        "description_version": prev_version + 1,
        "description_source": DESCRIPTION_SOURCE_MANUAL,
        "description_history_size": len(node.get("description_history") or []) + 1,
    }


async def generate_draft(
    db, node_type: str, node_id: str, force_regenerate: bool = False
) -> dict:
    """POST 草稿生成：调用 LLM_SUMMARY_MODEL，不写正式描述，写审计 draft_description"""
    node = await _get_node(db, node_id)
    node_type = node.get("node_type") or node_type
    _require_description_node(node_type)

    if not LLM_SUMMARY_MODEL:
        raise LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法生成描述草稿")

    draft = await _call_summary_llm(node)
    await write_audit(
        db,
        action=AUDIT_ACTION_DRAFT_DESCRIPTION,
        object_ref=node_id,
        actor="",
        context={
            "node_type": node_type,
            "model": LLM_SUMMARY_MODEL,
            "force_regenerate": bool(force_regenerate),
            "description_version": node.get("description_version") or 0,
        },
    )
    return {
        "node_id": node_id,
        "draft": draft,
        "model": LLM_SUMMARY_MODEL,
        "source_versions": [
            {
                "version": node.get("version") or "",
                "description_version": node.get("description_version") or 0,
            }
        ],
    }


async def adopt_draft(
    db, node_type: str, node_id: str, content: dict, updated_by: str
) -> dict:
    """POST 草稿采纳：source=ai_adopted，版本化 + 历史，写审计 adopt_description"""
    _validate_description(content)
    node = await _get_node(db, node_id)
    node_type = node.get("node_type") or node_type
    _require_description_node(node_type)

    prev_version = node.get("description_version") or 0
    await _persist_description(
        db,
        node_id,
        content=content,
        source=DESCRIPTION_SOURCE_AI_ADOPTED,
        updated_by=updated_by,
        prev_version=prev_version,
    )
    await write_audit(
        db,
        action=AUDIT_ACTION_ADOPT_DESCRIPTION,
        object_ref=node_id,
        actor=updated_by,
        context={
            "node_type": node_type,
            "idempotency_key": description_idempotency_key(node_id, prev_version + 1),
        },
    )
    return {
        "node_id": node_id,
        "description_version": prev_version + 1,
        "description_source": DESCRIPTION_SOURCE_AI_ADOPTED,
    }


async def list_description_history(
    db, *, node_id: str, limit: int = DESCRIPTION_HISTORY_LIMIT
) -> list[dict]:
    """读取描述历史快照（最近 limit 个，新版本在前）"""
    node = await _get_node(db, node_id)
    history = node.get("description_history") or []
    return history[-limit:]


async def manage_textbook_description(
    db, *, op: str, node_id: str, payload: Optional[dict] = None
) -> dict:
    """F2 教材描述四端点共用入口（service-contract §7.2）

    op: get / save_manual / draft / adopt
    """
    payload = payload or {}
    node_type = payload.get("node_type") or ""
    if op == "get":
        return await get_description(db, node_type, node_id)
    if op == "save_manual":
        return await save_description(
            db,
            node_type,
            node_id,
            content=payload.get("description"),
            updated_by=payload.get("updated_by") or "",
        )
    if op == "draft":
        return await generate_draft(
            db,
            node_type,
            node_id,
            force_regenerate=bool(payload.get("force_regenerate", False)),
        )
    if op == "adopt":
        return await adopt_draft(
            db,
            node_type,
            node_id,
            content=payload.get("description"),
            updated_by=payload.get("updated_by") or "",
        )
    raise ValueError(f"不支持的 op: {op!r}")


# ---------------------------------------------------------------------------
# LLM 草稿生成
# ---------------------------------------------------------------------------

_DRAFT_SYSTEM_PROMPT = (
    "你是小学数学教材教研助手。根据给定教材节点信息，生成结构化的教材描述。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_DRAFT_USER_TEMPLATE = """请为以下数学教材节点生成教材描述。

节点信息：
- 类型：{node_type}
- 名称：{title}
- 编码：{code}
- 年级：{grade}（{semester}）
- 单元：{unit_title}
- 课时：{lesson_title}
- 能力点：{abilities}
- 先修依赖：{prereq}
- 内容类型：{content_type}

要求：
1. summary：一段总结，≤800 字，说明本节核心内容与教学定位。
2. key_points：关键要点列表，每项 ≤120 字。
3. typical_examples：典型例题引用列表，每项 {{"ref": "教材题号或题目引用", "note": "讲解要点"}}。
4. prerequisites：先修要点列表（自然语言）。
5. teaching_tips：教学提示列表（家长陪练用）。

输出 JSON 结构：
{{
  "summary": "...",
  "key_points": ["...", "..."],
  "typical_examples": [{{"ref": "...", "note": "..."}}],
  "prerequisites": ["...", "..."],
  "teaching_tips": ["...", "..."]
}}"""


def _call_chat_sync(client: OpenAI, model: str, prompt: str) -> str:
    """同步 chat 调用（在线程池中执行）"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()


async def _call_summary_llm(node: dict) -> dict:
    """调用 LLM_SUMMARY_MODEL 生成描述草稿（JSON 解析，失败抛 LLMGenerationError）"""
    from services.dialogue import _parse_json_response

    prompt = _DRAFT_USER_TEMPLATE.format(
        node_type=node.get("node_type") or "",
        title=node.get("title") or "",
        code=node.get("code") or "",
        grade=node.get("grade") or "",
        semester=node.get("semester") or "",
        unit_title=node.get("unit_title") or "",
        lesson_title=node.get("lesson_title") or "",
        abilities="、".join(node.get("abilities") or []) or "无",
        prereq="、".join(node.get("prereq") or []) or "无",
        content_type=node.get("content_type") or "",
    )
    try:
        client = _get_llm_client()
        response = await asyncio.to_thread(_call_chat_sync, client, LLM_SUMMARY_MODEL, prompt)
        draft = _parse_json_response(response)
    except Exception as e:
        logger.error(f"草稿生成失败 node={node.get('node_id')}: {type(e).__name__}: {e}")
        raise LLMGenerationError(f"草稿生成失败: {str(e)}") from e

    _validate_description(draft)
    return draft
