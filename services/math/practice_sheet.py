"""A4 练习纸生成服务（F3.1）

契约：
- service-contract.md §7.3 generatePracticeSheet（2026-08-19 扩展 source / knowledge_points / include_extended_points）
- data-model-contract.md §4.12.4 practice_sheet / practice_sheet_item、§4.12.10(a) source 扩展
- api-contract.md §3.10 POST /math/practice-sheet
- ADR-0010（错题练习纸：状态机 / 防背题 / 必审审计）、ADR-0021（A4 练习纸对齐 AI 知识点）

说明：
- 三种选题源：`wrong_book`（错题本，默认）/ `ai_knowledge`（F1 知识点清单）/ `mixed`（错题 + 知识点混合）。
- `ai_knowledge`：按 `knowledge_points[].name` 精确匹配 `curriculum_node.ai_summary.knowledge_points[]`，
  未匹配项抛 KnowledgePointNotMatchedError（路由层映射 400）；每知识点按 `question_count_per_knowledge`
  （MVP 默认 2）出变式题，LLM 返回不足时降为实际数量并记 warnings（"知识点题量不足"）。
- `include_extended_points=true`：从 `ai_summary.extended_points[]`（difficulty_band ∈ {入门, 普及, 竞赛}）
  额外出奥数拔高题，列在主练习后，难度按 band 标注。
- 落库 practice_sheet（含 source / knowledge_points / include_extended_points）+ sheet_render_job（queued）
  + audit_log（action=generate，context 扩展 source / knowledge_points）。
- 幂等：同 scholar 10 分钟内相同参数（签名一致）重复生成 → 返回最近已有练习纸，不重复调 LLM。
- 沿用 ADR-0010：items 洗牌、答案落库不出参、status=generated；渲染 / 相似度校验为后续渲染任务。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import secrets
import time
from collections import Counter, defaultdict
from typing import Any

from config import LLM_DISABLE_THINKING, LLM_SUMMARY_MODEL
from services.audit import AUDIT_ACTION_GENERATE, write_audit
from services.database import (
    CURRICULUM_NODE_COLLECTION,
    ERROR_RECORD_COLLECTION,
    PRACTICE_SHEET_COLLECTION,
    SHEET_RENDER_JOB_COLLECTION,
)
from services.math.knowledge_summary import (
    LLMNotConfiguredError,
    LLMResponseError,
    _get_llm_client,
)

logger = logging.getLogger("scholar-admin.math.practice_sheet")

# ---------------------------------------------------------------------------
# 常量（契约 §4.12.4 / §4.12.10(a) / api-contract §3.10）
# ---------------------------------------------------------------------------

# practice_sheet.source 枚举（契约 §4.12.10(a)）
SHEET_SOURCE_WRONG_BOOK = "wrong_book"
SHEET_SOURCE_AI_KNOWLEDGE = "ai_knowledge"
SHEET_SOURCE_MIXED = "mixed"
SHEET_SOURCES = (SHEET_SOURCE_WRONG_BOOK, SHEET_SOURCE_AI_KNOWLEDGE, SHEET_SOURCE_MIXED)

# practice_sheet.status 状态机（ADR-0010：generated → printed → returned → evaluated）
SHEET_STATUS_GENERATED = "generated"
SHEET_STATUS_PRINTED = "printed"
SHEET_STATUS_RETURNED = "returned"
SHEET_STATUS_EVALUATED = "evaluated"

# sheet_render_job.status（异步渲染管线，F3.1 仅入队）
RENDER_JOB_QUEUED = "queued"

# 版式模板（契约 §4.12.5；MVP 仅 standard）
SHEET_TEMPLATE_STANDARD = "standard"
SHEET_TEMPLATE_ID_STANDARD = "tpl_standard"
SHEET_TEMPLATE_VERSION = 1

# 变式等级（ADR-0010：MVP L1/L2；LLM 按知识点生成的题目记为 L1 基础变式）
VARIANT_LEVEL_L1 = "L1"

# 内部标记：奥数扩展题（洗牌时区分，落库前剥离，非契约字段）
_EXTENDED_ITEM_FLAG = "_extended"

# 默认参数（契约 §3.10）
DEFAULT_QUESTION_COUNT_PER_KNOWLEDGE = 2
DEFAULT_WRONG_BOOK_RATIO = 0.5
MAX_SHEET_NODES = 3                       # nodes ≤3（错题练习纸 A4 篇幅）
MAX_ERROR_RECORDS_SCAN = 200              # 错题聚合扫描上限
SHEET_REPEAT_WINDOW_MS = 10 * 60 * 1000   # 幂等窗口：10 分钟

# 出题并发与时延预算（超时根因修复，详见 scripts/math_practice_sheet_probe.py）：
# 小程序 callContainer 上限 15s（tcb.js timeout=15000）。知识点间出题互不依赖，
# 串行最多 6 次（3 基础 + 3 奥数）在 15s 内必然超时 → 有界并发 + 单次调用限时。
MAX_LLM_CONCURRENCY = 3                   # 同时进行中的 LLM 出题数（控第三方并发限流）
SHEET_LLM_CALL_TIMEOUT_SEC = 12.0         # 单次 LLM 调用上限（s）；超时不再整次重试
                                          # （2×timeout 会突破 15s 容器上限）

# 能力维度中文标签（出题 prompt 用）
_ABILITY_DIM_LABELS = {
    "arithmetic": "算术",
    "computation": "计算",
    "modeling": "建模",
    "reasoning": "推理",
}

# extended_points[].difficulty_band → 题目难度（1~5，奥数整体拔高）
_EXTENDED_BAND_DIFFICULTY = {"入门": 3, "普及": 4, "竞赛": 5}


# ---------------------------------------------------------------------------
# 业务异常（供路由层映射 HTTP 状态码）
# ---------------------------------------------------------------------------


class PracticeSheetError(Exception):
    """练习纸生成业务错误基类"""


class InvalidSourceError(PracticeSheetError):
    """source 不在合法枚举内"""


class MissingScholarError(PracticeSheetError):
    """缺 scholar_id"""


class KnowledgePointNotMatchedError(PracticeSheetError):
    """knowledge_points[].name 在 ai_summary.knowledge_points[] 中未匹配"""


class NoQuestionsAvailableError(PracticeSheetError):
    """无可用选题（错题本无记录 / 知识点清单为空）"""


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------


def _gen_sheet_id() -> str:
    return f"ps_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _gen_item_id() -> str:
    return f"psi_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _gen_job_id() -> str:
    return f"srj_{int(time.time() * 1000)}_{secrets.token_hex(4)}"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _sheet_signature(
    *,
    scholar_id: str,
    source: str,
    template_type: str,
    node_codes: list | None,
    knowledge_points: list | None,
    include_extended_points: bool,
) -> str:
    """幂等签名：同 scholar 同参数 → 相同签名（重复生成幂等）"""
    sig = {
        "scholar_id": scholar_id,
        "source": source,
        "template_type": template_type,
        "node_codes": sorted(node_codes or []),
        "knowledge_points": [
            {
                "name": kp.get("name"),
                "ability_dimensions": sorted(kp.get("ability_dimensions") or []),
            }
            for kp in (knowledge_points or [])
        ],
        "include_extended_points": bool(include_extended_points),
    }
    raw = json.dumps(sig, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _load_node_by_id(db, node_id: str) -> dict | None:
    """按 node_id 读 curriculum_node，不存在返回 None（不抛错）"""
    if not node_id:
        return None
    res = await db.query(
        CURRICULUM_NODE_COLLECTION, where={"node_id": node_id}, limit=1
    )
    records = res.get("records") or []
    return records[0] if records else None


async def _load_node_by_code(db, node_code: str) -> dict | None:
    """按 code 读 curriculum_node（error_record.node_code 对应 curriculum_node.code）"""
    if not node_code:
        return None
    res = await db.query(
        CURRICULUM_NODE_COLLECTION, where={"code": node_code}, limit=1
    )
    records = res.get("records") or []
    return records[0] if records else None


async def _load_all_summarized_nodes(db) -> list[dict]:
    """读取全部含 ai_summary 的 curriculum_node（数学教材节点规模有限，全量内存匹配）

    返回 [{node_id, code, title, grade, semester, unit_title, lesson_title,
           textbook_id, description_version, ai_summary}]。
    """
    _SELECT = {
        "node_id": 1,
        "code": 1,
        "title": 1,
        "grade": 1,
        "semester": 1,
        "unit_title": 1,
        "lesson_title": 1,
        "textbook_id": 1,
        "description_version": 1,
        "ai_summary": 1,
    }
    nodes: list[dict] = []
    for offset in range(0, 2000, 500):  # 教材节点有限，分批拉取防截断
        res = await db.query(
            CURRICULUM_NODE_COLLECTION, select=_SELECT, offset=offset, limit=500
        )
        batch = res.get("records") or []
        nodes.extend(batch)
        if len(batch) < 500:
            break
    return [n for n in nodes if isinstance(n.get("ai_summary"), dict)]


async def listKnowledgeSummaries(db) -> list[dict]:
    """全部已总结节点摘要（F3.3 小程序选题：勾选知识点用）

    复用 _load_all_summarized_nodes（与生成选题同一口径），
    仅透出节点信息与知识点/拓展点清单（不含完整描述文本）。
    """
    nodes = await _load_all_summarized_nodes(db)
    return [
        {
            "node_id": n.get("node_id"),
            "code": n.get("code"),
            "title": n.get("title"),
            "grade": n.get("grade"),
            "semester": n.get("semester"),
            "unit_title": n.get("unit_title"),
            "lesson_title": n.get("lesson_title"),
            "textbook_id": n.get("textbook_id"),
            "description_version": n.get("description_version"),
            "knowledge_points": (n.get("ai_summary") or {}).get("knowledge_points") or [],
            "extended_points": (n.get("ai_summary") or {}).get("extended_points") or [],
        }
        for n in nodes
    ]


# ---------------------------------------------------------------------------
# 选题源：ai_knowledge（契约 §3.10：按 name 精确匹配）
# ---------------------------------------------------------------------------


async def _resolve_knowledge_points(db, knowledge_points: list) -> list[dict]:
    """按 knowledge_points[].name 精确匹配 ai_summary.knowledge_points[]

    返回 [{request, node, kp}]；任一 name 未匹配 → KnowledgePointNotMatchedError。
    """
    nodes = await _load_all_summarized_nodes(db)
    resolved: list[dict] = []
    for req in knowledge_points or []:
        name = (req.get("name") or "").strip()
        if not name:
            raise KnowledgePointNotMatchedError("knowledge_points 每项必须含 name")
        hit_node, hit_kp = None, None
        for node in nodes:
            ai = node.get("ai_summary") or {}
            if ai.get("status") != "success":
                continue
            for kp in ai.get("knowledge_points") or []:
                if (kp.get("name") or "").strip() == name:
                    hit_node, hit_kp = node, kp
                    break
            if hit_kp:
                break
        if not hit_kp:
            raise KnowledgePointNotMatchedError(
                f"knowledge_points.name 未匹配: {name!r}"
            )
        resolved.append({"request": req, "node": hit_node, "kp": hit_kp})
    return resolved


def _find_extended_point(node: dict | None, kp_name: str) -> dict | None:
    """从节点 ai_summary.extended_points[] 取与知识点关联的奥数扩展点

    优先 related_knowledge_name 精确匹配，否则取该节点第一个扩展点；无则 None。
    """
    if not node:
        return None
    eps = (node.get("ai_summary") or {}).get("extended_points") or []
    if not eps:
        return None
    for ep in eps:
        if (ep.get("related_knowledge_name") or "") == kp_name:
            return ep
    return eps[0]


# ---------------------------------------------------------------------------
# 选题源：wrong_book（ADR-0010：按 error_record 聚合错题）
# ---------------------------------------------------------------------------


async def _select_wrong_book_kps(db, scholar_id: str) -> list[dict]:
    """聚合错题记录（error_record，F4 写入）为练习纸选题源

    返回 [{node_code, node_title, primary_error, occurrence, node}]，按 occurrence 降序，
    最多 MAX_SHEET_NODES 个知识点；无记录返回 []。
    """
    res = await db.query(
        ERROR_RECORD_COLLECTION,
        where={"scholar_id": scholar_id},
        limit=MAX_ERROR_RECORDS_SCAN,
    )
    records = res.get("records") or []
    if not records:
        return []

    by_node: dict[str, list] = defaultdict(list)
    for r in records:
        node_code = r.get("node_code")
        if node_code:
            by_node[node_code].append(r)

    kps: list[dict] = []
    for node_code, rs in by_node.items():
        err_counter = Counter(
            (r.get("primary_error") or "concept") for r in rs
        )
        primary_error = err_counter.most_common(1)[0][0]
        occurrence = max((r.get("occurrence") or 1) for r in rs)
        node = await _load_node_by_code(db, node_code)
        kps.append(
            {
                "node_code": node_code,
                "node_title": (node or {}).get("title") or node_code,
                "primary_error": primary_error,
                "occurrence": occurrence,
                "node": node,
            }
        )
    kps.sort(key=lambda x: -x["occurrence"])
    return kps[:MAX_SHEET_NODES]


# ---------------------------------------------------------------------------
# 出题（LLM 按知识点生成变式题；奥数题按 difficulty_band 标注难度）
# ---------------------------------------------------------------------------

_QUESTION_SYSTEM_PROMPT = (
    "你是小学数学命题助手。根据给定知识点信息，为小学生生成适合 A4 打印的练习题。"
    "只输出合法 JSON，不要输出任何其他文字、markdown 或解释。"
)

_QUESTION_USER_TEMPLATE = """请基于以下数学知识点生成 {count} 道{extended_prefix}练习题。

知识点：
- 名称：{name}
- 总结：{summary}
- 能力维度：{ability_dimensions}
- 教材：{textbook_id}（{title}）
- 年级：{grade}（{semester}）
- 单元/课时：{unit_title} / {lesson_title}
{difficulty_note}
要求：
1. 题目为短答题（口算或简答），数值独立可算，情境适合小学生。
2. 避免与教材例题雷同，多道题之间数值、情境应有变化。
3. 每道题输出 {{"question": "题干", "answer": "参考答案", "difficulty": 1~5 整数, "hint_card": "家长提示语（≤30字）"}}。

输出 JSON 结构：
{{
  "items": [
    {{"question": "...", "answer": "...", "difficulty": 3, "hint_card": "..."}}
  ]
}}"""


def _format_ability_dimensions(dims: list) -> str:
    labels = [_ABILITY_DIM_LABELS.get(d, d) for d in (dims or [])]
    return "、".join(labels) if labels else "—"


def _build_question_prompt(
    *,
    kp: dict,
    node: dict | None,
    count: int,
    extended: bool = False,
    extended_band: str = "",
) -> str:
    node = node or {}
    if extended:
        band_label = extended_band or "入门"
        difficulty_note = f"难度：奥数拔高题（{band_label}档，建议难度 4~5）。"
        extended_prefix = "奥数拔高"
    else:
        difficulty_note = "难度：对应教材基础巩固（建议难度 1~3 为主）。"
        extended_prefix = ""
    ability = _format_ability_dimensions(kp.get("ability_dimensions"))
    return _QUESTION_USER_TEMPLATE.format(
        count=count,
        extended_prefix=extended_prefix,
        name=kp.get("name") or "",
        summary=kp.get("summary") or "",
        ability_dimensions=ability,
        textbook_id=node.get("textbook_id") or "",
        title=node.get("title") or "",
        grade=node.get("grade") or "",
        semester=node.get("semester") or "",
        unit_title=node.get("unit_title") or "",
        lesson_title=node.get("lesson_title") or "",
        difficulty_note=difficulty_note,
    )


def _validate_question_items(result: dict) -> list[dict]:
    """校验 LLM 出题结构（契约 §4.12.4 practice_sheet_item 口径），非法抛 LLMResponseError"""
    if not isinstance(result, dict):
        raise LLMResponseError("出题结果必须为 JSON 对象")
    items = result.get("items")
    if not isinstance(items, list) or not items:
        raise LLMResponseError("出题结果 items 必须为非空数组")
    cleaned: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            raise LLMResponseError("items 每项必须为对象")
        if not it.get("question") or not it.get("answer"):
            raise LLMResponseError("items 每项必须含 question / answer")
        try:
            difficulty = int(it.get("difficulty"))
        except (TypeError, ValueError):
            difficulty = 3
        if not 1 <= difficulty <= 5:
            difficulty = 3
        cleaned.append(
            {
                "question": str(it["question"]),
                "answer": str(it["answer"]),
                "difficulty": difficulty,
                "hint_card": str(it.get("hint_card") or ""),
            }
        )
    return cleaned


def _call_chat_sync(client, model: str, prompt: str) -> str:
    """同步 chat 调用（在线程池中执行），复用 F1 总结模型

    超时根因修复（与 knowledge_summary/Judge 同口径）：
    - LLM_DISABLE_THINKING 时显式禁用 thinking —— 推理模型不禁用单次可 >60s，
      出题请求叠加后必然突破小程序 callContainer 15s 上限（探针 P1）；
    - 每次请求带 SHEET_LLM_CALL_TIMEOUT_SEC 上限，替代客户端默认 60s（探针 P3）。
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
        "timeout": SHEET_LLM_CALL_TIMEOUT_SEC,
    }
    if LLM_DISABLE_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


async def _generate_questions(
    *,
    kp: dict,
    node: dict | None,
    count: int,
    extended: bool = False,
    extended_band: str = "",
) -> list[dict]:
    """调用 LLM_SUMMARY_MODEL 基于知识点生成 count 道练习题（JSON 解析失败重试 1 次）

    返回 [{question, answer, difficulty, hint_card}]；LLM 返回数量不足 count 时按实际数量返回
    （调用方据此记"知识点题量不足"），0 道视为出题失败（LLMResponseError）。
    """
    if not LLM_SUMMARY_MODEL:
        raise LLMNotConfiguredError("LLM_SUMMARY_MODEL 未配置，无法出题")
    from services.dialogue import _parse_json_response

    prompt = _build_question_prompt(
        kp=kp,
        node=node,
        count=count,
        extended=extended,
        extended_band=extended_band,
    )
    client = _get_llm_client()
    last_err: Exception | None = None
    for attempt in (1, 2):  # 首次 + 重试 1 次
        try:
            response = await asyncio.to_thread(
                _call_chat_sync, client, LLM_SUMMARY_MODEL, prompt
            )
            result = _parse_json_response(response)
            items = _validate_question_items(result)
            if items:
                return items[:count]
        except Exception as e:  # noqa: BLE001 — 统一重试后仍失败抛 LLMResponseError
            last_err = e
            logger.warning(
                f"出题失败（第 {attempt} 次）kp={kp.get('name')!r} "
                f"extended={extended}: {type(e).__name__}: {e}"
            )
            if type(e).__name__ in ("APITimeoutError", "TimeoutError"):
                # 硬超时（SHEET_LLM_CALL_TIMEOUT_SEC 触发）：不再整次重试，
                # 否则 2×timeout 会突破 callContainer 15s 容器上限（探针 P3）
                break
    raise LLMResponseError(f"AI 出题失败: {last_err}")


def _band_to_difficulty(band: str) -> int:
    """奥数题 difficulty_band → 题目难度（契约：奥数题按 band 标注难度）"""
    return _EXTENDED_BAND_DIFFICULTY.get(band, 3)


# ---------------------------------------------------------------------------
# 幂等（契约：重复生成幂等）
# ---------------------------------------------------------------------------


async def _find_recent_sheet(
    db, scholar_id: str, signature: str, within_ms: int = SHEET_REPEAT_WINDOW_MS
) -> dict | None:
    """同 scholar 幂等窗口内签名一致的最近练习纸（status=generated）"""
    res = await db.query(
        PRACTICE_SHEET_COLLECTION,
        where={"scholar_id": scholar_id},
        order=[{"field": "generated_at", "direction": "desc"}],
        limit=5,
    )
    now = _now_ms()
    for r in res.get("records") or []:
        if r.get("idempotency_signature") != signature:
            continue
        if r.get("status") != SHEET_STATUS_GENERATED:
            continue
        if now - int(r.get("generated_at") or 0) <= within_ms:
            return r
    return None


# ---------------------------------------------------------------------------
# 落库 + 审计
# ---------------------------------------------------------------------------


async def _persist_sheet(
    db,
    *,
    sheet_id: str,
    scholar_id: str,
    template_ref: dict,
    nodes: list,
    primary_errors: list,
    difficulty_bands: list,
    items: list,
    source: str,
    knowledge_points: list,
    include_extended_points: bool,
    warnings: list,
    signature: str,
) -> dict:
    """落库 practice_sheet（契约 §4.12.4 + §4.12.10(a) source 扩展）+ sheet_render_job（queued）"""
    now = _now_ms()
    sheet = {
        "_id": sheet_id,
        "sheet_id": sheet_id,
        "scholar_id": scholar_id,
        "template_ref": template_ref,
        "nodes": nodes,
        "primary_errors": primary_errors,
        "difficulty_bands": difficulty_bands,
        "items": items,
        "status": SHEET_STATUS_GENERATED,
        # 家长核对二维码：渲染阶段生成（ADR-0010），MVP 落库占位
        "qrcode_ref": {"qr_url": "", "signature": "", "expires_at": 0},
        # 渲染产物：渲染任务完成后填充（ADR-0010）
        "file_refs": {"pdf": "", "png": "", "preview": ""},
        # §4.12.10(a) source 扩展
        "source": source,
        "knowledge_points": knowledge_points or [],
        "include_extended_points": bool(include_extended_points),
        # 非契约内部字段：幂等签名 / 提示
        "idempotency_signature": signature,
        "warnings": warnings,
        "generated_at": now,
        "updated_at": now,
    }
    await db.insert(PRACTICE_SHEET_COLLECTION, sheet)

    job_id = _gen_job_id()
    await db.insert(
        SHEET_RENDER_JOB_COLLECTION,
        {
            "_id": job_id,
            "job_id": job_id,
            "sheet_id": sheet_id,
            "status": RENDER_JOB_QUEUED,
            "created_at": now,
        },
    )
    return sheet


def _schedule_render(db, sheet_id: str) -> None:
    """F3.2 接入渲染：落库后后台异步执行 A4 渲染（renderSheetJob）

    - 不阻塞生成接口；渲染失败/未装 Chromium 由 job 状态机兜底（failed/degraded），
      前端轮询 GET /math/practice-sheet/{id} 可见 file_refs。
    - 进程内 create_task 仅尽力而为：部署侧另有 scripts/render_sheet_jobs.py 定时兜底。
    """
    try:
        from services.math.a4_renderer import renderSheetJob

        asyncio.get_running_loop().create_task(renderSheetJob(db, sheet_id))
        logger.info(f"[F3.2] 已提交渲染任务 sheet_id={sheet_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[F3.2] 渲染任务提交失败（将由脚本兜底）: {e}")


def _to_public_sheet(sheet: dict) -> dict:
    """落库文档 → 接口出参（api-contract §3.10：不含 answer / hint_card）"""
    return {
        "sheet_id": sheet.get("sheet_id"),
        "status": sheet.get("status"),
        "source": sheet.get("source"),
        "template_ref": sheet.get("template_ref"),
        "nodes": sheet.get("nodes") or [],
        "primary_errors": sheet.get("primary_errors") or [],
        "difficulty_bands": sheet.get("difficulty_bands") or [],
        "items": [
            {k: it.get(k) for k in ("item_id", "question", "node_code", "target_error", "variant_level", "difficulty", "source_kp") if it.get(k) not in (None, "")}
            for it in (sheet.get("items") or [])
        ],
        "qrcode_ref": sheet.get("qrcode_ref"),
        "file_refs": sheet.get("file_refs"),
    }


async def getPracticeSheetById(db, *, sheet_id: str) -> dict | None:
    """按 sheet_id 查询练习纸公开详情（F3.3 小程序端轮询渲染状态）

    返回 _to_public_sheet 结构，并附加 render_status（sheet_render_job
    最新状态：queued/rendering/success/failed/degraded），供前端区分
    "渲染中 / 已成功 / 渲染失败"。练习纸不存在返回 None。
    二维码扫码校验（签名/过期/账号绑定）由 S3 家长核对页按契约实现，
    本函数仅提供公开详情查询。
    """
    res = await db.query(
        PRACTICE_SHEET_COLLECTION,
        where={"sheet_id": sheet_id},
        limit=1,
    )
    sheets = res.get("records") or []
    if not sheets:
        return None
    sheet = sheets[0]

    data = _to_public_sheet(sheet)
    data["render_status"] = ""
    jobs = await db.query(
        SHEET_RENDER_JOB_COLLECTION,
        where={"sheet_id": sheet_id},
        order=[{"field": "created_at", "direction": "desc"}],
        limit=1,
    )
    job = (jobs.get("records") or [None])[0]
    if job:
        data["render_status"] = job.get("status") or ""
    return data


# ---------------------------------------------------------------------------
# F3 主入口：generatePracticeSheet（service-contract §7.3 / api-contract §3.10）
# ---------------------------------------------------------------------------


async def generatePracticeSheet(
    db,
    *,
    scholar_id: str,
    template_type: str = SHEET_TEMPLATE_STANDARD,
    node_codes: list | None = None,
    primary_errors: list | None = None,
    difficulty_bands: list | None = None,
    source: str = SHEET_SOURCE_WRONG_BOOK,
    knowledge_points: list | None = None,
    include_extended_points: bool = False,
    wrong_book_ratio: float = DEFAULT_WRONG_BOOK_RATIO,
    actor: str = "",
) -> dict:
    """生成 A4 练习纸（三种选题源）

    前置校验：
    - 缺 scholar_id → MissingScholarError（路由层映射 400）
    - source 非法 → InvalidSourceError（400）
    - ai_knowledge / mixed 缺 knowledge_points 或 name 未匹配 → KnowledgePointNotMatchedError（400）
    - 无可用选题 → NoQuestionsAvailableError（400）
    - LLM_SUMMARY_MODEL 未配置 → LLMNotConfiguredError（400）；出题失败 → LLMResponseError（500）

    幂等：同 scholar 10 分钟内相同参数重复生成 → 返回最近已有练习纸（不重复调 LLM / 不重复写库）。
    防背题：items 洗牌；答案仅落库，出参不含；奥数题列在主练习后。
    """
    scholar_id = (scholar_id or "").strip()
    if not scholar_id:
        raise MissingScholarError("缺少 scholar_id")
    if source not in SHEET_SOURCES:
        raise InvalidSourceError(f"source 必须为 {list(SHEET_SOURCES)} 之一")

    ai_used = source in (SHEET_SOURCE_AI_KNOWLEDGE, SHEET_SOURCE_MIXED)
    if ai_used and not knowledge_points:
        raise KnowledgePointNotMatchedError("ai_knowledge / mixed 需要提供 knowledge_points")

    signature = _sheet_signature(
        scholar_id=scholar_id,
        source=source,
        template_type=template_type,
        node_codes=node_codes,
        knowledge_points=knowledge_points,
        include_extended_points=include_extended_points,
    )
    recent = await _find_recent_sheet(db, scholar_id, signature)
    if recent:
        logger.info(f"练习纸幂等命中 sheet_id={recent.get('sheet_id')}")
        _schedule_render(db, recent.get("sheet_id") or "")
        return _to_public_sheet(recent)

    # ------------------------------------------------------------------ 选题
    warnings: list[str] = []
    wrong_kps: list[dict] = []
    ai_kps: list[dict] = []
    if source == SHEET_SOURCE_WRONG_BOOK:
        wrong_kps = await _select_wrong_book_kps(db, scholar_id)
    elif source == SHEET_SOURCE_AI_KNOWLEDGE:
        ai_kps = await _resolve_knowledge_points(db, knowledge_points)
    else:  # mixed
        wrong_kps = await _select_wrong_book_kps(db, scholar_id)
        ai_kps = await _resolve_knowledge_points(db, knowledge_points)
        # 错题知识点占比 ≈ wrong_book_ratio，总知识点 ≤3
        target_wrong = max(
            0, round((len(wrong_kps) + len(ai_kps)) * wrong_book_ratio)
        )
        wrong_kps = wrong_kps[: min(target_wrong, len(wrong_kps))]
        remaining = MAX_SHEET_NODES - len(wrong_kps)
        ai_kps = ai_kps[:remaining]
        if len(wrong_kps) + len(ai_kps) > MAX_SHEET_NODES:
            ai_kps = ai_kps[: MAX_SHEET_NODES - len(wrong_kps)]

    if not wrong_kps and not ai_kps:
        raise NoQuestionsAvailableError(
            "无可用选题（错题本无记录且知识点清单为空）"
        )

    # ------------------------------------------------------------------ 出题
    nodes: list[dict] = []
    primary_errors: list[dict] = []
    bands: list[dict] = []
    items: list[dict] = []

    def _make_item(
        q: dict,
        *,
        node_code: str,
        target_error: str,
        source_kp: str = "",
        difficulty: int | None = None,
        extended: bool = False,
    ) -> dict:
        """题目记录 → practice_sheet_item（奥数题打内部标记，落库前剥离）"""
        item = {
            "item_id": _gen_item_id(),
            "sheet_id": "",  # 组装后回填
            "question": q["question"],
            "answer": q["answer"],
            "hint_card": q["hint_card"],
            "node_code": node_code,
            "target_error": target_error,
            "variant_level": VARIANT_LEVEL_L1,
            "difficulty": q["difficulty"] if difficulty is None else difficulty,
            "source_kp": source_kp,
            "sim_check": {"status": "skipped"},  # 防背题相似度校验：渲染任务阶段启用
        }
        if extended:
            item[_EXTENDED_ITEM_FLAG] = True
        return item

    # 知识点间出题互不依赖 → 有界并发调 LLM（超时根因修复 P2）：
    # 串行最多 6 次（3 基础 + 3 奥数）在 callContainer 15s 上限下极易超时；
    # 并行 + MAX_LLM_CONCURRENCY 后墙钟 ≈ ceil(N/并发) 波 × 单次时延。
    _llm_sem = asyncio.Semaphore(MAX_LLM_CONCURRENCY)

    async def _gen_basic(
        *,
        node_code: str,
        node_title: str,
        target_error: str,
        kp: dict,
        node: dict | None,
        count: int,
        source_kp: str = "",
    ) -> list[dict]:
        async with _llm_sem:
            questions = await _generate_questions(kp=kp, node=node, count=count)
        if len(questions) < count:
            warnings.append(f"知识点题量不足: {node_title or node_code}")
        return [
            _make_item(
                q,
                node_code=node_code,
                target_error=target_error,
                source_kp=source_kp,
            )
            for q in questions
        ]

    async def _gen_extended(
        *,
        node_code: str,
        node_title: str,
        kp: dict | None,
        node: dict | None,
        band: str,
    ) -> list[dict]:
        """奥数拔高题（每知识点 1 道；仅 include_extended_points 且命中扩展点才出）"""
        if not kp:
            return []
        async with _llm_sem:
            questions = await _generate_questions(
                kp=kp, node=node, count=1, extended=True, extended_band=band
            )
        return [
            _make_item(
                q,
                node_code=node_code,
                target_error="",
                difficulty=_band_to_difficulty(band),
                extended=True,
            )
            for q in questions[:1]
        ]

    async def _exec_wrong_group() -> tuple[list, list, list, list]:
        """wrong_book 源：知识点 = 聚合出的错题知识点（每点 2 题），组内并行"""
        plan: list[dict] = []
        for wk in wrong_kps:
            node_code = wk["node_code"]
            target_error = wk["primary_error"]
            band = "挑战" if (wk.get("occurrence") or 1) >= 3 else "巩固"
            # 出题上下文优先取节点 F1 ai_summary 的知识点总结，增强题目质量
            node = wk.get("node") or {}
            ai = node.get("ai_summary") or {}
            first_kp = (ai.get("knowledge_points") or [{}])[0]
            plan.append(
                {
                    "node_code": node_code,
                    "node_title": wk.get("node_title") or node_code,
                    "target_error": target_error,
                    "band": band,
                    "kp": {
                        "name": first_kp.get("name") or wk.get("node_title") or node_code,
                        "summary": first_kp.get("summary") or node.get("title") or "",
                        "ability_dimensions": first_kp.get("ability_dimensions") or [],
                    },
                    "node": node,
                }
            )
        if not plan:
            return [], [], [], []
        results = await asyncio.gather(
            *(
                _gen_basic(
                    node_code=p["node_code"],
                    node_title=p["node_title"],
                    target_error=p["target_error"],
                    kp=p["kp"],
                    node=p["node"],
                    count=DEFAULT_QUESTION_COUNT_PER_KNOWLEDGE,
                )
                for p in plan
            )
        )
        g_nodes: list[dict] = []
        g_errors: list[dict] = []
        g_bands: list[dict] = []
        g_items: list[dict] = []
        for p, generated in zip(plan, results):
            g_nodes.append({"node_code": p["node_code"], "node_title": p["node_title"]})
            g_errors.append({"node_code": p["node_code"], "type": p["target_error"]})
            g_bands.append({"node_code": p["node_code"], "band": p["band"]})
            g_items.extend(generated)
        return g_nodes, g_errors, g_bands, g_items

    async def _exec_ai_group() -> tuple[list, list, list]:
        """ai_knowledge / mixed 源：知识点 = 匹配出的 AI 知识点（每点 2 题 + 奥数 1 题）"""
        plan: list[dict] = []
        for entry in ai_kps:
            kp = entry["kp"]
            node = entry["node"]
            node_id = kp.get("source_node_id") or ""
            node_for_code = await _load_node_by_id(db, node_id) if node_id else node
            node_code = (node_for_code or {}).get("code") or kp.get("name") or ""
            node_title = (
                (node_for_code or {}).get("title")
                or (node or {}).get("title")
                or kp.get("name")
                or ""
            )
            ep = None
            if include_extended_points:
                ep = _find_extended_point(node, kp.get("name") or "")
            plan.append(
                {
                    "node_code": node_code,
                    "node_title": node_title,
                    "kp": kp,
                    "node": node,
                    "ep": ep,
                    "ext_band": (ep.get("difficulty_band") or "入门") if ep else "",
                }
            )
        if not plan:
            return [], [], []
        n = len(plan)
        results = await asyncio.gather(
            *(
                _gen_basic(
                    node_code=p["node_code"],
                    node_title=p["node_title"],
                    target_error="",
                    kp=p["kp"],
                    node=p["node"],
                    count=DEFAULT_QUESTION_COUNT_PER_KNOWLEDGE,
                    source_kp=p["kp"].get("name") or "",
                )
                for p in plan
            ),
            *(
                _gen_extended(
                    node_code=p["node_code"],
                    node_title=p["node_title"],
                    kp=p["ep"],
                    node=p["node"],
                    band=p["ext_band"],
                )
                for p in plan
            ),
        )
        g_nodes: list[dict] = []
        g_bands: list[dict] = []
        g_items: list[dict] = []
        for idx, p in enumerate(plan):
            basic_gen = results[idx]
            ext_gen = results[n + idx]
            g_nodes.append({"node_code": p["node_code"], "node_title": p["node_title"]})
            g_bands.append({"node_code": p["node_code"], "band": "巩固"})
            g_items.extend(basic_gen)
            g_items.extend(ext_gen)
        return g_nodes, g_bands, g_items

    # mixed：错题组与 AI 组相互独立 → 组间也并发（共享同一并发上限）
    (w_nodes, w_errors, w_bands, w_items), (a_nodes, a_bands, a_items) = (
        await asyncio.gather(_exec_wrong_group(), _exec_ai_group())
    )
    # 顺序归位（与历史输出一致：wrong 在前、ai 在后），保证 nodes/题序确定性
    nodes = w_nodes + a_nodes
    primary_errors = w_errors
    bands = w_bands + a_bands
    items = w_items + a_items

    if not items:
        raise NoQuestionsAvailableError("无可用选题（出题结果为空）")

    # 防背题（ADR-0010）：主练习题洗牌，奥数扩展题保持列在主练习后
    rng = random.Random(int(time.time() * 1000))
    main_pool = [it for it in items if not it.get(_EXTENDED_ITEM_FLAG)]
    ext_pool = [it for it in items if it.get(_EXTENDED_ITEM_FLAG)]
    rng.shuffle(main_pool)
    items = main_pool + ext_pool

    # 回填 sheet_id 并剥离内部标记（出参 / 落库不含 _extended）
    sheet_id = _gen_sheet_id()
    for it in items:
        it["sheet_id"] = sheet_id
        it.pop(_EXTENDED_ITEM_FLAG, None)

    # ------------------------------------------------------------------ 落库
    template_ref = {
        "template_id": SHEET_TEMPLATE_ID_STANDARD,
        "template_type": template_type or SHEET_TEMPLATE_STANDARD,
        "version": SHEET_TEMPLATE_VERSION,
    }
    sheet = await _persist_sheet(
        db,
        sheet_id=sheet_id,
        scholar_id=scholar_id,
        template_ref=template_ref,
        nodes=nodes,
        primary_errors=primary_errors,
        difficulty_bands=bands,
        items=items,
        source=source,
        knowledge_points=knowledge_points or [],
        include_extended_points=include_extended_points,
        warnings=warnings,
        signature=signature,
    )

    # ------------------------------------------------------------------ 审计
    await write_audit(
        db,
        action=AUDIT_ACTION_GENERATE,
        object_ref=sheet_id,
        actor=actor,
        context={
            "source": source,
            "knowledge_points": [
                {"name": kp.get("name", "")} for kp in (knowledge_points or [])
            ],
            "include_extended_points": bool(include_extended_points),
            "template_type": template_type,
            "nodes": nodes,
            "primary_errors": primary_errors,
            "difficulty_bands": bands,
            "item_count": len(items),
        },
    )
    _schedule_render(db, sheet_id)
    return _to_public_sheet(sheet)

