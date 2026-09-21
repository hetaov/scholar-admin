"""复习推荐 LLM Provider（v4 R4 第二段；ADR-0016 修订 2026-09-21）

职责边界（设计文档 §3.5 A1~A6 / D7）：
- **候选集由规则层确定性给出**（已学且未掌握的组，按到期句数降序）——本模块**不选组**；
- LLM 只做两件事：① 排序 ② 每组一句话 `reason_text`（≤120 字）；
- 模型返回必须**严格匹配候选集**：`group_id` 越界 / 重复 / 缺项 → 解析失败（任务置 failed，
  调用方回退规则排序）——即「不得新增/替换/删除候选」。

与 `services/providers/translation_eval.py` 同构（同一火山方舟 OpenAI 兼容调用姿势），
区别：模型取 `REVIEW_RECOMMEND_MODEL`（生成类）、超时取 `REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS`。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from config import (
    REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS,
    REVIEW_RECOMMEND_MODEL,
    VOLCANO_API_KEY,
    VOLCANO_BASE_URL,
)
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("scholar-admin.review_recommend")

# 失败阶段 / 错误码（对齐 data-model-contract §4.16 error 对象口径）
STAGE_LLM = "llm"
STAGE_PARSE = "parse"
ERR_LLM_TIMEOUT = "LLM_TIMEOUT"
ERR_LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
ERR_INVALID_OUTPUT = "INVALID_OUTPUT"

REASON_MAX_CHARS = 120

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


class ReviewRecommendError(Exception):
    """业务失败（路由/任务层转为 error 对象）。"""

    def __init__(self, error_code: str, failure_stage: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.failure_stage = failure_stage
        self.detail = detail

    def to_dict(self, llm_timeout_seconds: int | None = None, raw=None) -> dict:
        return {
            "error_code": self.error_code,
            "error_detail": self.detail,
            "failure_stage": self.failure_stage,
            "llm_timeout_seconds": llm_timeout_seconds,
            "raw": raw,
        }


def build_review_recommend_messages(candidates: list[dict]) -> list[dict]:
    """构造排序 + 理由的提示词（候选集由调用方给出，模型不得增删）。

    Args:
        candidates: [{group_id, group_title, group_status, group_mastery,
                      last_studied_at, due_sentence_count, sentence_count, sentence_samples?}]
    """
    lines = []
    for c in candidates:
        lines.append(
            "- group_id={gid}｜组名={title}｜组掌握状态={status}（数字越小越弱，0-4）"
            "｜掌握度={mastery}｜最近学习={last}｜到期句数={due}｜组内句数={total}".format(
                gid=c.get("group_id") or "",
                title=c.get("group_title") or "",
                status=c.get("group_status", 0),
                mastery=round(float(c.get("group_mastery") or 0.0) * 100),
                last=c.get("last_studied_at") or "",
                due=c.get("due_sentence_count", 0),
                total=c.get("sentence_count", 0),
            )
        )
    system = (
        "你是英语学习复习调度助手。给定同一本教材内**已筛选好的候选组**（规则层已确定集合，"
        "你**不得新增、替换或删除任何 group_id**），请按「该先复习哪一组」排序，并为每组写一句"
        f"不超过 {REASON_MAX_CHARS} 字的中文理由。\n"
        "排序依据：学习时间越久越靠前；到期句数越多越靠前；组掌握状态越弱（数字越小）越靠前；"
        "结合记忆规律（间隔复习）解释理由。\n"
        "严格输出 JSON 对象："
        '{"recommendations": [{"group_id": "<原样照抄候选中的 id>", "priority": 1, '
        '"reason_text": "<中文一句话理由>"}]}\n'
        "要求：必须覆盖**全部**候选、每个 group_id 恰好一次、priority 从 1 连续递增。"
    )
    user = "候选组（共 %d 组）：\n%s" % (len(candidates), "\n".join(lines))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _call_review_recommend_llm(
    messages: list[dict], temperature: float = 0.2
) -> str | None:
    """同步调用火山方舟（OpenAI 兼容）；凭据缺失 / 调用失败返回 None。

    超时由外层 `call_review_recommend_llm` 的 `asyncio.wait_for` 兜底（强取消）。
    """
    if not (VOLCANO_API_KEY and REVIEW_RECOMMEND_MODEL):
        logger.warning("[review_recommend] 未配置火山方舟凭据/模型，无法生成推荐")
        return None
    import requests

    resp = requests.post(
        f"{VOLCANO_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {VOLCANO_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": REVIEW_RECOMMEND_MODEL,
            "messages": messages,
            "temperature": temperature,
            # 强制 JSON 输出（OpenAI 兼容字段，方舟支持）——注意须为对象（数组不被接受）
            "response_format": {"type": "json_object"},
        },
        timeout=REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS,
    )
    if resp.status_code != 200:
        logger.error(
            "[review_recommend] 火山方舟返回 %s: %s", resp.status_code, resp.text[:200]
        )
        return None
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def call_review_recommend_llm(
    messages: list[dict],
    timeout_seconds: int | None = None,
) -> str | None:
    """LLM 调用异步包装（同步请求丢线程池 + `asyncio.wait_for` 超时强取消）。

    Raises:
        ReviewRecommendError: 超时 → error_code=LLM_TIMEOUT（stage=llm）
    """
    timeout = timeout_seconds or REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS
    try:
        return await asyncio.wait_for(
            run_in_threadpool(_call_review_recommend_llm, messages),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error("[review_recommend] LLM 调用超过 %ss 未返回 → LLM_TIMEOUT", timeout)
        raise ReviewRecommendError(
            ERR_LLM_TIMEOUT,
            STAGE_LLM,
            f"LLM 调用超过 {timeout}s 未返回（REVIEW_RECOMMEND_LLM_TIMEOUT_SECONDS={timeout}）",
        )


def parse_review_recommend_output(
    content: str | None,
    candidate_ids: list[str],
) -> list[dict] | None:
    """解析并**强校验**模型输出（A2/D7：不得新增/替换/删除候选）。

    校验规则（任一不满足 → None，任务置 failed，调用方回退规则排序）：
    - 输出为 JSON（容忍代码块包裹），形态 `{"recommendations": [...]}` 或裸数组；
    - 每项 `group_id` 必须 ∈ 候选集且不重复；
    - **必须覆盖全部候选**（缺项视为「删除候选」→ 非法）。

    容错：`reason_text` 缺失/非法 → 空串；超长 → 截断至 REASON_MAX_CHARS；`priority` 非法 → 按数组顺序补。

    Returns:
        [{group_id, priority, reason_text}]（顺序 = 模型给出的排序）| None
    """
    if not content:
        return None
    match = _JSON_BLOCK_RE.search(content)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict):
        items = parsed.get("recommendations")
    else:
        items = parsed
    if not isinstance(items, list) or not items:
        return None

    allowed = set(candidate_ids)
    seen: set[str] = set()
    out: list[dict] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        gid = str(item.get("group_id") or "")
        if gid not in allowed or gid in seen:
            logger.warning(
                "[review_recommend] 模型返回越界/重复 group_id=%r（候选 %d 组）", gid, len(allowed)
            )
            return None
        seen.add(gid)
        reason = str(item.get("reason_text") or "").strip()[:REASON_MAX_CHARS]
        priority = item.get("priority")
        if not isinstance(priority, int) or priority <= 0:
            priority = idx + 1
        out.append({"group_id": gid, "priority": priority, "reason_text": reason})

    if seen != allowed:
        logger.warning(
            "[review_recommend] 模型返回缺项（%d/%d）→ 视为删除候选，判非法",
            len(seen), len(allowed),
        )
        return None
    return out
