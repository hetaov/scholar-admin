"""评估证据接口（契约 api-contract §3.9，P0 后置评估 v1）

POST /evaluation/{ref}/evaluate — 对指定 attempt/turn/session 触发或重算评估
GET  /evaluation/{id}          — 查询评估与原始证据（证据不可改、评价可重算）

ref 支持三种引用前缀（多态）：
- learning_attempt:<id>   — 学习尝试（文本评估，证据来自尝试记录）
- conversation_turn:<id>  — 会话轮次（文本评估）
- speech:<id>             — SOE-N 语音评测存档（语音评估，证据来自 speech_evaluation）

原则（§9-1/§9-2）：
- 证据（raw/parsed）只读不写；重算（force=true）只更新评价部分，幂等。
- 低置信（confidence < 0.6）不回写 SkillState —— 本模块只产出 eval_verdict，
  状态回写由会话/训练路由按 LOW_CONFIDENCE_THRESHOLD 门控。
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from services.database import CloudBaseNoSQLClient
from services.dependencies import get_db
from services.evaluation_engine import (
    EVALUATION_COLLECTION,
    evaluate_speech,
    evaluate_text,
)
from services.speech_eval import SPEECH_EVALUATION_COLLECTION

logger = logging.getLogger("scholar-admin.evaluation")

router = APIRouter(tags=["evaluation"])


class EvalResponse(BaseModel):
    success: bool
    code: str = "OK"
    message: Optional[str] = None
    data: Optional[dict] = None


class EvalTriggerRequest(BaseModel):
    """重算请求体：force=true 时对同引用重算（幂等）；false 时已有评估直接返回。"""

    force: bool = Field(False, description="是否强制重算（默认命中已有评估直接返回）")


def _parse_ref(ref: str) -> Optional[tuple[str, str]]:
    """解析 ref 为 (kind, doc_id)；非法返回 None。"""
    if not ref or ":" not in ref:
        return None
    kind, doc_id = ref.split(":", 1)
    if kind not in ("learning_attempt", "conversation_turn", "speech"):
        return None
    if not doc_id:
        return None
    return kind, doc_id


async def _load_evidence(
    db: CloudBaseNoSQLClient, kind: str, doc_id: str
) -> Optional[dict]:
    """按 ref 加载证据文档：speech → speech_evaluation；其余按主键查询。"""
    collection = (
        SPEECH_EVALUATION_COLLECTION
        if kind == "speech"
        else "learning_attempt"
        if kind == "learning_attempt"
        else "conversation_turn"
    )
    result = await db.query(collection, where={"_id": doc_id}, limit=1)
    records = result.get("records") or []
    return records[0] if records else None


def _build_verdict(kind: str, evidence: dict, body: EvalTriggerRequest) -> Optional[dict]:
    """由证据产出 eval_verdict（文本走 L1+L2，语音走 L1 规则）。"""
    if kind == "speech":
        parsed = evidence.get("parsed") or {}
        return evaluate_speech(parsed)

    original = (evidence.get("original_text") or evidence.get("content") or "").strip()
    response = (
        (evidence.get("user_input") or evidence.get("utterance") or "").strip()
        if kind == "learning_attempt"
        else (evidence.get("utterance") or "").strip()
    )
    return evaluate_text(original, response)


@router.post("/evaluation/{ref}/evaluate", response_model=EvalResponse)
async def evaluation_trigger(
    ref: str,
    body: EvalTriggerRequest,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> EvalResponse:
    parsed = _parse_ref(ref)
    if parsed is None:
        return EvalResponse(
            success=False, code="INVALID_INPUT", message="ref 格式非法（期望 kind:id）"
        )
    kind, doc_id = parsed

    evidence = await _load_evidence(db, kind, doc_id)
    if evidence is None:
        return EvalResponse(
            success=False,
            code="NOT_FOUND",
            message=f"{kind} 记录不存在: {doc_id}",
        )

    # 命中已有评估（非强制重算）→ 直接返回（幂等）
    existing = await db.query(
        EVALUATION_COLLECTION,
        where={"attempt_ref": ref},
        order=[{"field": "created_at", "direction": "desc"}],
        limit=1,
    )
    if not body.force and (existing.get("records") or []):
        doc = existing["records"][0]
        return EvalResponse(success=True, data=doc)

    verdict = _build_verdict(kind, evidence, body)
    if verdict is None:
        return EvalResponse(
            success=False, code="EVAL_FAILED", message="评估失败（证据不可用）"
        )

    evaluation_doc = {
        "attempt_ref": ref,
        "evidence_ref": f"{kind}:{doc_id}",
        "level": "turn" if kind == "conversation_turn" else "attempt",
        "type": "speech" if kind == "speech" else "text",
        "score": verdict["score"],
        "confidence": verdict["confidence"],
        "verdict": {
            "meaningful": verdict["meaningful"],
            "faithfulness": verdict["faithfulness"],
            "anomaly": verdict["anomaly"],
        },
        "raw": {
            "rubric_verdict": verdict,
            "evidence_snapshot": {
                "original": (
                    evidence.get("original_text") or evidence.get("content") or ""
                ),
                "response": (
                    evidence.get("user_input")
                    or evidence.get("utterance")
                    or ""
                ),
            },
        },
        "created_at": int(time.time() * 1000),
        "updated_at": int(time.time() * 1000),
    }

    if body.force and (existing.get("records") or []):
        # 重算：更新同引用的最新评估（幂等，保留最新一次）
        target = existing["records"][0]
        await db.update(
            EVALUATION_COLLECTION,
            where={"_id": target["_id"]},
            data={"$set": {**evaluation_doc, "_id": target["_id"]}},
        )
        evaluation_doc["_id"] = target["_id"]
    else:
        inserted = await db.insert(EVALUATION_COLLECTION, evaluation_doc)
        evaluation_doc["_id"] = (inserted.get("ids") or [None])[0]

    return EvalResponse(success=True, data=evaluation_doc)


@router.get("/evaluation/{id}", response_model=EvalResponse)
async def evaluation_get(
    id: str,
    db: CloudBaseNoSQLClient = Depends(get_db),
) -> EvalResponse:
    if not id.strip():
        return EvalResponse(
            success=False, code="INVALID_INPUT", message="evaluation id 不能为空"
        )
    result = await db.query(EVALUATION_COLLECTION, where={"_id": id}, limit=1)
    records = result.get("records") or []
    if not records:
        return EvalResponse(success=False, code="NOT_FOUND", message="评估记录不存在")
    return EvalResponse(success=True, data=records[0])
