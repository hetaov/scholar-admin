"""对话匹配接口"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from services.dependencies import get_db
from services.dialogue import match_dialogue

logger = logging.getLogger("scholar-admin.routes.dialogue")
router = APIRouter(tags=["对话匹配"])


@router.post("/match/dialogue")
async def match_dialogue_endpoint(data: dict):
    """对话匹配 — 根据输入英文句，从已学语句中匹配或生成问答对

    请求体：
    {
      \"scholarId\": \"6d758f346a6daee000859c332ed11089\",
      \"sentence\": \"I go to school by bus every day.\"
    }

    返回：
    {
      \"success\": true,
      \"data\": {
        \"type\": \"qa\",
        \"statement\": \"...\",
        \"question\": \"...\",
        \"source\": \"matched|generated\"
      },
      \"is_question\": false
    }
    """
    scholar_id = data.get("scholarId", "")
    input_sentence = data.get("sentence", "")

    if not scholar_id:
        raise HTTPException(status_code=400, detail="缺少参数 scholarId")
    if not input_sentence:
        raise HTTPException(status_code=400, detail="缺少参数 sentence")

    try:
        db = get_db()

        # 1. 从 learning_mastery_tracking 获取该学者的所有 sentence_id
        tracking_result = await db.query(
            collection="learning_mastery_tracking",
            where={"scholar_id": scholar_id},
        )
        records = tracking_result.get("records", [])
        if not records:
            return {"success": False, "error": "该学者暂无已学语句", "data": None}

        sentence_ids = list(
            {r.get("sentence_id") for r in records if r.get("sentence_id")}
        )
        if not sentence_ids:
            return {"success": False, "error": "未找到已学语句 ID", "data": None}

        # 2. 从 sentence 集合获取语句文本（分批查询，$in 上限 100 条）
        learned_sentences: list[dict] = []
        for i in range(0, len(sentence_ids), 100):
            batch = sentence_ids[i : i + 100]
            sentence_result = await db.query(
                collection="sentence",
                where={"sentence_id": {"$in": batch}},
                limit=100,
            )
            for rec in sentence_result.get("records", []):
                learned_sentences.append(
                    {
                        "text": rec.get("text", ""),
                        "translation": rec.get("translation", ""),
                    }
                )

        logger.info(
            f"[match] scholar={scholar_id}, "
            f"已学={len(sentence_ids)} 句, "
            f'输入="{input_sentence}"'
        )

        # 3. 执行 LangGraph 工作流
        result = await match_dialogue(
            input_sentence=input_sentence,
            scholar_id=scholar_id,
            learned_sentences=learned_sentences,
        )
        return result

    except Exception as e:
        logger.error(f"[match] 对话匹配异常: {e}")
        raise HTTPException(status_code=500, detail=f"对话匹配失败: {str(e)}")
