"""系统接口：健康检查 + 集合列表"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from config import ENV_ID, REGION
from services.dependencies import get_db

logger = logging.getLogger("scholar-admin.routes.system")
router = APIRouter(tags=["系统"])


@router.get("/")
async def root():
    return {
        "service": "Scholar Admin API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@router.get("/health")
async def health():
    try:
        db = get_db()
        await db.query(collection="sentence_v2", where={}, limit=1)
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/collections")
async def list_collections():
    """列出所有集合及其配置信息"""
    return [
        {
            "name": "study_attempt",
            "description": "学习事件表 — append-only, 每次学习行为写一条(替代旧 tracking 写入)",
            "indexes": ["scholar_id", "sentence_id", "skill_code", "session_id"],
        },
        {
            "name": "study_session",
            "description": "学习会话表 — start 创建 / end 结算, 回填 duration_sec 与 attempt_count",
            "indexes": ["scholar_id", "textbook_id", "status"],
        },
        {
            "name": "textbook_v2",
            "description": "教材 v2 — 内容模型分层后的教材主表(替代 textbook)",
            "indexes": ["textbook_id"],
        },
        {
            "name": "chapter",
            "description": "章节表 — 教材 → 章",
            "indexes": ["chapter_id", "textbook_id", "order"],
        },
        {
            "name": "lesson",
            "description": "课表 — 章 → 课",
            "indexes": ["lesson_id", "chapter_id", "textbook_id", "order"],
        },
        {
            "name": "sentence_v2",
            "description": "语句 v2 — 内容模型分层后的句子表(替代 sentence)",
            "indexes": ["sentence_id", "chapter_id", "lesson_id", "textbook_id"],
        },
        {
            "name": "skill",
            "description": "能力定义表 — 能力种子数据(translation/listening/speaking/reading)",
            "indexes": ["skill_code"],
        },
        {
            "name": "skill_state",
            "description": "能力状态表 — 学者 × 句子 × 能力 的当前状态",
            "indexes": ["scholar_id", "sentence_id", "skill_code", "lesson_id"],
        },
        {
            "name": "scholar_book",
            "description": "学者×教材关联表 — 断点续学(current_chapter_id/current_lesson_id) + 累计时长(total_time_spent) + 学科标识(subject_type)",
            "indexes": ["scholar_id", "textbook_id", "status", "last_studied_at", "subject_type"],
        },
    ]


@router.get("/collections/{name}")
async def get_collection(name: str):
    """获取指定集合的文档结构（采样 1 条文档）"""
    try:
        db = get_db()
        result = await db.query(collection=name, where={}, limit=1)
        records = result.get("records", [])
        return {
            "collection": name,
            "sample": records[0] if records else None,
            "total_estimated": result.get("total", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")
