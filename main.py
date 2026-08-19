"""Scholar Admin API — FastAPI 主入口

接口拆分：
- 系统      → services/routes_system.py
- CRUD      → services/routes_crud.py
- 追踪/教材  → services/routes_tracking.py
- 学习状态   → services/routes_state.py
- 对话匹配   → services/routes_dialogue.py
- AI 识别   → services/routes_vision.py
- 教材构建   → services/routes_build.py
- 管理       → services/routes_admin.py
- 评估       → services/routes_eval.py
- 语音合成   → services/routes_tts.py
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import PORT, RENDER_OUTPUT_DIR, RENDER_STATIC_URL_PREFIX
from services.auth import require_paid_user
from services.routes_system import router as system_router
from services.routes_crud import router as crud_router
from services.routes_tracking import router as tracking_router
from services.routes_state import router as state_router
from services.routes_dialogue import router as dialogue_router
from services.routes_vision import router as vision_router
from services.routes_build import router as build_router
from services.routes_admin import router as admin_router
from services.routes_conversation import router as conversation_router
from services.routes_eval import router as eval_router
from services.routes_evaluation import router as evaluation_router
from services.routes_training import router as training_router
from services.routes_tts import router as tts_router
from services.routes_planner import router as planner_router
from services.routes_math import router as math_router

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("scholar-admin")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Scholar Admin API",
    description="学者管理后台 — 数据库 CRUD + AI 教材识别 + 对话匹配",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 路由挂载
# 调用付费 AI 能力（ASR/LLM/TTS/视觉/评测/内容生成）的路由必须通过白名单鉴权；
# 基础数据路由（学习记录上报/查询等）保持开放，保证小程序基础功能可用。
# ---------------------------------------------------------------------------

# 免费/基础能力路由：保持开放
_FREE_ROUTERS = [
    system_router,
    crud_router,
    tracking_router,
    state_router,
    admin_router,
]

# 付费 AI 能力路由：require_paid_user 白名单鉴权
_PAID_ROUTERS = [
    dialogue_router,
    vision_router,
    build_router,
    eval_router,
    evaluation_router,
    conversation_router,
    training_router,
    tts_router,
    planner_router,
    math_router,
]

for _router in _FREE_ROUTERS:
    app.include_router(_router)

for _router in _PAID_ROUTERS:
    app.include_router(_router, dependencies=[Depends(require_paid_user)])

# ---------------------------------------------------------------------------
# F3.2 练习纸渲染产物静态服务（PDF/PNG/预览图；目录启动时创建）
# file_refs 引用 RENDER_STATIC_URL_PREFIX（默认 /static/sheets）下的相对 URL
# ---------------------------------------------------------------------------

Path(RENDER_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
app.mount(
    RENDER_STATIC_URL_PREFIX,
    StaticFiles(directory=RENDER_OUTPUT_DIR),
    name="sheet_artifacts",
)

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
