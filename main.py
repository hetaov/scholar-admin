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
"""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import PORT
from services.routes_system import router as system_router
from services.routes_crud import router as crud_router
from services.routes_tracking import router as tracking_router
from services.routes_state import router as state_router
from services.routes_dialogue import router as dialogue_router
from services.routes_vision import router as vision_router
from services.routes_build import router as build_router
from services.routes_admin import router as admin_router
from services.routes_eval import router as eval_router

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

# 挂载所有路由
app.include_router(system_router)
app.include_router(crud_router)
app.include_router(tracking_router)
app.include_router(state_router)
app.include_router(dialogue_router)
app.include_router(vision_router)
app.include_router(build_router)
app.include_router(admin_router)
app.include_router(eval_router)

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {PORT}...")
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
