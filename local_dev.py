"""本地联调入口（临时）：全部路由挂载到 FakeDB 内存库，无需腾讯云凭证。

用法: python local_dev.py   # 监听 127.0.0.1:8081
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI  # noqa: E402

from tests.fakes.fake_db import FakeDB  # noqa: E402
from services import (  # noqa: E402
    routes_admin,
    routes_build,
    routes_crud,
    routes_dialogue,
    routes_state,
    routes_system,
    routes_tracking,
    routes_vision,
)

fake_db = FakeDB()
for mod in (
    routes_system,
    routes_crud,
    routes_tracking,
    routes_state,
    routes_dialogue,
    routes_vision,
    routes_build,
    routes_admin,
):
    mod.get_db = lambda: fake_db

app = FastAPI(title="Scholar Admin API (Local FakeDB)")
for mod in (
    routes_system,
    routes_crud,
    routes_tracking,
    routes_state,
    routes_dialogue,
    routes_vision,
    routes_build,
    routes_admin,
):
    app.include_router(mod.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8081)
