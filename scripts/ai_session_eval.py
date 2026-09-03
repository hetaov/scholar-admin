#!/usr/bin/env python3
"""沉浸式 AI 会话域 · scholar-admin 接口混元评估脚本

配套评估方案：scholar-skill/docs_v1/沉浸式AI会话域接口混元评估方案.md
被评功能设计稿：scholar-skill/docs_v1/沉浸式学习页中的AI会话功能优化.md（§3 会话 v2 / §4 场景角色 / §5 提示分级 / §6 融合生成）

评估分四层门禁：
  L0 环境门禁   —— /health 可达（可自启 uvicorn）、AUTH_MODE、混元/火山凭据存在性
  L1 可用性门禁 —— 拉取 /openapi.json 校验接口是否注册 + 契约探针（无 LLM 成本）
                 → 保证接口可用：已实现接口必须 READY；尚未实现的草案接口
                   如实标记 NOT_REGISTERED(待实现)，默认不阻断评估
  L2 功能链路   —— 已实现接口端到端：/ai/session/v2（C6 开场→轮询 / C7 续轮→轮询）、
                 /match/dialogue.task 异步、/eval/translate/v2 异步、
                 /conversation/scenario→turn×2→history，收集 AI 产物与耗时
  L3 混元质量   —— LLM-as-Judge（Judge≠Generator：生成=火山方舟，评分=混元），
                 按评估方案 §5.3 评分维度表打分（0~1，阈值 HUNYUAN_EVAL_PASS_THRESHOLD 默认 0.7）

用法：
  # 服务已运行在 8080；学者需有已学句数据（否则 L2 记为 NEEDS_DATA）
  python scripts/ai_session_eval.py --scholar-id <真实学者id>

  # 服务未启动：脚本自动拉起 uvicorn
  python scripts/ai_session_eval.py --port 8080 --scholar-id <真实学者id>

  # 只跑 L0+L1 可用性门禁（秒级，无 LLM 成本）
  python scripts/ai_session_eval.py --port 8080 --no-live --no-judge

  # 会话 v2 转正后：全接口可用性 + 会话 v2 异步链路冒烟（不调混元 Judge）
  python scripts/ai_session_eval.py --port 8080 --scholar-id xxx --no-judge

  # 作 CI 门禁（草案接口必须注册且可用；strict 将 NEEDS_DATA 视为失败）
  python scripts/ai_session_eval.py --port 8080 --scholar-id xxx --require-ai-session-v2 --strict

  # 限制混元 Judge 调用次数（成本控制）
  python scripts/ai_session_eval.py --port 8080 --scholar-id xxx --judge-limit 5

退出码：0=通过；1=必需门禁失败；2=Judge 等环境配置缺失
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.exit("[ERROR] 缺少 httpx，请先安装依赖：pip install httpx")

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover
    sys.exit("[ERROR] 缺少 openai，请先安装依赖：pip install openai")

# 保证 `python scripts/xxx.py` 与 `python -m scripts.xxx` 均可运行（项目根入 sys.path）
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 复用 config 加载 .env（凭据源与 main.py 一致）
from config import (  # noqa: E402
    AUTH_MODE,
    HUNYUAN_BASE_URL,
    HUNYUAN_EVAL_MODEL,
    HUNYUAN_EVAL_PASS_THRESHOLD,
    HUNYUAN_SECRET_KEY,
    HUNYUAN_TIMEOUT_SECONDS,
    PORT,
    VOLCANO_API_KEY,
    VOLCANO_CHAT_MODEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ai_session_eval")

POLL_TERMINAL = {"success", "failed"}

# ---------------------------------------------------------------------------
# 被评接口注册表（对齐评估方案 §2）
#   status: implemented=已实现（可用性必需） / draft=设计稿草案（默认不阻断）
# ---------------------------------------------------------------------------

INTERFACES = [
    {
        "id": "ai_session_v2_submit",
        "label": "沉浸式会话 v2 提交（契约 §3.12①）",
        "method": "POST",
        "path": "/ai/session/v2",
        "status": "implemented",
        "family": "ai_session_v2",
    },
    {
        "id": "ai_session_v2_task",
        "label": "沉浸式会话 v2 查询（契约 §3.12②）",
        "method": "GET",
        "path": "/ai/session/v2/task/{task_id}",
        "status": "implemented",
        "family": "ai_session_v2",
    },
    {
        "id": "match_dialogue",
        "label": "对话匹配（现状会话回复 /match/dialogue）",
        "method": "POST",
        "path": "/match/dialogue",
        "status": "implemented",
        "family": "match_dialogue",
    },
    {
        "id": "match_dialogue_task",
        "label": "对话匹配异步提交 /match/dialogue/task",
        "method": "POST",
        "path": "/match/dialogue/task",
        "status": "implemented",
        "family": "match_dialogue",
    },
    {
        "id": "match_dialogue_task_query",
        "label": "对话匹配异步查询 /match/dialogue/task/{task_id}",
        "method": "GET",
        "path": "/match/dialogue/task/{task_id}",
        "status": "implemented",
        "family": "match_dialogue",
    },
    {
        "id": "eval_translate",
        "label": "评估 v1 同步（语音转写现状 /eval/translate）",
        "method": "POST",
        "path": "/eval/translate",
        "status": "implemented",
        "family": "eval",
    },
    {
        "id": "eval_translate_v2",
        "label": "评估 v2 异步提交 /eval/translate/v2",
        "method": "POST",
        "path": "/eval/translate/v2",
        "status": "implemented",
        "family": "eval",
    },
    {
        "id": "eval_translate_v2_query",
        "label": "评估 v2 异步查询 /eval/translate/v2/task/{task_id}",
        "method": "GET",
        "path": "/eval/translate/v2/task/{task_id}",
        "status": "implemented",
        "family": "eval",
    },
    {
        "id": "conversation_scenario",
        "label": "会话链路 开场 /conversation/scenario",
        "method": "POST",
        "path": "/conversation/scenario",
        "status": "implemented",
        "family": "conversation",
    },
    {
        "id": "conversation_turn",
        "label": "会话链路 续轮 /conversation/turn",
        "method": "POST",
        "path": "/conversation/turn",
        "status": "implemented",
        "family": "conversation",
    },
    {
        "id": "conversation_history",
        "label": "会话链路 历史 /conversation/history",
        "method": "GET",
        "path": "/conversation/history",
        "status": "implemented",
        "family": "conversation",
    },
]

# 业务样例（对齐设计稿 §3.3 商务谈判示例，评估方案 §6）
TARGET_SENTENCE = (
    "We are willing to consider a discount if the order volume is substantial."
)
REVIEW_SENTENCE = "We need to clarify the payment terms."

# 同步生成类接口的客户端超时（秒）：/conversation/turn、/conversation/history
# （可能触发 end_session_with_summary LLM 小结）、/match/dialogue 后端为实时
# LLM 生成，真实耗时可达 1–4 分钟（2026-09-03 实测 ~3–4 分钟）；
# 客户端超时若小于生成耗时将被判 REQUEST_FAIL timed out。
# 异步提交/轮询类提交超时仍为 15s，终态等待上限见 --max-wait（默认 300）。
SYNC_LLM_CALL_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def ms(sec: float) -> str:
    return f"{sec * 1000:.0f}ms" if sec < 1 else f"{sec:.2f}s"


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ensure_server(base_url: str, port: int, autostart: bool) -> subprocess.Popen | None:
    """确认服务可达；不可达且允许自启则拉起 uvicorn，返回子进程句柄。"""
    try:
        r = httpx.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            logger.info(f"服务已就绪: {base_url}/health")
            return None
    except Exception:
        pass
    if not autostart:
        sys.exit(
            f"[ERROR] {base_url} 不可达。请先运行 `python main.py`，"
            f"或加 --port 让脚本自动拉起服务。"
        )
    logger.info(f"服务未启动，自动拉起: python -m uvicorn main:app --port {port}")
    log_path = Path("/tmp/scholar_ai_session_eval_server.log")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(HERE),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):  # 最多等 30s
        if proc.poll() is not None:
            sys.exit(
                f"[ERROR] uvicorn 启动失败（退出码 {proc.returncode}），"
                f"日志见 {log_path}"
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"服务已拉起: {base_url}（日志: {log_path}）")
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"[ERROR] uvicorn 30s 内未就绪，日志见 {log_path}")


def load_openapi(client: httpx.Client, base_url: str) -> set[tuple[str, str]]:
    """拉取 /openapi.json → {(METHOD, path_without_params), ...}。"""
    r = client.get(f"{base_url}/openapi.json", timeout=10)
    r.raise_for_status()
    spec = r.json()
    routes: set[tuple[str, str]] = set()
    for path, methods in (spec.get("paths") or {}).items():
        norm = path.split("{", 1)[0].rstrip("/")
        for m in methods:
            if m.lower() in ("get", "post", "put", "delete", "patch"):
                routes.add((m.upper(), norm))
    return routes


def path_is_registered(routes: set[tuple[str, str]], method: str, path: str) -> bool:
    """注册判断：去路径参数段前缀比对。"""
    norm = path.split("{", 1)[0].rstrip("/")
    for m, p in routes:
        if m == method.upper() and p == norm:
            return True
    # GET 查询类路径无参数前缀时完全匹配
    return (method.upper(), norm) in routes


# ---------------------------------------------------------------------------
# L1 可用性门禁：契约探针（无 LLM/DB 成本）
# ---------------------------------------------------------------------------

MISSING_ID = "__eval_missing__"


def probe_endpoint(client: httpx.Client, base_url: str, entry: dict) -> dict:
    """对单个接口做注册 + 契约探针。返回 {route_registered, probe_status, detail}。"""
    if entry["status"] == "draft":
        # 草案接口：注册与否都要如实记录，不打探针（未实现时 POST/GET 会 404）
        return {"probe_status": "NOT_REGISTERED"}

    routes = load_openapi(client, base_url)  # 每次调用代价小，缓存由上层负责
    registered = path_is_registered(routes, entry["method"], entry["path"])
    if not registered:
        return {"probe_status": "NOT_REGISTERED", "detail": "openapi 中无此路由"}

    try:
        method, path = entry["method"], entry["path"]
        if entry["id"] in ("match_dialogue", "match_dialogue_task"):
            resp = client.post(f"{base_url}{path}", json={}, timeout=10)
            ok = resp.status_code == 400  # 参数护栏：缺 scholarId/sentence → HTTP 400
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code}"}
        if entry["id"] == "eval_translate":
            # v1 schema 中 original_text 必填（{} 会 422），给最小合法值以触发业务错误护栏
            resp = client.post(f"{base_url}{path}",
                               json={"original_text": "It is a watch."}, timeout=10)
            body = resp.json()
            ok = body.get("success") is False and body.get("code") == "INVALID_INPUT"
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code} code={body.get('code')}"}
        if entry["id"] == "eval_translate_v2":
            resp = client.post(f"{base_url}{path}", json={}, timeout=10)
            body = resp.json()
            ok = body.get("success") is False and body.get("code") == "INVALID_INPUT"
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code} code={body.get('code')}"}
        if entry["id"] in ("match_dialogue_task_query", "eval_translate_v2_query"):
            resp = client.get(f"{base_url}{path.format(task_id=MISSING_ID)}", timeout=10)
            ok = resp.status_code == 404  # 任务不存在 → 404
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code}"}
        if entry["id"] == "ai_session_v2_submit":
            # 契约 §3.12：字段可选化 + 手动校验，{} → 200 + success=false + INVALID_INPUT
            resp = client.post(f"{base_url}{path}", json={}, timeout=10)
            body = resp.json()
            ok = body.get("success") is False and body.get("code") == "INVALID_INPUT"
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code} code={body.get('code')}"}
        if entry["id"] == "ai_session_v2_task":
            resp = client.get(f"{base_url}{path.format(task_id=MISSING_ID)}", timeout=10)
            ok = resp.status_code == 404  # 任务不存在/过期 → 404
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code}"}
        if entry["id"] in ("conversation_scenario", "conversation_turn"):
            resp = client.post(f"{base_url}{path}", json={}, timeout=10)
            body = resp.json()
            ok = body.get("success") is False and body.get("code") == "INVALID_INPUT"
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code} code={body.get('code')}"}
        if entry["id"] == "conversation_history":
            resp = client.get(f"{base_url}{path}", params={"session_id": MISSING_ID}, timeout=10)
            body = resp.json()
            ok = body.get("success") is False and body.get("code") == "NOT_FOUND"
            return {"probe_status": "OK" if ok else "CONTRACT_ERR",
                    "detail": f"HTTP {resp.status_code} code={body.get('code')}"}
    except Exception as e:  # noqa: BLE001
        return {"probe_status": "CONTRACT_ERR", "detail": f"{type(e).__name__}: {e}"}
    return {"probe_status": "CONTRACT_ERR", "detail": "未定义探针"}


def run_availability(client: httpx.Client, base_url: str) -> list[dict]:
    """L1：全部接口的注册 + 探针结果矩阵。"""
    print("\n===== [L1 可用性门禁] 接口注册 + 契约探针 =====")
    routes = load_openapi(client, base_url)
    results: list[dict] = []
    for entry in INTERFACES:
        if entry["status"] == "draft":
            registered = path_is_registered(routes, entry["method"], entry["path"])
            probe = "READY" if registered else "NOT_REGISTERED"
            detail = "已在 openapi 注册" if registered else "未实现（docs_v1 §3.3 草案，待契约落地）"
        else:
            r = probe_endpoint(client, base_url, entry)
            probe = r["probe_status"]
            detail = r.get("detail", "")
        # 已实现接口：契约探针 OK 即 READY；草案接口：注册即 READY
        available = probe in ("OK", "READY")
        res = {
            "id": entry["id"],
            "label": entry["label"],
            "method": entry["method"],
            "path": entry["path"],
            "status": entry["status"],
            "probe": probe,
            "detail": detail,
            "available": available,
        }
        results.append(res)
        if available:
            mark = "✓"
        elif entry["status"] == "draft":
            mark = "⚠(草案)"
        else:
            mark = "✗"
        print(f"  {mark} [{entry['status']:<12}] {entry['label']}: {probe} {detail}")
    return results


# ---------------------------------------------------------------------------
# L2 功能链路：异步提交 + 轮询
# ---------------------------------------------------------------------------


def poll_task(
    client: httpx.Client,
    base_url: str,
    task_id: str,
    task_api: str,
    *,
    poll_interval: float,
    max_wait: float,
    result_key: str,
    task_id_key: str,
    status_key: str,
) -> dict | None:
    """通用轮询：直到 success/failed 或超时。返回 {task_id,status,result,polls}。"""
    poll_start = time.perf_counter()
    polls = 0
    last_status = "pending"
    result: dict | None = None
    while True:
        if time.perf_counter() - poll_start > max_wait:
            print(f"    [轮询] 超过 {ms(max_wait)} 仍未终态（status={last_status}），放弃")
            return None
        time.sleep(poll_interval)
        polls += 1
        t1 = time.perf_counter()
        try:
            r = client.get(f"{base_url}{task_api.format(task_id=task_id)}", timeout=20)
        except Exception as e:  # noqa: BLE001
            print(f"    [轮询] 请求异常: {type(e).__name__}: {e}")
            return None
        dt = time.perf_counter() - t1
        try:
            data = r.json()["data"]
        except Exception:
            print(f"    [轮询] HTTP {r.status_code} 非预期响应: {r.text[:200]}")
            return None
        last_status = str(data.get(status_key) or "").lower()
        result = data.get("result") or {}
        extra = ""
        if r.status_code == 404:
            print("    [轮询] 404：任务不存在/已过期")
            return None
        if data.get("error"):
            extra = f" error={str(data['error'])[:60]}"
        elif result:
            brief = json.dumps(result, ensure_ascii=False)[:100]
            extra = f" result={brief}"
        print(f"    [轮询] #{polls} 耗时={ms(dt)} status={last_status}{extra}")
        if last_status in POLL_TERMINAL:
            break
    return {
        task_id_key: task_id,
        "status": last_status,
        "result": result,
        "polls": polls,
    }


def run_eval_v2_flow(
    client: httpx.Client,
    base_url: str,
    *,
    poll_interval: float,
    max_wait: float,
) -> dict:
    """C5：异步翻译评估（对照先例）——文字路径 ec（英原句 + 中文译文）。"""
    print("\n----- [L2/C5] /eval/translate/v2 异步链路（ec 文字直评） -----")
    case = {
        "original_text": TARGET_SENTENCE,
        "user_input": "如果订单量足够大，我们愿意考虑给予折扣。",
        "scholar_id": "eval_smoke",
    }
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/eval/translate/v2", json=case, timeout=10)
    except Exception as e:  # noqa: BLE001
        return {"case": "C5_eval_v2", "status": "REQUEST_FAIL", "error": str(e)}
    submit_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    if not body.get("success"):
        return {"case": "C5_eval_v2", "status": "BIZ_FAIL",
                "error": f"{body.get('code')} {body.get('message')}"}
    task_id = body["data"]["task_id"]
    print(f"    [提交] HTTP {resp.status_code} 耗时={ms(submit_ms / 1000)} task_id={task_id}")

    polled = poll_task(
        client, base_url, task_id, "/eval/translate/v2/task/{task_id}",
        poll_interval=poll_interval, max_wait=max_wait,
        result_key="result", task_id_key="task_id", status_key="status",
    )
    if polled is None:
        return {"case": "C5_eval_v2", "task_id": task_id,
                "status": "POLL_TIMEOUT", "submit_ms": submit_ms}
    return {
        "case": "C5_eval_v2",
        "task_id": task_id,
        "submit_ms": submit_ms,
        "status": polled["status"],
        "result": polled["result"],
    }


SESSION_V2_SCENARIO = {
    "scene_id": "negotiation",
    "title": "商务谈判 · 折扣条件",
    "scene": "Learner is a sales representative negotiating an order discount with a buyer.",
    "goal": "Secure a discount by offering order-volume terms",
    "constraints": "Stay in role; guide the learner to produce target sentences.",
}
SESSION_V2_ROLES = {
    "ai_role": {
        "name": "Buyer",
        "identity": "Procurement lead of a large client",
        "style": "polite but firm",
        "goal": "Nudge the learner to justify a discount request",
    },
    "learner_role": {"name": "Sales Rep", "identity": "Vendor representative"},
}
SESSION_V2_GROUPS = [
    {
        "kind": "new",
        "sentences": [
            {
                "sentence_id": "tg_discount",
                "content": TARGET_SENTENCE,
            }
        ],
    },
    {
        "kind": "review",
        "sentences": [
            {"sentence_id": "rev_payment", "content": REVIEW_SENTENCE}
        ],
    },
]


def run_ai_session_v2_flow(
    client: httpx.Client,
    base_url: str,
    *,
    poll_interval: float,
    max_wait: float,
) -> list[dict]:
    """C6/C7：沉浸式会话 v2 异步链路（start 开场 → 轮询 → turn 续轮 → 轮询）。

    素材/场景/角色自包含（契约 §3.12），scholar_id 仅留痕，无需学者真实已学句；
    产物含 ai_text/hint/suggested_targets，供 L3 混元按 ai_session_v2 维度评分。
    """
    print("\n----- [L2/C6/C7] /ai/session/v2 会话 v2 异步链路 -----")
    out: list[dict] = []

    # C6 start：场景/角色/素材注入 → 提交毫秒级返回 → 轮询到终态
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/ai/session/v2", json={
            "scholar_id": "eval_smoke",
            "mode": "start",
            "scenario": SESSION_V2_SCENARIO,
            "roles": SESSION_V2_ROLES,
            "groups": SESSION_V2_GROUPS,
            "preferred_type": "auto",
        }, timeout=10)
    except Exception as e:  # noqa: BLE001
        return [{"case": "C6_ai_session_v2_start", "status": "REQUEST_FAIL", "error": str(e)}]
    submit_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    if not body.get("success"):
        return [{"case": "C6_ai_session_v2_start", "status": "BIZ_FAIL",
                 "error": f"{body.get('code')} {body.get('message')}"}]
    data = body["data"]
    session_id = data["session_id"]
    print(f"    [start 提交] HTTP {resp.status_code} 耗时={ms(submit_ms / 1000)} "
          f"task_id={data['task_id']} session_id={session_id}")
    polled = poll_task(
        client, base_url, data["task_id"], "/ai/session/v2/task/{task_id}",
        poll_interval=poll_interval, max_wait=max_wait,
        result_key="result", task_id_key="task_id", status_key="status",
    )
    if polled is None:
        out.append({"case": "C6_ai_session_v2_start", "task_id": data["task_id"],
                    "status": "POLL_TIMEOUT", "submit_ms": submit_ms})
        return out
    out.append({
        "case": "C6_ai_session_v2_start", "session_id": session_id,
        "task_id": data["task_id"], "submit_ms": submit_ms,
        "status": polled["status"], "result": polled["result"],
    })
    if polled["status"] != "success":
        return out  # 开场失败不再续轮

    # C7 turn：续轮（带本轮作答 + assisted 上报）→ 提交 → 轮询
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/ai/session/v2", json={
            "scholar_id": "eval_smoke",
            "mode": "turn",
            "session_id": session_id,
            "user_input": "If you increase the volume, we can talk about a discount.",
            "preferred_type": "auto",
            "assisted": False,
        }, timeout=10)
    except Exception as e:  # noqa: BLE001
        out.append({"case": "C7_ai_session_v2_turn", "status": "REQUEST_FAIL", "error": str(e)})
        return out
    submit_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    if not body.get("success"):
        out.append({"case": "C7_ai_session_v2_turn", "status": "BIZ_FAIL",
                    "error": f"{body.get('code')} {body.get('message')}"})
        return out
    data = body["data"]
    print(f"    [turn 提交] HTTP {resp.status_code} 耗时={ms(submit_ms / 1000)} "
          f"task_id={data['task_id']}")
    polled = poll_task(
        client, base_url, data["task_id"], "/ai/session/v2/task/{task_id}",
        poll_interval=poll_interval, max_wait=max_wait,
        result_key="result", task_id_key="task_id", status_key="status",
    )
    if polled is None:
        out.append({"case": "C7_ai_session_v2_turn", "task_id": data["task_id"],
                    "status": "POLL_TIMEOUT", "submit_ms": submit_ms})
        return out
    out.append({
        "case": "C7_ai_session_v2_turn", "session_id": session_id,
        "task_id": data["task_id"], "submit_ms": submit_ms,
        "status": polled["status"], "result": polled["result"],
    })
    return out


def run_match_dialogue_flow(
    client: httpx.Client,
    base_url: str,
    scholar_id: str,
    *,
    poll_interval: float,
    max_wait: float,
    use_async: bool,
) -> dict:
    """C1/C2：对话匹配（沉浸式会话回复现状链路），scenario/sessionId 透传走异步。"""
    name = "C1_match_dialogue_task" if use_async else "C2_match_dialogue"
    print(f"\n----- [L2/{name}] /match/dialogue{'/task' if use_async else ''} "
          f"scholar_id={scholar_id} -----")
    payload = {
        "scholarId": scholar_id,
        "sentence": TARGET_SENTENCE,
    }
    if use_async:
        payload.update({
            "scenario": "business_negotiation",
            "sessionId": f"eval_s_{int(time.time() * 1000)}",
        })
    t0 = time.perf_counter()
    try:
        path = "/match/dialogue/task" if use_async else "/match/dialogue"
        # 异步任务仅提交（毫秒级，15s 充裕）；同步接口要等后端 LLM 实时生成（分钟级）
        timeout = 15 if use_async else SYNC_LLM_CALL_TIMEOUT
        resp = client.post(f"{base_url}{path}", json=payload, timeout=timeout)
    except Exception as e:  # noqa: BLE001
        return {"case": name, "status": "REQUEST_FAIL", "error": str(e)}
    submit_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    if resp.status_code != 200 or not body.get("success"):
        # 404/400 或业务失败：可能学者无已学句
        detail = body.get("error") or body.get("detail") or body.get("message") or ""
        if resp.status_code == 200 and "暂无已学语句" in str(detail):
            return {"case": name, "status": "NEEDS_DATA",
                    "error": "学者无已学语句（--scholar-id 需指向有学习记录的学者）"}
        return {"case": name, "status": "BIZ_FAIL",
                "error": f"HTTP {resp.status_code}: {str(detail)[:200]}"}

    if not use_async:
        result = body.get("data") or {}
        print(f"    [同步] 耗时={ms(submit_ms / 1000)} type={result.get('type')} "
              f"source={result.get('source')}")
        return {"case": name, "submit_ms": submit_ms, "status": "success",
                "result": result, "is_question": body.get("is_question")}

    task_id = body["data"]["taskId"]
    print(f"    [提交] HTTP {resp.status_code} 耗时={ms(submit_ms / 1000)} "
          f"taskId={task_id} status={body['data']['status']}")
    polled = poll_task(
        client, base_url, task_id, "/match/dialogue/task/{task_id}",
        poll_interval=poll_interval, max_wait=max_wait,
        result_key="result", task_id_key="taskId", status_key="status",
    )
    if polled is None:
        return {"case": name, "taskId": task_id, "status": "POLL_TIMEOUT",
                "submit_ms": submit_ms}
    return {"case": name, "taskId": task_id, "submit_ms": submit_ms,
            "status": polled["status"], "result": polled["result"]}


def run_conversation_flow(
    client: httpx.Client,
    base_url: str,
    scholar_id: str,
    *,
    poll_interval: float,
    max_wait: float,
) -> list[dict]:
    """C3/C4：/conversation/scenario → turn×2 → history（会话推进 + 产物收集）。"""
    print(f"\n----- [L2/C3/C4] /conversation/* 会话链路 scholar_id={scholar_id} -----")
    out: list[dict] = []

    # 开场
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/conversation/scenario", json={
            "scholar_id": scholar_id,
            "scenario": "business_negotiation",
            "topic": "Product pricing negotiation",
        }, timeout=120)  # 开场含 pre_assess + 图初始化 + 开场白 LLM 生成，首次 DB 访问可能较慢
    except Exception as e:  # noqa: BLE001
        return [{"case": "C3_scenario", "status": "REQUEST_FAIL", "error": str(e)}]
    scenario_ms = (time.perf_counter() - t0) * 1000
    body = resp.json()
    if not body.get("success"):
        return [{"case": "C3_scenario", "status": "BIZ_FAIL",
                 "error": f"{body.get('code')} {body.get('message')}"}]
    data = body["data"]
    session_id = data["session_id"]
    print(f"    [开场] 耗时={ms(scenario_ms / 1000)} session_id={session_id} "
          f"difficulty={data.get('difficulty')} "
          f"reply={str(data.get('reply') or '')[:60]!r}")
    out.append({"case": "C3_scenario", "status": "success", "submit_ms": scenario_ms,
                "result": {k: data.get(k) for k in
                           ("session_id", "difficulty", "pre_assessment", "cold_start",
                            "reply", "state")}})

    # 两轮 turn：第一轮正常表达；第二轮弱答（触发 hint/rephrase 递进，C4）
    turns = [
        ("C3_turn_good", "The price is a bit high for us. Can you offer a better deal?"),
        ("C4_turn_weak", "Discount. Big order. You know."),
    ]
    for case_id, utterance in turns:
        t0 = time.perf_counter()
        try:
            resp = client.post(f"{base_url}/conversation/turn", json={
                "session_id": session_id,
                "utterance": utterance,
                "mode": "text",
            }, timeout=SYNC_LLM_CALL_TIMEOUT)  # 后端实时生成 AI 回复（分钟级）
        except Exception as e:  # noqa: BLE001
            out.append({"case": case_id, "status": "REQUEST_FAIL", "error": str(e)})
            continue
        turn_ms = (time.perf_counter() - t0) * 1000
        tbody = resp.json()
        if not tbody.get("success"):
            out.append({"case": case_id, "status": "BIZ_FAIL",
                        "error": f"{tbody.get('code')} {tbody.get('message')}"})
            continue
        tdata = tbody["data"]
        state = tdata.get("state") or {}
        print(f"    [{case_id}] 耗时={ms(turn_ms / 1000)} stage={state.get('stage')} "
              f"reply={str(tdata.get('reply'))[:60]!r}")
        out.append({
            "case": case_id, "status": "success", "submit_ms": turn_ms,
            "result": {
                "reply": tdata.get("reply"),
                "state": state,
                "eval_verdict": (tdata.get("eval_verdict") or {}) if tdata.get("eval_verdict") else None,
            },
        })

    # 历史
    try:
        resp = client.get(f"{base_url}/conversation/history",
                          params={"session_id": session_id},
                          timeout=SYNC_LLM_CALL_TIMEOUT)  # 未结束会话首查会生成 LLM 小结
        hbody = resp.json()
        if hbody.get("success"):
            sess = (hbody.get("data") or {}).get("session") or {}
            print(f"    [history] stage/ended 判定 OK，summary={'有' if sess.get('summary') else '无(未结束)'}")
            out.append({"case": "C3_history", "status": "success",
                        "result": {"has_summary": bool(sess.get("summary")),
                                   "turn_count": len((hbody.get("data") or {}).get("turns") or [])}})
        else:
            out.append({"case": "C3_history", "status": "BIZ_FAIL",
                        "error": f"{hbody.get('code')} {hbody.get('message')}"})
    except Exception as e:  # noqa: BLE001
        out.append({"case": "C3_history", "status": "REQUEST_FAIL", "error": str(e)})
    return out


# ---------------------------------------------------------------------------
# L3 混元 Judge（LLM-as-Judge，Judge≠Generator）
# ---------------------------------------------------------------------------

# 评分维度表（评估方案 §5.3）：按 family 区分
RUBRICS: dict[str, dict] = {
    "match_dialogue": {
        "system": (
            "你是 scholar-admin「沉浸式英语会话域」质量评审专家。"
            "对话内容由火山方舟生成，你仅做独立质量评分。只输出合法 JSON。"
        ),
        "dimensions": [
            ("引导而非代答", 0.25, "AI 是否在引导学习者产出目标表达，而非直接代答或替学习者把整句说完"),
            ("素材牵引", 0.25, "输出是否围绕输入句/目标句合理设问或回应，不离题、不空泛"),
            ("场景与角色一致性", 0.15, "输出是否贴合请求中的 scenario/topic 语境"),
            ("语言自然度与正确性", 0.20, "英文表达自然、语法正确、语气符合角色"),
            ("结构化契约", 0.15, "返回字段(type/statement/question/source 等)齐备且语义正确"),
        ],
    },
    "conversation": {
        "system": (
            "你是 scholar-admin「沉浸式英语会话域」质量评审专家。"
            "会话回复由 LLM 生成，你仅做独立质量评分。只输出合法 JSON。"
        ),
        "dimensions": [
            ("引导而非代答", 0.25, "AI 回复是否引导学习者说英语，而非直接用中文或代答"),
            ("素材牵引", 0.25, "回复是否围绕当前目标句/话题推进，能牵引学习者用出目标表达"),
            ("提示递进合理性", 0.15, "出现 hint/rephrase 时：hint 不得给整句答案；rephrase 更简单但同义；档位合理不跳级"),
            ("语言自然度与正确性", 0.20, "英文自然、语法正确"),
            ("结构化契约", 0.15, "reply/state{stage,hint,rephrased}/eval_verdict 语义正确、字段合法"),
        ],
    },
    "ai_session_v2": {
        "system": (
            "你是 scholar-admin「沉浸式英语会话域」质量评审专家。"
            "输出由会话生成模型产生，你按沉浸式功能设计稿做质量评分。只输出合法 JSON。"
        ),
        "dimensions": [
            ("场景/角色三要素生效", 0.20, "开场/回复是否体现 scenario.title/scene/goal 与 roles 人设，不跳出角色直接点评语法"),
            ("素材绑定与新句必用", 0.20, "AI 是否制造语境让学习者用出 new(kind=new) 目标句，开场不直接展示句子原文"),
            ("复习句埋伏", 0.15, "kind=review 旧句是否被 AI 自然埋伏引诱学习者说回（若本轮含复习素材）"),
            ("提示不给整句答案", 0.15, "hint 为词块/骨架/对照等渐进引导，不得直接给整句答案"),
            ("语言自然度与正确性", 0.15, "英文自然、语法正确、符合 ai_role.style"),
            ("结构化契约", 0.15, "content_type 合法(dialogue/retell/fill/task)、ai_text/hint/suggested_targets/assisted 字段语义正确"),
        ],
    },
}

_JUDGE_USER_TEMPLATE = """请评估以下 AI 产物质量。

接口：{interface_label}
请求要点：{request_summary}
AI 实际输出（JSON）：
{output_json}

评分维度（评分 0.0~1.0，权重加权得 score）：
{dimensions}

输出 JSON（不要任何解释/markdown）：
{{"score": 0.0~1.0, "dimensions": [{{"name": "维度名", "score": 0.0, "comment": "一句评价"}}], "feedback": "总体评价与改进建议", "issues": ["问题1"]}}"""


def _parse_eval_response(text: str) -> dict:
    """解析混元返回 JSON（容错：去 markdown fence / 提取首尾花括号）。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    return json.loads(text)


async def _judge_one(
    artifact: dict,
    family: str,
    interface_label: str,
    sem: asyncio.Semaphore,
) -> dict:
    """调混元评估单个产物。返回 {"score": float, "feedback", "issues", "dimensions"}。"""
    rubric = RUBRICS.get(family) or RUBRICS["conversation"]
    dim_lines = "\n".join(
        f"- {name}（权重 {w}）：{desc}" for name, w, desc in rubric["dimensions"]
    )
    prompt = _JUDGE_USER_TEMPLATE.format(
        interface_label=interface_label,
        request_summary=artifact.get("request_summary", ""),
        output_json=json.dumps(artifact.get("output") or {}, ensure_ascii=False),
        dimensions=dim_lines,
    )
    try:
        async with sem:
            # 混元 OpenAI 兼容网关调用：走 OpenAI SDK（与网关官方示例一致）。
            # 不使用 response_format=json_object：实测该网关对 hy4-preview 推理模型
            # 开启 json_object 时 content 为空（耗时更长），改由 prompt 约束 +
            # 下方 _parse_eval_response 容错解析（去 markdown fence / 提取花括号）。
            client = AsyncOpenAI(
                api_key=HUNYUAN_SECRET_KEY,
                base_url=HUNYUAN_BASE_URL,
                timeout=HUNYUAN_TIMEOUT_SECONDS,
                max_retries=0,
            )
            resp = await client.chat.completions.create(
                model=HUNYUAN_EVAL_MODEL,
                messages=[
                    {"role": "system", "content": rubric["system"]},
                    {"role": "user", "content": prompt},
                ],
            )
            content = (resp.choices[0].message.content or "").strip()
            if not content:
                # 推理类模型可能把答复放在 reasoning_content 而未落到 content
                raise ValueError(
                    f"Judge 响应 content 为空（model={HUNYUAN_EVAL_MODEL}，"
                    f"base_url={HUNYUAN_BASE_URL} 须为 OpenAI 兼容 /chat/completions 网关）"
                )
            result = _parse_eval_response(content)
            score = max(0.0, min(1.0, float(result.get("score", -1.0))))
            return {
                "score": score,
                "feedback": str(result.get("feedback", "")),
                "issues": result.get("issues", []),
                "dimensions": result.get("dimensions", []),
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(f"混元评估失败，降级跳过: {type(e).__name__}: {e}")
        return {"score": -1.0, "feedback": f"评估跳过: {e}", "issues": [], "dimensions": []}


async def _judge_all(tasks: list) -> list:
    """在统一事件循环内并发执行全部混元评分任务。

    Python 3.14 兼容：`asyncio.gather(*tasks)` 必须在运行中的 loop 内调用
    （3.12 前会隐式自建默认 loop，3.14 起直接抛 RuntimeError），
    因此不能写成 `asyncio.run(asyncio.gather(*tasks))`。
    """
    return await asyncio.gather(*tasks)


def build_artifacts(
    func_results: list[dict],
    *,
    conversation_session_id: str | None = None,
) -> list[dict]:
    """把 L2 产物转成可评分对象（仅含 AI 文本的产物才送评）。"""
    artifacts: list[dict] = []
    for r in func_results:
        if r.get("status") != "success":
            continue
        case = r["case"]
        if case.startswith("C5"):
            continue  # 翻译评估由引擎自带评分口径，不送混元
        if case == "C3_history":
            continue  # 无 AI 文本
        if case.startswith("C6") or case.startswith("C7"):
            family = "ai_session_v2"
            label = (
                "POST /ai/session/v2（start 开场）"
                if case == "C6_ai_session_v2_start"
                else "POST /ai/session/v2（turn 续轮）"
            )
            summary = (
                "scenario=商务谈判·折扣条件；roles=Buyer/Sales Rep；"
                "groups=new(折扣目标句)+review(付款条款)；preferred_type=auto"
                if case == "C6_ai_session_v2_start"
                else "续轮 session_id 装载上下文；user_input=…volume…discount…；assisted=false"
            )
            output = r.get("result") or {}
        elif case.startswith("C1") or case.startswith("C2"):
            family = "match_dialogue"
            label = "POST /match/dialogue(/task) 对话匹配回复"
            summary = (
                f"scholar_id=…; sentence=「{TARGET_SENTENCE}」; "
                f"scenario/sessionId 随任务透传" if case.startswith("C1")
                else f"sentence=「{TARGET_SENTENCE}」"
            )
            output = r.get("result") or {}
        elif case == "C3_scenario":
            family = "conversation"
            label = "POST /conversation/scenario 会话开场"
            summary = "scenario=business_negotiation; topic=Product pricing negotiation"
            output = r.get("result") or {}
        else:  # C3_turn_* / C4_turn_*
            family = "conversation"
            label = f"POST /conversation/turn 续轮（{case}）"
            summary = "会话续轮，含 utterance 与 AI reply/state"
            output = r.get("result") or {}
        artifacts.append({
            "case": case,
            "family": family,
            "interface_label": label,
            "request_summary": summary,
            "output": output,
        })
    return artifacts


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------


def print_report(report: dict) -> None:
    print("\n\n===== 评估汇总 =====")
    av = report["availability"]
    print("[L1 可用性]")
    for r in av:
        mark = "✓" if r["available"] else ("⚠(草案)" if r["status"] == "draft" else "✗")
        print(f"  {mark} {r['label']}: {r['probe']}")
    print("[L2 功能链路]")
    for f in report["functions"]:
        mark = "✓" if f["status"] in ("success",) else "⚠" if f["status"] in (
            "NEEDS_DATA",) else "✗"
        extra = ""
        if "submit_ms" in f:
            extra += f" 提交={ms(f['submit_ms'] / 1000)}"
        if f.get("error"):
            extra += f" {str(f['error'])[:100]}"
        print(f"  {mark} {f['case']}: {f['status']}{extra}")
    print("[L3 混元质量]")
    for j in report["judgments"]:
        s = j["score"]
        if s < 0:
            mark, label = "⚠", "JUDGE_SKIPPED"
        else:
            mark = "✓" if s >= HUNYUAN_EVAL_PASS_THRESHOLD else "✗"
            label = f"score={s:.2f} (阈值 {HUNYUAN_EVAL_PASS_THRESHOLD})"
        print(f"  {mark} {j['case']} [{j['family']}]: {label}")
    if report["judgments"]:
        scores = [j["score"] for j in report["judgments"] if j["score"] >= 0]
        if scores:
            print(f"  → 被评 {len(scores)} 个产物平均分 = {sum(scores) / len(scores):.2f}")
    print(f"\n[结论] {'PASS' if report['pass'] else 'FAIL'}")
    for reason in report["reasons"]:
        print(f"  - {reason}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="沉浸式 AI 会话域 scholar-admin 接口混元评估（L0 环境 → L1 可用性 → L2 功能 → L3 混元质量）"
    )
    parser.add_argument("--base-url", default=None, help="服务地址（默认 http://127.0.0.1:{port}）")
    parser.add_argument("--port", type=int, default=PORT or 8080,
                        help="本地服务端口（服务未启动时自动拉起，默认取 config.PORT / 8080）")
    parser.add_argument("--scholar-id", default=os.environ.get("EVAL_SCHOLAR_ID", ""),
                        help="被评估学者 ID（需有已学句数据；缺省仅跑可用性门禁与 NEEDS_DATA 提示）")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="轮询间隔秒（默认 1s）")
    parser.add_argument("--max-wait", type=float, default=300,
                        help="等待终态上限秒（默认 300；LLM 生成任务真实耗时可达数分钟）")
    parser.add_argument("--no-live", action="store_true", help="跳过 L2 功能链路（只跑可用性门禁）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 L3 混元质量评分")
    parser.add_argument("--judge-limit", type=int, default=8, help="混元 Judge 最多调用数（默认 8）")
    parser.add_argument("--require-ai-session-v2", action="store_true",
                        help="v2 草案接口必须注册可用（落地后 CI 门禁用）")
    parser.add_argument("--strict", action="store_true", help="将 NEEDS_DATA/JUDGE_SKIPPED 视为失败")
    parser.add_argument("--report-path", type=Path,
                        default=Path("ai_session_eval_report.json"), help="JSON 报告输出路径")
    parser.add_argument("--no-autostart", action="store_true", help="服务不可达时不自动拉起")
    parser.add_argument("--keep-server", action="store_true", help="自拉起的服务测试完不关闭")
    args = parser.parse_args()

    print("===== [L0 环境门禁] =====")
    print(f"  AUTH_MODE={AUTH_MODE or 'dev(默认，无需鉴权头)'}")
    hunyuan_ok = bool(HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL)
    volcano_ok = bool(VOLCANO_API_KEY and VOLCANO_CHAT_MODEL)
    print(f"  混元 Judge: HUNYUAN_EVAL_MODEL={HUNYUAN_EVAL_MODEL or '(未配置)'} | "
          f"SECRET_KEY={'已配置' if HUNYUAN_SECRET_KEY else '未配置'} | "
          f"PASS_THRESHOLD={HUNYUAN_EVAL_PASS_THRESHOLD}")
    print(f"  火山生成: VOLCANO_CHAT_MODEL={'已配置' if volcano_ok else '未配置'}")
    if not hunyuan_ok and not args.no_judge:
        print("[WARN] 混元凭据未配置（HUNYUAN_SECRET_KEY/HUNYUAN_EVAL_MODEL），L3 将全部 JUDGE_SKIPPED")
    elif not args.no_judge:
        # 凭据存在但网关/密钥不匹配：默认值回落会让 L3 实际全部跳过，提前指出配置方向
        if HUNYUAN_BASE_URL.rstrip("/").endswith("hunyuan.tencentcloudapi.com"):
            print("[WARN] HUNYUAN_BASE_URL 仍为腾讯云 OpenAPI 默认值（hunyuan.tencentcloudapi.com）："
                  "该端点回 HTTP 200 + 错误 JSON（无 choices），不兼容 Bearer /chat/completions 调用，"
                  "L3 将全部 JUDGE_SKIPPED。请在 .env 设 HUNYUAN_BASE_URL=https://api.hunyuan.cloud.tencent.com/v1")
        if not os.environ.get("HUNYUAN_SECRET_KEY") and os.environ.get("TENCENTCLOUD_SECRETKEY"):
            print("[WARN] HUNYUAN_SECRET_KEY 未显式配置（回落 TENCENTCLOUD_SECRETKEY）："
                  "Bearer 鉴权需混元控制台 API 密钥，云 SecretKey 不可用于该网关，L3 将 401/403")
    if not args.scholar_id and not args.no_live:
        print("[WARN] 未指定 --scholar-id，L2 功能链路将标记 NEEDS_DATA（可用性门禁不受影响）")

    report: dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "judge_model": HUNYUAN_EVAL_MODEL,
            "pass_threshold": HUNYUAN_EVAL_PASS_THRESHOLD,
            "scholar_id": args.scholar_id or "(未指定)",
            "source_docs": [
                "scholar-skill/docs_v1/沉浸式学习页中的AI会话功能优化.md",
                "scholar-skill/docs_v1/沉浸式AI会话域接口混元评估方案.md",
            ],
        },
        "availability": [],
        "functions": [],
        "judgments": [],
        "summary": {},
        "pass": False,
        "reasons": [],
    }

    # 服务可达（自启）
    port = args.port
    base_url = args.base_url or f"http://127.0.0.1:{port}"
    proc = ensure_server(base_url, port, autostart=not args.no_autostart)
    reasons: list[str] = []
    try:
        with httpx.Client(timeout=30.0) as client:
            # L1 可用性门禁
            availability = run_availability(client, base_url)
            report["availability"] = availability
            implemented = [r for r in availability if r["status"] == "implemented"]
            drafts = [r for r in availability if r["status"] == "draft"]
            if any(not r["available"] for r in implemented):
                for r in implemented:
                    if not r["available"]:
                        reasons.append(f"已实现接口不可用: {r['id']} -> {r['probe']} {r['detail']}")
            if args.require_ai_session_v2 and any(not r["available"] for r in drafts):
                for r in drafts:
                    if not r["available"]:
                        reasons.append(f"v2 草案接口未就绪（--require-ai-session-v2 已开启）: {r['id']}")
            elif not args.require_ai_session_v2 and drafts and any(not r["available"] for r in drafts):
                print("\n[L1] 提示：/ai/session/v2* 为 docs_v1 §3.3 草案，尚未实现（预期状态，不阻断）。"
                      "契约落 docs_v2 并实现后，请加 --require-ai-session-v2 验收。")

            # L2 功能链路（仅当学者 ID 给定；可用性 READY 的已实现接口才跑）
            if not args.no_live and args.scholar_id:
                ai_v2_ready = all(
                    any(r["id"] == i and r["available"] for r in availability)
                    for i in ("ai_session_v2_submit", "ai_session_v2_task")
                )
                if ai_v2_ready:
                    report["functions"].extend(run_ai_session_v2_flow(
                        client, base_url, poll_interval=args.poll_interval,
                        max_wait=args.max_wait))
                if any(r["id"] == "eval_translate_v2" and r["available"] for r in availability):
                    report["functions"].append(run_eval_v2_flow(
                        client, base_url, poll_interval=args.poll_interval,
                        max_wait=args.max_wait))
                for use_async in (True, False):
                    key = "match_dialogue_task" if use_async else "match_dialogue"
                    if any(r["id"] == key and r["available"] for r in availability):
                        report["functions"].append(run_match_dialogue_flow(
                            client, base_url, args.scholar_id,
                            poll_interval=args.poll_interval, max_wait=args.max_wait,
                            use_async=use_async))
                conv_ready = all(
                    any(r["id"] == i and r["available"] for r in availability)
                    for i in ("conversation_scenario", "conversation_turn", "conversation_history")
                )
                if conv_ready:
                    report["functions"].extend(run_conversation_flow(
                        client, base_url, args.scholar_id,
                        poll_interval=args.poll_interval, max_wait=args.max_wait))
            elif not args.no_live and not args.scholar_id:
                reasons.append("未指定 --scholar-id，L2 功能链路未运行（可用性门禁已通过）")
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            logger.info("已关闭自动拉起的本地服务")

    # L2 判定：success 之外按严重级分类
    needs_data = [f for f in report["functions"] if f["status"] == "NEEDS_DATA"]
    biz_or_req_fail = [f for f in report["functions"]
                       if f["status"] not in ("success", "NEEDS_DATA")]
    if biz_or_req_fail:
        for f in biz_or_req_fail:
            reasons.append(f"功能链路失败: {f['case']} -> {f.get('status')} {f.get('error', '')[:120]}")
    if args.strict and needs_data:
        for f in needs_data:
            reasons.append(f"[strict] NEEDS_DATA 视为失败: {f['case']}")

    # L3 混元质量
    if not args.no_judge:
        artifacts = build_artifacts(report["functions"])
        if not artifacts:
            if args.scholar_id:
                print("\n[L3] 无可用产物（L2 均未产出 AI 文本），跳过评分")
        else:
            sem = asyncio.Semaphore(2)
            tasks = [
                _judge_one(a, a["family"], a["interface_label"], sem)
                for a in artifacts[: args.judge_limit]
            ]
            results = asyncio.run(_judge_all(tasks))
            for artifact, judged in zip(artifacts[: args.judge_limit], results):
                report["judgments"].append({
                    "case": artifact["case"],
                    "family": artifact["family"],
                    "interface": artifact["interface_label"],
                    **judged,
                })
            judged_scores = [j["score"] for j in report["judgments"] if j["score"] >= 0]
            skipped = [j for j in report["judgments"] if j["score"] < 0]
            low = [j for j in report["judgments"]
                   if 0 <= j["score"] < HUNYUAN_EVAL_PASS_THRESHOLD]
            for j in low:
                reasons.append(
                    f"混元评分未达标: {j['case']} score={j['score']:.2f} "
                    f"< {HUNYUAN_EVAL_PASS_THRESHOLD} | {str(j['feedback'])[:120]}")
            if args.strict and skipped:
                reasons.append(f"[strict] JUDGE_SKIPPED 视为失败: {len(skipped)} 个产物")
            elif skipped and not judged_scores:
                first_fb = str(skipped[0].get("feedback") or "")[:160]
                reasons.append(
                    "混元 Judge 全部跳过（凭据/网关/超时），请人工复核或检查 HUNYUAN_* 配置"
                    + (f"；首个原因: {first_fb}" if first_fb else "")
                )

    # 汇总判定
    implemented = [r for r in report["availability"] if r["status"] == "implemented"]
    drafts = [r for r in report["availability"] if r["status"] == "draft"]
    report["summary"] = {
        "availability_ok": bool(implemented)
        and len(implemented) == sum(1 for r in implemented if r["available"]),
        "implemented_count": len(implemented),
        "draft_count": len(drafts),
        "function_ok": len(biz_or_req_fail) == 0,
        "needs_data_count": len(needs_data),
        "judge_count": len(report["judgments"]),
        "judge_pass_count": sum(
            1 for j in report["judgments"]
            if 0 <= j["score"] and j["score"] >= HUNYUAN_EVAL_PASS_THRESHOLD),
    }
    # exit code 判定
    report["pass"] = not reasons
    report["reasons"] = reasons
    # 文件输出
    out_path = args.report_path
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[报告] 已写入 {out_path}")

    print_report(report)

    if report["pass"]:
        sys.exit(0)
    if not (HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL) and not args.no_judge:
        sys.exit(2)  # Judge 环境缺失
    sys.exit(1)


if __name__ == "__main__":
    main()
