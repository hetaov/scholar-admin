#!/usr/bin/env python3
"""AI 对话生成域（批量面 `/ai/dialogue/v1`）端到端冒烟 + 回归门禁脚本（设计稿 T6 / §8.3 / §8.4）。

配套设计稿：`scholar-skill/docs_v1/AI会话/AI英语对话生成设计.md`
范式对齐：
- `scripts/ai_session_eval.py`（L0~L3 分层门禁 + 退出码 0/1/2 + `--strict`）
- `scripts/translation_v2_verify.py`（自启 uvicorn + 提交/轮询耗时打印 + 产物落盘）

分层门禁：
  L0 环境门禁   —— 服务可达（可自启 uvicorn）、AUTH_MODE、火山/混元凭据、本地 NC2 语料
  L1 可用性门禁 —— `/openapi.json` 契约探针（6 个 `/ai/dialogue/v1/*` 路由注册 + 错误码契约，
                   零 LLM 成本；契约缺失 → 退出码 2）
  L2 功能链路   —— `GET /corpus` → `POST /generate` → 轮询终态 → `/checkpoints` →
                   `/user-input` →（failed 时）`/resume`；断言 §8.4 功能口径；
                   产物落盘 `data/nc2/generated/*.json` + 汇总指标
  L3 混元质量   —— LLM-as-Judge（Judge≠Generator：生成=火山方舟，评分=混元），
                   按 §8.4 质量口径打分（0~1，阈值 `HUNYUAN_EVAL_PASS_THRESHOLD` 默认 0.7）

用法::

  # 服务未启动：脚本自动拉起 uvicorn（并强制开启批量生成开关 + 图 + 断点）
  python scripts/ai_dialogue_gen_eval.py

  # 指向已运行的服务（不自动拉起）
  python scripts/ai_dialogue_gen_eval.py --port 8080 --no-autostart

  # 只跑 L0+L1 可用性门禁（秒级、零 LLM 成本；可进 CI 前置）
  python scripts/ai_dialogue_gen_eval.py --no-live --no-judge

  # 全链路 + 混元质量评分（生成=火山，评分=混元）
  python scripts/ai_dialogue_gen_eval.py --judge-limit 2

  # CI 门禁：WARN 也视为失败（覆盖降级/召回不足/Judge 跳过/无断点均须为 0）
  python scripts/ai_dialogue_gen_eval.py --strict

退出码：0=通过；1=门禁失败；2=契约缺失 / Judge 环境缺失（与 `ai_session_eval.py` 同口径）。
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
    DIALOGUE_CORPUS_DIR,
    DIALOGUE_GEN_CHECKPOINT_ENABLED,
    DIALOGUE_GEN_ENABLED,
    DIALOGUE_GEN_GRAPH_ENABLED,
    DIALOGUE_GEN_MAX_RETRY,
    DIALOGUE_LOCAL_SCHOLAR_ID,
    HUNYUAN_BASE_URL,
    HUNYUAN_EVAL_MODEL,
    HUNYUAN_EVAL_PASS_THRESHOLD,
    HUNYUAN_SECRET_KEY,
    HUNYUAN_TIMEOUT_SECONDS,
    PORT,
    VOLCANO_API_KEY,
    VOLCANO_CHAT_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_dialogue_gen_eval")

POLL_TERMINAL = {"success", "failed"}
MISSING_ID = "__eval_missing__"

# 生成形态白名单（与 services/providers/dialogue_gen.py 对齐；脚本不 import services，避免重依赖）
CONTENT_TYPES = ("dialogue", "non_dialogue")
NON_DIALOGUE_SUB_TYPES = ("retell", "fill", "task")
PROMPT_LANGS = ("zh", "en")
RECALL_MIN, RECALL_MAX = 2, 6
NOTES_RECALL_INSUFFICIENT = "recall_insufficient"
NOTES_COVERAGE_INCOMPLETE = "coverage_incomplete"
NOTES_FALLBACK = "fallback_non_dialogue"
NOTES_RESUME_WITHOUT_CHECKPOINT = "resume_without_checkpoint"

# ---------------------------------------------------------------------------
# 被评接口注册表（对齐设计稿 §5）
#   status: implemented=已实现（可用性必需） / draft=设计稿草案（默认不阻断）
# ---------------------------------------------------------------------------

_GENERATE_PATH = "/ai/dialogue/v1/generate"
_TASK_PATH = "/ai/dialogue/v1/task/{task_id}"

INTERFACES = [
    {
        "id": "dialogue_gen_submit",
        "label": "提交批量对话生成（§5 ①）",
        "method": "POST",
        "path": _GENERATE_PATH,
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_submit_guard",
        "label": "提交护栏 · task_group 必填（TASK_GROUP_REQUIRED）",
        "method": "POST",
        "path": _GENERATE_PATH,
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_type_guard",
        "label": "提交护栏 · preferred_type 白名单（TYPE_NOT_SUPPORTED）",
        "method": "POST",
        "path": _GENERATE_PATH,
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_task",
        "label": "查询生成任务（§5 ②）",
        "method": "GET",
        "path": _TASK_PATH,
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_checkpoints",
        "label": "断点轨迹（§5 ②'，T4）",
        "method": "GET",
        "path": "/ai/dialogue/v1/task/{task_id}/checkpoints",
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_resume",
        "label": "断点续写（§5 ③，T4）",
        "method": "POST",
        "path": "/ai/dialogue/v1/task/{task_id}/resume",
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_user_input",
        "label": "用户作答评测（§5 ④，T5）",
        "method": "POST",
        "path": "/ai/dialogue/v1/task/{task_id}/user-input",
        "status": "implemented",
    },
    {
        "id": "dialogue_gen_corpus",
        "label": "本地 NC2 语料（§3.2，T2）",
        "method": "GET",
        "path": "/ai/dialogue/v1/corpus",
        "status": "implemented",
    },
]

# 探针用最小合法任务组（不落库：护栏探针在 TASK_GROUP_REQUIRED/TYPE_NOT_SUPPORTED 处短路）
_PROBE_GROUP = {
    "lesson_id": "l_nc2_01",
    "group_id": "g_nc2_01_a",
    "group_label": "eval probe",
    "sentences": [{"sentence_id": "s_eval_probe", "content": "It is only a probe."}],
}

# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def ms(sec: float) -> str:
    return f"{sec * 1000:.0f}ms" if sec < 1 else f"{sec:.2f}s"


def pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _safe_json(resp: httpx.Response) -> dict:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}


def resolve_corpus_dir() -> Path:
    path = Path(DIALOGUE_CORPUS_DIR)
    return path if path.is_absolute() else HERE / path


def corpus_file_path() -> Path:
    return resolve_corpus_dir() / "corpus.json"


def resolve_generated_dir(override: str | None) -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else HERE / path
    return resolve_corpus_dir() / "generated"


def ensure_server(
    base_url: str,
    port: int,
    autostart: bool,
    *,
    app: str = "main:app",
    server_env: dict | None = None,
) -> subprocess.Popen | None:
    """确认服务可达；不可达且允许自启则拉起 uvicorn（注入批量生成开关），返回子进程句柄。"""
    try:
        r = httpx.get(f"{base_url}/health", timeout=3)
        if r.status_code == 200:
            logger.info(f"服务已就绪: {base_url}/health")
            return None
    except Exception:  # noqa: BLE001
        pass
    if not autostart:
        sys.exit(
            f"[ERROR] {base_url} 不可达。请先运行 `python main.py`，"
            f"或去掉 --no-autostart 让脚本自动拉起服务。"
        )
    env = dict(os.environ)
    env.update(server_env or {})
    logger.info(
        f"服务未启动，自动拉起: python -m uvicorn {app} --port {port} "
        f"(DIALOGUE_GEN_ENABLED={env.get('DIALOGUE_GEN_ENABLED')}, "
        f"GRAPH={env.get('DIALOGUE_GEN_GRAPH_ENABLED')}, "
        f"CHECKPOINT={env.get('DIALOGUE_GEN_CHECKPOINT_ENABLED')})"
    )
    log_path = Path("/tmp/scholar_ai_dialogue_gen_eval_server.log")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            app,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(HERE),
        env=env,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):  # 最多等 30s
        if proc.poll() is not None:
            sys.exit(
                f"[ERROR] uvicorn 启动失败（退出码 {proc.returncode}），日志见 {log_path}"
            )
        try:
            r = httpx.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                logger.info(f"服务已拉起: {base_url}（日志: {log_path}）")
                return proc
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    proc.kill()
    sys.exit(f"[ERROR] uvicorn 30s 内未就绪，日志见 {log_path}")


# ---------------------------------------------------------------------------
# L1 可用性门禁：openapi 注册 + 契约探针（无 LLM/DB 成本）
# ---------------------------------------------------------------------------


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
    if (method.upper(), norm) in routes:
        return True
    return any(m == method.upper() and p == norm for m, p in routes)


def _probe_verdict(ok: bool, resp: httpx.Response, body: dict) -> dict:
    code = body.get("code")
    detail = f"HTTP {resp.status_code}" + (f" code={code}" if code else "")
    return {"probe_status": "OK" if ok else "CONTRACT_ERR", "detail": detail}


def probe_endpoint(client: httpx.Client, base_url: str, entry: dict) -> dict:
    """对单个接口做注册 + 契约探针。返回 {probe_status, detail}。"""
    if entry["status"] == "draft":
        return {"probe_status": "NOT_REGISTERED", "detail": "草案接口（未实现）"}

    routes = load_openapi(client, base_url)
    if not path_is_registered(routes, entry["method"], entry["path"]):
        return {"probe_status": "NOT_REGISTERED", "detail": "openapi 中无此路由"}

    iid = entry["id"]
    path = entry["path"]
    try:
        if iid == "dialogue_gen_submit":
            # 契约 §5：字段可选 + 手动校验 → {} → 200 + success=false + INVALID_INPUT
            resp = client.post(f"{base_url}{path}", json={}, timeout=10)
            body = _safe_json(resp)
            ok = (
                resp.status_code == 200
                and body.get("success") is False
                and body.get("code") == "INVALID_INPUT"
            )
            return _probe_verdict(ok, resp, body)
        if iid == "dialogue_gen_submit_guard":
            resp = client.post(f"{base_url}{path}", json={"scholar_id": "eval_probe"}, timeout=10)
            body = _safe_json(resp)
            ok = body.get("success") is False and body.get("code") == "TASK_GROUP_REQUIRED"
            return _probe_verdict(ok, resp, body)
        if iid == "dialogue_gen_type_guard":
            resp = client.post(
                f"{base_url}{path}",
                json={
                    "scholar_id": "eval_probe",
                    "task_group": _PROBE_GROUP,
                    "preferred_type": "retell",
                },
                timeout=10,
            )
            body = _safe_json(resp)
            ok = body.get("success") is False and body.get("code") == "TYPE_NOT_SUPPORTED"
            return _probe_verdict(ok, resp, body)
        if iid in ("dialogue_gen_task", "dialogue_gen_checkpoints"):
            resp = client.get(f"{base_url}{path.format(task_id=MISSING_ID)}", timeout=10)
            ok = resp.status_code == 404  # 任务不存在/过期 → 404
            return _probe_verdict(ok, resp, _safe_json(resp))
        if iid == "dialogue_gen_resume":
            resp = client.post(
                f"{base_url}{path.format(task_id=MISSING_ID)}", json={}, timeout=10
            )
            ok = resp.status_code == 404
            return _probe_verdict(ok, resp, _safe_json(resp))
        if iid == "dialogue_gen_user_input":
            # 空 text 在任务查询前短路 → INVALID_INPUT（零 DB 成本）
            resp = client.post(
                f"{base_url}{path.format(task_id=MISSING_ID)}", json={}, timeout=10
            )
            body = _safe_json(resp)
            ok = body.get("success") is False and body.get("code") == "INVALID_INPUT"
            return _probe_verdict(ok, resp, body)
        if iid == "dialogue_gen_corpus":
            resp = client.get(f"{base_url}{path}", timeout=10)
            body = _safe_json(resp)
            ok = body.get("success") is True and isinstance(
                (body.get("data") or {}).get("corpus_available"), bool
            )
            return _probe_verdict(ok, resp, body)
    except Exception as e:  # noqa: BLE001
        return {"probe_status": "CONTRACT_ERR", "detail": f"{type(e).__name__}: {e}"}
    return {"probe_status": "CONTRACT_ERR", "detail": "未定义探针"}


def run_availability(client: httpx.Client, base_url: str) -> list[dict]:
    """L1：全部接口的注册 + 探针结果矩阵。"""
    print("\n===== [L1 可用性门禁] 接口注册 + 契约探针（零 LLM 成本） =====")
    results: list[dict] = []
    for entry in INTERFACES:
        if entry["status"] == "draft":
            routes = load_openapi(client, base_url)
            registered = path_is_registered(routes, entry["method"], entry["path"])
            probe = "READY" if registered else "NOT_REGISTERED"
            detail = "已在 openapi 注册" if registered else "未实现（草案）"
        else:
            probed = probe_endpoint(client, base_url, entry)
            probe = probed["probe_status"]
            detail = probed.get("detail", "")
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
        mark = "✓" if available else ("⚠(草案)" if entry["status"] == "draft" else "✗")
        print(f"  {mark} [{entry['status']:<11}] {entry['label']}: {probe} {detail}")
    return results


# ---------------------------------------------------------------------------
# L2 功能链路：语料 → 提交 → 轮询 → 断点 → 用户作答 →（failed 时）续写
# ---------------------------------------------------------------------------


def poll_task(
    client: httpx.Client,
    base_url: str,
    task_id: str,
    *,
    poll_interval: float,
    max_wait: float,
) -> dict | None:
    """通用轮询：直到 success/failed 或超时。返回 {status,result,error,resumable,polls}。"""
    poll_start = time.perf_counter()
    polls = 0
    last_status = "pending"
    data: dict = {}
    while True:
        if time.perf_counter() - poll_start > max_wait:
            print(f"    [轮询] 超过 {ms(max_wait)} 仍未终态（status={last_status}），放弃")
            return None
        time.sleep(poll_interval)
        polls += 1
        t1 = time.perf_counter()
        try:
            r = client.get(f"{base_url}/ai/dialogue/v1/task/{task_id}", timeout=20)
        except Exception as e:  # noqa: BLE001
            print(f"    [轮询] 请求异常: {type(e).__name__}: {e}")
            return None
        dt = time.perf_counter() - t1
        if r.status_code == 404:
            print("    [轮询] 404：任务不存在/已过期")
            return None
        body = _safe_json(r)
        data = body.get("data") or {}
        last_status = str(data.get("status") or "").lower()
        extra = ""
        if data.get("error"):
            extra = f" error={str(data['error'])[:80]}"
        elif data.get("result"):
            result = data["result"] or {}
            extra = (
                f" content_type={result.get('content_type')} "
                f"coverage={json.dumps(result.get('coverage'), ensure_ascii=False)}"
            )
        print(
            f"    [轮询] #{polls} 耗时={ms(dt)} status={last_status} "
            f"checkpoint_id={data.get('checkpoint_id')}{extra}"
        )
        if last_status in POLL_TERMINAL:
            break
    return {
        "status": last_status,
        "result": data.get("result"),
        "error": data.get("error"),
        "resumable": bool(data.get("resumable")),
        "checkpoint_id": data.get("checkpoint_id"),
        "polls": polls,
    }


def fetch_corpus(client: httpx.Client, base_url: str, scholar_id: str) -> dict:
    """GET /ai/dialogue/v1/corpus → {status, data|error}。"""
    try:
        resp = client.get(
            f"{base_url}/ai/dialogue/v1/corpus",
            params={"scholar_id": scholar_id},
            timeout=20,
        )
    except Exception as e:  # noqa: BLE001
        return {"status": "REQUEST_FAIL", "error": str(e)}
    body = _safe_json(resp)
    if not body.get("success"):
        return {
            "status": "BIZ_FAIL",
            "error": f"{body.get('code')} {body.get('message')}",
        }
    return {"status": "success", "data": body.get("data") or {}}


def pick_task_group(
    corpus_data: dict, *, group_index: int | None = None, lesson_id: str | None = None
) -> dict | None:
    """选任务组：显式序号/课号优先；缺省取句子最多的组（覆盖信号最强，确定性）。"""
    groups = [g for g in (corpus_data.get("task_groups") or []) if g.get("sentences")]
    if not groups:
        return None
    if lesson_id:
        filtered = [g for g in groups if str(g.get("lesson_id")) == str(lesson_id)]
        groups = filtered or groups
    if group_index is not None:
        return groups[group_index % len(groups)]
    return max(groups, key=lambda g: (len(g.get("sentences") or []), str(g.get("group_id") or "")))


def build_generate_payload(
    corpus_data: dict,
    scholar_id: str,
    *,
    prompt_lang: str,
    recall_enabled: bool,
    top_k: int,
    group_index: int | None,
    lesson_id: str | None,
) -> tuple[dict | None, dict | None]:
    """由语料任务组 + 本地学者指标构造 §5 提交入参（返回 payload, group）。"""
    group = pick_task_group(corpus_data, group_index=group_index, lesson_id=lesson_id)
    if group is None:
        return None, None
    label = str(group.get("group_label") or group.get("group_id") or "").strip()
    learner = corpus_data.get("learner") or {}
    payload = {
        "scholar_id": scholar_id,
        "task_group": group,
        "scenario": {
            "background": f"围绕「{label}」的日常对话场景，两位角色自然交谈，把任务组句子用进对话。",
            "goal": f"自然用出「{label}」中的句子",
        },
        "roles": [
            {"code": "A", "name": "Tom", "identity": "friend, warm and talkative"},
            {"code": "B", "name": "Anna", "identity": "classmate, curious"},
        ],
        "recall": {"enabled": bool(recall_enabled), "top_k": int(top_k)},
        "metrics": {
            "enabled": True,
            "weak_skills": list(learner.get("weak_skills") or [])[:5],
        },
        "prompt_lang": prompt_lang,
        "preferred_type": "auto",
    }
    return payload, group


def _notes_of(result: dict) -> str:
    return str((result or {}).get("notes") or "")


def evaluate_result_checks(
    result: dict,
    *,
    prompt_lang: str,
    recall_enabled: bool,
    submit_ms: float,
    submit_budget_ms: float,
    checkpoint_enabled: bool,
    checkpoints_count: int,
) -> list[dict]:
    """按 §8.4 功能口径逐项断言生成结果（PASS/FAIL/WARN/SKIP）。"""
    checks: list[dict] = []
    result = result or {}
    notes = _notes_of(result)
    coverage = result.get("coverage") or {}
    required_total = int(coverage.get("required_total") or 0)
    required_used = int(coverage.get("required_used") or 0)

    # 3 必用任务组
    if required_total == 0:
        checks.append({"id": "coverage", "status": "FAIL", "detail": "coverage 缺失/无必用句"})
    elif required_used >= required_total:
        checks.append({
            "id": "coverage",
            "status": "PASS",
            "detail": f"required_used={required_used}/{required_total}（全覆盖）",
        })
    elif NOTES_FALLBACK in notes or NOTES_COVERAGE_INCOMPLETE in notes:
        checks.append({
            "id": "coverage",
            "status": "WARN",
            "detail": f"required_used={required_used}/{required_total}，降级留痕 notes={notes}",
        })
    else:
        checks.append({
            "id": "coverage",
            "status": "FAIL",
            "detail": f"required_used={required_used}/{required_total} 且无降级留痕",
        })

    # 4 召回 2-6
    recalled = result.get("recalled_sentence_ids") or []
    if not recall_enabled:
        checks.append({"id": "recall", "status": "SKIP", "detail": "recall.enabled=false（按请求未开启）"})
    elif RECALL_MIN <= len(recalled) <= RECALL_MAX:
        checks.append({"id": "recall", "status": "PASS", "detail": f"条数={len(recalled)}∈[{RECALL_MIN},{RECALL_MAX}]"})
    elif NOTES_RECALL_INSUFFICIENT in notes:
        checks.append({"id": "recall", "status": "WARN", "detail": f"条数={len(recalled)} 不足，留痕 recall_insufficient"})
    else:
        checks.append({"id": "recall", "status": "FAIL", "detail": f"条数={len(recalled)} 越界且无留痕"})

    # 5 形态
    content_type = str(result.get("content_type") or "")
    sub_type = result.get("sub_type")
    if content_type not in CONTENT_TYPES:
        checks.append({"id": "form", "status": "FAIL", "detail": f"content_type={content_type!r} 非法"})
    elif content_type == "non_dialogue" and str(sub_type) not in NON_DIALOGUE_SUB_TYPES:
        checks.append({"id": "form", "status": "FAIL", "detail": f"non_dialogue 的 sub_type={sub_type!r} 非法"})
    else:
        checks.append({"id": "form", "status": "PASS", "detail": f"content_type={content_type}, sub_type={sub_type}"})

    # 6 提示语言
    prompts = result.get("prompts") or []
    if not prompts:
        checks.append({"id": "prompt_lang", "status": "WARN", "detail": "无 prompts（可接受，缺样例）"})
    else:
        bad = [p for p in prompts if str((p or {}).get("lang")) != prompt_lang]
        checks.append({
            "id": "prompt_lang",
            "status": "FAIL" if bad else "PASS",
            "detail": f"prompts={len(prompts)} 条，lang 期望={prompt_lang}" + (f"，异常 {len(bad)} 条" if bad else ""),
        })

    # 7 异步契约：提交毫秒级返回
    if submit_ms <= submit_budget_ms:
        checks.append({"id": "async", "status": "PASS", "detail": f"提交耗时={ms(submit_ms / 1000)}≤{submit_budget_ms:.0f}ms"})
    else:
        checks.append({"id": "async", "status": "FAIL", "detail": f"提交耗时={ms(submit_ms / 1000)}>{submit_budget_ms:.0f}ms"})

    # 指标
    metrics = result.get("metrics") or {}
    if metrics.get("score") is not None and metrics.get("level"):
        checks.append({"id": "metrics", "status": "PASS", "detail": f"metrics.level={metrics.get('level')} score={metrics.get('score')}"})
    else:
        checks.append({"id": "metrics", "status": "FAIL", "detail": "result.metrics 缺 score/level"})

    # 10 断点续写（仅观测 checkpoint 轨迹）
    if checkpoints_count > 0:
        checks.append({"id": "checkpoints", "status": "PASS", "detail": f"checkpoint 轨迹 {checkpoints_count} 条"})
    elif checkpoint_enabled:
        checks.append({"id": "checkpoints", "status": "WARN", "detail": "断点开关开启但轨迹为空"})
    else:
        checks.append({"id": "checkpoints", "status": "SKIP", "detail": "DIALOGUE_GEN_CHECKPOINT_ENABLED=0（未接入断点）"})
    return checks


def run_generate_flow(
    client: httpx.Client,
    base_url: str,
    *,
    scholar_id: str,
    prompt_lang: str,
    recall_enabled: bool,
    top_k: int,
    group_index: int | None,
    lesson_id: str | None,
    poll_interval: float,
    max_wait: float,
    submit_budget_ms: float,
    checkpoint_enabled: bool,
    try_resume: bool,
) -> dict:
    """L2 主链路：语料 → 提交 → 轮询 → 断点轨迹 → 用户作答 →（failed 时）续写。"""
    flow: dict = {
        "corpus": None,
        "payload": None,
        "submit": None,
        "task": None,
        "checkpoints": [],
        "user_input": None,
        "resume": None,
        "checks": [],
    }

    print("\n----- [L2/A] 本地语料 GET /ai/dialogue/v1/corpus -----")
    flow["corpus"] = fetch_corpus(client, base_url, scholar_id)
    corpus = flow["corpus"]
    if corpus["status"] != "success":
        print(f"  ✗ 语料读取失败: {corpus.get('error')}")
        return flow
    data = corpus["data"]
    lessons = data.get("lessons") or []
    print(
        f"  ✓ corpus_available={data.get('corpus_available')} book={data.get('book')} "
        f"lessons={len(lessons)} task_groups={len(data.get('task_groups') or [])} "
        f"scholar={data.get('scholar_id')}"
    )

    payload, group = build_generate_payload(
        data,
        scholar_id,
        prompt_lang=prompt_lang,
        recall_enabled=recall_enabled,
        top_k=top_k,
        group_index=group_index,
        lesson_id=lesson_id,
    )
    if payload is None:
        print("  ✗ 语料无可选任务组")
        return flow
    flow["payload"] = payload
    print(
        f"  任务组: {group.get('lesson_id')}/{group.get('group_id')} "
        f"「{group.get('group_label')}」 sentences={len(group.get('sentences') or [])}"
    )
    print(
        f"  recall={payload['recall']} metrics.weak_skills={payload['metrics']['weak_skills']} "
        f"prompt_lang={prompt_lang} top_role_count={len(payload['roles'])}"
    )

    print("\n----- [L2/B] 提交 POST /ai/dialogue/v1/generate（应毫秒级返回） -----")
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}{_GENERATE_PATH}", json=payload, timeout=20)
    except Exception as e:  # noqa: BLE001
        flow["submit"] = {"status": "REQUEST_FAIL", "error": str(e)}
        print(f"  ✗ 提交异常: {e}")
        return flow
    submit_ms = (time.perf_counter() - t0) * 1000
    body = _safe_json(resp)
    if not body.get("success"):
        flow["submit"] = {
            "status": "BIZ_FAIL",
            "error": f"{body.get('code')} {body.get('message')}",
            "submit_ms": submit_ms,
        }
        print(f"  ✗ 提交业务失败: {body.get('code')} {body.get('message')}（耗时 {ms(submit_ms / 1000)}）")
        return flow
    task_data = body.get("data") or {}
    task_id = task_data.get("task_id")
    flow["submit"] = {
        "status": "success",
        "task_id": task_id,
        "task_status": task_data.get("status"),
        "submit_ms": submit_ms,
    }
    print(f"  ✓ HTTP {resp.status_code} 提交耗时={ms(submit_ms / 1000)} task_id={task_id} status={task_data.get('status')}")

    print("\n----- [L2/C] 轮询 GET /ai/dialogue/v1/task/{task_id} -----")
    polled = poll_task(
        client, base_url, task_id, poll_interval=poll_interval, max_wait=max_wait
    )
    if polled is None:
        flow["task"] = {"task_id": task_id, "status": "POLL_TIMEOUT"}
        return flow
    flow["task"] = {"task_id": task_id, **polled}
    result = polled.get("result") or {}

    print("\n----- [L2/D] 断点轨迹 GET /ai/dialogue/v1/task/{task_id}/checkpoints -----")
    try:
        cresp = client.get(
            f"{base_url}/ai/dialogue/v1/task/{task_id}/checkpoints", timeout=20
        )
        cbody = _safe_json(cresp)
        checkpoints = (cbody.get("data") or {}).get("checkpoints") or []
    except Exception as e:  # noqa: BLE001
        checkpoints = []
        print(f"  ⚠ checkpoint 查询异常: {e}")
    flow["checkpoints"] = checkpoints
    if checkpoints:
        for cp in checkpoints:
            print(f"  · {cp.get('stage'):<16} retry={cp.get('retry_count')} cp={str(cp.get('checkpoint_id'))[:12]}")
    else:
        print("  （空轨迹：断点开关关闭或无 checkpoint）")

    if polled["status"] == "success" and result:
        print("\n----- [L2/E] 用户作答 POST /ai/dialogue/v1/task/{task_id}/user-input -----")
        target = ""
        for turn in reversed(result.get("turns") or []):
            if (turn or {}).get("target_sentence_id"):
                target = (turn or {}).get("text") or ""
                break
        if not target:
            sentences = (payload.get("task_group") or {}).get("sentences") or []
            target = str((sentences[-1] if sentences else {}).get("content") or "")
        try:
            uresp = client.post(
                f"{base_url}/ai/dialogue/v1/task/{task_id}/user-input",
                json={"text": target},
                timeout=120,
            )
            ubody = _safe_json(uresp)
            flow["user_input"] = {
                "status": "success" if ubody.get("success") else "BIZ_FAIL",
                "data": ubody.get("data"),
                "error": None if ubody.get("success") else f"{ubody.get('code')} {ubody.get('message')}",
            }
            if ubody.get("success"):
                d = ubody.get("data") or {}
                print(
                    f"  ✓ score={d.get('score')} meaningful={d.get('meaningful')} "
                    f"faithfulness={d.get('faithfulness')} confidence={d.get('confidence')} "
                    f"written_back={d.get('written_back')}"
                )
            else:
                print(f"  ✗ 业务失败: {ubody.get('code')} {ubody.get('message')}")
        except Exception as e:  # noqa: BLE001
            flow["user_input"] = {"status": "REQUEST_FAIL", "error": str(e)}
            print(f"  ✗ 异常: {e}")
    elif polled["status"] == "failed":
        print(f"  ⚠ 生成失败，跳过用户作答: error={polled.get('error')}")

    # failed 且有断点 → 续写（T4/T6）
    if try_resume and polled["status"] == "failed":
        print("\n----- [L2/F] 断点续写 POST /ai/dialogue/v1/task/{task_id}/resume -----")
        if not polled.get("resumable"):
            flow["resume"] = {"status": "SKIP", "detail": "resumable=false（无断点游标）"}
            print("  ⚠ 任务不可续写（resumable=false）")
        else:
            try:
                rresp = client.post(
                    f"{base_url}/ai/dialogue/v1/task/{task_id}/resume", json={}, timeout=20
                )
                rbody = _safe_json(rresp)
                if not rbody.get("success"):
                    flow["resume"] = {
                        "status": "BIZ_FAIL",
                        "error": f"{rbody.get('code')} {rbody.get('message')}",
                    }
                    print(f"  ✗ 续写业务失败: {rbody.get('code')} {rbody.get('message')}")
                else:
                    print(f"  ✓ 续写已调度 status={(rbody.get('data') or {}).get('status')}")
                    rpoll = poll_task(
                        client,
                        base_url,
                        task_id,
                        poll_interval=poll_interval,
                        max_wait=max_wait,
                    )
                    flow["resume"] = {
                        "status": (rpoll or {}).get("status") or "POLL_TIMEOUT",
                        "result": (rpoll or {}).get("result"),
                        "error": (rpoll or {}).get("error"),
                        "polls": (rpoll or {}).get("polls"),
                    }
                    flow["task"] = {"task_id": task_id, **(rpoll or {})}
                    if rpoll and rpoll.get("status") == "success":
                        flow["task"]["resumed"] = True
            except Exception as e:  # noqa: BLE001
                flow["resume"] = {"status": "REQUEST_FAIL", "error": str(e)}
                print(f"  ✗ 异常: {e}")

    # 逐项口径断言
    final_result = (flow.get("task") or {}).get("result") or result
    flow["checks"] = evaluate_result_checks(
        final_result,
        prompt_lang=prompt_lang,
        recall_enabled=recall_enabled,
        submit_ms=submit_ms,
        submit_budget_ms=submit_budget_ms,
        checkpoint_enabled=checkpoint_enabled,
        checkpoints_count=len(checkpoints),
    )
    print("\n----- [L2/G] §8.4 功能口径断言 -----")
    for c in flow["checks"]:
        mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "·"}.get(c["status"], "?")
        print(f"  {mark} [{c['status']}] {c['id']}: {c['detail']}")
    if flow.get("resume") and flow["resume"].get("status") == "success":
        print("  ✓ [resume] 续跑终态 success（retry_count 不重置见 checkpoint 轨迹）")
    return flow


# ---------------------------------------------------------------------------
# 产物落盘
# ---------------------------------------------------------------------------


def persist_artifacts(
    generated_dir: Path,
    flow: dict,
    *,
    scholar_id: str,
    meta: dict,
) -> Path:
    """落盘单次生成的完整产物 + 追加 index.json（对齐 §8.3「落盘 data/nc2/generated/*.json」）。"""
    generated_dir.mkdir(parents=True, exist_ok=True)
    task_id = ((flow.get("task") or {}).get("task_id")) or f"eval_{int(time.time())}"
    artifact = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scholar_id": scholar_id,
        "meta": meta,
        "request": flow.get("payload"),
        "submit": flow.get("submit"),
        "task": flow.get("task"),
        "checkpoints": flow.get("checkpoints"),
        "user_input": flow.get("user_input"),
        "resume": flow.get("resume"),
        "checks": flow.get("checks"),
    }
    path = generated_dir / f"{task_id}.json"
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")

    index_path = generated_dir / "index.json"
    index: list = []
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            index = loaded if isinstance(loaded, list) else []
        except Exception:  # noqa: BLE001
            index = []
    result = (flow.get("task") or {}).get("result") or {}
    index.append(
        {
            "task_id": task_id,
            "generated_at": artifact["generated_at"],
            "file": path.name,
            "status": (flow.get("task") or {}).get("status"),
            "content_type": result.get("content_type"),
            "coverage": result.get("coverage"),
            "recalled": len(result.get("recalled_sentence_ids") or []),
            "notes": result.get("notes"),
        }
    )
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# L3 混元 Judge（LLM-as-Judge，Judge≠Generator）
# ---------------------------------------------------------------------------

RUBRICS: dict[str, dict] = {
    "dialogue_gen": {
        "system": (
            "你是 scholar-admin「AI 对话生成域（批量生成面）」质量评审专家。"
            "内容由火山方舟生成，你仅做独立质量评分。只输出合法 JSON。"
        ),
        "dimensions": [
            ("任务组必用句覆盖", 0.25, "coverage.required_used/required_total 是否达 100%，或降级留痕合理（fallback_non_dialogue/coverage_incomplete）"),
            ("召回融合自然度", 0.15, "recalled 学习语句（若有）是否自然融入情境，不生硬堆砌、不喧宾夺主"),
            ("形态与场景适配", 0.15, "content_type/sub_type 是否贴合任务组；dialogue 为 A/B/C 多角色自然来往，non_dialogue 的 retell/fill/task 题面清晰"),
            ("提示引导性", 0.20, "prompts 是否引导学习者复述/用出目标句、不直接给整句答案，且语言与 prompt_lang 一致"),
            ("语言自然度与角色一致性", 0.15, "英文口语自然、语法正确，说话人符合 roles/背景设定"),
            ("结构化契约", 0.10, "turns/prompts/used_sentence_ids/recalled_sentence_ids/coverage 字段齐备且语义正确"),
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
    artifact: dict, sem: asyncio.Semaphore
) -> dict:
    """调混元评估单个产物。返回 {"score", "feedback", "issues", "dimensions"}。"""
    rubric = RUBRICS["dialogue_gen"]
    dim_lines = "\n".join(
        f"- {name}（权重 {w}）：{desc}" for name, w, desc in rubric["dimensions"]
    )
    prompt = _JUDGE_USER_TEMPLATE.format(
        interface_label=artifact.get("interface_label", ""),
        request_summary=artifact.get("request_summary", ""),
        output_json=json.dumps(artifact.get("output") or {}, ensure_ascii=False),
        dimensions=dim_lines,
    )
    try:
        async with sem:
            # 混元 OpenAI 兼容网关：不使用 response_format=json_object（对推理模型可能 content 为空），
            # 由 prompt 约束 + _parse_eval_response 容错解析。
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
    """在统一事件循环内并发执行全部混元评分任务（Python 3.14 兼容）。"""
    return await asyncio.gather(*tasks)


def build_judge_artifact(flow: dict) -> dict | None:
    """把 L2 生成产物转成可评分对象（仅 success 有 result 才送评）。"""
    task = flow.get("task") or {}
    result = task.get("result")
    if task.get("status") != "success" or not isinstance(result, dict) or not result:
        return None
    payload = flow.get("payload") or {}
    group = payload.get("task_group") or {}
    return {
        "case": "L2_dialogue_gen",
        "family": "dialogue_gen",
        "interface_label": "POST /ai/dialogue/v1/generate",
        "request_summary": (
            f"task_group={group.get('lesson_id')}/{group.get('group_id')}"
            f"（{len(group.get('sentences') or [])} 句必用）；"
            f"recall={payload.get('recall')}；prompt_lang={payload.get('prompt_lang')}；"
            f"roles={len(payload.get('roles') or [])}；preferred_type=auto"
        ),
        "output": result,
    }


# ---------------------------------------------------------------------------
# §8.4 验收清单（功能口径）
# ---------------------------------------------------------------------------


def _read_corpus_file(path: Path) -> dict:
    """直接读取本地 corpus.json（--no-live 时清单仍可校验语料结构）。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def build_checklist(
    *,
    flow: dict,
    corpus: dict | None,
    corpus_file: Path,
    generated_dir: Path,
    checkpoint_enabled: bool,
    live_ran: bool,
) -> list[dict]:
    """按 §8.4 逐条给出可自动化验证状态（PASS/WARN/FAIL/SKIP/MANUAL）。"""
    checks: list[dict] = []
    result = ((flow.get("task") or {}).get("result")) or {}
    notes = _notes_of(result)
    by_id = {c["id"]: c for c in (flow.get("checks") or [])}

    def add(num, item, expected, status, detail):
        checks.append({
            "num": num,
            "item": item,
            "expected": expected,
            "status": status,
            "detail": detail,
        })

    # 1 本地语料（L2 未跑时回落直接读 corpus.json）
    corpus = corpus or _read_corpus_file(corpus_file)
    lessons = corpus.get("lessons") or []
    if corpus_file.exists() and len(lessons) >= 20:
        add(1, "本地语料", "20 篇结构，corpus.json 与线上同构", "PASS",
            f"{corpus_file.name} lessons={len(lessons)}")
    elif corpus_file.exists():
        add(1, "本地语料", "20 篇结构", "WARN",
            f"corpus.json 存在但 lessons={len(lessons)}（先跑 scripts/prepare_nc2_corpus.py）")
    else:
        add(1, "本地语料", "20 篇结构", "FAIL", f"缺 {corpus_file}")

    # 2 模拟学习（脚本侧仅验证指标摘要；回写口径由 scripts/simulate_learning.py + 单测覆盖）
    learner = (corpus or {}).get("learner") or {}
    if learner.get("available"):
        add(2, "模拟学习", "skill_state 指标；confidence<0.6 不回写", "PASS",
            f"skill_state_count={learner.get('skill_state_count')} avg_mastery={learner.get('avg_mastery')} "
            f"avg_confidence={learner.get('avg_confidence')}")
    elif live_ran:
        add(2, "模拟学习", "有指标", "WARN", "learner 无指标（先跑 scripts/simulate_learning.py）")
    else:
        add(2, "模拟学习", "有指标", "SKIP", "未跑 L2")

    # 3~7 由 L2 断言映射
    mapping = [
        (3, "必用任务组", "coverage.required_used 覆盖或降级留痕", "coverage"),
        (4, "召回 2-6", "recalled 条数∈[2,6] 或 RECALL_INSUFFICIENT 留痕", "recall"),
        (5, "形态", "dialogue 优先；否则 non_dialogue + sub_type", "form"),
        (6, "提示语言", "prompts[].lang == 请求 prompt_lang", "prompt_lang"),
        (7, "异步契约", "提交毫秒级；状态可轮询；失败不静默", "async"),
    ]
    for num, item, expected, key in mapping:
        c = by_id.get(key)
        if not c:
            add(num, item, expected, "SKIP", "未跑 L2")
        else:
            add(num, item, expected, c["status"], c["detail"])

    # 8 本地可调试
    can_write = False
    try:
        generated_dir.mkdir(parents=True, exist_ok=True)
        probe = generated_dir / ".eval_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        can_write = True
    except Exception:  # noqa: BLE001
        can_write = False
    add(8, "本地可调试", "pytest 全绿不触网；前端可降级演示",
        "PASS" if can_write else "WARN",
        f"落盘目录可写={can_write}；pytest 全量需另行执行（见 --report 汇总的 regression 说明）")

    # 9 页面优先
    page = HERE.parent / "scholar-admin-web" / "src" / "pages" / "english" / "dialogue-gen.tsx"
    service = HERE.parent / "scholar-admin-web" / "src" / "services" / "dialogueGen.ts"
    if page.exists() and service.exists():
        add(9, "页面优先", "页面 mock 可演示 + 真实接口渲染", "PASS",
            f"{page.name} / {service.name} 存在")
    else:
        add(9, "页面优先", "页面存在", "WARN", f"缺 {page.name} 或 {service.name}")

    # 10 断点续写
    resume = flow.get("resume")
    cps = flow.get("checkpoints") or []
    if not live_ran:
        add(10, "断点续写", "checkpoint 可续跑", "SKIP", "未跑 L2")
    elif resume and resume.get("status") == "success":
        add(10, "断点续写", "failed/卡死可 resume 续跑且不重跑已完成节点", "PASS",
            f"续跑终态 success（polls={resume.get('polls')}）")
    elif cps:
        add(10, "断点续写", "checkpoint 可续跑", "PASS",
            f"轨迹 {len(cps)} 条，末阶段={cps[-1].get('stage')}")
    elif checkpoint_enabled:
        add(10, "断点续写", "checkpoint 可续跑", "WARN", "断点开关开启但轨迹为空")
    else:
        add(10, "断点续写", "checkpoint 可续跑", "SKIP", "DIALOGUE_GEN_CHECKPOINT_ENABLED=0")

    # 11 零侵入
    add(11, "零侵入", "既有接口逐断言不变；既有全量测试零失败",
        "MANUAL", "由 `pytest` 全量回归覆盖（本脚本不改既有路径/集合）")

    # 16 两面隔离
    add(16, "两面隔离", "批量面不写 ai_session*",
        "MANUAL", "由 tests/integration/test_routes_dialogue_gen.py 断言覆盖")

    if notes:
        print(f"  [note] result.notes={notes}")
    return checks


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------


def print_report(report: dict) -> None:
    print("\n\n===== 评估汇总 =====")
    print("[L1 可用性]")
    for r in report["availability"]:
        mark = "✓" if r["available"] else ("⚠(草案)" if r["status"] == "draft" else "✗")
        print(f"  {mark} {r['label']}: {r['probe']}")

    live = report.get("live") or {}
    print("[L2 功能链路]")
    if not live.get("ran"):
        print("  （未运行：--no-live）")
    else:
        first = live.get("first_fail")
        if first:
            print(f"  ✗ {first}")
        submit = live.get("submit") or {}
        task = live.get("task") or {}
        print(
            f"  {'✓' if submit.get('status') == 'success' else '✗'} 提交: "
            f"{submit.get('status')} {ms((submit.get('submit_ms') or 0) / 1000)}"
            f" task_id={submit.get('task_id')}"
        )
        print(f"  {'✓' if task.get('status') == 'success' else '⚠'} 终态: {task.get('status')} polls={task.get('polls')}")
        for c in live.get("checks") or []:
            mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "·"}.get(c["status"], "?")
            print(f"    {mark} [{c['status']}] {c['id']}: {c['detail']}")

    print("[§8.4 验收清单]")
    for c in report.get("checklist") or []:
        mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "·", "MANUAL": "☐"}.get(c["status"], "?")
        print(f"  {mark} #{c['num']} {c['item']}: {c['status']} —— {c['detail']}")

    print("[L3 混元质量]")
    if not report["judgments"]:
        print("  （未评分：--no-judge 或无产物）")
    for j in report["judgments"]:
        s = j["score"]
        if s < 0:
            mark, label = "⚠", "JUDGE_SKIPPED"
        else:
            mark = "✓" if s >= HUNYUAN_EVAL_PASS_THRESHOLD else "✗"
            label = f"score={s:.2f} (阈值 {HUNYUAN_EVAL_PASS_THRESHOLD})"
        print(f"  {mark} {j['case']} [{j['family']}]: {label}")
    scores = [j["score"] for j in report["judgments"] if j["score"] >= 0]
    if scores:
        print(f"  → 平均分 = {sum(scores) / len(scores):.2f}")

    print(f"\n[结论] {'PASS' if report['pass'] else 'FAIL'}")
    for reason in report["reasons"]:
        print(f"  - {reason}")


def checklist_reasons(checklist: list[dict], *, strict: bool) -> list[str]:
    """清单中的 FAIL/WARN 转为门禁理由（strict 下 WARN 亦失败）。"""
    reasons: list[str] = []
    for c in checklist:
        if c["status"] == "FAIL":
            reasons.append(f"§8.4 #{c['num']} {c['item']} 失败: {c['detail']}")
        elif c["status"] == "WARN" and strict:
            reasons.append(f"[strict] §8.4 #{c['num']} {c['item']} WARN 视为失败: {c['detail']}")
    return reasons


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AI 对话生成域（批量面 /ai/dialogue/v1）本地冒烟 + 回归门禁"
            "（L0 环境 → L1 可用性 → L2 功能 → L3 混元质量）"
        )
    )
    parser.add_argument("--base-url", default=None, help="服务地址（默认 http://127.0.0.1:{port}）")
    parser.add_argument(
        "--port",
        type=int,
        default=PORT or 8080,
        help="本地服务端口（服务未启动时自动拉起，默认取 config.PORT / 8080）",
    )
    parser.add_argument("--app", default="main:app", help="uvicorn 应用入口（默认 main:app）")
    parser.add_argument(
        "--scholar-id",
        default=DIALOGUE_LOCAL_SCHOLAR_ID,
        help=f"被评估学者 ID（默认本地模拟学者 {DIALOGUE_LOCAL_SCHOLAR_ID}）",
    )
    parser.add_argument("--lesson-id", default=None, help="指定任务组所属课（默认取句子最多的组）")
    parser.add_argument("--group-index", type=int, default=None, help="指定任务组下标（默认自动）")
    parser.add_argument("--prompt-lang", default="zh", choices=list(PROMPT_LANGS), help="轮间提示语言（默认 zh）")
    parser.add_argument("--top-k", type=int, default=4, help="召回条数（图内 clamp 到 [2,6]，默认 4）")
    parser.add_argument("--no-recall", action="store_true", help="关闭召回（recall.enabled=false）")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="轮询间隔秒（默认 1s）")
    parser.add_argument(
        "--max-wait", type=float, default=300, help="等待终态上限秒（默认 300；LLM 生成可达数分钟）"
    )
    parser.add_argument(
        "--submit-budget-ms", type=float, default=2000, help="提交毫秒级预算（默认 2000ms）"
    )
    parser.add_argument("--no-live", action="store_true", help="跳过 L2 功能链路（只跑可用性门禁）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 L3 混元质量评分")
    parser.add_argument("--judge-limit", type=int, default=4, help="混元 Judge 最多调用数（默认 4）")
    parser.add_argument("--no-resume", action="store_true", help="生成失败时不尝试断点续写")
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="自启服务时不强制开启断点开关（DIALOGUE_GEN_CHECKPOINT_ENABLED 保持缺省）",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="自启服务时不强制开启生成图（DIALOGUE_GEN_GRAPH_ENABLED 保持缺省）",
    )
    parser.add_argument("--strict", action="store_true", help="将 WARN/JUDGE_SKIPPED 视为失败")
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="产物落盘目录（默认 DIALOGUE_CORPUS_DIR/generated）",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("ai_dialogue_gen_eval_report.json"),
        help="JSON 报告输出路径",
    )
    parser.add_argument("--no-autostart", action="store_true", help="服务不可达时不自动拉起")
    parser.add_argument("--keep-server", action="store_true", help="自拉起的服务测试完不关闭")
    args = parser.parse_args()

    corpus_file = corpus_file_path()
    generated_dir = resolve_generated_dir(args.generated_dir)

    # ---------------- L0 环境门禁 ----------------
    print("===== [L0 环境门禁] =====")
    print(f"  AUTH_MODE={AUTH_MODE or 'dev(默认，无需鉴权头)'}")
    print(
        f"  批量生成开关: DIALOGUE_GEN_ENABLED={int(DIALOGUE_GEN_ENABLED)} | "
        f"GRAPH={DIALOGUE_GEN_GRAPH_ENABLED} | CHECKPOINT={DIALOGUE_GEN_CHECKPOINT_ENABLED} | "
        f"MAX_RETRY={DIALOGUE_GEN_MAX_RETRY}"
    )
    volcano_ok = bool(VOLCANO_API_KEY and VOLCANO_CHAT_MODEL)
    hunyuan_ok = bool(HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL)
    print(
        f"  火山生成: VOLCANO_CHAT_MODEL={'已配置' if volcano_ok else '未配置'} | "
        f"API_KEY={'已配置' if VOLCANO_API_KEY else '未配置'}"
    )
    print(
        f"  混元 Judge: HUNYUAN_EVAL_MODEL={HUNYUAN_EVAL_MODEL or '(未配置)'} | "
        f"SECRET_KEY={'已配置' if HUNYUAN_SECRET_KEY else '未配置'} | "
        f"PASS_THRESHOLD={HUNYUAN_EVAL_PASS_THRESHOLD}"
    )
    print(
        f"  本地语料: {corpus_file} ({'存在' if corpus_file.exists() else '缺失'}) | "
        f"产物目录: {generated_dir}"
    )
    if not volcano_ok and not args.no_live:
        print("[WARN] 火山方舟凭据缺失 → L2 生成将 failed(LLM_UNAVAILABLE)")
    if not hunyuan_ok and not args.no_judge:
        print("[WARN] 混元凭据未配置 → L3 将全部 JUDGE_SKIPPED")
    elif not args.no_judge:
        if HUNYUAN_BASE_URL.rstrip("/").endswith("hunyuan.tencentcloudapi.com"):
            print(
                "[WARN] HUNYUAN_BASE_URL 仍为腾讯云 OpenAPI 默认值：不兼容 Bearer /chat/completions，"
                "L3 将全部 JUDGE_SKIPPED。请在 .env 设 "
                "HUNYUAN_BASE_URL=https://api.hunyuan.cloud.tencent.com/v1"
            )
        if not os.environ.get("HUNYUAN_SECRET_KEY") and os.environ.get("TENCENTCLOUD_SECRETKEY"):
            print(
                "[WARN] HUNYUAN_SECRET_KEY 未显式配置（回落 TENCENTCLOUD_SECRETKEY）："
                "Bearer 鉴权需混元控制台 API 密钥，L3 可能 401/403"
            )
    if not corpus_file.exists():
        print("[WARN] 本地语料缺失 → L2 的 /corpus 将 corpus_available=false；先跑 scripts/prepare_nc2_corpus.py")

    # ---------------- 报告骨架 ----------------
    report: dict = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "base_url": args.base_url or f"http://127.0.0.1:{args.port}",
            "scholar_id": args.scholar_id,
            "prompt_lang": args.prompt_lang,
            "judge_model": HUNYUAN_EVAL_MODEL,
            "pass_threshold": HUNYUAN_EVAL_PASS_THRESHOLD,
            "dialogue_gen_enabled": int(DIALOGUE_GEN_ENABLED),
            "graph_enabled": bool(DIALOGUE_GEN_GRAPH_ENABLED),
            "checkpoint_enabled": bool(DIALOGUE_GEN_CHECKPOINT_ENABLED),
            "max_retry": DIALOGUE_GEN_MAX_RETRY,
            "source_docs": [
                "scholar-skill/docs_v1/AI会话/AI英语对话生成设计.md",
            ],
            "server_app": args.app,
        },
        "availability": [],
        "live": {"ran": False},
        "checklist": [],
        "judgments": [],
        "artifacts": [],
        "summary": {},
        "pass": False,
        "reasons": [],
    }

    # ---------------- 服务可达（自启时强制开启批量生成开关） ----------------
    port = args.port
    base_url = args.base_url or f"http://127.0.0.1:{port}"
    server_env = {"DIALOGUE_GEN_ENABLED": "1"}
    if not args.no_graph:
        server_env["DIALOGUE_GEN_GRAPH_ENABLED"] = "1"
    if not args.no_checkpoint:
        server_env["DIALOGUE_GEN_CHECKPOINT_ENABLED"] = "1"
    proc = ensure_server(
        base_url,
        port,
        autostart=not args.no_autostart,
        app=args.app,
        server_env=server_env,
    )
    # 仅当服务由本脚本拉起时，才能确认断点开关被注入；指向既有服务时以 config 值为准
    effective_checkpoint = bool(DIALOGUE_GEN_CHECKPOINT_ENABLED) or (
        proc is not None and not args.no_checkpoint
    )

    reasons: list[str] = []
    missing_routes: list[dict] = []
    flow: dict = {"checks": [], "checkpoints": []}
    try:
        with httpx.Client(timeout=30.0) as client:
            # ---------------- L1 ----------------
            availability = run_availability(client, base_url)
            report["availability"] = availability
            missing_routes = [
                r for r in availability if r["status"] == "implemented" and not r["available"]
            ]
            for r in missing_routes:
                reasons.append(
                    f"接口契约缺失/不符: {r['id']} -> {r['probe']} {r['detail']}"
                )

            # ---------------- L2 ----------------
            if not args.no_live:
                if missing_routes:
                    print("\n[L2] 存在契约缺失，跳过功能链路")
                else:
                    flow = run_generate_flow(
                        client,
                        base_url,
                        scholar_id=args.scholar_id,
                        prompt_lang=args.prompt_lang,
                        recall_enabled=not args.no_recall,
                        top_k=args.top_k,
                        group_index=args.group_index,
                        lesson_id=args.lesson_id,
                        poll_interval=args.poll_interval,
                        max_wait=args.max_wait,
                        submit_budget_ms=args.submit_budget_ms,
                        checkpoint_enabled=effective_checkpoint,
                        try_resume=not args.no_resume,
                    )
                    task = flow.get("task") or {}
                    report["live"] = {
                        "ran": True,
                        "corpus": flow.get("corpus"),
                        "submit": flow.get("submit"),
                        "task": task,
                        "checkpoints": len(flow.get("checkpoints") or []),
                        "user_input": flow.get("user_input"),
                        "resume": flow.get("resume"),
                        "checks": flow.get("checks"),
                    }
                    # 功能链路失败原因
                    if (flow.get("corpus") or {}).get("status") != "success":
                        report["live"]["first_fail"] = (
                            f"语料读取失败: {(flow.get('corpus') or {}).get('error')}"
                        )
                        reasons.append(report["live"]["first_fail"])
                    elif (flow.get("submit") or {}).get("status") != "success":
                        report["live"]["first_fail"] = (
                            f"提交失败: {(flow.get('submit') or {}).get('error')}"
                        )
                        reasons.append(report["live"]["first_fail"])
                    elif task.get("status") != "success":
                        fail = f"生成未成功: status={task.get('status')} error={task.get('error')}"
                        report["live"]["first_fail"] = fail
                        reasons.append(fail)
                    for c in flow.get("checks") or []:
                        if c["status"] == "FAIL":
                            reasons.append(f"§8.4 断言失败 [{c['id']}]: {c['detail']}")
                        elif c["status"] == "WARN" and args.strict:
                            reasons.append(f"[strict] §8.4 断言 WARN [{c['id']}]: {c['detail']}")
                    # 用户作答
                    ui = flow.get("user_input")
                    if ui and ui.get("status") != "success":
                        reasons.append(f"用户作答评测失败: {ui.get('status')} {ui.get('error')}")
                    # 续写
                    resume = flow.get("resume")
                    if resume and resume.get("status") not in (None, "success", "SKIP"):
                        reasons.append(
                            f"断点续写失败: {resume.get('status')} {resume.get('error')}"
                        )

                    # 产物落盘
                    try:
                        path = persist_artifacts(
                            generated_dir,
                            flow,
                            scholar_id=args.scholar_id,
                            meta=report["meta"],
                        )
                        report["artifacts"].append(str(path))
                        print(f"\n[产物] 已落盘 {path}")
                    except Exception as e:  # noqa: BLE001
                        reasons.append(f"产物落盘失败: {e}")

            # ---------------- §8.4 清单 ----------------
            checklist = build_checklist(
                flow=flow,
                corpus=(flow.get("corpus") or {}).get("data") if flow.get("corpus") else None,
                corpus_file=corpus_file,
                generated_dir=generated_dir,
                checkpoint_enabled=effective_checkpoint,
                live_ran=bool(report["live"].get("ran")),
            )
            report["checklist"] = checklist
            reasons.extend(checklist_reasons(checklist, strict=args.strict))
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            logger.info("已关闭自动拉起的本地服务")

    # ---------------- L3 混元质量 ----------------
    if not args.no_judge and args.judge_limit > 0:
        artifact = build_judge_artifact(flow)
        if artifact is None:
            if report["live"].get("ran"):
                print("\n[L3] 无可用产物（L2 未成功产出 result），跳过评分")
        else:
            print("\n[L3] 混元质量评分中 ...")
            sem = asyncio.Semaphore(2)
            results = asyncio.run(_judge_all([_judge_one(artifact, sem)]))
            judged = results[0] if results else {"score": -1.0, "feedback": "无结果"}
            report["judgments"].append(
                {
                    "case": artifact["case"],
                    "family": artifact["family"],
                    "interface": artifact["interface_label"],
                    **judged,
                }
            )
            scores = [j["score"] for j in report["judgments"] if j["score"] >= 0]
            low = [j for j in report["judgments"] if 0 <= j["score"] < HUNYUAN_EVAL_PASS_THRESHOLD]
            for j in low:
                reasons.append(
                    f"混元评分未达标: {j['case']} score={j['score']:.2f} "
                    f"< {HUNYUAN_EVAL_PASS_THRESHOLD} | {str(j['feedback'])[:160]}"
                )
            if not scores:
                first_fb = str(report["judgments"][0].get("feedback") or "")[:160]
                reasons.append(
                    "混元 Judge 跳过（凭据/网关/超时），请检查 HUNYUAN_* 配置"
                    + (f"；首个原因: {first_fb}" if first_fb else "")
                )
                if args.strict:
                    reasons.append("[strict] JUDGE_SKIPPED 视为失败")

    # ---------------- 汇总判定 ----------------
    implemented = [r for r in report["availability"] if r["status"] == "implemented"]
    report["summary"] = {
        "routes_total": len(report["availability"]),
        "routes_ok": sum(1 for r in report["availability"] if r["available"]),
        "routes_missing": len(missing_routes) if report["availability"] else 0,
        "live_ran": bool(report["live"].get("ran")),
        "checklist_pass": sum(1 for c in report["checklist"] if c["status"] == "PASS"),
        "checklist_warn": sum(1 for c in report["checklist"] if c["status"] == "WARN"),
        "checklist_fail": sum(1 for c in report["checklist"] if c["status"] == "FAIL"),
        "checklist_manual": sum(1 for c in report["checklist"] if c["status"] == "MANUAL"),
        "judge_count": len(report["judgments"]),
        "judge_pass_count": sum(
            1
            for j in report["judgments"]
            if j["score"] >= 0 and j["score"] >= HUNYUAN_EVAL_PASS_THRESHOLD
        ),
        "implemented_count": len(implemented),
    }
    deduped: list[str] = []
    for reason in reasons:
        if reason not in deduped:
            deduped.append(reason)
    report["pass"] = not deduped
    report["reasons"] = deduped

    # 文件输出
    out_path = args.report_path
    if not out_path.is_absolute():
        out_path = HERE / out_path
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] 已写入 {out_path}")

    print_report(report)

    # ---------------- 退出码 ----------------
    if report["pass"]:
        sys.exit(0)
    # 2 = 契约缺失 / Judge 环境缺失
    if missing_routes:
        sys.exit(2)
    if not args.no_judge and not (HUNYUAN_SECRET_KEY and HUNYUAN_EVAL_MODEL):
        sys.exit(2)
    sys.exit(1)


if __name__ == "__main__":
    main()
