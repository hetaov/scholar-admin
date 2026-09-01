#!/usr/bin/env python3
"""翻译评估 v2 接口本地耗时验证脚本（中译英 ce / 英译中 ec）

覆盖三个层面的耗时：
  1) HTTP 全链路
     POST /eval/translate/v2            提交 → task_id(pending)，测「提交接口返回耗时」
     GET  /eval/translate/v2/task/{id}  轮询 → success/failed，测「到终态总耗时」
  2) 纯 LLM 单次评分耗时（进程内直连 evaluate_translation_v2，不经过 HTTP）
  3) 单次 CloudBase DB 读写 RTT（insert/query/update，进程内直连）

用法：
  # 服务已运行在 8080（python main.py）
  python scripts/translation_v2_verify.py

  # 服务未启动：脚本自动拉起 uvicorn main:app
  python scripts/translation_v2_verify.py --port 8080

  # 只做 LLM/DB 微观探针，不碰 HTTP
  python scripts/translation_v2_verify.py --probe-only

  # 语音路径（真实 mp3，测 ASR + LLM）
  python scripts/translation_v2_verify.py --voice-audio /path/to/xx.mp3

  # 完整跑 N 轮取 min/avg/max
  python scripts/translation_v2_verify.py --rounds 3 --poll-interval 1
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

# 保证 `python scripts/xxx.py` 与 `python -m scripts.xxx` 均可运行（项目根入 sys.path）
HERE = Path(__file__).resolve().parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 复用 config 加载 .env（凭据源与 main.py 一致）
from config import (
    AUTH_MODE,
    TRANSLATION_LLM_TIMEOUT_SECONDS,
    VOLCANO_API_KEY,
    VOLCANO_CHAT_MODEL,
    VOLCANO_VISION_MODEL,
)

# VOLCANO_CHAT_MODEL 是否「显式配置」还是「回退到视觉模型」
_CHAT_MODEL_EXPLICIT = bool(os.environ.get("VOLCANO_CHAT_MODEL"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("translation_v2_verify")

POLL_TERMINAL = {"success", "failed"}

# 测试样例（对齐 e2e/integration 用例）
CASES = {
    "ec": {  # 英译中：英文原句 + 中文译文
        "original_text": "It is a watch.",
        "user_input": "它是一块手表。",
    },
    "ce": {  # 中译英：中文原句 + 英文译文
        "original_text": "这是一块手表。",
        "user_input": "It is a watch.",
    },
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
    log_path = Path("/tmp/scholar_translation_v2_server.log")
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


# ---------------------------------------------------------------------------
# 进程内微观探针：LLM 单次耗时 / DB 单次 RTT
# ---------------------------------------------------------------------------


def probe_llm(rounds: int) -> float | None:
    """直连 evaluate_translation_v2，测纯 LLM 评分耗时（不经过 HTTP/DB）。

    Returns:
        平均耗时秒（ec 模式），用于汇总页优化建议。
    """
    from services.providers.translation_eval import evaluate_translation_v2

    print("\n===== [探针] 纯 LLM 单次评分耗时（evaluate_translation_v2 直连） =====")
    if not (VOLCANO_API_KEY and VOLCANO_CHAT_MODEL):
        print("[WARN] 未配置 VOLCANO_API_KEY / VOLCANO_CHAT_MODEL，跳过 LLM 探针")
        return None
    print(f"  模型: {VOLCANO_CHAT_MODEL}"
          f"（{'显式配置 VOLCANO_CHAT_MODEL' if _CHAT_MODEL_EXPLICIT else '未配置，回退到 VOLCANO_VISION_MODEL 视觉模型'}）")
    print(f"  超时上限: TRANSLATION_LLM_TIMEOUT_SECONDS={TRANSLATION_LLM_TIMEOUT_SECONDS}s")
    avg = None
    for mode, case in CASES.items():
        lat = []
        for i in range(rounds):
            t0 = time.perf_counter()
            try:
                res = asyncio.run(
                    evaluate_translation_v2(
                        mode, case["original_text"], case["user_input"]
                    )
                )
                dt = time.perf_counter() - t0
                lat.append(dt)
                print(
                    f"  [{mode}] 第{i + 1}次: {ms(dt)}  status={res.get('status')} "
                    f"feedback={str(res.get('feedback'))[:30]}"
                )
            except Exception as e:
                print(f"  [{mode}] 第{i + 1}次: 失败 {type(e).__name__}: {e}")
        if lat:
            print(
                f"  [{mode}] min={ms(min(lat))} avg={ms(sum(lat) / len(lat))} "
                f"max={ms(max(lat))}（{len(lat)} 次）"
            )
            if mode == "ec":
                avg = sum(lat) / len(lat)
    return avg


def probe_db(rounds: int) -> float | None:
    """直连 CloudBaseNoSQLClient，测 insert/query/update/delete 单次 RTT。

    Returns:
        insert 平均 RTT 秒（提交接口耗时 = 1 次 insert），用于汇总页优化建议。
    """
    from services.database import CloudBaseNoSQLClient

    print("\n===== [探针] CloudBase DB 单次操作 RTT（tcb.tencentcloudapi.com） =====")
    try:
        db = CloudBaseNoSQLClient()
    except Exception as e:
        print(f"[WARN] 无法创建 DB 客户端: {e}")
        return None
    probe_task_id = f"tr_probe_{int(time.time() * 1000)}"
    doc = {
        "task_id": probe_task_id,
        "original_text": "probe",
        "status": "pending",
        "created_at": int(time.time() * 1000),
    }

    async def _measure_all() -> tuple[list[float], list[float], list[float]]:
        """在单一事件循环内顺序执行全部操作（共享 AsyncClient 复用同一连接池）。"""
        ins: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            await db.insert("translation_task", doc)
            ins.append(time.perf_counter() - t0)
        qry: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            await db.query("translation_task", where={"task_id": probe_task_id})
            qry.append(time.perf_counter() - t0)
        upd: list[float] = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            await db.update(
                "translation_task",
                where={"task_id": probe_task_id},
                data={"$set": {"status": "processing"}},
            )
            upd.append(time.perf_counter() - t0)
        return ins, qry, upd

    avg = None
    try:
        ins, qry, upd = asyncio.run(_measure_all())
        if ins:
            print(f"  insert ×{rounds}: " + " ".join(ms(v) for v in ins) +
                  f"  → avg={ms(sum(ins) / len(ins))}")
            avg = sum(ins) / len(ins)
        if qry:
            print(f"  query  ×{rounds}: " + " ".join(ms(v) for v in qry) +
                  f"  → avg={ms(sum(qry) / len(qry))}")
        if upd:
            print(f"  update ×{rounds}: " + " ".join(ms(v) for v in upd) +
                  f"  → avg={ms(sum(upd) / len(upd))}")
    finally:
        asyncio.run(db.delete("translation_task", where={"task_id": probe_task_id}))
        logger.info(f"[探针] 已清理探针任务 {probe_task_id}")
    return avg


# ---------------------------------------------------------------------------
# HTTP 全链路：提交 + 轮询到终态
# ---------------------------------------------------------------------------


def run_http_case(
    client: httpx.Client,
    base_url: str,
    name: str,
    case: dict,
    poll_interval: float,
    max_wait: float,
) -> dict | None:
    print(f"\n----- HTTP 链路 [{name}] "
          f"original_text={case['original_text']!r} user_input={case['user_input']!r} -----")

    # 1) 提交
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/eval/translate/v2", json=case)
    except Exception as e:
        print(f"  [提交] 请求异常: {type(e).__name__}: {e}")
        return None
    submit_dt = time.perf_counter() - t0
    try:
        body = resp.json()
    except Exception:
        print(f"  [提交] HTTP {resp.status_code}，非 JSON 响应: {resp.text[:200]}")
        return None
    print(f"  [提交] HTTP {resp.status_code} 耗时={ms(submit_dt)}")
    if not body.get("success"):
        print(f"  [提交] 业务失败: {body.get('code')} {body.get('message')}")
        return None
    task_id = body["data"]["task_id"]
    status0 = body["data"]["status"]
    print(f"  [提交] task_id={task_id} status={status0}（异步，pending 即返回）")

    # 2) 轮询到终态
    poll_start = time.perf_counter()
    polls = []
    last_status = status0
    while True:
        if time.perf_counter() - poll_start > max_wait:
            print(f"  [轮询] 超过 {ms(max_wait)} 仍未终态，放弃（status={last_status}）")
            return None
        time.sleep(poll_interval)
        t1 = time.perf_counter()
        try:
            r = client.get(f"{base_url}/eval/translate/v2/task/{task_id}")
        except Exception as e:
            print(f"  [轮询] 请求异常: {type(e).__name__}: {e}")
            return None
        dt = time.perf_counter() - t1
        polls.append(dt)
        try:
            data = r.json()["data"]
        except Exception:
            print(f"  [轮询] HTTP {r.status_code} 非预期响应: {r.text[:200]}")
            return None
        last_status = data["status"]
        if r.status_code == 404:
            print("  [轮询] 404：任务不存在/已过期")
            return None
        extra = ""
        if data.get("result"):
            extra = f" status={data['result'].get('status')}"
        if data.get("error"):
            extra = f" error={str(data['error'])[:60]}"
        print(
            f"  [轮询] #{len(polls)} 耗时={ms(dt)} status={last_status}{extra}"
        )
        if last_status in POLL_TERMINAL:
            break

    total_dt = time.perf_counter() - t0
    worker_dt = total_dt - submit_dt
    print(
        f"  [汇总] {name}: 提交={ms(submit_dt)}，轮询{len(polls)}次到终态={ms(total_dt)}，"
        f"后台执行(终态-提交)≈{ms(worker_dt)}，终态={last_status}"
    )
    return {
        "name": name,
        "task_id": task_id,
        "submit_ms": submit_dt * 1000,
        "total_ms": total_dt * 1000,
        "worker_ms": worker_dt * 1000,
        "poll_count": len(polls),
        "poll_avg_ms": (sum(polls) / len(polls) * 1000) if polls else 0,
        "status": last_status,
    }


def run_voice_case(
    client: httpx.Client,
    base_url: str,
    audio_path: Path,
    poll_interval: float,
    max_wait: float,
) -> dict | None:
    """语音路径（ce 中译英）：中文原句 + 用户英文口语录音。"""
    audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()
    case = {
        "original_text": "这是一块手表。",
        "audio_base64": audio_b64,
        "voice_format": "mp3",
    }
    print(f"\n----- HTTP 链路 [voice/ce] audio={audio_path}（{len(audio_b64) // 1024}KB b64）-----")
    t0 = time.perf_counter()
    try:
        resp = client.post(f"{base_url}/eval/translate/v2", json=case)
    except Exception as e:
        print(f"  [提交] 请求异常: {type(e).__name__}: {e}")
        return None
    submit_dt = time.perf_counter() - t0
    print(f"  [提交] HTTP {resp.status_code} 耗时={ms(submit_dt)}")
    body = resp.json()
    if not body.get("success"):
        print(f"  [提交] 业务失败: {body.get('code')} {body.get('message')}")
        return None
    task_id = body["data"]["task_id"]
    print(f"  [提交] task_id={task_id} status={body['data']['status']}")

    while True:
        if time.perf_counter() - t0 > max_wait:
            print("  [轮询] 超时放弃")
            return None
        time.sleep(poll_interval)
        t1 = time.perf_counter()
        r = client.get(f"{base_url}/eval/translate/v2/task/{task_id}")
        dt = time.perf_counter() - t1
        data = r.json()["data"]
        extra = ""
        if data.get("result"):
            extra = f" transcription={data['result'].get('transcription')!r}"
        if data.get("error"):
            extra = f" error={str(data['error'])[:60]}"
        print(f"  [轮询] 耗时={ms(dt)} status={data['status']}{extra}")
        if data["status"] in POLL_TERMINAL:
            break

    total_dt = time.perf_counter() - t0
    print(
        f"  [汇总] voice/ce: 提交={ms(submit_dt)}，到终态={ms(total_dt)}，"
        f"后台执行≈{ms(total_dt - submit_dt)}，终态={data['status']}"
    )
    return {
        "name": "voice/ce",
        "submit_ms": submit_dt * 1000,
        "total_ms": total_dt * 1000,
        "worker_ms": (total_dt - submit_dt) * 1000,
        "status": data["status"],
    }


# ---------------------------------------------------------------------------
# 汇总与优化建议
# ---------------------------------------------------------------------------


def print_summary(results: list[dict], llm_avg_s: float | None, db_avg_ms: float | None) -> None:
    print("\n\n===== 汇总 =====")
    print(f"{'链路':<12}{'提交耗时':<12}{'到终态总耗时':<14}{'后台执行':<12}{'轮询次数':<8}{'终态'}")
    for r in results:
        print(
            f"{r['name']:<12}{ms(r['submit_ms'] / 1000):<12}"
            f"{ms(r['total_ms'] / 1000):<14}{ms(r['worker_ms'] / 1000):<12}"
            f"{r.get('poll_count', '-'):<8}{r['status']}"
        )

    print("\n===== 耗时构成与优化点（依据本次实测） =====")
    hints = []
    db_str = ms(db_avg_ms / 1000) if db_avg_ms else "未测(--no-probe)"
    llm_str = ms(llm_avg_s) if llm_avg_s else "未测(--no-probe)"

    # 提交接口
    if results:
        submit_avg = sum(r["submit_ms"] for r in results) / len(results)
        print(f"[提交接口] POST /eval/translate/v2 平均 {ms(submit_avg / 1000)}。"
              f"耗时 = 1 次 DB insert 往返（实测 {db_str}）+ 参数校验。"
              f"设计文档称『毫秒级』，实际是相对 LLM 秒级而言，DB 网络往返不可省略（契约要求先落库再返回）。"
              f"1/50 概率内联巡检已移出提交热路径（见 services/background_tasks.py）。")

    # 后台执行
    if results:
        worker_avg = sum(r["worker_ms"] for r in results) / len(results)
        print(f"[后台执行] run_translation_task 平均 {ms(worker_avg / 1000)}。"
              f"构成 = claim(1 次 DB update) + LLM 评分(实测 {llm_str})"
              f" + 终态双写并行(1 次 DB update + 1 次 DB insert，asyncio.gather)。")
        if llm_avg_s and worker_avg:
            ratio = llm_avg_s / (worker_avg / 1000) * 100
            print(f"          其中 LLM 评分占比 ≈ {ratio:.0f}%，是绝对大头。")
            hints.append(
                "LLM 是最大耗时项（占比最大）：当前 VOLCANO_CHAT_MODEL 未在 .env 配置，"
                "回退到 VOLCANO_VISION_MODEL（视觉模型）做纯文本评分——建议在 .env 配置专用"
                "文本对话模型接入点（如 doubao-lite / doubao-seed-1.6-lite），通常可显著降延迟与成本。"
            )
            hints.append(
                f"TRANSLATION_LLM_TIMEOUT_SECONDS={TRANSLATION_LLM_TIMEOUT_SECONDS}s 默认 300s，"
                "而前端轮询上限仅 60s（5s×12）。建议降到 30~60s，让失败尽早收敛。"
            )

    print("\n建议优化清单：")
    for i, h in enumerate(hints, 1):
        print(f"  {i}. {h}")
    print("\n（注：以上为本地实测数据；线上 CloudRun 的网络延迟与本机不同，建议以线上日志交叉验证。）")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="翻译评估 v2 接口耗时验证（ce 中译英 / ec 英译中）")
    parser.add_argument("--base-url", default=None, help="服务地址（默认 http://127.0.0.1:{port}）")
    parser.add_argument("--port", type=int, default=8080, help="本地服务端口（服务未启动时自动拉起，默认 8080）")
    parser.add_argument("--rounds", type=int, default=1, help="每个链路重复轮数（默认 1）")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="轮询间隔秒（默认 1s）")
    parser.add_argument("--max-wait", type=float, default=180, help="等待终态上限秒（默认 180）")
    parser.add_argument("--probe-llm-rounds", type=int, default=3, help="LLM 微观探针次数（默认 3）")
    parser.add_argument("--probe-db-rounds", type=int, default=3, help="DB 微观探针次数（默认 3）")
    parser.add_argument("--voice-audio", type=Path, default=None, help="语音路径（mp3，测 voice/ce 链路）")
    parser.add_argument("--no-probe", action="store_true", help="跳过 LLM/DB 微观探针")
    parser.add_argument("--probe-only", action="store_true", help="只跑 LLM/DB 微观探针，不跑 HTTP")
    parser.add_argument("--no-autostart", action="store_true", help="服务不可达时不自动拉起")
    parser.add_argument("--keep-server", action="store_true", help="自拉起的服务测试完不关闭")
    args = parser.parse_args()

    print("===== 环境信息 =====")
    print(f"  AUTH_MODE={AUTH_MODE or 'dev(默认，无需鉴权头)'}")
    print(f"  VOLCANO_API_KEY={'已配置' if VOLCANO_API_KEY else '未配置'} | "
          f"VOLCANO_CHAT_MODEL={'已配置' if _CHAT_MODEL_EXPLICIT else '未配置(回退视觉模型)'} | "
          f"VOLCANO_VISION_MODEL={'已配置' if VOLCANO_VISION_MODEL else '未配置'}")
    print(f"  TRANSLATION_LLM_TIMEOUT_SECONDS={TRANSLATION_LLM_TIMEOUT_SECONDS}s")

    # 微观探针
    llm_avg_s: float | None = None
    db_avg_ms: float | None = None
    if not args.no_probe:
        llm_avg_s = probe_llm(args.probe_llm_rounds)
        db_avg = probe_db(args.probe_db_rounds)
        db_avg_ms = db_avg * 1000 if db_avg else None

    if args.probe_only:
        return

    # HTTP 链路
    port = args.port
    base_url = args.base_url or f"http://127.0.0.1:{port}"
    proc = ensure_server(base_url, port, autostart=not args.no_autostart)
    results: list[dict] = []
    try:
        with httpx.Client(timeout=30.0) as client:
            for _ in range(args.rounds):
                for name, case in CASES.items():
                    r = run_http_case(
                        client, base_url, name, case,
                        args.poll_interval, args.max_wait,
                    )
                    if r:
                        results.append(r)
            if args.voice_audio:
                if not args.voice_audio.exists():
                    print(f"[WARN] 音频不存在: {args.voice_audio}")
                else:
                    r = run_voice_case(
                        client, base_url, args.voice_audio,
                        args.poll_interval, args.max_wait,
                    )
                    if r:
                        results.append(r)
    finally:
        if proc is not None and not args.keep_server:
            proc.terminate()
            logger.info("已关闭自动拉起的本地服务")

    print_summary(results, llm_avg_s, db_avg_ms)


if __name__ == "__main__":
    main()
