"""新概念英语教材批量构建脚本 — 按 /build/nce 端点分批写入

用法::

    # 默认构建 Book 2 全 96 课,每批 10 课
    python scripts/build_nce_batch.py

    # 指定册数 + 批次大小
    python scripts/build_nce_batch.py --book 2 --batch-size 10

    # 只跑指定课号范围(用于补跑失败批次)
    python scripts/build_nce_batch.py --book 2 --start 11 --end 20

    # 自定义服务地址(默认本地 8080)
    python scripts/build_nce_batch.py --base-url http://127.0.0.1:8080

断点文件:data/nc2_build_progress.json(按 book 区分)
- completed: 已成功批次列表(含 textbook_id / lesson_count / sentence_count / 时间戳)
- failed:    失败批次列表(含 error / attempts)
- next_batch: 下一批的 start/end;脚本启动时从这里继续

中断(Ctrl+C)会保存当前状态,再次启动自动从 next_batch 继续,不重复写库。
失败的批次不会卡住主流程,会在最后总结里列出供手动重跑。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger("build_nce_batch")

# NCE 各册总课数(对齐 services.build.build_nce.NCE_BOOKS)
NCE_TOTAL_LESSONS = {"1": 144, "2": 96, "3": 60, "4": 48}

PROGRESS_DIR = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------------------
# 进度文件
# ---------------------------------------------------------------------------


def progress_path(book: str) -> Path:
    return PROGRESS_DIR / f"nc{book}_build_progress.json"


def load_progress(book: str) -> dict:
    """读断点;不存在返回初始结构。"""
    p = progress_path(book)
    if not p.exists():
        return {"book": book, "completed": [], "failed": [], "next_batch": None}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"进度文件损坏,重置: {e}")
        return {"book": book, "completed": [], "failed": [], "next_batch": None}


def save_progress(book: str, progress: dict) -> None:
    p = progress_path(book)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# HTTP 调用
# ---------------------------------------------------------------------------


def call_build_nce(
    base_url: str, book: str, start: int, end: int, timeout: int = 300
) -> dict:
    """POST /build/nce,返回响应 dict;失败抛 Exception。"""
    url = f"{base_url.rstrip('/')}/build/nce"
    payload = json.dumps(
        {"book": book, "start_lesson": start, "end_lesson": end}
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    data = json.loads(body)
    if not data.get("success"):
        raise RuntimeError(f"接口返回 success=false: {data}")
    return data


# ---------------------------------------------------------------------------
# 批次迭代
# ---------------------------------------------------------------------------


def iter_batches(
    total: int,
    batch_size: int,
    start_override: int | None,
    end_override: int | None,
) -> list[tuple[int, int]]:
    """生成 (start, end) 批次列表,end 含本课。

    - 默认从 1 跑到 total,按 batch_size 切分
    - 指定 start/end:从 start 跑到 end,仍按 batch_size 切分
      (用于跳过已写库的前几课,例如烟测写了 1-2 后用 --start 3 跑后面)
    """
    s = max(1, start_override or 1)
    e_end = min(total, end_override or total)
    batches: list[tuple[int, int]] = []
    while s <= e_end:
        e = min(s + batch_size - 1, e_end)
        batches.append((s, e))
        s = e + 1
    return batches


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def run(
    book: str,
    batch_size: int,
    base_url: str,
    start_override: int | None,
    end_override: int | None,
    timeout: int,
    sleep_between: float,
) -> int:
    total = NCE_TOTAL_LESSONS[book]
    progress = load_progress(book)
    completed_set = {
        (b["start"], b["end"]) for b in progress.get("completed", [])
    }

    batches = list(iter_batches(total, batch_size, start_override, end_override))
    print(f"=== NCE Book {book} 共 {total} 课,本轮待跑 {len(batches)} 批 ===")
    print(f"已完成批次: {len(completed_set)}")

    fail_count_this_run = 0
    success_count_this_run = 0

    for idx, (s, e) in enumerate(batches, 1):
        # 跳过已完成的批次(只在非 override 模式下跳,override 是手动补跑)
        if (s, e) in completed_set and start_override is None:
            print(f"[{idx}/{len(batches)}] Lesson {s}-{e} 已完成,跳过")
            continue

        print(f"[{idx}/{len(batches)}] Lesson {s}-{e} 调用中...", end=" ", flush=True)
        t0 = time.time()
        try:
            resp = call_build_nce(base_url, book, s, e, timeout=timeout)
            elapsed = time.time() - t0
            data = resp.get("data", {})
            entry = {
                "start": s,
                "end": e,
                "lesson_count": data.get("unit_count", 0),
                "sentence_count": data.get("total_sentences", 0),
                "textbook_id": data.get("textbook_id", ""),
                "reused_textbook": resp.get("reused_textbook", False),
                "elapsed_sec": round(elapsed, 1),
                "ts": int(time.time()),
            }
            progress.setdefault("completed", []).append(entry)
            completed_set.add((s, e))
            progress["next_batch"] = None  # 进行中,不存 next
            save_progress(book, progress)
            success_count_this_run += 1
            print(
                f"OK 课={entry['lesson_count']} 句={entry['sentence_count']} "
                f"tb={entry['textbook_id']} ({elapsed:.1f}s)"
            )
        except Exception as e_err:
            elapsed = time.time() - t0
            err_entry = {
                "start": s,
                "end": e,
                "error": str(e_err)[:300],
                "elapsed_sec": round(elapsed, 1),
                "ts": int(time.time()),
                "attempts": 1,
            }
            # 已有同批次失败记录 → attempts+1
            existing = next(
                (f for f in progress.get("failed", []) if f["start"] == s and f["end"] == e),
                None,
            )
            if existing:
                existing["attempts"] = existing.get("attempts", 0) + 1
                existing["error"] = str(e_err)[:300]
                existing["ts"] = int(time.time())
            else:
                progress.setdefault("failed", []).append(err_entry)
            save_progress(book, progress)
            fail_count_this_run += 1
            print(f"FAIL ({elapsed:.1f}s): {e_err}")

        # 批间小睡,避免对 LLM 网关压力
        if idx < len(batches) and sleep_between > 0:
            time.sleep(sleep_between)

    # 总结
    print("\n=== 本轮总结 ===")
    print(f"成功批次: {success_count_this_run}")
    print(f"失败批次: {fail_count_this_run}")

    # 是否全部完成
    all_batches = set(
        (s, e) for s, e in iter_batches(total, batch_size, None, None)
    )
    missing = all_batches - completed_set
    if not missing:
        print(f"\n✓ Book {book} 全部 {total} 课已完成!")
    else:
        print(f"\n⚠ 还有 {len(missing)} 批未完成:")
        for s, e in sorted(missing):
            print(f"  - Lesson {s}-{e}")
        print(f"  补跑命令: python scripts/build_nce_batch.py --book {book} --start <s> --end <e>")

    # 失败批次明细
    if progress.get("failed"):
        print(f"\n失败批次明细(共 {len(progress['failed'])} 批,含历史):")
        for f in progress["failed"]:
            print(
                f"  - Lesson {f['start']}-{f['end']}: "
                f"attempts={f.get('attempts', 1)} error={f.get('error', '')[:100]}"
            )

    return 0 if fail_count_this_run == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="NCE 批量构建(分批 + 断点续跑)")
    ap.add_argument("--book", default="2", choices=["1", "2", "3", "4"], help="册数(默认 2)")
    ap.add_argument("--batch-size", type=int, default=10, help="每批课数(默认 10)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080", help="服务地址")
    ap.add_argument("--start", type=int, default=None, help="只跑指定起始课(手动补跑用)")
    ap.add_argument("--end", type=int, default=None, help="只跑指定结束课(手动补跑用)")
    ap.add_argument("--timeout", type=int, default=300, help="单批次超时秒(默认 300)")
    ap.add_argument("--sleep-between", type=float, default=1.0, help="批间小睡秒(默认 1.0)")
    ap.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 覆盖模式下 batch_size 无意义,固定为 1 批
    if args.start is not None:
        run(
            book=args.book,
            batch_size=args.batch_size,
            base_url=args.base_url,
            start_override=args.start,
            end_override=args.end,
            timeout=args.timeout,
            sleep_between=args.sleep_between,
        )
        return

    sys.exit(run(
        book=args.book,
        batch_size=args.batch_size,
        base_url=args.base_url,
        start_override=None,
        end_override=None,
        timeout=args.timeout,
        sleep_between=args.sleep_between,
    ))


if __name__ == "__main__":
    main()
