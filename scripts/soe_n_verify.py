#!/usr/bin/env python3
"""SOE-N 句级口语评测最小调用封装（F1-2 验证用）

腾讯云口语评测（新版 SOE-N）接入要点：
- 传输：WSS WebSocket（wss://soe.cloud.tencent.com/soe/api/<appid>），非 TC3 REST
- 鉴权：URL 查询参数 HMAC-SHA1 + Base64 签名（官方 SDK 内部处理）
- 官方 Python SDK：tencentcloud-speech-sdk-python（GitHub: TencentCloud/tencentcloud-speech-sdk-python）
  ⚠️ 2026-08-17 实测：**PyPI 上不存在该包**（pip 安装报 "No matching distribution"），
  仓库为纯源码分发（无 setup.py），需 git clone 后把仓库根目录加入 sys.path 引用。
  （注意：tencentcloud-sdk-python-soe 是旧版 SOE 的 SDK，不适用于 SOE-N）
- 录音模式：rec_mode=1 一次性上传完整音频（≤60s），适合后端"拿完整录音再评测"

安装（一次）：
    git clone --depth 1 https://github.com/TencentCloud/tencentcloud-speech-sdk-python.git \\
        scholar-admin/vendor/tencentcloud-speech-sdk-python
    pip install websocket-client requests      # SDK 依赖；readme 建议 websocket-client==0.48，新版实测兼容

用法：
    export TENCENTCLOUD_SECRETID=... TENCENTCLOUD_SECRETKEY=...   # 与 scholar-admin/config.py 同源
    python scripts/soe_n_verify.py --appid 1306xxx --audio test16k.mp3 \
        --ref-text "The quick brown fox jumps over the lazy dog" \
        [--sdk-dir 仓库根目录]   # 默认自动探测 scholar-admin/vendor/tencentcloud-speech-sdk-python

输出：on_recognition_complete 的完整 JSON，并打印关键评测字段。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("soe_n_verify")

# 复用 config.py 的凭据源：自动加载 scholar-admin/.env（不覆盖已 export 的环境变量）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

APPID = os.environ.get("TCB_APPID", "")  # 腾讯云账号 AppID，或 --appid 传入
SECRET_ID = os.environ.get("TENCENTCLOUD_SECRETID", "")
SECRET_KEY = os.environ.get("TENCENTCLOUD_SECRETKEY", "")
SESSION_TOKEN = os.environ.get("TENCENTCLOUD_SESSIONTOKEN", "")


class SoeNListener:
    """评测结果回调：最终结果在 on_recognition_complete 中返回"""

    def on_recognition_start(self, response):
        logger.info("评测开始 voice_id=%s", response.get("voice_id"))

    def on_intermediate_result(self, response):
        pass  # 录音模式（rec_mode=1）不关注中间结果

    def on_recognition_complete(self, response):
        print(json.dumps(response, ensure_ascii=False))
        for key in ("SuggestedScore", "PronAccuracy", "PronFluency", "PronCompletion"):
            if key in response:
                print(f"{key} = {response[key]}")
        words = response.get("Words")
        if words:
            print(f"Words 词级数量 = {len(words)}（含 Word/MatchTag）")

    def on_fail(self, response):
        logger.error("评测失败: %s", json.dumps(response, ensure_ascii=False))
        sys.exit(1)


def run(
    appid: str,
    audio_path: Path,
    ref_text: str,
    eval_mode: int = 1,
    voice_format: int = 2,
    engine: str = "16k_en",
    sdk_dir: Path | None = None,
) -> None:
    """录音模式（rec_mode=1）一次性上传完整音频做句级评测"""
    if not (SECRET_ID and SECRET_KEY):
        sys.exit(
            "未配置 TENCENTCLOUD_SECRETID/SECRETKEY（与 scholar-admin/config.py 同源，"
            "可写 scholar-admin/.env 或 export）"
        )
    if not audio_path.exists():
        sys.exit(f"音频不存在: {audio_path}")

    # 官方 SDK 是纯源码分发（PyPI 无包）：优先用 --sdk-dir/默认 vendor 目录，其次碰运气导入
    if sdk_dir is None:
        sdk_dir = (
            Path(__file__).resolve().parent.parent / "vendor" / "tencentcloud-speech-sdk-python"
        )
    sdk_dir = sdk_dir.expanduser().resolve()
    if sdk_dir.is_dir():
        sys.path.insert(0, str(sdk_dir))

    try:
        from common import credential
        from soe import speaking_assessment
    except ImportError:
        sys.exit(
            "缺少官方 SDK 源码，请先执行:\n"
            "  git clone --depth 1 https://github.com/TencentCloud/tencentcloud-speech-sdk-python.git \\\n"
            "      scholar-admin/vendor/tencentcloud-speech-sdk-python\n"
            "然后重试（或 --sdk-dir 指定仓库根目录）"
        )

    cred = credential.Credential(SECRET_ID, SECRET_KEY, SESSION_TOKEN or None)
    recognizer = speaking_assessment.SpeakingAssessment(appid, cred, engine, SoeNListener())
    recognizer.set_eval_mode(eval_mode)        # 1 = 句子
    recognizer.set_rec_mode(1)                 # 1 = 录音模式（一次性上传完整音频）
    recognizer.set_ref_text(ref_text)          # 句级 ≤30 词
    recognizer.set_text_mode(0)                # 0 = 普通文本
    recognizer.set_voice_format(voice_format)  # 2 = mp3（0=pcm / 1=wav / 2=mp3 / 4=speex）
    # recognizer.set_score_coeff(1.0)          # 苛刻指数 1.0~4.0（按年龄段调整，按需启用）

    recognizer.start()
    try:
        with open(audio_path, "rb") as f:
            recognizer.write(f.read())
    except Exception:
        logger.exception("音频上传异常")
        raise
    finally:
        recognizer.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="SOE-N 句级评测最小调用（F1-2 验证）")
    parser.add_argument("--appid", default=APPID, help="腾讯云账号 AppID（控制台账号信息）")
    parser.add_argument("--audio", required=True, help="音频路径（16k / 单声道 / mp3 或 wav）")
    parser.add_argument("--ref-text", required=True, help="评测参考文本（句级 ≤30 词）")
    parser.add_argument("--eval-mode", type=int, default=1, help="1=句子（默认）")
    parser.add_argument("--voice-format", type=int, default=2, help="2=mp3（默认；0=pcm/1=wav/2=mp3/4=speex）")
    parser.add_argument("--engine", default="16k_en", help="16k_en（默认）/ 16k_zh")
    parser.add_argument(
        "--sdk-dir",
        type=Path,
        default=None,
        help="官方 SDK 仓库根目录（默认自动探测 scholar-admin/vendor/tencentcloud-speech-sdk-python）",
    )
    args = parser.parse_args()

    if not args.appid:
        sys.exit("缺少 AppID：--appid 或环境变量 TCB_APPID（控制台账号信息）")
    run(
        args.appid, Path(args.audio), args.ref_text,
        args.eval_mode, args.voice_format, args.engine, args.sdk_dir,
    )


if __name__ == "__main__":
    main()
