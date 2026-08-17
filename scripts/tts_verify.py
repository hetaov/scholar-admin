#!/usr/bin/env python3
"""腾讯云 TTS（TextToVoice）最小调用封装（F3/3.2 验证用）

接入要点：
- 传输：TC3 HTTP REST（tts.tencentcloudapi.com），官方 PyPI 包 tencentcloud-sdk-python-tts
  （与 SOE-N 的 WSS + 源码分发 SDK 不同，TTS 走标准 TencentCloud SDK 全家桶）
- 参数（对齐契约 api-contract §3.5）：PrimaryLanguage=2（英文）、Codec=mp3、SampleRate=16000、
  ModelType=1（普通音色）、VoiceType=101001（英文默认音色）
- 文本限制：TextToVoice.Text 最大 150 字节（UTF-8）；契约层上限 200 字符
- ✅ 2026-08-17 验证通过：
  a) 首次：UnsupportedOperation.ServerNotOpen（TTS 服务未开通）→ 需先在控制台开通"语音合成"
     （官方指引：https://cloud.tencent.com/document/product/1073/56640），类比 F1-1 开通 SOE-N；
  b) 开通后曾报 UnsupportedOperation.PkgExhausted（资源包配额耗尽）→ 配额恢复后已复测通过，
     2026-08-17 成功合成 "The quick brown fox jumps over the lazy dog"（mp3/16k，见验证记录）

用法：
    export TENCENTCLOUD_SECRETID=... TENCENTCLOUD_SECRETKEY=...   # 与 scholar-admin/config.py 同源
    python scripts/tts_verify.py --text "The quick brown fox jumps over the lazy dog"
    python scripts/tts_verify.py --text "..." --out /tmp/tts.mp3    # 可选：保存 mp3 到本地

输出：Audio base64 长度 + 解码后 mp3 头校验（ID3 / frame sync）+ RequestId/SessionId。
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

# 复用 services.tts 的 Provider（凭据源与 config.py 一致，自动加载 scholar-admin/.env）
from services.tts import get_tts_provider


def run(text: str, out_path: Path | None) -> None:
    provider = get_tts_provider()
    if not provider.available:
        sys.exit(
            "未配置 TENCENTCLOUD_SECRETID/SECRETKEY（与 scholar-admin/config.py 同源，"
            "可写 scholar-admin/.env 或 export）"
        )
    raw = provider.synthesize(text)
    if raw is None:
        sys.exit(
            "合成失败（TTS_UNAVAILABLE）。若提示 TTS service is not open，请先在控制台开通"
            "「语音合成」（https://cloud.tencent.com/document/product/1073/56640）后重试"
        )
    audio = raw["Audio"]
    audio_bytes = base64.b64decode(audio)
    id3 = audio_bytes[:3] == b"ID3"
    frame_sync = audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0
    print(f"audio base64 长度 = {len(audio)}")
    print(f"audio bytes = {len(audio_bytes)}")
    print(f"mp3 头校验: ID3={id3} | frame sync={frame_sync}")
    print(f"RequestId = {raw.get('RequestId')}")
    print(f"SessionId = {raw.get('SessionId')}")
    if out_path is not None:
        out_path.write_bytes(audio_bytes)
        print(f"已保存: {out_path}")
    print("TTS VERIFY OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="腾讯云 TTS 最小调用（F3/3.2 验证）")
    parser.add_argument("--text", required=True, help="待合成英文文本（≤150 字节，契约上限 200 字符）")
    parser.add_argument("--out", type=Path, default=None, help="可选：保存 mp3 到本地文件")
    args = parser.parse_args()
    run(args.text, args.out)


if __name__ == "__main__":
    main()
