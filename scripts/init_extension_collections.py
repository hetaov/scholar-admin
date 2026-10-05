"""一次性脚本：创建英文语句扩展相关集合与索引（幂等）

创建五个集合：
  - english_extension_point（AI 抽取结果 / 缓存；point_key 幂等）      §4.24
  - extension_task（抽取 / 评测统一任务；24h TTL 由后台 cleanup 清理） §4.25
  - extension_review（学习者判断 overlay 补丁）                       §4.26
  - extension_review_log（学习历史，append-only）                     §4.27
  - extension_round（**语言点造句多轮会话**，服务端有状态；24h TTL）    §4.28

索引：
  - english_extension_point : point_key（唯一）、sentence_id
  - extension_task          : task_id（唯一）、expires_at
  - extension_review        : (scholar_id, sentence_id) **唯一**、scholar_id
  - extension_review_log    : log_id（唯一）、(scholar_id, sentence_id)、(scholar_id, at)
  - extension_round         : round_id（唯一）、(scholar_id, sentence_id)、expires_at

用法（在 scholar-admin 项目根目录执行）：
    python -m scripts.init_extension_collections
    # 或
    python scripts/init_extension_collections.py

已存在集合 / 同名索引时直接跳过（幂等，可重复执行）。
集合名可通过环境变量 EXTENSION_POINT_COLLECTION / EXTENSION_TASK_COLLECTION /
EXTENSION_REVIEW_COLLECTION / EXTENSION_REVIEW_LOG_COLLECTION /
EXTENSION_ROUND_COLLECTION 覆盖。
需要 CloudBase 环境变量（.env / cloudbaserc.json 注入），本地若无凭据会提示。

索引创建失败只告警不中断：唯一索引是写侧约束的兜底（E4' 全量替换自身即幂等），
不是功能前置条件；若云端不支持 CREATEINDEXES，功能仍可用，需在控制台手工补建。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402  加载 EXTENSION_*_COLLECTION
from services.dependencies import get_db  # noqa: E402


# (集合名, [索引 spec, ...])，按 data-model-contract §4.24~§4.28
COLLECTION_INDEXES: list[tuple[str, list[dict]]] = [
    (
        config.EXTENSION_POINT_COLLECTION,
        [
            {"key": {"point_key": 1}, "name": "point_key", "unique": True},
            {"key": {"sentence_id": 1}, "name": "sentence_id"},
        ],
    ),
    (
        config.EXTENSION_TASK_COLLECTION,
        [
            {"key": {"task_id": 1}, "name": "task_id", "unique": True},
            {"key": {"expires_at": 1}, "name": "expires_at"},
        ],
    ),
    (
        config.EXTENSION_REVIEW_COLLECTION,
        [
            {
                "key": {"scholar_id": 1, "sentence_id": 1},
                "name": "scholar_sentence",
                "unique": True,
            },
            {"key": {"scholar_id": 1}, "name": "scholar_id"},
        ],
    ),
    (
        config.EXTENSION_REVIEW_LOG_COLLECTION,
        [
            {"key": {"log_id": 1}, "name": "log_id", "unique": True},
            {
                "key": {"scholar_id": 1, "sentence_id": 1},
                "name": "scholar_sentence",
            },
            {"key": {"scholar_id": 1, "at": -1}, "name": "scholar_at"},
        ],
    ),
    (
        config.EXTENSION_ROUND_COLLECTION,
        [
            {"key": {"round_id": 1}, "name": "round_id", "unique": True},
            {
                "key": {"scholar_id": 1, "sentence_id": 1},
                "name": "scholar_sentence",
            },
            {"key": {"expires_at": 1}, "name": "expires_at"},
        ],
    ),
]


async def main() -> None:
    db = get_db()
    for name in [c for c, _ in COLLECTION_INDEXES]:
        if await db.check_collection(name):
            print(f"集合 `{name}` 已存在，跳过创建（幂等）")
            continue
        await db.create_collection(name)
        print(f"集合 `{name}` 创建成功")

    for name, indexes in COLLECTION_INDEXES:
        for spec in indexes:
            try:
                await db.create_indexes(name, [spec])
                print(f"索引 `{name}.{spec['name']}` 就绪（unique={spec.get('unique', False)}）")
            except Exception as e:  # noqa: BLE001  索引失败不中断，仅告警
                print(
                    f"[WARN] 索引 `{name}.{spec['name']}` 创建失败：{e}\n"
                    f"       功能不受影响（写侧自带幂等），请在 CloudBase 控制台手工补建"
                )


if __name__ == "__main__":
    asyncio.run(main())
