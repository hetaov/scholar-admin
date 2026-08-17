#!/usr/bin/env python3
"""检查 scholar-admin 数据模型与目标模型(design.md)的差距。

命名策略:为避免与现有集合冲突、不影响线上数据,目标模型中凡是
与旧集合同名的集合一律加 _v2 后缀(如 textbook -> textbook_v2、
sentence -> sentence_v2);其余全新集合(如 chapter/lesson/skill 等)
本就不与旧表冲突,保持原名。

本脚本扫描 services/ 等目录下所有 collection 参数,识别集合名,与目标集合
清单对比,输出。集合名有两种写法都会识别:
  - 字符串字面量: collection="textbook_v2"
  - 本文件定义的常量引用: collection=CHAPTER(配合顶部 CHAPTER = "chapter")
    —— 常量需在同一 .py 文件内以顶层赋值定义(如 services/models_content.py),
    跨文件导入的常量不展开(其定义文件自身即可被扫描到)。输出:
  - 已存在集合(目标新表已建并被代码使用)
  - 缺失集合(目标新表待创建)
  - 待迁移集合(旧表仍在使用,标注迁移目标)
  - 已清理集合(旧表已不在代码中出现,可安全下线)

用法:
    python3 scripts/check_schema.py                  # 默认扫描项目根
    python3 scripts/check_schema.py path/to/services # 指定目录
退出码: 0 = 无差距, 1 = 存在差距(缺失或待迁移), 2 = 参数错误。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# 目标集合清单 —— 与 references/target-model.md 保持一致
# 注意:与旧集合同名的目标集合已加 _v2 后缀,避免影响现有数据
TARGET_COLLECTIONS = {
    "textbook_v2": "教材(替代旧 textbook)",
    "chapter": "章节",
    "lesson": "课(替代旧 unit)",
    "sentence_v2": "句子(替代旧 sentence)",
    "skill": "能力定义",
    "skill_state": "学者×句子×能力 状态",
    "study_attempt": "学习事件(append-only)",
    "study_session": "学习会话",
    "scholar_book": "学者×教材 关联",
    "knowledge_point": "知识点(可选)",
    # P2 扩展（2026-08-17，见 03-change/proposals/2026-08-16-P2-后续扩展功能完整设计.md）
    "badge": "徽章定义(P2/F8)",
    "scholar_badge": "学者已获得徽章(P2/F8)",
}

# 旧集合 → 迁移目标(旧表仍可能被现有代码使用,迁移完成后从代码中移除)
LEGACY_COLLECTIONS = {
    "textbook": "迁移至 textbook_v2",
    "unit": "迁移至 lesson",
    "paragraph": "内容并入 sentence_v2 或按需保留",
    "sentence": "迁移至 sentence_v2",
    "learning_mastery_tracking": "迁移至 skill_state + study_attempt",
}

# 集合名写法一:字符串字面量(collection="textbook" / collection: "textbook")
_LITERAL_RE = re.compile(r"""collection\s*[=:]\s*["']([^"']+)["']""")
# 集合名写法二:常量引用(collection=CHAPTER), 值需在同一文件顶层定义
_CONST_REF_RE = re.compile(r"""collection\s*[=:]\s*([A-Za-z_][A-Za-z0-9_]*)""")
# 模块顶层集合名常量定义, 如 CHAPTER = "chapter"(行首无缩进)
_CONST_DEF_RE = re.compile(r"""^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*["']([^"']+)["']\s*$""", re.MULTILINE)

_SKILL_DIR_MARK = "scholar-refactor-skill"  # 跳过本 skill 自身,避免字面量自匹配


def _scan_py_file(text: str, found: set[str]) -> None:
    """解析单个 .py 文件中的集合名: 字面量 + 本文件定义的常量引用。"""
    found.update(_LITERAL_RE.findall(text))
    consts: dict[str, str] = dict(_CONST_DEF_RE.findall(text))  # 本文件顶层常量 {名称: 值}
    names: list[str] = _CONST_REF_RE.findall(text)
    for name in names:
        if name in consts:
            found.add(consts[name])


def scan_collections(root: Path) -> set[str]:
    """扫描 root 下所有 .py 文件中出现的集合名(字面量 + 常量引用)。"""
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        if _SKILL_DIR_MARK in path.parts:
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        _scan_py_file(text, found)
    return found


def find_project_root() -> Path:
    """向上查找包含 services/ 目录的祖先作为项目根。"""
    p = Path(__file__).resolve().parent
    while not (p / "services").is_dir() and p.parent != p:
        p = p.parent
    return p if (p / "services").is_dir() else Path.cwd()


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else find_project_root()
    if not root.is_dir():
        print(f"[错误] 目录不存在: {root}")
        return 2

    current = scan_collections(root)
    # 仅保留"像集合名"的 token,忽略无关字符串
    current = {c for c in current if not c.startswith(("_", "{"))}

    matched = sorted(TARGET_COLLECTIONS.keys() & current)
    missing = sorted(TARGET_COLLECTIONS.keys() - current)
    migrating = sorted(current & LEGACY_COLLECTIONS.keys())
    cleaned = sorted(LEGACY_COLLECTIONS.keys() - current)

    print("=" * 60)
    print(f"扫描目录: {root}")
    print(f"当前实际使用集合 ({len(current)}): {', '.join(sorted(current)) or '-'}")
    print("-" * 60)
    print(f"[已存在] 目标新表已建 ({len(matched)})")
    for name in matched:
        print(f"  ✅ {name:24s} {TARGET_COLLECTIONS[name]}")
    print(f"[缺失] 目标新表待建 ({len(missing)})")
    for name in missing:
        print(f"  ❌ {name:24s} {TARGET_COLLECTIONS[name]}")
    print(f"[待迁移] 旧表仍在使用 ({len(migrating)})")
    for name in migrating:
        print(f"  🔁 {name:24s} {LEGACY_COLLECTIONS[name]}")
    print(f"[已清理] 旧表已下线 ({len(cleaned)})")
    for name in cleaned:
        print(f"  ✔  {name:24s} {LEGACY_COLLECTIONS[name]}")
    print("=" * 60)

    if not missing and not migrating:
        print("✅ 无差距:新表已就绪且旧表已全部迁移清理。")
        return 0
    print("⚠️  存在差距,请按 SKILL.md 中 references/execution-guide.md 的 Phase 顺序重构。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
