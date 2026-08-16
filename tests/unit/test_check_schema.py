"""check_schema.py(差距脚本)单元测试。

覆盖核心识别逻辑:集合名的两种写法(字符串字面量 / 本文件顶层常量引用)、
未定义常量不误报、skill 目录自跳过、以及项目根扫描结果与目标模型一致。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs" / "english-learning-plan" / "scholar-refactor-skill"
    / "scripts" / "check_schema.py"
)


@pytest.fixture(scope="module")
def check_schema():
    """加载 docs 下的 check_schema.py 脚本为模块。"""
    spec = importlib.util.spec_from_file_location("check_schema", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_schema"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _scan_py_file: 单文件集合名识别
# ---------------------------------------------------------------------------


def test_literal_collection(check_schema):
    found: set[str] = set()
    check_schema._scan_py_file('await db.query(collection="foo", where={})', found)
    assert found == {"foo"}


def test_colon_literal_collection(check_schema):
    found: set[str] = set()
    check_schema._scan_py_file('db.query(collection: "bar")', found)
    assert found == {"bar"}


def test_const_reference_collection(check_schema):
    text = 'CHAPTER = "chapter"\nawait db.query(collection=CHAPTER, where={})'
    found: set[str] = set()
    check_schema._scan_py_file(text, found)
    assert found == {"chapter"}


def test_undefined_const_not_collected(check_schema):
    """collection=name 中 name 若是函数参数等未定义常量, 不纳入集合名。"""
    found: set[str] = set()
    check_schema._scan_py_file("await db.query(collection=name, where={})", found)
    assert found == set()


def test_imported_const_not_collected(check_schema):
    """跨文件导入的常量在本文件不展开(定义文件自身会被扫描到)。"""
    text = (
        "from services.models_content import CHAPTER\n"
        "await db.query(collection=CHAPTER, where={})"
    )
    found: set[str] = set()
    check_schema._scan_py_file(text, found)
    assert found == set()


# ---------------------------------------------------------------------------
# scan_collections: 目录扫描
# ---------------------------------------------------------------------------


def test_scan_collections_tmp(tmp_path, check_schema):
    (tmp_path / "a.py").write_text(
        'x = db.query(collection="bar")', encoding="utf-8"
    )
    (tmp_path / "b.py").write_text(
        'CHAPTER = "chapter"\ny = db.query(collection=CHAPTER)', encoding="utf-8"
    )
    assert check_schema.scan_collections(tmp_path) == {"bar", "chapter"}


def test_scan_collections_skips_skill_dir(tmp_path, check_schema):
    skill_dir = tmp_path / "scholar-refactor-skill"
    skill_dir.mkdir()
    (skill_dir / "x.py").write_text(
        'db.query(collection="skill")', encoding="utf-8"
    )
    assert check_schema.scan_collections(tmp_path) == set()


def test_scan_collections_skips_hidden_dirs(tmp_path, check_schema):
    hidden = tmp_path / ".venv"
    hidden.mkdir()
    (hidden / "x.py").write_text(
        'db.query(collection="sentence_v2")', encoding="utf-8"
    )
    assert check_schema.scan_collections(tmp_path) == set()


# ---------------------------------------------------------------------------
# 项目根扫描: 与 target-model 的目标状态对齐
# ---------------------------------------------------------------------------


def test_project_root_matches_phase6(check_schema):
    """Phase 6 完成后: 4 个新表已使用, 5 个旧表已全部清理(无引用)。"""
    root = check_schema.find_project_root()
    found = check_schema.scan_collections(root)
    assert {"textbook_v2", "chapter", "lesson", "sentence_v2"} <= found
    assert {"textbook", "unit", "paragraph", "sentence", "learning_mastery_tracking"}.isdisjoint(found)
