"""G0.2 单元测试 — audit 必审动作 12 → 17 类扩展（SOP DM-4）

覆盖 SOP §5 G0.2 验收标准 + 契约 §4.12.10(b) 必审清单第 ⑤ 组：

  - create_math_textbook / update_math_textbook / delete_math_textbook
  - import_math_nodes / manual_edit_summary

核心断言：
  1. REQUIRED_AUDIT_ACTIONS 可 import， len == 17
  2. MUST_AUDIT_ACTIONS 别名仍可 import（向后兼容，已有写审计不破坏），且 == REQUIRED_AUDIT_ACTIONS（同一集合或相同）
  3. 5 个新增 AUDIT_ACTION_* 常量存在，取值与契约 DM-4 分组 ⑤ 完全一致（snake_case 动词_对象对齐既有 generate_knowledge_summary / scan_upload 风格）
  4. 5 个新增值均在必审集合中
  5. 原 12 类必审动作（ADR-0011 5 + F2 3 + F1 1 + F4 3）仍在集合中（零破坏）
  6. 原 12 个 AUDIT_ACTION_* 常量值完全不变（零破坏）
"""
from __future__ import annotations


# ===========================================================================
# 1. REQUIRED_AUDIT_ACTIONS 别名 + 长度（契约 DM-4 计数：17 类）
# ===========================================================================


class TestRequiredAuditActionsLength:
    def test_required_audit_actions_import_and_len_17(self):
        """契约 DM-4：12 → 17 类，必审动作全集大小 = 17。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        assert REQUIRED_AUDIT_ACTIONS is not None
        assert len(REQUIRED_AUDIT_ACTIONS) == 17

    def test_must_audit_actions_alias_still_works_and_equals_required(self):
        """向后兼容：原有 MUST_AUDIT_ACTIONS 常量仍可 import（零破坏既有调用处），且内容等价于 REQUIRED。"""
        from services.audit import MUST_AUDIT_ACTIONS, REQUIRED_AUDIT_ACTIONS

        assert isinstance(MUST_AUDIT_ACTIONS, frozenset)
        # 两个集合的值完全等价（不管是否是同一个对象）
        assert set(MUST_AUDIT_ACTIONS) == set(REQUIRED_AUDIT_ACTIONS)


# ===========================================================================
# 2. 5 个新增 ACTION 常量 — 名称 / 取值（契约 §4.12.10(b) 第 ⑤ 组）
# ===========================================================================


class TestNewActionConstantsForGManagement:
    EXPECTED = {
        "AUDIT_ACTION_CREATE_MATH_TEXTBOOK": "create_math_textbook",
        "AUDIT_ACTION_UPDATE_MATH_TEXTBOOK": "update_math_textbook",
        "AUDIT_ACTION_DELETE_MATH_TEXTBOOK": "delete_math_textbook",
        "AUDIT_ACTION_IMPORT_MATH_NODES": "import_math_nodes",
        "AUDIT_ACTION_MANUAL_EDIT_SUMMARY": "manual_edit_summary",
    }

    def test_all_5_constants_are_importable_with_correct_values(self):
        """5 个常量都存在且值严格等于契约分组 ⑤ 的命名：snake_case 动词_对象，前缀对齐既有 ACTION_* 命名。"""
        import services.audit as audit

        for const_name, expected_value in self.EXPECTED.items():
            actual = getattr(audit, const_name, "<NOT_FOUND>")
            assert actual != "<NOT_FOUND>", f"常量缺失：{const_name}"
            assert actual == expected_value, (
                f"{const_name} 取值与契约不符："
                f"实际={actual!r}，期望={expected_value!r}"
            )

    def test_all_5_values_are_in_required_audit_actions(self):
        """新增 5 值必须全部包含在 REQUIRED_AUDIT_ACTIONS 必审全集里。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        for expected_value in self.EXPECTED.values():
            assert expected_value in REQUIRED_AUDIT_ACTIONS, (
                f"新增动作 {expected_value!r} 未加入 REQUIRED_AUDIT_ACTIONS 必审全集"
            )


# ===========================================================================
# 3. 向后兼容：原有 12 类动作 + 12 个常量值完全不变（零破坏）
# ===========================================================================


class TestExistingActionsNotBroken:
    # 原有 12 个动作全集：ADR-0011 5 + F2 3 + F1 1 + F4 3
    EXISTING_VALUES = {
        # ① ADR-0011 原生 5 类（F3 练习纸）
        "generate",
        "share",
        "download",
        "return",
        "modify",
        # ② F2 描述 3 类
        "edit_description",
        "draft_description",
        "adopt_description",
        # ③ F1 总结 1 类
        "generate_knowledge_summary",
        # ④ F4 扫描 3 类
        "scan_upload",
        "scan_classify",
        "scan_correct",
    }
    # 原有 12 个 AUDIT_ACTION_* 常量名 → 值
    EXISTING_CONST_MAP = {
        "AUDIT_ACTION_GENERATE": "generate",
        "AUDIT_ACTION_SHARE": "share",
        "AUDIT_ACTION_DOWNLOAD": "download",
        "AUDIT_ACTION_RETURN": "return",
        "AUDIT_ACTION_MODIFY": "modify",
        "AUDIT_ACTION_EDIT_DESCRIPTION": "edit_description",
        "AUDIT_ACTION_DRAFT_DESCRIPTION": "draft_description",
        "AUDIT_ACTION_ADOPT_DESCRIPTION": "adopt_description",
        "AUDIT_ACTION_GENERATE_KNOWLEDGE_SUMMARY": "generate_knowledge_summary",
        "AUDIT_ACTION_SCAN_UPLOAD": "scan_upload",
        "AUDIT_ACTION_SCAN_CLASSIFY": "scan_classify",
        "AUDIT_ACTION_SCAN_CORRECT": "scan_correct",
    }

    def test_existing_12_values_are_all_still_in_required(self):
        """新增 5 类不应替换或移除原有 12 类 — 必审全集仍包含所有历史值。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        missing = self.EXISTING_VALUES - set(REQUIRED_AUDIT_ACTIONS)
        assert not missing, (
            f"原有必审动作丢失！未出现在 REQUIRED_AUDIT_ACTIONS 中：{missing}"
        )

    def test_existing_12_constants_values_unchanged(self):
        """原有 12 个常量（AUDIT_ACTION_*）的值严格不变，避免破坏 audit.write_audit 已有调用。"""
        import services.audit as audit

        for const_name, expected_value in self.EXISTING_CONST_MAP.items():
            actual = getattr(audit, const_name, "<NOT_FOUND>")
            assert actual != "<NOT_FOUND>", f"原有常量丢失：{const_name}"
            assert actual == expected_value, (
                f"原有常量值被错误修改！{const_name}: "
                f"实际={actual!r}, 历史原值={expected_value!r}"
            )
