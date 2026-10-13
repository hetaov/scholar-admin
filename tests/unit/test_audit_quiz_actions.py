"""单元测试 — audit 必审动作 24 → 27 类扩展（QZ 数学同类型题，ADR-0035 D8）

对应任务：docs_v1/数学/同类型题-原子任务清单.md B13；契约 data-model-contract §4.12.10(b) 第 ⑨ 组。

核心断言：
  1. `REQUIRED_AUDIT_ACTIONS` 可 import，len == 27（⑧ 组 24 类 + QZ 3 类）
  2. `MUST_AUDIT_ACTIONS` 别名仍可 import 且与 REQUIRED 等价（向后兼容）
  3. 新增 `AUDIT_ACTION_QUIZ_*` 3 个常量存在且取值与契约 ⑨ 组完全一致
  4. 3 个新值全部在必审集合内
  5. 原有 24 类必审动作与 24 个常量值零破坏
"""
from __future__ import annotations


# ===========================================================================
# 1. 必审动作全集长度（契约 DM-4 计数：24 → 27 类）
# ===========================================================================


class TestRequiredAuditActionsLength:
    def test_required_audit_actions_len_27(self):
        """契约 DM-4 ⑨ 组：QZ 数学同类型题 3 类并入后全集 = 27 类。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        assert REQUIRED_AUDIT_ACTIONS is not None
        assert len(REQUIRED_AUDIT_ACTIONS) == 27

    def test_must_audit_actions_alias_equals_required(self):
        """向后兼容：`MUST_AUDIT_ACTIONS` 历史别名仍可用且内容等价于 REQUIRED。"""
        from services.audit import MUST_AUDIT_ACTIONS, REQUIRED_AUDIT_ACTIONS

        assert isinstance(MUST_AUDIT_ACTIONS, frozenset)
        assert set(MUST_AUDIT_ACTIONS) == set(REQUIRED_AUDIT_ACTIONS)


# ===========================================================================
# 2. QZ 组 3 个 ACTION 常量 — 名称 / 取值（ADR-0035 D8）
# ===========================================================================


class TestNewActionConstantsForQuiz:
    EXPECTED = {
        "AUDIT_ACTION_QUIZ_RECOGNIZE": "quiz_recognize",  # 接口 28 拍照题识别
        "AUDIT_ACTION_QUIZ_GENERATE": "quiz_generate",    # 接口 29 出同类型题
        "AUDIT_ACTION_QUIZ_RENDER": "quiz_render",        # 接口 30 确认打印 / 异步渲染
    }

    def test_3_constants_importable_with_correct_values(self):
        """3 个常量均存在且取值严格等于契约 ⑨ 组命名（snake_case，动词_对象）。"""
        import services.audit as audit

        for const_name, expected_value in self.EXPECTED.items():
            actual = getattr(audit, const_name, "<NOT_FOUND>")
            assert actual != "<NOT_FOUND>", f"常量缺失：{const_name}"
            assert actual == expected_value, (
                f"{const_name} 取值与契约不符：实际={actual!r}，期望={expected_value!r}"
            )

    def test_3_values_are_in_required_audit_actions(self):
        """「未登记的动作」告警必须消失 —— 3 值均在必审全集中。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        for expected_value in self.EXPECTED.values():
            assert expected_value in REQUIRED_AUDIT_ACTIONS, (
                f"新增动作 {expected_value!r} 未加入 REQUIRED_AUDIT_ACTIONS 必审全集"
            )


# ===========================================================================
# 3. 向后兼容：原有 24 类动作与常量值零破坏
# ===========================================================================


class TestExistingActionsNotBroken:
    EXISTING_VALUES = {
        # ① ADR-0011 原生 5 类
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
        # ⑤ G 管理端 5 类
        "create_math_textbook",
        "update_math_textbook",
        "delete_math_textbook",
        "import_math_nodes",
        "manual_edit_summary",
        # ⑥ E 英语语句管理 3 类
        "create_english_sentences",
        "edit_english_sentence",
        "delete_english_sentence",
        # ⑦ SG 语句组管理 3 类
        "create_sentence_group",
        "edit_sentence_group",
        "delete_sentence_group",
        # ⑧ ED 英语批量去重 1 类
        "deduplicate_english_sentences",
    }

    def test_existing_24_values_are_all_still_in_required(self):
        """新增 3 类不应替换或移除原有 24 类 —— 必审全集仍包含所有历史值。"""
        from services.audit import REQUIRED_AUDIT_ACTIONS

        missing = self.EXISTING_VALUES - set(REQUIRED_AUDIT_ACTIONS)
        assert not missing, f"原有必审动作丢失：{missing}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
