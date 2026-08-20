"""冷启动策略（设计文档 §5.6）— 先验默认 + 证据稀疏保护 + cold_start 标记

- §5.6.1 先验默认值：无历史时 SkillState 返回确定性先验（config 可调），零外部调用。
- §5.6.2 证据稀疏保护：attempt_count < MIN_EVIDENCE 时回写权重乘以
  `attempt_count / MIN_EVIDENCE`（如第 1 次尝试只贡献 1/3），防单次偶然污染。
- §5.6.5 数据契约：无历史时评估/会话接口返回 `"cold_start": true` 标记 + 先验默认，
  前端可据此展示引导（不配初始诊断，附录 B-5）。
"""
from __future__ import annotations

from config import COLD_START_DIFFICULTY, COLD_START_MASTERY, MIN_EVIDENCE

# 标准引导序列（§5.6.3）：无历史时 Activity 推荐回退该序列（非弱项驱动）
COLD_START_SEQUENCE = ("content", "shadowing", "translation", "listening")


def cold_start_prior(difficulty: int | None = None) -> dict:
    """返回冷启动先验默认（§5.6.1）。"""
    return {
        "mastery": COLD_START_MASTERY,
        "confidence": 0.0,  # 无证据
        "stability": 0.0,  # 不稳
        "difficulty": int(difficulty or COLD_START_DIFFICULTY),
    }


def sparse_evidence_weight(attempt_count: int) -> float:
    """证据稀疏打折系数（§5.6.2）：attempt_count < MIN_EVIDENCE 时线性打折。

    - 第 1 次尝试 → 1/3 权重；第 2 次 → 2/3；≥3 次 → 1.0（满权重）。
    - 冷启动期不触发"连续低可信降权"惩罚分支（§5.6.2 末句：只记录不惩罚）。
    """
    if attempt_count <= 0:
        return 0.0
    if attempt_count >= MIN_EVIDENCE:
        return 1.0
    return round(attempt_count / MIN_EVIDENCE, 4)


def is_sparse(attempt_count: int) -> bool:
    """证据是否稀疏（< MIN_EVIDENCE）。"""
    return attempt_count < MIN_EVIDENCE


def cold_start_flag(*, has_history: bool) -> bool:
    """`"cold_start": true` 标记（§5.6.5）：无历史时评估/会话接口返回该标记。"""
    return not has_history


async def has_skill_history(db, *, scholar_id: str, skill_code: str | None = None) -> bool:
    """是否有学习历史（§5.6.5）：skill_state 存在该学者记录（可限定 skill）。

    无历史 → 冷启动（返回先验默认 + "cold_start": true 标记）。
    """
    where: dict = {"scholar_id": scholar_id}
    if skill_code:
        where["skill_code"] = skill_code
    result = await db.query(collection="skill_state", where=where, limit=1)
    return bool(result.get("records"))
