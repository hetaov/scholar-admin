---
name: scholar-data-model-refactor
description: >
  按 docs/english-learning-plan/design.md 的四层架构设计（内容模型 + 学习状态 + 学习事件 + 聚合），
  把 scholar-admin 项目当前的平铺数据模型（textbook/unit/paragraph/sentence +
  learning_mastery_tracking）逐步重构为生产级模型（Scholar/ScholarBook/Book/Chapter/Lesson/
  Sentence + Skill/SkillState/StudyAttempt/StudySession + KnowledgePoint）。
  本 skill 在用户要求"按 design.md 重构数据模型"、"落地四层架构"、"拆分学习状态与学习事件"、
  "迁移 learning_mastery_tracking"、"把 design 转化为可执行步骤"时使用。
  使用前必须保留 design.md 原文，本 skill 只新增目录不改动该文件。
---

# Scholar 数据模型分步重构 Skill

## 目标

将 scholar-admin 从当前"教材内容 + 平铺学习记录"模型，重构为 design.md 定义的四层架构：

> **教材结构决定"学什么"（内容模型），Skill 决定"学会什么能力"，StudyAttempt 记录"学过什么"（事件），SkillState 表示"现在会多少"（状态），Progress/Mastery 从这些数据向上聚合到 Lesson → Chapter → Book。**

## 核心原则（重构时必须遵守）

1. **内容与状态分离**：Book/Chapter/Lesson/Sentence 只描述教材内容，不保存任何用户进度。
2. **状态与事件分离**：SkillState 保存"当前状态"（可覆盖），StudyAttempt 保存"历史事件"（append-only，永不修改）。
3. **Skill 维度独立**：一个 Sentence 的 Translation/Listening/Speaking 等能力各自独立记录，禁止在 Sentence 上平铺状态字段。
4. **进度动态计算**：Book 自身不存用户进度，Lesson/Chapter/Book 的 Progress/Mastery 由 SkillState 向上聚合。
5. **兼容过渡**：每步都保留旧字段/旧接口，满足"完成一步、验收一步、不破坏线上"。
6. **新旧集合隔离**：与旧集合同名的目标集合一律加 `_v2` 后缀（`textbook_v2`/`sentence_v2`），新表独立创建、旧表照常服务，互不影响，方便之后把旧数据迁移过来（详见 `references/target-model.md` 第 0 节命名策略）。
7. **测试同步维护**：每个新增集合/接口/纯函数必须伴随测试用例（单元或集成），并同步更新 `references/testing-guide.md` 的测试矩阵；验收标准必须包含"新增/更新测试通过"。禁止只改功能不碰测试。
8. **清理时机（延迟清理红线）**：旧表与旧数据**严禁提前清理**。迁移阶段（Phase 1-5）只写新表、旧表只读，任何 Phase 都不得删除旧表或旧数据（Phase 3 的"停写旧表"仅指停止写入，不等于删除）。旧表下线/旧数据清理只允许发生在 **Phase 6 末尾**，且必须同时满足前置条件：新表与旧数据迁移结束、迁移核对通过、全量测试通过——顺序为"先迁移核对 → 再全量回归 → 最后清理"，缺一不可。

## 工作流程（严格按序执行，一步验收通过再进入下一步）

执行本 skill 时，按 `references/execution-guide.md` 的 Phase 0 ~ Phase 6 顺序逐步进行。
每步都包含：目标、改动文件/集合、操作步骤、验收标准。**只有当前步骤验收通过才可进入下一步。**

阶段总览：

| Phase | 名称 | 交付物 | 依赖 |
|-------|------|--------|------|
| 0 | 基线确认 | 测试全绿 + 现状快照 | 无 |
| 1 | 内容模型分层 | textbook_v2/chapter/lesson/sentence_v2 新集合 + 层级引用 | Phase 0 |
| 2 | 能力模型 | skill 集合 + skill_state 集合 + 状态上报接口 | Phase 1 |
| 3 | 事件模型 | study_attempt + study_session 集合 + 事件上报接口 | Phase 2 |
| 4 | 聚合计算 | progress/mastery 聚合模块 + 重构 tracking/stats | Phase 2/3 |
| 5 | 关联与计划 | scholar_book 集合 + 学习计划字段 | Phase 1 |
| 6 | 清理与文档 | **迁移核对 + 全量回归通过后**再下线旧表（learning_mastery_tracking/textbook/unit/paragraph/sentence）、更新文档、全量测试 | 全部 |

## 参考资源

- `references/target-model.md` — design.md 的 CloudBase NoSQL 目标集合设计（含字段映射、唯一键、索引建议、状态枚举）。
- `references/current-state.md` — 当前数据模型与代码盘点、与目标的差距清单。
- `references/execution-guide.md` — 每个 Phase 的可执行细节（文件级改动、集合结构、接口签名、验收标准）。
- `references/testing-guide.md` — 测试体系规范：unit/integration 分层、目录结构与命名、各 Phase 测试矩阵、维护规则。
- `scripts/check_schema.py` — 差距检查脚本：扫描 `services/` 下实际使用的集合名，与目标集合清单对比，输出"已存在/缺失/需废弃"清单。

## 执行指引

1. **开始前**：运行 `python3 -m pytest tests/ -q` 确认基线全绿；若环境缺依赖（如 langgraph），先 `pip install -r requirements.txt` 或记录为既有问题。
2. **每步开始**：阅读 `references/execution-guide.md` 中对应 Phase，按其"改动文件"清单定位代码。
3. **每步结束**：按该 Phase 的"验收标准"逐项自检，并补充/更新**单元测试与集成测试**（结构、用例模板、维护规则见 `references/testing-guide.md`；用 `python3 -m pytest tests/ -q` 确认新增测试通过）。
4. **数据迁移**：涉及 CloudBase 集合的 Phase（1/2/3/5），先执行 `scripts/check_schema.py` 输出差距清单，再按 execution-guide 中的迁移命令样例操作（先备份、小批量、可回滚）。**迁移只写新表、旧表只读，任何 Phase 都不得删除旧表/旧数据**；旧表下线与旧数据清理仅允许在 Phase 6 末尾，且必须在迁移核对与全量测试通过之后执行（见核心原则 8）。
5. **结束**：全部 Phase 完成后，运行全量测试与 `scripts/check_schema.py`，确认差距为 0，更新 README 与 docs 说明。
