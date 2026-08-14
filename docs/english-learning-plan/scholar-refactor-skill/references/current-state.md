# 当前状态盘点（基线快照）

> 本文件是重构**开始前**的现状记录。随 Phase 0 确认，后续每个 Phase 完成后应回到本文件核对差距清单。
> 验证方式：`python3 scripts/check_schema.py`（位于本 skill `scripts/` 下）。

## 1. 技术栈

- **后端**：FastAPI（Python 3.12+），入口 `main.py`，路由挂载于 `services/routes_*.py`。
- **数据库**：腾讯云 CloudBase（NoSQL 文档型），数据访问封装在 `services/database.py`（`db.query / db.insert / db.update`），依赖注入 `services/dependencies.py` 的 `get_db()`。
- **测试**：`pytest`，用例在 `tests/`（现有 `tests/test_tracking_stats.py`）。
- 已知环境问题：本地**未安装 `langgraph`**（`import langgraph` 报 ModuleNotFoundError），会导致 `import main` 失败；测试直接导入 `services.*` 不依赖它，不影响基线测试。后续需要运行 `main.py` 时先执行 `pip install -r requirements.txt`。

## 2. 现有集合（5 个）

| 集合 | 用途 | 关键字段（现状） | 备注 |
|------|------|------------------|------|
| `textbook` | 教材 | `_id`、`title`、`text_book_id` | 字段名为 `text_book_id`（下划线）；目标新表为 `textbook_v2`，旧表保持不动 |
| `unit` | 课 | `unit_id`、`title`、`text_book_id` | 目标新表为 `lesson`（旧 unit 数据迁移过去） |
| `paragraph` | 段落 | 文本块 | 目标模型未单列 paragraph，内容并入 `sentence_v2` 或按需保留 |
| `sentence` | 句子 | `sentence_id`、`unit_id`、`text_book_id`、`order`、`text` | **无 chapter/lesson 引用**；目标新表为 `sentence_v2`，旧表保持不动 |
| `learning_mastery_tracking` | 学习追踪 | `scholar_id`、`sentence_id`、`time_spent`、`status`、`score`、`mastery`、`last_study_time` | 平铺"一学一记录"，目标由 `skill_state` + `study_attempt` 取代 |

> **命名策略**：为避免影响现有数据，目标集合中与旧集合同名的统一加 `_v2`（`textbook_v2`/`sentence_v2`）；其余全新集合（chapter/lesson/skill/skill_state/study_attempt/study_session/scholar_book/knowledge_point）不与旧表冲突，保持原名。新旧并行走，**旧表/旧数据严禁提前清理**——Phase 6 数据迁移核对 + 全量测试通过后，才统一切读并下线旧表。

## 3. 现有接口（与本次重构相关）

| 接口 | 文件 | 说明 | 重构去向 |
|------|------|------|----------|
| `GET /tracking/{scholar_id}` | routes_tracking.py | 按学者查 learning_mastery_tracking | Phase 2/4 改查 skill_state |
| `POST /tracking/stats` | routes_tracking.py | 上报 record_list 计算进度（tracking_stats.py 纯函数） | Phase 4 改为从 skill_state/study_attempt 聚合；`tracking_stats.py` 保留为纯函数测试目标 |
| `POST /build/textbook` 等构建接口 | routes_build.py / routes_vision.py | 写 textbook/unit/paragraph/sentence | Phase 1 扩展写 chapter/lesson |

## 4. 与目标的差距清单（重构总账）

| # | 差距 | 影响 | 对应 Phase |
|---|------|------|-----------|
| G1 | 无 `chapter` 集合；sentence/unit 无 chapter 引用 | 无法按章节管理内容与进度 | Phase 1 |
| G2 | 无 `lesson` 集合；`unit` 语义与设计不符 | 内容层级缺一层 | Phase 1 |
| G3 | 字段名不统一：`text_book_id` vs `textbook_id`，`unit_id` vs `lesson_id` | 查询/接口易错 | Phase 1（过渡期双写） |
| G4 | 无 `skill` / `skill_state`，状态平铺在 learning_mastery_tracking | 无法按"能力"独立度量；学习次数、复习调度无落点 | Phase 2 |
| G5 | 无 `study_attempt` / `study_session`，历史事件不可追溯 | 时间/次数统计不可重算，滚动学习调度缺数据源 | Phase 3 |
| G6 | 进度靠客户端上报一次性计算（tracking/stats 的 record_list） | 服务端无可信状态源，聚合口径不可复现 | Phase 4 |
| G7 | 无 `scholar_book`，断点续学/教材级状态缺失 | 无法表达"学者-教材"关系 | Phase 5 |
| G8 | `learning_mastery_tracking` 仍在多条代码路径被读写（routes_tracking / routes_admin / routes_dialogue / routes_build） | 双写维护成本高 | Phase 6 废弃 |

## 5. 代码引用地图（重构改动前必看，Phase 0 已于 2026-08-13 重新生成存档）

- `services/routes_tracking.py`：读 `learning_mastery_tracking`（`GET /tracking/{scholar_id}`，行 29）；`POST /tracking/stats` 读 `sentence`/`unit`（按 `text_book_id` 查询，行 101/125）。
- `services/routes_build.py`：写 `textbook`/`unit`/`paragraph`/`sentence`（构建与 repair，行 30-120、505-564）；写 `learning_mastery_tracking`（repair 补齐，行 641/673）。
- `services/routes_admin.py`：批量删除 `textbook`/`unit`/`paragraph`/`sentence`/`learning_mastery_tracking`（行 37-113）；合并教材（读 sentence、更新 text_book_id）。
- `services/routes_dialogue.py`：读 `learning_mastery_tracking`（行 49-51）、读 `sentence`（行 69）。
- `services/routes_vision.py`：写 `unit`/`paragraph`/`sentence`（`text_book_id` 字段，行 27-47）。
- `services/routes_system.py`：集合清单与索引定义（含 sentence/unit/paragraph/text_book_id、learning_mastery_tracking）。
- `services/tracking_stats.py`：纯函数统计（保留，作为 Phase 4 聚合模块的参照与测试对象）。
- `tests/test_tracking_stats.py`：覆盖 tracking_stats 与 /tracking/stats 接口。

> 重新生成命令（本机无 `rg`，用 grep）：`grep -rn --include="*.py" -E "learning_mastery_tracking|text_book_id|paragraph" services/`

## 6. Phase 0 基线确认记录（2026-08-13）

- 全量测试：`python3 -m pytest tests/ -q` → **42 passed**（既有 32 + 新增 10：unit 6 + integration 4），用例清单快照存档 `/tmp/baseline_tests.txt`。
- `check_schema.py`：现状 5 集合、**缺失 10**（textbook_v2/chapter/lesson/sentence_v2/skill/skill_state/study_attempt/study_session/scholar_book/knowledge_point）、**待迁移 5**（textbook/unit/paragraph/sentence/learning_mastery_tracking）—— 与第 4 节差距清单一致。
- 环境：`langgraph` 未安装（不阻塞测试，运行 main 前需安装）。
- 结论：**基线已锁定**，无既有失败项，可以进入 Phase 1（内容模型分层）。

## 7. Phase 1 完成记录（2026-08-13，内容模型分层）

### 目标达成
内容层级 `textbook_v2 → chapter → lesson → sentence_v2` 已落地：新内容写入新表，旧表 `textbook`/`unit`/`paragraph`/`sentence` 保持只读不动（双写过渡）。

### 新增集合与文件
| 项 | 说明 |
|----|------|
| `textbook_v2` | 教材主表（替代 textbook，`version=1` + 冗余计数） |
| `chapter` | 章（`textbook_id` + `order`） |
| `lesson` | 课（`chapter_id`；过渡期 `lesson_id` 沿用旧 `unit_id` 值） |
| `sentence_v2` | 句子（`chapter_id` + `lesson_id`；过渡期保留 `unit_id`/`text_book_id`，Phase 6 移除） |
| `services/models_content.py` | 查询辅助 + 文档构建纯函数 + `write_content_v2` 双写 |
| `services/migrate_content_v2.py` | 幂等迁移脚本（**只写新表、只读旧表**） |
| `services/routes_build.py` | `_write_to_db` 双写新表，返回新增 `"v2"` 统计 |
| `services/routes_vision.py` | 视觉识别双写新表（无教材不建 textbook_v2） |
| `services/routes_system.py` | `/collections` 新增 4 个新表清单 |

### 迁移红线遵守
- 迁移脚本只写新表，旧表仅读取，**任何路径均未删除/修改旧表**（见 `tests/unit/test_migrate_content_v2.py` 旧表数据原样保留断言）。
- 旧表下线仅允许 Phase 6 末尾（前置：迁移核对通过 + 全量回归通过）。

### 测试与差距核对
- 全量测试：`python3 -m pytest tests/ -q` → **78 passed**（Phase 0 基线 42 + Phase 1 新增 27 + check_schema 识别测试 9）。
- `check_schema.py`（已支持常量引用识别）输出：**已存在 4**（textbook_v2/chapter/lesson/sentence_v2）、**缺失 6**（skill/skill_state/study_attempt/study_session/scholar_book/knowledge_point，属 Phase 2/3/5）、**待迁移 5**（textbook/unit/paragraph/sentence/learning_mastery_tracking）、**已清理 0**。

### 验收核对（对照 execution-guide Phase 1）
- [x] chapter/lesson 可按 `textbook_id + order` 有序查询（`get_chapters`/`get_lessons` 单测覆盖）。
- [x] `sentence_v2` 每条含 `chapter_id` + `lesson_id`，且与旧 `unit_id` 对应关系正确（集成测试断言 `lesson_id == unit_id`）。
- [x] 旧 `textbook`/`sentence` 未被修改（迁移测试 `== SEED` 断言）。
- [x] 旧接口行为不变，测试全绿（`tests/test_tracking_stats.py` 仍在 42 基线内）。
- [x] 新增测试覆盖构建 chapter/lesson、sentence_v2 引用字段完整性。

### 待办（进入真实环境时）
1. 在 CloudBase 控制台为 `textbook_v2`（textbook_id）、`chapter`（textbook_id + order）、`lesson`（chapter_id）、`sentence_v2`（chapter_id/lesson_id）建索引。
2. 运行存量迁移（小批量、可回滚）：`python3 -m services.migrate_content_v2`。
3. 迁移后用 `scripts/check_schema.py` + 抽样核对新旧数据一致性（覆盖 5% 样本）。

## 8. Phase 2 完成记录（2026-08-13，能力模型）

### 目标达成
"学者 × 句子 × 能力"状态模型已落地：`skill_state` 按复合键 `{scholar_id}_{sentence_id}_{skill_code}` 单条存储，`attempt_count`（学习次数）与 `last_studied_at`/`next_review_at`（滚动调度）有明确落点；新写入统一英文状态枚举，旧中文状态词经归一化收敛。

### 新增集合与文件
| 项 | 说明 |
|----|------|
| `skill` | 能力定义种子数据（translation/listening/speaking/reading + mastery/learned 阈值），`seed_skills` 幂等预置 |
| `skill_state` | 学者×句子×能力 当前状态，复合键 `{scholar_id}_{sentence_id}_{skill_code}` |
| `services/models_learning.py` | 状态枚举 + 中文归一化（`normalize_status`）+ 滚动调度公式（`review_interval_seconds`/`compute_next_review_at`）+ `upsert_skill_state` + `seed_skills` |
| `services/routes_state.py` | `POST /tracking/state` 上报接口（status/score/mastery/time_spent，返回最新状态） |
| `services/migrate_learning.py` | `learning_mastery_tracking → skill_state` 幂等迁移（**只写新表、只读旧表**） |
| `services/routes_tracking.py` | `GET /tracking/{scholar_id}` 优先查 skill_state，无记录回退旧表（过渡兼容） |
| `services/routes_system.py` | `/collections` 新增 `skill`/`skill_state` 清单 |

### 关键设计
- **upsert 幂等**：同复合键重复上报只累加 `attempt_count`、刷新 `last_studied_at`，不产生新记录。
- **滚动调度**：`next_review_at = last_studied_at + interval(attempt_count, mastery_score)`；基础间隔 1/3/7/14/30 天随次数递增，mastery ≥ 80 间隔 ×1.5，< 60 间隔 ÷2 且状态置 `review_due`。
- **状态收敛**：中文（已学/已学会/已掌握/学习中/未学…）→ 英文枚举（learned/mastered/learning/not_started/review_due）；显式"已掌握/已学"优先保留，低掌握度推导 `review_due`。
- **迁移映射**：`status`→`normalize_status`、`score/mastery`→`mastery_score(0-100)`、`study_count`→`attempt_count`（缺省 1）、`last_study_time`→`last_studied_at`、`lesson_id` 按 `sentence_id` 回填旧 `unit_id`。

### 测试与差距核对
- 全量测试：`python3 -m pytest tests/ -q` → **124 passed**（Phase 1 后 78 + Phase 2 新增 46：unit 35 + integration 6 + migrate 5）。
- 新增测试覆盖：upsert 幂等性、attempt_count 累加、复合键冲突（不同 skill_code/sentence 各自独立）、中文状态词转换、滚动调度边界、迁移幂等 + 旧表原样保留、接口参数校验。
- `check_schema.py` 输出：**已存在 6**（+skill/skill_state）、**缺失 4**（study_attempt/study_session/scholar_book/knowledge_point，属 Phase 3/5）、**待迁移 5**（不变）、**已清理 0**。

### 验收核对（对照 execution-guide Phase 2）
- [x] 同一学者对同一句子同能力只产生一条 skill_state，重复上报只累加 attempt_count（`test_repeat_accumulates_attempt_count`）。
- [x] `POST /tracking/state` 支持 status/score/mastery/time_spent，返回最新状态（集成测试）。
- [x] 迁移后 skill_state 数据量与 learning_mastery_tracking 一致（3 条 → 3 条，幂等重复执行 0 新建）。
- [x] 新增测试覆盖 upsert 幂等、attempt_count 累加、中文状态词转换、复合键冲突。
- [x] 旧接口 `GET /tracking/{scholar_id}` 行为不变：skill_state 有数据时返回新表，无数据回退旧表（兼容未迁移数据）。
- [x] 旧表未被修改（`test_old_table_not_modified` == SEED）。

### 待办（进入真实环境时）
1. 运行存量迁移（小批量、可回滚）：`python3 -m services.migrate_learning`（可选 `--skill-code`，默认 translation）。
2. 为 `skill_state`（scholar_id/sentence_id/skill_code/lesson_id 复合索引）、`skill`（skill_code）建索引。
3. `POST /tracking/state` 上线后，前端从 `POST /tracking/stats` 的临时 record_list 上报切换到新接口。

## 9. Phase 3 完成记录（2026-08-13，事件模型）

### 目标达成
append-only 事件流已落地：每次学习行为写 `study_attempt`，每次学习过程写 `study_session`，为"学习时间 / 学习次数 / 滚动调度"提供可信数据源；同时**停写**旧表 `learning_mastery_tracking`（只读迁移数据保留）。

### 新增集合与文件
| 项 | 说明 |
|----|------|
| `study_attempt` | 学习事件表 — 每次学习行为追加一条（纯 insert，无 update 调用路径） |
| `study_session` | 学习会话表 — `POST /tracking/session/start` 创建（active），`POST /tracking/session/end` 结算（ended，回填 `ended_at`/`duration_sec`/`attempt_count`） |
| `services/events.py` | `infer_attempt_type` / `normalize_attempt_status` / `build_attempt_doc` / `build_session_doc` / `record_attempt` / `start_session` / `end_session` / `count_session_attempts` |
| `services/routes_state.py` | `POST /tracking/state` 同步写 study_attempt（返回 `{"state", "attempt"}`）；新增会话 start/end 接口 |
| `tests/unit/test_events.py` | 事件模块纯函数 + FakeDB 写入/结算/隔离单测（12） |
| `tests/integration/test_session_endpoints.py` | 会话 start/end 接口集成测试（7） |

### 关键设计
- **只增不改**：`record_attempt` 仅 `insert`，事件不可变；同一复合键可产生多条 attempt（与 skill_state 的 upsert 语义互补）。
- **类型推断**：`attempt_type` 未传时按 `skill_code` 推断（translation→translate、listening→listen、speaking→speak、reading→read），未知回落 `quiz`。
- **状态归一**：`attempt_status` 统一英文枚举（correct/incorrect/completed/abandoned），中文/非法值回落 `completed`。
- **会话结算**：`duration_sec = ended_at - started_at`，`attempt_count` = 会话内事件数（`count_session_attempts`）；无会话归属的事件不计入任何会话。
- **停写旧表**：`routes_build.py` repair 步骤 8 改为仅日志提示（不再 insert `learning_mastery_tracking`），返回的 `scholars_tracked`/`new_tracking_records` 恒为 0。旧表读取路径（GET /tracking 回退 / routes_admin 清理 / routes_dialogue / migrate）全部保留，仅停写。

### 测试与差距核对
- 全量测试：`python3 -m pytest tests/ -q` → **148 passed**（Phase 2 后 124 + Phase 3 新增 24：unit 12 + integration 12）。
- `check_schema.py` 输出：**已存在 8**（+study_attempt/study_session）、**缺失 2**（scholar_book/knowledge_point，属 Phase 5）、**待迁移 5**（不变，含 learning_mastery_tracking——仅剩读取路径，Phase 6 清理）。

### 验收核对（对照 execution-guide Phase 3）
- [x] study_attempt 只增不改（`test_events.py` 断言 study_attempt 集合内仅 insert；`events.py` 无 update 调用）。
- [x] study_session 的 duration_sec = ended_at - started_at，attempt_count 与会话内事件数一致（`TestSessionFlow::test_start_then_end`、`test_settles_session_with_attempt_count`）。
- [x] 线上新增行为不再产生新的 learning_mastery_tracking 文档（repair 步骤 8 停写，新增测试仅覆盖事件路径；`rg "learning_mastery_tracking" services/` 仅剩读取/清理/迁移/清单引用）。
- [x] 新增测试覆盖：事件写入（test_events.py + test_state_endpoints.py）、会话结算（test_session_endpoints.py）、重复会话互不干扰（`TestSessionAttemptIsolation`）。

### 待办（进入真实环境时）
1. 为 `study_attempt`（scholar_id/sentence_id/skill_code/session_id 索引）、`study_session`（scholar_id/textbook_id/status 索引）建索引。
2. 前端从 `POST /tracking/stats` record_list 切换到 `POST /tracking/state` + `POST /tracking/session/start` + `POST /tracking/session/end`（每次学习过程先 start，学习中多次 state 上报带 session_id，结束调 end）。
3. Phase 6 迁移核对通过后，才可下线 `learning_mastery_tracking`。

## 10. Phase 4 完成记录（2026-08-13，聚合计算与接口重构）

### 目标达成
进度/掌握度改为**服务端可复现的动态聚合**：Sentence → Lesson → Chapter → Book 逐级向上，按 skill 维度独立统计；`POST /tracking/stats` 不再依赖客户端 `record_list`，直接由 `skill_state` + 内容层级聚合。

### 新增/修改文件
| 项 | 说明 |
|----|------|
| `services/progress.py` | 聚合纯函数模块：`sentence_progress` / `pick_state` / `mastery_distribution` / `merge_distributions` / `lesson_progress` / `chapter_progress` / `book_progress` / `sum_time_spent` / `aggregate_progress` |
| `services/routes_tracking.py` | `POST /tracking/stats` 双路径：无 `record_list` → 服务端聚合（skill_state + chapter/lesson/sentence_v2 + study_attempt 时长）；带 `record_list` → 走旧 `compute_tracking_stats` 兼容入口（契约不变） |
| `tests/unit/test_progress.py` | 聚合纯函数单测（19） |
| `tests/integration/test_stats_aggregation.py` | stats 接口集成测试（12）：服务端聚合 / skill 过滤 / 复现 / 校验 / 兼容入口 |

### 关键设计
- **逐级加权**：lesson 按句均值 → chapter 按课内句子数加权 → book 按章内句子数加权，无学习记录句子按 progress=0 计入分母，保证与内容结构分母一致。
- **按 skill 独立**：同句多条 skill_state 在未指定 skill_code 时取 progress 最高（乐观聚合）；指定时只取该能力，各维度仅反映该能力。
- **时长聚合**：`total_time_spent` = `study_attempt.time_spent` 求和（非法值按 0），为 Phase 5 `scholar_book.total_time_spent` 预留；`format_duration` 复用 tracking_stats。
- **兼容层**：`tracking_stats.compute_tracking_stats` 完整保留，旧测试 `test_tracking_stats.py` 5 项全部通过。
- **返回结构**：`{scholar_id, text_book_id, skill_code, summary, chapters[{lessons}], lessons, units(兼容), sentences(兼容)}`；`summary` 含 textbook_progress / 三级计数 / mastery_distribution / 时长。

### 测试与差距核对
- 全量测试：`python3 -m pytest tests/ -q` → **179 passed**（Phase 3 后 148 + Phase 4 新增 31）。
- `check_schema.py` 输出：**已存在 8**（skill/skill_state/study_attempt/study_session/chapter/lesson/sentence_v2/textbook_v2）、**缺失 2**（scholar_book=Phase 5 / knowledge_point）、**待迁移 5**（旧表，Phase 6 清理）。

### 验收核对（对照 execution-guide Phase 4）
- [x] 同一输入下聚合结果可复现（`test_reproducible`：两次调用 `==`）。
- [x] `/tracking/stats` 响应包含 lesson/chapter/book 三级 progress 与 mastery 分布（`summary` + `chapters[].lessons[]`）。
- [x] 按 skill_code 过滤时，各维度结果仅反映该能力（`test_skill_code_filter` / `test_skill_filter_only_reflects_that_skill`）。
- [x] 旧测试 `test_tracking_stats.py` 仍通过；新增聚合模块测试（unit 19 + integration 12）通过。

### 待办（进入真实环境时）
1. 真实环境为 `chapter`/`lesson`/`sentence_v2`/`textbook_v2` 内容层级保证数据完整（聚合分母依赖内容结构）。
2. 前端 `POST /tracking/stats` 切换到服务端聚合形态（传 `scholar_id + textbook_id + 可选 skill_code`，去掉 record_list 上报）。
3. Phase 5 将 `total_time_spent` 落库到 `scholar_book`（周期重算或端侧结算写回）。

## 11. Phase 5 完成记录（2026-08-13，关联与学习计划）

### 目标达成
"学者 × 教材"关联已落地：`scholar_book` 以复合键 `{scholar_id}_{textbook_id}` 唯一，承载断点（`current_chapter_id`/`current_lesson_id`）、最后学习时间（`last_studied_at`）与累计时长（`total_time_spent`）；学习会话结算时端侧回写，教材列表接口聚合进度。

### 新增/修改文件
| 项 | 说明 |
|----|------|
| `services/models_scholar_book.py` | `scholar_book_id` / `build_scholar_book_doc` / `upsert_scholar_book`（存在则更新断点+增量累加时长，不存在则插入）/ `touch_scholar_book`（会话结算回写）/ `list_scholar_books` |
| `services/routes_tracking.py` | `GET /scholar/{scholar_id}/books`（我的教材列表，复用 `_aggregate_progress_for_book` 聚合进度）；`PUT /scholar/{scholar_id}/books/{textbook_id}/position`（更新断点，至少传一个字段否则 400） |
| `services/routes_state.py` | `POST /tracking/session/end` 结算成功后调用 `touch_scholar_book` 回写（刷新 `last_studied_at`、`total_time_spent += duration_sec`；无 textbook_id 的会话不落库） |
| `services/routes_system.py` | `/collections` 新增 `scholar_book` 清单（含断点/时长/状态/时间索引） |
| `tests/unit/test_scholar_book.py` | 纯函数 + FakeDB 单测（12）：复合键 / 文档构建 / 首次插入 / 重复加入幂等 / 时长增量累加 / 时间刷新 / 按学者隔离 / 列表排序 / 结算回写 / 无教材不落库 |
| `tests/integration/test_scholar_book_endpoints.py` | 接口集成测试（12）：首次加入 / 断点更新 / 重复幂等 / 参数校验 / 空列表 / 列表含进度 / 断点取回 / 多教材隔离 / 会话结算回写与多次累加 |

### 关键设计
- **复合键唯一**：`scholar_book_id = {scholar_id}_{textbook_id}`，upsert 以 `_id` 定位，重复加入/更新断点不产生新记录（验收标准 1）。
- **结算回写**：会话结束即回写，`total_time_spent` 增量累加 `duration_sec`；`last_studied_at` 取 `ended_at`。断点由 `PUT position` 显式维护，结算不动断点。
- **列表复用聚合**：`GET /scholar/{scholar_id}/books` 对每本教材调用 Phase 4 服务端聚合（`_aggregate_progress_for_book`），返回 `summary`（含 textbook_progress / 三级计数 / mastery_distribution / 时长），前端可基于此展示教材级进度。
- **兼容**：`POST /tracking/session/end` 返回结构不变（`data` 仍为会话文档），仅新增库内回写，既有会话测试不受影响。

### 测试与差距核对
- 全量测试：`python3 -m pytest tests/ -q` → **203 passed**（Phase 4 后 179 + Phase 5 新增 24：unit 12 + integration 12）。
- `check_schema.py` 输出：**已存在 9**（+scholar_book）、**缺失 1**（knowledge_point=可选，非必须）、**待迁移 5**（旧表，Phase 6 清理）。

### 验收核对（对照 execution-guide Phase 5）
- [x] 一个学者对同一教材只有一条 scholar_book 记录（`test_repeated_join_idempotent` / `test_repeated_join_idempotent` 接口幂等 / 多次会话结算仍单条）。
- [x] 断点更新后，重新获取教材列表能取回 current_lesson_id（`test_breakpoint_retrieved_after_update`）。
- [x] 新增测试覆盖首次加入、断点更新、重复加入幂等（`TestPutPosition` 5 项 + `TestUpsert` 幂等/累加）。

### 待办（进入真实环境时）
1. 为 `scholar_book`（scholar_id / textbook_id / status / last_studied_at）建索引。
2. 前端接入：学习流程 `start → 多次 state（带 session_id）→ end` 后自动回写教材时长；断点由 `PUT position` 维护；教材列表由 `GET /scholar/{scholar_id}/books` 拉取。
3. `knowledge_point` 为可选集合，若产品需要知识点级掌握度再补齐（当前聚合到教材级已闭环）。
