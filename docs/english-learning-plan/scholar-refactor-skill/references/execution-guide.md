# 分步重构执行指南（Phase 0 → Phase 6）

> 每个 Phase 结构：**目标 / 改动文件与集合 / 操作步骤 / 验收标准**。
> 严格执行"**完成一步、验收一步、再进下一步**"。所有涉及 CloudBase 集合的步骤，先小批量测试、可回滚。
> **清理红线**：旧表与旧数据严禁提前清理。迁移只写新表、旧表保持只读；只有在新表数据迁移结束、迁移核对与全量测试通过后（**仅 Phase 6 末尾**）才允许下线/删除旧表。任何前置 Phase 中出现"删除旧表/旧数据"动作即为违规。

---

## Phase 0 — 基线确认（无代码改动）

### 目标
锁定重构起点，确认当前一切可运行、可测试，为后续对比留基准。

### 操作步骤
1. 安装依赖：`pip install -r requirements.txt`（如环境缺 langgraph，单独记录，不阻塞）。
2. 跑全量测试：`python3 -m pytest tests/ -q`，确认全绿。
3. 生成现状快照：`python3 -m pytest tests/ -q --collect-only > /tmp/baseline_tests.txt`。
4. 存档现状：运行 `scripts/check_schema.py` 输出"已存在/缺失/需废弃"清单，与 `references/current-state.md` 核对。
5. 更新 `references/current-state.md` 的"代码引用地图"（重新执行 `rg -n "learning_mastery_tracking|unit|paragraph|text_book_id" services/`）。

### 验收标准
- [ ] 全量测试通过（记录失败项及原因，均为既有问题）。
- [ ] `check_schema.py` 输出：现状 5 集合（textbook/unit/paragraph/sentence/learning_mastery_tracking），**缺失 10 集合**（textbook_v2/chapter/lesson/sentence_v2/skill/skill_state/study_attempt/study_session/scholar_book/knowledge_point），**待迁移 5 集合**（textbook/unit/paragraph/sentence/learning_mastery_tracking）。
- [ ] 引用地图已更新存档。

---

## Phase 1 — 内容模型分层（chapter + lesson + sentence_v2）

### 目标
补齐内容层级：`textbook_v2 → chapter → lesson → sentence_v2`，让句子归课、课归章、章归教材。**新内容写入 v2 新表，旧表 textbook/sentence 保持不动**，互不影响。

### 改动文件与集合
- 新增集合：`chapter`、`lesson`。
- **新建集合（替代旧表，不与旧数据冲突）**：`textbook_v2`（替代 `textbook`）、`sentence_v2`（替代 `sentence`）。
- 新增文件：`services/models_content.py`（查询辅助 + `write_content_v2` 双写）、`services/migrate_content_v2.py`（幂等迁移脚本，**只写新表、只读旧表**）。
- 修改文件：`services/routes_build.py`（构建双写新表）、`services/routes_vision.py`（视觉识别双写新表）、`services/routes_system.py`（`/collections` 集合清单）。
- 说明：`routes_admin.py` 仅操作旧表，Phase 1 不扩展其删除/合并范围到新表（新表管理逻辑留给后续 Phase），避免触碰清理红线。

### 操作步骤
1. 在 `services/` 新增数据访问辅助（如 `services/models_content.py`）：
   - `get_chapters(textbook_id)` / `get_lessons(chapter_id)` / `get_sentences_by_lesson(lesson_id)`，均指向新表。
2. 修改构建流程（routes_build.py / routes_vision.py）：构建 `textbook_v2` 后先建 chapter，再按 chapter 建 lesson（transition：**新生成的 `lesson_id` 与旧 `unit_id` 值保持一致**，如 `unit_3` → `lesson` 集合中 `lesson_id=unit_3`，便于迁移）。
3. `sentence_v2` 文档包含 `chapter_id`、`lesson_id` 引用字段；**过渡期**保留 `unit_id`、`text_book_id`（便于核对迁移）。
4. 数据迁移：旧 `textbook` → `textbook_v2`（全量复制 + `version=1`），旧 `sentence` → `sentence_v2`（全量复制并按旧 `unit` 归属回填 `chapter_id`/`lesson_id`）。用 `scripts/check_schema.py` 对照"待迁移"清单，小批量执行、可回滚（迁移命令样例见文末）。

### 验收标准
- [ ] 新增的 chapter/lesson 集合可按 `textbook_id + order` 有序查询。
- [ ] `textbook_v2`/`sentence_v2` 数据完整，`sentence_v2` 每条都有 `chapter_id` + `lesson_id`，且与旧 `unit_id` 对应关系正确（抽查 5% 样本）。
- [ ] 旧 `textbook`/`sentence` 集合**未被修改**（线上只读路径不受影响）。
- [ ] 旧接口（GET /tracking/{scholar_id}、POST /tracking/stats）行为不变，测试仍全绿。
- [ ] 新增单元测试覆盖：构建 chapter/lesson、sentence_v2 引用字段完整性。

---

## Phase 2 — 能力模型（skill + skill_state）

### 目标
把"一学一记录"的平铺状态，改造成"学者 × 句子 × 能力"的状态模型；补上学习次数（study_count/attempt_count）与最后学习时间（last_studied_at）的落点，支撑滚动学习调度。

### 改动文件与集合
- 新增集合：`skill`、`skill_state`。
- 新增文件：`services/models_learning.py`（状态读写）、`services/routes_state.py`（上报接口）、`services/migrate_learning.py`（幂等迁移脚本，**只写新表、只读旧表**）。
- 修改文件：`services/routes_tracking.py`（查询改走 skill_state，无记录回退旧表过渡兼容）、`services/routes_system.py`（`/collections` 集合清单）、`services/main.py`（注册 state 路由）。

### 操作步骤
1. 预置 skill 种子数据（`translation`/`listening`/`speaking`/`reading` 及阈值）。
2. 实现 `skill_state` 读写：`upsert_skill_state(scholar_id, sentence_id, skill_code, **update)`——按复合键 `{scholar_id}_{sentence_id}_{skill_code}` upsert；`attempt_count` 累加、`last_studied_at` 刷新。
3. 新增接口 `POST /tracking/state`（上报单句单能力状态，替代原 record_list 的临时上报），并把 `learning_mastery_tracking` 的旧字段逐一映射到 skill_state（见 target-model.md 第 3 节说明）。
4. 数据迁移：将现有 `learning_mastery_tracking` 记录按 `sentence_id` 回填 `lesson_id`，写入 `skill_state`（`attempt_count` 取原 study_count 或 1，`last_studied_at` 取 last_study_time）。
5. 中文状态词收敛：`is_learned` 的中文兼容保留于 tracking_stats（纯函数兼容层），新写入统一英文枚举。

### 验收标准
- [ ] 同一学者对同一句子同能力只产生一条 skill_state，重复上报只累加 attempt_count。
- [ ] `POST /tracking/state` 支持 status/score/mastery/time_spent，返回最新状态。
- [ ] 迁移后 skill_state 数据量与 learning_mastery_tracking 一致（除按能力拆分）。
- [ ] 新增测试：upsert 幂等性、attempt_count 累加、中文状态词转换、复合键冲突。

---

## Phase 3 — 事件模型（study_attempt + study_session）

### 目标
建立 append-only 事件流：每次学习行为写 study_attempt，每次学习过程写 study_session，为"学习时间 / 学习次数 / 滚动调度"提供可信数据源。

### 改动文件与集合
- 新增集合：`study_attempt`、`study_session`。
- 新增文件：`services/events.py`（写入辅助）。
- 修改文件：`services/routes_state.py`（上报时同步写事件）、`services/routes_build.py`（若构建过程写 tracking，改到 Phase 3 后停写旧表）。

### 操作步骤
1. 实现 `record_attempt(...)` / `record_session(...)`：纯插入、不更新（append-only）。
2. 在 `POST /tracking/state` 中同时写 study_attempt（attempt_type 由前端传或按 skill_code 推断）。
3. 新增 `POST /tracking/session/start` 与 `/tracking/session/end`（或单接口合并），创建/结算 study_session，累加 duration_sec。
4. 停写旧表：从 Phase 3 起，新增学习行为**不再写** `learning_mastery_tracking`（只读迁移数据），避免双写。**注意：停写 ≠ 删除**，旧表及其数据必须保留到 Phase 6 迁移核对 + 全量测试通过后才能清理。

### 验收标准
- [x] study_attempt 只增不改（无 update 调用路径）。
- [x] study_session 的 duration_sec = ended_at - started_at，attempt_count 与会话内事件数一致。
- [x] 线上新增行为不再产生新的 learning_mastery_tracking 文档。
- [x] 新增测试：事件写入、会话结算、重复会话互不干扰。

---

## Phase 4 — 聚合计算与接口重构（progress / mastery）

### 目标
进度与掌握度改为**服务端可复现的动态聚合**：Sentence → Lesson → Chapter → Book 逐级向上，按 skill 维度独立统计。

### 改动文件
- 新增：`services/progress.py`（聚合模块，纯函数便于测试）。
- 修改：`services/routes_tracking.py`（`POST /tracking/stats` 改为服务端计算）、`services/tracking_stats.py`（保留供旧测试与兼容）。

### 操作步骤
1. 实现 `progress.py` 纯函数（以 `tracking_stats.py` 为参照）：
   - `sentence_progress(state)`：由 mastery_score / status 得 0-1。
   - `lesson_progress(states, skill_code)`：课内句子进度均值。
   - `chapter_progress(lessons)` / `book_progress(chapters)`：逐级加权平均。
   - `mastery_distribution(states)`：learned/mastered 占比分布。
2. `POST /tracking/stats` 改为：入参 `scholar_id + textbook_id（可选 skill_code）`，服务端从 skill_state 拉数聚合，**不再依赖客户端 record_list**（record_list 仅作 Phase 2/3 兼容入口，可选）。
3. 学习时间聚合：`scholar_book.total_time_spent` 由 study_attempt.time_spent 求和（或周期重算 job）。
4. 保留 `tracking_stats.compute_tracking_stats` 供旧测试；新增测试全部走新聚合路径。

### 验收标准
- [x] 同一输入下聚合结果可复现（两次调用一致）。
- [x] `/tracking/stats` 响应包含 lesson/chapter/book 三级 progress 与 mastery 分布。
- [x] 按 skill_code 过滤时，各维度结果仅反映该能力。
- [x] 旧测试 `test_tracking_stats.py` 仍通过（兼容层）；新增聚合模块测试通过。

---

## Phase 5 — 关联与学习计划（scholar_book）

### 目标
建立"学者 × 教材"关联，支持断点续学、教材级状态与累计时长。

### 改动文件与集合
- 新增集合：`scholar_book`。
- 修改文件：`services/routes_tracking.py`（新增/调整接口）、`services/routes_state.py`。

### 操作步骤
1. 实现 `scholar_book` upsert（键 `{scholar_id}_{textbook_id}`），记录 current_chapter_id / current_lesson_id / last_studied_at / total_time_spent。
2. 新增接口：
   - `GET /scholar/{scholar_id}/books` — 我的教材列表（含进度）。
   - `PUT /scholar/{scholar_id}/books/{textbook_id}/position` — 更新断点。
3. 学习会话结算时回写 last_studied_at 与 total_time_spent。

### 验收标准
- [x] 一个学者对同一教材只有一条 scholar_book 记录。
- [x] 断点更新后，重新获取教材列表能取回 current_lesson_id。
- [x] 新增测试覆盖：首次加入、断点更新、重复加入幂等。

> Phase 5 验收记录（2026-08-13）：全量 `pytest tests/ -q` → **203 passed**（新增 24）；
> `scripts/check_schema.py` → 已存在 9（含 scholar_book）、缺失 1（knowledge_point=可选）、待迁移 5（Phase 6 清理）。
> 详见 `references/current-state.md` 第 11 节。

---

## Phase 6 — 清理与文档收尾

### 目标
废弃旧集合与旧字段，统一命名，完成文档与全量验证。

### 操作步骤
1. 代码清理：移除对 `learning_mastery_tracking` 的所有读写（routes_tracking / routes_admin / routes_dialogue / routes_build），`unit_id`/`text_book_id` 字段只读兼容后移除。
2. 数据清理（**最后一步，前置条件全部满足才可执行，顺序不可颠倒**）：
   - 前置条件 a（迁移结束并核对通过）：确认 `skill_state`/`study_attempt`/`study_session`/`scholar_book` 数据完整，`textbook_v2`/`sentence_v2` 数据完整且与旧表抽样核对一致（覆盖 5% 样本）；
   - 前置条件 b（测试通过）：`python3 -m pytest tests/ -q` 全绿，`scripts/check_schema.py` 确认新表已就绪；
   - **a、b 全部满足后**，才归档并下线 `learning_mastery_tracking` 与旧 `textbook`/`sentence`/`unit`/`paragraph`（先备份导出 JSON，再下线）。
3. 命名统一：全量替换 `text_book_id → textbook_id`、`unit_id → lesson_id`（更新 `references/current-state.md` 的接口表）。
4. 文档：更新 README、接口文档、`docs/english-learning-plan/03-roadmap.md` 状态。
5. 终验：全量测试 + `scripts/check_schema.py` 差距归零（"缺失"为 0 且"待迁移"为 0）。

### 验收标准
- [ ] 旧表下线仅发生在 Phase 6 末尾，且顺序可追溯为"迁移核对通过 → 全量回归通过 → 才清理"；Phase 0-5 全程无删除旧表/旧数据的动作。
- [ ] `rg -n "learning_mastery_tracking|unit_id|text_book_id" services/ tests/` 无有效命中。
- [ ] 全量 pytest 通过。
- [ ] `check_schema.py` 输出：**缺失 0、待迁移 0、已清理 5**（textbook/unit/paragraph/sentence/learning_mastery_tracking）。
- [ ] 文档与代码一致。

---

## 附录：数据迁移命令样例（CloudBase）

> 以下为思路示例，实际以项目 `services/database.py` 封装为准；迁移前先对目标集合做备份（导出 JSON），单批 ≤ 500 条，可回滚。
> **迁移后不删除旧表**：先用 `scripts/check_schema.py` 与抽样核对新旧数据一致性、再跑全量测试；确认无误后（仅 Phase 6 末尾）才归档并下线旧表。

```python
# === 1) 旧 textbook → textbook_v2（全量复制，新增 version 字段）
# 对旧 textbook 每条：insert textbook_v2 = {**doc, "version": 1}

# === 2) 旧 sentence → sentence_v2（全量复制 + 回填层级引用）
# 2.1) 对每个旧 textbook：先建 chapter（按旧 unit 的 group 规则或教材自有章节目录）
# 2.2) 对每个旧 unit：建 lesson（lesson_id = unit_id），得到 chapter_id
# 2.3) 批量写 sentence_v2：
#      where={"text_book_id": tb_id, "unit_id": uid} → insert sentence_v2 = {**doc,
#           "chapter_id": chapter_id, "lesson_id": lesson_id}

# === 3) 迁移 learning_mastery_tracking → skill_state（skill_state 为全新集合，直接写）
# 1) 按 scholar_id + sentence_id 归组（同一句多技能则各写一条）
# 2) upsert skill_state：status 映射、mastery_score=score、attempt_count=study_count or 1、
#    last_studied_at=last_study_time、next_review_at=按滚动调度公式初算
```

## 附录：滚动学习调度公式（Phase 2 落点）

`next_review_at = last_studied_at + interval(attempt_count, mastery_score)`
- 基础间隔：1天 → 3天 → 7天 → 14天 → 30天（随 attempt_count 递增）。
- 掌握度高（mastery_score ≥ 80）：间隔 × 1.5；低（< 60）：间隔 ÷ 2 且状态置 `review_due`。
- 具体实现放入 Phase 2 的 `services/models_learning.py`，单测覆盖边界。
