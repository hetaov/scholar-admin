# 目标数据模型（design.md → CloudBase NoSQL 映射）

> 本文件从 `../design.md` 提取设计意图，映射为 scholar-admin 实际可落地的 **CloudBase NoSQL 集合**。
> CloudBase 为文档型数据库，因此设计上采用"**每条记录一个主键 + 冗余引用字段 + 唯一复合键约束**"模拟关系：
> - 内容集合（textbook_v2/chapter/lesson/sentence_v2）：只存内容，**不存任何用户进度**。
> - 状态集合（skill_state）：一学一能力一条记录，保存**当前状态**，可更新。
> - 事件集合（study_attempt/study_session）：**append-only**，只插入不修改。
> - 进度（progress/mastery）不落库，由 skill_state **动态聚合**。

## 0. 命名策略（新旧数据隔离，先并行走，后迁移）

为**不影响现有数据、方便后续迁移**，目标集合名采用以下规则：

| 规则 | 说明 |
|------|------|
| 与旧集合同名 → 加 `_v2` | `textbook → textbook_v2`、`sentence → sentence_v2`。旧表照常服务线上，新表独立建，互不覆盖 |
| 全新集合 → 保持原名 | `chapter`/`lesson`/`skill`/`skill_state`/`study_attempt`/`study_session`/`scholar_book`/`knowledge_point` 在现有代码中不存在，直接用原名无冲突 |
| 旧表迁移目标 | `unit → lesson`、`paragraph` 内容并入 `sentence_v2`、`learning_mastery_tracking → skill_state + study_attempt`、`textbook → textbook_v2`、`sentence → sentence_v2` |

> **迁移节奏**：Phase 1-5 只写新表（或新旧双写），旧表只读；**旧表与旧数据严禁提前清理**——Phase 6 必须先完成数据迁移核对与全量测试回归，全部通过后才统一切换读路径并下线旧表。`scripts/check_schema.py` 输出"已存在 / 缺失 / 待迁移 / 已清理"四类清单跟踪进度（"已清理"只允许在 Phase 6 末尾出现）。

## 1. 内容模型

### textbook_v2（教材，新建集合，替代旧 textbook）

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `textbook_id` | string | 主键 |
| `title` | string | 教材名 |
| `grade` / `level` | string | 年级/级别 |
| `version` | int | 内容版本（默认 1，内容改版时 +1） |
| `chapter_count` | int | 章节数（冗余，便于列表页） |
| `lesson_count` | int | 课数（冗余） |
| `sentence_count` | int | 句子数（冗余） |

> 迁移来源：旧 `textbook` 集合全量复制并新增 `version=1` 等字段；`_id` 沿用旧 `textbook_id`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `textbook_id` | string | 主键 |
| `title` | string | 教材名 |
| `grade` / `level` | string | 年级/级别 |
| `version` | int | 内容版本（默认 1，内容改版时 +1） |
| `chapter_count` | int | 章节数（冗余，便于列表页） |
| `lesson_count` | int | 课数（冗余） |
| `sentence_count` | int | 句子数（冗余） |

### chapter（章节，新建集合）

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `chapter_id` | string | 主键 |
| `textbook_id` | string | 教材引用（**复合索引 A**） |
| `order` | int | 章节序号 |
| `title` | string | 标题 |
| `lesson_count` | int | 课数冗余 |

### lesson（课，新建集合；现有 unit 升级而来）

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `lesson_id` | string | 主键（过渡期可直接沿用 unit_id 值） |
| `chapter_id` | string | 章节引用（**复合索引 A 之二**） |
| `textbook_id` | string | 教材引用（冗余，便于直查） |
| `order` | int | 课序号 |
| `title` | string | 标题 |
| `sentence_count` | int | 句子数冗余 |

### sentence_v2（句子，新建集合，替代旧 sentence）

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `sentence_id` | string | 主键（沿用旧 `sentence_id` 值，便于关联迁移） |
| `textbook_id` | string | 教材引用（指向 textbook_v2） |
| `chapter_id` | string | 章节引用（Phase 1 补齐） |
| `lesson_id` | string | 课引用（Phase 1 补齐） |
| `unit_id` | string | 旧字段（仅迁移期保留，Phase 6 移除） |
| `order` | int | 句序 |
| `text` | string | 原文 |
| `translation` | string | 译文 |
| `audio_url` | string | 音频 |
| `knowledge_point_ids` | string[] | 关联知识点（可选） |

> 迁移来源：旧 `sentence` 全量复制，并按旧 `unit` 归属回填 `chapter_id` / `lesson_id`。

## 2. 关联模型

### scholar_book（新建集合）— 学者 × 教材

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `scholar_book_id` | string | 主键（`{scholar_id}_{textbook_id}`） |
| `scholar_id` | string | 学者引用（**复合索引 B**） |
| `textbook_id` | string | 教材引用 |
| `status` | string | `not_started` / `learning` / `completed` |
| `current_chapter_id` | string | 当前章节（断点续学） |
| `current_lesson_id` | string | 当前课（断点续学） |
| `total_time_spent` | number | 累计学习时长（秒，聚合冗余，可周期重算） |
| `last_studied_at` | date | 最近学习时间 |
| `started_at` / `completed_at` | date | 起止时间 |

## 3. 学习状态模型

### skill（能力定义，新建集合）

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `skill_code` | string | 主键，如 `translation` / `listening` / `speaking` / `reading` |
| `name` | string | 展示名 |
| `mastery_threshold` | number | 判为 mastered 的阈值（0-1，默认 0.8） |
| `learned_threshold` | number | 判为 learned 的阈值（0-1，默认 0.6） |

### skill_state（新建集合）— 每个"学者 × 句子 × 能力"一条当前状态

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `state_id` | string | 主键（`{scholar_id}_{sentence_id}_{skill_code}`） |
| `scholar_id` | string | 学者引用（**复合索引 C**：scholar_id + lesson_id + skill_code） |
| `sentence_id` | string | 句子引用 |
| `lesson_id` | string | 课引用（冗余，便于按课聚合） |
| `skill_code` | string | 能力代码 |
| `status` | string | `not_started` / `learning` / `learned` / `mastered` / `review_due` |
| `mastery_score` | number | 掌握分数 0-100 |
| `progress` | number | 当前进度 0-1 |
| `attempt_count` | int | 学习次数（**StudyAttempt 计数**） |
| `last_studied_at` | date | 最近学习时间 |
| `next_review_at` | date | 下次复习时间（滚动学习调度用） |
| `created_at` / `updated_at` | date | 时间戳 |

> **说明**：`learning_mastery_tracking` 的每个字段都能映射进这里 —— `time_spent` 归入事件聚合与 `scholar_book.total_time_spent`，`status/score/mastery` 归入 `skill_state`，`study_count` 归入 `attempt_count`，`last_study_time` 归入 `last_studied_at`。

## 4. 学习事件模型

### study_attempt（新建集合）— append-only 事件日志

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `attempt_id` | string | 主键（可用自增/时间戳） |
| `scholar_id` | string | 学者引用（**复合索引 D**：scholar_id + sentence_id + skill_code + created_at） |
| `sentence_id` | string | 句子引用 |
| `lesson_id` | string | 课引用（冗余） |
| `skill_code` | string | 能力代码 |
| `attempt_type` | string | `read` / `translate` / `listen` / `speak` / `quiz` |
| `status` | string | 完成状态（`correct` / `incorrect` / `completed` / `abandoned`） |
| `score` | number | 本次得分 0-100（可选） |
| `time_spent` | number | 本次时长（秒） |
| `session_id` | string | 所属会话 |
| `created_at` | date | 事件时间（不可变） |

### study_session（新建集合）— 学习会话

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `session_id` | string | 主键 |
| `scholar_id` | string | 学者引用（**复合索引 E**：scholar_id + started_at） |
| `textbook_id` | string | 教材引用 |
| `started_at` / `ended_at` | date | 起止时间 |
| `duration_sec` | number | 时长（秒） |
| `attempt_count` | int | 会话内尝试次数（冗余） |
| `device` / `source` | string | 设备/来源（可选） |

## 5. 知识点（可选，后期）

### knowledge_point（新建集合）— 学习目标/知识点

| 字段 | 类型 | 说明 |
|------|------|------|
| `_id` / `kp_id` | string | 主键 |
| `code` | string | 如 `vocab_001` / `grammar_003` |
| `type` | string | `vocabulary` / `grammar` / `phrase` / `sentence_pattern` |
| `title` | string | 描述 |
| `sentence_ids` | string[] | 关联句子 |
| `difficulty` | number | 难度 1-5 |

## 6. 状态枚举（全局统一）

| 枚举 | 取值 |
|------|------|
| `skill_state.status` | `not_started` / `learning` / `learned` / `mastered` / `review_due` |
| `scholar_book.status` | `not_started` / `learning` / `completed` |
| `study_attempt.attempt_type` | `read` / `translate` / `listen` / `speak` / `quiz` |
| `study_attempt.status` | `correct` / `incorrect` / `completed` / `abandoned` |
| 中文兼容 | 现有 `tracking_stats.is_learned` 的中文状态词（已学/未学/学习中/掌握）在 Phase 2 收敛为英文枚举 |

## 7. 索引设计（CloudBase 建议）

| 索引 | 集合 | 字段 | 用途 |
|------|------|------|------|
| A | sentence_v2 / lesson / chapter | `textbook_id` + `order` | 按教材取内容 |
| B | scholar_book | `scholar_id` + `textbook_id` | 我的教材列表 |
| C | skill_state | `scholar_id` + `lesson_id` + `skill_code` | 按课聚合掌握度 |
| D | study_attempt | `scholar_id` + `sentence_id` + `skill_code` + `created_at` | 按句/能力查历史 |
| E | study_session | `scholar_id` + `started_at` | 按时间查会话 |

> CloudBase 索引在控制台按以上字段创建，DDL 类操作见 `execution-guide.md` 中对应 Phase 的迁移命令样例。
