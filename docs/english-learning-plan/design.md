
1. **教材是学习的主线**：Book → Chapter → Lesson → Sentence。
2. **学习方式不能共享一个掌握度**：Conversation、Translation、Listening、Speaking 等必须独立记录。

我建议生产级设计不要只做 7 张表，而是采用 **“内容模型 + 学习状态 + 学习事件 + 聚合”** 四层结构。

---

# 一、整体架构

```text
Scholar
   │
   ├──────────────┐
   │              │
   ↓              ↓
Enrollment      StudyAttempt
   │
   ↓
Book
   │
   ↓
Chapter
   │
   ↓
Lesson
   │
   ↓
Sentence
   │
   ├── SkillState ──→ Skill
   │
   └── StudyAttempt
```

其中：

```text
Book / Chapter / Lesson / Sentence
```

是**教材内容**。

```text
SkillState
```

是：

> 学习者目前掌握到什么程度。

而：

```text
StudyAttempt
```

是：

> 学习者曾经做过什么。

这是整个设计最重要的分离。

---

# 二、Scholar

`Scholar` 不建议直接叫 `User`，如果你的产品本身就是学习系统，`Scholar` 很适合表达“学习者”。

```sql
CREATE TABLE scholar (
    id              BIGINT PRIMARY KEY,
    external_id     VARCHAR(100),
    
    display_name    VARCHAR(100),
    avatar_url      VARCHAR(500),

    locale          VARCHAR(20) NOT NULL DEFAULT 'zh-CN',
    timezone        VARCHAR(50) NOT NULL DEFAULT 'Asia/Shanghai',

    status          SMALLINT NOT NULL DEFAULT 1,

    created_at      TIMESTAMP NOT NULL,
    updated_at      TIMESTAMP NOT NULL,

    UNIQUE (external_id)
);
```

### status

```text
0 = disabled
1 = active
2 = archived
```

如果未来接 Auth0、Clerk、Firebase、自己的 OAuth 等：

```text
external_id
```

可以对应外部用户 ID。

---

# 三、Book

一本教材是一个独立的学习资源。

```sql
CREATE TABLE book (
    id                  BIGINT PRIMARY KEY,

    title               VARCHAR(255) NOT NULL,
    subtitle            VARCHAR(500),

    description         TEXT,

    language            VARCHAR(20) NOT NULL,
    target_language     VARCHAR(20),

    cover_url           VARCHAR(500),

    version             VARCHAR(50),

    publisher           VARCHAR(255),
    isbn                VARCHAR(50),

    total_chapters      INT NOT NULL DEFAULT 0,
    total_lessons       INT NOT NULL DEFAULT 0,
    total_sentences     INT NOT NULL DEFAULT 0,

    status              SMALLINT NOT NULL DEFAULT 1,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL
);
```

这里有一个非常重要的设计：

### Book 不属于 Scholar

不要：

```text
Book
 └── scholar_id
```

因为一本教材可以被很多学习者学习。

正确关系是：

```text
Scholar
   │
   └── ScholarBook
             │
             ↓
            Book
```

---

# 四、ScholarBook

虽然你没有把它列进 7 张表，但是**生产环境强烈建议增加这一张表**。

它代表：

> 某个学习者正在学习某本教材。

```sql
CREATE TABLE scholar_book (
    id                  BIGINT PRIMARY KEY,

    scholar_id          BIGINT NOT NULL,
    book_id             BIGINT NOT NULL,

    status              SMALLINT NOT NULL DEFAULT 1,

    started_at          TIMESTAMP,
    last_studied_at     TIMESTAMP,

    current_chapter_id  BIGINT,
    current_lesson_id   BIGINT,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL,

    UNIQUE (scholar_id, book_id),

    FOREIGN KEY (scholar_id) REFERENCES scholar(id),
    FOREIGN KEY (book_id) REFERENCES book(id)
);
```

例如：

```text
Tao
 │
 ├── English Grammar Book
 │
 └── Oxford English
```

这张表以后还能承载：

```text
favorite
learning_goal
start_date
target_date
daily_minutes
```

等学习计划信息。

---

# 五、Chapter

```sql
CREATE TABLE chapter (
    id                  BIGINT PRIMARY KEY,

    book_id             BIGINT NOT NULL,

    chapter_no          INT NOT NULL,

    title               VARCHAR(255) NOT NULL,
    subtitle            VARCHAR(500),

    description         TEXT,

    status              SMALLINT NOT NULL DEFAULT 1,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL,

    UNIQUE (book_id, chapter_no),

    FOREIGN KEY (book_id) REFERENCES book(id)
);
```

例如：

```text
Book
 │
 ├── Chapter 1
 ├── Chapter 2
 ├── Chapter 3
 └── Chapter 4
```

---

# 六、Lesson

```sql
CREATE TABLE lesson (
    id                  BIGINT PRIMARY KEY,

    chapter_id          BIGINT NOT NULL,

    lesson_no           INT NOT NULL,

    title               VARCHAR(255) NOT NULL,
    subtitle            VARCHAR(500),

    description         TEXT,

    status              SMALLINT NOT NULL DEFAULT 1,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL,

    UNIQUE (chapter_id, lesson_no),

    FOREIGN KEY (chapter_id) REFERENCES chapter(id)
);
```

于是：

```text
Book
 └── Chapter
      └── Lesson
```

---

# 七、Sentence

这里我建议稍微慎重。

**Sentence 不一定只是字符串。**

因为以后你很可能需要：

* 英文
* 中文
* 音频
* IPA
* 单词
* AI 解释
* 场景
* 难度
* 来源

所以：

```sql
CREATE TABLE sentence (
    id                  BIGINT PRIMARY KEY,

    lesson_id           BIGINT NOT NULL,

    sentence_no         INT NOT NULL,

    text                TEXT NOT NULL,
    translation         TEXT,

    audio_url           VARCHAR(500),

    difficulty          SMALLINT,

    metadata            JSON,

    status              SMALLINT NOT NULL DEFAULT 1,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL,

    UNIQUE (lesson_id, sentence_no),

    FOREIGN KEY (lesson_id) REFERENCES lesson(id)
);
```

---

# 八、但是 Sentence 最好不要直接绑定 Skill

这是一个很关键的设计。

不要：

```text
Sentence
 ├── translation_status
 ├── conversation_status
 ├── listening_status
 └── speaking_status
```

因为这样以后增加：

```text
dictation
shadowing
reading
grammar
```

就需要不断修改表结构。

应该增加：

# Skill

```sql
CREATE TABLE skill (
    id              SMALLINT PRIMARY KEY,

    code            VARCHAR(50) NOT NULL,
    name            VARCHAR(100) NOT NULL,

    description     TEXT,

    status          SMALLINT NOT NULL DEFAULT 1,

    UNIQUE (code)
);
```

例如：

```text
translation
conversation
listening
speaking
reading
dictation
```

---

# 九、SkillState

这是整个系统的核心表。

它表示：

> Scholar 对某个 Sentence 的某个 Skill 当前掌握状态。

```sql
CREATE TABLE skill_state (
    id                  BIGINT PRIMARY KEY,

    scholar_id          BIGINT NOT NULL,
    sentence_id         BIGINT NOT NULL,
    skill_id            SMALLINT NOT NULL,

    status              SMALLINT NOT NULL DEFAULT 0,

    mastery_score       DECIMAL(5,2) NOT NULL DEFAULT 0,

    attempt_count       INT NOT NULL DEFAULT 0,
    correct_count       INT NOT NULL DEFAULT 0,

    consecutive_correct INT NOT NULL DEFAULT 0,
    consecutive_wrong   INT NOT NULL DEFAULT 0,

    first_studied_at    TIMESTAMP,
    last_studied_at     TIMESTAMP,

    next_review_at      TIMESTAMP,

    created_at          TIMESTAMP NOT NULL,
    updated_at          TIMESTAMP NOT NULL,

    UNIQUE (scholar_id, sentence_id, skill_id),

    FOREIGN KEY (scholar_id) REFERENCES scholar(id),
    FOREIGN KEY (sentence_id) REFERENCES sentence(id),
    FOREIGN KEY (skill_id) REFERENCES skill(id)
);
```

---

# 十、这里要区分 `status` 和 `mastery_score`

我非常建议你保留两个字段。

### status

用于 UI：

```text
0 = not_started
1 = learning
2 = practicing
3 = familiar
4 = mastered
5 = maintaining
```

### mastery_score

用于算法：

```text
0.00 ~ 100.00
```

例如：

```text
status = 4
mastery_score = 87.35
```

这样：

> UI 可以显示“已掌握”。

而算法仍然知道：

> 87.35 和 96.82 并不一样。

---

# 十一、StudyAttempt

这个表记录**每一次学习行为**。

不要修改历史数据。

```sql
CREATE TABLE study_attempt (
    id                  BIGINT PRIMARY KEY,

    scholar_id          BIGINT NOT NULL,

    sentence_id         BIGINT NOT NULL,
    skill_id            SMALLINT NOT NULL,

    skill_state_id      BIGINT,

    session_id          BIGINT,

    score               SMALLINT NOT NULL,

    duration_ms         INT,

    attempt_no          INT,

    answer              TEXT,
    expected_answer     TEXT,

    error_type          VARCHAR(50),

    hints_used          SMALLINT NOT NULL DEFAULT 0,

    source              VARCHAR(50),

    metadata            JSON,

    studied_at          TIMESTAMP NOT NULL,
    created_at          TIMESTAMP NOT NULL,

    FOREIGN KEY (scholar_id) REFERENCES scholar(id),
    FOREIGN KEY (sentence_id) REFERENCES sentence(id),
    FOREIGN KEY (skill_id) REFERENCES skill(id),
    FOREIGN KEY (skill_state_id) REFERENCES skill_state(id)
);
```

---

# 十二、为什么 StudyAttempt 和 SkillState 必须分开？

例如：

第一次：

```text
Translation
score = 2
```

第二次：

```text
Translation
score = 3
```

第三次：

```text
Translation
score = 5
```

StudyAttempt：

```text
2
3
5
```

全部保留。

而 SkillState：

```text
mastery_score = 76.2
status = mastered
```

只保存**当前状态**。

所以：

```text
StudyAttempt = Event Log
SkillState    = Current State
```

这是非常典型的生产级设计。

---

# 十三、Session 也建议增加

虽然不是你现在要求的表，但是我强烈建议有。

因为一次学习可能包含：

```text
打开 Lesson
 ↓
AI Conversation
 ↓
Translation
 ↓
Listening
 ↓
结束
```

它们应该属于一次学习 Session。

```sql
CREATE TABLE study_session (
    id              BIGINT PRIMARY KEY,

    scholar_id      BIGINT NOT NULL,
    book_id         BIGINT,

    started_at      TIMESTAMP NOT NULL,
    ended_at        TIMESTAMP,

    duration_ms     INT,

    source          VARCHAR(50),

    metadata        JSON,

    created_at      TIMESTAMP NOT NULL,

    FOREIGN KEY (scholar_id) REFERENCES scholar(id),
    FOREIGN KEY (book_id) REFERENCES book(id)
);
```

然后：

```text
StudySession
     │
     ├── StudyAttempt
     ├── StudyAttempt
     ├── StudyAttempt
     └── StudyAttempt
```

这样以后你可以回答：

> 今天学习了多久？

> 今天学习了哪一章？

> 今天完成了多少次练习？

---

# 十四、完整关系

最终我建议：

```text
                         Scholar
                            │
             ┌──────────────┼───────────────┐
             │              │               │
             ↓              ↓               ↓
       ScholarBook    StudySession     StudyAttempt
             │                              │
             ↓                              │
            Book                            │
             │                              │
             ↓                              │
          Chapter                           │
             │                              │
             ↓                              │
           Lesson                           │
             │                              │
             ↓                              │
          Sentence ─────────────────────────┤
             │                              │
             ↓                              │
         SkillState ←───────────────────────┘
             │
             ↓
           Skill
```

---

# 十五、教材进度怎么计算？

这个设计最大的好处就是：

**Book 本身不需要保存某个用户的进度。**

例如：

```text
Sentence
   ↓
SkillState
   ↓
Scholar
```

然后动态计算。

---

## Lesson Progress

假设 Lesson 有：

```text
20 sentences
```

其中：

```text
12 sentences 已经学习
```

那么：

```text
Progress = 12 / 20 = 60%
```

---

## Lesson Mastery

假设每个 Sentence 的 SkillState：

```text
5
4
3
5
2
...
```

计算：

```text
Mastery =
AVG(mastery_score)
```

---

# 十六、Chapter

Chapter：

```text
Chapter
 ├── Lesson 1
 ├── Lesson 2
 ├── Lesson 3
 └── Lesson 4
```

Chapter Progress：

```text
SUM(learned sentences)
/
SUM(total sentences)
```

而不是：

```text
AVG(lesson progress)
```

因为每个 Lesson 的句子数量可能不同。

---

# 十七、Book

最终：

```text
Book
 ├── Chapter 1
 ├── Chapter 2
 ├── Chapter 3
 └── ...
```

可以得到：

```text
Book Progress
Book Mastery
```

例如：

```text
English Book

Progress       62%
Mastery        57%

12 Chapters
8 Chapters started
4 Chapters mastered

450 Sentences
286 Learned
164 Not started

42 Need Review
```

---

# 十八、推荐增加 Material / Content Version

如果这是**生产级教材系统**，还有一个很容易被忽略的问题：

> 教材内容以后可能会修改。

例如：

```text
Sentence 1024
```

今天：

```text
I am going to school.
```

以后教材更新：

```text
I'm going to school.
```

这会导致历史学习记录出现问题。

所以建议至少在 `Book` 上：

```text
version
```

更严格一点，可以增加：

```text
book_version
```

例如：

```text
Book
 │
 ├── Version 1.0
 │     └── Sentences
 │
 └── Version 2.0
       └── Sentences
```

对于你的 AI 学习系统，我认为这是值得做的。

---

# 十九、索引设计

生产环境不要忘记索引。

### ScholarBook

```sql
CREATE INDEX idx_scholar_book_scholar
ON scholar_book(scholar_id);

CREATE INDEX idx_scholar_book_book
ON scholar_book(book_id);
```

### Chapter

```sql
CREATE INDEX idx_chapter_book
ON chapter(book_id);
```

### Lesson

```sql
CREATE INDEX idx_lesson_chapter
ON lesson(chapter_id);
```

### Sentence

```sql
CREATE INDEX idx_sentence_lesson
ON sentence(lesson_id);
```

### SkillState

这个最重要：

```sql
CREATE INDEX idx_skill_state_scholar_skill
ON skill_state(scholar_id, skill_id);

CREATE INDEX idx_skill_state_review
ON skill_state(scholar_id, next_review_at);

CREATE INDEX idx_skill_state_sentence
ON skill_state(sentence_id);

CREATE INDEX idx_skill_state_mastery
ON skill_state(scholar_id, mastery_score);
```

### StudyAttempt

```sql
CREATE INDEX idx_attempt_scholar_time
ON study_attempt(scholar_id, studied_at);

CREATE INDEX idx_attempt_sentence
ON study_attempt(sentence_id);

CREATE INDEX idx_attempt_state
ON study_attempt(skill_state_id);

CREATE INDEX idx_attempt_session
ON study_attempt(session_id);
```

---

# 二十、我会再加一层：Learning Objective

这是你这个产品未来可能非常有价值的一张表。

因为：

> Sentence ≠ Knowledge Point

例如：

```text
I have been working here for three years.
```

它可能同时包含：

```text
Present Perfect Continuous
for + duration
work
```

所以未来可以：

```text
Sentence
   ├── KnowledgePoint
   │     ├── Present Perfect
   │     ├── Present Perfect Continuous
   │     └── for/since
   │
   └── Skill
         ├── Translation
         ├── Conversation
         └── Listening
```

这样你最终就不只是：

> **这句话掌握了吗？**

而可以回答：

> **这个学习者在本教材的现在完成进行时知识点上掌握得怎么样？**

这对于你之前提到的**错题分析、知识薄弱点、强化练习**会非常重要。

---

# 二十一、所以最终生产级架构，我会定成

```text
                    ┌──────────────┐
                    │   Scholar    │
                    └──────┬───────┘
                           │
                    ScholarBook
                           │
                           ↓
┌──────────────────────────────────────────┐
│                  BOOK                    │
│                                          │
│  Chapter                                 │
│    │                                     │
│    └── Lesson                            │
│          │                               │
│          └── Sentence                    │
│                │                         │
│                ├──── KnowledgePoint      │
│                │                         │
│                └──── Skill               │
│                       │                  │
│                       ↓                  │
│                  SkillState              │
│                       ↑                  │
│                       │                  │
│                  StudyAttempt            │
│                       ↑                  │
│                       │                  │
│                  StudySession            │
└──────────────────────────────────────────┘
```

### 核心原则可以浓缩成一句话：

> **教材结构决定“学什么”，Skill 决定“学会什么能力”，StudyAttempt 记录“学过什么”，SkillState 表示“现在会多少”，Progress/Mastery 则从这些数据向上聚合到 Lesson → Chapter → Book。**

这套结构特别适合你现在这个产品，因为以后你加入 **AI 对话、中译英、听力、口语、错题、间隔重复、知识薄弱点、AI 自动生成练习题**，基本都不需要推翻现有的数据模型。

如果用 PostgreSQL，我还会进一步把它设计成 **UUID/ULID、软删除、乐观锁 `version`、JSONB、时间分区的 StudyAttempt、唯一约束和事务更新 SkillState** 的完整 DDL；尤其 `StudyAttempt` 数据量将来会远大于其他表，应该从一开始就考虑它的增长方式。
