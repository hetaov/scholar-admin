# CloudBase NoSQL API + 火山引擎 AI 图片识别

基于 FastAPI 的 CloudBase 文档型数据库 RESTful API 服务 + 火山引擎豆包视觉模型英文教材图片识别服务。

## 项目结构

```
scholar-admin/
├── config.py              # 项目配置（.env 自动加载）
├── main.py                # FastAPI 应用入口 + API 路由
├── services/
│   ├── __init__.py
│   ├── database.py        # CloudBase NoSQL 客户端（TC3 签名 + CRUD 封装）
│   └── volcano.py         # 火山引擎豆包视觉模型服务
├── requirements.txt       # Python 依赖
├── Dockerfile             # 容器构建文件
├── cloudbaserc.json       # CloudBase 部署配置
├── .env.example           # 环境变量模板（可提交 git）
├── .env                   # 真实环境变量（gitignore 保护）
└── README.md
```

## 启动方式

### 1. 本地开发

**前置要求：Python 3.11+**

```bash
# 创建并激活虚拟环境（强烈推荐，避免与系统 Python 混用）
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量（推荐方式：复制模板文件并填写）
cp .env.example .env
# 编辑 .env，填入 VOLCANO_API_KEY、TENCENTCLOUD_SECRETID 等

# 启动服务（默认端口 8080）
python main.py

# 或指定端口
PORT=3000 python main.py
```

> **注意：所有脚本（`main.py`、`pytest` 等）都必须在 `.venv` 环境下运行。**
> 若未激活虚拟环境，`python` 可能解析到系统 Python（如 conda 的 Python 3.14），
> 其依赖与项目不一致，会导致 `.env` 中的腾讯云密钥加载异常，请求被判定为无效凭证
> （报 `AuthFailure.SignatureFailure` 等）。可用 `which python` 确认是否指向 `.venv/bin/python`。

启动后访问 http://localhost:8080/docs 查看 Swagger API 文档。

### 2. Docker 运行

```bash
# 构建镜像
docker build -t scholar-admin .

# 运行容器
docker run -p 8080:8080 \
  -e TCB_ENV_ID="your-env-id" \
  -e TENCENTCLOUD_SECRETID="your-secret-id" \
  -e TENCENTCLOUD_SECRETKEY="your-secret-key" \
  scholar-admin
```

### 3. CloudBase CloudRun 部署

项目已包含 `cloudbaserc.json` 和 `Dockerfile`，可直接部署到 CloudBase 云托管：

```bash
# 使用 CloudBase CLI 部署
tcb cloudrun deploy
```

CloudRun 环境下，`TENCENTCLOUD_SECRETID` / `TENCENTCLOUD_SECRETKEY` / `TENCENTCLOUD_SESSIONTOKEN` 会自动注入，**无需手动配置这三项**。

> **SOE-N 语音评测依赖官方 SDK 源码**（`vendor/tencentcloud-speech-sdk-python`，PyPI 无包，纯源码分发）。`.gitignore` 排除了 `vendor/`，因此从 **git 仓库拉取构建** 的 CloudRun 部署产物不含该目录——Dockerfile 已在构建期兜底：若镜像缺 `vendor/tencentcloud-speech-sdk-python/common` 则在线 `git clone` 官方仓库。若构建机无法访问 GitHub，需先 `git add -f vendor/tencentcloud-speech-sdk-python` 入库后再部署。

## 配置项说明

所有配置通过**环境变量**注入，定义在 `config.py` 中：

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `TCB_ENV_ID` | `knowlege-graph-env-d7cwud346b70b` | CloudBase 环境 ID，**必须配置** |
| `TCB_REGION` | `ap-shanghai` | 区域，如 `ap-shanghai`、`ap-guangzhou` |
| `TENCENTCLOUD_SECRETID` | (空) | 腾讯云 API SecretId，**本地运行必须配置** |
| `TENCENTCLOUD_SECRETKEY` | (空) | 腾讯云 API SecretKey，**本地运行必须配置** |
| `TENCENTCLOUD_SESSIONTOKEN` | (空) | 临时凭证 Token（CloudRun 自动注入，本地留空） |
| `PORT` | `8080` | 服务监听端口 |
| `VOLCANO_API_KEY` | (空) | 火山方舟 API Key，**使用图片识别功能必须配置** |
| `VOLCANO_VISION_MODEL` | `doubao-1.5-vision-pro-32k` | 豆包视觉模型名称 |

> **推荐：** 本地开发使用 `.env` 文件管理密钥。复制 `.env.example` 为 `.env` 并填写真实值，程序启动自动加载（`.env` 已加入 `.gitignore`，不会被提交）。

### 关键注意事项

1. **本地运行需要配置 API 密钥**
   - `TENCENTCLOUD_SECRETID` 和 `TENCENTCLOUD_SECRETKEY` 在本地为空字符串，不配置会导致 API 签名失败。
   - 在 [腾讯云访问管理](https://console.cloud.tencent.com/cam/capi) 创建或获取密钥。

2. **CloudRun 环境会自动注入凭证**
   - 在 CloudRun 中运行时，平台会自动通过环境变量注入 `TENCENTCLOUD_SECRETID`、`TENCENTCLOUD_SECRETKEY`、`TENCENTCLOUD_SESSIONTOKEN`，**不要手动在云托管控制台设置这些值**，否则可能覆盖自动注入导致失败。

3. **环境 ID 必须与 CloudBase 环境一致**
   - `TCB_ENV_ID` 默认值 `knowlege-graph-env-d7cwud346b70b` 是项目创建时绑定的环境，如需切换请修改环境变量。

4. **权限要求**
   - 使用 CloudBase 数据库前，确保该环境的「数据库」服务已启用。
   - API 密钥对应的账号需要有 CloudBase 数据库的操作权限（如 `QcloudTCBFullAccess`）。

## API 接口

启动后访问 `/docs` 查看完整 Swagger 文档。主要接口：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 服务信息 |
| `GET` | `/health` | 健康检查 + 集合列表 |
| `GET` | `/collections` | 列出所有集合 |
| `GET` | `/collections/{name}` | 检查集合是否存在 |
| `POST` | `/collections/{collection}/query` | 查询文档 |
| `POST` | `/collections/{collection}/insert` | 插入文档 |
| `PUT` | `/collections/{collection}/update` | 更新文档 |
| `DELETE` | `/collections/{collection}/delete` | 删除文档 |
| `GET` | `/collections/{collection}/count` | 统计文档数量 |
| `POST` | `/vision/recognize` | 图片上传识别（multipart） |
| `POST` | `/vision/recognize-url` | 图片 URL 识别（JSON） |
| `POST` | `/vision/recognize-base64` | 图片 Base64 识别（JSON） |

### 查询示例

```bash
# 健康检查
curl http://localhost:8080/health

# 查询文档
curl -X POST http://localhost:8080/collections/users/query \
  -H "Content-Type: application/json" \
  -d '{"where": {"age": {"$gt": 18}}, "limit": 10}'

# 插入文档
curl -X POST http://localhost:8080/collections/users/insert \
  -H "Content-Type: application/json" \
  -d '{"data": {"name": "张三", "age": 25}}'

# 更新文档
curl -X PUT http://localhost:8080/collections/users/update \
  -H "Content-Type: application/json" \
  -d '{"where": {"name": "张三"}, "data": {"$set": {"age": 26}}}'

# 删除文档
curl -X DELETE http://localhost:8080/collections/users/delete \
  -H "Content-Type: application/json" \
  -d '{"where": {"name": "张三"}}'
```

### 图片识别示例

#### 1. 上传本地图片识别

```bash
curl -X POST http://localhost:8080/vision/recognize \
  -F "file=@/path/to/english-textbook.jpg"
```

#### 2. 通过图片 URL 识别

```bash
curl -X POST http://localhost:8080/vision/recognize-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/textbook-page.jpg"}'
```

#### 3. 通过 Base64 编码识别

适用于前端 Canvas 截图、拍照后直接传 Base64 的场景。

```bash
# 生成 Base64 并直接调用
BASE64=$(base64 -i /path/to/image.jpg | tr -d '\n')
curl -X POST http://localhost:8080/vision/recognize-base64 \
  -H "Content-Type: application/json" \
  -d "{\"base64\": \"$BASE64\", \"mime_type\": \"image/jpeg\"}"

# 也支持完整的 data:image 前缀
curl -X POST http://localhost:8080/vision/recognize-base64 \
  -H "Content-Type: application/json" \
  -d '{"base64": "data:image/jpeg;base64,iVBORw0KGgo...", "mime_type": "image/jpeg"}'
```

#### 返回示例

```json
{
  "language": "en",
  "material_type": "textbook",
  "title": "The Past and Present of Transport",
  "sentences": [
    {
      "index": 1,
      "text": "In the past, people traveled by horse.",
      "translation": "过去，人们骑马出行。",
      "level": "A2",
      "keywords": ["traveled", "horse", "past"]
    },
    {
      "index": 2,
      "text": "Nowadays, high-speed trains connect major cities.",
      "translation": "如今，高铁连接了各大城市。",
      "level": "B1",
      "keywords": ["nowadays", "high-speed trains", "connect", "major"]
    }
  ],
  "total_sentences": 2,
  "summary": "一张关于交通工具今昔对比的英语教材图片。",
  "_storage": {
    "stored": true,
    "lesson_id": "lesson_9f3a2b1c...",
    "sentence_count": 2
  }
}
```

> 识别结果会**自动存储**到 CloudBase 文档型数据库的 `chapter`、`lesson`、`sentence_v2` 集合中（带 `textbook_id` 时同步写入/复用 `textbook_v2`）。非英文材料会跳过存储，`_storage.stored` 为 `false`。

### 存储数据模型

| 集合 | 说明 | 关键字段 |
|------|------|----------|
| `textbook_v2` | 教材 | `textbook_id`, `title`, `version` |
| `chapter` | 章 | `chapter_id`, `textbook_id`, `title` |
| `lesson` | 课（原 unit 语义） | `lesson_id`, `chapter_id`, `textbook_id`, `title` |
| `sentence_v2` | 单句 | `sentence_id`, `lesson_id`, `chapter_id`, `textbook_id`, `text`, `translation`, `level`, `keywords`, `index` |

可通过已有的 CRUD 接口查询存储内容：

```bash
# 查询所有教材
curl -X POST http://localhost:8080/collections/textbook_v2/query \
  -H "Content-Type: application/json" \
  -d '{}'

# 查询某个 lesson 下的所有句子
curl -X POST http://localhost:8080/collections/sentence_v2/query \
  -H "Content-Type: application/json" \
  -d '{"where": {"lesson_id": "lesson_9f3a2b1c..."}}'
```

## 技术栈

- **FastAPI** — Web 框架，自动生成 OpenAPI 文档
- **uvicorn** — ASGI 服务器
- **httpx** — 异步 HTTP 客户端，调用腾讯云 API
- **OpenAI SDK** — 兼容接口调用火山引擎豆包视觉模型
- **python-dotenv** — 从 `.env` 文件加载环境变量
- **腾讯云 API v3 TC3-HMAC-SHA256** 签名机制
