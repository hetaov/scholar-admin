# CloudBase NoSQL API

基于 FastAPI 的 CloudBase 文档型数据库 RESTful API 服务，提供集合管理和文档 CRUD 操作接口。

## 项目结构

```
scholar-admin/
├── config.py              # 项目配置（环境变量读取）
├── main.py                # FastAPI 应用入口 + API 路由
├── services/
│   ├── __init__.py
│   └── database.py        # CloudBase NoSQL 客户端（TC3 签名 + CRUD 封装）
├── requirements.txt       # Python 依赖
├── Dockerfile             # 容器构建文件
├── cloudbaserc.json       # CloudBase 部署配置
└── README.md
```

## 启动方式

### 1. 本地开发

**前置要求：Python 3.11+**

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（见下方配置说明）
export TCB_ENV_ID="your-env-id"
export TENCENTCLOUD_SECRETID="your-secret-id"
export TENCENTCLOUD_SECRETKEY="your-secret-key"

# 启动服务（默认端口 8080）
python main.py

# 或指定端口
PORT=3000 python main.py
```

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

## 技术栈

- **FastAPI** — Web 框架，自动生成 OpenAPI 文档
- **uvicorn** — ASGI 服务器
- **httpx** — 异步 HTTP 客户端，调用腾讯云 API
- **腾讯云 API v3 TC3-HMAC-SHA256** 签名机制
