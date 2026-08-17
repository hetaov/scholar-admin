FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/

# 复制项目文件
COPY . .

# 官方 SOE-N SDK（tencentcloud-speech-sdk-python）为纯源码分发（PyPI 无包，见 scripts/soe_n_verify.py），
# 且 .gitignore 排除了 vendor/，git 仓库构建时不会随 COPY 进入镜像。
# 若 vendor 缺失则构建期在线拉取（本地 docker build 上下文含 vendor 时跳过，离线构建不受影响）：
RUN if [ ! -d vendor/tencentcloud-speech-sdk-python/common ]; then \
        apt-get update && apt-get install -y --no-install-recommends git \
        && git clone --depth 1 https://github.com/TencentCloud/tencentcloud-speech-sdk-python.git \
            vendor/tencentcloud-speech-sdk-python \
        && apt-get purge -y git && apt-get autoremove -y \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "main.py"]
