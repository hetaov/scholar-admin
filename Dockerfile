FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/

# F3.2 A4 练习纸渲染：安装 Chromium（playwright）与中文字体
# - playwright install --with-deps 会安装 Chromium 及其系统依赖（libnss3/libatk 等）
# - fonts-noto-cjk 保证中文与常用数学符号正常渲染（A4 PDF/PNG）
# - PLAYWRIGHT_DOWNLOAD_HOST 可指定国内镜像加速浏览器下载
RUN PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright/ \
    python -m playwright install --with-deps chromium \
    && apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

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
