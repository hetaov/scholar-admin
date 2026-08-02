FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/

# 复制项目文件
COPY . .

# 暴露端口
EXPOSE 8080

# 启动服务
CMD ["python", "main.py"]
