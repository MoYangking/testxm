# 选择体积较小的 Python 运行时镜像
FROM python:3.11-slim


# 工作目录（容器内）
WORKDIR /app

# 拷贝代码（确保与本 Dockerfile 同目录的 app.py 就是你的服务代码文件）
COPY app.py /app/app.py

# 安装运行所需依赖（无需系统编译依赖）
# 注意：如果你有 requirements.txt，也可以改为：COPY requirements.txt /app && pip install -r requirements.txt
RUN pip install -U fastapi "uvicorn[standard]" httpx pydantic

# 对外暴露端口（默认 8000）
EXPOSE 8000

# 可选：通过环境变量 PORT 控制监听端口（未设置时默认为 8000）
# 不建议在镜像里写死 Cookie 等敏感信息，改为运行时通过环境变量或文件注入
ENV PORT=8000

# 启动命令（使用 uvicorn 启动 FastAPI）
# 说明：
# - app:app 指的是 app.py 中的 app 实例
# - --host 0.0.0.0 允许外部访问
# - --port 读取环境变量 PORT，缺省 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
