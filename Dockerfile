# 使用多阶段构建 .NET 项目
FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:9.0-bookworm-slim AS build-env

WORKDIR /root/build
ARG TARGETARCH

COPY c /root/build

RUN dotnet publish -p:DebugType="none" -a $TARGETARCH -f "net9.0" -o "/root/out" "Lagrange.OneBot"

# 运行环境
FROM python:3.10-slim

WORKDIR /app

# 安装 Supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    supervisor \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制 .NET 项目
COPY --from=build-env /root/out /app/dotnet/bin
COPY c/Lagrange.OneBot/Resources/docker-entrypoint.sh /app/dotnet/bin/docker-entrypoint.sh

# 复制 Python 项目
COPY python /app/python

# 赋予执行权限
RUN chmod +x /app/dotnet/bin/docker-entrypoint.sh

# 安装 Python 依赖
RUN python -m venv /app/python/venv && \
    /app/python/venv/bin/pip install --upgrade pip && \
    /app/python/venv/bin/pip install -r /app/python/requirements.txt --no-cache-dir && \
    /app/python/venv/bin/pip install socksio wechatpy cryptography --no-cache-dir

# 复制 supervisord 配置
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 暴露端口
EXPOSE 6185 6186

# 使用 Supervisor 管理进程
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
