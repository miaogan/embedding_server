# 使用Python 3.12作为基础镜像
FROM python:3.12.11
# 设置工作目录
WORKDIR /app

# 复制requirements文件并安装Python依赖
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 复制应用代码
COPY app/ ./app

# 创建模型目录
RUN mkdir -p /models/bge-m3

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "app/main.py"]
