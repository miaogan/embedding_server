from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import logging
import os

# --- 配置 ---
# 优先从环境变量读取配置，如果没有则使用默认值
MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", 'E:\\bge-m3')  # 例如: /models/bge-m3 或 E:\bge-m3
DEVICE = os.getenv("EMBEDDING_DEVICE", 'cpu')  # 例如: cpu, cuda:0
USE_FP16 = os.getenv("EMBEDDING_USE_FP16", 'False').lower() == 'true'
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 512))

app = FastAPI(title="BGE-M3 Embedding Service", description="提供文本嵌入生成功能的 API")


# --- Pydantic 模型用于请求和响应 ---
class EmbeddingRequest(BaseModel):
    texts: List[str]


class EmbeddingData(BaseModel):
    # 注意：FastAPI/Pydantic 对稀疏向量 (dict) 的原生支持可能有限，
    # 这里使用 List[Dict] 或 List[List[float]] (密集) 和 List[Dict[int, float]] (稀疏) 来表示
    sparse: List[Dict[int, float]]
    dense: List[List[float]]


class EmbeddingResponse(BaseModel):
    success: bool
    data: Optional[EmbeddingData] = None
    error: Optional[str] = None


# --- 初始化嵌入模型 ---
print("正在加载 BGE-M3 模型...")
logger = logging.getLogger("uvicorn")
try:
    embedding_func = BGEM3EmbeddingFunction(
        model_name=MODEL_PATH,
        device=DEVICE,
        use_fp16=USE_FP16
    )
    print("BGE-M3 模型加载成功。")
    logger.info("BGE-M3 模型加载成功。")
except Exception as e:
    error_msg = f"模型加载失败: {e}"
    print(error_msg)
    logger.error(error_msg)
    embedding_func = None


# --- API 端点 ---
@app.post("/embed", response_model=EmbeddingResponse, status_code=status.HTTP_200_OK)
async def get_embeddings(request: EmbeddingRequest):
    """
    接收 JSON 格式的文本列表，返回对应的稀疏和稠密嵌入。
    """
    if not embedding_func:
        logger.error("嵌入模型未正确加载")
        raise HTTPException(status_code=500, detail="嵌入模型未正确加载")

    texts = request.texts
    if not texts:
        logger.warning("收到空的文本列表")
        raise HTTPException(status_code=400, detail="文本列表不能为空")

    # 可选：截断过长文本
    texts = [text[:MAX_TEXT_LENGTH] for text in texts]

    try:
        logger.info(f"正在为 {len(texts)} 个文本生成嵌入...")
        embeddings = embedding_func(texts)
        logger.info("嵌入生成完成。")

        # 确保返回的数据是可序列化的标准 Python 类型
        # pymilvus 的 sparse 向量通常是 csr_array 或 dict, 需要转换
        sparse_vectors = []
        for sparse_vec in embeddings["sparse"]:
            if hasattr(sparse_vec, 'todok'):  # 如果是 csr_array
                # 转换为 {index: value} 字典
                dok_dict = sparse_vec.todok()
                sparse_vectors.append({int(k): float(v) for k, v in dok_dict.items()})
            elif isinstance(sparse_vec, dict):
                sparse_vectors.append({int(k): float(v) for k, v in sparse_vec.items()})
            else:
                # 如果已经是期望格式或需要其他处理
                sparse_vectors.append(sparse_vec)

        dense_vectors = [vec.tolist() for vec in embeddings["dense"]]

        return EmbeddingResponse(
            success=True,
            data=EmbeddingData(sparse=sparse_vectors, dense=dense_vectors)
        )
    except Exception as e:
        error_msg = f"嵌入生成过程中出错: {e}"
        logger.exception(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# 可选：添加一个健康检查端点
@app.get("/health")
async def health_check():
    if embedding_func:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}


if __name__ == "__main__":
    import uvicorn

    # 从环境变量获取 host 和 port，便于容器化部署
    host = os.getenv("EMBEDDING_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("EMBEDDING_SERVICE_PORT", 5000))
    # reload=True 适用于开发环境
    uvicorn.run("main:app", host=host, port=port, reload=False, log_level="info")



