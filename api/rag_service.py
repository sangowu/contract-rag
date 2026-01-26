from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
from contextlib import asynccontextmanager
from loguru import logger
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rag.retrieval import retrieve_top_k, retrieve_top_k_hybrid, rerank_results, get_reranker_model
from src.rag.embedding import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务生命周期管理：启动时预加载模型"""
    # 启动时预加载模型（预热）
    logger.info("Preloading models on startup...")
    try:
        # 预加载embedding模型
        logger.info("Preloading embedding model...")
        get_model()
        logger.success("Embedding model loaded")
        
        # 预加载reranker模型
        logger.info("Preloading reranker model...")
        get_reranker_model()
        logger.success("Reranker model loaded")
        
        logger.success("All models preloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to preload some models: {e}, will load on first request")
    
    yield  # 服务运行
    
    # 关闭时清理（如果需要）
    logger.info("Shutting down API service...")


app = FastAPI(
    title="RAG Service API",
    description="Contract Intelligence Analysis RAG Service",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    """检索请求模型"""
    query: str
    top_k_retrieval: int = 100
    file_name: Optional[str] = None
    use_hybrid: bool = True
    use_rerank: bool = True
    top_k_reranked: int = 10


class SearchResponse(BaseModel):
    """检索响应模型"""
    ok: bool
    data: List[Dict[str, Union[str, float, int]]]  # 允许字符串、浮点数和整数类型
    error: Optional[str] = None


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "service": "RAG Service API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


@app.post("/api/retrieval/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    检索相关合同条款
    
    Args:
        request: 检索请求，包含查询、top_k等参数
    
    Returns:
        SearchResponse: 包含检索结果的响应
    
    Raises:
        HTTPException: 当检索失败时
    """
    try:
        logger.info(f"Search request: query='{request.query[:50]}...', top_k={request.top_k_retrieval}, file_name={request.file_name}")
        
        # 执行检索
        if request.use_hybrid:
            retrieved_data = retrieve_top_k_hybrid(
                query=request.query,
                top_k_shown=request.top_k_retrieval,
                file_name=request.file_name,
                top_k_retrieval=request.top_k_retrieval
            )
        else:
            retrieved_data = retrieve_top_k(
                query=request.query,
                top_k_shown=request.top_k_retrieval,
                file_name=request.file_name,
                top_k_retrieval=request.top_k_retrieval
            )
        
        # 如果需要重排序
        if request.use_rerank and retrieved_data:
            retrieved_data = rerank_results(
                query=request.query,
                candidate_chunks=retrieved_data,
                top_k=request.top_k_reranked
            )
        
        logger.info(f"Retrieved {len(retrieved_data)} results")
        
        return SearchResponse(
            ok=True,
            data=retrieved_data
        )
        
    except Exception as e:
        error_msg = str(e)
        error_type = type(e).__name__
        logger.error(f"Error during search: {error_type}: {error_msg}", exc_info=True)
        
        # 返回更详细的错误信息
        return SearchResponse(
            ok=False,
            data=[],
            error=f"{error_type}: {error_msg}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
