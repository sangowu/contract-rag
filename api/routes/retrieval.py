"""
检索服务路由
"""

import time
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.schemas import (
    RetrievalRequest,
    RetrievalResponse,
    ChunkResult,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
)

# 延迟导入 RAG 模块
_retrieval_module = None
_embedding_module = None


def get_retrieval_module():
    """延迟加载检索模块"""
    global _retrieval_module
    if _retrieval_module is None:
        from src.rag import retrieval
        _retrieval_module = retrieval
    return _retrieval_module


def get_embedding_module():
    """延迟加载嵌入模块"""
    global _embedding_module
    if _embedding_module is None:
        from src.rag import embedding
        _embedding_module = embedding
    return _embedding_module


router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


def convert_to_chunk_result(data: Dict[str, Any]) -> ChunkResult:
    """将检索结果转换为 ChunkResult"""
    return ChunkResult(
        chunk_id=str(data.get("chunk_id", "")),
        clause_text=data.get("clause_text", ""),
        parent_clause_text=data.get("parent_clause_text"),
        file_name=data.get("file_name", ""),
        clause_type=data.get("clause_type"),
        page_num=data.get("page_num"),
        score=data.get("score"),
        rerank_score=data.get("rerank_score"),
        bbox_json=data.get("bbox_json"),
        parent_id=data.get("parent_id"),
    )


@router.post("/search", response_model=RetrievalResponse)
async def search(request: RetrievalRequest):
    """
    检索相关文档
    
    支持向量检索、混合检索和重排序
    
    Args:
        request: 检索请求参数
    
    Returns:
        检索结果列表
    """
    start_time = time.time()
    
    try:
        retrieval = get_retrieval_module()
        
        logger.info(f"Search request: query='{request.query[:50]}...', top_k={request.top_k}")
        
        # 执行检索
        if request.use_hybrid:
            retrieved_data = retrieval.retrieve_top_k_hybrid(
                query=request.query,
                top_k_shown=request.top_k_retrieval,
                file_name=request.file_name,
                top_k_retrieval=request.top_k_retrieval,
            )
        else:
            retrieved_data = retrieval.retrieve_top_k(
                query=request.query,
                top_k_shown=request.top_k_retrieval,
                file_name=request.file_name,
                top_k_retrieval=request.top_k_retrieval,
            )
        
        # 重排序
        if request.use_rerank and retrieved_data:
            retrieved_data = retrieval.rerank_results(
                query=request.query,
                candidate_chunks=retrieved_data,
                top_k=request.rerank_top_k,
            )
        
        # 转换结果
        results = [convert_to_chunk_result(d) for d in retrieved_data]
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Retrieved {len(results)} results in {elapsed_ms:.1f}ms")
        
        return RetrievalResponse(
            ok=True,
            data=results,
            query=request.query,
            total_retrieved=len(results),
            retrieval_time_ms=elapsed_ms,
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return RetrievalResponse(
            ok=False,
            data=[],
            error=str(e),
            query=request.query,
            retrieval_time_ms=elapsed_ms,
        )


@router.post("/batch", response_model=BatchRetrievalResponse)
async def batch_search(request: BatchRetrievalRequest):
    """
    批量检索
    
    Args:
        request: 批量检索请求
    
    Returns:
        批量检索结果
    """
    start_time = time.time()
    
    try:
        results = []
        
        for query_request in request.queries:
            result = await search(query_request)
            results.append(result)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BatchRetrievalResponse(
            ok=True,
            results=results,
            total_time_ms=elapsed_ms,
        )
        
    except Exception as e:
        logger.error(f"Batch search failed: {e}", exc_info=True)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return BatchRetrievalResponse(
            ok=False,
            results=[],
            error=str(e),
            total_time_ms=elapsed_ms,
        )


@router.get("/stats")
async def get_stats():
    """
    获取检索服务统计信息
    """
    try:
        retrieval = get_retrieval_module()
        
        # 获取索引统计
        chunk_df = retrieval.get_chunk_df()
        
        return {
            "ok": True,
            "data": {
                "total_chunks": len(chunk_df) if chunk_df is not None else 0,
                "total_files": chunk_df["file_name"].nunique() if chunk_df is not None else 0,
                "index_loaded": chunk_df is not None,
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"ok": False, "error": str(e)}
