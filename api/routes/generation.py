"""
生成服务路由
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
import json

from api.schemas import (
    GenerationRequest,
    GenerationResponse,
    RAGRequest,
    RAGResponse,
)

# 延迟导入模块
_llm_module = None
_retrieval_module = None


def get_llm_module():
    """延迟加载 LLM 模块"""
    global _llm_module
    if _llm_module is None:
        from src.inference import llm_inference
        _llm_module = llm_inference
    return _llm_module


def get_retrieval_module():
    """延迟加载检索模块"""
    global _retrieval_module
    if _retrieval_module is None:
        from src.rag import retrieval
        _retrieval_module = retrieval
    return _retrieval_module


router = APIRouter(prefix="/generation", tags=["Generation"])


def build_prompt(query: str, contexts: List[str], system_prompt: Optional[str] = None) -> str:
    """构建 LLM 提示"""
    context_text = "\n\n".join(contexts)
    
    if system_prompt:
        prompt = f"""{system_prompt}

Question: {query}

Relevant contract clauses:
{context_text}

Answer:"""
    else:
        prompt = f"""You are a helpful contract analysis assistant. Based on the provided contract clauses, answer the question accurately and concisely.

Question: {query}

Relevant contract clauses:
{context_text}

Answer:"""
    
    return prompt


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest):
    """
    生成答案
    
    基于提供的上下文生成答案
    
    Args:
        request: 生成请求
    
    Returns:
        生成的答案
    """
    start_time = time.time()
    
    try:
        llm = get_llm_module()
        
        if not request.contexts:
            return GenerationResponse(
                ok=False,
                error="No contexts provided",
                generation_time_ms=(time.time() - start_time) * 1000,
            )
        
        # 构建提示
        prompt = build_prompt(request.query, request.contexts, request.system_prompt)
        
        logger.info(f"Generating answer for: '{request.query[:50]}...'")
        
        # 生成答案
        answer = llm.llm_generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Generated answer in {elapsed_ms:.1f}ms")
        
        return GenerationResponse(
            ok=True,
            answer=answer,
            model="qwen3-8b",  # 从配置获取
            generation_time_ms=elapsed_ms,
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        elapsed_ms = (time.time() - start_time) * 1000
        
        return GenerationResponse(
            ok=False,
            error=str(e),
            generation_time_ms=elapsed_ms,
        )


@router.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """
    完整 RAG 查询
    
    执行检索 + 生成的完整 RAG 流程
    
    Args:
        request: RAG 请求
    
    Returns:
        RAG 响应，包含答案和上下文
    """
    total_start = time.time()
    
    try:
        retrieval = get_retrieval_module()
        llm = get_llm_module()
        
        logger.info(f"RAG query: '{request.query[:50]}...'")
        
        # 1. 检索
        retrieval_start = time.time()
        
        retrieved_data = retrieval.retrieve_top_k_hybrid(
            query=request.query,
            top_k_shown=request.top_k * 2,  # 检索更多，重排序后取 top_k
            file_name=request.file_name,
            top_k_retrieval=request.top_k * 3,
        )
        
        if request.use_rerank and retrieved_data:
            retrieved_data = retrieval.rerank_results(
                query=request.query,
                candidate_chunks=retrieved_data,
                top_k=request.top_k,
            )
        
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        # 2. 构建上下文
        contexts = [d.get("clause_text", "") for d in retrieved_data[:request.top_k]]
        
        # 3. 生成
        generation_start = time.time()
        
        prompt = build_prompt(request.query, contexts)
        answer = llm.llm_generate(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        
        generation_time = (time.time() - generation_start) * 1000
        total_time = (time.time() - total_start) * 1000
        
        # 4. 构建响应
        response = RAGResponse(
            ok=True,
            answer=answer,
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
        )
        
        # 可选返回上下文
        if request.return_contexts:
            response.contexts = [
                {
                    "chunk_id": d.get("chunk_id", ""),
                    "text": d.get("clause_text", ""),
                    "file_name": d.get("file_name", ""),
                    "score": d.get("rerank_score", d.get("score")),
                    "bbox_json": d.get("bbox_json"),
                }
                for d in retrieved_data[:request.top_k]
            ]
        
        # 构建引用信息
        response.citations = [
            {
                "index": i + 1,
                "chunk_id": d.get("chunk_id", ""),
                "file_name": d.get("file_name", ""),
                "page_num": d.get("page_num"),
                "bbox_json": d.get("bbox_json"),
            }
            for i, d in enumerate(retrieved_data[:request.top_k])
        ]
        
        logger.info(f"RAG completed in {total_time:.1f}ms (retrieval: {retrieval_time:.1f}ms, generation: {generation_time:.1f}ms)")
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}", exc_info=True)
        total_time = (time.time() - total_start) * 1000
        
        return RAGResponse(
            ok=False,
            error=str(e),
            total_time_ms=total_time,
        )


@router.post("/stream")
async def rag_stream(request: RAGRequest):
    """
    流式 RAG 查询
    
    以 Server-Sent Events 格式返回流式响应
    """
    async def generate_stream():
        try:
            retrieval = get_retrieval_module()
            llm = get_llm_module()
            
            # 1. 检索 (非流式)
            yield f"data: {json.dumps({'type': 'status', 'content': 'Retrieving relevant documents...'})}\n\n"
            
            retrieved_data = retrieval.retrieve_top_k_hybrid(
                query=request.query,
                top_k_shown=request.top_k * 2,
                file_name=request.file_name,
            )
            
            if request.use_rerank and retrieved_data:
                yield f"data: {json.dumps({'type': 'status', 'content': 'Reranking results...'})}\n\n"
                retrieved_data = retrieval.rerank_results(
                    query=request.query,
                    candidate_chunks=retrieved_data,
                    top_k=request.top_k,
                )
            
            # 发送引用信息
            citations = [
                {
                    "index": i + 1,
                    "chunk_id": d.get("chunk_id", ""),
                    "file_name": d.get("file_name", ""),
                }
                for i, d in enumerate(retrieved_data[:request.top_k])
            ]
            yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
            
            # 2. 生成 (模拟流式)
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating answer...'})}\n\n"
            
            contexts = [d.get("clause_text", "") for d in retrieved_data[:request.top_k]]
            prompt = build_prompt(request.query, contexts)
            
            # TODO: 实现真正的流式生成
            answer = llm.llm_generate(prompt, max_tokens=request.max_tokens)
            
            # 分块发送答案
            chunk_size = 20
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
