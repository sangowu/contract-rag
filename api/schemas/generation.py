"""
生成服务请求/响应模型
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class GenerationRequest(BaseModel):
    """生成请求"""
    query: str = Field(..., description="用户问题")
    contexts: List[str] = Field(default_factory=list, description="检索到的上下文")
    
    # 生成选项
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.1, ge=0, le=2, description="生成温度")
    
    # 系统提示
    system_prompt: Optional[str] = Field(default=None, description="自定义系统提示")


class GenerationResponse(BaseModel):
    """生成响应"""
    ok: bool = True
    answer: str = ""
    error: Optional[str] = None
    
    # 元信息
    model: str = ""
    tokens_used: int = 0
    generation_time_ms: float = 0


class RAGRequest(BaseModel):
    """完整 RAG 请求 (检索 + 生成)"""
    query: str = Field(..., description="用户问题")
    file_name: Optional[str] = Field(default=None, description="指定文件名")
    
    # 检索选项
    top_k: int = Field(default=10, ge=1, le=50, description="检索结果数量")
    use_rerank: bool = Field(default=True, description="是否使用重排序")
    
    # 生成选项
    max_tokens: int = Field(default=512, ge=1, le=4096, description="最大生成长度")
    temperature: float = Field(default=0.1, ge=0, le=2, description="生成温度")
    
    # 返回选项
    return_contexts: bool = Field(default=True, description="是否返回上下文")
    return_scores: bool = Field(default=True, description="是否返回分数")


class RAGResponse(BaseModel):
    """完整 RAG 响应"""
    ok: bool = True
    answer: str = ""
    error: Optional[str] = None
    
    # 检索信息
    contexts: Optional[List[Dict[str, Any]]] = None
    
    # 引用信息 (用于前端高亮)
    citations: Optional[List[Dict[str, Any]]] = None
    
    # 性能指标
    retrieval_time_ms: float = 0
    generation_time_ms: float = 0
    total_time_ms: float = 0


class StreamChunk(BaseModel):
    """流式响应块"""
    type: str = "text"  # text, citation, done
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
