"""
检索服务请求/响应模型
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """检索请求"""
    query: str = Field(..., description="查询问题")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    file_name: Optional[str] = Field(default=None, description="指定文件名过滤")
    
    # 检索选项
    use_hybrid: bool = Field(default=True, description="是否使用混合检索")
    use_rerank: bool = Field(default=True, description="是否使用重排序")
    
    # 高级选项
    top_k_retrieval: int = Field(default=50, ge=1, le=200, description="初始检索数量")
    rerank_top_k: int = Field(default=10, ge=1, le=50, description="重排序后返回数量")


class ChunkResult(BaseModel):
    """单个检索结果"""
    chunk_id: str
    clause_text: str
    parent_clause_text: Optional[str] = None
    file_name: str
    clause_type: Optional[str] = None
    page_num: Optional[int] = None
    
    # 分数
    score: Optional[float] = None
    rerank_score: Optional[float] = None
    
    # BBox 信息 (用于前端高亮)
    bbox_json: Optional[str] = None
    
    # 父子关系
    parent_id: Optional[str] = None


class RetrievalResponse(BaseModel):
    """检索响应"""
    ok: bool = True
    data: List[ChunkResult] = Field(default_factory=list)
    error: Optional[str] = None
    
    # 元信息
    query: str = ""
    total_retrieved: int = 0
    retrieval_time_ms: float = 0


class BatchRetrievalRequest(BaseModel):
    """批量检索请求"""
    queries: List[RetrievalRequest]
    parallel: bool = Field(default=True, description="是否并行处理")


class BatchRetrievalResponse(BaseModel):
    """批量检索响应"""
    ok: bool = True
    results: List[RetrievalResponse] = Field(default_factory=list)
    error: Optional[str] = None
    total_time_ms: float = 0
