"""
基础请求/响应模型

定义所有 API 共用的基础模型
"""

from typing import Any, Optional, List, Dict, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime


T = TypeVar('T')


class BaseResponse(BaseModel, Generic[T]):
    """统一响应格式"""
    ok: bool = True
    data: Optional[T] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseResponse[T]):
    """分页响应格式"""
    total: int = 0
    page: int = 1
    page_size: int = 20
    has_more: bool = False


class ErrorDetail(BaseModel):
    """错误详情"""
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    """健康检查状态"""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float = 0
    models_loaded: Dict[str, bool] = Field(default_factory=dict)
    memory_usage_mb: float = 0


class FileInfo(BaseModel):
    """文件信息"""
    filename: str
    file_size: int
    content_type: str
    upload_time: datetime = Field(default_factory=datetime.now)
