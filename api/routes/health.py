"""
健康检查路由
"""

import time
import psutil
from fastapi import APIRouter, Depends
from loguru import logger

from api.schemas import HealthStatus, BaseResponse


router = APIRouter(prefix="/health", tags=["Health"])

# 服务启动时间
_start_time = time.time()

# 模型加载状态
_models_status = {
    "embedding": False,
    "reranker": False,
    "llm": False,
}


def set_model_status(model_name: str, loaded: bool):
    """设置模型加载状态"""
    global _models_status
    if model_name in _models_status:
        _models_status[model_name] = loaded


def get_models_status() -> dict:
    """获取模型加载状态"""
    return _models_status.copy()


@router.get("", response_model=BaseResponse[HealthStatus])
async def health_check():
    """
    健康检查端点
    
    返回服务状态、运行时间、模型加载状态和内存使用情况
    """
    try:
        # 计算运行时间
        uptime = time.time() - _start_time
        
        # 获取内存使用
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        
        status = HealthStatus(
            status="healthy",
            version="2.0.0",
            uptime_seconds=uptime,
            models_loaded=get_models_status(),
            memory_usage_mb=round(memory_mb, 2),
        )
        
        return BaseResponse(ok=True, data=status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return BaseResponse(
            ok=False,
            error=str(e),
            data=HealthStatus(status="unhealthy")
        )


@router.get("/ready")
async def readiness_check():
    """
    就绪检查端点
    
    检查所有必要的服务是否已准备好处理请求
    """
    models = get_models_status()
    
    # 至少需要 embedding 模型加载完成
    if not models.get("embedding", False):
        return {"ready": False, "reason": "Embedding model not loaded"}
    
    return {"ready": True}


@router.get("/live")
async def liveness_check():
    """
    存活检查端点
    
    简单的存活检查，用于 Kubernetes 等编排系统
    """
    return {"alive": True}
