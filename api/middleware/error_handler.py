"""
错误处理中间件
"""

import time
import traceback
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    全局错误处理中间件
    
    捕获所有未处理的异常并返回统一的错误响应
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            # 记录错误
            error_id = f"{int(time.time())}"
            logger.error(
                f"Unhandled error [{error_id}]: {type(e).__name__}: {str(e)}\n"
                f"Path: {request.url.path}\n"
                f"Method: {request.method}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            
            # 返回错误响应
            return JSONResponse(
                status_code=500,
                content={
                    "ok": False,
                    "error": "Internal server error",
                    "error_code": "INTERNAL_ERROR",
                    "error_id": error_id,
                    "detail": str(e) if logger.level("DEBUG").no <= 10 else None,
                }
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    
    记录所有请求的路径、方法、耗时等信息
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # 记录请求
        logger.debug(f"Request: {request.method} {request.url.path}")
        
        # 处理请求
        response = await call_next(request)
        
        # 计算耗时
        duration_ms = (time.time() - start_time) * 1000
        
        # 记录响应
        log_level = "info" if response.status_code < 400 else "warning"
        getattr(logger, log_level)(
            f"Response: {request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.1f}ms"
        )
        
        # 添加响应头
        response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"
        
        return response


class CORSMiddleware:
    """
    CORS 中间件配置
    
    用于配置跨域请求
    """
    
    @staticmethod
    def get_config():
        return {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
