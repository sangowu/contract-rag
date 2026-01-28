"""
CUAD 合同助手 - FastAPI 应用

主应用入口，整合所有路由和中间件

使用方式:
    # 开发模式
    python scripts/run_api.py --mode dev
    
    # 生产模式
    python scripts/run_api.py --mode prod
    
    # 或直接使用 uvicorn
    uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# 确保项目根目录在 path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置和 GPU 管理
from src.core.config import get_config, init_config
from src.core.gpu_manager import init_gpu_manager, get_gpu_config, GPUManager

# 导入路由
from api.routes import (
    health_router,
    retrieval_router,
    generation_router,
    pdf_router,
)

# 导入中间件
from api.middleware import (
    ErrorHandlerMiddleware,
    RequestLoggingMiddleware,
)


# =============================================================================
# 应用生命周期
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    启动时:
    - 初始化配置
    - 初始化 GPU 管理器
    - 预加载模型
    
    关闭时:
    - 清理资源
    """
    logger.info("=" * 60)
    logger.info("Starting CUAD Contract Assistant API...")
    logger.info("=" * 60)
    
    # 初始化配置
    mode = os.environ.get("APP_MODE", "prod")
    force_gpu_mode = os.environ.get("FORCE_GPU_MODE", None)  # 可选: 强制 GPU 模式
    
    try:
        config = init_config(mode=mode)
        logger.info(f"Configuration loaded: mode={mode}")
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        config = None
    
    # 初始化 GPU 管理器
    try:
        gpu_config = init_gpu_manager(
            project_root=project_root,
            force_mode=force_gpu_mode,
        )
        logger.info(f"GPU Manager initialized: {gpu_config.mode}")
        logger.info(f"  LLM Model: {gpu_config.llm_model_name}")
        logger.info(f"  LLM Devices: {gpu_config.llm_devices}")
        logger.info(f"  Other Devices: {gpu_config.other_devices}")
    except Exception as e:
        logger.error(f"Failed to initialize GPU Manager: {e}")
        raise
    
    # 预加载模型
    await preload_models()
    
    logger.success("API service started successfully")
    
    yield  # 服务运行
    
    # 关闭时清理
    logger.info("Shutting down API service...")
    await cleanup()
    GPUManager.reset()  # 重置 GPU 管理器
    logger.info("API service stopped")


async def preload_models():
    """预加载模型"""
    from api.routes.health import set_model_status
    
    # 获取 GPU 配置
    try:
        gpu_config = get_gpu_config()
        logger.info(f"Preloading models in {gpu_config.mode} mode...")
    except Exception as e:
        logger.warning(f"GPU config not available: {e}")
        gpu_config = None
    
    # 1. 预加载 vLLM (LLM 模型)
    try:
        logger.info("Preloading vLLM...")
        from src.utils.model_loading import get_vllm
        get_vllm()  # 使用 GPU 管理器的配置
        set_model_status("llm", True)
        logger.success("vLLM loaded")
    except Exception as e:
        logger.warning(f"Failed to load vLLM: {e}")
        set_model_status("llm", False)
    
    # 2. 预加载 embedding 模型
    try:
        logger.info("Preloading embedding model...")
        from src.rag.embedding import get_model
        get_model()
        set_model_status("embedding", True)
        logger.success("Embedding model loaded")
    except Exception as e:
        logger.warning(f"Failed to load embedding model: {e}")
        set_model_status("embedding", False)
    
    # 3. 预加载 reranker 模型
    try:
        logger.info("Preloading reranker model...")
        from src.rag.retrieval import get_reranker_model
        get_reranker_model()
        set_model_status("reranker", True)
        logger.success("Reranker model loaded")
    except Exception as e:
        logger.warning(f"Failed to load reranker model: {e}")
        set_model_status("reranker", False)


async def cleanup():
    """清理资源"""
    try:
        from src.utils.model_loading import release_all_models, release_vllm
        release_vllm()  # 先释放 vLLM
        release_all_models()  # 释放其他模型
        logger.info("All models released")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")


# =============================================================================
# 创建应用
# =============================================================================

def create_app() -> FastAPI:
    """
    创建 FastAPI 应用
    
    Returns:
        FastAPI 应用实例
    """
    app = FastAPI(
        title="CUAD Contract Assistant API",
        description="""
        合同智能分析助手 API
        
        功能:
        - 文档检索: 基于向量和 BM25 的混合检索
        - 智能问答: 基于检索增强的答案生成
        - PDF 处理: PDF 解析、BBox 提取、表格处理
        - 评估服务: RAGAS 和传统指标评估
        
        版本: 2.0.0
        """,
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # 添加中间件
    add_middleware(app)
    
    # 注册路由
    register_routes(app)
    
    return app


def add_middleware(app: FastAPI):
    """添加中间件"""
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 错误处理
    app.add_middleware(ErrorHandlerMiddleware)
    
    # 请求日志
    app.add_middleware(RequestLoggingMiddleware)


def register_routes(app: FastAPI):
    """注册路由"""
    # 根路由
    @app.get("/")
    async def root():
        return {
            "service": "CUAD Contract Assistant API",
            "version": "2.0.0",
            "status": "running",
            "docs": "/docs",
        }
    
    # 注册路由器
    app.include_router(health_router)
    app.include_router(retrieval_router, prefix="/api")
    app.include_router(generation_router, prefix="/api")
    app.include_router(pdf_router, prefix="/api")


# 创建应用实例
app = create_app()


# =============================================================================
# 直接运行
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
    )
