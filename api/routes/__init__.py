"""
API 路由模块
"""

from api.routes.health import router as health_router
from api.routes.retrieval import router as retrieval_router
from api.routes.generation import router as generation_router
from api.routes.pdf import router as pdf_router

__all__ = [
    'health_router',
    'retrieval_router',
    'generation_router',
    'pdf_router',
]
