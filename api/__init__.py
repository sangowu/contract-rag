"""
CUAD 合同助手 - API 模块

提供 RESTful API 服务:
- /api/retrieval - 文档检索
- /api/generation - 答案生成
- /api/pdf - PDF 处理
- /health - 健康检查

使用方式:
    # 启动服务
    python scripts/run_api.py --mode dev
    
    # 或使用 uvicorn
    uvicorn api.app:app --reload
"""

from api.app import app, create_app

__all__ = ['app', 'create_app']
