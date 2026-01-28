"""
API 请求/响应模型
"""

from api.schemas.base import (
    BaseResponse,
    PaginatedResponse,
    ErrorDetail,
    HealthStatus,
    FileInfo,
)

from api.schemas.retrieval import (
    RetrievalRequest,
    RetrievalResponse,
    ChunkResult,
    BatchRetrievalRequest,
    BatchRetrievalResponse,
)

from api.schemas.generation import (
    GenerationRequest,
    GenerationResponse,
    RAGRequest,
    RAGResponse,
    StreamChunk,
)

from api.schemas.pdf import (
    PDFUploadResponse,
    PDFParseRequest,
    PDFParseResponse,
    PDFStatusResponse,
    PDFListResponse,
    BBoxInfo,
    TextBlockInfo,
    TableInfo,
)

__all__ = [
    # Base
    'BaseResponse',
    'PaginatedResponse',
    'ErrorDetail',
    'HealthStatus',
    'FileInfo',
    # Retrieval
    'RetrievalRequest',
    'RetrievalResponse',
    'ChunkResult',
    'BatchRetrievalRequest',
    'BatchRetrievalResponse',
    # Generation
    'GenerationRequest',
    'GenerationResponse',
    'RAGRequest',
    'RAGResponse',
    'StreamChunk',
    # PDF
    'PDFUploadResponse',
    'PDFParseRequest',
    'PDFParseResponse',
    'PDFStatusResponse',
    'PDFListResponse',
    'BBoxInfo',
    'TextBlockInfo',
    'TableInfo',
]
