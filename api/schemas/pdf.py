"""
PDF 服务请求/响应模型
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class PDFUploadResponse(BaseModel):
    """PDF 上传响应"""
    ok: bool = True
    error: Optional[str] = None
    
    # 文件信息
    file_id: str = ""
    filename: str = ""
    file_size: int = 0
    
    # 状态
    status: str = "uploaded"  # uploaded, processing, completed, failed
    upload_time: datetime = Field(default_factory=datetime.now)


class PDFParseRequest(BaseModel):
    """PDF 解析请求"""
    file_id: str = Field(..., description="文件 ID")
    
    # 解析选项
    extract_tables: bool = Field(default=True, description="是否提取表格")
    extract_bbox: bool = Field(default=True, description="是否提取 BBox")
    use_ocr: bool = Field(default=False, description="是否使用 OCR")
    
    # 分块选项
    chunk_size: int = Field(default=500, ge=100, le=2000, description="分块大小")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="分块重叠")


class BBoxInfo(BaseModel):
    """边界框信息"""
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float


class TextBlockInfo(BaseModel):
    """文本块信息"""
    block_id: str
    text: str
    bbox: BBoxInfo
    block_type: str = "text"
    page_num: int


class TableInfo(BaseModel):
    """表格信息"""
    table_id: str
    page_num: int
    row_count: int
    col_count: int
    markdown: str
    summary: Optional[str] = None
    bbox: BBoxInfo


class PDFParseResponse(BaseModel):
    """PDF 解析响应"""
    ok: bool = True
    error: Optional[str] = None
    
    # 文件信息
    file_id: str = ""
    filename: str = ""
    page_count: int = 0
    
    # 解析结果
    text_blocks: List[TextBlockInfo] = Field(default_factory=list)
    tables: List[TableInfo] = Field(default_factory=list)
    
    # 分块结果
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    chunk_count: int = 0
    
    # 性能
    parse_time_ms: float = 0
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PDFStatusResponse(BaseModel):
    """PDF 处理状态响应"""
    ok: bool = True
    file_id: str
    status: str  # uploaded, processing, completed, failed
    progress: float = 0  # 0-100
    message: str = ""
    
    # 完成时的结果
    result: Optional[PDFParseResponse] = None


class PDFListResponse(BaseModel):
    """PDF 列表响应"""
    ok: bool = True
    files: List[Dict[str, Any]] = Field(default_factory=list)
    total: int = 0
