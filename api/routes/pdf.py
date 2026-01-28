"""
PDF 服务路由
"""

import os
import time
import uuid
import shutil
from typing import Dict, Any
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from loguru import logger

from api.schemas import (
    PDFUploadResponse,
    PDFParseRequest,
    PDFParseResponse,
    PDFStatusResponse,
    PDFListResponse,
    TextBlockInfo,
    TableInfo,
    BBoxInfo,
)

# 延迟导入 PDF 模块
_pdf_module = None


def get_pdf_module():
    """延迟加载 PDF 模块"""
    global _pdf_module
    if _pdf_module is None:
        from src import pdf
        _pdf_module = pdf
    return _pdf_module


router = APIRouter(prefix="/pdf", tags=["PDF"])

# 文件存储
UPLOAD_DIR = Path("/root/autodl-tmp/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 处理状态存储 (生产环境应使用 Redis)
_processing_status: Dict[str, Dict[str, Any]] = {}


def get_file_path(file_id: str) -> Path:
    """获取文件路径"""
    return UPLOAD_DIR / f"{file_id}.pdf"


@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传 PDF 文件
    
    Args:
        file: PDF 文件
    
    Returns:
        上传结果，包含文件 ID
    """
    try:
        # 验证文件类型
        if not file.filename.lower().endswith('.pdf'):
            return PDFUploadResponse(
                ok=False,
                error="Only PDF files are allowed",
            )
        
        # 生成文件 ID
        file_id = str(uuid.uuid4())
        
        # 保存文件
        file_path = get_file_path(file_id)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        file_size = len(content)
        
        # 记录状态
        _processing_status[file_id] = {
            "status": "uploaded",
            "filename": file.filename,
            "file_size": file_size,
            "progress": 0,
        }
        
        logger.info(f"Uploaded PDF: {file.filename} -> {file_id} ({file_size} bytes)")
        
        return PDFUploadResponse(
            ok=True,
            file_id=file_id,
            filename=file.filename,
            file_size=file_size,
            status="uploaded",
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return PDFUploadResponse(
            ok=False,
            error=str(e),
        )


@router.post("/parse", response_model=PDFParseResponse)
async def parse_pdf(request: PDFParseRequest, background_tasks: BackgroundTasks = None):
    """
    解析 PDF 文件
    
    Args:
        request: 解析请求
    
    Returns:
        解析结果
    """
    start_time = time.time()
    
    try:
        pdf_module = get_pdf_module()
        
        # 检查文件是否存在
        file_path = get_file_path(request.file_id)
        if not file_path.exists():
            return PDFParseResponse(
                ok=False,
                error=f"File not found: {request.file_id}",
            )
        
        # 更新状态
        if request.file_id in _processing_status:
            _processing_status[request.file_id]["status"] = "processing"
            _processing_status[request.file_id]["progress"] = 10
        
        logger.info(f"Parsing PDF: {request.file_id}")
        
        # 解析 PDF
        doc = pdf_module.parse_pdf(
            str(file_path),
            extract_tables=request.extract_tables,
            extract_bbox=request.extract_bbox,
        )
        
        # 更新进度
        if request.file_id in _processing_status:
            _processing_status[request.file_id]["progress"] = 50
        
        # 转换文本块
        text_blocks = []
        for block in doc.all_text_blocks:
            text_blocks.append(TextBlockInfo(
                block_id=block.block_id,
                text=block.text,
                bbox=BBoxInfo(
                    page_num=block.bbox.page_num,
                    x0=block.bbox.x0,
                    y0=block.bbox.y0,
                    x1=block.bbox.x1,
                    y1=block.bbox.y1,
                ),
                block_type=block.block_type,
                page_num=block.bbox.page_num,
            ))
        
        # 转换表格
        tables = []
        for table in doc.all_tables:
            tables.append(TableInfo(
                table_id=table.block_id,
                page_num=table.bbox.page_num,
                row_count=table.row_count,
                col_count=table.col_count,
                markdown=table.raw_markdown,
                summary=table.summary,
                bbox=BBoxInfo(
                    page_num=table.bbox.page_num,
                    x0=table.bbox.x0,
                    y0=table.bbox.y0,
                    x1=table.bbox.x1,
                    y1=table.bbox.y1,
                ),
            ))
        
        # 生成 chunks
        if request.file_id in _processing_status:
            _processing_status[request.file_id]["progress"] = 75
        
        chunks = doc.to_chunks(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        
        # 完成
        elapsed_ms = (time.time() - start_time) * 1000
        
        if request.file_id in _processing_status:
            _processing_status[request.file_id]["status"] = "completed"
            _processing_status[request.file_id]["progress"] = 100
        
        logger.info(f"Parsed PDF in {elapsed_ms:.1f}ms: {doc.page_count} pages, {len(text_blocks)} blocks, {len(chunks)} chunks")
        
        return PDFParseResponse(
            ok=True,
            file_id=request.file_id,
            filename=_processing_status.get(request.file_id, {}).get("filename", ""),
            page_count=doc.page_count,
            text_blocks=text_blocks,
            tables=tables,
            chunks=chunks,
            chunk_count=len(chunks),
            parse_time_ms=elapsed_ms,
            metadata=doc.metadata,
        )
        
    except Exception as e:
        logger.error(f"Parse failed: {e}", exc_info=True)
        
        if request.file_id in _processing_status:
            _processing_status[request.file_id]["status"] = "failed"
            _processing_status[request.file_id]["error"] = str(e)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return PDFParseResponse(
            ok=False,
            error=str(e),
            file_id=request.file_id,
            parse_time_ms=elapsed_ms,
        )


@router.get("/status/{file_id}", response_model=PDFStatusResponse)
async def get_status(file_id: str):
    """
    获取 PDF 处理状态
    
    Args:
        file_id: 文件 ID
    
    Returns:
        处理状态
    """
    if file_id not in _processing_status:
        return PDFStatusResponse(
            ok=False,
            file_id=file_id,
            status="not_found",
            message="File not found",
        )
    
    status_info = _processing_status[file_id]
    
    return PDFStatusResponse(
        ok=True,
        file_id=file_id,
        status=status_info.get("status", "unknown"),
        progress=status_info.get("progress", 0),
        message=status_info.get("error", ""),
    )


@router.get("/list", response_model=PDFListResponse)
async def list_files():
    """
    列出所有上传的 PDF 文件
    """
    files = []
    
    for file_id, info in _processing_status.items():
        files.append({
            "file_id": file_id,
            "filename": info.get("filename", ""),
            "file_size": info.get("file_size", 0),
            "status": info.get("status", "unknown"),
        })
    
    return PDFListResponse(
        ok=True,
        files=files,
        total=len(files),
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """
    删除 PDF 文件
    
    Args:
        file_id: 文件 ID
    """
    try:
        # 删除文件
        file_path = get_file_path(file_id)
        if file_path.exists():
            file_path.unlink()
        
        # 删除状态
        if file_id in _processing_status:
            del _processing_status[file_id]
        
        logger.info(f"Deleted PDF: {file_id}")
        
        return {"ok": True, "message": f"File {file_id} deleted"}
        
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return {"ok": False, "error": str(e)}
