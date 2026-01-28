"""
PDF 解析器主模块

功能:
- 解析 PDF 文件提取文本
- 提取 BBox 坐标信息
- 检测和处理表格
- 支持 OCR (扫描件)
- 异常处理和降级策略

使用方式:
    from src.pdf import PDFParser, parse_pdf
    
    # 方式1: 快捷函数
    doc = parse_pdf("contract.pdf")
    
    # 方式2: 完整控制
    parser = PDFParser()
    doc = parser.parse("contract.pdf")
    chunks = doc.to_chunks()
"""

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from src.core.config import get_config
from src.pdf.exceptions import (
    PDFParseError,
    PDFCorruptedError,
    PDFEncryptedError,
    PDFTooLargeError,
    PDFPageLimitError,
    UnsupportedFormatError,
)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class BoundingBox:
    """边界框坐标"""
    page_num: int      # 页码 (从0开始)
    x0: float          # 左上角 x
    y0: float          # 左上角 y
    x1: float          # 右下角 x
    y1: float          # 右下角 y
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BoundingBox':
        return cls(**d)
    
    @classmethod
    def from_fitz_rect(cls, rect: fitz.Rect, page_num: int) -> 'BoundingBox':
        """从 PyMuPDF Rect 创建"""
        return cls(
            page_num=page_num,
            x0=rect.x0,
            y0=rect.y0,
            x1=rect.x1,
            y1=rect.y1,
        )
    
    def __repr__(self) -> str:
        return f"BBox(p{self.page_num}: [{self.x0:.1f},{self.y0:.1f}]-[{self.x1:.1f},{self.y1:.1f}])"


@dataclass
class TextBlock:
    """文本块"""
    block_id: str
    text: str
    bbox: BoundingBox
    block_type: str = "text"     # text / title / list
    font_size: float = 0.0
    font_name: str = ""
    is_bold: bool = False
    confidence: float = 1.0      # OCR 置信度
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['bbox'] = self.bbox.to_dict()
        return d


@dataclass
class TableBlock:
    """表格块"""
    block_id: str
    bbox: BoundingBox
    raw_markdown: str = ""
    raw_html: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    row_count: int = 0
    col_count: int = 0
    summary: str = ""            # LLM 生成的摘要
    context_before: str = ""     # 表格前的上下文
    context_after: str = ""      # 表格后的上下文
    block_type: str = "table"
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['bbox'] = self.bbox.to_dict()
        return d


@dataclass
class PDFPage:
    """PDF 页面"""
    page_num: int
    width: float
    height: float
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[TableBlock] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """获取页面全文"""
        texts = [block.text for block in self.text_blocks]
        return "\n".join(texts)


@dataclass
class PDFDocument:
    """解析后的 PDF 文档"""
    file_path: str
    file_name: str
    page_count: int
    pages: List[PDFPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parse_errors: List[str] = field(default_factory=list)
    
    @property
    def full_text(self) -> str:
        """获取完整文本"""
        return "\n\n".join(page.full_text for page in self.pages)
    
    @property
    def all_text_blocks(self) -> List[TextBlock]:
        """获取所有文本块"""
        blocks = []
        for page in self.pages:
            blocks.extend(page.text_blocks)
        return blocks
    
    @property
    def all_tables(self) -> List[TableBlock]:
        """获取所有表格"""
        tables = []
        for page in self.pages:
            tables.extend(page.tables)
        return tables
    
    def to_chunks(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        include_tables: bool = True,
        compatible_format: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        将文档转换为 chunks (与 cuad_v1_chunks.csv 完全兼容的格式)
        
        chunk_id 格式: {file_name}::{clause_type}::{contract_idx}::{chunk_idx}::{start_char}:{end_char}
        
        Args:
            chunk_size: chunk 大小 (child chunk)
            chunk_overlap: 重叠大小
            include_tables: 是否包含表格
            compatible_format: 是否使用与原始数据兼容的格式
        
        Returns:
            chunk 列表，格式与 cuad_v1_chunks.csv 完全一致
        """
        chunks = []
        contract_idx = 0  # PDF 文档的合同索引
        
        # 合并所有页面的文本块，按页面分组作为 "段落"
        paragraphs = []
        current_paragraph = []
        current_page = -1
        
        for block in self.all_text_blocks:
            text = block.text.strip()
            if not text:
                continue
            
            # 如果换页了，保存当前段落
            if block.bbox.page_num != current_page:
                if current_paragraph:
                    paragraphs.append({
                        'text': '\n'.join(current_paragraph),
                        'page_num': current_page,
                    })
                current_paragraph = []
                current_page = block.bbox.page_num
            
            current_paragraph.append(text)
        
        # 保存最后一个段落
        if current_paragraph:
            paragraphs.append({
                'text': '\n'.join(current_paragraph),
                'page_num': current_page,
            })
        
        # 对每个段落应用 Parent-Child 策略
        chunk_idx = 0
        for para in paragraphs:
            parent_text = para['text']
            page_num = para['page_num']
            
            # 对于 PDF 解析，clause_type 使用 "PDF_Page_{page_num}" 格式
            # 这样可以区分不同页面的内容
            clause_type = f"PDF_Page_{page_num}"
            
            # Parent chunk（完整段落）
            parent_start = 0
            parent_end = len(parent_text)
            
            # 如果文本较短，直接作为一个 chunk
            if len(parent_text) <= chunk_size:
                # chunk_id 格式与原始数据一致
                chunk_id = f"{self.file_name}::{clause_type}::{contract_idx}::{chunk_idx}::{parent_start}:{parent_end}"
                chunks.append({
                    'file_name': self.file_name,
                    'chunk_id': chunk_id,
                    'clause_type': clause_type,
                    'clause_text': parent_text,
                    'parent_clause_text': parent_text,
                    'contract_idx': contract_idx,
                    'chunk_idx': chunk_idx,
                    'start_char': parent_start,
                    'end_char': parent_end,
                    'has_answer': False,  # PDF 解析不知道是否有答案
                    'source': 'pdf',
                })
                chunk_idx += 1
            else:
                # Parent-Child 分块策略：将长文本分割成 child chunks
                start = 0
                child_idx = 0
                
                while start < len(parent_text):
                    end = min(start + chunk_size, len(parent_text))
                    child_text = parent_text[start:end].strip()
                    
                    if child_text:
                        # chunk_id 格式与原始数据一致
                        chunk_id = f"{self.file_name}::{clause_type}::{contract_idx}::{chunk_idx}::{start}:{end}"
                        chunks.append({
                            'file_name': self.file_name,
                            'chunk_id': chunk_id,
                            'clause_type': clause_type,
                            'clause_text': child_text,  # Child chunk 文本
                            'parent_clause_text': parent_text,  # Parent 完整文本
                            'contract_idx': contract_idx,
                            'chunk_idx': chunk_idx,
                            'start_char': start,
                            'end_char': end,
                            'has_answer': False,
                            'source': 'pdf',
                        })
                        chunk_idx += 1
                        child_idx += 1
                    
                    if end >= len(parent_text):
                        break
                    start = max(0, end - chunk_overlap)
        
        # 处理表格
        if include_tables:
            for table in self.all_tables:
                search_text = table.summary if table.summary else table.raw_markdown
                if not search_text:
                    continue
                
                clause_type = f"PDF_Table_{table.bbox.page_num}"
                start_char = 0
                end_char = len(search_text)
                
                chunk_id = f"{self.file_name}::{clause_type}::{contract_idx}::{chunk_idx}::{start_char}:{end_char}"
                chunks.append({
                    'file_name': self.file_name,
                    'chunk_id': chunk_id,
                    'clause_type': clause_type,
                    'clause_text': search_text,
                    'parent_clause_text': table.raw_markdown,
                    'contract_idx': contract_idx,
                    'chunk_idx': chunk_idx,
                    'start_char': start_char,
                    'end_char': end_char,
                    'has_answer': False,
                    'source': 'pdf',
                })
                chunk_idx += 1
        
        return chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'file_path': self.file_path,
            'file_name': self.file_name,
            'page_count': self.page_count,
            'pages': [
                {
                    'page_num': p.page_num,
                    'width': p.width,
                    'height': p.height,
                    'text_blocks': [b.to_dict() for b in p.text_blocks],
                    'tables': [t.to_dict() for t in p.tables],
                }
                for p in self.pages
            ],
            'metadata': self.metadata,
            'parse_errors': self.parse_errors,
        }


# =============================================================================
# PDF 解析器
# =============================================================================

class PDFParser:
    """PDF 解析器"""
    
    def __init__(self):
        """初始化解析器"""
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载 PDF 配置"""
        try:
            config = get_config()
            # 检查是否有 pdf 配置
            if hasattr(config, 'pdf'):
                return {
                    'max_pages': getattr(config.pdf.parser, 'max_pages', 500),
                    'max_file_size_mb': getattr(config.pdf.parser, 'max_file_size_mb', 100),
                    'bbox_granularity': getattr(config.pdf.bbox, 'granularity', 'line'),
                    'table_enabled': getattr(config.pdf.table, 'enabled', True),
                    'ocr_enabled': getattr(config.pdf.ocr, 'enabled', True),
                }
        except Exception as e:
            logger.warning(f"Failed to load PDF config, using defaults: {e}")
        
        # 默认配置
        return {
            'max_pages': 500,
            'max_file_size_mb': 100,
            'bbox_granularity': 'line',
            'table_enabled': True,
            'ocr_enabled': True,
        }
    
    def parse(
        self,
        file_path: str,
        password: Optional[str] = None,
        extract_tables: bool = True,
        extract_bbox: bool = True,
    ) -> PDFDocument:
        """
        解析 PDF 文件
        
        Args:
            file_path: PDF 文件路径
            password: 密码 (如果加密)
            extract_tables: 是否提取表格
            extract_bbox: 是否提取 BBox
        
        Returns:
            PDFDocument 对象
        
        Raises:
            PDFParseError: 解析失败
        """
        file_path = str(file_path)
        
        # 验证文件
        self._validate_file(file_path)
        
        # 打开 PDF
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise PDFCorruptedError(file_path, str(e))
        
        # 检查加密
        if doc.is_encrypted:
            if password:
                if not doc.authenticate(password):
                    raise PDFEncryptedError(file_path)
            else:
                raise PDFEncryptedError(file_path)
        
        # 检查页数
        if doc.page_count > self.config['max_pages']:
            raise PDFPageLimitError(file_path, doc.page_count, self.config['max_pages'])
        
        # 提取元数据
        metadata = self._extract_metadata(doc)
        
        # 解析页面
        pages = []
        parse_errors = []
        
        for page_num in range(doc.page_count):
            try:
                page = self._parse_page(doc, page_num, extract_tables, extract_bbox)
                pages.append(page)
            except Exception as e:
                error_msg = f"Failed to parse page {page_num}: {e}"
                logger.warning(error_msg)
                parse_errors.append(error_msg)
                # 创建空页面
                fitz_page = doc[page_num]
                pages.append(PDFPage(
                    page_num=page_num,
                    width=fitz_page.rect.width,
                    height=fitz_page.rect.height,
                ))
        
        doc.close()
        
        file_name = Path(file_path).name
        
        result = PDFDocument(
            file_path=file_path,
            file_name=file_name,
            page_count=len(pages),
            pages=pages,
            metadata=metadata,
            parse_errors=parse_errors,
        )
        
        logger.info(f"Parsed PDF: {file_name}, pages={len(pages)}, "
                   f"text_blocks={len(result.all_text_blocks)}, "
                   f"tables={len(result.all_tables)}")
        
        return result
    
    def _validate_file(self, file_path: str):
        """验证文件"""
        path = Path(file_path)
        
        if not path.exists():
            raise PDFParseError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise PDFParseError(f"Not a file: {file_path}")
        
        # 检查扩展名
        if path.suffix.lower() not in ['.pdf']:
            raise UnsupportedFormatError(file_path, path.suffix)
        
        # 检查文件大小
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config['max_file_size_mb']:
            raise PDFTooLargeError(file_path, file_size_mb, self.config['max_file_size_mb'])
    
    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """提取 PDF 元数据"""
        metadata = {}
        
        try:
            meta = doc.metadata
            if meta:
                metadata = {
                    'title': meta.get('title', ''),
                    'author': meta.get('author', ''),
                    'subject': meta.get('subject', ''),
                    'keywords': meta.get('keywords', ''),
                    'creator': meta.get('creator', ''),
                    'producer': meta.get('producer', ''),
                    'creation_date': meta.get('creationDate', ''),
                    'mod_date': meta.get('modDate', ''),
                }
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _parse_page(
        self,
        doc: fitz.Document,
        page_num: int,
        extract_tables: bool,
        extract_bbox: bool,
    ) -> PDFPage:
        """解析单个页面"""
        page = doc[page_num]
        
        # 页面尺寸
        width = page.rect.width
        height = page.rect.height
        
        # 提取文本块
        text_blocks = self._extract_text_blocks(page, page_num, extract_bbox)
        
        # 提取表格
        tables = []
        if extract_tables:
            tables = self._extract_tables(page, page_num)
        
        return PDFPage(
            page_num=page_num,
            width=width,
            height=height,
            text_blocks=text_blocks,
            tables=tables,
        )
    
    def _extract_text_blocks(
        self,
        page: fitz.Page,
        page_num: int,
        extract_bbox: bool,
    ) -> List[TextBlock]:
        """提取文本块"""
        blocks = []
        granularity = self.config['bbox_granularity']
        
        if granularity == 'block':
            # 按块提取
            block_list = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            
            for idx, block in enumerate(block_list):
                if block["type"] != 0:  # 跳过图片块
                    continue
                
                # 合并行文本
                text_parts = []
                for line in block.get("lines", []):
                    line_text = ""
                    for span in line.get("spans", []):
                        line_text += span.get("text", "")
                    text_parts.append(line_text)
                
                text = "\n".join(text_parts).strip()
                if not text:
                    continue
                
                bbox = BoundingBox(
                    page_num=page_num,
                    x0=block["bbox"][0],
                    y0=block["bbox"][1],
                    x1=block["bbox"][2],
                    y1=block["bbox"][3],
                ) if extract_bbox else BoundingBox(page_num, 0, 0, 0, 0)
                
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_b{idx}",
                    text=text,
                    bbox=bbox,
                ))
        
        elif granularity == 'line':
            # 按行提取
            block_list = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
            line_idx = 0
            
            for block in block_list:
                if block["type"] != 0:
                    continue
                
                for line in block.get("lines", []):
                    text = ""
                    font_size = 0
                    font_name = ""
                    
                    for span in line.get("spans", []):
                        text += span.get("text", "")
                        if not font_size:
                            font_size = span.get("size", 0)
                            font_name = span.get("font", "")
                    
                    text = text.strip()
                    if not text:
                        continue
                    
                    bbox = BoundingBox(
                        page_num=page_num,
                        x0=line["bbox"][0],
                        y0=line["bbox"][1],
                        x1=line["bbox"][2],
                        y1=line["bbox"][3],
                    ) if extract_bbox else BoundingBox(page_num, 0, 0, 0, 0)
                    
                    blocks.append(TextBlock(
                        block_id=f"p{page_num}_l{line_idx}",
                        text=text,
                        bbox=bbox,
                        font_size=font_size,
                        font_name=font_name,
                    ))
                    line_idx += 1
        
        else:
            # 简单文本提取
            text = page.get_text("text").strip()
            if text:
                bbox = BoundingBox.from_fitz_rect(page.rect, page_num) if extract_bbox else BoundingBox(page_num, 0, 0, 0, 0)
                blocks.append(TextBlock(
                    block_id=f"p{page_num}_full",
                    text=text,
                    bbox=bbox,
                ))
        
        return blocks
    
    def _extract_tables(self, page: fitz.Page, page_num: int) -> List[TableBlock]:
        """
        提取表格
        
        注意: PyMuPDF 的表格检测能力有限，这里使用简单的启发式方法
        完整的表格检测建议使用 Marker 或 Camelot
        """
        tables = []
        
        try:
            # 尝试使用 PyMuPDF 内置的表格检测
            found_tables = page.find_tables()
            
            for idx, table in enumerate(found_tables):
                # 获取表格数据
                table_data = table.extract()
                
                if not table_data or not table_data[0]:
                    continue
                
                headers = table_data[0] if table_data else []
                rows = table_data[1:] if len(table_data) > 1 else []
                
                # 生成 Markdown
                markdown = self._table_to_markdown(headers, rows)
                
                # 生成 HTML
                html = self._table_to_html(headers, rows)
                
                # BBox
                bbox = BoundingBox(
                    page_num=page_num,
                    x0=table.bbox[0],
                    y0=table.bbox[1],
                    x1=table.bbox[2],
                    y1=table.bbox[3],
                )
                
                tables.append(TableBlock(
                    block_id=f"p{page_num}_t{idx}",
                    bbox=bbox,
                    raw_markdown=markdown,
                    raw_html=html,
                    headers=headers,
                    rows=rows,
                    row_count=len(rows),
                    col_count=len(headers) if headers else 0,
                ))
        
        except Exception as e:
            logger.debug(f"Table extraction failed on page {page_num}: {e}")
        
        return tables
    
    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """将表格转换为 Markdown"""
        if not headers:
            return ""
        
        lines = []
        
        # 表头
        header_line = "| " + " | ".join(str(h or "") for h in headers) + " |"
        lines.append(header_line)
        
        # 分隔线
        separator = "|" + "|".join("---" for _ in headers) + "|"
        lines.append(separator)
        
        # 数据行
        for row in rows:
            # 确保行长度与表头一致
            padded_row = list(row) + [""] * (len(headers) - len(row))
            row_line = "| " + " | ".join(str(cell or "") for cell in padded_row[:len(headers)]) + " |"
            lines.append(row_line)
        
        return "\n".join(lines)
    
    def _table_to_html(self, headers: List[str], rows: List[List[str]]) -> str:
        """将表格转换为 HTML"""
        if not headers:
            return ""
        
        html_parts = ["<table>"]
        
        # 表头
        html_parts.append("<thead><tr>")
        for h in headers:
            html_parts.append(f"<th>{h or ''}</th>")
        html_parts.append("</tr></thead>")
        
        # 数据行
        html_parts.append("<tbody>")
        for row in rows:
            html_parts.append("<tr>")
            padded_row = list(row) + [""] * (len(headers) - len(row))
            for cell in padded_row[:len(headers)]:
                html_parts.append(f"<td>{cell or ''}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody>")
        
        html_parts.append("</table>")
        
        return "".join(html_parts)


# =============================================================================
# 便捷函数
# =============================================================================

def parse_pdf(
    file_path: str,
    password: Optional[str] = None,
    extract_tables: bool = True,
    extract_bbox: bool = True,
) -> PDFDocument:
    """
    解析 PDF 文件 (便捷函数)
    
    Args:
        file_path: PDF 文件路径
        password: 密码 (如果加密)
        extract_tables: 是否提取表格
        extract_bbox: 是否提取 BBox
    
    Returns:
        PDFDocument 对象
    
    Example:
        >>> doc = parse_pdf("contract.pdf")
        >>> print(f"Pages: {doc.page_count}")
        >>> print(f"Full text: {doc.full_text[:500]}")
        >>> chunks = doc.to_chunks()
    """
    parser = PDFParser()
    return parser.parse(
        file_path,
        password=password,
        extract_tables=extract_tables,
        extract_bbox=extract_bbox,
    )


def parse_pdf_to_chunks(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    解析 PDF 并直接返回 chunks
    
    Args:
        file_path: PDF 文件路径
        chunk_size: chunk 大小
        chunk_overlap: 重叠大小
        **kwargs: 传递给 parse_pdf 的其他参数
    
    Returns:
        chunk 列表
    """
    doc = parse_pdf(file_path, **kwargs)
    return doc.to_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# =============================================================================
# 批量处理函数
# =============================================================================

@dataclass
class PDFParseResult:
    """单个 PDF 解析结果"""
    file_path: str
    file_name: str
    success: bool
    page_count: int = 0
    text_block_count: int = 0
    table_count: int = 0
    char_count: int = 0
    chunk_count: int = 0
    parse_time_ms: float = 0.0
    error_message: str = ""
    avg_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_pdf_batch(
    pdf_paths: List[str],
    output_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    save_chunks: bool = True,
    save_full_text: bool = True,
    save_json: bool = True,
) -> List[PDFParseResult]:
    """
    批量解析 PDF 文件
    
    Args:
        pdf_paths: PDF 文件路径列表
        output_dir: 输出目录
        chunk_size: chunk 大小
        chunk_overlap: 重叠大小
        save_chunks: 是否保存 chunks
        save_full_text: 是否保存全文
        save_json: 是否保存 JSON 格式
    
    Returns:
        解析结果列表
    """
    import time
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    if save_chunks:
        (output_path / "chunks").mkdir(exist_ok=True)
    if save_full_text:
        (output_path / "text").mkdir(exist_ok=True)
    if save_json:
        (output_path / "json").mkdir(exist_ok=True)
    
    parser = PDFParser()
    results = []
    all_chunks = []
    
    for idx, pdf_path in enumerate(pdf_paths):
        file_name = Path(pdf_path).name
        logger.info(f"[{idx+1}/{len(pdf_paths)}] Parsing: {file_name}")
        
        start_time = time.perf_counter()
        
        try:
            # 解析 PDF
            doc = parser.parse(pdf_path)
            chunks = doc.to_chunks(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            parse_time = (time.perf_counter() - start_time) * 1000
            
            # 计算平均置信度
            confidences = [b.confidence for b in doc.all_text_blocks if b.confidence < 1.0]
            avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
            
            result = PDFParseResult(
                file_path=pdf_path,
                file_name=file_name,
                success=True,
                page_count=doc.page_count,
                text_block_count=len(doc.all_text_blocks),
                table_count=len(doc.all_tables),
                char_count=len(doc.full_text),
                chunk_count=len(chunks),
                parse_time_ms=parse_time,
                avg_confidence=avg_conf,
            )
            
            # 保存结果
            base_name = Path(file_name).stem
            
            if save_full_text:
                text_path = output_path / "text" / f"{base_name}.txt"
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(doc.full_text)
            
            if save_json:
                json_path = output_path / "json" / f"{base_name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
            
            if save_chunks:
                all_chunks.extend(chunks)
            
            logger.success(f"  Done: {doc.page_count} pages, {len(chunks)} chunks, {parse_time:.1f}ms")
            
        except Exception as e:
            parse_time = (time.perf_counter() - start_time) * 1000
            result = PDFParseResult(
                file_path=pdf_path,
                file_name=file_name,
                success=False,
                parse_time_ms=parse_time,
                error_message=str(e),
            )
            logger.error(f"  Failed: {e}")
        
        results.append(result)
    
    # 保存所有 chunks 到 CSV（列顺序与 cuad_v1_chunks.csv 一致）
    if save_chunks and all_chunks:
        import pandas as pd
        chunks_df = pd.DataFrame(all_chunks)
        
        # 确保列顺序与原始数据一致
        column_order = [
            'file_name', 'chunk_id', 'clause_type', 'clause_text', 
            'parent_clause_text', 'contract_idx', 'chunk_idx', 
            'start_char', 'end_char', 'has_answer', 'source'
        ]
        # 只保留存在的列
        existing_cols = [c for c in column_order if c in chunks_df.columns]
        chunks_df = chunks_df[existing_cols]
        
        chunks_path = output_path / "all_chunks.csv"
        chunks_df.to_csv(chunks_path, index=False)
        logger.info(f"Saved {len(all_chunks)} chunks to {chunks_path}")
    
    return results


def find_pdf_files(
    directory: str,
    recursive: bool = True,
    max_files: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> List[str]:
    """
    查找目录下的 PDF 文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归搜索
        max_files: 最大文件数
        shuffle: 是否随机打乱
        seed: 随机种子
    
    Returns:
        PDF 文件路径列表
    """
    import random
    from pathlib import Path
    
    path = Path(directory)
    
    if recursive:
        pdf_files = list(path.rglob("*.pdf")) + list(path.rglob("*.PDF"))
    else:
        pdf_files = list(path.glob("*.pdf")) + list(path.glob("*.PDF"))
    
    pdf_paths = [str(f) for f in pdf_files]
    
    if shuffle:
        random.seed(seed)
        random.shuffle(pdf_paths)
    
    if max_files and len(pdf_paths) > max_files:
        pdf_paths = pdf_paths[:max_files]
    
    return pdf_paths
