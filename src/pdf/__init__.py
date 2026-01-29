"""
CUAD PDF 解析模块

核心功能:
- PDF 文本提取 (PyMuPDF)
- BBox 坐标提取 (用于前端高亮)
- 表格检测与处理
- OCR 支持 (扫描件)

使用方式:
    from src.pdf import parse_pdf, PDFParser, PDFDocument
    
    # 快捷方式
    doc = parse_pdf("contract.pdf")
    chunks = doc.to_chunks()
    
    # 完整控制
    parser = PDFParser()
    doc = parser.parse("contract.pdf", extract_tables=True)
"""

from src.pdf.parser import (
    # 主类
    PDFParser,
    PDFDocument,
    PDFPage,
    # 数据结构
    BoundingBox,
    TextBlock,
    TableBlock,
    PDFParseResult,
    # 便捷函数
    parse_pdf,
    parse_pdf_to_chunks,
    # 批量处理
    parse_pdf_batch,
    find_pdf_files,
)

from src.pdf.exceptions import (
    PDFParseError,
    PDFCorruptedError,
    PDFEncryptedError,
    PDFTooLargeError,
    PDFPageLimitError,
    OCRFailedError,
    TableExtractionError,
    UnsupportedFormatError,
)

from src.pdf.ocr_engine import (
    OCREngine,
    OCRResult,
    OCRLine,
    ocr_image,
    pdf_page_to_image,
)

__all__ = [
    # 主类
    'PDFParser',
    'PDFDocument',
    'PDFPage',
    # 数据结构
    'BoundingBox',
    'TextBlock',
    'TableBlock',
    'PDFParseResult',
    # 便捷函数
    'parse_pdf',
    'parse_pdf_to_chunks',
    # 批量处理
    'parse_pdf_batch',
    'find_pdf_files',
    # 异常
    'PDFParseError',
    'PDFCorruptedError',
    'PDFEncryptedError',
    'PDFTooLargeError',
    'PDFPageLimitError',
    'OCRFailedError',
    'TableExtractionError',
    'UnsupportedFormatError',
    # OCR
    'OCREngine',
    'OCRResult',
    'OCRLine',
    'ocr_image',
    'pdf_page_to_image',
]
