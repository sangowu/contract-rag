"""
PDF 解析异常定义

异常层级:
    PDFParseError (基类)
    ├── PDFCorruptedError     - 文件损坏
    ├── PDFEncryptedError     - 文件加密
    ├── PDFTooLargeError      - 文件过大
    ├── PDFPageLimitError     - 页数超限
    ├── OCRFailedError        - OCR 失败
    └── TableExtractionError  - 表格提取失败
"""

from typing import Optional


class PDFParseError(Exception):
    """PDF 解析基础异常"""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        self.message = message
        self.file_path = file_path
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.file_path:
            return f"{self.message} (file: {self.file_path})"
        return self.message


class PDFCorruptedError(PDFParseError):
    """PDF 文件损坏"""
    
    def __init__(self, file_path: str, detail: Optional[str] = None):
        message = "PDF file is corrupted or invalid"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message, file_path)


class PDFEncryptedError(PDFParseError):
    """PDF 文件加密未解锁"""
    
    def __init__(self, file_path: str):
        super().__init__("PDF file is encrypted and password is required", file_path)


class PDFTooLargeError(PDFParseError):
    """PDF 文件过大"""
    
    def __init__(self, file_path: str, file_size_mb: float, max_size_mb: float):
        self.file_size_mb = file_size_mb
        self.max_size_mb = max_size_mb
        message = f"PDF file too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
        super().__init__(message, file_path)


class PDFPageLimitError(PDFParseError):
    """PDF 页数超限"""
    
    def __init__(self, file_path: str, page_count: int, max_pages: int):
        self.page_count = page_count
        self.max_pages = max_pages
        message = f"PDF has too many pages: {page_count} (max: {max_pages})"
        super().__init__(message, file_path)


class OCRFailedError(PDFParseError):
    """OCR 识别失败"""
    
    def __init__(self, file_path: str, page_num: Optional[int] = None, detail: Optional[str] = None):
        self.page_num = page_num
        message = "OCR recognition failed"
        if page_num is not None:
            message = f"{message} on page {page_num}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message, file_path)


class TableExtractionError(PDFParseError):
    """表格提取失败"""
    
    def __init__(self, file_path: str, page_num: Optional[int] = None, detail: Optional[str] = None):
        self.page_num = page_num
        message = "Table extraction failed"
        if page_num is not None:
            message = f"{message} on page {page_num}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message, file_path)


class UnsupportedFormatError(PDFParseError):
    """不支持的文件格式"""
    
    def __init__(self, file_path: str, detected_format: Optional[str] = None):
        message = "Unsupported file format"
        if detected_format:
            message = f"{message}: {detected_format}"
        super().__init__(message, file_path)
