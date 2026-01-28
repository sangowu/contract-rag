"""
PDF 解析器测试
"""

import os
import pytest
from pathlib import Path

# 配置初始化
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.core.config import init_config

# 初始化配置
init_config(mode='test')

from src.pdf import (
    parse_pdf,
    PDFParser,
    PDFDocument,
    BoundingBox,
    TextBlock,
    PDFParseError,
    PDFCorruptedError,
)


# 测试数据目录
TEST_PDF_DIR = "/root/autodl-tmp/data/raw/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements"


def get_test_pdf() -> str:
    """获取测试 PDF 文件路径"""
    if not os.path.exists(TEST_PDF_DIR):
        pytest.skip(f"Test PDF directory not found: {TEST_PDF_DIR}")
    
    pdf_files = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        pytest.skip("No PDF files found in test directory")
    
    return os.path.join(TEST_PDF_DIR, pdf_files[0])


class TestBoundingBox:
    """BoundingBox 测试"""
    
    def test_create_bbox(self):
        bbox = BoundingBox(page_num=0, x0=10.0, y0=20.0, x1=100.0, y1=50.0)
        assert bbox.page_num == 0
        assert bbox.x0 == 10.0
        assert bbox.y1 == 50.0
    
    def test_bbox_to_dict(self):
        bbox = BoundingBox(page_num=1, x0=10.0, y0=20.0, x1=100.0, y1=50.0)
        d = bbox.to_dict()
        assert d['page_num'] == 1
        assert d['x0'] == 10.0
    
    def test_bbox_from_dict(self):
        d = {'page_num': 2, 'x0': 10.0, 'y0': 20.0, 'x1': 100.0, 'y1': 50.0}
        bbox = BoundingBox.from_dict(d)
        assert bbox.page_num == 2


class TestPDFParser:
    """PDFParser 测试"""
    
    def test_parse_pdf(self):
        """测试基本 PDF 解析"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        
        assert isinstance(doc, PDFDocument)
        assert doc.page_count > 0
        assert len(doc.pages) == doc.page_count
        assert doc.file_name.endswith('.pdf')
    
    def test_parse_pdf_text_blocks(self):
        """测试文本块提取"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        
        blocks = doc.all_text_blocks
        assert len(blocks) > 0
        
        # 检查文本块结构
        block = blocks[0]
        assert isinstance(block, TextBlock)
        assert block.text
        assert isinstance(block.bbox, BoundingBox)
    
    def test_parse_pdf_bbox(self):
        """测试 BBox 提取"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path, extract_bbox=True)
        
        blocks = doc.all_text_blocks
        assert len(blocks) > 0
        
        # 检查 BBox 有效性
        for block in blocks[:10]:
            bbox = block.bbox
            assert bbox.page_num >= 0
            assert bbox.x1 >= bbox.x0
            assert bbox.y1 >= bbox.y0
    
    def test_parse_pdf_to_chunks(self):
        """测试转换为 chunks"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        chunks = doc.to_chunks()
        
        assert len(chunks) > 0
        
        # 检查 chunk 结构
        chunk = chunks[0]
        assert 'file_name' in chunk
        assert 'chunk_id' in chunk
        assert 'clause_text' in chunk
        assert 'bbox_json' in chunk
        assert 'page_num' in chunk
    
    def test_parse_nonexistent_file(self):
        """测试不存在的文件"""
        with pytest.raises(PDFParseError):
            parse_pdf("/nonexistent/path/file.pdf")
    
    def test_full_text(self):
        """测试完整文本提取"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        
        full_text = doc.full_text
        assert len(full_text) > 0
        assert isinstance(full_text, str)


class TestPDFDocument:
    """PDFDocument 测试"""
    
    def test_document_metadata(self):
        """测试文档元数据"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        
        assert isinstance(doc.metadata, dict)
        # 元数据可能为空，但字段应该存在
        assert 'title' in doc.metadata or doc.metadata == {}
    
    def test_document_to_dict(self):
        """测试转换为字典"""
        pdf_path = get_test_pdf()
        doc = parse_pdf(pdf_path)
        
        d = doc.to_dict()
        assert 'file_path' in d
        assert 'file_name' in d
        assert 'page_count' in d
        assert 'pages' in d


if __name__ == "__main__":
    # 简单测试运行
    print("Running PDF parser tests...")
    
    pdf_path = get_test_pdf()
    print(f"Test PDF: {pdf_path}")
    
    # 测试解析
    doc = parse_pdf(pdf_path)
    print(f"Pages: {doc.page_count}")
    print(f"Text blocks: {len(doc.all_text_blocks)}")
    print(f"Tables: {len(doc.all_tables)}")
    
    # 测试 chunks
    chunks = doc.to_chunks()
    print(f"Chunks: {len(chunks)}")
    
    print("\nAll tests passed!")
