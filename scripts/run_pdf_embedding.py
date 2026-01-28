#!/usr/bin/env python
"""
PDF 解析结果 Embedding 脚本

功能:
- 读取 PDF 解析后的 chunks 数据
- 使用 Embedding 模型生成向量
- 保存到 ChromaDB（collection 名称体现 PDF 来源）
- 构建 BM25 索引

使用方式:
    # 使用默认路径
    python scripts/run_pdf_embedding.py --mode test
    
    # 指定输入文件
    python scripts/run_pdf_embedding.py --mode test --input data/pdf_parsed_results/all_chunks.csv
    
    # 指定 collection 名称
    python scripts/run_pdf_embedding.py --mode test --collection contracts_pdf
"""

import os
import sys
import argparse
import time
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import chromadb
from loguru import logger
from tqdm import tqdm
from rank_bm25 import BM25L

from src.core.config import init_config, get_config
from src.utils.seed_utils import set_global_seed, get_seed


# =============================================================================
# 配置
# =============================================================================

DEFAULT_INPUT_PATH = "data/pdf_parsed_results/all_chunks.csv"
DEFAULT_COLLECTION_NAME = "contracts_pdf"
DEFAULT_BM25_INDEX_NAME = "bm25_pdf_index.pkl"


# =============================================================================
# Embedding 函数
# =============================================================================

def load_embedding_model():
    """加载 Embedding 模型"""
    from src.rag.embedding import get_model
    return get_model()


def tokenize_for_bm25(text: str) -> List[str]:
    """BM25 分词"""
    import re
    import unicodedata
    
    # 规范化
    text = unicodedata.normalize("NFKC", text.lower())
    # 分词
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_bm25_index(texts: List[str]) -> BM25L:
    """构建 BM25 索引"""
    tokenized = [tokenize_for_bm25(t) for t in texts]
    return BM25L(tokenized)


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PDF Embedding Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='test',
        choices=['test', 'dev', 'prod'],
        help='Configuration mode (default: test)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default=None,
        help=f'Input CSV file path (default: {DEFAULT_INPUT_PATH})'
    )
    
    parser.add_argument(
        '--collection',
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f'ChromaDB collection name (default: {DEFAULT_COLLECTION_NAME})'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Embedding batch size (default: from config)'
    )
    
    parser.add_argument(
        '--recreate',
        action='store_true',
        help='Recreate collection if exists (delete old data)'
    )
    
    parser.add_argument(
        '--skip-bm25',
        action='store_true',
        help='Skip BM25 index building'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化配置
    logger.info("Initializing configuration...")
    config = init_config(mode=args.mode)
    project_root = config.app.project_root
    
    # 设置全局随机种子
    seed = set_global_seed()
    logger.info(f"Global random seed: {seed}")
    
    # 确定输入路径
    if args.input:
        input_path = args.input if os.path.isabs(args.input) else os.path.join(project_root, args.input)
    else:
        input_path = os.path.join(project_root, DEFAULT_INPUT_PATH)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        logger.info("Please run 'python scripts/run_pdf_parsing.py' first to generate chunks.")
        return
    
    # 加载数据
    logger.info(f"Loading chunks from: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} chunks from {df['file_name'].nunique()} files")
    
    # 过滤空文本
    df = df[df['clause_text'].notna() & (df['clause_text'].str.strip() != '')]
    logger.info(f"After filtering empty texts: {len(df)} chunks")
    
    if len(df) == 0:
        logger.error("No valid chunks found!")
        return
    
    # 获取配置参数
    batch_size = args.batch_size if args.batch_size else config.models.embedding.batch_size
    
    # ChromaDB 路径（使用 PDF 专用目录）
    chroma_base_path = Path(config.retrieval.vector_db.persist_directory).parent
    chroma_pdf_path = chroma_base_path / "chroma_db_pdf"
    chroma_pdf_path.mkdir(parents=True, exist_ok=True)
    
    # BM25 路径
    bm25_base_path = Path(config.retrieval.bm25.index_path)
    bm25_pdf_path = bm25_base_path / DEFAULT_BM25_INDEX_NAME
    bm25_base_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"ChromaDB path: {chroma_pdf_path}")
    logger.info(f"BM25 index path: {bm25_pdf_path}")
    logger.info(f"Collection name: {args.collection}")
    logger.info(f"Batch size: {batch_size}")
    
    # 加载 Embedding 模型
    logger.info("Loading embedding model...")
    model = load_embedding_model()
    logger.success("Embedding model loaded")
    
    # 初始化 ChromaDB
    logger.info("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=str(chroma_pdf_path))
    
    # 检查是否需要重建
    existing_collections = [c.name for c in client.list_collections()]
    if args.collection in existing_collections:
        if args.recreate:
            logger.warning(f"Deleting existing collection: {args.collection}")
            client.delete_collection(args.collection)
        else:
            logger.warning(f"Collection '{args.collection}' already exists. Use --recreate to overwrite.")
            # 获取现有数量
            existing = client.get_collection(args.collection)
            logger.info(f"Existing collection has {existing.count()} documents")
            
            # 询问是否继续（追加模式）
            logger.info("Continuing in append mode...")
    
    collection = client.get_or_create_collection(name=args.collection)
    logger.info(f"Collection '{args.collection}' ready, current count: {collection.count()}")
    
    # 准备数据
    texts = df['clause_text'].tolist()
    chunk_ids = df['chunk_id'].tolist()
    
    # 准备 metadata
    metadata_cols = ['file_name', 'chunk_type', 'page_num', 'chunk_idx', 'source']
    metadatas = []
    for _, row in df.iterrows():
        meta = {}
        for col in metadata_cols:
            if col in row and pd.notna(row[col]):
                val = row[col]
                # ChromaDB metadata 只支持 str, int, float, bool
                if isinstance(val, (int, float, bool)):
                    meta[col] = val
                else:
                    meta[col] = str(val)
        
        # 添加 parent_text 截断版本
        parent_text = row.get('parent_clause_text', '')
        if parent_text and pd.notna(parent_text):
            meta['parent_text_preview'] = str(parent_text)[:500]
        
        metadatas.append(meta)
    
    # 批量 Embedding 和插入
    logger.info("=" * 60)
    logger.info("Starting embedding and indexing...")
    logger.info("=" * 60)
    
    total_batches = (len(texts) + batch_size - 1) // batch_size
    start_time = time.perf_counter()
    
    for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        batch_ids = chunk_ids[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        
        # 生成 embeddings
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        
        # 插入 ChromaDB
        collection.upsert(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metas,
            documents=batch_texts,
        )
    
    embedding_time = time.perf_counter() - start_time
    
    logger.info("=" * 60)
    logger.info(f"Embedding completed in {embedding_time:.1f}s")
    logger.info(f"Collection '{args.collection}' now has {collection.count()} documents")
    logger.info("=" * 60)
    
    # 构建 BM25 索引
    if not args.skip_bm25:
        logger.info("Building BM25 index...")
        bm25_start = time.perf_counter()
        
        bm25_index = build_bm25_index(texts)
        
        # 保存索引和映射
        bm25_data = {
            'index': bm25_index,
            'chunk_ids': chunk_ids,
            'file_names': df['file_name'].tolist(),
        }
        
        with open(bm25_pdf_path, 'wb') as f:
            pickle.dump(bm25_data, f)
        
        bm25_time = time.perf_counter() - bm25_start
        logger.success(f"BM25 index saved to: {bm25_pdf_path} ({bm25_time:.1f}s)")
    
    # 打印摘要
    total_time = time.perf_counter() - start_time
    
    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    print(f"  Input File:       {input_path}")
    print(f"  Total Chunks:     {len(df)}")
    print(f"  Unique Files:     {df['file_name'].nunique()}")
    print(f"  Collection:       {args.collection}")
    print(f"  ChromaDB Path:    {chroma_pdf_path}")
    print(f"  BM25 Index:       {bm25_pdf_path if not args.skip_bm25 else 'Skipped'}")
    print(f"  Embedding Time:   {embedding_time:.1f}s")
    print(f"  Total Time:       {total_time:.1f}s")
    print(f"  Throughput:       {len(df) / embedding_time:.1f} chunks/s")
    print("=" * 60 + "\n")
    
    logger.success("PDF embedding completed!")
    
    return {
        "collection_name": args.collection,
        "chroma_path": str(chroma_pdf_path),
        "bm25_path": str(bm25_pdf_path) if not args.skip_bm25 else None,
        "chunk_count": len(df),
        "file_count": df['file_name'].nunique(),
    }


if __name__ == "__main__":
    main()
