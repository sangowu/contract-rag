"""
CUAD Embedding 模块

功能:
- 加载 Embedding 模型
- 管理 ChromaDB 向量库
- BM25 索引构建
- Parent-Child 分块策略
"""

import os
import re
import pickle
import unicodedata
from pathlib import Path
from typing import List, Dict, Optional

import chromadb
import pandas as pd
from loguru import logger
from rank_bm25 import BM25L
from sentence_transformers import SentenceTransformer

# 配置导入
from src.core.config import get_config


# =============================================================================
# 全局缓存
# =============================================================================

_model: Optional[SentenceTransformer] = None
_collection = None
_chroma_client = None


# =============================================================================
# 路径获取
# =============================================================================

def get_embedding_paths():
    """
    从配置获取 embedding 相关路径
    
    Returns:
        dict: 包含所有路径的字典
    """
    config = get_config()
    
    chroma_path = Path(config.retrieval.vector_db.persist_directory)
    bm25_path = Path(config.retrieval.bm25.index_path)
    
    # 确保目录存在
    chroma_path.mkdir(parents=True, exist_ok=True)
    bm25_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'chunk_path': str(config.data.chunks_path),
        'gold_answers_path': str(config.data.gold_answers_path),
        'embedding_model': config.models.embedding.path,
        'chroma_db_path': str(chroma_path),
        'bm25_index_path': str(bm25_path),
        'collection_name': config.retrieval.vector_db.collection_name,
        'batch_size': config.models.embedding.batch_size,
    }


# =============================================================================
# 模型和数据库管理
# =============================================================================

def get_chroma_client():
    """获取 ChromaDB 客户端 (单例)"""
    global _chroma_client
    if _chroma_client is None:
        paths = get_embedding_paths()
        _chroma_client = chromadb.PersistentClient(path=paths['chroma_db_path'])
        logger.info(f"ChromaDB client initialized at {paths['chroma_db_path']}")
    return _chroma_client


def get_model() -> SentenceTransformer:
    """
    获取 embedding 模型，根据 GPU 配置自动选择设备
    
    策略:
    - 双 GPU: 使用 GPU1 (与 LLM 分开)
    - 单 GPU: 使用 GPU0 (与 LLM 共享)
    - 无 GPU: 使用 CPU
    
    Returns:
        SentenceTransformer 模型对象
    """
    global _model
    if _model is None:
        import torch
        
        paths = get_embedding_paths()
        model_path = paths['embedding_model']
        
        # 根据 GPU 数量选择设备
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                device = "cuda:1"  # 双 GPU: 使用 GPU1
                logger.info(f"Loading embedding model on GPU1: {model_path}")
            else:
                device = "cuda:0"  # 单 GPU: 使用 GPU0
                logger.info(f"Loading embedding model on GPU0: {model_path}")
        else:
            device = "cpu"
            logger.warning(f"No GPU available, using CPU for embedding model: {model_path}")
        
        _model = SentenceTransformer(model_path, device=device)
    return _model


def get_collection():
    """获取 ChromaDB collection (单例)"""
    global _collection
    if _collection is None:
        paths = get_embedding_paths()
        client = get_chroma_client()
        _collection = client.get_or_create_collection(name=paths['collection_name'])
        logger.info(f"ChromaDB collection '{paths['collection_name']}' loaded")
    return _collection


# =============================================================================
# 文本处理
# =============================================================================

def normalize_text(s: str) -> str:
    """标准化文本"""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()


# =============================================================================
# BM25 索引
# =============================================================================

def build_bm25_index(chunk_texts: List[str]):
    """
    构建 BM25 索引
    
    Args:
        chunk_texts: 文本列表
    
    Returns:
        (BM25 索引, tokenized corpus)
    """
    paths = get_embedding_paths()
    
    tokenized_corpus = []
    for text in chunk_texts:
        normalized = normalize_text(text)
        tokens = re.findall(r'\b\w+\b', normalized)
        tokenized_corpus.append(tokens)

    bm25 = BM25L(tokenized_corpus)
    
    bm25_path = Path(paths['bm25_index_path'])
    bm25_path.mkdir(parents=True, exist_ok=True)
    index_file = bm25_path / "bm25_index.pkl"
    
    with open(index_file, 'wb') as f:
        pickle.dump(bm25, f)
    
    logger.success(f"Built BM25 index with {len(tokenized_corpus)} documents at {index_file}")
    return bm25, tokenized_corpus


# =============================================================================
# Parent-Child 分块
# =============================================================================

def parent_child_chunking(
    df_chunk: pd.DataFrame,
    child_chunk_size: Optional[int] = None,
    overlap: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Parent-Child 分块策略
    
    Args:
        df_chunk: chunk DataFrame
        child_chunk_size: 子块大小 (默认从配置读取)
        overlap: 重叠大小 (默认从配置读取)
    
    Returns:
        包含 parent 和 child 信息的字典
    """
    config = get_config()
    
    if child_chunk_size is None:
        child_chunk_size = config.retrieval.parent_child.child_chunk_size
    if overlap is None:
        overlap = config.retrieval.parent_child.overlap

    child_ids = []
    child_texts = []
    child_to_parent = {}
    parent_ids = []
    parent_texts = []

    for _, row in df_chunk.iterrows():
        file_name = str(row["file_name"])
        clause_type = str(row["clause_type"])
        contract_idx = int(row["contract_idx"])
        parent_id = str(row["chunk_id"])
        parent_text = row.get("parent_clause_text", row["clause_text"])
        if not isinstance(parent_text, str):
            parent_text = str(parent_text)
        parent_text = normalize_text(parent_text)

        parent_ids.append(parent_id)
        parent_texts.append(parent_text)

        text_len = len(parent_text)
        start = 0
        child_idx = 0

        while start < text_len:
            end = min(start + child_chunk_size, text_len)
            child_text = parent_text[start:end].strip()
            if not child_text:
                break
            child_id = f"{file_name}::{clause_type}::{contract_idx}::{child_idx}::{start}:{end}"
            child_ids.append(child_id)
            child_texts.append(child_text)
            child_to_parent[child_id] = parent_id
            child_idx += 1
            if end == text_len:
                break
            start = max(0, end - overlap)

    logger.info(f"Created {len(parent_ids)} parents and {len(child_ids)} child chunks")
    return {
        "parent_ids": parent_ids,
        "parent_texts": parent_texts,
        "child_ids": child_ids,
        "child_texts": child_texts,
        "child_to_parent": child_to_parent,
    }


# =============================================================================
# 主初始化函数
# =============================================================================

def initialize_embeddings():
    """
    初始化 embeddings：构建 BM25 索引并上传到 ChromaDB
    
    注意: 此函数直接使用 data_loader 生成的 chunks 数据，
    不再进行二次 Parent-Child 切分。chunks 数据已经包含：
    - chunk_id: 唯一标识符
    - clause_text: child chunk 文本 (用于检索)
    - parent_clause_text: parent 完整文本 (用于生成)
    """
    paths = get_embedding_paths()
    config = get_config()
    
    logger.info(f"Loading chunks from {paths['chunk_path']}...")
    df_chunk = pd.read_csv(paths['chunk_path'])
    
    # 检查 Parent-Child 结构
    has_parent_child = 'parent_clause_text' in df_chunk.columns
    if has_parent_child:
        diff_count = (df_chunk['clause_text'] != df_chunk['parent_clause_text']).sum()
        diff_ratio = diff_count / len(df_chunk) * 100
        logger.info(f"Parent-Child structure: {diff_count}/{len(df_chunk)} ({diff_ratio:.1f}%) have different parent")
    else:
        logger.warning("parent_clause_text column not found, using clause_text as parent")
    
    model = get_model()
    collection = get_collection()

    # 直接使用已处理好的 chunks 数据（不再二次切分）
    chunk_ids = df_chunk['chunk_id'].astype(str).tolist()
    chunk_texts = df_chunk['clause_text'].astype(str).apply(normalize_text).tolist()
    
    # 获取 parent 文本（用于 BM25 和 metadata）
    if has_parent_child:
        parent_texts = df_chunk['parent_clause_text'].astype(str).apply(normalize_text).tolist()
    else:
        parent_texts = chunk_texts.copy()
    
    # 构建 metadata
    chunk_meta = []
    for _, row in df_chunk.iterrows():
        parent_text = normalize_text(str(row.get('parent_clause_text', row['clause_text'])))
        chunk_meta.append({
            "file_name": str(row["file_name"]),
            "clause_type": str(row["clause_type"]),
            "parent_text": parent_text,
        })

    # 构建 BM25 索引
    # 使用 parent_texts 建立索引（更完整的文本，更好的关键词匹配）
    # 注意：parent_texts 与 chunk_df 一一对应，这样 BM25 检索时可以通过索引位置找到对应的 chunk
    # 多个 child 可能共享同一个 parent 文本，但这没关系，BM25 会为每个位置打分
    logger.info(f"Building BM25 index with {len(chunk_texts)} documents (using parent_clause_text)")
    build_bm25_index(chunk_texts)  # 使用 parent 文本，更好的关键词覆盖

    # 计算 embeddings（使用 child 文本）
    batch_size = paths['batch_size']
    logger.info(f"Computing embeddings for {len(chunk_texts)} chunks with batch_size={batch_size}...")
    
    embeddings = model.encode(
        chunk_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # 上传到 ChromaDB
    upsert_batch_size = 4096
    total_items = len(chunk_ids)
    logger.info(f"Upserting {total_items} chunk embeddings in batches of {upsert_batch_size}")

    for start in range(0, total_items, upsert_batch_size):
        end = min(start + upsert_batch_size, total_items)
        batch_ids = chunk_ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        metadatas = []
        for offset in range(len(batch_ids)):
            idx = start + offset
            meta = chunk_meta[idx]
            
            metadatas.append({
                "chunk_id": batch_ids[offset],
                "file_name": meta["file_name"],
                "clause_type": meta["clause_type"],
                "clause_text": chunk_texts[idx],
                "parent_text": meta["parent_text"],
            })

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted items {start}..{end-1}")

    logger.success(f"Upserted {total_items} chunk embeddings to ChromaDB")


if __name__ == "__main__":
    from src.core.config import parse_args_and_init
    
    # 解析命令行参数并初始化配置
    parse_args_and_init()
    
    # 初始化 embeddings
    initialize_embeddings()
