import os
import re
import pickle
import unicodedata
from pathlib import Path
from typing import List, Dict

import chromadb
import pandas as pd
from loguru import logger
from rank_bm25 import BM25L
from sentence_transformers import SentenceTransformer

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
EMBEDDING_MODEL = "/root/autodl-tmp/model/sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "/root/autodl-tmp/data/indexes/embeddings/chroma_db"
BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25"

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
_model = None
_collection = None

def get_model():
    """
    获取embedding模型，固定在GPU1
    
    Returns:
        SentenceTransformer模型对象
    """
    global _model
    if _model is None:
        import torch
        # 检查GPU1是否可用
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device = f"cuda:1"  # 使用GPU1
            logger.info(f"Loading embedding model on GPU1")
        else:
            device = "cpu"
            logger.warning("GPU1 not available, using CPU for embedding model")
        _model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    return _model  

def get_collection():
    global _collection
    if _collection is None:
        _collection = chroma_client.get_or_create_collection(name="cuad_chunks")
    return _collection

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def build_bm25_index(chunk_texts: List[str]):
    tokenized_corpus = []
    for text in chunk_texts:
        normalized = normalize_text(text)
        tokens = re.findall(r'\b\w+\b', normalized)
        tokenized_corpus.append(tokens)

    bm25 = BM25L(tokenized_corpus) 
    Path(BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    index_file = Path(BM25_INDEX_PATH) / "bm25_index.pkl"
    with open(index_file, 'wb') as f:
        pickle.dump(bm25, f)
    
    logger.success(f"Built BM25 index with {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus

def parent_child_chunking(
    df_chunk: pd.DataFrame,
    child_chunk_size: int = 500,
    overlap: int = 100
) -> Dict[str, List[str]]:

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

def initialize_embeddings():
    df_chunk = pd.read_csv(CHUNK_PATH)
    model = get_model()
    collection = get_collection()

    chunking = parent_child_chunking(
        df_chunk,
        child_chunk_size=500,
        overlap=100,
    )

    parent_texts = chunking["parent_texts"]
    child_ids = chunking["child_ids"]
    child_texts = chunking["child_texts"]
    child_to_parent = chunking["child_to_parent"]

    parent_meta = {
        row["chunk_id"]: {
            "file_name": row["file_name"],
            "clause_type": row["clause_type"],
            "parent_text": normalize_text(
                row.get("parent_clause_text", row["clause_text"])
            ),
        }
        for _, row in df_chunk.iterrows()
    }

    build_bm25_index(parent_texts)

    embeddings = model.encode(
        child_texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    batch_size = 4096
    total_items = len(child_ids)
    logger.info(f"Upserting {total_items} child embeddings in batches of {batch_size}")

    for start in range(0, total_items, batch_size):
        end = min(start + batch_size, total_items)
        batch_ids = child_ids[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        metadatas = []
        for offset, cid in enumerate(batch_ids):
            parent_id = child_to_parent[cid]
            pm = parent_meta[parent_id]

            metadatas.append({
                "chunk_id": cid,                        
                "parent_id": parent_id,                    
                "file_name": pm["file_name"],
                "clause_type": pm["clause_type"],
                "clause_text": child_texts[start + offset], 
                "parent_text": pm["parent_text"],          
            })

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Upserted child items {start}..{end-1}")

    logger.success(f"Upserted {total_items} child embeddings to ChromaDB")


if __name__ == "__main__":
    initialize_embeddings()