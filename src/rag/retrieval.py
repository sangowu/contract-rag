from typing import List, Dict
from loguru import logger
from rank_bm25 import BM25L
from pathlib import Path
import re, os, sys, ast, unicodedata
from .embedding import get_model, get_collection, normalize_text

BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25"
CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"

def load_bm25_index():
    if Path(BM25_INDEX_PATH).exists():
        return BM25L.load(path=BM25_INDEX_PATH)
    else:
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")

def bm25_search(query: str, k: int = 10, file_name: str | None = None):
    try:
        bm25 = load_bm25_index()
        normalized_query = normalize_text(query)
        query_tokens = re.findall(r'\b\w+\b', normalized_query)
        scores = bm25.get_scores(query_tokens)

        all_indices = scores.argsort()[::-1]
        
        if file_name:
            filterd_results = []
            for idx in all_indices:
                doc_file_name = chunk_df.loc[idx, 'file_name']
        return scores
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}. Query: {query}")
        return []

def retrieve_top_k(
    query: str,
    k: int = 10,
    file_name: str | None = None,
    top_k_retrieval: int = 100,
) -> List[Dict[str, str]]:
    try:
        model = get_model()
        collection = get_collection()
        q_emb = model.encode([query], normalize_embeddings=True)[0]

        where = {"file_name": file_name} if file_name else None
        
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=max(k, top_k_retrieval),
            include=['metadatas'],
            where=where,
        )
        
        ids = results["ids"][0]
        metas = results["metadatas"][0]

        ids = ids[:k]
        metas = metas[:k]

        return [
            {
                "chunk_id": cid,
                "clause_text": m.get("clause_text", ""),
                "file_name": m.get("file_name", ""),
                "clause_type": m.get("clause_type", ""),
            }
            for cid, m in zip(ids, metas)
        ]
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []
