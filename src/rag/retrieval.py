from typing import List, Dict
from loguru import logger
from pathlib import Path
import re, pandas as pd, pickle
from .embedding import get_model, get_collection, normalize_text

BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25/bm25_index.pkl"
CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
_chunk_df = None

def get_chunk_df():
    global _chunk_df
    if _chunk_df is None:
        _chunk_df = pd.read_csv(CHUNK_PATH)
    return _chunk_df

def load_bm25_index():
    if Path(BM25_INDEX_PATH).exists():
        with open(BM25_INDEX_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")

def bm25_search(query: str, k: int = 10, file_name: str | None = None):
    try:
        chunk_df = get_chunk_df()
        bm25 = load_bm25_index()
        normalized_query = normalize_text(query)
        query_tokens = re.findall(r'\b\w+\b', normalized_query)
        scores = bm25.get_scores(query_tokens)

        all_indices = scores.argsort()[::-1]
        
        if file_name:
            file_mask = chunk_df['file_name'] == file_name
            file_indices = chunk_df[file_mask].index.tolist()
            
            file_scores = scores[file_indices]
            file_sorted_indices = file_scores.argsort()[::-1][:k]
            top_k_indices = [file_indices[i] for i in file_sorted_indices]
        else:
            top_k_indices = all_indices[:k]

        return [
            {
                "chunk_id": chunk_df.loc[idx, 'chunk_id'],
                "clause_text": chunk_df.loc[idx, 'clause_text'],
                "file_name": chunk_df.loc[idx, 'file_name'],
                "clause_type": chunk_df.loc[idx, 'clause_type'],
            }
            for idx in top_k_indices
        ]
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

def retrieve_top_k_hybrid(
    query: str,
    k: int = 10,
    file_name: str | None = None,
    top_k_retrieval: int = 100,
    rrf_k: int = 20, 
) -> List[Dict[str, str]]:
    try:
        bm25_results = bm25_search(query, k=top_k_retrieval, file_name=file_name)
        bm25_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(bm25_results)}
        retrieval_results = retrieve_top_k(query, k=k, file_name=file_name, top_k_retrieval=top_k_retrieval)
        retrieval_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(retrieval_results)}

        rrf_scores = {}
        all_chunk_ids = set(bm25_ranks.keys()) | set(retrieval_ranks.keys())
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 0.3 / (rrf_k + bm25_ranks[chunk_id])
            if chunk_id in retrieval_ranks:
                score += 0.7 / (rrf_k + retrieval_ranks[chunk_id])
            rrf_scores[chunk_id] = score
        
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        chunk_df = get_chunk_df()
        results = []
        for chunk_id, _ in sorted_chunks:
            row = chunk_df[chunk_df['chunk_id'] == chunk_id]
            if not row.empty:
                results.append({
                    "chunk_id": chunk_id,
                    "clause_text": row['clause_text'].iloc[0],
                    "file_name": row['file_name'].iloc[0],
                    "clause_type": row['clause_type'].iloc[0],
                })

        return results
    except Exception as e:
        logger.error(f"Error during hybrid retrieval: {e}")
        return []