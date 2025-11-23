from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path
import torch
import re, pandas as pd, pickle
from .embedding import get_model, get_collection, normalize_text
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25/bm25_index.pkl"
CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
RERANKER_MODEL = "/root/autodl-tmp/model/Qwen3-Reranker-4B"
_chunk_df = None
_reranker_tokenizer = None
_reranker_model = None

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

def load_reranker():
    global _reranker_tokenizer, _reranker_model
    if _reranker_model is None:
        if not Path(RERANKER_MODEL).exists():
            raise FileNotFoundError(f"Reranker model not found at {RERANKER_MODEL}")

        logger.info(f"Loading Qwen3-Reranker from {RERANKER_MODEL}")

        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            RERANKER_MODEL,
            padding_side="left",
            trust_remote_code=True,
        )

        global max_length
        max_length = 1024

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        _reranker_model = AutoModelForCausalLM.from_pretrained(
            RERANKER_MODEL,
            dtype="auto", 
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()

        logger.success("Qwen3-Reranker loaded successfully")
    return _reranker_tokenizer, _reranker_model

def format_instruction(instruction: str | None, query: str, doc: str) -> str:
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output

def bm25_search(query: str, top_k_retrieval: int = 10, file_name: str | None = None):
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
            file_sorted_indices = file_scores.argsort()[::-1][:top_k_retrieval]
            top_k_indices = [file_indices[i] for i in file_sorted_indices]
        else:
            top_k_indices = all_indices[:top_k_retrieval]

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
    top_k_shown: int | None = None,
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
            n_results=top_k_retrieval,
            include=['metadatas'],
            where=where,
        )
        
        ids = results["ids"][0]
        metas = results["metadatas"][0]

        ids = ids[:top_k_shown if top_k_shown else top_k_retrieval]
        metas = metas[:top_k_shown if top_k_shown else top_k_retrieval]

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
    top_k_shown: int = 10,
    file_name: str | None = None,
    top_k_retrieval: int = 100,
    rrf_k: int = 20, 
) -> List[Dict[str, str]]:
    try:
        bm25_results = bm25_search(query, top_k_retrieval=top_k_retrieval, file_name=file_name)
        bm25_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(bm25_results)}
        retrieval_results = retrieve_top_k(query, file_name=file_name, top_k_retrieval=top_k_retrieval)
        retrieval_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(retrieval_results)}

        rrf_scores = {}
        all_chunk_ids = set(bm25_ranks.keys()) | set(retrieval_ranks.keys())
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in bm25_ranks:
                score += 0.4 / (rrf_k + bm25_ranks[chunk_id])
            if chunk_id in retrieval_ranks:
                score += 0.6 / (rrf_k + retrieval_ranks[chunk_id])
            rrf_scores[chunk_id] = score
        
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_shown]
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

def rerank_results(
    query: str,
    candidate_chunks: List[Dict[str, str]],
    top_k: Optional[int] = None,
    batch_size: int = 4,
) -> List[Dict[str, str]]:
    if not candidate_chunks:
        return []
    
    tokenizer, model = load_reranker()

    yes_ids = tokenizer.encode(" yes", add_special_tokens=False)
    no_ids  = tokenizer.encode(" no", add_special_tokens=False)
    if len(yes_ids) != 1 or len(no_ids) != 1:
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids  = tokenizer.encode("no", add_special_tokens=False)
    assert len(yes_ids) == 1 and len(no_ids) == 1
    token_true_id, token_false_id = yes_ids[0], no_ids[0]
    
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    docs = [c.get("clause_text","") for c in candidate_chunks]
    pairs = [format_instruction(task, query, d) for d in docs]
    
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            batch_scores = model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_relevance_scores = batch_scores[:, 1].exp().tolist()
            
            scores.extend(batch_relevance_scores)
    
    scored_candidates = [(score, cand) for score, cand in zip(scores, candidate_chunks)]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    
    if top_k:
        results = [cand for _, cand in scored_candidates[:top_k]]
    else:
        results = [cand for _, cand in scored_candidates]
    
    for i, (score, _) in enumerate(scored_candidates[:top_k] if top_k else scored_candidates):
        results[i]['rerank_score'] = score
    
    logger.info(f"Reranked {len(results)} candidates, top score: {max(scores) if scores else 0}")
    return results