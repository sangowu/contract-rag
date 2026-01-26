from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path
import torch
import re, pandas as pd, pickle
from .embedding import get_model, get_collection, normalize_text
from src.utils.model_loading import load_reranker

BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25/bm25_index.pkl"
CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
RERANKER_MODEL = "/root/autodl-tmp/model/Qwen3-Reranker-4B"
_chunk_df = None
RERANKER_MAX_LENGTH = 256  

# Reranker模型和tokenizer缓存
_reranker_tokenizer = None
_reranker_model = None
_reranker_token_ids = None  # 缓存yes/no token IDs
_reranker_device = None  # 缓存模型设备  

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
        # 优化：使用convert_to_numpy=True避免GPU内存占用
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

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
    top_k_shown: int = 20,
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

def get_reranker_model():
    """
    获取reranker模型和tokenizer
    
    Returns:
        (tokenizer, model, token_ids, device) 元组
    """
    global _reranker_tokenizer, _reranker_model, _reranker_token_ids, _reranker_device
    
    if _reranker_tokenizer is None or _reranker_model is None:
        logger.info("Loading reranker model (first time or cache cleared)...")
        _reranker_tokenizer, _reranker_model = load_reranker(RERANKER_MODEL)
        
        yes_ids = _reranker_tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = _reranker_tokenizer.encode(" no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            yes_ids = _reranker_tokenizer.encode("yes", add_special_tokens=False)
            no_ids = _reranker_tokenizer.encode("no", add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1
        _reranker_token_ids = (yes_ids[0], no_ids[0])
        
        try:
            if hasattr(_reranker_model, 'device'):
                _reranker_device = _reranker_model.device
            else:
                first_param = next(_reranker_model.parameters(), None)
                _reranker_device = first_param.device if first_param is not None else torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        except Exception:
            _reranker_device = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")
        
        _reranker_model.eval()  
    
    return _reranker_tokenizer, _reranker_model, _reranker_token_ids, _reranker_device


def rerank_results(
    query: str,
    candidate_chunks: List[Dict[str, str]],
    top_k: Optional[int] = None,
    batch_size: int = 512,  # 优化：利用剩余10G显存，增大batch_size以提升吞吐量
) -> List[Dict[str, str]]:
    """
    重排序候选文档（高性能优化版本）
    
    Args:
        query: 查询文本
        candidate_chunks: 候选文档列表
        top_k: 返回top k结果
        batch_size: 批处理大小（已优化以利用剩余显存）
    
    Returns:
        重排序后的文档列表
    """
    if not candidate_chunks:
        return []
    
    # 使用缓存的模型和tokenizer
    tokenizer, model, (token_true_id, token_false_id), model_device = get_reranker_model()
    
    # 准备输入对
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    docs = [c.get("clause_text", "") for c in candidate_chunks]
    pairs = [format_instruction(task, query, d) for d in docs]
    
    # 优化：对于大量数据，预先tokenize可以减少循环开销
    # 但如果数据量太大（>10000），仍然分批tokenize以避免CPU内存压力
    pre_tokenize_threshold = 10000
    if len(pairs) <= pre_tokenize_threshold:
        # 预先tokenize所有pairs，减少循环中的开销
        all_inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors='pt'
        )
        # 移动到GPU设备（一次性传输，减少开销）
        all_inputs = {k: v.to(model_device, non_blocking=True) for k, v in all_inputs.items()}
        use_pre_tokenized = True
    else:
        # 对于超大数据集，仍然分批tokenize
        all_inputs = None
        use_pre_tokenized = False
    
    # 优化：使用torch张量存储scores，避免频繁的CPU-GPU数据传输
    all_scores = []
    
    with torch.inference_mode():
        # 动态调整batch_size以避免OOM
        actual_batch_size = batch_size
        num_batches = (len(pairs) + actual_batch_size - 1) // actual_batch_size
        
        for i in range(0, len(pairs), actual_batch_size):
            end_idx = min(i + actual_batch_size, len(pairs))
            batch_pairs = pairs[i:end_idx]
            
            if use_pre_tokenized:
                # 从预处理的输入中提取当前batch
                batch_inputs = {
                    'input_ids': all_inputs['input_ids'][i:end_idx],
                    'attention_mask': all_inputs['attention_mask'][i:end_idx]
                }
            else:
                # 分批tokenize（用于超大数据集）
                batch_inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=RERANKER_MAX_LENGTH,
                    return_tensors='pt'
                )
                batch_inputs = {k: v.to(model_device, non_blocking=True) for k, v in batch_inputs.items()}
            
            # 前向传播（优化：只获取最后一个token的logits）
            outputs = model(**batch_inputs)
            batch_scores = outputs.logits[:, -1, :]
            
            # 优化：直接提取yes/no logits并计算分数，减少中间变量
            # 使用更高效的方式：直接索引并计算softmax
            true_logits = batch_scores[:, token_true_id]
            false_logits = batch_scores[:, token_false_id]
            
            # 优化：使用log_softmax + exp，数值更稳定且可能更快
            # 或者直接使用softmax（对于2个值，性能差异不大）
            logits = torch.stack([false_logits, true_logits], dim=1)
            batch_relevance_scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            
            # 优化：保持在GPU上，最后统一转换
            all_scores.append(batch_relevance_scores)
            
            # 释放中间变量
            del outputs, batch_scores, logits, true_logits, false_logits
            if not use_pre_tokenized:
                del batch_inputs  # 如果是分批tokenize，立即释放
            
            # 优化：减少清理频率，每16个batch清理一次（因为batch_size更大）
            if (i // actual_batch_size + 1) % 16 == 0:
                torch.cuda.empty_cache()
    
    # 优化：在GPU上合并所有scores，然后一次性转换到CPU
    all_scores_tensor = torch.cat(all_scores, dim=0)
    del all_scores
    if use_pre_tokenized:
        del all_inputs
    
    # 优化：使用torch的topk替代Python的sort，更快（直接在GPU上操作）
    if top_k and top_k < len(candidate_chunks):
        top_k_values, top_k_indices = torch.topk(all_scores_tensor, k=top_k, largest=True)
        top_k_indices = top_k_indices.cpu().tolist()
        results = [candidate_chunks[idx] for idx in top_k_indices]
        top_scores = top_k_values.cpu().tolist()
        max_score = top_k_values.max().item()  # 获取最大分数用于日志
        del top_k_values, top_k_indices
    else:
        # 如果不需要top_k或top_k >= len(scores)，使用传统排序
        scores = all_scores_tensor.cpu().tolist()
        scored_candidates = [(score, cand) for score, cand in zip(scores, candidate_chunks)]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = [cand for _, cand in scored_candidates]
        top_scores = [score for score, _ in scored_candidates]
        max_score = max(scores) if scores else 0.0
    
    del all_scores_tensor
    
    # 添加rerank_score
    for i, score in enumerate(top_scores):
        results[i]['rerank_score'] = float(score)
    
    logger.debug(f"Reranked {len(results)} candidates, top score: {max_score:.4f}")
    return results