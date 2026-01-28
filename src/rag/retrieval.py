"""
CUAD 检索模块

功能:
- BM25 检索
- 向量检索
- 混合检索 (RRF)
- Reranker 重排序
"""

from typing import List, Dict, Optional
from loguru import logger
from pathlib import Path
import torch
import re
import pandas as pd
import pickle

from .embedding import get_model, get_collection, normalize_text
from src.utils.model_loading import load_reranker
from src.core.config import get_config


# =============================================================================
# 全局缓存
# =============================================================================

_chunk_df: Optional[pd.DataFrame] = None
_bm25_index = None

# Reranker 模型缓存
_reranker_tokenizer = None
_reranker_model = None
_reranker_token_ids = None
_reranker_device = None

# 配置缓存
RERANKER_MAX_LENGTH = 256


# =============================================================================
# 路径和配置获取
# =============================================================================

def get_retrieval_paths():
    """
    从配置获取检索相关路径
    
    Returns:
        dict: 包含所有路径的字典
    """
    config = get_config()
    
    # BM25 索引文件名（支持配置，默认使用 PDF 索引）
    bm25_index_file = getattr(config.retrieval.bm25, 'index_file', 'bm25_pdf_index.pkl')
    
    return {
        'bm25_index_path': str(Path(config.retrieval.bm25.index_path) / bm25_index_file),
        'chunk_path': str(config.data.chunks_path),
        'reranker_model': config.models.reranker.path,
        'reranker_top_k': config.retrieval.rerank.top_k,
        'hybrid_top_k': config.retrieval.hybrid.top_k,
    }


# =============================================================================
# 数据加载
# =============================================================================

def get_chunk_df() -> pd.DataFrame:
    """获取 chunk DataFrame (缓存)"""
    global _chunk_df
    if _chunk_df is None:
        paths = get_retrieval_paths()
        _chunk_df = pd.read_csv(paths['chunk_path'])
        logger.info(f"Loaded chunk DataFrame from {paths['chunk_path']}")
    return _chunk_df


def load_bm25_index():
    """加载 BM25 索引 (缓存)"""
    global _bm25_index
    if _bm25_index is None:
        paths = get_retrieval_paths()
        bm25_path = Path(paths['bm25_index_path'])
        
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                _bm25_index = pickle.load(f)
            logger.info(f"Loaded BM25 index from {bm25_path}")
        else:
            raise FileNotFoundError(f"BM25 index not found at {bm25_path}")
    
    return _bm25_index


# =============================================================================
# Reranker
# =============================================================================

def format_instruction(instruction: Optional[str], query: str, doc: str) -> str:
    """格式化 reranker 输入"""
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)
    return output


def get_reranker_model():
    """
    获取 reranker 模型和 tokenizer (缓存)
    
    Returns:
        (tokenizer, model, token_ids, device) 元组
    """
    global _reranker_tokenizer, _reranker_model, _reranker_token_ids, _reranker_device
    
    if _reranker_tokenizer is None or _reranker_model is None:
        paths = get_retrieval_paths()
        model_path = paths['reranker_model']
        
        logger.info(f"Loading reranker model from {model_path}...")
        _reranker_tokenizer, _reranker_model = load_reranker(model_path)
        
        # 获取 yes/no token IDs
        yes_ids = _reranker_tokenizer.encode(" yes", add_special_tokens=False)
        no_ids = _reranker_tokenizer.encode(" no", add_special_tokens=False)
        if len(yes_ids) != 1 or len(no_ids) != 1:
            yes_ids = _reranker_tokenizer.encode("yes", add_special_tokens=False)
            no_ids = _reranker_tokenizer.encode("no", add_special_tokens=False)
        assert len(yes_ids) == 1 and len(no_ids) == 1
        _reranker_token_ids = (yes_ids[0], no_ids[0])
        
        # 获取设备
        try:
            if hasattr(_reranker_model, 'device'):
                _reranker_device = _reranker_model.device
            else:
                first_param = next(_reranker_model.parameters(), None)
                _reranker_device = first_param.device if first_param is not None else torch.device(
                    "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
                )
        except Exception:
            _reranker_device = torch.device(
                "cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
            )
        
        _reranker_model.eval()
        logger.info(f"Reranker model loaded on {_reranker_device}")
    
    return _reranker_tokenizer, _reranker_model, _reranker_token_ids, _reranker_device


# =============================================================================
# 检索函数
# =============================================================================

def bm25_search(
    query: str, 
    top_k_retrieval: int = 10, 
    file_name: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    BM25 检索
    
    Args:
        query: 查询文本
        top_k_retrieval: 返回数量
        file_name: 过滤文件名 (可选)
    
    Returns:
        检索结果列表
    """
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
    top_k_shown: Optional[int] = None,
    file_name: Optional[str] = None,
    top_k_retrieval: int = 100,
) -> List[Dict[str, str]]:
    """
    向量检索
    
    Args:
        query: 查询文本
        top_k_shown: 返回数量
        file_name: 过滤文件名 (可选)
        top_k_retrieval: 检索数量
    
    Returns:
        检索结果列表
    """
    try:
        model = get_model()
        collection = get_collection()
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
    file_name: Optional[str] = None,
    top_k_retrieval: int = 100,
    rrf_k: int = 20, 
) -> List[Dict[str, str]]:
    """
    混合检索 (RRF: Reciprocal Rank Fusion)
    
    Args:
        query: 查询文本
        top_k_shown: 返回数量
        file_name: 过滤文件名 (可选)
        top_k_retrieval: 检索数量
        rrf_k: RRF 参数
    
    Returns:
        检索结果列表
    """
    try:
        # BM25 检索
        bm25_results = bm25_search(query, top_k_retrieval=top_k_retrieval, file_name=file_name)
        bm25_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(bm25_results)}
        
        # 向量检索
        retrieval_results = retrieve_top_k(query, file_name=file_name, top_k_retrieval=top_k_retrieval)
        retrieval_ranks = {result['chunk_id']: rank+1 for rank, result in enumerate(retrieval_results)}

        # RRF 融合
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
    batch_size: int = 512,
) -> List[Dict[str, str]]:
    """
    重排序候选文档 (高性能优化版本)
    
    Args:
        query: 查询文本
        candidate_chunks: 候选文档列表
        top_k: 返回 top k 结果 (默认从配置读取)
        batch_size: 批处理大小
    
    Returns:
        重排序后的文档列表
    """
    if not candidate_chunks:
        return []
    
    # 默认 top_k 从配置读取
    if top_k is None:
        paths = get_retrieval_paths()
        top_k = paths['reranker_top_k']
    
    # 使用缓存的模型和 tokenizer
    tokenizer, model, (token_true_id, token_false_id), model_device = get_reranker_model()
    
    # 准备输入对
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    docs = [c.get("clause_text", "") for c in candidate_chunks]
    pairs = [format_instruction(task, query, d) for d in docs]
    
    # 预处理阈值
    pre_tokenize_threshold = 10000
    if len(pairs) <= pre_tokenize_threshold:
        all_inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=RERANKER_MAX_LENGTH,
            return_tensors='pt'
        )
        all_inputs = {k: v.to(model_device, non_blocking=True) for k, v in all_inputs.items()}
        use_pre_tokenized = True
    else:
        all_inputs = None
        use_pre_tokenized = False
    
    all_scores = []
    
    with torch.inference_mode():
        actual_batch_size = batch_size
        
        for i in range(0, len(pairs), actual_batch_size):
            end_idx = min(i + actual_batch_size, len(pairs))
            batch_pairs = pairs[i:end_idx]
            
            if use_pre_tokenized:
                batch_inputs = {
                    'input_ids': all_inputs['input_ids'][i:end_idx],
                    'attention_mask': all_inputs['attention_mask'][i:end_idx]
                }
            else:
                batch_inputs = tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=RERANKER_MAX_LENGTH,
                    return_tensors='pt'
                )
                batch_inputs = {k: v.to(model_device, non_blocking=True) for k, v in batch_inputs.items()}
            
            outputs = model(**batch_inputs)
            batch_scores = outputs.logits[:, -1, :]
            
            true_logits = batch_scores[:, token_true_id]
            false_logits = batch_scores[:, token_false_id]
            
            logits = torch.stack([false_logits, true_logits], dim=1)
            batch_relevance_scores = torch.nn.functional.softmax(logits, dim=1)[:, 1]
            
            all_scores.append(batch_relevance_scores)
            
            del outputs, batch_scores, logits, true_logits, false_logits
            if not use_pre_tokenized:
                del batch_inputs
            
            if (i // actual_batch_size + 1) % 16 == 0:
                torch.cuda.empty_cache()
    
    all_scores_tensor = torch.cat(all_scores, dim=0)
    del all_scores
    if use_pre_tokenized:
        del all_inputs
    
    # 使用 torch 的 topk
    if top_k and top_k < len(candidate_chunks):
        top_k_values, top_k_indices = torch.topk(all_scores_tensor, k=top_k, largest=True)
        top_k_indices = top_k_indices.cpu().tolist()
        results = [candidate_chunks[idx] for idx in top_k_indices]
        top_scores = top_k_values.cpu().tolist()
        max_score = top_k_values.max().item()
        del top_k_values, top_k_indices
    else:
        scores = all_scores_tensor.cpu().tolist()
        scored_candidates = [(score, cand) for score, cand in zip(scores, candidate_chunks)]
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        results = [cand for _, cand in scored_candidates]
        top_scores = [score for score, _ in scored_candidates]
        max_score = max(scores) if scores else 0.0
    
    del all_scores_tensor
    
    # 添加 rerank_score
    for i, score in enumerate(top_scores):
        results[i]['rerank_score'] = float(score)
    
    logger.debug(f"Reranked {len(results)} candidates, top score: {max_score:.4f}")
    return results
