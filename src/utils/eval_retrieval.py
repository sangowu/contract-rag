from typing import List, Callable, Tuple, Dict
from tqdm import tqdm
import ast  
import torch
import gc
import pandas as pd
from loguru import logger

from src.utils.plot import plot_hits, plot_rrs, plot_recalls
from src.utils.query_builder import build_query
from src.utils.eval_answers import eval_one
from config.cuad_meta import get_answer_type

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")

def release_all_models():
    try:
        import src.rag.embedding as embedding_module
        if hasattr(embedding_module, '_model') and embedding_module._model is not None:
            del embedding_module._model
            embedding_module._model = None
            logger.info("Embedding model released")
        
        if hasattr(embedding_module, '_collection'):
            embedding_module._collection = None
        
        import src.inference.llm_inference as llm_module
        if hasattr(llm_module, 'transformers_model') and llm_module.transformers_model is not None:
            del llm_module.transformers_model
            llm_module.transformers_model = None
            logger.info("Transformers model released")
        
        if hasattr(llm_module, 'tokenizer'):
            llm_module.tokenizer = None
        
        for _ in range(3):
            clear_gpu_memory()
        
        logger.success("All models released and GPU memory cleared")
        
    except Exception as e:
        logger.warning(f"Failed to release some models: {e}")
        clear_gpu_memory()


def hit_at_k(retrieved: List[str], gold: List[str], k: int = 10) -> int:
    """
    Check if any of the top k retrieved chunk_ids are in the gold chunk_ids.
    """

    top_k = retrieved[:k]
    return int(any(cid in top_k for cid in gold))

def mrr_at_k(retrieved: List[str], gold: List[str], k: int = 10) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) at k.
    """
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved: List[str], gold: List[str], k: int = 10) -> float:
    """
    Calculate the Recall at k.
    """
    top_k = set(retrieved[:k])
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    hits = len(top_k & gold_set)
    return hits / len(gold_set)

def _ensure_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return [x]
    return x

def gold_chunk_coverage(
    gold_df: pd.DataFrame,
    chunks_df: pd.DataFrame,
    gold_ids_col: str = "gold_chunk_ids",
    key_cols: Tuple[str, str] = ("file_name", "clause_type"),
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    gold = gold_df.copy()
    gold[gold_ids_col] = gold[gold_ids_col].apply(_ensure_list)

    chunk_id_set = set(chunks_df["chunk_id"])

    def _sample_stats(row):
        gold_ids = row[gold_ids_col] or []
        present = [cid for cid in gold_ids if cid in chunk_id_set]
        missing = [cid for cid in gold_ids if cid not in chunk_id_set]
        strict = int(len(missing) == 0 and len(gold_ids) > 0)  
        loose  = int(len(present) > 0)                       
        return pd.Series({
            "strict_cover": strict,
            "loose_cover": loose,
            "n_gold_ids": len(gold_ids),
            "n_present": len(present),
            "n_missing": len(missing),
            "missing_ids": present if not strict else [], 
            "present_ids": present,
        })

    cov = gold.join(gold.apply(_sample_stats, axis=1))

    overall = {
        "strict_coverage": cov["strict_cover"].mean() if len(cov) else 0.0,
        "loose_coverage":  cov["loose_cover"].mean()  if len(cov) else 0.0,
        "avg_gold_ids_per_sample": cov["n_gold_ids"].mean() if len(cov) else 0.0,
    }

    by_clause = cov.groupby("clause_type")[["strict_cover", "loose_cover"]].mean().reset_index()
    by_file   = cov.groupby("file_name")[["strict_cover", "loose_cover"]].mean().reset_index()

    missing_tbl = cov[cov["n_missing"] > 0][
        ["file_name", "clause_type", "missing_ids", "n_missing"]
    ].sort_values(["n_missing"], ascending=False)

    gold_keys  = gold[list(key_cols)].drop_duplicates()
    chunk_keys = chunks_df[list(key_cols)].drop_duplicates()
    key_diff = gold_keys.merge(chunk_keys, on=list(key_cols), how="left", indicator=True)
    missing_keys = key_diff[key_diff["_merge"] == "left_only"].drop(columns=["_merge"])

    return overall, by_clause, by_file, missing_tbl, missing_keys

def evaluate_retrieval(
    gold_df: pd.DataFrame,
    retrieve_fn: Callable[[str, int], List[Dict[str, str]]],
    k: int = 10,
    top_k_retrieved: int = 50,
    plot: bool = True,
    plot_loc: str = "vanilla_retrieval",
) -> Tuple[float, float, float]:
    if len(gold_df) > 0 and isinstance(gold_df['gold_chunk_ids'].iloc[0], str):
        gold_df = gold_df.copy()
        gold_df['gold_chunk_ids'] = gold_df['gold_chunk_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    hits: List[int] = []
    rrs: List[float] = []
    recalls: List[float] = []

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="evaluating retrieval"):
        query = row['query']
        file_name = row['file_name']
        gold_chunk_ids = row['gold_chunk_ids']
        gold_chunk_ids = [str(cid) for cid in gold_chunk_ids]
        retrieved_data: List[Dict[str, str]] = retrieve_fn(query, k=top_k_retrieved, file_name=file_name)
        retrieved_ids = [str(data['chunk_id']) for data in retrieved_data]
        hits.append(hit_at_k(retrieved_ids, gold_chunk_ids, k))
        rrs.append(mrr_at_k(retrieved_ids, gold_chunk_ids, k))
        recalls.append(recall_at_k(retrieved_ids, gold_chunk_ids, k))

    logger.info(f"Hits: {hits[:10]}")
    logger.info(f"RRs: {rrs[:10]}")
    logger.info(f"Recalls: {recalls[:10]}")
    if plot:
        log_hits = pd.DataFrame(hits, columns=['hits'])
        log_rrs = pd.DataFrame(rrs, columns=['rrs'])
        log_recalls = pd.DataFrame(recalls, columns=['recalls'])
        plot_hits(log_hits, plot_loc)
        plot_rrs(log_rrs, plot_loc)
        plot_recalls(log_recalls, plot_loc)
    else:
        logger.info("No plots generated")

    hit_rate_at_k = sum(hits) / len(hits) if hits else 0.0
    recall_at_k_value = sum(recalls) / len(recalls) if recalls else 0.0
    mrr_at_k_value = sum(rrs) / len(rrs) if rrs else 0.0

    logger.success(f"Recall@{k}: {recall_at_k_value:.4f}, MRR@{k}: {mrr_at_k_value:.4f}, Hit Rate@{k}: {hit_rate_at_k:.4f}")

    return hit_rate_at_k, recall_at_k_value, mrr_at_k_value

def evaluate_e2e(
    gold_df: pd.DataFrame,
    retrieve_fn: Callable[[str, int], List[Dict[str, str]]],
    answer_fn: Callable[[str, List[Dict[str, str]]], str],
    top_k_shown: int = 10,
    top_k_retrieved: int = 50,
    plot: bool = True,
    plot_loc: str = "vanilla_e2e",
) -> pd.DataFrame:

    if len(gold_df) > 0 and isinstance(gold_df['gold_chunk_ids'].iloc[0], str):
        gold_df = gold_df.copy()
        gold_df['gold_chunk_ids'] = gold_df['gold_chunk_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    hits = []
    rrs = []
    recalls = []
    records = []

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="evaluating e2e"):
        category = row["clause_type"]
        gold_answer = row["gold_answer_text"]
        gold_chunk_ids = row["gold_chunk_ids"]
        file_name = row["file_name"]
        if "query" in row and isinstance(row["query"], str):
            query = row["query"]
        else:
            query = build_query(category)

        retrieved_data: List[Dict[str, str]] = retrieve_fn(query, k=top_k_retrieved, file_name=file_name)
        retrieved_ids = [data['chunk_id'] for data in retrieved_data]

        hit = hit_at_k(retrieved_ids, gold_chunk_ids, top_k_shown)
        rr = mrr_at_k(retrieved_ids, gold_chunk_ids, top_k_shown)
        rec = recall_at_k(retrieved_ids, gold_chunk_ids, top_k_shown)

        hits.append(hit)
        rrs.append(rr)
        recalls.append(rec)

        model_answer = answer_fn(query, retrieved_data)

        ans_metrics = eval_one(category, gold_answer, model_answer)

        records.append({
            "category": category,
            "answer_type": get_answer_type(category),
            "query": query,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "hit@k": hit,
            "rr@k": rr,
            "recall@k": rec,
            **ans_metrics,   
        })

    if plot:
        log_hits = pd.DataFrame(hits, columns=['hits'])
        log_rrs = pd.DataFrame(rrs, columns=['rrs'])
        log_recalls = pd.DataFrame(recalls, columns=['recalls'])
        plot_hits(log_hits, plot_loc)
        plot_rrs(log_rrs, plot_loc)
        plot_recalls(log_recalls, plot_loc)
    else:
        logger.info("No plots generated")

    result_df = pd.DataFrame(records)
    return result_df