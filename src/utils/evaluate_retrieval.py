from typing import List, Callable, Tuple, Dict
from tqdm import tqdm
import ast
import pandas as pd
from loguru import logger

from src.utils.plot import plot_hits, plot_rrs, plot_recalls
from src.utils.query_builder import build_query
from src.utils.eval_answers import eval_one, get_answer_type
from config.cuad_meta import get_answer_type


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

def evaluate_retrieval(
    gold_df: pd.DataFrame,
    retrieve_fn: Callable[[str, int], List[str]],
    k: int = 10,
    top_k_retrieved: int = 50,
    plot: bool = True,
    plot_loc: str = "vanilla_retrieval",
) -> Tuple[float, float, float]:
    """
    Evaluate the retrieval performance.
    """
    if len(gold_df) > 0 and isinstance(gold_df['gold_chunk_ids'].iloc[0], str):
        gold_df = gold_df.copy()
        gold_df['gold_chunk_ids'] = gold_df['gold_chunk_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    hits: List[int] = []
    rrs: List[float] = []
    recalls: List[float] = []

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="evaluating retrieval"):
        query = row['query']
        gold_chunk_ids = row['gold_chunk_ids']
        retrieved_data: List[Dict[str, str]] = retrieve_fn(query, top_k_retrieved)
        retrieved_ids = [data['chunk_id'] for data in retrieved_data]

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
    retrieve_fn: Callable[[str, int], List[str]],
    answer_fn: Callable[[str, List[str]], str],
    k: int = 10,
    top_k_retrieved: int = 50,
) -> pd.DataFrame:
    """
    Evaluate the end-to-end performance.
    """
    if len(gold_df) > 0 and isinstance(gold_df['gold_chunk_ids'].iloc[0], str):
        gold_df = gold_df.copy()
        gold_df['gold_chunk_ids'] = gold_df['gold_chunk_ids'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    records = []

    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="evaluating e2e"):
        category = row["category"]
        gold_answer = row["answer"]
        gold_chunk_ids = row["gold_chunk_ids"]

        if "query" in row and isinstance(row["query"], str):
            query = row["query"]
        else:
            query = build_query(category)

        retrieved_ids: List[str] = retrieve_fn(query, top_k_retrieved)
        hit = hit_at_k(retrieved_ids, gold_chunk_ids, k)
        rr = mrr_at_k(retrieved_ids, gold_chunk_ids, k)
        rec = recall_at_k(retrieved_ids, gold_chunk_ids, k)

        model_answer = answer_fn(query, retrieved_ids)

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

    result_df = pd.DataFrame(records)
    return result_df