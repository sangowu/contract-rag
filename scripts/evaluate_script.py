import sys, os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.eval_retrieval import evaluate_reranked_e2e, gold_chunk_coverage, evaluate_reranked_e2e_optimized
from src.rag.retrieval import retrieve_top_k_hybrid, retrieve_top_k
from src.inference.llm_inference import llm_generate

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked.csv"

def answer_fn(query: str, retrieved_data: List[Dict[str, str]]) -> str:

    ctx_chunks = [data['clause_text'] for data in retrieved_data]
    context = "\n\n".join(ctx_chunks)

    prompt = f"""
    Question: {query}
    Relevant contract clauses: {context}
    Answer as concisely as possible.
    """
    return llm_generate(prompt)

if __name__ == "__main__":
    df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)
    # chunks_df = pd.read_csv(CHUNK_PATH)
    # df_gold_answers_sample = df_gold_answers.head(20)

    # overall, by_clause, by_file, missing_tbl, missing_keys = gold_chunk_coverage(df_gold_answers, chunks_df)

    # logger.info(f"Gold->Chunk Strict Coverage: {overall['strict_coverage']:.4f}")
    # logger.info(f"Gold->Chunk Loose  Coverage: {overall['loose_coverage']:.4f}")
    # logger.info(f"Avg gold IDs per sample: {overall['avg_gold_ids_per_sample']:.2f}")
    # logger.info(f"="*50)
    # logger.info(f"Missing keys rows: {len(missing_keys)}")
    # logger.info(f"Top missing:\n{missing_tbl.head(5)}")
    # logger.info(f"="*50)

    e2e_results = evaluate_reranked_e2e_optimized(
        gold_df=df_gold_answers,
        retrieve_fn=retrieve_top_k_hybrid,
        top_k_shown=20,
        top_k_retrieved=100,
        top_k_reranked=10,
        plot_loc="reranked_e2e_batch_parent_child",
    )
    logger.info(f"E2E Results: {e2e_results}")
    E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked_parent_child.csv"
    e2e_results.to_csv(E2E_RESULTS_PATH, index=False)
    logger.success(f"Saved e2e evaluation results to {E2E_RESULTS_PATH}")

    os.system("/usr/bin/shutdown")