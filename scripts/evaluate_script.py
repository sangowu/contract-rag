import pandas as pd
import sys
import os
from typing import List
from loguru import logger
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.evaluate_retrieval import evaluate_retrieval
from src.rag.embedding import retrieve_top_k
from src.utils.evaluate_retrieval import evaluate_e2e
from src.inference.llm_inference import llm_generate

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_vanilla.csv"


def answer_fn(query: str, retrieved_ids: List[str]) -> str:
    """
    Answer the question based on the retrieved chunk_ids.
    """
    df_chunk = pd.read_csv(CHUNK_PATH)
    chunk_id_to_text = dict(zip(df_chunk['chunk_id'], df_chunk['clause_text']))
    ctx_chunks = [chunk_id_to_text[cid] for cid in retrieved_ids if cid in chunk_id_to_text]
    context = "\n\n".join(ctx_chunks)

    prompt = f"""
    Question: {query}
    Relevant contract clauses: {context}
    Answer as concisely as possible.
    """
    return llm_generate(prompt)

if __name__ == "__main__":
    df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)

    hit_rate, recall, mrr = evaluate_retrieval(
        gold_df=df_gold_answers, 
        retrieve_fn=retrieve_top_k, 
        k=10, 
        top_k_retrieved=50,
    )

    logger.info(f"Hit Rate at K: {hit_rate}")
    logger.info(f"Recall at K: {recall}")
    logger.info(f"MRR: {mrr}")

    # e2e_results = evaluate_e2e(
    #     gold_df=df_gold_answers,
    #     retrieve_fn=retrieve_top_k,
    #     answer_fn=answer_fn,
    #     k=10,
    #     top_k_retrieved=50,
    # )
    # logger.info(f"E2E Results: {e2e_results}")

    # e2e_results.to_csv(E2E_RESULTS_PATH, index=False)
    # logger.success(f"Saved e2e evaluation results to {E2E_RESULTS_PATH}")