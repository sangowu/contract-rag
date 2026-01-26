import sys, os
import pandas as pd
from pathlib import Path
from typing import List, Dict
from loguru import logger
import requests
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils.eval_retrieval import evaluate_reranked_e2e, gold_chunk_coverage, evaluate_reranked_e2e_optimized
from src.inference.llm_inference import llm_generate

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked.csv"
RETRIEVAL_API_URL = "http://127.0.0.1:8000/api/retrieval/search"

def answer_fn(query: str, retrieved_data: List[Dict[str, str]]) -> str:

    ctx_chunks = [data['clause_text'] for data in retrieved_data]
    context = "\n\n".join(ctx_chunks)

    prompt = f"""
    Question: {query}
    Relevant contract clauses: {context}
    Answer as concisely as possible.
    """
    return llm_generate(prompt)

def retrieve_via_api(
    query: str,
    top_k_retrieval: int = 50,
    file_name: str | None = None,
    **kwargs,
) -> List[Dict[str, str]]:

    payload = {
        "query": query,
        "top_k_retrieval": top_k_retrieval,
        "file_name": file_name,
    }

    # 增加超时时间，因为首次请求需要加载模型
    resp = requests.post(
        RETRIEVAL_API_URL,
        json=payload,
        timeout=300,  # 增加到5分钟，给模型加载和首次推理足够时间
    )
    
    # 先获取响应数据，即使状态码不是200
    try:
        data = resp.json()
    except Exception:
        # 如果无法解析JSON，说明可能是服务器错误
        resp.raise_for_status()
    
    # 检查业务逻辑错误
    if not data.get("ok", True):
        error_msg = data.get("error", "Unknown error")
        logger.error(f"Retrieval API returned error: {error_msg}")
        raise RuntimeError(f"Retrieval API error: {error_msg}")
    
    # 检查HTTP状态码
    resp.raise_for_status()
    
    return data["data"]

if __name__ == "__main__":
    df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)

    e2e_results = evaluate_reranked_e2e_optimized(
        gold_df=df_gold_answers,
        retrieve_fn=retrieve_via_api,
        top_k_shown=30,
        top_k_retrieved=50,
        top_k_reranked=10,
        plot_loc="reranked_e2e_batch_parent_child_api",
    )
    logger.info(f"E2E Results: {e2e_results}")
    E2E_RESULTS_PATH = "/root/autodl-tmp/results/csv/cuad_v1_e2e_reranked_parent_child_api.csv"
    e2e_results.to_csv(E2E_RESULTS_PATH, index=False)
    logger.success(f"Saved e2e evaluation results to {E2E_RESULTS_PATH}")

    os.system("/usr/bin/shutdown")