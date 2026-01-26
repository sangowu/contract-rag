import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

import pandas as pd
from loguru import logger
from inference.llm_inference import llm_generate
# from rag.embedding import initialize_embeddings


GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)

if __name__ == "__main__":
    logger.info("Starting LLM script...")
    # initialize_embeddings()
    max_queries = 2
    for i, row in df_gold_answers.iterrows():
        if i >= max_queries:
            break
        prompt = row['query']
        answer = llm_generate(prompt)
        logger.info(f"Answer {i+1}: {answer}")
    logger.success("LLM script completed successfully")