import pandas as pd
from loguru import logger

CHUNK_PATH = "./data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "./data/answers/CUAD_v1/cuad_v1_gold_answers.csv"

df_chunk = pd.read_csv(CHUNK_PATH)
df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)

class DataChecker:
    def __init__(self, df_chunk: pd.DataFrame, df_gold_answers: pd.DataFrame):
        self.df_chunk = df_chunk
        self.df_gold_answers = df_gold_answers

    def check_data(self):
        logger.info("Checking data...")
        logger.info(f"Number of unique files in chunks: {df_chunk['file_name'].nunique()}")
        logger.info(f"Number of unique files in gold answers: {df_gold_answers['file_name'].nunique()}")
        logger.info(f"Number of null gold chunk ids in gold answers: {df_gold_answers['gold_chunk_ids'].isnull().sum()}")


if __name__ == "__main__":
    data_checker = DataChecker(df_chunk, df_gold_answers).check_data()