import json
import os
import pandas as pd
from tqdm import tqdm
from loguru import logger
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.query_builder import build_query


def load_data_csv(data_path):
    df = pd.read_csv(data_path)
    return df

def load_data_json(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

os.makedirs("/root/autodl-tmp/data/processed/CUAD_v1", exist_ok=True)
os.makedirs("/root/autodl-tmp/data/answers/CUAD_v1", exist_ok=True)
DATA_PATH = "/root/autodl-tmp/data/raw/CUAD_v1/master_clauses.csv"
CHUNK_OUTPUT_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_OUTPUT_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
df_raw = load_data_csv(DATA_PATH)

class RawToChunkRecordsProcessor:
    def __init__(self, df_raw: pd.DataFrame):
        self.df_raw = df_raw
        self.df_chunks = df_raw.copy()
        self.answer_columns = [c for c in self.df_chunks.columns if c.endswith('-Answer')]
        self.df_chunks.drop(self.answer_columns, axis=1, inplace=True)
        self.clause_text_cols = [c for c in self.df_chunks.columns if c not in ['Filename']]
        self.records = []

    def process(self) -> pd.DataFrame:
        for idx, row in tqdm(self.df_chunks.iterrows(), total=len(self.df_chunks)):
            filename = row['Filename']

            for col in self.clause_text_cols:
                text = row[col]
                if pd.isna(text):
                    continue
                ans_col = f"{col}-Answer"
                if ans_col in self.df_raw.columns:
                    ans_val = self.df_raw.loc[idx, ans_col]
                    has_answer = pd.notna(ans_val) and str(ans_val).strip() != ''
                else:
                    has_answer = False
                
                chunk_idx = 0
                start_char = 0
                end_char = len(text)
                chunk_id = f"{filename}::{col}::{idx}::{chunk_idx}::{start_char}:{end_char}"  
                self.records.append({
                    'file_name': filename,
                    'chunk_id': chunk_id,
                    'clause_type': col,
                    'clause_text': text,
                    'parent_clause_text': text,
                    'contract_idx': idx,
                    'chunk_idx': chunk_idx,
                    'start_char': start_char,
                    'end_char': end_char,
                    'has_answer': has_answer,
                    'source': 'master_clauses',
                })

        df_records = pd.DataFrame(self.records)
        df_records.to_csv(CHUNK_OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(df_records)} chunks to {CHUNK_OUTPUT_PATH}")

        return df_records

class RawToGoldAnswersProcessor:
    def __init__(self, df_raw: pd.DataFrame, chunk_df: pd.DataFrame):
        self.df_raw = df_raw
        self.chunk_df = chunk_df
        self.df_gold_answers = df_raw.copy()
        self.answer_columns = [c for c in self.df_gold_answers.columns if c.endswith('-Answer')]
        self.clause_text_cols = [c for c in self.df_gold_answers.columns if c not in ['Filename'] and c not in self.answer_columns]
        self.records = []

    def process(self) -> pd.DataFrame:
        for idx, row in tqdm(self.df_gold_answers.iterrows(), total=len(self.df_gold_answers)):
            filename = row['Filename']
 
            for col_ans in self.answer_columns:
                ans_val = row[col_ans]
                if pd.isna(ans_val):
                    continue
                gold_answer_text = str(ans_val).strip()
                clause_type = col_ans.replace('-Answer', '')

                mask = (
                    (self.chunk_df['file_name'] == filename) &
                    (self.chunk_df['clause_type'] == clause_type) &
                    (self.chunk_df['contract_idx'] == idx)
                )
                matched_chunk_ids = self.chunk_df.loc[mask, 'chunk_id'].tolist()

                if not matched_chunk_ids:
                    logger.warning(f"No chunk found for {filename}::{clause_type}::{idx}")
                    continue

                matched_chunk_ids.sort(key=lambda x: int(x.split('::')[3]) if len(x.split('::')) >= 4 else 0)
                sample_id = f"{filename}::{clause_type}::{idx}"
                query = build_query(clause_type)

                self.records.append({
                    'sample_id': sample_id,
                    'file_name': filename,
                    'gold_chunk_ids': matched_chunk_ids,
                    'clause_type': clause_type,
                    'query' : query,
                    'gold_answer_text': gold_answer_text,
                    'contract_idx': idx,
                    'source': 'master_clauses',
                })
        df_gold = pd.DataFrame(self.records)
        df_gold.to_csv(GOLD_ANSWERS_OUTPUT_PATH, index=False)
        logger.info(f"Saved {len(df_gold)} gold answers to {GOLD_ANSWERS_OUTPUT_PATH}")
        return df_gold


if __name__ == "__main__":
    logger.info("Loading raw data...")
    df_raw = load_data_csv(DATA_PATH)

    logger.info("Building CUAD_v1 chunk records...")
    chunk_records = RawToChunkRecordsProcessor(df_raw).process()
    logger.info("Building CUAD_v1 gold answers...")
    gold_answers = RawToGoldAnswersProcessor(df_raw, chunk_records).process()

    logger.info("Done!")


            