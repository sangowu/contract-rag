import pandas as pd
import pickle
from typing import List, Dict
from loguru import logger
from sentence_transformers import SentenceTransformer
import re, os, sys, ast, unicodedata, chromadb
from pathlib import Path
from rank_bm25 import BM25L
from loguru import logger

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
EMBEDDING_MODEL = "/root/autodl-tmp/model/sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "/root/autodl-tmp/data/indexes/embeddings/chroma_db"
BM25_INDEX_PATH = "/root/autodl-tmp/data/indexes/bm25"
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
_model = None
_collection = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model  

def get_collection():
    global _collection
    if _collection is None:
        _collection = chroma_client.get_collection(name="cuad_chunks")
    return _collection

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()

def build_bm25_index(chunk_texts: List[str]):
    tokenized_corpus = []
    for text in chunk_texts:
        normalized = normalize_text(text)
        tokens = re.findall(r'\b\w+\b', normalized)
        tokenized_corpus.append(tokens)

    bm25 = BM25L(tokenized_corpus) 
    Path(BM25_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    index_file = Path(BM25_INDEX_PATH) / "bm25_index.pkl"
    with open(index_file, 'wb') as f:
        pickle.dump(bm25, f)
    
    logger.success(f"Built BM25 index with {len(tokenized_corpus)} documents")
    return bm25, tokenized_corpus

def initialize_embeddings():
    df_chunk = pd.read_csv(CHUNK_PATH)
    model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_ids = df_chunk['chunk_id'].astype(str).tolist()
    file_names = df_chunk['file_name'].astype(str).tolist()
    clause_types = df_chunk['clause_type'].astype(str).tolist()

    chunk_texts = []
    for text in df_chunk['clause_text']:
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list):
                natural_text = ", ".join(str(item) for item in items)
            else:
                natural_text = str(text)
        except:
            natural_text = str(text)
        chunk_texts.append(normalize_text(natural_text))

    build_bm25_index(chunk_texts)

    embeddings = model.encode(
        chunk_texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        )

    collection = chroma_client.get_or_create_collection(name="cuad_chunks")

    batch_size = 4096
    total_items = len(chunk_ids)
    logger.info(f"Upserting {total_items} embeddings in batches of {batch_size}")

    for i in range(0, total_items, batch_size):
        j = min(i + batch_size, total_items)
        collection.upsert(
            ids=chunk_ids[i:j],                     
            embeddings=embeddings[i:j].tolist(),    
            metadatas=[
                {
                    "chunk_id": chunk_ids[k],
                    "file_name": file_names[k],      
                    "clause_type": clause_types[k],   
                    "clause_text": chunk_texts[k],    
                }
                for k in range(i, j)
            ],
        )
        logger.info(f"Upserted items {i}..{j-1}")

    logger.success(f"Upserted {total_items} embeddings to ChromaDB")

if __name__ == "__main__":
    initialize_embeddings()