import pandas as pd
from typing import List, Tuple
from loguru import logger
from sentence_transformers import SentenceTransformer
import chromadb
import os

CHUNK_PATH = "/root/autodl-tmp/data/processed/CUAD_v1/cuad_v1_chunks.csv"
GOLD_ANSWERS_PATH = "/root/autodl-tmp/data/answers/CUAD_v1/cuad_v1_gold_answers.csv"
EMBEDDING_MODEL = "/root/autodl-tmp/model/sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DB_PATH = "/root/autodl-tmp/data/embeddings/chroma_db"
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

def initialize_embeddings():
    df_chunk = pd.read_csv(CHUNK_PATH)

    model = SentenceTransformer(EMBEDDING_MODEL)
    chunk_ids = df_chunk['chunk_id'].tolist()
    chunk_texts = df_chunk['clause_text'].tolist()

    embeddings = model.encode(
        chunk_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        )

    collection = chroma_client.get_or_create_collection(name="cuad_chunks")
    if collection.count() == 0:
        batch_size = 4096
        total_items = len(chunk_ids)
        logger.info(f"Adding {total_items} embeddings in batches of {batch_size}")

        for i in range(0, total_items, batch_size):
            end_idx  = min(i + batch_size, total_items)
            batch_embeddings = embeddings[i:end_idx].tolist()
            batch_metadatas = [
                {"chunk_id": cid, "clause_text": text} for cid, text in zip(chunk_ids[i:end_idx], chunk_texts[i:end_idx])
            ]
            batch_ids = chunk_ids[i:end_idx]

            collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids,
            )
            logger.info(f"Added batch {i//batch_size + 1}: items {i} to {end_idx-1}")

        logger.success(f"Added {total_items} embeddings to ChromaDB")
    else:
        logger.info("Embeddings already exist in ChromaDB, skipping addition")

def retrieve_top_k(query: str, k: int = 10) -> List[List[str]]:
    try:
        model = get_model()
        collection = get_collection()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        
        results = collection.query(
            query_embeddings=[q_emb.tolist()],
            n_results=k,
            include=['metadatas'],
        )
        
        chunk_ids = results['ids'][0]
        chunk_texts = [meta['clause_text'] for meta in results['metadatas'][0]]

        chunks = [{"chunk_id": chunk_id, "clause_text": chunk_text} for chunk_id, chunk_text in zip(chunk_ids, chunk_texts)]
        
        return chunks
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return []

if __name__ == "__main__":
    initialize_embeddings()
    df_gold_answers = pd.read_csv(GOLD_ANSWERS_PATH)
    row = df_gold_answers.iloc[0]
    gold_answer = row['gold_answer_text']
    gold_chunk_ids = row['gold_chunk_ids']
    query = row['query']
    topk_chunks = retrieve_top_k(query, k=10)
    logger.info(f"Top 10 chunk ids: {topk_chunks}")
    logger.info(f"Gold answer: {gold_answer}")
    logger.info(f"Gold chunk ids: {gold_chunk_ids}")
    logger.info(f"Query: {query}")