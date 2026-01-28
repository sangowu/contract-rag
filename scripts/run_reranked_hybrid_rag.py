#!/usr/bin/env python
"""
Reranked Hybrid RAG End-to-End Test Script

Features:
- Hybrid retrieval (vector + BM25 with RRF fusion)
- Reranker-based re-ranking of hybrid results
- Evaluation of retrieval and generation quality

Usage:
    # Full test
    python scripts/run_reranked_hybrid_rag.py --mode test
    
    # Test with limited samples
    python scripts/run_reranked_hybrid_rag.py --mode test --max-samples 100
    
    # Adjust weights and top-k
    python scripts/run_reranked_hybrid_rag.py --mode test --bm25-weight 0.4 --vector-weight 0.6 --rerank-top-k 10
    
    # Show GPU configuration
    python scripts/run_reranked_hybrid_rag.py --show-gpu
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import pandas as pd

# Core modules
from src.core.config import init_config, get_config
from src.core.gpu_manager import init_gpu_manager, get_gpu_config
from src.data.data_loader import load_gold_answers
from src.rag.embedding import get_model
from src.rag.retrieval import (
    retrieve_top_k_hybrid,
    load_bm25_index,
    rerank_results,
    get_reranker_model,
)
from src.inference.llm_inference import llm_generate
from src.evaluation import EvaluationPipeline, EvaluationConfig
from src.utils.seed_utils import set_global_seed, get_seed


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOP_K = 10
DEFAULT_TOP_K_RETRIEVAL = 50
DEFAULT_RRF_K = 20
DEFAULT_RERANK_TOP_K = 10
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_BATCH_SIZE = 512


# =============================================================================
# GPU Information Display
# =============================================================================

def show_gpu_info() -> None:
    """Display GPU configuration information."""
    try:
        gpu_config = get_gpu_config()
        print("\n" + "=" * 60)
        print("GPU Configuration")
        print("=" * 60)
        print(f"  Mode: {gpu_config.mode}")
        print(f"  GPU Count: {gpu_config.gpu_count}")
        print(f"  LLM Model: {gpu_config.llm_model_name}")
        print(f"  LLM Devices: {gpu_config.llm_devices}")
        print(f"  LLM GPU Util: {gpu_config.llm_gpu_memory_utilization}")
        print(f"  Other Devices: {gpu_config.other_devices}")
        if gpu_config.total_memory_gb:
            for i, mem in enumerate(gpu_config.total_memory_gb):
                print(f"  GPU {i} Memory: {mem} GB")
        print("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"GPU Manager not initialized: {e}")


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Reranked Hybrid RAG End-to-End Test (Vector + BM25 + RRF + Reranker)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='test',
        choices=['test', 'dev', 'prod'],
        help='Configuration mode (default: test)'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=DEFAULT_TOP_K,
        help=f'Top K for final retrieval results after reranking (default: {DEFAULT_TOP_K})'
    )
    
    parser.add_argument(
        '--top-k-retrieval',
        type=int,
        default=DEFAULT_TOP_K_RETRIEVAL,
        help=f'Top K for each retrieval method before fusion (default: {DEFAULT_TOP_K_RETRIEVAL})'
    )
    
    parser.add_argument(
        '--rrf-k',
        type=int,
        default=DEFAULT_RRF_K,
        help=f'RRF k parameter (default: {DEFAULT_RRF_K})'
    )
    
    parser.add_argument(
        '--rerank-top-k',
        type=int,
        default=DEFAULT_RERANK_TOP_K,
        help=f'Top K results to keep after reranking (default: {DEFAULT_RERANK_TOP_K})'
    )
    
    parser.add_argument(
        '--rerank-batch-size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size for reranker inference (default: {DEFAULT_BATCH_SIZE})'
    )
    
    parser.add_argument(
        '--bm25-weight',
        type=float,
        default=DEFAULT_BM25_WEIGHT,
        help=f'BM25 weight in RRF fusion (default: {DEFAULT_BM25_WEIGHT})'
    )
    
    parser.add_argument(
        '--vector-weight',
        type=float,
        default=DEFAULT_VECTOR_WEIGHT,
        help=f'Vector weight in RRF fusion (default: {DEFAULT_VECTOR_WEIGHT})'
    )
    
    parser.add_argument(
        '--show-gpu',
        action='store_true',
        help='Show GPU configuration and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/reranked_hybrid_rag',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for version management (default: reranked_hybrid_rag)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


# =============================================================================
# Retrieval Functions
# =============================================================================

def hybrid_retrieve_with_rerank(
    query: str,
    file_name: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    rerank_batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval with reranker (Vector + BM25 + RRF + Reranker).
    
    This function performs:
    1. Hybrid retrieval using BM25 and vector search with RRF fusion
    2. Reranker-based re-ranking of the hybrid results
    
    Args:
        query: Query text.
        file_name: Filter by file name (optional).
        top_k: Final number of results to return.
        top_k_retrieval: Number of results per retrieval method before fusion.
        rrf_k: RRF fusion parameter.
        rerank_top_k: Number of top results to keep after reranking.
        rerank_batch_size: Batch size for reranker inference.
    
    Returns:
        List of reranked retrieval results.
    """
    # Step 1: Hybrid retrieval (BM25 + Vector + RRF)
    # Get more candidates than needed for reranking
    hybrid_top_k = max(top_k_retrieval, rerank_top_k * 2)
    hybrid_results = retrieve_top_k_hybrid(
        query=query,
        top_k_shown=hybrid_top_k,
        file_name=file_name,
        top_k_retrieval=top_k_retrieval,
        rrf_k=rrf_k,
    )
    
    if not hybrid_results:
        return []
    
    # Step 2: Rerank the hybrid results
    reranked_results = rerank_results(
        query=query,
        candidate_chunks=hybrid_results,
        top_k=rerank_top_k,
        batch_size=rerank_batch_size,
    )
    
    if not reranked_results:
        return []
    
    # Step 3: Format output with rank information
    retrieved = []
    for i, r in enumerate(reranked_results[:top_k]):
        retrieved.append({
            'chunk_id': r.get('chunk_id', ''),
            'clause_text': r.get('clause_text', ''),
            'file_name': r.get('file_name', ''),
            'clause_type': r.get('clause_type', ''),
            'rerank_score': r.get('rerank_score', 0.0),
            'rank': i + 1,
        })
    
    return retrieved


def reranked_hybrid_generate(query: str, contexts: List[str]) -> str:
    """
    Generate answer using LLM based on reranked contexts.
    
    Args:
        query: Query text.
        contexts: Retrieved context texts.
    
    Returns:
        Generated answer string.
    """
    if not contexts:
        return "No relevant context found."
    
    # Use top 5 contexts
    context_str = "\n\n".join(contexts[:5])
    
    # Construct prompt
    prompt = f"""Based on the following contract clauses, answer the question concisely.

Contract clauses:
{context_str}

Question: {query}

Answer:"""
    
    return llm_generate(prompt)


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def run_reranked_hybrid_rag_evaluation(
    gold_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    top_k_retrieval: int = DEFAULT_TOP_K_RETRIEVAL,
    rrf_k: int = DEFAULT_RRF_K,
    rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    rerank_batch_size: int = DEFAULT_BATCH_SIZE,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    output_dir: str = 'results/reranked_hybrid_rag',
    experiment_name: Optional[str] = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run Reranked Hybrid RAG evaluation.
    
    Args:
        gold_df: Gold standard DataFrame.
        top_k: Final top K results to return.
        top_k_retrieval: Top K for each retrieval method before fusion.
        rrf_k: RRF parameter.
        rerank_top_k: Top K after reranking.
        rerank_batch_size: Batch size for reranker.
        bm25_weight: BM25 weight in RRF.
        vector_weight: Vector search weight in RRF.
        output_dir: Output directory for results.
        experiment_name: Experiment name.
        generate_plots: Whether to generate visualization plots.
    
    Returns:
        Evaluation summary dictionary.
    
    Raises:
        FileNotFoundError: If BM25 index is not found.
    """
    from tqdm import tqdm
    import json
    import ast
    
    # Generate experiment name
    if experiment_name is None:
        experiment_name = "reranked_hybrid_rag"
    
    logger.info("=" * 60)
    logger.info("Starting Reranked Hybrid RAG Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Total samples: {len(gold_df)}")
    logger.info(f"  Top K (final): {top_k}")
    logger.info(f"  Top K (retrieval): {top_k_retrieval}")
    logger.info(f"  RRF k: {rrf_k}")
    logger.info(f"  Rerank Top K: {rerank_top_k}")
    logger.info(f"  Rerank Batch Size: {rerank_batch_size}")
    logger.info(f"  BM25 weight: {bm25_weight}")
    logger.info(f"  Vector weight: {vector_weight}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Generate plots: {generate_plots}")
    logger.info("=" * 60)
    
    # Preload models and indexes
    logger.info("Loading embedding model...")
    get_model()
    logger.success("Embedding model loaded")
    
    logger.info("Loading BM25 index...")
    try:
        load_bm25_index()
        logger.success("BM25 index loaded")
    except FileNotFoundError as e:
        logger.error(f"BM25 index not found: {e}")
        logger.error("Please run: python -m src.rag.embedding --mode prod")
        raise
    
    logger.info("Loading reranker model...")
    get_reranker_model()
    logger.success("Reranker model loaded")
    
    # Prepare evaluation data
    eval_data = []
    
    for idx, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Evaluating"):
        start_time = time.time()
        
        query = row.get('query', '')
        file_name = row.get('file_name', '')
        gold_answer = row.get('gold_answer_text', '')
        
        # Parse gold_chunk_ids
        gold_ids = row.get('gold_chunk_ids', [])
        if isinstance(gold_ids, str):
            try:
                gold_ids = json.loads(gold_ids)
            except json.JSONDecodeError:
                try:
                    gold_ids = ast.literal_eval(gold_ids)
                except (ValueError, SyntaxError):
                    gold_ids = [gold_ids]
        
        # Hybrid retrieval with reranking
        retrieval_start = time.time()
        retrieved = hybrid_retrieve_with_rerank(
            query=query,
            file_name=file_name,
            top_k=top_k,
            top_k_retrieval=top_k_retrieval,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
        )
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        # Extract results
        retrieved_ids = [r['chunk_id'] for r in retrieved]
        retrieved_contexts = [r['clause_text'] for r in retrieved]
        rerank_scores = [r.get('rerank_score', 0.0) for r in retrieved]
        
        # Generate answer
        generation_start = time.time()
        pred_answer = reranked_hybrid_generate(query, retrieved_contexts)
        generation_time_ms = (time.time() - generation_start) * 1000
        
        total_time_ms = (time.time() - start_time) * 1000
        
        eval_data.append({
            'query': query,
            'file_name': file_name,
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'retrieved_ids': retrieved_ids,
            'gold_ids': [str(gid) for gid in gold_ids],
            'retrieved_contexts': retrieved_contexts,
            'rerank_scores': rerank_scores,
            'answer_type': row.get('answer_type', 'text'),
            'clause_type': row.get('clause_type', ''),
            # Time metrics
            'retrieval_time_ms': retrieval_time_ms,
            'generation_time_ms': generation_time_ms,
            'total_time_ms': total_time_ms,
        })
    
    # Run evaluation
    logger.info("\nCalculating metrics...")
    
    eval_config = EvaluationConfig(
        k_values=[1, 3, 5, 10],
        ragas_enabled=False,  # Disable full RAGAS, use simplified version
        wandb_enabled=False,
        output_dir=output_dir,
        # Plot configuration
        plot_enabled=generate_plots,
        plot_include_performance=True,
        plot_include_quality=True,
        plot_include_correlation=False,
    )
    
    # Create evaluation pipeline
    pipeline = EvaluationPipeline(
        config=eval_config,
        experiment_name=experiment_name,
    )
    
    # Batch evaluation
    pipeline.evaluate_batch(eval_data)
    
    # RAGAS evaluation (requires LLM)
    if eval_config.ragas_enabled:
        try:
            logger.info("Running RAGAS evaluation...")
            pipeline.evaluate_with_ragas()
        except Exception as e:
            logger.warning(f"RAGAS evaluation skipped: {e}")
    
    # Print summary
    pipeline.print_summary()
    
    # Save results (auto-generate CSV + JSON + MD + plots)
    output_files = pipeline.save_results()
    
    return pipeline.get_summary()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> Optional[Dict[str, Any]]:
    """Main function."""
    args = parse_args()
    
    # Show GPU info and exit
    if args.show_gpu:
        show_gpu_info()
        return None
    
    # Initialize configuration
    logger.info("Initializing configuration...")
    config = init_config(mode=args.mode)
    project_root = config.app.project_root
    
    # Set global random seed (ensure reproducibility)
    seed = set_global_seed()
    logger.info(f"Global random seed set to: {seed}")
    
    # Initialize GPU manager
    logger.info("Initializing GPU manager...")
    gpu_config = init_gpu_manager(project_root=project_root)
    show_gpu_info()
    
    # Load gold standard data
    logger.info("Loading gold standard data...")
    gold_df = load_gold_answers()
    logger.info(f"Loaded {len(gold_df)} gold answer samples")
    
    if args.max_samples and args.max_samples < len(gold_df):
        gold_df = gold_df.sample(n=args.max_samples, random_state=get_seed())
        logger.info(f"Sampled {len(gold_df)} samples for testing")
    
    # Run evaluation
    try:
        summary = run_reranked_hybrid_rag_evaluation(
            gold_df,
            top_k=args.top_k,
            top_k_retrieval=args.top_k_retrieval,
            rrf_k=args.rrf_k,
            rerank_top_k=args.rerank_top_k,
            rerank_batch_size=args.rerank_batch_size,
            bm25_weight=args.bm25_weight,
            vector_weight=args.vector_weight,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            generate_plots=not args.no_plots,
        )
        
        logger.success("\n" + "=" * 60)
        logger.success("Reranked Hybrid RAG Evaluation Complete!")
        logger.success("=" * 60)
        
        return summary
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
        return None
    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        raise
    finally:
        # Cleanup resources
        from src.utils.model_loading import release_all_models
        logger.info("Cleaning up resources...")
        release_all_models()


if __name__ == "__main__":
    main()
