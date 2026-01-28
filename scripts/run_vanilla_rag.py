#!/usr/bin/env python
"""
Vanilla RAG 端到端测试脚本

功能:
- 最基础的 RAG 流程（无增强）
- 仅使用向量检索（无 BM25、无 Rerank）
- 评估初始 benchmark

使用方式:
    # 完整测试
    python scripts/run_vanilla_rag.py --mode test
    
    # 仅测试前 10 个样本
    python scripts/run_vanilla_rag.py --mode test --max-samples 10
    
    # 查看 GPU 配置
    python scripts/run_vanilla_rag.py --show-gpu
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any
from pathlib import Path

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import pandas as pd

# 核心模块
from src.core.config import init_config, get_config
from src.core.gpu_manager import init_gpu_manager, get_gpu_config
from src.data.data_loader import load_gold_answers
from src.rag.embedding import get_model
from src.rag.retrieval import retrieve_top_k
from src.inference.llm_inference import llm_generate
from src.utils.seed_utils import set_global_seed, get_seed
from src.evaluation import EvaluationPipeline, EvaluationConfig


def show_gpu_info():
    """显示 GPU 配置信息"""
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Vanilla RAG End-to-End Test",
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
        default=10,
        help='Top K for retrieval (default: 10)'
    )
    
    parser.add_argument(
        '--show-gpu',
        action='store_true',
        help='Show GPU configuration and exit'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/vanilla_rag',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name for version management (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Disable plot generation'
    )
    
    return parser.parse_args()


def vanilla_retrieve(
    query: str,
    file_name: str = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Vanilla 检索（仅向量检索）
    
    Args:
        query: 查询文本
        file_name: 文件名过滤
        top_k: 返回结果数量
    
    Returns:
        检索结果列表
    """
    # 使用 retrieval 模块的向量检索
    results = retrieve_top_k(
        query=query,
        top_k_shown=top_k,
        file_name=file_name,
        top_k_retrieval=top_k,
    )
    
    if not results:
        return []
    
    # 转换格式（添加 rank）
    retrieved = []
    for i, r in enumerate(results):
        retrieved.append({
            'chunk_id': r.get('chunk_id', ''),
            'clause_text': r.get('clause_text', ''),
            'file_name': r.get('file_name', ''),
            'score': 1.0,  # retrieve_top_k 不返回分数
            'rank': i + 1,
        })
    
    return retrieved


def vanilla_generate(query: str, contexts: List[str]) -> str:
    """
    Vanilla 生成（基础 prompt）
    
    Args:
        query: 查询文本
        contexts: 检索到的上下文
    
    Returns:
        生成的答案
    """
    if not contexts:
        return "No relevant context found."
    
    # 合并上下文
    context_str = "\n\n".join(contexts[:5])  # 只用前5个
    
    # 简单的 prompt
    prompt = f"""Based on the following contract clauses, answer the question concisely.

Contract clauses:
{context_str}

Question: {query}

Answer:"""
    
    return llm_generate(prompt)


def run_vanilla_rag_evaluation(
    gold_df: pd.DataFrame,
    top_k: int = 10,
    output_dir: str = 'results/vanilla_rag',
    experiment_name: str = None,
    generate_plots: bool = True,
) -> Dict[str, Any]:
    """
    运行 Vanilla RAG 评估
    
    Args:
        gold_df: 金标准数据
        top_k: 检索 top k
        output_dir: 输出目录
        experiment_name: 实验名称（用于图表版本管理）
        generate_plots: 是否生成可视化图表
    
    Returns:
        评估结果摘要
    """
    from tqdm import tqdm
    import json
    import ast
    
    # 生成实验名称（不带时间戳，便于覆盖更新）
    if experiment_name is None:
        experiment_name = "vanilla_rag"
    
    logger.info("=" * 60)
    logger.info("Starting Vanilla RAG Evaluation")
    logger.info("=" * 60)
    logger.info(f"  Experiment: {experiment_name}")
    logger.info(f"  Total samples: {len(gold_df)}")
    logger.info(f"  Top K: {top_k}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Generate plots: {generate_plots}")
    logger.info("=" * 60)
    
    # 预加载模型
    logger.info("Loading embedding model...")
    get_model()
    logger.success("Embedding model loaded")
    
    # 准备评估数据
    eval_data = []
    
    for idx, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Evaluating"):
        start_time = time.time()
        
        query = row.get('query', '')
        file_name = row.get('file_name', '')
        gold_answer = row.get('gold_answer_text', '')
        
        # 解析 gold_chunk_ids
        gold_ids = row.get('gold_chunk_ids', [])
        if isinstance(gold_ids, str):
            try:
                gold_ids = json.loads(gold_ids)
            except:
                try:
                    gold_ids = ast.literal_eval(gold_ids)
                except:
                    gold_ids = [gold_ids]
        
        # Vanilla 检索
        retrieval_start = time.time()
        retrieved = vanilla_retrieve(query, file_name, top_k)
        retrieval_time_ms = (time.time() - retrieval_start) * 1000
        
        # 提取结果
        retrieved_ids = [r['chunk_id'] for r in retrieved]
        retrieved_contexts = [r['clause_text'] for r in retrieved]
        
        # Vanilla 生成
        generation_start = time.time()
        pred_answer = vanilla_generate(query, retrieved_contexts)
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
            'answer_type': row.get('answer_type', 'text'),
            'clause_type': row.get('clause_type', ''),
            # 时间指标
            'retrieval_time_ms': retrieval_time_ms,
            'generation_time_ms': generation_time_ms,
            'total_time_ms': total_time_ms,
        })
    
    # 运行评估（使用内置版本管理和自动绘图）
    logger.info("\nCalculating metrics...")
    
    eval_config = EvaluationConfig(
        k_values=[1, 3, 5, 10],
        ragas_enabled=False,  # 禁用完整版 RAGAS，使用简化版
        wandb_enabled=False,
        output_dir=output_dir,
        # 绘图配置
        plot_enabled=generate_plots,
        plot_include_performance=True,
        plot_include_quality=True,
        plot_include_correlation=False,
    )
    
    # 创建评估管道
    pipeline = EvaluationPipeline(
        config=eval_config,
        experiment_name=experiment_name,
    )
    
    # 批量评估
    pipeline.evaluate_batch(eval_data)
    
    # RAGAS 评估（需要 LLM）
    if eval_config.ragas_enabled:
        try:
            logger.info("Running RAGAS evaluation...")
            pipeline.evaluate_with_ragas()
        except Exception as e:
            logger.warning(f"RAGAS evaluation skipped: {e}")
    
    # 打印摘要
    pipeline.print_summary()
    
    # 保存结果（自动生成 CSV + JSON + MD + 图表）
    output_files = pipeline.save_results()
    
    return pipeline.get_summary()


def main():
    """主函数"""
    args = parse_args()
    
    # 显示 GPU 信息并退出
    if args.show_gpu:
        show_gpu_info()
        return
    
    # 初始化配置
    logger.info("Initializing configuration...")
    config = init_config(mode=args.mode)
    project_root = config.app.project_root
    
    # 设置全局随机种子（确保实验可复现）
    seed = set_global_seed()
    logger.info(f"Global random seed set to: {seed}")
    
    # 初始化 GPU 管理器
    logger.info("Initializing GPU manager...")
    gpu_config = init_gpu_manager(project_root=project_root)
    show_gpu_info()
    
    # 加载金标准数据
    logger.info("Loading gold standard data...")
    gold_df = load_gold_answers()
    logger.info(f"Loaded {len(gold_df)} gold answer samples")
    
    if args.max_samples and args.max_samples < len(gold_df):
        gold_df = gold_df.sample(n=args.max_samples, random_state=get_seed())
        logger.info(f"Sampled {len(gold_df)} samples for testing")
    
    # 运行评估
    try:
        summary = run_vanilla_rag_evaluation(
            gold_df,
            top_k=args.top_k,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            generate_plots=not args.no_plots,
        )
        
        logger.success("\n" + "=" * 60)
        logger.success("Vanilla RAG Evaluation Complete!")
        logger.success("=" * 60)
        
        return summary
        
    except KeyboardInterrupt:
        logger.warning("\nEvaluation interrupted by user")
    except Exception as e:
        logger.error(f"\nEvaluation failed: {e}")
        raise
    finally:
        # 清理资源
        from src.utils.model_loading import release_all_models
        logger.info("Cleaning up resources...")
        release_all_models()


if __name__ == "__main__":
    main()
