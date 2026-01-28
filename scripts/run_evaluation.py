#!/usr/bin/env python
"""
自动化评估脚本

功能:
- 加载金标准数据
- 调用 RAG API 进行检索和生成
- 计算传统指标和 RAGAS 指标
- 记录到 WandB
- 保存结果

使用方式:
    # 基本评估
    python scripts/run_evaluation.py --mode test
    
    # 启用 WandB
    python scripts/run_evaluation.py --mode test --wandb
    
    # 指定样本数
    python scripts/run_evaluation.py --mode test --max-samples 100
    
    # 启用 RAGAS (需要 LLM)
    python scripts/run_evaluation.py --mode test --ragas
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
import requests
from loguru import logger

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import init_config, get_config
from src.evaluation import (
    EvaluationPipeline,
    EvaluationConfig,
)
from src.utils.seed_utils import set_global_seed


# =============================================================================
# 配置
# =============================================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Script")
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='test',
                        choices=['test', 'dev', 'prod'],
                        help='Environment mode')
    parser.add_argument('--api-url', type=str, default=DEFAULT_API_URL,
                        help='RAG API base URL')
    
    # 评估参数
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Top K for retrieval')
    parser.add_argument('--k-values', type=str, default='1,3,5,10,20',
                        help='K values for metrics (comma-separated)')
    
    # 功能开关
    parser.add_argument('--wandb', action='store_true',
                        help='Enable WandB tracking')
    parser.add_argument('--wandb-project', type=str, default='cuad-assistant',
                        help='WandB project name')
    parser.add_argument('--ragas', action='store_true',
                        help='Enable RAGAS evaluation (requires LLM)')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip LLM generation, only evaluate retrieval')
    
    # 输出参数
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Experiment name (e.g., "baseline", "with_rerank")')
    parser.add_argument('--experiment-version', type=str, default=None,
                        help='Experiment version (e.g., "v1", "v2")')
    
    # 绘图参数
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable automatic plot generation')
    
    return parser.parse_args()


# =============================================================================
# 数据加载
# =============================================================================

def load_gold_data(config, max_samples: Optional[int] = None) -> pd.DataFrame:
    """加载金标准数据"""
    from src.data.data_loader import load_gold_answers
    
    df = load_gold_answers()
    logger.info(f"Loaded {len(df)} gold answer samples")
    
    if max_samples and max_samples < len(df):
        from src.utils.seed_utils import get_seed
        df = df.sample(n=max_samples, random_state=get_seed())
        logger.info(f"Sampled {len(df)} samples for evaluation")
    
    return df


# =============================================================================
# API 调用 (带时间记录)
# =============================================================================

import time as time_module


def call_retrieval_api(
    query: str,
    file_name: str,
    top_k: int = 10,
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """
    调用检索 API
    
    Returns:
        包含结果和时间的字典:
        {
            "data": [...],
            "retrieval_time_ms": 125.3,
            "rerank_time_ms": 45.2,  # 如果 API 返回
        }
    """
    start_time = time_module.time()
    
    try:
        response = requests.post(
            f"{api_url}/api/retrieval/search",
            json={
                "query": query,
                "top_k_retrieval": top_k,
                "file_name": file_name,
            },
            timeout=300,
        )
        
        elapsed_ms = (time_module.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok", True):
            logger.warning(f"API error: {data.get('error', 'Unknown')}")
            return {"data": [], "retrieval_time_ms": elapsed_ms}
        
        # 使用 API 返回的时间，如果没有则使用客户端测量的时间
        api_time = data.get("retrieval_time_ms", elapsed_ms)
        
        return {
            "data": data.get("data", []),
            "retrieval_time_ms": api_time,
        }
    
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time_module.time() - start_time) * 1000
        logger.error(f"API request failed: {e}")
        return {"data": [], "retrieval_time_ms": elapsed_ms, "error": str(e)}


def call_generation_api(
    query: str,
    contexts: List[str],
    api_url: str = DEFAULT_API_URL,
) -> Dict[str, Any]:
    """
    调用生成 API
    
    Returns:
        包含结果和时间的字典:
        {
            "answer": "...",
            "generation_time_ms": 890.5,
            "tokens_used": 128,  # 如果 API 返回
        }
    """
    start_time = time_module.time()
    
    try:
        response = requests.post(
            f"{api_url}/api/generation/generate",
            json={
                "query": query,
                "contexts": contexts,
            },
            timeout=300,
        )
        
        elapsed_ms = (time_module.time() - start_time) * 1000
        
        response.raise_for_status()
        data = response.json()
        
        # 使用 API 返回的时间，如果没有则使用客户端测量的时间
        api_time = data.get("generation_time_ms", elapsed_ms)
        
        return {
            "answer": data.get("answer", ""),
            "generation_time_ms": api_time,
            "tokens_used": data.get("tokens_used", 0),
        }
    
    except requests.exceptions.RequestException as e:
        elapsed_ms = (time_module.time() - start_time) * 1000
        logger.error(f"Generation API failed: {e}")
        return {"answer": "", "generation_time_ms": elapsed_ms, "error": str(e)}


# =============================================================================
# 评估流程
# =============================================================================

def prepare_evaluation_data(
    gold_df: pd.DataFrame,
    api_url: str,
    top_k: int = 10,
    skip_generation: bool = False,
) -> List[Dict[str, Any]]:
    """准备评估数据 (带时间记录)"""
    from tqdm import tqdm
    import ast
    
    eval_data = []
    
    for _, row in tqdm(gold_df.iterrows(), total=len(gold_df), desc="Preparing data"):
        sample_start = time_module.time()
        
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
        
        # 调用检索 API (带时间)
        retrieval_result = call_retrieval_api(query, file_name, top_k, api_url)
        retrieved = retrieval_result.get("data", [])
        retrieval_time_ms = retrieval_result.get("retrieval_time_ms", 0)
        
        # 提取检索结果
        retrieved_ids = [
            str(r.get("parent_id", r.get("chunk_id", "")))
            for r in retrieved
        ]
        retrieved_contexts = [
            r.get("clause_text", "")
            for r in retrieved
        ]
        
        # 生成答案 (带时间)
        pred_answer = ""
        generation_time_ms = 0
        tokens_used = 0
        
        if not skip_generation and retrieved_contexts:
            generation_result = call_generation_api(query, retrieved_contexts, api_url)
            pred_answer = generation_result.get("answer", "")
            generation_time_ms = generation_result.get("generation_time_ms", 0)
            tokens_used = generation_result.get("tokens_used", 0)
        
        # 计算端到端时间
        total_time_ms = (time_module.time() - sample_start) * 1000
        
        # 推断答案类型
        answer_type = row.get('answer_type', 'text')
        
        eval_data.append({
            'query': query,
            'file_name': file_name,
            'gold_answer': gold_answer,
            'pred_answer': pred_answer,
            'retrieved_ids': retrieved_ids,
            'gold_ids': [str(gid) for gid in gold_ids],
            'retrieved_contexts': retrieved_contexts,
            'answer_type': answer_type,
            'clause_type': row.get('clause_type', ''),
            # 时间指标
            'retrieval_time_ms': retrieval_time_ms,
            'generation_time_ms': generation_time_ms,
            'total_time_ms': total_time_ms,
            'tokens_used': tokens_used,
        })
    
    return eval_data


def run_evaluation(
    eval_data: List[Dict[str, Any]],
    config: EvaluationConfig,
    experiment_name: str = None,
    experiment_version: str = None,
    run_ragas: bool = False,
) -> Dict[str, float]:
    """
    运行评估
    
    Args:
        eval_data: 评估数据
        config: 评估配置
        experiment_name: 实验名称
        experiment_version: 实验版本
        run_ragas: 是否运行 RAGAS 评估
    
    Returns:
        评估摘要
    """
    # 创建评估管道（带版本管理）
    pipeline = EvaluationPipeline(
        config=config,
        experiment_name=experiment_name,
        experiment_version=experiment_version,
    )
    
    # 初始化 WandB
    if config.wandb_enabled:
        pipeline.init_wandb(
            experiment_name=experiment_name,
            config_dict={
                'num_samples': len(eval_data),
                'k_values': config.k_values,
                'ragas_enabled': config.ragas_enabled,
            }
        )
    
    # 批量评估
    logger.info(f"Evaluating {len(eval_data)} samples...")
    pipeline.evaluate_batch(eval_data)
    
    # RAGAS 评估
    if run_ragas and config.ragas_enabled:
        logger.info("Running RAGAS evaluation...")
        try:
            pipeline.evaluate_with_ragas()
        except Exception as e:
            logger.warning(f"RAGAS evaluation skipped: {e}")
    
    # 打印摘要
    pipeline.print_summary()
    
    # 保存结果（自动生成 CSV + JSON + 图表）
    pipeline.save_results()
    
    # 结束
    pipeline.finish()
    
    return pipeline.get_summary()


# =============================================================================
# 主函数
# =============================================================================

def main():
    args = parse_args()
    
    # 初始化配置
    config = init_config(mode=args.mode)
    logger.info(f"Initialized config with mode: {args.mode}")
    
    # 设置全局随机种子（确保实验可复现）
    seed = set_global_seed()
    logger.info(f"Global random seed set to: {seed}")
    
    # 解析 K 值
    k_values = [int(k) for k in args.k_values.split(',')]
    
    # 创建评估配置（含绘图配置）
    eval_config = EvaluationConfig(
        k_values=k_values,
        ragas_enabled=args.ragas,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        output_dir=args.output_dir,
        # 绘图配置
        plot_enabled=not getattr(args, 'no_plots', False),
        plot_include_performance=True,
        plot_include_quality=True,
    )
    
    # 加载金标准数据
    gold_df = load_gold_data(config, max_samples=args.max_samples)
    
    # 准备评估数据
    logger.info("Preparing evaluation data...")
    eval_data = prepare_evaluation_data(
        gold_df,
        api_url=args.api_url,
        top_k=args.top_k,
        skip_generation=args.skip_generation,
    )
    
    # 运行评估
    experiment_name = args.experiment_name or f"eval_{args.mode}"
    summary = run_evaluation(
        eval_data,
        eval_config,
        experiment_name=experiment_name,
        experiment_version=getattr(args, 'experiment_version', None),
        run_ragas=args.ragas,
    )
    
    logger.success("Evaluation complete!")
    return summary


if __name__ == "__main__":
    main()
