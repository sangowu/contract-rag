"""
CUAD 评估模块

核心组件:
- 传统指标: Hit@K, MRR, Recall, F1, Exact Match
- RAGAS 指标: Faithfulness, Answer Relevancy, Context Precision/Recall
- WandB 追踪: 实验记录和对比
- 自动可视化: 图表自动生成

使用方式:
    from src.evaluation import EvaluationPipeline, evaluate_rag_results, quick_evaluate
    
    # 使用评估管道（带版本管理）
    pipeline = EvaluationPipeline(
        experiment_name="baseline",
        experiment_version="v1",
    )
    results = pipeline.evaluate_batch(data)
    pipeline.save_results()  # 自动生成 CSV + JSON + 图表
    
    # 便捷函数
    summary = evaluate_rag_results(
        results,
        experiment_name="my_test",
        plot_enabled=True,
    )
    
    # 快速评估
    pipeline = quick_evaluate(results, name="quick_test")
"""

# 核心指标
from src.evaluation.metrics import (
    RetrievalMetrics,
    AnswerMetrics,
    BatchEvaluator,
    LatencyMetrics,
)

# 评估管道
from src.evaluation.pipeline import (
    EvaluationPipeline,
    EvaluationConfig,
    evaluate_rag_results,
    quick_evaluate,
)

# WandB 追踪
from src.evaluation.wandb_tracker import (
    WandBTracker,
    WandBConfig,
    init_wandb,
    get_tracker,
    log_metrics,
    finish_wandb,
)

# RAGAS 评估 (简化版总是可用)
from src.evaluation.ragas_evaluator import (
    SimplifiedRAGASMetrics,
    evaluate_simplified,
    RAGAS_AVAILABLE,
)

# RAGAS 完整版 (需要 ragas 库)
if RAGAS_AVAILABLE:
    from src.evaluation.ragas_evaluator import (
        RAGASEvaluator,
        RAGASConfig,
        evaluate_with_ragas,
    )

__all__ = [
    # 指标
    'RetrievalMetrics',
    'AnswerMetrics',
    'BatchEvaluator',
    'LatencyMetrics',
    # 管道
    'EvaluationPipeline',
    'EvaluationConfig',
    'evaluate_rag_results',
    'quick_evaluate',
    # WandB
    'WandBTracker',
    'WandBConfig',
    'init_wandb',
    'get_tracker',
    'log_metrics',
    'finish_wandb',
    # RAGAS
    'RAGAS_AVAILABLE',
]

if RAGAS_AVAILABLE:
    __all__.extend([
        'RAGASEvaluator',
        'RAGASConfig',
        'SimplifiedRAGASMetrics',
        'evaluate_with_ragas',
        'evaluate_simplified',
    ])
