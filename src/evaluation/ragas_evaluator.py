"""
RAGAS 评估器

RAGAS (Retrieval Augmented Generation Assessment) 提供以下指标:
- Faithfulness: 答案是否基于检索的上下文
- Answer Relevancy: 答案与问题的相关性
- Context Precision: 上下文的精确性
- Context Recall: 上下文的召回率
- Context Relevancy: 上下文与问题的相关性

使用方式:
    from src.evaluation.ragas_evaluator import RAGASEvaluator
    
    evaluator = RAGASEvaluator()
    results = evaluator.evaluate(dataset)
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

# RAGAS 导入
RAGAS_AVAILABLE = False
ragas_evaluate = None
faithfulness = None
answer_relevancy = None
context_precision = None
context_recall = None
Dataset = None

try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    try:
        from datasets import Dataset
    except ImportError:
        Dataset = None
    RAGAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAGAS not available: {e}")


@dataclass
class RAGASConfig:
    """RAGAS 配置"""
    
    metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])
    
    llm_model: str = ""  # 用于评估的 LLM 模型
    embeddings_model: str = ""  # 用于评估的 Embedding 模型
    
    batch_size: int = 10
    timeout: int = 300
    
    @classmethod
    def from_config(cls, config) -> 'RAGASConfig':
        """从全局配置创建"""
        try:
            ragas_cfg = config.evaluation.ragas
            return cls(
                metrics=getattr(ragas_cfg, 'metrics', cls.metrics),
                llm_model=getattr(ragas_cfg, 'llm_model', ''),
                embeddings_model=getattr(ragas_cfg, 'embeddings_model', ''),
            )
        except Exception:
            return cls()


class RAGASEvaluator:
    """
    RAGAS 评估器
    
    用于评估 RAG 系统的质量，包括:
    - Faithfulness: 生成的答案是否忠于检索的上下文
    - Answer Relevancy: 答案是否回答了问题
    - Context Precision: 检索的上下文是否精确
    - Context Recall: 检索的上下文是否覆盖了答案
    """
    
    def __init__(self, config: Optional[RAGASConfig] = None):
        """
        初始化 RAGAS 评估器
        
        Args:
            config: RAGAS 配置
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("RAGAS is not installed. Please install with: pip install ragas")
        
        self.config = config or RAGASConfig()
        self._metrics = self._get_metrics()
    
    def _get_metrics(self) -> List:
        """获取评估指标"""
        metric_mapping = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        
        metrics = []
        for name in self.config.metrics:
            if name in metric_mapping and metric_mapping[name] is not None:
                metrics.append(metric_mapping[name])
            elif name not in metric_mapping:
                logger.warning(f"Unknown RAGAS metric: {name}")
        
        return metrics
    
    def prepare_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str] = None,
    ):
        """
        准备 RAGAS 评估数据集
        
        Args:
            questions: 问题列表
            answers: 模型答案列表
            contexts: 检索上下文列表 (每个问题对应多个上下文)
            ground_truths: 金标准答案列表 (可选)
        
        Returns:
            HuggingFace Dataset
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        
        if ground_truths:
            data["ground_truth"] = ground_truths
        
        return Dataset.from_dict(data)
    
    def evaluate(
        self,
        dataset: Dataset = None,
        questions: List[str] = None,
        answers: List[str] = None,
        contexts: List[List[str]] = None,
        ground_truths: List[str] = None,
        llm = None,
        embeddings = None,
    ) -> Dict[str, Any]:
        """
        执行 RAGAS 评估
        
        Args:
            dataset: 已准备好的数据集，或者提供以下参数:
            questions: 问题列表
            answers: 模型答案列表
            contexts: 检索上下文列表
            ground_truths: 金标准答案列表
            llm: 用于评估的 LLM (可选)
            embeddings: 用于评估的 Embeddings (可选)
        
        Returns:
            评估结果字典
        """
        # 准备数据集
        if dataset is None:
            if questions is None or answers is None or contexts is None:
                raise ValueError("Must provide dataset or (questions, answers, contexts)")
            dataset = self.prepare_dataset(questions, answers, contexts, ground_truths)
        
        logger.info(f"Starting RAGAS evaluation on {len(dataset)} samples...")
        logger.info(f"Metrics: {[m.name for m in self._metrics]}")
        
        try:
            # 执行评估
            result = ragas_evaluate(
                dataset=dataset,
                metrics=self._metrics,
                llm=llm,
                embeddings=embeddings,
            )
            
            # 转换结果
            scores = {}
            for metric in self._metrics:
                if hasattr(result, metric.name):
                    scores[metric.name] = getattr(result, metric.name)
                elif metric.name in result:
                    scores[metric.name] = result[metric.name]
            
            # 添加样本级别的结果
            if hasattr(result, 'to_pandas'):
                scores['_detailed'] = result.to_pandas().to_dict('records')
            
            logger.success(f"RAGAS evaluation complete: {scores}")
            return scores
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise
    
    def evaluate_from_results(
        self,
        results: List[Dict[str, Any]],
        question_key: str = "query",
        answer_key: str = "model_answer",
        context_key: str = "retrieved_contexts",
        ground_truth_key: str = "gold_answer",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        从评估结果列表进行 RAGAS 评估
        
        Args:
            results: 评估结果列表
            question_key: 问题字段名
            answer_key: 答案字段名
            context_key: 上下文字段名
            ground_truth_key: 金标准字段名
        
        Returns:
            RAGAS 评估结果
        """
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for r in results:
            questions.append(r.get(question_key, ""))
            answers.append(r.get(answer_key, ""))
            
            # 处理上下文
            ctx = r.get(context_key, [])
            if isinstance(ctx, str):
                ctx = [ctx]
            contexts.append(ctx)
            
            # 金标准
            gt = r.get(ground_truth_key, "")
            ground_truths.append(gt)
        
        return self.evaluate(
            questions=questions,
            answers=answers,
            contexts=contexts,
            ground_truths=ground_truths if any(ground_truths) else None,
            **kwargs,
        )


# =============================================================================
# 简化版 RAGAS 计算 (不依赖 LLM)
# =============================================================================

class SimplifiedRAGASMetrics:
    """
    简化版 RAGAS 指标
    
    不需要 LLM 即可计算的近似指标:
    - Context Coverage: 答案中有多少内容来自上下文
    - Overlap Score: 答案与上下文的词汇重叠度
    """
    
    def __init__(self):
        from src.evaluation.metrics import AnswerMetrics
        self.answer_metrics = AnswerMetrics()
    
    def context_coverage(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        上下文覆盖率: 答案 tokens 中有多少在上下文中出现
        
        近似于 Faithfulness
        """
        answer_tokens = set(self.answer_metrics.tokenize(answer))
        if not answer_tokens:
            return 0.0
        
        context_text = " ".join(contexts)
        context_tokens = set(self.answer_metrics.tokenize(context_text))
        
        if not context_tokens:
            return 0.0
        
        overlap = len(answer_tokens & context_tokens)
        return overlap / len(answer_tokens)
    
    def answer_context_overlap(
        self,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        答案-上下文重叠度
        
        F1 风格的重叠度计算
        """
        answer_tokens = self.answer_metrics.tokenize(answer)
        context_text = " ".join(contexts)
        context_tokens = self.answer_metrics.tokenize(context_text)
        
        if not answer_tokens or not context_tokens:
            return 0.0
        
        # 计算 F1
        return self.answer_metrics.f1_score(context_text, answer)
    
    def ground_truth_coverage(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        上下文对金标准的覆盖率
        
        近似于 Context Recall
        """
        gt_tokens = set(self.answer_metrics.tokenize(ground_truth))
        if not gt_tokens:
            return 0.0
        
        context_text = " ".join(contexts)
        context_tokens = set(self.answer_metrics.tokenize(context_text))
        
        if not context_tokens:
            return 0.0
        
        overlap = len(gt_tokens & context_tokens)
        return overlap / len(gt_tokens)
    
    def compute_all(
        self,
        answer: str,
        contexts: List[str],
        ground_truth: str = None,
    ) -> Dict[str, float]:
        """
        计算所有简化指标
        """
        results = {
            "context_coverage": self.context_coverage(answer, contexts),
            "answer_context_overlap": self.answer_context_overlap(answer, contexts),
        }
        
        if ground_truth:
            results["ground_truth_coverage"] = self.ground_truth_coverage(contexts, ground_truth)
        
        return results


# =============================================================================
# 便捷函数
# =============================================================================

def evaluate_with_ragas(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str] = None,
    metrics: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    便捷的 RAGAS 评估函数
    
    Args:
        questions: 问题列表
        answers: 答案列表
        contexts: 上下文列表
        ground_truths: 金标准列表
        metrics: 评估指标列表
    
    Returns:
        评估结果
    """
    config = RAGASConfig(metrics=metrics) if metrics else None
    evaluator = RAGASEvaluator(config)
    return evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths,
        **kwargs,
    )


def evaluate_simplified(
    answer: str,
    contexts: List[str],
    ground_truth: str = None,
) -> Dict[str, float]:
    """
    简化版 RAGAS 评估 (不需要 LLM)
    """
    evaluator = SimplifiedRAGASMetrics()
    return evaluator.compute_all(answer, contexts, ground_truth)
