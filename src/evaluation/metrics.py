"""
传统评估指标

包括:
- 检索指标: Hit@K, MRR@K, Recall@K, Precision@K, NDCG@K
- 答案指标: F1, Exact Match, Accuracy

使用方式:
    from src.evaluation.metrics import RetrievalMetrics, AnswerMetrics
    
    # 检索指标
    metrics = RetrievalMetrics()
    score = metrics.hit_at_k(retrieved_ids, gold_ids, k=10)
    
    # 答案指标
    metrics = AnswerMetrics()
    score = metrics.f1_score(gold_answer, pred_answer)
"""

import re
import math
from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass, field
from loguru import logger


# =============================================================================
# 检索指标
# =============================================================================

@dataclass
class RetrievalMetrics:
    """检索评估指标"""
    
    def hit_at_k(self, retrieved: List[str], gold: List[str], k: int = 10) -> int:
        """
        Hit@K: 检查 top-k 中是否至少有一个相关文档
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k: 前 k 个结果
        
        Returns:
            1 if hit, 0 otherwise
        """
        top_k = set(retrieved[:k])
        gold_set = set(gold)
        return int(len(top_k & gold_set) > 0)
    
    def mrr_at_k(self, retrieved: List[str], gold: List[str], k: int = 10) -> float:
        """
        MRR@K: Mean Reciprocal Rank
        返回第一个相关文档的倒数排名
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k: 前 k 个结果
        
        Returns:
            1/rank of first relevant doc, 0 if not found
        """
        gold_set = set(gold)
        for rank, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in gold_set:
                return 1.0 / rank
        return 0.0
    
    def recall_at_k(self, retrieved: List[str], gold: List[str], k: int = 10) -> float:
        """
        Recall@K: 召回率
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k: 前 k 个结果
        
        Returns:
            Ratio of relevant docs found in top-k
        """
        if not gold:
            return 0.0
        top_k = set(retrieved[:k])
        gold_set = set(gold)
        hits = len(top_k & gold_set)
        return hits / len(gold_set)
    
    def precision_at_k(self, retrieved: List[str], gold: List[str], k: int = 10) -> float:
        """
        Precision@K: 精确率
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k: 前 k 个结果
        
        Returns:
            Ratio of relevant docs in top-k
        """
        if k == 0:
            return 0.0
        top_k = set(retrieved[:k])
        gold_set = set(gold)
        hits = len(top_k & gold_set)
        return hits / k
    
    def ndcg_at_k(self, retrieved: List[str], gold: List[str], k: int = 10) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k: 前 k 个结果
        
        Returns:
            NDCG score
        """
        gold_set = set(gold)
        
        # 计算 DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in gold_set:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # 计算 IDCG (理想情况下的 DCG)
        ideal_k = min(len(gold), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def compute_all(
        self,
        retrieved: List[str],
        gold: List[str],
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """
        计算所有检索指标
        
        Args:
            retrieved: 检索到的文档 ID 列表
            gold: 金标准文档 ID 列表
            k_values: K 值列表
        
        Returns:
            包含所有指标的字典
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        results = {}
        for k in k_values:
            results[f'hit@{k}'] = self.hit_at_k(retrieved, gold, k)
            results[f'mrr@{k}'] = self.mrr_at_k(retrieved, gold, k)
            results[f'recall@{k}'] = self.recall_at_k(retrieved, gold, k)
            results[f'precision@{k}'] = self.precision_at_k(retrieved, gold, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, gold, k)
        
        return results


# =============================================================================
# 答案指标
# =============================================================================

@dataclass
class AnswerMetrics:
    """答案评估指标"""
    
    def normalize_text(self, text: str) -> str:
        """标准化文本"""
        if text is None:
            return ""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        normalized = self.normalize_text(text)
        if not normalized:
            return []
        return normalized.split()
    
    def exact_match(self, gold: str, pred: str) -> float:
        """
        Exact Match: 完全匹配
        
        Args:
            gold: 金标准答案
            pred: 预测答案
        
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return float(self.normalize_text(gold) == self.normalize_text(pred))
    
    def f1_score(self, gold: str, pred: str) -> float:
        """
        F1 Score: Token 级别的 F1
        
        Args:
            gold: 金标准答案
            pred: 预测答案
        
        Returns:
            F1 score
        """
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)
        
        if not gold_tokens or not pred_tokens:
            return 0.0
        
        common = Counter(gold_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def precision_score(self, gold: str, pred: str) -> float:
        """
        Precision: Token 级别的精确率
        """
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)
        
        if not pred_tokens:
            return 0.0
        
        common = Counter(gold_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        
        return num_same / len(pred_tokens)
    
    def recall_score(self, gold: str, pred: str) -> float:
        """
        Recall: Token 级别的召回率
        """
        gold_tokens = self.tokenize(gold)
        pred_tokens = self.tokenize(pred)
        
        if not gold_tokens:
            return 0.0
        
        common = Counter(gold_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        
        return num_same / len(gold_tokens)
    
    def bool_accuracy(self, gold: str, pred: str) -> float:
        """
        布尔答案准确率
        """
        gold_bool = self._normalize_bool(gold)
        pred_bool = self._normalize_bool(pred)
        
        if pred_bool is None:
            return 0.0
        
        return float(gold_bool == pred_bool)
    
    def _normalize_bool(self, text: str) -> Optional[str]:
        """标准化布尔值"""
        if text is None:
            return None
        
        t = text.lower().strip()
        
        if any(w in t for w in ["yes", "there is", "it does", "true"]):
            return "yes"
        if any(w in t for w in ["no", "there is no", "does not", "false"]):
            return "no"
        
        return None
    
    def compute_by_type(
        self,
        gold: str,
        pred: str,
        answer_type: str,
    ) -> Dict[str, float]:
        """
        根据答案类型计算指标
        
        Args:
            gold: 金标准答案
            pred: 预测答案
            answer_type: 答案类型 (bool, text, date, etc.)
        
        Returns:
            指标字典
        """
        answer_type = answer_type.lower()
        
        if answer_type == "bool":
            return {
                "accuracy": self.bool_accuracy(gold, pred),
            }
        elif answer_type in ["date", "duration", "location", "entity"]:
            return {
                "exact_match": self.exact_match(gold, pred),
            }
        else:  # text, list, extractive
            return {
                "f1": self.f1_score(gold, pred),
                "precision": self.precision_score(gold, pred),
                "recall": self.recall_score(gold, pred),
                "exact_match": self.exact_match(gold, pred),
            }
    
    def compute_all(self, gold: str, pred: str) -> Dict[str, float]:
        """
        计算所有答案指标
        """
        return {
            "f1": self.f1_score(gold, pred),
            "precision": self.precision_score(gold, pred),
            "recall": self.recall_score(gold, pred),
            "exact_match": self.exact_match(gold, pred),
            "bool_accuracy": self.bool_accuracy(gold, pred),
        }


# =============================================================================
# 批量评估
# =============================================================================

@dataclass
class BatchEvaluator:
    """批量评估器"""
    
    retrieval_metrics: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer_metrics: AnswerMetrics = field(default_factory=AnswerMetrics)
    
    def evaluate_retrieval_batch(
        self,
        results: List[Dict[str, Any]],
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """
        批量评估检索结果
        
        Args:
            results: 包含 retrieved_ids 和 gold_ids 的结果列表
            k_values: K 值列表
        
        Returns:
            平均指标
        """
        if k_values is None:
            k_values = [1, 3, 5, 10, 20]
        
        if not results:
            return {}
        
        # 收集所有指标
        all_metrics = {f'{m}@{k}': [] for k in k_values for m in ['hit', 'mrr', 'recall', 'precision', 'ndcg']}
        
        for r in results:
            retrieved = r.get('retrieved_ids', [])
            gold = r.get('gold_ids', [])
            
            metrics = self.retrieval_metrics.compute_all(retrieved, gold, k_values)
            for key, value in metrics.items():
                all_metrics[key].append(value)
        
        # 计算平均值
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        return avg_metrics
    
    def evaluate_answer_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        批量评估答案结果
        
        Args:
            results: 包含 gold_answer, pred_answer, answer_type 的结果列表
        
        Returns:
            平均指标
        """
        if not results:
            return {}
        
        # 按类型分组评估
        all_metrics = {
            'f1': [],
            'precision': [],
            'recall': [],
            'exact_match': [],
            'bool_accuracy': [],
        }
        
        for r in results:
            gold = r.get('gold_answer', '')
            pred = r.get('pred_answer', '')
            answer_type = r.get('answer_type', 'text')
            
            metrics = self.answer_metrics.compute_by_type(gold, pred, answer_type)
            
            for key, value in metrics.items():
                if key in all_metrics:
                    all_metrics[key].append(value)
        
        # 计算平均值
        avg_metrics = {}
        for key, values in all_metrics.items():
            if values:
                avg_metrics[f'avg_{key}'] = sum(values) / len(values)
        
        return avg_metrics
    
    def evaluate_e2e_batch(
        self,
        results: List[Dict[str, Any]],
        k_values: List[int] = None,
    ) -> Dict[str, float]:
        """
        端到端批量评估
        
        Args:
            results: 完整结果列表
            k_values: K 值列表
        
        Returns:
            所有指标的平均值
        """
        retrieval_metrics = self.evaluate_retrieval_batch(results, k_values)
        answer_metrics = self.evaluate_answer_batch(results)
        
        return {**retrieval_metrics, **answer_metrics}


# =============================================================================
# 延迟指标
# =============================================================================

@dataclass
class LatencyMetrics:
    """
    延迟指标计算
    
    用于分析系统响应时间:
    - 检索延迟
    - 生成延迟
    - 端到端延迟
    - 分位数统计
    """
    
    def compute_stats(self, latencies: List[float]) -> Dict[str, float]:
        """
        计算延迟统计
        
        Args:
            latencies: 延迟值列表 (ms)
        
        Returns:
            统计指标字典
        """
        if not latencies:
            return {}
        
        import numpy as np
        arr = np.array(latencies)
        
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }
    
    def compute_all(
        self,
        results: List[Dict[str, Any]],
        retrieval_key: str = "retrieval_time_ms",
        generation_key: str = "generation_time_ms",
        total_key: str = "total_time_ms",
    ) -> Dict[str, Dict[str, float]]:
        """
        计算所有延迟指标
        
        Args:
            results: 评估结果列表
            retrieval_key: 检索时间字段名
            generation_key: 生成时间字段名
            total_key: 总时间字段名
        
        Returns:
            各阶段延迟统计
        """
        stats = {}
        
        # 检索延迟
        retrieval_times = [
            r[retrieval_key] for r in results 
            if retrieval_key in r and r[retrieval_key] is not None
        ]
        if retrieval_times:
            stats["retrieval"] = self.compute_stats(retrieval_times)
        
        # 生成延迟
        generation_times = [
            r[generation_key] for r in results 
            if generation_key in r and r[generation_key] is not None
        ]
        if generation_times:
            stats["generation"] = self.compute_stats(generation_times)
        
        # 端到端延迟
        total_times = [
            r[total_key] for r in results 
            if total_key in r and r[total_key] is not None
        ]
        if total_times:
            stats["total"] = self.compute_stats(total_times)
        
        # 重排序延迟 (如果有)
        rerank_times = [
            r["rerank_time_ms"] for r in results 
            if "rerank_time_ms" in r and r["rerank_time_ms"] is not None
        ]
        if rerank_times:
            stats["rerank"] = self.compute_stats(rerank_times)
        
        return stats
    
    def compute_throughput(
        self,
        results: List[Dict[str, Any]],
        total_key: str = "total_time_ms",
    ) -> Dict[str, float]:
        """
        计算吞吐量指标
        
        Args:
            results: 评估结果列表
            total_key: 总时间字段名
        
        Returns:
            吞吐量指标
        """
        if not results:
            return {}
        
        total_times = [
            r[total_key] for r in results 
            if total_key in r and r[total_key] is not None
        ]
        
        if not total_times:
            return {}
        
        total_time_sec = sum(total_times) / 1000  # 转换为秒
        
        return {
            "samples_count": len(results),
            "total_time_sec": total_time_sec,
            "queries_per_second": len(results) / total_time_sec if total_time_sec > 0 else 0,
            "avg_time_per_query_ms": sum(total_times) / len(total_times),
        }
    
    def compute_timeout_stats(
        self,
        results: List[Dict[str, Any]],
        timeout_threshold_ms: float = 5000,
        total_key: str = "total_time_ms",
    ) -> Dict[str, Any]:
        """
        计算超时统计
        
        Args:
            results: 评估结果列表
            timeout_threshold_ms: 超时阈值 (ms)
            total_key: 总时间字段名
        
        Returns:
            超时统计
        """
        if not results:
            return {}
        
        total_times = [
            r.get(total_key, 0) for r in results
        ]
        
        timeouts = sum(1 for t in total_times if t > timeout_threshold_ms)
        
        return {
            "timeout_threshold_ms": timeout_threshold_ms,
            "timeout_count": timeouts,
            "timeout_rate": timeouts / len(results) if results else 0,
            "within_sla_count": len(results) - timeouts,
            "within_sla_rate": (len(results) - timeouts) / len(results) if results else 0,
        }
    
    def format_summary(self, stats: Dict[str, Dict[str, float]]) -> str:
        """格式化延迟摘要"""
        lines = ["\n--- Latency Summary ---"]
        
        for stage, metrics in stats.items():
            if metrics:
                lines.append(f"\n{stage.upper()}:")
                lines.append(f"  Mean: {metrics.get('mean', 0):.1f}ms")
                lines.append(f"  P50:  {metrics.get('p50', 0):.1f}ms")
                lines.append(f"  P90:  {metrics.get('p90', 0):.1f}ms")
                lines.append(f"  P99:  {metrics.get('p99', 0):.1f}ms")
        
        return "\n".join(lines)
