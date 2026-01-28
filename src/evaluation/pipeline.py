"""
统一评估管道

整合所有评估组件:
- 传统指标 (Hit@K, MRR, Recall, F1)
- RAGAS 指标 (Faithfulness, Answer Relevancy, Context Precision/Recall)
- WandB 追踪
- 自动可视化图表生成

使用方式:
    from src.evaluation import EvaluationPipeline
    
    pipeline = EvaluationPipeline(experiment_name="v1_baseline")
    results = pipeline.evaluate(eval_data)
    pipeline.save_results()  # 自动保存 CSV、JSON、图表
"""

import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

from src.evaluation.metrics import RetrievalMetrics, AnswerMetrics, BatchEvaluator, LatencyMetrics
from src.evaluation.wandb_tracker import WandBTracker, WandBConfig


@dataclass
class EvaluationConfig:
    """评估配置"""
    
    # 检索指标
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    
    # RAGAS 配置
    ragas_enabled: bool = True
    ragas_metrics: List[str] = field(default_factory=lambda: [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ])
    
    # WandB 配置
    wandb_enabled: bool = False
    wandb_project: str = "cuad-assistant"
    wandb_experiment: str = ""
    
    # 输出配置
    output_dir: str = ""
    save_detailed: bool = True
    
    # 可视化配置 (新增)
    plot_enabled: bool = True
    plot_dir: str = "results/plots"
    plot_include_performance: bool = True
    plot_include_quality: bool = True
    plot_include_correlation: bool = False
    
    @classmethod
    def from_config(cls, config) -> 'EvaluationConfig':
        """从全局配置创建"""
        try:
            eval_cfg = config.evaluation
            return cls(
                k_values=getattr(eval_cfg.traditional, 'k_values', cls.k_values),
                ragas_enabled=getattr(eval_cfg.ragas, 'enabled', True),
                ragas_metrics=getattr(eval_cfg.ragas, 'metrics', cls.ragas_metrics),
                wandb_enabled=getattr(eval_cfg.wandb, 'enabled', False),
                wandb_project=getattr(eval_cfg.wandb, 'project', 'cuad-assistant'),
                output_dir=getattr(eval_cfg.output, 'results_dir', ''),
            )
        except Exception:
            return cls()


class EvaluationPipeline:
    """
    统一评估管道
    
    整合所有评估组件，提供端到端的评估功能:
    - 检索指标计算
    - 答案质量评估
    - 延迟统计
    - 自动可视化
    - 版本管理
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        experiment_name: Optional[str] = None,
        experiment_version: Optional[str] = None,
    ):
        """
        初始化评估管道
        
        Args:
            config: 评估配置
            experiment_name: 实验名称 (用于版本管理和图表命名)
            experiment_version: 实验版本 (如 "v1", "v2_bm25")
        """
        self.config = config or EvaluationConfig()
        
        # 版本管理
        self.experiment_name = experiment_name
        self.experiment_version = experiment_version
        self._init_version_info()
        
        # 初始化评估器
        self.retrieval_metrics = RetrievalMetrics()
        self.answer_metrics = AnswerMetrics()
        self.batch_evaluator = BatchEvaluator()
        self.latency_metrics = LatencyMetrics()
        
        # WandB 追踪器
        self.wandb_tracker: Optional[WandBTracker] = None
        
        # RAGAS 评估器 (延迟初始化)
        self._ragas_evaluator = None
        
        # 结果存储
        self.results: List[Dict[str, Any]] = []
        self.summary: Dict[str, float] = {}
        self.latency_summary: Dict[str, Dict[str, float]] = {}
        
        # 输出文件路径记录
        self.output_files: Dict[str, str] = {}
    
    def _init_version_info(self) -> None:
        """初始化版本信息"""
        self.created_at = datetime.now()
        self.timestamp = self.created_at.strftime("%Y%m%d_%H%M%S")
        
        # 生成完整实验标识（不带时间戳，便于覆盖更新）
        if self.experiment_name is None:
            self.experiment_name = "evaluation"
        
        if self.experiment_version:
            self.full_experiment_id = f"{self.experiment_version}_{self.experiment_name}"
        else:
            self.full_experiment_id = self.experiment_name
        
        logger.info(f"Experiment ID: {self.full_experiment_id}")
    
    def init_wandb(
        self,
        experiment_name: str = None,
        config_dict: Dict[str, Any] = None,
        **kwargs,
    ) -> bool:
        """
        初始化 WandB 追踪
        
        Args:
            experiment_name: 实验名称
            config_dict: 配置字典
        
        Returns:
            是否成功初始化
        """
        if not self.config.wandb_enabled:
            return False
        
        self.wandb_tracker = WandBTracker(
            project=self.config.wandb_project,
            experiment_name=experiment_name or self.config.wandb_experiment,
            **kwargs,
        )
        
        return self.wandb_tracker.init(config_dict=config_dict)
    
    def evaluate_single(
        self,
        query: str,
        gold_answer: str,
        pred_answer: str,
        retrieved_ids: List[str],
        gold_ids: List[str],
        retrieved_contexts: List[str] = None,
        answer_type: str = "text",
        **extra_fields,
    ) -> Dict[str, Any]:
        """
        评估单个样本
        
        Args:
            query: 查询问题
            gold_answer: 金标准答案
            pred_answer: 预测答案
            retrieved_ids: 检索到的文档 ID
            gold_ids: 金标准文档 ID
            retrieved_contexts: 检索到的上下文文本
            answer_type: 答案类型
            **extra_fields: 额外字段
        
        Returns:
            评估结果字典
        """
        result = {
            "query": query,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "answer_type": answer_type,
            **extra_fields,
        }
        
        # 检索指标
        for k in self.config.k_values:
            result[f"hit@{k}"] = self.retrieval_metrics.hit_at_k(retrieved_ids, gold_ids, k)
            result[f"mrr@{k}"] = self.retrieval_metrics.mrr_at_k(retrieved_ids, gold_ids, k)
            result[f"recall@{k}"] = self.retrieval_metrics.recall_at_k(retrieved_ids, gold_ids, k)
            result[f"precision@{k}"] = self.retrieval_metrics.precision_at_k(retrieved_ids, gold_ids, k)
        
        # 答案指标
        answer_results = self.answer_metrics.compute_by_type(gold_answer, pred_answer, answer_type)
        result.update(answer_results)
        
        # 简化版 RAGAS 指标 (不需要 LLM)
        if retrieved_contexts:
            from src.evaluation.ragas_evaluator import SimplifiedRAGASMetrics
            simplified = SimplifiedRAGASMetrics()
            ragas_simple = simplified.compute_all(pred_answer, retrieved_contexts, gold_answer)
            for k, v in ragas_simple.items():
                result[f"simple_{k}"] = v
        
        return result
    
    def evaluate_batch(
        self,
        data: List[Dict[str, Any]],
        query_key: str = "query",
        gold_answer_key: str = "gold_answer",
        pred_answer_key: str = "pred_answer",
        retrieved_ids_key: str = "retrieved_ids",
        gold_ids_key: str = "gold_ids",
        contexts_key: str = "retrieved_contexts",
        answer_type_key: str = "answer_type",
        progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量评估
        
        Args:
            data: 评估数据列表
            *_key: 各字段的 key 名称
            progress: 是否显示进度
        
        Returns:
            评估结果列表
        """
        from tqdm import tqdm
        
        self.results = []
        iterator = tqdm(data, desc="Evaluating") if progress else data
        
        for item in iterator:
            result = self.evaluate_single(
                query=item.get(query_key, ""),
                gold_answer=item.get(gold_answer_key, ""),
                pred_answer=item.get(pred_answer_key, ""),
                retrieved_ids=item.get(retrieved_ids_key, []),
                gold_ids=item.get(gold_ids_key, []),
                retrieved_contexts=item.get(contexts_key, []),
                answer_type=item.get(answer_type_key, "text"),
                # 保留其他字段
                **{k: v for k, v in item.items() if k not in [
                    query_key, gold_answer_key, pred_answer_key,
                    retrieved_ids_key, gold_ids_key, contexts_key, answer_type_key
                ]}
            )
            self.results.append(result)
        
        # 计算摘要
        self._compute_summary()
        
        # 记录到 WandB
        if self.wandb_tracker and self.wandb_tracker.is_active:
            self.wandb_tracker.log_evaluation_results(
                retrieval_metrics=self._get_retrieval_summary(),
                answer_metrics=self._get_answer_summary(),
                detailed_results=self.results[:100],  # 限制数量
            )
        
        return self.results
    
    def evaluate_with_ragas(
        self,
        llm = None,
        embeddings = None,
    ) -> Dict[str, float]:
        """
        使用 RAGAS 进行评估 (需要 LLM)
        
        Args:
            llm: 用于评估的 LLM
            embeddings: 用于评估的 Embeddings
        
        Returns:
            RAGAS 评估结果
        """
        if not self.config.ragas_enabled:
            logger.warning("RAGAS evaluation is disabled")
            return {}
        
        if not self.results:
            logger.warning("No results to evaluate with RAGAS")
            return {}
        
        try:
            from src.evaluation.ragas_evaluator import RAGASEvaluator, RAGASConfig
            
            if self._ragas_evaluator is None:
                ragas_config = RAGASConfig(metrics=self.config.ragas_metrics)
                self._ragas_evaluator = RAGASEvaluator(ragas_config)
            
            # 准备数据
            questions = [r.get("query", "") for r in self.results]
            answers = [r.get("pred_answer", "") for r in self.results]
            contexts = [r.get("retrieved_contexts", []) for r in self.results]
            ground_truths = [r.get("gold_answer", "") for r in self.results]
            
            # 执行 RAGAS 评估
            ragas_results = self._ragas_evaluator.evaluate(
                questions=questions,
                answers=answers,
                contexts=contexts,
                ground_truths=ground_truths,
                llm=llm,
                embeddings=embeddings,
            )
            
            # 更新摘要
            for k, v in ragas_results.items():
                if not k.startswith('_'):
                    self.summary[f"ragas_{k}"] = v
            
            # 记录到 WandB
            if self.wandb_tracker and self.wandb_tracker.is_active:
                self.wandb_tracker.log_metrics(ragas_results, prefix="ragas")
            
            return ragas_results
            
        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}")
            return {}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}
    
    def _compute_summary(self) -> None:
        """计算汇总指标"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        
        # 检索指标平均值
        for k in self.config.k_values:
            for metric in ['hit', 'mrr', 'recall', 'precision']:
                col = f"{metric}@{k}"
                if col in df.columns:
                    self.summary[f"avg_{col}"] = df[col].mean()
        
        # 答案指标平均值
        for metric in ['f1', 'exact_match', 'precision', 'recall', 'accuracy']:
            if metric in df.columns:
                self.summary[f"avg_{metric}"] = df[metric].mean(skipna=True)
        
        # 简化 RAGAS 指标
        for metric in ['simple_context_coverage', 'simple_answer_context_overlap', 'simple_ground_truth_coverage']:
            if metric in df.columns:
                self.summary[f"avg_{metric}"] = df[metric].mean(skipna=True)
        
        # 按答案类型分组
        if 'answer_type' in df.columns:
            type_groups = df.groupby('answer_type').agg({
                col: 'mean' for col in df.columns if df[col].dtype in ['float64', 'int64']
            })
            self.summary['_by_answer_type'] = type_groups.to_dict()
        
        # 延迟指标
        self._compute_latency_summary()
    
    def _compute_latency_summary(self) -> None:
        """计算延迟统计"""
        if not self.results:
            return
        
        # 计算各阶段延迟统计
        self.latency_summary = self.latency_metrics.compute_all(self.results)
        
        # 添加到主摘要
        for stage, stats in self.latency_summary.items():
            for metric, value in stats.items():
                self.summary[f"latency_{stage}_{metric}"] = value
        
        # 计算吞吐量
        throughput = self.latency_metrics.compute_throughput(self.results)
        for k, v in throughput.items():
            self.summary[f"throughput_{k}"] = v
        
        # 计算超时统计 (默认 5 秒)
        timeout_stats = self.latency_metrics.compute_timeout_stats(self.results)
        for k, v in timeout_stats.items():
            self.summary[f"timeout_{k}"] = v
    
    def _get_retrieval_summary(self) -> Dict[str, float]:
        """获取检索指标摘要"""
        return {k: v for k, v in self.summary.items() 
                if any(m in k for m in ['hit@', 'mrr@', 'recall@', 'precision@', 'ndcg@'])}
    
    def _get_answer_summary(self) -> Dict[str, float]:
        """获取答案指标摘要"""
        return {k: v for k, v in self.summary.items() 
                if any(m in k for m in ['f1', 'exact_match', 'accuracy', 'precision', 'recall'])
                and '@' not in k}
    
    def _get_latency_summary(self) -> Dict[str, float]:
        """获取延迟指标摘要"""
        return {k: v for k, v in self.summary.items() 
                if k.startswith('latency_') or k.startswith('throughput_') or k.startswith('timeout_')}
    
    def get_summary(self) -> Dict[str, float]:
        """获取完整摘要"""
        return self.summary
    
    def get_results_df(self) -> pd.DataFrame:
        """获取结果 DataFrame"""
        return pd.DataFrame(self.results)
    
    def save_results(
        self,
        output_path: str = None,
        include_summary: bool = True,
        include_plots: bool = None,
    ) -> Dict[str, str]:
        """
        保存评估结果（CSV、JSON 摘要、可视化图表）
        
        Args:
            output_path: 输出文件路径
            include_summary: 是否包含摘要文件
            include_plots: 是否生成图表 (None 时使用 config.plot_enabled)
        
        Returns:
            输出文件路径字典 {"csv": ..., "summary": ..., "plots_dir": ...}
        """
        if output_path is None:
            output_dir = self.config.output_dir or "results/evaluation"
            output_path = os.path.join(output_dir, f"{self.full_experiment_id}.csv")
        
        # 确保目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        df = self.get_results_df()
        df.to_csv(output_path, index=False)
        logger.success(f"Saved evaluation results to {output_path}")
        self.output_files["csv"] = output_path
        
        # 保存摘要
        if include_summary:
            summary_path = output_path.replace('.csv', '_summary.json')
            summary_data = self._prepare_summary_for_save()
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved summary to {summary_path}")
            self.output_files["summary"] = summary_path
        
        # 生成 Markdown 报告
        try:
            report_path = output_path.replace('.csv', '_report.md')
            self._generate_markdown_report(report_path)
            self.output_files["report"] = report_path
        except Exception as e:
            logger.warning(f"Failed to generate markdown report: {e}")
        
        # 生成图表
        if include_plots is None:
            include_plots = self.config.plot_enabled
        
        if include_plots:
            plots_dir = self.generate_plots()
            if plots_dir:
                self.output_files["plots_dir"] = plots_dir
        
        # 打印输出文件汇总
        self._print_output_summary()
        
        return self.output_files
    
    def _prepare_summary_for_save(self) -> Dict[str, Any]:
        """准备用于保存的摘要数据"""
        summary_data = {
            # 元信息
            "experiment_id": self.full_experiment_id,
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "created_at": self.created_at.isoformat(),
            "timestamp": self.timestamp,
            "total_samples": len(self.results),
            
            # 指标摘要
            "metrics": {
                k: v for k, v in self.summary.items() 
                if not k.startswith('_') and isinstance(v, (int, float, str, list, dict))
            },
            
            # 延迟摘要
            "latency": self.latency_summary,
            
            # 配置信息
            "config": {
                "k_values": self.config.k_values,
                "ragas_enabled": self.config.ragas_enabled,
                "wandb_enabled": self.config.wandb_enabled,
            }
        }
        return summary_data
    
    def generate_plots(self, plot_dir: str = None) -> Optional[str]:
        """
        生成可视化图表
        
        Args:
            plot_dir: 图表输出目录 (None 时使用 config.plot_dir)
        
        Returns:
            图表目录路径
        """
        if not self.results:
            logger.warning("No results to plot")
            return None
        
        try:
            from src.utils.plot_enhanced import plot_all_metrics
            
            # 确定输出目录
            if plot_dir is None:
                plot_dir = self.config.plot_dir or "results/plots"
            
            # 获取结果数据
            df = self.get_results_df()
            
            # 生成图表
            logger.info(f"Generating plots for experiment: {self.full_experiment_id}")
            plot_all_metrics(
                df,
                loc=self.full_experiment_id,
                include_performance=self.config.plot_include_performance,
                include_quality=self.config.plot_include_quality,
                include_correlation=self.config.plot_include_correlation,
            )
            
            # 返回图表目录
            plots_output_dir = os.path.join(plot_dir, self.full_experiment_id)
            logger.success(f"Plots saved to: {plots_output_dir}")
            
            return plots_output_dir
            
        except ImportError as e:
            logger.warning(f"Plot generation failed - missing dependency: {e}")
            logger.warning("Install with: pip install matplotlib seaborn scipy")
            return None
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
            return None
    
    def _generate_markdown_report(self, output_path: str) -> str:
        """
        生成 Markdown 格式的评估报告
        
        Args:
            output_path: 输出文件路径
        
        Returns:
            报告文件路径
        """
        df = self.get_results_df()
        lines = []
        
        # 标题
        lines.append(f"# {self.full_experiment_id} Evaluation Report")
        lines.append("")
        lines.append(f"Auto-generated evaluation report.")
        lines.append("")
        lines.append(f"- **Experiment**: {self.experiment_name}")
        lines.append(f"- **Created**: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **Total Samples**: {len(self.results)}")
        lines.append("")
        
        # 总体指标摘要
        lines.append("## Overall Metrics")
        lines.append("")
        
        # 检索指标表格
        lines.append("### Retrieval Metrics")
        lines.append("")
        lines.append("| K | Hit@K | MRR@K | Recall@K | Precision@K |")
        lines.append("|---|-------|-------|----------|-------------|")
        for k in self.config.k_values:
            hit = self.summary.get(f'avg_hit@{k}', 0)
            mrr = self.summary.get(f'avg_mrr@{k}', 0)
            recall = self.summary.get(f'avg_recall@{k}', 0)
            precision = self.summary.get(f'avg_precision@{k}', 0)
            lines.append(f"| {k} | {hit:.4f} | {mrr:.4f} | {recall:.4f} | {precision:.4f} |")
        lines.append("")
        
        # 答案质量指标
        lines.append("### Answer Quality Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric in ['f1', 'exact_match', 'precision', 'recall']:
            key = f'avg_{metric}'
            if key in self.summary:
                lines.append(f"| {metric.replace('_', ' ').title()} | {self.summary[key]:.4f} |")
        lines.append("")
        
        # 简化版 RAGAS 指标
        simple_metrics = {
            'simple_context_coverage': 'Context Coverage (≈Faithfulness)',
            'simple_answer_context_overlap': 'Answer-Context Overlap',
            'simple_ground_truth_coverage': 'Ground Truth Coverage (≈Context Recall)',
        }
        has_simple = any(f'avg_{k}' in self.summary for k in simple_metrics.keys())
        if has_simple:
            lines.append("### Simplified RAGAS Metrics")
            lines.append("")
            lines.append("| Metric | Value | Description |")
            lines.append("|--------|-------|-------------|")
            for key, desc in simple_metrics.items():
                avg_key = f'avg_{key}'
                if avg_key in self.summary:
                    lines.append(f"| {key} | {self.summary[avg_key]:.4f} | {desc} |")
            lines.append("")
        
        # 延迟指标
        if self.latency_summary:
            lines.append("### Latency Metrics")
            lines.append("")
            lines.append("| Stage | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) |")
            lines.append("|-------|-----------|----------|----------|----------|")
            for stage in ['retrieval', 'generation', 'total']:
                if stage in self.latency_summary:
                    stats = self.latency_summary[stage]
                    lines.append(f"| {stage.title()} | {stats.get('mean', 0):.1f} | {stats.get('p50', 0):.1f} | {stats.get('p90', 0):.1f} | {stats.get('p99', 0):.1f} |")
            lines.append("")
        
        # 按类别统计
        if 'clause_type' in df.columns:
            lines.append("## Results by Category")
            lines.append("")
            
            # 获取最大 k 值的列名
            max_k = max(self.config.k_values)
            hit_col = f'hit@{max_k}'
            mrr_col = f'mrr@{max_k}'
            recall_col = f'recall@{max_k}'
            
            # 按类别分组
            grouped = df.groupby('clause_type').agg({
                hit_col: 'mean' if hit_col in df.columns else lambda x: 0,
                mrr_col: 'mean' if mrr_col in df.columns else lambda x: 0,
                recall_col: 'mean' if recall_col in df.columns else lambda x: 0,
                'f1': 'mean' if 'f1' in df.columns else lambda x: 0,
                'exact_match': 'mean' if 'exact_match' in df.columns else lambda x: 0,
                'clause_type': 'count',
            }).rename(columns={'clause_type': 'count'})
            
            # 按 hit 排序
            if hit_col in grouped.columns:
                grouped = grouped.sort_values(hit_col, ascending=False)
            
            for category in grouped.index[:10]:  # 只显示前10个类别
                cat_data = grouped.loc[category]
                cat_df = df[df['clause_type'] == category]
                
                lines.append(f"### {category}")
                lines.append("")
                lines.append(f"- **Samples**: {int(cat_data['count'])}")
                if hit_col in grouped.columns:
                    lines.append(f"- **Hit@{max_k}**: {cat_data[hit_col]:.4f}")
                if mrr_col in grouped.columns:
                    lines.append(f"- **MRR@{max_k}**: {cat_data[mrr_col]:.4f}")
                if 'f1' in grouped.columns:
                    lines.append(f"- **F1**: {cat_data['f1']:.4f}")
                if 'exact_match' in grouped.columns:
                    lines.append(f"- **Exact Match**: {cat_data['exact_match']:.4f}")
                lines.append("")
                
                # 显示该类别的示例（最多3个）
                examples = cat_df.head(3)
                for i, (_, row) in enumerate(examples.iterrows(), 1):
                    lines.append(f"#### Example {i}")
                    lines.append("")
                    lines.append(f"- **Query**: {row.get('query', 'N/A')}")
                    lines.append(f"- **Gold Answer**: {row.get('gold_answer', 'N/A')}")
                    lines.append(f"- **Pred Answer**: {row.get('pred_answer', 'N/A')}")
                    
                    # 指标
                    metrics_str = []
                    if hit_col in row:
                        metrics_str.append(f"hit@{max_k}={row[hit_col]:.4f}")
                    if mrr_col in row:
                        metrics_str.append(f"mrr@{max_k}={row[mrr_col]:.4f}")
                    if 'f1' in row:
                        metrics_str.append(f"f1={row['f1']:.4f}")
                    if 'exact_match' in row:
                        metrics_str.append(f"em={row['exact_match']:.4f}")
                    
                    if metrics_str:
                        lines.append(f"- **Metrics**: {', '.join(metrics_str)}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        # 写入文件
        report_content = "\n".join(lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Saved markdown report to {output_path}")
        return output_path
    
    def _print_output_summary(self) -> None:
        """打印输出文件汇总"""
        if not self.output_files:
            return
        
        print("\n" + "-" * 50)
        print("OUTPUT FILES")
        print("-" * 50)
        
        for file_type, path in self.output_files.items():
            print(f"  {file_type:12s}: {path}")
        
        print("-" * 50)
    
    def print_summary(self) -> None:
        """打印评估摘要"""
        if not self.summary:
            logger.warning("No summary available")
            return
        
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        # 检索指标
        print("\n--- Retrieval Metrics ---")
        for k in self.config.k_values:
            hit = self.summary.get(f'avg_hit@{k}', 0)
            mrr = self.summary.get(f'avg_mrr@{k}', 0)
            recall = self.summary.get(f'avg_recall@{k}', 0)
            print(f"  @{k:2d}: Hit={hit:.4f}, MRR={mrr:.4f}, Recall={recall:.4f}")
        
        # 答案指标
        print("\n--- Answer Metrics ---")
        for metric in ['f1', 'exact_match', 'accuracy']:
            key = f'avg_{metric}'
            if key in self.summary:
                print(f"  {metric:12s}: {self.summary[key]:.4f}")
        
        # RAGAS 指标
        ragas_metrics = {k: v for k, v in self.summary.items() if k.startswith('ragas_')}
        if ragas_metrics:
            print("\n--- RAGAS Metrics ---")
            for k, v in ragas_metrics.items():
                print(f"  {k:20s}: {v:.4f}")
        
        # 简化 RAGAS 指标
        simple_metrics = {k: v for k, v in self.summary.items() if k.startswith('avg_simple_')}
        if simple_metrics:
            print("\n--- Simplified RAGAS Metrics ---")
            for k, v in simple_metrics.items():
                print(f"  {k:30s}: {v:.4f}")
        
        # 延迟指标
        if self.latency_summary:
            print("\n--- Latency Metrics ---")
            for stage, stats in self.latency_summary.items():
                if stats:
                    print(f"\n  {stage.upper()}:")
                    print(f"    Mean:  {stats.get('mean', 0):>8.1f} ms")
                    print(f"    P50:   {stats.get('p50', 0):>8.1f} ms")
                    print(f"    P90:   {stats.get('p90', 0):>8.1f} ms")
                    print(f"    P99:   {stats.get('p99', 0):>8.1f} ms")
                    print(f"    Min:   {stats.get('min', 0):>8.1f} ms")
                    print(f"    Max:   {stats.get('max', 0):>8.1f} ms")
            
            # 吞吐量
            qps = self.summary.get('throughput_queries_per_second', 0)
            if qps > 0:
                print(f"\n  THROUGHPUT:")
                print(f"    QPS: {qps:.2f} queries/sec")
            
            # 超时
            timeout_rate = self.summary.get('timeout_timeout_rate', 0)
            if timeout_rate > 0:
                print(f"\n  TIMEOUT:")
                print(f"    Rate: {timeout_rate*100:.2f}%")
        
        print("\n" + "=" * 60)
    
    def finish(self) -> None:
        """结束评估管道"""
        if self.wandb_tracker:
            self.wandb_tracker.log_summary(self.summary)
            self.wandb_tracker.finish()


# =============================================================================
# 便捷函数
# =============================================================================

def evaluate_rag_results(
    results: List[Dict[str, Any]],
    experiment_name: str = None,
    experiment_version: str = None,
    config: EvaluationConfig = None,
    wandb_enabled: bool = False,
    wandb_project: str = "cuad-assistant",
    output_path: str = None,
    plot_enabled: bool = True,
) -> Dict[str, float]:
    """
    便捷的 RAG 评估函数
    
    Args:
        results: 评估结果列表
        experiment_name: 实验名称 (如 "baseline", "with_rerank")
        experiment_version: 实验版本 (如 "v1", "v2")
        config: 评估配置
        wandb_enabled: 是否启用 WandB
        wandb_project: WandB 项目名
        output_path: 输出路径
        plot_enabled: 是否生成图表
    
    Returns:
        评估摘要
    
    Example:
        >>> summary = evaluate_rag_results(
        ...     results=eval_data,
        ...     experiment_name="vanilla_rag",
        ...     experiment_version="v1",
        ...     plot_enabled=True,
        ... )
    """
    if config is None:
        config = EvaluationConfig(
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            plot_enabled=plot_enabled,
        )
    
    pipeline = EvaluationPipeline(
        config,
        experiment_name=experiment_name,
        experiment_version=experiment_version,
    )
    
    if config.wandb_enabled:
        pipeline.init_wandb()
    
    pipeline.evaluate_batch(results)
    
    if output_path:
        pipeline.save_results(output_path)
    else:
        pipeline.save_results()
    
    pipeline.print_summary()
    pipeline.finish()
    
    return pipeline.get_summary()


def quick_evaluate(
    results: List[Dict[str, Any]],
    name: str = "quick_eval",
    version: str = None,
    save: bool = True,
    plot: bool = True,
) -> EvaluationPipeline:
    """
    快速评估函数
    
    Args:
        results: 评估数据
        name: 实验名称
        version: 版本号
        save: 是否保存结果
        plot: 是否生成图表
    
    Returns:
        EvaluationPipeline 实例 (可进一步操作)
    
    Example:
        >>> pipeline = quick_evaluate(eval_data, name="my_test", version="v1")
        >>> print(pipeline.summary)
        >>> df = pipeline.get_results_df()
    """
    config = EvaluationConfig(
        plot_enabled=plot,
        k_values=[1, 3, 5, 10],
    )
    
    pipeline = EvaluationPipeline(
        config,
        experiment_name=name,
        experiment_version=version,
    )
    
    pipeline.evaluate_batch(results)
    
    if save:
        pipeline.save_results()
    
    pipeline.print_summary()
    
    return pipeline
