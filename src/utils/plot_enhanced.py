"""
增强版绘图工具

新增功能:
- 性能指标可视化（延迟分布、吞吐量）
- 答案质量对比图
- 多版本对比图
- 时间序列图
- 自动版本管理
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
from typing import Optional, List, Dict, Any
from datetime import datetime
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PlotManager:
    """绘图管理器"""
    
    def __init__(self, base_dir: str = 'results/plots'):
        """
        初始化绘图管理器
        
        Args:
            base_dir: 基础输出目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        """获取实验目录"""
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    def save_plot(self, experiment_name: str, filename: str):
        """保存当前图表"""
        exp_dir = self.get_experiment_dir(experiment_name)
        filepath = exp_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {filepath}")
        return filepath


# =============================================================================
# 基础绘图函数（兼容原有接口）
# =============================================================================

def plot_hits(df: pd.DataFrame, loc: str, column: str = 'hit@10'):
    """
    绘制命中率分布
    
    Args:
        df: 数据框
        loc: 实验名称
        column: 使用的列名
    """
    plt.figure(figsize=(8, 5))
    
    if column in df.columns:
        # 如果是 0/1 值，统计命中和未命中
        hits = df[column].value_counts().sort_index()
    else:
        logger.warning(f"Column {column} not found, using first boolean column")
        hits = df.iloc[:, 0].value_counts().sort_index()
    
    labels = ['Miss' if x == 0 else 'Hit' if x == 1 else str(x) for x in hits.index]
    values = hits.values
    colors = ['#ff6b6b', '#51cf66'][:len(values)]
    
    bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Result', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Hit Distribution ({column}) - {loc}', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.15)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2, 
            height + max(values) * 0.01,
            f'{int(height)} ({height/sum(values)*100:.1f}%)',
            ha='center', va='bottom', fontsize=10
        )
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_hits.png")


def plot_rrs(df: pd.DataFrame, loc: str, column: str = 'mrr@10', max_categories: Optional[int] = 20):
    """
    绘制 MRR 分布
    
    Args:
        df: 数据框
        loc: 实验名称
        column: 使用的列名
        max_categories: 最多显示类别数
    """
    plt.figure(figsize=(10, 5))
    
    if column not in df.columns:
        logger.warning(f"Column {column} not found")
        return
    
    # 统计分布
    counts = df[column].value_counts().sort_index(ascending=False)
    if max_categories and len(counts) > max_categories:
        counts = counts.head(max_categories)
    
    labels = [f'{idx:.2f}' if isinstance(idx, float) else str(idx) for idx in counts.index]
    values = counts.values
    colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
    
    bars = plt.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('MRR Values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'MRR Distribution ({column}) - {loc}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
    
    max_val = values.max()
    plt.ylim(0, max_val * 1.2)
    
    # 添加数值标签（只在前10个上显示）
    for i, bar in enumerate(bars[:10]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max_val * 0.01,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=8
        )
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_mrr.png")


def plot_recalls(df: pd.DataFrame, loc: str, column: str = 'recall@10', max_categories: Optional[int] = 20):
    """
    绘制召回率分布
    
    Args:
        df: 数据框
        loc: 实验名称
        column: 使用的列名
        max_categories: 最多显示类别数
    """
    plt.figure(figsize=(10, 5))
    
    if column not in df.columns:
        logger.warning(f"Column {column} not found")
        return
    
    counts = df[column].value_counts().sort_index(ascending=False)
    if max_categories and len(counts) > max_categories:
        counts = counts.head(max_categories)
    
    labels = [f'{idx:.2f}' if isinstance(idx, float) else str(idx) for idx in counts.index]
    values = counts.values
    colors = plt.cm.plasma(np.linspace(0, 1, len(values)))
    
    bars = plt.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
    plt.xlabel('Recall Values', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Recall Distribution ({column}) - {loc}', fontsize=14, fontweight='bold')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right', fontsize=9)
    
    max_val = values.max()
    plt.ylim(0, max_val * 1.2)
    
    for i, bar in enumerate(bars[:10]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max_val * 0.01,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=8
        )
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_recall.png")


def plot_category_hits(
    df: pd.DataFrame,
    loc: str,
    category_col: str = 'clause_type',
    hit_col: str = 'hit@10',
    max_categories: Optional[int] = 20,
):
    """
    绘制各类别的命中率
    
    Args:
        df: 数据框
        loc: 实验名称
        category_col: 类别列名
        hit_col: 命中率列名
        max_categories: 最多显示类别数
    """
    if category_col not in df.columns or hit_col not in df.columns:
        logger.warning(f"Columns {category_col} or {hit_col} not found")
        return
    
    grouped = (
        df.groupby(category_col)
        .agg(hit_k=(hit_col, "mean"), count=(category_col, "count"))
        .sort_values("hit_k", ascending=True)
    )
    
    if max_categories and len(grouped) > max_categories:
        grouped = grouped.tail(max_categories)
    
    plt.figure(figsize=(10, max(6, len(grouped) * 0.4)))
    
    colors = plt.cm.RdYlGn(grouped['hit_k'].values)
    bars = plt.barh(grouped.index, grouped["hit_k"], color=colors, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for i, (idx, row) in enumerate(grouped.iterrows()):
        plt.text(
            row['hit_k'] + 0.02,
            i,
            f"{row['hit_k']:.2f} (n={int(row['count'])})",
            va='center',
            fontsize=9
        )
    
    plt.xlabel(f'{hit_col}', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.title(f'{hit_col} by Category - {loc}', fontsize=14, fontweight='bold')
    plt.xlim(0, min(1.0, grouped['hit_k'].max() * 1.15))
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_hit_by_category.png")


# =============================================================================
# 新增功能: 性能指标可视化
# =============================================================================

def plot_latency_distribution(
    df: pd.DataFrame,
    loc: str,
    time_columns: List[str] = None,
):
    """
    绘制延迟分布图
    
    Args:
        df: 数据框
        loc: 实验名称
        time_columns: 时间列名列表
    """
    if time_columns is None:
        time_columns = ['retrieval_time_ms', 'generation_time_ms', 'total_time_ms']
    
    # 过滤存在的列
    available_cols = [col for col in time_columns if col in df.columns]
    if not available_cols:
        logger.warning("No time columns found")
        return
    
    fig, axes = plt.subplots(1, len(available_cols), figsize=(6 * len(available_cols), 5))
    if len(available_cols) == 1:
        axes = [axes]
    
    for ax, col in zip(axes, available_cols):
        data = df[col].dropna()
        
        # 直方图 + KDE
        ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # KDE 曲线
        from scipy import stats
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # 统计信息
        mean_val = data.mean()
        median_val = data.median()
        p95_val = data.quantile(0.95)
        
        ax.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}ms')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}ms')
        ax.axvline(p95_val, color='red', linestyle='--', linewidth=2, label=f'P95: {p95_val:.1f}ms')
        
        ax.set_xlabel('Time (ms)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(col.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_latency_distribution.png")


def plot_latency_boxplot(df: pd.DataFrame, loc: str):
    """
    绘制延迟箱线图
    
    Args:
        df: 数据框
        loc: 实验名称
    """
    time_columns = ['retrieval_time_ms', 'generation_time_ms', 'total_time_ms']
    available_cols = [col for col in time_columns if col in df.columns]
    
    if not available_cols:
        logger.warning("No time columns found")
        return
    
    plt.figure(figsize=(10, 6))
    
    data_to_plot = [df[col].dropna() for col in available_cols]
    labels = [col.replace('_time_ms', '').replace('_', ' ').title() for col in available_cols]
    
    bp = plt.boxplot(
        data_to_plot,
        labels=labels,
        patch_artist=True,
        notch=True,
        showmeans=True,
    )
    
    # 美化箱线图
    colors = ['lightblue', 'lightgreen', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title(f'Latency Comparison - {loc}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_latency_boxplot.png")


def plot_answer_quality_metrics(df: pd.DataFrame, loc: str):
    """
    绘制答案质量指标（包含简化版 RAGAS 指标）
    
    Args:
        df: 数据框
        loc: 实验名称
    """
    # 答案质量指标
    answer_metrics = ['f1', 'precision', 'recall', 'exact_match']
    # 简化版 RAGAS 指标
    simple_ragas_metrics = ['simple_context_coverage', 'simple_answer_context_overlap', 'simple_ground_truth_coverage']
    
    # 查找可用的指标
    available_answer = [m for m in answer_metrics if m in df.columns]
    available_ragas = [m for m in simple_ragas_metrics if m in df.columns]
    
    if not available_answer and not available_ragas:
        logger.warning("No answer quality metrics found")
        return
    
    # 创建组合图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Answer Quality & Simplified RAGAS Metrics - {loc}', fontsize=14, fontweight='bold')
    
    # 第一行：答案质量指标
    colors_answer = ['coral', 'lightgreen', 'skyblue', 'gold']
    for i, (ax, metric) in enumerate(zip(axes[0], available_answer[:3])):
        data = df[metric].dropna()
        
        ax.hist(data, bins=20, alpha=0.7, color=colors_answer[i], edgecolor='black')
        
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏第一行多余的子图
    for ax in axes[0][len(available_answer[:3]):]:
        ax.axis('off')
    
    # 第二行：简化版 RAGAS 指标
    colors_ragas = ['mediumpurple', 'mediumseagreen', 'darkorange']
    ragas_titles = {
        'simple_context_coverage': 'Context Coverage\n(≈Faithfulness)',
        'simple_answer_context_overlap': 'Answer-Context\nOverlap',
        'simple_ground_truth_coverage': 'Ground Truth Coverage\n(≈Context Recall)',
    }
    
    for i, (ax, metric) in enumerate(zip(axes[1], available_ragas[:3])):
        data = df[metric].dropna()
        
        ax.hist(data, bins=20, alpha=0.7, color=colors_ragas[i], edgecolor='black')
        
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(ragas_titles.get(metric, metric), fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏第二行多余的子图
    for ax in axes[1][len(available_ragas[:3]):]:
        ax.axis('off')
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_answer_quality.png")


def plot_metrics_heatmap(df: pd.DataFrame, loc: str):
    """
    绘制指标相关性热图
    
    Args:
        df: 数据框
        loc: 实验名称
    """
    # 选择数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 过滤出指标列
    metric_cols = [col for col in numeric_cols if any(
        keyword in col for keyword in ['hit', 'mrr', 'recall', 'precision', 'f1', 'exact']
    )]
    
    if len(metric_cols) < 2:
        logger.warning("Not enough metrics for heatmap")
        return
    
    # 计算相关性
    corr = df[metric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f'Metrics Correlation Heatmap - {loc}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_metrics_correlation.png")


# =============================================================================
# 新增功能: 多版本对比
# =============================================================================

def plot_version_comparison(
    results_dict: Dict[str, pd.DataFrame],
    metrics: List[str],
    output_dir: str = 'results/plots/comparison',
):
    """
    绘制多版本对比图
    
    Args:
        results_dict: {版本名: DataFrame} 字典
        metrics: 要对比的指标列表
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        versions = []
        values = []
        errors = []
        
        for version, df in results_dict.items():
            if metric in df.columns:
                versions.append(version)
                values.append(df[metric].mean())
                errors.append(df[metric].std())
        
        if not versions:
            continue
        
        bars = ax.bar(versions, values, yerr=errors, capsize=5, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10
            )
        
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xticklabels(versions, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filepath = output_path / 'version_comparison.png'
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved comparison plot: {filepath}")


# =============================================================================
# 组合图: 将多个子图合并为一张图
# =============================================================================

def plot_combined_retrieval_metrics(
    df: pd.DataFrame,
    loc: str,
    hit_col: str = 'hit@10',
    mrr_col: str = 'mrr@10',
    recall_col: str = 'recall@10',
):
    """
    绘制组合检索指标图（2x2 子图）
    
    包含:
    - Hits 分布
    - Recall 分布
    - MRR 分布
    - Latency 箱线图
    
    Args:
        df: 数据框
        loc: 实验名称
        hit_col: 命中率列名
        mrr_col: MRR 列名
        recall_col: 召回率列名
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Retrieval Metrics Overview - {loc}', fontsize=16, fontweight='bold', y=1.02)
    
    # ============ 子图1: Hits 分布 ============
    ax1 = axes[0, 0]
    if hit_col in df.columns:
        hits = df[hit_col].value_counts().sort_index()
        labels = ['Miss' if x == 0 else 'Hit' if x == 1 else str(x) for x in hits.index]
        values = hits.values
        colors = ['#ff6b6b', '#51cf66'][:len(values)]
        
        bars = ax1.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Result', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title(f'Hit Distribution ({hit_col})', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, max(values) * 1.15)
        
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                height + max(values) * 0.01,
                f'{int(height)} ({height/sum(values)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10
            )
    else:
        ax1.text(0.5, 0.5, f'{hit_col} not found', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Hit Distribution', fontsize=12, fontweight='bold')
    
    # ============ 子图2: Recall 分布 ============
    ax2 = axes[0, 1]
    if recall_col in df.columns:
        counts = df[recall_col].value_counts().sort_index(ascending=False)
        if len(counts) > 15:
            counts = counts.head(15)
        
        labels = [f'{idx:.2f}' if isinstance(idx, float) else str(idx) for idx in counts.index]
        values = counts.values
        colors = plt.cm.plasma(np.linspace(0, 1, len(values)))
        
        bars = ax2.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Recall Values', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title(f'Recall Distribution ({recall_col})', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylim(0, values.max() * 1.2)
    else:
        ax2.text(0.5, 0.5, f'{recall_col} not found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Recall Distribution', fontsize=12, fontweight='bold')
    
    # ============ 子图3: MRR 分布 ============
    ax3 = axes[1, 0]
    if mrr_col in df.columns:
        counts = df[mrr_col].value_counts().sort_index(ascending=False)
        if len(counts) > 15:
            counts = counts.head(15)
        
        labels = [f'{idx:.2f}' if isinstance(idx, float) else str(idx) for idx in counts.index]
        values = counts.values
        colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
        
        bars = ax3.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('MRR Values', fontsize=11)
        ax3.set_ylabel('Count', fontsize=11)
        ax3.set_title(f'MRR Distribution ({mrr_col})', fontsize=12, fontweight='bold')
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax3.set_ylim(0, values.max() * 1.2)
    else:
        ax3.text(0.5, 0.5, f'{mrr_col} not found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('MRR Distribution', fontsize=12, fontweight='bold')
    
    # ============ 子图4: Latency 箱线图 ============
    ax4 = axes[1, 1]
    time_columns = ['retrieval_time_ms', 'generation_time_ms', 'total_time_ms']
    available_cols = [col for col in time_columns if col in df.columns]
    
    if available_cols:
        data_to_plot = [df[col].dropna() for col in available_cols]
        labels = [col.replace('_time_ms', '').replace('_', ' ').title() for col in available_cols]
        
        bp = ax4.boxplot(
            data_to_plot,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True,
        )
        
        box_colors = ['lightblue', 'lightgreen', 'lightyellow']
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        
        ax4.set_ylabel('Time (ms)', fontsize=11)
        ax4.set_title('Latency Comparison', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        ax4.text(0.5, 0.5, 'No latency data found', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Latency Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    manager = PlotManager()
    manager.save_plot(loc, f"{loc}_combined_metrics.png")


# =============================================================================
# 统一接口: 生成所有图表
# =============================================================================

def plot_all_metrics(
    df: pd.DataFrame,
    loc: str,
    include_performance: bool = True,
    include_quality: bool = True,
    include_correlation: bool = False,
    include_combined: bool = True,
    include_separate: bool = False,
):
    """
    生成所有评估图表
    
    Args:
        df: 评估结果数据框
        loc: 实验名称
        include_performance: 是否包含性能图表（延迟分布）
        include_quality: 是否包含答案质量图表
        include_correlation: 是否包含相关性热图
        include_combined: 是否生成组合图（hits/recall/mrr/latency 合并为一张图）
        include_separate: 是否生成单独的基础图表（hits/recall/mrr/latency_boxplot）
    """
    logger.info(f"Generating plots for experiment: {loc}")
    
    # 组合图（推荐：4个子图合并为一张）
    if include_combined:
        try:
            plot_combined_retrieval_metrics(df, loc)
        except Exception as e:
            logger.warning(f"Failed to plot combined metrics: {e}")
    
    # 基础图表（单独的图，默认不生成）
    if include_separate:
        plot_hits(df, loc)
        plot_rrs(df, loc)
        plot_recalls(df, loc)
        plot_latency_boxplot(df, loc)
    
    # 分类命中率图（始终生成）
    plot_category_hits(df, loc)
    
    # 性能图表（延迟分布）
    if include_performance:
        try:
            plot_latency_distribution(df, loc)
        except Exception as e:
            logger.warning(f"Failed to plot performance metrics: {e}")
    
    # 答案质量图表
    if include_quality:
        try:
            plot_answer_quality_metrics(df, loc)
        except Exception as e:
            logger.warning(f"Failed to plot answer quality metrics: {e}")
    
    # 相关性热图
    if include_correlation:
        try:
            plot_metrics_heatmap(df, loc)
        except Exception as e:
            logger.warning(f"Failed to plot correlation heatmap: {e}")
    
    logger.success(f"All plots generated for: {loc}")


if __name__ == "__main__":
    # 测试代码
    from src.utils.seed_utils import set_global_seed
    set_global_seed()  # 从配置读取种子
    
    test_df = pd.DataFrame({
        'hit@10': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
        'mrr@10': np.random.uniform(0, 1, 100),
        'recall@10': np.random.uniform(0, 1, 100),
        'f1_score': np.random.uniform(0, 0.8, 100),
        'precision': np.random.uniform(0, 0.9, 100),
        'recall': np.random.uniform(0, 0.8, 100),
        'exact_match': np.random.choice([0, 1], 100, p=[0.9, 0.1]),
        'retrieval_time_ms': np.random.normal(150, 30, 100),
        'generation_time_ms': np.random.normal(2000, 400, 100),
        'total_time_ms': np.random.normal(2300, 450, 100),
        'clause_type': np.random.choice(['Termination', 'Payment', 'Liability'], 100),
    })
    
    plot_all_metrics(test_df, 'test_enhanced', include_correlation=True)
    logger.success("Test plots generated!")
