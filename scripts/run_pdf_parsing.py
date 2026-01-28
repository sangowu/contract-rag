#!/usr/bin/env python
"""
PDF 解析评估脚本

功能:
- 批量解析 PDF 文件
- 与金标准 CSV 对比评估准确性
- 统计解析耗时、成功率等指标
- 生成评估报告

使用方式:
    # 测试模式：随机 100 份 PDF
    python scripts/run_pdf_parsing.py --mode test --max-files 100
    
    # 完整评估
    python scripts/run_pdf_parsing.py --mode prod
    
    # 仅解析（不评估）
    python scripts/run_pdf_parsing.py --mode test --skip-eval
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.core.config import init_config, get_config
from src.utils.seed_utils import set_global_seed, get_seed
from src.pdf.parser import (
    parse_pdf_batch,
    find_pdf_files,
    PDFParseResult,
)


# =============================================================================
# 配置
# =============================================================================

DEFAULT_PDF_DIR = "data/raw/CUAD_v1/full_contract_pdf"
DEFAULT_OUTPUT_DIR = "data/pdf_parsed_results"
DEFAULT_RESULTS_DIR = "results/pdf_parsing"


# =============================================================================
# 评估指标
# =============================================================================

@dataclass
class ParsingMetrics:
    """解析指标"""
    total_files: int = 0
    success_count: int = 0
    fail_count: int = 0
    success_rate: float = 0.0
    
    total_pages: int = 0
    total_text_blocks: int = 0
    total_tables: int = 0
    total_chars: int = 0
    total_chunks: int = 0
    
    avg_pages_per_file: float = 0.0
    avg_chunks_per_file: float = 0.0
    avg_chars_per_file: float = 0.0
    
    total_parse_time_ms: float = 0.0
    avg_parse_time_ms: float = 0.0
    min_parse_time_ms: float = 0.0
    max_parse_time_ms: float = 0.0
    
    avg_confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccuracyMetrics:
    """准确性指标（与金标准对比）"""
    files_matched: int = 0
    files_total: int = 0
    match_rate: float = 0.0
    
    # 文本覆盖率：解析文本包含金标准文本的比例
    avg_text_coverage: float = 0.0
    min_text_coverage: float = 0.0
    max_text_coverage: float = 0.0
    
    # 字符级别的召回率
    avg_char_recall: float = 0.0
    
    # 按条款类型的覆盖率
    coverage_by_clause_type: Dict[str, float] = None
    
    def __post_init__(self):
        if self.coverage_by_clause_type is None:
            self.coverage_by_clause_type = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 评估函数
# =============================================================================

def compute_parsing_metrics(results: List[PDFParseResult]) -> ParsingMetrics:
    """计算解析指标"""
    metrics = ParsingMetrics()
    
    metrics.total_files = len(results)
    metrics.success_count = sum(1 for r in results if r.success)
    metrics.fail_count = metrics.total_files - metrics.success_count
    metrics.success_rate = metrics.success_count / metrics.total_files if metrics.total_files > 0 else 0
    
    successful = [r for r in results if r.success]
    
    if successful:
        metrics.total_pages = sum(r.page_count for r in successful)
        metrics.total_text_blocks = sum(r.text_block_count for r in successful)
        metrics.total_tables = sum(r.table_count for r in successful)
        metrics.total_chars = sum(r.char_count for r in successful)
        metrics.total_chunks = sum(r.chunk_count for r in successful)
        
        metrics.avg_pages_per_file = metrics.total_pages / len(successful)
        metrics.avg_chunks_per_file = metrics.total_chunks / len(successful)
        metrics.avg_chars_per_file = metrics.total_chars / len(successful)
        
        parse_times = [r.parse_time_ms for r in successful]
        metrics.total_parse_time_ms = sum(parse_times)
        metrics.avg_parse_time_ms = sum(parse_times) / len(parse_times)
        metrics.min_parse_time_ms = min(parse_times)
        metrics.max_parse_time_ms = max(parse_times)
        
        confidences = [r.avg_confidence for r in successful]
        metrics.avg_confidence = sum(confidences) / len(confidences)
    
    return metrics


def compute_text_coverage(parsed_text: str, gold_texts: List[str]) -> float:
    """
    计算文本覆盖率
    
    Args:
        parsed_text: 解析得到的文本
        gold_texts: 金标准文本列表
    
    Returns:
        覆盖率 (0-1)
    """
    if not gold_texts:
        return 1.0
    
    parsed_lower = parsed_text.lower()
    covered = 0
    total = 0
    
    for gold_text in gold_texts:
        if not gold_text or gold_text == "[]":
            continue
        
        # 清理金标准文本
        clean_text = gold_text.strip().lower()
        if clean_text.startswith("[") and clean_text.endswith("]"):
            # 尝试解析列表
            try:
                import ast
                items = ast.literal_eval(gold_text)
                if isinstance(items, list):
                    for item in items:
                        if item and str(item).strip():
                            total += 1
                            if str(item).lower() in parsed_lower:
                                covered += 1
                    continue
            except:
                pass
        
        # 直接匹配
        total += 1
        if clean_text in parsed_lower:
            covered += 1
    
    return covered / total if total > 0 else 1.0


def evaluate_accuracy(
    results: List[PDFParseResult],
    gold_df: pd.DataFrame,
    parsed_output_dir: str,
) -> AccuracyMetrics:
    """
    评估解析准确性
    
    Args:
        results: 解析结果
        gold_df: 金标准 DataFrame
        parsed_output_dir: 解析输出目录
    
    Returns:
        准确性指标
    """
    metrics = AccuracyMetrics()
    
    # 构建文件名到解析文本的映射
    text_dir = Path(parsed_output_dir) / "text"
    parsed_texts = {}
    
    for result in results:
        if result.success:
            text_file = text_dir / f"{Path(result.file_name).stem}.txt"
            if text_file.exists():
                with open(text_file, 'r', encoding='utf-8') as f:
                    parsed_texts[result.file_name] = f.read()
    
    # 按文件名分组金标准数据
    gold_by_file = gold_df.groupby('file_name')
    
    coverages = []
    coverage_by_type = {}
    
    for file_name, parsed_text in parsed_texts.items():
        if file_name not in gold_by_file.groups:
            continue
        
        metrics.files_matched += 1
        
        gold_rows = gold_by_file.get_group(file_name)
        
        # 获取所有金标准文本
        gold_texts = gold_rows['clause_text'].dropna().tolist()
        
        # 计算覆盖率
        coverage = compute_text_coverage(parsed_text, gold_texts)
        coverages.append(coverage)
        
        # 按条款类型统计
        for _, row in gold_rows.iterrows():
            clause_type = row.get('clause_type', 'Unknown')
            clause_text = row.get('clause_text', '')
            
            if clause_type not in coverage_by_type:
                coverage_by_type[clause_type] = {'covered': 0, 'total': 0}
            
            if clause_text and clause_text != "[]":
                coverage_by_type[clause_type]['total'] += 1
                if compute_text_coverage(parsed_text, [clause_text]) > 0.5:
                    coverage_by_type[clause_type]['covered'] += 1
    
    metrics.files_total = len(parsed_texts)
    metrics.match_rate = metrics.files_matched / metrics.files_total if metrics.files_total > 0 else 0
    
    if coverages:
        metrics.avg_text_coverage = sum(coverages) / len(coverages)
        metrics.min_text_coverage = min(coverages)
        metrics.max_text_coverage = max(coverages)
        metrics.avg_char_recall = metrics.avg_text_coverage  # 简化处理
    
    # 计算按条款类型的覆盖率
    for clause_type, stats in coverage_by_type.items():
        if stats['total'] > 0:
            metrics.coverage_by_clause_type[clause_type] = stats['covered'] / stats['total']
    
    return metrics


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(
    parsing_metrics: ParsingMetrics,
    accuracy_metrics: Optional[AccuracyMetrics],
    results: List[PDFParseResult],
    output_dir: str,
) -> str:
    """
    生成评估报告
    
    Args:
        parsing_metrics: 解析指标
        accuracy_metrics: 准确性指标
        results: 解析结果列表
        output_dir: 输出目录
    
    Returns:
        报告文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 生成 Markdown 报告
    lines = [
        "# PDF Parsing Evaluation Report",
        "",
        f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Total Files**: {parsing_metrics.total_files}",
        "",
        "## Parsing Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Success Rate | {parsing_metrics.success_rate:.2%} |",
        f"| Success Count | {parsing_metrics.success_count} |",
        f"| Failed Count | {parsing_metrics.fail_count} |",
        f"| Total Pages | {parsing_metrics.total_pages} |",
        f"| Total Text Blocks | {parsing_metrics.total_text_blocks} |",
        f"| Total Tables | {parsing_metrics.total_tables} |",
        f"| Total Characters | {parsing_metrics.total_chars:,} |",
        f"| Total Chunks | {parsing_metrics.total_chunks} |",
        "",
        "### Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Avg Pages/File | {parsing_metrics.avg_pages_per_file:.1f} |",
        f"| Avg Chunks/File | {parsing_metrics.avg_chunks_per_file:.1f} |",
        f"| Avg Chars/File | {parsing_metrics.avg_chars_per_file:,.0f} |",
        f"| Avg Parse Time | {parsing_metrics.avg_parse_time_ms:.1f} ms |",
        f"| Min Parse Time | {parsing_metrics.min_parse_time_ms:.1f} ms |",
        f"| Max Parse Time | {parsing_metrics.max_parse_time_ms:.1f} ms |",
        f"| Total Parse Time | {parsing_metrics.total_parse_time_ms/1000:.1f} s |",
        f"| Avg Confidence | {parsing_metrics.avg_confidence:.4f} |",
        "",
    ]
    
    # 准确性指标
    if accuracy_metrics:
        lines.extend([
            "## Accuracy Metrics (vs Gold Standard)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Files Matched | {accuracy_metrics.files_matched} / {accuracy_metrics.files_total} |",
            f"| Match Rate | {accuracy_metrics.match_rate:.2%} |",
            f"| Avg Text Coverage | {accuracy_metrics.avg_text_coverage:.2%} |",
            f"| Min Text Coverage | {accuracy_metrics.min_text_coverage:.2%} |",
            f"| Max Text Coverage | {accuracy_metrics.max_text_coverage:.2%} |",
            "",
        ])
        
        if accuracy_metrics.coverage_by_clause_type:
            lines.extend([
                "### Coverage by Clause Type",
                "",
                "| Clause Type | Coverage |",
                "|-------------|----------|",
            ])
            
            # 按覆盖率排序
            sorted_types = sorted(
                accuracy_metrics.coverage_by_clause_type.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for clause_type, coverage in sorted_types[:20]:  # 只显示前20个
                lines.append(f"| {clause_type} | {coverage:.2%} |")
            
            lines.append("")
    
    # 失败案例
    failed = [r for r in results if not r.success]
    if failed:
        lines.extend([
            "## Failed Cases",
            "",
            f"Total failed: {len(failed)}",
            "",
        ])
        
        for i, r in enumerate(failed[:10], 1):  # 只显示前10个
            lines.extend([
                f"### {i}. {r.file_name}",
                "",
                f"- **Error**: {r.error_message}",
                "",
            ])
    
    # 保存报告
    report_path = output_path / "parsing_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Report saved to: {report_path}")
    
    # 保存 JSON 摘要
    summary = {
        "timestamp": timestamp,
        "parsing_metrics": parsing_metrics.to_dict(),
        "accuracy_metrics": accuracy_metrics.to_dict() if accuracy_metrics else None,
    }
    
    summary_path = output_path / "parsing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Summary saved to: {summary_path}")
    
    # 保存详细结果 CSV
    results_df = pd.DataFrame([r.to_dict() for r in results])
    results_csv_path = output_path / "parsing_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    
    logger.info(f"Results saved to: {results_csv_path}")
    
    return str(report_path)


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PDF Parsing Evaluation",
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
        '--max-files',
        type=int,
        default=100,
        help='Maximum PDF files to process (default: 100 for test mode)'
    )
    
    parser.add_argument(
        '--pdf-dir',
        type=str,
        default=None,
        help=f'PDF directory (default: {DEFAULT_PDF_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for parsed results (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f'Results directory for evaluation metrics (default: {DEFAULT_RESULTS_DIR})'
    )
    
    parser.add_argument(
        '--skip-eval',
        action='store_true',
        help='Skip accuracy evaluation (only parse)'
    )
    
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='Shuffle PDF files (default: True)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Chunk size (default: from config)'
    )
    
    parser.add_argument(
        '--chunk-overlap',
        type=int,
        default=None,
        help='Chunk overlap (default: from config)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 初始化配置
    logger.info("Initializing configuration...")
    config = init_config(mode=args.mode)
    project_root = config.app.project_root
    
    # 设置全局随机种子
    seed = set_global_seed()
    logger.info(f"Global random seed: {seed}")
    
    # 确定 PDF 目录
    if args.pdf_dir:
        pdf_dir = args.pdf_dir
    else:
        pdf_dir = os.path.join(project_root, DEFAULT_PDF_DIR)
    
    logger.info(f"PDF directory: {pdf_dir}")
    
    # 调整 test 模式的参数
    max_files = args.max_files
    if args.mode == 'test' and args.max_files is None:
        max_files = 100
    elif args.mode == 'prod':
        max_files = None  # 处理所有文件
    
    # 查找 PDF 文件
    logger.info("Finding PDF files...")
    pdf_paths = find_pdf_files(
        pdf_dir,
        recursive=True,
        max_files=max_files,
        shuffle=args.shuffle,
        seed=get_seed(),
    )
    
    logger.info(f"Found {len(pdf_paths)} PDF files to process")
    
    if not pdf_paths:
        logger.error("No PDF files found!")
        return
    
    # 从配置读取分块参数（命令行参数优先）
    chunk_size = args.chunk_size if args.chunk_size is not None else config.data.chunking.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else config.data.chunking.chunk_overlap
    
    logger.info(f"Chunk config: size={chunk_size}, overlap={chunk_overlap}")
    
    # 解析 PDF
    output_dir = os.path.join(project_root, args.output_dir)
    logger.info(f"Output directory: {output_dir}")
    
    logger.info("=" * 60)
    logger.info("Starting PDF parsing...")
    logger.info("=" * 60)
    
    start_time = time.perf_counter()
    
    results = parse_pdf_batch(
        pdf_paths,
        output_dir=output_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        save_chunks=True,
        save_full_text=True,
        save_json=True,
    )
    
    total_time = time.perf_counter() - start_time
    
    logger.info("=" * 60)
    logger.info(f"Parsing completed in {total_time:.1f}s")
    logger.info("=" * 60)
    
    # 计算解析指标
    parsing_metrics = compute_parsing_metrics(results)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)
    print(f"  Total Files:    {parsing_metrics.total_files}")
    print(f"  Success:        {parsing_metrics.success_count} ({parsing_metrics.success_rate:.1%})")
    print(f"  Failed:         {parsing_metrics.fail_count}")
    print(f"  Total Pages:    {parsing_metrics.total_pages}")
    print(f"  Total Chunks:   {parsing_metrics.total_chunks}")
    print(f"  Avg Parse Time: {parsing_metrics.avg_parse_time_ms:.1f} ms")
    print(f"  Total Time:     {total_time:.1f} s")
    print("=" * 60 + "\n")
    
    # 准确性评估
    accuracy_metrics = None
    if not args.skip_eval:
        logger.info("Loading gold standard data for evaluation...")
        
        try:
            from src.data.data_loader import get_data_paths, load_data_csv
            paths = get_data_paths()
            gold_df = load_data_csv(paths['chunks'])
            
            logger.info(f"Loaded {len(gold_df)} gold standard records from {paths['chunks']}")
            
            logger.info("Evaluating accuracy...")
            accuracy_metrics = evaluate_accuracy(results, gold_df, output_dir)
            
            print("\n" + "=" * 60)
            print("ACCURACY SUMMARY")
            print("=" * 60)
            print(f"  Files Matched:      {accuracy_metrics.files_matched} / {accuracy_metrics.files_total}")
            print(f"  Avg Text Coverage:  {accuracy_metrics.avg_text_coverage:.1%}")
            print(f"  Min Text Coverage:  {accuracy_metrics.min_text_coverage:.1%}")
            print(f"  Max Text Coverage:  {accuracy_metrics.max_text_coverage:.1%}")
            print("=" * 60 + "\n")
            
        except Exception as e:
            logger.warning(f"Accuracy evaluation failed: {e}")
            logger.warning("Skipping accuracy evaluation")
    
    # 生成报告
    results_dir = os.path.join(project_root, args.results_dir)
    report_path = generate_report(
        parsing_metrics,
        accuracy_metrics,
        results,
        results_dir,
    )
    
    logger.success(f"Evaluation complete! Report: {report_path}")
    
    # 返回输出文件路径
    return {
        "output_dir": output_dir,
        "results_dir": results_dir,
        "report_path": report_path,
        "chunks_csv": os.path.join(output_dir, "all_chunks.csv"),
    }


if __name__ == "__main__":
    main()
