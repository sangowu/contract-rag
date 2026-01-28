"""
CUAD 数据加载器

功能:
- 加载原始数据 (CSV/JSON)
- 生成 chunk 记录
- 生成金标准答案 (支持 RAGAS 评估)
- 支持数据采样 (测试模式)
- RAGAS 格式转换
"""

import json
import os
import ast
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from typing import Optional, List, Dict, Any

# 配置导入
from src.core.config import get_config

# 工具导入
from src.utils.query_builder import build_query


# =============================================================================
# 数据加载函数
# =============================================================================

def load_data_csv(data_path: str) -> pd.DataFrame:
    """加载 CSV 数据文件"""
    df = pd.read_csv(data_path)
    return df


def load_data_json(data_path: str) -> pd.DataFrame:
    """加载 JSON 数据文件"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


def get_data_paths() -> Dict[str, str]:
    """
    从配置获取数据路径
    
    Returns:
        dict: 包含所有数据路径的字典
    """
    config = get_config()
    
    # 确保目录存在
    processed_path = Path(config.data.processed_path)
    answers_path = Path(config.data.answers_path)
    processed_path.mkdir(parents=True, exist_ok=True)
    answers_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'raw_path': config.data.raw_path,
        'processed_path': str(processed_path),
        'answers_path': str(answers_path),
        'master_clauses': str(config.data.master_clauses_path),
        'chunks': str(config.data.chunks_path),
        'gold_answers': str(config.data.gold_answers_path),
    }


def sample_data(df: pd.DataFrame, ratio: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """
    数据采样 (用于测试模式)
    
    Args:
        df: 原始数据
        ratio: 采样比例 (0.0-1.0)
        seed: 随机种子
    
    Returns:
        采样后的数据
    """
    if ratio >= 1.0:
        return df
    
    n_samples = max(1, int(len(df) * ratio))
    sampled = df.sample(n=n_samples, random_state=seed)
    logger.info(f"Sampled {n_samples}/{len(df)} rows ({ratio*100:.1f}%)")
    return sampled.reset_index(drop=True)


# =============================================================================
# 答案类型推断
# =============================================================================

# Boolean 类型的 clause types
BOOLEAN_CLAUSE_TYPES = {
    'Most Favored Nation',
    'Competitive Restriction Exception',
    'Non-Compete',
    'Exclusivity',
    'No-Solicit Of Customers',
    'No-Solicit Of Employees',
    'Non-Disparagement',
    'Termination For Convenience',
    'Rofr/Rofo/Rofn',
    'Change Of Control',
    'Anti-Assignment',
    'Revenue/Profit Sharing',
    'Price Restrictions',
    'Minimum Commitment',
    'Volume Restriction',
    'Ip Ownership Assignment',
    'Joint Ip Ownership',
    'License Grant',
    'Non-Transferable License',
    'Affiliate License-Licensor',
    'Affiliate License-Licensee',
    'Unlimited/All-You-Can-Eat-License',
    'Irrevocable Or Perpetual License',
    'Source Code Escrow',
    'Post-Termination Services',
    'Audit Rights',
    'Uncapped Liability',
    'Cap On Liability',
    'Liquidated Damages',
    'Warranty Duration',
    'Insurance',
    'Covenant Not To Sue',
    'Third Party Beneficiary',
}

# 日期类型的 clause types
DATE_CLAUSE_TYPES = {
    'Agreement Date',
    'Effective Date',
    'Expiration Date',
}

# 实体/名称类型
ENTITY_CLAUSE_TYPES = {
    'Document Name',
    'Governing Law',
}

# 列表类型 (多个值)
LIST_CLAUSE_TYPES = {
    'Parties',
}


def infer_answer_type(clause_type: str, answer_text: str) -> str:
    """
    推断答案类型
    
    Args:
        clause_type: 条款类型
        answer_text: 答案文本
    
    Returns:
        答案类型: 'boolean', 'date', 'entity', 'list', 'extractive', 'duration'
    """
    # 规范化答案
    answer_lower = answer_text.lower().strip()
    
    # 1. Boolean 类型
    if clause_type in BOOLEAN_CLAUSE_TYPES:
        return 'boolean'
    
    # 也检查答案本身是否是 Yes/No
    if answer_lower in ('yes', 'no'):
        return 'boolean'
    
    # 2. 日期类型
    if clause_type in DATE_CLAUSE_TYPES:
        return 'date'
    
    # 3. 实体类型
    if clause_type in ENTITY_CLAUSE_TYPES:
        return 'entity'
    
    # 4. 列表类型
    if clause_type in LIST_CLAUSE_TYPES:
        return 'list'
    
    # 5. 时长类型 (Renewal Term, Notice Period 等)
    if 'term' in clause_type.lower() or 'period' in clause_type.lower():
        return 'duration'
    
    # 6. 默认为抽取式
    return 'extractive'


# =============================================================================
# Parent-Child 分块配置
# =============================================================================

# 默认分块参数
DEFAULT_CHILD_CHUNK_SIZE = 500   # Child chunk 大小
DEFAULT_CHUNK_OVERLAP = 100      # Child chunk 重叠
DEFAULT_MIN_CHUNK_SIZE = 50      # 最小 chunk 大小（避免过短的片段）


# =============================================================================
# Chunk 记录处理器
# =============================================================================

class RawToChunkRecordsProcessor:
    """
    将原始数据转换为 chunk 记录
    
    支持 Parent-Child 分块策略:
    - 短文本 (< child_chunk_size): parent_clause_text == clause_text
    - 长文本 (>= child_chunk_size): 切分成多个 child chunks，共享同一个 parent_clause_text
    """
    
    def __init__(
        self,
        df_raw: pd.DataFrame,
        output_path: Optional[str] = None,
        child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        enable_parent_child: bool = True,
    ):
        """
        Args:
            df_raw: 原始数据 DataFrame
            output_path: 输出路径 (可选，默认从配置读取)
            child_chunk_size: Child chunk 大小 (默认 500)
            chunk_overlap: Child chunk 重叠 (默认 100)
            enable_parent_child: 是否启用 Parent-Child 分块 (默认 True)
        """
        self.df_raw = df_raw
        self.df_chunks = df_raw.copy()
        self.answer_columns = [c for c in self.df_chunks.columns if c.endswith('-Answer')]
        self.df_chunks.drop(self.answer_columns, axis=1, inplace=True)
        self.clause_text_cols = [c for c in self.df_chunks.columns if c not in ['Filename', 'Unnamed: 0']]
        self.records = []
        
        # Parent-Child 分块参数
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_parent_child = enable_parent_child
        
        # 输出路径
        if output_path:
            self.output_path = output_path
        else:
            paths = get_data_paths()
            self.output_path = paths['chunks']
    
    def _split_text_into_chunks(self, text: str) -> List[tuple]:
        """
        将长文本切分成多个 child chunks
        
        Args:
            text: 原始文本
        
        Returns:
            List of (start_char, end_char, chunk_text) tuples
        """
        text_str = str(text)
        text_len = len(text_str)
        
        # 短文本不需要切分
        if text_len <= self.child_chunk_size:
            return [(0, text_len, text_str)]
        
        chunks = []
        start = 0
        
        while start < text_len:
            end = min(start + self.child_chunk_size, text_len)
            chunk_text = text_str[start:end].strip()
            
            # 跳过过短的片段
            if len(chunk_text) >= DEFAULT_MIN_CHUNK_SIZE:
                chunks.append((start, end, chunk_text))
            
            # 如果已经到达末尾，停止
            if end >= text_len:
                break
            
            # 下一个 chunk 的起始位置（考虑重叠）
            start = max(0, end - self.chunk_overlap)
            
            # 防止无限循环
            if start >= text_len:
                break
        
        return chunks if chunks else [(0, text_len, text_str)]

    def process(self) -> pd.DataFrame:
        """
        处理数据并保存 chunk 记录
        
        Parent-Child 分块策略:
        - 每个原始文本作为 parent_clause_text
        - 长文本切分成多个 child chunks
        - 每个 child 的 clause_text 是切分后的片段
        - 所有 child 共享同一个 parent_clause_text
        """
        parent_count = 0
        child_count = 0
        
        for idx, row in tqdm(self.df_chunks.iterrows(), total=len(self.df_chunks), desc="Processing chunks"):
            filename = row['Filename']

            for col in self.clause_text_cols:
                text = row[col]
                if pd.isna(text):
                    continue
                
                # 检查是否有答案
                ans_col = f"{col}-Answer"
                if ans_col in self.df_raw.columns:
                    ans_val = self.df_raw.loc[idx, ans_col]
                    has_answer = pd.notna(ans_val) and str(ans_val).strip() != '' and str(ans_val).lower() != "not present"
                else:
                    has_answer = False
                
                # 原始完整文本作为 parent
                parent_text = str(text)
                parent_count += 1
                
                # 决定是否进行 Parent-Child 分块
                if self.enable_parent_child:
                    child_chunks = self._split_text_into_chunks(text)
                else:
                    # 不启用时，parent == child
                    child_chunks = [(0, len(parent_text), parent_text)]
                
                # 为每个 child chunk 创建记录
                for chunk_idx, (start_char, end_char, chunk_text) in enumerate(child_chunks):
                    chunk_id = f"{filename}::{col}::{idx}::{chunk_idx}::{start_char}:{end_char}"
                    
                    self.records.append({
                        'file_name': filename,
                        'chunk_id': chunk_id,
                        'clause_type': col,
                        'clause_text': chunk_text,           # Child chunk 文本
                        'parent_clause_text': parent_text,   # Parent 完整文本
                        'contract_idx': idx,
                        'chunk_idx': chunk_idx,
                        'start_char': start_char,
                        'end_char': end_char,
                        'has_answer': has_answer,
                        'source': 'master_clauses',
                    })
                    child_count += 1

        df_records = pd.DataFrame(self.records)
        df_records.to_csv(self.output_path, index=False)
        
        # 统计信息
        logger.info(f"Saved {len(df_records)} chunks to {self.output_path}")
        logger.info(f"  - Parent texts: {parent_count}")
        logger.info(f"  - Child chunks: {child_count}")
        logger.info(f"  - Avg children per parent: {child_count/parent_count:.2f}" if parent_count > 0 else "")
        
        # 检查 Parent-Child 结构
        diff_count = (df_records['clause_text'] != df_records['parent_clause_text']).sum()
        diff_ratio = diff_count / len(df_records) * 100 if len(df_records) > 0 else 0
        logger.info(f"  - Parent != Child: {diff_count} ({diff_ratio:.1f}%)")

        return df_records


# =============================================================================
# 金标准答案处理器
# =============================================================================

class RawToGoldAnswersProcessor:
    """将原始数据转换为金标准答案 (支持 RAGAS 评估)"""
    
    def __init__(self, df_raw: pd.DataFrame, chunk_df: pd.DataFrame, output_path: Optional[str] = None):
        """
        Args:
            df_raw: 原始数据 DataFrame
            chunk_df: chunk 记录 DataFrame
            output_path: 输出路径 (可选，默认从配置读取)
        """
        self.df_raw = df_raw
        self.chunk_df = chunk_df
        self.df_gold_answers = df_raw.copy()
        self.answer_columns = [c for c in self.df_gold_answers.columns if c.endswith('-Answer')]
        self.clause_text_cols = [c for c in self.df_gold_answers.columns if c not in ['Filename'] and c not in self.answer_columns]
        self.records = []
        
        # 输出路径
        if output_path:
            self.output_path = output_path
        else:
            paths = get_data_paths()
            self.output_path = paths['gold_answers']
        
        # 构建 chunk_id -> clause_text 的映射 (child text)
        self._chunk_text_map = dict(zip(
            self.chunk_df['chunk_id'].astype(str),
            self.chunk_df['clause_text'].astype(str)
        ))
        
        # 构建 chunk_id -> parent_clause_text 的映射 (parent text)
        if 'parent_clause_text' in self.chunk_df.columns:
            self._parent_text_map = dict(zip(
                self.chunk_df['chunk_id'].astype(str),
                self.chunk_df['parent_clause_text'].astype(str)
            ))
        else:
            self._parent_text_map = self._chunk_text_map

    def _get_context_text(self, chunk_ids: List[str], use_parent: bool = True) -> str:
        """
        根据 chunk IDs 获取上下文文本
        
        Args:
            chunk_ids: chunk ID 列表
            use_parent: 是否使用 parent_clause_text (默认 True)
        
        Returns:
            合并的上下文文本 (已去重)
        """
        text_map = self._parent_text_map if use_parent else self._chunk_text_map
        
        # 使用集合去重（多个 child 可能共享同一个 parent）
        unique_texts = []
        seen_texts = set()
        
        for cid in chunk_ids:
            text = text_map.get(cid, "")
            if text and text not in seen_texts:
                unique_texts.append(text)
                seen_texts.add(text)
        
        return "\n\n".join(unique_texts)

    def process(self) -> pd.DataFrame:
        """处理数据并保存金标准答案"""
        for idx, row in tqdm(self.df_gold_answers.iterrows(), total=len(self.df_gold_answers), desc="Processing gold answers"):
            filename = row['Filename']
 
            for col_ans in self.answer_columns:
                ans_val = row[col_ans]
                if pd.isna(ans_val) or str(ans_val).strip() == "" or str(ans_val).lower() == "not present":
                    continue
                gold_answer_text = str(ans_val).strip()
                clause_type = col_ans.replace('-Answer', '')

                mask = (
                    (self.chunk_df['file_name'] == filename) &
                    (self.chunk_df['clause_type'] == clause_type) &
                    (self.chunk_df['contract_idx'] == idx)
                )
                matched_chunk_ids = self.chunk_df.loc[mask, 'chunk_id'].tolist()

                if not matched_chunk_ids:
                    logger.warning(f"No chunk found for {filename}::{clause_type}::{idx}")
                    continue

                matched_chunk_ids.sort(key=lambda x: int(x.split('::')[3]) if len(x.split('::')) >= 4 else 0)
                sample_id = f"{filename}::{clause_type}::{idx}"
                query = build_query(clause_type)
                
                # 获取参考上下文文本 (RAGAS 需要)
                gold_context_text = self._get_context_text(matched_chunk_ids)
                
                # 推断答案类型
                answer_type = infer_answer_type(clause_type, gold_answer_text)

                self.records.append({
                    'sample_id': sample_id,
                    'file_name': filename,
                    'gold_chunk_ids': json.dumps(matched_chunk_ids),  # JSON 格式
                    'clause_type': clause_type,
                    'query': query,
                    'gold_answer_text': gold_answer_text,
                    'gold_context_text': gold_context_text,  # 新增: 参考上下文
                    'answer_type': answer_type,              # 新增: 答案类型
                    'contract_idx': idx,
                    'source': 'master_clauses',
                })
                
        df_gold = pd.DataFrame(self.records)
        df_gold.to_csv(self.output_path, index=False)
        logger.info(f"Saved {len(df_gold)} gold answers to {self.output_path}")
        
        # 打印答案类型统计
        type_counts = df_gold['answer_type'].value_counts()
        logger.info(f"Answer type distribution:\n{type_counts.to_string()}")
        
        return df_gold


# =============================================================================
# RAGAS 格式转换
# =============================================================================

def load_gold_answers(path: Optional[str] = None) -> pd.DataFrame:
    """
    加载金标准答案
    
    Args:
        path: 文件路径 (可选，默认从配置读取)
    
    Returns:
        金标准答案 DataFrame
    """
    if path is None:
        paths = get_data_paths()
        path = paths['gold_answers']
    
    df = pd.read_csv(path)
    
    # 解析 JSON 格式的 gold_chunk_ids
    if 'gold_chunk_ids' in df.columns:
        def parse_chunk_ids(x):
            if pd.isna(x):
                return []
            try:
                # 尝试 JSON 解析
                return json.loads(x)
            except json.JSONDecodeError:
                # 回退到 ast.literal_eval (兼容旧格式)
                try:
                    return ast.literal_eval(x)
                except:
                    return []
        
        df['gold_chunk_ids'] = df['gold_chunk_ids'].apply(parse_chunk_ids)
    
    return df


def convert_to_ragas_format(
    gold_answers_df: pd.DataFrame,
    model_outputs: Optional[Dict[str, List]] = None
) -> Dict[str, List]:
    """
    将金标准答案转换为 RAGAS 评估格式
    
    Args:
        gold_answers_df: 金标准答案 DataFrame
        model_outputs: 模型输出 (可选)
            - 'answers': 模型生成的答案列表
            - 'contexts': 检索到的上下文列表 (每个元素是 List[str])
    
    Returns:
        RAGAS 格式的字典:
        {
            'question': List[str],
            'ground_truth': List[str],
            'reference_contexts': List[List[str]],  # RAGAS Context Recall 需要
            'answer': List[str],                     # 模型生成 (可选)
            'contexts': List[List[str]],             # 检索结果 (可选)
        }
    
    Example:
        >>> from datasets import Dataset
        >>> ragas_data = convert_to_ragas_format(gold_df, model_outputs)
        >>> dataset = Dataset.from_dict(ragas_data)
        >>> from ragas import evaluate
        >>> results = evaluate(dataset, metrics=[...])
    """
    ragas_data = {
        'question': gold_answers_df['query'].tolist(),
        'ground_truth': gold_answers_df['gold_answer_text'].tolist(),
    }
    
    # 添加参考上下文 (如果存在)
    if 'gold_context_text' in gold_answers_df.columns:
        # RAGAS 需要 List[List[str]] 格式
        ragas_data['reference_contexts'] = [
            [ctx] if ctx else [] 
            for ctx in gold_answers_df['gold_context_text'].tolist()
        ]
    
    # 添加模型输出 (如果提供)
    if model_outputs:
        if 'answers' in model_outputs:
            ragas_data['answer'] = model_outputs['answers']
        if 'contexts' in model_outputs:
            ragas_data['contexts'] = model_outputs['contexts']
    
    return ragas_data


def load_gold_answers_for_ragas(
    path: Optional[str] = None,
    sample_ratio: Optional[float] = None
) -> Dict[str, List]:
    """
    加载金标准答案并转换为 RAGAS 格式
    
    Args:
        path: 文件路径 (可选)
        sample_ratio: 采样比例 (可选)
    
    Returns:
        RAGAS 格式的数据字典
    
    Example:
        >>> ragas_data = load_gold_answers_for_ragas(sample_ratio=0.1)
        >>> print(f"Questions: {len(ragas_data['question'])}")
    """
    df = load_gold_answers(path)
    
    # 数据采样
    if sample_ratio and sample_ratio < 1.0:
        config = get_config()
        df = sample_data(df, ratio=sample_ratio, seed=config.data.sampling.seed)
    
    return convert_to_ragas_format(df)


def get_answer_type_groups(gold_answers_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    按答案类型分组
    
    Args:
        gold_answers_df: 金标准答案 DataFrame
    
    Returns:
        按答案类型分组的字典
    
    Example:
        >>> groups = get_answer_type_groups(df)
        >>> boolean_df = groups['boolean']
        >>> extractive_df = groups['extractive']
    """
    if 'answer_type' not in gold_answers_df.columns:
        logger.warning("answer_type column not found, returning empty groups")
        return {}
    
    return {
        answer_type: group_df 
        for answer_type, group_df in gold_answers_df.groupby('answer_type')
    }


# =============================================================================
# 主处理函数
# =============================================================================

def process_cuad_data(
    sample_ratio: Optional[float] = None,
    enable_parent_child: bool = True,
    child_chunk_size: int = DEFAULT_CHILD_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple:
    """
    处理 CUAD 数据集的主函数
    
    Args:
        sample_ratio: 采样比例 (可选，默认从配置读取)
        enable_parent_child: 是否启用 Parent-Child 分块 (默认 True)
        child_chunk_size: Child chunk 大小 (默认 500)
        chunk_overlap: Child chunk 重叠 (默认 100)
    
    Returns:
        (chunk_records, gold_answers) 元组
    """
    config = get_config()
    paths = get_data_paths()
    
    # 确定采样比例
    if sample_ratio is None:
        if config.data.sampling.enabled:
            sample_ratio = config.data.sampling.ratio
        else:
            sample_ratio = 1.0
    
    seed = config.data.sampling.seed
    
    logger.info(f"Loading raw data from {paths['master_clauses']}...")
    df_raw = load_data_csv(paths['master_clauses'])
    
    # 数据采样 (测试模式)
    if sample_ratio < 1.0:
        df_raw = sample_data(df_raw, ratio=sample_ratio, seed=seed)

    logger.info("Building CUAD_v1 chunk records...")
    logger.info(f"  - Parent-Child enabled: {enable_parent_child}")
    logger.info(f"  - Child chunk size: {child_chunk_size}")
    logger.info(f"  - Chunk overlap: {chunk_overlap}")
    
    chunk_records = RawToChunkRecordsProcessor(
        df_raw,
        child_chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
        enable_parent_child=enable_parent_child,
    ).process()
    
    logger.info("Building CUAD_v1 gold answers...")
    gold_answers = RawToGoldAnswersProcessor(df_raw, chunk_records).process()

    logger.info("Done!")
    return chunk_records, gold_answers


# =============================================================================
# 命令行入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    from src.core.config import init_config
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="CUAD Data Loader with Parent-Child Support")
    parser.add_argument('--mode', type=str, default='prod', choices=['test', 'dev', 'prod'],
                        help='Environment mode')
    parser.add_argument('--sample-ratio', type=float, default=None,
                        help='Override sampling ratio')
    parser.add_argument('--check-ragas', action='store_true',
                        help='Check RAGAS format conversion')
    
    # Parent-Child 分块参数
    parser.add_argument('--enable-parent-child', action='store_true', default=True,
                        help='Enable Parent-Child chunking (default: True)')
    parser.add_argument('--disable-parent-child', action='store_true',
                        help='Disable Parent-Child chunking')
    parser.add_argument('--child-chunk-size', type=int, default=DEFAULT_CHILD_CHUNK_SIZE,
                        help=f'Child chunk size (default: {DEFAULT_CHILD_CHUNK_SIZE})')
    parser.add_argument('--chunk-overlap', type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f'Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})')
    
    args = parser.parse_args()
    
    # 确定是否启用 Parent-Child
    enable_parent_child = not args.disable_parent_child
    
    # 初始化配置
    init_config(mode=args.mode)
    
    # 处理数据
    chunk_records, gold_answers = process_cuad_data(
        sample_ratio=args.sample_ratio,
        enable_parent_child=enable_parent_child,
        child_chunk_size=args.child_chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    # 检查 RAGAS 格式
    if args.check_ragas:
        logger.info("\n=== RAGAS Format Check ===")
        ragas_data = convert_to_ragas_format(gold_answers)
        logger.info(f"Questions: {len(ragas_data['question'])}")
        logger.info(f"Ground truths: {len(ragas_data['ground_truth'])}")
        if 'reference_contexts' in ragas_data:
            logger.info(f"Reference contexts: {len(ragas_data['reference_contexts'])}")
        
        # 显示示例
        logger.info("\n=== Sample Data ===")
        for i in range(min(3, len(ragas_data['question']))):
            logger.info(f"\n[{i+1}]")
            logger.info(f"  Question: {ragas_data['question'][i][:100]}...")
            logger.info(f"  Ground Truth: {ragas_data['ground_truth'][i][:100]}...")
            if 'reference_contexts' in ragas_data and ragas_data['reference_contexts'][i]:
                ctx = ragas_data['reference_contexts'][i][0]
                logger.info(f"  Context: {ctx[:100]}...")
