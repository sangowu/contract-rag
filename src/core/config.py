"""
CUAD 合同助手 - 配置管理器

功能:
1. 单例模式 - 全局只加载一次
2. 支持 --mode 参数 (test/dev/prod)
3. 支持配置合并 (base + env)
4. 支持环境变量替换 ${VAR} 和 ${VAR:default}
5. 提供类型安全的属性访问
6. 路径自动解析

使用方式:
    from src.core.config import get_config, init_config
    
    # 初始化 (通常在入口脚本中调用一次)
    init_config(mode="test")
    
    # 获取配置
    config = get_config()
    print(config.models.llm.name)
    print(config.data.raw_path)
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field
from copy import deepcopy


# =============================================================================
# 配置数据类
# =============================================================================

@dataclass
class LLMModelConfig:
    """单个 LLM 模型配置"""
    name: str = ""
    path: str = ""


@dataclass
class LLMGPUUtilConfig:
    """LLM GPU 利用率配置"""
    single_gpu: float = 0.45
    dual_gpu: float = 0.85


@dataclass
class LLMConfig:
    """LLM 配置（支持多模型）"""
    # 默认模型（双 GPU 使用）
    default: LLMModelConfig = field(default_factory=LLMModelConfig)
    # 备用模型（单 GPU 使用）
    fallback: LLMModelConfig = field(default_factory=LLMModelConfig)
    # 通用参数
    max_tokens: int = 4096
    temperature: float = 0.1
    max_model_len: int = 8192
    # GPU 利用率
    gpu_memory_utilization: LLMGPUUtilConfig = field(default_factory=LLMGPUUtilConfig)
    
    # 兼容性属性
    @property
    def name(self) -> str:
        """兼容旧代码"""
        return self.default.name
    
    @property
    def path(self) -> str:
        """兼容旧代码"""
        return self.default.path


@dataclass
class EmbeddingConfig:
    name: str = ""
    path: str = ""
    batch_size: int = 32
    dimension: int = 384


@dataclass
class RerankerConfig:
    name: str = ""
    path: str = ""
    top_k: int = 20


@dataclass
class ModelsConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)


@dataclass
class SamplingConfig:
    enabled: bool = False
    ratio: float = 1.0
    seed: int = 42
    cache_enabled: bool = False


@dataclass
class FilesConfig:
    master_clauses: str = "master_clauses.csv"
    chunks: str = "cuad_v1_chunks.csv"
    gold_answers: str = "cuad_v1_gold_answers.csv"


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100


@dataclass
class PDFParsedConfig:
    """PDF 解析数据配置"""
    enabled: bool = True  # 默认启用 PDF 解析数据
    chunks_path: str = ""  # PDF 解析后的 chunks 路径


@dataclass
class DataConfig:
    base_dir: str = ""
    raw_path: str = ""
    processed_path: str = ""
    answers_path: str = ""
    gold_standard_path: str = ""
    cache_path: str = ""
    files: FilesConfig = field(default_factory=FilesConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    pdf_parsed: PDFParsedConfig = field(default_factory=PDFParsedConfig)
    
    # 便捷属性：完整文件路径
    @property
    def master_clauses_path(self) -> Path:
        return Path(self.raw_path) / self.files.master_clauses
    
    @property
    def chunks_path(self) -> Path:
        """获取 chunks 路径（优先使用 PDF 解析数据）"""
        # 如果启用 PDF 解析数据且路径存在，优先使用
        if self.pdf_parsed.enabled and self.pdf_parsed.chunks_path:
            pdf_path = Path(self.pdf_parsed.chunks_path)
            if pdf_path.exists():
                return pdf_path
        # 否则使用原始预处理数据
        return Path(self.processed_path) / self.files.chunks
    
    @property
    def original_chunks_path(self) -> Path:
        """获取原始预处理 chunks 路径（用于金标准评估）"""
        return Path(self.processed_path) / self.files.chunks
    
    @property
    def gold_answers_path(self) -> Path:
        return Path(self.answers_path) / self.files.gold_answers


@dataclass
class VectorDBConfig:
    type: str = "chroma"
    persist_directory: str = ""
    collection_name: str = "contracts"


@dataclass
class BM25Config:
    index_path: str = ""
    index_file: str = "bm25_pdf_index.pkl"  # 默认使用 PDF 索引


@dataclass
class HybridConfig:
    enabled: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k: int = 50


@dataclass
class ParentChildConfig:
    enabled: bool = True
    parent_chunk_size: int = 2000
    child_chunk_size: int = 500
    overlap: int = 100


@dataclass
class RerankConfig:
    enabled: bool = True
    top_k: int = 10


@dataclass
class RetrievalConfig:
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    parent_child: ParentChildConfig = field(default_factory=ParentChildConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)


@dataclass
class SemanticCacheConfig:
    enabled: bool = False
    backend: str = "redis"
    similarity_threshold: float = 0.95
    ttl_seconds: int = 86400
    max_entries: int = 10000


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""


@dataclass
class CacheConfig:
    semantic: SemanticCacheConfig = field(default_factory=SemanticCacheConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class RAGASConfig:
    enabled: bool = True
    metrics: list = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    ])


@dataclass
class TraditionalMetricsConfig:
    metrics: list = field(default_factory=lambda: [
        "f1", "exact_match", "recall_at_k", "mrr", "hit_at_k"
    ])
    k_values: list = field(default_factory=lambda: [1, 3, 5, 10, 20])


@dataclass
class WandBConfig:
    enabled: bool = False
    project: str = "cuad-assistant"
    entity: str = ""


@dataclass
class OutputConfig:
    results_dir: str = ""
    csv_dir: str = ""
    plots_dir: str = ""
    reports_dir: str = ""


@dataclass
class EvaluationConfig:
    ragas: RAGASConfig = field(default_factory=RAGASConfig)
    traditional: TraditionalMetricsConfig = field(default_factory=TraditionalMetricsConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = ""
    console: bool = True


# -----------------------------------------------------------------------------
# PDF 配置
# -----------------------------------------------------------------------------

@dataclass
class PDFParserConfig:
    backend: str = "pymupdf"
    max_pages: int = 500
    max_file_size_mb: int = 100
    timeout_seconds: int = 300


@dataclass
class PDFOCRConfig:
    enabled: bool = True
    engine: str = "paddleocr"
    languages: list = field(default_factory=lambda: ["en", "ch"])
    confidence_threshold: float = 0.6
    fallback_on_failure: bool = True


@dataclass
class PDFTableConfig:
    enabled: bool = True
    max_tokens: int = 2000
    summary_enabled: bool = True
    context_chars: int = 100


@dataclass
class PDFBBoxConfig:
    enabled: bool = True
    granularity: str = "line"
    include_images: bool = False


@dataclass
class PDFOutputConfig:
    format: str = "markdown"
    preserve_layout: bool = True


@dataclass
class PDFConfig:
    parser: PDFParserConfig = field(default_factory=PDFParserConfig)
    ocr: PDFOCRConfig = field(default_factory=PDFOCRConfig)
    table: PDFTableConfig = field(default_factory=PDFTableConfig)
    bbox: PDFBBoxConfig = field(default_factory=PDFBBoxConfig)
    output: PDFOutputConfig = field(default_factory=PDFOutputConfig)


@dataclass
class AppConfig:
    name: str = "CUAD Assistant"
    version: str = "1.0.0"
    project_root: str = ""


@dataclass
class SeedConfig:
    """全局随机数种子配置，确保实验可复现"""
    global_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    python_seed: int = 42
    cuda_deterministic: bool = True


# =============================================================================
# 主配置类
# =============================================================================

@dataclass
class Config:
    """主配置类，包含所有配置项"""
    app: AppConfig = field(default_factory=AppConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)  # 全局随机种子配置
    models: ModelsConfig = field(default_factory=ModelsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)  # 新增 PDF 配置
    
    # 元信息
    _mode: str = field(default="prod", repr=False)
    _config_dir: str = field(default="", repr=False)


# =============================================================================
# 配置加载器
# =============================================================================

class ConfigLoader:
    """配置加载器 - 单例模式"""
    
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        pass
    
    def load(self, mode: str = "prod", config_dir: Optional[str] = None) -> Config:
        """
        加载配置
        
        Args:
            mode: 环境模式 (test/dev/prod)
            config_dir: 配置目录路径，默认为项目根目录下的 config/
        
        Returns:
            Config 对象
        """
        # 确定配置目录
        if config_dir is None:
            # 默认：项目根目录/config
            project_root = self._find_project_root()
            config_dir = os.path.join(project_root, "config")
        
        self._config_dir = config_dir
        self._project_root = self._find_project_root()
        
        # 加载 base.yaml
        base_path = os.path.join(config_dir, "base.yaml")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base config not found: {base_path}")
        
        base_config = self._load_yaml(base_path)
        
        # 加载环境配置并合并
        env_path = os.path.join(config_dir, f"{mode}.yaml")
        if os.path.exists(env_path):
            env_config = self._load_yaml(env_path)
            merged_config = self._deep_merge(base_config, env_config)
        else:
            merged_config = base_config
        
        # 替换环境变量
        merged_config = self._substitute_env_vars(merged_config)
        
        # 转换为 Config 对象
        self._config = self._dict_to_config(merged_config, mode, config_dir)
        
        return self._config
    
    def get(self) -> Config:
        """获取当前配置"""
        if self._config is None:
            raise RuntimeError("Config not initialized. Call load() first or use init_config().")
        return self._config
    
    def _find_project_root(self) -> str:
        """查找项目根目录（包含 config/ 目录的位置）"""
        current = os.path.dirname(os.path.abspath(__file__))
        
        # 向上查找，直到找到 config/ 目录
        for _ in range(10):  # 最多向上10层
            if os.path.exists(os.path.join(current, "config")):
                return current
            parent = os.path.dirname(current)
            if parent == current:  # 已到根目录
                break
            current = parent
        
        # 回退到默认值
        return "/root/autodl-tmp"
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """加载 YAML 文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并两个字典"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """递归替换环境变量"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._replace_var(config)
        else:
            return config
    
    def _replace_var(self, value: str) -> str:
        """替换字符串中的变量"""
        # 模式: ${VAR} 或 ${VAR:default}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            
            # 特殊变量
            if var_name == "PROJECT_ROOT":
                return self._project_root
            
            # 环境变量
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            elif default is not None:
                return default
            else:
                return match.group(0)  # 保持原样
        
        return re.sub(pattern, replacer, value)
    
    def _dict_to_config(self, d: Dict, mode: str, config_dir: str) -> Config:
        """将字典转换为 Config 对象"""
        config = Config()
        config._mode = mode
        config._config_dir = config_dir
        
        # App
        if 'app' in d:
            config.app = AppConfig(**{k: v for k, v in d['app'].items() if k in AppConfig.__dataclass_fields__})
        
        # Models
        if 'models' in d:
            m = d['models']
            if 'llm' in m:
                llm_dict = m['llm']
                config.models.llm = LLMConfig()
                
                # 解析 default 和 fallback
                if 'default' in llm_dict:
                    config.models.llm.default = LLMModelConfig(**llm_dict['default'])
                if 'fallback' in llm_dict:
                    config.models.llm.fallback = LLMModelConfig(**llm_dict['fallback'])
                
                # 解析通用参数
                if 'max_tokens' in llm_dict:
                    config.models.llm.max_tokens = llm_dict['max_tokens']
                if 'temperature' in llm_dict:
                    config.models.llm.temperature = llm_dict['temperature']
                if 'max_model_len' in llm_dict:
                    config.models.llm.max_model_len = llm_dict['max_model_len']
                
                # 解析 GPU 利用率
                if 'gpu_memory_utilization' in llm_dict:
                    gpu_util = llm_dict['gpu_memory_utilization']
                    config.models.llm.gpu_memory_utilization = LLMGPUUtilConfig(
                        single_gpu=gpu_util.get('single_gpu', 0.45),
                        dual_gpu=gpu_util.get('dual_gpu', 0.85),
                    )
                
            if 'embedding' in m:
                config.models.embedding = EmbeddingConfig(**{k: v for k, v in m['embedding'].items() if k in EmbeddingConfig.__dataclass_fields__})
            if 'reranker' in m:
                config.models.reranker = RerankerConfig(**{k: v for k, v in m['reranker'].items() if k in RerankerConfig.__dataclass_fields__})
        
        # Data
        if 'data' in d:
            data = d['data']
            data_fields = {k: v for k, v in data.items() if k in DataConfig.__dataclass_fields__ and k not in ['files', 'chunking', 'sampling', 'pdf_parsed']}
            config.data = DataConfig(**data_fields)
            
            if 'files' in data:
                config.data.files = FilesConfig(**{k: v for k, v in data['files'].items() if k in FilesConfig.__dataclass_fields__})
            if 'chunking' in data:
                config.data.chunking = ChunkingConfig(**{k: v for k, v in data['chunking'].items() if k in ChunkingConfig.__dataclass_fields__})
            if 'sampling' in data:
                config.data.sampling = SamplingConfig(**{k: v for k, v in data['sampling'].items() if k in SamplingConfig.__dataclass_fields__})
            if 'pdf_parsed' in data:
                config.data.pdf_parsed = PDFParsedConfig(**{k: v for k, v in data['pdf_parsed'].items() if k in PDFParsedConfig.__dataclass_fields__})
        
        # Retrieval
        if 'retrieval' in d:
            r = d['retrieval']
            if 'vector_db' in r:
                config.retrieval.vector_db = VectorDBConfig(**{k: v for k, v in r['vector_db'].items() if k in VectorDBConfig.__dataclass_fields__})
            if 'bm25' in r:
                config.retrieval.bm25 = BM25Config(**{k: v for k, v in r['bm25'].items() if k in BM25Config.__dataclass_fields__})
            if 'hybrid' in r:
                config.retrieval.hybrid = HybridConfig(**{k: v for k, v in r['hybrid'].items() if k in HybridConfig.__dataclass_fields__})
            if 'parent_child' in r:
                config.retrieval.parent_child = ParentChildConfig(**{k: v for k, v in r['parent_child'].items() if k in ParentChildConfig.__dataclass_fields__})
            if 'rerank' in r:
                config.retrieval.rerank = RerankConfig(**{k: v for k, v in r['rerank'].items() if k in RerankConfig.__dataclass_fields__})
        
        # Cache
        if 'cache' in d:
            c = d['cache']
            if 'semantic' in c:
                config.cache.semantic = SemanticCacheConfig(**{k: v for k, v in c['semantic'].items() if k in SemanticCacheConfig.__dataclass_fields__})
            if 'redis' in c:
                config.cache.redis = RedisConfig(**{k: v for k, v in c['redis'].items() if k in RedisConfig.__dataclass_fields__})
        
        # API
        if 'api' in d:
            config.api = APIConfig(**{k: v for k, v in d['api'].items() if k in APIConfig.__dataclass_fields__})
        
        # Evaluation
        if 'evaluation' in d:
            e = d['evaluation']
            if 'ragas' in e:
                config.evaluation.ragas = RAGASConfig(**{k: v for k, v in e['ragas'].items() if k in RAGASConfig.__dataclass_fields__})
            if 'traditional' in e:
                config.evaluation.traditional = TraditionalMetricsConfig(**{k: v for k, v in e['traditional'].items() if k in TraditionalMetricsConfig.__dataclass_fields__})
            if 'wandb' in e:
                config.evaluation.wandb = WandBConfig(**{k: v for k, v in e['wandb'].items() if k in WandBConfig.__dataclass_fields__})
            if 'output' in e:
                config.evaluation.output = OutputConfig(**{k: v for k, v in e['output'].items() if k in OutputConfig.__dataclass_fields__})
        
        # Logging
        if 'logging' in d:
            config.logging = LoggingConfig(**{k: v for k, v in d['logging'].items() if k in LoggingConfig.__dataclass_fields__})
        
        # PDF
        if 'pdf' in d:
            p = d['pdf']
            if 'parser' in p:
                config.pdf.parser = PDFParserConfig(**{k: v for k, v in p['parser'].items() if k in PDFParserConfig.__dataclass_fields__})
            if 'ocr' in p:
                config.pdf.ocr = PDFOCRConfig(**{k: v for k, v in p['ocr'].items() if k in PDFOCRConfig.__dataclass_fields__})
            if 'table' in p:
                config.pdf.table = PDFTableConfig(**{k: v for k, v in p['table'].items() if k in PDFTableConfig.__dataclass_fields__})
            if 'bbox' in p:
                config.pdf.bbox = PDFBBoxConfig(**{k: v for k, v in p['bbox'].items() if k in PDFBBoxConfig.__dataclass_fields__})
            if 'output' in p:
                config.pdf.output = PDFOutputConfig(**{k: v for k, v in p['output'].items() if k in PDFOutputConfig.__dataclass_fields__})
        
        return config


# =============================================================================
# 全局访问接口
# =============================================================================

_loader = ConfigLoader()


def init_config(mode: str = "prod", config_dir: Optional[str] = None) -> Config:
    """
    初始化配置（通常在应用入口调用一次）
    
    Args:
        mode: 环境模式 (test/dev/prod)
        config_dir: 配置目录路径
    
    Returns:
        Config 对象
    
    Example:
        >>> init_config(mode="test")
        >>> config = get_config()
        >>> print(config.data.sampling.ratio)  # 0.1
    """
    return _loader.load(mode=mode, config_dir=config_dir)


def get_config() -> Config:
    """
    获取配置（需先调用 init_config）
    
    Returns:
        Config 对象
    
    Raises:
        RuntimeError: 如果未初始化配置
    """
    return _loader.get()


def parse_args_and_init() -> Config:
    """
    从命令行参数解析 mode 并初始化配置
    
    支持:
        --mode test/dev/prod
        --config-dir /path/to/config
    
    Returns:
        Config 对象
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--mode', type=str, default='prod', choices=['test', 'dev', 'prod'],
                        help='Environment mode: test, dev, or prod')
    parser.add_argument('--config-dir', type=str, default=None,
                        help='Path to config directory')
    
    args, _ = parser.parse_known_args()
    
    return init_config(mode=args.mode, config_dir=args.config_dir)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 测试配置加载
    import sys
    
    # 解析命令行参数
    config = parse_args_and_init()
    
    print(f"=" * 60)
    print(f"配置加载测试 - Mode: {config._mode}")
    print(f"=" * 60)
    
    print(f"\n[App]")
    print(f"  name: {config.app.name}")
    print(f"  version: {config.app.version}")
    print(f"  project_root: {config.app.project_root}")
    
    print(f"\n[Models]")
    print(f"  LLM: {config.models.llm.name}")
    print(f"  LLM Path: {config.models.llm.path}")
    print(f"  Embedding: {config.models.embedding.name}")
    print(f"  Reranker: {config.models.reranker.name}")
    
    print(f"\n[Data]")
    print(f"  raw_path: {config.data.raw_path}")
    print(f"  processed_path: {config.data.processed_path}")
    print(f"  master_clauses_path: {config.data.master_clauses_path}")
    print(f"  sampling.enabled: {config.data.sampling.enabled}")
    print(f"  sampling.ratio: {config.data.sampling.ratio}")
    
    print(f"\n[Retrieval]")
    print(f"  vector_db.persist_directory: {config.retrieval.vector_db.persist_directory}")
    print(f"  hybrid.top_k: {config.retrieval.hybrid.top_k}")
    print(f"  rerank.top_k: {config.retrieval.rerank.top_k}")
    
    print(f"\n[Cache]")
    print(f"  semantic.enabled: {config.cache.semantic.enabled}")
    print(f"  semantic.backend: {config.cache.semantic.backend}")
    
    print(f"\n[Logging]")
    print(f"  level: {config.logging.level}")
    print(f"  file: {config.logging.file}")
    
    print(f"\n" + "=" * 60)
    print("配置加载成功!")
