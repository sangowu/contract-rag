"""
CUAD 合同助手 - 核心模块

导出:
    - Config: 配置数据类
    - get_config: 获取配置
    - init_config: 初始化配置
    - parse_args_and_init: 从命令行解析并初始化配置
    - GPUManager, GPUConfig: GPU 管理
    - init_gpu_manager, get_gpu_config: GPU 管理便捷函数
"""

from src.core.config import (
    Config,
    get_config,
    init_config,
    parse_args_and_init,
    # 子配置类 (按需导入)
    ModelsConfig,
    DataConfig,
    RetrievalConfig,
    CacheConfig,
    APIConfig,
    EvaluationConfig,
    LoggingConfig,
    LLMConfig,
    LLMModelConfig,
)

from src.core.gpu_manager import (
    GPUManager,
    GPUConfig,
    init_gpu_manager,
    get_gpu_config,
)

__all__ = [
    # 配置
    'Config',
    'get_config',
    'init_config',
    'parse_args_and_init',
    'ModelsConfig',
    'DataConfig',
    'RetrievalConfig',
    'CacheConfig',
    'APIConfig',
    'EvaluationConfig',
    'LoggingConfig',
    'LLMConfig',
    'LLMModelConfig',
    # GPU 管理
    'GPUManager',
    'GPUConfig',
    'init_gpu_manager',
    'get_gpu_config',
]
