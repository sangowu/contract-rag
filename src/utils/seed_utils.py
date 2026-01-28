"""
CUAD 合同助手 - 随机数种子管理工具

功能:
1. 统一管理所有随机数种子 (Python random, NumPy, PyTorch, CUDA)
2. 从配置文件读取种子值
3. 确保实验可复现性

使用方式:
    from src.utils.seed_utils import set_global_seed, get_seed
    
    # 设置全局种子（通常在程序入口调用一次）
    set_global_seed()
    
    # 获取种子值（用于需要显式传递 seed 的场景）
    seed = get_seed()
"""

import os
import random
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 全局种子缓存
_GLOBAL_SEED: Optional[int] = None


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    设置全局随机数种子，确保实验可复现
    
    Args:
        seed: 随机种子值。如果为 None，则从配置文件读取
    
    Returns:
        实际使用的种子值
    
    Note:
        此函数会设置以下随机数生成器的种子:
        - Python random 模块
        - NumPy random
        - PyTorch (CPU 和 CUDA)
        - 环境变量 PYTHONHASHSEED
    """
    global _GLOBAL_SEED
    
    # 获取种子值
    if seed is None:
        seed = _get_seed_from_config()
    
    _GLOBAL_SEED = seed
    
    # 设置 Python random
    random.seed(seed)
    logger.debug(f"Set Python random seed: {seed}")
    
    # 设置环境变量（影响 hash 函数的随机性）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 设置 NumPy
    try:
        import numpy as np
        np.random.seed(seed)
        logger.debug(f"Set NumPy random seed: {seed}")
    except ImportError:
        logger.debug("NumPy not installed, skipping numpy seed")
    
    # 设置 PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        logger.debug(f"Set PyTorch manual seed: {seed}")
        
        # 设置 CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多 GPU 情况
            logger.debug(f"Set CUDA manual seed: {seed}")
            
            # 设置 CUDA 确定性模式
            cuda_deterministic = _get_cuda_deterministic_from_config()
            if cuda_deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.debug("Set CUDA deterministic mode: True")
            else:
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
                logger.debug("Set CUDA deterministic mode: False (faster but non-deterministic)")
    except ImportError:
        logger.debug("PyTorch not installed, skipping torch seed")
    
    logger.info(f"Global random seed set to: {seed}")
    return seed


def get_seed() -> int:
    """
    获取当前全局种子值
    
    Returns:
        当前种子值。如果尚未设置，返回配置中的默认值
    """
    global _GLOBAL_SEED
    
    if _GLOBAL_SEED is not None:
        return _GLOBAL_SEED
    
    return _get_seed_from_config()


def _get_seed_from_config() -> int:
    """
    从配置文件获取种子值
    
    Returns:
        种子值，默认为 42
    """
    try:
        from src.core.config import get_config
        config = get_config()
        if hasattr(config, 'seed') and hasattr(config.seed, 'global_seed'):
            return config.seed.global_seed
    except Exception as e:
        logger.debug(f"Failed to get seed from config: {e}")
    
    # 默认值
    return 42


def _get_cuda_deterministic_from_config() -> bool:
    """
    从配置文件获取 CUDA 确定性模式设置
    
    Returns:
        是否启用 CUDA 确定性模式，默认为 True
    """
    try:
        from src.core.config import get_config
        config = get_config()
        if hasattr(config, 'seed') and hasattr(config.seed, 'cuda_deterministic'):
            return config.seed.cuda_deterministic
    except Exception as e:
        logger.debug(f"Failed to get cuda_deterministic from config: {e}")
    
    # 默认启用确定性模式
    return True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker 初始化函数，确保多进程数据加载的可复现性
    
    用于 PyTorch DataLoader:
        loader = DataLoader(
            dataset,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(get_seed())
        )
    
    Args:
        worker_id: worker 进程 ID
    """
    worker_seed = get_seed() + worker_id
    
    import numpy as np
    np.random.seed(worker_seed)
    random.seed(worker_seed)
