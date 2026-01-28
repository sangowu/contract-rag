"""
GPU 管理模块

功能:
- 自动检测 GPU 数量
- 根据 GPU 数量选择合适的模型
- 管理设备分配策略
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from loguru import logger


def _get_model_path(project_root: str, local_name: str, hf_repo_id: str) -> str:
    """
    获取模型路径：优先使用本地路径，不存在则使用 HuggingFace repo id
    
    Args:
        project_root: 项目根目录
        local_name: 本地模型文件夹名称
        hf_repo_id: HuggingFace 仓库 ID (如 "Qwen/Qwen3-4B-Instruct-2507")
    
    Returns:
        模型路径（本地路径或 HuggingFace repo id）
    """
    local_path = os.path.join(project_root, "model", local_name)
    if os.path.exists(local_path):
        logger.info(f"Using local model: {local_path}")
        return local_path
    else:
        logger.info(f"Local model not found, using HuggingFace: {hf_repo_id}")
        return hf_repo_id


@dataclass
class GPUConfig:
    """GPU 配置"""
    gpu_count: int
    mode: str  # "single_gpu" | "dual_gpu" | "cpu_only"
    
    # LLM 配置
    llm_model_name: str
    llm_model_path: str
    llm_devices: List[int]
    llm_tensor_parallel: int
    llm_gpu_memory_utilization: float
    
    # 其他模型配置 (embedding, reranker)
    other_devices: List[int]
    other_primary_device: int
    
    # 显存信息 (GB)
    total_memory_gb: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """验证配置"""
        if self.gpu_count == 0 and self.mode != "cpu_only":
            raise ValueError("No GPU available but mode is not cpu_only")
    
    @property
    def llm_device_str(self) -> str:
        """获取 LLM 设备字符串"""
        if not self.llm_devices:
            return "cpu"
        return f"cuda:{self.llm_devices[0]}"
    
    @property
    def other_device_str(self) -> str:
        """获取其他模型设备字符串"""
        if not self.other_devices:
            return "cpu"
        return f"cuda:{self.other_devices[0]}"


class GPUManager:
    """
    GPU 管理器 (单例)
    
    负责检测 GPU 并提供设备分配策略
    """
    
    _instance: Optional['GPUManager'] = None
    _config: Optional[GPUConfig] = None
    
    # 模型配置
    MODEL_CONFIGS = {
        "single_gpu": {
            "name": "Qwen/Qwen3-4B-Instruct-2507",
            "path_key": "llm_4b_path",  # 从配置文件读取
            "gpu_util": 0.45,  # 单 GPU 需要与其他模型共享
        },
        "dual_gpu": {
            "name": "Qwen/Qwen3-8B",
            "path_key": "llm_8b_path",
            "gpu_util": 0.85,  # 独占 GPU0
        },
    }
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        if GPUManager._config is not None:
            return
    
    @classmethod
    def detect_gpus(cls) -> tuple[int, List[float]]:
        """
        检测可用 GPU 数量和显存
        
        Returns:
            (gpu_count, memory_list_gb)
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, running on CPU")
            return 0, []
        
        gpu_count = torch.cuda.device_count()
        memory_list = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024 ** 3)
            memory_list.append(round(memory_gb, 2))
            logger.info(f"GPU {i}: {props.name}, {memory_gb:.2f} GB")
        
        return gpu_count, memory_list
    
    @classmethod
    def initialize(cls, project_root: str, force_mode: Optional[str] = None) -> GPUConfig:
        """
        初始化 GPU 配置
        
        Args:
            project_root: 项目根目录
            force_mode: 强制使用的模式 ("single_gpu" | "dual_gpu" | "cpu_only")
        
        Returns:
            GPUConfig 对象
        """
        gpu_count, memory_list = cls.detect_gpus()
        
        # 确定模式
        if force_mode:
            mode = force_mode
            logger.info(f"Using forced GPU mode: {mode}")
        elif gpu_count == 0:
            mode = "cpu_only"
        elif gpu_count == 1:
            mode = "single_gpu"
        else:  # gpu_count >= 2
            mode = "dual_gpu"
        
        logger.info(f"GPU mode: {mode} (detected {gpu_count} GPU(s))")
        
        # 根据模式配置
        if mode == "cpu_only":
            config = cls._create_cpu_config()
        elif mode == "single_gpu":
            config = cls._create_single_gpu_config(project_root, memory_list)
        else:  # dual_gpu
            config = cls._create_dual_gpu_config(project_root, memory_list)
        
        cls._config = config
        cls._log_config(config)
        
        return config
    
    @classmethod
    def _create_cpu_config(cls) -> GPUConfig:
        """创建 CPU 模式配置"""
        return GPUConfig(
            gpu_count=0,
            mode="cpu_only",
            llm_model_name="Qwen/Qwen3-4B-Instruct-2507",
            llm_model_path="",  # CPU 模式可能不支持
            llm_devices=[],
            llm_tensor_parallel=1,
            llm_gpu_memory_utilization=0.0,
            other_devices=[],
            other_primary_device=-1,
            total_memory_gb=[],
        )
    
    @classmethod
    def _create_single_gpu_config(cls, project_root: str, memory_list: List[float]) -> GPUConfig:
        """
        创建单 GPU 配置
        
        策略: 使用较小的 4B 模型，所有模型共享 GPU0
        """
        model_config = cls.MODEL_CONFIGS["single_gpu"]
        
        return GPUConfig(
            gpu_count=1,
            mode="single_gpu",
            llm_model_name=model_config["name"],
            # 优先使用本地路径，不存在则使用 HuggingFace repo id
            llm_model_path=_get_model_path(project_root, "Qwen3-4B-Instruct-2507", model_config["name"]),
            llm_devices=[0],
            llm_tensor_parallel=1,
            llm_gpu_memory_utilization=model_config["gpu_util"],
            other_devices=[0],  # 共享 GPU0
            other_primary_device=0,
            total_memory_gb=memory_list,
        )
    
    @classmethod
    def _create_dual_gpu_config(cls, project_root: str, memory_list: List[float]) -> GPUConfig:
        """
        创建双 GPU 配置
        
        策略: 
        - LLM (8B) 独占 GPU0
        - Embedding + Reranker 使用 GPU1
        """
        model_config = cls.MODEL_CONFIGS["dual_gpu"]
        
        return GPUConfig(
            gpu_count=min(2, len(memory_list)),
            mode="dual_gpu",
            llm_model_name=model_config["name"],
            # 优先使用本地路径，不存在则使用 HuggingFace repo id
            llm_model_path=_get_model_path(project_root, "Qwen3-8B", model_config["name"]),
            llm_devices=[0],
            llm_tensor_parallel=1,
            llm_gpu_memory_utilization=model_config["gpu_util"],
            other_devices=[1],  # Embedding/Reranker 用 GPU1
            other_primary_device=1,
            total_memory_gb=memory_list[:2],
        )
    
    @classmethod
    def _log_config(cls, config: GPUConfig):
        """记录配置信息"""
        logger.info("=" * 60)
        logger.info("GPU Configuration Summary")
        logger.info("=" * 60)
        logger.info(f"  Mode: {config.mode}")
        logger.info(f"  GPU Count: {config.gpu_count}")
        logger.info(f"  LLM Model: {config.llm_model_name}")
        logger.info(f"  LLM Devices: {config.llm_devices}")
        logger.info(f"  LLM GPU Utilization: {config.llm_gpu_memory_utilization}")
        logger.info(f"  Other Model Devices: {config.other_devices}")
        if config.total_memory_gb:
            for i, mem in enumerate(config.total_memory_gb):
                logger.info(f"  GPU {i} Memory: {mem} GB")
        logger.info("=" * 60)
    
    @classmethod
    def get_config(cls) -> GPUConfig:
        """
        获取当前 GPU 配置
        
        Returns:
            GPUConfig 对象
        
        Raises:
            RuntimeError: 如果尚未初始化
        """
        if cls._config is None:
            raise RuntimeError(
                "GPUManager not initialized. Call GPUManager.initialize() first."
            )
        return cls._config
    
    @classmethod
    def is_initialized(cls) -> bool:
        """检查是否已初始化"""
        return cls._config is not None
    
    @classmethod
    def reset(cls):
        """重置配置（用于测试）"""
        cls._config = None
        cls._instance = None


def get_gpu_config() -> GPUConfig:
    """
    获取 GPU 配置的便捷函数
    
    Returns:
        GPUConfig 对象
    """
    return GPUManager.get_config()


def init_gpu_manager(project_root: str, force_mode: Optional[str] = None) -> GPUConfig:
    """
    初始化 GPU 管理器的便捷函数
    
    Args:
        project_root: 项目根目录
        force_mode: 强制模式
    
    Returns:
        GPUConfig 对象
    """
    return GPUManager.initialize(project_root, force_mode)
