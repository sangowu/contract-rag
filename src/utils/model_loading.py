"""
模型加载工具模块

功能:
- vLLM 模型加载（唯一的 LLM 推理后端）
- Tokenizer 加载
- Embedding/Reranker 模型加载
- GPU 设备分配
- 资源清理
"""

import os
import gc
import atexit
from typing import Optional, Tuple, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM
from loguru import logger

from src.core.gpu_manager import GPUManager, GPUConfig, get_gpu_config


# 模型缓存
_cache = {}

# vLLM 实例 (全局单例)
_vllm_instance: Optional[LLM] = None

# 是否已注册清理函数
_cleanup_registered: bool = False


def qwen_4bit_nf4_config(compute_dtype=torch.bfloat16) -> BitsAndBytesConfig:
    """
    创建 Qwen 模型的 4bit NF4 量化配置
    
    Args:
        compute_dtype: 计算数据类型
    
    Returns:
        BitsAndBytesConfig 对象
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def get_tokenizer(model_path: str, **tok_kwargs) -> Any:
    """
    获取 tokenizer，带缓存机制
    
    Args:
        model_path: 模型路径
        **tok_kwargs: tokenizer 的其他参数
    
    Returns:
        tokenizer 对象
    """
    key = ("tok", model_path, tuple(sorted(tok_kwargs.items())))
    if key not in _cache:
        _cache[key] = AutoTokenizer.from_pretrained(
            model_path,
            **tok_kwargs
        )
        logger.info(f"Tokenizer loaded: {model_path}")
    return _cache[key]


def get_vllm(
    model_path: Optional[str] = None,
    dtype: str = "auto",
    max_len: Optional[int] = None,
    gpu_util: Optional[float] = None,
    trust_remote_code: bool = True,
) -> LLM:
    """
    获取 vLLM 模型实例 (全局单例)
    
    根据 GPU 配置自动选择模型和设备分配。
    
    Args:
        model_path: 模型路径 (None 时从 GPU 配置自动获取)
        dtype: 数据类型
        max_len: 最大序列长度 (None 时从配置获取)
        gpu_util: GPU 内存利用率 (None 时从 GPU 配置获取)
        trust_remote_code: 是否信任远程代码
    
    Returns:
        vLLM LLM 实例
    """
    global _vllm_instance
    
    # 如果已有实例，直接返回
    if _vllm_instance is not None:
        logger.debug("Returning cached vLLM instance")
        return _vllm_instance
    
    # 获取 GPU 配置
    gpu_config = get_gpu_config()
    
    # 使用 GPU 配置的默认值
    if model_path is None:
        model_path = gpu_config.llm_model_path
    if gpu_util is None:
        gpu_util = gpu_config.llm_gpu_memory_utilization
    if max_len is None:
        # 从应用配置获取
        try:
            from src.core.config import get_config
            app_config = get_config()
            max_len = app_config.models.llm.max_model_len
        except:
            max_len = 8192
    
    logger.info(f"Initializing vLLM...")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  GPU Mode: {gpu_config.mode}")
    logger.info(f"  GPU Utilization: {gpu_util}")
    logger.info(f"  Max Length: {max_len}")
    logger.info(f"  Tensor Parallel: {gpu_config.llm_tensor_parallel}")
    
    # 设置 CUDA_VISIBLE_DEVICES 以控制 vLLM 使用的 GPU
    if gpu_config.llm_devices:
        devices_str = ",".join(map(str, gpu_config.llm_devices))
        # 注意: vLLM 在初始化时读取此环境变量
        # 如果已经有 CUDA 上下文，可能不会生效
        current_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if current_visible is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices_str
            logger.info(f"Set CUDA_VISIBLE_DEVICES={devices_str} for vLLM")
        else:
            logger.info(f"Using existing CUDA_VISIBLE_DEVICES={current_visible}")
    
    try:
        _vllm_instance = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_len,
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=gpu_config.llm_tensor_parallel,
            trust_remote_code=trust_remote_code,
        )
        logger.success(f"vLLM initialized: {gpu_config.llm_model_name}")
        
        # 注册清理函数
        _register_cleanup()
        
        return _vllm_instance
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        
        # 如果是双 GPU 模式失败，尝试降级到单 GPU 模式
        if gpu_config.mode == "dual_gpu":
            logger.warning("Attempting fallback to single GPU mode...")
            return _fallback_to_single_gpu(dtype, max_len, trust_remote_code)
        
        raise


def _fallback_to_single_gpu(
    dtype: str,
    max_len: int,
    trust_remote_code: bool
) -> LLM:
    """
    降级到单 GPU 模式
    """
    global _vllm_instance
    
    try:
        from src.core.config import get_config
        app_config = get_config()
        fallback_path = app_config.models.llm.fallback.path
        fallback_name = app_config.models.llm.fallback.name
    except:
        fallback_path = "/root/autodl-tmp/model/Qwen3-4B-Instruct-2507"
        fallback_name = "Qwen3-4B"
    
    logger.warning(f"Falling back to smaller model: {fallback_name}")
    
    _vllm_instance = LLM(
        model=fallback_path,
        dtype=dtype,
        max_model_len=max_len,
        gpu_memory_utilization=0.45,
        tensor_parallel_size=1,
        trust_remote_code=trust_remote_code,
    )
    
    logger.success(f"Fallback vLLM initialized: {fallback_name}")
    
    # 注册清理函数
    _register_cleanup()
    
    return _vllm_instance


def load_embedding_model(model_path: str) -> Any:
    """
    加载 Embedding 模型
    
    根据 GPU 配置分配到正确的设备。
    
    Args:
        model_path: 模型路径
    
    Returns:
        SentenceTransformer 模型
    """
    from sentence_transformers import SentenceTransformer
    
    gpu_config = get_gpu_config()
    device = gpu_config.other_device_str
    
    key = ("embedding", model_path, device)
    if key not in _cache:
        logger.info(f"Loading embedding model on {device}: {model_path}")
        _cache[key] = SentenceTransformer(model_path, device=device)
        logger.success(f"Embedding model loaded on {device}")
    
    return _cache[key]


def load_reranker(model_path: str) -> Tuple[Any, Any]:
    """
    加载重排序模型
    
    根据 GPU 配置分配到正确的设备。
    
    Args:
        model_path: 模型路径
    
    Returns:
        (tokenizer, model) 元组
    """
    gpu_config = get_gpu_config()
    device_id = gpu_config.other_primary_device
    
    # 确定设备映射
    if device_id >= 0 and torch.cuda.is_available():
        device_map = f"cuda:{device_id}"
        logger.info(f"Loading reranker on GPU{device_id}: {model_path}")
    else:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker with device_map={device_map}")
    
    # 清理 GPU 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    tok = get_tokenizer(model_path, padding_side="left", trust_remote_code=True)
    
    key = ("reranker_model", model_path, device_map)
    if key not in _cache:
        _cache[key] = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=qwen_4bit_nf4_config(),
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
        logger.success(f"Reranker model loaded")
    
    return tok, _cache[key]


def clear_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")


def release_vllm():
    """释放 vLLM 实例"""
    global _vllm_instance
    if _vllm_instance is not None:
        del _vllm_instance
        _vllm_instance = None
        clear_gpu_memory()
        logger.info("vLLM instance released")


def release_all_models():
    """释放所有模型，清理 GPU 内存"""
    global _cache, _vllm_instance
    
    logger.info("Releasing all models...")
    
    # 释放 vLLM
    release_vllm()
    
    # 清理缓存
    if _cache:
        logger.info(f"Clearing {len(_cache)} cached models")
        _cache.clear()
    
    # 多次 GC 确保完全释放
    for _ in range(3):
        gc.collect()
        clear_gpu_memory()
    
    logger.success("All models released")


def get_model_info() -> dict:
    """
    获取当前模型配置信息
    
    Returns:
        包含模型信息的字典
    """
    gpu_config = get_gpu_config()
    
    return {
        "gpu_mode": gpu_config.mode,
        "gpu_count": gpu_config.gpu_count,
        "llm_model": gpu_config.llm_model_name,
        "llm_devices": gpu_config.llm_devices,
        "other_devices": gpu_config.other_devices,
        "vllm_loaded": _vllm_instance is not None,
        "cached_models": list(_cache.keys()),
    }


# =============================================================================
# 兼容性别名 (保持向后兼容)
# =============================================================================

def get_hf_causallm(*args, **kwargs):
    """
    [已弃用] HuggingFace CausalLM 加载
    
    请使用 get_vllm() 进行 LLM 推理
    """
    logger.warning(
        "get_hf_causallm is deprecated for LLM inference. "
        "Use get_vllm() instead. This function now only works for reranker models."
    )
    # 仅用于 reranker 等非生成任务
    model_path = args[0] if args else kwargs.get("model_path")
    quant = kwargs.get("quant")
    device_map = kwargs.get("device_map", "auto")
    trust_remote_code = kwargs.get("trust_remote_code", True)
    
    key = ("hf", model_path, repr(quant), device_map, trust_remote_code)
    if key not in _cache:
        load_kwargs = dict(
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        if quant is not None:
            load_kwargs["quantization_config"] = quant
        _cache[key] = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs
        ).eval()
    return _cache[key]


# =============================================================================
# 资源清理
# =============================================================================

def cleanup_resources():
    """
    清理模型资源，释放 GPU 显存
    
    在程序退出时自动调用，也可手动调用。
    """
    global _vllm_instance, _cache
    
    logger.debug("Cleaning up model resources...")
    
    # 清理 vLLM 实例
    if _vllm_instance is not None:
        try:
            del _vllm_instance
            _vllm_instance = None
            logger.debug("vLLM instance released")
        except Exception as e:
            logger.debug(f"Error releasing vLLM: {e}")
    
    # 清理缓存的模型
    for key in list(_cache.keys()):
        try:
            del _cache[key]
        except Exception:
            pass
    _cache.clear()
    
    # 清理 CUDA 缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 销毁分布式进程组（解决 NCCL 警告）
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            logger.debug("Distributed process group destroyed")
        except Exception as e:
            logger.debug(f"Error destroying process group: {e}")
    
    logger.debug("Model resources cleaned up")


def _register_cleanup():
    """注册退出时的清理函数"""
    global _cleanup_registered
    if not _cleanup_registered:
        atexit.register(cleanup_resources)
        _cleanup_registered = True
        logger.debug("Cleanup function registered")
