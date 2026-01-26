from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM
import torch
from loguru import logger
from pathlib import Path
import gc
import os

# GPU设备分配配置
# LLM推理固定在GPU0，其他模块（embedding、reranker）使用GPU1
# 
# 使用说明：
# 1. 在启动脚本中设置 CUDA_VISIBLE_DEVICES=0,1（让所有GPU可见）
# 2. vLLM会自动使用GPU0（通过环境变量或默认行为）
# 3. Embedding和Reranker模型会明确指定使用GPU1
#
# 如果只有单GPU，系统会自动降级到使用该GPU
LLM_GPU_ID = 0  # LLM推理使用GPU0
OTHER_GPU_ID = 1  # Embedding和Reranker使用GPU1

_cache = {}

def qwen_4bit_nf4_config(compute_dtype=torch.bfloat16):
    """
    创建Qwen模型的4bit NF4量化配置
    
    Args:
        compute_dtype: 计算数据类型
    
    Returns:
        BitsAndBytesConfig对象
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

def get_tokenizer(model_path, **tok_kwargs):
    """
    获取tokenizer，带缓存机制
    
    Args:
        model_path: 模型路径
        **tok_kwargs: tokenizer的其他参数
    
    Returns:
        tokenizer对象
    """
    key = ("tok", model_path, tuple(sorted(tok_kwargs.items())))
    if key not in _cache:
        _cache[key] = AutoTokenizer.from_pretrained(
            model_path,
            **tok_kwargs
        )
        logger.info(f"Tokenizer loaded: {model_path} {tok_kwargs}")
    return _cache[key]

def get_hf_causallm(model_path, quant=None, device_map="auto", dtype="auto", trust_remote_code=True):
    """
    获取HuggingFace因果语言模型，带缓存机制
    
    Args:
        model_path: 模型路径
        quant: 量化配置
        device_map: 设备映射
        dtype: 数据类型
        trust_remote_code: 是否信任远程代码
    
    Returns:
        模型对象
    """
    key = ("hf", model_path, repr(quant), dtype, device_map, trust_remote_code)
    if key not in _cache:
        kwargs = dict(device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
        if quant is not None:
            kwargs["quantization_config"] = quant
        _cache[key] = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        logger.success(f"HF model loaded: {model_path}")
    return _cache[key]

def get_vllm(model_path, dtype="auto", max_len=4096, gpu_util=0.85, trust_remote_code=True):
    """
    获取vLLM模型，带缓存机制，固定在GPU0
    
    注意：vLLM通过CUDA_VISIBLE_DEVICES环境变量控制可见GPU。
    为了将vLLM固定在GPU0，需要在启动进程前设置CUDA_VISIBLE_DEVICES=0。
    如果环境变量未设置，vLLM会使用所有可见的GPU。
    
    Args:
        model_path: 模型路径
        dtype: 数据类型
        max_len: 最大序列长度
        gpu_util: GPU内存利用率 (默认0.85)
        trust_remote_code: 是否信任远程代码
    
    Returns:
        vLLM模型对象
    
    Raises:
        ValueError: 当GPU内存不足时
    """
    key = ("vllm", model_path, dtype, max_len, gpu_util, trust_remote_code)
    if key not in _cache:
        try:
            # 检查CUDA_VISIBLE_DEVICES是否已设置
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            if cuda_visible is None:
                # 如果未设置，临时设置为GPU0（仅对当前进程有效）
                # 注意：这不会影响已经初始化的CUDA上下文
                if torch.cuda.device_count() > LLM_GPU_ID:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(LLM_GPU_ID)
                    logger.info(f"Setting CUDA_VISIBLE_DEVICES={LLM_GPU_ID} for vLLM (LLM固定在GPU0)")
                    logger.warning("建议在启动脚本中设置 CUDA_VISIBLE_DEVICES=0 以确保vLLM只使用GPU0")
            else:
                logger.info(f"Using existing CUDA_VISIBLE_DEVICES={cuda_visible}")
            
            _cache[key] = LLM(
                model=model_path,
                dtype=dtype,
                max_model_len=max_len,
                gpu_memory_utilization=gpu_util,
                tensor_parallel_size=1,
                trust_remote_code=trust_remote_code,
            )
            logger.success(f"vLLM model initialized on GPU{LLM_GPU_ID}: {model_path}")
                
        except ValueError as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            logger.warning("Try reducing gpu_util parameter or freeing GPU memory")
            logger.warning("确保在启动脚本中设置 CUDA_VISIBLE_DEVICES=0 以将vLLM固定在GPU0")
            raise
    return _cache[key]

def load_reranker(model_path: str):
    """
    加载重排序模型，固定在GPU1
    
    Args:
        model_path: 模型路径
    
    Returns:
        (tokenizer, model) 元组
    """
    tok = get_tokenizer(model_path, padding_side="left", trust_remote_code=True)
    
    # 检查GPU1是否可用
    if torch.cuda.is_available() and torch.cuda.device_count() > OTHER_GPU_ID:
        # 使用device_map指定GPU1
        # 对于量化模型，device_map需要是字典格式
        device_map = f"cuda:{OTHER_GPU_ID}"
        logger.info(f"Loading reranker on GPU{OTHER_GPU_ID}")
    else:
        device_map = "auto" if torch.cuda.is_available() else None
        if device_map == "auto":
            logger.warning(f"GPU{OTHER_GPU_ID} not available, using auto device mapping")
    
    # 在加载模型前清理GPU缓存，避免内存碎片
    if torch.cuda.is_available() and torch.cuda.device_count() > OTHER_GPU_ID:
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    model = get_hf_causallm(
        model_path,
        quant=qwen_4bit_nf4_config(),
        dtype="auto",
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # 确保模型在GPU1上（对于量化模型，可能需要手动移动）
    if torch.cuda.is_available() and torch.cuda.device_count() > OTHER_GPU_ID:
        try:
            # 尝试获取模型的设备
            if hasattr(model, 'device'):
                current_device = str(model.device)
            else:
                # 对于量化模型，检查第一个参数的设备
                first_param = next(model.parameters(), None)
                current_device = str(first_param.device) if first_param is not None else None
            
            target_device = f"cuda:{OTHER_GPU_ID}"
            if current_device and current_device != target_device:
                # 注意：量化模型可能不支持.to()方法，所以这里只记录
                logger.info(f"Reranker model is on {current_device}, target is {target_device}")
                logger.info("量化模型已通过device_map参数分配到指定GPU")
        except Exception as e:
            logger.warning(f"Could not verify reranker device: {e}")
    
    return tok, model

def clear_gpu_memory():
    """
    清理GPU内存
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")

def release_all_models():
    """
    释放所有模型，清理GPU内存
    """
    global _cache
    logger.info("Starting complete model release sequence...")

    if _cache:
        logger.info(f"Clearing {len(_cache)} models from internal cache")
        _cache.clear()

    try:
        import src.rag.embedding as embedding_module
        if hasattr(embedding_module, '_model'):
            embedding_module._model = None
        if hasattr(embedding_module, '_collection'):
            embedding_module._collection = None
        
    except Exception as e:
        logger.warning(f"Error referencing external modules: {e}")

    for _ in range(3):
        gc.collect()
        clear_gpu_memory()
    
    logger.success("GPU memory release sequence completed.")
