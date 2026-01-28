"""
CUAD LLM 推理模块

功能:
- 批量生成回答 (vLLM)
- 单条生成回答
- 流式生成 (用于 API)

注意: 本模块仅支持 vLLM 后端，已移除 HuggingFace 直接推理
"""

from typing import List, Optional, Iterator, Dict, Any
from vllm import SamplingParams
from loguru import logger

from src.utils.model_loading import get_tokenizer, get_vllm
from src.utils.prompting import build_chat_prompts, postprocess
from src.core.config import get_config
from src.core.gpu_manager import get_gpu_config


def get_llm_config() -> Dict[str, Any]:
    """
    从配置获取 LLM 相关参数
    
    Returns:
        dict: 包含 LLM 配置的字典
    """
    config = get_config()
    gpu_config = get_gpu_config()
    
    return {
        'model_name': gpu_config.llm_model_name,
        'model_path': gpu_config.llm_model_path,
        'max_tokens': config.models.llm.max_tokens,
        'temperature': config.models.llm.temperature,
        'max_model_len': config.models.llm.max_model_len,
        'gpu_mode': gpu_config.mode,
        'gpu_util': gpu_config.llm_gpu_memory_utilization,
    }


def llm_generate_batch(
    prompts: List[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: float = 0.95,
    stop_sequences: Optional[List[str]] = None,
) -> List[str]:
    """
    使用 vLLM 批量生成回答
    
    Args:
        prompts: 提示词列表
        max_tokens: 最大生成 token 数 (默认从配置读取)
        temperature: 采样温度 (默认从配置读取)
        top_p: nucleus 采样参数
        stop_sequences: 停止序列
    
    Returns:
        生成的回答列表
    """
    llm_config = get_llm_config()
    model_path = llm_config['model_path']
    
    # 使用配置默认值
    if max_tokens is None:
        max_tokens = llm_config['max_tokens']
    if temperature is None:
        temperature = llm_config['temperature']
    if stop_sequences is None:
        stop_sequences = ["<|im_end|>", "\n\n", "###", "<|endoftext|>"]
    
    logger.info(f"Generating {len(prompts)} responses with {llm_config['model_name']}")
    logger.debug(f"  GPU mode: {llm_config['gpu_mode']}")
    logger.debug(f"  Max tokens: {max_tokens}, Temperature: {temperature}")
    
    # 获取 tokenizer 和构建聊天格式
    tokenizer = get_tokenizer(model_path, trust_remote_code=True)
    batch_texts = build_chat_prompts(tokenizer, prompts)
    
    # 获取 vLLM 实例
    llm = get_vllm()
    
    # 采样参数
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop_sequences,
    )
    
    # 生成
    outputs = llm.generate(batch_texts, sampling_params)
    
    # 后处理
    results = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text
        text = postprocess(text)
        results.append(text)
        
        if i < 3:  # 只记录前几条
            logger.debug(f"Generated [{i+1}]: {text[:100]}...")
    
    logger.info(f"Batch generation completed: {len(results)} responses")
    return results


def llm_generate(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> str:
    """
    生成单个回答
    
    Args:
        prompt: 提示词
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        **kwargs: 其他参数传递给 llm_generate_batch
    
    Returns:
        生成的回答
    """
    results = llm_generate_batch(
        [prompt],
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    return results[0]


def llm_generate_stream(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: float = 0.95,
) -> Iterator[str]:
    """
    流式生成回答 (用于 API 流式响应)
    
    注意: vLLM 原生不支持真正的流式生成（除非使用 vLLM server）。
    此函数模拟流式输出，将完整生成结果分块返回。
    
    Args:
        prompt: 提示词
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        top_p: nucleus 采样参数
    
    Yields:
        生成的文本片段
    """
    # 生成完整回答
    full_response = llm_generate(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # 模拟流式输出：按词/字符分块
    # 可以调整 chunk_size 控制流式效果
    chunk_size = 5  # 每次输出的字符数
    
    for i in range(0, len(full_response), chunk_size):
        chunk = full_response[i:i + chunk_size]
        yield chunk


async def llm_generate_stream_async(
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> Iterator[str]:
    """
    异步流式生成 (用于 FastAPI)
    
    Args:
        prompt: 提示词
        max_tokens: 最大生成 token 数
        temperature: 采样温度
    
    Yields:
        生成的文本片段
    """
    import asyncio
    
    # 在线程池中运行同步生成
    loop = asyncio.get_event_loop()
    full_response = await loop.run_in_executor(
        None,
        lambda: llm_generate(prompt, max_tokens, temperature)
    )
    
    # 流式输出
    chunk_size = 5
    for i in range(0, len(full_response), chunk_size):
        chunk = full_response[i:i + chunk_size]
        yield chunk
        await asyncio.sleep(0.01)  # 小延迟以模拟流式效果


# =============================================================================
# 兼容性别名
# =============================================================================

def llm_generate_full_dataset(
    prompts: List[str],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    gpu_util: float = 0.85,  # 此参数现在由 GPU 管理器控制
) -> List[str]:
    """
    [兼容性函数] 批量生成回答
    
    Args:
        prompts: 提示词列表
        max_tokens: 最大生成 token 数
        temperature: 采样温度
        gpu_util: GPU 利用率 (现在由 GPU 管理器自动设置)
    
    Returns:
        生成的回答列表
    """
    if gpu_util != 0.85:
        logger.warning(
            "gpu_util parameter is now managed by GPUManager and will be ignored. "
            "Use init_gpu_manager() to configure GPU settings."
        )
    
    return llm_generate_batch(
        prompts,
        max_tokens=max_tokens,
        temperature=temperature,
    )
