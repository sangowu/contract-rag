from typing import List
from vllm import SamplingParams
from loguru import logger
from vllm import LLM
from src.utils.model_loading import get_tokenizer, get_hf_causallm, get_vllm
from src.utils.prompting import build_chat_prompts, postprocess

MODEL_NAME = "/root/autodl-tmp/model/Qwen3-8B"

def llm_generate_full_dataset(
    prompts: List[str],
    max_tokens: int = 1024, 
    temperature: float = 0.0, 
    gpu_util: float = 0.85,   
) -> List[str]:
    """
    使用vLLM批量生成回答
    
    Args:
        prompts: 提示词列表
        max_tokens: 最大生成token数
        temperature: 采样温度
        gpu_util: GPU内存利用率 (默认0.85)
    
    Returns:
        生成的回答列表
    """
    logger.info(f"Initializing vLLM for {len(prompts)} prompts...")
    
    tokenizer = get_tokenizer(MODEL_NAME, trust_remote_code=True)
    
    batch_texts = build_chat_prompts(tokenizer, prompts)
    
    llm = get_vllm(MODEL_NAME, dtype="auto", max_len=8192, gpu_util=gpu_util, trust_remote_code=True)
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "\n\n", "###", "<|endoftext|>"], 
    )
    
    outputs = llm.generate(batch_texts, sampling_params)
    
    results = []
    for output in outputs:
        text = output.outputs[0].text
        text = postprocess(text)
        results.append(text)
        
    return results


def llm_generate_batch(
    prompts: List[str],
    use_vllm: bool = True,
    batch_size: int = 4,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
    max_input_len: int = 4096,
) -> List[str]:
    """
    批量生成回答
    
    Args:
        prompts: 提示词列表
        use_vllm: 是否使用vLLM
        batch_size: 批处理大小
        max_tokens: 最大生成token数
        temperature: 采样温度
        top_p: nucleus采样参数
        max_input_len: 最大输入长度
    
    Returns:
        生成的回答列表
    """
    tokenizer = get_tokenizer(MODEL_NAME)

    batch_texts = build_chat_prompts(tokenizer, prompts)

    if use_vllm:
        llm = get_vllm(MODEL_NAME, dtype="auto", max_len=max_input_len, gpu_util=0.85)
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["\n\n", "###", "<|endoftext|>"],
        )
        outputs = llm.generate(batch_texts, sampling_params)

        results = []
        for i, o in enumerate(outputs):
            text = postprocess(o.outputs[0].text)
            results.append(text)
            if i < 3:
                logger.debug(f"Generated {i+1}: {text[:100]}...")
        return results

    else:
        model = get_hf_causallm(MODEL_NAME, dtype="auto", device_map="auto", trust_remote_code=True)
        results = []

        for i in range(0, len(batch_texts), batch_size):
            sub_texts = batch_texts[i:i+batch_size]
            inputs = tokenizer(
                sub_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_len, 
            ).to(model.device)

            ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j, seq in enumerate(ids):
                in_len = inputs.input_ids.shape[1]
                out_ids = seq[in_len:]
                text = postprocess(tokenizer.decode(out_ids, skip_special_tokens=True))
                results.append(text)

        return results

def llm_generate(
    prompt: str,
    use_vllm: bool = False,
    **kwargs
) -> str:
    """
    生成单个回答
    
    Args:
        prompt: 提示词
        use_vllm: 是否使用vLLM
        **kwargs: 其他参数传递给llm_generate_batch
    
    Returns:
        生成的回答
    """
    return llm_generate_batch([prompt], use_vllm=use_vllm, batch_size=1, **kwargs)[0]
