from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from loguru import logger

MODEL_NAME = "/root/autodl-tmp/model/Qwen3-8B"

_tokenizer = None
_transformers_model = None
_vllm_model = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        logger.info("Loading tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info("Tokenizer loaded")
    return _tokenizer

def get_transformers_model():
    global _transformers_model
    if _transformers_model is None:
        logger.info("Loading transformers model...")
        _transformers_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype="auto",
            device_map="auto",
        )
        logger.success("Transformers model loaded")
    return _transformers_model

def get_vllm_model() -> LLM:
    global _vllm_model
    if _vllm_model is None:
        logger.info("Initializing vLLM model...")
        _vllm_model = LLM(
            model=MODEL_NAME,
            dtype="auto",
            max_model_len=16384,
            gpu_memory_utilization=0.9,  
            tensor_parallel_size=1, 
            enforce_eager=False,  
        )
        logger.success("vLLM model initialized")
    return _vllm_model

def llm_generate_batch(
    prompts: List[str], 
    use_vllm: bool = True,
    batch_size: int = 8,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> List[str]:

    if use_vllm:
        return _llm_generate_batch_vllm(prompts, max_tokens, temperature, top_p)
    else:
        return _llm_generate_batch_transformers(prompts, batch_size, max_tokens, temperature, top_p)

def _llm_generate_batch_vllm(
    prompts: List[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:

    llm = get_vllm_model()
    
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["\n\n", "###", "<|endoftext|>"], 
    )
    
    # logger.info(f"Starting vLLM batch inference for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        results.append(generated_text)
        if i < 3:  
            logger.debug(f"Generated {i+1}: {generated_text[:100]}...")
    
    # logger.success(f"vLLM batch inference completed for {len(prompts)} prompts")
    return results

def _llm_generate_batch_transformers(
    prompts: List[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:

    tokenizer = get_tokenizer()
    transformers_model = get_transformers_model()

    results = []
    
    # logger.info(f"Starting transformers batch inference for {len(prompts)} prompts...")
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        batch_messages = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": "You are a helpful contract review assistant."},
                {"role": "user", "content": prompt}
            ]
            batch_messages.append(messages)
        
        batch_texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for messages in batch_messages
        ]
        
        model_inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(transformers_model.device)
        
        generated_ids = transformers_model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        for j, generated_seq in enumerate(generated_ids):
            input_len = len(model_inputs.input_ids[j])
            output_ids = generated_seq[input_len:].tolist()
            
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            content = content.split("\n\n")[0] if "\n\n" in content else content
            results.append(content)
            
            if len(results) <= 3:  
                logger.debug(f"Generated {len(results)}: {content[:100]}...")
    
    # logger.success(f"Transformers batch inference completed for {len(prompts)} prompts")
    return results

def llm_generate(
    full_prompt: str, 
    use_vllm: bool = False,  
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:

    results = llm_generate_batch(
        [full_prompt], 
        use_vllm=use_vllm,
        batch_size=1,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return results[0]

def llm_generate_vllm(prompt: str, max_tokens: int = 1024) -> str:
    return llm_generate(prompt, use_vllm=True, max_tokens=max_tokens)

def llm_generate_batch_vllm(prompts: List[str], max_tokens: int = 1024) -> List[str]:
    return llm_generate_batch(prompts, use_vllm=True, max_tokens=max_tokens)