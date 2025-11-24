from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM
import torch
from loguru import logger
from pathlib import Path
import gc

_cache = {}

def qwen_4bit_nf4_config(compute_dtype=torch.bfloat16):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

def get_tokenizer(model_path, **tok_kwargs):
    key = ("tok", model_path, tuple(sorted(tok_kwargs.items())))
    if key not in _cache:
        _cache[key] = AutoTokenizer.from_pretrained(
            model_path,
            **tok_kwargs
        )
        logger.info(f"Tokenizer loaded: {model_path} {tok_kwargs}")
    return _cache[key]

def get_hf_causallm(model_path, quant=None, device_map="auto", dtype="auto", trust_remote_code=True):
    key = ("hf", model_path, repr(quant), dtype, device_map, trust_remote_code)
    if key not in _cache:
        kwargs = dict(device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
        if quant is not None:
            kwargs["quantization_config"] = quant
        _cache[key] = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        logger.success(f"HF model loaded: {model_path}")
    return _cache[key]

def get_vllm(model_path, dtype="auto", max_len=4096, gpu_util=0.9, trust_remote_code=True):
    key = ("vllm", model_path, dtype, max_len, gpu_util, trust_remote_code)
    if key not in _cache:
        _cache[key] = LLM(
            model=model_path,
            dtype=dtype,
            max_model_len=max_len,
            gpu_memory_utilization=gpu_util,
            tensor_parallel_size=1,
            trust_remote_code=trust_remote_code,
        )
        logger.success(f"vLLM model initialized: {model_path}")
    return _cache[key]

def load_reranker(model_path: str):
    tok = get_tokenizer(model_path, padding_side="left", trust_remote_code=True)
    model = get_hf_causallm(
        model_path,
        quant=qwen_4bit_nf4_config(),
        dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    return tok, model

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")

def release_all_models():
    global _cache
    logger.info("Starting complete model release sequence...")

    if _cache:
        logger.info(f"Clearing {_cache.keys()} from internal cache")
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