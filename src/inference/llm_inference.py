from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger
from src.rag.embedding import retrieve_top_k

MODEL_NAME = "/root/autodl-tmp/model/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
    )

def llm_generate(full_prompt: str) -> str:
    """
    Generate the answer based on the prompt.
    """
    messages = [
        {"role": "system", "content": "You are a helpful contract review assistant."},
        {"role": "user", "content": full_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    logger.success(f"Generated answer: {content}")
    return content