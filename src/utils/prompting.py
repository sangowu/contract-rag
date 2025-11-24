SYSTEM_PROMPT = "You are a helpful contract review assistant."

def build_chat_prompts(tokenizer, prompts):
    msgs = [
        [{"role":"system","content":SYSTEM_PROMPT},
         {"role":"user","content":p}]
        for p in prompts
    ]
    return [tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ) for m in msgs]

def postprocess(text):
    text = text.strip()
    return text.split("\n\n")[0] if "\n\n" in text else text