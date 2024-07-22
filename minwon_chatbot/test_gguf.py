from llama_cpp import Llama
model_path = "./llama2_with_lora_merged_q4.gguf"

# llm = LlamaCpp(model_path=model_path, max_tokens=2000, temperature=1.0, verbose=True)
llm = Llama(model_path=model_path)
prompt = """
Question: 어떻게 해야 집에 갈 수 있을까요?
"""
output = llm(prompt, max_tokens=32)
print(output)