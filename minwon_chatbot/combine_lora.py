from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model_checkpoint = "meta-llama/Llama-2-7b-chat-hf"
lora_path = "./llama2_RAG_hackerton_mk3/checkpoint-1200"

base_model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map='auto')
config = LoraConfig.from_pretrained(lora_path)

model = get_peft_model(base_model, config)
tokenizer = AutoTokenizer.from_pretrained(lora_path)

model = model.merge_and_unload()
model.save_pretrained("./llama2_with_mk3", from_pt=True)