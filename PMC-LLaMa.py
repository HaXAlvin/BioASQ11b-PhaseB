import transformers
import torch
import os

if torch.has_mps:
    device = 'mps'
elif torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
else:
    device = 'cpu'
print(f"Device: {device}")
# exit()
tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/PMC_LLAMA_7B')
model = transformers.LlamaForCausalLM.from_pretrained('chaoyi-wu/PMC_LLAMA_7B', device_map="auto")

sentence = 'Hello, doctor' 
batch = tokenizer(
    sentence,
    return_tensors="pt", 
    add_special_tokens=False
).to(device)

with torch.no_grad():
    generated = model.generate(inputs = batch["input_ids"], max_length=1024, do_sample=True, top_k=50)
    print('model predict: ',tokenizer.decode(generated[0]))