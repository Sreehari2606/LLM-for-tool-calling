import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re

BASE_MODEL = "google/gemma-3-270m"
LORA_DIR = "./gemma_finetuned"
DATA_FILE = "./data.jsonl"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
SPECIAL_TOKEN = "[TOOL_CALL]"
if SPECIAL_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKEN]})

device = "cuda" if torch.cuda.is_available() else "cpu"

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto" if device == "cuda" else None,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()
model.to(device)

# Load tool data and extract only the "answer" part
tool_data = {}
with open(DATA_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        query = entry["query"].lower().strip()
        answer_text = entry["answer"]
        # Extract JSON inside [TOOL_CALL] chat{...}
        match = re.search(r'"answer"\s*:\s*"(.*?)"', answer_text)
        if match:
            answer_text_clean = match.group(1)
        else:
            answer_text_clean = answer_text
        tool_data[query] = answer_text_clean

def generate(prompt, max_tokens=128, temperature=0.7):
    cleaned_prompt = prompt.lower().strip()

    # Handle math expressions
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\.]+", prompt):
        try:
            return str(eval(prompt))
        except:
            pass

    # Check tool data
    if cleaned_prompt in tool_data:
        return tool_data[cleaned_prompt]

    # Generate using model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    except RuntimeError as e:
        if "CUDA" in str(e):
            inputs = inputs.to("cpu")
            model.to("cpu")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature
                )
            model.to(device)
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text
        else:
            raise e

print("Tool-calling ready. Type exit to quit.\n")

while True:
    prompt = input("Enter prompt: ").strip()
    if prompt.lower() == "exit":
        break
    try:
        output = generate(prompt)
        print("\nOutput:", output, "\n")
    except Exception as e:
        print("Error generating:", e)
