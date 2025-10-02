import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# ------------------------

MODEL_NAME = "google/gemma-3-270m"
DATA_FILE = "data.jsonl" 
MAX_LENGTH = 128
OUTPUT_DIR = "./gemma_finetuned"
BATCH_SIZE = 1  
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

special_tokens_dict = {"additional_special_tokens": ["[TOOL_CALL]"]}
tokenizer.add_special_tokens(special_tokens_dict)

print("Configuring 2-bit quantization for low VRAM...")
quant_config = BitsAndBytesConfig(
    load_in_2bit=True,
    bnb_2bit_quant_type="nf4",
    bnb_2bit_compute_dtype=torch.float16
)

print("Loading model in 2-bit mode...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
)

model.resize_token_embeddings(len(tokenizer))

print("Attaching LoRA adapter...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_FILE)

def tokenize(batch):
    model_inputs = tokenizer(
        batch["query"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    labels = tokenizer(
        batch["answer"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset["train"].map(
    tokenize, batched=True, remove_columns=["query", "answer"]
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    fp16=True,
    optim="paged_adamw_32bit",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed successfully!")
