import os
import json
import math
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

ADAPTER_PATH = "results/smollm2-1.7b-qlora"

TEST_FILE = "data/processed/test.jsonl"

MAX_LENGTH = 64

OUTPUT_DIR = "results/smollm2-1.7b-evaluation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SMOLLM2 1.7B - PERPLEXITY EVALUATION")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading base SmolLM2 model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

base_model.config.use_cache = True


# ============================================================
# LOAD QLORA ADAPTER
# ============================================================

print("\nLoading QLoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("Adapter loaded successfully.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

dataset = load_dataset(
    "json",
    data_files=TEST_FILE,
    split="train"
)

print("Test samples:", len(dataset))


# ============================================================
# PERPLEXITY
# ============================================================

print("\nCalculating perplexity...")

total_loss = 0.0
total_tokens = 0

for i, example in enumerate(dataset):

    text = example["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    labels = input_ids.clone()

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    loss = outputs.loss

    num_tokens = labels.numel()

    total_loss += loss.item() * num_tokens
    total_tokens += num_tokens

    if (i + 1) % 500 == 0:
        print(f"Processed: {i + 1}/{len(dataset)}")


# ============================================================
# FINAL METRICS
# ============================================================

average_loss = total_loss / total_tokens

perplexity = math.exp(average_loss)


print("\n" + "=" * 60)
print("SMOLLM2 EVALUATION COMPLETED")
print("=" * 60)

print(f"Average Loss : {average_loss:.6f}")
print(f"Perplexity   : {perplexity:.4f}")
print(f"Test Samples : {len(dataset)}")


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": BASE_MODEL,
    "adapter": ADAPTER_PATH,
    "test_samples": len(dataset),
    "average_loss": average_loss,
    "perplexity": perplexity
}

output_file = os.path.join(
    OUTPUT_DIR,
    "perplexity.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResults saved to:")
print(output_file)