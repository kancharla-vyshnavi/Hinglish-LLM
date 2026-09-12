import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "results/qwen2.5-3b-qlora/checkpoint-22500"

TEST_FILE = "data/processed/test.jsonl"

MAX_LENGTH = 64


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("QWEN 3B PERPLEXITY EVALUATION")
print("=" * 60)

test_dataset = load_dataset(
    "json",
    data_files={"test": TEST_FILE}
)["test"]

print(f"Test samples: {len(test_dataset)}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL IN 4-BIT
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)


# ============================================================
# LOAD QLORA ADAPTER
# ============================================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("✅ QWEN 3B MODEL LOADED")
print("Adapter:", ADAPTER_PATH)


# ============================================================
# PERPLEXITY EVALUATION
# ============================================================

total_loss = 0.0
total_tokens = 0

print("\n🚀 Calculating perplexity...")

with torch.no_grad():

    for i, example in enumerate(test_dataset):

        text = example["text"]

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss

        num_tokens = attention_mask.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

        if (i + 1) % 500 == 0:
            print(f"Processed: {i + 1}/{len(test_dataset)}")


# ============================================================
# FINAL PERPLEXITY
# ============================================================

average_loss = total_loss / total_tokens

perplexity = math.exp(average_loss)


print("\n" + "=" * 60)
print("✅ QWEN 3B PERPLEXITY EVALUATION COMPLETED")
print("=" * 60)

print(f"Average Loss : {average_loss:.6f}")
print(f"Perplexity   : {perplexity:.4f}")
print(f"Test Samples : {len(test_dataset)}")
print("=" * 60)