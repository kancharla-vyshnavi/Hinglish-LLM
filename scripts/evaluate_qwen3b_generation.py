import json
import re
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "results/qwen2.5-3b-qlora/checkpoint-22500"

TEST_FILE = "data/processed/test.jsonl"

OUTPUT_DIR = "results/qwen2.5-3b-generation"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "generated_samples.jsonl")
METRICS_FILE = os.path.join(OUTPUT_DIR, "generation_metrics.json")

NUM_SAMPLES = 500
MAX_INPUT_LENGTH = 64
MAX_NEW_TOKENS = 64


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("QWEN 3B GENERATION EVALUATION")
print("=" * 60)

dataset = load_dataset(
    "json",
    data_files={"test": TEST_FILE}
)["test"]

dataset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))

print("Test samples used:", len(dataset))


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL
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
# LOAD LORA ADAPTER
# ============================================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

print("✅ QWEN 3B MODEL LOADED")
print("Adapter:", ADAPTER_PATH)


# ============================================================
# GENERATION
# ============================================================

generated_texts = []
records = []

print("\n🚀 Generating responses...")

for i, example in enumerate(dataset):

    text = example["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id
        )

    # Only keep newly generated tokens
    input_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[0][input_length:]

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    ).strip()

    generated_texts.append(generated_text)

    records.append({
        "id": i,
        "source": example.get("source", ""),
        "input": text,
        "generated": generated_text
    })

    if (i + 1) % 50 == 0:
        print(f"Generated: {i + 1}/{len(dataset)}")


# ============================================================
# SAVE GENERATED SAMPLES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# METRICS
# ============================================================

all_words = []
all_bigrams = []

total_generated_words = 0
repeated_words = 0

for text in generated_texts:

    words = re.findall(r"\b\w+\b", text.lower())

    all_words.extend(words)

    bigrams = list(zip(words, words[1:]))
    all_bigrams.extend(bigrams)

    total_generated_words += len(words)

    repeated_words += len(words) - len(set(words))


# Distinct-1
distinct_1 = (
    len(set(all_words)) / len(all_words)
    if all_words else 0
)

# Distinct-2
distinct_2 = (
    len(set(all_bigrams)) / len(all_bigrams)
    if all_bigrams else 0
)

# Repetition rate
repetition_rate = (
    repeated_words / total_generated_words
    if total_generated_words else 0
)


# ============================================================
# LANGUAGE MIXING
# ============================================================

english_words = 0
total_words = 0

for text in generated_texts:

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    english_words += len(words)

    total_words += len(
        re.findall(r"\b\w+\b", text)
    )

english_ratio = (
    english_words / total_words
    if total_words else 0
)


# ============================================================
# CHARACTER DIVERSITY
# ============================================================

all_chars = "".join(generated_texts)

character_diversity = (
    len(set(all_chars)) / len(all_chars)
    if all_chars else 0
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "model": BASE_MODEL,
    "adapter": ADAPTER_PATH,
    "num_samples": len(dataset),
    "distinct_1": distinct_1,
    "distinct_2": distinct_2,
    "repetition_rate": repetition_rate,
    "english_ratio": english_ratio,
    "character_diversity": character_diversity
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("✅ QWEN 3B GENERATION EVALUATION COMPLETED")
print("=" * 60)

print(f"Samples              : {len(dataset)}")
print(f"Distinct-1           : {distinct_1:.4f}")
print(f"Distinct-2           : {distinct_2:.4f}")
print(f"Repetition Rate      : {repetition_rate:.4f}")
print(f"English Ratio        : {english_ratio:.4f}")
print(f"Character Diversity  : {character_diversity:.4f}")

print("\nSaved:")
print(OUTPUT_FILE)
print(METRICS_FILE)

print("=" * 60)