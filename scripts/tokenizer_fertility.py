import json
import os
import numpy as np
from transformers import AutoTokenizer

# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/processed/test.jsonl"

MODELS = {
    "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B": "Qwen/Qwen2.5-3B-Instruct",
    "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Bharat-Tiny-LLM-v3": "eulogik/Bharat-Tiny-LLM-v3",
}

OUTPUT_DIR = "results/final_evaluation"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "tokenizer_fertility.json"
)

# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 60)
print("TOKENIZER FERTILITY ANALYSIS")
print("=" * 60)

texts = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        text = item.get("text", "").strip()

        if text:
            texts.append(text)

print(f"\nTest samples loaded: {len(texts):,}")

# ============================================================
# CALCULATE FERTILITY
# ============================================================

results = {}

for model_name, model_id in MODELS.items():

    print("\n" + "-" * 60)
    print(f"Loading tokenizer: {model_name}")
    print(f"Model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )

    fertility_values = []
    total_tokens = 0
    total_words = 0

    for text in texts:

        # Whitespace-separated words
        words = text.split()

        if len(words) == 0:
            continue

        # Tokenize without special tokens
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        num_words = len(words)
        num_tokens = len(tokens)

        fertility = num_tokens / num_words

        fertility_values.append(fertility)

        total_tokens += num_tokens
        total_words += num_words

    fertility_array = np.array(fertility_values)

    results[model_name] = {
        "model_id": model_id,
        "samples": len(fertility_values),
        "total_words": int(total_words),
        "total_tokens": int(total_tokens),
        "average_fertility": float(np.mean(fertility_array)),
        "median_fertility": float(np.median(fertility_array)),
        "std_fertility": float(np.std(fertility_array)),
        "min_fertility": float(np.min(fertility_array)),
        "max_fertility": float(np.max(fertility_array)),
    }

    print(f"Samples            : {len(fertility_values):,}")
    print(f"Total words        : {total_words:,}")
    print(f"Total tokens       : {total_tokens:,}")
    print(f"Average fertility  : {np.mean(fertility_array):.4f}")
    print(f"Median fertility   : {np.median(fertility_array):.4f}")
    print(f"Std deviation      : {np.std(fertility_array):.4f}")
    print(f"Minimum             : {np.min(fertility_array):.4f}")
    print(f"Maximum             : {np.max(fertility_array):.4f}")

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n" + "=" * 60)
print("TOKENIZER FERTILITY ANALYSIS COMPLETED")
print("=" * 60)

print(f"\nSaved to:")
print(OUTPUT_FILE)

print("\nSummary:")
print("-" * 60)

for model_name, values in results.items():
    print(
        f"{model_name:<25} "
        f"{values['average_fertility']:.4f}"
    )