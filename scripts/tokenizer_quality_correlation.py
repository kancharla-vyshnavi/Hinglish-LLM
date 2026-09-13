import json
import os
import numpy as np

# ============================================================
# FILES
# ============================================================

FERTILITY_FILE = "results/final_evaluation/tokenizer_fertility.json"

METRIC_FILES = {
    "Qwen2.5-1.5B": "results/final_evaluation/qwen25_15b_metrics.json",
    "Qwen2.5-3B": "results/final_evaluation/qwen25_3b_metrics.json",
    "SmolLM2-1.7B": "results/final_evaluation/smollm2_17b_metrics.json",
    "Bharat-Tiny-LLM-v3": "results/final_evaluation/bharat_metrics.json",
}

OUTPUT_FILE = "results/final_evaluation/tokenizer_quality_correlation.json"

# ============================================================
# LOAD FERTILITY
# ============================================================

with open(FERTILITY_FILE, "r", encoding="utf-8") as f:
    fertility = json.load(f)

# ============================================================
# LOAD GENERATION METRICS
# ============================================================

metrics = {}

for model, path in METRIC_FILES.items():

    if not os.path.exists(path):
        print(f"WARNING: File not found: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        metrics[model] = json.load(f)

# ============================================================
# BUILD DATA
# ============================================================

models = []

for model in fertility:

    if model not in metrics:
        continue

    models.append(model)

fertility_values = [
    fertility[m]["average_fertility"]
    for m in models
]

ppl_values = [
    metrics[m]["perplexity"]
    for m in models
]

d1_values = [
    metrics[m]["distinct_1"]
    for m in models
]

d2_values = [
    metrics[m]["distinct_2"]
    for m in models
]

repetition_values = [
    metrics[m]["repetition_rate"]
    for m in models
]

mixing_values = [
    metrics[m]["language_mixing_ratio"]
    for m in models
]

# ============================================================
# PEARSON CORRELATION
# ============================================================

def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


correlations = {
    "fertility_vs_perplexity": pearson(
        fertility_values,
        ppl_values
    ),

    "fertility_vs_distinct_1": pearson(
        fertility_values,
        d1_values
    ),

    "fertility_vs_distinct_2": pearson(
        fertility_values,
        d2_values
    ),

    "fertility_vs_repetition": pearson(
        fertility_values,
        repetition_values
    ),

    "fertility_vs_language_mixing": pearson(
        fertility_values,
        mixing_values
    ),
}

# ============================================================
# SAVE MODEL-LEVEL DATA
# ============================================================

model_data = {}

for i, model in enumerate(models):

    model_data[model] = {
        "average_fertility": fertility_values[i],
        "perplexity": ppl_values[i],
        "distinct_1": d1_values[i],
        "distinct_2": d2_values[i],
        "repetition_rate": repetition_values[i],
        "language_mixing_ratio": mixing_values[i],
    }

output = {
    "models": model_data,
    "pearson_correlations": correlations,
    "number_of_models": len(models),
    "note": (
        "Correlations are exploratory model-level correlations "
        "based on four models and should not be interpreted as "
        "causal evidence."
    ),
}

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(
        output,
        f,
        indent=4
    )

# ============================================================
# DISPLAY
# ============================================================

print("=" * 65)
print("TOKENIZER FERTILITY vs GENERATION QUALITY")
print("=" * 65)

print("\nModel-level values:")
print("-" * 65)

for model in models:
    print(
        f"{model:<25} "
        f"Fertility={fertility[model]['average_fertility']:.4f} | "
        f"PPL={metrics[model]['perplexity']:.4f}"
    )

print("\nPearson correlations:")
print("-" * 65)

for name, value in correlations.items():
    print(f"{name:<40}: {value:.4f}")

print("\nSaved to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print(
    "These are exploratory correlations across only 4 models. "
    "They do not establish causation."
)

print("=" * 65)