import os
import json
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PATHS
# ============================================================

METRICS_FILE = "results/final_evaluation/all_models_metrics.json"
GRAPH_DIR = "results/final_evaluation/graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# LOAD ACTUAL RESULTS
# ============================================================

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)


models = list(results.keys())

print("=" * 70)
print("LOADED ACTUAL EVALUATION RESULTS")
print("=" * 70)

for model in models:
    print(model)
    print(results[model])


# ============================================================
# EXTRACT METRICS
# ============================================================

ppl = [
    results[m]["perplexity"]
    for m in models
]

distinct_1 = [
    results[m]["distinct_1"]
    for m in models
]

distinct_2 = [
    results[m]["distinct_2"]
    for m in models
]

repetition = [
    results[m]["repetition_rate"]
    for m in models
]

mixing = [
    results[m]["language_mixing_ratio"]
    for m in models
]


# ============================================================
# SHORT MODEL NAMES
# ============================================================

short_names = [
    "Qwen2.5-1.5B",
    "Qwen2.5-3B",
    "SmolLM2-1.7B",
    "Bharat-Tiny-LLM-v3"
]


# ============================================================
# FUNCTION FOR BAR GRAPHS
# ============================================================

def create_bar_graph(
    values,
    title,
    ylabel,
    filename,
    higher_is_better=True
):

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        short_names,
        values
    )

    plt.title(
        title,
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel(
        "Models",
        fontsize=12
    )

    plt.ylabel(
        ylabel,
        fontsize=12
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    # Add exact values above bars
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()

    path = os.path.join(
        GRAPH_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", path)


# ============================================================
# 1. PERPLEXITY
# ============================================================

create_bar_graph(
    ppl,
    "Perplexity Comparison",
    "Perplexity (Lower is Better)",
    "perplexity_comparison.png",
    higher_is_better=False
)


# ============================================================
# 2. DISTINCT-1
# ============================================================

create_bar_graph(
    distinct_1,
    "Distinct-1 Comparison",
    "Distinct-1 Score (Higher is Better)",
    "distinct1_comparison.png"
)


# ============================================================
# 3. DISTINCT-2
# ============================================================

create_bar_graph(
    distinct_2,
    "Distinct-2 Comparison",
    "Distinct-2 Score (Higher is Better)",
    "distinct2_comparison.png"
)


# ============================================================
# 4. REPETITION RATE
# ============================================================

create_bar_graph(
    repetition,
    "Repetition Rate Comparison",
    "Repetition Rate (Lower is Better)",
    "repetition_comparison.png",
    higher_is_better=False
)


# ============================================================
# 5. LANGUAGE MIXING RATIO
# ============================================================

create_bar_graph(
    mixing,
    "Language Mixing Ratio Comparison",
    "Language Mixing Ratio",
    "language_mixing_comparison.png"
)


# ============================================================
# COMBINED SUMMARY JSON
# ============================================================

summary = {}

for i, model in enumerate(models):

    summary[model] = {
        "perplexity": ppl[i],
        "distinct_1": distinct_1[i],
        "distinct_2": distinct_2[i],
        "repetition_rate": repetition[i],
        "language_mixing_ratio": mixing[i]
    }


summary_file = os.path.join(
    GRAPH_DIR,
    "comparison_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("GRAPH GENERATION COMPLETED")
print("=" * 70)

print("\nGraphs saved inside:")
print(GRAPH_DIR)

print("\nGenerated files:")

print("1. perplexity_comparison.png")
print("2. distinct1_comparison.png")
print("3. distinct2_comparison.png")
print("4. repetition_comparison.png")
print("5. language_mixing_comparison.png")
print("6. comparison_summary.json")

print("\nAll values were read directly from:")
print(METRICS_FILE)