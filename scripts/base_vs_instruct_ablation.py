import json
import os
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

INSTRUCT_FILE = (
    "results/final_evaluation/qwen25_3b_metrics.json"
)

BASE_FILE = (
    "results/final_evaluation/qwen25_3b_base_metrics.json"
)

OUTPUT_DIR = (
    "results/final_evaluation"
)

GRAPH_DIR = (
    "results/final_evaluation/graphs"
)

COMPARISON_FILE = (
    "results/final_evaluation/"
    "qwen25_3b_base_vs_instruct.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 70)
print("QWEN2.5-3B BASE vs INSTRUCT ABLATION")
print("=" * 70)

with open(
    INSTRUCT_FILE,
    "r",
    encoding="utf-8"
) as f:
    instruct = json.load(f)

with open(
    BASE_FILE,
    "r",
    encoding="utf-8"
) as f:
    base = json.load(f)


# ============================================================
# COMMON METRICS ONLY
# ============================================================

metrics = [
    "perplexity",
    "distinct_1",
    "distinct_2",
    "repetition_rate",
    "language_mixing_ratio"
]


# ============================================================
# COMPARISON
# ============================================================

comparison = {
    "instruct_model": "Qwen2.5-3B-Instruct + QLoRA",
    "base_model": "Qwen2.5-3B-Base + QLoRA",

    "same_dataset": True,
    "same_training_steps": 22500,
    "seed": 42,

    "metrics": {}
}


for metric in metrics:

    instruct_value = instruct.get(metric)
    base_value = base.get(metric)

    if (
        instruct_value is None
        or base_value is None
    ):
        print(
            f"Skipping missing metric: {metric}"
        )
        continue

    difference = (
        base_value
        - instruct_value
    )

    if instruct_value != 0:

        percentage_change = (
            difference
            / abs(instruct_value)
        ) * 100

    else:

        percentage_change = None

    comparison["metrics"][metric] = {

        "instruct":
            instruct_value,

        "base":
            base_value,

        "absolute_difference":
            difference,

        "percentage_change":
            percentage_change
    }


# ============================================================
# SAVE COMPARISON
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)

with open(
    COMPARISON_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        comparison,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("ABLATION RESULTS")
print("=" * 70)

for metric, values in comparison[
    "metrics"
].items():

    print(f"\n{metric}")

    print(
        f"  Instruct : "
        f"{values['instruct']:.6f}"
    )

    print(
        f"  Base     : "
        f"{values['base']:.6f}"
    )

    print(
        f"  Difference: "
        f"{values['absolute_difference']:.6f}"
    )

    if values[
        "percentage_change"
    ] is not None:

        print(
            f"  Change   : "
            f"{values['percentage_change']:.2f}%"
        )


# ============================================================
# GRAPH 1 — PERPLEXITY
# ============================================================

models = [
    "Qwen2.5-3B\nInstruct",
    "Qwen2.5-3B\nBase"
]

ppl_values = [
    instruct["perplexity"],
    base["perplexity"]
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    models,
    ppl_values
)

plt.ylabel(
    "Perplexity ↓"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Perplexity"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_perplexity.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 2 — DISTINCT-1
# ============================================================

values = [
    instruct["distinct_1"],
    base["distinct_1"]
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    models,
    values
)

plt.ylabel(
    "Distinct-1 ↑"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Distinct-1"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_distinct1.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 3 — DISTINCT-2
# ============================================================

values = [
    instruct["distinct_2"],
    base["distinct_2"]
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    models,
    values
)

plt.ylabel(
    "Distinct-2 ↑"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Distinct-2"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_distinct2.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 4 — REPETITION
# ============================================================

values = [
    instruct["repetition_rate"],
    base["repetition_rate"]
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    models,
    values
)

plt.ylabel(
    "Repetition Rate ↓"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Repetition"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_repetition.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 5 — LANGUAGE MIXING
# ============================================================

values = [
    instruct["language_mixing_ratio"],
    base["language_mixing_ratio"]
]

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    models,
    values
)

plt.ylabel(
    "Language Mixing Ratio"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Language Mixing"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_language_mixing.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# GRAPH 6 — OVERALL COMPARISON
# ============================================================

metric_names = [
    "Distinct-1",
    "Distinct-2",
    "Repetition",
    "Language Mixing"
]

instruct_values = [
    instruct["distinct_1"],
    instruct["distinct_2"],
    instruct["repetition_rate"],
    instruct["language_mixing_ratio"]
]

base_values = [
    base["distinct_1"],
    base["distinct_2"],
    base["repetition_rate"],
    base["language_mixing_ratio"]
]

x = range(
    len(metric_names)
)

width = 0.35

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    [i - width / 2 for i in x],
    instruct_values,
    width=width,
    label="Instruct"
)

plt.bar(
    [i + width / 2 for i in x],
    base_values,
    width=width,
    label="Base"
)

plt.xticks(
    list(x),
    metric_names
)

plt.ylabel(
    "Metric Value"
)

plt.title(
    "Qwen2.5-3B Base vs Instruct: Generation Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "base_vs_instruct_generation_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BASE vs INSTRUCT ABLATION COMPLETED")
print("=" * 70)

print(
    "\nComparison saved:"
)

print(COMPARISON_FILE)

print(
    "\nGraphs saved:"
)

print(GRAPH_DIR)

print("\nGenerated graphs:")

print("1. base_vs_instruct_perplexity.png")
print("2. base_vs_instruct_distinct1.png")
print("3. base_vs_instruct_distinct2.png")
print("4. base_vs_instruct_repetition.png")
print("5. base_vs_instruct_language_mixing.png")
print("6. base_vs_instruct_generation_comparison.png")

print("=" * 70)