import json
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "results/final_evaluation/llm_judge_summary.json"
)

OUTPUT_DIR = Path(
    "results/final_evaluation/graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


metrics = data["metrics"]

models = list(metrics.keys())


# ============================================================
# METRIC DISPLAY NAMES
# ============================================================

metric_names = {
    "fluency": "Fluency",
    "coherence": "Coherence",
    "relevance": "Relevance",
    "code_mixing_quality": "Code-Mixing Quality",
    "overall_quality": "Overall Quality"
}


# ============================================================
# CREATE INDIVIDUAL GRAPHS
# ============================================================

for metric_key, metric_title in metric_names.items():

    values = [
        metrics[model][metric_key]
        for model in models
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        models,
        values
    )

    plt.title(
        f"LLM-as-a-Judge: {metric_title}"
    )

    plt.ylabel(
        "Score (1–5)"
    )

    plt.ylim(
        0,
        5
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    # Display values above bars
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR /
        f"llm_judge_{metric_key}.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


# ============================================================
# COMBINED GRAPH
# ============================================================

all_values = []

for metric_key in metric_names:

    values = [
        metrics[model][metric_key]
        for model in models
    ]

    all_values.append(values)


plt.figure(
    figsize=(13, 7)
)

x_positions = range(
    len(models)
)

number_of_metrics = len(
    metric_names
)

bar_width = 0.15

for i, (metric_key, metric_title) in enumerate(
    metric_names.items()
):

    values = [
        metrics[model][metric_key]
        for model in models
    ]

    positions = [
        x + (i - 2) * bar_width
        for x in x_positions
    ]

    plt.bar(
        positions,
        values,
        width=bar_width,
        label=metric_title
    )


plt.title(
    "LLM-as-a-Judge Comparison of Hinglish LLMs"
)

plt.ylabel(
    "Score (1–5)"
)

plt.ylim(
    0,
    5
)

plt.xticks(
    list(x_positions),
    models,
    rotation=20,
    ha="right"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

combined_file = (
    OUTPUT_DIR /
    "llm_judge_comparison.png"
)

plt.savefig(
    combined_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved: {combined_file}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("LLM JUDGE GRAPHS COMPLETED")
print("=" * 70)

print()
print(
    f"Graphs saved in: {OUTPUT_DIR}"
)

print()
print("Metrics:")
for metric in metric_names.values():
    print(f"  - {metric}")

print()
print("=" * 70)