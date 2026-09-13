import json
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "results/final_evaluation"
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
# ACTUAL FINAL RESULTS
# ============================================================

models = {
    "Qwen2.5-1.5B": {
        "ppl": 8.1047,
        "trainable_pct": 1.0283,
        "throughput": 1.105
    },
    "Qwen2.5-3B": {
        "ppl": 7.3397,
        "trainable_pct": 0.8734,
        "throughput": 0.619
    },
    "SmolLM2-1.7B": {
        "ppl": 7.5022,
        "trainable_pct": 0.9883,
        "throughput": 0.907
    },
    "Bharat-Tiny-LLM-v3": {
        "ppl": 10.1036,
        "trainable_pct": 0.3151,
        "throughput": 1.060
    }
}

# ============================================================
# SAVE ANALYSIS DATA
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "efficiency_quality_analysis.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(models, f, indent=4)

# ============================================================
# GRAPH 1: PPL VS TRAINABLE PARAMETERS
# ============================================================

plt.figure(figsize=(9, 6))

for model, values in models.items():
    plt.scatter(
        values["trainable_pct"],
        values["ppl"],
        s=100,
        label=model
    )

    plt.annotate(
        model,
        (
            values["trainable_pct"],
            values["ppl"]
        ),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.xlabel("Trainable Parameters (%)")
plt.ylabel("Perplexity (Lower is Better)")
plt.title("Efficiency–Quality Frontier")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "efficiency_quality_frontier.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GRAPH 2: GENERATION THROUGHPUT
# ============================================================

names = list(models.keys())
throughput = [
    models[m]["throughput"]
    for m in names
]

plt.figure(figsize=(9, 5))

bars = plt.bar(names, throughput)

plt.ylabel("Generation Throughput (samples/sec)")
plt.xlabel("Model")
plt.title("Generation Efficiency Comparison")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, throughput):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "generation_throughput_comparison.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# PRINT SUMMARY
# ============================================================

print("=" * 65)
print("EFFICIENCY–QUALITY ANALYSIS COMPLETED")
print("=" * 65)

print("\nModel Summary:")
print("-" * 65)

for model, values in models.items():
    print(
        f"{model:<25} "
        f"PPL={values['ppl']:.4f} | "
        f"Trainable={values['trainable_pct']:.4f}% | "
        f"Throughput={values['throughput']:.3f} samples/s"
    )

print("\nSaved:")
print("1. results/final_evaluation/efficiency_quality_analysis.json")
print("2. results/final_evaluation/graphs/efficiency_quality_frontier.png")
print("3. results/final_evaluation/graphs/generation_throughput_comparison.png")

print("\nNote:")
print(
    "Training time and peak training GPU memory were not included "
    "because they were not recorded during the original training runs."
)

print("=" * 65)