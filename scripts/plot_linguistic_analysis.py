import json
import os
import matplotlib.pyplot as plt

INPUT_FILE = "results/final_evaluation/linguistic_analysis.json"
GRAPH_DIR = "results/final_evaluation/graphs"

os.makedirs(GRAPH_DIR, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

models = list(data.keys())

# ============================================================
# 1. CODE-MIXED SAMPLE RATIO
# ============================================================

code_mixed = [
    data[model]["code_mixed_sample_ratio"] * 100
    for model in models
]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, code_mixed)

plt.xlabel("Model")
plt.ylabel("Code-Mixed Samples (%)")
plt.title("Code-Mixed Sample Ratio Across Models")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, code_mixed):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "code_mixed_sample_ratio.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. AVERAGE LANGUAGE SWITCHES
# ============================================================

switches = [
    data[model]["average_language_switches"]
    for model in models
]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, switches)

plt.xlabel("Model")
plt.ylabel("Average Language Switches")
plt.title("Average Language Switching Across Models")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, switches):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "average_language_switches.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. AVERAGE CMI
# ============================================================

cmi = [
    data[model]["average_cmi"]
    for model in models
]

plt.figure(figsize=(9, 5))

bars = plt.bar(models, cmi)

plt.xlabel("Model")
plt.ylabel("Average CMI")
plt.title("Code-Mixing Index Across Models")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, cmi):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        GRAPH_DIR,
        "code_mixing_index.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# DONE
# ============================================================

print("=" * 65)
print("LINGUISTIC ANALYSIS GRAPHS COMPLETED")
print("=" * 65)

print("\nSaved graphs:")

print(
    "1.",
    os.path.join(
        GRAPH_DIR,
        "code_mixed_sample_ratio.png"
    )
)

print(
    "2.",
    os.path.join(
        GRAPH_DIR,
        "average_language_switches.png"
    )
)

print(
    "3.",
    os.path.join(
        GRAPH_DIR,
        "code_mixing_index.png"
    )
)

print("=" * 65)