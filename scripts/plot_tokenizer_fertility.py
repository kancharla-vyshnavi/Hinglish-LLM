import json
import os
import matplotlib.pyplot as plt

INPUT_FILE = "results/final_evaluation/tokenizer_fertility.json"
OUTPUT_DIR = "results/final_evaluation/graphs"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "tokenizer_fertility_comparison.png"
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

models = list(data.keys())
fertility = [
    data[model]["average_fertility"]
    for model in models
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.figure(figsize=(9, 5))
bars = plt.bar(models, fertility)

plt.ylabel("Average Tokenizer Fertility")
plt.xlabel("Model")
plt.title("Tokenizer Fertility Comparison Across Models")
plt.xticks(rotation=20, ha="right")

for bar, value in zip(bars, fertility):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print("=" * 60)
print("TOKENIZER FERTILITY GRAPH COMPLETED")
print("=" * 60)
print(f"\nSaved to:\n{OUTPUT_FILE}")