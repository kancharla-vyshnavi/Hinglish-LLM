import json
import os
import matplotlib.pyplot as plt

INPUT_FILE = "results/final_evaluation/error_taxonomy_frequency.json"
OUTPUT_FILE = "results/final_evaluation/graphs/error_taxonomy_frequency.png"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

categories = []
percentages = []

for item in data["categories"]:
    categories.append(item["category"])
    percentages.append(item["percentage_of_all_samples"])

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

plt.figure(figsize=(10, 6))

bars = plt.bar(categories, percentages)

plt.xlabel("Error Category")
plt.ylabel("Detected Samples (%)")
plt.title("Automatic Error Taxonomy Across 500 Generated Samples")

plt.xticks(
    rotation=20,
    ha="right"
)

for bar, value in zip(bars, percentages):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 60)
print("ERROR TAXONOMY GRAPH COMPLETED")
print("=" * 60)

print(f"\nSaved to:")
print(OUTPUT_FILE)