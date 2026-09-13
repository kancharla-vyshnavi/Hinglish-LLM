import json
from collections import Counter

INPUT_FILE = "results/final_evaluation/error_analysis_candidates.jsonl"
OUTPUT_FILE = "results/final_evaluation/error_taxonomy_frequency.json"

counter = Counter()
total_examples = 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        categories = record.get("error_categories", [])

        if categories:
            total_examples += 1

            for category in categories:
                counter[category] += 1

results = []

for category, count in counter.most_common():
    percentage = (count / total_examples) * 100 if total_examples else 0

    results.append({
        "category": category,
        "count": count,
        "percentage": round(percentage, 2)
    })

output = {
    "total_selected_examples": total_examples,
    "category_distribution": results
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)

print("=" * 60)
print("ERROR TAXONOMY FREQUENCY ANALYSIS")
print("=" * 60)

print(f"\nTotal selected examples: {total_examples}")

print("\nCategory distribution:")
print("-" * 60)

for item in results:
    print(
        f"{item['category']:<35} "
        f"{item['count']:>3} "
        f"({item['percentage']:.2f}%)"
    )

print("\nSaved:")
print(OUTPUT_FILE)

print("=" * 60)