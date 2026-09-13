import json
import os
from collections import Counter

SPLITS = {
    "train": "data/processed/train.jsonl",
    "validation": "data/processed/validation.jsonl",
    "test": "data/processed/test.jsonl"
}

OUTPUT_FILE = "results/final_evaluation/provenance_analysis.json"


def load_sources(path):
    sources = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            sources.append(record.get("source", "UNKNOWN"))

    return sources


results = {}

print("=" * 70)
print("DATASET PROVENANCE / SOURCE DISTRIBUTION ANALYSIS")
print("=" * 70)

for split_name, path in SPLITS.items():

    sources = load_sources(path)
    counts = Counter(sources)
    total = len(sources)

    print(f"\n{split_name.upper()}")
    print("-" * 70)
    print(f"Total samples: {total:,}")

    split_result = {
        "total_samples": total,
        "sources": {}
    }

    for source, count in sorted(counts.items()):
        percentage = (count / total * 100) if total else 0

        print(
            f"{source:<30} "
            f"{count:>8,} "
            f"({percentage:>6.2f}%)"
        )

        split_result["sources"][source] = {
            "count": count,
            "percentage": percentage
        }

    results[split_name] = split_result


# ============================================================
# SOURCE PRESENCE CHECK
# ============================================================

all_sources = set()

for split in results.values():
    all_sources.update(split["sources"].keys())

print("\n" + "=" * 70)
print("SOURCE PRESENCE CHECK")
print("=" * 70)

for source in sorted(all_sources):

    present = []

    for split_name in SPLITS:
        if source in results[split_name]["sources"]:
            present.append(split_name)

    print(
        f"{source:<30} -> "
        f"{', '.join(present)}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("PROVENANCE ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nSaved report:")
print(OUTPUT_FILE)