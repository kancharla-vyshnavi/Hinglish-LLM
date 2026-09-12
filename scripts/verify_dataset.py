import json
from collections import Counter

files = {
    "Train": "data/processed/train.jsonl",
    "Validation": "data/processed/validation.jsonl",
    "Test": "data/processed/test.jsonl",
}

print("=" * 70)
print("              DATASET VERIFICATION")
print("=" * 70)

total = 0

for name, filepath in files.items():

    count = 0
    sources = Counter()
    empty = 0

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            count += 1
            sources[item["source"]] += 1

            if not item.get("text", "").strip():
                empty += 1

    total += count

    print(f"\n{name}")
    print("-" * 50)
    print(f"Total examples : {count:,}")
    print(f"Empty examples : {empty}")
    print("Sources:")

    for source, source_count in sources.items():
        percentage = source_count / count * 100
        print(f"  {source:<28}: {source_count:,} ({percentage:.2f}%)")


print("\n" + "=" * 70)
print(f"TOTAL EXAMPLES : {total:,}")
print("=" * 70)

if total == 370000:
    print("✅ EXACT 370,000 DATASET VERIFIED")
else:
    print("❌ TOTAL COUNT IS NOT 370,000")