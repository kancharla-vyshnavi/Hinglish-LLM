import json
import random
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "data/processed/merged_cleaned.jsonl"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"
TEST_FILE = "data/processed/test.jsonl"

TOTAL_SIZE = 370000
TRAIN_SIZE = 360000
VAL_SIZE = 5000
TEST_SIZE = 5000

SEED = 42


# ============================================================
# CHECK SIZES
# ============================================================

if TRAIN_SIZE + VAL_SIZE + TEST_SIZE != TOTAL_SIZE:
    raise ValueError("Train + Validation + Test sizes do not equal 370,000")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("              CREATING 370K DATASET SPLIT")
print("=" * 70)

print("\nLoading cleaned dataset...")

data = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)

            if "text" not in item:
                continue

            if not item["text"].strip():
                continue

            data.append(item)

        except json.JSONDecodeError:
            continue


print(f"Available examples: {len(data):,}")


# ============================================================
# CHECK TOTAL DATA
# ============================================================

if len(data) < TOTAL_SIZE:
    raise ValueError(
        f"Need {TOTAL_SIZE:,} examples, "
        f"but only {len(data):,} are available."
    )


# ============================================================
# GROUP BY SOURCE
# ============================================================

source_groups = defaultdict(list)

for item in data:
    source = item.get("source", "UNKNOWN")
    source_groups[source].append(item)


print("\nSource distribution:")
print("-" * 50)

for source, items in source_groups.items():
    print(f"{source:<30}: {len(items):,}")


# ============================================================
# PROPORTIONAL SOURCE SAMPLING
# ============================================================

print("\nSelecting exactly 370,000 examples...")

random.seed(SEED)

source_names = list(source_groups.keys())
total_available = sum(len(source_groups[s]) for s in source_names)

allocations = {}

# Initial proportional allocation
for source in source_names:
    proportion = len(source_groups[source]) / total_available

    allocations[source] = int(
        proportion * TOTAL_SIZE
    )


# ============================================================
# FIX ROUNDING DIFFERENCE
# ============================================================

current_total = sum(allocations.values())

remaining = TOTAL_SIZE - current_total

# Add remaining examples based on largest source sizes
sorted_sources = sorted(
    source_names,
    key=lambda s: len(source_groups[s]),
    reverse=True
)

index = 0

while remaining > 0:
    source = sorted_sources[index % len(sorted_sources)]

    if allocations[source] < len(source_groups[source]):
        allocations[source] += 1
        remaining -= 1

    index += 1


# ============================================================
# SAMPLE FROM EACH SOURCE
# ============================================================

selected_data = []

print("\nSelected source distribution:")
print("-" * 60)

for source in source_names:

    count = allocations[source]

    sampled = random.sample(
        source_groups[source],
        count
    )

    selected_data.extend(sampled)

    percentage = (count / TOTAL_SIZE) * 100

    print(
        f"{source:<30}: "
        f"{count:>8,} "
        f"({percentage:>6.2f}%)"
    )


print("-" * 60)
print(f"{'TOTAL':<30}: {len(selected_data):>8,}")


# ============================================================
# FINAL SHUFFLE
# ============================================================

random.shuffle(selected_data)


# ============================================================
# SPLIT
# ============================================================

train_data = selected_data[:TRAIN_SIZE]

val_data = selected_data[
    TRAIN_SIZE:
    TRAIN_SIZE + VAL_SIZE
]

test_data = selected_data[
    TRAIN_SIZE + VAL_SIZE:
]


# ============================================================
# VERIFY
# ============================================================

print("\nFinal split:")
print("-" * 50)

print(f"Train       : {len(train_data):,}")
print(f"Validation  : {len(val_data):,}")
print(f"Test        : {len(test_data):,}")
print(f"Total       : {len(train_data) + len(val_data) + len(test_data):,}")


if len(train_data) != TRAIN_SIZE:
    raise ValueError("Train size mismatch!")

if len(val_data) != VAL_SIZE:
    raise ValueError("Validation size mismatch!")

if len(test_data) != TEST_SIZE:
    raise ValueError("Test size mismatch!")


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_jsonl(data, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# SAVE FILES
# ============================================================

print("\nSaving files...")

save_jsonl(train_data, TRAIN_FILE)
save_jsonl(val_data, VAL_FILE)
save_jsonl(test_data, TEST_FILE)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("              DATASET SPLIT COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(f"✅ {TRAIN_FILE}")
print(f"✅ {VAL_FILE}")
print(f"✅ {TEST_FILE}")

print("\nFinal dataset:")
print("✅ Train       = 360,000")
print("✅ Validation  =   5,000")
print("✅ Test        =   5,000")
print("✅ Total       = 370,000")

print("\n🎯 Ready for QLoRA training!")