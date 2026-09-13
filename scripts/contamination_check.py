import json
import os
import re

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"
TEST_FILE = "data/processed/test.jsonl"

OUTPUT_FILE = "results/final_evaluation/contamination_check.json"


def normalize_text(text):
    """
    Normalize text for exact contamination checking.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)

    return text


def load_jsonl(path):
    texts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            text = record.get("text", "")

            normalized = normalize_text(text)

            if normalized:
                texts.append(normalized)

    return texts


def check_overlap(set_a, set_b):
    return set_a.intersection(set_b)


print("=" * 70)
print("DATA CONTAMINATION / PROVENANCE CHECK")
print("=" * 70)

print("\nLoading datasets...")

train = load_jsonl(TRAIN_FILE)
validation = load_jsonl(VAL_FILE)
test = load_jsonl(TEST_FILE)

print(f"Train samples       : {len(train):,}")
print(f"Validation samples  : {len(validation):,}")
print(f"Test samples        : {len(test):,}")


# Convert to sets
train_set = set(train)
validation_set = set(validation)
test_set = set(test)


# ============================================================
# OVERLAP CHECKS
# ============================================================

train_test_overlap = check_overlap(train_set, test_set)
train_val_overlap = check_overlap(train_set, validation_set)
val_test_overlap = check_overlap(validation_set, test_set)


# ============================================================
# PERCENTAGES
# ============================================================

train_test_percentage = (
    len(train_test_overlap) / len(test_set) * 100
    if test_set else 0
)

train_val_percentage = (
    len(train_val_overlap) / len(validation_set) * 100
    if validation_set else 0
)

val_test_percentage = (
    len(val_test_overlap) / len(test_set) * 100
    if test_set else 0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("EXACT NORMALIZED TEXT OVERLAP")
print("=" * 70)

print(
    f"\nTrain ↔ Test       : "
    f"{len(train_test_overlap):,} "
    f"({train_test_percentage:.4f}% of test)"
)

print(
    f"Train ↔ Validation : "
    f"{len(train_val_overlap):,} "
    f"({train_val_percentage:.4f}% of validation)"
)

print(
    f"Validation ↔ Test  : "
    f"{len(val_test_overlap):,} "
    f"({val_test_percentage:.4f}% of test)"
)


# ============================================================
# CONTAMINATION STATUS
# ============================================================

contamination_free = (
    len(train_test_overlap) == 0
    and len(train_val_overlap) == 0
    and len(val_test_overlap) == 0
)

print("\n" + "=" * 70)

if contamination_free:
    print("RESULT: NO EXACT NORMALIZED TEXT OVERLAP DETECTED")
else:
    print("RESULT: OVERLAP DETECTED")

print("=" * 70)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "method": {
        "type": "exact_normalized_text_match",
        "normalization": [
            "lowercase",
            "strip leading/trailing whitespace",
            "collapse consecutive whitespace"
        ]
    },
    "dataset_sizes": {
        "train": len(train),
        "validation": len(validation),
        "test": len(test)
    },
    "overlap": {
        "train_test": {
            "count": len(train_test_overlap),
            "percentage_of_test": train_test_percentage
        },
        "train_validation": {
            "count": len(train_val_overlap),
            "percentage_of_validation": train_val_percentage
        },
        "validation_test": {
            "count": len(val_test_overlap),
            "percentage_of_test": val_test_percentage
        }
    },
    "exact_contamination_detected": not contamination_free
}


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
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


print(f"\nSaved report:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("CONTAMINATION CHECK COMPLETED")
print("=" * 70)