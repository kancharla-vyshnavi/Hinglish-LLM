import json
import re
from pathlib import Path

# ============================================================
# PREPROCESS + MERGE ALL 4 HINGLISH DATASETS
# ============================================================

INPUT_FILES = [
    ("data/interim/hinmix_420k.jsonl", "HINMIX"),
    ("data/raw/comi_lingua_tn.jsonl", "COMI-LINGUA-TN"),
    ("data/raw/hinge_hinglish.jsonl", "HinGE"),
    ("data/raw/conversations_53477.jsonl", "Hinglish-Conversations"),
]

OUTPUT_FILE = Path("data/processed/merged_cleaned.jsonl")


def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


print("=" * 70)
print("       HINGLISH DATA PREPROCESSING")
print("=" * 70)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

seen = set()

total_input = 0
valid_examples = 0
duplicates = 0
invalid = 0

source_counts = {}

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as out:

    for file_path, source_name in INPUT_FILES:

        path = Path(file_path)

        print(f"\nProcessing: {source_name}")
        print(f"File     : {file_path}")

        if not path.exists():
            print("❌ FILE NOT FOUND - SKIPPING")
            continue

        source_count = 0

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                total_input += 1

                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue

                text = clean_text(item.get("text", ""))

                if not text:
                    invalid += 1
                    continue

                # Case-insensitive exact duplicate check
                key = text.lower()

                if key in seen:
                    duplicates += 1
                    continue

                seen.add(key)

                output_item = {
                    "text": text,
                    "source": source_name
                }

                out.write(
                    json.dumps(
                        output_item,
                        ensure_ascii=False
                    ) + "\n"
                )

                valid_examples += 1
                source_count += 1

        source_counts[source_name] = source_count

        print(
            f"Unique examples added: {source_count:,}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("              PREPROCESSING COMPLETE")
print("=" * 70)

print(f"\nTotal input examples : {total_input:,}")
print(f"Valid unique examples: {valid_examples:,}")
print(f"Duplicates removed   : {duplicates:,}")
print(f"Invalid/empty        : {invalid:,}")

print("\nSource distribution:")

for source, count in source_counts.items():
    print(f"{source:30s}: {count:,}")

print("\nOutput file:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("✅ MERGED CLEANED DATASET CREATED")
print("=" * 70)