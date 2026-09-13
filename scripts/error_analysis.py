import json
import re
import os
from collections import Counter

INPUT_FILE = "results/final_evaluation/qwen25_3b_samples.jsonl"

OUTPUT_CANDIDATES = (
    "results/final_evaluation/error_analysis_candidates.jsonl"
)

OUTPUT_FREQUENCY = (
    "results/final_evaluation/error_taxonomy_frequency.json"
)

MAX_EXAMPLES_PER_CATEGORY = 3


# ============================================================
# ERROR DETECTORS
# ============================================================

def detect_token_corruption(text):
    """
    Detect obvious Unicode replacement/corruption characters.
    """
    corruption_chars = [
        "�",
        "\ufffd"
    ]

    return any(char in text for char in corruption_chars)


def detect_truncation(text):
    """
    Detect outputs that appear unfinished.

    Conservative rules:
    - very short output
    - ends with incomplete punctuation/structure
    - unfinished sentence markers
    """

    text = text.strip()

    if len(text.split()) < 5:
        return True

    # Common unfinished endings
    unfinished_patterns = [
        r"\.\.\.$",
        r"…$",
        r":$",
        r",$",
        r"\band$",
        r"\bor$",
        r"\bbut$",
        r"\bki$",
        r"\bke$",
        r"\bhai ki$",
        r"\bto$"
    ]

    for pattern in unfinished_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def detect_word_repetition(text):
    """
    Detect meaningful repeated words or phrases.

    Requires repeated consecutive words/phrases,
    avoiding normal repeated function words.
    """

    words = re.findall(r"\b[\w'-]+\b", text.lower())

    if len(words) < 8:
        return False

    # Repeated 3-word phrase
    for i in range(len(words) - 5):
        phrase1 = words[i:i + 3]
        phrase2 = words[i + 3:i + 6]

        if phrase1 == phrase2:
            return True

    # Repeated 2-word phrase
    for i in range(len(words) - 3):
        phrase1 = words[i:i + 2]
        phrase2 = words[i + 2:i + 4]

        if phrase1 == phrase2:
            return True

    # Same meaningful word repeated 3+ times
    stopwords = {
        "hai", "ka", "ki", "ke", "ko",
        "yeh", "yah", "to", "aur",
        "the", "is", "a", "an", "of",
        "in", "on", "for"
    }

    counts = Counter(words)

    for word, count in counts.items():
        if (
            count >= 3
            and len(word) >= 4
            and word not in stopwords
        ):
            return True

    return False


def detect_emoji_repetition(text):
    """
    Detect excessive repeated emoji usage.
    """

    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U0001F1E6-\U0001F1FF"
        "]+"
    )

    emojis = emoji_pattern.findall(text)

    if not emojis:
        return False

    emoji_chars = []

    for group in emojis:
        emoji_chars.extend(list(group))

    counts = Counter(emoji_chars)

    # Same emoji repeated 3+ times
    return any(count >= 3 for count in counts.values())


def detect_style_deviation(text):
    """
    Conservative detection of outputs that strongly
    deviate from conversational Hinglish style.

    This is heuristic and should be reported as such.
    """

    text_lower = text.lower()

    # Excessive meta/system-like language
    meta_patterns = [
        "as an ai language model",
        "i cannot fulfill",
        "i am unable to",
        "here is a detailed explanation",
        "as a large language model"
    ]

    if any(pattern in text_lower for pattern in meta_patterns):
        return True

    # Very long repetitive formatting
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) >= 8:
        return True

    return False


# ============================================================
# LOAD SAMPLES
# ============================================================

samples = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        record = json.loads(line)

        text = record.get("generated_text", "").strip()

        if text:
            samples.append({
                "id": record.get("id"),
                "generated_text": text
            })


print("=" * 70)
print("AUTOMATIC ERROR ANALYSIS")
print("=" * 70)

print(f"\nTotal generated samples: {len(samples)}")


# ============================================================
# DETECT ERRORS
# ============================================================

detectors = {
    "Token corruption": detect_token_corruption,
    "Likely truncation / unfinished output": detect_truncation,
    "Word/phrase repetition": detect_word_repetition,
    "Emoji repetition": detect_emoji_repetition,
    "Style deviation": detect_style_deviation
}


category_records = {
    category: []
    for category in detectors
}

category_counts = Counter()


for sample in samples:

    text = sample["generated_text"]

    detected_categories = []

    for category, detector in detectors.items():

        try:
            detected = detector(text)

        except Exception:
            detected = False

        if detected:

            detected_categories.append(category)

            category_counts[category] += 1

            category_records[category].append({
                "id": sample["id"],
                "generated_text": text,
                "error_categories": [category]
            })


# ============================================================
# SELECT REPRESENTATIVE EXAMPLES AUTOMATICALLY
# ============================================================

selected_examples = []

for category in detectors:

    examples = category_records[category]

    # Deterministic selection:
    # first 3 automatically detected examples
    selected = examples[:MAX_EXAMPLES_PER_CATEGORY]

    selected_examples.extend(selected)


# ============================================================
# SAVE REPRESENTATIVE EXAMPLES
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CANDIDATES),
    exist_ok=True
)

with open(
    OUTPUT_CANDIDATES,
    "w",
    encoding="utf-8"
) as f:

    for record in selected_examples:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# FREQUENCY ANALYSIS
# ============================================================

frequency = []

for category in detectors:

    count = category_counts[category]

    percentage = (
        count / len(samples) * 100
        if samples
        else 0
    )

    frequency.append({
        "category": category,
        "count": count,
        "percentage_of_all_samples": round(
            percentage,
            2
        ),
        "representative_examples_selected": min(
            count,
            MAX_EXAMPLES_PER_CATEGORY
        )
    })


frequency_output = {
    "total_samples": len(samples),
    "automatic_detection": True,
    "categories": frequency
}


with open(
    OUTPUT_FREQUENCY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        frequency_output,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nError frequency across ALL samples:")
print("-" * 70)

for item in frequency:

    print(
        f"{item['category']:<40}"
        f"{item['count']:>4} "
        f"({item['percentage_of_all_samples']:.2f}%)"
    )


print("\nRepresentative examples selected:")
print("-" * 70)

for category in detectors:

    print(
        f"{category:<40}"
        f"{min(category_counts[category], MAX_EXAMPLES_PER_CATEGORY)}"
    )


print("\nSaved:")
print(OUTPUT_CANDIDATES)
print(OUTPUT_FREQUENCY)

print("=" * 70)