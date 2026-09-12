import json
import re
from pathlib import Path
from collections import Counter


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "results/final_evaluation/qwen25_3b_samples.jsonl"
)

OUTPUT_FILE = Path(
    "results/final_evaluation/error_analysis_candidates.jsonl"
)


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())


def get_words(text):
    return re.findall(
        r"\b[\w']+\b",
        text.lower(),
        flags=re.UNICODE
    )


# ============================================================
# 1. TOKEN CORRUPTION
# ============================================================

def has_token_corruption(text):

    patterns = [
        "�",
        "Ã",
        "Â",
        "ðŸ",
        "â€",
        "tsmallgt",
        "smallgt",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


# ============================================================
# 2. EMOJI REPETITION
# ============================================================

def has_emoji_repetition(text):

    symbols = re.findall(
        r"[^\w\s.,!?;:'\"()\[\]{}#/@$%&*+=<>_\-]",
        text,
        flags=re.UNICODE
    )

    if len(symbols) < 5:
        return False

    counts = Counter(symbols)

    # Same emoji/symbol repeated at least 5 times
    return counts.most_common(1)[0][1] >= 5


# ============================================================
# 3. TRUE WORD / PHRASE REPETITION
# ============================================================

def has_real_repetition(text):

    words = get_words(text)

    if len(words) < 10:
        return False

    # --------------------------------------------------------
    # A. Same word repeated immediately
    # --------------------------------------------------------
    #
    # Example:
    # "very very good"
    #
    # --------------------------------------------------------

    for i in range(len(words) - 1):

        if words[i] == words[i + 1]:

            return True


    # --------------------------------------------------------
    # B. Same 3-word sequence repeated immediately
    # --------------------------------------------------------
    #
    # Example:
    #
    # "main ghar ja main ghar ja"
    #
    # --------------------------------------------------------

    for i in range(len(words) - 5):

        first = tuple(
            words[i:i + 3]
        )

        second = tuple(
            words[i + 3:i + 6]
        )

        if first == second:

            return True


    # --------------------------------------------------------
    # C. Same 4-word sequence repeated immediately
    # --------------------------------------------------------
    #
    # Example:
    #
    # "main ghar ja raha main ghar ja raha"
    #
    # --------------------------------------------------------

    for i in range(len(words) - 7):

        first = tuple(
            words[i:i + 4]
        )

        second = tuple(
            words[i + 4:i + 8]
        )

        if first == second:

            return True


    # --------------------------------------------------------
    # D. Strong repeated 2-word phrase
    #
    # Require the same phrase to appear at least THREE times.
    # This prevents normal conversational repetition from
    # being incorrectly classified.
    # --------------------------------------------------------

    phrase_counts = Counter(
        tuple(words[i:i + 2])
        for i in range(len(words) - 1)
    )

    for phrase, count in phrase_counts.items():

        if count >= 3:

            return True


    return False


# ============================================================
# 4. LIKELY TRUNCATION
# ============================================================

def likely_truncated(text):

    text = clean_text(text)

    if len(text) < 35:
        return False

    # Proper sentence ending
    if text[-1] in ".!?。！？":

        return False

    # Remove trailing symbols/emojis
    stripped = re.sub(
        r"[^\w\s]+$",
        "",
        text,
        flags=re.UNICODE
    ).strip()

    if not stripped:
        return False

    words = get_words(stripped)

    if len(words) < 8:
        return False

    last_word = words[-1].lower()

    # Strong incomplete endings only
    strong_endings = {
        "ki",
        "ke",
        "ka",
        "ko",
        "se",
        "mein",
        "me",
        "par",
        "aur",
        "lekin",
        "kyunki",
        "because",
        "but",
        "and",
        "with",
        "for",
        "from",
        "about",
        "how",
        "what",
        "which",
        "where",
        "when",
        "to",
    }

    if last_word in strong_endings:

        return True

    # Strong multi-word endings
    lower = stripped.lower()

    strong_phrases = [
        "tumhare liye",
        "mere liye",
        "aapke liye",
        "mere saath",
        "aapke saath",
        "iske baare",
        "uske baare",
        "aisa lagta",
        "how to",
        "what to",
        "ways to",
        "tips on",
        "because of",
    ]

    for phrase in strong_phrases:

        if lower.endswith(phrase):

            return True

    # Single-character final word
    if len(last_word) == 1:

        return True

    return False


# ============================================================
# 5. STYLE DEVIATION
# ============================================================

def has_style_deviation(text):

    hashtags = re.findall(
        r"#\w+",
        text
    )

    return len(hashtags) >= 4


# ============================================================
# CLASSIFY
# ============================================================

def classify(text):

    categories = []

    if has_token_corruption(text):

        categories.append(
            "Token corruption"
        )

    if has_emoji_repetition(text):

        categories.append(
            "Emoji repetition"
        )

    if has_real_repetition(text):

        categories.append(
            "Word/phrase repetition"
        )

    if has_style_deviation(text):

        categories.append(
            "Style deviation"
        )

    if likely_truncated(text):

        categories.append(
            "Likely truncation / unfinished output"
        )

    if not categories:

        categories.append(
            "No obvious error"
        )

    return categories


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    samples = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# ============================================================
# ANALYSIS
# ============================================================

results = []

for sample in samples:

    text = clean_text(
        sample.get(
            "generated_text",
            ""
        )
    )

    results.append({
        "id": sample["id"],
        "generated_text": text,
        "error_categories": classify(text)
    })


# ============================================================
# CATEGORY COUNTS
# ============================================================

category_counts = Counter()

for item in results:

    for category in item["error_categories"]:

        category_counts[category] += 1


print()
print("=" * 70)
print("AUTOMATIC ERROR ANALYSIS - FINAL")
print("=" * 70)

print(
    f"Total samples: {len(samples)}"
)

print()
print("CATEGORY COUNTS")
print("-" * 70)


categories = [
    "No obvious error",
    "Likely truncation / unfinished output",
    "Word/phrase repetition",
    "Token corruption",
    "Emoji repetition",
    "Style deviation",
]


for category in categories:

    count = category_counts.get(
        category,
        0
    )

    percentage = (
        count / len(samples) * 100
    )

    print(
        f"{category:40s}: "
        f"{count:3d} "
        f"({percentage:.1f}%)"
    )


# ============================================================
# SELECT EXACTLY 3 PER ERROR CATEGORY
# ============================================================

selected = []

used_ids = set()

selection_categories = [
    "Token corruption",
    "Likely truncation / unfinished output",
    "Word/phrase repetition",
    "Emoji repetition",
    "Style deviation",
]


for category in selection_categories:

    # Prefer examples containing ONLY this category
    candidates = [
        item
        for item in results
        if item["id"] not in used_ids
        and item["error_categories"] == [category]
    ]

    # Sort by length so examples are informative
    candidates.sort(
        key=lambda x: len(
            x["generated_text"]
        ),
        reverse=True
    )

    # If fewer than 3 clean examples exist,
    # use multi-category examples.
    if len(candidates) < 3:

        candidates = [
            item
            for item in results
            if item["id"] not in used_ids
            and category in item["error_categories"]
        ]

        candidates.sort(
            key=lambda x: len(
                x["generated_text"]
            ),
            reverse=True
        )

    chosen = 0

    for item in candidates:

        if item["id"] in used_ids:
            continue

        selected.append(item)

        used_ids.add(
            item["id"]
        )

        chosen += 1

        if chosen == 3:
            break


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for item in selected:

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 70)
print("SELECTED REPRESENTATIVE EXAMPLES")
print("=" * 70)


for i, item in enumerate(
    selected,
    start=1
):

    print()

    print(
        f"Example {i}"
    )

    print(
        f"ID       : "
        f"{item['id']}"
    )

    print(
        f"Category : "
        f"{', '.join(item['error_categories'])}"
    )

    print(
        f"Output   : "
        f"{item['generated_text']}"
    )


# ============================================================
# VERIFY DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("REPRESENTATIVE EXAMPLE DISTRIBUTION")
print("=" * 70)


distribution = Counter()

for item in selected:

    # Determine the category used for selection
    for category in selection_categories:

        if category in item["error_categories"]:

            distribution[category] += 1

            break


for category in selection_categories:

    print(
        f"{category:40s}: "
        f"{distribution.get(category, 0)}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL ERROR ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Total samples: {len(samples)}"
)

print(
    f"Selected examples: {len(selected)}"
)

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("=" * 70)