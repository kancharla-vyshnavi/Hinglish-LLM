import json
import re
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

METRICS_FILE = Path("results/final_evaluation/qwen25_3b_metrics.json")
SAMPLES_FILE = Path("results/final_evaluation/qwen25_3b_samples.jsonl")
OUTPUT_FILE = Path("results/final_evaluation/qwen25_3b_extended_metrics.json")


# ============================================================
# LOAD EXISTING METRICS
# ============================================================

with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)


# ============================================================
# LOAD GENERATED SAMPLES
# ============================================================

samples = []

with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        item = json.loads(line)

        if "generated_text" in item:
            samples.append(item["generated_text"])


if not samples:
    raise ValueError("No generated samples found.")


# ============================================================
# TOKENIZATION
# ============================================================

def words(text):
    return re.findall(r"\b[\w']+\b", text.lower(), flags=re.UNICODE)


# ============================================================
# CMI — CODE-MIXING INDEX
# ============================================================
#
# CMI = percentage of tokens that belong to the
# less-dominant language.
#
# Here we use a practical Roman-Hinglish heuristic:
# English words are detected using a controlled English
# vocabulary. Remaining alphabetic tokens are treated as
# Hindi/Hinglish-side tokens.
#
# This is a descriptive automatic measure, not a perfect
# linguistic language identifier.
# ============================================================

ENGLISH_WORDS = {
    "a", "about", "after", "again", "all", "also", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being",
    "but", "by", "can", "could", "did", "do", "does", "doing", "done",
    "for", "from", "get", "getting", "give", "go", "going", "good",
    "got", "had", "has", "have", "he", "her", "here", "him", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "know", "like", "look", "make", "me", "more", "most", "my",
    "need", "no", "not", "now", "of", "on", "one", "only", "or",
    "other", "our", "out", "over", "please", "really", "say", "see",
    "she", "should", "so", "some", "something", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "think", "this", "those", "to", "too", "up", "us", "very",
    "want", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "yes", "you", "your",
    "career", "creative", "activity", "ideas", "traditional",
    "identity", "outfit", "trendy", "specific", "reaction"
}


def calculate_cmi(text):
    token_list = words(text)

    if not token_list:
        return 0.0

    english_count = sum(
        1 for token in token_list
        if token in ENGLISH_WORDS
    )

    total = len(token_list)

    hindi_side_count = total - english_count

    if total == 0:
        return 0.0

    dominant = max(english_count, hindi_side_count)

    # Code-Mixing Index:
    # less-dominant language proportion
    cmi = min(english_count, hindi_side_count) / total

    return cmi


# ============================================================
# AVERAGE GENERATION LENGTH
# ============================================================

lengths = [len(words(text)) for text in samples]

average_generation_length = (
    sum(lengths) / len(lengths)
)


# ============================================================
# UNIQUE SENTENCE RATIO
# ============================================================

def split_sentences(text):
    sentences = re.split(r"[.!?]+", text)
    return [
        s.strip().lower()
        for s in sentences
        if s.strip()
    ]


all_sentences = []

for text in samples:
    all_sentences.extend(split_sentences(text))


if all_sentences:
    unique_sentence_ratio = (
        len(set(all_sentences)) / len(all_sentences)
    )
else:
    unique_sentence_ratio = 0.0


# ============================================================
# AVERAGE WORD LENGTH
# ============================================================

all_words = []

for text in samples:
    all_words.extend(words(text))

if all_words:
    average_word_length = (
        sum(len(word) for word in all_words)
        / len(all_words)
    )
else:
    average_word_length = 0.0


# ============================================================
# AVERAGE CMI
# ============================================================

cmi_values = [
    calculate_cmi(text)
    for text in samples
]

average_cmi = (
    sum(cmi_values) / len(cmi_values)
)


# ============================================================
# BUILD EXTENDED RESULTS
# ============================================================

extended_metrics = {
    "model": metrics.get("model"),
    "base_model": metrics.get("base_model"),

    # Existing evaluation metrics
    "perplexity": metrics.get("perplexity"),
    "distinct_1": metrics.get("distinct_1"),
    "distinct_2": metrics.get("distinct_2"),
    "repetition_rate": metrics.get("repetition_rate"),
    "language_mixing_ratio": metrics.get(
        "language_mixing_ratio"
    ),

    # New metrics
    "cmi": average_cmi,
    "average_generation_length_words": average_generation_length,
    "unique_sentence_ratio": unique_sentence_ratio,
    "average_word_length": average_word_length,

    # Evaluation information
    "generation_samples": len(samples),
    "cmi_note": (
        "CMI is calculated using a controlled English-word "
        "vocabulary and is intended as a descriptive "
        "automatic code-mixing measure."
    )
}


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
    json.dump(
        extended_metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("QWEN2.5-3B EXTENDED EVALUATION")
print("=" * 70)

print(f"Perplexity                 : {extended_metrics['perplexity']:.4f}")
print(f"Distinct-1                 : {extended_metrics['distinct_1']:.4f}")
print(f"Distinct-2                 : {extended_metrics['distinct_2']:.4f}")
print(f"Repetition Rate            : {extended_metrics['repetition_rate']:.4f}")
print(f"Language Mixing Ratio     : {extended_metrics['language_mixing_ratio']:.4f}")
print(f"CMI                        : {extended_metrics['cmi']:.4f}")
print(
    f"Avg Generation Length     : "
    f"{extended_metrics['average_generation_length_words']:.2f} words"
)
print(
    f"Unique Sentence Ratio     : "
    f"{extended_metrics['unique_sentence_ratio']:.4f}"
)
print(
    f"Average Word Length       : "
    f"{extended_metrics['average_word_length']:.2f} characters"
)

print("=" * 70)
print(f"Samples used               : {len(samples)}")
print(f"Saved to                   : {OUTPUT_FILE}")
print("=" * 70)