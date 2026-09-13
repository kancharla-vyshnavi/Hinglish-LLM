import json
import os
import re
from statistics import mean

# ============================================================
# INPUT FILES
# ============================================================

MODEL_FILES = {
    "Qwen2.5-1.5B":
        "results/final_evaluation/samples/Qwen25_15B_samples.jsonl",

    "Qwen2.5-3B":
        "results/final_evaluation/samples/Qwen25_3B_samples.jsonl",

    "SmolLM2-1.7B":
        "results/final_evaluation/samples/SmolLM2_17B_samples.jsonl",

    "Bharat-Tiny-LLM-v3":
        "results/final_evaluation/samples/Bharat_Tiny_LLM_v3_samples.jsonl"
}

OUTPUT_FILE = (
    "results/final_evaluation/linguistic_analysis.json"
)


# ============================================================
# CONTROLLED ENGLISH VOCABULARY
# ============================================================

ENGLISH_WORDS = {
    "the", "is", "are", "am", "was", "were", "be", "been",
    "being", "a", "an", "and", "or", "but", "if", "then",
    "because", "so", "for", "from", "with", "without",
    "about", "into", "over", "under", "after", "before",
    "this", "that", "these", "those", "it", "its",
    "you", "your", "we", "our", "they", "their",
    "he", "she", "his", "her", "me", "my", "i",
    "can", "could", "will", "would", "should", "may",
    "might", "must", "do", "does", "did",
    "have", "has", "had",
    "not", "very", "more", "most", "also", "just",
    "only", "really", "good", "bad", "best", "better",
    "important", "different", "same", "new", "old",
    "time", "day", "way", "people", "person",
    "thing", "things", "work", "life", "world",
    "school", "college", "student", "study", "learn",
    "learning", "exam", "question", "answer",
    "help", "start", "stop", "use", "using",
    "make", "made", "know", "think", "want",
    "need", "like", "love", "go", "come",
    "get", "give", "take", "tell", "see",
    "look", "feel", "try", "keep",
    "easy", "hard", "simple", "possible",
    "sure", "okay", "yes", "no",
    "please", "thanks", "thank",
    "today", "tomorrow", "yesterday",
    "now", "later", "here", "there",
    "online", "offline", "phone", "computer",
    "internet", "data", "code", "coding",
    "python", "java", "project", "model",
    "machine", "learning", "ai",
    "english", "language", "sentence", "word",
    "idea", "problem", "solution", "example",
    "first", "second", "third", "next",
    "start", "finish", "complete",
    "happy", "sad", "interesting",
    "experience", "history", "historical",
    "beautiful", "natural", "conversation",
    "message", "information", "understand",
    "understanding", "practice", "improve"
}


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?",
        text.lower()
    )


# ============================================================
# ENGLISH WORD DETECTION
# ============================================================

def is_english_word(word):
    return word in ENGLISH_WORDS


# ============================================================
# ANALYZE ONE TEXT
# ============================================================

def analyze_text(text):

    words = tokenize(text)

    if not words:
        return {
            "word_count": 0,
            "english_words": 0,
            "non_english_words": 0,
            "english_word_ratio": 0.0,
            "non_english_word_ratio": 0.0,
            "code_mixed": False,
            "switch_count": 0,
            "cmi": 0.0
        }

    language_labels = []

    english_count = 0

    for word in words:

        if is_english_word(word):
            language_labels.append("EN")
            english_count += 1
        else:
            language_labels.append("HI")

    non_english_count = len(words) - english_count

    english_ratio = english_count / len(words)

    non_english_ratio = non_english_count / len(words)

    # --------------------------------------------------------
    # Language switching
    # --------------------------------------------------------

    switch_count = 0

    for i in range(1, len(language_labels)):

        if language_labels[i] != language_labels[i - 1]:
            switch_count += 1

    # --------------------------------------------------------
    # Code-mixed sentence
    # --------------------------------------------------------

    code_mixed = (
        english_count > 0
        and non_english_count > 0
    )

    # --------------------------------------------------------
    # Descriptive CMI
    #
    # CMI = 2 * min(EN, HI) / total
    #
    # Higher value = more balanced mixing.
    # This is a heuristic because Romanized Hindi
    # is difficult to identify automatically.
    # --------------------------------------------------------

    cmi = (
        2 * min(
            english_count,
            non_english_count
        ) / len(words)
    )

    return {
        "word_count": len(words),
        "english_words": english_count,
        "non_english_words": non_english_count,
        "english_word_ratio": english_ratio,
        "non_english_word_ratio": non_english_ratio,
        "code_mixed": code_mixed,
        "switch_count": switch_count,
        "cmi": cmi
    }


# ============================================================
# LOAD JSONL
# ============================================================

def load_samples(file_path):

    samples = []

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            text = record.get(
                "generated_text",
                ""
            ).strip()

            if text:
                samples.append(text)

    return samples


# ============================================================
# MAIN ANALYSIS
# ============================================================

all_results = {}

print("=" * 75)
print("HINGLISH LINGUISTIC / CODE-SWITCHING ANALYSIS")
print("=" * 75)

for model, file_path in MODEL_FILES.items():

    print(f"\nAnalyzing: {model}")

    if not os.path.exists(file_path):

        print(
            f"WARNING: File not found:\n{file_path}"
        )

        continue

    samples = load_samples(file_path)

    analyses = [
        analyze_text(text)
        for text in samples
    ]

    if not analyses:
        continue

    total_words = sum(
        x["word_count"]
        for x in analyses
    )

    total_english = sum(
        x["english_words"]
        for x in analyses
    )

    total_non_english = sum(
        x["non_english_words"]
        for x in analyses
    )

    english_ratio = (
        total_english / total_words
        if total_words
        else 0
    )

    non_english_ratio = (
        total_non_english / total_words
        if total_words
        else 0
    )

    code_mixed_count = sum(
        x["code_mixed"]
        for x in analyses
    )

    code_mixed_ratio = (
        code_mixed_count / len(analyses)
        if analyses
        else 0
    )

    average_switches = mean(
        x["switch_count"]
        for x in analyses
    )

    average_cmi = mean(
        x["cmi"]
        for x in analyses
    )

    all_results[model] = {
        "samples_analyzed": len(analyses),
        "total_words": total_words,
        "english_words": total_english,
        "non_english_words": total_non_english,
        "english_word_ratio": round(
            english_ratio,
            4
        ),
        "non_english_word_ratio": round(
            non_english_ratio,
            4
        ),
        "code_mixed_samples": code_mixed_count,
        "code_mixed_sample_ratio": round(
            code_mixed_ratio,
            4
        ),
        "average_language_switches": round(
            average_switches,
            4
        ),
        "average_cmi": round(
            average_cmi,
            4
        )
    }

    print(
        f"Samples: {len(analyses)}"
    )

    print(
        f"English ratio: "
        f"{english_ratio:.4f}"
    )

    print(
        f"Non-English ratio: "
        f"{non_english_ratio:.4f}"
    )

    print(
        f"Code-mixed ratio: "
        f"{code_mixed_ratio:.4f}"
    )

    print(
        f"Average switches: "
        f"{average_switches:.4f}"
    )

    print(
        f"Average CMI: "
        f"{average_cmi:.4f}"
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
        all_results,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 75)
print("LINGUISTIC ANALYSIS COMPLETED")
print("=" * 75)

print(
    f"\nSaved to:\n{OUTPUT_FILE}"
)