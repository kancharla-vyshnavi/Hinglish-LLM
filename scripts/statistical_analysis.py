import json
import random
import re
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "final_evaluation"

QWEN_FILE = RESULTS_DIR / "qwen25_3b_samples.jsonl"
SMOL_FILE = RESULTS_DIR / "smollm2_17b_samples.jsonl"

OUTPUT_FILE = RESULTS_DIR / "statistical_significance.json"

N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
SEED = 42


# ============================================================
# LOAD SAMPLES
# ============================================================

def load_samples(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            text = row.get("generated_text", "")

            if text:
                samples.append(text.strip())

    return samples


# ============================================================
# TOKENIZATION
# ============================================================

def words(text):
    return re.findall(r"\b[\w']+\b", text.lower())


# ============================================================
# PER-SAMPLE METRICS
# ============================================================

def distinct_1(text):
    tokens = words(text)

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def distinct_2(text):
    tokens = words(text)

    if len(tokens) < 2:
        return 0.0

    bigrams = list(zip(tokens, tokens[1:]))

    return len(set(bigrams)) / len(bigrams)


def repetition_rate(text):
    tokens = words(text)

    if not tokens:
        return 0.0

    return 1.0 - (len(set(tokens)) / len(tokens))


def language_mixing_ratio(text):
    """
    Simple descriptive heuristic.
    English words are estimated using common English function/content words.
    """

    english_words = {
        "i", "you", "he", "she", "it", "we", "they",
        "is", "am", "are", "was", "were", "be",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "should",
        "the", "a", "an", "and", "or", "but",
        "if", "then", "because", "for", "from",
        "with", "this", "that", "these", "those",
        "what", "why", "when", "where", "how",
        "my", "your", "his", "her", "our", "their",
        "in", "on", "at", "to", "of", "by",
        "not", "yes", "no", "very", "good", "bad",
        "really", "just", "like", "know", "think",
        "want", "need", "make", "go", "come",
        "get", "give", "take", "see", "look",
        "help", "please", "thanks", "thank",
        "okay", "ok"
    }

    tokens = words(text)

    if not tokens:
        return 0.0

    english_count = sum(
        1 for token in tokens
        if token in english_words
    )

    return english_count / len(tokens)


def cmi(text):
    """
    Descriptive Code-Mixing Index heuristic.

    CMI = percentage of tokens classified as English
    among tokens considered for language mixing.
    """

    tokens = words(text)

    if not tokens:
        return 0.0

    english_words = {
        "i", "you", "he", "she", "it", "we", "they",
        "is", "am", "are", "was", "were", "be",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "can", "could", "should",
        "the", "a", "an", "and", "or", "but",
        "if", "then", "because", "for", "from",
        "with", "this", "that", "these", "those",
        "what", "why", "when", "where", "how",
        "my", "your", "his", "her", "our", "their",
        "in", "on", "at", "to", "of", "by",
        "not", "yes", "no", "very", "good", "bad",
        "really", "just", "like", "know", "think",
        "want", "need", "make", "go", "come",
        "get", "give", "take", "see", "look",
        "help", "please", "thanks", "thank",
        "okay", "ok"
    }

    english_count = sum(
        1 for token in tokens
        if token in english_words
    )

    return (english_count / len(tokens)) * 100.0


def unique_sentence_ratio(text):
    """
    For an individual sample, this is 1 if all whitespace-separated
    tokens are unique and decreases with duplication.
    """

    tokens = words(text)

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def avg_word_length(text):
    tokens = words(text)

    if not tokens:
        return 0.0

    return sum(len(token) for token in tokens) / len(tokens)


def generation_length(text):
    return len(words(text))


# ============================================================
# CALCULATE METRICS FOR EVERY SAMPLE
# ============================================================

def calculate_metrics(samples):

    metrics = {
        "distinct_1": [],
        "distinct_2": [],
        "repetition_rate": [],
        "language_mixing_ratio": [],
        "cmi": [],
        "unique_sentence_ratio": [],
        "avg_word_length": [],
        "generation_length": []
    }

    for text in samples:

        metrics["distinct_1"].append(
            distinct_1(text)
        )

        metrics["distinct_2"].append(
            distinct_2(text)
        )

        metrics["repetition_rate"].append(
            repetition_rate(text)
        )

        metrics["language_mixing_ratio"].append(
            language_mixing_ratio(text)
        )

        metrics["cmi"].append(
            cmi(text)
        )

        metrics["unique_sentence_ratio"].append(
            unique_sentence_ratio(text)
        )

        metrics["avg_word_length"].append(
            avg_word_length(text)
        )

        metrics["generation_length"].append(
            generation_length(text)
        )

    return {
        key: np.array(value, dtype=float)
        for key, value in metrics.items()
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95):

    rng = np.random.default_rng(SEED)

    n = len(values)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        sample = rng.choice(
            values,
            size=n,
            replace=True
        )

        bootstrap_means[i] = np.mean(sample)

    alpha = 1.0 - confidence

    lower = np.percentile(
        bootstrap_means,
        100 * (alpha / 2)
    )

    upper = np.percentile(
        bootstrap_means,
        100 * (1 - alpha / 2)
    )

    return {
        "mean": float(np.mean(values)),
        "lower_95_ci": float(lower),
        "upper_95_ci": float(upper)
    }


# ============================================================
# BOOTSTRAP DIFFERENCE
# ============================================================

def bootstrap_difference(
    qwen_values,
    smol_values,
    n_bootstrap=1000
):

    rng = np.random.default_rng(SEED)

    n_qwen = len(qwen_values)
    n_smol = len(smol_values)

    differences = np.empty(n_bootstrap)

    for i in range(n_bootstrap):

        q_sample = rng.choice(
            qwen_values,
            size=n_qwen,
            replace=True
        )

        s_sample = rng.choice(
            smol_values,
            size=n_smol,
            replace=True
        )

        differences[i] = (
            np.mean(q_sample)
            - np.mean(s_sample)
        )

    lower = np.percentile(
        differences,
        2.5
    )

    upper = np.percentile(
        differences,
        97.5
    )

    mean_difference = (
        np.mean(qwen_values)
        - np.mean(smol_values)
    )

    return {
        "qwen_minus_smol_mean_difference":
            float(mean_difference),

        "lower_95_ci":
            float(lower),

        "upper_95_ci":
            float(upper),

        "significant_at_95_percent":
            bool(
                lower > 0 or upper < 0
            )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BOOTSTRAP STATISTICAL ANALYSIS")
    print("=" * 70)

    print("\nLoading samples...")

    qwen_samples = load_samples(QWEN_FILE)
    smol_samples = load_samples(SMOL_FILE)

    print(f"Qwen2.5-3B samples    : {len(qwen_samples)}")
    print(f"SmolLM2-1.7B samples  : {len(smol_samples)}")

    if len(qwen_samples) == 0 or len(smol_samples) == 0:
        raise RuntimeError("No samples found.")

    print("\nCalculating per-sample metrics...")

    qwen_metrics = calculate_metrics(qwen_samples)
    smol_metrics = calculate_metrics(smol_samples)

    results = {
        "comparison": {
            "model_1": "Qwen2.5-3B",
            "model_2": "SmolLM2-1.7B",
            "samples_model_1": len(qwen_samples),
            "samples_model_2": len(smol_samples),
            "bootstrap_iterations": N_BOOTSTRAP,
            "confidence_level": "95%"
        },

        "model_results": {},

        "between_model_comparison": {}
    }

    # --------------------------------------------------------
    # Individual model confidence intervals
    # --------------------------------------------------------

    for model_name, metrics in [
        ("Qwen2.5-3B", qwen_metrics),
        ("SmolLM2-1.7B", smol_metrics)
    ]:

        results["model_results"][model_name] = {}

        for metric_name, values in metrics.items():

            results["model_results"][model_name][metric_name] = (
                bootstrap_ci(
                    values,
                    N_BOOTSTRAP,
                    CONFIDENCE
                )
            )

    # --------------------------------------------------------
    # Difference between models
    # --------------------------------------------------------

    for metric_name in qwen_metrics.keys():

        results["between_model_comparison"][metric_name] = (
            bootstrap_difference(
                qwen_metrics[metric_name],
                smol_metrics[metric_name],
                N_BOOTSTRAP
            )
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
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
            indent=4
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for metric_name in qwen_metrics.keys():

        qwen_result = (
            results["model_results"]
            ["Qwen2.5-3B"]
            [metric_name]
        )

        smol_result = (
            results["model_results"]
            ["SmolLM2-1.7B"]
            [metric_name]
        )

        comparison = (
            results["between_model_comparison"]
            [metric_name]
        )

        print(f"\n{metric_name}")
        print(
            f"  Qwen 3B     : "
            f"{qwen_result['mean']:.4f} "
            f"[{qwen_result['lower_95_ci']:.4f}, "
            f"{qwen_result['upper_95_ci']:.4f}]"
        )

        print(
            f"  SmolLM2     : "
            f"{smol_result['mean']:.4f} "
            f"[{smol_result['lower_95_ci']:.4f}, "
            f"{smol_result['upper_95_ci']:.4f}]"
        )

        print(
            f"  Difference  : "
            f"{comparison['qwen_minus_smol_mean_difference']:.4f}"
        )

        print(
            f"  95% CI diff : "
            f"[{comparison['lower_95_ci']:.4f}, "
            f"{comparison['upper_95_ci']:.4f}]"
        )

        print(
            f"  Significant : "
            f"{comparison['significant_at_95_percent']}"
        )

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()