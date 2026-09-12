import json
from pathlib import Path
from collections import defaultdict


# ============================================================
# FILES
# ============================================================

INPUT_FILE = Path(
    "results/final_evaluation/llm_judge_results_fixed.jsonl"
)

OUTPUT_FILE = Path(
    "results/final_evaluation/llm_judge_summary.json"
)


# ============================================================
# REQUIRED SCORE FIELDS
# ============================================================

SCORE_KEYS = [
    "fluency",
    "coherence",
    "relevance",
    "code_mixing_quality",
    "overall_quality"
]


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"File not found:\n{INPUT_FILE}"
        )

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                print(
                    "WARNING: Skipping invalid JSON line."
                )

    return rows


# ============================================================
# VALIDATE ROW
# ============================================================

def valid_row(row):

    if "model" not in row:
        return False

    if "scores" not in row:
        return False

    scores = row["scores"]

    if not isinstance(scores, dict):
        return False

    for key in SCORE_KEYS:

        if key not in scores:
            return False

        value = scores[key]

        if not isinstance(
            value,
            (int, float)
        ):
            return False

        if value < 1 or value > 5:
            return False

    return True


# ============================================================
# CALCULATE SUMMARY
# ============================================================

def calculate_summary(rows):

    model_scores = defaultdict(
        lambda: {
            key: []
            for key in SCORE_KEYS
        }
    )

    valid_count = 0
    invalid_count = 0

    for row in rows:

        if not valid_row(row):

            invalid_count += 1
            continue

        model = row["model"]
        scores = row["scores"]

        valid_count += 1

        for key in SCORE_KEYS:

            model_scores[model][key].append(
                scores[key]
            )

    summary = {}

    for model, scores in model_scores.items():

        n = len(
            scores["overall_quality"]
        )

        summary[model] = {
            "samples_evaluated": n,

            "fluency": round(
                sum(scores["fluency"]) / n,
                4
            ),

            "coherence": round(
                sum(scores["coherence"]) / n,
                4
            ),

            "relevance": round(
                sum(scores["relevance"]) / n,
                4
            ),

            "code_mixing_quality": round(
                sum(
                    scores["code_mixing_quality"]
                ) / n,
                4
            ),

            "overall_quality": round(
                sum(
                    scores["overall_quality"]
                ) / n,
                4
            )
        }

    return summary, valid_count, invalid_count


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    summary,
    total_rows,
    valid_count,
    invalid_count
):

    output = {
        "judge_model": "openai/gpt-oss-20b",
        "provider": "groq",

        "total_rows_in_file": total_rows,

        "valid_evaluations": valid_count,

        "invalid_evaluations": invalid_count,

        "evaluation_scale": "1-5",

        "metrics": summary
    }

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
            output,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary,
    total_rows,
    valid_count,
    invalid_count
):

    print()
    print("=" * 75)
    print("LLM-AS-A-JUDGE SUMMARY")
    print("=" * 75)

    print()
    print(
        f"Judge model       : openai/gpt-oss-20b"
    )

    print(
        f"Provider           : Groq"
    )

    print(
        f"Total rows         : {total_rows}"
    )

    print(
        f"Valid evaluations  : {valid_count}"
    )

    print(
        f"Invalid evaluations: {invalid_count}"
    )

    print()
    print("-" * 75)

    for model in sorted(summary.keys()):

        metrics = summary[model]

        print()
        print(
            model
        )

        print(
            f"  Samples evaluated   : "
            f"{metrics['samples_evaluated']}"
        )

        print(
            f"  Fluency             : "
            f"{metrics['fluency']:.4f}/5"
        )

        print(
            f"  Coherence           : "
            f"{metrics['coherence']:.4f}/5"
        )

        print(
            f"  Relevance           : "
            f"{metrics['relevance']:.4f}/5"
        )

        print(
            f"  Code-mixing quality : "
            f"{metrics['code_mixing_quality']:.4f}/5"
        )

        print(
            f"  Overall quality     : "
            f"{metrics['overall_quality']:.4f}/5"
        )

    print()
    print("=" * 75)

    print()
    print(
        f"Summary saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 75)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("GENERATING LLM JUDGE SUMMARY")
    print("=" * 75)

    rows = load_results()

    print()
    print(
        f"Loaded rows: {len(rows)}"
    )

    summary, valid_count, invalid_count = (
        calculate_summary(rows)
    )

    save_summary(
        summary,
        len(rows),
        valid_count,
        invalid_count
    )

    print_summary(
        summary,
        len(rows),
        valid_count,
        invalid_count
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()