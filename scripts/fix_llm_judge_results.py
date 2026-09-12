import json
from pathlib import Path


INPUT_FILE = Path(
    "results/final_evaluation/llm_judge_results.jsonl"
)

OUTPUT_FILE = Path(
    "results/final_evaluation/llm_judge_results_fixed.jsonl"
)


REQUIRED_KEYS = [
    "fluency",
    "coherence",
    "relevance",
    "code_mixing_quality",
    "overall_quality"
]


def valid_score(value):
    return (
        isinstance(value, (int, float))
        and 1 <= value <= 5
    )


def main():

    print("=" * 70)
    print("FIXING LLM JUDGE RESULTS")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print()
        print("ERROR: Input file not found:")
        print(INPUT_FILE)
        return

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
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print()
    print("Original saved rows:", len(rows))

    fixed_rows = []

    old_format = 0
    new_format = 0
    invalid = 0

    # --------------------------------------------------------
    # Convert all rows to ONE standard format
    # --------------------------------------------------------

    for row in rows:

        # ====================================================
        # FORMAT 1
        # Old format:
        #
        # "fluency": 3,
        # "coherence": 4,
        # ...
        # ====================================================

        if all(
            key in row
            and valid_score(row[key])
            for key in REQUIRED_KEYS
        ):

            scores = {
                key: int(row[key])
                for key in REQUIRED_KEYS
            }

            fixed_row = {
                "sample_id": row.get(
                    "sample_id"
                ),

                "model": row.get(
                    "model"
                ),

                "input": row.get(
                    "input",
                    ""
                ),

                "generated_output": row.get(
                    "generated_output",
                    ""
                ),

                "scores": scores,

                "raw_response": row.get(
                    "raw_judge_response",
                    row.get(
                        "raw_response",
                        ""
                    )
                ),

                "judge_model": row.get(
                    "judge_model",
                    "openai/gpt-oss-20b"
                ),

                "provider": row.get(
                    "provider",
                    "groq"
                )
            }

            fixed_rows.append(
                fixed_row
            )

            old_format += 1

        # ====================================================
        # FORMAT 2
        # New format:
        #
        # "scores": {
        #     "fluency": 3,
        #     ...
        # }
        # ====================================================

        elif "scores" in row:

            scores = row.get(
                "scores"
            )

            if (
                isinstance(scores, dict)
                and all(
                    key in scores
                    and valid_score(scores[key])
                    for key in REQUIRED_KEYS
                )
            ):

                fixed_row = {
                    "sample_id": row.get(
                        "sample_id"
                    ),

                    "model": row.get(
                        "model"
                    ),

                    "input": row.get(
                        "input",
                        ""
                    ),

                    "generated_output": row.get(
                        "generated_output",
                        ""
                    ),

                    "scores": {
                        key: int(scores[key])
                        for key in REQUIRED_KEYS
                    },

                    "raw_response": row.get(
                        "raw_response",
                        ""
                    ),

                    "judge_model": row.get(
                        "judge_model",
                        "openai/gpt-oss-20b"
                    ),

                    "provider": row.get(
                        "provider",
                        "groq"
                    )
                }

                fixed_rows.append(
                    fixed_row
                )

                new_format += 1

            else:
                invalid += 1

        else:
            invalid += 1

    # --------------------------------------------------------
    # Remove duplicate model + sample combinations
    # --------------------------------------------------------

    unique_rows = {}

    for row in fixed_rows:

        key = (
            str(row["model"]),
            str(row["sample_id"])
        )

        # Keep the latest valid version
        unique_rows[key] = row

    final_rows = list(
        unique_rows.values()
    )

    # --------------------------------------------------------
    # Save fixed file
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in final_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CONVERSION COMPLETED")
    print("=" * 70)

    print(
        f"Original rows       : {len(rows)}"
    )

    print(
        f"Old-format rows     : {old_format}"
    )

    print(
        f"New-format rows     : {new_format}"
    )

    print(
        f"Invalid rows        : {invalid}"
    )

    print(
        f"After deduplication : {len(final_rows)}"
    )

    print()
    print("Fixed file:")
    print(OUTPUT_FILE)

    # --------------------------------------------------------
    # Per-model counts
    # --------------------------------------------------------

    model_counts = {}

    for row in final_rows:

        model = row["model"]

        model_counts[model] = (
            model_counts.get(model, 0) + 1
        )

    print()
    print("=" * 70)
    print("PER-MODEL COUNTS")
    print("=" * 70)

    for model, count in sorted(
        model_counts.items()
    ):

        print(
            f"{model:<30} : {count}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()