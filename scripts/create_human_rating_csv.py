import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "results"
    / "final_evaluation"
    / "human_evaluation_sheet.jsonl"
)

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "final_evaluation"
    / "human_rating_sheet.csv"
)


def main():

    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    fieldnames = [
        "sample_id",
        "model",
        "input",
        "generated_output",
        "fluency",
        "coherence",
        "relevance",
        "code_mixing_quality",
        "overall_quality",
        "comments"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "model": row["model"],
                "input": row["input"],
                "generated_output": row["generated_output"],
                "fluency": "",
                "coherence": "",
                "relevance": "",
                "code_mixing_quality": "",
                "overall_quality": "",
                "comments": ""
            })

    print("=" * 70)
    print("HUMAN RATING SHEET CREATED")
    print("=" * 70)

    print(f"Total rows: {len(rows)}")

    print("\nRating scale:")
    print("1 = Very Poor")
    print("2 = Poor")
    print("3 = Average")
    print("4 = Good")
    print("5 = Excellent")

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()