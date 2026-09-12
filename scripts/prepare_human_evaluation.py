import json
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TEST_FILE = BASE_DIR / "data" / "processed" / "test.jsonl"
SAMPLES_DIR = BASE_DIR / "results" / "final_evaluation" / "samples"

OUTPUT_FILE = (
    BASE_DIR
    / "results"
    / "final_evaluation"
    / "human_evaluation_sheet.jsonl"
)

N_SAMPLES = 100
SEED = 42

MODEL_FILES = {
    "Qwen2.5-1.5B":
        SAMPLES_DIR / "Qwen25_15B_samples.jsonl",

    "Qwen2.5-3B":
        SAMPLES_DIR / "Qwen25_3B_samples.jsonl",

    "SmolLM2-1.7B":
        SAMPLES_DIR / "SmolLM2_17B_samples.jsonl",

    "Bharat-Tiny-LLM-v3":
        SAMPLES_DIR / "Bharat_Tiny_LLM_v3_samples.jsonl",
}


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def main():

    print("=" * 70)
    print("PREPARING HUMAN EVALUATION DATASET")
    print("=" * 70)

    test_rows = load_jsonl(TEST_FILE)

    model_outputs = {}

    for model, path in MODEL_FILES.items():

        rows = load_jsonl(path)

        model_outputs[model] = [
            row["generated_text"].strip()
            for row in rows
        ]

        print(f"{model}: {len(rows)} outputs")

    # Make sure all models have enough aligned samples
    lengths = [len(v) for v in model_outputs.values()]

    if len(test_rows) < N_SAMPLES:
        raise ValueError("Not enough test samples.")

    if min(lengths) < N_SAMPLES:
        raise ValueError("Not enough generated samples.")

    # Same 100 examples for every model
    random.seed(SEED)

    selected_indices = sorted(
        random.sample(
            range(N_SAMPLES),
            N_SAMPLES
        )
    )

    # Create output
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    count = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for idx in selected_indices:

            input_text = test_rows[idx]["text"]

            for model in MODEL_FILES:

                output_text = model_outputs[model][idx]

                row = {
                    "sample_id": idx,
                    "model": model,
                    "input": input_text,
                    "generated_output": output_text,

                    "fluency": "",
                    "coherence": "",
                    "relevance": "",
                    "code_mixing_quality": "",
                    "overall_quality": "",

                    "comments": ""
                }

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    ) + "\n"
                )

                count += 1

    print()
    print(f"Common samples used : {N_SAMPLES}")
    print(f"Models               : {len(MODEL_FILES)}")
    print(f"Total evaluation rows: {count}")

    print()
    print("Rating scale:")
    print("1 = Very poor")
    print("2 = Poor")
    print("3 = Average")
    print("4 = Good")
    print("5 = Excellent")

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()