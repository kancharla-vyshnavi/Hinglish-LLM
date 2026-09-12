import os
import json
import time
import re
from pathlib import Path
from collections import defaultdict

from huggingface_hub import InferenceClient


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "results/final_evaluation/human_evaluation_sheet.jsonl"
)

OUTPUT_FILE = Path(
    "results/final_evaluation/llm_judge_results.jsonl"
)

SUMMARY_FILE = Path(
    "results/final_evaluation/llm_judge_summary.json"
)

PROVIDER = "groq"
MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5
DELAY_BETWEEN_REQUESTS = 0.5

MAX_TOKENS = 1000
TEMPERATURE = 0.0


# ============================================================
# JUDGE RUBRIC
# ============================================================

RUBRIC = """
You are evaluating a Hindi-English code-mixed text generation.

You will receive:
- INPUT: the original user text
- GENERATED OUTPUT: the model-generated continuation

Evaluate ONLY the generated output with respect to the input.

Give exactly five scores from 1 to 5.

1. Fluency
How natural, grammatical, readable, and smooth is the generated text?

2. Coherence
Is the generated text internally logical, understandable, and reasonably complete?

3. Relevance
Does the generated text remain relevant to the given input?

4. Code-mixing quality
Does the Hindi-English mixing feel natural and meaningful?
Do NOT reward random or unnecessary switching between Hindi and English.

5. Overall quality
Give an overall quality score considering all the above factors.

Scoring scale:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return ONLY valid JSON.

Required JSON format:

{
  "fluency": 1,
  "coherence": 1,
  "relevance": 1,
  "code_mixing_quality": 1,
  "overall_quality": 1
}
"""


# ============================================================
# LOAD INPUT DATA
# ============================================================

def load_input_rows():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                rows.append(row)

            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Could not parse line {line_number}: {e}"
                )

    return rows


# ============================================================
# LOAD PREVIOUS RESULTS
# ============================================================

def load_previous_results():

    if not OUTPUT_FILE.exists():
        return []

    results = []

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
                results.append(row)

            except json.JSONDecodeError:
                print(
                    f"WARNING: Invalid saved result at line "
                    f"{line_number}. Skipping."
                )

    return results


# ============================================================
# CREATE RESUME KEY
# ============================================================

def make_key(row):
    """
    Unique key for a model + sample.
    """

    model = str(row.get("model", ""))
    sample_id = str(row.get("sample_id", ""))

    return f"{model}|||{sample_id}"


# ============================================================
# VALIDATE SCORES
# ============================================================

def validate_scores(scores):

    required_keys = [
        "fluency",
        "coherence",
        "relevance",
        "code_mixing_quality",
        "overall_quality"
    ]

    if not isinstance(scores, dict):
        return False

    for key in required_keys:

        if key not in scores:
            return False

        value = scores[key]

        if not isinstance(value, (int, float)):
            return False

        if value < 1 or value > 5:
            return False

        # Scores should normally be integers
        if float(value) != int(value):
            return False

    return True


# ============================================================
# EXTRACT JSON FROM JUDGE RESPONSE
# ============================================================

def extract_json(text):

    if not text:
        return None

    text = text.strip()

    # --------------------------------------------------------
    # Method 1: Entire response is JSON
    # --------------------------------------------------------

    try:

        obj = json.loads(text)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    # --------------------------------------------------------
    # Method 2: Remove markdown code fences
    # --------------------------------------------------------

    cleaned = re.sub(
        r"```(?:json)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    cleaned = cleaned.replace("```", "").strip()

    try:

        obj = json.loads(cleaned)

        if isinstance(obj, dict):
            return obj

    except Exception:
        pass

    # --------------------------------------------------------
    # Method 3: Find JSON object inside response
    # --------------------------------------------------------

    matches = re.findall(
        r"\{.*?\}",
        text,
        flags=re.DOTALL
    )

    for match in matches:

        try:

            obj = json.loads(match)

            if isinstance(obj, dict):
                return obj

        except Exception:
            continue

    return None


# ============================================================
# BUILD JUDGE PROMPT
# ============================================================

def build_prompt(input_text, generated_output):

    prompt = f"""
{RUBRIC}

INPUT:
{input_text}

GENERATED OUTPUT:
{generated_output}

Return ONLY the JSON object.
"""

    return prompt


# ============================================================
# CALL LLM JUDGE
# ============================================================

def call_judge(client, input_text, generated_output):

    prompt = build_prompt(
        input_text,
        generated_output
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict evaluator. "
                            "Return only valid JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            # ------------------------------------------------
            # Extract response content safely
            # ------------------------------------------------

            content = ""

            if response is not None:

                if hasattr(response, "choices"):

                    if response.choices:

                        message = response.choices[0].message

                        if message is not None:

                            content = getattr(
                                message,
                                "content",
                                ""
                            )

            if content is None:
                content = ""

            content = str(content).strip()

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            scores = extract_json(content)

            if scores is not None:

                if validate_scores(scores):

                    cleaned_scores = {
                        "fluency": int(scores["fluency"]),
                        "coherence": int(scores["coherence"]),
                        "relevance": int(scores["relevance"]),
                        "code_mixing_quality": int(
                            scores["code_mixing_quality"]
                        ),
                        "overall_quality": int(
                            scores["overall_quality"]
                        )
                    }

                    return {
                        "scores": cleaned_scores,
                        "raw_response": content
                    }

            last_error = (
                "Could not extract valid scores from response"
            )

        except Exception as e:

            last_error = str(e)

        if attempt < MAX_RETRIES:

            print(
                f"      Retry {attempt}/{MAX_RETRIES} "
                f"after {RETRY_WAIT_SECONDS}s..."
            )

            time.sleep(RETRY_WAIT_SECONDS)

    return {
        "error": last_error
    }


# ============================================================
# SAVE SINGLE RESULT
# ============================================================

def append_result(result):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                result,
                ensure_ascii=False
            )
            + "\n"
        )


# ============================================================
# CALCULATE SUMMARY
# ============================================================

def calculate_summary(results):

    required_keys = [
        "fluency",
        "coherence",
        "relevance",
        "code_mixing_quality",
        "overall_quality"
    ]

    model_scores = defaultdict(
        lambda: {
            key: []
            for key in required_keys
        }
    )

    # --------------------------------------------------------
    # Read valid results only
    # --------------------------------------------------------

    for row in results:

        # Skip malformed/incomplete rows
        if "scores" not in row:
            continue

        scores = row["scores"]

        if not validate_scores(scores):
            continue

        model = row.get(
            "model",
            "Unknown"
        )

        for key in required_keys:

            model_scores[model][key].append(
                scores[key]
            )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    summary = {}

    for model, score_dict in model_scores.items():

        count = len(
            score_dict["overall_quality"]
        )

        if count == 0:
            continue

        summary[model] = {
            "samples_evaluated": count,

            "fluency": round(
                sum(score_dict["fluency"]) / count,
                4
            ),

            "coherence": round(
                sum(score_dict["coherence"]) / count,
                4
            ),

            "relevance": round(
                sum(score_dict["relevance"]) / count,
                4
            ),

            "code_mixing_quality": round(
                sum(
                    score_dict["code_mixing_quality"]
                ) / count,
                4
            ),

            "overall_quality": round(
                sum(
                    score_dict["overall_quality"]
                ) / count,
                4
            )
        }

    return summary


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(summary):

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(summary):

    print()
    print("=" * 70)
    print("LLM-AS-A-JUDGE SUMMARY")
    print("=" * 70)

    if not summary:

        print("No valid judge results found.")
        return

    for model, metrics in summary.items():

        print()
        print("-" * 70)
        print(model)
        print("-" * 70)

        print(
            f"Samples evaluated       : "
            f"{metrics['samples_evaluated']}"
        )

        print(
            f"Fluency                 : "
            f"{metrics['fluency']:.4f}/5"
        )

        print(
            f"Coherence               : "
            f"{metrics['coherence']:.4f}/5"
        )

        print(
            f"Relevance               : "
            f"{metrics['relevance']:.4f}/5"
        )

        print(
            f"Code-mixing quality     : "
            f"{metrics['code_mixing_quality']:.4f}/5"
        )

        print(
            f"Overall quality         : "
            f"{metrics['overall_quality']:.4f}/5"
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    print()
    print(f"Provider : {PROVIDER}")
    print(f"Model    : {MODEL}")

    # --------------------------------------------------------
    # Load input
    # --------------------------------------------------------

    input_rows = load_input_rows()

    print(
        f"Total input rows : {len(input_rows)}"
    )

    if len(input_rows) == 0:

        print("ERROR: No input rows found.")
        return

    # --------------------------------------------------------
    # Load existing results
    # --------------------------------------------------------

    previous_results = load_previous_results()

    print(
        f"Previously saved : {len(previous_results)}"
    )

    # --------------------------------------------------------
    # Build resume set
    # --------------------------------------------------------

    completed_keys = set()

    for row in previous_results:

        if "scores" in row:

            scores = row.get("scores")

            if validate_scores(scores):

                completed_keys.add(
                    make_key(row)
                )

    print(
        f"Valid completed  : {len(completed_keys)}"
    )

    # --------------------------------------------------------
    # Connect to Groq through HF
    # --------------------------------------------------------

    print()
    print("Connecting to Hugging Face Inference Provider...")

    try:

        client = InferenceClient(
            provider=PROVIDER
        )

    except Exception as e:

        print()
        print("ERROR: Could not create InferenceClient.")
        print(e)
        return

    print("Connection initialized.")
    print()

    # --------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------

    successful_this_run = 0
    failed_this_run = 0

    total = len(input_rows)

    for position, row in enumerate(
        input_rows,
        start=1
    ):

        model = row.get(
            "model",
            "Unknown"
        )

        sample_id = row.get(
            "sample_id",
            position
        )

        key = make_key(row)

        # ----------------------------------------------------
        # Resume support
        # ----------------------------------------------------

        if key in completed_keys:

            print(
                f"[{position}/{total}] "
                f"{model} | sample {sample_id} "
                f"-> already completed"
            )

            continue

        input_text = row.get(
            "input",
            ""
        )

        generated_output = row.get(
            "generated_output",
            ""
        )

        if not input_text or not generated_output:

            print(
                f"[{position}/{total}] "
                f"{model} | sample {sample_id} "
                f"-> SKIPPED: missing input/output"
            )

            failed_this_run += 1
            continue

        print(
            f"[{position}/{total}] "
            f"{model} | sample {sample_id}"
        )

        print(
            "      Sending request..."
        )

        judge_result = call_judge(
            client,
            input_text,
            generated_output
        )

        # ----------------------------------------------------
        # Successful evaluation
        # ----------------------------------------------------

        if "scores" in judge_result:

            result = {
                "sample_id": sample_id,
                "model": model,

                "input": input_text,
                "generated_output": generated_output,

                "scores": judge_result["scores"],

                "raw_response": judge_result.get(
                    "raw_response",
                    ""
                ),

                "judge_model": MODEL,
                "provider": PROVIDER
            }

            append_result(result)

            completed_keys.add(key)

            successful_this_run += 1

            scores = result["scores"]

            avg = sum(
                scores.values()
            ) / len(scores)

            print(
                "      SUCCESS"
            )

            print(
                f"      Fluency       : "
                f"{scores['fluency']}/5"
            )

            print(
                f"      Coherence     : "
                f"{scores['coherence']}/5"
            )

            print(
                f"      Relevance     : "
                f"{scores['relevance']}/5"
            )

            print(
                f"      Code-mixing   : "
                f"{scores['code_mixing_quality']}/5"
            )

            print(
                f"      Overall       : "
                f"{scores['overall_quality']}/5"
            )

            print(
                f"      Average       : "
                f"{avg:.2f}/5"
            )

        # ----------------------------------------------------
        # Failed evaluation
        # ----------------------------------------------------

        else:

            failed_this_run += 1

            print(
                "      FAILED"
            )

            print(
                f"      Reason: "
                f"{judge_result.get('error', 'Unknown error')}"
            )

        # ----------------------------------------------------
        # Small delay between API calls
        # ----------------------------------------------------

        time.sleep(
            DELAY_BETWEEN_REQUESTS
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    all_results = load_previous_results()

    valid_results = []

    for row in all_results:

        if "scores" not in row:
            continue

        if validate_scores(
            row["scores"]
        ):

            valid_results.append(row)

    print()
    print("=" * 70)
    print("LLM JUDGE EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"Total input rows : {len(input_rows)}"
    )

    print(
        f"Successful       : {successful_this_run}"
    )

    print(
        f"Failed this run  : {failed_this_run}"
    )

    print(
        f"Total saved      : {len(valid_results)}"
    )

    # --------------------------------------------------------
    # Calculate summary safely
    # --------------------------------------------------------

    summary = calculate_summary(
        all_results
    )

    save_summary(summary)

    print_summary(summary)

    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Results : {OUTPUT_FILE}"
    )

    print(
        f"Summary : {SUMMARY_FILE}"
    )

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # Completion status
    # --------------------------------------------------------

    if len(valid_results) >= len(input_rows):

        print(
            "STATUS: ALL INPUT ROWS HAVE VALID JUDGE RESULTS"
        )

    else:

        remaining = (
            len(input_rows)
            - len(valid_results)
        )

        print(
            f"STATUS: {remaining} "
            f"VALID RESULTS STILL MISSING"
        )

        print(
            "Run the same command again to resume."
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()