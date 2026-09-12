import json
import re
from huggingface_hub import InferenceClient


def extract_json(text):
    """
    Find the first JSON object containing the required
    evaluation fields.
    """

    required_keys = [
        "fluency",
        "coherence",
        "relevance",
        "code_mixing_quality",
        "overall_quality",
    ]

    # Try direct JSON first
    try:
        data = json.loads(text)

        if all(key in data for key in required_keys):
            return data

    except Exception:
        pass

    # Search for JSON object inside the response
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)

    for match in matches:
        try:
            data = json.loads(match)

            if all(key in data for key in required_keys):
                return data

        except Exception:
            continue

    return None


def validate_scores(data):
    """
    Check that all scores are integers from 1 to 5.
    """

    required_keys = [
        "fluency",
        "coherence",
        "relevance",
        "code_mixing_quality",
        "overall_quality",
    ]

    if data is None:
        return False

    for key in required_keys:

        if key not in data:
            return False

        try:
            value = float(data[key])
        except Exception:
            return False

        if value < 1 or value > 5:
            return False

    return True


def main():

    print("=" * 70)
    print("LLM-AS-A-JUDGE TEST")
    print("=" * 70)

    # ------------------------------------------------------------
    # Hugging Face → Groq
    # ------------------------------------------------------------

    client = InferenceClient(
        provider="groq"
    )

    model = "openai/gpt-oss-20b"

    print("\nProvider : Groq")
    print("Model    :", model)

    # ------------------------------------------------------------
    # Example input/output
    # ------------------------------------------------------------

    input_text = (
        "Mujhe weekend par kya karna chahiye?"
    )

    generated_output = (
        "Weekend par aap friends ke saath movie dekh sakte ho "
        "aur thoda relax bhi kar sakte ho."
    )

    # ------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------

    prompt = f"""
You are evaluating a Hindi-English code-mixed (Hinglish)
text generation system.

Evaluate the generated response.

INPUT:
{input_text}

GENERATED RESPONSE:
{generated_output}

Give exactly five scores from 1 to 5.

FLUENCY:
How natural, readable, and grammatically acceptable is
the generated response?

COHERENCE:
Is the response logically connected and understandable?

RELEVANCE:
Does the response directly address the input?

CODE_MIXING_QUALITY:
Is the Hindi-English mixing natural and appropriate?

OVERALL_QUALITY:
Overall quality of the generated response.

SCORING:
1 = Very Poor
2 = Poor
3 = Average
4 = Good
5 = Excellent

Return a JSON object containing ONLY these five fields:

{{
  "fluency": 4,
  "coherence": 4,
  "relevance": 4,
  "code_mixing_quality": 4,
  "overall_quality": 4
}}

Do not add any other fields.
"""

    print("\nSending request...")
    print("-" * 70)

    # ------------------------------------------------------------
    # API CALL
    # ------------------------------------------------------------

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator. "
                        "Evaluate the response and provide "
                        "five scores."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=1000,
            temperature=0.0,
        )

    except Exception as e:

        print("\nAPI ERROR")
        print("-" * 70)
        print(type(e).__name__)
        print(str(e))
        print("-" * 70)

        return

    # ------------------------------------------------------------
    # Get response content
    # ------------------------------------------------------------

    if not response.choices:

        print("\nERROR: No choices returned.")
        return

    message = response.choices[0].message

    content = getattr(message, "content", None)

    reasoning = getattr(message, "reasoning", None)

    print("\nCONTENT")
    print("-" * 70)

    if content:
        print(content)
    else:
        print("[No normal content returned]")

    print("-" * 70)

    # ------------------------------------------------------------
    # Extract JSON
    # ------------------------------------------------------------

    combined_text = ""

    if content:
        combined_text += content + "\n"

    if reasoning:
        combined_text += reasoning + "\n"

    result = extract_json(combined_text)

    # ------------------------------------------------------------
    # If JSON was found
    # ------------------------------------------------------------

    if result is not None and validate_scores(result):

        print("\nJUDGE SCORES")
        print("-" * 70)

        print(
            f"Fluency             : "
            f"{result['fluency']}/5"
        )

        print(
            f"Coherence           : "
            f"{result['coherence']}/5"
        )

        print(
            f"Relevance           : "
            f"{result['relevance']}/5"
        )

        print(
            f"Code-mixing quality : "
            f"{result['code_mixing_quality']}/5"
        )

        print(
            f"Overall quality     : "
            f"{result['overall_quality']}/5"
        )

        average = (
            float(result["fluency"])
            + float(result["coherence"])
            + float(result["relevance"])
            + float(result["code_mixing_quality"])
            + float(result["overall_quality"])
        ) / 5

        print("-" * 70)

        print(
            f"Average score       : "
            f"{average:.2f}/5"
        )

        print("-" * 70)

        print("\nVALID JSON:")
        print(json.dumps(result, indent=2))

        print("\n" + "=" * 70)
        print("LLM-AS-A-JUDGE TEST PASSED")
        print("=" * 70)

        return

    # ------------------------------------------------------------
    # Failure
    # ------------------------------------------------------------

    print("\nJSON COULD NOT BE EXTRACTED")
    print("-" * 70)

    if reasoning:
        print("Reasoning was returned by the model.")

    if not content:
        print("No normal content was returned.")

    print("\nFull response object:")
    print(response)

    print("-" * 70)


if __name__ == "__main__":
    main()