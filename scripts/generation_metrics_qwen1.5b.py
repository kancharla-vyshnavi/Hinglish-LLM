import os
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "results/qwen2.5-1.5b-qlora"

TEST_FILE = "data/processed/test.jsonl"

OUTPUT_DIR = "results/qwen2.5-1.5b-generation"

NUM_GENERATION_SAMPLES = 500

MAX_INPUT_LENGTH = 64

MAX_NEW_TOKENS = 32


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("      QWEN 1.5B HINGLISH GENERATION EVALUATION")
print("=" * 70)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

test_data = []

with open(TEST_FILE, "r", encoding="utf-8") as f:

    for line in f:
        item = json.loads(line)
        test_data.append(item)


print(f"Total test examples : {len(test_data)}")

if len(test_data) != 5000:

    print("WARNING: Expected 5,000 test examples.")


# Use first 500 samples for generation
generation_data = test_data[:NUM_GENERATION_SAMPLES]

print(f"Generation samples  : {len(generation_data)}")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("\nConfiguring 4-bit NF4 quantization...")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)


# ============================================================
# LOAD BASE MODEL
# ============================================================

print("\nLoading Qwen 1.5B base model...")

base_model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    trust_remote_code=True
)


# ============================================================
# LOAD TRAINED QLORA ADAPTER
# ============================================================

print("\nLoading trained QLoRA adapter...")

model = PeftModel.from_pretrained(

    base_model,

    ADAPTER_PATH
)

model.eval()

print("QLoRA model loaded successfully.")


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def tokenize_words(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


def distinct_n(text, n):

    tokens = tokenize_words(text)

    if len(tokens) < n:
        return 0.0

    ngrams = []

    for i in range(len(tokens) - n + 1):

        ngrams.append(
            tuple(tokens[i:i+n])
        )

    if len(ngrams) == 0:
        return 0.0

    return len(set(ngrams)) / len(ngrams)


def repetition_rate(text):

    tokens = tokenize_words(text)

    if len(tokens) <= 1:
        return 0.0

    unique_tokens = len(set(tokens))

    return 1.0 - (
        unique_tokens / len(tokens)
    )


def english_ratio(text):

    all_words = re.findall(
        r"\b\w+\b",
        text
    )

    if len(all_words) == 0:
        return 0.0

    english_words = re.findall(
        r"[a-zA-Z]+",
        text
    )

    return len(english_words) / len(all_words)


def character_diversity(text):

    chars = [
        c.lower()
        for c in text
        if c.isalpha()
    ]

    if len(chars) == 0:
        return 0.0

    return len(set(chars)) / len(chars)


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST OUTPUTS")
print("=" * 70)

results = []

distinct1_scores = []
distinct2_scores = []
repetition_scores = []
english_scores = []
character_diversity_scores = []


for idx, item in enumerate(
    tqdm(
        generation_data,
        desc="Generating"
    )
):

    input_text = item["text"]

    # --------------------------------------------------------
    # TOKENIZE INPUT
    # --------------------------------------------------------

    inputs = tokenizer(

        input_text,

        return_tensors="pt",

        truncation=True,

        max_length=MAX_INPUT_LENGTH
    )


    # Move tensors to model device

    inputs = {

        key: value.to(model.device)

        for key, value in inputs.items()

    }


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    with torch.no_grad():

        output_ids = model.generate(

            **inputs,

            max_new_tokens=MAX_NEW_TOKENS,

            do_sample=False,

            num_beams=1,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id

        )


    # --------------------------------------------------------
    # REMOVE INPUT TOKENS
    # --------------------------------------------------------

    generated_ids = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]


    generated_text = tokenizer.decode(

        generated_ids,

        skip_special_tokens=True

    ).strip()


    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    d1 = distinct_n(
        generated_text,
        1
    )

    d2 = distinct_n(
        generated_text,
        2
    )

    rep = repetition_rate(
        generated_text
    )

    eng = english_ratio(
        generated_text
    )

    char_div = character_diversity(
        generated_text
    )


    # --------------------------------------------------------
    # STORE METRICS
    # --------------------------------------------------------

    distinct1_scores.append(d1)

    distinct2_scores.append(d2)

    repetition_scores.append(rep)

    english_scores.append(eng)

    character_diversity_scores.append(char_div)


    # --------------------------------------------------------
    # STORE SAMPLE
    # --------------------------------------------------------

    results.append({

        "input": input_text,

        "generated": generated_text,

        "distinct_1": d1,

        "distinct_2": d2,

        "repetition_rate": rep,

        "english_ratio": eng,

        "character_diversity": char_div

    })


# ============================================================
# AVERAGE FUNCTION
# ============================================================

def average(values):

    if len(values) == 0:

        return 0.0

    return sum(values) / len(values)


# ============================================================
# FINAL METRICS
# ============================================================

metrics = {

    "model":
        MODEL_NAME,

    "adapter":
        ADAPTER_PATH,

    "test_set_size":
        len(test_data),

    "generation_samples":
        NUM_GENERATION_SAMPLES,

    "max_input_length":
        MAX_INPUT_LENGTH,

    "max_new_tokens":
        MAX_NEW_TOKENS,

    "distinct_1":
        average(distinct1_scores),

    "distinct_2":
        average(distinct2_scores),

    "repetition_rate":
        average(repetition_scores),

    "english_ratio":
        average(english_scores),

    "character_diversity":
        average(character_diversity_scores)

}


# ============================================================
# SAVE METRICS
# ============================================================

metrics_file = os.path.join(

    OUTPUT_DIR,

    "generation_metrics.json"

)


with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(

        metrics,

        f,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# SAVE GENERATED SAMPLES
# ============================================================

samples_file = os.path.join(

    OUTPUT_DIR,

    "generated_samples.jsonl"

)


with open(

    samples_file,

    "w",

    encoding="utf-8"

) as f:

    for result in results:

        f.write(

            json.dumps(

                result,

                ensure_ascii=False

            ) + "\n"

        )


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("       GENERATION METRICS COMPLETED")
print("=" * 70)

print(f"\nModel              : {MODEL_NAME}")

print(f"Test set           : {len(test_data)}")

print(
    f"Generation samples : {NUM_GENERATION_SAMPLES}"
)

print("\nGeneration Metrics")
print("-" * 50)

print(
    f"Distinct-1         : {metrics['distinct_1']:.4f}"
)

print(
    f"Distinct-2         : {metrics['distinct_2']:.4f}"
)

print(
    f"Repetition Rate    : {metrics['repetition_rate']:.4f}"
)

print(
    f"English Ratio      : {metrics['english_ratio']:.4f}"
)

print(
    f"Character Diversity: {metrics['character_diversity']:.4f}"
)


print("\nFiles saved:")

print(
    f"Metrics : {metrics_file}"
)

print(
    f"Samples : {samples_file}"
)

print("\n" + "=" * 70)
print("              EVALUATION READY")
print("=" * 70)