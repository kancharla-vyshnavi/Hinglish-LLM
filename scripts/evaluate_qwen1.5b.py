# ============================================================
# Qwen 2.5 1.5B QLoRA - Test Evaluation
# Hindi-English Code-Mixed Text
# ============================================================

import json
import math
import re
import os

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTER_PATH = "results/qwen2.5-1.5b-qlora"

TEST_FILE = "data/processed/test.jsonl"

MAX_LENGTH = 64

MAX_EVAL_SAMPLES = 5000

GENERATION_MAX_NEW_TOKENS = 64

OUTPUT_FILE = "results/qwen2.5-1.5b-evaluation.json"


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available.")

print("=" * 70)
print("QWEN 2.5 1.5B QLORA - TEST EVALUATION")
print("=" * 70)

print(f"GPU : {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_jsonl(path):

    texts = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            text = item.get("text", "").strip()

            if text:
                texts.append(text)

    return texts


print("\nLoading test dataset...")

test_texts = load_jsonl(TEST_FILE)

print(f"Test examples : {len(test_texts):,}")


if len(test_texts) != 5000:
    raise ValueError(
        f"Expected 5,000 test examples, "
        f"found {len(test_texts):,}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# LOAD BASE MODEL IN 4-BIT
# ============================================================

print("\nLoading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
)

base_model.config.use_cache = True


# ============================================================
# LOAD TRAINED QLORA ADAPTER
# ============================================================

print("\nLoading trained QLoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("✅ QLoRA adapter loaded successfully.")


# ============================================================
# PERPLEXITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING PERPLEXITY")
print("=" * 70)

total_negative_log_likelihood = 0.0
total_tokens = 0

with torch.no_grad():

    for i, text in enumerate(test_texts):

        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = encoded["input_ids"].to("cuda")

        attention_mask = encoded["attention_mask"].to("cuda")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        loss = outputs.loss

        num_tokens = (
            attention_mask.sum().item() - 1
        )

        if num_tokens > 0:

            total_negative_log_likelihood += (
                loss.item() * num_tokens
            )

            total_tokens += num_tokens

        if (i + 1) % 500 == 0:

            print(
                f"Processed {i + 1:,}/{len(test_texts):,}"
            )


if total_tokens == 0:

    raise RuntimeError(
        "No valid tokens found for perplexity calculation."
    )


average_nll = (
    total_negative_log_likelihood
    / total_tokens
)

perplexity = math.exp(average_nll)


print(f"\nPerplexity : {perplexity:.4f}")


# ============================================================
# GENERATION METRICS
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING GENERATION METRICS")
print("=" * 70)


def tokenize_words(text):

    return re.findall(
        r"\b[\w']+\b",
        text.lower(),
    )


def distinct_n(tokens, n):

    if len(tokens) < n:

        return 0.0

    ngrams = [
        tuple(tokens[i:i+n])
        for i in range(len(tokens) - n + 1)
    ]

    if len(ngrams) == 0:

        return 0.0

    return len(set(ngrams)) / len(ngrams)


def repetition_rate(tokens):

    if len(tokens) == 0:

        return 0.0

    return 1.0 - (
        len(set(tokens)) / len(tokens)
    )


def language_ratio(text):

    tokens = tokenize_words(text)

    if len(tokens) == 0:

        return 0.0, 0.0

    english = 0
    roman_hindi = 0

    for token in tokens:

        if re.fullmatch(
            r"[a-zA-Z]+",
            token
        ):

            english += 1

        elif re.search(
            r"[a-zA-Z]",
            token
        ):

            roman_hindi += 1

    total = len(tokens)

    return (
        english / total,
        roman_hindi / total,
    )


# ============================================================
# GENERATE SAMPLES
# ============================================================

generated_texts = []

# Evaluate a representative subset for generation.
GENERATION_SAMPLES = min(
    MAX_EVAL_SAMPLES,
    1000
)

print(
    f"\nGenerating {GENERATION_SAMPLES:,} test outputs..."
)


for i in range(GENERATION_SAMPLES):

    text = test_texts[i]

    # Use first part as prompt
    words = text.split()

    if len(words) > 12:

        prompt = " ".join(words[:12])

    else:

        prompt = text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    inputs = {
        key: value.to("cuda")
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=GENERATION_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    generated_texts.append(generated)

    if (i + 1) % 100 == 0:

        print(
            f"Generated {i + 1:,}/{GENERATION_SAMPLES:,}"
        )


# ============================================================
# CALCULATE GENERATION METRICS
# ============================================================

all_tokens = []

distinct1_values = []
distinct2_values = []
repetition_values = []

english_ratios = []
roman_hindi_ratios = []


for text in generated_texts:

    tokens = tokenize_words(text)

    all_tokens.extend(tokens)

    distinct1_values.append(
        distinct_n(tokens, 1)
    )

    distinct2_values.append(
        distinct_n(tokens, 2)
    )

    repetition_values.append(
        repetition_rate(tokens)
    )

    english_ratio, roman_hindi_ratio = (
        language_ratio(text)
    )

    english_ratios.append(
        english_ratio
    )

    roman_hindi_ratios.append(
        roman_hindi_ratio
    )


def mean(values):

    if not values:

        return 0.0

    return sum(values) / len(values)


distinct1 = mean(
    distinct1_values
)

distinct2 = mean(
    distinct2_values
)

repetition = mean(
    repetition_values
)

english_ratio = mean(
    english_ratios
)

roman_hindi_ratio = mean(
    roman_hindi_ratios
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"\nPerplexity       : {perplexity:.4f}"
)

print(
    f"Distinct-1       : {distinct1:.4f}"
)

print(
    f"Distinct-2       : {distinct2:.4f}"
)

print(
    f"Repetition Rate  : {repetition:.4f}"
)

print(
    f"English Ratio    : {english_ratio:.4f}"
)

print(
    f"Roman-Hindi Ratio: {roman_hindi_ratio:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": BASE_MODEL,

    "adapter": ADAPTER_PATH,

    "test_samples": len(test_texts),

    "generation_samples": GENERATION_SAMPLES,

    "metrics": {

        "perplexity": perplexity,

        "distinct_1": distinct1,

        "distinct_2": distinct2,

        "repetition_rate": repetition,

        "english_ratio": english_ratio,

        "roman_hindi_ratio": roman_hindi_ratio,
    },
}


os.makedirs(
    "results",
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
        indent=4,
        ensure_ascii=False,
    )


print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print("\n🎯 Qwen 1.5B evaluation completed!")