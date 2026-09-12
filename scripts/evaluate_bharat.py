import os
import math
import time
import re
import json

import torch
import pandas as pd

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_MODEL = "eulogik/Bharat-Tiny-LLM-v3"

# Use the FINAL checkpoint
MODEL_PATH = "results/bharat/checkpoint-22500"

TEST_FILE = "data/processed/test.jsonl"

OUTPUT_DIR = "results/final_evaluation/bharat"

MAX_LENGTH = 64
MAX_NEW_TOKENS = 64

# Generation batch size
GEN_BATCH_SIZE = 4

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Controlled English word list
ENGLISH_WORDS = {
    "the", "is", "are", "was", "were", "am",
    "you", "your", "i", "we", "they", "he", "she",
    "this", "that", "these", "those",
    "what", "why", "when", "where", "how",
    "can", "could", "should", "would",
    "will", "shall", "do", "does", "did",
    "have", "has", "had",
    "good", "bad", "best", "better",
    "very", "really", "also", "just",
    "please", "thanks", "thank",
    "today", "tomorrow", "yesterday",
    "hello", "hi", "okay", "yes", "no",
    "and", "or", "but", "because",
    "for", "from", "with", "about",
    "in", "on", "at", "to", "of",
    "my", "me", "your", "our", "their",
    "not", "never", "always",
    "know", "think", "want", "need",
    "like", "love", "make", "go",
    "come", "give", "take", "get",
    "work", "time", "day", "people",
    "life", "friend", "friends",
    "help", "problem", "important"
}


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("=" * 80)
print("          BHARAT TINY LLM - 5 METRIC EVALUATION")
print("=" * 80)

print(f"\nDevice: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("\nLoading test dataset...")

test_dataset = load_dataset(
    "json",
    data_files=TEST_FILE,
    split="train"
)

print(f"Test samples: {len(test_dataset)}")


# ============================================================
# 3. LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# IMPORTANT for batched generation
tokenizer.padding_side = "left"


# ============================================================
# 4. LOAD MODEL
# ============================================================

print("\nLoading trained model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH
)

model.eval()

print("✓ Model loaded successfully")


# ============================================================
# 5. PERPLEXITY
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING PERPLEXITY")
print("=" * 80)

ppl_start = time.time()

losses = []

for start in range(0, len(test_dataset), GEN_BATCH_SIZE):

    batch = test_dataset[start:start + GEN_BATCH_SIZE]

    texts = batch["text"]

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    with torch.inference_mode():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )

    logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(
        reduction="none"
    )

    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    token_losses = token_losses.view(
        shift_labels.size()
    )

    for i in range(token_losses.size(0)):

        valid_losses = token_losses[i][
            shift_mask[i].bool()
        ]

        if len(valid_losses) > 0:
            losses.append(
                valid_losses.mean().item()
            )

    del input_ids
    del attention_mask
    del outputs
    del logits

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if start % (GEN_BATCH_SIZE * 100) == 0:
        print(
            f"PPL progress: "
            f"{min(start + GEN_BATCH_SIZE, len(test_dataset))}/"
            f"{len(test_dataset)}"
        )


average_loss = sum(losses) / len(losses)

perplexity = math.exp(average_loss)

ppl_time = time.time() - ppl_start

print(f"\nAverage Loss : {average_loss:.6f}")
print(f"Perplexity   : {perplexity:.4f}")
print(f"PPL Time     : {ppl_time:.2f} sec")


# ============================================================
# 6. GENERATE RESPONSES
# ============================================================

print("\n" + "=" * 80)
print("GENERATING MODEL RESPONSES")
print("=" * 80)

generation_start = time.time()

# Same 500-sample evaluation used for other models
generation_dataset = test_dataset.select(
    range(min(500, len(test_dataset)))
)

texts = generation_dataset["text"]

generated_texts = []

total = len(texts)

for start in range(0, total, GEN_BATCH_SIZE):

    batch_texts = texts[
        start:start + GEN_BATCH_SIZE
    ]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        k: v.to(model.device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Remove the input portion
    input_length = inputs["input_ids"].shape[1]

    generated_only = outputs[:, input_length:]

    decoded = tokenizer.batch_decode(
        generated_only,
        skip_special_tokens=True
    )

    generated_texts.extend(decoded)

    del inputs
    del outputs

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    completed = min(
        start + GEN_BATCH_SIZE,
        total
    )

    print(
        f"Generation progress: "
        f"{completed}/{total}"
    )


generation_time = time.time() - generation_start

print(
    f"\nGeneration completed in "
    f"{generation_time:.2f} seconds"
)


# ============================================================
# 7. DISTINCT-1
# ============================================================

def calculate_distinct_1(texts):

    total_tokens = 0
    unique_tokens = set()

    for text in texts:

        tokens = text.lower().split()

        total_tokens += len(tokens)

        unique_tokens.update(tokens)

    if total_tokens == 0:
        return 0.0

    return len(unique_tokens) / total_tokens


# ============================================================
# 8. DISTINCT-2
# ============================================================

def calculate_distinct_2(texts):

    total_bigrams = 0
    unique_bigrams = set()

    for text in texts:

        tokens = text.lower().split()

        bigrams = list(
            zip(tokens, tokens[1:])
        )

        total_bigrams += len(bigrams)

        unique_bigrams.update(bigrams)

    if total_bigrams == 0:
        return 0.0

    return len(unique_bigrams) / total_bigrams


# ============================================================
# 9. REPETITION RATE
# ============================================================

def calculate_repetition_rate(texts):

    repetition_scores = []

    for text in texts:

        tokens = text.lower().split()

        if len(tokens) == 0:
            continue

        unique_count = len(set(tokens))

        repetition = 1 - (
            unique_count / len(tokens)
        )

        repetition_scores.append(
            repetition
        )

    if len(repetition_scores) == 0:
        return 0.0

    return sum(repetition_scores) / len(
        repetition_scores
    )


# ============================================================
# 10. LANGUAGE MIXING RATIO
# ============================================================

def calculate_language_mixing(texts):

    ratios = []

    for text in texts:

        words = re.findall(
            r"\b[a-zA-Z]+\b",
            text.lower()
        )

        if len(words) == 0:
            continue

        english_count = sum(
            1
            for word in words
            if word in ENGLISH_WORDS
        )

        ratio = english_count / len(words)

        ratios.append(ratio)

    if len(ratios) == 0:
        return 0.0

    return sum(ratios) / len(ratios)


# ============================================================
# 11. CALCULATE METRICS
# ============================================================

distinct_1 = calculate_distinct_1(
    generated_texts
)

distinct_2 = calculate_distinct_2(
    generated_texts
)

repetition_rate = calculate_repetition_rate(
    generated_texts
)

language_mixing_ratio = calculate_language_mixing(
    generated_texts
)


# ============================================================
# 12. SAVE GENERATED SAMPLES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

samples_file = os.path.join(
    OUTPUT_DIR,
    "generated_samples.jsonl"
)

with open(
    samples_file,
    "w",
    encoding="utf-8"
) as f:

    for i in range(len(generated_texts)):

        record = {
            "input": texts[i],
            "generated": generated_texts[i]
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# 13. SAVE METRICS
# ============================================================

metrics = {
    "model": "Bharat-Tiny-LLM-v3",
    "base_model": BASE_MODEL,
    "checkpoint": MODEL_PATH,

    "perplexity": round(
        perplexity,
        4
    ),

    "distinct_1": round(
        distinct_1,
        4
    ),

    "distinct_2": round(
        distinct_2,
        4
    ),

    "repetition_rate": round(
        repetition_rate,
        4
    ),

    "language_mixing_ratio": round(
        language_mixing_ratio,
        4
    ),

    "test_samples": len(test_dataset),
    "generation_samples": len(generated_texts),

    "ppl_time_seconds": round(
        ppl_time,
        2
    ),

    "generation_time_seconds": round(
        generation_time,
        2
    )
}

metrics_file = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)

with open(
    metrics_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("RESULTS: Bharat-Tiny-LLM-v3")
print("=" * 80)

print(
    f"Perplexity             : {perplexity:.4f}"
)

print(
    f"Distinct-1             : {distinct_1:.4f}"
)

print(
    f"Distinct-2             : {distinct_2:.4f}"
)

print(
    f"Repetition Rate        : {repetition_rate:.4f}"
)

print(
    f"Language Mixing Ratio  : "
    f"{language_mixing_ratio:.4f}"
)

print(
    f"Evaluation Time        : "
    f"{ppl_time + generation_time:.2f} sec"
)

print("=" * 80)

print("\n✓ Bharat evaluation completed successfully!")

print(
    f"\nMetrics saved to:\n{metrics_file}"
)

print(
    f"Generated samples saved to:\n{samples_file}"
)