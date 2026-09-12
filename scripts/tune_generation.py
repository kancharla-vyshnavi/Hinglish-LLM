import os
import json
import math
import re
import time
from collections import Counter

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# SETTINGS
# ============================================================

TEST_FILE = "data/processed/test.jsonl"

# Tune on 500 fixed held-out samples
NUM_SAMPLES = 500

MAX_LENGTH = 64
MAX_NEW_TOKENS = 64

BATCH_SIZE = 4

OUTPUT_DIR = "results/generation_tuning/qwen2.5-3b"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER = "results/qwen2.5-3b-qlora"


# ============================================================
# GENERATION SETTINGS TO TEST
# ============================================================

CONFIGURATIONS = [

    {
        "name": "config_1",
        "temperature": 0.6,
        "top_p": 0.85,
        "repetition_penalty": 1.05
    },

    {
        "name": "config_2",
        "temperature": 0.7,
        "top_p": 0.90,
        "repetition_penalty": 1.05
    },

    {
        "name": "config_3",
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.05
    },

    {
        "name": "config_4",
        "temperature": 0.6,
        "top_p": 0.90,
        "repetition_penalty": 1.10
    },

    {
        "name": "config_5",
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.10
    },

    {
        "name": "config_6",
        "temperature": 0.8,
        "top_p": 0.90,
        "repetition_penalty": 1.10
    },

    {
        "name": "config_7",
        "temperature": 0.7,
        "top_p": 0.90,
        "repetition_penalty": 1.15
    },

    {
        "name": "config_8",
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.15
    }
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "=" * 80)
print("       GENERATION PARAMETER TUNING")
print("=" * 80)

print(f"Device : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    texts = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            item = json.loads(line)

            text = item.get("text", "")

            if text is None:
                continue

            text = str(text).strip()

            if text:
                texts.append(text)

    return texts


test_texts = load_test_data()

generation_texts = test_texts[:NUM_SAMPLES]

print(f"Test samples loaded : {len(test_texts)}")
print(f"Tuning samples      : {len(generation_texts)}")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

print("Tokenizer loaded")


# ============================================================
# 4-BIT CONFIGURATION
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen2.5-3B in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True
)

print("Base model loaded")

print("\nLoading QLoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER
)

model.eval()

print("QLoRA adapter loaded")


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_batch(batch_texts):

    encoded = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    return (
        encoded["input_ids"].to(model.device),
        encoded["attention_mask"].to(model.device)
    )


# ============================================================
# GENERATION
# ============================================================

def generate_texts(config):

    generated_texts = []

    total_batches = math.ceil(
        len(generation_texts) / BATCH_SIZE
    )

    for batch_start in range(
        0,
        len(generation_texts),
        BATCH_SIZE
    ):

        batch_texts = generation_texts[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_number = (
            batch_start // BATCH_SIZE
        ) + 1

        if batch_number % 10 == 1 or batch_number == total_batches:

            processed = min(
                batch_start + BATCH_SIZE,
                len(generation_texts)
            )

            print(
                f"  Progress: "
                f"{processed}/{len(generation_texts)}"
            )

        input_ids, attention_mask = tokenize_batch(
            batch_texts
        )

        with torch.inference_mode():

            output_ids = model.generate(

                input_ids=input_ids,

                attention_mask=attention_mask,

                max_new_tokens=MAX_NEW_TOKENS,

                do_sample=True,

                temperature=config["temperature"],

                top_p=config["top_p"],

                repetition_penalty=config[
                    "repetition_penalty"
                ],

                pad_token_id=tokenizer.pad_token_id,

                eos_token_id=tokenizer.eos_token_id
            )

        for row in range(
            output_ids.shape[0]
        ):

            generated_ids = output_ids[
                row,
                input_ids.shape[1]:
            ]

            output_text = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            output_text = re.sub(
                r"\s+",
                " ",
                output_text
            ).strip()

            generated_texts.append(
                output_text
            )

        del input_ids
        del attention_mask
        del output_ids

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return generated_texts


# ============================================================
# WORD TOKENIZER
# ============================================================

def word_tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower(),
        flags=re.UNICODE
    )


# ============================================================
# DISTINCT-N
# ============================================================

def calculate_distinct(texts, n):

    all_ngrams = []

    for text in texts:

        tokens = word_tokenize(text)

        if len(tokens) < n:
            continue

        for i in range(
            len(tokens) - n + 1
        ):

            all_ngrams.append(
                tuple(tokens[i:i + n])
            )

    if not all_ngrams:
        return 0.0

    return (
        len(set(all_ngrams))
        /
        len(all_ngrams)
    )


# ============================================================
# REPETITION RATE
# ============================================================

def calculate_repetition_rate(texts):

    total_tokens = 0
    repeated_tokens = 0

    for text in texts:

        tokens = word_tokenize(text)

        if not tokens:
            continue

        counts = Counter(tokens)

        total_tokens += len(tokens)

        for count in counts.values():

            if count > 1:

                repeated_tokens += (
                    count - 1
                )

    if total_tokens == 0:
        return 0.0

    return (
        repeated_tokens /
        total_tokens
    )


# ============================================================
# CONTROLLED ENGLISH WORD LIST
# ============================================================

ENGLISH_WORDS = {
    "the", "a", "an", "and", "or", "but",
    "if", "then", "than", "because", "so",
    "that", "this", "these", "those",
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "will", "would",
    "shall", "should",
    "may", "might", "must",
    "not", "no", "yes",
    "i", "me", "my", "mine",
    "you", "your", "yours",
    "he", "him", "his",
    "she", "her", "hers",
    "we", "us", "our", "ours",
    "they", "them", "their", "theirs",
    "what", "why", "when", "where",
    "who", "whom", "which", "how",
    "in", "on", "at", "by", "for",
    "from", "with", "about", "into",
    "over", "under", "after", "before",
    "between", "during", "through",
    "to", "of", "as",
    "hello", "hi", "hey",
    "thanks", "thank", "please", "sorry",
    "okay", "ok",
    "good", "bad", "better", "best",
    "very", "really", "just", "also",
    "only", "more", "most",
    "some", "any", "many", "much",
    "today", "tomorrow", "yesterday",
    "now", "here", "there",
    "time", "day", "people", "person",
    "thing", "things", "work", "school",
    "college", "home",
    "friend", "friends", "family",
    "one", "two", "three",
    "first", "second",
    "new", "old", "big", "small",
    "right", "wrong"
}


# ============================================================
# LANGUAGE MIXING
# ============================================================

def calculate_language_mixing(texts):

    total_tokens = 0
    english_tokens = 0

    for text in texts:

        tokens = word_tokenize(text)

        for token in tokens:

            total_tokens += 1

            if token in ENGLISH_WORDS:
                english_tokens += 1

    if total_tokens == 0:
        return 0.0

    return (
        english_tokens /
        total_tokens
    )


# ============================================================
# RUN TUNING
# ============================================================

all_results = []

for config in CONFIGURATIONS:

    print("\n" + "=" * 80)

    print(
        f"TESTING {config['name']}"
    )

    print("=" * 80)

    print(
        f"Temperature        : "
        f"{config['temperature']}"
    )

    print(
        f"Top-p              : "
        f"{config['top_p']}"
    )

    print(
        f"Repetition penalty : "
        f"{config['repetition_penalty']}"
    )

    start_time = time.time()

    generated_texts = generate_texts(
        config
    )

    distinct_1 = calculate_distinct(
        generated_texts,
        1
    )

    distinct_2 = calculate_distinct(
        generated_texts,
        2
    )

    repetition = calculate_repetition_rate(
        generated_texts
    )

    mixing = calculate_language_mixing(
        generated_texts
    )

    elapsed = time.time() - start_time

    result = {

        "configuration":
            config["name"],

        "temperature":
            config["temperature"],

        "top_p":
            config["top_p"],

        "repetition_penalty":
            config["repetition_penalty"],

        "distinct_1":
            round(distinct_1, 4),

        "distinct_2":
            round(distinct_2, 4),

        "repetition_rate":
            round(repetition, 4),

        "language_mixing_ratio":
            round(mixing, 4),

        "time_seconds":
            round(elapsed, 2)
    }

    all_results.append(result)

    print("\nRESULT")

    print(
        f"Distinct-1            : "
        f"{distinct_1:.4f}"
    )

    print(
        f"Distinct-2            : "
        f"{distinct_2:.4f}"
    )

    print(
        f"Repetition Rate        : "
        f"{repetition:.4f}"
    )

    print(
        f"Language Mixing Ratio  : "
        f"{mixing:.4f}"
    )

    print(
        f"Time                   : "
        f"{elapsed:.2f} sec"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    OUTPUT_DIR,
    "tuning_results.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n\n")

print("=" * 100)
print("                 GENERATION TUNING RESULTS")
print("=" * 100)

print(
    f"{'Config':<12}"
    f"{'Temp':>10}"
    f"{'Top-p':>10}"
    f"{'RepPen':>10}"
    f"{'D-1':>12}"
    f"{'D-2':>12}"
    f"{'Repeat':>12}"
    f"{'Mixing':>12}"
)

print("-" * 100)

for result in all_results:

    print(
        f"{result['configuration']:<12}"
        f"{result['temperature']:>10.2f}"
        f"{result['top_p']:>10.2f}"
        f"{result['repetition_penalty']:>10.2f}"
        f"{result['distinct_1']:>12.4f}"
        f"{result['distinct_2']:>12.4f}"
        f"{result['repetition_rate']:>12.4f}"
        f"{result['language_mixing_ratio']:>12.4f}"
    )

print("-" * 100)

print(
    f"\nResults saved to:\n{output_file}"
)

print("\n" + "=" * 100)
print("                  TUNING COMPLETED")
print("=" * 100)