import json
import os
import re
import time
import math
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B"

ADAPTER_PATH = "results/qwen2.5-3b-base-qlora"

TEST_FILE = "data/processed/test.jsonl"

OUTPUT_DIR = "results/final_evaluation"

METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_3b_base_metrics.json"
)

SAMPLES_FILE = os.path.join(
    OUTPUT_DIR,
    "qwen25_3b_base_samples.jsonl"
)

MAX_LENGTH = 64

GENERATION_SAMPLES = 100

BATCH_SIZE = 4

SEED = 42


# ============================================================
# SETUP
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print("QWEN2.5-3B BASE + QLoRA ABLATION EVALUATION")
print("=" * 70)

print(f"\nBase model : {BASE_MODEL}")
print(f"Adapter    : {ADAPTER_PATH}")
print(f"Test file  : {TEST_FILE}")

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

dataset = load_dataset(
    "json",
    data_files=TEST_FILE,
    split="train"
)

print(
    f"Test samples: {len(dataset):,}"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


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

print("\nLoading Qwen2.5-3B BASE model...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

model.config.use_cache = False


# ============================================================
# LOAD LoRA ADAPTER
# ============================================================

print("\nLoading trained Base-model LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH
)

model.eval()

print("Adapter loaded successfully.")


# ============================================================
# DETERMINE DEVICE
# ============================================================

device = next(
    model.parameters()
).device

print("Model device:", device)


# ============================================================
# PERPLEXITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING PERPLEXITY")
print("=" * 70)

ppl_start_time = time.time()

total_negative_log_likelihood = 0.0

total_valid_tokens = 0

model.config.use_cache = False

with torch.no_grad():

    for start_idx in range(
        0,
        len(dataset),
        BATCH_SIZE
    ):

        end_idx = min(
            start_idx + BATCH_SIZE,
            len(dataset)
        )

        batch_texts = dataset[
            start_idx:end_idx
        ]["text"]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )

        input_ids = inputs["input_ids"]

        attention_mask = inputs[
            "attention_mask"
        ]

        # ----------------------------------------------------
        # IMPORTANT:
        # Ignore padding tokens during loss calculation.
        # ----------------------------------------------------

        labels = input_ids.clone()

        labels[
            attention_mask == 0
        ] = -100

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        labels = labels.to(device)

        outputs = model(
            **inputs,
            labels=labels
        )

        # ----------------------------------------------------
        # Calculate token-level negative log likelihood
        # manually so padding does not affect PPL.
        # ----------------------------------------------------

        logits = outputs.logits

        shift_logits = logits[
            :, :-1, :
        ].contiguous()

        shift_labels = labels[
            :, 1:
        ].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="sum",
            ignore_index=-100
        )

        batch_loss = loss_fct(
            shift_logits.view(
                -1,
                shift_logits.size(-1)
            ),
            shift_labels.view(-1)
        )

        valid_tokens = (
            shift_labels != -100
        ).sum().item()

        total_negative_log_likelihood += (
            batch_loss.item()
        )

        total_valid_tokens += valid_tokens

        processed = end_idx

        if (
            processed % 500 == 0
            or processed == len(dataset)
        ):
            print(
                f"Processed "
                f"{processed:,}/{len(dataset):,}"
            )


ppl_time = (
    time.time()
    - ppl_start_time
)


# ============================================================
# FINAL PPL
# ============================================================

average_negative_log_likelihood = (
    total_negative_log_likelihood
    / total_valid_tokens
)

perplexity = math.exp(
    average_negative_log_likelihood
)

print(
    f"\nTotal valid tokens : "
    f"{total_valid_tokens:,}"
)

print(
    f"Average NLL/token  : "
    f"{average_negative_log_likelihood:.6f}"
)

print(
    f"Perplexity         : "
    f"{perplexity:.4f}"
)

print(
    f"PPL evaluation time: "
    f"{ppl_time:.2f} seconds"
)


# ============================================================
# GENERATION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SAMPLES")
print("=" * 70)

generation_start = time.time()

generation_records = []

number_to_generate = min(
    GENERATION_SAMPLES,
    len(dataset)
)

for i in range(
    number_to_generate
):

    text = dataset[i]["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,

            max_new_tokens=64,

            do_sample=True,

            temperature=0.7,

            top_p=0.9,

            repetition_penalty=1.05,

            pad_token_id=tokenizer.pad_token_id,

            eos_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    generation_records.append(
        {
            "id": i,
            "generated_text": generated
        }
    )

    if (
        (i + 1) % 10 == 0
        or (i + 1) == number_to_generate
    ):
        print(
            f"Generated "
            f"{i + 1}/{number_to_generate}"
        )


generation_time = (
    time.time()
    - generation_start
)


print(
    f"\nGeneration time: "
    f"{generation_time:.2f} seconds"
)


# ============================================================
# SAVE GENERATION SAMPLES
# ============================================================

with open(
    SAMPLES_FILE,
    "w",
    encoding="utf-8"
) as f:

    for record in generation_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print(
    f"Saved samples:"
    f"\n{SAMPLES_FILE}"
)


# ============================================================
# WORD TOKENIZATION
# ============================================================

def tokenize_words(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ============================================================
# COLLECT WORDS AND BIGRAMS
# ============================================================

all_words = []

all_bigrams = []

repeated_bigram_count = 0

total_words = 0


for record in generation_records:

    words = tokenize_words(
        record["generated_text"]
    )

    all_words.extend(words)

    total_words += len(words)

    if len(words) > 1:

        bigrams = list(
            zip(
                words[:-1],
                words[1:]
            )
        )

        all_bigrams.extend(
            bigrams
        )

        repeated_bigram_count += (
            len(bigrams)
            - len(set(bigrams))
        )


# ============================================================
# DISTINCT-1
# ============================================================

if len(all_words) > 0:

    distinct_1 = (
        len(set(all_words))
        / len(all_words)
    )

else:

    distinct_1 = 0.0


# ============================================================
# DISTINCT-2
# ============================================================

if len(all_bigrams) > 0:

    distinct_2 = (
        len(set(all_bigrams))
        / len(all_bigrams)
    )

else:

    distinct_2 = 0.0


# ============================================================
# REPETITION RATE
# ============================================================

if len(all_bigrams) > 0:

    repetition_rate = (
        repeated_bigram_count
        / len(all_bigrams)
    )

else:

    repetition_rate = 0.0


# ============================================================
# LANGUAGE MIXING RATIO
# ============================================================

english_words = {
    "the",
    "is",
    "are",
    "was",
    "were",
    "and",
    "or",
    "but",
    "for",
    "with",
    "this",
    "that",
    "what",
    "why",
    "how",
    "can",
    "will",
    "should",
    "you",
    "your",
    "i",
    "we",
    "they",
    "he",
    "she",
    "it",
    "my",
    "to",
    "in",
    "on",
    "of",
    "from",
    "a",
    "an",
    "be",
    "have",
    "has",
    "do",
    "does",
    "not",
    "yes",
    "no",
    "please",
    "good",
    "very",
    "learn",
    "learning",
    "code",
    "coding",
    "data",
    "model",
    "project",
    "study",
    "exam",
    "college",
    "class",
    "problem",
    "solution",
    "time",
    "day",
    "today",
    "tomorrow",
    "help"
}


english_count = 0


for word in all_words:

    if word in english_words:

        english_count += 1


if len(all_words) > 0:

    language_mixing_ratio = (
        english_count
        / len(all_words)
    )

else:

    language_mixing_ratio = 0.0


# ============================================================
# AVERAGE GENERATION LENGTH
# ============================================================

generation_lengths = []

for record in generation_records:

    words = tokenize_words(
        record["generated_text"]
    )

    generation_lengths.append(
        len(words)
    )


if generation_lengths:

    average_generation_length = (
        sum(generation_lengths)
        / len(generation_lengths)
    )

else:

    average_generation_length = 0.0


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "Qwen2.5-3B-Base",

    "base_model":
        BASE_MODEL,

    "adapter":
        ADAPTER_PATH,

    "test_samples":
        len(dataset),

    "perplexity":
        perplexity,

    "average_nll":
        average_negative_log_likelihood,

    "distinct_1":
        distinct_1,

    "distinct_2":
        distinct_2,

    "repetition_rate":
        repetition_rate,

    "language_mixing_ratio":
        language_mixing_ratio,

    "average_generation_length":
        average_generation_length,

    "ppl_time_seconds":
        ppl_time,

    "generation_time_seconds":
        generation_time,

    "generation_samples":
        len(generation_records),

    "generation_throughput_samples_per_second":
        (
            len(generation_records)
            / generation_time
            if generation_time > 0
            else 0.0
        )
}


with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("BASE ABLATION EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nPerplexity      : "
    f"{perplexity:.4f}"
)

print(
    f"Distinct-1      : "
    f"{distinct_1:.4f}"
)

print(
    f"Distinct-2      : "
    f"{distinct_2:.4f}"
)

print(
    f"Repetition      : "
    f"{repetition_rate:.4f}"
)

print(
    f"Language mixing : "
    f"{language_mixing_ratio:.4f}"
)

print(
    f"Avg gen length  : "
    f"{average_generation_length:.2f}"
)

print(
    f"Throughput      : "
    f"{metrics['generation_throughput_samples_per_second']:.3f} "
    f"samples/sec"
)

print(
    "\nSaved metrics:"
)

print(METRICS_FILE)

print(
    "\nSaved samples:"
)

print(SAMPLES_FILE)

print("=" * 70)