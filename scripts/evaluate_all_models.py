import os
import json
import math
import time
import re
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

TEST_FILE = "data/processed/test.jsonl"
OUTPUT_DIR = "results/final_evaluation"

MODELS = {
    "Qwen2.5-1.5B": {
        "base": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter": "results/qwen2.5-1.5b-qlora",
        "output": "qwen25_15b_metrics.json",
    },

    "Qwen2.5-3B": {
        "base": "Qwen/Qwen2.5-3B-Instruct",
        "adapter": "results/qwen2.5-3b-qlora",
        "output": "qwen25_3b_metrics.json",
    },

    "SmolLM2-1.7B": {
        "base": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "adapter": "results/smollm2-1.7b-qlora",
        "output": "smollm2_17b_metrics.json",
    },

    "Bharat-Tiny-LLM-v3": {
        "base": "eulogik/Bharat-Tiny-LLM-v3",
        "adapter": "results/bharat/checkpoint-22500",
        "output": "bharat_metrics.json",
    },
}


# ============================================================
# SETTINGS
# ============================================================

MAX_LENGTH = 64

# Fast generation
GEN_SAMPLES = 100
GEN_BATCH_SIZE = 4
MAX_NEW_TOKENS = 32

# PPL
PPL_BATCH_SIZE = 4

TEMPERATURE = 0.7
TOP_P = 0.90
REPETITION_PENALTY = 1.10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ALL MODELS EVALUATION")
print("=" * 70)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# LOAD TEST DATA
# ============================================================

texts = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        text = row.get("text", "")

        if text and text.strip():
            texts.append(text.strip())

print("\nTest samples:", len(texts))


# ============================================================
# 4-BIT CONFIG
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# ENGLISH WORD LIST
# ============================================================

ENGLISH_WORDS = {
    "the", "is", "are", "was", "were",
    "and", "or", "but", "for", "with",
    "this", "that", "what", "when",
    "where", "why", "how", "can",
    "could", "should", "would",
    "will", "have", "has", "had",
    "do", "does", "did",
    "not", "yes", "no",
    "good", "bad", "very",
    "please", "okay", "thanks",
    "thank", "hello", "today",
    "tomorrow", "yesterday",
    "time", "work", "home",
    "school", "college",
    "friend", "friends",
    "help", "need", "want",
    "know", "think", "make",
    "go", "come", "get",
    "give", "take", "use",
    "new", "old", "one",
    "two", "three"
}


# ============================================================
# LANGUAGE MIXING
# ============================================================

def calculate_mixing_ratio(generated_texts):

    total_words = 0
    english_words = 0

    for text in generated_texts:

        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

        total_words += len(words)

        for word in words:
            if word in ENGLISH_WORDS:
                english_words += 1

    if total_words == 0:
        return 0.0

    return english_words / total_words


# ============================================================
# REPETITION RATE
# ============================================================

def calculate_repetition_rate(generated_texts):

    repetition_scores = []

    for text in generated_texts:

        words = text.lower().split()

        if len(words) == 0:
            repetition_scores.append(0.0)
            continue

        unique_words = len(set(words))

        repetition = 1.0 - (unique_words / len(words))

        repetition_scores.append(repetition)

    if not repetition_scores:
        return 0.0

    return sum(repetition_scores) / len(repetition_scores)


# ============================================================
# DISTINCT
# ============================================================

def calculate_distinct(generated_texts):

    all_words = []

    for text in generated_texts:
        all_words.extend(text.lower().split())

    if len(all_words) == 0:
        return 0.0, 0.0

    unique_1 = len(set(all_words))

    distinct_1 = unique_1 / len(all_words)

    bigrams = list(zip(all_words, all_words[1:]))

    if len(bigrams) == 0:
        distinct_2 = 0.0
    else:
        distinct_2 = len(set(bigrams)) / len(bigrams)

    return distinct_1, distinct_2


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_perplexity(model, tokenizer):

    print("\nCalculating perplexity...")

    model.eval()

    total_nll = 0.0
    total_tokens = 0

    start_time = time.time()

    for start in range(0, len(texts), PPL_BATCH_SIZE):

        batch_texts = texts[start:start + PPL_BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        # IMPORTANT:
        # Ignore padding tokens in loss calculation
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

        # Calculate exact token-level NLL
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum"
        )

        valid_tokens = (shift_labels != -100).sum().item()

        total_nll += loss.item()
        total_tokens += valid_tokens

        processed = min(start + len(batch_texts), len(texts))

        if processed % 1000 == 0 or processed == len(texts):
            print(f"  PPL progress: {processed}/{len(texts)}")

    avg_nll = total_nll / total_tokens

    perplexity = math.exp(avg_nll)

    elapsed = time.time() - start_time

    print(f"PPL: {perplexity:.4f}")
    print(f"PPL time: {elapsed:.2f} sec")

    return perplexity, elapsed


# ============================================================
# GENERATION
# ============================================================

def generate_samples(model, tokenizer):

    print("\nGenerating samples...")

    model.eval()

    generation_texts = texts[:GEN_SAMPLES]

    generated_outputs = []

    start_time = time.time()

    tokenizer.padding_side = "left"

    for start in range(0, len(generation_texts), GEN_BATCH_SIZE):

        batch_texts = generation_texts[
            start:start + GEN_BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        input_length = input_ids.shape[1]

        generated_ids = outputs[:, input_length:]

        decoded = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        generated_outputs.extend(decoded)

        processed = min(
            start + len(batch_texts),
            len(generation_texts)
        )

        print(
            f"  Generation progress: "
            f"{processed}/{len(generation_texts)}"
        )

    elapsed = time.time() - start_time

    d1, d2 = calculate_distinct(generated_outputs)

    repetition = calculate_repetition_rate(
        generated_outputs
    )

    mixing = calculate_mixing_ratio(
        generated_outputs
    )

    print(f"Distinct-1: {d1:.4f}")
    print(f"Distinct-2: {d2:.4f}")
    print(f"Repetition Rate: {repetition:.4f}")
    print(f"Language Mixing Ratio: {mixing:.4f}")
    print(f"Generation time: {elapsed:.2f} sec")

    return {
        "distinct_1": d1,
        "distinct_2": d2,
        "repetition_rate": repetition,
        "language_mixing_ratio": mixing,
        "generation_time_seconds": elapsed,
        "samples": generated_outputs
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

all_results = {}

for model_name, config in MODELS.items():

    print("\n")
    print("=" * 70)
    print("MODEL:", model_name)
    print("=" * 70)

    model_start = time.time()

    try:

        # ----------------------------------------------------
        # TOKENIZER
        # ----------------------------------------------------

        print("\nLoading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            config["base"],
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        # ----------------------------------------------------
        # BASE MODEL
        # ----------------------------------------------------

        print("\nLoading base model:")
        print(config["base"])

        model = AutoModelForCausalLM.from_pretrained(
            config["base"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16
        )

        print("Base model loaded.")

        # ----------------------------------------------------
        # ADAPTER
        # ----------------------------------------------------

        print("\nLoading adapter:")
        print(config["adapter"])

        model = PeftModel.from_pretrained(
            model,
            config["adapter"]
        )

        print("Adapter loaded.")

        # ----------------------------------------------------
        # PPL
        # ----------------------------------------------------

        ppl, ppl_time = calculate_perplexity(
            model,
            tokenizer
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        gen_results = generate_samples(
            model,
            tokenizer
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        result = {
            "model": model_name,
            "base_model": config["base"],
            "adapter": config["adapter"],

            "perplexity": ppl,

            "distinct_1": gen_results["distinct_1"],
            "distinct_2": gen_results["distinct_2"],
            "repetition_rate": gen_results["repetition_rate"],
            "language_mixing_ratio": gen_results[
                "language_mixing_ratio"
            ],

            "ppl_time_seconds": ppl_time,
            "generation_time_seconds":
                gen_results["generation_time_seconds"],

            "ppl_samples": len(texts),
            "generation_samples": GEN_SAMPLES,

            "evaluation_time_seconds":
                time.time() - model_start
        }

        all_results[model_name] = result

        # ----------------------------------------------------
        # SAVE INDIVIDUAL RESULT
        # ----------------------------------------------------

        output_file = os.path.join(
            OUTPUT_DIR,
            config["output"]
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=4
            )

        print("\nSaved:", output_file)

        # ----------------------------------------------------
        # SAVE GENERATED SAMPLES
        # ----------------------------------------------------

        samples_dir = os.path.join(
            OUTPUT_DIR,
            "samples"
        )

        os.makedirs(samples_dir, exist_ok=True)

        sample_file = os.path.join(
            samples_dir,
            model_name.replace(".", "").replace("-", "_")
            + "_samples.jsonl"
        )

        with open(
            sample_file,
            "w",
            encoding="utf-8"
        ) as f:

            for text in gen_results["samples"]:

                f.write(
                    json.dumps(
                        {"generated_text": text},
                        ensure_ascii=False
                    )
                    + "\n"
                )

        # ----------------------------------------------------
        # CLEAN GPU MEMORY
        # ----------------------------------------------------

        del model
        del tokenizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:

        print("\nERROR:", model_name)
        print(e)

        all_results[model_name] = {
            "model": model_name,
            "error": str(e)
        }


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

combined_file = os.path.join(
    OUTPUT_DIR,
    "all_models_metrics.json"
)

with open(
    combined_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_results,
        f,
        indent=4
    )


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 95)
print("FINAL MODEL COMPARISON")
print("=" * 95)

print(
    f"{'Model':<28}"
    f"{'PPL':>10}"
    f"{'D1':>10}"
    f"{'D2':>10}"
    f"{'Repeat':>12}"
    f"{'Mixing':>12}"
)

print("-" * 95)

for model_name, result in all_results.items():

    if "error" in result:

        print(
            f"{model_name:<28}"
            f"{'ERROR':>10}"
        )

        continue

    print(
        f"{model_name:<28}"
        f"{result['perplexity']:>10.4f}"
        f"{result['distinct_1']:>10.4f}"
        f"{result['distinct_2']:>10.4f}"
        f"{result['repetition_rate']:>12.4f}"
        f"{result['language_mixing_ratio']:>12.4f}"
    )


print("=" * 95)
print("\nCombined results saved to:")
print(combined_file)

print("\nEvaluation completed.")