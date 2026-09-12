# ============================================================
# Qwen 2.5 1.5B - QLoRA Training
# Hindi-English Code-Mixed Text
# RTX 4050 6GB - Windows
# RESUME FROM CHECKPOINT ENABLED
# ============================================================

import os
import json
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "results/qwen2.5-1.5b-qlora"

EXPECTED_TRAIN_SIZE = 360000
EXPECTED_VAL_SIZE = 5000

MAX_LENGTH = 64

# RTX 4050 optimized
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

# 4 x 4 = 16
EFFECTIVE_BATCH_SIZE = (
    PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
)

# 360,000 / 16 = 22,500 steps
MAX_STEPS = 22500

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# 2. GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. Check your PyTorch/CUDA setup."
    )

print("=" * 70)
print("GPU INFORMATION")
print("=" * 70)

print(f"GPU              : {torch.cuda.get_device_name(0)}")
print(f"CUDA available   : {torch.cuda.is_available()}")

total_memory = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)

print(f"GPU memory       : {total_memory:.2f} GB")

# GPU performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")


# ============================================================
# 3. LOAD JSONL
# ============================================================

def load_jsonl(path):

    texts = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            item = json.loads(line)

            text = item.get("text", "")

            if text:
                texts.append(text)

    return texts


print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

train_texts = load_jsonl(TRAIN_FILE)
val_texts = load_jsonl(VAL_FILE)

print(f"Training examples   : {len(train_texts):,}")
print(f"Validation examples : {len(val_texts):,}")


# ============================================================
# 4. DATASET SIZE CHECK
# ============================================================

if len(train_texts) != EXPECTED_TRAIN_SIZE:

    raise ValueError(
        f"Expected {EXPECTED_TRAIN_SIZE:,} training examples, "
        f"found {len(train_texts):,}"
    )


if len(val_texts) != EXPECTED_VAL_SIZE:

    raise ValueError(
        f"Expected {EXPECTED_VAL_SIZE:,} validation examples, "
        f"found {len(val_texts):,}"
    )


# ============================================================
# 5. CREATE DATASETS
# ============================================================

train_dataset = Dataset.from_dict({
    "text": train_texts
})

val_dataset = Dataset.from_dict({
    "text": val_texts
})

print("\nDataset verification:")
print(f"Train      : {len(train_dataset):,}")
print(f"Validation : {len(val_dataset):,}")


# ============================================================
# 6. TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"


# ============================================================
# 7. TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )


print("\nTokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="Tokenizing train",
)

print("Tokenizing validation dataset...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,
    remove_columns=["text"],
    desc="Tokenizing validation",
)


# ============================================================
# 8. QLORA CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("QLORA CONFIGURATION")
print("=" * 70)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# 9. LOAD MODEL
# ============================================================

print("\nLoading Qwen 2.5 1.5B model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.float16,
)

model.config.use_cache = False


# ============================================================
# 10. PREPARE MODEL FOR QLORA
# ============================================================

model = prepare_model_for_kbit_training(model)


# ============================================================
# 11. LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    bias="none",
    task_type="CAUSAL_LM",
)


# ============================================================
# 12. APPLY LORA
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)

print("\n" + "=" * 70)
print("TRAINABLE PARAMETERS")
print("=" * 70)

model.print_trainable_parameters()


# ============================================================
# 13. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 14. FIND LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint(output_dir):

    if not os.path.exists(output_dir):
        return None

    checkpoints = []

    for name in os.listdir(output_dir):

        full_path = os.path.join(output_dir, name)

        if (
            os.path.isdir(full_path)
            and name.startswith("checkpoint-")
        ):

            try:
                step = int(
                    name.replace("checkpoint-", "")
                )

                checkpoints.append(
                    (step, full_path)
                )

            except ValueError:
                pass

    if not checkpoints:
        return None

    checkpoints.sort(
        key=lambda x: x[0]
    )

    return checkpoints[-1][1]


latest_checkpoint = find_latest_checkpoint(
    OUTPUT_DIR
)


# ============================================================
# 15. TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CONFIGURATION")
print("=" * 70)

print(f"Model                 : {MODEL_NAME}")
print(f"Train examples        : {len(train_dataset):,}")
print(f"Validation examples   : {len(val_dataset):,}")
print(f"Max sequence length   : {MAX_LENGTH}")

print(f"\nPer-device batch      : {PER_DEVICE_BATCH_SIZE}")
print(
    f"Gradient accumulation : "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(f"Effective batch       : {EFFECTIVE_BATCH_SIZE}")

print(f"\nLearning rate         : {LEARNING_RATE}")
print(f"Weight decay          : {WEIGHT_DECAY}")

print(f"\nTotal training steps  : {MAX_STEPS:,}")

print("\nQLoRA:")
print("  4-bit NF4")
print("  Double quantization")
print("  LoRA r=8")
print("  LoRA alpha=16")
print("  LoRA dropout=0.05")

print("\nGPU optimization:")
print("  TF32 enabled")
print("  Gradient checkpointing OFF")
print("  8-bit AdamW optimizer")
print("  Batch size = 4")
print("  Gradient accumulation = 4")

print("\nWindows:")
print("  DataLoader workers = 0")


# ============================================================
# 16. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,

    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    max_steps=MAX_STEPS,

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    lr_scheduler_type="cosine",

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="paged_adamw_8bit",

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    fp16=True,

    # --------------------------------------------------------
    # Gradient checkpointing
    # --------------------------------------------------------

    gradient_checkpointing=False,

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=2000,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=100,

    # --------------------------------------------------------
    # Saving
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=5000,

    save_total_limit=2,

    # --------------------------------------------------------
    # WINDOWS SAFE DATA LOADING
    # --------------------------------------------------------

    dataloader_num_workers=0,

    dataloader_pin_memory=True,

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,

    data_seed=SEED,

    # --------------------------------------------------------
    # PEFT
    # --------------------------------------------------------

    remove_unused_columns=False,

    # --------------------------------------------------------
    # Reporting
    # --------------------------------------------------------

    report_to="none",
)


# ============================================================
# 17. CREATE TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,
)


# ============================================================
# 18. RESUME INFORMATION
# ============================================================

print("\n" + "=" * 70)

if latest_checkpoint:

    print("🔄 CHECKPOINT FOUND")
    print("=" * 70)

    print(f"Latest checkpoint:")
    print(latest_checkpoint)

    print(
        "\nTraining will RESUME from the latest checkpoint."
    )

else:

    print("🆕 NO CHECKPOINT FOUND")
    print("=" * 70)

    print(
        "\nTraining will START FROM STEP 0."
    )


# ============================================================
# 19. START / RESUME TRAINING
# ============================================================

print("\n" + "=" * 70)
print("🚀 STARTING QLORA TRAINING")
print("=" * 70)

print(f"Training examples : {EXPECTED_TRAIN_SIZE:,}")
print(f"Validation        : {EXPECTED_VAL_SIZE:,}")
print(f"Effective batch   : {EFFECTIVE_BATCH_SIZE}")
print(f"Total steps       : {MAX_STEPS:,}")

if latest_checkpoint:

    print(
        f"\n🔄 RESUMING FROM:"
    )

    print(latest_checkpoint)

    print(
        "\nExpected continuation:"
    )

    print(
        "10,000 → 22,500 steps"
    )

else:

    print(
        "\nStarting from step 0."
    )

print("\nGPU monitor:")
print("nvidia-smi -l 1")

print("=" * 70)


# ============================================================
# 20. TRAIN
# ============================================================

if latest_checkpoint:

    train_result = trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    train_result = trainer.train()


# ============================================================
# 21. SAVE FINAL ADAPTER
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINAL QLORA ADAPTER")
print("=" * 70)

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)

print("\n✅ Final adapter saved to:")
print(OUTPUT_DIR)


# ============================================================
# 22. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION")
print("=" * 70)

eval_results = trainer.evaluate()

for key, value in eval_results.items():

    if isinstance(value, float):

        print(
            f"{key:25s}: {value:.6f}"
        )

    else:

        print(
            f"{key:25s}: {value}"
        )


# ============================================================
# 23. GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print("\n" + "=" * 70)
    print("FINAL GPU MEMORY")
    print("=" * 70)

    print(
        f"Allocated : {allocated:.2f} GB"
    )

    print(
        f"Reserved  : {reserved:.2f} GB"
    )


# ============================================================
# 24. COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("🎯 QLORA TRAINING COMPLETED")
print("=" * 70)

print("\nModel:")
print("Qwen/Qwen2.5-1.5B-Instruct")

print("\nMethod:")
print("QLoRA + 4-bit NF4 + LoRA")

print("\nDataset:")
print("360,000 training examples")
print("5,000 validation examples")

print("\nTraining:")
print("Effective batch size : 16")
print("Total training steps : 22,500")

print("\nAdapter:")
print(OUTPUT_DIR)

print("\n🎯 READY FOR TEST EVALUATION!")

print("=" * 70)