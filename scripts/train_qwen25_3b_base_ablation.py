import os
import torch

from datasets import load_dataset
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
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "results/qwen2.5-3b-base-qlora"

MAX_LENGTH = 64

SEED = 42


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 70)
print("QWEN2.5-3B BASE — QLoRA ABLATION TRAINING")
print("=" * 70)

print(f"\nModel       : {MODEL_NAME}")
print(f"Output      : {OUTPUT_DIR}")
print(f"Train file  : {TRAIN_FILE}")
print(f"Validation  : {VAL_FILE}")
print(f"Max length  : {MAX_LENGTH}")
print(f"Seed        : {SEED}")

print("\nCUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# DATASET
# ============================================================

print("\nLoading datasets...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

val_dataset = load_dataset(
    "json",
    data_files=VAL_FILE,
    split="train"
)

print(f"Training samples  : {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names
)

print("Tokenization completed.")


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
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16
)

model.config.use_cache = False


# ============================================================
# PREPARE FOR K-BIT TRAINING
# ============================================================

print("\nPreparing model for k-bit training...")

model = prepare_model_for_kbit_training(model)


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\nApplying LoRA...")

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
        "down_proj"
    ],

    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()


# ============================================================
# DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,
    weight_decay=0.01,

    max_steps=22500,

    lr_scheduler_type="cosine",

    fp16=True,

    tf32=torch.cuda.is_available(),

    gradient_checkpointing=False,

    optim="paged_adamw_8bit",

    logging_steps=100,

    eval_strategy="steps",
    eval_steps=2000,

    save_strategy="steps",
    save_steps=5000,

    save_total_limit=2,

    dataloader_num_workers=0,

    remove_unused_columns=False,

    seed=SEED,
    data_seed=SEED,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)

trainer.train()


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print("\nSaving final model adapter...")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# FINAL EVALUATION LOSS
# ============================================================

print("\nRunning final validation evaluation...")

eval_results = trainer.evaluate()

print("\nFinal validation results:")

for key, value in eval_results.items():
    print(f"{key}: {value}")


# ============================================================
# SAVE TRAINING SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "training_summary.txt"
)

with open(summary_file, "w", encoding="utf-8") as f:

    f.write("Qwen2.5-3B Base QLoRA Ablation\n")
    f.write("=" * 50 + "\n")

    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Dataset: 370K Hinglish corpus\n")
    f.write(f"Train samples: {len(train_dataset)}\n")
    f.write(f"Validation samples: {len(val_dataset)}\n")

    f.write("\nTraining configuration:\n")
    f.write("4-bit NF4 + double quantization\n")
    f.write("FP16 compute\n")
    f.write("LoRA r=8\n")
    f.write("LoRA alpha=16\n")
    f.write("LoRA dropout=0.05\n")
    f.write("Batch size=4\n")
    f.write("Gradient accumulation=4\n")
    f.write("Effective batch size=16\n")
    f.write("Learning rate=2e-4\n")
    f.write("Weight decay=0.01\n")
    f.write("Cosine scheduler\n")
    f.write("Max steps=22500\n")
    f.write("Seed=42\n")

    f.write("\nEvaluation results:\n")

    for key, value in eval_results.items():
        f.write(f"{key}: {value}\n")


print("\n" + "=" * 70)
print("BASE MODEL ABLATION TRAINING COMPLETED")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT_DIR)