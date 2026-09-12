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

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "results/qwen2.5-3b-qlora"

MAX_LENGTH = 64

# SAME SETTINGS AS QWEN 1.5B
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

MAX_STEPS = 22500

SEED = 42


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("        QWEN 2.5 - 3B QLORA TRAINING")
print("=" * 60)

if torch.cuda.is_available():

    print("\nGPU detected:")
    print(torch.cuda.get_device_name(0))

    gpu_memory = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU Memory: {gpu_memory:.2f} GB")

else:

    print("\nWARNING: CUDA GPU NOT DETECTED!")
    print("Training will be extremely slow.")

print()


# ============================================================
# DATASET
# ============================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

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

print(f"\nTraining examples   : {len(train_dataset)}")
print(f"Validation examples : {len(val_dataset)}")


# ============================================================
# TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("Loading tokenizer...")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
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
        padding="max_length"
    )


print("\nTokenizing training dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train dataset"
)

print("\nTokenizing validation dataset...")

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=val_dataset.column_names,
    desc="Tokenizing validation dataset"
)


# ============================================================
# 4-BIT QLORA CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("Configuring 4-bit QLoRA...")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading Qwen 2.5 3B model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
    trust_remote_code=True,
)

model.config.use_cache = False


# ============================================================
# PREPARE MODEL FOR K-BIT TRAINING
# ============================================================

print("\nPreparing model for k-bit training...")

model = prepare_model_for_kbit_training(model)


# ============================================================
# LORA CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("Configuring LoRA...")
print("=" * 60)

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
# APPLY LORA
# ============================================================

print("\nApplying LoRA...")

model = get_peft_model(
    model,
    lora_config
)

print("\nTrainable parameters:")

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

print("\n" + "=" * 60)
print("Configuring training...")
print("=" * 60)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    max_steps=MAX_STEPS,

    lr_scheduler_type="cosine",

    fp16=True,

    tf32=True,

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

    report_to="none",

    seed=SEED,

    data_seed=SEED,
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,
)


# ============================================================
# FIND LATEST CHECKPOINT
# ============================================================

latest_checkpoint = None

if os.path.exists(OUTPUT_DIR):

    checkpoints = [

        os.path.join(OUTPUT_DIR, folder)

        for folder in os.listdir(OUTPUT_DIR)

        if folder.startswith("checkpoint-")

        and os.path.isdir(
            os.path.join(OUTPUT_DIR, folder)
        )

    ]

    if checkpoints:

        latest_checkpoint = max(

            checkpoints,

            key=lambda x: int(
                os.path.basename(x).split("-")[1]
            )

        )


# ============================================================
# TRAINING INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("        TRAINING CONFIGURATION")
print("=" * 60)

print(f"\nModel                  : {MODEL_NAME}")
print(f"Training examples     : {len(train_dataset)}")
print(f"Validation examples   : {len(val_dataset)}")

print(f"\nBatch size             : {BATCH_SIZE}")

print(
    f"Gradient accumulation : "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch size  : "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(f"Learning rate          : {LEARNING_RATE}")
print(f"Max training steps     : {MAX_STEPS}")


# ============================================================
# CHECKPOINT INFORMATION
# ============================================================

if latest_checkpoint:

    checkpoint_step = int(
        os.path.basename(latest_checkpoint).split("-")[1]
    )

    remaining_steps = MAX_STEPS - checkpoint_step

    print("\n" + "=" * 60)
    print("        CHECKPOINT FOUND")
    print("=" * 60)

    print(f"\nLatest checkpoint : {latest_checkpoint}")
    print(f"Completed steps   : {checkpoint_step}")
    print(f"Remaining steps   : {remaining_steps}")

    print(
        f"\nTraining will continue from "
        f"step {checkpoint_step}."
    )

else:

    print("\nNo checkpoint found.")
    print("Training will start from scratch.")


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("        STARTING QLORA TRAINING")
print("=" * 60)

if latest_checkpoint:

    trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    trainer.train()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("Saving final QLoRA adapter...")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("        QWEN 2.5 - 3B TRAINING COMPLETED")
print("=" * 60)

print("\nModel:")
print(MODEL_NAME)

print("\nMethod:")
print("QLoRA + 4-bit NF4 + LoRA")

print("\nDataset:")
print("360,000 training examples")
print("5,000 validation examples")

print("\nTraining:")
print("Effective batch size : 16")
print("Total training steps : 22,500")

print("\nOutput:")
print(OUTPUT_DIR)

print("\nFinal Qwen 3B adapter saved successfully!")

print("\nREADY FOR TEST EVALUATION!")

print("\n" + "=" * 60)
print("        QWEN 3B TRAINING FINISHED")
print("=" * 60)