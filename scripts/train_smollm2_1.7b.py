import os
import glob
import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


# ============================================================
# SMOLLM2 - 1.7B QLORA TRAINING
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

TRAIN_FILE = "data/processed/train.jsonl"
VAL_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "results/smollm2-1.7b-qlora"

MAX_LENGTH = 64


# ============================================================
# SAME SETTINGS FOR ALL 4 MODELS
# ============================================================

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

LEARNING_RATE = 2e-4

MAX_STEPS = 22500


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("        SMOLLM2 - 1.7B QLORA TRAINING")
print("=" * 70)


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU not detected. Please check your PyTorch CUDA setup."
    )

print("\nGPU detected:")
print(torch.cuda.get_device_name(0))


# Enable TF32 for faster NVIDIA GPU computation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\nLoading training dataset...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_FILE,
    split="train"
)

validation_dataset = load_dataset(
    "json",
    data_files=VAL_FILE,
    split="train"
)

print(f"Training examples   : {len(train_dataset)}")
print(f"Validation examples: {len(validation_dataset)}")


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
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )


print("\nTokenizing training dataset...")

tokenized_train = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

print("Tokenizing validation dataset...")

tokenized_validation = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=validation_dataset.column_names
)


# ============================================================
# 4-BIT NF4 QLORA CONFIG
# ============================================================

print("\nConfiguring 4-bit NF4 QLoRA...")

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16
)


# ============================================================
# LOAD SMOLLM2 1.7B
# ============================================================

print("\nLoading SmolLM2 1.7B model...")

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    dtype=torch.float16,

    trust_remote_code=True
)


# ============================================================
# PREPARE MODEL FOR QLORA
# ============================================================

print("\nPreparing model for QLoRA...")

model = prepare_model_for_kbit_training(
    model
)


# ============================================================
# LORA CONFIGURATION
# ============================================================

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)


# ============================================================
# APPLY LORA
# ============================================================

model = get_peft_model(

    model,

    lora_config
)


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

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

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # SAME FOR ALL 4 MODELS
    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    lr_scheduler_type="cosine",

    fp16=True,

    tf32=True,

    gradient_checkpointing=False,

    optim="paged_adamw_8bit",

    max_steps=MAX_STEPS,

    logging_steps=100,

    eval_steps=2000,

    save_steps=5000,

    eval_strategy="steps",

    save_strategy="steps",

    save_total_limit=2,

    # Windows safe
    dataloader_num_workers=0,

    remove_unused_columns=False,

    seed=42,

    data_seed=42,

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    data_collator=data_collator
)


# ============================================================
# CHECK EXISTING CHECKPOINT
# ============================================================

checkpoints = glob.glob(

    os.path.join(
        OUTPUT_DIR,
        "checkpoint-*"
    )
)

latest_checkpoint = None


if checkpoints:

    checkpoints.sort(

        key=lambda x: int(
            x.split("-")[-1]
        )
    )

    latest_checkpoint = checkpoints[-1]

    print("\nExisting checkpoint found:")

    print(latest_checkpoint)

    print(
        "\nTraining will resume "
        "from this checkpoint."
    )


# ============================================================
# START TRAINING
# ============================================================

print("\n" + "=" * 70)

print("              STARTING QLORA TRAINING")

print("=" * 70)


print(f"\nModel                 : {MODEL_NAME}")

print(
    f"Train examples        : "
    f"{len(train_dataset)}"
)

print(
    f"Validation examples   : "
    f"{len(validation_dataset)}"
)

print(
    f"Batch size             : "
    f"{BATCH_SIZE}"
)

print(
    f"Gradient accumulation  : "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Effective batch size   : "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Total training steps   : "
    f"{MAX_STEPS}"
)

print(
    f"Learning rate          : "
    f"{LEARNING_RATE}"
)


if latest_checkpoint:

    trainer.train(
        resume_from_checkpoint=latest_checkpoint
    )

else:

    trainer.train()


# ============================================================
# SAVE FINAL ADAPTER
# ============================================================

print("\nSaving final QLoRA adapter...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("          QLORA TRAINING COMPLETED")

print("=" * 70)


print("\nModel:")
print(MODEL_NAME)


print("\nMethod:")
print("QLoRA + 4-bit NF4 + LoRA")


print("\nDataset:")
print("360,000 training examples")
print("5,000 validation examples")


print("\nTraining:")

print(
    f"Effective batch size : "
    f"{BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
)

print(
    f"Total training steps : "
    f"{MAX_STEPS}"
)


print("\nAdapter:")
print(OUTPUT_DIR)


print("\n🎯 READY FOR TEST EVALUATION!")