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
# 1. CONFIGURATION
# ============================================================

MODEL_NAME = "eulogik/Bharat-Tiny-LLM-v3"

TRAIN_FILE = "data/processed/train.jsonl"
VALIDATION_FILE = "data/processed/validation.jsonl"

OUTPUT_DIR = "results/bharat"

# IMPORTANT:
# This checkpoint must exist before running.
CHECKPOINT_PATH = "results/bharat/checkpoint-16000"

MAX_LENGTH = 512


# ============================================================
# 2. CHECK DEVICE
# ============================================================

print("=" * 70)
print("        BHARAT TINY LLM - RESUME QLoRA TRAINING")
print("=" * 70)

print("\nChecking device...")

if torch.cuda.is_available():

    print("CUDA available ✅")
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

else:

    print("CUDA not available ❌")
    print("Training will be extremely slow on CPU.")


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\nChecking dataset files...")

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"Training file not found:\n{TRAIN_FILE}"
    )

if not os.path.exists(VALIDATION_FILE):
    raise FileNotFoundError(
        f"Validation file not found:\n{VALIDATION_FILE}"
    )

# Check checkpoint
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT_PATH}\n\n"
        "Run: dir results\\bharat"
    )

print("Training file found ✅")
print("Validation file found ✅")
print("Checkpoint found ✅")

print("\nCheckpoint:")
print(CHECKPOINT_PATH)


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("Loading dataset...")
print("=" * 70)

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

print("\nDataset loaded successfully ✅")

print("Training samples   :", len(train_dataset))
print("Validation samples :", len(validation_dataset))

print("\nDataset columns:")
print(train_dataset.column_names)


# ============================================================
# 5. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 70)
print("Loading tokenizer...")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded successfully ✅")
print("Vocabulary size:", len(tokenizer))


# ============================================================
# 6. QUANTIZATION CONFIG
# ============================================================

print("\n" + "=" * 70)
print("Configuring 4-bit quantization...")
print("=" * 70)

if torch.cuda.is_available():

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("4-bit NF4 quantization enabled ✅")

else:

    bnb_config = None

    print("Quantization disabled because CUDA is unavailable.")


# ============================================================
# 7. LOAD BHARAT MODEL
# ============================================================

print("\n" + "=" * 70)
print("Loading Bharat model...")
print("=" * 70)

if torch.cuda.is_available():

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

else:

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

print("\nBharat model loaded successfully ✅")


# ============================================================
# 8. PREPARE MODEL FOR QLoRA
# ============================================================

if torch.cuda.is_available():

    print("\nPreparing model for k-bit training...")

    model = prepare_model_for_kbit_training(model)

    print("Model prepared for QLoRA ✅")


# ============================================================
# 9. LoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("Configuring LoRA...")
print("=" * 70)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",
)


# ============================================================
# 10. ATTACH LoRA
# ============================================================

model = get_peft_model(
    model,
    lora_config,
)

print("\nLoRA attached successfully ✅")

model.print_trainable_parameters()


# ============================================================
# 11. DATA FORMATTING
# ============================================================

print("\n" + "=" * 70)
print("Preparing dataset...")
print("=" * 70)


def format_example(example):

    # Case 1: text
    if "text" in example and example["text"]:
        return str(example["text"])

    # Case 2: instruction + response
    if "instruction" in example and "response" in example:

        return (
            f"Instruction:\n"
            f"{example['instruction']}\n\n"
            f"Response:\n"
            f"{example['response']}"
        )

    # Case 3: prompt + response
    if "prompt" in example and "response" in example:

        return (
            f"Prompt:\n"
            f"{example['prompt']}\n\n"
            f"Response:\n"
            f"{example['response']}"
        )

    # Case 4: input + output
    if "input" in example and "output" in example:

        return (
            f"Input:\n"
            f"{example['input']}\n\n"
            f"Output:\n"
            f"{example['output']}"
        )

    # Case 5: question + answer
    if "question" in example and "answer" in example:

        return (
            f"Question:\n"
            f"{example['question']}\n\n"
            f"Answer:\n"
            f"{example['answer']}"
        )

    raise ValueError(
        "Unknown dataset format. Available columns: "
        + str(list(example.keys()))
    )


# ============================================================
# 12. TOKENIZATION
# ============================================================

print("\nTokenizing dataset...")


def tokenize_function(example):

    text = format_example(example)

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    return tokenized


tokenized_train = train_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=train_dataset.column_names,
    desc="Tokenizing train",
)

tokenized_validation = validation_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=validation_dataset.column_names,
    desc="Tokenizing validation",
)

print("\nTokenization completed ✅")


# ============================================================
# 13. DATA COLLATOR
# ============================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)


# ============================================================
# 14. TRAINING ARGUMENTS
# ============================================================

print("\n" + "=" * 70)
print("Creating training configuration...")
print("=" * 70)

training_args = TrainingArguments(

    output_dir=OUTPUT_DIR,

    # Continue the original 1 epoch
    num_train_epochs=1,

    per_device_train_batch_size=2,

    per_device_eval_batch_size=2,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    weight_decay=0.01,

    logging_steps=50,

    # Keep evaluation
    eval_strategy="steps",
    eval_steps=500,

    # Keep checkpoint saving
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    fp16=torch.cuda.is_available(),

    report_to="none",

    remove_unused_columns=False,

    gradient_checkpointing=True,

    optim=(
        "paged_adamw_8bit"
        if torch.cuda.is_available()
        else "adamw_torch"
    ),
)


# ============================================================
# 15. TRAINER
# ============================================================

print("\nCreating Trainer...")

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_train,

    eval_dataset=tokenized_validation,

    processing_class=tokenizer,

    data_collator=data_collator,
)

print("Trainer created successfully ✅")


# ============================================================
# 16. RESUME TRAINING
# ============================================================

print("\n" + "=" * 70)
print("       RESUMING TRAINING FROM CHECKPOINT")
print("=" * 70)

print("\nCheckpoint:")
print(CHECKPOINT_PATH)

print("\nPrevious progress: approximately 16,000 steps")
print("Remaining progress: approximately 6,500 steps")

print("\nResuming training... 🚀")


trainer.train(
    resume_from_checkpoint=CHECKPOINT_PATH
)


# ============================================================
# 17. SAVE FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("Saving trained model...")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved successfully ✅")

print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# 18. FINAL EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("Running final evaluation...")
print("=" * 70)

evaluation_results = trainer.evaluate()

print("\nEvaluation results:")

for key, value in evaluation_results.items():

    if isinstance(value, float):

        print(f"{key}: {value:.4f}")

    else:

        print(f"{key}: {value}")


# ============================================================
# 19. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("          BHARAT TRAINING COMPLETED ✅")
print("=" * 70)

print("\nModel location:")
print(OUTPUT_DIR)

print("\nDone brooo 🚀🔥")