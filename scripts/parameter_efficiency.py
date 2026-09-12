import torch
import os

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from safetensors.torch import load_file


MODELS = {
    "Qwen2.5-1.5B": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "results/qwen2.5-1.5b-qlora"
    ),

    "Qwen2.5-3B": (
        "Qwen/Qwen2.5-3B-Instruct",
        "results/qwen2.5-3b-qlora"
    ),

    "SmolLM2-1.7B": (
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "results/smollm2-1.7b-qlora"
    ),

    "Bharat-Tiny-LLM-v3": (
        "eulogik/Bharat-Tiny-LLM-v3",
        "results/bharat"
    ),
}


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
# PROCESS MODELS
# ============================================================

for name, (base_model, adapter_path) in MODELS.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=bnb_config,
    )

    print("Loading LoRA adapter...")

    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )

    # --------------------------------------------------------
    # TOTAL PARAMETERS
    # --------------------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    # --------------------------------------------------------
    # ACTUAL SAVED LORA PARAMETERS
    # --------------------------------------------------------

    adapter_file = os.path.join(
        adapter_path,
        "adapter_model.safetensors"
    )

    adapter_state = load_file(
        adapter_file,
        device="cpu"
    )

    lora_params = sum(
        tensor.numel()
        for tensor in adapter_state.values()
    )

    # --------------------------------------------------------
    # PERCENTAGE
    # --------------------------------------------------------

    percentage = (
        lora_params / total_params
    ) * 100

    print()
    print(f"Total parameters     : {total_params:,}")
    print(f"LoRA parameters      : {lora_params:,}")
    print(f"Trainable percentage : {percentage:.4f}%")

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("PARAMETER EFFICIENCY ANALYSIS COMPLETED")
print("=" * 70)