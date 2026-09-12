# Efficient Adaptation and Comparative Evaluation of Pretrained Causal Language Models for Hindi-English Code-Mixed Text Generation

## Overview

Hindi-English code-mixed language, commonly referred to as Hinglish,
combines Hindi and English within the same sentence and frequently uses
Romanized Hindi.

This project investigates how efficiently pretrained causal language
models can be adapted for Hinglish text generation using
Parameter-Efficient Fine-Tuning (PEFT), specifically QLoRA.

The study performs a controlled comparison of four pretrained causal
language models under the same primary adaptation and evaluation
framework.

---

## Research Questions

The study investigates the following questions:

1. Which pretrained causal language model adapts most effectively to
   Hindi-English code-mixed text using QLoRA?

2. How does model scale affect Hinglish generation quality?

3. Can parameter-efficient adaptation achieve competitive generation
   quality under constrained computational resources?

4. Which Hinglish characteristics, including Romanization, spelling
   variation, English insertion, and code-switching, affect generation
   quality?

---

## Models

| Model | Size | Role |
|---|---:|---|
| Qwen2.5-1.5B-Instruct | 1.5B | General pretrained causal model |
| Qwen2.5-3B-Instruct | 3B | Larger general pretrained causal model |
| SmolLM2-1.7B-Instruct | 1.7B | Compact pretrained causal model |
| Bharat-Tiny-LLM-v3 | 1.7B | Hindi/Hinglish-specialized pretrained model |

Bharat-Tiny-LLM-v3 is treated as a specialized pretrained baseline
because it was already designed for Hindi/Hinglish.

---

## Dataset

Four Hinglish-related datasets were combined:

- HINMIX
- COMI-LINGUA-TN
- HinGE
- Hinglish Everyday Conversations

After preprocessing and duplicate removal:

```text
Valid unique examples: 2,435,094