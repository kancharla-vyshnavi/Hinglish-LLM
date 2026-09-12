# Related Work

## 1. Hindi and Indic Language Modeling

Research on Indian-language NLP has explored pretrained language models
specifically designed for Indic languages.

Models such as IndicBERT and MuRIL demonstrated the usefulness of
language-specific and multilingual pretraining for Indian-language
understanding tasks.

These approaches established strong pretrained representations for
Indic languages, but Hindi-English code-mixed text introduces additional
challenges because Hindi may be written in Roman script and combined
with English within the same utterance.

---

## 2. Hindi-English Code-Mixed NLP

Hindi-English code-mixed text has been studied across tasks such as
sentiment analysis, language identification, question answering, and
text generation.

Code-mixed NLP differs from conventional monolingual NLP because a
single sentence may contain words from multiple languages, transliterated
Hindi, spelling variations, and English insertions.

These characteristics create challenges for tokenization, language
identification, language modeling, and generation quality.

---

## 3. Hinglish Language Models

Previous work has also explored Hindi/Hinglish language models trained
or adapted specifically for code-mixed text.

One relevant direction is training a Hindi/Hinglish transformer language
model from scratch using a large code-mixed corpus.

The publicly available Hindi/Hinglish Transformer LLM project used a
custom tokenizer and trained a transformer language model from scratch
on a large Hindi/Hinglish token corpus.

This approach differs fundamentally from the present study.

The existing approach focuses on building a language model from scratch,
whereas this work investigates parameter-efficient adaptation of
multiple pretrained causal language models using QLoRA.

---

## 4. Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) methods aim to adapt pretrained
models without updating all model parameters.

Low-Rank Adaptation (LoRA) introduces trainable low-rank matrices into
selected model layers while keeping the original model weights frozen.

QLoRA extends this approach by combining LoRA with low-bit
quantization, enabling adaptation of relatively large language models
with substantially reduced memory requirements.

In this work, QLoRA is used as the common adaptation method across the
evaluated models.

QLoRA itself is not treated as the primary novelty of the study.

---

## 5. Research Gap

Existing research demonstrates the effectiveness of pretrained
language models, multilingual models, code-mixed NLP systems, and
parameter-efficient adaptation techniques.

However, there is limited controlled comparison of multiple
pretrained causal language models specifically adapted for
Hindi-English code-mixed generation under constrained computational
resources.

In particular, the interaction between:

- model scale,
- pretrained model family,
- Hinglish adaptation,
- lexical diversity,
- repetition,
- language mixing,
- generation quality, and
- computational efficiency

requires further systematic evaluation.

---

## 6. Positioning of the Present Work

The present study addresses this gap through a controlled comparative
evaluation of four pretrained causal language models:

1. Qwen2.5-1.5B-Instruct
2. Qwen2.5-3B-Instruct
3. SmolLM2-1.7B-Instruct
4. Bharat-Tiny-LLM-v3

All models are adapted using the same primary QLoRA training protocol
and evaluated using a common held-out test set.

The study combines automatic generation metrics, code-mixing measures,
statistical analysis, error analysis, efficiency measurements, and
LLM-as-a-judge evaluation.

This design enables analysis of both generation quality and the
computational trade-offs associated with adapting different pretrained
causal language models for Hinglish text generation.

---

## 7. Difference from Training-from-Scratch Approaches

A key distinction between this study and Hindi/Hinglish language-model
training-from-scratch approaches is the adaptation strategy.

Training from scratch requires learning the model's language
representations and parameters from the available corpus.

In contrast, the present work starts from pretrained causal language
models and adapts them to Hinglish using parameter-efficient QLoRA.

Therefore, the research question is not whether a new language model
architecture can be created, but whether existing pretrained causal
models can be efficiently adapted for Hindi-English code-mixed
generation under constrained computational resources.

---

## 8. Summary

The related literature motivates three important observations:

1. Hindi and other Indic languages benefit from specialized or
   multilingual pretrained representations.

2. Hindi-English code-mixed text introduces linguistic challenges that
   are not fully captured by standard monolingual evaluation.

3. Parameter-efficient adaptation provides a practical approach for
   adapting pretrained causal language models when computational
   resources are limited.

Building on these observations, this work provides a controlled
multi-model evaluation focused specifically on Hinglish generation.