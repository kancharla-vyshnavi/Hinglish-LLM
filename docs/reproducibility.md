# Reproducibility

## 1. Experimental Environment

The experiments were conducted using:

- Operating System: Windows
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- GPU VRAM: 6 GB
- CPU: Intel Core i7-14650HX
- System RAM: 16 GB
- Python: 3.11

The primary development environment used VS Code with a Python virtual
environment.

---

## 2. Software Environment

The main software components used in the experiments include:

- PyTorch
- Transformers
- PEFT
- Accelerate
- bitsandbytes
- Hugging Face Datasets

The experiments use 4-bit quantization with NF4 and double
quantization for QLoRA-based adaptation.

---

## 3. Dataset Construction

Four Hinglish-related datasets were combined:

- HINMIX
- COMI-LINGUA-TN
- HinGE
- Hinglish Everyday Conversations

After preprocessing and duplicate removal:

Total valid unique examples: 2,435,094

A fixed subset of 370,000 examples was selected.

The final dataset contains:

- Training: 360,000
- Validation: 5,000
- Test: 5,000

Random seed:

`42`

---

## 4. Models

The following pretrained causal language models were evaluated:

### Qwen2.5-1.5B-Instruct

`Qwen/Qwen2.5-1.5B-Instruct`

### Qwen2.5-3B-Instruct

`Qwen/Qwen2.5-3B-Instruct`

### SmolLM2-1.7B-Instruct

`HuggingFaceTB/SmolLM2-1.7B-Instruct`

### Bharat-Tiny-LLM-v3

`eulogik/Bharat-Tiny-LLM-v3`

Bharat-Tiny-LLM-v3 is a Hindi/Hinglish-specialized pretrained model.
Therefore, it is interpreted as a specialized pretrained baseline that
receives further QLoRA adaptation rather than as a general-purpose
pretrained model.

---

## 5. QLoRA Configuration

The main experiments use the following common configuration:

- Quantization: 4-bit
- Quantization type: NF4
- Double quantization: Enabled
- Compute dtype: FP16
- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Optimizer: paged AdamW 8-bit
- Learning rate: 2e-4
- Weight decay: 0.01
- Scheduler: cosine
- Maximum sequence length: 64
- Per-device batch size: 4
- Gradient accumulation steps: 4
- Effective batch size: 16
- Maximum training steps: 22,500
- Random seed: 42
- Data seed: 42
- Gradient checkpointing: Disabled
- DataLoader workers: 0

For the primary Qwen and SmolLM2 experiments, LoRA adapters target:

- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj
- down_proj

The saved Bharat-Tiny-LLM-v3 adapter uses:

- q_proj
- k_proj
- v_proj
- o_proj

This difference is preserved from the actual saved experiment artifact.

---

## 6. Evaluation

All models were evaluated using the same held-out test set.

Automatic metrics include:

- Perplexity
- Distinct-1
- Distinct-2
- Repetition Rate
- Language Mixing Ratio
- Code-Mixing Index
- Unique Sentence Ratio
- Average Generation Length

Additional evaluation includes:

- Statistical bootstrap analysis
- Automatic error analysis
- Efficiency analysis
- LLM-as-a-judge evaluation

---

## 7. Generation Evaluation

Generation samples were collected using the same evaluation procedure
for the compared models.

Generation throughput was measured from the saved evaluation results.

The reported throughput is based on 100 generated samples per model.

---

## 8. Statistical Evaluation

The statistical analysis compares Qwen2.5-3B and SmolLM2-1.7B using
generation-level measurements.

The analysis uses:

- 500 samples per model
- 1,000 bootstrap iterations
- 95% confidence intervals

---

## 9. LLM-as-a-Judge

The qualitative evaluation uses:

`openai/gpt-oss-20b`

through the Groq inference provider.

The judge evaluates:

- Fluency
- Coherence
- Relevance
- Code-mixing quality
- Overall quality

Each dimension is scored from 1 to 5.

Model identity is not supplied to the judge during evaluation.

A total of 244 valid evaluations were obtained because inference
provider credits were exhausted before the planned evaluation set was
completed.

---

## 10. Randomness Control

The following seeds were fixed where applicable:

- Dataset sampling seed: 42
- Dataset splitting seed: 42
- Training seed: 42
- Data seed: 42

The fixed seed helps make dataset construction and training procedures
more reproducible.

---

## 11. Repository Structure

The project is organized approximately as follows:

```text
HindiHinglish-LLM/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── scripts/
│   ├── preprocessing scripts
│   ├── dataset preparation
│   ├── training scripts
│   ├── evaluation scripts
│   ├── statistical analysis
│   └── error analysis
│
├── results/
│   └── final_evaluation/
│
├── docs/
│
├── requirements.txt
│
└── README.md