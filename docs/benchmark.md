# Benchmark and Evaluation Protocol

## 1. Dataset

The study evaluates pretrained causal language models adapted for
Hindi-English code-mixed (Hinglish) text generation.

The final corpus was constructed by combining four Hinglish-related
datasets:

1. HINMIX
2. COMI-LINGUA-TN
3. HinGE
4. Hinglish Everyday Conversations

After preprocessing and duplicate removal, the combined corpus contained
2,435,094 valid unique text examples.

A controlled subset of 370,000 examples was selected for model adaptation.

The final split consisted of:

- Training set: 360,000 examples
- Validation set: 5,000 examples
- Test set: 5,000 examples

A fixed random seed of 42 was used for dataset sampling and splitting.

---

## 2. Evaluation Objective

The evaluation focuses on the ability of adapted causal language models
to generate fluent, diverse, coherent, and appropriately code-mixed
Hindi-English text.

The evaluation does not treat text generation as a conventional
single-label classification problem. Therefore, accuracy alone is not
an appropriate primary evaluation measure.

Instead, multiple complementary metrics are used.

---

## 3. Automatic Evaluation Metrics

### 3.1 Perplexity

Perplexity measures how well a language model predicts the evaluation
text.

Lower perplexity indicates better language-modeling performance.

Perplexity is calculated on the held-out test set and is used as one of
the primary quantitative measures.

---

### 3.2 Distinct-1

Distinct-1 measures unigram diversity in generated text.

It is calculated as:

Distinct-1 = Number of unique unigrams / Total number of unigrams

Higher values indicate greater lexical diversity.

---

### 3.3 Distinct-2

Distinct-2 measures bigram diversity.

It is calculated as:

Distinct-2 = Number of unique bigrams / Total number of bigrams

Higher values indicate greater diversity in generated sequences.

---

### 3.4 Repetition Rate

Repetition rate measures the proportion of repeated textual content in
generated outputs.

Lower repetition indicates that the model is less likely to produce
redundant or repetitive generations.

---

### 3.5 Language Mixing Ratio

Language Mixing Ratio is used as a descriptive automatic measure of the
extent of English lexical insertion within generated Hinglish text.

Because Hinglish is a code-mixed language variety, this metric provides
an additional view of the model's language-mixing behavior.

The metric should not be interpreted as a universal measure of
code-mixing quality.

---

### 3.6 Code-Mixing Index (CMI)

Code-Mixing Index is reported as an additional descriptive measure of
language mixing.

CMI is particularly relevant because the target language variety
contains Hindi-English lexical mixing.

The implementation used in this study relies on automatic lexical
classification and should therefore be interpreted as an approximate
automatic measure rather than a linguistically perfect gold-standard
annotation.

---

### 3.7 Unique Sentence Ratio

Unique Sentence Ratio measures the proportion of generated sentences
that are unique.

Higher values indicate lower duplication across generated outputs.

---

### 3.8 Average Generation Length

Average generation length is reported to provide additional information
about the characteristics of generated responses.

It is not interpreted as a direct quality score.

---

## 4. LLM-as-a-Judge Evaluation

Automatic metrics alone may not fully capture perceived generation
quality.

Therefore, an additional LLM-as-a-judge evaluation was conducted.

The evaluation considered five dimensions:

- Fluency
- Coherence
- Relevance
- Code-mixing quality
- Overall quality

Each dimension was scored on a 1–5 scale.

The judge model was:

`openai/gpt-oss-20b`

through the Groq inference provider.

The model identity was not provided to the judge during scoring in order
to reduce model-name-based bias.

Due to inference-provider credit limitations, 244 valid evaluations were
obtained from the planned evaluation subset.

The valid evaluation counts were:

- Bharat-Tiny-LLM-v3: 60
- Qwen2.5-1.5B: 62
- Qwen2.5-3B: 61
- SmolLM2-1.7B: 61

No scores were fabricated for unavailable evaluations.

---

## 5. Statistical Analysis

Bootstrap confidence intervals were used to examine differences in
generation-level metrics between the Qwen2.5-3B and SmolLM2-1.7B models.

The analysis used 500 generated samples per model and 1,000 bootstrap
iterations with 95% confidence intervals.

This analysis provides an estimate of uncertainty around the observed
metric differences.

---

## 6. Error Analysis

Automatic error analysis was performed on generated outputs.

The analysis considered categories including:

- Token corruption
- Likely truncation or unfinished output
- Word or phrase repetition
- Emoji repetition
- Style deviation

The purpose of the error analysis is to identify qualitative failure
patterns that are not fully captured by aggregate automatic metrics.

---

## 7. Efficiency Evaluation

Model efficiency is evaluated using:

- Trainable parameter count
- Percentage of trainable parameters
- Generation throughput
- Evaluation runtime

Training and inference settings were kept as consistent as possible
across models.

Because historical peak GPU memory and complete training-runtime
measurements were not recorded for every run, these values are not
reported as measured experimental results.

---

## 8. Model Selection

The final model selection is based on multiple complementary evaluation
criteria rather than a single metric.

Qwen2.5-3B was selected as the overall best-performing model primarily
because it achieved the lowest test-set perplexity among the evaluated
models while also maintaining strong diversity and low repetition.

Other models showed strengths on individual metrics.

For example, SmolLM2-1.7B achieved the highest Distinct-2 score, while
Bharat-Tiny-LLM-v3 achieved the lowest repetition rate.

This demonstrates the importance of multi-dimensional evaluation.

---

## 9. Reproducibility

The experiments use a fixed random seed of 42 for dataset sampling,
splitting, and training-related data operations.

The final train, validation, and test sizes are fixed.

QLoRA configuration, model configurations, preprocessing scripts,
evaluation scripts, and generated evaluation artifacts are maintained
within the project repository.

Large datasets, virtual environments, and model checkpoints are excluded
from the Git repository because of their storage requirements.