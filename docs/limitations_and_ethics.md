# Limitations, Ethics, and Bias

## 1. Limitations

### 1.1 Limited Computational Resources

The experiments were conducted under constrained computational
resources, including a 6 GB VRAM laptop GPU.

As a result, parameter-efficient QLoRA adaptation was used instead of
full-parameter fine-tuning for the main multi-model comparison.

Therefore, the results should not be interpreted as a direct comparison
with unrestricted full fine-tuning.

---

### 1.2 Limited Model Scale

The evaluated models are relatively small causal language models
compared with large-scale commercial and research LLMs.

The conclusions therefore primarily apply to small and medium-sized
models that can be adapted under constrained computational conditions.

---

### 1.3 Dataset Composition

The final corpus combines multiple Hinglish-related datasets with
different sources, collection procedures, domains, and linguistic
characteristics.

Although source-aware sampling was used, differences between datasets
may influence model behavior.

The dataset should therefore not be considered a perfectly balanced
representation of all Hindi-English code-mixed communication.

---

### 1.4 Automatic Code-Mixing Metrics

Language Mixing Ratio and CMI are used as descriptive automatic measures.

Automatic lexical classification cannot perfectly distinguish Hindi,
English, transliterated Hindi, named entities, abbreviations, and other
language forms.

Consequently, these metrics should not be interpreted as definitive
linguistic annotations.

---

### 1.5 LLM-as-a-Judge Coverage

The planned LLM-as-a-judge evaluation contained 400 selected
model-output evaluations.

Due to inference-provider credit limitations, 244 valid evaluations
were obtained:

- Bharat-Tiny-LLM-v3: 60
- Qwen2.5-1.5B: 62
- Qwen2.5-3B: 61
- SmolLM2-1.7B: 61

Therefore, the LLM-as-a-judge results should be considered partial
qualitative evidence rather than a complete human evaluation.

No unavailable scores were fabricated.

---

### 1.6 Human Evaluation

The study uses an LLM-as-a-judge evaluation rather than a large-scale
multi-rater human evaluation.

Although this provides scalable qualitative assessment, LLM-based
judgment may introduce evaluator-specific preferences and biases.

Future work should include multiple independent human raters with
inter-rater agreement analysis.

---

### 1.7 Generation Evaluation

Generation quality is multi-dimensional.

Metrics such as perplexity, Distinct-1, Distinct-2, repetition rate, and
code-mixing measures capture different properties of generated text but
do not fully represent semantic correctness, factuality, helpfulness,
or cultural appropriateness.

Therefore, no single metric is treated as a complete measure of
generation quality.

---

### 1.8 Historical Training Measurements

Complete historical peak GPU-memory measurements and training-runtime
measurements were not recorded consistently for every completed
training run.

These values are therefore not reported as measured experimental
results.

Instead, the study reports efficiency quantities that were reliably
available from the saved experiment artifacts, such as trainable
parameter counts and generation throughput.

---

## 2. Ethical Considerations

### 2.1 Sensitive Language Data

Hinglish text may contain personal information, offensive language,
slang, stereotypes, or culturally sensitive expressions.

Dataset use should therefore follow the licensing and usage conditions
of the original datasets.

The project does not intentionally attempt to infer sensitive personal
attributes from users.

---

### 2.2 Bias and Representation

Hindi-English code-mixed language is highly diverse.

Differences in region, age, education, social context, spelling,
Romanization conventions, and English usage can result in substantial
variation in Hinglish.

Because the training datasets do not necessarily represent all
Hinglish-speaking populations equally, the adapted models may reproduce
or amplify biases present in the source data.

---

### 2.3 Romanization Variation

Hinglish is commonly written using multiple Romanization styles.

For example, the same Hindi expression can be written using different
spellings.

Such variation may affect tokenization, language identification,
perplexity, and generation quality.

The study therefore treats Romanization variation as an important
linguistic characteristic rather than assuming a single standardized
Hinglish spelling system.

---

### 2.4 Code-Mixing Bias

The amount and type of English insertion can vary considerably across
speakers and contexts.

A higher language-mixing score should not automatically be interpreted
as better Hinglish.

The goal is to evaluate whether generated text exhibits appropriate
and coherent code-mixing rather than maximizing English usage.

---

### 2.5 Model-Generated Content

Generated outputs may contain factual errors, inappropriate statements,
or hallucinated information.

The models should therefore not be treated as authoritative sources of
medical, legal, financial, or other high-stakes information.

---

## 3. Responsible Interpretation

The results should be interpreted as an empirical comparison of
parameter-efficient adaptation strategies for Hinglish text generation.

The findings do not establish that one model is universally superior for
all Hindi-English code-mixed applications.

Instead, the experiments demonstrate how different pretrained causal
language models behave under a controlled adaptation and evaluation
setting.

Future research should expand the evaluation to larger models,
additional Hinglish datasets, multiple human raters, broader domains,
and more comprehensive factuality and safety evaluation.