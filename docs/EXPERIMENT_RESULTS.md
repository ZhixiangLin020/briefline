# Experiment Results

Briefline evaluates an end-to-end news intelligence pipeline built around graph-based semantic data selection, multi-task fine-tuning, agentic verification, and optimized inference. Semantic data selection reduced the training corpus by **86.83%**. The validation-selected model improved held-out multi-task performance by **10.56%**, while the agentic verification workflow reduced the **RAGAS-derived hallucination score by 41.2%**. vLLM also delivered a measured **over 3× end-to-end inference speedup**.

## Results at a Glance

| Evaluation | Reference | Briefline | Change |
|---|---:|---:|---:|
| Training corpus size | 547,036 original | 72,026 selected | **−86.83%** |
| Test combined model score | 0.7293 base model | 0.8063 fine-tuned | **+10.56%** |
| RAGAS faithfulness | 0.9166 before verification | 0.9510 after verification | **+3.44 points** |
| RAGAS-derived hallucination score | 0.0834 before verification | 0.0490 after verification | **−41.2%** |
| End-to-end inference | Hugging Face Transformers | vLLM | **>3× faster** |

The corpus-size result measures semantic data selection before training. The combined model score compares the base and fine-tuned models on held-out multi-task test sets. The RAGAS results compare generated content before and after agentic verification, while the inference result compares the Hugging Face Transformers and vLLM execution paths.

## Graph-Based Semantic Data Selection

Briefline uses semantic similarity graphs and community detection to identify overlapping examples and select representative training data from CNN/DailyMail and KPTime.

| Dataset | Original Training Set | Selected Training Set | Retention Rate | Corpus Reduction |
|---|---:|---:|---:|---:|
| CNN/DailyMail | 287,113 | 37,739 | 13.14% | 86.86% |
| KPTime | 259,923 | 34,287 | 13.19% | 86.81% |
| **Combined** | **547,036** | **72,026** | **13.17%** | **86.83%** |

The process reduced the combined training corpus from **547,036 to 72,026 examples**, retaining **13.17%** of the original data. This represents an **86.83% reduction**, equivalent to approximately **7.60× corpus compaction**.

## Multi-Task Model Performance

Briefline adapts `Qwen/Qwen2.5-3B-Instruct` with AdaLoRA using selected examples from two long-form news datasets:

- **CNN/DailyMail:** article summarization;
- **KPTime:** topic classification and keyphrase generation.

The full experiment used **37,739 CNN/DailyMail** and **34,287 KPTime** training examples. Evaluation used 1,000 validation and 1,000 held-out test examples from each dataset.

The combined score equally weights CNN/DailyMail summarization performance and KPTime category/keyphrase performance using RoBERTa-based MoverScore. Checkpoint selection was based only on validation performance; the test split was reserved for final reporting.

## Parameter-Efficient Adaptation

AdaLoRA made **239.5 million parameters** trainable, representing **7.20% of the full model**. This allowed Briefline to specialize Qwen2.5-3B for multiple long-text news tasks without full-model fine-tuning.

## Agentic Verification and RAGAS Faithfulness

The multi-model verification workflow retrieves supporting evidence, uses an LLM judge to evaluate generated content, and selectively corrects flagged outputs. The workflow was evaluated across **2,492 valid comparisons**.

RAGAS faithfulness increased from **0.9166 to 0.9510**, indicating that the verified content was more strongly grounded in the retrieved source articles.

Briefline also reports the complementary metric:

```text
RAGAS-derived hallucination score = 1 − faithfulness
```

The derived score decreased from **0.0834 to 0.0490**, representing a **41.2% relative reduction**.

## Inference Efficiency

Briefline uses vLLM to optimize model serving for long-text workloads. Compared with the Hugging Face Transformers inference path, vLLM delivered a measured **over 3× end-to-end inference speedup**.

## Evaluation Protocol

The data-selection results compare the original training splits with the selected corpus. Multi-task performance compares the base and fine-tuned models on the same held-out test sets. RAGAS metrics compare content before and after agentic verification, while inference performance compares the Transformers and vLLM execution paths.

Checkpoint selection was based only on validation performance. Held-out test metrics were used only for final reporting.

For complete execution and verification instructions, see the [Model Pipeline Guide](MODEL_PIPELINE.md), [RAG and Frontend Guide](RAG_FRONTEND_INTEGRATION.md), and [Verification and Smoke Tests](VERIFICATION.md).
