# Experiment Results

Briefline evaluates a fine-tuned Qwen2.5-3B model across long-form summarization, topic classification, and keyphrase generation. The selected model improved held-out multi-task performance by **10.56%**, while the downstream RAG pipeline reduced its **RAGAS-derived hallucination score by 41.2%**.

## Results at a Glance

| Evaluation | Baseline | Briefline | Change |
|---|---:|---:|---:|
| Test combined model score | 0.7293 | 0.8063 | **+10.56%** |
| RAGAS faithfulness | 0.9166 | 0.9510 | **+3.44 points** |
| RAGAS-derived hallucination score | 0.0834 | 0.0490 | **−41.2%** |

The model score evaluates the fine-tuned model's multi-task outputs. The RAGAS results evaluate whether downstream answers remain supported by retrieved news sources.

## Multi-Task Model Performance

Briefline adapts `Qwen/Qwen2.5-3B-Instruct` with AdaLoRA using two long-form news datasets:

- **CNN/DailyMail:** article summarization;
- **KPTime:** topic classification and keyphrase generation.

The full experiment used **37,739 CNN/DailyMail** and **34,287 KPTime** training examples. Evaluation used 1,000 validation and 1,000 held-out test examples from each dataset.

The combined score equally weights CNN/DailyMail summarization performance and KPTime category/keyphrase performance using RoBERTa-based MoverScore. Checkpoint selection was based only on validation performance; the test split was reserved for final reporting.

## RAGAS Faithfulness

The downstream RAG pipeline was evaluated across **2,492 valid comparisons**. RAGAS faithfulness increased from **0.9166 to 0.9510**, indicating stronger support from the retrieved source articles.

Briefline reports the complementary metric:

```text
RAGAS-derived hallucination score = 1 − faithfulness
```

The derived score decreased from **0.0834 to 0.0490**, a **41.2% relative reduction**.

## Parameter-Efficient Adaptation

AdaLoRA made **239.5 million parameters** trainable, representing **7.20% of the full model**. This allowed Briefline to specialize Qwen2.5-3B for multiple long-text news tasks without full-model fine-tuning.

## Evaluation Protocol

The reported results use the complete experiment datasets and validation-based model selection. Held-out test metrics are used only for final reporting.

For complete execution and verification instructions, see the [Model Pipeline Guide](MODEL_PIPELINE.md), [RAG and Frontend Guide](RAG_FRONTEND_INTEGRATION.md), and [Verification and Smoke Tests](VERIFICATION.md).
