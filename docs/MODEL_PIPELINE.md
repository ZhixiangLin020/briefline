# Model Pipeline Guide

This guide documents the formal Briefline model experiment from data curation through held-out evaluation. The commands in the main path use the complete datasets and do not enable smoke mode.

## Run Map

| Goal | Sections |
|---|---|
| Reproduce the reported experiment | 1–7 |
| Resume interrupted training | 6 |
| Pass the selected adapter to RAG | 8 |
| Validate the pipeline with bounded data | [Verification and Smoke Tests](VERIFICATION.md#3-model-pipeline-smoke-test) |

## 1. Reproduce the Reported Run

A run matches the reported model experiment when all of the following hold:

- the complete prepared CNN/DailyMail and KPTime datasets are used;
- no data command includes `--limit`;
- training and evaluation run without `--smoke-test`;
- the prepared dataset manifests and training manifest pass the recorded provenance checks;
- checkpoint selection uses validation metrics only;
- test metrics remain report-only.

The reported full run used these dataset sizes:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| CNN/DailyMail | 37,739 | 1,000 | 1,000 |
| KPTime | 34,287 | 1,000 | 1,000 |

If upstream data revisions or dependency changes produce different fingerprints, record the new manifests and treat the output as a new experiment rather than an exact rerun.

## 2. Define Paths, Install, and Verify the Runtime

Run from the repository root—the directory containing `pyproject.toml`. Its directory name is not significant. Choose one absolute, writable workspace outside the Git checkout and reuse it throughout the workflow:

```bash
export BRIEFLINE_WORKSPACE="/absolute/path/to/briefline_workspace"
export HF_HOME="$BRIEFLINE_WORKSPACE/hf_cache"
export HF_DATASETS_CACHE="$BRIEFLINE_WORKSPACE/hf_datasets_cache"
export BRIEFLINE_DATA_ROOT="$BRIEFLINE_WORKSPACE/data"
export BRIEFLINE_RUN_ROOT="$BRIEFLINE_WORKSPACE/runs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$BRIEFLINE_DATA_ROOT" "$BRIEFLINE_RUN_ROOT"

python scripts/install_dependencies.py
python -m briefline check-env
python scripts/verify_vllm_flash_attn.py
```

Use `python scripts/install_dependencies.py --with-rag` only when the same environment will also run the RAG backend.

### Colab note

In Google Colab, environment variables set through `!export` do not persist between shell cells. Clone or extract the repository to `/content/briefline`, then use:

```python
%cd /content/briefline
%env BRIEFLINE_WORKSPACE=/content/briefline_workspace
%env HF_HOME=/content/briefline_workspace/hf_cache
%env HF_DATASETS_CACHE=/content/briefline_workspace/hf_datasets_cache
%env BRIEFLINE_DATA_ROOT=/content/briefline_workspace/data
%env BRIEFLINE_RUN_ROOT=/content/briefline_workspace/runs
```

Prefix subsequent shell commands with `!`. If dependency installation requires a runtime restart, repeat `%cd` and the `%env` lines.

## 3. Prepare the Full Datasets

### CNN/DailyMail

```bash
python -m briefline data \
  --dataset cnn_dm \
  --stage all \
  --seed 42 \
  --output-dir "$BRIEFLINE_DATA_ROOT/cnn_dm"
```

### KPTime

```bash
python -m briefline data \
  --dataset kptimes \
  --stage all \
  --seed 42 \
  --task-mode both \
  --output-dir "$BRIEFLINE_DATA_ROOT/kptimes"
```

`--stage all` executes three ordered stages:

1. `select` filters examples and reduces semantic duplication.
2. `prepare` tokenizes examples and writes Hugging Face datasets.
3. `validate` checks schema, shapes, row counts, and metadata.

The prepared outputs are:

```text
$BRIEFLINE_DATA_ROOT/cnn_dm/prepared
$BRIEFLINE_DATA_ROOT/kptimes/prepared
```

Both paths must be extracted directories readable by `datasets.load_from_disk()`, not ZIP archives.

### Important data parameters

| Parameter | Default | Purpose |
|---|---:|---|
| `--stage` | required | Run `select`, `prepare`, `validate`, or `all` |
| `--seed` | `42` | Controls deterministic sampling and selection |
| `--device` | `cuda` | Device used by semantic selection components |
| `--num-proc` | `8` | Multiprocessing workers for data preparation |
| `--batch-size` | `512` | Batch size for data-processing operations |
| `--task-mode` | `both` | KPTime output contract for category and keyphrase tasks |
| `--force-rebuild` | disabled | Rebuild outputs instead of reusing compatible artifacts |
| `--limit` | unset | Optional row cap for bounded verification; omit it for full experiments |

Dataset-specific defaults, including HNSW and Leiden settings, live in `data_processing/config.py`. Preparation logic lives in `data_processing/cnn_dm.py` and `data_processing/kptimes.py`.

### Data artifacts

Each dataset output records:

- `manifest.json` under the prepared dataset;
- `run_metadata.json` under the dataset output root;
- split sizes and fingerprints;
- tokenizer identity and tokenization contract;
- preparation parameters and relevant package versions.

Retain these manifests with the prepared datasets so later runs can verify data provenance.

## 4. Configure Formal Training

Create a local working copy and keep it out of commits:

```bash
cp configs/original_experiment.yaml configs/local_experiment.yaml
```

Shell variables are not expanded inside the YAML loader. Resolve the workspace path first, then write the resulting absolute paths into `configs/local_experiment.yaml`. The example below assumes `BRIEFLINE_WORKSPACE=/absolute/path/to/briefline_workspace`.

The template has two sections:

```yaml
data:
  cnn_dm_dataset: /absolute/path/to/briefline_workspace/data/cnn_dm/prepared
  kptimes_dataset: /absolute/path/to/briefline_workspace/data/kptimes/prepared
  seed: 42
  smoke_test: false

training:
  output_dir: /absolute/path/to/briefline_workspace/runs/full
  best_model_dir: /absolute/path/to/briefline_workspace/runs/full/best_model
  model_name_or_path: Qwen/Qwen2.5-3B-Instruct
  roberta_path: FacebookAI/roberta-large
  resume_from_checkpoint: null
  dry_run: false
```

`roberta_path` is used by validation scoring. Public runs use `FacebookAI/roberta-large`; keep this value fixed across all compared checkpoints.

### Frozen training configuration

Algorithmic settings are centralized in `training/config.py` so formal runs share one recorded configuration.

| Group | Recorded setting |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Epochs | 6 |
| Batch size | 8 |
| Gradient accumulation | 1 |
| Learning rate | `4e-4` |
| Scheduler | Cosine with minimum learning-rate ratio `0.1` |
| Warmup | `0.05` of training |
| Precision | BF16 training/evaluation, TF32 enabled |
| Evaluation/save cadence | `0.05` of total training steps |
| Task sampling | Epoch-dependent CNN/DM:KPTime schedule from 0.7:0.3 to 0.5:0.5 |
| AdaLoRA rank | Initial `128`, target `90` |
| AdaLoRA alpha/dropout | `128` / `0.05` |
| AdaLoRA target modules | `all-linear` |

The recorded run exposed 239,500,800 trainable parameters out of 3,324,884,732 total parameters, or 7.20%.

## 5. Run Full Training

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline train \
  --config configs/local_experiment.yaml
```

Training consumes the prepared datasets produced in Section 3, keeping data curation and model optimization as explicit pipeline stages.

The formal training path is complete when these files exist:

```bash
test -f "$BRIEFLINE_RUN_ROOT/full/run_manifest.json"
test -f "$BRIEFLINE_RUN_ROOT/full/training_result.json"
test -f "$BRIEFLINE_RUN_ROOT/full/best_model/best_k_metrics.json"
```

### Training outputs

| Artifact | Purpose |
|---|---|
| `run_manifest.json` | Data fingerprints, model identity, configuration, and reproduction status |
| `training_result.json` | Final training summary |
| `checkpoint-*` | Trainer and adapter state for recovery/evaluation |
| `best_model/best_k_metrics.json` | Validation-ranked checkpoint candidates consumed by evaluation |
| `best_model/logs/` | Checkpoint-selection history and diagnostics |

Use separate output directories for bounded verification and formal experiments.

## 6. Resume an Interrupted Run

Resume with the original output paths and an explicit checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline train \
  --config configs/local_experiment.yaml \
  --resume-from-checkpoint "$BRIEFLINE_RUN_ROOT/full/checkpoint-N"
```

Replace `checkpoint-N` with the checkpoint directory selected for recovery.

Keep the following together:

- the checkpoint directory and trainer state;
- sampler state;
- the associated `best_model/best_k_metrics.json` history;
- the same prepared dataset directories and manifests;
- the same `output_dir` and `best_model_dir` relationships.

A strict resume uses the complete checkpoint and trainer state. Adapter weights alone remain sufficient for inference.

## 7. Run Full Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline evaluate \
  --cnn-dm-dataset "$BRIEFLINE_DATA_ROOT/cnn_dm/prepared" \
  --kptimes-dataset "$BRIEFLINE_DATA_ROOT/kptimes/prepared" \
  --base-model-path Qwen/Qwen2.5-3B-Instruct \
  --tokenizer-path Qwen/Qwen2.5-3B-Instruct \
  --roberta-path FacebookAI/roberta-large \
  --best-model-dir "$BRIEFLINE_RUN_ROOT/full/best_model" \
  --output-dir "$BRIEFLINE_RUN_ROOT/full_evaluation" \
  --temp-merged-model-dir "$BRIEFLINE_RUN_ROOT/tmp_merged_models"
```

The evaluator:

1. reads checkpoint candidates from `best_model/best_k_metrics.json`;
2. evaluates the base model and candidate adapters under the same decoding contract;
3. temporarily merges each adapter for vLLM inference;
4. computes CNN/DailyMail and KPTime task scores;
5. selects the checkpoint with the highest full-validation combined score;
6. reports held-out test metrics without using them for selection.

The fixed generation defaults include `max_new_tokens=128`, `temperature=0`, `top_p=1`, and `repetition_penalty=1.02`.

### Evaluation outputs

| Artifact | Purpose |
|---|---|
| `evaluation_manifest.json` | Models, datasets, runtime, decoding, and provenance |
| `metrics_long.csv` | Long-form metric records by model, task, and split |
| `summary_vs_base.csv` | Compact candidate comparison and deltas from the base model |
| `best_finetuned_by_full_valid_combo.json` | Validation-selected checkpoint |
| `runtime.csv` | Inference and evaluation timing |
| `predictions_minimal.csv` | Compact predictions for inspection |
| `model_io/*.jsonl` | Model inputs and raw outputs for auditability |

See [Experiment Results](EXPERIMENT_RESULTS.md) for the recorded values and metric definitions.

Evaluation is complete when the validation-selected checkpoint record exists:

```bash
test -f "$BRIEFLINE_RUN_ROOT/full_evaluation/best_finetuned_by_full_valid_combo.json"
```

## 8. Pass the Selected Adapter to RAG

The evaluator records the validation-selected adapter in:

```text
$BRIEFLINE_RUN_ROOT/full_evaluation/best_finetuned_by_full_valid_combo.json
```

Its `best_model_path` field points to the adapter artifact evaluated under the selection rule. Export it without manually guessing a checkpoint:

```bash
export ADAPTER_PATH="$(python -c 'import json, os; p=os.path.join(os.environ["BRIEFLINE_RUN_ROOT"], "full_evaluation", "best_finetuned_by_full_valid_combo.json"); print(json.load(open(p))["best_model_path"])')"

test -f "$ADAPTER_PATH/adapter_config.json"
find "$ADAPTER_PATH" -maxdepth 1 \
  \( -name 'adapter_model*.safetensors' -o -name 'adapter_model*.bin' \) \
  -print -quit | grep -q .
```

Pass the resulting value to `python -m briefline rag --adapter-path "$ADAPTER_PATH"`. This keeps the downstream application aligned with the checkpoint selected by the validation protocol.

## Bounded Verification

For the complete limited-data workflow—data preparation, smoke training, smoke evaluation, and artifact checks—follow the [Model Pipeline Smoke Test](VERIFICATION.md#3-model-pipeline-smoke-test).

Verification artifacts use separate directories from formal outputs. Reported metrics come from the full evaluation workflow above.
