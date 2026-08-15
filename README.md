# Briefline

<h3 align="center">
  An end-to-end news intelligence system built around multi-task fine-tuning, multi-model verification, and hybrid retrieval.
</h3>
<p align="center">
  <strong>Explore the live application:</strong>&nbsp;
  <a href="https://briefline.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white" alt="Open the Briefline live demo">
  </a>
</p>

Briefline is an efficient news intelligence system powered by multi-task fine-tuning, hybrid retrieval, and agentic verification.

The system uses a multi-task fine-tuned model to automate news-processing tasks, alongside an agentic verification workflow that evaluates and selectively refines model-generated content, improving factual reliability. It also uses RAG-based retrieval to recommend related news based on the stories users explore.

The system delivered a **10.56%** improvement in **Multi-Task Performance**, reduced the **Hallucination Score** by **41.2%**, and achieved over **3× Faster** inference with vLLM.

<table align="center">
  <thead>
    <tr>
      <th align="center">Multi-Task Performance</th>
      <th align="center">Hallucination Score</th>
      <th align="center">Inference Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>+10.56%</strong></td>
      <td align="center"><strong>−41.2%</strong></td>
      <td align="center"><strong>&gt;3× faster</strong></td>
    </tr>
  </tbody>
</table>

## System Architecture

### Training Workflow<br>

<p align="center">
  <img src="docs/assets/briefline_model_training_workflow.svg" alt="Briefline model training workflow" width="90%">
</p>

### Application Workflow<br>

<p align="center">
  <img src="docs/assets/briefline_application_workflow.svg" alt="Briefline model application workflow" width="90%">
</p>

## Project Highlights

- **Long-Text Multi-Task Learning**<br>
  Builds shared news-domain capabilities by jointly learning multiple tasks from long-form reporting.

- **From Benchmarks to Current News**<br>
  Although trained on historical benchmark datasets, the system is applied to newly published reporting from a different source and time period.

- **Multi-Model Orchestration with Faithfulness Evaluation**<br>
  Brings complementary models together in an agentic workflow, with faithfulness evaluation measuring how well refined content remains grounded in source evidence.

- **LLM-as-a-Judge with Conditional Routing**<br>
  Uses an LLM judge to automatically evaluate each output and dynamically route flagged content for targeted correction.

- **Hybrid Retrieval and ColBERT Reranking**<br>
  Improves retrieval coverage and ranking precision, surfacing more relevant evidence and related news.

- **Graph-Based Semantic Deduplication**<br>
  Identifies semantic overlap across the training corpus to reduce redundancy and create a cleaner, more diverse dataset.

- **Efficient Fine-Tuning and Inference**<br>
  Optimizes model adaptation and serving for long-text workloads, delivering a measured over 3× inference speedup.

## Environment Requirements

| Component | Requirement |
|---|---|
| Python | Python 3.12 |
| Prebuilt artifacts | Git LFS is required to download the adapter and curated dataset archives |
| Data selection, training, evaluation, and RAG backend | Linux x86_64 and one CUDA GPU; NVIDIA A100 or A800 recommended for the full workflow |
| RAG services | PostgreSQL and Weaviate |
| Frontend | CPU-only environment is sufficient |

The pinned GPU stack targets **CUDA 12.8** with **PyTorch 2.8.0**, **vLLM 0.10.2**, and **FlashAttention 2.8.3.post1**.

## Choose a Run Path

| Goal | Start here | Scope and success signal |
|---|---|---|
| Check the repository before using a GPU | [Repository checks](docs/VERIFICATION.md#1-repository-checks) | CPU validation; repository checks pass |
| Verify the pinned GPU stack | [GPU runtime check](docs/VERIFICATION.md#2-gpu-runtime-check) | CUDA validation; environment checks pass |
| Verify the model pipeline with a bounded GPU run | [Model pipeline smoke test](docs/VERIFICATION.md#3-model-pipeline-smoke-test) | Bounded workflow; smoke evaluation and selected-model record are produced |
| Reproduce the reported model experiment | [Full model experiment](#full-model-experiment) | Full workflow; training manifests and held-out results are produced |
| Use the trained adapter and curated datasets | [Prebuilt artifacts](#prebuilt-artifacts) | Download with Git LFS, extract, and provide the local paths |
| Validate RAG configuration without live requests | [RAG preflight](docs/VERIFICATION.md#4-rag-preflight) | Configuration validation; manifest reports `preflight_ok` |
| Process Guardian articles through RAG | [RAG and Frontend Guide](docs/RAG_FRONTEND_INTEGRATION.md) | Current-news workflow; manifest reports `completed` or `no_new_records` |
| Open an already populated news database | [CPU-only frontend](#cpu-only-frontend) | Read-only interface; Streamlit page opens |

Reported metrics come from the full model experiment; smoke mode is a bounded integration check.

## Prebuilt Artifacts

The trained adapter and curated CNN/DailyMail and KPTime datasets can be downloaded as compressed archives from [`artifacts/adapter`](artifacts/adapter) and [`artifacts/dataset`](artifacts/dataset); supporting metric summaries are available in [`artifacts/results`](artifacts/results). 

After cloning the repository with Git LFS, extract the archives and pass the extracted directories to the corresponding path inputs—for example, `ADAPTER_PATH`, `data.cnn_dm_dataset`, and `data.kptimes_dataset`—to run the relevant workflow without rebuilding these artifacts.

## Full Model Experiment

This section is the shortest path through the formal data, training, and evaluation workflow. It intentionally does not use sample limits or `--smoke-test`.

<details>
<summary><strong>Show full experiment reproduction steps</strong></summary>

### 1. Enter the repository and define shared paths

Run every project command from the repository root—the directory that contains `pyproject.toml`, `README.md`, and the inner `briefline/` Python package. The repository directory itself may have any name; renaming it does not change the commands.

Choose one absolute, writable directory outside the repository for caches, prepared data, checkpoints, and evaluation outputs. Only `BRIEFLINE_WORKSPACE` needs to be changed:

```bash
# Replace this value with an absolute writable path on your machine.
export BRIEFLINE_WORKSPACE="/absolute/path/to/briefline_workspace"

export HF_HOME="$BRIEFLINE_WORKSPACE/hf_cache"
export HF_DATASETS_CACHE="$BRIEFLINE_WORKSPACE/hf_datasets_cache"
export BRIEFLINE_DATA_ROOT="$BRIEFLINE_WORKSPACE/data"
export BRIEFLINE_RUN_ROOT="$BRIEFLINE_WORKSPACE/runs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$BRIEFLINE_DATA_ROOT" "$BRIEFLINE_RUN_ROOT"
```

The commands below quote these variables, so the selected workspace may contain spaces. Keep large outputs outside the Git checkout to avoid accidentally staging datasets or checkpoints.

For Google Colab, clone or extract the repository to `/content/briefline`, then use the equivalent notebook setup:

```python
%cd /content/briefline
%env BRIEFLINE_WORKSPACE=/content/briefline_workspace
%env HF_HOME=/content/briefline_workspace/hf_cache
%env HF_DATASETS_CACHE=/content/briefline_workspace/hf_datasets_cache
%env BRIEFLINE_DATA_ROOT=/content/briefline_workspace/data
%env BRIEFLINE_RUN_ROOT=/content/briefline_workspace/runs
```

In Colab, prefix subsequent shell commands with `!`. If dependency installation restarts the runtime, repeat the `%cd` and `%env` cell. In any environment, the remaining commands reuse `BRIEFLINE_DATA_ROOT` and `BRIEFLINE_RUN_ROOT`, so the output from one stage becomes the input to the next.

### 2. Install the training environment

```bash
python scripts/install_dependencies.py
python -m briefline check-env
```

To install the RAG dependencies in the same environment, use `python scripts/install_dependencies.py --with-rag`.

### 3. Build the complete prepared datasets

```bash
python -m briefline data \
  --dataset cnn_dm \
  --stage all \
  --seed 42 \
  --output-dir "$BRIEFLINE_DATA_ROOT/cnn_dm"

python -m briefline data \
  --dataset kptimes \
  --stage all \
  --seed 42 \
  --task-mode both \
  --output-dir "$BRIEFLINE_DATA_ROOT/kptimes"
```

`--stage all` runs selection, preparation, and validation. The trainer-ready datasets are written to:

```text
$BRIEFLINE_DATA_ROOT/cnn_dm/prepared
$BRIEFLINE_DATA_ROOT/kptimes/prepared
```

Each prepared dataset contains a `manifest.json` recording row counts, fingerprints, tokenizer provenance, and preparation parameters. Do not add `--limit` when reproducing the reported experiment.

### 4. Configure and run full training

Copy the recorded experiment template instead of editing the tracked file:

```bash
cp configs/original_experiment.yaml configs/local_experiment.yaml
```

Edit these fields in `configs/local_experiment.yaml`:

| Field | Value to provide |
|---|---|
| `data.cnn_dm_dataset` | Absolute path corresponding to `$BRIEFLINE_DATA_ROOT/cnn_dm/prepared` |
| `data.kptimes_dataset` | Absolute path corresponding to `$BRIEFLINE_DATA_ROOT/kptimes/prepared` |
| `training.output_dir` | Absolute path corresponding to `$BRIEFLINE_RUN_ROOT/full` |
| `training.best_model_dir` | Absolute path corresponding to `$BRIEFLINE_RUN_ROOT/full/best_model` |
| `training.model_name_or_path` | `Qwen/Qwen2.5-3B-Instruct` or a local snapshot |
| `training.roberta_path` | `FacebookAI/roberta-large` or the original local snapshot |

Shell variables are not expanded inside YAML. Write the resolved absolute values into `configs/local_experiment.yaml`, not literal strings such as `$BRIEFLINE_DATA_ROOT/...`.

Then launch the formal run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline train \
  --config configs/local_experiment.yaml
```

The algorithm, AdaLoRA, loss, sampling, and `TrainingArguments` values are frozen in `training/config.py`. Training writes checkpoints, run metadata, and validation-ranked candidates under the configured run directory.

### 5. Evaluate the saved candidates

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

Evaluation reads candidate checkpoints from `best_model/best_k_metrics.json`; it does not retrain the model or rebuild data. The selected checkpoint is determined by the combined validation score. Test results are report-only.

Evaluation writes the validation-selected model record used by the RAG workflow.

</details>

See the [Model Pipeline Guide](docs/MODEL_PIPELINE.md) for the complete parameter contract, resume procedure, output manifests, and the optional YAML-driven end-to-end command.

## RAG Workflow

<details>
<summary><strong>Show RAG setup and integration details</strong></summary>

### 1. Install the backend environment

From the repository root, install the base GPU stack and both RAG overlays together:

```bash
python scripts/install_dependencies.py --with-rag
```

The live generation and Judge stages require the GPU environment. For a configuration-only preflight on a machine without a visible GPU, follow the [`--allow-no-cuda` procedure](docs/VERIFICATION.md#4-rag-preflight); it does not make the live backend CPU-compatible.

### 2. Provide the external services

PostgreSQL and Weaviate are provisioned as external services.

The database user must be able to create tables and indexes. Backend stages automatically create the core PostgreSQL tables, the Judge table, the recommendation table, and the Weaviate evidence collection when first needed. The taxonomy table is created separately by `python -m briefline taxonomy`.

### 3. Export settings and resolve the selected adapter

Copy the variable names from `.env.example` into exported environment variables, Colab Secrets, or Streamlit Secrets. The project does not automatically load `.env` files, and real credentials must not be committed.

Store RAG artifacts under the same user-selected workspace, or replace these values with another writable location:

```bash
export RAG_ARTIFACT_DIR="$BRIEFLINE_WORKSPACE/rag"
export RAG_TEMP_ROOT="$BRIEFLINE_WORKSPACE/rag/runtime"
```

After full evaluation, load the validation-selected adapter path:

```bash
export ADAPTER_PATH="$(python -c 'import json, os; p=os.path.join(os.environ["BRIEFLINE_RUN_ROOT"], "full_evaluation", "best_finetuned_by_full_valid_combo.json"); print(json.load(open(p))["best_model_path"])')"
test -f "$ADAPTER_PATH/adapter_config.json"
```

The RAG workflow consumes the validation-selected model produced by the evaluation stage through `ADAPTER_PATH`.

For RAG preflight, production execution, taxonomy setup, and recovery procedures, see the [RAG and Frontend Guide](docs/RAG_FRONTEND_INTEGRATION.md).

</details>

## CPU-Only Frontend

<details>
<summary><strong>Show frontend setup and launch steps</strong></summary>

The frontend reads PostgreSQL and does not load Torch, vLLM, an adapter, or a retrieval model. It expects `raw_articles`, `judge_results`, `category_broad_mapping`, and `article_recommendations` to have been populated by the backend and taxonomy workflows.

```bash
python -m pip install -r requirements-frontend.txt
export DATABASE_URL='postgresql://user:password@host:5432/database'

python -m briefline frontend \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

Open `http://localhost:8501`. Launch the frontend after the RAG and taxonomy workflows have populated the required tables.

</details>

## Configuration Map

<details>
<summary><strong>Show configuration file reference</strong></summary>

| File | Purpose |
|---|---|
| `configs/original_experiment.yaml` | Recorded full-training paths and model inputs; copy before editing |
| `configs/pipeline_all.example.yaml` | Data-to-evaluation orchestration template; replace smoke settings before a formal run |
| `configs/smoke_test.yaml` | Bounded integration check; not an experimental reproduction |
| `requirements-dev.txt` | Lightweight dependencies for CPU repository checks |
| `training/config.py` | Frozen optimizer, schedule, sampling, loss, and AdaLoRA settings |
| `evaluation/config.py` | Evaluation runtime and decoding defaults |
| `.env.example` | Names of RAG, model-path, artifact-path, and frontend variables |
| `rag/config.py` | RAG CLI defaults and validation |

</details>

## Verification

Repository, GPU, model-pipeline, RAG, and frontend checks are centralized in [Verification and Smoke Tests](docs/VERIFICATION.md).

## Repository Layout

<details>
<summary><strong>Show repository structure</strong></summary>

```text
briefline/          unified CLI and runtime checks
artifacts/           trained adapter, curated datasets, and metric summaries (Git LFS)
data_processing/    CNN/DailyMail and KPTime curation
training/           multi-task AdaLoRA training
evaluation/         vLLM generation, scoring, and selection
rag/                incremental Guardian RAG stages
frontend/           read-only Streamlit application
configs/            formal, pipeline, and smoke configurations
scripts/            installation, runtime, and documentation verification
tests/              CPU regression tests
docs/               pipeline, results, deployment, and verification guides
```

</details>

## Documentation

- [Model Pipeline Guide](docs/MODEL_PIPELINE.md)
- [Experiment Results](docs/EXPERIMENT_RESULTS.md)
- [RAG and Frontend Guide](docs/RAG_FRONTEND_INTEGRATION.md)
- [Verification and Smoke Tests](docs/VERIFICATION.md)

Run `python -m briefline <command> --help` for the full CLI reference.
