# Verification and Smoke Tests

This guide provides bounded checks for Briefline without mixing their outputs with the formal experiment. Choose the lowest verification level that covers the component you need to inspect.

## Verification Levels

| Level | GPU | External services | What it verifies |
|---|---:|---|---|
| Repository checks | No | None | Python syntax, local documentation links, CLI contracts, configuration rules, and regression tests |
| GPU runtime check | Yes | None | CUDA, PyTorch, vLLM, and FlashAttention compatibility |
| Model pipeline smoke | Yes | Hugging Face access | Bounded data preparation, training, checkpoint discovery, and evaluation |
| RAG preflight | No* | Runtime settings | Required variables, adapter structure, and optional dependency overlays |
| RAG smoke | Yes | PostgreSQL, Weaviate, Guardian, OpenAI, and Google APIs | Bounded ingestion-to-faithfulness workflow |
| Frontend verification | No | Populated PostgreSQL | Database contract and Streamlit startup |

These levels are cumulative only when their prerequisites overlap. For example, frontend verification does not require the GPU environment, while RAG smoke does.

`*` RAG preflight can run without a visible GPU when the environment is installed with `--allow-no-cuda`. This flag skips the hardware requirement during installation verification; it does not make the live generation or Judge stages CPU-compatible.

## Common Setup

Run commands from the repository root—the directory containing `pyproject.toml`. The repository directory may have any name. Choose an absolute, writable workspace outside the Git checkout:

```bash
export BRIEFLINE_WORKSPACE="/absolute/path/to/briefline_workspace"
export HF_HOME="$BRIEFLINE_WORKSPACE/hf_cache"
export HF_DATASETS_CACHE="$BRIEFLINE_WORKSPACE/hf_datasets_cache"
export BRIEFLINE_SMOKE_DATA_ROOT="$BRIEFLINE_WORKSPACE/smoke/data"
export BRIEFLINE_SMOKE_RUN_ROOT="$BRIEFLINE_WORKSPACE/smoke/runs"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" \
  "$BRIEFLINE_SMOKE_DATA_ROOT" "$BRIEFLINE_SMOKE_RUN_ROOT"
```

In Colab, use `/content/briefline_workspace` as `BRIEFLINE_WORKSPACE`, set the derived values with `%env`, and prefix shell commands with `!`. If dependency installation restarts the runtime, return to the repository root and set the variables again.

## 1. Repository Checks

Install the lightweight check dependencies, then run the repository checks. These commands do not install the model stack, load models, call external services, or require a CUDA device.

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python scripts/check_markdown_links.py
python -m unittest discover -s tests -t .
```

Success criteria:

- `compileall` exits without an error;
- the Markdown checker reports that every local file and heading target exists;
- the test runner ends with `OK` and exit code `0`.

These tests exercise imports, CLI routing, data and training configuration, checkpoint bookkeeping, RAG stage scoping, recovery rules, and secret redaction. They do not prove that GPU kernels or external services are available.

## 2. GPU Runtime Check

Install the pinned model environment before running the checks:

```bash
python scripts/install_dependencies.py
python -m briefline check-env
python scripts/verify_vllm_flash_attn.py
```

The expected runtime is Python 3.12 on Linux x86_64 with CUDA 12.8, PyTorch 2.8.0, vLLM 0.10.2, and FlashAttention 2.8.3.post1. Both verification commands must exit with code `0`.

This level validates the installed stack. It does not train or evaluate a model.

## 3. Model Pipeline Smoke Test

The model smoke test follows the complete execution order:

```text
bounded data preparation → smoke training → smoke evaluation → artifact checks
```

It writes to dedicated smoke directories so limited datasets and checkpoints cannot be confused with formal experiment artifacts.

### 3.1 Prepare bounded datasets

```bash
python -m briefline data \
  --dataset cnn_dm \
  --stage all \
  --limit 500 \
  --seed 42 \
  --output-dir "$BRIEFLINE_SMOKE_DATA_ROOT/cnn_dm"

python -m briefline data \
  --dataset kptimes \
  --stage all \
  --limit 500 \
  --seed 42 \
  --task-mode both \
  --output-dir "$BRIEFLINE_SMOKE_DATA_ROOT/kptimes"
```

Confirm the prepared datasets and their provenance manifests:

```bash
test -f "$BRIEFLINE_SMOKE_DATA_ROOT/cnn_dm/prepared/manifest.json"
test -f "$BRIEFLINE_SMOKE_DATA_ROOT/kptimes/prepared/manifest.json"
```

### 3.2 Run smoke training

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline train \
  --cnn-dm-dataset "$BRIEFLINE_SMOKE_DATA_ROOT/cnn_dm/prepared" \
  --kptimes-dataset "$BRIEFLINE_SMOKE_DATA_ROOT/kptimes/prepared" \
  --output-dir "$BRIEFLINE_SMOKE_RUN_ROOT/training" \
  --best-model-dir "$BRIEFLINE_SMOKE_RUN_ROOT/training/best_model" \
  --model-name-or-path Qwen/Qwen2.5-3B-Instruct \
  --roberta-path FacebookAI/roberta-large \
  --smoke-test
```

Confirm that training produced its run record and checkpoint index:

```bash
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/training/run_manifest.json"
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/training/training_result.json"
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/training/best_model/best_k_metrics.json"
```

Smoke training uses bounded in-memory subsets and does not modify the prepared source datasets.

### 3.3 Evaluate smoke checkpoints

```bash
CUDA_VISIBLE_DEVICES=0 python -m briefline evaluate \
  --cnn-dm-dataset "$BRIEFLINE_SMOKE_DATA_ROOT/cnn_dm/prepared" \
  --kptimes-dataset "$BRIEFLINE_SMOKE_DATA_ROOT/kptimes/prepared" \
  --base-model-path Qwen/Qwen2.5-3B-Instruct \
  --tokenizer-path Qwen/Qwen2.5-3B-Instruct \
  --roberta-path FacebookAI/roberta-large \
  --best-model-dir "$BRIEFLINE_SMOKE_RUN_ROOT/training/best_model" \
  --output-dir "$BRIEFLINE_SMOKE_RUN_ROOT/evaluation" \
  --temp-merged-model-dir "$BRIEFLINE_SMOKE_RUN_ROOT/tmp_merged_models" \
  --smoke-test
```

Confirm the evaluation outputs:

```bash
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/evaluation/evaluation_manifest.json"
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/evaluation/summary_vs_base.csv"
test -f "$BRIEFLINE_SMOKE_RUN_ROOT/evaluation/best_finetuned_by_full_valid_combo.json"
```

The final JSON proves that the bounded data-to-selection loop completed. It does not make the smoke-selected checkpoint suitable for experimental reporting.

## 4. RAG Preflight

Choose the installation command that matches the check being performed.

For preflight followed by a live GPU workflow:

```bash
python scripts/install_dependencies.py --with-rag
```

For configuration-only preflight on a Linux x86_64 machine without a visible GPU:

```bash
python scripts/install_dependencies.py --with-rag --allow-no-cuda
```

The second command still installs the pinned CUDA-targeted package stack so that its imports and versions can be checked. It only relaxes the final visible-GPU assertion.

Provide the variables listed in `.env.example`. For `--stages all`, the relevant settings include Guardian, PostgreSQL, OpenAI, Weaviate, Google, model paths, artifact paths, and a valid adapter.

To reuse the adapter selected by the model smoke evaluation:

```bash
export ADAPTER_PATH="$(python -c 'import json, os; p=os.path.join(os.environ["BRIEFLINE_SMOKE_RUN_ROOT"], "evaluation", "best_finetuned_by_full_valid_combo.json"); print(json.load(open(p))["best_model_path"])')"
export RAG_ARTIFACT_DIR="$BRIEFLINE_WORKSPACE/smoke/rag"
export RAG_TEMP_ROOT="$BRIEFLINE_WORKSPACE/smoke/rag/runtime"

test -f "$ADAPTER_PATH/adapter_config.json"
find "$ADAPTER_PATH" -maxdepth 1 \
  \( -name 'adapter_model*.safetensors' -o -name 'adapter_model*.bin' \) \
  -print -quit | grep -q .
```

Run preflight:

```bash
python -m briefline rag \
  --mode smoke \
  --stages all \
  --max-new-articles 10 \
  --max-pending-articles 10 \
  --only-current-run \
  --recover-pending-generation \
  --use-colbert \
  --adapter-path "$ADAPTER_PATH" \
  --preflight-only
```

Confirm the manifest status:

```bash
python -c 'import json, os; p=os.path.join(os.environ["RAG_ARTIFACT_DIR"], "last_run.json"); d=json.load(open(p)); assert d["status"] == "preflight_ok", d; print(p, d["status"])'
```

Preflight validates paths, required settings, adapter files, and installed overlays without fetching articles or loading the generation and Judge models. It does not prove that remote services are reachable; the RAG smoke run performs the live integration check.

## 5. RAG Smoke Test

Run the same bounded configuration without `--preflight-only`:

```bash
python -m briefline rag \
  --mode smoke \
  --stages all \
  --max-new-articles 10 \
  --max-pending-articles 10 \
  --only-current-run \
  --recover-pending-generation \
  --use-colbert \
  --adapter-path "$ADAPTER_PATH"
```

The workflow can fetch at most 10 new articles and add at most 10 eligible pending articles. Downstream stages remain restricted to the explicit new/recovered work set.

Confirm the outcome:

```bash
python -c 'import json, os; p=os.path.join(os.environ["RAG_ARTIFACT_DIR"], "last_run.json"); d=json.load(open(p)); assert d["status"] in {"completed", "no_new_records"}, d; print(p, d["status"])'
```

Status meanings:

| Status | Meaning |
|---|---|
| `completed` | Every requested stage completed for the eligible work set |
| `no_new_records` | No new or recoverable articles required downstream work |
| `failed` | The manifest contains an `error` object with the failure type and message |

The run also records stage results, source-ID scope, completed IDs, recovery decisions, and artifact paths in `last_run.json`.

## 6. Frontend Verification

The frontend check uses a CPU-only environment and a PostgreSQL database already populated by the RAG and taxonomy workflows.

Install the frontend dependencies and set the database URL:

```bash
python -m pip install -r requirements-frontend.txt
export DATABASE_URL='postgresql://user:password@host:5432/database'
```

Confirm the required table contract:

```bash
python - <<'PY'
import os
import psycopg

required = (
    "raw_articles",
    "judge_results",
    "category_broad_mapping",
    "article_recommendations",
)

with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT name, to_regclass(name) FROM unnest(%s::text[]) AS required(name)",
            (list(required),),
        )
        missing = [name for name, relation in cursor.fetchall() if relation is None]

if missing:
    raise SystemExit("Missing frontend tables: " + ", ".join(missing))

print("Frontend database contract is ready.")
PY
```

Start Streamlit:

```bash
python -m briefline frontend \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

Open `http://localhost:8501` and verify that the home page, article details, and related-article navigation load from PostgreSQL. The frontend does not create tables, ingest articles, or run model inference.

## Verification Scope

Bounded data and smoke runs validate pipeline integration. The reported model and RAG metrics come from the complete experiment workflows. Repository checks, RAG preflight, and frontend verification each validate their stated component contracts; GPU runtime and live-service behavior are covered by their dedicated checks.

For the formal paths, use the [Model Pipeline Guide](MODEL_PIPELINE.md) and [RAG and Frontend Guide](RAG_FRONTEND_INTEGRATION.md). The reporting rules and recorded metrics are in [Experiment Results](EXPERIMENT_RESULTS.md).

## Troubleshooting Entry Points

| Failure | First check |
|---|---|
| CLI option or required path is unclear | `python -m briefline <command> --help` |
| CUDA, vLLM, or FlashAttention mismatch | `python -m briefline check-env` |
| Training cannot find data | Confirm both paths are extracted `datasets.load_from_disk()` directories |
| Evaluation finds no candidates | Inspect `best_model/best_k_metrics.json` |
| RAG preflight fails | Check the reported variable, adapter file, or optional dependency |
| RAG execution fails | Inspect `RAG_ARTIFACT_DIR/last_run.json` and its `error` field |
| Similarity cannot find its collection | Run retrieval first; it creates the evidence collection |
| Frontend reports missing tables | Complete the RAG and taxonomy workflows before launching it |
