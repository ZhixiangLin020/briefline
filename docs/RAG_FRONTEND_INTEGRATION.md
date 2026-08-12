# RAG and Frontend Guide

This guide covers the production-scale Guardian workflow, service configuration, restart behavior, faithfulness evaluation, and the independent CPU-only frontend. The main path uses `--mode full`; bounded checks are maintained in the centralized verification guide.

## First-Run Sequence

1. Install the RAG environment.
2. Connect Briefline to PostgreSQL and Weaviate Cloud.
3. Export the variables listed in `.env.example`.
4. Resolve the validation-selected adapter produced by model evaluation.
5. Run production preflight, then the full backend workflow.
6. Run taxonomy generation once the backend has Judge results.
7. Start the CPU-only frontend.

Briefline initializes its application tables and Weaviate collection against the configured service endpoints.

For bounded validation before the production-scale workflow, use the centralized [RAG Preflight and RAG Smoke Test](VERIFICATION.md#4-rag-preflight).

## 1. Pipeline Scope

The backend has six ordered stages:

```text
fetch → generation → retrieval → judge → similarity → faithfulness
```

| Stage | Responsibility | Main runtime dependency |
|---|---|---|
| `fetch` | Fetch Guardian articles and insert new rows | Guardian API, PostgreSQL |
| `generation` | Merge the selected adapter and generate structured article outputs | CUDA GPU, vLLM, base model and adapter |
| `retrieval` | Chunk source articles, write vectors, and build evidence packets | OpenAI API, Weaviate, PostgreSQL |
| `judge` | Run the two-stage Qwen3 judge and persist confirmed answers | CUDA GPU, judge model, PostgreSQL |
| `similarity` | Build ordered related-article recommendations | OpenAI API, Weaviate, PostgreSQL |
| `faithfulness` | Compare original and Judge-final outputs with RAGAS | Google API, RAGAS, PostgreSQL |

Taxonomy generation is managed separately from `--stages all` so it can be refreshed independently. The CPU-only frontend reads the populated database as a separate application layer.

## 2. Install the Backend Environment

Run project commands from the repository root—the directory containing `pyproject.toml`. Its directory name is not significant.

```bash
python scripts/install_dependencies.py --with-rag
python -m briefline check-env
python scripts/verify_vllm_flash_attn.py
```

The verified shared stack includes:

```text
Python 3.12
torch==2.8.0+cu128
vllm==0.10.2
transformers==4.56.2
huggingface-hub==0.34.4
flash-attn==2.8.3.post1
```

The combined installer resolves and verifies the base GPU stack and RAG overlays before the workflow starts.

For configuration-only preflight on Linux x86_64 without a visible GPU, install with `python scripts/install_dependencies.py --with-rag --allow-no-cuda` and follow [RAG Preflight](VERIFICATION.md#4-rag-preflight). Generation and Judge execution still require a CUDA GPU.

ColBERT reranking is optional. Use `--use-colbert` for hybrid retrieval with ColBERT reranking, or `--no-use-colbert` for the standard hybrid retrieval path.

## 3. Provision External Services

Briefline requires:

- an existing PostgreSQL database whose configured user can create tables and indexes;
- an existing authenticated Weaviate Cloud cluster reachable from the backend;
- API access for the Guardian, OpenAI, and Google-backed stages used in the selected run.

Use existing managed service endpoints or separately provisioned instances, then verify connectivity during preflight.

### Schema ownership

| Resource | How it is initialized |
|---|---|
| `raw_articles`, `model_outputs`, `generation_task_status` | Created automatically by Guardian fetch/generation code |
| `judge_results` | Created automatically by the Judge stage |
| `article_recommendations` | Created automatically by the similarity stage |
| `category_broad_mapping` | Created or synchronized by the separate `briefline taxonomy` command |
| Weaviate evidence collection | Created automatically by retrieval when absent |
| Similarity properties | Added to the existing evidence collection by similarity |

The stage order matters: similarity expects the evidence collection created by retrieval, and the frontend expects the PostgreSQL tables plus `category_broad_mapping` to exist.

## 4. Configure Settings and Select an Adapter

Use `.env.example` as a variable-name reference. Load values through exported environment variables, Colab Secrets, or Streamlit Secrets, and keep credentials out of tracked files.

| Variable | Used by | Required |
|---|---|---|
| `GUARDIAN_API_KEY` | Guardian ingestion | `fetch` |
| `DATABASE_URL` | Backend state and frontend reads | PostgreSQL-backed stages and frontend |
| `OPENAI_API_KEY` | Embeddings and retrieval | `retrieval`, `similarity` |
| `WEAVIATE_URL` | Vector storage | `retrieval`, `similarity` |
| `WEAVIATE_API_KEY` | Vector storage authentication | `retrieval`, `similarity` |
| `GOOGLE_API_KEY` | Taxonomy and RAGAS evaluation | `taxonomy`, `faithfulness` |
| `HF_TOKEN` | Gated or authenticated model downloads | Optional |
| `ADAPTER_PATH` | Selected PEFT adapter | `generation` unless passed by CLI |
| `BASE_MODEL_PATH` | Base generation model | Defaults to `Qwen/Qwen2.5-3B-Instruct` |
| `JUDGE_MODEL_PATH` | Judge model | Defaults to `Qwen/Qwen3-14B` |
| `RAG_ARTIFACT_DIR` | Run manifests and outputs | Optional path override |
| `RAG_TEMP_ROOT` | Temporary merged-model cache | Optional path override |
| `SITE_NAME`, `SITE_TAGLINE`, `THEME_ACCENT_COLOR` | Frontend branding | Optional |

Example shell structure:

```bash
export BRIEFLINE_WORKSPACE="/absolute/path/to/briefline_workspace"

export GUARDIAN_API_KEY='replace_me'
export DATABASE_URL='postgresql://user:password@host:5432/database'
export OPENAI_API_KEY='replace_me'
export WEAVIATE_URL='https://replace_me.weaviate.network'
export WEAVIATE_API_KEY='replace_me'
export GOOGLE_API_KEY='replace_me'

export BASE_MODEL_PATH='Qwen/Qwen2.5-3B-Instruct'
export JUDGE_MODEL_PATH='Qwen/Qwen3-14B'
export RAG_ARTIFACT_DIR="$BRIEFLINE_WORKSPACE/rag"
export RAG_TEMP_ROOT="$BRIEFLINE_WORKSPACE/rag/runtime"
```

Replace `BRIEFLINE_WORKSPACE` with an absolute writable path. Use real secret values only in the runtime environment or secret manager; do not paste credentials into tracked files.

`--adapter-path` takes precedence over `ADAPTER_PATH`. A valid adapter directory contains:

```text
adapter_config.json
adapter_model.safetensors   # or adapter_model.bin / sharded equivalent
```

If the adapter includes `tokenizer_config.json`, that tokenizer is used; otherwise the tokenizer comes from the base model path.

### Resolve the adapter selected by model evaluation

After following the full model pipeline, export the exact validation-selected path:

```bash
export BRIEFLINE_RUN_ROOT="$BRIEFLINE_WORKSPACE/runs"
export ADAPTER_PATH="$(python -c 'import json, os; p=os.path.join(os.environ["BRIEFLINE_RUN_ROOT"], "full_evaluation", "best_finetuned_by_full_valid_combo.json"); print(json.load(open(p))["best_model_path"])')"

test -f "$ADAPTER_PATH/adapter_config.json"
find "$ADAPTER_PATH" -maxdepth 1 \
  \( -name 'adapter_model*.safetensors' -o -name 'adapter_model*.bin' \) \
  -print -quit | grep -q .
```

The source JSON is written by evaluation after comparing candidates and selecting by validation score. Complete model evaluation before resolving this path.

## 5. Preflight the Full Workflow

Run preflight before fetching data or loading models:

```bash
python -m briefline rag \
  --mode full \
  --stages all \
  --max-new-articles 2500 \
  --max-pending-articles 2500 \
  --only-current-run \
  --recover-pending-generation \
  --use-colbert \
  --adapter-path "$ADAPTER_PATH" \
  --preflight-only
```

Preflight validates configuration, required variables, model structure, and optional runtime overlays without running the external workflow or loading models. See [RAG Preflight](VERIFICATION.md#4-rag-preflight) for the verification command and expected outcome.

## 6. Run the Production-Scale Workflow

Remove `--preflight-only` from the validated command:

```bash
python -m briefline rag \
  --mode full \
  --stages all \
  --max-new-articles 2500 \
  --max-pending-articles 2500 \
  --only-current-run \
  --recover-pending-generation \
  --use-colbert \
  --adapter-path "$ADAPTER_PATH"
```

### Important run controls

| Parameter | Meaning |
|---|---|
| `--mode full` | Uses the formal backend defaults rather than bounded smoke behavior |
| `--stages all` | Runs all six ordered stages, including faithfulness |
| `--max-new-articles` | Upper bound on newly fetched and accepted articles |
| `--max-pending-articles` | Shared bound for eligible incomplete historical work |
| `--only-current-run` | Restricts downstream processing to the run's explicit new/recovered scope |
| `--recover-pending-generation` | Includes bounded incomplete work alongside newly inserted rows |
| `--use-colbert` | Enables ColBERT reranking when the optional overlay is installed |
| `--adapter-path` | Overrides `ADAPTER_PATH` for this invocation |
| `--cleanup-merged-model` | Removes the selected temporary merged model after a successful run |

Full mode defaults to a maximum of 2,500 new articles. Explicit bounds keep GPU, API, and service usage predictable.

At first use, fetch/generation create the core PostgreSQL tables, retrieval creates the Weaviate evidence collection, Judge creates `judge_results`, and similarity creates `article_recommendations` and extends the evidence collection schema.

See [RAG Smoke Test](VERIFICATION.md#5-rag-smoke-test) for the final verification command and outcome definitions.

## 7. Current-Run Scope and Recovery

With `--only-current-run`, the pipeline works from a de-duplicated union of:

- source IDs inserted by the current fetch;
- bounded recent raw articles missing required generation outputs;
- bounded rows eligible to resume at later incomplete stages.

After each stage, only IDs confirmed complete are forwarded. Duplicate articles with complete outputs are not regenerated.

Every run writes a manifest at:

```text
artifacts/rag/last_run.json
```

The manifest records inserted IDs, recovered IDs by stage, completed stages, downstream scope, output paths, and failures. It does not record secret values.

### Resume from a previous manifest

```bash
python -m briefline rag \
  --mode full \
  --stages generation,retrieval,judge,similarity,faithfulness \
  --only-current-run \
  --source-ids-file artifacts/rag/last_run.json \
  --adapter-path "$ADAPTER_PATH"
```

This resumes the supplied source-ID scope without fetching again. Stage recovery checks PostgreSQL state so already completed work can be skipped and eligible incomplete work can continue.

Temporary resized and merged models are cached under `RAG_TEMP_ROOT`. The cache identity includes the base model, tokenizer, adapter path, adapter file identity, and target vocabulary size, preventing an adapter change from silently reusing an incompatible merged model.

## 8. Faithfulness Evaluation

When `faithfulness` is included in the integrated workflow, evaluation defaults to the current run's exact eligible source IDs.

To rerun the most recent scope independently:

```bash
python -m briefline faithfulness
```

To evaluate another manifest:

```bash
python -m briefline faithfulness \
  --source-ids-file /path/to/run_manifest.json
```

To bound a rerun and score only rows whose final highlight changed:

```bash
python -m briefline faithfulness \
  --run-n 50 \
  --only-changed-highlight
```

The default scope is run-specific. Request a global historical evaluation explicitly:

```bash
python -m briefline faithfulness --all-eligible
```

Standalone results and resumable state are written under `artifacts/rag/faithfulness` unless `--output-dir` is provided. Metric definitions and the reported comparison scope are in [Experiment Results](EXPERIMENT_RESULTS.md).

## 9. Taxonomy Management

The frontend reuses the `category_broad_mapping` table. Rebuild it only when intentionally replacing the existing mapping:

```bash
python -m briefline taxonomy
```

Taxonomy generation is not part of `python -m briefline rag --stages all`.

Run it after Judge results exist and before the first frontend launch. The command creates or synchronizes `category_broad_mapping`; rerun it only when intentionally rebuilding that mapping.

## 10. Launch the CPU-Only Frontend

Install the isolated frontend requirements on any machine that can reach PostgreSQL:

```bash
python -m pip install -r requirements-frontend.txt
export DATABASE_URL='postgresql://user:password@host:5432/database'

python -m briefline frontend \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true
```

The frontend reads:

```text
raw_articles
judge_results
category_broad_mapping
article_recommendations
```

The frontend is a CPU-only, read-only PostgreSQL client. Ingestion, model serving, retrieval, and schema initialization remain backend responsibilities.

Launch it after the RAG workflow and taxonomy command have populated the required tables, then open `http://localhost:8501`.

### Colab access

For a temporary Colab demonstration, keep Streamlit running in the background:

```bash
nohup python -m briefline frontend \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  > /tmp/briefline_frontend.log 2>&1 &
```

Then obtain the notebook proxy URL:

```python
from google.colab import output

frontend_url = output.eval_js("google.colab.kernel.proxyPort(8501)")
print(frontend_url)
```

Disabling CORS and XSRF protection is appropriate only for this temporary notebook proxy pattern, not a public deployment.

## 11. Operational Checks

Before a formal run, confirm:

- `python -m briefline check-env` passes in the GPU environment;
- PostgreSQL and Weaviate are reachable from the backend;
- the adapter directory contains its configuration and weights;
- external API quotas cover the explicit article bounds;
- RAG artifact and temporary-model paths have sufficient space;
- `artifacts/rag/last_run.json` is retained when recovery or auditability matters;
- the frontend host can reach PostgreSQL without receiving backend model credentials.

The bounded RAG command, manifest assertions, and frontend verification procedure are maintained in [Verification and Smoke Tests](VERIFICATION.md).
