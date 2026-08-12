"""Stage orchestration for the incremental Guardian RAG backend."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from .config import RAGRunConfig
from .model_cache import (
    build_merged_model_identity,
    build_resized_base_identity,
    cache_key,
    write_cache_identity,
)


LANGCHAIN_COMPAT_VERSIONS = {
    "langchain": "0.1.20",
    "langchain-community": "0.0.38",
    "langchain-core": "0.1.52",
    "langchain-text-splitters": "0.0.2",
}
COLBERT_REQUIRED_VERSIONS = {
    "ragatouille": "0.0.9.post2",
    "sentence-transformers": "3.4.1",
    **LANGCHAIN_COMPAT_VERSIONS,
}
FAITHFULNESS_REQUIRED_VERSIONS = {
    "ragas": "0.4.3",
    **LANGCHAIN_COMPAT_VERSIONS,
    "langchain-openai": "0.1.7",
}


def _secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


def _required_secrets(config: RAGRunConfig) -> set[str]:
    required: set[str] = set()
    if "fetch" in config.stages:
        required.update({"GUARDIAN_API_KEY", "DATABASE_URL"})
    if "generation" in config.stages or "judge" in config.stages:
        required.add("DATABASE_URL")
    if "retrieval" in config.stages or "similarity" in config.stages:
        required.update(
            {"DATABASE_URL", "OPENAI_API_KEY", "WEAVIATE_URL", "WEAVIATE_API_KEY"}
        )
    if "faithfulness" in config.stages:
        required.update({"DATABASE_URL", "GOOGLE_API_KEY"})
    return required


def _verify_required_versions(
    required_versions: Dict[str, str],
) -> tuple[Dict[str, str], list[str]]:
    from packaging.version import InvalidVersion, Version

    installed_versions: Dict[str, str] = {}
    version_errors: list[str] = []
    for distribution_name, expected_version in required_versions.items():
        try:
            actual_version = importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            version_errors.append(f"{distribution_name} is not installed")
            continue
        installed_versions[distribution_name] = actual_version
        try:
            version_matches = Version(actual_version) == Version(expected_version)
        except InvalidVersion:
            version_matches = False
        if not version_matches:
            version_errors.append(
                f"{distribution_name}=={actual_version} is installed; "
                f"expected {distribution_name}=={expected_version}"
            )
    return installed_versions, version_errors


def _verify_colbert_environment() -> Dict[str, str]:
    """Verify the text-only ColBERT import in an isolated subprocess."""

    installed_versions, version_errors = _verify_required_versions(
        COLBERT_REQUIRED_VERSIONS
    )

    if version_errors:
        raise RuntimeError(
            "ColBERT dependency preflight failed: "
            + "; ".join(version_errors)
            + ". Install requirements-rag-colbert.txt in this environment."
        )

    environment = os.environ.copy()
    environment["USE_TF"] = "0"
    environment["USE_TORCH"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "from ragatouille import RAGPretrainedModel; "
                "print('RAGatouille import OK')",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "ColBERT dependency preflight timed out while importing RAGatouille."
        ) from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown import error").strip()
        raise RuntimeError(
            "ColBERT dependency preflight could not import RAGatouille in the "
            "text-only environment. Install requirements-rag-colbert.txt and "
            f"rerun preflight. Import error: {details[-2000:]}"
        )

    return installed_versions


def _verify_faithfulness_environment() -> Dict[str, str]:
    """Verify the exact Ragas/LangChain imports before any pipeline stage runs."""

    installed_versions, version_errors = _verify_required_versions(
        FAITHFULNESS_REQUIRED_VERSIONS
    )
    if version_errors:
        raise RuntimeError(
            "Faithfulness dependency preflight failed: "
            + "; ".join(version_errors)
            + ". Run scripts/install_dependencies.py --with-rag."
        )

    environment = os.environ.copy()
    environment["USE_TF"] = "0"
    environment["USE_TORCH"] = "1"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                "import instructor; "
                "from langchain_openai import OpenAIEmbeddings; "
                "from ragas.llms import InstructorLLM; "
                "from ragas.metrics.collections import Faithfulness; "
                "print('Faithfulness imports OK')",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Faithfulness dependency preflight timed out while importing Ragas."
        ) from exc

    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown import error").strip()
        raise RuntimeError(
            "Faithfulness dependency preflight could not import the installed "
            "Ragas/LangChain stack. Run scripts/install_dependencies.py "
            f"--with-rag and rerun preflight. Import error: {details[-2000:]}"
        )

    return installed_versions


def preflight(config: RAGRunConfig) -> Dict[str, Any]:
    config.validate()
    missing = sorted(name for name in _required_secrets(config) if not _secret(name))
    if missing:
        raise RuntimeError(
            "Missing required environment variables or Colab Secrets: "
            + ", ".join(missing)
        )

    faithfulness_versions = None
    if "faithfulness" in config.stages:
        faithfulness_versions = _verify_faithfulness_environment()

    colbert_versions = None
    if "similarity" in config.stages and config.use_colbert:
        colbert_versions = _verify_colbert_environment()

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    config.temp_root.mkdir(parents=True, exist_ok=True)
    config.retrieval_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "ok",
        "required_settings": sorted(_required_secrets(config)),
        "adapter_path": None if config.adapter_path is None else str(config.adapter_path),
    }
    if colbert_versions is not None:
        report["colbert_versions"] = colbert_versions
    if faithfulness_versions is not None:
        report["faithfulness_versions"] = faithfulness_versions
    return report


def _write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _configure_module_paths(config: RAGRunConfig) -> None:
    os.environ["RAG_ARTIFACT_DIR"] = str(config.artifact_dir)
    os.environ["RAG_TEMP_ROOT"] = str(config.temp_root)
    if config.adapter_path is not None:
        os.environ["ADAPTER_PATH"] = str(config.adapter_path)


def _run_fetch(config: RAGRunConfig) -> Dict[str, Any]:
    from . import guardian_pipeline as guardian

    guardian_api_key = _secret("GUARDIAN_API_KEY")
    if not guardian_api_key:
        raise RuntimeError("GUARDIAN_API_KEY is required for the fetch stage.")
    return guardian.fetch_and_store_guardian_articles(
        api_key=guardian_api_key,
        from_date=config.guardian_from_date,
        to_date=config.guardian_to_date,
        max_new_articles=config.max_new_articles,
        seen_ids_path=str(config.temp_root / "guardian_seen_ids.json"),
    )


def _deduplicate_source_ids(source_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(source_id) for source_id in source_ids if source_id))


def _load_recoverable_pending_source_ids(
    config: RAGRunConfig,
    exclude_source_ids: Sequence[str],
) -> list[str]:
    if (
        not config.recover_pending_generation
        or not config.only_current_run
        or "generation" not in config.stages
    ):
        return []

    from . import guardian_pipeline as guardian

    limit = config.max_pending_articles or config.max_new_articles
    return _deduplicate_source_ids(
        guardian.load_pending_generation_source_ids(
            limit=limit,
            exclude_source_ids=list(exclude_source_ids),
        )
    )


def _load_recoverable_downstream_source_ids(
    config: RAGRunConfig,
    *,
    exclude_source_ids: Sequence[str],
    remaining_limit: int,
) -> tuple[list[str], list[str]]:
    """Load bounded retrieval/Judge and similarity recovery scopes."""

    if (
        not config.recover_pending_generation
        or not config.only_current_run
        or remaining_limit <= 0
    ):
        return [], []

    recover_retrieval = "retrieval" in config.stages and "judge" in config.stages
    recover_similarity = "similarity" in config.stages
    if not recover_retrieval and not recover_similarity:
        return [], []

    database_url = _secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required for pending-stage recovery.")

    from .stage_recovery import (
        load_pending_retrieval_source_ids,
        load_pending_similarity_source_ids,
    )

    excluded = _deduplicate_source_ids(exclude_source_ids)
    retrieval_source_ids: list[str] = []
    similarity_source_ids: list[str] = []

    if recover_retrieval:
        retrieval_source_ids = _deduplicate_source_ids(
            load_pending_retrieval_source_ids(
                database_url=database_url,
                limit=remaining_limit,
                exclude_source_ids=excluded,
            )
        )
        remaining_limit -= len(retrieval_source_ids)
        excluded = _deduplicate_source_ids([*excluded, *retrieval_source_ids])

    if recover_similarity and remaining_limit > 0:
        similarity_source_ids = _deduplicate_source_ids(
            load_pending_similarity_source_ids(
                database_url=database_url,
                limit=remaining_limit,
                exclude_source_ids=excluded,
            )
        )

    return retrieval_source_ids, similarity_source_ids


def _merge_explicit_scope(
    source_ids: Optional[Sequence[str]],
    additional_source_ids: Sequence[str],
) -> Optional[list[str]]:
    if source_ids is None:
        return None
    return _deduplicate_source_ids([*source_ids, *additional_source_ids])


def _tokenizer_compatibility_kwargs(tokenizer_path: str | Path) -> Dict[str, Any]:
    """Translate a Transformers-v5 tokenizer field in memory for v4.56."""

    config_path = Path(tokenizer_path) / "tokenizer_config.json"
    if not config_path.is_file():
        return {}

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read adapter tokenizer config: {config_path}") from exc

    extra_special_tokens = payload.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return {}

    additional_special_tokens = payload.get("additional_special_tokens", [])
    if additional_special_tokens is None:
        additional_special_tokens = []
    if not isinstance(additional_special_tokens, list):
        raise ValueError(
            "Adapter tokenizer_config.json must store additional_special_tokens "
            "as a list."
        )

    combined: list[str] = []
    for token in [*additional_special_tokens, *extra_special_tokens]:
        if not isinstance(token, str):
            raise ValueError(
                "The Transformers-v5 extra_special_tokens compatibility path "
                "supports string tokens only."
            )
        if token not in combined:
            combined.append(token)

    return {
        "extra_special_tokens": {},
        "additional_special_tokens": combined,
    }


def _generation_batch_limit(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
) -> int:
    return config.max_new_articles if source_ids is None else len(source_ids)


def _run_generation(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    from briefline.runtime import ensure_runtime_compatible

    ensure_runtime_compatible("Guardian RAG generation")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    from . import guardian_pipeline as guardian

    if config.adapter_path is None:
        raise RuntimeError("adapter_path is required for the generation stage.")

    adapter_path = str(config.adapter_path)
    tokenizer_path = (
        adapter_path
        if (config.adapter_path / "tokenizer_config.json").is_file()
        else config.base_model_path
    )
    tokenizer_compatibility_kwargs = _tokenizer_compatibility_kwargs(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        **tokenizer_compatibility_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    adapter_vocab_size = guardian.infer_adapter_saved_vocab_size(adapter_path)
    target_vocab_size = int(adapter_vocab_size or len(tokenizer))
    if adapter_vocab_size is not None and len(tokenizer) != target_vocab_size:
        raise ValueError(
            f"Tokenizer length ({len(tokenizer)}) does not match adapter vocabulary "
            f"size ({target_vocab_size})."
        )

    resized_identity = build_resized_base_identity(
        original_base_model_path=config.base_model_path,
        tokenizer_path=tokenizer_path,
        target_vocab_size=target_vocab_size,
    )
    resized_cache_key = cache_key(resized_identity, prefix="base")
    resized_base_path = str(
        config.temp_root / "resized_base_models" / resized_cache_key
    )

    merged_identity = build_merged_model_identity(
        original_base_model_path=config.base_model_path,
        adapter_path=adapter_path,
        tokenizer_path=tokenizer_path,
        target_vocab_size=target_vocab_size,
    )
    merged_cache_key = cache_key(merged_identity, prefix="adapter")
    merged_model_path = str(
        config.temp_root / "merged_models" / merged_cache_key
    )

    prepared_base = guardian.prepare_resized_base_model_if_needed(
        original_base_model_path=config.base_model_path,
        tokenizer=tokenizer,
        target_vocab_size=target_vocab_size,
        output_path=resized_base_path,
        dtype=torch.float16,
        device="cpu",
    )
    if Path(prepared_base).resolve() == Path(resized_base_path).resolve():
        write_cache_identity(prepared_base, resized_identity)

    merged_model_path = guardian.merge_peft_adapter_to_temp_full_model(
        base_model_path=prepared_base,
        adapter_path=adapter_path,
        output_path=merged_model_path,
        tokenizer=tokenizer,
        merge_dtype=torch.float16,
        merge_device="cpu",
    )
    if tokenizer_compatibility_kwargs:
        tokenizer.save_pretrained(merged_model_path)
    write_cache_identity(merged_model_path, merged_identity)

    processor_path = str(config.temp_root / "answer_only_repetition_processor_vllm.py")
    guardian.patch_vllm_stdout_for_colab()
    guardian.write_custom_logits_processor_module(processor_path)
    guardian.ensure_content_on_pythonpath(processor_path)
    importlib.invalidate_caches()
    importlib.import_module("answer_only_repetition_processor_vllm")

    llm = None
    try:
        llm = LLM(
            model=merged_model_path,
            tokenizer=merged_model_path,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=4096,
            enforce_eager=True,
            disable_log_stats=True,
            logits_processors=[
                "answer_only_repetition_processor_vllm:"
                "WrappedAnswerOnlyRepetitionPenaltyProcessor"
            ],
        )
        exclude_token_ids = guardian.build_exclude_token_ids(tokenizer)
        generation_df, _ = guardian.run_guardian_generation_from_postgres(
            llm=llm,
            tokenizer=tokenizer,
            limit=_generation_batch_limit(config, source_ids),
            source_ids=None if source_ids is None else list(source_ids),
            allowed_sections=guardian.GUARDIAN_ALLOWED_SECTIONS,
            require_allowed_section=True,
            highlight_max_tokens=guardian.HIGHLIGHT_MAX_TOKENS,
            both_max_tokens=guardian.BOTH_MAX_TOKENS,
            answer_repetition_penalty=guardian.ANSWER_REPETITION_PENALTY,
            exclude_token_ids=exclude_token_ids,
            print_each=config.print_each,
        )
        completed_source_ids = (
            None
            if source_ids is None
            else guardian.load_generation_complete_source_ids(list(source_ids))
        )
        return {
            "generated_rows": 0 if generation_df is None else int(len(generation_df)),
            "completed_source_ids": completed_source_ids,
            "tokenizer_v5_compatibility": bool(tokenizer_compatibility_kwargs),
            "merged_model_path": merged_model_path,
            "resized_base_cache_key": resized_cache_key,
            "merged_model_cache_key": merged_cache_key,
        }
    finally:
        guardian.unload_vllm_model(llm, name="llm")
        guardian.cleanup_cuda()


def _run_retrieval(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    import weaviate
    from weaviate.auth import AuthApiKey

    from . import guardian_pipeline as guardian

    openai_api_key = _secret("OPENAI_API_KEY")
    weaviate_url = _secret("WEAVIATE_URL")
    weaviate_api_key = _secret("WEAVIATE_API_KEY")
    if not openai_api_key or not weaviate_url or not weaviate_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY, WEAVIATE_URL, and WEAVIATE_API_KEY are required "
            "for retrieval."
        )
    os.environ["OPENAI_API_KEY"] = openai_api_key

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=AuthApiKey(weaviate_api_key),
    )
    try:
        if not client.is_ready():
            raise RuntimeError("Weaviate is not ready.")
        embedder = guardian.OpenAIEmbeddingClient(
            model=guardian.EMBEDDING_MODEL,
            dimensions=guardian.EMBEDDING_DIMENSIONS,
            batch_size=128,
        )
        result = guardian.run_batch_highlight_retrieval_pipeline(
            client=client,
            embedder=embedder,
            collection_name=config.collection_name,
            source_ids=None if source_ids is None else list(source_ids),
            row_indices=None,
            max_samples=None,
            require_format_ok=False,
            overlap_ratio=guardian.RETRIEVAL_OVERLAP_RATIO,
            top_k=guardian.RETRIEVAL_TOP_K,
            hybrid_alpha=guardian.RETRIEVAL_HYBRID_ALPHA,
            min_score=guardian.RETRIEVAL_MIN_SCORE,
            index_ready_wait_seconds=guardian.WEAVIATE_INDEX_READY_WAIT_SECONDS,
            top_n_to_print=guardian.RETRIEVAL_TOP_N_TO_PRINT,
            print_search_text=guardian.RETRIEVAL_PRINT_SEARCH_TEXT,
            save_dir=str(config.retrieval_dir),
            skip_existing=False,
            fail_fast=False,
            search_max_workers=guardian.RETRIEVAL_SEARCH_MAX_WORKERS,
            summary_csv_path="guardian_retrieval_batch_summary.csv",
            summary_json_path="guardian_retrieval_batch_summary.json",
            packets_jsonl_path="guardian_retrieval_packets.jsonl",
        )
        summary_df = result["summary_df"]
        completed_source_ids = None
        if source_ids is not None:
            successful_source_ids: set[str] = set()
            if (
                summary_df is not None
                and not summary_df.empty
                and {"source_id", "status"}.issubset(summary_df.columns)
            ):
                successful_rows = summary_df.loc[
                    summary_df["status"].astype(str).eq("ok"),
                    "source_id",
                ]
                successful_source_ids = {
                    str(source_id) for source_id in successful_rows.tolist()
                }
            completed_source_ids = [
                str(source_id)
                for source_id in source_ids
                if str(source_id) in successful_source_ids
            ]
        return {
            "summary_rows": int(len(summary_df)),
            "failed_rows": int(len(result["failed_rows"])),
            "packets_jsonl": result["aggregate_paths"]["packets_jsonl"],
            "completed_source_ids": completed_source_ids,
        }
    finally:
        client.close()


def _run_judge(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
    packets_jsonl: Optional[str],
) -> Dict[str, Any]:
    from briefline.runtime import ensure_runtime_compatible

    ensure_runtime_compatible("Guardian judge")
    from . import judge_pipeline as judge

    packet_path = packets_jsonl or str(
        config.retrieval_dir / "guardian_retrieval_packets.jsonl"
    )
    limit = None
    if source_ids is not None:
        limit = len(source_ids)
    elif config.mode == "smoke":
        limit = config.max_new_articles

    judge_llm = None
    judge_tokenizer = None
    try:
        judge_llm, judge_tokenizer = judge.load_unified_judge_model_for_reuse(
            unload_existing_generation_model=False,
            model_path=config.judge_model_path,
        )
        result_df = judge.run_unified_guardian_judge_with_loaded_model(
            judge_llm=judge_llm,
            judge_tokenizer=judge_tokenizer,
            packets_jsonl_path=packet_path,
            max_packets=limit,
            max_ck_samples=limit,
            source_ids=None if source_ids is None else list(source_ids),
        )
        completed_source_ids = None
        if source_ids is not None:
            database_url = _secret("DATABASE_URL")
            if not database_url:
                raise RuntimeError("DATABASE_URL is required for Judge completion checks.")
            from .stage_recovery import load_judge_complete_source_ids

            completed_source_ids = load_judge_complete_source_ids(
                database_url=database_url,
                source_ids=source_ids,
            )
        return {
            "judged_rows": int(len(result_df)),
            "completed_source_ids": completed_source_ids,
        }
    finally:
        judge.close_judge_vllm_model(
            judge_llm=judge_llm,
            judge_tokenizer=judge_tokenizer,
        )


def _run_similarity(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    from . import similarity_pipeline as similarity

    result = similarity.run_guardian_similar_articles_pipeline(
        process_n=None if source_ids is not None else (
            config.max_new_articles if config.mode == "smoke" else None
        ),
        use_colbert=config.use_colbert,
        source_ids=source_ids,
        collection_name=config.collection_name,
    )
    completed_source_ids = None
    if source_ids is not None:
        recommendation_records = result["recommendation_records"]
        if (
            hasattr(recommendation_records, "columns")
            and "source_id" in recommendation_records.columns
        ):
            completed_set = {
                str(source_id)
                for source_id in recommendation_records["source_id"].tolist()
            }
            completed_source_ids = [
                str(source_id)
                for source_id in source_ids
                if str(source_id) in completed_set
            ]
        else:
            completed_source_ids = list(source_ids)
    return {
        "query_articles": int(len(result["query_table"])),
        "recommendation_rows": int(len(result["recommendation_records"])),
        "sync_stats": result["sync_stats"],
        "completed_source_ids": completed_source_ids,
    }


def _run_faithfulness(
    config: RAGRunConfig,
    source_ids: Optional[Sequence[str]],
) -> Dict[str, Any]:
    """Evaluate only the current incremental scope unless global mode is explicit."""
    scoped_source_ids: Optional[tuple[str, ...]]
    if config.faithfulness_all_eligible:
        scoped_source_ids = None
        scope_name = "all_eligible"
    else:
        scoped_source_ids = tuple(str(value) for value in (source_ids or ()) if value)
        scope_name = "current_run"
        if not scoped_source_ids:
            return {
                "scope": scope_name,
                "selected_rows": 0,
                "completed_rows_in_scope": 0,
                "status": "skipped_no_source_ids",
            }

    output_dir = config.artifact_dir / "faithfulness"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from . import faithfulness_pipeline as pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Faithfulness dependencies are not installed or import-compatible. "
            "Run scripts/install_dependencies.py --with-rag before using the "
            "faithfulness stage."
        ) from exc

    pipeline.OUTPUT_DIR = output_dir
    pipeline.OUT_PATH = output_dir / "faithfulness_original_vs_final_results.csv"
    pipeline.ERROR_LOG_PATH = output_dir / "faithfulness_original_vs_final_errors.csv"
    pipeline.RUN_N = config.faithfulness_run_n
    pipeline.START_AT = 0
    pipeline.ONLY_CHANGED_HIGHLIGHT = config.faithfulness_only_changed_highlight
    pipeline.SOURCE_IDS = scoped_source_ids

    _, run_df, result_df = asyncio.run(pipeline.run_faithfulness_pipeline())

    completed_rows_in_scope = 0
    if result_df is not None and not result_df.empty:
        if scoped_source_ids is None:
            completed_rows_in_scope = int(len(result_df))
        elif "source_id" in result_df.columns:
            completed_rows_in_scope = int(
                result_df["source_id"].astype(str).isin(scoped_source_ids).sum()
            )

    return {
        "scope": scope_name,
        "source_id_count": None if scoped_source_ids is None else len(scoped_source_ids),
        "selected_rows": 0 if run_df is None else int(len(run_df)),
        "completed_rows_in_scope": completed_rows_in_scope,
        "output_path": str(pipeline.OUT_PATH),
        "status": "completed",
    }


def _safe_cleanup_merged_model(config: RAGRunConfig, merged_model_path: str) -> None:
    target = Path(merged_model_path).resolve()
    root = config.temp_root.resolve()
    if target == root or root not in target.parents:
        raise RuntimeError(
            f"Refusing to remove a merged model outside the configured temp root: {target}"
        )
    if target.is_dir():
        shutil.rmtree(target)


def run_pipeline(config: RAGRunConfig) -> Dict[str, Any]:
    from briefline.runtime import configure_pytorch_backend

    configure_pytorch_backend()
    _configure_module_paths(config)
    preflight_report = preflight(config)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "status": "preflight_ok" if config.preflight_only else "running",
        "started_at": started_at,
        "finished_at": None,
        "config": config.to_dict(),
        "preflight": preflight_report,
        "inserted_source_ids": [],
        "recovered_pending_source_ids": [],
        "recovered_pending_retrieval_source_ids": [],
        "recovered_pending_similarity_source_ids": [],
        "generation_candidate_source_ids": (
            list(config.source_ids) if "generation" in config.stages else []
        ),
        "generation_completed_source_ids": [],
        "downstream_source_ids": list(config.source_ids),
        "stage_results": {},
    }
    _write_manifest(config.run_manifest_path, manifest)

    if config.preflight_only:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(config.run_manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return manifest

    incremental_source_ids: Optional[list[str]] = (
        list(config.source_ids) if config.source_ids else None
    )
    current_source_ids: Optional[list[str]] = (
        list(config.source_ids) if config.only_current_run and config.source_ids else None
    )
    merged_model_path: Optional[str] = None
    packets_jsonl: Optional[str] = None
    recovered_pending_retrieval_source_ids: list[str] = []
    recovered_pending_similarity_source_ids: list[str] = []

    try:
        for stage in config.stages:
            if stage == "fetch":
                result = _run_fetch(config)
                inserted_source_ids = list(result.get("inserted_source_ids", []))
                pending_source_ids = _load_recoverable_pending_source_ids(
                    config,
                    inserted_source_ids,
                )
                inserted_source_id_set = set(inserted_source_ids)
                pending_source_ids = [
                    source_id
                    for source_id in pending_source_ids
                    if source_id not in inserted_source_id_set
                ]
                generation_work_source_ids = _deduplicate_source_ids(
                    [*inserted_source_ids, *pending_source_ids]
                )
                pending_limit = config.max_pending_articles or config.max_new_articles
                remaining_pending_limit = max(
                    0,
                    pending_limit - len(pending_source_ids),
                )
                (
                    recovered_pending_retrieval_source_ids,
                    recovered_pending_similarity_source_ids,
                ) = _load_recoverable_downstream_source_ids(
                    config,
                    exclude_source_ids=generation_work_source_ids,
                    remaining_limit=remaining_pending_limit,
                )
                incremental_source_ids = generation_work_source_ids
                manifest["inserted_source_ids"] = inserted_source_ids
                manifest["recovered_pending_source_ids"] = pending_source_ids
                manifest["recovered_pending_retrieval_source_ids"] = list(
                    recovered_pending_retrieval_source_ids
                )
                manifest["recovered_pending_similarity_source_ids"] = list(
                    recovered_pending_similarity_source_ids
                )
                manifest["generation_candidate_source_ids"] = (
                    generation_work_source_ids
                    if "generation" in config.stages
                    else []
                )
                manifest["downstream_source_ids"] = _deduplicate_source_ids(
                    [
                        *generation_work_source_ids,
                        *recovered_pending_retrieval_source_ids,
                        *recovered_pending_similarity_source_ids,
                    ]
                )
                if config.only_current_run:
                    current_source_ids = generation_work_source_ids
                manifest["stage_results"][stage] = {
                    "fetched": len(result.get("articles", [])),
                    "storage_stats": result.get("storage_stats", {}),
                    "recovered_pending": len(pending_source_ids),
                    "recovered_pending_retrieval": len(
                        recovered_pending_retrieval_source_ids
                    ),
                    "recovered_pending_similarity": len(
                        recovered_pending_similarity_source_ids
                    ),
                }

                downstream_requested = any(
                    candidate in config.stages for candidate in (
                        "generation",
                        "retrieval",
                        "judge",
                        "similarity",
                        "faithfulness",
                    )
                )
                has_incremental_work = bool(
                    current_source_ids
                    or recovered_pending_retrieval_source_ids
                    or recovered_pending_similarity_source_ids
                )
                if (
                    config.only_current_run
                    and downstream_requested
                    and not has_incremental_work
                ):
                    if config.faithfulness_all_eligible and "faithfulness" in config.stages:
                        manifest["incremental_downstream_status"] = "skipped_no_new_records"
                    else:
                        manifest["status"] = "no_new_records"
                        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                        _write_manifest(config.run_manifest_path, manifest)
                        print(
                            "No new or pending records require downstream processing."
                        )
                        return manifest

            elif stage == "generation":
                if current_source_ids == []:
                    manifest["stage_results"][stage] = {
                        "status": "skipped_no_source_ids"
                    }
                else:
                    result = _run_generation(config, current_source_ids)
                    merged_model_path = result.get("merged_model_path")
                    manifest["stage_results"][stage] = result
                    if current_source_ids is not None:
                        completed_source_ids = result.get("completed_source_ids")
                        if completed_source_ids is not None:
                            current_source_ids = _deduplicate_source_ids(
                                completed_source_ids
                            )
                        incremental_source_ids = list(current_source_ids)
                        manifest["generation_completed_source_ids"] = list(
                            current_source_ids
                        )
                        manifest["downstream_source_ids"] = _deduplicate_source_ids(
                            [
                                *current_source_ids,
                                *recovered_pending_retrieval_source_ids,
                                *recovered_pending_similarity_source_ids,
                            ]
                        )

            elif stage == "retrieval":
                retrieval_source_ids = _merge_explicit_scope(
                    current_source_ids,
                    recovered_pending_retrieval_source_ids,
                )
                if retrieval_source_ids == []:
                    manifest["stage_results"][stage] = {
                        "status": "skipped_no_source_ids"
                    }
                    current_source_ids = []
                else:
                    result = _run_retrieval(config, retrieval_source_ids)
                    packets_jsonl = result.get("packets_jsonl")
                    manifest["stage_results"][stage] = result
                    current_source_ids = retrieval_source_ids
                    if current_source_ids is not None:
                        completed_source_ids = result.get("completed_source_ids")
                        if completed_source_ids is not None:
                            current_source_ids = _deduplicate_source_ids(
                                completed_source_ids
                            )
                        incremental_source_ids = list(current_source_ids)
                        manifest["downstream_source_ids"] = _deduplicate_source_ids(
                            [
                                *current_source_ids,
                                *recovered_pending_similarity_source_ids,
                            ]
                        )

            elif stage == "judge":
                if current_source_ids == []:
                    manifest["stage_results"][stage] = {
                        "status": "skipped_no_source_ids"
                    }
                else:
                    result = _run_judge(
                        config,
                        current_source_ids,
                        packets_jsonl,
                    )
                    manifest["stage_results"][stage] = result
                    if current_source_ids is not None:
                        completed_source_ids = result.get("completed_source_ids")
                        if completed_source_ids is not None:
                            current_source_ids = _deduplicate_source_ids(
                                completed_source_ids
                            )
                        incremental_source_ids = list(current_source_ids)
                        manifest["downstream_source_ids"] = _deduplicate_source_ids(
                            [
                                *current_source_ids,
                                *recovered_pending_similarity_source_ids,
                            ]
                        )

            elif stage == "similarity":
                similarity_source_ids = _merge_explicit_scope(
                    current_source_ids,
                    recovered_pending_similarity_source_ids,
                )
                if similarity_source_ids == []:
                    manifest["stage_results"][stage] = {
                        "status": "skipped_no_source_ids"
                    }
                    current_source_ids = []
                else:
                    result = _run_similarity(config, similarity_source_ids)
                    manifest["stage_results"][stage] = result
                    current_source_ids = similarity_source_ids
                    if current_source_ids is not None:
                        completed_source_ids = result.get("completed_source_ids")
                        if completed_source_ids is not None:
                            current_source_ids = _deduplicate_source_ids(
                                completed_source_ids
                            )
                        incremental_source_ids = list(current_source_ids)
                        manifest["downstream_source_ids"] = list(
                            current_source_ids
                        )

            elif stage == "faithfulness":
                faithfulness_source_ids = (
                    None
                    if config.faithfulness_all_eligible
                    else incremental_source_ids
                )
                manifest["stage_results"][stage] = _run_faithfulness(
                    config,
                    faithfulness_source_ids,
                )

            _write_manifest(config.run_manifest_path, manifest)

        if config.cleanup_merged_model and merged_model_path:
            _safe_cleanup_merged_model(config, merged_model_path)
            manifest["merged_model_removed"] = True

        manifest["status"] = "completed"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(config.run_manifest_path, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(config.run_manifest_path, manifest)
        raise
