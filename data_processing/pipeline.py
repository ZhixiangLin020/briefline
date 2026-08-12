"""Orchestration for the two data pipelines.

This module contains no notebook state and performs no work at import time.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import math
import os
from dataclasses import asdict
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import ClusterConfig, PipelineConfig


REQUIRED_TRAINING_COLUMNS = (
    "input_ids",
    "attention_mask",
    "labels",
    "loss_weights",
)
REQUIRED_SPLITS = ("train", "validation", "test")

# These values are the arguments that were explicitly used to build the
# trainer-ready datasets in the original experiment.  Keep them here rather
# than relying on builder defaults: changing a default must not silently change
# the reproduction dataset.
CNN_DM_ORIGINAL_PREPARE_PARAMETERS: Dict[str, Any] = {
    "train_take": 1.0,
    "valid_take": 1_000,
    "test_take": 1_000,
    "highlight_prefix_text": "highlight: ",
    "article_max_tokens": 2_500,
    "highlight_prefix_weight": 0.2,
    "highlight_body_weight": 1.0,
    "terminal_active_weight": 1.0,
    "terminal_masked_weight": 0.0,
    "terminal_loss_mode": "final_only",
    "to_chat_template": True,
}

KPTIMES_ORIGINAL_PREPARE_PARAMETERS: Dict[str, Any] = {
    "task_mode": "both",
    "train_samples": 1.0,
    "valid_samples": 1_000,
    "test_samples": 1_000,
    "body_max_tokens": 2_000,
    "keyword_prefix_text": "keywords: ",
    "keyword_separator_list": [",", ";"],
    "keyword_token_idf_temperature": 0.15,
    "keyword_token_idf_cap": 2,
    "prefix_weight": 0.2,
    "separator_weight": 0.8,
    "terminal_active_weight": 1.0,
}


def _require_packages(packages: Iterable[str], *, stage: str) -> None:
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Missing packages for the {stage} stage: {joined}. "
            "Run `python scripts/install_dependencies.py` before running the pipeline."
        )


def _load_tokenizer(model_name: str):
    _require_packages(["transformers"], stage="prepare")
    from transformers import AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def _limit_dataset_dict(ds_dict, limit: Optional[int]):
    if limit is None:
        return ds_dict
    from datasets import DatasetDict

    return DatasetDict(
        {
            split: ds.select(range(min(int(limit), len(ds))))
            for split, ds in ds_dict.items()
        }
    )


def _package_versions() -> Dict[str, Optional[str]]:
    names = [
        "datasets",
        "faiss-cpu",
        "igraph",
        "leidenalg",
        "numpy",
        "pandas",
        "sentence-transformers",
        "torch",
        "transformers",
    ]
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _write_metadata(cfg: PipelineConfig, result: Dict[str, Any]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(cfg).items()
        },
        "package_versions": _package_versions(),
        "result": result,
    }
    path = cfg.output_dir / "run_metadata.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_parameters(cfg: PipelineConfig) -> Dict[str, Any]:
    if cfg.dataset == "cnn_dm":
        params = dict(CNN_DM_ORIGINAL_PREPARE_PARAMETERS)
        params["train_take"] = cfg.limit if cfg.limit is not None else params["train_take"]
        params["valid_take"] = cfg.limit if cfg.limit is not None else params["valid_take"]
        params["test_take"] = cfg.limit if cfg.limit is not None else params["test_take"]
        return params

    params = dict(KPTIMES_ORIGINAL_PREPARE_PARAMETERS)
    params["task_mode"] = cfg.task_mode
    params["train_samples"] = cfg.limit if cfg.limit is not None else params["train_samples"]
    params["valid_samples"] = cfg.limit if cfg.limit is not None else params["valid_samples"]
    params["test_samples"] = cfg.limit if cfg.limit is not None else params["test_samples"]
    return params


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _tokenizer_contract(tokenizer) -> Optional[Dict[str, Any]]:
    if tokenizer is None:
        return None
    vocab = tokenizer.get_vocab()
    chat_template = getattr(tokenizer, "chat_template", None)
    raw_model_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = None
    if isinstance(raw_model_limit, Integral):
        raw_model_limit = int(raw_model_limit)
        if 0 < raw_model_limit < 1_000_000_000:
            model_limit = raw_model_limit
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(len(tokenizer)),
        "vocab_sha256": _sha256_json(vocab),
        "chat_template_sha256": (
            None
            if chat_template is None
            else hashlib.sha256(str(chat_template).encode("utf-8")).hexdigest()
        ),
        "model_max_length": model_limit,
        # pad_token_id is deliberately excluded: the original training
        # notebook overwrites pad_token with eos_token after loading.
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
    }


def _write_prepared_manifest(cfg: PipelineConfig, ds_out, *, tokenizer=None) -> Path:
    """Write the immutable data/training hand-off manifest.

    The manifest is an extra JSON file beside the Hugging Face dataset files;
    it does not modify any row or dataset fingerprint.
    """

    prepare_parameters = _prepare_parameters(cfg)
    hash_payload = {
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "tokenizer_name": cfg.tokenizer_name,
        "prepare_parameters": prepare_parameters,
    }
    canonical = json.dumps(hash_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    split_manifest: Dict[str, Any] = {}
    for split, ds in ds_out.items():
        split_manifest[str(split)] = {
            "rows": int(len(ds)),
            "columns": list(ds.column_names),
            "fingerprint": getattr(ds, "_fingerprint", None),
        }

    payload = {
        "schema_version": 2,
        "dataset": cfg.dataset,
        "tokenizer_name": cfg.tokenizer_name,
        "seed": int(cfg.seed),
        "task_mode": cfg.task_mode if cfg.dataset == "kptimes" else "highlight",
        "required_splits": list(REQUIRED_SPLITS),
        "required_training_columns": list(REQUIRED_TRAINING_COLUMNS),
        "prepare_parameters": prepare_parameters,
        "prepare_config_hash": config_hash,
        "package_versions": _package_versions(),
        "tokenizer_contract": _tokenizer_contract(tokenizer),
        "splits": split_manifest,
    }
    manifest_path = cfg.output_dir / "prepared" / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _cluster_config(cfg: PipelineConfig) -> ClusterConfig:
    return ClusterConfig(
        take_n=cfg.limit or ClusterConfig.take_n,
        device=cfg.device,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )


def _select_cnn_dm(cfg: PipelineConfig) -> Dict[str, Any]:
    _require_packages(
        ["datasets", "faiss", "igraph", "leidenalg", "pyarrow", "sentence_transformers"],
        stage="cnn_dm select",
    )
    from .cnn_dm import run_cnn_dm_end2end

    selected_dir = cfg.output_dir / "selected"
    artifacts_dir = cfg.output_dir / "artifacts"
    embedding_dir = cfg.cache_dir / "embeddings"

    if selected_dir.exists() and not cfg.force_rebuild:
        return {"selected_dir": str(selected_dir), "selection": "reused"}

    result = run_cnn_dm_end2end(
        cfg=_cluster_config(cfg),
        emb_cache_dir=str(embedding_dir),
        artifacts_out_dir=str(artifacts_dir),
        load_emb_if_exists=not cfg.force_rebuild,
        seed=cfg.seed,
        picked_out_dir=str(selected_dir),
        reload_picked_if_exists=not cfg.force_rebuild,
        load_records=True,
        build_repmap=False,
    )
    return {
        "selected_dir": str(selected_dir),
        "artifacts_dir": str(artifacts_dir),
        "selection": "built",
        "selected_rows": int(len(result["picked_raw"])),
    }


def _select_kptimes(cfg: PipelineConfig) -> Dict[str, Any]:
    _require_packages(
        ["datasets", "faiss", "igraph", "leidenalg", "pyarrow", "sentence_transformers"],
        stage="kptimes select",
    )
    from .kptimes import build_kptimes_dedup_dataset_v2, load_kptimes_raw

    selected_dir = cfg.output_dir / "selected"
    if selected_dir.exists() and not cfg.force_rebuild:
        return {"selected_dir": str(selected_dir), "selection": "reused"}

    ds_raw = _limit_dataset_dict(load_kptimes_raw(str(cfg.cache_dir / "datasets")), cfg.limit)
    picked, summary, _manifest, picked_idx, _small = build_kptimes_dedup_dataset_v2(
        cfg=_cluster_config(cfg),
        ds_raw=ds_raw,
        split="train",
        protect_n=40,
        seed=cfg.seed,
        cluster_prefer_side="max",
        num_proc=cfg.num_proc,
        batch_size=cfg.batch_size,
        include_body=True,
        body_max_words=220,
        body_max_chars=0,
        sample_growth="sqrt",
        sample_tau=1.0,
        sample_cap=10_000,
        gap_side="right",
        log_base=4.0,
        cache_root=str(cfg.cache_dir / "kptimes"),
        overwrite_prepared=cfg.force_rebuild,
        overwrite_emb=cfg.force_rebuild,
        strict_prepared_meta=True,
        strict_emb_meta=True,
        out_ds_dir=str(selected_dir),
        save_to_disk=True,
    )
    return {
        "selected_dir": str(selected_dir),
        "selection": "built",
        "selected_rows": int(len(picked)),
        "selected_indices": int(len(picked_idx)),
        "summary": {
            key: value
            for key, value in summary.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }


def select_data(cfg: PipelineConfig) -> Dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    if cfg.dataset == "cnn_dm":
        return _select_cnn_dm(cfg)
    return _select_kptimes(cfg)


def _prepare_cnn_dm(cfg: PipelineConfig, tokenizer) -> Dict[str, Any]:
    from .cnn_dm import build_cnn_dm_highlight_trainer_dataset_v4

    selected_dir = cfg.output_dir / "selected"
    prepared_dir = cfg.output_dir / "prepared"
    if not selected_dir.exists():
        raise FileNotFoundError(f"Selected CNN/DailyMail dataset not found: {selected_dir}")

    params = _prepare_parameters(cfg)
    result = build_cnn_dm_highlight_trainer_dataset_v4(
        tokenizer=tokenizer,
        picked_train_dir=str(selected_dir),
        auto_build_picked_if_missing=False,
        train_take=params["train_take"],
        valid_take=params["valid_take"],
        test_take=params["test_take"],
        seed=cfg.seed,
        highlight_prefix_text=params["highlight_prefix_text"],
        article_max_tokens=params["article_max_tokens"],
        highlight_prefix_weight=params["highlight_prefix_weight"],
        highlight_body_weight=params["highlight_body_weight"],
        terminal_active_weight=params["terminal_active_weight"],
        terminal_masked_weight=params["terminal_masked_weight"],
        terminal_loss_mode=params["terminal_loss_mode"],
        to_chat_template=params["to_chat_template"],
        num_proc=cfg.num_proc,
        batch_size=cfg.batch_size,
        print_checks=True,
        save_final_to_disk=True,
        output_dir=str(prepared_dir),
    )
    manifest_path = _write_prepared_manifest(
        cfg,
        result["dataset"],
        tokenizer=tokenizer,
    )
    return {
        "prepared_dir": str(prepared_dir),
        "splits": {name: len(ds) for name, ds in result["dataset"].items()},
        "manifest": str(manifest_path),
    }


def _prepare_kptimes(cfg: PipelineConfig, tokenizer) -> Dict[str, Any]:
    from .kptimes import build_kptimes_title_cls_trainer_dataset_v3

    selected_dir = cfg.output_dir / "selected"
    prepared_dir = cfg.output_dir / "prepared"
    if not selected_dir.exists():
        raise FileNotFoundError(f"Selected KPTimes dataset not found: {selected_dir}")

    params = _prepare_parameters(cfg)
    ds_out, constraints = build_kptimes_title_cls_trainer_dataset_v3(
        tokenizer=tokenizer,
        task_mode=params["task_mode"],
        train_samples=params["train_samples"],
        valid_samples=params["valid_samples"],
        test_samples=params["test_samples"],
        seed=cfg.seed,
        body_max_tokens=params["body_max_tokens"],
        picked_train_dir=str(selected_dir),
        auto_build_picked_if_missing=False,
        save_root=str(cfg.cache_dir / "kptimes_processing"),
        force_rebuild=cfg.force_rebuild,
        num_proc=cfg.num_proc,
        batch_size=cfg.batch_size,
        return_constraints=True,
        keyword_prefix_text=params["keyword_prefix_text"],
        keyword_separator_list=params["keyword_separator_list"],
        keyword_token_idf_temperature=params["keyword_token_idf_temperature"],
        keyword_token_idf_cap=params["keyword_token_idf_cap"],
        prefix_weight=params["prefix_weight"],
        separator_weight=params["separator_weight"],
        terminal_active_weight=params["terminal_active_weight"],
        save_final_to_disk=False,
        print_checks=True,
    )
    ds_out.save_to_disk(str(prepared_dir))
    manifest_path = _write_prepared_manifest(cfg, ds_out, tokenizer=tokenizer)
    return {
        "prepared_dir": str(prepared_dir),
        "splits": {name: len(ds) for name, ds in ds_out.items()},
        "constraints_created": constraints is not None,
        "manifest": str(manifest_path),
    }


def prepare_data(cfg: PipelineConfig) -> Dict[str, Any]:
    _require_packages(["datasets", "transformers"], stage=f"{cfg.dataset} prepare")
    tokenizer = _load_tokenizer(cfg.tokenizer_name)
    if cfg.dataset == "cnn_dm":
        return _prepare_cnn_dm(cfg, tokenizer)
    return _prepare_kptimes(cfg, tokenizer)


def _validate_prepared_row(row, *, dataset_name: str, row_idx: int, tokenizer) -> None:
    try:
        lengths = {
            name: len(row[name]) for name in REQUIRED_TRAINING_COLUMNS
        }
    except TypeError as exc:
        raise ValueError(
            f"Prepared split {dataset_name!r} row {row_idx} token fields must be sequences"
        ) from exc
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"Prepared split {dataset_name!r} row {row_idx} has misaligned token fields: "
            f"{lengths}"
        )
    sequence_length = lengths["input_ids"]
    if sequence_length == 0:
        raise ValueError(
            f"Prepared split {dataset_name!r} row {row_idx} is an empty sequence"
        )

    vocab_size = int(len(tokenizer))
    raw_model_limit = getattr(tokenizer, "model_max_length", None)
    model_limit = None
    if isinstance(raw_model_limit, Integral):
        raw_model_limit = int(raw_model_limit)
        if 0 < raw_model_limit < 1_000_000_000:
            model_limit = raw_model_limit
    if model_limit is not None and sequence_length > model_limit:
        raise ValueError(
            f"Prepared split {dataset_name!r} row {row_idx} has sequence length "
            f"{sequence_length}, exceeding tokenizer/model limit {model_limit}"
        )

    for field_name in ("input_ids", "attention_mask", "labels"):
        for position, value in enumerate(row[field_name]):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(
                    f"Prepared split {dataset_name!r} row {row_idx} "
                    f"{field_name}[{position}] must be an integer"
                )

    for position, token_id in enumerate(row["input_ids"]):
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            raise ValueError(
                f"Prepared split {dataset_name!r} row {row_idx} "
                f"input_ids[{position}]={token_id} is outside vocabulary "
                f"[0, {vocab_size})"
            )
    for position, value in enumerate(row["attention_mask"]):
        if int(value) not in (0, 1):
            raise ValueError(
                f"Prepared split {dataset_name!r} row {row_idx} "
                f"attention_mask[{position}]={value} is not 0 or 1"
            )
    for position, label in enumerate(row["labels"]):
        label = int(label)
        if label != -100 and not 0 <= label < vocab_size:
            raise ValueError(
                f"Prepared split {dataset_name!r} row {row_idx} "
                f"labels[{position}]={label} is invalid"
            )
    for position, value in enumerate(row["loss_weights"]):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(
                f"Prepared split {dataset_name!r} row {row_idx} "
                f"loss_weights[{position}] must be numeric"
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"Prepared split {dataset_name!r} row {row_idx} contains "
                f"non-finite loss_weights[{position}]"
            )


def validate_data(cfg: PipelineConfig) -> Dict[str, Any]:
    _require_packages(["datasets", "transformers"], stage=f"{cfg.dataset} validate")
    from datasets import DatasetDict, load_from_disk

    prepared_dir = cfg.output_dir / "prepared"
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {prepared_dir}")
    ds_out = load_from_disk(str(prepared_dir))
    if not isinstance(ds_out, DatasetDict):
        raise TypeError(
            f"Prepared training data must be a DatasetDict with {REQUIRED_SPLITS}; "
            f"got {type(ds_out).__name__}."
        )

    missing_splits = [split for split in REQUIRED_SPLITS if split not in ds_out]
    if missing_splits:
        raise ValueError(
            f"Prepared dataset is missing required splits: {missing_splits}; "
            f"actual splits: {list(ds_out.keys())}"
        )

    tokenizer = _load_tokenizer(cfg.tokenizer_name)
    for split in REQUIRED_SPLITS:
        ds = ds_out[split]
        if len(ds) == 0:
            raise ValueError(f"Prepared split is empty: {split}")
        missing_columns = [name for name in REQUIRED_TRAINING_COLUMNS if name not in ds.column_names]
        if missing_columns:
            raise ValueError(
                f"Prepared split {split!r} is missing training columns: {missing_columns}"
            )

        for row_idx in range(len(ds)):
            row = ds[row_idx]
            _validate_prepared_row(
                row,
                dataset_name=split,
                row_idx=row_idx,
                tokenizer=tokenizer,
            )

    checked = 0
    if cfg.dataset == "cnn_dm":
        from .cnn_dm import verify_cnn_dm_highlight_trainer_dataset_consistency

        for ds in ds_out.values():
            verify_cnn_dm_highlight_trainer_dataset_consistency(
                ds,
                tokenizer,
                n_check=None,
            )
            checked += len(ds)
    else:
        from .kptimes import verify_trainer_dataset_consistency

        for ds in ds_out.values():
            verify_trainer_dataset_consistency(ds, tokenizer, n_check=None)
            checked += len(ds)
    manifest_path = _write_prepared_manifest(cfg, ds_out, tokenizer=tokenizer)
    return {
        "prepared_dir": str(prepared_dir),
        "validated_rows": checked,
        "manifest": str(manifest_path),
    }


def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run one explicitly requested stage and return a JSON-safe summary."""

    cfg = config.normalized()
    result: Dict[str, Any] = {"dataset": cfg.dataset, "stage": cfg.stage}
    if cfg.stage in {"select", "all"}:
        result["select"] = select_data(cfg)
    if cfg.stage in {"prepare", "all"}:
        result["prepare"] = prepare_data(cfg)
    if cfg.stage in {"validate", "all"}:
        result["validate"] = validate_data(cfg)
    _write_metadata(cfg, result)
    return result
