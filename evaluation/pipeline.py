"""Wrapper around the preserved full vLLM evaluation implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .config import EvaluationRunConfig
from training.config import validate_original_model_config


def _deterministic_subset(dataset, n: int, seed: int):
    if n >= len(dataset):
        return dataset
    order = np.random.default_rng(int(seed)).permutation(len(dataset))[: int(n)]
    return dataset.select(order.tolist())


def _load_smoke_eval_map(config: EvaluationRunConfig):
    from datasets import load_from_disk

    cfg = config.normalized()
    n = int(cfg.max_samples_per_split_per_dataset)
    cnn = load_from_disk(str(cfg.cnn_dm_dataset))
    kpt = load_from_disk(str(cfg.kptimes_dataset))
    return {
        "validation": {
            "cnn_dm": _deterministic_subset(cnn["validation"], n, cfg.seed + 401),
            "kptime": _deterministic_subset(kpt["validation"], n, cfg.seed + 402),
        },
        "test": {
            "cnn_dm": _deterministic_subset(cnn["test"], n, cfg.seed + 501),
            "kptime": _deterministic_subset(kpt["test"], n, cfg.seed + 502),
        },
    }


def run_evaluation(config: EvaluationRunConfig) -> Dict[str, Any]:
    cfg = config.normalized()
    import torch
    from transformers import AutoConfig

    validate_original_model_config(
        AutoConfig.from_pretrained(str(cfg.base_model_path))
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = cfg.manifest()
    manifest_path = cfg.output_dir / "evaluation_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if cfg.smoke_test:
        print("RUN MODE: SMOKE TEST — NOT A REPRODUCTION RUN", flush=True)

    from .vllm_pipeline import run_mixed_full_valid_test_eval_vllm

    dataset_obj = _load_smoke_eval_map(cfg) if cfg.smoke_test else None
    outputs = run_mixed_full_valid_test_eval_vllm(
        dataset_path=None,
        dataset_obj=dataset_obj,
        cnn_dm_dataset_path=(None if dataset_obj is not None else str(cfg.cnn_dm_dataset)),
        kptime_dataset_path=(None if dataset_obj is not None else str(cfg.kptimes_dataset)),
        model_artifact_paths={
            alias: str(path) for alias, path in cfg.model_artifact_paths.items()
        },
        base_model_path=str(cfg.base_model_path),
        tokenizer_path=str(cfg.tokenizer_path),
        roberta_path=str(cfg.roberta_path),
        output_dir=str(cfg.output_dir),
        # CLI paths may be Hugging Face model IDs. Reuse the cache when it is
        # complete and allow from_pretrained/vLLM to download missing files.
        local_files_only=False,
        decoding_cfg=cfg.decoding_config(),
        auto_merge_peft_adapters=True,
        temp_merged_model_dir=str(cfg.temp_merged_model_dir),
        delete_temp_merged_models=True,
        merge_dtype=torch.float16,
        merge_device="cpu",
        merge_max_shard_size="5GB",
        auto_prepare_resized_base=True,
        delete_temp_resized_base_model=True,
        strict_tokenizer_adapter_vocab_match=True,
        vllm_dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        max_model_len=None,
        enforce_eager=False,
        disable_log_stats=True,
        seed=cfg.seed,
        roberta_dtype=torch.float16,
        roberta_device="cuda",
        save_model_io=True,
        save_predictions_csv=True,
        save_csv=True,
        use_tqdm=True,
        print_first_batch=True,
        continue_on_error=True,
    )
    return {
        "status": "completed",
        "manifest_path": str(manifest_path),
        "best_by_full_valid": outputs.get("best_by_full_valid"),
        "paths": outputs.get("paths", {}),
    }
