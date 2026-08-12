"""Frozen vLLM evaluation protocol from the original experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ORIGINAL_EVALUATION_SEED = 42


@dataclass(frozen=True)
class EvaluationRunConfig:
    cnn_dm_dataset: Path
    kptimes_dataset: Path
    base_model_path: Path
    roberta_path: Path
    output_dir: Path
    model_artifact_paths: Mapping[str, Path]
    tokenizer_path: Optional[Path] = None
    temp_merged_model_dir: Path = Path("/dev/shm/tmp_merged_models")
    smoke_test: bool = False
    max_samples_per_split_per_dataset: Optional[int] = None
    seed: int = 42

    def normalized(self) -> "EvaluationRunConfig":
        models = {str(alias): Path(path) for alias, path in self.model_artifact_paths.items()}
        if "base" not in models:
            models = {"base": Path(self.base_model_path), **models}
        seed = int(self.seed)
        if not self.smoke_test and seed != ORIGINAL_EVALUATION_SEED:
            raise ValueError(
                f"Full evaluation freezes seed={ORIGINAL_EVALUATION_SEED}; "
                f"got seed={seed}. Use smoke_test=True only for a reduced "
                "non-reproduction run."
            )

        sample_n = self.max_samples_per_split_per_dataset
        if self.smoke_test:
            sample_n = 50 if sample_n is None else int(sample_n)
        elif sample_n is not None:
            raise ValueError(
                "max_samples_per_split_per_dataset is test-only; pass smoke_test=True"
            )
        if sample_n is not None and sample_n <= 0:
            raise ValueError("max_samples_per_split_per_dataset must be > 0")
        return replace(
            self,
            cnn_dm_dataset=Path(self.cnn_dm_dataset),
            kptimes_dataset=Path(self.kptimes_dataset),
            base_model_path=Path(self.base_model_path),
            roberta_path=Path(self.roberta_path),
            output_dir=Path(self.output_dir),
            model_artifact_paths=models,
            tokenizer_path=(
                Path(self.base_model_path)
                if self.tokenizer_path is None
                else Path(self.tokenizer_path)
            ),
            temp_merged_model_dir=Path(self.temp_merged_model_dir),
            max_samples_per_split_per_dataset=sample_n,
            seed=seed,
        )

    @staticmethod
    def decoding_config() -> Dict[str, Any]:
        return {
            "max_new_tokens": 128,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.02,
            "answer_prefix_len": 2,
            "request_chunk_size": 1024,
            "encoder_batch_size": 512,
        }

    def manifest(self) -> Dict[str, Any]:
        cfg = self.normalized()
        return {
            "run_mode": "smoke_test" if cfg.smoke_test else "full_evaluation",
            "smoke_test_warning": (
                "NOT A REPRODUCTION RUN" if cfg.smoke_test else None
            ),
            "cnn_dm_dataset": str(cfg.cnn_dm_dataset),
            "kptimes_dataset": str(cfg.kptimes_dataset),
            "base_model_path": str(cfg.base_model_path),
            "tokenizer_path": str(cfg.tokenizer_path),
            "roberta_path": str(cfg.roberta_path),
            "model_artifact_paths": {
                alias: str(path) for alias, path in cfg.model_artifact_paths.items()
            },
            "output_dir": str(cfg.output_dir),
            "seed": cfg.seed,
            "max_samples_per_split_per_dataset": cfg.max_samples_per_split_per_dataset,
            "decoding": cfg.decoding_config(),
            "merge": {
                "auto_merge_peft_adapters": True,
                "delete_temp_merged_models": True,
                "dtype": "float16",
                "device": "cpu",
                "max_shard_size": "5GB",
                "auto_prepare_resized_base": True,
            },
            "vllm": {
                "dtype": "float16",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.90,
                "max_model_len": None,
                "enforce_eager": False,
                "disable_log_stats": True,
            },
            "selection_rule": "highest validation_combo_mover_score; test is report-only",
        }
