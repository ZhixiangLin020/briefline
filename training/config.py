"""Frozen data-loading configuration for the original training experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ORIGINAL_EPOCH_RATIO_SCHEDULE: Tuple[Tuple[float, float], ...] = (
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
)

ORIGINAL_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_ROBERTA_MODEL = "FacebookAI/roberta-large"
ORIGINAL_SEED = 42
ORIGINAL_MODEL_CONFIG_SIGNATURE: Dict[str, Any] = {
    "model_type": "qwen2",
    "hidden_size": 2_048,
    "intermediate_size": 11_008,
    "num_hidden_layers": 36,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32_768,
    "vocab_size": 151_936,
}
ORIGINAL_DATASET_COUNTS: Dict[str, int] = {
    "train_cnn_dm": 37_739,
    "train_kptime": 34_287,
    "validation_cnn_dm": 1_000,
    "validation_kptime": 1_000,
    "test_cnn_dm": 1_000,
    "test_kptime": 1_000,
}


def validate_original_model_config(model_config) -> None:
    mismatches = {}
    for name, expected in ORIGINAL_MODEL_CONFIG_SIGNATURE.items():
        actual = getattr(model_config, name, None)
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    quantization_config = getattr(model_config, "quantization_config", None)
    if quantization_config is not None:
        mismatches["quantization_config"] = {
            "expected": None,
            "actual": quantization_config,
        }
    if mismatches:
        raise ValueError(
            "model_name_or_path is not the frozen Qwen/Qwen2.5-3B-Instruct "
            f"base configuration: {mismatches}"
        )


@dataclass(frozen=True)
class TrainingRunConfig:
    """Runtime paths plus test-only scale controls.

    Algorithm parameters are deliberately not exposed as generic CLI
    overrides. They are returned by :meth:`training_arguments_kwargs`,
    :meth:`sft_config_kwargs`, and :meth:`peft_spec_kwargs` exactly as recorded
    in the original experiment.
    """

    data: "TrainingDataConfig"
    output_dir: Path
    best_model_dir: Optional[Path] = None
    model_name_or_path: str = ORIGINAL_MODEL_NAME
    roberta_path: str = DEFAULT_ROBERTA_MODEL
    resume_from_checkpoint: Optional[Path] = None
    smoke_epochs: int = 3
    dry_run: bool = False

    def normalized(self) -> "TrainingRunConfig":
        data = self.data.normalized()
        output_dir = Path(self.output_dir)
        best_model_dir = (
            Path(self.best_model_dir)
            if self.best_model_dir is not None
            else output_dir / "best_model"
        )
        smoke_epochs = int(self.smoke_epochs)
        if smoke_epochs <= 0:
            raise ValueError("smoke_epochs must be > 0")
        if not data.smoke_test and smoke_epochs != 3:
            raise ValueError(
                "smoke_epochs is test-only and cannot override the frozen six-epoch full run."
            )
        return replace(
            self,
            data=data,
            output_dir=output_dir,
            best_model_dir=best_model_dir,
            resume_from_checkpoint=(
                None
                if self.resume_from_checkpoint is None
                else Path(self.resume_from_checkpoint)
            ),
            smoke_epochs=smoke_epochs,
            dry_run=bool(self.dry_run),
        )

    @property
    def num_train_epochs(self) -> int:
        return self.smoke_epochs if self.data.smoke_test else 6

    def training_arguments_kwargs(self) -> Dict[str, Any]:
        """Frozen TrainingArguments values from the recorded notebook."""

        return {
            "output_dir": str(self.output_dir),
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 1,
            "learning_rate": 4e-4,
            # Preserved as recorded. The custom compute_loss does not call the
            # Trainer label smoother, so this value remains non-operative.
            "label_smoothing_factor": 0.05,
            "lr_scheduler_type": "cosine_with_min_lr",
            "lr_scheduler_kwargs": {"min_lr_rate": 0.1},
            "warmup_ratio": 0.05,
            "optim": "adamw_torch",
            "num_train_epochs": self.num_train_epochs,
            "logging_steps": 50,
            "save_steps": 0.05,
            "eval_steps": 0.05,
            "save_total_limit": 1,
            "remove_unused_columns": False,
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_combo_mover_score",
            # Preserved even though the independent top-k callback uses True.
            "greater_is_better": False,
            "bf16": True,
            "bf16_full_eval": True,
            "tf32": True,
        }

    def sft_config_kwargs(self, *, samples_per_epoch: int) -> Dict[str, Any]:
        return {
            "best_metric_name": "eval_combo_mover_score",
            "train_preview_every": 2_000,
            "train_preview_num_samples": 8,
            "train_preview_max_new_tokens": 150,
            "train_preview_do_sample": False,
            "train_preview_num_beams": 1,
            "train_preview_answer_prefix_len": 2,
            "train_preview_repetition_penalty": 1.0,
            "train_preview_no_repeat_ngram_size": 0,
            "assistant_loss_weight": 1.0,
            "use_dynamic_prompt_loss_weight": True,
            "prompt_loss_weight_start": 0.04,
            "prompt_loss_weight_end": 0.01,
            "prompt_loss_decay_start_type": "progress",
            "prompt_loss_decay_start_value": 0.0,
            "prompt_loss_decay_anchor_type": "progress",
            "prompt_loss_decay_anchor_value": 0.8,
            "prompt_loss_decay_bin_ratio": 0.01,
            "loss_normalization": "sample_mean",
            "eval_run_loss": True,
            # The formal step-1416... training logs use 300 per task. The
            # separate 400-sample/global_step=0 output is stale notebook state.
            "eval_metric_sample_size": 300,
            "eval_metric_sample_mode": "fixed",
            "eval_metric_sample_seed": 42,
            "eval_metric_max_new_tokens": 512,
            "eval_metric_do_sample": False,
            "eval_metric_num_beams": 1,
            "eval_metric_repetition_penalty": 1.0,
            "eval_metric_answer_prefix_len": 2,
            "eval_metric_generation_batch_size": 4,
            "eval_metric_no_repeat_ngram_size": 0,
            "eval_metric_length_bucket": True,
            "eval_metric_show_progress": True,
            "eval_mover_model_path": self.roberta_path,
            "eval_mover_max_length": 512,
            "eval_mover_first_layer_index": 0,
            "eval_mover_dtype": "float16",
            "eval_mover_device": None,
            "eval_mover_encoder_batch_size": 512,
            "eval_mover_cache_encoder": True,
            "eval_mover_release_encoder_after_eval": False,
            "epoch_ratio_schedule": [
                {"cnn_dm": cnn_ratio, "kptime": kpt_ratio}
                for cnn_ratio, kpt_ratio in ORIGINAL_EPOCH_RATIO_SCHEDULE
            ],
            "samples_per_epoch": int(samples_per_epoch),
            "epoch_sampling_seed": 42,
            "epoch_sampling_shuffle": True,
            "epoch_sampling_verbose": True,
            "epoch_sampling_save_state": True,
            "epoch_sampling_replay_last_epoch_on_resume": True,
            "eval_cleanup_cuda_cache_after_eval": True,
            "eval_cleanup_cuda_cache_verbose": False,
        }

    @staticmethod
    def peft_spec_kwargs(*, total_steps: int) -> Dict[str, Any]:
        return {
            "method": "adalora",
            "r": 128,
            "alpha": 128,
            "dropout": 0.05,
            "init_r": 128,
            "target_r": 90,
            "total_step": int(total_steps),
            "target_modules": "all-linear",
            "tinit": int(total_steps * 0.4),
            "tfinal": int(total_steps * 0.2),
            "deltaT": 250,
            "use_rslora": False,
            "use_dora": False,
            "init_lora_weights": True,
        }

    @staticmethod
    def reproduction_eligibility(counts: Dict[str, int]) -> Dict[str, Any]:
        mismatches = {
            name: {"expected": expected, "actual": counts.get(name)}
            for name, expected in ORIGINAL_DATASET_COUNTS.items()
            if counts.get(name) != expected
        }
        return {
            "eligible": not mismatches,
            "mismatches": mismatches,
        }


@dataclass(frozen=True)
class TrainingDataConfig:
    """Paths and scale controls for the training-data hand-off.

    Full mode keeps the original full-split sampling and ordering behavior.
    Dataset-size overrides are accepted only in explicit smoke-test mode so a
    reduced run cannot be mistaken for an experiment reproduction.
    """

    cnn_dm_dataset: Path
    kptimes_dataset: Path
    seed: int = 42
    smoke_test: bool = False
    max_train_samples_per_dataset: Optional[int] = None
    max_validation_samples_per_dataset: Optional[int] = None
    max_test_samples_per_dataset: Optional[int] = None
    samples_per_epoch: Optional[int] = None

    def normalized(self) -> "TrainingDataConfig":
        train_n = self.max_train_samples_per_dataset
        validation_n = self.max_validation_samples_per_dataset
        test_n = self.max_test_samples_per_dataset
        samples_per_epoch = self.samples_per_epoch

        seed = int(self.seed)
        if not self.smoke_test and seed != ORIGINAL_SEED:
            raise ValueError(
                f"Full training freezes seed={ORIGINAL_SEED}; got seed={seed}. "
                "Use smoke_test=True only for a reduced non-reproduction run."
            )

        overrides = (train_n, validation_n, test_n, samples_per_epoch)
        if not self.smoke_test and any(value is not None for value in overrides):
            raise ValueError(
                "Reduced dataset sizes are test-only. Pass smoke_test=True; "
                "a reproduction run must use the complete prepared datasets."
            )

        if self.smoke_test:
            train_n = 500 if train_n is None else int(train_n)
            validation_n = 50 if validation_n is None else int(validation_n)
            test_n = 50 if test_n is None else int(test_n)
            samples_per_epoch = 500 if samples_per_epoch is None else int(samples_per_epoch)

        named_values = {
            "max_train_samples_per_dataset": train_n,
            "max_validation_samples_per_dataset": validation_n,
            "max_test_samples_per_dataset": test_n,
            "samples_per_epoch": samples_per_epoch,
        }
        for name, value in named_values.items():
            if value is not None and int(value) <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")

        return replace(
            self,
            cnn_dm_dataset=Path(self.cnn_dm_dataset),
            kptimes_dataset=Path(self.kptimes_dataset),
            seed=seed,
            max_train_samples_per_dataset=train_n,
            max_validation_samples_per_dataset=validation_n,
            max_test_samples_per_dataset=test_n,
            samples_per_epoch=samples_per_epoch,
        )

    @property
    def run_mode(self) -> str:
        return "smoke_test" if self.smoke_test else "reproduction"
