"""YAML-driven data, training, and evaluation execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("pipeline config must contain a YAML mapping")
    return payload


def _run_data_stage(name: str, raw: Dict[str, Any]):
    from data_processing.config import PipelineConfig
    from data_processing.pipeline import run_pipeline as run_data

    cfg = PipelineConfig(
        dataset=name,
        stage=str(raw.get("stage", "all")),
        output_dir=Path(raw["output_dir"]),
        cache_dir=(None if raw.get("cache_dir") is None else Path(raw["cache_dir"])),
        limit=raw.get("limit"),
        seed=int(raw.get("seed", 42)),
        device=str(raw.get("device", "cuda")),
        num_proc=int(raw.get("num_proc", 8)),
        batch_size=int(raw.get("batch_size", 512)),
        tokenizer_name=str(raw.get("tokenizer_name", "Qwen/Qwen2.5-3B-Instruct")),
        task_mode=str(raw.get("task_mode", "both")),
        force_rebuild=bool(raw.get("force_rebuild", False)),
    )
    return run_data(cfg)


def _read_best_models(best_model_dir: Path) -> Dict[str, Path]:
    payload = json.loads(
        (best_model_dir / "best_k_metrics.json").read_text(encoding="utf-8")
    )
    return {
        f"step{int(record['step'])}": Path(record["path"])
        for record in payload.get("best_records", [])
    }


def run_all(config_path: Path) -> Dict[str, Any]:
    raw = _load_yaml(config_path)
    data_raw = dict(raw.get("data") or {})
    training_raw = dict(raw.get("training") or {})
    evaluation_raw = dict(raw.get("evaluation") or {})
    results: Dict[str, Any] = {"data": {}}

    if not bool(training_raw.get("dry_run", False)):
        from briefline.runtime import ensure_runtime_compatible

        # One preflight covers both training and the later in-process vLLM
        # evaluation because every stage shares the same pinned stack.
        ensure_runtime_compatible("all-stage training/evaluation")

    prepared_paths: Dict[str, Path] = {}
    for name in ("cnn_dm", "kptimes"):
        stage_raw = dict(data_raw.get(name) or {})
        if stage_raw and bool(stage_raw.get("enabled", True)):
            results["data"][name] = _run_data_stage(name, stage_raw)
            prepared_paths[name] = Path(stage_raw["output_dir"]) / "prepared"

    from training.config import (
        DEFAULT_ROBERTA_MODEL,
        TrainingDataConfig,
        TrainingRunConfig,
    )
    from training.pipeline import run_training

    cnn_path = training_raw.get("cnn_dm_dataset", prepared_paths.get("cnn_dm"))
    kpt_path = training_raw.get("kptimes_dataset", prepared_paths.get("kptimes"))
    if cnn_path is None or kpt_path is None:
        raise ValueError(
            "all mode needs both training dataset paths, either from data outputs "
            "or training.cnn_dm_dataset / training.kptimes_dataset"
        )
    smoke = bool(training_raw.get("smoke_test", False))
    training_data = TrainingDataConfig(
        cnn_dm_dataset=Path(cnn_path),
        kptimes_dataset=Path(kpt_path),
        seed=int(training_raw.get("seed", 42)),
        smoke_test=smoke,
        max_train_samples_per_dataset=training_raw.get(
            "max_train_samples_per_dataset"
        ),
        max_validation_samples_per_dataset=training_raw.get(
            "max_validation_samples_per_dataset"
        ),
        max_test_samples_per_dataset=training_raw.get(
            "max_test_samples_per_dataset"
        ),
        samples_per_epoch=training_raw.get("samples_per_epoch"),
    )
    training_cfg = TrainingRunConfig(
        data=training_data,
        output_dir=Path(training_raw["output_dir"]),
        best_model_dir=(
            None
            if training_raw.get("best_model_dir") is None
            else Path(training_raw["best_model_dir"])
        ),
        model_name_or_path=str(
            training_raw.get("model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
        ),
        roberta_path=str(
            training_raw.get("roberta_path", DEFAULT_ROBERTA_MODEL)
        ),
        resume_from_checkpoint=(
            None
            if training_raw.get("resume_from_checkpoint") is None
            else Path(training_raw["resume_from_checkpoint"])
        ),
        smoke_epochs=int(training_raw.get("smoke_epochs", 3)),
        dry_run=bool(training_raw.get("dry_run", False)),
    ).normalized()
    results["training"] = run_training(training_cfg)

    if bool(evaluation_raw.get("enabled", True)) and not training_cfg.dry_run:
        from evaluation.config import EvaluationRunConfig
        from evaluation.pipeline import run_evaluation

        best_dir = Path(training_cfg.best_model_dir)
        models = {
            alias: Path(path)
            for alias, path in dict(
                evaluation_raw.get("model_artifact_paths") or {}
            ).items()
        }
        if not models:
            models = _read_best_models(best_dir)
        base_path = Path(
            evaluation_raw.get("base_model_path", training_cfg.model_name_or_path)
        )
        eval_cfg = EvaluationRunConfig(
            cnn_dm_dataset=Path(
                evaluation_raw.get(
                    "cnn_dm_dataset", training_cfg.data.cnn_dm_dataset
                )
            ),
            kptimes_dataset=Path(
                evaluation_raw.get(
                    "kptimes_dataset", training_cfg.data.kptimes_dataset
                )
            ),
            base_model_path=base_path,
            tokenizer_path=(
                None
                if evaluation_raw.get("tokenizer_path") is None
                else Path(evaluation_raw["tokenizer_path"])
            ),
            roberta_path=Path(
                evaluation_raw.get("roberta_path", training_cfg.roberta_path)
            ),
            output_dir=Path(evaluation_raw["output_dir"]),
            model_artifact_paths={"base": base_path, **models},
            temp_merged_model_dir=Path(
                evaluation_raw.get(
                    "temp_merged_model_dir", "/dev/shm/tmp_merged_models"
                )
            ),
            smoke_test=bool(evaluation_raw.get("smoke_test", smoke)),
            max_samples_per_split_per_dataset=evaluation_raw.get(
                "max_samples_per_split_per_dataset"
            ),
            seed=int(evaluation_raw.get("seed", 42)),
        ).normalized()
        results["evaluation"] = run_evaluation(eval_cfg)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run data, training, and evaluation stages from one YAML file."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_all(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
