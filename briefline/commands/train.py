"""Independent training entry point for two user-specified HF DatasetDict paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from training.config import (
    DEFAULT_ROBERTA_MODEL,
    TrainingDataConfig,
    TrainingRunConfig,
)
from training.pipeline import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen CNN/DM + KPTime AdaLoRA training pipeline."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cnn-dm-dataset", type=Path)
    parser.add_argument("--kptimes-dataset", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--best-model-dir", type=Path)
    parser.add_argument("--model-name-or-path")
    parser.add_argument("--roberta-path")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true", default=None)
    parser.add_argument("--max-train-samples-per-dataset", type=int)
    parser.add_argument("--max-validation-samples-per-dataset", type=int)
    parser.add_argument("--max-test-samples-per-dataset", type=int)
    parser.add_argument("--samples-per-epoch", type=int)
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser


def _read_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("training config must contain a YAML mapping")
    data = dict(payload.get("data") or {})
    training = dict(payload.get("training") or {})
    flat = {key: value for key, value in payload.items() if key not in {"data", "training"}}
    return {**flat, **data, **training}


def _choose(cli_value: Any, mapping: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else mapping.get(key, default)


def config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
    raw = _read_config(args.config)
    cnn_path = _choose(args.cnn_dm_dataset, raw, "cnn_dm_dataset")
    kpt_path = _choose(args.kptimes_dataset, raw, "kptimes_dataset")
    output_dir = _choose(args.output_dir, raw, "output_dir")
    missing = [
        name
        for name, value in (
            ("cnn_dm_dataset", cnn_path),
            ("kptimes_dataset", kpt_path),
            ("output_dir", output_dir),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing required training paths: " + ", ".join(missing)
        )

    data = TrainingDataConfig(
        cnn_dm_dataset=Path(cnn_path),
        kptimes_dataset=Path(kpt_path),
        seed=int(_choose(args.seed, raw, "seed", 42)),
        smoke_test=bool(_choose(args.smoke_test, raw, "smoke_test", False)),
        max_train_samples_per_dataset=_choose(
            args.max_train_samples_per_dataset,
            raw,
            "max_train_samples_per_dataset",
        ),
        max_validation_samples_per_dataset=_choose(
            args.max_validation_samples_per_dataset,
            raw,
            "max_validation_samples_per_dataset",
        ),
        max_test_samples_per_dataset=_choose(
            args.max_test_samples_per_dataset,
            raw,
            "max_test_samples_per_dataset",
        ),
        samples_per_epoch=_choose(args.samples_per_epoch, raw, "samples_per_epoch"),
    )
    return TrainingRunConfig(
        data=data,
        output_dir=Path(output_dir),
        best_model_dir=(
            None
            if _choose(args.best_model_dir, raw, "best_model_dir") is None
            else Path(_choose(args.best_model_dir, raw, "best_model_dir"))
        ),
        model_name_or_path=str(
            _choose(
                args.model_name_or_path,
                raw,
                "model_name_or_path",
                "Qwen/Qwen2.5-3B-Instruct",
            )
        ),
        roberta_path=str(
            _choose(
                args.roberta_path,
                raw,
                "roberta_path",
                DEFAULT_ROBERTA_MODEL,
            )
        ),
        resume_from_checkpoint=(
            None
            if _choose(args.resume_from_checkpoint, raw, "resume_from_checkpoint") is None
            else Path(_choose(args.resume_from_checkpoint, raw, "resume_from_checkpoint"))
        ),
        smoke_epochs=int(_choose(args.smoke_epochs, raw, "smoke_epochs", 3)),
        dry_run=bool(_choose(args.dry_run, raw, "dry_run", False)),
    ).normalized()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    if not config.dry_run:
        from briefline.runtime import ensure_runtime_compatible

        ensure_runtime_compatible("training")
    result = run_training(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
