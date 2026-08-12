"""Independent full validation/test evaluation entry point."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml

from evaluation.config import EvaluationRunConfig
from evaluation.pipeline import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate base and PEFT checkpoints with the frozen vLLM protocol."
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cnn-dm-dataset", type=Path)
    parser.add_argument("--kptimes-dataset", type=Path)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--roberta-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--temp-merged-model-dir", type=Path)
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        metavar="ALIAS=PATH",
        help="Model or adapter artifact; repeat for multiple candidates.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        default=None,
        help="Adapter checkpoint path; alias is derived from its step name.",
    )
    parser.add_argument(
        "--best-model-dir",
        type=Path,
        help="Read candidates from best_k_metrics.json written during training.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke-test", action="store_true", default=None)
    parser.add_argument("--max-samples-per-split-per-dataset", type=int)
    return parser


def _read_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("evaluation config must contain a YAML mapping")
    nested = dict(payload.get("evaluation") or {})
    flat = {key: value for key, value in payload.items() if key != "evaluation"}
    return {**flat, **nested}


def _choose(cli_value: Any, raw: Dict[str, Any], key: str, default: Any = None) -> Any:
    return cli_value if cli_value is not None else raw.get(key, default)


def _alias_for_checkpoint(path: Path) -> str:
    match = re.search(r"step[-_]?([0-9]+)", path.name, flags=re.IGNORECASE)
    return f"step{match.group(1)}" if match else path.name


def _parse_model_entries(values) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for value in values or []:
        if "=" not in str(value):
            raise ValueError(f"--model must use ALIAS=PATH, got {value!r}")
        alias, path = str(value).split("=", 1)
        alias = alias.strip()
        if not alias or not path.strip():
            raise ValueError(f"invalid --model value: {value!r}")
        out[alias] = Path(path.strip())
    return out


def _models_from_best_dir(path: Optional[Path]) -> Dict[str, Path]:
    if path is None:
        return {}
    path = Path(path)
    manifest_path = path / "best_k_metrics.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: Dict[str, Path] = {}
    for record in payload.get("best_records", []):
        step = int(record["step"])
        out[f"step{step}"] = Path(record["path"])
    return out


def config_from_args(args: argparse.Namespace) -> EvaluationRunConfig:
    raw = _read_config(args.config)
    required_values = {
        "cnn_dm_dataset": _choose(args.cnn_dm_dataset, raw, "cnn_dm_dataset"),
        "kptimes_dataset": _choose(args.kptimes_dataset, raw, "kptimes_dataset"),
        "base_model_path": _choose(args.base_model_path, raw, "base_model_path"),
        "roberta_path": _choose(args.roberta_path, raw, "roberta_path"),
        "output_dir": _choose(args.output_dir, raw, "output_dir"),
    }
    missing = [name for name, value in required_values.items() if value is None]
    if missing:
        raise ValueError("Missing required evaluation paths: " + ", ".join(missing))

    models = {
        alias: Path(path)
        for alias, path in dict(raw.get("model_artifact_paths") or {}).items()
    }
    models.update(_models_from_best_dir(_choose(args.best_model_dir, raw, "best_model_dir")))
    for checkpoint in args.checkpoint or []:
        models[_alias_for_checkpoint(checkpoint)] = checkpoint
    models.update(_parse_model_entries(args.model))

    base = Path(required_values["base_model_path"])
    models = {"base": base, **{k: v for k, v in models.items() if k != "base"}}
    return EvaluationRunConfig(
        cnn_dm_dataset=Path(required_values["cnn_dm_dataset"]),
        kptimes_dataset=Path(required_values["kptimes_dataset"]),
        base_model_path=base,
        tokenizer_path=(
            None
            if _choose(args.tokenizer_path, raw, "tokenizer_path") is None
            else Path(_choose(args.tokenizer_path, raw, "tokenizer_path"))
        ),
        roberta_path=Path(required_values["roberta_path"]),
        output_dir=Path(required_values["output_dir"]),
        model_artifact_paths=models,
        temp_merged_model_dir=Path(
            _choose(
                args.temp_merged_model_dir,
                raw,
                "temp_merged_model_dir",
                "/dev/shm/tmp_merged_models",
            )
        ),
        smoke_test=bool(_choose(args.smoke_test, raw, "smoke_test", False)),
        max_samples_per_split_per_dataset=_choose(
            args.max_samples_per_split_per_dataset,
            raw,
            "max_samples_per_split_per_dataset",
        ),
        seed=int(_choose(args.seed, raw, "seed", 42)),
    ).normalized()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    from briefline.runtime import ensure_runtime_compatible

    ensure_runtime_compatible("evaluation")
    result = run_evaluation(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
