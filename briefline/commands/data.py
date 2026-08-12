"""Command-line entry point for data selection and preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from data_processing.config import PipelineConfig
from data_processing.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible data pipeline.")
    parser.add_argument("--dataset", required=True, choices=("cnn_dm", "kptimes"))
    parser.add_argument(
        "--stage",
        default="all",
        choices=("select", "prepare", "all", "validate"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--task-mode",
        default="both",
        choices=("category", "keywords", "both"),
        help="KPTimes preparation task; ignored for CNN/DailyMail.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PipelineConfig(
        dataset=args.dataset,
        stage=args.stage,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        limit=args.limit,
        seed=args.seed,
        device=args.device,
        num_proc=args.num_proc,
        batch_size=args.batch_size,
        tokenizer_name=args.tokenizer_name,
        task_mode=args.task_mode,
        force_rebuild=args.force_rebuild,
    )
    result = run_pipeline(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

