"""CLI entry point for the incremental Guardian RAG backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rag.config import build_parser, config_from_args
from rag.orchestrator import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args, PROJECT_ROOT)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

