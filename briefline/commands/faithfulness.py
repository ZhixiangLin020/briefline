"""Run the optional original-vs-final highlight faithfulness evaluation."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional, Sequence, Tuple

from rag.config import load_source_ids


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_MANIFEST = PROJECT_ROOT / "artifacts" / "rag" / "last_run.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-n", type=int, default=None)
    parser.add_argument("--start-at", type=int, default=0)
    parser.add_argument("--only-changed-highlight", action="store_true")
    parser.add_argument(
        "--source-ids-file",
        type=Path,
        default=None,
        help=(
            "JSON list or RAG run manifest containing inserted_source_ids. "
            "Defaults to artifacts/rag/last_run.json."
        ),
    )
    parser.add_argument(
        "--all-eligible",
        action="store_true",
        help=(
            "Evaluate all eligible PostgreSQL rows instead of restricting the run "
            "to source IDs from the latest RAG manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rag" / "faithfulness",
    )
    return parser


def resolve_source_ids(
    *,
    source_ids_file: Optional[Path],
    all_eligible: bool,
) -> Optional[Tuple[str, ...]]:
    """Resolve the exact source IDs to evaluate, or None for an explicit global run."""
    if all_eligible:
        if source_ids_file is not None:
            raise ValueError("--all-eligible cannot be combined with --source-ids-file.")
        return None

    manifest_path = (
        source_ids_file.expanduser().resolve()
        if source_ids_file is not None
        else Path(
            os.environ.get(
                "RAG_RUN_MANIFEST",
                str(DEFAULT_RUN_MANIFEST),
            )
        ).expanduser().resolve()
    )
    return load_source_ids(manifest_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    source_ids = resolve_source_ids(
        source_ids_file=args.source_ids_file,
        all_eligible=bool(args.all_eligible),
    )

    if source_ids == ():
        print("No newly inserted source IDs were found. Faithfulness evaluation skipped.")
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    os.environ["FAITHFULNESS_OUTPUT_DIR"] = str(output_dir)

    from rag import faithfulness_pipeline as pipeline

    pipeline.OUTPUT_DIR = output_dir
    pipeline.OUT_PATH = output_dir / "faithfulness_original_vs_final_results.csv"
    pipeline.ERROR_LOG_PATH = output_dir / "faithfulness_original_vs_final_errors.csv"
    pipeline.RUN_N = args.run_n
    pipeline.START_AT = int(args.start_at)
    pipeline.ONLY_CHANGED_HIGHLIGHT = bool(args.only_changed_highlight)
    pipeline.SOURCE_IDS = source_ids
    asyncio.run(pipeline.run_faithfulness_pipeline())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
