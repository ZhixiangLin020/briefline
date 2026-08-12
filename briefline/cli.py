"""Single command-line entry point for all Briefline workflows."""

from __future__ import annotations

import importlib
import sys
from typing import Dict, Optional, Sequence, Tuple

from . import __version__
from .runtime import configure_pytorch_backend


COMMANDS: Dict[str, Tuple[str, str]] = {
    "data": (
        "briefline.commands.data",
        "select, prepare, or validate training data",
    ),
    "train": (
        "briefline.commands.train",
        "run AdaLoRA multi-task training",
    ),
    "evaluate": (
        "briefline.commands.evaluate",
        "evaluate base models and PEFT checkpoints",
    ),
    "all": (
        "briefline.commands.all_stages",
        "run data, training, and evaluation from YAML",
    ),
    "rag": (
        "briefline.commands.rag",
        "run the incremental Guardian RAG backend",
    ),
    "taxonomy": (
        "briefline.commands.taxonomy",
        "generate and persist the frontend taxonomy",
    ),
    "faithfulness": (
        "briefline.commands.faithfulness",
        "run standalone RAGAS faithfulness evaluation",
    ),
    "frontend": (
        "briefline.commands.frontend",
        "start the Streamlit frontend",
    ),
    "check-env": (
        "briefline.runtime",
        "verify the pinned GPU runtime",
    ),
}


def _usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = [
        "Usage: python -m briefline <command> [options]",
        "       briefline <command> [options]",
        "",
        "Commands:",
    ]
    lines.extend(
        f"  {name:<{width}}  {description}"
        for name, (_, description) in COMMANDS.items()
    )
    lines.extend(
        [
            "",
            "Run 'python -m briefline <command> --help' for command options.",
        ]
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if args[0] == "--version":
        print(__version__)
        return 0

    command, rest = args[0], args[1:]
    spec = COMMANDS.get(command)
    if spec is None:
        print(f"Unknown command: {command}\n\n{_usage()}", file=sys.stderr)
        return 2

    configure_pytorch_backend()
    module = importlib.import_module(spec[0])
    return int(module.main(rest) or 0)
