"""Install the verified Torch/vLLM/FlashAttention stack without compilation."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from briefline.runtime import REQUIRED_VERSIONS, verify_environment


DEFAULT_REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
COMPLETE_RAG_REQUIREMENTS = (
    PROJECT_ROOT / "requirements-rag-evaluation.txt",
    PROJECT_ROOT / "requirements-rag-colbert.txt",
)
PYTORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v2.8.3.post1/"
    "flash_attn-2.8.3.post1%2Bcu12torch2.8cxx11abiTRUE-"
    "cp312-cp312-linux_x86_64.whl"
)
STACK_DISTRIBUTIONS = frozenset(REQUIRED_VERSIONS)


def _normalized_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise ValueError(f"Unsupported requirement line: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def load_project_requirements(path: Path) -> Tuple[List[str], Dict[str, str]]:
    """Return non-stack requirements after validating every stack pin."""

    requirements: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement", "-c", "--constraint")):
            raise ValueError(
                "requirements.txt must be self-contained; nested requirements "
                f"are not supported: {line!r}"
            )
        name = _normalized_name(line)
        if name in requirements:
            raise ValueError(f"Duplicate requirement for {name!r}")
        requirements[name] = line

    for name, version in REQUIRED_VERSIONS.items():
        expected = f"{name}=={version}"
        actual = requirements.get(name)
        if actual != expected:
            raise ValueError(
                f"requirements.txt must pin {name!r} as {expected!r}; "
                f"found {actual!r}"
            )

    extras = [
        requirement
        for name, requirement in requirements.items()
        if name not in STACK_DISTRIBUTIONS
    ]
    return extras, requirements


def build_commands(
    requirements_path: Path,
    *,
    include_rag: bool = False,
) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """Build the exact binary-only stack sequence plus project dependencies."""

    requirements_path = requirements_path.resolve()
    extras, _ = load_project_requirements(requirements_path)
    python_pip = [sys.executable, "-m", "pip", "install"]
    rag_arguments: List[str] = []
    if include_rag:
        for overlay_path in COMPLETE_RAG_REQUIREMENTS:
            if not overlay_path.is_file():
                raise FileNotFoundError(overlay_path)
            rag_arguments.extend(["--requirement", str(overlay_path)])

    commands: List[Tuple[str, List[str]]] = [
        (
            "PyTorch 2.8.0 CUDA 12.8 wheels",
            [
                *python_pip,
                "torch==2.8.0",
                "torchvision==0.23.0",
                "torchaudio==2.8.0",
                "--index-url",
                PYTORCH_INDEX_URL,
            ],
        ),
        (
            "vLLM 0.10.2",
            [
                *python_pip,
                "--constraint",
                str(requirements_path),
                "vllm==0.10.2",
            ],
        ),
        (
            "official FlashAttention wheel",
            [*python_pip, "--no-deps", FLASH_ATTN_WHEEL_URL],
        ),
        (
            "Transformers 4.56.2",
            [
                *python_pip,
                "--no-deps",
                "--force-reinstall",
                "transformers==4.56.2",
            ],
        ),
        (
            "huggingface-hub 0.34.4",
            [
                *python_pip,
                "--no-deps",
                "--force-reinstall",
                "huggingface-hub==0.34.4",
            ],
        ),
        (
            "remaining project dependencies",
            [
                *python_pip,
                "--constraint",
                str(requirements_path),
                *extras,
                *rag_arguments,
            ],
        ),
    ]
    pip_check = [sys.executable, "-m", "pip", "check"]
    return commands, pip_check


def _validate_wheel_runtime() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "The official FlashAttention wheel is tagged for Python 3.12; "
            f"this interpreter is Python {sys.version_info.major}.{sys.version_info.minor}."
        )
    if not sys.platform.startswith("linux"):
        raise RuntimeError("The official FlashAttention wheel requires Linux.")
    machine = platform.machine().lower()
    if machine not in {"x86_64", "amd64"}:
        raise RuntimeError(
            "The official FlashAttention wheel requires Linux x86_64; "
            f"found {machine!r}."
        )


def _run(command: Sequence[str]) -> None:
    print("[RUN] " + shlex.join(command), flush=True)
    subprocess.run(list(command), check=True)


def _project_dependency_closure(roots: Sequence[str]) -> set[str]:
    """Return installed runtime dependencies reachable from project roots."""

    from packaging.markers import default_environment
    from packaging.requirements import InvalidRequirement, Requirement

    marker_environment = default_environment()
    marker_environment["extra"] = ""
    closure: set[str] = set()
    pending = [_normalized_name(name) for name in roots]

    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue

        for raw_requirement in distribution.requires or ():
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate(
                marker_environment
            ):
                continue
            dependency = _normalized_name(requirement.name)
            if dependency not in closure:
                pending.append(dependency)

    return closure


def _run_project_pip_check(
    command: Sequence[str], project_distributions: Sequence[str]
) -> None:
    """Fail only for dependency problems in this project's package graph.

    Colab may ship unrelated packages whose metadata is already inconsistent.
    Those ambient issues must not turn a valid project installation into a
    failure, and the installer must not uninstall unrelated notebook packages.
    """

    print("[RUN] " + shlex.join(command), flush=True)
    completed = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    if completed.returncode == 0:
        if output:
            print(output, flush=True)
        return
    if not output:
        raise RuntimeError("pip check failed without reporting a dependency issue.")

    project_graph = _project_dependency_closure(project_distributions)
    project_issues: List[str] = []
    ambient_issues: List[str] = []
    for line in output.splitlines():
        issue = line.strip()
        if not issue:
            continue
        distribution = _normalized_name(issue.split(maxsplit=1)[0])
        if distribution in project_graph:
            project_issues.append(issue)
        else:
            ambient_issues.append(issue)

    if project_issues:
        raise RuntimeError(
            "Project dependency check failed:\n" + "\n".join(project_issues)
        )
    if ambient_issues:
        print(
            "[PIP CHECK] Ignoring conflicts from preinstalled packages outside "
            "the project dependency graph:",
            flush=True,
        )
        for issue in ambient_issues:
            print(f"  - {issue}", flush=True)


def _overlay_distribution_names(paths: Sequence[Path]) -> List[str]:
    """Return package names declared by optional requirements and their includes."""

    names: List[str] = []
    pending = [path.resolve() for path in paths]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                pending.append((path.parent / line[3:].strip()).resolve())
                continue
            if line.startswith("--requirement "):
                pending.append((path.parent / line.split(maxsplit=1)[1]).resolve())
                continue
            if line.startswith(("-c", "--constraint")):
                continue
            name = _normalized_name(line)
            if name not in names:
                names.append(name)
    return names


def _verify_complete_rag_imports() -> None:
    """Import both optional RAG stacks in a clean PyTorch-only subprocess."""

    environment = os.environ.copy()
    environment["USE_TF"] = "0"
    environment["USE_TORCH"] = "1"
    command = [
        sys.executable,
        "-c",
        (
            "import instructor; "
            "from langchain_openai import OpenAIEmbeddings; "
            "from ragatouille import RAGPretrainedModel; "
            "from ragas.llms import InstructorLLM; "
            "from ragas.metrics.collections import Faithfulness; "
            "print('Complete RAG imports OK')"
        ),
    ]
    print("[RUN] " + shlex.join(command), flush=True)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Complete RAG dependency import check timed out.") from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout or "unknown import error").strip()
        raise RuntimeError(
            "Complete RAG dependency import check failed:\n" + details[-4000:]
        )
    if completed.stdout.strip():
        print(completed.stdout.strip(), flush=True)


def install(
    *,
    requirements_path: Path = DEFAULT_REQUIREMENTS,
    dry_run: bool = False,
    require_cuda: bool = True,
    include_rag: bool = False,
) -> None:
    requirements_path = requirements_path.resolve()
    if not requirements_path.is_file():
        raise FileNotFoundError(requirements_path)

    commands, pip_check = build_commands(
        requirements_path,
        include_rag=include_rag,
    )
    _, declared_requirements = load_project_requirements(requirements_path)
    if dry_run:
        for index, (label, command) in enumerate(commands, start=1):
            print(f"PHASE {index}/{len(commands)} [{label}]: {shlex.join(command)}")
        print("VERIFY: " + shlex.join(pip_check))
        return

    _validate_wheel_runtime()
    for index, (label, command) in enumerate(commands, start=1):
        print(f"[PHASE {index}/{len(commands)}] {label}", flush=True)
        try:
            _run(command)
        except subprocess.CalledProcessError as exc:
            if label == "official FlashAttention wheel":
                raise RuntimeError(
                    "The official precompiled FlashAttention wheel could not be "
                    "installed. Source compilation is disabled and was not attempted."
                ) from exc
            raise

    project_distributions = list(declared_requirements)
    if include_rag:
        project_distributions.extend(
            _overlay_distribution_names(COMPLETE_RAG_REQUIREMENTS)
        )
    _run_project_pip_check(pip_check, project_distributions)
    report = verify_environment(require_cuda=require_cuda)
    if include_rag:
        _verify_complete_rag_imports()
    print("[INSTALLATION COMPLETE]", flush=True)
    print(report, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install the verified Torch 2.8/vLLM 0.10.2 stack and the official "
            "FlashAttention wheel without compiling FlashAttention."
        )
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        default=DEFAULT_REQUIREMENTS,
        help="Unified requirements file (default: project requirements.txt).",
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help=(
            "Install the complete RAG evaluation and ColBERT overlays together "
            "in one dependency resolution and verify both import stacks."
        ),
    )
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Do not require a visible CUDA GPU during final verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the installation commands without changing the environment.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    install(
        requirements_path=args.requirements,
        dry_run=args.dry_run,
        require_cuda=not args.allow_no_cuda,
        include_rag=args.with_rag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
