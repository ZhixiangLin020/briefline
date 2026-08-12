"""Version and binary checks for the unified A100/A800 GPU environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"
EXPECTED_PYTHON = (3, 12)

REQUIRED_VERSIONS: Dict[str, str] = {
    "torch": "2.8.0",
    "torchvision": "0.23.0",
    "torchaudio": "2.8.0",
    "vllm": "0.10.2",
    "flash-attn": "2.8.3.post1",
    "transformers": "4.56.2",
    "huggingface-hub": "0.34.4",
}


def configure_pytorch_backend() -> None:
    """Select the project's PyTorch backend before importing ML libraries."""

    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"


def installed_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _without_local_suffix(version: str) -> str:
    """Accept CUDA/local wheel suffixes while enforcing the public version."""

    return str(version).split("+", 1)[0]


def version_issues(
    required: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Optional[str]]]:
    expected_versions = REQUIRED_VERSIONS if required is None else required
    issues: Dict[str, Dict[str, Optional[str]]] = {}
    for distribution, expected in expected_versions.items():
        actual = installed_version(distribution)
        if actual is None or _without_local_suffix(actual) != expected:
            issues[distribution] = {"expected": expected, "actual": actual}
    return issues


def _probe_source(*, require_cuda: bool) -> str:
    return f"""
import json
import torch
import flash_attn
from flash_attn import flash_attn_func

cuda_available = bool(torch.cuda.is_available())
if {require_cuda!r} and not cuda_available:
    raise RuntimeError("CUDA is not available to the unified GPU environment")

def run_flashattention_kernel():
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, q, q, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    if out.shape != q.shape:
        raise RuntimeError(f"Unexpected FlashAttention output shape: {{out.shape}}")

flash_before = (getattr(flash_attn, "__version__", "unknown"), flash_attn.__file__)
if cuda_available:
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected Torch CUDA 12.8, found {{torch.version.cuda}}")
    if not bool(torch._C._GLIBCXX_USE_CXX11_ABI):
        raise RuntimeError("Torch must use CXX11 ABI TRUE for the selected wheel")
    run_flashattention_kernel()

import vllm
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, RequestLogitsProcessor

if cuda_available:
    run_flashattention_kernel()
flash_after = (getattr(flash_attn, "__version__", "unknown"), flash_attn.__file__)
if flash_before != flash_after:
    raise RuntimeError(f"FlashAttention changed after importing vLLM: {{flash_before}} -> {{flash_after}}")

payload = {{
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
    "cuda_available": cuda_available,
    "flash_attn": flash_after[0],
    "flash_attn_path": flash_after[1],
    "flash_attn_func": callable(flash_attn_func),
    "vllm": getattr(vllm, "__version__", "unknown"),
    "vllm_v1_logits_api": bool(AdapterLogitsProcessor and RequestLogitsProcessor),
}}
if cuda_available:
    payload["gpu_name"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
print(json.dumps(payload, sort_keys=True))
"""


def binary_probe(*, require_cuda: bool = True) -> Dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-c", _probe_source(require_cuda=require_cuda)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "The Torch/vLLM/FlashAttention compatibility probe failed. "
            "Run `python scripts/install_dependencies.py`; it uses the official wheel "
            "and never falls back to source compilation.\n" + details
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("The binary probe returned no result.")
    return json.loads(lines[-1])


def verify_environment(*, require_cuda: bool = True) -> Dict[str, object]:
    configure_pytorch_backend()
    if sys.version_info[:2] != EXPECTED_PYTHON:
        raise RuntimeError(
            "The selected FlashAttention wheel requires Python 3.12; "
            f"found Python {sys.version_info.major}.{sys.version_info.minor}."
        )
    issues = version_issues()
    if issues:
        formatted = ", ".join(
            f"{name}: expected {item['expected']}, found {item['actual'] or 'missing'}"
            for name, item in sorted(issues.items())
        )
        raise RuntimeError(
            "The active Python environment does not match requirements.txt: "
            f"{formatted}. Run `python scripts/install_dependencies.py`."
        )
    probe = binary_probe(require_cuda=require_cuda)
    return {
        "status": "ok",
        "python": sys.executable,
        "requirements": str(REQUIREMENTS_PATH),
        "versions": dict(REQUIRED_VERSIONS),
        "binary_probe": probe,
    }


def ensure_runtime_compatible(stage: str) -> Dict[str, object]:
    try:
        report = verify_environment(require_cuda=True)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot start the {stage} stage because the unified GPU "
            f"environment check failed. {exc}"
        ) from exc
    print(
        "[UNIFIED ENVIRONMENT OK] "
        + json.dumps(report["binary_probe"], ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the unified Torch/vLLM/FlashAttention environment."
    )
    parser.add_argument(
        "--allow-no-cuda",
        action="store_true",
        help="Run import/version checks without requiring a visible CUDA GPU.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = verify_environment(require_cuda=not args.allow_no_cuda)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
