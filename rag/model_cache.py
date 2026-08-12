"""Deterministic cache identities for temporary merged model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping


CACHE_IDENTITY_FILENAME = "cache_identity.json"
CACHE_IDENTITY_SCHEMA_VERSION = 1

_ADAPTER_PATTERNS = (
    "adapter_config.json",
    "adapter_model*.safetensors",
    "adapter_model*.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
)

_BASE_MODEL_PATTERNS = (
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "pytorch_model*.bin",
)

_TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)

_HASH_CONTENT_LIMIT_BYTES = 16 * 1024 * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_files(directory: Path, patterns: Iterable[str]) -> list[Path]:
    matches: dict[str, Path] = {}
    for pattern in patterns:
        for path in directory.glob(pattern):
            if path.is_file():
                matches[path.name] = path
    return [matches[name] for name in sorted(matches)]


def _file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    identity: dict[str, Any] = {
        "name": path.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if stat.st_size <= _HASH_CONTENT_LIMIT_BYTES:
        identity["sha256"] = _sha256_file(path)
    return identity


def _artifact_identity(value: str | os.PathLike[str], patterns: Iterable[str]) -> dict[str, Any]:
    raw_value = str(value)
    path = Path(raw_value).expanduser()
    if not path.exists():
        return {
            "kind": "identifier",
            "value": raw_value,
        }

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "file",
            "path": str(resolved),
            "file": _file_identity(resolved),
        }

    return {
        "kind": "directory",
        "path": str(resolved),
        "files": [_file_identity(item) for item in _matching_files(resolved, patterns)],
    }


def build_resized_base_identity(
    *,
    original_base_model_path: str,
    tokenizer_path: str,
    target_vocab_size: int,
) -> dict[str, Any]:
    """Build the cache identity for a temporary resized base model."""
    return {
        "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "artifact_type": "resized_base_model",
        "base_model": _artifact_identity(original_base_model_path, _BASE_MODEL_PATTERNS),
        "tokenizer": _artifact_identity(tokenizer_path, _TOKENIZER_PATTERNS),
        "target_vocab_size": int(target_vocab_size),
    }


def build_merged_model_identity(
    *,
    original_base_model_path: str,
    adapter_path: str,
    tokenizer_path: str,
    target_vocab_size: int,
) -> dict[str, Any]:
    """Build the cache identity for a base model merged with a PEFT adapter."""
    return {
        "schema_version": CACHE_IDENTITY_SCHEMA_VERSION,
        "artifact_type": "merged_peft_model",
        "base_model": _artifact_identity(original_base_model_path, _BASE_MODEL_PATTERNS),
        "adapter": _artifact_identity(adapter_path, _ADAPTER_PATTERNS),
        "tokenizer": _artifact_identity(tokenizer_path, _TOKENIZER_PATTERNS),
        "target_vocab_size": int(target_vocab_size),
    }


def cache_key(identity: Mapping[str, Any], *, prefix: str) -> str:
    """Return a short deterministic directory name for an identity payload."""
    canonical = json.dumps(
        identity,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:24]
    return f"{prefix}-{digest}"


def write_cache_identity(directory: str | os.PathLike[str], identity: Mapping[str, Any]) -> Path:
    """Atomically write a human-readable identity manifest into a cache directory."""
    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)
    target = directory_path / CACHE_IDENTITY_FILENAME
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
