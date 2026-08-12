"""Load and adapt the two trainer-ready Hugging Face datasets.

This module preserves the sampling calls and seed offsets used by the original
notebook.  It does not tokenize, concatenate, or overwrite either input.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .config import TrainingDataConfig


REQUIRED_SPLITS: Tuple[str, ...] = ("train", "validation", "test")
REQUIRED_TRAINING_COLUMNS: Tuple[str, ...] = (
    "input_ids",
    "attention_mask",
    "labels",
    "loss_weights",
)
OPTIONAL_TRAINING_COLUMNS: Tuple[str, ...] = ("full_labels",)


@dataclass
class TrainingDataBundle:
    train_pools: Dict[str, Any]
    eval_pools: Dict[str, Any]
    test_pools: Dict[str, Any]
    counts: Dict[str, int]
    samples_per_epoch: int
    run_mode: str
    seed: int
    source_paths: Dict[str, str]
    source_manifests: Dict[str, Optional[Dict[str, Any]]]

    def summary(self) -> Dict[str, Any]:
        return {
            "run_mode": self.run_mode,
            "seed": self.seed,
            "samples_per_epoch": self.samples_per_epoch,
            "counts": dict(self.counts),
            "source_paths": dict(self.source_paths),
            "source_manifests": {
                name: {
                    "present": manifest is not None,
                    "schema_version": (
                        None if manifest is None else manifest.get("schema_version")
                    ),
                    "dataset": None if manifest is None else manifest.get("dataset"),
                    "tokenizer_name": (
                        None if manifest is None else manifest.get("tokenizer_name")
                    ),
                    "prepare_config_hash": (
                        None
                        if manifest is None
                        else manifest.get("prepare_config_hash")
                    ),
                    "package_versions": (
                        None
                        if manifest is None
                        else manifest.get("package_versions")
                    ),
                    "tokenizer_contract": (
                        None
                        if manifest is None
                        else manifest.get("tokenizer_contract")
                    ),
                }
                for name, manifest in self.source_manifests.items()
            },
        }


def _require_datasets_package() -> None:
    if importlib.util.find_spec("datasets") is None:
        raise RuntimeError(
            "The training data loader requires Hugging Face datasets. "
            "Install the project's data dependencies before loading data from disk."
        )


def _check_dataset_path(path: Path, dataset_name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} dataset path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(
            f"{dataset_name} must point to an extracted Hugging Face dataset directory, "
            f"not an archive or file: {path}"
        )


def _read_manifest(dataset_path: Path) -> Optional[Dict[str, Any]]:
    path = dataset_path / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_dataset_dict_shape(ds_dict: Mapping[str, Any], dataset_name: str) -> None:
    missing = [split for split in REQUIRED_SPLITS if split not in ds_dict]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required splits {missing}; "
            f"actual splits: {list(ds_dict.keys())}"
        )


def _require_integer_tokens(
    values,
    *,
    field_name: str,
    dataset_name: str,
    row_index: int,
) -> None:
    for position, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"{dataset_name} row={row_index} field={field_name!r} "
                f"position={position} must be an integer token value; "
                f"got {type(value).__name__}"
            )


def validate_trainer_split(
    ds,
    *,
    dataset_name: str,
    vocab_size: Optional[int] = None,
    max_sequence_length: Optional[int] = None,
) -> None:
    """Validate every row in the split used by this run."""

    columns = list(getattr(ds, "column_names", []))
    missing = [name for name in REQUIRED_TRAINING_COLUMNS if name not in columns]
    if missing:
        raise ValueError(
            f"{dataset_name} is missing trainer columns {missing}; actual columns: {columns}"
        )
    if len(ds) == 0:
        raise ValueError(f"{dataset_name} is empty")

    if vocab_size is not None and int(vocab_size) <= 0:
        raise ValueError(f"vocab_size must be > 0, got {vocab_size}")
    if max_sequence_length is not None and int(max_sequence_length) <= 0:
        raise ValueError(
            f"max_sequence_length must be > 0, got {max_sequence_length}"
        )

    for row_index in range(len(ds)):
        row = ds[row_index]
        try:
            lengths = {name: len(row[name]) for name in REQUIRED_TRAINING_COLUMNS}
        except TypeError as exc:
            raise ValueError(
                f"{dataset_name} row={row_index} token fields must be sequences"
            ) from exc
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"{dataset_name} row={row_index} has misaligned token fields: {lengths}"
            )
        sequence_length = lengths["input_ids"]
        if sequence_length == 0:
            raise ValueError(f"{dataset_name} row={row_index} is an empty sequence")
        if (
            max_sequence_length is not None
            and sequence_length > int(max_sequence_length)
        ):
            raise ValueError(
                f"{dataset_name} row={row_index} has sequence length "
                f"{sequence_length}, exceeding model limit {max_sequence_length}"
            )

        _require_integer_tokens(
            row["input_ids"],
            field_name="input_ids",
            dataset_name=dataset_name,
            row_index=row_index,
        )
        _require_integer_tokens(
            row["attention_mask"],
            field_name="attention_mask",
            dataset_name=dataset_name,
            row_index=row_index,
        )
        _require_integer_tokens(
            row["labels"],
            field_name="labels",
            dataset_name=dataset_name,
            row_index=row_index,
        )

        for position, token_id in enumerate(row["input_ids"]):
            token_id = int(token_id)
            if token_id < 0 or (
                vocab_size is not None and token_id >= int(vocab_size)
            ):
                allowed = (
                    "non-negative"
                    if vocab_size is None
                    else f"inside tokenizer vocabulary [0, {vocab_size})"
                )
                raise ValueError(
                    f"{dataset_name} row={row_index} input_ids[{position}]={token_id} "
                    f"must be {allowed}"
                )
        for position, mask_value in enumerate(row["attention_mask"]):
            if int(mask_value) not in (0, 1):
                raise ValueError(
                    f"{dataset_name} row={row_index} attention_mask[{position}]="
                    f"{mask_value} is not 0 or 1"
                )
        for position, label in enumerate(row["labels"]):
            label = int(label)
            if label == -100:
                continue
            if label < 0 or (vocab_size is not None and label >= int(vocab_size)):
                raise ValueError(
                    f"{dataset_name} row={row_index} labels[{position}]={label} "
                    f"is neither -100 nor a valid tokenizer ID"
                )
        for position, value in enumerate(row["loss_weights"]):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(
                    f"{dataset_name} row={row_index} loss_weights[{position}] "
                    f"must be numeric; got {type(value).__name__}"
                )
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"{dataset_name} row={row_index} contains non-finite "
                    f"loss_weights[{position}]"
                )


def _select_with_original_rng(ds, n: int, *, seed: int, replacement: bool = False):
    """Exact sampling primitive from the original notebook."""

    total = len(ds)
    n = int(n)
    if n < 0:
        raise ValueError("sample size cannot be negative")
    if n == 0:
        return ds.select([])
    if total == 0:
        raise ValueError("cannot sample from an empty dataset")
    if n > total and not replacement:
        raise ValueError(f"requested {n} samples from a dataset containing only {total}")

    rng = np.random.default_rng(seed)
    if replacement:
        indices = rng.choice(total, size=n, replace=True).tolist()
    else:
        indices = rng.permutation(total)[:n].tolist()
    return ds.select(indices)


def _keep_training_columns(ds):
    keep = [
        name
        for name in REQUIRED_TRAINING_COLUMNS + OPTIONAL_TRAINING_COLUMNS
        if name in ds.column_names
    ]
    remove = [name for name in ds.column_names if name not in keep]
    return ds.remove_columns(remove) if remove else ds


def _requested_count(ds, configured: Optional[int]) -> int:
    if configured is None:
        return len(ds)
    if int(configured) > len(ds):
        raise ValueError(
            f"requested {configured} samples from a split containing only {len(ds)}; "
            "smoke subsets use sampling without replacement"
        )
    return int(configured)


def _prepare_source_splits(
    ds_dict: Mapping[str, Any],
    *,
    source_name: str,
    config: TrainingDataConfig,
    source_offset: int,
) -> Tuple[Any, Any, Any]:
    _validate_dataset_dict_shape(ds_dict, source_name)

    train_n = _requested_count(ds_dict["train"], config.max_train_samples_per_dataset)
    validation_n = _requested_count(
        ds_dict["validation"], config.max_validation_samples_per_dataset
    )
    test_n = _requested_count(ds_dict["test"], config.max_test_samples_per_dataset)

    # Preserve the original seed offsets exactly:
    # cnn_dm train sample/shuffle +101/+201; kptime +102/+202.
    train = _select_with_original_rng(
        ds_dict["train"], train_n, seed=config.seed + 101 + source_offset
    ).shuffle(seed=config.seed + 201 + source_offset)
    validation = _select_with_original_rng(
        ds_dict["validation"],
        validation_n,
        seed=config.seed + 401 + source_offset,
    )
    test = _select_with_original_rng(
        ds_dict["test"],
        test_n,
        seed=config.seed + 501 + source_offset,
    )

    prepared = (
        _keep_training_columns(train),
        _keep_training_columns(validation),
        _keep_training_columns(test),
    )
    for split_name, split in zip(REQUIRED_SPLITS, prepared):
        validate_trainer_split(
            split,
            dataset_name=f"{source_name}/{split_name}",
        )
    return prepared


def _manifest_dataset_name(source_name: str) -> str:
    return "kptimes" if source_name == "kptime" else source_name


def _validate_source_manifest(
    manifest: Optional[Dict[str, Any]],
    ds_dict: Mapping[str, Any],
    *,
    source_name: str,
) -> None:
    if manifest is None:
        return
    if not isinstance(manifest, dict):
        raise ValueError(f"{source_name} manifest must contain a JSON object")

    required = (
        "schema_version",
        "dataset",
        "tokenizer_name",
        "prepare_parameters",
        "prepare_config_hash",
        "splits",
    )
    missing = [name for name in required if manifest.get(name) is None]
    if missing:
        raise ValueError(
            f"{source_name} manifest is missing required provenance fields: {missing}"
        )
    expected_name = _manifest_dataset_name(source_name)
    if str(manifest["dataset"]).strip().lower() != expected_name:
        raise ValueError(
            f"{source_name} path contains a manifest for dataset "
            f"{manifest['dataset']!r}, expected {expected_name!r}"
        )

    split_manifest = manifest["splits"]
    if not isinstance(split_manifest, dict):
        raise ValueError(f"{source_name} manifest field 'splits' must be an object")
    for split in REQUIRED_SPLITS:
        record = split_manifest.get(split)
        if not isinstance(record, dict) or "rows" not in record:
            raise ValueError(
                f"{source_name} manifest is missing row metadata for split {split!r}"
            )
        recorded_rows = int(record["rows"])
        actual_rows = len(ds_dict[split])
        if recorded_rows != actual_rows:
            raise ValueError(
                f"{source_name}/{split} manifest row count is {recorded_rows}, "
                f"but load_from_disk returned {actual_rows}"
            )


def _check_manifest_compatibility(manifests: Mapping[str, Optional[Dict[str, Any]]]) -> None:
    present = [manifest for manifest in manifests.values() if manifest is not None]
    if len(present) < 2:
        return
    tokenizer_names = {manifest.get("tokenizer_name") for manifest in present}
    tokenizer_names.discard(None)
    if len(tokenizer_names) > 1:
        raise ValueError(
            f"The two prepared datasets record incompatible tokenizers: {sorted(tokenizer_names)}"
        )


def _tokenizer_name_matches(recorded: str, configured: str) -> bool:
    recorded = str(recorded).rstrip("/\\")
    configured = str(configured).rstrip("/\\")
    if recorded == configured:
        return True
    return Path(recorded).name == Path(configured).name


def validate_manifest_tokenizer_for_training(
    manifests: Mapping[str, Optional[Dict[str, Any]]],
    *,
    model_name_or_path: str,
) -> None:
    for source_name, manifest in manifests.items():
        if manifest is None:
            continue
        recorded = manifest.get("tokenizer_name")
        if Path(model_name_or_path).is_dir():
            # A local directory may have any user-chosen name. Its actual
            # vocabulary/template contract is checked after loading.
            continue
        if not recorded or not _tokenizer_name_matches(recorded, model_name_or_path):
            raise ValueError(
                f"{source_name} was prepared with tokenizer {recorded!r}, but training "
                f"uses {model_name_or_path!r}"
            )


def _sha256_json(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_loaded_tokenizer_contract(
    manifests: Mapping[str, Optional[Dict[str, Any]]],
    tokenizer,
) -> None:
    actual_vocab = None
    actual_vocab_hash = None
    actual_chat_template = getattr(tokenizer, "chat_template", None)
    actual_chat_hash = (
        None
        if actual_chat_template is None
        else hashlib.sha256(str(actual_chat_template).encode("utf-8")).hexdigest()
    )
    for source_name, manifest in manifests.items():
        if manifest is None:
            continue
        contract = manifest.get("tokenizer_contract")
        if contract is None:
            continue
        if not isinstance(contract, dict):
            raise ValueError(
                f"{source_name} manifest tokenizer_contract must be an object"
            )

        mismatches: Dict[str, Dict[str, Any]] = {}
        expected_vocab_size = contract.get("vocab_size")
        if expected_vocab_size is not None and int(expected_vocab_size) != len(tokenizer):
            mismatches["vocab_size"] = {
                "prepared": int(expected_vocab_size),
                "training": int(len(tokenizer)),
            }

        expected_vocab_hash = contract.get("vocab_sha256")
        if expected_vocab_hash is not None:
            if actual_vocab is None:
                actual_vocab = tokenizer.get_vocab()
                actual_vocab_hash = _sha256_json(actual_vocab)
            if str(expected_vocab_hash) != actual_vocab_hash:
                mismatches["vocab_sha256"] = {
                    "prepared": str(expected_vocab_hash),
                    "training": actual_vocab_hash,
                }

        expected_chat_hash = contract.get("chat_template_sha256")
        if expected_chat_hash != actual_chat_hash:
            mismatches["chat_template_sha256"] = {
                "prepared": expected_chat_hash,
                "training": actual_chat_hash,
            }

        for field_name in ("bos_token_id", "eos_token_id", "unk_token_id"):
            expected = contract.get(field_name)
            actual = getattr(tokenizer, field_name, None)
            if expected != actual:
                mismatches[field_name] = {
                    "prepared": expected,
                    "training": actual,
                }

        if mismatches:
            raise ValueError(
                f"{source_name} tokenizer contract is incompatible with the "
                f"tokenizer loaded for training: {mismatches}"
            )


def validate_bundle_tokenizer_compatibility(
    bundle: TrainingDataBundle,
    tokenizer,
    *,
    model_config=None,
) -> None:
    """Validate every selected row against the tokenizer used for training."""

    _validate_loaded_tokenizer_contract(bundle.source_manifests, tokenizer)
    vocab_size = int(len(tokenizer))
    raw_limits = [getattr(tokenizer, "model_max_length", None)]
    if model_config is not None:
        raw_limits.extend(
            getattr(model_config, name, None)
            for name in (
                "max_position_embeddings",
                "max_sequence_length",
                "seq_length",
                "n_positions",
            )
        )
    meaningful_limits = [
        int(value)
        for value in raw_limits
        if isinstance(value, Integral) and 0 < int(value) < 1_000_000_000
    ]
    # Transformers uses enormous sentinel values for tokenizers without a
    # meaningful maximum. Use the strictest real tokenizer/model limit.
    max_sequence_length = min(meaningful_limits) if meaningful_limits else None

    for pool_group, pools in (
        ("train", bundle.train_pools),
        ("validation", bundle.eval_pools),
        ("test", bundle.test_pools),
    ):
        for source_name, split in pools.items():
            validate_trainer_split(
                split,
                dataset_name=f"{source_name}/{pool_group}",
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
            )


def build_training_data_bundle(
    cnn_dm,
    kptimes,
    config: TrainingDataConfig,
    *,
    source_manifests: Optional[Mapping[str, Optional[Dict[str, Any]]]] = None,
) -> TrainingDataBundle:
    """Build separate task pools without changing the source datasets."""

    cfg = config.normalized()
    manifests = dict(source_manifests or {"cnn_dm": None, "kptime": None})
    _validate_dataset_dict_shape(cnn_dm, "cnn_dm")
    _validate_dataset_dict_shape(kptimes, "kptime")
    _validate_source_manifest(
        manifests.get("cnn_dm"),
        cnn_dm,
        source_name="cnn_dm",
    )
    _validate_source_manifest(
        manifests.get("kptime"),
        kptimes,
        source_name="kptime",
    )
    _check_manifest_compatibility(manifests)

    cnn_train, cnn_eval, cnn_test = _prepare_source_splits(
        cnn_dm,
        source_name="cnn_dm",
        config=cfg,
        source_offset=0,
    )
    kpt_train, kpt_eval, kpt_test = _prepare_source_splits(
        kptimes,
        source_name="kptime",
        config=cfg,
        source_offset=1,
    )

    counts = {
        "train_cnn_dm": len(cnn_train),
        "train_kptime": len(kpt_train),
        "validation_cnn_dm": len(cnn_eval),
        "validation_kptime": len(kpt_eval),
        "test_cnn_dm": len(cnn_test),
        "test_kptime": len(kpt_test),
    }
    samples_per_epoch = (
        int(cfg.samples_per_epoch)
        if cfg.samples_per_epoch is not None
        else len(cnn_train)
    )

    return TrainingDataBundle(
        train_pools={"cnn_dm": cnn_train, "kptime": kpt_train},
        eval_pools={"cnn_dm": cnn_eval, "kptime": kpt_eval},
        test_pools={"cnn_dm": cnn_test, "kptime": kpt_test},
        counts=counts,
        samples_per_epoch=samples_per_epoch,
        run_mode=cfg.run_mode,
        seed=cfg.seed,
        source_paths={
            "cnn_dm": str(cfg.cnn_dm_dataset),
            "kptime": str(cfg.kptimes_dataset),
        },
        source_manifests=manifests,
    )


def load_training_data(config: TrainingDataConfig) -> TrainingDataBundle:
    """Load two explicit HF DatasetDict paths and create training views."""

    cfg = config.normalized()
    _check_dataset_path(cfg.cnn_dm_dataset, "cnn_dm")
    _check_dataset_path(cfg.kptimes_dataset, "kptimes")
    _require_datasets_package()

    from datasets import DatasetDict, load_from_disk

    cnn_dm = load_from_disk(str(cfg.cnn_dm_dataset))
    kptimes = load_from_disk(str(cfg.kptimes_dataset))
    for name, value in (("cnn_dm", cnn_dm), ("kptimes", kptimes)):
        if not isinstance(value, DatasetDict):
            raise TypeError(
                f"{name} must be a Hugging Face DatasetDict with train/validation/test; "
                f"got {type(value).__name__}"
            )

    manifests = {
        "cnn_dm": _read_manifest(cfg.cnn_dm_dataset),
        "kptime": _read_manifest(cfg.kptimes_dataset),
    }
    return build_training_data_bundle(
        cnn_dm,
        kptimes,
        cfg,
        source_manifests=manifests,
    )
