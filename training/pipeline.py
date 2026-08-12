"""End-to-end training assembly without changing the recorded algorithm."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Dict

from .config import (
    ORIGINAL_MODEL_CONFIG_SIGNATURE,
    TrainingRunConfig,
    validate_original_model_config,
)
from .data import (
    TrainingDataBundle,
    load_training_data,
    validate_bundle_tokenizer_compatibility,
    validate_manifest_tokenizer_for_training,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_run_manifest(
    config: TrainingRunConfig,
    bundle: TrainingDataBundle,
) -> Dict[str, Any]:
    cfg = config.normalized()
    algorithm = {
        "training_arguments": cfg.training_arguments_kwargs(),
        "sft_config": cfg.sft_config_kwargs(samples_per_epoch=bundle.samples_per_epoch),
        "peft_formula": {
            "method": "adalora",
            "init_r": 128,
            "target_r": 90,
            "alpha": 128,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "tinit": "int(total_optimizer_steps * 0.4)",
            "tfinal": "int(total_optimizer_steps * 0.2)",
            "deltaT": 250,
        },
        "warmup": {
            "forward_steps": 2,
            "backward_steps": 1,
            "pick_longest_from": 10_000,
            "warm_optimizer_state": True,
        },
        "top_k": 4,
        "base_model_config_signature": dict(ORIGINAL_MODEL_CONFIG_SIGNATURE),
    }
    digest_input = json.dumps(_jsonable(algorithm), sort_keys=True, separators=(",", ":"))
    reproduction = cfg.reproduction_eligibility(bundle.counts)
    missing_manifests = [
        name
        for name, manifest in bundle.source_manifests.items()
        if manifest is None
    ]
    provenance_issues: Dict[str, Any] = {}
    if missing_manifests:
        provenance_issues["missing_dataset_manifests"] = missing_manifests
    missing_version_info = [
        name
        for name, manifest in bundle.source_manifests.items()
        if manifest is not None and not manifest.get("package_versions")
    ]
    if missing_version_info:
        provenance_issues["missing_package_versions"] = missing_version_info
    missing_tokenizer_contract = [
        name
        for name, manifest in bundle.source_manifests.items()
        if manifest is not None and not manifest.get("tokenizer_contract")
    ]
    if missing_tokenizer_contract:
        provenance_issues["missing_tokenizer_contract"] = missing_tokenizer_contract
    if provenance_issues:
        reproduction["eligible"] = False
        reproduction["provenance_issues"] = provenance_issues
    if cfg.dry_run:
        reproduction["eligible"] = False
        reproduction.setdefault("provenance_issues", {})[
            "model_config_validation"
        ] = "not performed in dry_run"

    return {
        "run_mode": bundle.run_mode,
        "smoke_test_warning": (
            "NOT A REPRODUCTION RUN" if cfg.data.smoke_test else None
        ),
        "data": bundle.summary(),
        "experiment_reproduction": reproduction,
        "model_name_or_path": cfg.model_name_or_path,
        "roberta_path": cfg.roberta_path,
        "output_dir": str(cfg.output_dir),
        "best_model_dir": str(cfg.best_model_dir),
        "resume_from_checkpoint": (
            None
            if cfg.resume_from_checkpoint is None
            else str(cfg.resume_from_checkpoint)
        ),
        "algorithm": algorithm,
        "algorithm_config_hash": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
    }


def _build_training_arguments(config: TrainingRunConfig):
    from transformers import TrainingArguments

    kwargs = config.training_arguments_kwargs()
    parameters = inspect.signature(TrainingArguments).parameters
    eval_key = "evaluation_strategy" if "evaluation_strategy" in parameters else "eval_strategy"
    kwargs[eval_key] = "steps"
    return TrainingArguments(**kwargs)


def _apply_original_tokenizer_padding(tokenizer) -> None:
    """Restore the unconditional padding assignment used by the notebook."""

    tokenizer.pad_token = tokenizer.eos_token


def _validate_resume_preflight(config: TrainingRunConfig) -> None:
    if config.resume_from_checkpoint is None:
        return
    if not config.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            "resume_from_checkpoint must point to an existing Trainer checkpoint "
            f"directory: {config.resume_from_checkpoint}"
        )
    best_k_manifest = config.best_model_dir / "best_k_metrics.json"
    if not best_k_manifest.is_file():
        raise FileNotFoundError(
            "Strict resume requires the historical top-k manifest so candidates "
            f"from before the checkpoint are not lost: {best_k_manifest}"
        )


def run_training(config: TrainingRunConfig) -> Dict[str, Any]:
    cfg = config.normalized()
    _validate_resume_preflight(cfg)
    bundle = load_training_data(cfg.data)
    validate_manifest_tokenizer_for_training(
        bundle.source_manifests,
        model_name_or_path=cfg.model_name_or_path,
    )

    tokenizer = None
    model_config = None
    if not cfg.dry_run:
        # Heavy dependencies remain lazy for data-only commands and dry runs.
        import torch
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        _apply_original_tokenizer_padding(tokenizer)
        model_config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        validate_original_model_config(model_config)
        validate_bundle_tokenizer_compatibility(
            bundle,
            tokenizer,
            model_config=model_config,
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.best_model_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(cfg, bundle)
    manifest_path = cfg.output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if cfg.data.smoke_test:
        print("RUN MODE: SMOKE TEST — NOT A REPRODUCTION RUN", flush=True)
    elif not manifest["experiment_reproduction"]["eligible"]:
        print(
            "RUN MODE: FULL TRAINING WITH A NON-ORIGINAL DATA SNAPSHOT — "
            "NOT AN EXACT EXPERIMENT REPRODUCTION",
            flush=True,
        )
    else:
        print("RUN MODE: ORIGINAL EXPERIMENT REPRODUCTION CANDIDATE", flush=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    if cfg.dry_run:
        return {
            "status": "dry_run",
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        }

    from .callbacks import EvalTableSaveCallback, SaveBestPeftCallback
    from .collator import SFTAnswerOnlyCollator
    from .modeling import (
        AdaLoraRankAllocatorCallback,
        PeftSpec,
        apply_peft_for_causal_lm,
        build_base_model,
        infer_total_optim_steps,
        print_trainable_parameters,
    )
    from .trainer import SFTConfig, SFTTrainer
    from .warmup import trainer_warmup

    collator = SFTAnswerOnlyCollator(tokenizer)
    training_args = _build_training_arguments(cfg)
    sft_config = SFTConfig(
        **cfg.sft_config_kwargs(samples_per_epoch=bundle.samples_per_epoch)
    )

    total_steps = infer_total_optim_steps(
        training_args,
        train_len=sft_config.samples_per_epoch,
    )
    peft_spec = PeftSpec(**cfg.peft_spec_kwargs(total_steps=total_steps))

    model = build_base_model(
        cfg.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        qlora_bits=None,
    )
    model = apply_peft_for_causal_lm(
        model,
        tokenizer=tokenizer,
        peft=peft_spec,
        is_kbit_training=False,
        gradient_checkpointing=True,
        train_embeddings_for_new_tokens=False,
    )

    eval_log_dir = cfg.best_model_dir / "logs"
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_pools=bundle.train_pools,
        eval_dataset=bundle.eval_pools,
        data_collator=collator,
        processing_class=tokenizer,
        sft_config=sft_config,
        callbacks=[
            EvalTableSaveCallback(output_dir=str(eval_log_dir)),
            SaveBestPeftCallback(
                output_dir=str(cfg.best_model_dir),
                metric_name="auto",
                tokenizer=tokenizer,
                greater_is_better=True,
                top_k=4,
                resume_existing=cfg.resume_from_checkpoint is not None,
            ),
        ],
    )
    trainer.add_callback(AdaLoraRankAllocatorCallback(model=trainer.model))

    print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if not hasattr(trainer, "current_gradient_accumulation_steps"):
        trainer.current_gradient_accumulation_steps = max(
            1,
            int(trainer.args.gradient_accumulation_steps),
        )

    trainer_warmup(
        trainer,
        warmup_forward_steps=2,
        warmup_backward_steps=1,
        pick_longest_from=10_000,
        also_warmup_optimizer_state=True,
    )
    model.config.use_cache = False
    train_output = trainer.train(
        resume_from_checkpoint=(
            None
            if cfg.resume_from_checkpoint is None
            else str(cfg.resume_from_checkpoint)
        )
    )
    trainer._run_train_preview()

    result = {
        "status": "completed",
        "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
        "total_optimizer_steps": int(total_steps),
        "train_metrics": _jsonable(getattr(train_output, "metrics", {})),
        "manifest_path": str(manifest_path),
        "best_k_manifest": str(cfg.best_model_dir / "best_k_metrics.json"),
    }
    result_path = cfg.output_dir / "training_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["result_path"] = str(result_path)
    return result
