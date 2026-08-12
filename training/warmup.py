"""Non-consuming CUDA/optimizer warm-up preserved from the original run."""

import torch
import math


def _get_train_pools_from_trainer(trainer):
    """
    Retrieve the original train_pools from the updated SFTTrainer.
    Fall back to trainer.train_dataset when train_pools is not enabled.
    """
    wrapper = getattr(trainer, "_epoch_ratio_train_dataset", None)

    if wrapper is not None and hasattr(wrapper, "train_pools"):
        return dict(wrapper.train_pools)

    if getattr(trainer, "train_dataset", None) is not None:
        return {"train": trainer.train_dataset}

    raise ValueError("Cannot find train_pools or trainer.train_dataset for warmup.")


def _filter_warmup_feature(row):
    """
    Keep only fields required by the collator and compute_loss.
    Do not pass non-tensor fields such as PromptText, AnswerText, or source.
    """
    keep_cols = {
        "input_ids",
        "attention_mask",
        "labels",
        "loss_weights",
        "full_labels",
    }
    return {k: v for k, v in dict(row).items() if k in keep_cols}


def _build_non_consuming_warmup_batch(
    trainer,
    *,
    pick_longest_from: int = 32,
    batch_size: int | None = None,
):
    """
    Do not use trainer.get_train_dataloader().
    This avoids calling epoch_ratio_sampler.__iter__(), advancing sampler.epoch,
    or updating usage_counts.
    """
    train_pools = _get_train_pools_from_trainer(trainer)

    if trainer.data_collator is None:
        raise ValueError("trainer.data_collator is required for warmup.")

    if batch_size is None:
        batch_size = getattr(trainer, "_train_batch_size", None)

        if batch_size is None:
            batch_size = getattr(trainer.args, "train_batch_size", None)

        if batch_size is None:
            batch_size = getattr(trainer.args, "per_device_train_batch_size", 1)

    batch_size = int(batch_size)
    pick_longest_from = int(pick_longest_from)

    source_names = list(train_pools.keys())
    per_source_quota = max(
        1,
        math.ceil(pick_longest_from / max(1, len(source_names))),
    )

    candidates = []

    for source_name in source_names:
        ds = train_pools[source_name]
        n = min(len(ds), per_source_quota)

        for i in range(n):
            row = _filter_warmup_feature(ds[int(i)])

            if "input_ids" not in row:
                continue

            seq_len = len(row["input_ids"])
            candidates.append((seq_len, source_name, i, row))

    if not candidates:
        raise ValueError("No valid warmup candidates found. Expected rows with input_ids.")

    # Select a relatively long batch to better warm up peak GPU memory and kernels.
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = candidates[:batch_size]

    features = [row for _, _, _, row in selected]
    batch = trainer.data_collator(features)

    picked_seq_len = max(x[0] for x in selected)
    picked_sources = [
        {
            "source": source_name,
            "index": int(idx),
            "seq_len": int(seq_len),
        }
        for seq_len, source_name, idx, _ in selected
    ]

    return batch, picked_seq_len, picked_sources


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _snapshot_trainer_log_state(trainer):
    """
    Warm-up calls compute_loss and training_step.
    Save the related state here and restore it afterward to avoid contaminating
    the actual training logs.
    """
    keys = [
        "_log_ce_prompt_weighted_sum",
        "_log_ce_assistant_only_sum",
        "_log_cnt",
        "_printed_loss_weight_debug",
        "_last_prompt_loss_weight",
        "_last_learning_rate",
        "_last_loss_normalization",
    ]

    state = {}
    for k in keys:
        if hasattr(trainer, k):
            state[k] = getattr(trainer, k)

    return state


def _restore_trainer_log_state(trainer, state):
    for k, v in state.items():
        setattr(trainer, k, v)


def trainer_warmup(
    trainer,
    *,
    warmup_forward_steps: int = 2,
    warmup_backward_steps: int = 1,
    pick_longest_from: int = 32,
    also_warmup_optimizer_state: bool = True,
):
    """
    Non-consuming Trainer warm-up.

    Preserve the main effects of the original warm-up:
      - forward warmup
      - backward warmup
      - AMP / bf16 / fp16 path warmup
      - loss_weights / prompt-weighted loss path warmup
      - optional optimizer state warmup

    Avoid mutating the updated dynamic sampler:
      - do not call trainer.get_train_dataloader()
      - do not advance sampler.epoch
      - do not update sampler.usage_counts
      - do not change current_epoch_global_indices
    """
    args = trainer.args

    using_deepspeed = bool(getattr(args, "deepspeed", None))
    using_fsdp = bool(getattr(args, "fsdp", None))

    model = trainer.model
    was_training = model.training
    model.train()

    device = args.device

    try:
        param_device = next(model.parameters()).device
    except StopIteration:
        param_device = device

    if (not using_deepspeed) and (not using_fsdp) and (str(param_device) == "cpu"):
        model.to(device)

    # Save sampler state as a safeguard. This function should not touch the
    # sampler, but the snapshot prevents future changes from affecting it.
    sampler = getattr(trainer, "_epoch_ratio_sampler", None)
    sampler_state = None
    sampler_epoch_before = None

    if sampler is not None:
        sampler_state = sampler.state_dict()
        sampler_epoch_before = int(getattr(sampler, "epoch", 0))

    trainer_log_state = _snapshot_trainer_log_state(trainer)

    try:
        best_batch, best_len, picked_sources = _build_non_consuming_warmup_batch(
            trainer,
            pick_longest_from=pick_longest_from,
        )

        best_batch = _move_batch_to_device(best_batch, device)

        use_autocast = bool(args.fp16 or args.bf16) and torch.cuda.is_available()
        amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)

        # ---------- 1) forward warm-up ----------
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        with torch.no_grad():
            for _ in range(int(warmup_forward_steps)):
                if use_autocast and amp_dtype is not None:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        _ = trainer.compute_loss(
                            model,
                            best_batch,
                            return_outputs=False,
                        )
                else:
                    _ = trainer.compute_loss(
                        model,
                        best_batch,
                        return_outputs=False,
                    )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ---------- 2) Backward warm-up without optimizer.step ----------
        for _ in range(int(warmup_backward_steps)):
            model.zero_grad(set_to_none=True)

            # Use Trainer.training_step to preserve the AMP, accelerator,
            # loss-scaling, and backward paths.
            _ = trainer.training_step(model, best_batch)

            model.zero_grad(set_to_none=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # ---------- 3) optimizer state warm-up ----------
        if also_warmup_optimizer_state and (not using_deepspeed) and (not using_fsdp):
            if trainer.optimizer is None:
                trainer.create_optimizer()

            opt = trainer.optimizer

            # Back up lr and weight_decay, then temporarily set them to zero
            # so opt.step() does not modify the weights.
            backup = []
            for g in opt.param_groups:
                backup.append((g.get("lr", None), g.get("weight_decay", None)))

                if "lr" in g:
                    g["lr"] = 0.0

                if "weight_decay" in g:
                    g["weight_decay"] = 0.0

            # Create zero gradients so optimizers such as AdamW initialize state.
            for group in opt.param_groups:
                for p in group["params"]:
                    if p.requires_grad:
                        if p.grad is None:
                            p.grad = torch.zeros_like(
                                p,
                                memory_format=torch.preserve_format,
                            )
                        else:
                            p.grad.zero_()

            opt.step()

            # Reset optimizer step counters to zero to avoid affecting Adam bias correction.
            for group in opt.param_groups:
                for p in group["params"]:
                    st = opt.state.get(p, None)

                    if st and "step" in st:
                        if torch.is_tensor(st["step"]):
                            st["step"].zero_()
                        else:
                            st["step"] = 0

            opt.zero_grad(set_to_none=True)

            # Restore lr and weight_decay.
            for g, (lr, wd) in zip(opt.param_groups, backup):
                if lr is not None:
                    g["lr"] = lr

                if wd is not None:
                    g["weight_decay"] = wd

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        print(
            f"[warmup] done. picked_seq_len={best_len}, "
            f"forward={warmup_forward_steps}, "
            f"backward={warmup_backward_steps}, "
            f"opt_state={also_warmup_optimizer_state and (not using_deepspeed) and (not using_fsdp)}"
        )
        print(f"[warmup] picked_sources={picked_sources}")

    finally:
        # Restore the sampler so warm-up cannot advance epochs or update usage_counts.
        if sampler is not None and sampler_state is not None:
            sampler.load_state_dict(
                sampler_state,
                replay_current_epoch=False,
            )

            sampler_epoch_after = int(getattr(sampler, "epoch", 0))
            if sampler_epoch_before != sampler_epoch_after:
                print(
                    f"[warmup][WARN] sampler epoch restored from {sampler_epoch_after} "
                    f"back to {sampler_epoch_before}"
                )

        # Restore Trainer log state so warm-up does not affect actual training logs.
        _restore_trainer_log_state(trainer, trainer_log_state)

        if not was_training:
            model.eval()
