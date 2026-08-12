"""Weighted SFT loss primitives copied from the original Trainer implementation."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


IGNORE_INDEX = -100


def resolve_loss_normalization(loss_normalization: str) -> str:
    name = str(loss_normalization or "token_mean").strip().lower()
    if name in {"token", "token_mean", "global_token_mean", "weight_mean"}:
        return "token_mean"
    if name in {"sample", "sample_mean", "example_mean", "sequence_mean", "per_sample_mean"}:
        return "sample_mean"
    raise ValueError("loss_normalization must be 'token_mean' or 'sample_mean'.")


def shift_logits_and_labels(logits: Any, labels: Any) -> Tuple[Any, Any]:
    return logits[:, :-1, :], labels[:, 1:]


def ce_from_logits_and_labels_shifted(
    logits_s: Any,
    labels_s: Any,
    ignore_index: int = IGNORE_INDEX,
):
    """Return the original per-token shifted CE tensor without reducing it."""

    try:
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Weighted training loss requires PyTorch.") from exc
    return functional.cross_entropy(
        logits_s.reshape(-1, logits_s.size(-1)),
        labels_s.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels_s)


def weighted_ce_from_shifted_logits_labels(
    logits_s: Any,
    labels_s: Any,
    loss_weights_s: Optional[Any] = None,
    attention_mask_s: Optional[Any] = None,
    ignore_index: int = IGNORE_INDEX,
):
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("Weighted training loss requires PyTorch.") from exc

    per_token_ce = functional.cross_entropy(
        logits_s.reshape(-1, logits_s.size(-1)),
        labels_s.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels_s)
    valid_mask = labels_s.ne(ignore_index)
    if attention_mask_s is not None:
        valid_mask = valid_mask & attention_mask_s.to(torch.bool)
    if loss_weights_s is None:
        effective_weights = valid_mask.to(per_token_ce.dtype)
    else:
        effective_weights = loss_weights_s.to(per_token_ce.dtype)
        effective_weights = effective_weights.masked_fill(~valid_mask, 0.0)
    denominator = effective_weights.sum().clamp_min(1.0)
    return (per_token_ce * effective_weights).sum() / denominator


def reduce_weighted_loss_from_per_token_values(
    per_token_loss: Any,
    valid_mask: Any,
    loss_weights_s: Optional[Any] = None,
    loss_normalization: str = "sample_mean",
    return_details: bool = False,
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Weighted training loss requires PyTorch.") from exc

    loss_normalization = resolve_loss_normalization(loss_normalization)
    valid_mask = valid_mask.to(torch.bool)
    if loss_weights_s is None:
        effective_weights = valid_mask.to(per_token_loss.dtype)
    else:
        effective_weights = loss_weights_s.to(per_token_loss.dtype)
        effective_weights = effective_weights.masked_fill(~valid_mask, 0.0)

    if not return_details:
        weighted_loss = per_token_loss * effective_weights
        if loss_normalization == "token_mean":
            denominator = effective_weights.sum().clamp_min(1.0)
            return {"loss": weighted_loss.sum() / denominator}
        sample_denominator = effective_weights.sum(dim=1)
        sample_numerator = weighted_loss.sum(dim=1)
        active_sample_mask = sample_denominator.gt(0)
        active_sample_float = active_sample_mask.to(per_token_loss.dtype)
        sample_loss = sample_numerator / sample_denominator.clamp_min(1.0)
        active_sample_count = active_sample_float.sum().clamp_min(1.0)
        return {
            "loss": (sample_loss * active_sample_float).sum() / active_sample_count
        }

    weighted_loss = per_token_loss * effective_weights
    token_denominator = effective_weights.sum().clamp_min(1.0)
    token_mean_loss = weighted_loss.sum() / token_denominator
    sample_denominator = effective_weights.sum(dim=1)
    sample_numerator = weighted_loss.sum(dim=1)
    active_sample_mask = sample_denominator.gt(0)
    active_sample_float = active_sample_mask.to(per_token_loss.dtype)
    sample_loss = sample_numerator / sample_denominator.clamp_min(1.0)
    active_sample_count = active_sample_float.sum().clamp_min(1.0)
    sample_mean_loss = (sample_loss * active_sample_float).sum() / active_sample_count
    final_loss = token_mean_loss if loss_normalization == "token_mean" else sample_mean_loss
    return {
        "loss": final_loss,
        "token_mean_loss": token_mean_loss.detach(),
        "sample_mean_loss": sample_mean_loss.detach(),
        "sample_loss": sample_loss.detach(),
        "sample_denom": sample_denominator.detach(),
        "sample_numer": sample_numerator.detach(),
        "active_sample_mask": active_sample_mask.detach(),
        "active_sample_count": active_sample_count.detach(),
        "token_denom": token_denominator.detach(),
    }


def build_full_supervision_labels(
    input_ids: Any,
    attention_mask: Optional[Any] = None,
    ignore_index: int = IGNORE_INDEX,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Weighted training loss requires PyTorch.") from exc
    full_labels = input_ids.clone()
    if attention_mask is not None:
        full_labels = full_labels.masked_fill(~attention_mask.to(torch.bool), ignore_index)
    return full_labels


def build_prompt_weighted_shifted_loss_weights(
    assistant_labels_s: Any,
    full_labels_s: Any,
    prompt_loss_weight: float,
    assistant_loss_weight: float = 1.0,
    assistant_base_loss_weights_s: Optional[Any] = None,
    ignore_index: int = IGNORE_INDEX,
):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Weighted training loss requires PyTorch.") from exc

    prompt_loss_weight = float(prompt_loss_weight)
    assistant_loss_weight = float(assistant_loss_weight)
    output = torch.zeros_like(full_labels_s, dtype=torch.float32)
    assistant_mask_s = assistant_labels_s.ne(ignore_index)
    full_valid_mask_s = full_labels_s.ne(ignore_index)
    prompt_mask_s = full_valid_mask_s & (~assistant_mask_s)
    if prompt_loss_weight != 0.0:
        output.masked_fill_(prompt_mask_s, prompt_loss_weight)
    if assistant_base_loss_weights_s is None:
        if assistant_loss_weight != 0.0:
            output.masked_fill_(assistant_mask_s, assistant_loss_weight)
    else:
        base = assistant_base_loss_weights_s.to(torch.float32)
        base = base.masked_fill(~assistant_mask_s, 0.0)
        output.add_(base * assistant_loss_weight)
    return output


def clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resolve_schedule_point_progress(
    *, point_type: str, point_value: float, num_train_epochs: Optional[float]
) -> float:
    point_type = str(point_type or "progress").lower()
    point_value = float(point_value)
    if point_type == "progress":
        return clip01(point_value)
    if point_type == "epoch":
        if num_train_epochs is None or float(num_train_epochs) <= 0:
            raise ValueError("num_train_epochs must be > 0 when point_type is 'epoch'.")
        return clip01(point_value / float(num_train_epochs))
    raise ValueError("schedule point type must be 'progress' or 'epoch'.")


def resolve_progress_from_trainer_state(*, global_step: int, max_steps: int) -> float:
    if max_steps is None or int(max_steps) <= 0:
        return 0.0
    return clip01(float(global_step) / float(max_steps))


def resolve_scheduled_prompt_loss_weight(
    *,
    use_dynamic: bool,
    fixed_prompt_loss_weight: float,
    start_weight: float,
    end_weight: float,
    start_progress: float,
    anchor_progress: float,
    current_progress: float,
    bin_ratio: float,
) -> float:
    if not bool(use_dynamic):
        return float(fixed_prompt_loss_weight)
    start_weight = float(start_weight)
    end_weight = float(end_weight)
    start_progress = clip01(start_progress)
    anchor_progress = clip01(anchor_progress)
    current_progress = clip01(current_progress)
    bin_ratio = float(bin_ratio)
    if bin_ratio <= 0:
        raise ValueError("prompt_loss_decay_bin_ratio must be > 0")
    if anchor_progress < start_progress:
        raise ValueError("prompt_loss_decay_anchor must be >= prompt_loss_decay_start")
    if abs(anchor_progress - start_progress) <= 1e-12:
        return end_weight
    if current_progress <= start_progress:
        return start_weight
    if current_progress >= anchor_progress:
        return end_weight

    local_span = anchor_progress - start_progress
    local_progress = clip01((current_progress - start_progress) / local_span)
    num_bins = max(1, int(math.ceil(1.0 / bin_ratio)))
    if num_bins == 1:
        return end_weight
    bin_index = min(int(local_progress / bin_ratio), num_bins - 1)
    fraction = float(bin_index) / float(num_bins - 1)
    return start_weight + (end_weight - start_weight) * fraction


# Original private names are retained for notebook-level regression checks.
_resolve_loss_normalization = resolve_loss_normalization
_shift_logits_and_labels = shift_logits_and_labels
_ce_from_logits_and_labels_shifted = ce_from_logits_and_labels_shifted
_weighted_ce_from_shifted_logits_labels = weighted_ce_from_shifted_logits_labels
_reduce_weighted_loss_from_per_token_values = reduce_weighted_loss_from_per_token_values
_build_full_supervision_labels = build_full_supervision_labels
_build_prompt_weighted_shifted_loss_weights = build_prompt_weighted_shifted_loss_weights
_clip01 = clip01
_resolve_schedule_point_progress = resolve_schedule_point_progress
_resolve_progress_from_trainer_state = resolve_progress_from_trainer_state
_resolve_scheduled_prompt_loss_weight = resolve_scheduled_prompt_loss_weight
