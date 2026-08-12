"""Original custom Trainer, modularized from the training notebook.

The class body and in-training semantic evaluation methods are preserved from
the recorded experiment. Data sampling and loss primitives are imported from
the separately regression-tested modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Sequence, Literal, List
import gc
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    Trainer,
)

from .config import DEFAULT_ROBERTA_MODEL

try:
    import ot
except Exception:
    # Preserve the original notebook behavior: POT is an executable dependency
    # of the in-training MoverScore evaluation, so install it immediately when
    # the environment does not already provide an importable ``ot`` module.
    print("[INSTALL] POT", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])
    import ot

from .losses import (
    IGNORE_INDEX,
    _build_full_supervision_labels,
    _build_prompt_weighted_shifted_loss_weights,
    _ce_from_logits_and_labels_shifted,
    _reduce_weighted_loss_from_per_token_values,
    _resolve_loss_normalization,
    _resolve_progress_from_trainer_state,
    _resolve_schedule_point_progress,
    _resolve_scheduled_prompt_loss_weight,
    _shift_logits_and_labels,
)
from .sampling import (
    _BalancedEpochRatioSampler,
    _TwoSourceTrainerDataset,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class RepetitionPenaltyLogitsProcessorExceptTokens(LogitsProcessor):
    """
    Apply repetition penalty only to the answer after prompt_ignore_length.
    This matches the behavior of Hugging Face repetition_penalty:
      - penalize tokens that have already appeared in the history
      - completely exempt tokens listed in exempt_token_ids
    """

    def __init__(
        self,
        penalty: float,
        prompt_ignore_length: int = 0,
        exempt_token_ids: Optional[Sequence[int]] = None,
    ):
        penalty = float(penalty)
        if penalty <= 0:
            raise ValueError("penalty must be > 0")
        self.penalty = penalty
        self.prompt_ignore_length = int(prompt_ignore_length)
        self.exempt_token_ids = set(int(x) for x in (exempt_token_ids or []))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.penalty == 1.0:
            return scores

        for b in range(input_ids.size(0)):
            seq = input_ids[b]

            if seq.size(0) <= self.prompt_ignore_length:
                continue

            answer_ids = seq[self.prompt_ignore_length:]
            if answer_ids.numel() == 0:
                continue

            seen = torch.unique(answer_ids)
            if seen.numel() == 0:
                continue

            if self.exempt_token_ids:
                keep_mask = torch.tensor(
                    [int(t.item()) not in self.exempt_token_ids for t in seen],
                    device=seen.device,
                    dtype=torch.bool,
                )
                seen = seen[keep_mask]

            if seen.numel() == 0:
                continue

            token_scores = scores[b, seen]
            token_scores = torch.where(
                token_scores < 0,
                token_scores * self.penalty,
                token_scores / self.penalty,
            )
            scores[b, seen] = token_scores

        return scores






class RepetitionPenaltyLogitsProcessorExceptTokensBatch(LogitsProcessor):
    """
    Batched version used by mover eval generation.

    It supports left-padded generation inputs and beam expansion.  Each original
    sample can have a different prompt_ignore_length.  The answer region starts
    at prompt_ignore_lengths[base_sample_index].
    """

    def __init__(
        self,
        penalty: float,
        prompt_ignore_lengths: Sequence[int],
        exempt_token_ids: Optional[Sequence[int]] = None,
        num_beams: int = 1,
    ):
        penalty = float(penalty)
        if penalty <= 0:
            raise ValueError("penalty must be > 0")
        self.penalty = penalty
        self.prompt_ignore_lengths = [int(x) for x in prompt_ignore_lengths]
        self.exempt_token_ids = set(int(x) for x in (exempt_token_ids or []))
        self.num_beams = max(1, int(num_beams or 1))

    def _base_index(self, row_index: int) -> int:
        if len(self.prompt_ignore_lengths) <= 1:
            return 0
        # HF generation expands rows as: sample0 beams..., sample1 beams..., ...
        return min(int(row_index) // self.num_beams, len(self.prompt_ignore_lengths) - 1)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.penalty == 1.0 or len(self.prompt_ignore_lengths) == 0:
            return scores

        for b in range(input_ids.size(0)):
            base_b = self._base_index(b)
            ignore_len = int(self.prompt_ignore_lengths[base_b])

            seq = input_ids[b]
            if seq.size(0) <= ignore_len:
                continue

            answer_ids = seq[ignore_len:]
            if answer_ids.numel() == 0:
                continue

            seen = torch.unique(answer_ids)
            if seen.numel() == 0:
                continue

            if self.exempt_token_ids:
                keep_mask = torch.tensor(
                    [int(t.item()) not in self.exempt_token_ids for t in seen],
                    device=seen.device,
                    dtype=torch.bool,
                )
                seen = seen[keep_mask]

            if seen.numel() == 0:
                continue

            token_scores = scores[b, seen]
            token_scores = torch.where(
                token_scores < 0,
                token_scores * self.penalty,
                token_scores / self.penalty,
            )
            scores[b, seen] = token_scores

        return scores


@dataclass
class SFTConfig:
    prompt_loss_weight: float = 0.0
    assistant_loss_weight: float = 1.0
    full_loss_labels_key: str = "full_labels"
    prompt_loss_use_input_ids_as_full_labels: bool = True

    # =========================
    # loss normalization
    # =========================
    # token_mean:
    #   Average all active tokens and weights together.
    # sample_mean:
    #   Average within each sample, then average across active samples.
    loss_normalization: Literal["token_mean", "sample_mean"] = "sample_mean"
    loss_debug_first_n: int = 8

    use_dynamic_prompt_loss_weight: bool = False
    prompt_loss_weight_start: float = 0.0
    prompt_loss_weight_end: float = 0.0

    # Start point: prompt weight remains at start before this point.
    prompt_loss_decay_start_type: Literal["progress", "epoch"] = "progress"
    prompt_loss_decay_start_value: float = 0.0

    # Anchor point: prompt weight reaches end here and remains there afterward.
    prompt_loss_decay_anchor_type: Literal["progress", "epoch"] = "progress"
    prompt_loss_decay_anchor_value: float = 1.0

    # Bin ratio within the local [start, anchor] interval.
    prompt_loss_decay_bin_ratio: float = 0.1

    # =========================
    # best metric
    # =========================
    # Used only to align SaveBestPeftCallback with TrainingArguments.
    # The official metric no longer uses eval_valid_mover_final_score;
    # metric_name="auto" is recommended for the callback.
    best_metric_name: str = "eval_combo_mover_score"

    # =========================
    # train preview
    # Preserve all original preview parameters.
    # =========================
    train_preview_every: int = 0
    train_preview_num_samples: int = 5
    train_preview_indices: Optional[Sequence[int]] = None
    train_preview_max_new_tokens: int = 128
    train_preview_do_sample: bool = False
    train_preview_num_beams: int = 1
    train_preview_answer_prefix_len: int = 2
    train_preview_repetition_penalty: float = 1.10
    train_preview_no_repeat_ngram_size: int = 0


    # =========================
    # eval metrics
    # =========================
    eval_run_loss: bool = True


    # Semantic metric sampling; <=0 evaluates the full validation set.
    eval_metric_sample_size: int = 100
    eval_metric_sample_mode: Literal["fixed", "per_eval_random"] = "fixed"
    eval_metric_sample_seed: int = 42

    # Summary metric generation settings.
    # This batch size controls model.generate before the official semantic evaluation;
    # per_device_eval_batch_size controls only the validation-loss dataloader.
    eval_metric_max_new_tokens: int = 128
    eval_metric_do_sample: bool = False
    eval_metric_num_beams: int = 1
    eval_metric_repetition_penalty: float = 1.10
    eval_metric_answer_prefix_len: int = 2
    eval_metric_generation_batch_size: int = 16
    eval_metric_no_repeat_ngram_size: int = 0
    eval_metric_length_bucket: bool = True
    eval_metric_show_progress: bool = True



    # MoverScore-like OT / RoBERTa encoder settings.
    eval_mover_model_path: str = DEFAULT_ROBERTA_MODEL
    eval_mover_max_length: int = 512
    eval_mover_first_layer_index: int = 0
    eval_mover_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    eval_mover_device: Optional[str] = None
    eval_mover_encoder_batch_size: int = 8
    eval_mover_cache_encoder: bool = True
    eval_mover_release_encoder_after_eval: bool = False

    # Backward-compatible parameter names: if an older external notebook still
    # passes eval_mover_*, prefer eval_metric_* and use the old values as fallbacks.
    eval_run_mover: bool = True
    eval_mover_sample_size: int = 100
    eval_mover_sample_mode: Literal["fixed", "per_eval_random"] = "fixed"
    eval_mover_sample_seed: int = 42
    eval_mover_max_new_tokens: int = 128
    eval_mover_do_sample: bool = False
    eval_mover_num_beams: int = 1
    eval_mover_repetition_penalty: float = 1.10
    eval_mover_answer_prefix_len: int = 2
    eval_mover_generation_batch_size: int = 16

    # =========================
    # epoch-level dynamic source mixing
    # =========================
    # When train_pools is passed to SFTTrainer, these fields control
    # how cnn_dm / kptime are sampled at each epoch.  If the current epoch
    # exceeds the schedule length, the final schedule point is reused.
    epoch_ratio_schedule: Optional[Sequence[Dict[str, float]]] = None
    samples_per_epoch: Optional[int] = None
    epoch_sampling_seed: int = 42
    epoch_sampling_shuffle: bool = True
    epoch_sampling_verbose: bool = True
    epoch_sampling_save_state: bool = True
    # If resuming from a mid-epoch checkpoint, replay the saved epoch index order
    # once so Trainer's skipped-batch logic can line up with the original stream.
    # For checkpoints saved exactly at an epoch boundary, replay is automatically
    # disabled based on trainer_state.json.
    epoch_sampling_replay_last_epoch_on_resume: bool = True

    # Clear the CUDA cache after evaluation.
    eval_cleanup_cuda_cache_after_eval: bool = True
    eval_cleanup_cuda_cache_verbose: bool = False

class SFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        sft_config: Optional[SFTConfig] = None,
        train_pools: Optional[Dict[str, Any]] = None,
        epoch_ratio_schedule: Optional[Sequence[Dict[str, float]]] = None,
        samples_per_epoch: Optional[int] = None,
        epoch_sampling_seed: Optional[int] = None,
        epoch_sampling_shuffle: Optional[bool] = None,
        epoch_sampling_verbose: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sft_config = sft_config or SFTConfig()

        self._log_ce_prompt_weighted_sum = 0.0
        self._log_ce_assistant_only_sum = 0.0
        self._log_cnt = 0
        self._last_prompt_loss_weight = float(getattr(self.sft_config, "prompt_loss_weight", 0.0) or 0.0)
        self._last_learning_rate = 0.0
        self._last_loss_normalization = _resolve_loss_normalization(
            getattr(self.sft_config, "loss_normalization", "sample_mean")
        )

        self._train_preview_indices_cache: Optional[List[int]] = None
        self._last_train_preview_step: int = -1

        # One-time diagnostic output control.
        self._printed_loss_weight_debug = False

        # Tokenizer fallback-inference cache.

        # RoBERTa encoder cache for MoverScore-like OT.
        self._mover_tokenizer = None
        self._mover_model = None
        self._mover_encoder_key = None


        # Epoch-level cnn_dm / kptime dynamic mixing state.
        self._use_epoch_ratio_sampling = False
        self._epoch_ratio_train_dataset = None
        self._epoch_ratio_sampler = None
        self._setup_epoch_ratio_sampling(
            train_pools=train_pools,
            epoch_ratio_schedule=epoch_ratio_schedule,
            samples_per_epoch=samples_per_epoch,
            epoch_sampling_seed=epoch_sampling_seed,
            epoch_sampling_shuffle=epoch_sampling_shuffle,
            epoch_sampling_verbose=epoch_sampling_verbose,
        )

    def _setup_epoch_ratio_sampling(
        self,
        *,
        train_pools: Optional[Dict[str, Any]],
        epoch_ratio_schedule: Optional[Sequence[Dict[str, float]]],
        samples_per_epoch: Optional[int],
        epoch_sampling_seed: Optional[int],
        epoch_sampling_shuffle: Optional[bool],
        epoch_sampling_verbose: Optional[bool],
    ) -> None:
        """
        Optional dynamic train sampling setup.

        If train_pools is None, the Trainer keeps the original Hugging Face
        Trainer behavior exactly and uses self.train_dataset.

        If train_pools is provided, self.train_dataset is replaced by a wrapper
        over train_pools and get_train_dataloader() will use a custom sampler
        that rebuilds epoch samples according to epoch_ratio_schedule.
        """
        if train_pools is None:
            return

        if not isinstance(train_pools, dict) or len(train_pools) == 0:
            raise ValueError("train_pools must be a non-empty dict when provided.")

        # Prefer the requested public names.  Fall back to insertion order only
        # to avoid breaking experimentation with equivalent source names.
        if "cnn_dm" in train_pools and "kptime" in train_pools:
            source_names = ["cnn_dm", "kptime"]
        else:
            source_names = [str(k) for k in train_pools.keys()]

        if epoch_ratio_schedule is None:
            epoch_ratio_schedule = getattr(self.sft_config, "epoch_ratio_schedule", None)
        if samples_per_epoch is None:
            samples_per_epoch = getattr(self.sft_config, "samples_per_epoch", None)
        if epoch_sampling_seed is None:
            epoch_sampling_seed = int(getattr(self.sft_config, "epoch_sampling_seed", 42) or 42)
        if epoch_sampling_shuffle is None:
            epoch_sampling_shuffle = bool(getattr(self.sft_config, "epoch_sampling_shuffle", True))
        if epoch_sampling_verbose is None:
            epoch_sampling_verbose = bool(getattr(self.sft_config, "epoch_sampling_verbose", True))

        wrapper = _TwoSourceTrainerDataset(
            train_pools=train_pools,
            source_names=source_names,
        )

        process_rank = int(getattr(self.args, "process_index", 0) or 0)
        world_size = int(getattr(self.args, "world_size", 1) or 1)

        sampler = _BalancedEpochRatioSampler(
            source_lengths=wrapper.lengths,
            source_offsets=wrapper.offsets,
            source_names=wrapper.source_names,
            epoch_ratio_schedule=epoch_ratio_schedule,
            samples_per_epoch=samples_per_epoch,
            seed=int(epoch_sampling_seed),
            shuffle_epoch_indices=bool(epoch_sampling_shuffle),
            process_rank=process_rank,
            num_processes=world_size,
            verbose=bool(epoch_sampling_verbose),
        )

        self._use_epoch_ratio_sampling = True
        self._epoch_ratio_train_dataset = wrapper
        self._epoch_ratio_sampler = sampler

        # Keep Trainer preview and other Trainer internals pointing to a valid
        # train_dataset.  Actual epoch selection is controlled by sampler.
        self.train_dataset = wrapper

        if self.is_world_process_zero():
            print("\n[Dynamic epoch source mixing enabled]", flush=True)
            print(f"sources={wrapper.source_names}", flush=True)
            print(f"source_lengths={wrapper.lengths}", flush=True)
            print(f"samples_per_epoch={sampler.samples_per_epoch}", flush=True)
            print(f"epoch_ratio_schedule={sampler.epoch_ratio_schedule}", flush=True)

    def get_train_dataloader(self):
        """
        Preserve original Trainer behavior unless train_pools was provided.

        With train_pools, this returns a DataLoader whose sampler creates a new
        balanced cnn_dm/kptime sample list every epoch.  The collator and
        compute_loss paths are unchanged, so loss_weights are still padded by
        the collator and consumed by compute_loss exactly as before.
        """
        if not getattr(self, "_use_epoch_ratio_sampling", False):
            return super().get_train_dataloader()

        if self._epoch_ratio_train_dataset is None or self._epoch_ratio_sampler is None:
            raise RuntimeError("Epoch ratio sampling is enabled but dataset/sampler is missing.")

        if self.data_collator is None:
            raise ValueError("data_collator must be set when using epoch ratio sampling.")

        batch_size = getattr(self, "_train_batch_size", None)
        if batch_size is None:
            batch_size = self.args.train_batch_size

        num_workers = int(getattr(self.args, "dataloader_num_workers", 0) or 0)

        dataloader_kwargs = {
            "batch_size": int(batch_size),
            "sampler": self._epoch_ratio_sampler,
            "collate_fn": self.data_collator,
            "drop_last": bool(getattr(self.args, "dataloader_drop_last", False)),
            "num_workers": num_workers,
            "pin_memory": bool(getattr(self.args, "dataloader_pin_memory", True)),
        }

        if num_workers > 0:
            persistent_workers = bool(getattr(self.args, "dataloader_persistent_workers", False))
            dataloader_kwargs["persistent_workers"] = persistent_workers

            prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = prefetch_factor

        dataloader = DataLoader(
            self._epoch_ratio_train_dataset,
            **dataloader_kwargs,
        )
        return self.accelerator.prepare(dataloader)

    _EPOCH_RATIO_SAMPLER_STATE_FILE = "epoch_ratio_sampler_state.pt"

    def _get_epoch_ratio_sampler_checkpoint_dir(self, trial=None) -> str:
        run_dir = self.args.output_dir
        get_output_dir = getattr(self, "_get_output_dir", None)
        if callable(get_output_dir):
            try:
                run_dir = get_output_dir(trial=trial)
            except TypeError:
                try:
                    run_dir = get_output_dir(trial)
                except Exception:
                    run_dir = self.args.output_dir
            except Exception:
                run_dir = self.args.output_dir

        return os.path.join(run_dir, f"checkpoint-{int(self.state.global_step)}")

    def _save_epoch_ratio_sampler_state(self, checkpoint_dir: str) -> None:
        if not getattr(self, "_use_epoch_ratio_sampling", False):
            return
        if self._epoch_ratio_sampler is None:
            return
        if not bool(getattr(self.sft_config, "epoch_sampling_save_state", True)):
            return
        if not self.is_world_process_zero():
            return

        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, self._EPOCH_RATIO_SAMPLER_STATE_FILE)
        payload = self._epoch_ratio_sampler.state_dict()
        torch.save(payload, path)

        if bool(getattr(self.sft_config, "epoch_sampling_verbose", True)):
            print(f"[Epoch mixing sampler state saved] {path}", flush=True)

    @staticmethod
    def _load_torch_checkpoint_payload(path: str) -> Dict[str, Any]:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _is_checkpoint_epoch_boundary(checkpoint_dir: str) -> bool:
        state_path = os.path.join(checkpoint_dir, "trainer_state.json")
        if not os.path.exists(state_path):
            return False

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            epoch_value = state.get("epoch", None)
            if epoch_value is None:
                return False
            epoch_float = float(epoch_value)
            return abs(epoch_float - round(epoch_float)) <= 1e-9
        except Exception:
            return False

    def _load_epoch_ratio_sampler_state(self, checkpoint_dir: Optional[str]) -> None:
        if not getattr(self, "_use_epoch_ratio_sampling", False):
            return
        if self._epoch_ratio_sampler is None:
            return
        if checkpoint_dir is None:
            return

        path = os.path.join(str(checkpoint_dir), self._EPOCH_RATIO_SAMPLER_STATE_FILE)
        if not os.path.exists(path):
            if self.is_world_process_zero():
                print(
                    f"[WARN] Epoch mixing sampler state not found in checkpoint: {path}",
                    flush=True,
                )
            return

        payload = self._load_torch_checkpoint_payload(path)

        replay_cfg = bool(
            getattr(self.sft_config, "epoch_sampling_replay_last_epoch_on_resume", True)
        )
        # At exact epoch-boundary checkpoints, the next dataloader should advance
        # to the next epoch, so replaying the just-finished epoch would be wrong.
        replay_current_epoch = replay_cfg and (not self._is_checkpoint_epoch_boundary(str(checkpoint_dir)))

        self._epoch_ratio_sampler.load_state_dict(
            payload,
            replay_current_epoch=replay_current_epoch,
        )

        if self.is_world_process_zero():
            print(
                "[Epoch mixing sampler state loaded] "
                f"{path} replay_current_epoch={replay_current_epoch}",
                flush=True,
            )

    def _save_checkpoint(self, *args, **kwargs):
        out = super()._save_checkpoint(*args, **kwargs)

        if getattr(self, "_use_epoch_ratio_sampling", False):
            trial = kwargs.get("trial", None)
            if trial is None and len(args) >= 2:
                trial = args[1]
            checkpoint_dir = self._get_epoch_ratio_sampler_checkpoint_dir(trial=trial)
            self._save_epoch_ratio_sampler_state(checkpoint_dir)

        return out

    def _load_from_checkpoint(self, *args, **kwargs):
        resume_from_checkpoint = None
        if len(args) >= 1:
            resume_from_checkpoint = args[0]
        if resume_from_checkpoint is None:
            resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)

        out = super()._load_from_checkpoint(*args, **kwargs)
        self._load_epoch_ratio_sampler_state(resume_from_checkpoint)
        return out

    def _get_current_learning_rate_value(self) -> float:
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            try:
                return float(optimizer.param_groups[0]["lr"])
            except Exception:
                pass
        return float(getattr(self.args, "learning_rate", 0.0) or 0.0)

    def _get_current_prompt_loss_weight_value(self) -> float:
        num_train_epochs = getattr(self.args, "num_train_epochs", None)

        start_progress = _resolve_schedule_point_progress(
            point_type=getattr(self.sft_config, "prompt_loss_decay_start_type", "progress"),
            point_value=getattr(self.sft_config, "prompt_loss_decay_start_value", 0.0),
            num_train_epochs=num_train_epochs,
        )

        anchor_progress = _resolve_schedule_point_progress(
            point_type=getattr(self.sft_config, "prompt_loss_decay_anchor_type", "progress"),
            point_value=getattr(self.sft_config, "prompt_loss_decay_anchor_value", 1.0),
            num_train_epochs=num_train_epochs,
        )

        current_progress = _resolve_progress_from_trainer_state(
            global_step=int(getattr(self.state, "global_step", 0) or 0),
            max_steps=int(getattr(self.state, "max_steps", 0) or 0),
        )

        return float(
            _resolve_scheduled_prompt_loss_weight(
                use_dynamic=getattr(self.sft_config, "use_dynamic_prompt_loss_weight", False),
                fixed_prompt_loss_weight=getattr(self.sft_config, "prompt_loss_weight", 0.0),
                start_weight=getattr(self.sft_config, "prompt_loss_weight_start", 0.0),
                end_weight=getattr(self.sft_config, "prompt_loss_weight_end", 0.0),
                start_progress=start_progress,
                anchor_progress=anchor_progress,
                current_progress=current_progress,
                bin_ratio=getattr(self.sft_config, "prompt_loss_decay_bin_ratio", 0.1),
            )
        )

    def log(self, logs: Dict[str, Any], *args, **kwargs) -> None:
        printed_train_compare = None

        if self.model is not None and self.model.training and self._log_cnt > 0:
            logs = dict(logs)

            loss_ce_prompt_weighted_raw = self._log_ce_prompt_weighted_sum / self._log_cnt
            loss_ce_assistant_only_raw = self._log_ce_assistant_only_sum / self._log_cnt

            ga_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)

            # Match the definition used by Training Loss in the Trainer table.
            loss_ce_prompt_weighted_display = loss_ce_prompt_weighted_raw
            loss_ce_assistant_only_display = loss_ce_assistant_only_raw
            logs["loss"] = loss_ce_prompt_weighted_display

            logs["loss_ce_prompt_weighted"] = loss_ce_prompt_weighted_display
            logs["loss_ce_assistant_only"] = loss_ce_assistant_only_display
            logs["loss_ce"] = logs["loss_ce_prompt_weighted"]
            logs["prompt_loss_weight_current"] = float(self._last_prompt_loss_weight)
            logs["learning_rate_current"] = float(self._last_learning_rate)
            logs["loss_is_sample_mean"] = (
                1.0 if getattr(self, "_last_loss_normalization", "token_mean") == "sample_mean" else 0.0
            )

            printed_train_compare = {
                "step": int(getattr(self.state, "global_step", 0) or 0),
                "epoch": float(logs["epoch"]) if "epoch" in logs else None,
                "ga_steps": ga_steps,
                "loss_ce_prompt_weighted": float(logs["loss_ce_prompt_weighted"]),
                "loss_ce_assistant_only": float(logs["loss_ce_assistant_only"]),
                "prompt_loss_weight_current": float(logs["prompt_loss_weight_current"]),
                "learning_rate_current": float(logs["learning_rate_current"]),
                "loss_normalization": str(getattr(self, "_last_loss_normalization", "token_mean")),
            }

            self._log_ce_prompt_weighted_sum = 0.0
            self._log_ce_assistant_only_sum = 0.0
            self._log_cnt = 0

        out = super().log(logs, *args, **kwargs)

        if printed_train_compare is not None and self.is_world_process_zero():
            epoch_str = (
                f"{printed_train_compare['epoch']:.4f}"
                if printed_train_compare["epoch"] is not None
                else "NA"
            )
            print(
                "[TRAIN-LOSS-COMPARE] "
                f"step={printed_train_compare['step']} "
                f"epoch={epoch_str} "
                f"ga_steps={printed_train_compare['ga_steps']} "
                f"loss_ce_prompt_weighted={printed_train_compare['loss_ce_prompt_weighted']:.6f} "
                f"loss_ce_assistant_only={printed_train_compare['loss_ce_assistant_only']:.6f} "
                f"lr={printed_train_compare['learning_rate_current']:.8f} "
                f"prompt_w={printed_train_compare['prompt_loss_weight_current']:.6f} "
                f"loss_norm={printed_train_compare['loss_normalization']}",
                flush=True,
            )

        if self.model is not None and self.model.training and self._should_run_train_preview():
            try:
                self._run_train_preview()
            except Exception as e:
                print(f"[WARN] train preview failed: {e}", flush=True)


        return out

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ):
        labels: torch.Tensor = inputs["labels"]
        loss_weights_full: Optional[torch.Tensor] = inputs.get("loss_weights", None)

        full_loss_labels_key = str(getattr(self.sft_config, "full_loss_labels_key", "full_labels"))
        full_labels: Optional[torch.Tensor] = inputs.get(full_loss_labels_key, None)

        loss_normalization = _resolve_loss_normalization(
            getattr(self.sft_config, "loss_normalization", "sample_mean")
        )
        self._last_loss_normalization = loss_normalization

        model_inputs = {
            k: v for k, v in inputs.items()
            if k not in {"labels", "loss_weights", full_loss_labels_key}
        }

        out_adapt = model(**model_inputs, use_cache=False)
        logits_adapt: torch.Tensor = out_adapt.logits

        logits_a_s, labels_s = _shift_logits_and_labels(logits_adapt, labels)

        attention_mask = model_inputs.get("attention_mask", None)
        attention_mask_s = None
        if attention_mask is not None:
            attention_mask_s = attention_mask[:, 1:]

        loss_weights_s = None
        if loss_weights_full is not None:
            loss_weights_s = loss_weights_full[:, 1:]

        prompt_loss_weight = self._get_current_prompt_loss_weight_value()
        assistant_loss_weight = float(getattr(self.sft_config, "assistant_loss_weight", 1.0) or 0.0)
        self._last_prompt_loss_weight = float(prompt_loss_weight)
        self._last_learning_rate = self._get_current_learning_rate_value()

        full_labels_source = full_loss_labels_key
        if full_labels is None:
            use_input_ids_fallback = bool(
                getattr(self.sft_config, "prompt_loss_use_input_ids_as_full_labels", True)
            )
            if use_input_ids_fallback:
                full_labels = _build_full_supervision_labels(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=attention_mask,
                    ignore_index=IGNORE_INDEX,
                )
                full_labels_source = "input_ids_fallback"
            else:
                raise ValueError(
                    f"Prompt-weighted loss requires `{full_loss_labels_key}` in inputs "
                    f"or prompt_loss_use_input_ids_as_full_labels=True."
                )

        _, full_labels_s = _shift_logits_and_labels(logits_adapt, full_labels)

        # ============================================================
        # Use one model forward pass without a second inference pass.
        # assistant-only and prompt-weighted use different targets:
        #   - assistant-only uses labels_s exclusively
        #   - prompt-weighted uses full_labels_s exclusively
        #
        # GPU memory optimization:
        #   - compute the prompt-weighted CE used for training and immediately
        #     reduce it to a scalar;
        #   - release the [B, T] prompt-CE intermediate before computing the
        #     assistant-only CE used for logging;
        #   - build shifted prompt weights directly instead of slicing a
        #     full-length weight tensor.
        # Evaluation and training semantics remain unchanged: no second model
        # forward pass is added and the targets are unchanged.
        # ============================================================
        need_loss_debug = (not self._printed_loss_weight_debug) and self.is_world_process_zero()

        assistant_valid_mask_s = labels_s.ne(IGNORE_INDEX)
        full_valid_mask_s = full_labels_s.ne(IGNORE_INDEX)

        if attention_mask_s is not None:
            attn_bool_s = attention_mask_s.to(torch.bool)
            assistant_valid_mask_s = assistant_valid_mask_s & attn_bool_s
            full_valid_mask_s = full_valid_mask_s & attn_bool_s
            del attn_bool_s

        prompt_weighted_loss_weights_s = _build_prompt_weighted_shifted_loss_weights(
            assistant_labels_s=labels_s,
            full_labels_s=full_labels_s,
            prompt_loss_weight=prompt_loss_weight,
            assistant_loss_weight=assistant_loss_weight,
            assistant_base_loss_weights_s=loss_weights_s,
            ignore_index=IGNORE_INDEX,
        )

        per_token_ce_full = _ce_from_logits_and_labels_shifted(
            logits_s=logits_a_s,
            labels_s=full_labels_s,
            ignore_index=IGNORE_INDEX,
        )

        prompt_weighted_ce_out = _reduce_weighted_loss_from_per_token_values(
            per_token_loss=per_token_ce_full,
            valid_mask=full_valid_mask_s,
            loss_weights_s=prompt_weighted_loss_weights_s,
            loss_normalization=loss_normalization,
            return_details=need_loss_debug,
        )
        loss_ce_prompt_weighted = prompt_weighted_ce_out["loss"]

        # The main training loss is already reduced. Release the [B, T] CE
        # immediately so it does not coexist with the logging CE in memory.
        del per_token_ce_full

        with torch.no_grad():
            per_token_ce_assistant = _ce_from_logits_and_labels_shifted(
                logits_s=logits_a_s.detach(),
                labels_s=labels_s,
                ignore_index=IGNORE_INDEX,
            )
            assistant_ce_out = _reduce_weighted_loss_from_per_token_values(
                per_token_loss=per_token_ce_assistant,
                valid_mask=assistant_valid_mask_s,
                loss_weights_s=loss_weights_s,
                loss_normalization=loss_normalization,
                return_details=need_loss_debug,
            )
            loss_ce_assistant_only = assistant_ce_out["loss"]

        del per_token_ce_assistant

        if need_loss_debug:
            prompt_valid_mask_s = full_valid_mask_s & (~assistant_valid_mask_s)

            debug_n = int(getattr(self.sft_config, "loss_debug_first_n", 8) or 8)
            debug_n = max(1, debug_n)

            print("\n[LOSS_WEIGHT DEBUG - PRINT ONCE]", flush=True)
            print("input keys:", list(inputs.keys()), flush=True)
            print("labels shape:", tuple(labels.shape), flush=True)
            print("full_labels_source:", full_labels_source, flush=True)
            print("loss_normalization:", loss_normalization, flush=True)
            print("TRAINING_LOSS_USED: prompt_weighted_ce_" + loss_normalization, flush=True)
            print("prompt_loss_weight:", prompt_loss_weight, flush=True)
            print("assistant_loss_weight:", assistant_loss_weight, flush=True)

            # Check whether labels_s and full_labels_s agree at active assistant
            # positions. assistant-only CE is computed separately from labels_s,
            # so a mismatch does not contaminate that loss; it only indicates
            # that the two supervision targets differ.
            mismatch_mask = assistant_valid_mask_s & labels_s.ne(full_labels_s)
            mismatch_count = int(mismatch_mask.sum().detach().cpu().item())
            print("assistant/full label mismatch count:", mismatch_count, flush=True)
            if mismatch_count > 0:
                print(
                    "[INFO] assistant labels and full_labels differ on active assistant positions. "
                    "assistant-only CE uses labels_s; prompt-weighted CE uses full_labels_s.",
                    flush=True,
                )

            assistant_token_count_per_sample = assistant_valid_mask_s.sum(dim=1)
            prompt_token_count_per_sample = prompt_valid_mask_s.sum(dim=1)
            full_token_count_per_sample = full_valid_mask_s.sum(dim=1)

            print(
                "assistant_token_count_per_sample:",
                assistant_token_count_per_sample[:debug_n].detach().cpu().tolist(),
                flush=True,
            )
            print(
                "prompt_token_count_per_sample   :",
                prompt_token_count_per_sample[:debug_n].detach().cpu().tolist(),
                flush=True,
            )
            print(
                "full_token_count_per_sample     :",
                full_token_count_per_sample[:debug_n].detach().cpu().tolist(),
                flush=True,
            )

            print("shifted assistant token count:", int(assistant_valid_mask_s.sum().item()), flush=True)
            print("shifted prompt token count   :", int(prompt_valid_mask_s.sum().item()), flush=True)
            print("shifted full token count     :", int(full_valid_mask_s.sum().item()), flush=True)

            if loss_weights_full is None:
                print("assistant loss_weights_full: None", flush=True)
            else:
                print("assistant loss_weights_full shape:", tuple(loss_weights_full.shape), flush=True)
                print("assistant loss_weights_full dtype:", loss_weights_full.dtype, flush=True)

            active_prompt_weighted = prompt_weighted_loss_weights_s[full_valid_mask_s]
            if active_prompt_weighted.numel() > 0:
                print(
                    "prompt-weighted active weight unique sample:",
                    torch.unique(active_prompt_weighted)[:20].detach().cpu().tolist(),
                    flush=True,
                )
                print(
                    "prompt-weighted first active weights:",
                    active_prompt_weighted[:30].detach().cpu().tolist(),
                    flush=True,
                )

            print(
                "assistant_sample_denom:",
                assistant_ce_out["sample_denom"][:debug_n].detach().cpu().tolist(),
                flush=True,
            )
            print(
                "prompt_weighted_sample_denom:",
                prompt_weighted_ce_out["sample_denom"][:debug_n].detach().cpu().tolist(),
                flush=True,
            )

            print(
                "assistant_sample_loss_first_n:",
                assistant_ce_out["sample_loss"][:debug_n].detach().cpu().tolist(),
                flush=True,
            )
            print(
                "prompt_weighted_sample_loss_first_n:",
                prompt_weighted_ce_out["sample_loss"][:debug_n].detach().cpu().tolist(),
                flush=True,
            )

            print(
                "assistant_active_sample_count:",
                float(assistant_ce_out["active_sample_count"].detach().cpu()),
                flush=True,
            )
            print(
                "prompt_weighted_active_sample_count:",
                float(prompt_weighted_ce_out["active_sample_count"].detach().cpu()),
                flush=True,
            )

            print(
                "assistant_ce_token_mean_debug:",
                float(assistant_ce_out["token_mean_loss"].detach().cpu()),
                flush=True,
            )
            print(
                "assistant_ce_sample_mean:",
                float(assistant_ce_out["sample_mean_loss"].detach().cpu()),
                flush=True,
            )
            print(
                "prompt_weighted_ce_token_mean_debug:",
                float(prompt_weighted_ce_out["token_mean_loss"].detach().cpu()),
                flush=True,
            )
            print(
                "prompt_weighted_ce_sample_mean:",
                float(prompt_weighted_ce_out["sample_mean_loss"].detach().cpu()),
                flush=True,
            )

            print(
                "assistant-only CE USED:",
                float(loss_ce_assistant_only.detach().cpu()),
                flush=True,
            )
            print(
                "prompt-weighted CE USED:",
                float(loss_ce_prompt_weighted.detach().cpu()),
                flush=True,
            )

            self._printed_loss_weight_debug = True

            # These debug tensors and index results are used only for one-time
            # output. Delete them afterward to avoid extending GPU references.
            try:
                del prompt_valid_mask_s, mismatch_mask
                del assistant_token_count_per_sample, prompt_token_count_per_sample, full_token_count_per_sample
                del active_prompt_weighted
            except Exception:
                pass

        # Only scalar losses and log values are needed after this point.
        del prompt_weighted_loss_weights_s
        del assistant_valid_mask_s, full_valid_mask_s
        del assistant_ce_out, prompt_weighted_ce_out

        loss = loss_ce_prompt_weighted

        if model.training:
            self._log_ce_prompt_weighted_sum += float(loss_ce_prompt_weighted.detach().cpu())
            self._log_ce_assistant_only_sum += float(loss_ce_assistant_only.detach().cpu())
            self._log_cnt += 1

        if return_outputs:
            return loss, out_adapt
        return loss

    @staticmethod
    def _strip_summary_tags(text: str) -> str:
        text = text.replace("[SUMMARY]", "").replace("[/SUMMARY]", "")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _to_1d_long_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            t = t.view(-1)
        return t.long()

    @staticmethod
    def _normalize_preview_text(text: str) -> str:
        if text is None:
            return ""
        text = str(text).replace("\u3000", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _extract_answer_payload(cls, text: str) -> str:
        text = cls._normalize_preview_text(text)
        if not text:
            return ""

        text = re.sub(r"<\|.*?\|>", " ", text)
        text = cls._normalize_preview_text(text)

        pos_candidates = [p for p in (text.find(":"), text.find("：")) if p != -1]
        if pos_candidates:
            text = text[min(pos_candidates) + 1:]

        text = re.sub(r"<\|.*?\|>", " ", text)
        text = cls._normalize_preview_text(text)
        return text

    @classmethod
    def _canonicalize_item(cls, item: str) -> str:
        item = cls._normalize_preview_text(item)
        item = item.strip("\"'[](){}")
        item = cls._normalize_preview_text(item)
        return item.lower()

    @classmethod
    def _payload_to_items(cls, payload: str) -> List[str]:
        payload = cls._normalize_preview_text(payload)
        if not payload:
            return []

        parts = re.split(r"[,，;；\n\t]+", payload)

        items: List[str] = []
        seen = set()
        for part in parts:
            part = cls._normalize_preview_text(part)
            if not part:
                continue

            key = cls._canonicalize_item(part)
            if not key:
                continue
            if key in seen:
                continue

            seen.add(key)
            items.append(part)

        return items

    @staticmethod
    def _safe_div(n: float, d: float) -> float:
        return float(n) / float(d) if d else 0.0

    @staticmethod
    def _dedup_preserve_order(items):
        out = []
        seen = set()
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out



    @classmethod
    def _compute_item_set_metrics(
        cls,
        gold_items: List[str],
        pred_items: List[str],
    ) -> Dict[str, Any]:
        gold_map: Dict[str, str] = {}
        for x in gold_items:
            k = cls._canonicalize_item(x)
            if k and k not in gold_map:
                gold_map[k] = x

        pred_map: Dict[str, str] = {}
        for x in pred_items:
            k = cls._canonicalize_item(x)
            if k and k not in pred_map:
                pred_map[k] = x

        gold_keys = set(gold_map.keys())
        pred_keys = set(pred_map.keys())

        matched_keys = gold_keys & pred_keys
        missing_keys = [k for k in gold_map.keys() if k not in pred_keys]
        extra_keys = [k for k in pred_map.keys() if k not in gold_keys]

        tp = len(matched_keys)
        fp = len(extra_keys)
        fn = len(missing_keys)

        precision = cls._safe_div(tp, tp + fp)
        recall = cls._safe_div(tp, tp + fn)
        f1 = cls._safe_div(2 * precision * recall, precision + recall)
        exact_match = (gold_keys == pred_keys)

        return {
            "gold_items": list(gold_map.values()),
            "pred_items": list(pred_map.values()),
            "matched_items": [gold_map[k] for k in gold_map.keys() if k in pred_keys],
            "missing_items": [gold_map[k] for k in missing_keys],
            "extra_items": [pred_map[k] for k in extra_keys],
            "gold_count": len(gold_keys),
            "pred_count": len(pred_keys),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
        }

    def _should_run_train_preview(self) -> bool:
        every = int(getattr(self.sft_config, "train_preview_every", 0) or 0)
        if every <= 0:
            return False
        if self.train_dataset is None:
            return False
        if self.model is None:
            return False
        if not self.is_world_process_zero():
            return False
        if self.state.global_step <= 0:
            return False
        if self.state.global_step % every != 0:
            return False
        if self.state.global_step == self._last_train_preview_step:
            return False
        return True

    def _resolve_preview_dataset_and_indices(self, split: str):
        # Only train preview is supported. Eval preview code has been removed;
        # formal validation is reported through evaluate() metrics only.
        if split != "train":
            raise ValueError("Only train preview is supported. Eval preview has been removed.")

        override_dataset = getattr(self, "_train_preview_override_dataset", None)
        if override_dataset is not None:
            dataset_name = getattr(self, "_train_preview_override_dataset_name", "train")
            indices = list(getattr(self, "_train_preview_override_indices", []) or [])
            return override_dataset, dataset_name, indices

        dataset = self.train_dataset
        dataset_name = "train"
        if dataset is None:
            return None, dataset_name, []
        indices = self._resolve_train_preview_indices()
        return dataset, dataset_name, indices
    def _decode_generated_text_from_output_ids(
        self,
        *,
        tokenizer,
        gen_ids: torch.LongTensor,
        input_len: int,
        kept_answer_prefix_ids: torch.Tensor,
        keep_prefix_len: int,
    ) -> str:
        pred_suffix_ids = gen_ids[0, input_len:]
        pred_suffix_text = tokenizer.decode(
            pred_suffix_ids, skip_special_tokens=False
        )
        pred_suffix_text = self._strip_summary_tags(pred_suffix_text)

        if keep_prefix_len > 0:
            kept_answer_prefix_text = tokenizer.decode(
                kept_answer_prefix_ids,
                skip_special_tokens=False,
            )
            pred_text = kept_answer_prefix_text + pred_suffix_text
        else:
            pred_text = pred_suffix_text

        return self._strip_summary_tags(pred_text)

    def _run_dataset_preview(self, split: str = "train") -> None:
        if split != "train":
            raise ValueError("Only train preview is supported. Eval preview has been removed.")

        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            print(f"[{split}-preview] skipped: tokenizer/processing_class is missing.", flush=True)
            return

        dataset, dataset_name, indices = self._resolve_preview_dataset_and_indices(split)
        if dataset is None:
            print(f"[{split}-preview] skipped: dataset is missing.", flush=True)
            return
        if len(indices) == 0:
            print(f"[{split}-preview] skipped: no preview indices available.", flush=True)
            return

        model = self.model
        gen_model = _unwrap_model(model)
        was_training = model.training
        model.eval()

        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0

        max_new_tokens = int(
            getattr(self.sft_config, "train_preview_max_new_tokens", 0)
            or getattr(self.args, "generation_max_length", None)
            or 128
        )
        do_sample = bool(getattr(self.sft_config, "train_preview_do_sample", False))
        num_beams = int(getattr(self.sft_config, "train_preview_num_beams", 1) or 1)
        keep_prefix_len_cfg = int(getattr(self.sft_config, "train_preview_answer_prefix_len", 2) or 2)
        repetition_penalty = float(
            getattr(self.sft_config, "train_preview_repetition_penalty", 1.0) or 1.0
        )

        gen_kwargs = dict(
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

        no_repeat_ngram_size = int(getattr(self.sft_config, "train_preview_no_repeat_ngram_size", 0) or 0)
        if no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

        old_use_cache = getattr(gen_model.config, "use_cache", None)
        if old_use_cache is not None:
            gen_model.config.use_cache = True

        device = self.args.device

        print("\n" + "=" * 100, flush=True)
        print(
            f"[{split}-preview] global_step={self.state.global_step}  dataset={dataset_name}  indices={indices}",
            flush=True,
        )
        print("=" * 100, flush=True)

        preview_stats = {
            "sample_count": 0,
            "exact_match_count": 0,
            "tp_total": 0,
            "fp_total": 0,
            "fn_total": 0,
            "macro_precision_sum": 0.0,
            "macro_recall_sum": 0.0,
            "macro_f1_sum": 0.0,
            "gold_count_sum": 0,
            "pred_count_sum": 0,
        }

        try:
            with torch.no_grad():
                for i, ds_idx in enumerate(indices):
                    try:
                        raw = dataset[ds_idx]

                        if "input_ids" not in raw or "labels" not in raw:
                            print("-" * 100, flush=True)
                            print(f"[{i}] {split}_idx={ds_idx}", flush=True)
                            print(
                                "Skipped: sample does not contain `input_ids` or `labels`.",
                                flush=True,
                            )
                            continue

                        input_ids = self._to_1d_long_tensor(raw["input_ids"]).unsqueeze(0).to(device)
                        labels = self._to_1d_long_tensor(raw["labels"]).unsqueeze(0).to(device)

                        if "attention_mask" in raw and raw["attention_mask"] is not None:
                            attention_mask = (
                                self._to_1d_long_tensor(raw["attention_mask"]).unsqueeze(0).to(device)
                            )
                        else:
                            attention_mask = torch.ones_like(
                                input_ids, dtype=torch.long, device=device
                            )

                        answer_pos = labels.ne(IGNORE_INDEX)
                        answer_indices = answer_pos[0].nonzero(as_tuple=False).squeeze(-1)

                        ref_ids = labels[0][labels[0].ne(IGNORE_INDEX)]
                        ref_text = tokenizer.decode(ref_ids, skip_special_tokens=False)
                        ref_text = self._strip_summary_tags(ref_text)

                        keep_prefix_len = 0
                        kept_answer_prefix_ids = ref_ids[:0]
                        if answer_indices.numel() > 0 and ref_ids.numel() > 0:
                            keep_prefix_len = min(
                                keep_prefix_len_cfg,
                                int(answer_indices.numel()),
                                int(ref_ids.numel()),
                            )
                            kept_answer_prefix_ids = ref_ids[:keep_prefix_len]

                        if answer_indices.numel() > 0:
                            prompt_len = int(answer_indices[0].item())
                        else:
                            prompt_len = int(input_ids.size(1))

                        prompt_only_ids = input_ids[0, :prompt_len].to(device)
                        kept_answer_prefix_ids = kept_answer_prefix_ids.to(device)

                        gen_input_ids_1d = torch.cat(
                            [prompt_only_ids, kept_answer_prefix_ids], dim=0
                        )
                        gen_input_ids = gen_input_ids_1d.unsqueeze(0)
                        gen_attention_mask = torch.ones_like(
                            gen_input_ids, dtype=torch.long, device=device
                        )

                        prompt_text = tokenizer.decode(
                            gen_input_ids[0], skip_special_tokens=False
                        )

                        local_gen_kwargs = dict(gen_kwargs)
                        processors = LogitsProcessorList()
                        existing_processors = local_gen_kwargs.pop("logits_processor", None)
                        if existing_processors is not None:
                            if isinstance(existing_processors, LogitsProcessorList):
                                processors.extend(list(existing_processors))
                            elif isinstance(existing_processors, list):
                                processors.extend(existing_processors)
                            else:
                                processors.append(existing_processors)

                        if repetition_penalty > 1.0:
                            processors.append(
                                RepetitionPenaltyLogitsProcessorExceptTokens(
                                    penalty=float(repetition_penalty),
                                    prompt_ignore_length=int(prompt_len),
                                    exempt_token_ids=None,
                                )
                            )

                        if len(processors) > 0:
                            local_gen_kwargs["logits_processor"] = processors

                        gen_ids = gen_model.generate(
                            input_ids=gen_input_ids,
                            attention_mask=gen_attention_mask,
                            **local_gen_kwargs,
                        )

                        L = int(gen_input_ids.shape[1])
                        pred_text = self._decode_generated_text_from_output_ids(
                            tokenizer=tokenizer,
                            gen_ids=gen_ids,
                            input_len=L,
                            kept_answer_prefix_ids=kept_answer_prefix_ids,
                            keep_prefix_len=keep_prefix_len,
                        )

                        gold_payload = self._extract_answer_payload(ref_text)
                        pred_payload = self._extract_answer_payload(pred_text)

                        gold_items = self._payload_to_items(gold_payload)
                        pred_items = self._payload_to_items(pred_payload)

                        item_metrics = self._compute_item_set_metrics(
                            gold_items=gold_items,
                            pred_items=pred_items,
                        )

                        preview_stats["sample_count"] += 1
                        preview_stats["exact_match_count"] += int(item_metrics["exact_match"])
                        preview_stats["tp_total"] += int(item_metrics["tp"])
                        preview_stats["fp_total"] += int(item_metrics["fp"])
                        preview_stats["fn_total"] += int(item_metrics["fn"])
                        preview_stats["macro_precision_sum"] += float(item_metrics["precision"])
                        preview_stats["macro_recall_sum"] += float(item_metrics["recall"])
                        preview_stats["macro_f1_sum"] += float(item_metrics["f1"])
                        preview_stats["gold_count_sum"] += int(item_metrics["gold_count"])
                        preview_stats["pred_count_sum"] += int(item_metrics["pred_count"])

                        meta_parts = [f"{split}_idx={ds_idx}"]
                        for key in ("task", "task_name", "dataset"):
                            if key in raw and isinstance(raw[key], str):
                                meta_parts.append(f"{key}={raw[key]}")
                        header = "  ".join(meta_parts)

                        print("-" * 100, flush=True)
                        print(f"[{i}] {header}", flush=True)
                        print("[PROMPT]", flush=True)
                        print(prompt_text if prompt_text else "<EMPTY PROMPT>", flush=True)
                        print("\n[GOLD]", flush=True)
                        print(ref_text if ref_text else "<EMPTY GOLD>", flush=True)
                        print("\n[PRED]", flush=True)
                        print(pred_text if pred_text else "<EMPTY PRED>", flush=True)

                        print("\n[PARSED]", flush=True)
                        print(f"gold_payload={gold_payload!r}", flush=True)
                        print(f"pred_payload={pred_payload!r}", flush=True)

                        if "kptime" in str(dataset_name).lower():
                            ref_fields = self._mover_extract_field_payloads(ref_text)
                            pred_fields = self._mover_extract_field_payloads(pred_text)
                            print("\n[PARSED KPTime FIELDS]", flush=True)
                            print("Reference Categories:", ref_fields.get("categories_payload", ""), flush=True)
                            print("Reference Keywords  :", ref_fields.get("keywords_payload", ""), flush=True)
                            print("Prediction Categories:", pred_fields.get("categories_payload", ""), flush=True)
                            print("Prediction Keywords  :", pred_fields.get("keywords_payload", ""), flush=True)

                        print(f"gold_items={item_metrics['gold_items']}", flush=True)
                        print(f"pred_items={item_metrics['pred_items']}", flush=True)

                        print("\n[ITEM-METRICS]", flush=True)
                        print(
                            f"gold_count={item_metrics['gold_count']}  "
                            f"pred_count={item_metrics['pred_count']}  "
                            f"tp={item_metrics['tp']}  fp={item_metrics['fp']}  fn={item_metrics['fn']}",
                            flush=True,
                        )
                        print(
                            f"precision={item_metrics['precision']:.4f}  "
                            f"recall={item_metrics['recall']:.4f}  "
                            f"f1={item_metrics['f1']:.4f}  "
                            f"exact_match={item_metrics['exact_match']}",
                            flush=True,
                        )

                        print("\n[ITEM-DIFF]", flush=True)
                        print(f"missing_items={item_metrics['missing_items']}", flush=True)
                        print(f"extra_items={item_metrics['extra_items']}", flush=True)

                    except Exception as e:
                        print("-" * 100, flush=True)
                        print(
                            f"[WARN] {split}-preview sample failed: idx={ds_idx}, error={e}",
                            flush=True,
                        )
                        continue

            n = int(preview_stats["sample_count"])
            if n > 0:
                tp_total = int(preview_stats["tp_total"])
                fp_total = int(preview_stats["fp_total"])
                fn_total = int(preview_stats["fn_total"])

                micro_precision = self._safe_div(tp_total, tp_total + fp_total)
                micro_recall = self._safe_div(tp_total, tp_total + fn_total)
                micro_f1 = self._safe_div(
                    2 * micro_precision * micro_recall,
                    micro_precision + micro_recall,
                )

                macro_precision = self._safe_div(preview_stats["macro_precision_sum"], n)
                macro_recall = self._safe_div(preview_stats["macro_recall_sum"], n)
                macro_f1 = self._safe_div(preview_stats["macro_f1_sum"], n)
                exact_match_acc = self._safe_div(preview_stats["exact_match_count"], n)

                avg_gold_count = self._safe_div(preview_stats["gold_count_sum"], n)
                avg_pred_count = self._safe_div(preview_stats["pred_count_sum"], n)

                print("-" * 100, flush=True)
                print(f"[{split.upper()}-PREVIEW-SUMMARY]", flush=True)
                print(
                    f"samples={n}  exact_match_acc={exact_match_acc:.4f}  "
                    f"avg_gold_count={avg_gold_count:.4f}  avg_pred_count={avg_pred_count:.4f}",
                    flush=True,
                )
                print(
                    f"micro_precision={micro_precision:.4f}  "
                    f"micro_recall={micro_recall:.4f}  "
                    f"micro_f1={micro_f1:.4f}",
                    flush=True,
                )
                print(
                    f"macro_precision={macro_precision:.4f}  "
                    f"macro_recall={macro_recall:.4f}  "
                    f"macro_f1={macro_f1:.4f}",
                    flush=True,
                )
                print(
                    f"tp_total={tp_total}  fp_total={fp_total}  fn_total={fn_total}",
                    flush=True,
                )

        finally:
            if old_use_cache is not None:
                gen_model.config.use_cache = old_use_cache
            if was_training:
                model.train()

        print("=" * 100 + "\n", flush=True)

    def _run_train_preview(self) -> None:
        if not getattr(self, "_use_epoch_ratio_sampling", False):
            self._run_dataset_preview(split="train")
            self._last_train_preview_step = int(self.state.global_step)
            return

        wrapper = getattr(self, "_epoch_ratio_train_dataset", None)
        sampler = getattr(self, "_epoch_ratio_sampler", None)
        if wrapper is None or sampler is None:
            self._run_dataset_preview(split="train")
            self._last_train_preview_step = int(self.state.global_step)
            return

        source_names = list(getattr(wrapper, "source_names", []) or [])
        if "cnn_dm" in source_names and "kptime" in source_names:
            source_names = ["cnn_dm", "kptime"]
        if len(source_names) < 2:
            self._run_dataset_preview(split="train")
            self._last_train_preview_step = int(self.state.global_step)
            return

        total_n = int(getattr(self.sft_config, "train_preview_num_samples", 5) or 5)
        total_n = max(1, total_n)
        n_first = (total_n + 1) // 2
        n_second = total_n // 2
        per_source_target = {source_names[0]: n_first, source_names[1]: n_second}

        current_global = getattr(sampler, "current_epoch_global_indices", None)
        grouped_local = {name: [] for name in source_names}

        if current_global:
            for global_idx in current_global:
                try:
                    source_name, local_idx = wrapper._locate(int(global_idx))
                except Exception:
                    continue
                if source_name in grouped_local and len(grouped_local[source_name]) < per_source_target[source_name]:
                    grouped_local[source_name].append(int(local_idx))
                if all(len(grouped_local[name]) >= per_source_target[name] for name in source_names):
                    break

        # Fallback: if no current epoch has been sampled yet, draw deterministic
        # preview rows from each source pool directly.
        for name in source_names:
            target = per_source_target[name]
            if len(grouped_local[name]) >= target:
                continue
            pool_len = len(wrapper.train_pools[name])
            used = set(grouped_local[name])
            for idx in range(pool_len):
                if idx in used:
                    continue
                grouped_local[name].append(int(idx))
                if len(grouped_local[name]) >= target:
                    break

        display_names = {
            "cnn_dm": "CNN/DM",
            "kptime": "KPTime",
        }

        print("\n" + "=" * 100, flush=True)
        print(
            f"[TRAIN PREVIEW] global_step={self.state.global_step} epoch={getattr(self.state, 'epoch', None)}",
            flush=True,
        )
        print("=" * 100, flush=True)

        old_dataset = getattr(self, "_train_preview_override_dataset", None)
        old_name = getattr(self, "_train_preview_override_dataset_name", None)
        old_indices = getattr(self, "_train_preview_override_indices", None)

        try:
            for name in source_names:
                indices = grouped_local.get(name, [])
                if not indices:
                    continue
                print("\n" + "=" * 100, flush=True)
                print(f"===== {display_names.get(name, name)} =====", flush=True)
                print("=" * 100, flush=True)

                self._train_preview_override_dataset = wrapper.train_pools[name]
                self._train_preview_override_dataset_name = f"train:{name}"
                self._train_preview_override_indices = indices
                self._run_dataset_preview(split="train")
        finally:
            self._train_preview_override_dataset = old_dataset
            self._train_preview_override_dataset_name = old_name
            self._train_preview_override_indices = old_indices
            self._last_train_preview_step = int(self.state.global_step)

    def _resolve_train_preview_indices(self) -> List[int]:
        if self.train_dataset is None:
            return []

        manual = getattr(self.sft_config, "train_preview_indices", None)
        if manual:
            return [int(i) for i in manual if 0 <= int(i) < len(self.train_dataset)]

        n = int(getattr(self.sft_config, "train_preview_num_samples", 5) or 5)
        return list(range(min(n, len(self.train_dataset))))

    def _should_run_eval_loss(self) -> bool:
        return bool(getattr(self.sft_config, "eval_run_loss", True))

    def _should_run_eval_mover(self) -> bool:
        return bool(getattr(self.sft_config, "eval_run_mover", True))

    def _maybe_subsample_dataset(
        self,
        ds,
        sample_size: int = 0,
        sample_mode: str = "fixed",
        sample_seed: int = 42,
    ):
        sample_size = int(sample_size or 0)
        if sample_size <= 0:
            return ds

        try:
            total_n = len(ds)
        except Exception:
            return ds

        if total_n <= sample_size:
            return ds

        sample_mode = str(sample_mode or "fixed").lower()
        base_seed = int(sample_seed or 42)

        if sample_mode == "per_eval_random":
            seed = base_seed + int(getattr(self.state, "global_step", 0) or 0)
        elif sample_mode == "fixed":
            seed = base_seed
        else:
            raise ValueError("sample_mode must be 'fixed' or 'per_eval_random'.")

        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total_n), sample_size))

        if hasattr(ds, "select"):
            return ds.select(indices)

        return torch.utils.data.Subset(ds, indices)

    def _left_pad_prompt_batch(
        self,
        seqs: List[torch.Tensor],
        pad_id: int,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(seqs) == 0:
            raise ValueError("seqs must not be empty")

        max_len = max(int(x.numel()) for x in seqs)

        input_ids = torch.full(
            (len(seqs), max_len),
            fill_value=int(pad_id),
            dtype=seqs[0].dtype,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(seqs), max_len),
            dtype=torch.long,
            device=device,
        )

        for i, s in enumerate(seqs):
            L = int(s.numel())
            input_ids[i, max_len - L:] = s
            attention_mask[i, max_len - L:] = 1

        return input_ids, attention_mask

    def _resolve_eval_metric_indices(
        self,
        dataset_len: int,
        sample_size: int,
        sample_mode: str = "fixed",
        sample_seed: int = 42,
    ) -> List[int]:
        if dataset_len <= 0:
            return []

        sample_size = int(sample_size or 0)

        if sample_size <= 0 or sample_size >= dataset_len:
            return list(range(dataset_len))

        sample_mode = str(sample_mode or "fixed").lower().strip()

        if sample_mode == "fixed":
            rng = random.Random(int(sample_seed))
        elif sample_mode == "per_eval_random":
            step = int(getattr(self.state, "global_step", 0) or 0)
            rng = random.Random(int(sample_seed) + step)
        else:
            raise ValueError("sample_mode must be 'fixed' or 'per_eval_random'.")

        return sorted(rng.sample(range(dataset_len), sample_size))

    @staticmethod
    def _get_single_token_id(tokenizer, text: str) -> Optional[int]:
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        if len(ids) == 1:
            return int(ids[0])
        return None

    def _build_eval_eos_token_ids(self, tokenizer):
        eos_ids = []

        if tokenizer.eos_token_id is not None:
            eos_ids.append(int(tokenizer.eos_token_id))

        for tok in ["<|end|>", "<|endoftext|>"]:
            tid = self._get_single_token_id(tokenizer, tok)
            if tid is not None:
                eos_ids.append(tid)

        eos_ids = sorted(set(eos_ids))

        if len(eos_ids) == 0:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    @staticmethod
    def _mover_remove_special_tokens(text: str) -> str:
        text = str(text)
        for tok in [
            "<|end|>",
            "<|endoftext|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
        ]:
            text = text.replace(tok, "")
        return text.strip()

    @staticmethod
    def _mover_normalize_space(text: str) -> str:
        text = str(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def _mover_clean_payload_text(cls, text: str) -> str:
        text = cls._mover_remove_special_tokens(text)
        text = cls._mover_normalize_space(text)
        return text.strip()

    @classmethod
    def _clean_summary_for_metric(cls, text: str) -> str:
        """
        Summary-only metric cleaning for MoverScore-like OT.

        It removes only leading summary prefixes, not later colons in content:
            Summary: The report says: inflation increased.
        ->
            The report says: inflation increased.
        """
        text = cls._mover_remove_special_tokens(text)
        text = str(text).replace("[SUMMARY]", " ").replace("[/SUMMARY]", " ")
        text = re.sub(r"<\|[^>]+?\|>", " ", text)
        text = cls._mover_normalize_space(text)
        text = re.sub(
            r"^\s*(summary|summaries|\u6458\u8981)\s*[:：]\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        return cls._mover_normalize_space(text)

    @classmethod
    def _mover_extract_field_payloads(cls, text: str) -> Dict[str, Any]:
        """
        Extract:
          categories: ...
          keywords: ...

        This matches the logic of the standalone MoverScore-like OT script.
        """
        text_clean = cls._mover_remove_special_tokens(text)
        lower = text_clean.lower()

        cat_match = re.search(r"categories\s*:", lower)
        kw_match = re.search(r"keywords\s*:", lower)

        categories_payload = ""
        keywords_payload = ""

        has_categories = cat_match is not None
        has_keywords = kw_match is not None
        line_order_valid = False

        if cat_match and kw_match:
            cat_start = cat_match.end()
            kw_label_start = kw_match.start()
            kw_start = kw_match.end()

            if cat_match.start() < kw_match.start():
                categories_payload = text_clean[cat_start:kw_label_start]
                keywords_payload = text_clean[kw_start:]
                line_order_valid = True
            else:
                keywords_payload = text_clean[kw_start:cat_match.start()]
                categories_payload = text_clean[cat_start:]
                line_order_valid = False

        elif cat_match:
            categories_payload = text_clean[cat_match.end():]

        elif kw_match:
            keywords_payload = text_clean[kw_match.end():]

        return {
            "categories_payload": cls._mover_clean_payload_text(categories_payload),
            "keywords_payload": cls._mover_clean_payload_text(keywords_payload),
            "has_categories": has_categories,
            "has_keywords": has_keywords,
            "line_order_valid": line_order_valid,
        }

    def _resolve_mover_dtype(self):
        dtype = getattr(self.sft_config, "eval_mover_dtype", "float16")

        if isinstance(dtype, torch.dtype):
            return dtype

        dtype = str(dtype).lower().strip()

        if dtype in {"fp16", "float16", "torch.float16"}:
            return torch.float16
        if dtype in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp32", "float32", "torch.float32"}:
            return torch.float32

        raise ValueError(
            "eval_mover_dtype must be one of: "
            "'float16', 'bfloat16', 'float32'."
        )

    def _resolve_mover_device(self) -> str:
        device = getattr(self.sft_config, "eval_mover_device", None)
        if device is not None:
            return str(device)

        if torch.cuda.is_available():
            return str(self.args.device)

        return "cpu"

    def _get_mover_encoder(self):
        """
        Load and cache the RoBERTa encoder.
        """
        model_path = str(getattr(self.sft_config, "eval_mover_model_path"))
        device = self._resolve_mover_device()
        dtype = self._resolve_mover_dtype()

        if device == "cpu":
            dtype = torch.float32

        cache_encoder = bool(getattr(self.sft_config, "eval_mover_cache_encoder", True))
        encoder_key = (model_path, str(dtype), device)

        if (
            cache_encoder
            and self._mover_tokenizer is not None
            and self._mover_model is not None
            and self._mover_encoder_key == encoder_key
        ):
            return self._mover_tokenizer, self._mover_model

        if self.is_world_process_zero():
            print(
                "[MoverScore-like OT] loading encoder: "
                f"path={model_path!r}, dtype={dtype}, device={device}",
                flush=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        try:
            model = AutoModel.from_pretrained(
                model_path,
                dtype=dtype,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )

        model = model.to(device)
        model.eval()

        if cache_encoder:
            self._mover_tokenizer = tokenizer
            self._mover_model = model
            self._mover_encoder_key = encoder_key

        return tokenizer, model

    def _release_mover_encoder(self) -> None:
        self._mover_tokenizer = None
        self._mover_model = None
        self._mover_encoder_key = None

    @torch.no_grad()
    def _mover_encode_texts_first_last_avg_hidden_batch(
        self,
        texts: Sequence[str],
        tokenizer,
        model,
        max_length: int = 512,
        first_layer_index: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract (first_hidden + last_hidden) / 2 for one batch.
        Preserve the evaluation method while shortening the lifetime of large
        tensors on the GPU.
        """
        device = next(model.parameters()).device

        texts = [self._mover_clean_payload_text(x) for x in texts]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if first_layer_index < 0 or first_layer_index >= len(hidden_states):
            available = len(hidden_states) - 1
            del outputs, hidden_states, inputs
            raise ValueError(
                f"first_layer_index={first_layer_index} out of range. "
                f"Available hidden states: 0 ~ {available}"
            )

        # Preserve the original evaluation definition:
        # (first_hidden + last_hidden) / 2. Use clone plus in-place add/mul to
        # avoid large intermediate tensors for addition and division.
        mixed_hidden = hidden_states[first_layer_index].clone()
        mixed_hidden.add_(hidden_states[-1])
        mixed_hidden.mul_(0.5)

        # mixed_hidden is now independent; release full hidden states and outputs.
        del outputs, hidden_states

        attention_mask = inputs["attention_mask"].to(torch.bool)
        input_ids = inputs["input_ids"]

        special_mask = torch.zeros_like(
            attention_mask,
            dtype=torch.bool,
        )

        for special_id in tokenizer.all_special_ids:
            special_mask.logical_or_(input_ids.eq(int(special_id)))

        valid_mask = attention_mask & (~special_mask)

        # Later scoring uses only mixed_hidden and valid_mask. Do not return
        # input_ids, so the encoded result does not retain them.
        del input_ids, attention_mask, special_mask, inputs

        return {
            "mixed_hidden": mixed_hidden,
            "valid_mask": valid_mask,
            "texts": texts,
        }

    @staticmethod
    def _mover_score_from_batch_hidden(
        mixed_hidden: torch.Tensor,
        valid_mask: torch.Tensor,
        pred_index: int,
        ref_index: int,
        clamp_similarity: bool = True,
    ) -> Dict[str, float]:
        pred_hidden = mixed_hidden[pred_index][valid_mask[pred_index]]
        ref_hidden = mixed_hidden[ref_index][valid_mask[ref_index]]

        pred_count = int(pred_hidden.size(0))
        ref_count = int(ref_hidden.size(0))

        if pred_count == 0 and ref_count == 0:
            del pred_hidden, ref_hidden
            return {
                "score": 1.0,
                "ot_distance": 0.0,
                "pred_token_count": 0,
                "ref_token_count": 0,
            }

        if pred_count == 0 or ref_count == 0:
            del pred_hidden, ref_hidden
            return {
                "score": 0.0,
                "ot_distance": 1.0,
                "pred_token_count": pred_count,
                "ref_token_count": ref_count,
            }

        pred_hidden = F.normalize(
            pred_hidden.float(),
            p=2,
            dim=-1,
        )

        ref_hidden = F.normalize(
            ref_hidden.float(),
            p=2,
            dim=-1,
        )

        sim = pred_hidden @ ref_hidden.T
        del pred_hidden, ref_hidden

        if clamp_similarity:
            sim.clamp_(min=0.0, max=1.0)

        # Original semantics: dist = (1 - sim).clamp_min(0). Reuse sim storage
        # here to reduce matrix-sized temporary tensors.
        sim.neg_()
        sim.add_(1.0)
        sim.clamp_min_(0.0)

        dist_np = sim.detach().cpu().numpy().astype(np.float64)
        del sim

        pred_weights = np.ones(pred_count, dtype=np.float64)
        ref_weights = np.ones(ref_count, dtype=np.float64)

        pred_weights = pred_weights / pred_weights.sum()
        ref_weights = ref_weights / ref_weights.sum()

        ot_distance = float(
            ot.emd2(
                pred_weights,
                ref_weights,
                dist_np,
            )
        )
        del dist_np, pred_weights, ref_weights

        score = 1.0 - ot_distance
        score = max(0.0, min(1.0, float(score)))

        return {
            "score": score,
            "ot_distance": ot_distance,
            "pred_token_count": pred_count,
            "ref_token_count": ref_count,
        }

    def _compute_category_keyword_mover_like_score(
        self,
        prediction: str,
        reference: str,
        roberta_tokenizer,
        roberta_model,
        max_length: int = 512,
        first_layer_index: int = 0,
    ) -> Dict[str, Any]:
        pred_parsed = self._mover_extract_field_payloads(prediction)
        ref_parsed = self._mover_extract_field_payloads(reference)

        pred_category = pred_parsed["categories_payload"]
        pred_keyword = pred_parsed["keywords_payload"]

        ref_category = ref_parsed["categories_payload"]
        ref_keyword = ref_parsed["keywords_payload"]

        batch_texts = [
            pred_category,
            pred_keyword,
            ref_category,
            ref_keyword,
        ]

        encoded = self._mover_encode_texts_first_last_avg_hidden_batch(
            texts=batch_texts,
            tokenizer=roberta_tokenizer,
            model=roberta_model,
            max_length=max_length,
            first_layer_index=first_layer_index,
        )

        mixed_hidden = encoded["mixed_hidden"]
        valid_mask = encoded["valid_mask"]

        category_raw = self._mover_score_from_batch_hidden(
            mixed_hidden=mixed_hidden,
            valid_mask=valid_mask,
            pred_index=0,
            ref_index=2,
            clamp_similarity=True,
        )

        keyword_raw = self._mover_score_from_batch_hidden(
            mixed_hidden=mixed_hidden,
            valid_mask=valid_mask,
            pred_index=1,
            ref_index=3,
            clamp_similarity=True,
        )

        category_score = float(category_raw["score"])
        keyword_score = float(keyword_raw["score"])
        final_score = float((category_score + keyword_score) / 2.0)

        return {
            "pred_category_payload": pred_category,
            "ref_category_payload": ref_category,
            "pred_keyword_payload": pred_keyword,
            "ref_keyword_payload": ref_keyword,
            "category_score": category_score,
            "keyword_score": keyword_score,
            "final_score": final_score,
        }

    def _eval_mover_like_from_texts(
        self,
        preds: Sequence[str],
        refs: Sequence[str],
        prefix_name: str = "valid",
    ) -> Dict[str, float]:
        """
        Batched MoverScore-like OT evaluation.

        Previous version encoded 4 texts per sample in a loop.  This version
        encodes multiple pred/ref pairs in one RoBERTa forward pass:
            [pred_cat, pred_kw, ref_cat, ref_kw] * batch_size
        OT is still computed per sample on CPU, but the expensive encoder pass is batched.
        """
        roberta_tokenizer, roberta_model = self._get_mover_encoder()

        max_length = int(getattr(self.sft_config, "eval_mover_max_length", 512) or 512)
        first_layer_index = int(getattr(self.sft_config, "eval_mover_first_layer_index", 0) or 0)
        encoder_batch_size = int(getattr(self.sft_config, "eval_mover_encoder_batch_size", 8) or 8)
        encoder_batch_size = max(1, encoder_batch_size)

        category_scores: List[float] = []
        keyword_scores: List[float] = []
        final_scores: List[float] = []

        pair_count = min(len(preds), len(refs))

        if self.is_world_process_zero():
            print(
                f"[MoverScore-like OT] scoring pairs={pair_count} "
                f"encoder_pair_batch_size={encoder_batch_size}",
                flush=True,
            )

        for start in range(0, pair_count, encoder_batch_size):
            end = min(start + encoder_batch_size, pair_count)
            batch_texts: List[str] = []

            for pred, ref in zip(preds[start:end], refs[start:end]):
                pred_parsed = self._mover_extract_field_payloads(pred)
                ref_parsed = self._mover_extract_field_payloads(ref)

                batch_texts.extend(
                    [
                        pred_parsed["categories_payload"],
                        pred_parsed["keywords_payload"],
                        ref_parsed["categories_payload"],
                        ref_parsed["keywords_payload"],
                    ]
                )

            encoded = self._mover_encode_texts_first_last_avg_hidden_batch(
                texts=batch_texts,
                tokenizer=roberta_tokenizer,
                model=roberta_model,
                max_length=max_length,
                first_layer_index=first_layer_index,
            )

            mixed_hidden = encoded["mixed_hidden"]
            valid_mask = encoded["valid_mask"]

            local_pair_count = end - start
            for local_i in range(local_pair_count):
                base = local_i * 4

                category_raw = self._mover_score_from_batch_hidden(
                    mixed_hidden=mixed_hidden,
                    valid_mask=valid_mask,
                    pred_index=base,
                    ref_index=base + 2,
                    clamp_similarity=True,
                )

                keyword_raw = self._mover_score_from_batch_hidden(
                    mixed_hidden=mixed_hidden,
                    valid_mask=valid_mask,
                    pred_index=base + 1,
                    ref_index=base + 3,
                    clamp_similarity=True,
                )

                category_score = float(category_raw["score"])
                keyword_score = float(keyword_raw["score"])
                final_score = float((category_score + keyword_score) / 2.0)

                category_scores.append(category_score)
                keyword_scores.append(keyword_score)
                final_scores.append(final_score)

            # Release large temporary tensors before the next RoBERTa batch.
            del encoded, mixed_hidden, valid_mask

        n = max(len(final_scores), 1)

        avg_category = float(sum(category_scores) / n) if category_scores else 0.0
        avg_keyword = float(sum(keyword_scores) / n) if keyword_scores else 0.0
        avg_final = float(sum(final_scores) / n) if final_scores else 0.0

        return {
            f"eval_{prefix_name}_mover_category_score": avg_category,
            f"eval_{prefix_name}_mover_keyword_score": avg_keyword,
            f"eval_{prefix_name}_mover_final_score": avg_final,
        }

    # ------------------------------------------------------------
    # Summary mover metric
    # ------------------------------------------------------------
    @torch.no_grad()
    def _eval_mover_summary_from_texts(
        self,
        preds: Sequence[str],
        refs: Sequence[str],
        prefix_name: str = "valid",
    ) -> Dict[str, float]:
        """
        Summary-only MoverScore-like OT.

        The older category/keyword Mover functions remain above for compatibility.
        The official summary evaluation uses this function and returns only
        eval_{prefix}_mover_score, without the category/keyword/final columns.
        """
        roberta_tokenizer, roberta_model = self._get_mover_encoder()

        max_length = int(getattr(self.sft_config, "eval_mover_max_length", 512) or 512)
        first_layer_index = int(getattr(self.sft_config, "eval_mover_first_layer_index", 0) or 0)
        encoder_batch_size = int(getattr(self.sft_config, "eval_mover_encoder_batch_size", 8) or 8)
        encoder_batch_size = max(1, encoder_batch_size)

        pair_count = min(len(preds), len(refs))
        if pair_count == 0:
            return {f"eval_{prefix_name}_mover_score": 0.0}

        clean_preds = [self._clean_summary_for_metric(x) for x in list(preds[:pair_count])]
        clean_refs = [self._clean_summary_for_metric(x) for x in list(refs[:pair_count])]

        if self.is_world_process_zero():
            print(
                f"[MoverScore-like OT] scoring summary pairs={pair_count} "
                f"encoder_pair_batch_size={encoder_batch_size} max_length={max_length}",
                flush=True,
            )

        scores: List[float] = []
        batch_starts = list(range(0, pair_count, encoder_batch_size))
        if bool(getattr(self.sft_config, "eval_metric_show_progress", True)) and self.is_world_process_zero():
            try:
                from tqdm.auto import tqdm
                batch_starts = tqdm(batch_starts, desc="[MoverScore-like OT]", leave=False)
            except Exception:
                pass

        for start in batch_starts:
            end = min(start + encoder_batch_size, pair_count)
            batch_texts: List[str] = []
            for pred, ref in zip(clean_preds[start:end], clean_refs[start:end]):
                batch_texts.extend([pred, ref])

            encoded = self._mover_encode_texts_first_last_avg_hidden_batch(
                texts=batch_texts,
                tokenizer=roberta_tokenizer,
                model=roberta_model,
                max_length=max_length,
                first_layer_index=first_layer_index,
            )

            mixed_hidden = encoded["mixed_hidden"]
            valid_mask = encoded["valid_mask"]

            local_pair_count = end - start
            for local_i in range(local_pair_count):
                base = local_i * 2
                raw = self._mover_score_from_batch_hidden(
                    mixed_hidden=mixed_hidden,
                    valid_mask=valid_mask,
                    pred_index=base,
                    ref_index=base + 1,
                    clamp_similarity=True,
                )
                scores.append(float(raw["score"]))

            del encoded, mixed_hidden, valid_mask

        avg_score = float(sum(scores) / max(len(scores), 1)) if scores else 0.0
        return {f"eval_{prefix_name}_mover_score": avg_score}



    @torch.no_grad()
    def _collect_eval_predictions_and_refs(
        self,
        ds,
        sample_size: int = 0,
        sample_mode: str = "fixed",
        sample_seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Batched generation used by mover eval.
        """
        tokenizer = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Tokenizer/processing_class is required to compute mover metrics.")

        model = self.model
        gen_model = _unwrap_model(model)
        was_training = model.training
        model.eval()

        preds: List[str] = []
        refs: List[str] = []

        dataset_len = len(ds)
        indices = self._resolve_eval_metric_indices(
            dataset_len=dataset_len,
            sample_size=int(sample_size or 0),
            sample_mode=sample_mode,
            sample_seed=int(sample_seed or 42),
        )

        eos_ids = self._build_eval_eos_token_ids(tokenizer)

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            if isinstance(eos_ids, list) and len(eos_ids) > 0:
                pad_id = int(eos_ids[0])
            elif isinstance(eos_ids, int):
                pad_id = int(eos_ids)
            elif tokenizer.eos_token_id is not None:
                pad_id = int(tokenizer.eos_token_id)
            else:
                pad_id = 0

        max_new_tokens = int(
            getattr(self.sft_config, "eval_metric_max_new_tokens", 0)
            or getattr(self.sft_config, "eval_mover_max_new_tokens", 0)
            or getattr(self.args, "generation_max_length", None)
            or 128
        )

        answer_prefix_len_cfg = int(
            getattr(self.sft_config, "eval_metric_answer_prefix_len", None)
            if getattr(self.sft_config, "eval_metric_answer_prefix_len", None) is not None
            else getattr(self.sft_config, "eval_mover_answer_prefix_len", 2)
        )

        do_sample = bool(
            getattr(self.sft_config, "eval_metric_do_sample", None)
            if getattr(self.sft_config, "eval_metric_do_sample", None) is not None
            else getattr(self.sft_config, "eval_mover_do_sample", False)
        )
        num_beams = int(
            getattr(self.sft_config, "eval_metric_num_beams", 0)
            or getattr(self.sft_config, "eval_mover_num_beams", 1)
            or 1
        )
        repetition_penalty = float(
            getattr(self.sft_config, "eval_metric_repetition_penalty", 0.0)
            or getattr(self.sft_config, "eval_mover_repetition_penalty", 1.10)
            or 1.0
        )

        generation_batch_size = int(
            getattr(self.sft_config, "eval_metric_generation_batch_size", 0)
            or getattr(self.sft_config, "eval_mover_generation_batch_size", 4)
            or 4
        )
        generation_batch_size = max(1, generation_batch_size)

        gen_kwargs_base = dict(
            do_sample=do_sample,
            num_beams=num_beams,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_ids,
        )

        no_repeat_ngram_size = int(getattr(self.sft_config, "eval_metric_no_repeat_ngram_size", 0) or 0)
        if no_repeat_ngram_size > 0:
            gen_kwargs_base["no_repeat_ngram_size"] = no_repeat_ngram_size

        if bool(getattr(self.sft_config, "eval_metric_length_bucket", True)):
            try:
                indices = sorted(
                    indices,
                    key=lambda idx: len(ds[int(idx)]["input_ids"]) if "input_ids" in ds[int(idx)] else 0,
                )
            except Exception:
                pass

        if self.is_world_process_zero():
            print(
                f"[Mover eval generation] samples={len(indices)} "
                f"generation_batch_size={generation_batch_size} "
                f"max_new_tokens={max_new_tokens} beams={num_beams}",
                flush=True,
            )

        device = self.args.device

        old_use_cache = getattr(gen_model.config, "use_cache", None)
        if old_use_cache is not None:
            gen_model.config.use_cache = True

        batch_starts = list(range(0, len(indices), generation_batch_size))
        if bool(getattr(self.sft_config, "eval_metric_show_progress", True)) and self.is_world_process_zero():
            try:
                from tqdm.auto import tqdm
                batch_starts = tqdm(batch_starts, desc="[Mover eval generation]", leave=False)
            except Exception:
                pass

        try:
            for batch_start in batch_starts:
                batch_indices = indices[batch_start:batch_start + generation_batch_size]

                gen_seqs: List[torch.Tensor] = []
                ref_texts: List[str] = []
                answer_prefix_ids_cpu_list: List[torch.Tensor] = []
                prompt_lens_without_left_pad: List[int] = []

                for ds_idx in batch_indices:
                    raw = ds[int(ds_idx)]

                    if "input_ids" not in raw or "labels" not in raw:
                        continue

                    # Keep per-sample preprocessing on the CPU to avoid many
                    # short-lived small tensors on the GPU.
                    input_ids_1d = self._to_1d_long_tensor(raw["input_ids"])
                    labels_1d = self._to_1d_long_tensor(raw["labels"])

                    if "attention_mask" in raw and raw["attention_mask"] is not None:
                        attention_mask_1d = self._to_1d_long_tensor(raw["attention_mask"])
                    else:
                        attention_mask_1d = torch.ones_like(input_ids_1d, dtype=torch.long)

                    answer_pos = labels_1d.ne(IGNORE_INDEX)
                    answer_indices = answer_pos.nonzero(as_tuple=False).squeeze(-1)

                    if answer_indices.numel() > 0:
                        answer_start = int(answer_indices[0].item())
                    else:
                        answer_start = int(attention_mask_1d.sum().item())

                    answer_start = max(0, min(answer_start, int(input_ids_1d.numel())))

                    prompt_mask = attention_mask_1d[:answer_start].to(torch.bool)
                    prompt_ids = input_ids_1d[:answer_start][prompt_mask]
                    prompt_len = int(prompt_ids.numel())

                    ref_ids = labels_1d[labels_1d.ne(IGNORE_INDEX)]

                    answer_prefix_len = max(0, int(answer_prefix_len_cfg))
                    answer_prefix_len = min(answer_prefix_len, int(ref_ids.numel()))

                    answer_prefix_ids_cpu = ref_ids[:answer_prefix_len]

                    gen_input_ids_1d = torch.cat(
                        [
                            prompt_ids,
                            answer_prefix_ids_cpu,
                        ],
                        dim=0,
                    )

                    if gen_input_ids_1d.numel() == 0:
                        gen_input_ids_1d = torch.tensor(
                            [int(pad_id)],
                            dtype=torch.long,
                        )
                        prompt_len = int(gen_input_ids_1d.numel())

                    ref_text = tokenizer.decode(
                        ref_ids,
                        skip_special_tokens=False,
                    ).strip()

                    gen_seqs.append(gen_input_ids_1d)
                    ref_texts.append(ref_text)
                    answer_prefix_ids_cpu_list.append(answer_prefix_ids_cpu)
                    prompt_lens_without_left_pad.append(prompt_len)

                if len(gen_seqs) == 0:
                    continue

                max_len = max(int(x.numel()) for x in gen_seqs)

                batch_input_ids = torch.full(
                    (len(gen_seqs), max_len),
                    fill_value=int(pad_id),
                    dtype=torch.long,
                )
                batch_attention_mask = torch.zeros(
                    (len(gen_seqs), max_len),
                    dtype=torch.long,
                )

                prompt_lens_after_left_pad: List[int] = []

                for b, seq in enumerate(gen_seqs):
                    L = int(seq.numel())
                    pad_offset = max_len - L
                    batch_input_ids[b, pad_offset:] = seq
                    batch_attention_mask[b, pad_offset:] = 1
                    prompt_lens_after_left_pad.append(
                        int(pad_offset + prompt_lens_without_left_pad[b])
                    )

                # Move the completed padded batch to the GPU/target device once.
                batch_input_ids = batch_input_ids.to(device, non_blocking=True)
                batch_attention_mask = batch_attention_mask.to(device, non_blocking=True)

                local_gen_kwargs = dict(gen_kwargs_base)

                processors = LogitsProcessorList()
                existing_processors = local_gen_kwargs.pop("logits_processor", None)
                if existing_processors is not None:
                    if isinstance(existing_processors, LogitsProcessorList):
                        processors.extend(list(existing_processors))
                    elif isinstance(existing_processors, list):
                        processors.extend(existing_processors)
                    else:
                        processors.append(existing_processors)

                if repetition_penalty > 1.0:
                    processors.append(
                        RepetitionPenaltyLogitsProcessorExceptTokensBatch(
                            penalty=repetition_penalty,
                            prompt_ignore_lengths=prompt_lens_after_left_pad,
                            exempt_token_ids=None,
                            num_beams=num_beams,
                        )
                    )

                if len(processors) > 0:
                    local_gen_kwargs["logits_processor"] = processors

                outputs = gen_model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    **local_gen_kwargs,
                )

                input_width = int(batch_input_ids.shape[1])

                for b in range(len(gen_seqs)):
                    generated_suffix_ids = outputs[b, input_width:].detach().cpu()

                    full_prediction_ids = torch.cat(
                        [
                            answer_prefix_ids_cpu_list[b],
                            generated_suffix_ids,
                        ],
                        dim=0,
                    )

                    pred_text = tokenizer.decode(
                        full_prediction_ids,
                        skip_special_tokens=False,
                    ).strip()

                    preds.append(pred_text)
                    refs.append(ref_texts[b])

                del batch_input_ids, batch_attention_mask, outputs
                del gen_seqs, ref_texts, answer_prefix_ids_cpu_list
                del prompt_lens_without_left_pad, prompt_lens_after_left_pad

        finally:
            if old_use_cache is not None:
                gen_model.config.use_cache = old_use_cache

            if was_training:
                model.train()

        return preds, refs

    def _cleanup_eval_cuda_cache(self, tag: str = "eval_end") -> None:
        if not bool(getattr(self.sft_config, "eval_cleanup_cuda_cache_after_eval", True)):
            return

        release_encoder = bool(
            getattr(self.sft_config, "eval_mover_release_encoder_after_eval", False)
        )

        if release_encoder:
            self._release_mover_encoder()


        verbose = bool(getattr(self.sft_config, "eval_cleanup_cuda_cache_verbose", False))

        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                before_alloc = torch.cuda.memory_allocated(device)
                before_reserved = torch.cuda.memory_reserved(device)
            except Exception:
                before_alloc = before_reserved = None
        else:
            before_alloc = before_reserved = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

            if verbose and self.is_world_process_zero():
                try:
                    device = torch.cuda.current_device()
                    after_alloc = torch.cuda.memory_allocated(device)
                    after_reserved = torch.cuda.memory_reserved(device)
                    print(
                        f"[EVAL-CUDA-CLEANUP] tag={tag} "
                        f"allocated={before_alloc/1024**3:.3f}GB->{after_alloc/1024**3:.3f}GB "
                        f"reserved={before_reserved/1024**3:.3f}GB->{after_reserved/1024**3:.3f}GB "
                        f"mover_encoder_cached={self._mover_model is not None}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[WARN] eval cuda cleanup verbose print failed: {e}", flush=True)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Final mixed-task evaluate:
        - loss is computed for each validation dataset.
        - cnn_dm uses summary-level MoverScore-like OT.
        - kptime uses field-aware MoverScore-like OT:
            categories and keywords are extracted and scored separately,
            then averaged into eval_kptime_mover_final_score.
        - eval_combo_mover_score is the primary best-model metric:
            0.5 * eval_cnn_dm_mover_score
          + 0.5 * eval_kptime_mover_final_score.
        """
        if metric_key_prefix != "eval":
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        run_eval_loss = self._should_run_eval_loss()
        run_mover = self._should_run_eval_mover()

        if eval_dataset is None:
            raise ValueError(
                "eval_dataset is None. Please pass eval_dataset or set trainer.eval_dataset."
            )

        metrics: Dict[str, float] = {}

        def _as_eval_dict(ds_obj):
            if isinstance(ds_obj, dict):
                return dict(ds_obj)
            return {"valid": ds_obj}

        eval_datasets = _as_eval_dict(eval_dataset)

        # -------------------------
        # 1) Loss per dataset
        # -------------------------
        if run_eval_loss:
            for name, ds in eval_datasets.items():
                name = str(name)
                eval_dataloader = self.get_eval_dataloader(ds)

                output = self.evaluation_loop(
                    eval_dataloader,
                    description=f"Evaluation {name}",
                    prediction_loss_only=True,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"eval_{name}",
                )

                loss_key = f"eval_{name}_loss"
                if loss_key in output.metrics:
                    metrics[loss_key] = float(output.metrics[loss_key])

            loss_values = []
            for name in ("cnn_dm", "kptime"):
                key = f"eval_{name}_loss"
                if key in metrics:
                    loss_values.append(float(metrics[key]))

            if len(loss_values) == 2:
                combo_loss = 0.5 * loss_values[0] + 0.5 * loss_values[1]
                metrics["eval_combo_loss"] = float(combo_loss)
                metrics["eval_loss"] = float(combo_loss)
            elif "valid" in eval_datasets and "eval_valid_loss" in metrics:
                metrics["eval_loss"] = float(metrics["eval_valid_loss"])
            elif loss_values:
                combo_loss = float(sum(loss_values) / len(loss_values))
                metrics["eval_combo_loss"] = combo_loss
                metrics["eval_loss"] = combo_loss

        # -------------------------
        # 2) Mover metrics per dataset
        # -------------------------
        if run_mover:
            metric_sample_size = int(
                getattr(self.sft_config, "eval_metric_sample_size", None)
                if getattr(self.sft_config, "eval_metric_sample_size", None) is not None
                else getattr(self.sft_config, "eval_mover_sample_size", 0)
            )
            metric_sample_mode = str(
                getattr(self.sft_config, "eval_metric_sample_mode", None)
                if getattr(self.sft_config, "eval_metric_sample_mode", None) is not None
                else getattr(self.sft_config, "eval_mover_sample_mode", "fixed")
            )
            metric_sample_seed = int(
                getattr(self.sft_config, "eval_metric_sample_seed", None)
                if getattr(self.sft_config, "eval_metric_sample_seed", None) is not None
                else getattr(self.sft_config, "eval_mover_sample_seed", 42)
            )

            for name, ds in eval_datasets.items():
                name = str(name)
                preds, refs = self._collect_eval_predictions_and_refs(
                    ds,
                    sample_size=metric_sample_size,
                    sample_mode=metric_sample_mode,
                    sample_seed=metric_sample_seed,
                )

                normalized_name = name.lower().replace("-", "_")
                if normalized_name == "kptime":
                    metric_dict = self._eval_mover_like_from_texts(
                        preds,
                        refs,
                        prefix_name=name,
                    )
                else:
                    # cnn_dm and non-kptime datasets are treated as summary/highlight tasks.
                    metric_dict = self._eval_mover_summary_from_texts(
                        preds,
                        refs,
                        prefix_name=name,
                    )

                metrics.update(metric_dict)

            cnn_mover = metrics.get("eval_cnn_dm_mover_score", None)
            kptime_mover = metrics.get("eval_kptime_mover_final_score", None)

            if cnn_mover is not None and kptime_mover is not None:
                metrics["eval_combo_mover_score"] = float(
                    0.5 * float(cnn_mover) + 0.5 * float(kptime_mover)
                )
            elif "eval_valid_mover_score" in metrics:
                metrics["eval_combo_mover_score"] = float(metrics["eval_valid_mover_score"])

        # -------------------------
        # 3) Single log / callback dispatch
        # -------------------------
        if metrics:
            self.log(metrics)
            self.control = self.callback_handler.on_evaluate(
                self.args,
                self.state,
                self.control,
                metrics,
            )

        self._cleanup_eval_cuda_cache(tag="evaluate_end")

        return metrics
