"""Importable vLLM V1 logits processor used by evaluation workers."""

from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence, Tuple

def _sanitize_positive_int_env(name: str, default: str = "1") -> None:
    value = os.environ.get(name)
    if value is None:
        return
    value_s = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", value_s):
        os.environ[name] = default

for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    _sanitize_positive_int_env(_name)

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, RequestLogitsProcessor


class TrainingStyleVLLMLogitsProcessor:
    def __init__(
        self,
        *,
        prompt_ignore_len: int,
        answer_prefix_ids: Sequence[int],
        penalty: float,
        separator_token_ids: Sequence[int],
        blocked_after_separator_token_ids: Sequence[int],
    ):
        penalty = float(penalty)
        if penalty <= 0:
            raise ValueError("penalty must be > 0")
        self.prompt_ignore_len = max(0, int(prompt_ignore_len))
        self.answer_prefix_ids = [int(x) for x in answer_prefix_ids]
        self.penalty = penalty
        self.separator_token_ids = set(int(x) for x in separator_token_ids)
        self.blocked_after_separator_token_ids = set(int(x) for x in blocked_after_separator_token_ids)

    @staticmethod
    def _to_list(x) -> List[int]:
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().view(-1).tolist()]
        return [int(v) for v in list(x)]

    def _build_answer_ids(self, *args) -> Tuple[List[int], torch.Tensor]:
        if len(args) == 2:
            output_token_ids, logits = args
            answer_ids = list(self.answer_prefix_ids) + self._to_list(output_token_ids)
            return answer_ids, logits

        if len(args) == 3:
            prompt_token_ids, output_token_ids, logits = args
            prompt_ids = self._to_list(prompt_token_ids)
            output_ids = self._to_list(output_token_ids)
            answer_prefix_from_prompt = prompt_ids[self.prompt_ignore_len:]
            answer_ids = list(answer_prefix_from_prompt) + output_ids
            return answer_ids, logits

        raise TypeError(
            "Unsupported vLLM logits processor signature. Expected "
            "(output_token_ids, logits) or (prompt_token_ids, output_token_ids, logits)."
        )

    def __call__(self, *args):
        answer_ids, logits = self._build_answer_ids(*args)

        if self.penalty != 1.0 and answer_ids:
            seen = sorted(set(int(x) for x in answer_ids))
            if self.separator_token_ids:
                seen = [x for x in seen if x not in self.separator_token_ids]

            if seen:
                idx = torch.tensor(seen, dtype=torch.long, device=logits.device)
                valid = (idx >= 0) & (idx < logits.shape[-1])
                idx = idx[valid]

                if idx.numel() > 0:
                    token_scores = logits[idx]
                    logits[idx] = torch.where(
                        token_scores < 0,
                        token_scores * self.penalty,
                        token_scores / self.penalty,
                    )

        if answer_ids:
            last_tok = int(answer_ids[-1])
            if last_tok in self.separator_token_ids:
                block_ids = sorted(self.separator_token_ids | self.blocked_after_separator_token_ids)
                if block_ids:
                    idx = torch.tensor(block_ids, dtype=torch.long, device=logits.device)
                    valid = (idx >= 0) & (idx < logits.shape[-1])
                    idx = idx[valid]

                    if idx.numel() > 0:
                        logits[idx] = float("-inf")

        return logits


class WrappedTrainingStyleVLLMLogitsProcessor(AdapterLogitsProcessor):
    EXTRA_KEY = "training_style_logits_processor"

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = getattr(params, "extra_args", None) or {}
        cfg = extra_args.get(cls.EXTRA_KEY)

        if cfg is None:
            return

        required = [
            "prompt_ignore_len",
            "answer_prefix_ids",
            "penalty",
            "separator_token_ids",
            "blocked_after_separator_token_ids",
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"Missing {cls.EXTRA_KEY} keys: {missing}")

        penalty = float(cfg["penalty"])
        if penalty <= 0:
            raise ValueError("penalty must be > 0")

        int(cfg["prompt_ignore_len"])

        for list_key in [
            "answer_prefix_ids",
            "separator_token_ids",
            "blocked_after_separator_token_ids",
        ]:
            if not isinstance(cfg[list_key], (list, tuple)):
                raise ValueError(f"{list_key} must be a list/tuple of token ids")
            for x in cfg[list_key]:
                int(x)

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        extra_args = getattr(params, "extra_args", None) or {}
        cfg = extra_args.get(self.EXTRA_KEY)

        if cfg is None:
            return None

        self.validate_params(params)

        return TrainingStyleVLLMLogitsProcessor(
            prompt_ignore_len=int(cfg["prompt_ignore_len"]),
            answer_prefix_ids=[int(x) for x in cfg["answer_prefix_ids"]],
            penalty=float(cfg["penalty"]),
            separator_token_ids=[int(x) for x in cfg["separator_token_ids"]],
            blocked_after_separator_token_ids=[
                int(x) for x in cfg["blocked_after_separator_token_ids"]
            ],
        )

