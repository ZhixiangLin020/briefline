# ============================================================
# Pure vLLM mixed full valid/test generation + MoverScore-like OT eval
#
# Dataset structure:
# {
#   "validation": {"cnn_dm": validation_cnn_dm, "kptime": validation_kptime},
#   "test": {"cnn_dm": test_cnn_dm, "kptime": test_kptime},
# }
#
# vLLM is used ONLY for decoder-only LLM generation.
# RoBERTa + MoverScore-like OT scoring remains ordinary PyTorch/Transformers.
#
# Main outputs:
# - metrics_long.csv
# - summary_vs_base.csv
# - runtime.csv
# - best_finetuned_by_full_valid_combo.json
# - predictions_minimal.csv
# - model_io/*.jsonl
# ============================================================

from __future__ import annotations

import gc
import json
import math
import os
import re

# Sanitize OpenMP-related environment variables before importing numpy/torch.
def _sanitize_positive_int_env(name: str, default: str = "1") -> None:
    value = os.environ.get(name)
    if value is None:
        return
    value_s = str(value).strip()
    if not re.fullmatch(r"[1-9][0-9]*", value_s):
        os.environ[name] = default

for _env_name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    _sanitize_positive_int_env(_env_name)

import subprocess
import sys
import time
import shutil

# ============================================================
# vLLM V1 custom logits processor mode
# ============================================================
# This script uses the current vLLM V1 custom logits processor interface:
#   - register the custom processor class at LLM(...) initialization time
#   - pass per-request configuration via SamplingParams(extra_args=...)
# Do NOT force VLLM_USE_V1=0 here. The old V0 per-request interface
# SamplingParams(logits_processors=[...]) is intentionally not used.
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the companion logits-processor module importable by vLLM worker processes.
_SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
_pythonpath_parts = [p for p in _existing_pythonpath.split(os.pathsep) if p]
if str(_SCRIPT_DIR) not in _pythonpath_parts:
    os.environ["PYTHONPATH"] = str(_SCRIPT_DIR) + (os.pathsep + _existing_pythonpath if _existing_pythonpath else "")

from training_style_logits_processor_vllm import (
    WrappedTrainingStyleVLLMLogitsProcessor as ImportableWrappedTrainingStyleVLLMLogitsProcessor,
)
IMPORTABLE_TRAINING_STYLE_LOGITS_PROCESSOR_FQCN = (
    "training_style_logits_processor_vllm:WrappedTrainingStyleVLLMLogitsProcessor"
)
TRAINING_STYLE_EXTRA_KEY = ImportableWrappedTrainingStyleVLLMLogitsProcessor.EXTRA_KEY

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datasets import load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    import ot
except Exception:
    print("[INSTALL] POT", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "POT"])
    import ot

try:
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.v1.sample.logits_processor import (
        AdapterLogitsProcessor,
        RequestLogitsProcessor,
    )
except Exception as e:
    raise ImportError(
        "This script requires a vLLM version with the V1 custom logits processor API. "
        "Run `python scripts/install_dependencies.py` to install the unified environment, "
        "then rerun."
    ) from e


IGNORE_INDEX = -100


# ============================================================
# 1. Basic helpers
# ============================================================

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def cleanup_vllm_engine(llm: Optional[Any]) -> None:
    """Best-effort cleanup between model_artifact_paths."""
    try:
        if llm is not None:
            del llm
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception:
        pass

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    cleanup_cuda()


def maybe_tqdm(iterable, desc: str = "", total: Optional[int] = None, use_tqdm: bool = True, leave: bool = False):
    if not use_tqdm:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True, leave=leave)
    except Exception:
        return iterable


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60.0:.2f}h"


def parse_torch_dtype(dtype: Optional[Any], default: torch.dtype) -> torch.dtype:
    if dtype is None:
        return default
    if isinstance(dtype, torch.dtype):
        return dtype
    s = str(dtype).lower().strip()
    if s in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def parse_vllm_dtype(dtype: Optional[Any]) -> str:
    if dtype is None:
        return "auto"
    if isinstance(dtype, torch.dtype):
        if dtype is torch.float16:
            return "float16"
        if dtype is torch.bfloat16:
            return "bfloat16"
        if dtype is torch.float32:
            return "float32"
    s = str(dtype).lower().strip().replace("torch.", "")
    if s in {"auto", "float16", "bfloat16", "float32", "half", "fp16", "bf16"}:
        if s == "fp16" or s == "half":
            return "float16"
        if s == "bf16":
            return "bfloat16"
        return s
    raise ValueError(f"Unsupported vLLM dtype: {dtype}")


def to_1d_long_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    if t.ndim != 1:
        t = t.view(-1)
    return t.long()


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unnamed"


def is_peft_adapter_dir(path: str) -> bool:
    return os.path.exists(os.path.join(str(path), "adapter_config.json"))


def read_peft_adapter_type(path: str) -> str:
    cfg_path = os.path.join(str(path), "adapter_config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return str(cfg.get("peft_type", "PEFT")).upper()
    except Exception:
        return "PEFT"


def infer_adapter_saved_vocab_size(adapter_path: str) -> Optional[int]:
    """Infer saved embedding/lm_head vocab size from a PEFT adapter checkpoint.

    Some PEFT checkpoints save resized `embed_tokens` / `lm_head` tensors.
    If the current base model has a different embedding size, PEFT loading fails.
    This function reads tensor shapes without fully materializing large tensors
    when safetensors slicing metadata is available.
    """
    adapter_path = str(adapter_path)
    if not os.path.isdir(adapter_path):
        return None

    safetensor_files = [
        os.path.join(adapter_path, fn)
        for fn in os.listdir(adapter_path)
        if fn.endswith(".safetensors")
    ]

    for fp in safetensor_files:
        try:
            from safetensors.torch import safe_open
        except Exception:
            break

        try:
            with safe_open(fp, framework="pt", device="cpu") as f:
                for k in f.keys():
                    lk = str(k).lower()
                    if ("embed_tokens.weight" in lk) or ("lm_head.weight" in lk):
                        try:
                            shape = tuple(f.get_slice(k).get_shape())
                        except Exception:
                            shape = tuple(f.get_tensor(k).shape)
                        if len(shape) >= 2:
                            return int(shape[0])
        except Exception:
            continue

    # Fallback for very old adapters saved as .bin. This may load tensors into RAM.
    bin_files = [
        os.path.join(adapter_path, fn)
        for fn in os.listdir(adapter_path)
        if fn.endswith(".bin") and ("adapter" in fn or "pytorch_model" in fn)
    ]
    for fp in bin_files:
        try:
            state = torch.load(fp, map_location="cpu")
            for k, v in state.items():
                lk = str(k).lower()
                if hasattr(v, "shape") and (("embed_tokens.weight" in lk) or ("lm_head.weight" in lk)):
                    shape = tuple(v.shape)
                    if len(shape) >= 2:
                        return int(shape[0])
        except Exception:
            continue

    return None


def collect_adapter_saved_vocab_sizes(model_artifact_paths: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for alias, path in model_artifact_paths.items():
        if is_peft_adapter_dir(str(path)):
            vocab_size = infer_adapter_saved_vocab_size(str(path))
            if vocab_size is not None:
                out[str(alias)] = int(vocab_size)
    return out


def get_model_embedding_vocab_size(model) -> int:
    return int(model.get_input_embeddings().weight.shape[0])


def maybe_resize_model_embeddings_to_vocab(
    model,
    *,
    target_vocab_size: int,
    reason: str = "",
):
    """Resize input/output embeddings if the base model vocab differs from target."""
    target_vocab_size = int(target_vocab_size)
    if target_vocab_size < 1000:
        raise ValueError(
            f"Refusing to resize model to suspicious vocab size={target_vocab_size}. "
            "This usually means the tokenizer was loaded incorrectly."
        )

    old_vocab_size = get_model_embedding_vocab_size(model)
    print("  base_embedding_vocab_size:", old_vocab_size, flush=True)
    print("  target_vocab_size:", target_vocab_size, flush=True)
    if reason:
        print("  resize_reason:", reason, flush=True)

    if old_vocab_size != target_vocab_size:
        print(f"[RESIZE TOKEN EMBEDDINGS] {old_vocab_size} -> {target_vocab_size}", flush=True)
        try:
            model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(target_vocab_size)
        model.config.vocab_size = target_vocab_size

    return model


def prepare_resized_base_model_if_needed(
    *,
    original_base_model_path: str,
    tokenizer,
    target_vocab_size: int,
    output_path: str,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    dtype: Optional[Any] = torch.float16,
    device: str = "cpu",
    max_shard_size: str = "2GB",
) -> str:
    """Create a temporary base model whose embedding vocab matches target_vocab_size.

    If the original base already matches, returns original_base_model_path and does not save.
    Otherwise saves a resized full model to output_path. Prefer placing output_path under
    /dev/shm so this temporary model does not consume persistent disk.
    """
    target_vocab_size = int(target_vocab_size)
    if target_vocab_size < 1000:
        raise ValueError(
            f"Suspicious target_vocab_size={target_vocab_size}. "
            "Do not use a broken checkpoint tokenizer; use the original base tokenizer."
        )

    original_base_model_path = str(original_base_model_path)
    output_path = str(output_path)

    dtype_t = parse_torch_dtype(dtype, default=torch.float16)
    device = str(device or "cpu").lower()
    if device == "cpu":
        device_map = {"": "cpu"}
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    print("[CHECK BASE VOCAB FOR RESIZE]", flush=True)
    print("  original_base_model_path:", original_base_model_path, flush=True)
    print("  tokenizer_len:", len(tokenizer), flush=True)
    print("  target_vocab_size:", target_vocab_size, flush=True)

    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            original_base_model_path,
            torch_dtype=dtype_t,
            device_map=device_map,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        old_vocab_size = get_model_embedding_vocab_size(model)
        print("  original_base_embedding_vocab_size:", old_vocab_size, flush=True)

        if old_vocab_size == target_vocab_size:
            print("[BASE VOCAB OK] no resized base needed", flush=True)
            return original_base_model_path

        if os.path.abspath(output_path) == os.path.abspath(original_base_model_path):
            raise ValueError("output_path for resized base must not equal original_base_model_path.")

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        model = maybe_resize_model_embeddings_to_vocab(
            model,
            target_vocab_size=target_vocab_size,
            reason="base/tokenizer or base/adapter vocab mismatch",
        )

        print("[SAVE TEMP RESIZED BASE]", output_path, flush=True)
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=str(max_shard_size),
        )
        tokenizer.save_pretrained(output_path)
        return output_path

    finally:
        try:
            del model
        except Exception:
            pass
        cleanup_cuda()


def choose_target_vocab_size_for_eval(
    *,
    tokenizer,
    adapter_vocab_sizes: Dict[str, int],
    strict_tokenizer_adapter_vocab_match: bool = True,
) -> int:
    tokenizer_vocab_size = int(len(tokenizer))
    if tokenizer_vocab_size < 1000:
        raise ValueError(
            f"Loaded tokenizer length is suspiciously small: {tokenizer_vocab_size}. "
            "Your tokenizer_path is likely wrong. Use the original base tokenizer path, not a broken adapter tokenizer."
        )

    unique_adapter_sizes = sorted(set(int(v) for v in adapter_vocab_sizes.values()))
    if unique_adapter_sizes:
        print("[ADAPTER SAVED VOCAB SIZES]", adapter_vocab_sizes, flush=True)
        if len(unique_adapter_sizes) != 1:
            raise ValueError(
                f"Adapters have different saved vocab sizes: {adapter_vocab_sizes}. "
                "Evaluate separately or use matching base/tokenizer for each group."
            )
        adapter_vocab_size = int(unique_adapter_sizes[0])
        if adapter_vocab_size != tokenizer_vocab_size:
            msg = (
                f"Adapter saved vocab size ({adapter_vocab_size}) != tokenizer length ({tokenizer_vocab_size}). "
                "For strict evaluation, tokenizer/model/adapter vocab must match."
            )
            if strict_tokenizer_adapter_vocab_match:
                raise ValueError(msg)
            print("[WARN] " + msg, flush=True)
        return adapter_vocab_size

    return tokenizer_vocab_size


def merge_peft_adapter_to_temp_full_model(
    *,
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    tokenizer,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    merge_dtype: Optional[Any] = torch.float16,
    merge_device: str = "cpu",
    max_shard_size: str = "2GB",
) -> str:
    """Merge a PEFT adapter into the base model and save a temporary full model for vLLM.

    This is intended for adapter types that vLLM cannot reliably load directly, such as AdaLoRA.
    The caller can delete output_path after vLLM has finished generation for this model.
    """
    try:
        from peft import PeftModel
    except Exception as e:
        raise ImportError(
            "PEFT is required to auto-merge adapter checkpoints. Run "
            "`python scripts/install_dependencies.py` to install the unified environment."
        ) from e

    adapter_type = read_peft_adapter_type(adapter_path)
    output_path = str(output_path)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    merge_dtype_t = parse_torch_dtype(merge_dtype, default=torch.float16)
    merge_device = str(merge_device or "cpu").lower()
    if merge_device == "cpu":
        device_map = {"": "cpu"}
    elif merge_device == "auto":
        device_map = "auto"
    else:
        device_map = {"": merge_device}

    print("[TEMP MERGE PEFT ADAPTER]", flush=True)
    print("  adapter_type:", adapter_type, flush=True)
    print("  base_model_path:", base_model_path, flush=True)
    print("  adapter_path:", adapter_path, flush=True)
    print("  output_path:", output_path, flush=True)
    print("  merge_dtype:", merge_dtype_t, flush=True)
    print("  merge_device:", merge_device, flush=True)

    base_model = None
    merged_model = None
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            str(base_model_path),
            torch_dtype=merge_dtype_t,
            device_map=device_map,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        base_model.eval()

        adapter_saved_vocab_size = infer_adapter_saved_vocab_size(str(adapter_path))
        tokenizer_vocab_size = int(len(tokenizer))
        print("  tokenizer_vocab_size:", tokenizer_vocab_size, flush=True)
        print("  adapter_saved_vocab_size:", adapter_saved_vocab_size, flush=True)
        if adapter_saved_vocab_size is not None:
            if tokenizer_vocab_size != int(adapter_saved_vocab_size):
                raise ValueError(
                    f"Tokenizer length ({tokenizer_vocab_size}) != adapter saved vocab size "
                    f"({adapter_saved_vocab_size}). Use the tokenizer that was used for training."
                )
            base_model = maybe_resize_model_embeddings_to_vocab(
                base_model,
                target_vocab_size=int(adapter_saved_vocab_size),
                reason="adapter checkpoint contains resized embedding/lm_head tensors",
            )

        merged_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_path),
            local_files_only=local_files_only,
        )
        merged_model.eval()
        merged_model = merged_model.merge_and_unload()

        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=str(max_shard_size),
        )
        tokenizer.save_pretrained(output_path)

    finally:
        try:
            del merged_model
        except Exception:
            pass
        try:
            del base_model
        except Exception:
            pass
        cleanup_cuda()

    print("[TEMP MERGE DONE]", output_path, flush=True)
    return output_path


# ============================================================
# 2. Token ids helpers
# ============================================================

def get_single_token_id(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def build_eval_eos_token_ids(tokenizer) -> List[int]:
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    for tok in ["<|end|>", "<|endoftext|>"]:
        tid = get_single_token_id(tokenizer, tok)
        if tid is not None:
            eos_ids.append(int(tid))
    return sorted(set(eos_ids))


def infer_separator_token_ids_from_tokenizer(tokenizer, separator_texts: Sequence[str] = (",",)) -> List[int]:
    out = set()
    candidates = []
    for sep in separator_texts:
        sep = str(sep)
        candidates.extend([sep, f" {sep}", f"{sep} "])
    for s in candidates:
        try:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if len(ids) == 1:
                out.add(int(ids[0]))
        except Exception:
            pass
    return sorted(out)


# ============================================================
# 3. vLLM custom logits processor
# ============================================================

class TrainingStyleVLLMLogitsProcessor:
    """
    Per-request vLLM logits processor that mirrors the HF training/eval decoding style:
      1. answer-region-only repetition penalty
      2. separator tokens are exempt from repetition penalty
      3. if last answer token is a separator, block another separator and EOS

    It supports both common vLLM callable signatures:
      - processor(output_token_ids, logits)
      - processor(prompt_token_ids, output_token_ids, logits)

    If vLLM passes only output_token_ids, answer_prefix_ids from construction is used.
    If vLLM passes prompt_token_ids, answer_prefix is recovered as prompt_token_ids[prompt_ignore_len:].
    """

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
    """vLLM V1 wrapper around the request-level training-style processor.

    vLLM V1 loads custom logits processors at engine initialization. Per-request
    options are passed through SamplingParams.extra_args under EXTRA_KEY.

    This preserves the original behavior:
      - repetition penalty is applied only to the answer region
      - comma/separator tokens are exempt from repetition penalty
      - if the last answer token is a separator, another separator and EOS are blocked
    """

    EXTRA_KEY = "training_style_logits_processor"

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = getattr(params, "extra_args", None) or {}
        cfg = extra_args.get(cls.EXTRA_KEY)
        if cfg is None:
            # Processor disabled for this request.
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

        for list_key in [
            "answer_prefix_ids",
            "separator_token_ids",
            "blocked_after_separator_token_ids",
        ]:
            if not isinstance(cfg[list_key], (list, tuple)):
                raise ValueError(f"{list_key} must be a list/tuple of token ids")
            for x in cfg[list_key]:
                int(x)

        int(cfg["prompt_ignore_len"])

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


# ============================================================
# 4. Dataset resolving
# ============================================================

def _has_keys(x) -> bool:
    return hasattr(x, "keys")


def _get_by_candidate_names(obj, candidate_names: Sequence[str], object_name: str):
    if not _has_keys(obj):
        raise TypeError(f"{object_name} is not dict-like. type={type(obj)}")
    keys = list(obj.keys())
    for name in candidate_names:
        if name in obj:
            return name, obj[name]
    raise KeyError(
        f"Cannot find {object_name}. candidate_names={list(candidate_names)}, available={keys}"
    )


def resolve_nested_eval_datasets(
    ds_out,
    *,
    validation_names: Sequence[str] = ("validation", "valid", "eval"),
    test_names: Sequence[str] = ("test",),
    task_names: Sequence[str] = ("cnn_dm", "kptime"),
) -> Dict[str, Dict[str, Any]]:
    validation_key, validation_obj = _get_by_candidate_names(ds_out, validation_names, "validation split")
    test_key, test_obj = _get_by_candidate_names(ds_out, test_names, "test split")

    out: Dict[str, Dict[str, Any]] = {}
    for split_alias, real_split_key, split_obj in [
        ("validation", validation_key, validation_obj),
        ("test", test_key, test_obj),
    ]:
        if not _has_keys(split_obj):
            raise TypeError(
                f"Split {real_split_key!r} must be dict-like and contain tasks {list(task_names)}. "
                f"Got type={type(split_obj)}"
            )
        out[split_alias] = {}
        available_tasks = list(split_obj.keys())
        for task in task_names:
            if task not in split_obj:
                raise KeyError(
                    f"Cannot find task={task!r} under split={real_split_key!r}. "
                    f"Available tasks={available_tasks}"
                )
            out[split_alias][task] = split_obj[task]
    return out




def _resolve_split_from_datasetdict(ds, split_candidates: Sequence[str], dataset_name: str):
    """Return one split from a DatasetDict-like object using candidate split names."""
    return _get_by_candidate_names(ds, split_candidates, f"{dataset_name} split")


def _resolve_nested_saved_dir(
    root: str,
    *,
    validation_names: Sequence[str],
    test_names: Sequence[str],
    task_names: Sequence[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load custom nested dirs such as root/validation/cnn_dm and root/test/kptime.

    This supports datasets saved by save_cnn_dm_kptime_result_to_disk(...), which are
    not themselves a Hugging Face DatasetDict at the root.
    """
    root_p = Path(root)
    if not root_p.exists() or not root_p.is_dir():
        return None

    def _find_split_dir(candidates: Sequence[str]) -> Optional[Path]:
        for name in candidates:
            p = root_p / str(name)
            if p.exists() and p.is_dir():
                return p
        return None

    validation_dir = _find_split_dir(validation_names)
    test_dir = _find_split_dir(test_names)
    if validation_dir is None or test_dir is None:
        return None

    for split_dir in [validation_dir, test_dir]:
        for task in task_names:
            if not (split_dir / str(task)).exists():
                return None

    out: Dict[str, Dict[str, Any]] = {"validation": {}, "test": {}}
    for split_alias, split_dir in [("validation", validation_dir), ("test", test_dir)]:
        for task in task_names:
            out[split_alias][task] = load_from_disk(str(split_dir / str(task)))
    return out


def load_eval_datasets_flexible(
    *,
    dataset_path: Optional[str] = None,
    dataset_obj: Optional[Any] = None,
    cnn_dm_dataset_path: Optional[str] = None,
    kptime_dataset_path: Optional[str] = None,
    validation_names: Sequence[str] = ("validation", "valid", "eval"),
    test_names: Sequence[str] = ("test",),
    task_names: Sequence[str] = ("cnn_dm", "kptime"),
) -> Dict[str, Dict[str, Any]]:
    """Load evaluation datasets in any of these formats:

    1) dataset_obj: an in-memory nested dict, e.g.
       {"validation": {"cnn_dm": ..., "kptime": ...}, "test": {...}}

    2) cnn_dm_dataset_path + kptime_dataset_path: two separate trainer DatasetDict dirs,
       each containing validation/test splits.

    3) dataset_path: a custom nested directory saved as
       dataset_path/validation/cnn_dm, dataset_path/validation/kptime,
       dataset_path/test/cnn_dm, dataset_path/test/kptime.

    4) dataset_path: a normal Hugging Face DatasetDict path whose loaded object is nested.
    """
    if dataset_obj is not None:
        print("[DATASET LOADER] using in-memory dataset_obj", flush=True)
        return resolve_nested_eval_datasets(
            dataset_obj,
            validation_names=validation_names,
            test_names=test_names,
            task_names=task_names,
        )

    if cnn_dm_dataset_path is not None or kptime_dataset_path is not None:
        if cnn_dm_dataset_path is None or kptime_dataset_path is None:
            raise ValueError("cnn_dm_dataset_path and kptime_dataset_path must be provided together.")
        print("[DATASET LOADER] using two separate DatasetDict paths", flush=True)
        print("  cnn_dm_dataset_path:", cnn_dm_dataset_path, flush=True)
        print("  kptime_dataset_path:", kptime_dataset_path, flush=True)
        ds_cnn_dm = load_from_disk(str(cnn_dm_dataset_path))
        ds_kptime = load_from_disk(str(kptime_dataset_path))

        _, val_cnn = _resolve_split_from_datasetdict(ds_cnn_dm, validation_names, "cnn_dm validation")
        _, test_cnn = _resolve_split_from_datasetdict(ds_cnn_dm, test_names, "cnn_dm test")
        _, val_kptime = _resolve_split_from_datasetdict(ds_kptime, validation_names, "kptime validation")
        _, test_kptime = _resolve_split_from_datasetdict(ds_kptime, test_names, "kptime test")

        return {
            "validation": {"cnn_dm": val_cnn, "kptime": val_kptime},
            "test": {"cnn_dm": test_cnn, "kptime": test_kptime},
        }

    if dataset_path is None:
        raise ValueError(
            "Provide one of: dataset_obj, dataset_path, or both cnn_dm_dataset_path and kptime_dataset_path."
        )

    nested = _resolve_nested_saved_dir(
        str(dataset_path),
        validation_names=validation_names,
        test_names=test_names,
        task_names=task_names,
    )
    if nested is not None:
        print("[DATASET LOADER] using custom nested saved directory", flush=True)
        print("  dataset_path:", dataset_path, flush=True)
        return nested

    print("[DATASET LOADER] using load_from_disk(dataset_path) + resolve_nested_eval_datasets", flush=True)
    ds_out = load_from_disk(str(dataset_path))
    return resolve_nested_eval_datasets(
        ds_out,
        validation_names=validation_names,
        test_names=test_names,
        task_names=task_names,
    )

# ============================================================
# 5. Decode prompt/reference from trainer-ready row
# ============================================================

def decode_prompt_and_reference_from_dataset_row(
    raw,
    tokenizer,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[str, str, torch.Tensor, torch.Tensor]:
    if "input_ids" not in raw or "labels" not in raw:
        raise KeyError(
            "Dataset row must contain `input_ids` and `labels`. "
            "If your dataset is raw text, adapt this function."
        )

    input_ids = to_1d_long_tensor(raw["input_ids"])
    labels = to_1d_long_tensor(raw["labels"])

    if "attention_mask" in raw and raw["attention_mask"] is not None:
        attn = to_1d_long_tensor(raw["attention_mask"])
        if attn.numel() == input_ids.numel():
            keep = attn.ne(0)
            input_ids = input_ids[keep]
            labels = labels[keep]

    answer_mask = labels.ne(ignore_index)
    answer_indices = answer_mask.nonzero(as_tuple=False).flatten()

    if answer_indices.numel() == 0:
        prompt_ids = input_ids
        ref_ids = input_ids[:0]
    else:
        first_answer_pos = int(answer_indices[0].item())
        prompt_ids = input_ids[:first_answer_pos]
        ref_ids = labels[answer_mask]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False).strip()
    reference_text = tokenizer.decode(ref_ids, skip_special_tokens=False).strip()
    return prompt_text, reference_text, prompt_ids.long(), ref_ids.long()


# ============================================================
# 6. Text cleaning / kptime parsing
# ============================================================

SPECIAL_TOKENS = [
    "<|end|>",
    "<|endoftext|>",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
]


def remove_special_tokens(text: str) -> str:
    text = "" if text is None else str(text)
    for tok in SPECIAL_TOKENS:
        text = text.replace(tok, " ")
    text = re.sub(r"<\|.*?\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_summary_for_metric(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("[SUMMARY]", " ").replace("[/SUMMARY]", " ")
    text = remove_special_tokens(text)
    text = re.sub(
        r"^\s*(summary|summaries|\u6458\u8981)\s*[:：]\s*",
        "",
        text,
        flags=re.I,
    )
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_payload_text(text: str) -> str:
    text = remove_special_tokens(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_kptime_field_payloads(text: str) -> Dict[str, Any]:
    text_clean = remove_special_tokens(text)
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
        "categories_payload": clean_payload_text(categories_payload),
        "keywords_payload": clean_payload_text(keywords_payload),
        "has_categories": bool(has_categories),
        "has_keywords": bool(has_keywords),
        "line_order_valid": bool(line_order_valid),
    }


# ============================================================
# 7. vLLM generation + JSONL model I/O saving
# ============================================================

def build_sampling_params_for_request(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_ids: Sequence[int],
    logits_processor_config: Dict[str, Any],
) -> SamplingParams:
    """Build vLLM SamplingParams with per-request custom logits config.

    Current vLLM V1 does not accept SamplingParams(logits_processors=[...]).
    The custom processor is registered once at LLM(...) initialization, and this
    function enables/configures it for each request via extra_args.
    """
    kwargs = dict(
        n=1,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        stop_token_ids=list(eos_token_ids) if eos_token_ids else None,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        extra_args={
            TRAINING_STYLE_EXTRA_KEY: logits_processor_config
        },
    )

    try:
        return SamplingParams(**kwargs, detokenize=False)
    except TypeError as e1:
        # Some vLLM versions do not expose detokenize. Retry without it.
        try:
            return SamplingParams(**kwargs)
        except TypeError as e2:
            if "extra_args" in str(e1) or "extra_args" in str(e2):
                raise RuntimeError(
                    "Current vLLM SamplingParams does not accept `extra_args`. "
                    "This script requires vLLM V1 custom logits processor support."
                ) from e2
            raise


def make_vllm_llm(
    *,
    model_path: str,
    tokenizer_path: str,
    trust_remote_code: bool,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: Optional[int],
    enforce_eager: bool,
    disable_log_stats: bool,
    seed: int,
    extra_llm_kwargs: Optional[Dict[str, Any]] = None,
) -> LLM:
    kwargs: Dict[str, Any] = dict(
        model=str(model_path),
        tokenizer=str(tokenizer_path),
        trust_remote_code=bool(trust_remote_code),
        dtype=dtype,
        tensor_parallel_size=int(tensor_parallel_size),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=bool(enforce_eager),
        disable_log_stats=bool(disable_log_stats),
        seed=int(seed),
    )
    if max_model_len is not None:
        kwargs["max_model_len"] = int(max_model_len)

    # vLLM V1: custom logits processors are loaded once when the engine starts.
    # Per-request settings are passed later through SamplingParams.extra_args.
    default_processors = [IMPORTABLE_TRAINING_STYLE_LOGITS_PROCESSOR_FQCN]
    if extra_llm_kwargs:
        extra_llm_kwargs = dict(extra_llm_kwargs)
        provided_processors = extra_llm_kwargs.pop("logits_processors", None)
        if provided_processors is None:
            kwargs["logits_processors"] = default_processors
        else:
            if not isinstance(provided_processors, (list, tuple)):
                provided_processors = [provided_processors]
            # Keep our processor first so strict training-style decoding is always present.
            kwargs["logits_processors"] = list(default_processors) + list(provided_processors)
        kwargs.update(extra_llm_kwargs)
    else:
        kwargs["logits_processors"] = default_processors

    return LLM(**kwargs)


def collect_predictions_for_dataset_vllm(
    *,
    llm: LLM,
    tokenizer,
    ds,
    model_alias: str,
    model_path: str,
    artifact_load_type: str,
    split_name: str,
    task_name: str,
    decoding_cfg: Dict[str, Any],
    output_dir: str,
    save_model_io: bool = True,
    use_tqdm: bool = True,
    print_first_batch: bool = True,
    request_chunk_size: int = 4096,
) -> List[Dict[str, Any]]:
    max_tokens = int(decoding_cfg.get("max_new_tokens", decoding_cfg.get("max_tokens", 128)))
    temperature = float(decoding_cfg.get("temperature", 0.0))
    top_p = float(decoding_cfg.get("top_p", 1.0))
    repetition_penalty = float(decoding_cfg.get("repetition_penalty", 1.10))
    answer_prefix_len = int(decoding_cfg.get("answer_prefix_len", 2))

    eos_token_ids = build_eval_eos_token_ids(tokenizer)
    separator_token_ids = infer_separator_token_ids_from_tokenizer(tokenizer, separator_texts=[","])
    blocked_after_separator_token_ids = list(eos_token_ids)

    io_fp = None
    io_path = None
    if save_model_io:
        io_dir = os.path.join(output_dir, "model_io")
        os.makedirs(io_dir, exist_ok=True)
        fname = (
            f"{sanitize_filename(model_alias)}."
            f"{sanitize_filename(split_name)}."
            f"{sanitize_filename(task_name)}.jsonl"
        )
        io_path = os.path.join(io_dir, fname)
        io_fp = open(io_path, "w", encoding="utf-8")

    total_samples = len(ds)
    request_chunk_size = max(1, int(request_chunk_size))
    total_chunks = (total_samples + request_chunk_size - 1) // request_chunk_size

    print(
        f"[vLLM GENERATE START] model={model_alias} split={split_name} task={task_name} "
        f"n={total_samples} chunk={request_chunk_size} max_tokens={max_tokens} "
        f"temperature={temperature} top_p={top_p} custom_logits_processor=True",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    start_time = time.time()

    try:
        chunk_starts = range(0, total_samples, request_chunk_size)
        for chunk_i, start in enumerate(
            maybe_tqdm(
                chunk_starts,
                desc=f"vLLM Generate {model_alias}/{split_name}/{task_name}",
                total=total_chunks,
                use_tqdm=use_tqdm,
                leave=False,
            ),
            start=1,
        ):
            end = min(start + request_chunk_size, total_samples)
            chunk_items: List[Dict[str, Any]] = []
            prompts_list: List[Dict[str, Any]] = []
            sampling_params_list: List[SamplingParams] = []

            for idx in range(start, end):
                raw = ds[int(idx)]
                prompt_text, reference_text, prompt_ids, ref_ids = decode_prompt_and_reference_from_dataset_row(
                    raw,
                    tokenizer,
                    ignore_index=IGNORE_INDEX,
                )

                keep_prefix_len = min(max(0, answer_prefix_len), int(ref_ids.numel()))
                answer_prefix_ids = ref_ids[:keep_prefix_len].long()
                model_input_ids = torch.cat([prompt_ids.long(), answer_prefix_ids], dim=0)

                if model_input_ids.numel() == 0:
                    pad_or_eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    model_input_ids = torch.tensor([int(pad_or_eos)], dtype=torch.long)

                model_input_id_list = [int(x) for x in model_input_ids.tolist()]
                answer_prefix_id_list = [int(x) for x in answer_prefix_ids.tolist()]
                prefix_text = tokenizer.decode(answer_prefix_id_list, skip_special_tokens=False)
                model_input_text = tokenizer.decode(model_input_id_list, skip_special_tokens=False).strip()

                logits_processor_config = {
                    "prompt_ignore_len": int(prompt_ids.numel()),
                    "answer_prefix_ids": [int(x) for x in answer_prefix_id_list],
                    "penalty": float(repetition_penalty),
                    "separator_token_ids": [int(x) for x in separator_token_ids],
                    "blocked_after_separator_token_ids": [
                        int(x) for x in blocked_after_separator_token_ids
                    ],
                }

                sampling_params = build_sampling_params_for_request(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_ids=eos_token_ids,
                    logits_processor_config=logits_processor_config,
                )

                chunk_items.append(
                    {
                        "idx": int(idx),
                        "prompt_text": str(prompt_text).strip(),
                        "reference_text": str(reference_text).strip(),
                        "answer_prefix_text": str(prefix_text),
                        "model_input_text": str(model_input_text),
                    }
                )
                prompts_list.append({"prompt_token_ids": model_input_id_list})
                sampling_params_list.append(sampling_params)

            if not chunk_items:
                continue

            # Token-id input for current vLLM: pass a list of TokensPrompt dicts.
            # Newer vLLM versions removed the `prompt_token_ids=` keyword from
            # LLM.generate(); token ids must be embedded in each prompt object.
            outputs = llm.generate(
                prompts=prompts_list,
                sampling_params=sampling_params_list,
                use_tqdm=False,
            )

            if len(outputs) != len(chunk_items):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(chunk_items)} requests."
                )

            for item, out in zip(chunk_items, outputs):
                if not out.outputs:
                    generated_token_ids: List[int] = []
                else:
                    generated_token_ids = [int(x) for x in list(out.outputs[0].token_ids)]

                suffix_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
                prediction_raw = (item["answer_prefix_text"] + suffix_text).strip()
                reference_raw = item["reference_text"]
                prompt_raw = item["prompt_text"]
                model_input_raw = item["model_input_text"]
                answer_prefix_raw = item["answer_prefix_text"]

                row: Dict[str, Any] = {
                    "model_alias": model_alias,
                    "model_path": str(model_path),
                    "artifact_load_type": artifact_load_type,
                    "split": split_name,
                    "task": task_name,
                    "idx": int(item["idx"]),
                    "prompt": prompt_raw,
                    "answer_prefix": answer_prefix_raw,
                    "model_input": model_input_raw,
                    "prediction_raw": prediction_raw,
                    "reference_raw": reference_raw,
                }

                if task_name == "cnn_dm":
                    row["prediction"] = clean_summary_for_metric(prediction_raw)
                    row["reference"] = clean_summary_for_metric(reference_raw)
                elif task_name == "kptime":
                    pred_parsed = extract_kptime_field_payloads(prediction_raw)
                    ref_parsed = extract_kptime_field_payloads(reference_raw)

                    row["prediction"] = clean_payload_text(prediction_raw)
                    row["reference"] = clean_payload_text(reference_raw)
                    row["pred_categories"] = pred_parsed["categories_payload"]
                    row["pred_keywords"] = pred_parsed["keywords_payload"]
                    row["ref_categories"] = ref_parsed["categories_payload"]
                    row["ref_keywords"] = ref_parsed["keywords_payload"]
                    row["pred_has_categories"] = pred_parsed["has_categories"]
                    row["pred_has_keywords"] = pred_parsed["has_keywords"]
                    row["ref_has_categories"] = ref_parsed["has_categories"]
                    row["ref_has_keywords"] = ref_parsed["has_keywords"]
                else:
                    row["prediction"] = clean_payload_text(prediction_raw)
                    row["reference"] = clean_payload_text(reference_raw)

                rows.append(row)

                if io_fp is not None:
                    compact: Dict[str, Any] = {
                        "model_alias": model_alias,
                        "split": split_name,
                        "task": task_name,
                        "idx": int(item["idx"]),
                        "prompt": prompt_raw,
                        "answer_prefix": answer_prefix_raw,
                        "model_input": model_input_raw,
                        "prediction": prediction_raw,
                        "reference": reference_raw,
                    }
                    if task_name == "kptime":
                        compact["parsed_prediction"] = {
                            "categories": row.get("pred_categories", ""),
                            "keywords": row.get("pred_keywords", ""),
                        }
                        compact["parsed_reference"] = {
                            "categories": row.get("ref_categories", ""),
                            "keywords": row.get("ref_keywords", ""),
                        }
                    io_fp.write(json.dumps(compact, ensure_ascii=False) + "\n")

            if print_first_batch and chunk_i == 1:
                print("\n[FIRST vLLM GENERATION CHUNK PREVIEW]", flush=True)
                for preview_row in rows[-min(len(rows), 2):]:
                    print("-" * 100, flush=True)
                    print(
                        f"model={model_alias} split={split_name} task={task_name} idx={preview_row['idx']}",
                        flush=True,
                    )
                    print("[MODEL_INPUT]", preview_row["model_input"][:1000], flush=True)
                    print("[PRED       ]", preview_row["prediction_raw"][:1000], flush=True)
                    print("[REF        ]", preview_row["reference_raw"][:1000], flush=True)

    finally:
        if io_fp is not None:
            io_fp.close()

    elapsed = time.time() - start_time
    print(
        f"[vLLM GENERATE DONE] model={model_alias} split={split_name} task={task_name} "
        f"n={len(rows)} elapsed={format_seconds(elapsed)} io={io_path}",
        flush=True,
    )
    return rows


# ============================================================
# 8. RoBERTa MoverScore-like OT
# ============================================================

def load_roberta_encoder(
    roberta_path: str,
    *,
    device: str,
    dtype: torch.dtype,
    local_files_only: bool = False,
):
    print("[LOAD ROBERTA TOKENIZER]", roberta_path, flush=True)
    roberta_tokenizer = AutoTokenizer.from_pretrained(
        roberta_path,
        local_files_only=local_files_only,
        use_fast=False,
    )

    print("[LOAD ROBERTA MODEL]", roberta_path, flush=True)
    roberta_model = AutoModel.from_pretrained(
        roberta_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    roberta_model.to(device)
    roberta_model.eval()
    return roberta_tokenizer, roberta_model


@torch.inference_mode()
def mover_encode_texts_first_last_avg_hidden_batch(
    *,
    texts: Sequence[str],
    roberta_tokenizer,
    roberta_model,
    max_length: int = 512,
    first_layer_index: int = 0,
) -> Dict[str, torch.Tensor]:
    device = model_device(roberta_model)
    texts = [clean_payload_text(x) for x in texts]

    inputs = roberta_tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = roberta_model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = outputs.hidden_states
    if first_layer_index < 0 or first_layer_index >= len(hidden_states):
        raise ValueError(
            f"first_layer_index={first_layer_index} out of range. "
            f"Available hidden states: 0 ~ {len(hidden_states) - 1}"
        )

    first_hidden = hidden_states[first_layer_index]
    last_hidden = hidden_states[-1]
    mixed_hidden = (first_hidden + last_hidden) / 2.0

    attention_mask = inputs["attention_mask"].bool()
    special_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    for sid in roberta_tokenizer.all_special_ids:
        special_mask |= inputs["input_ids"].eq(int(sid))
    valid_mask = attention_mask & (~special_mask)

    return {"mixed_hidden": mixed_hidden, "valid_mask": valid_mask}


def mover_score_from_batch_hidden(
    *,
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
        return {"score": 1.0, "ot_distance": 0.0, "pred_token_count": 0, "ref_token_count": 0}
    if pred_count == 0 or ref_count == 0:
        return {"score": 0.0, "ot_distance": 1.0, "pred_token_count": pred_count, "ref_token_count": ref_count}

    pred_hidden = F.normalize(pred_hidden.float(), p=2, dim=-1)
    ref_hidden = F.normalize(ref_hidden.float(), p=2, dim=-1)
    sim = pred_hidden @ ref_hidden.T
    if clamp_similarity:
        sim = sim.clamp(min=0.0, max=1.0)
    dist = (1.0 - sim).clamp_min(0.0)
    dist_np = dist.detach().cpu().numpy().astype(np.float64)

    pred_weights = np.ones(pred_count, dtype=np.float64)
    ref_weights = np.ones(ref_count, dtype=np.float64)
    pred_weights = pred_weights / pred_weights.sum()
    ref_weights = ref_weights / ref_weights.sum()

    ot_distance = float(ot.emd2(pred_weights, ref_weights, dist_np))
    score = max(0.0, min(1.0, float(1.0 - ot_distance)))
    return {"score": score, "ot_distance": ot_distance, "pred_token_count": pred_count, "ref_token_count": ref_count}


@torch.inference_mode()
def score_summary_mover_from_rows(
    *,
    rows: List[Dict[str, Any]],
    roberta_tokenizer,
    roberta_model,
    encoder_batch_size: int = 512,
    max_length: int = 512,
    first_layer_index: int = 0,
    use_tqdm: bool = True,
    desc: str = "",
) -> Dict[str, float]:
    pair_count = len(rows)
    if pair_count == 0:
        return {"mover_score": 0.0, "num_pairs": 0}

    scores: List[float] = []
    encoder_batch_size = max(1, int(encoder_batch_size))
    batch_starts = range(0, pair_count, encoder_batch_size)

    for start in maybe_tqdm(
        batch_starts,
        desc=desc or "Scoring summary",
        total=(pair_count + encoder_batch_size - 1) // encoder_batch_size,
        use_tqdm=use_tqdm,
        leave=False,
    ):
        end = min(start + encoder_batch_size, pair_count)
        batch_texts: List[str] = []
        for row in rows[start:end]:
            batch_texts.extend([
                clean_summary_for_metric(row["prediction_raw"]),
                clean_summary_for_metric(row["reference_raw"]),
            ])

        encoded = mover_encode_texts_first_last_avg_hidden_batch(
            texts=batch_texts,
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            max_length=max_length,
            first_layer_index=first_layer_index,
        )
        mixed_hidden = encoded["mixed_hidden"]
        valid_mask = encoded["valid_mask"]

        for i in range(end - start):
            base = i * 2
            raw_score = mover_score_from_batch_hidden(
                mixed_hidden=mixed_hidden,
                valid_mask=valid_mask,
                pred_index=base,
                ref_index=base + 1,
                clamp_similarity=True,
            )
            scores.append(float(raw_score["score"]))
        del encoded, mixed_hidden, valid_mask

    return {"mover_score": float(sum(scores) / max(len(scores), 1)), "num_pairs": int(pair_count)}


@torch.inference_mode()
def score_kptime_mover_from_rows(
    *,
    rows: List[Dict[str, Any]],
    roberta_tokenizer,
    roberta_model,
    encoder_batch_size: int = 512,
    max_length: int = 512,
    first_layer_index: int = 0,
    use_tqdm: bool = True,
    desc: str = "",
) -> Dict[str, float]:
    pair_count = len(rows)
    if pair_count == 0:
        return {
            "mover_category_score": 0.0,
            "mover_keyword_score": 0.0,
            "mover_final_score": 0.0,
            "num_pairs": 0,
        }

    category_scores: List[float] = []
    keyword_scores: List[float] = []
    final_scores: List[float] = []

    encoder_batch_size = max(1, int(encoder_batch_size))
    batch_starts = range(0, pair_count, encoder_batch_size)

    for start in maybe_tqdm(
        batch_starts,
        desc=desc or "Scoring kptime",
        total=(pair_count + encoder_batch_size - 1) // encoder_batch_size,
        use_tqdm=use_tqdm,
        leave=False,
    ):
        end = min(start + encoder_batch_size, pair_count)
        batch_texts: List[str] = []

        for row in rows[start:end]:
            pred_cat = row.get("pred_categories", "")
            pred_kw = row.get("pred_keywords", "")
            ref_cat = row.get("ref_categories", "")
            ref_kw = row.get("ref_keywords", "")
            if not any([pred_cat, pred_kw, ref_cat, ref_kw]):
                pred_parsed = extract_kptime_field_payloads(row.get("prediction_raw", ""))
                ref_parsed = extract_kptime_field_payloads(row.get("reference_raw", ""))
                pred_cat = pred_parsed["categories_payload"]
                pred_kw = pred_parsed["keywords_payload"]
                ref_cat = ref_parsed["categories_payload"]
                ref_kw = ref_parsed["keywords_payload"]
            batch_texts.extend([pred_cat, pred_kw, ref_cat, ref_kw])

        encoded = mover_encode_texts_first_last_avg_hidden_batch(
            texts=batch_texts,
            roberta_tokenizer=roberta_tokenizer,
            roberta_model=roberta_model,
            max_length=max_length,
            first_layer_index=first_layer_index,
        )
        mixed_hidden = encoded["mixed_hidden"]
        valid_mask = encoded["valid_mask"]

        for i in range(end - start):
            base = i * 4
            cat_raw = mover_score_from_batch_hidden(
                mixed_hidden=mixed_hidden,
                valid_mask=valid_mask,
                pred_index=base,
                ref_index=base + 2,
                clamp_similarity=True,
            )
            kw_raw = mover_score_from_batch_hidden(
                mixed_hidden=mixed_hidden,
                valid_mask=valid_mask,
                pred_index=base + 1,
                ref_index=base + 3,
                clamp_similarity=True,
            )
            cat_score = float(cat_raw["score"])
            kw_score = float(kw_raw["score"])
            final_score = float((cat_score + kw_score) / 2.0)
            category_scores.append(cat_score)
            keyword_scores.append(kw_score)
            final_scores.append(final_score)
        del encoded, mixed_hidden, valid_mask

    n = max(len(final_scores), 1)
    return {
        "mover_category_score": float(sum(category_scores) / n),
        "mover_keyword_score": float(sum(keyword_scores) / n),
        "mover_final_score": float(sum(final_scores) / n),
        "num_pairs": int(pair_count),
    }


# ============================================================
# 9. Main runner
# ============================================================

def build_default_decoding_cfg() -> Dict[str, Any]:
    return {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "repetition_penalty": 1.10,
        "answer_prefix_len": 2,
        "request_chunk_size": 4096,
        "encoder_batch_size": 512,
    }


def run_mixed_full_valid_test_eval_vllm(
    *,
    dataset_path: Optional[str] = None,
    dataset_obj: Optional[Any] = None,
    cnn_dm_dataset_path: Optional[str] = None,
    kptime_dataset_path: Optional[str] = None,
    model_artifact_paths: Dict[str, str],
    base_model_path: str,
    roberta_path: str,
    output_dir: str = "runs/mixed_full_valid_test_eval_vllm",

    validation_names: Sequence[str] = ("validation", "valid", "eval"),
    test_names: Sequence[str] = ("test",),
    task_names: Sequence[str] = ("cnn_dm", "kptime"),

    decoding_cfg: Optional[Dict[str, Any]] = None,

    tokenizer_path: Optional[str] = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,

    vllm_dtype: Optional[Any] = "auto",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: Optional[int] = None,
    enforce_eager: bool = False,
    disable_log_stats: bool = True,
    seed: int = 42,
    extra_llm_kwargs: Optional[Dict[str, Any]] = None,

    # PEFT/AdaLoRA convenience: if a model_artifact_paths entry is an adapter dir,
    # merge it to a temporary full model, run vLLM, then optionally delete the temp model.
    auto_merge_peft_adapters: bool = True,
    temp_merged_model_dir: Optional[str] = None,
    delete_temp_merged_models: bool = True,
    merge_dtype: Optional[Any] = torch.float16,
    merge_device: str = "cpu",
    merge_max_shard_size: str = "2GB",

    # If the base model embedding size differs from tokenizer/adapters, create a
    # temporary resized base model once, then use it for both base eval and adapter merge.
    auto_prepare_resized_base: bool = True,
    temp_resized_base_model_dir: Optional[str] = None,
    delete_temp_resized_base_model: bool = True,
    strict_tokenizer_adapter_vocab_match: bool = True,

    roberta_dtype: Optional[Any] = torch.float16,
    roberta_device: Optional[str] = None,

    save_model_io: bool = True,
    save_predictions_csv: bool = True,
    save_csv: bool = True,

    use_tqdm: bool = True,
    print_first_batch: bool = True,
    continue_on_error: bool = True,
) -> Dict[str, Any]:
    if not isinstance(model_artifact_paths, dict) or not model_artifact_paths:
        raise ValueError("model_artifact_paths must be a non-empty dict.")

    if "base" not in model_artifact_paths:
        print("[INFO] Adding base model to model_artifact_paths as alias='base'.", flush=True)
        model_artifact_paths = dict(model_artifact_paths)
        model_artifact_paths["base"] = base_model_path

    if not torch.cuda.is_available():
        raise RuntimeError("vLLM generation requires CUDA in this script.")

    tokenizer_path = str(tokenizer_path or base_model_path)
    vllm_dtype_s = parse_vllm_dtype(vllm_dtype)
    roberta_dtype_t = parse_torch_dtype(roberta_dtype, default=torch.float16)
    roberta_device = roberta_device or "cuda"

    decoding_cfg = dict(build_default_decoding_cfg() if decoding_cfg is None else decoding_cfg)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 120, flush=True)
    print("[PURE vLLM MIXED FULL VALID/TEST GENERATION EVAL]", flush=True)
    print("vLLM version:", getattr(vllm, "__version__", "unknown"), flush=True)
    print("dataset_path:", dataset_path, flush=True)
    print("dataset_obj is not None:", dataset_obj is not None, flush=True)
    print("cnn_dm_dataset_path:", cnn_dm_dataset_path, flush=True)
    print("kptime_dataset_path:", kptime_dataset_path, flush=True)
    print("base_model_path:", base_model_path, flush=True)
    print("tokenizer_path:", tokenizer_path, flush=True)
    print("local_files_only:", local_files_only, flush=True)
    print("roberta_path:", roberta_path, flush=True)
    print("output_dir:", output_dir, flush=True)
    print("vllm_dtype:", vllm_dtype_s, flush=True)
    print("tensor_parallel_size:", tensor_parallel_size, flush=True)
    print("gpu_memory_utilization:", gpu_memory_utilization, flush=True)
    print("decoding_cfg:", json.dumps(decoding_cfg, ensure_ascii=False, indent=2, default=str), flush=True)
    print("auto_merge_peft_adapters:", auto_merge_peft_adapters, flush=True)
    print("temp_merged_model_dir:", temp_merged_model_dir, flush=True)
    print("delete_temp_merged_models:", delete_temp_merged_models, flush=True)
    print("merge_dtype:", merge_dtype, flush=True)
    print("merge_device:", merge_device, flush=True)
    print("auto_prepare_resized_base:", auto_prepare_resized_base, flush=True)
    print("temp_resized_base_model_dir:", temp_resized_base_model_dir, flush=True)
    print("delete_temp_resized_base_model:", delete_temp_resized_base_model, flush=True)
    print("custom_logits_processor_mode:", "vllm_v1_engine_registered", flush=True)
    print("=" * 120, flush=True)

    # -------------------------
    # 1) Load tokenizer / dataset
    # -------------------------
    print("\n[LOAD TOKENIZER]", tokenizer_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("tokenizer.pad_token:", repr(tokenizer.pad_token), flush=True)
    print("tokenizer.eos_token:", repr(tokenizer.eos_token), flush=True)
    print("tokenizer.padding_side:", tokenizer.padding_side, flush=True)
    print("tokenizer_len:", len(tokenizer), flush=True)

    # -------------------------
    # 1.5) Prepare resized base if needed
    # -------------------------
    # Typical Qwen case observed in PEFT checkpoints:
    #   original base embedding vocab: 151936
    #   tokenizer len / adapter saved vocab: 151665
    # PEFT loading then fails unless the base is resized before loading adapter.
    prepared_base_model_path = str(base_model_path)
    prepared_tokenizer_path = tokenizer_path
    resized_base_path_for_cleanup: Optional[str] = None

    if auto_prepare_resized_base:
        adapter_vocab_sizes = collect_adapter_saved_vocab_sizes(model_artifact_paths)
        target_vocab_size = choose_target_vocab_size_for_eval(
            tokenizer=tokenizer,
            adapter_vocab_sizes=adapter_vocab_sizes,
            strict_tokenizer_adapter_vocab_match=strict_tokenizer_adapter_vocab_match,
        )

        resize_root = temp_resized_base_model_dir
        if resize_root is None:
            if temp_merged_model_dir is not None:
                resize_root = os.path.join(str(temp_merged_model_dir), f"_resized_base_vocab_{target_vocab_size}")
            else:
                resize_root = os.path.join(output_dir, f"_tmp_resized_base_vocab_{target_vocab_size}")

        maybe_resized_path = prepare_resized_base_model_if_needed(
            original_base_model_path=str(base_model_path),
            tokenizer=tokenizer,
            target_vocab_size=int(target_vocab_size),
            output_path=str(resize_root),
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            dtype=merge_dtype,
            device=merge_device,
            max_shard_size=merge_max_shard_size,
        )

        prepared_base_model_path = str(maybe_resized_path)
        if os.path.abspath(prepared_base_model_path) != os.path.abspath(str(base_model_path)):
            prepared_tokenizer_path = prepared_base_model_path
            resized_base_path_for_cleanup = prepared_base_model_path
            print("[USING TEMP RESIZED BASE]", prepared_base_model_path, flush=True)
            print("[USING TEMP RESIZED TOKENIZER]", prepared_tokenizer_path, flush=True)
        else:
            print("[USING ORIGINAL BASE]", prepared_base_model_path, flush=True)

    print("\n[LOAD DATASET]", flush=True)
    eval_map = load_eval_datasets_flexible(
        dataset_path=dataset_path,
        dataset_obj=dataset_obj,
        cnn_dm_dataset_path=cnn_dm_dataset_path,
        kptime_dataset_path=kptime_dataset_path,
        validation_names=validation_names,
        test_names=test_names,
        task_names=task_names,
    )

    for split_name, task_map in eval_map.items():
        for task_name, ds in task_map.items():
            print(f"[DATASET] split={split_name} task={task_name} size={len(ds)}", flush=True)

    # -------------------------
    # 2) vLLM generation
    # -------------------------
    all_prediction_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    model_aliases = list(model_artifact_paths.keys())

    for model_alias, model_path in model_artifact_paths.items():
        print("\n" + "=" * 140, flush=True)
        print(f"[EVAL MODEL WITH vLLM] alias={model_alias}", flush=True)
        print(f"[MODEL PATH] {model_path}", flush=True)
        print("=" * 140, flush=True)

        start_time = time.time()
        llm = None
        status = "ok"
        error = ""
        artifact_load_type = "vllm_full_model"
        original_model_path = str(model_path)
        effective_vllm_model_path = original_model_path
        temp_merged_path: Optional[str] = None

        # If this entry points to the original base, evaluate the prepared resized base instead.
        if os.path.abspath(original_model_path) == os.path.abspath(str(base_model_path)):
            effective_vllm_model_path = prepared_base_model_path

        try:
            if is_peft_adapter_dir(original_model_path):
                if not auto_merge_peft_adapters:
                    raise ValueError(
                        f"{original_model_path} looks like a PEFT adapter directory. "
                        "Set auto_merge_peft_adapters=True or provide a full/merged model path."
                    )

                adapter_type = read_peft_adapter_type(original_model_path)
                artifact_load_type = f"vllm_temp_merged_{adapter_type.lower()}_adapter"
                temp_root = str(temp_merged_model_dir or os.path.join(output_dir, "_tmp_merged_models"))
                temp_merged_path = os.path.join(temp_root, sanitize_filename(model_alias))

                effective_vllm_model_path = merge_peft_adapter_to_temp_full_model(
                    base_model_path=str(prepared_base_model_path),
                    adapter_path=original_model_path,
                    output_path=temp_merged_path,
                    tokenizer=tokenizer,
                    local_files_only=local_files_only,
                    trust_remote_code=trust_remote_code,
                    merge_dtype=merge_dtype,
                    merge_device=merge_device,
                    max_shard_size=merge_max_shard_size,
                )
            elif os.path.abspath(original_model_path) == os.path.abspath(str(base_model_path)):
                if os.path.abspath(effective_vllm_model_path) != os.path.abspath(original_model_path):
                    artifact_load_type = "vllm_temp_resized_base_full_model"
                else:
                    artifact_load_type = "vllm_base_full_model"

            llm = make_vllm_llm(
                model_path=str(effective_vllm_model_path),
                tokenizer_path=prepared_tokenizer_path,
                trust_remote_code=trust_remote_code,
                dtype=vllm_dtype_s,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                disable_log_stats=disable_log_stats,
                seed=seed,
                extra_llm_kwargs=extra_llm_kwargs,
            )

            for split_name, task_map in eval_map.items():
                for task_name, ds in task_map.items():
                    rows = collect_predictions_for_dataset_vllm(
                        llm=llm,
                        tokenizer=tokenizer,
                        ds=ds,
                        model_alias=model_alias,
                        model_path=original_model_path,
                        artifact_load_type=artifact_load_type,
                        split_name=split_name,
                        task_name=task_name,
                        decoding_cfg=decoding_cfg,
                        output_dir=output_dir,
                        save_model_io=save_model_io,
                        use_tqdm=use_tqdm,
                        print_first_batch=print_first_batch,
                        request_chunk_size=int(decoding_cfg.get("request_chunk_size", 4096)),
                    )
                    all_prediction_rows.extend(rows)

        except Exception as e:
            status = "error"
            error = repr(e)
            print(f"[ERROR] model_alias={model_alias}", flush=True)
            print(error, flush=True)
            if not continue_on_error:
                raise

        finally:
            elapsed_min = (time.time() - start_time) / 60.0
            runtime_rows.append(
                {
                    "model_alias": model_alias,
                    "model_path": str(model_path),
                    "artifact_load_type": artifact_load_type,
                    "status": status,
                    "error": error,
                    "elapsed_min_generation_only": float(elapsed_min),
                }
            )
            cleanup_vllm_engine(llm)
            llm = None
            if temp_merged_path and delete_temp_merged_models:
                try:
                    print(f"[DELETE TEMP MERGED MODEL] {temp_merged_path}", flush=True)
                    shutil.rmtree(temp_merged_path, ignore_errors=True)
                except Exception as delete_e:
                    print(f"[WARN] Failed to delete temp merged model {temp_merged_path}: {delete_e!r}", flush=True)

    if resized_base_path_for_cleanup and delete_temp_resized_base_model:
        try:
            print(f"[DELETE TEMP RESIZED BASE] {resized_base_path_for_cleanup}", flush=True)
            shutil.rmtree(resized_base_path_for_cleanup, ignore_errors=True)
        except Exception as delete_e:
            print(f"[WARN] Failed to delete temp resized base {resized_base_path_for_cleanup}: {delete_e!r}", flush=True)

    predictions_df = pd.DataFrame(all_prediction_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    if predictions_df.empty:
        raise RuntimeError("No predictions were generated. Please check dataset/model paths and vLLM loading.")

    if save_predictions_csv:
        pred_csv_path = os.path.join(output_dir, "predictions_minimal.csv")
        predictions_df.to_csv(pred_csv_path, index=False)
        print("[SAVE]", pred_csv_path, flush=True)

    # -------------------------
    # 3) Scoring
    # -------------------------
    print("\n[LOAD ROBERTA SCORER]", roberta_path, flush=True)
    roberta_tokenizer, roberta_model = load_roberta_encoder(
        roberta_path,
        device=roberta_device,
        dtype=roberta_dtype_t,
        local_files_only=False,
    )

    metrics_long_rows: List[Dict[str, Any]] = []

    try:
        for model_alias in model_aliases:
            for split_name in ["validation", "test"]:
                for task_name in task_names:
                    sub_df = predictions_df[
                        (predictions_df["model_alias"] == model_alias)
                        & (predictions_df["split"] == split_name)
                        & (predictions_df["task"] == task_name)
                    ].sort_values("idx")

                    rows = sub_df.to_dict("records")
                    if not rows:
                        continue

                    model_path = rows[0]["model_path"]
                    artifact_load_type = rows[0]["artifact_load_type"]

                    if task_name == "cnn_dm":
                        score_pack = score_summary_mover_from_rows(
                            rows=rows,
                            roberta_tokenizer=roberta_tokenizer,
                            roberta_model=roberta_model,
                            encoder_batch_size=int(decoding_cfg.get("encoder_batch_size", 512)),
                            max_length=512,
                            first_layer_index=0,
                            use_tqdm=use_tqdm,
                            desc=f"Score {model_alias}/{split_name}/cnn_dm",
                        )
                        metrics_long_rows.append(
                            {
                                "model_alias": model_alias,
                                "model_path": model_path,
                                "artifact_load_type": artifact_load_type,
                                "split": split_name,
                                "task": task_name,
                                "metric_name": f"{split_name}_cnn_dm_mover_score",
                                "metric_value": float(score_pack["mover_score"]),
                                "num_samples": int(score_pack["num_pairs"]),
                            }
                        )

                    elif task_name == "kptime":
                        score_pack = score_kptime_mover_from_rows(
                            rows=rows,
                            roberta_tokenizer=roberta_tokenizer,
                            roberta_model=roberta_model,
                            encoder_batch_size=int(decoding_cfg.get("encoder_batch_size", 512)),
                            max_length=512,
                            first_layer_index=0,
                            use_tqdm=use_tqdm,
                            desc=f"Score {model_alias}/{split_name}/kptime",
                        )
                        for short_name, value in [
                            ("mover_category_score", score_pack["mover_category_score"]),
                            ("mover_keyword_score", score_pack["mover_keyword_score"]),
                            ("mover_final_score", score_pack["mover_final_score"]),
                        ]:
                            metrics_long_rows.append(
                                {
                                    "model_alias": model_alias,
                                    "model_path": model_path,
                                    "artifact_load_type": artifact_load_type,
                                    "split": split_name,
                                    "task": task_name,
                                    "metric_name": f"{split_name}_kptime_{short_name}",
                                    "metric_value": float(value),
                                    "num_samples": int(score_pack["num_pairs"]),
                                }
                            )

    finally:
        try:
            del roberta_model
            del roberta_tokenizer
        except Exception:
            pass
        cleanup_cuda()

    metrics_df = pd.DataFrame(metrics_long_rows)

    # -------------------------
    # 4) Build summary + combo
    # -------------------------
    summary_rows: List[Dict[str, Any]] = []
    metric_lookup: Dict[Tuple[str, str], float] = {}
    for _, r in metrics_df.iterrows():
        metric_lookup[(str(r["model_alias"]), str(r["metric_name"]))] = float(r["metric_value"])

    model_meta = (
        predictions_df[["model_alias", "model_path", "artifact_load_type"]]
        .drop_duplicates("model_alias")
        .set_index("model_alias")
        .to_dict("index")
    )

    for model_alias in model_aliases:
        meta = model_meta.get(model_alias, {})
        row: Dict[str, Any] = {
            "model_alias": model_alias,
            "model_path": meta.get("model_path", str(model_artifact_paths[model_alias])),
            "artifact_load_type": meta.get("artifact_load_type", "unknown"),
        }

        for split_name in ["validation", "test"]:
            cnn_key = f"{split_name}_cnn_dm_mover_score"
            cat_key = f"{split_name}_kptime_mover_category_score"
            kw_key = f"{split_name}_kptime_mover_keyword_score"
            kptime_key = f"{split_name}_kptime_mover_final_score"
            combo_key = f"{split_name}_combo_mover_score"

            row[cnn_key] = metric_lookup.get((model_alias, cnn_key), np.nan)
            row[cat_key] = metric_lookup.get((model_alias, cat_key), np.nan)
            row[kw_key] = metric_lookup.get((model_alias, kw_key), np.nan)
            row[kptime_key] = metric_lookup.get((model_alias, kptime_key), np.nan)

            if math.isfinite(float(row[cnn_key])) and math.isfinite(float(row[kptime_key])):
                row[combo_key] = float(0.5 * row[cnn_key] + 0.5 * row[kptime_key])
            else:
                row[combo_key] = np.nan

            metrics_long_rows.append(
                {
                    "model_alias": model_alias,
                    "model_path": row["model_path"],
                    "artifact_load_type": row["artifact_load_type"],
                    "split": split_name,
                    "task": "combo",
                    "metric_name": combo_key,
                    "metric_value": row[combo_key],
                    "num_samples": np.nan,
                }
            )

        summary_rows.append(row)

    metrics_df = pd.DataFrame(metrics_long_rows)
    summary_df = pd.DataFrame(summary_rows)

    # -------------------------
    # 5) Delta vs base
    # -------------------------
    if "base" in set(summary_df["model_alias"]):
        base_row = summary_df[summary_df["model_alias"] == "base"].iloc[0]
        base_valid_combo = float(base_row.get("validation_combo_mover_score", np.nan))
        base_test_combo = float(base_row.get("test_combo_mover_score", np.nan))
        summary_df["delta_vs_base_validation_combo"] = (
            summary_df["validation_combo_mover_score"].astype(float) - base_valid_combo
        )
        summary_df["delta_vs_base_test_combo"] = (
            summary_df["test_combo_mover_score"].astype(float) - base_test_combo
        )
    else:
        summary_df["delta_vs_base_validation_combo"] = np.nan
        summary_df["delta_vs_base_test_combo"] = np.nan

    # -------------------------
    # 6) Select best by full validation combo
    # -------------------------
    finetuned_df = summary_df[summary_df["model_alias"] != "base"].copy()
    finetuned_df = finetuned_df[finetuned_df["validation_combo_mover_score"].notna()]

    best_by_full_valid = None
    if not finetuned_df.empty:
        best_idx = finetuned_df["validation_combo_mover_score"].astype(float).idxmax()
        best_row = finetuned_df.loc[best_idx].to_dict()
        summary_df["selected_best_finetuned_by_full_valid_combo"] = (
            summary_df["model_alias"] == best_row["model_alias"]
        )
        best_by_full_valid = {
            "best_model_alias": best_row["model_alias"],
            "best_model_path": best_row["model_path"],
            "artifact_load_type": best_row.get("artifact_load_type", "unknown"),
            "selection_metric": "validation_combo_mover_score",
            "best_validation_combo_mover_score": float(best_row["validation_combo_mover_score"]),
            "best_test_combo_mover_score": float(best_row["test_combo_mover_score"]),
            "delta_vs_base_validation_combo": (
                None if pd.isna(best_row.get("delta_vs_base_validation_combo", np.nan))
                else float(best_row["delta_vs_base_validation_combo"])
            ),
            "delta_vs_base_test_combo": (
                None if pd.isna(best_row.get("delta_vs_base_test_combo", np.nan))
                else float(best_row["delta_vs_base_test_combo"])
            ),
        }
    else:
        summary_df["selected_best_finetuned_by_full_valid_combo"] = False

    # -------------------------
    # 7) Save outputs
    # -------------------------
    paths: Dict[str, str] = {}
    if save_csv:
        metrics_path = os.path.join(output_dir, "metrics_long.csv")
        summary_path = os.path.join(output_dir, "summary_vs_base.csv")
        runtime_path = os.path.join(output_dir, "runtime.csv")
        best_path = os.path.join(output_dir, "best_finetuned_by_full_valid_combo.json")

        metrics_df.to_csv(metrics_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        runtime_df.to_csv(runtime_path, index=False)
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_by_full_valid, f, ensure_ascii=False, indent=2)

        paths.update(
            {
                "metrics_long": metrics_path,
                "summary_vs_base": summary_path,
                "runtime": runtime_path,
                "best_finetuned_by_full_valid_combo": best_path,
                "model_io_dir": os.path.join(output_dir, "model_io"),
            }
        )
        print("\n[SAVED FILES]", flush=True)
        for k, v in paths.items():
            print(f"{k}: {v}", flush=True)

    display_cols = [
        "model_alias",
        "validation_cnn_dm_mover_score",
        "validation_kptime_mover_final_score",
        "validation_combo_mover_score",
        "test_cnn_dm_mover_score",
        "test_kptime_mover_final_score",
        "test_combo_mover_score",
        "delta_vs_base_validation_combo",
        "delta_vs_base_test_combo",
        "selected_best_finetuned_by_full_valid_combo",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]

    print("\n[SUMMARY]", flush=True)
    print(summary_df[display_cols].to_string(index=False), flush=True)

    if best_by_full_valid is not None:
        print("\n[BEST FINETUNED BY FULL VALID COMBO]", flush=True)
        print(json.dumps(best_by_full_valid, ensure_ascii=False, indent=2), flush=True)

    return {
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "runtime_df": runtime_df,
        "predictions_df": predictions_df,
        "best_by_full_valid": best_by_full_valid,
        "paths": paths,
    }

# Notebook usage: call build_default_decoding_cfg() and run_mixed_full_valid_test_eval_vllm(...) from a separate cell.

# Auto-merge PEFT usage: adapter dirs in model_artifact_paths are merged one at a time into a temporary full model for vLLM, then deleted by default.

# Flexible dataset loading: use dataset_obj=out, dataset_path=<nested_saved_root>, or cnn_dm_dataset_path=... and kptime_dataset_path=... .
