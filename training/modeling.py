"""Qwen loading and PEFT/AdaLoRA setup preserved from the original notebook."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Literal

import torch
from transformers import AutoModelForCausalLM, TrainerCallback
from transformers import BitsAndBytesConfig

from peft import LoraConfig, AdaLoraConfig, TaskType, get_peft_model

try:
    # This is usually required for 4-bit/8-bit training (QLoRA or k-bit).
    from peft import prepare_model_for_kbit_training
except Exception:
    prepare_model_for_kbit_training = None


# =========================
# 1) small utils
# =========================
def print_trainable_parameters(model) -> None:
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / max(total, 1)
    print(f"[trainable] {trainable:,} / {total:,} ({pct:.2f}%)")


def guess_lora_target_modules(model) -> List[str]:
    """
    Infer target_modules when possible (names vary by architecture).
    - Phi-3 / Phi-3.5: qkv_proj/o_proj/gate_up_proj/down_proj
    - LLaMA/Qwen/Gemma/Mistral: q_proj/k_proj/v_proj/o_proj
    - GPT2: c_attn/c_proj
    - Falcon: query_key_value/dense
    If inference fails, require target_modules to be provided explicitly.
    """
    names = [n for n, _ in model.named_modules()]

    # Phi-3 / Phi-3.5 like
    phi_like = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    if any(any(k in n for n in names) for k in phi_like):
        return [k for k in phi_like if any(k in n for n in names)]

    # LLaMA-like
    llama_like = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if any(any(k in n for n in names) for k in llama_like):
        return [k for k in llama_like if any(k in n for n in names)]

    # GPT-2 like
    gpt2_like = ["c_attn", "c_proj"]
    if any(any(k in n for n in names) for k in gpt2_like):
        return [k for k in gpt2_like if any(k in n for n in names)]

    # Falcon-like
    falcon_like = ["query_key_value", "dense"]
    if any(any(k in n for n in names) for k in falcon_like):
        return [k for k in falcon_like if any(k in n for n in names)]

    raise ValueError(
        "Cannot guess LoRA target_modules for this model architecture.\n"
        "Please pass target_modules manually, e.g. ['q_proj','k_proj','v_proj','o_proj']."
    )


def infer_total_optim_steps(args, train_len: int) -> int:
    """
    total_step is the number of optimizer updates after gradient accumulation.
    """
    max_steps = getattr(args, "max_steps", 0)
    if isinstance(max_steps, int) and max_steps > 0:
        return max_steps

    world_size = int(getattr(args, "world_size", 1) or 1)
    per_device_bs = int(getattr(args, "per_device_train_batch_size", 1))
    ga = int(getattr(args, "gradient_accumulation_steps", 1))
    steps_per_epoch = math.ceil(train_len / max(per_device_bs * world_size * ga, 1))
    num_epochs = float(getattr(args, "num_train_epochs", 1))
    return int(steps_per_epoch * num_epochs)


def _convert_alpha_for_rslora(alpha: int, r: int) -> int:
    """
    Approximate a vanilla LoRA alpha that gives a similar scale under rsLoRA.

    vanilla: scale = alpha / r
    rsLoRA : scale = alpha / sqrt(r)

    To keep the initial scale approximately unchanged after enabling rsLoRA,
    use alpha_new ≈ alpha_old / sqrt(r).
    """
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r}")
    new_alpha = max(1, int(round(alpha / math.sqrt(r))))
    return new_alpha


# =========================
# 2) core: apply LoRA / AdaLoRA
# =========================
@dataclass
class PeftSpec:
    method: Literal["lora", "adalora"] = "lora"

    # Shared parameters
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    bias: str = "none"  # "none" | "all" | "lora_only"
    target_modules: Optional[Sequence[str] | str] = None

    # rsLoRA
    use_rslora: bool = False
    keep_alpha_scale_when_use_rslora: bool = False

    # DoRA
    use_dora: bool = False

    init_lora_weights: str | bool = True

    # Additional AdaLoRA parameters
    init_r: Optional[int] = None
    target_r: Optional[int] = None
    tinit: int = 0
    tfinal: int = 0
    deltaT: int = 1
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    total_step: Optional[int] = None


def apply_peft_for_causal_lm(
    model,
    *,
    tokenizer=None,
    peft: PeftSpec = PeftSpec(),
    is_kbit_training: bool = False,
    gradient_checkpointing: bool = True,
    train_embeddings_for_new_tokens: bool = False,
    verbose: bool = True,
):
    # (A) Resize embeddings and record the old/new vocabulary sizes.
    old_vocab_size = None
    new_vocab_size = None
    if tokenizer is not None:
        emb0 = model.get_input_embeddings()
        old_vocab_size = int(emb0.weight.size(0))
        new_vocab_size = int(len(tokenizer))
        if old_vocab_size != new_vocab_size:
            model.resize_token_embeddings(new_vocab_size)
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # (B) Prepare for k-bit training.
    if is_kbit_training:
        if prepare_model_for_kbit_training is None:
            raise RuntimeError("prepare_model_for_kbit_training unavailable; check peft version.")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    # (C) Enable checkpointing and disable the cache.
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # (D) target_modules
    target_modules = (
        peft.target_modules
        if peft.target_modules is not None
        else guess_lora_target_modules(model)
    )

    # (E) Resolve alpha.
    effective_alpha = peft.alpha
    if peft.use_rslora and peft.keep_alpha_scale_when_use_rslora:
        effective_alpha = _convert_alpha_for_rslora(peft.alpha, peft.r)

    if verbose:
        print(
            "[PEFT] "
            f"method={peft.method} "
            f"use_rslora={peft.use_rslora} "
            f"use_dora={peft.use_dora} "
            f"(r={peft.r}, alpha={peft.alpha} -> effective_alpha={effective_alpha})"
        )

    # (F) Build the configuration for the selected method.
    if peft.method == "lora":
        cfg_kwargs = dict(
            task_type=TaskType.CAUSAL_LM,
            r=peft.r,
            lora_alpha=effective_alpha,
            lora_dropout=peft.dropout,
            bias=peft.bias,
            target_modules=target_modules,
            use_rslora=peft.use_rslora,
            use_dora=peft.use_dora,
            init_lora_weights=peft.init_lora_weights,
        )

        try:
            cfg = LoraConfig(**cfg_kwargs)
        except TypeError as e:
            if peft.use_dora:
                raise TypeError(
                    "Current peft version does not support use_dora in LoraConfig. "
                    "Please upgrade peft."
                ) from e

            # Compatibility fallback for older versions.
            cfg_kwargs.pop("use_rslora", None)
            cfg_kwargs.pop("init_lora_weights", None)
            cfg = LoraConfig(**cfg_kwargs)

        model = get_peft_model(model, cfg)

    elif peft.method == "adalora":
        init_r = peft.init_r if peft.init_r is not None else peft.r
        target_r = peft.target_r if peft.target_r is not None else init_r

        if peft.total_step is None:
            raise ValueError("AdaLoRA needs peft.total_step (total optimizer steps).")

        cfg_kwargs = dict(
            task_type=TaskType.CAUSAL_LM,
            init_r=init_r,
            target_r=target_r,
            tinit=peft.tinit,
            tfinal=peft.tfinal,
            deltaT=peft.deltaT,
            beta1=peft.beta1,
            beta2=peft.beta2,
            orth_reg_weight=peft.orth_reg_weight,
            total_step=int(peft.total_step),
            lora_alpha=effective_alpha,
            lora_dropout=peft.dropout,
            bias=peft.bias,
            target_modules=target_modules,
            use_rslora=peft.use_rslora,
            use_dora=peft.use_dora,
            init_lora_weights=peft.init_lora_weights,
        )

        try:
            cfg = AdaLoraConfig(**cfg_kwargs)
        except TypeError as e:
            if peft.use_dora:
                raise TypeError(
                    "Current peft version does not support use_dora in AdaLoraConfig. "
                    "Please upgrade peft."
                ) from e

            # Compatibility fallback for some older PEFT versions.
            cfg_kwargs.pop("use_rslora", None)
            cfg_kwargs.pop("init_lora_weights", None)
            cfg = AdaLoraConfig(**cfg_kwargs)

        model = get_peft_model(model, cfg)

    else:
        raise ValueError(f"Unknown peft.method: {peft.method}")

    # (G) Train only the embedding rows for newly added tokens.
    if train_embeddings_for_new_tokens:
        if tokenizer is None or old_vocab_size is None or new_vocab_size is None:
            raise ValueError("train_embeddings_for_new_tokens=True requires tokenizer and resize path.")
        if new_vocab_size > old_vocab_size:
            def _zero_grad_on_old_rows(grad):
                grad[:old_vocab_size].zero_()
                return grad

            emb = model.get_input_embeddings()
            emb.weight.requires_grad = True
            emb.weight.register_hook(_zero_grad_on_old_rows)

            out_emb = model.get_output_embeddings()
            if out_emb is not None and out_emb is not emb:
                out_emb.weight.requires_grad = True
                out_emb.weight.register_hook(_zero_grad_on_old_rows)

    return model


class AdaLoraRankAllocatorCallback(TrainerCallback):
    def __init__(self, model=None):
        self._model = model
        self._last_called_step = None

    def _maybe_update(self, args, state, control, model=None, **kwargs):
        model = model or kwargs.get("model") or self._model
        if model is None:
            return control

        if self._last_called_step == state.global_step:
            return control

        base = getattr(model, "base_model", None)
        if base is None or not hasattr(base, "update_and_allocate"):
            return control

        has_grad = False
        for n, p in model.named_parameters():
            if p.requires_grad and ("lora_" in n or "adalora" in n):
                if p.grad is not None:
                    has_grad = True
                    break

        if not has_grad:
            return control

        base.update_and_allocate(state.global_step)
        self._last_called_step = state.global_step
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return self._maybe_update(args, state, control, **kwargs)

    def on_optimizer_step(self, args, state, control, **kwargs):
        return self._maybe_update(args, state, control, **kwargs)


# =========================
# 3) minimal usage
# =========================
def build_base_model(
    model_name_or_path: str,
    *,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    qlora_bits: Optional[int] = None,   # None | 4 | 8
    bnb_compute_dtype: Optional[torch.dtype] = None,
):
    quantization_config = None
    if qlora_bits is not None:
        if qlora_bits not in (4, 8):
            raise ValueError("qlora_bits must be None / 4 / 8")

        if bnb_compute_dtype is None:
            bnb_compute_dtype = (
                torch_dtype if torch_dtype in (torch.float16, torch.bfloat16)
                else torch.float16
            )

        if qlora_bits == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=bnb_compute_dtype,
            )
        else:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype if quantization_config is None else None,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=False,
    )
