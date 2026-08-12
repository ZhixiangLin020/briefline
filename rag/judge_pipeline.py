# -*- coding: utf-8 -*-
"""Clean standalone unified Guardian judge pipeline for Colab.

This file is self-contained. It includes the required highlight-judge helpers,
category/keyword-judge helpers, a unified two-stage scheduler, PostgreSQL input
for category/keyword judging, and PostgreSQL storage for confirmed results.

No __main__ block is included. Run the file in Colab, then call:

    judge_llm, judge_tokenizer = load_unified_judge_model_for_reuse()
    unified_df = run_unified_guardian_judge_with_loaded_model(
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
    )
    close_judge_vllm_model(judge_llm=judge_llm, judge_tokenizer=judge_tokenizer)
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:
    raise ImportError(
        "psycopg is required. Install the dependencies declared in "
        "requirements-rag.txt before running the judge pipeline."
    ) from exc


# ======================================================================================
# User-editable config
# ======================================================================================

# ----- Input and database settings -----
JUDGE_MODEL_PATH = "Qwen/Qwen3-14B"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ARTIFACT_DIR = Path(
    os.environ.get("RAG_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts" / "rag"))
)
RETRIEVAL_PACKETS_JSONL_PATH = str(
    RAG_ARTIFACT_DIR / "retrieval" / "guardian_retrieval_packets.jsonl"
)
RAW_ARTICLES_TABLE = "raw_articles"
MODEL_OUTPUTS_TABLE = "model_outputs"
JUDGE_RESULTS_TABLE = "judge_results"

# ----- Optional limits for testing. None = process all. -----
MAX_RETRIEVAL_PACKETS = None
MAX_CATEGORY_KEYWORD_SAMPLES = None

# ----- vLLM / model settings -----
JUDGE_MAX_MODEL_LEN = 16384
JUDGE_GPU_MEMORY_UTILIZATION = 0.92
JUDGE_TENSOR_PARALLEL_SIZE = 1
JUDGE_DTYPE = "auto"
JUDGE_ENFORCE_EAGER = True
JUDGE_DISABLE_LOG_STATS = True
JUDGE_ENABLE_PREFIX_CACHING = True

# Optional micro-batching. None = submit all prompts to vLLM at once.
JUDGE_VLLM_PROMPT_BATCH_SIZE = None

# ----- Unified judge generation settings -----
UNIFIED_STAGE1_ENABLE_THINKING = True
UNIFIED_STAGE2_ENABLE_THINKING = True
UNIFIED_STAGE1_MAX_TOKENS = 2048
UNIFIED_STAGE2_MAX_TOKENS = 4096
UNIFIED_TEMPERATURE = 0.0
UNIFIED_TOP_P = 1.0

# Stage-specific repetition control.
# Stage 1 is a short PASS/VERIFY/router task, so keep repetition control neutral.
# Stage 2 is the correction task and is more likely to overthink/revisit the same entities.
UNIFIED_STAGE1_REPETITION_PENALTY = 1.00
UNIFIED_STAGE1_FREQUENCY_PENALTY = 0.00
UNIFIED_STAGE2_REPETITION_PENALTY = 1.05
UNIFIED_STAGE2_FREQUENCY_PENALTY = 0.2

# Backward-compatible aliases for notebooks that may still inspect these names.
UNIFIED_REPETITION_PENALTY = UNIFIED_STAGE2_REPETITION_PENALTY
UNIFIED_FREQUENCY_PENALTY = UNIFIED_STAGE2_FREQUENCY_PENALTY

# ----- Task settings -----
JUDGE_EVIDENCE_TOP_N_PER_CLAIM = 5
CK_STAGE1_ARTICLE_FRACTION = 0.5
CK_STAGE1_EVIDENCE_SOURCE = "first_half_article_direct"

# If available in the notebook, this can be overridden before running the file.
GUARDIAN_ALLOWED_SECTIONS = globals().get("GUARDIAN_ALLOWED_SECTIONS", [])

# ----- Output prefixes used by parsers -----
JUDGE_ROUND2_ANSWER_PREFIX = "The correct answer is:"
CATEGORY_CORRECTION_PREFIX = "The correct category is:"
KEYWORD_CORRECTION_PREFIX = "The correct keywords are:"

# ----- Prompt texts. Keep unchanged unless intentionally revising judge behavior. -----
JUDGE_ROUND1_SYSTEM_TEXT = """You are a factual judge.

Use only the retrieved evidence.

Think carefully if thinking is enabled.
After any reasoning, output exactly one final label:

PASS

or

VERIFY

PASS = every factual claim in the generated answer is fully supported by the retrieved evidence.
VERIFY = any factual claim is unsupported, unclear, incomplete, or may need minimal revision.

Do not rewrite the answer in Round 1.
When unsure, output VERIFY."""

JUDGE_ROUND2_SYSTEM_TEXT ="""You are a factual revision judge continuing a prior verification conversation.
Use the full article as the source of truth.
Based on the full article, output the correct final answer.
Make only the minimum necessary change.
If a detail in the original answer is not supported by the article, remove or generalize that detail instead of reasoning about it repeatedly.
Assume the current year is 2026 when the article does not specify a year.
Current-year keyword guidance:
- "2026 World Cup", "2026 FIFA World Cup", "FIFA World Cup 2026", and similar references refer to the 2026 FIFA World Cup.
- "2026 U.S. midterm elections", "2026 US midterms", and similar references refer to the United States midterm elections in 2026.
- Do not reinterpret these keywords as events from earlier years merely because the article contains outdated predictions, previews, or historical comparisons.

Do not explain.
Output only the final answer.
Output format:
The correct answer is: <final answer>"""

CATEGORY_KEYWORD_STAGE1_SYSTEM_TEXT = """You are a factual consistency judge.

Check whether the news category and keywords are factually wrong.

If both category and keywords are semantically compatible with the article, output exactly:
PASS
DO NOT output anything else.

If further checking is needed, output exactly one of:
VERIFY
Route: category

VERIFY
Route: keywords

VERIFY
Route: both

Rules:
- PASS means no correction is needed.
- VERIFY means Stage 2 must check or correct the selected route.
- Only VERIFY when something is clearly factually wrong.
- Do not VERIFY because a category is broad, redundant, or less specific.
- Do not VERIFY to normalize taxonomy labels.
- A broad category is acceptable if it does not contradict the article.
- A broad keyword is acceptable if it refers to the correct topic, entity, event, place, or person.
- Output no explanation."""

CATEGORY_CORRECTION_SYSTEM_TEXT = """You are a category correction judge.

Use the full article as the source of truth.
Choose the single best category from the allowed category list.
Do not explain.

Output format:
The correct category is: <category>"""

KEYWORD_CORRECTION_SYSTEM_TEXT = """You are a keyword correction judge.

Use the full article as the source of truth.
Correct only keywords that clearly refer to the wrong entity, event, year, place, or person.
Assume the current year is 2026 when the article does not specify a year.
Current-year keyword guidance:
- "2026 World Cup", "2026 FIFA World Cup", "FIFA World Cup 2026", and similar references refer to the 2026 FIFA World Cup.
- "2026 U.S. midterm elections", "2026 US midterms", and similar references refer to the United States midterm elections in 2026.
- Do not reinterpret these keywords as events from earlier years merely because the article contains outdated predictions, previews, or historical comparisons.
Keep all correct keywords unchanged.
When there is ambiguity, choose the most article-supported keyword set and still output the final corrected keywords.
Do not explain.

Output format:
The correct keywords are: <comma-separated keywords>"""

COMBINED_CATEGORY_KEYWORD_CORRECTION_SYSTEM_TEXT = """You are a category and keyword correction judge.

Use the full article as the source of truth.
Choose the single best category from the allowed category list.
Correct only keywords that clearly refer to the wrong entity, event, year, place, or person.
Assume the current year is 2026 when the article does not specify a year.
Current-year keyword guidance:
- "2026 World Cup", "2026 FIFA World Cup", "FIFA World Cup 2026", and similar references refer to the 2026 FIFA World Cup.
- "2026 U.S. midterm elections", "2026 US midterms", and similar references refer to the United States midterm elections in 2026.
- Do not reinterpret these keywords as events from earlier years merely because the article contains outdated predictions, previews, or historical comparisons.
Keep all correct keywords unchanged.
Do not explain.

Output format:
The correct category is: <category>
The correct keywords are: <comma-separated keywords>"""

# ----- Unified job ordering for prefix-cache-friendly batching -----
UNIFIED_STAGE1_JOB_ORDER = {
    "highlight_round1": 0,
    "ck_stage1": 1,
}

UNIFIED_STAGE2_JOB_ORDER = {
    "highlight_round2": 0,
    "ck_stage2_category": 1,
    "ck_stage2_keywords": 2,
    "ck_stage2_both": 3,
}
GUARDIAN_ALLOWED_SECTIONS = (
    # score 5
    "australia-news",
    "business",
    "environment",
    "football",
    "media",
    "money",
    "politics",
    "science",
    "sport",
    "technology",
    "us-news",
    "world",

    # score 4
    "better-business",
    "business-to-business",
    "enterprise-network",
    "global-development",
    "small-business-network",

    # score 3
    "global-development-professionals-network",
    "government-computing-network",
    "inequality",
    "katine",
    "law",
    "local-government-network",
    "media-network",
    "news",
    "public-leaders-network",
    "social-enterprise-network",
    "tv-and-radio",
    "uk-news",
    "women-in-leadership",
    "working-in-development",

    # score 2
    "animals-farmed",
    "cardiff",
    "cities",
    "edinburgh",
    "education",
    "housing-network",
    "jobsadvice",
    "leeds",
    "society",
    "weather",
)
# ----- Unified in-memory result schema. No token fields by design. -----
UNIFIED_OUTPUT_COLUMNS = [
    "article_id",
    "source_id",
    "title",
    "packet_index",
    "highlight_model_input_rounds",
    "highlight_model_input",
    "ck_model_input_rounds",
    "ck_model_input",
    "generated_highlight",
    "final_highlight",
    "highlight_changed",
    "highlight_stage1_status",
    "highlight_entered_stage2",
    "highlight_stage2_parse_ok",
    "highlight_stage2_parse_note",
    "original_category",
    "final_category",
    "category_changed",
    "original_keywords",
    "final_keywords",
    "keywords_changed",
    "ck_stage1_status",
    "ck_stage1_route",
    "ck_stage1_parse_ok",
    "ck_stage1_parse_note",
    "ck_entered_stage2",
    "ck_stage2_parse_ok",
    "ck_stage2_parse_note",
    "any_changed",
    "any_parse_failed",
    "final_quality_status",
]


# ======================================================================================
# Generic helpers
# ======================================================================================

@contextlib.contextmanager
def _noop_suppress_stdout():
    yield


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def stringify_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        return ", ".join(str(v).strip() for v in x if str(v).strip()).strip()
    return str(x).strip()


def bool_value(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return bool(x)


def changed_value(original: Any, final: Any) -> bool:
    return safe_str(original).strip() != safe_str(final).strip()


def normalize_highlight_for_change(x: Any) -> str:
    """Normalize highlight text only for change detection.

    This intentionally ignores a leading "highlight:" label because that is a
    formatting prefix, not a factual/content revision. The returned text is
    used only for comparison; the saved final_highlight text is not modified.
    """
    text = safe_str(x).strip()
    text = re.sub(
        r"^\s*highlight\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text


def highlight_changed_value(original: Any, final: Any) -> bool:
    return normalize_highlight_for_change(original) != normalize_highlight_for_change(final)



def clean_vllm_text(text: str) -> str:
    text = "" if text is None else str(text)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>", "<|im_start|>"]:
        text = text.replace(tok, "")
    return text.strip()


def clean_judge_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return clean_vllm_text(text).strip()


def extract_input_ids(x: Any) -> List[int]:
    if isinstance(x, dict):
        return list(x["input_ids"])
    if hasattr(x, "data") and isinstance(getattr(x, "data"), dict) and "input_ids" in x.data:
        return list(x.data["input_ids"])
    return list(x)


def messages_to_debug_text(messages: List[Dict[str, str]]) -> str:
    blocks = []
    for message in messages:
        role = safe_str(message.get("role", "")).upper()
        content = safe_str(message.get("content", ""))
        blocks.append(f"[{role}]\n{content}")
    return "\n\n".join(blocks)


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# Backward-compatible aliases for notebooks that expect the old helper names.
_safe_str = safe_str
_stringify_text = stringify_text


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


def get_database_url() -> str:
    database_url = get_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from Colab Secrets.")
    return database_url


_POSTGRES_JUDGE_TABLE_INITIALIZED = False


def initialize_judge_results_table() -> None:
    global _POSTGRES_JUDGE_TABLE_INITIALIZED

    if _POSTGRES_JUDGE_TABLE_INITIALIZED:
        return

    statement = f"""
    CREATE TABLE IF NOT EXISTS {JUDGE_RESULTS_TABLE} (
        source_id TEXT PRIMARY KEY REFERENCES {RAW_ARTICLES_TABLE}(source_id) ON DELETE CASCADE,
        final_highlight TEXT NOT NULL,
        final_category TEXT NOT NULL,
        final_keywords TEXT NOT NULL,
        highlight_changed BOOLEAN NOT NULL,
        category_changed BOOLEAN NOT NULL,
        keywords_changed BOOLEAN NOT NULL,
        any_changed BOOLEAN NOT NULL,
        highlight_stage1_status TEXT NOT NULL,
        highlight_entered_stage2 BOOLEAN NOT NULL,
        highlight_stage2_parse_ok BOOLEAN NOT NULL,
        highlight_stage2_parse_note TEXT NOT NULL,
        ck_stage1_status TEXT NOT NULL,
        ck_stage1_route TEXT NOT NULL,
        ck_stage1_parse_ok BOOLEAN NOT NULL,
        ck_stage1_parse_note TEXT NOT NULL,
        ck_entered_stage2 BOOLEAN NOT NULL,
        ck_stage2_parse_ok BOOLEAN NOT NULL,
        ck_stage2_parse_note TEXT NOT NULL,
        any_parse_failed BOOLEAN NOT NULL,
        final_quality_status TEXT NOT NULL
    );
    """

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(statement)

    _POSTGRES_JUDGE_TABLE_INITIALIZED = True
    print("PostgreSQL judge results table is ready.")


# ======================================================================================
# Model lifecycle
# ======================================================================================

def patch_vllm_stdout_for_notebook() -> None:
    """Patch vLLM suppress_stdout for Colab/Jupyter when sys.stdout.fileno() fails."""
    try:
        import vllm.utils.system_utils as vllm_system_utils
        vllm_system_utils.suppress_stdout = _noop_suppress_stdout
        print("patched vllm.utils.system_utils.suppress_stdout")
    except Exception as exc:
        print("system_utils patch skipped:", repr(exc))

    try:
        import vllm.distributed.parallel_state as vllm_parallel_state
        vllm_parallel_state.suppress_stdout = _noop_suppress_stdout
        print("patched vllm.distributed.parallel_state.suppress_stdout")
    except Exception as exc:
        print("parallel_state patch skipped:", repr(exc))


def unload_generation_model_for_judge(
    *,
    global_names_to_delete: Optional[List[str]] = None,
) -> None:
    """Delete existing generation-model globals before loading the judge model."""
    if global_names_to_delete is None:
        global_names_to_delete = ["llm"]

    print("=" * 100)
    print("[UNLOAD GENERATION MODEL]")

    for name in global_names_to_delete:
        if name not in globals():
            continue
        try:
            obj = globals().get(name)
            if obj is not None:
                try:
                    engine = getattr(obj, "llm_engine", None)
                    executor = getattr(engine, "model_executor", None) if engine is not None else None
                    if executor is not None and hasattr(executor, "shutdown"):
                        executor.shutdown()
                        print(f"shutdown model_executor for global: {name}")
                except Exception as exc:
                    print(f"model_executor shutdown skipped for {name}:", repr(exc))
            print(f"delete global: {name}")
            del globals()[name]
        except Exception as exc:
            print(f"[WARN] failed to delete {name}: {repr(exc)}")

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        try:
            destroy_model_parallel()
            print("destroy_model_parallel done")
        except Exception as exc:
            print("destroy_model_parallel skipped:", repr(exc))
    except Exception as exc:
        print("vLLM model-parallel cleanup skipped:", repr(exc))

    cleanup_cuda()

    if torch.cuda.is_available():
        print(
            "[CUDA]",
            "allocated_gb=", round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb=", round(torch.cuda.memory_reserved() / 1024**3, 3),
        )


def load_unified_judge_vllm_model(
    *,
    model_path: str = JUDGE_MODEL_PATH,
    enable_prefix_caching: bool = JUDGE_ENABLE_PREFIX_CACHING,
):
    """Load the judge model once, with prefix caching enabled when supported."""
    print("=" * 100)
    print("[LOAD UNIFIED JUDGE MODEL]")
    print("model_path:", model_path)
    print("enable_prefix_caching:", bool(enable_prefix_caching))

    patch_vllm_stdout_for_notebook()

    judge_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    kwargs = {
        "model": model_path,
        "tokenizer": model_path,
        "trust_remote_code": True,
        "dtype": JUDGE_DTYPE,
        "tensor_parallel_size": JUDGE_TENSOR_PARALLEL_SIZE,
        "gpu_memory_utilization": JUDGE_GPU_MEMORY_UTILIZATION,
        "max_model_len": JUDGE_MAX_MODEL_LEN,
        "enforce_eager": JUDGE_ENFORCE_EAGER,
        "disable_log_stats": JUDGE_DISABLE_LOG_STATS,
    }
    if enable_prefix_caching:
        kwargs["enable_prefix_caching"] = True

    try:
        judge_llm = LLM(**kwargs)
    except TypeError as exc:
        if "enable_prefix_caching" not in kwargs:
            raise
        print("[WARN] vLLM did not accept enable_prefix_caching; retrying without it:", repr(exc))
        kwargs.pop("enable_prefix_caching", None)
        judge_llm = LLM(**kwargs)

    return judge_llm, judge_tokenizer


def load_unified_judge_model_for_reuse(
    *,
    delete_generation_globals: Optional[List[str]] = None,
    unload_existing_generation_model: bool = True,
    model_path: str = JUDGE_MODEL_PATH,
    enable_prefix_caching: bool = JUDGE_ENABLE_PREFIX_CACHING,
):
    """Load the judge model for repeated Colab calls."""
    if unload_existing_generation_model:
        unload_generation_model_for_judge(global_names_to_delete=delete_generation_globals or ["llm"])
    else:
        cleanup_cuda()

    return load_unified_judge_vllm_model(
        model_path=model_path,
        enable_prefix_caching=enable_prefix_caching,
    )


def close_judge_vllm_model(*, judge_llm=None, judge_tokenizer=None) -> None:
    print("=" * 100)
    print("[UNLOAD JUDGE MODEL]")

    try:
        if judge_llm is not None:
            engine = getattr(judge_llm, "llm_engine", None)
            executor = getattr(engine, "model_executor", None) if engine is not None else None
            if executor is not None and hasattr(executor, "shutdown"):
                executor.shutdown()
                print("judge model_executor shutdown done")
    except Exception as exc:
        print("judge model_executor shutdown skipped:", repr(exc))

    try:
        del judge_llm
    except Exception:
        pass
    try:
        del judge_tokenizer
    except Exception:
        pass

    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        try:
            destroy_model_parallel()
            print("destroy_model_parallel done")
        except Exception as exc:
            print("destroy_model_parallel skipped:", repr(exc))
    except Exception as exc:
        print("vLLM model-parallel cleanup skipped:", repr(exc))

    cleanup_cuda()


# Backward-compatible alias.
load_judge_model_for_reuse = load_unified_judge_model_for_reuse


# ======================================================================================
# Chat template and vLLM batch generation
# ======================================================================================

def apply_chat_messages_template(
    tokenizer,
    *,
    messages: List[Dict[str, str]],
    enable_thinking: bool,
) -> Dict[str, Any]:
    clean_messages = []
    for message in messages:
        role = safe_str(message.get("role", ""))
        content = stringify_text(message.get("content", ""))
        if not role:
            raise ValueError(f"Message without role: {message!r}")
        clean_messages.append({"role": role, "content": content})

    try:
        prompt_ids = extract_input_ids(
            tokenizer.apply_chat_template(
                clean_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
        )
    except TypeError:
        prompt_ids = extract_input_ids(
            tokenizer.apply_chat_template(
                clean_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    return {
        "messages": clean_messages,
        "prompt_ids": prompt_ids,
        "messages_debug_text": messages_to_debug_text(clean_messages),
    }


def decode_one_completion(tokenizer, output) -> Dict[str, Any]:
    completion = output.outputs[0]
    token_ids = list(completion.token_ids)
    raw_text = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return {
        "raw_text": raw_text,
        "clean_text": clean_judge_text(raw_text),
    }


def judge_generate_messages_batch(
    *,
    judge_llm,
    judge_tokenizer,
    messages_list: List[List[Dict[str, str]]],
    max_tokens: int,
    enable_thinking: bool,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    prompt_batch_size: Optional[int] = JUDGE_VLLM_PROMPT_BATCH_SIZE,
    use_tqdm: bool = True,
    desc: str = "judge batch generate",
) -> List[Dict[str, Any]]:
    """Batch generation from full chat message lists. No token fields are returned."""
    if not messages_list:
        return []

    built_items = [
        apply_chat_messages_template(
            judge_tokenizer,
            messages=messages,
            enable_thinking=enable_thinking,
        )
        for messages in messages_list
    ]

    prompts = [{"prompt_token_ids": item["prompt_ids"]} for item in built_items]

    sampling_params = SamplingParams(
        n=1,
        temperature=float(UNIFIED_TEMPERATURE),
        top_p=float(UNIFIED_TOP_P),
        max_tokens=int(max_tokens),
        repetition_penalty=float(repetition_penalty),
        frequency_penalty=float(frequency_penalty),
        stop=["<|im_end|>", "<|endoftext|>", "<|end|>"],
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )

    if prompt_batch_size is None or int(prompt_batch_size) <= 0:
        prompt_batch_size = len(prompts)
    prompt_batch_size = max(1, int(prompt_batch_size))

    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    ranges = list(range(0, len(prompts), prompt_batch_size))
    iterator = tqdm(ranges, desc=desc) if use_tqdm and len(ranges) > 1 else ranges

    for start in iterator:
        end = min(start + prompt_batch_size, len(prompts))
        batch_outputs = judge_llm.generate(
            prompts=prompts[start:end],
            sampling_params=sampling_params,
            use_tqdm=bool(use_tqdm and len(ranges) == 1),
        )

        if len(batch_outputs) != (end - start):
            raise RuntimeError(
                f"vLLM output count mismatch: got {len(batch_outputs)}, expected {end - start}."
            )

        for offset, output in enumerate(batch_outputs):
            i = start + offset
            decoded = decode_one_completion(judge_tokenizer, output)
            decoded["messages_debug_text"] = built_items[i]["messages_debug_text"]
            results[i] = decoded

    final_results: List[Dict[str, Any]] = []
    for i, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing judge output at index {i}.")
        final_results.append(result)
    return final_results


def build_model_full_context_text(
    *,
    stage_label: str,
    messages_debug_text: str,
    raw_output: str,
) -> str:
    """Build human-readable context for the existing *_model_input fields.

    It contains the exact chat messages sent to the model plus the raw assistant
    output returned by vLLM for that stage.
    """
    messages_debug_text = safe_str(messages_debug_text).strip()
    raw_output = safe_str(raw_output).strip()
    return (
        f"===== {safe_str(stage_label).strip()} =====\n"
        f"{messages_debug_text}\n\n"
        f"[ASSISTANT_OUTPUT]\n{raw_output}"
    ).strip()


def append_model_full_context_round(existing_context: str, new_round_context: str) -> str:
    existing_context = safe_str(existing_context).strip()
    new_round_context = safe_str(new_round_context).strip()
    if not existing_context:
        return new_round_context
    if not new_round_context:
        return existing_context
    return f"{existing_context}\n\n{'=' * 100}\n\n{new_round_context}".strip()


# ======================================================================================
# Highlight judge helpers
# ======================================================================================

def validate_retrieval_packet_for_judge(packet: Dict[str, Any]) -> None:
    required_keys = ["original_article", "generated_highlight", "claims"]
    missing = [k for k in required_keys if k not in packet or packet.get(k) in [None, ""]]
    if missing:
        raise ValueError(f"Saved retrieval packet missing required keys: {missing}")

    if not isinstance(packet.get("claims"), list):
        raise ValueError("Saved retrieval packet field 'claims' must be a list.")

    for i, claim in enumerate(packet["claims"]):
        if "claim_sentence" not in claim:
            raise ValueError(f"claims[{i}] missing claim_sentence")
        if "evidence_candidates" not in claim:
            raise ValueError(f"claims[{i}] missing evidence_candidates")


def load_saved_retrieval_packets_jsonl(
    *,
    packets_jsonl_path: str = RETRIEVAL_PACKETS_JSONL_PATH,
    max_packets: Optional[int] = MAX_RETRIEVAL_PACKETS,
    source_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    packets_jsonl_path = str(packets_jsonl_path)
    if not os.path.exists(packets_jsonl_path):
        raise FileNotFoundError(f"Packets JSONL not found: {packets_jsonl_path}")

    print("=" * 100)
    print("[LOAD SAVED RETRIEVAL PACKETS JSONL]")
    print("packets_jsonl_path:", packets_jsonl_path)

    allowed_source_ids = None
    if source_ids is not None:
        allowed_source_ids = {safe_str(value).strip() for value in source_ids}
        allowed_source_ids.discard("")
        if not allowed_source_ids:
            return []

    packets: List[Dict[str, Any]] = []
    with open(packets_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                packet = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {packets_jsonl_path}: {repr(exc)}")
            if not isinstance(packet, dict):
                raise ValueError(f"Packet at line {line_no} must be a JSON object.")

            packet_source_id = safe_str(
                packet.get("source_id") or packet.get("article_id")
            ).strip()
            if allowed_source_ids is not None and packet_source_id not in allowed_source_ids:
                continue

            packet["_packet_index"] = len(packets)
            validate_retrieval_packet_for_judge(packet)
            packets.append(packet)

            if max_packets is not None and len(packets) >= int(max_packets):
                break

    if not packets:
        raise ValueError(f"No packets loaded from {packets_jsonl_path}")

    print("loaded packets:", len(packets))
    return packets


def format_retrieved_evidence_from_packet(
    packet: Dict[str, Any],
    *,
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
    include_score: bool = False,
) -> str:
    claims = packet.get("claims", []) or []
    blocks = []

    for claim in claims:
        claim_idx = claim.get("claim_idx", "")
        claim_sentence = safe_str(claim.get("claim_sentence", ""))
        blocks.append(f"[Claim {claim_idx}]\n{claim_sentence}")

        evidence_candidates = claim.get("evidence_candidates", []) or []
        evidence_candidates = evidence_candidates[: int(top_n_per_claim)]

        if not evidence_candidates:
            blocks.append("[Retrieved evidence]\nNo retrieved evidence.")
            continue

        for ev in evidence_candidates:
            rank = ev.get("rank", "")
            chunk_text = safe_str(ev.get("chunk_text", ""))
            if include_score:
                score = ev.get("score", "")
                blocks.append(f"[Evidence {claim_idx}.{rank} | score={score}]\n{chunk_text}")
            else:
                blocks.append(f"[Evidence {claim_idx}.{rank}]\n{chunk_text}")

    return "\n\n".join(blocks).strip()


def build_judge_round1_user_text(
    packet: Dict[str, Any],
    *,
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
) -> str:
    generated_answer = safe_str(packet.get("generated_highlight", ""))
    retrieved_evidence = format_retrieved_evidence_from_packet(
        packet,
        top_n_per_claim=top_n_per_claim,
        include_score=False,
    )
    return f"""Generated answer:
{generated_answer}

Retrieved evidence:
{retrieved_evidence}

Decision:"""


def normalize_round1_status(text: str) -> str:
    clean = clean_judge_text(text).upper()
    clean = clean.replace("`", " ")
    clean = re.sub(r"[^A-Z]+", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    tokens = clean.split()

    if "VERIFY" in tokens:
        return "VERIFY"
    if "PASS" in tokens:
        return "PASS"
    if "VERIFY" in clean:
        return "VERIFY"
    if "PASS" in clean:
        return "PASS"
    return "VERIFY"


def build_round1_assistant_message_for_round2(round1_result: Dict[str, Any]) -> str:
    raw_text = safe_str(round1_result.get("round1_model_output", "")).strip()
    if raw_text:
        return raw_text
    status = safe_str(round1_result.get("round1_status", "VERIFY")).upper().strip()
    if status in {"PASS", "VERIFY"}:
        return status
    return "VERIFY"


def build_judge_round2_followup_user_text(packet: Dict[str, Any]) -> str:
    full_article_text = safe_str(packet.get("original_article", ""))
    generated_answer = safe_str(packet.get("generated_highlight", ""))
    return f"""Now produce the correct final answer using the full article.

Full article:
{full_article_text}

Original answer:
{generated_answer}"""


def build_judge_round2_messages(
    packet: Dict[str, Any],
    round1_result: Dict[str, Any],
    *,
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
) -> List[Dict[str, str]]:
    round1_user_text = build_judge_round1_user_text(packet, top_n_per_claim=top_n_per_claim)
    round1_assistant_text = build_round1_assistant_message_for_round2(round1_result)
    round2_user_text = build_judge_round2_followup_user_text(packet)
    return [
        {"role": "system", "content": JUDGE_ROUND2_SYSTEM_TEXT},
        {"role": "user", "content": round1_user_text},
        {"role": "assistant", "content": round1_assistant_text},
        {"role": "user", "content": round2_user_text},
    ]


def extract_round2_final_answer(text: str) -> Tuple[str, bool, str]:
    raw = clean_vllm_text(text).strip()
    if not raw:
        return "", False, "empty_output"

    raw_lower = raw.lower()
    if "<think>" in raw_lower and "</think>" not in raw_lower:
        return "", False, "unclosed_think_block_likely_truncated"

    clean = clean_judge_text(raw).strip()
    if not clean:
        return "", False, "empty_after_removing_think"

    pattern = r"^\s*" + re.escape(JUDGE_ROUND2_ANSWER_PREFIX) + r"\s*"
    without_prefix = re.sub(pattern, "", clean, count=1, flags=re.IGNORECASE).strip()
    if without_prefix != clean:
        return without_prefix, True, "prefix_at_start"

    m = re.search(re.escape(JUDGE_ROUND2_ANSWER_PREFIX), clean, flags=re.IGNORECASE)
    if m:
        return clean[m.end():].strip(), True, "prefix_found_later"

    if clean.lower().startswith("highlight:"):
        return clean, True, "missing_prefix_but_starts_with_highlight"

    return "", False, "missing_required_prefix"


# ======================================================================================
# Category/keyword helpers
# ======================================================================================

def ck_split_items(text: str) -> List[str]:
    text = safe_str(text)
    if not text:
        return []
    parts = re.split(r"[,;\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def ck_join_items(items: List[str]) -> str:
    return ", ".join([safe_str(x).strip() for x in items if safe_str(x).strip()])



def flatten_allowed_categories(allowed: Any = None) -> List[str]:
    if allowed is None:
        allowed = GUARDIAN_ALLOWED_SECTIONS

    items: List[str] = []

    if isinstance(allowed, dict):
        for _score, categories in allowed.items():
            if isinstance(categories, (list, tuple, set)):
                items.extend(
                    str(category).strip()
                    for category in categories
                    if str(category).strip()
                )
            elif categories is not None:
                value = str(categories).strip()
                if value:
                    items.append(value)

    elif isinstance(allowed, (list, tuple, set)):
        items.extend(
            str(category).strip()
            for category in allowed
            if str(category).strip()
        )

    elif allowed is not None:
        value = str(allowed).strip()
        if value:
            items.append(value)

    result = sorted(dict.fromkeys(items))

    if not result:
        raise ValueError("GUARDIAN_ALLOWED_SECTIONS must not be empty")

    return result


def format_allowed_categories_for_prompt(allowed_categories: List[str]) -> str:
    return "\n".join(f"- {cat}" for cat in allowed_categories)


def extract_generated_category_keywords(both_output_text: str) -> Dict[str, Any]:
    text = clean_vllm_text(both_output_text)

    cat_text = ""
    kw_text = ""

    m_cat = re.search(
        r"categories\s*:\s*(.*?)(?:\n\s*keywords\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_cat:
        cat_text = m_cat.group(1).strip()

    m_kw = re.search(r"keywords\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
    if m_kw:
        kw_text = m_kw.group(1).strip()

    categories = ck_split_items(cat_text)
    keywords = ck_split_items(kw_text)

    return {
        "generated_category_text": ck_join_items(categories) if categories else cat_text,
        "generated_categories": categories,
        "generated_keyword_text": ck_join_items(keywords) if keywords else kw_text,
        "generated_keywords": keywords,
    }


def first_fraction_article_text(article_text: str, fraction: Optional[float] = None) -> str:
    article_text = safe_str(article_text).strip()
    if not article_text:
        return ""
    if fraction is None:
        fraction = float(CK_STAGE1_ARTICLE_FRACTION)
    fraction = max(0.01, min(1.0, float(fraction)))
    end = max(1, int(math.ceil(len(article_text) * fraction)))
    return article_text[:end].strip()


def first_half_article_text(article_text: str) -> str:
    return first_fraction_article_text(article_text, fraction=float(CK_STAGE1_ARTICLE_FRACTION))


def load_category_keyword_samples_from_postgres(
    *,
    max_samples: Optional[int] = MAX_CATEGORY_KEYWORD_SAMPLES,
    source_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    source_filter = ""
    params: List[Any] = []
    if source_ids is not None:
        cleaned_source_ids = [
            safe_str(source_id).strip()
            for source_id in source_ids
            if safe_str(source_id).strip()
        ]
        if not cleaned_source_ids:
            return []
        source_filter = "AND output.source_id = ANY(%s)"
        params.append(cleaned_source_ids)

    query = f"""
    SELECT
        output.source_id,
        output.published_at AS published_at,
        article.title,
        article.body_text,
        article.summary,
        output.generated_clean,
        output.generated_raw
    FROM {MODEL_OUTPUTS_TABLE} AS output
    INNER JOIN {RAW_ARTICLES_TABLE} AS article
        ON article.source_id = output.source_id
    WHERE output.task = 'both'
    {source_filter}
    ORDER BY output.published_at DESC NULLS LAST, output.source_id
    """

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    samples: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for row in rows:
        row_dict = dict(row)
        source_id = safe_str(row_dict.get("source_id", "")).strip()
        published_at = safe_str(row_dict.get("published_at", "")).strip()
        title = safe_str(row_dict.get("title", "")).strip()
        both_output = safe_str(
            row_dict.get("generated_clean", "")
            or row_dict.get("generated_raw", "")
        ).strip()
        parsed = extract_generated_category_keywords(both_output)

        original_article = safe_str(row_dict.get("body_text", "")).strip()
        if not original_article:
            original_article = safe_str(row_dict.get("summary", "")).strip()

        if not source_id:
            skipped_rows.append({
                "source_id": source_id,
                "title": title,
                "reason": "empty_source_id",
            })
            continue

        if not original_article:
            skipped_rows.append({
                "source_id": source_id,
                "title": title,
                "reason": "empty_original_article",
            })
            continue

        stage1_article_text = first_half_article_text(original_article)
        samples.append({
            "sample_index": len(samples),
            "title": title,
            "source_id": source_id,
            "article_id": source_id,
            "published_at": published_at,
            "original_article": original_article,
            "stage1_article_text": stage1_article_text,
            "ck_stage1_evidence_source": CK_STAGE1_EVIDENCE_SOURCE,
            "generated_both_output": both_output,
            "generated_category": parsed["generated_category_text"],
            "generated_keywords": parsed["generated_keyword_text"],
            "generated_keyword_list": parsed["generated_keywords"],
        })

        if max_samples is not None and len(samples) >= int(max_samples):
            break

    print("=" * 100)
    print("[LOAD CATEGORY/KEYWORD JUDGE SAMPLES FROM POSTGRESQL]")
    print("both rows:", len(rows))
    print("loaded samples:", len(samples))
    print("skipped rows:", len(skipped_rows))
    print("stage1_article_fraction:", CK_STAGE1_ARTICLE_FRACTION)

    if skipped_rows:
        print("[SKIPPED ROWS - FIRST 5]")
        for item in skipped_rows[:5]:
            print(item)

    if not samples:
        raise ValueError(
            "No category/keyword judge samples were built from PostgreSQL."
        )
    return samples


def build_ck_stage1_user_text(sample: Dict[str, Any]) -> str:
    article_half = safe_str(
        sample.get("stage1_article_text", "")
    )

    if not article_half:
        article_half = first_half_article_text(
            sample.get("original_article", "")
        )

    published_at = safe_str(
        sample.get("published_at", "")
    ).strip()

    if not published_at:
        published_at = "Unknown"

    return f"""Title:
{safe_str(sample.get('title', ''))}

Publication time:
{published_at}

Category:
{safe_str(sample.get('generated_category', ''))}

Keywords:
{safe_str(sample.get('generated_keywords', ''))}

First half of article:
{article_half}

Decision:"""


def parse_ck_stage1_output(raw_text: str) -> Dict[str, Any]:
    clean = clean_judge_text(raw_text).strip()
    clean = clean_vllm_text(clean).strip()

    if not clean:
        return {
            "stage1_status": "VERIFY",
            "stage1_route": "both",
            "category_needs_correction": True,
            "keywords_needs_correction": True,
            "stage1_clean_text": "",
            "stage1_parse_ok": False,
            "stage1_parse_note": "empty_output",
        }

    if re.search(r"(^|\n)\s*PASS\b", clean, flags=re.IGNORECASE):
        return {
            "stage1_status": "PASS",
            "stage1_route": "none",
            "category_needs_correction": False,
            "keywords_needs_correction": False,
            "stage1_clean_text": "PASS",
            "stage1_parse_ok": True,
            "stage1_parse_note": "pass_short_circuit",
        }

    if not re.search(r"(^|\n)\s*VERIFY\b", clean, flags=re.IGNORECASE):
        return {
            "stage1_status": "VERIFY",
            "stage1_route": "both",
            "category_needs_correction": True,
            "keywords_needs_correction": True,
            "stage1_clean_text": clean,
            "stage1_parse_ok": False,
            "stage1_parse_note": "missing_pass_or_verify",
        }

    route_match = re.search(
        r"^\s*Route\s*:\s*(category|keywords|both)\b",
        clean,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if route_match:
        route = route_match.group(1).lower()
        parse_ok = True
        parse_note = "verify_with_route"
    else:
        route = "both"
        parse_ok = False
        parse_note = "verify_missing_route_default_both"

    return {
        "stage1_status": "VERIFY",
        "stage1_route": route,
        "category_needs_correction": route in {"category", "both"},
        "keywords_needs_correction": route in {"keywords", "both"},
        "stage1_clean_text": f"VERIFY\nRoute: {route}",
        "stage1_parse_ok": parse_ok,
        "stage1_parse_note": parse_note,
    }


def route_from_stage1_result(stage1_result: Dict[str, Any]) -> str:
    route = safe_str(stage1_result.get("stage1_route", "")).strip().lower()
    if route in {"none", "category", "keywords", "both"}:
        return route

    cat = bool(stage1_result.get("category_needs_correction"))
    kw = bool(stage1_result.get("keywords_needs_correction"))
    if cat and kw:
        return "both"
    if cat:
        return "category"
    if kw:
        return "keywords"
    return "none"


def build_category_correction_user_text(sample: Dict[str, Any], allowed_categories: List[str]) -> str:
    return f"""Allowed categories:
{format_allowed_categories_for_prompt(allowed_categories)}

Title:
{safe_str(sample.get('title', ''))}

Full article:
{safe_str(sample.get('original_article', ''))}

Original category:
{safe_str(sample.get('generated_category', ''))}

The correct category is:"""


def build_keyword_correction_user_text(sample: Dict[str, Any]) -> str:
    return f"""Title:
{safe_str(sample.get('title', ''))}

Full article:
{safe_str(sample.get('original_article', ''))}

Original keywords:
{safe_str(sample.get('generated_keywords', ''))}

The correct keywords are:"""


def build_combined_correction_user_text(sample: Dict[str, Any], allowed_categories: List[str]) -> str:
    return f"""Allowed categories:
{format_allowed_categories_for_prompt(allowed_categories)}

Title:
{safe_str(sample.get('title', ''))}

Full article:
{safe_str(sample.get('original_article', ''))}

Original category:
{safe_str(sample.get('generated_category', ''))}

Original keywords:
{safe_str(sample.get('generated_keywords', ''))}

The correct category is:
The correct keywords are:"""


def build_ck_stage2_messages(
    sample: Dict[str, Any],
    stage1_result: Dict[str, Any],
    *,
    route: str,
    allowed_categories: List[str],
) -> List[Dict[str, str]]:
    stage1_user_text = build_ck_stage1_user_text(sample)

    stage1_assistant_text = safe_str(stage1_result.get("stage1_clean_text", "")).strip()
    if not stage1_assistant_text:
        stage1_assistant_text = "PASS" if route == "none" else f"VERIFY\nRoute: {route}"

    if route == "category":
        system_text = CATEGORY_CORRECTION_SYSTEM_TEXT
        followup = build_category_correction_user_text(sample, allowed_categories)
    elif route == "keywords":
        system_text = KEYWORD_CORRECTION_SYSTEM_TEXT
        followup = build_keyword_correction_user_text(sample)
    elif route == "both":
        system_text = COMBINED_CATEGORY_KEYWORD_CORRECTION_SYSTEM_TEXT
        followup = build_combined_correction_user_text(sample, allowed_categories)
    else:
        raise ValueError(f"Unsupported route: {route}")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": stage1_user_text},
        {"role": "assistant", "content": stage1_assistant_text},
        {"role": "user", "content": followup},
    ]


def extract_prefixed_single_line(text: str, prefix: str) -> Tuple[str, bool, str]:
    raw = clean_vllm_text(text).strip()
    if not raw:
        return "", False, "empty_output"

    raw_lower = raw.lower()
    if "<think>" in raw_lower and "</think>" not in raw_lower:
        return "", False, "unclosed_think_block_likely_truncated"

    clean = clean_judge_text(raw).strip()
    if not clean:
        return "", False, "empty_after_removing_think"

    for line in clean.splitlines():
        line_s = line.strip()
        if line_s.lower().startswith(prefix.lower()):
            value = line_s[len(prefix):].strip()
            return value, bool(value), "prefix_line"

    m = re.search(re.escape(prefix), clean, flags=re.IGNORECASE)
    if m:
        rest = clean[m.end():].strip()
        value = rest.splitlines()[0].strip() if rest else ""
        return value, bool(value), "prefix_found_later"

    return "", False, "missing_required_prefix"


def extract_category_output(text: str, allowed_categories: List[str]) -> Tuple[str, bool, str]:
    allowed_set = set(allowed_categories)
    category, ok, note = extract_prefixed_single_line(text, CATEGORY_CORRECTION_PREFIX)
    category = category.strip().strip("`'\"")

    if ok and category in allowed_set:
        return category, True, note
    if ok and category:
        return category, False, f"{note};category_not_in_allowed_list"

    clean = clean_judge_text(text).strip().strip("`'\"")
    if clean in allowed_set:
        return clean, True, "missing_prefix_but_exact_allowed_category"

    return "", False, note


def extract_keywords_output(
    text: str,
) -> Tuple[str, bool, str]:
    raw = clean_vllm_text(text).strip()

    if not raw:
        return "", False, "empty_output"

    raw_lower = raw.lower()
    if "<think>" in raw_lower and "</think>" not in raw_lower:
        return "", False, "unclosed_think_block_likely_truncated"

    clean = clean_judge_text(raw).strip()

    if not clean:
        return "", False, "empty_after_removing_think"

    # Format 1:
    # The correct keywords are: keyword1, keyword2
    keywords, ok, note = extract_prefixed_single_line(
        clean,
        KEYWORD_CORRECTION_PREFIX,
    )

    normalized = ck_join_items(ck_split_items(keywords))

    if ok and normalized:
        return normalized, True, note

    # Format 2:
    # keyword1, keyword2
    #
    # This happens when the user prompt already ends with
    # "The correct keywords are:" and the model only completes the value.
    non_empty_lines = [
        line.strip()
        for line in clean.splitlines()
        if line.strip()
    ]

    if len(non_empty_lines) == 1:
        bare_keywords = (
            non_empty_lines[0]
            .strip()
            .strip("`")
            .strip()
        )

        normalized = ck_join_items(
            ck_split_items(bare_keywords)
        )

        if normalized:
            return (
                normalized,
                True,
                "missing_prefix_but_bare_keywords",
            )

    return "", False, note


def extract_combined_output(text: str, allowed_categories: List[str]) -> Tuple[str, str, bool, str]:
    category, cat_ok, cat_note = extract_category_output(text, allowed_categories)
    keywords, kw_ok, kw_note = extract_keywords_output(text)
    ok = bool(cat_ok and kw_ok)
    note = f"category={cat_note}; keywords={kw_note}"
    return category, keywords, ok, note


# ======================================================================================
# Unified scheduler state and jobs
# ======================================================================================

@dataclass
class JudgeJob:
    job_id: str
    job_type: str
    stage: int
    item_index: int
    messages: List[Dict[str, str]]
    route: str = ""
    generated: Dict[str, Any] = field(default_factory=dict)


def sort_jobs(jobs: List[JudgeJob], order: Dict[str, int]) -> List[JudgeJob]:
    return sorted(jobs, key=lambda job: (order.get(job.job_type, 999), job.item_index))


def run_unified_jobs_generate(
    *,
    judge_llm,
    judge_tokenizer,
    jobs: List[JudgeJob],
    max_tokens: int,
    enable_thinking: bool,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    prompt_batch_size: Optional[int] = JUDGE_VLLM_PROMPT_BATCH_SIZE,
    desc: str = "unified judge batch",
    use_tqdm: bool = True,
) -> List[JudgeJob]:
    if not jobs:
        return []

    messages_list = [job.messages for job in jobs]
    generated_items = judge_generate_messages_batch(
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
        messages_list=messages_list,
        max_tokens=int(max_tokens),
        enable_thinking=bool(enable_thinking),
        repetition_penalty=float(repetition_penalty),
        frequency_penalty=float(frequency_penalty),
        prompt_batch_size=prompt_batch_size,
        use_tqdm=use_tqdm,
        desc=desc,
    )

    if len(generated_items) != len(jobs):
        raise RuntimeError(f"Unified job output count mismatch: got {len(generated_items)}, expected {len(jobs)}.")

    for job, generated in zip(jobs, generated_items):
        job.generated = generated
    return jobs


def init_highlight_states(packets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for i, packet in enumerate(packets):
        generated_highlight = safe_str(packet.get("generated_highlight", ""))
        states.append({
            "packet_index": int(packet.get("_packet_index", i)),
            "title": safe_str(packet.get("title", "")),
            "source_id": safe_str(packet.get("source_id", "")),
            "article_id": safe_str(packet.get("article_id", "")),
            "generated_highlight": generated_highlight,
            "final_highlight": "",
            "highlight_changed": False,
            "highlight_stage1_status": "",
            "highlight_entered_stage2": False,
            "highlight_stage2_parse_ok": True,
            "highlight_stage2_parse_note": "not_run",
            "highlight_model_input_rounds": 0,
            "highlight_model_input": "",
            "_packet": packet,
            "_round1_result": {},
        })
    return states


def init_ck_states(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    states: List[Dict[str, Any]] = []
    for i, sample in enumerate(samples):
        original_category = safe_str(sample.get("generated_category", ""))
        original_keywords = safe_str(sample.get("generated_keywords", ""))
        states.append({
            "sample_index": sample.get("sample_index", i),
            "title": safe_str(sample.get("title", "")),
            "source_id": safe_str(sample.get("source_id", "")),
            "article_id": safe_str(sample.get("article_id", "")),
            "original_category": original_category,
            "final_category": "",
            "category_changed": False,
            "original_keywords": original_keywords,
            "final_keywords": "",
            "keywords_changed": False,
            "ck_stage1_status": "",
            "ck_stage1_route": "",
            "ck_stage1_parse_ok": True,
            "ck_stage1_parse_note": "not_run",
            "ck_entered_stage2": False,
            "ck_stage2_parse_ok": True,
            "ck_stage2_parse_note": "not_run",
            "ck_model_input_rounds": 0,
            "ck_model_input": "",
            "_sample": sample,
            "_stage1_result": {},
        })
    return states


def build_unified_stage1_jobs(
    *,
    packets: List[Dict[str, Any]],
    ck_samples: List[Dict[str, Any]],
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
) -> List[JudgeJob]:
    jobs: List[JudgeJob] = []

    for i, packet in enumerate(packets):
        messages = [
            {"role": "system", "content": JUDGE_ROUND1_SYSTEM_TEXT},
            {"role": "user", "content": build_judge_round1_user_text(packet, top_n_per_claim=top_n_per_claim)},
        ]
        jobs.append(JudgeJob(
            job_id=f"highlight_round1:{i}",
            job_type="highlight_round1",
            stage=1,
            item_index=i,
            messages=messages,
        ))

    for i, sample in enumerate(ck_samples):
        messages = [
            {"role": "system", "content": CATEGORY_KEYWORD_STAGE1_SYSTEM_TEXT},
            {"role": "user", "content": build_ck_stage1_user_text(sample)},
        ]
        jobs.append(JudgeJob(
            job_id=f"ck_stage1:{i}",
            job_type="ck_stage1",
            stage=1,
            item_index=i,
            messages=messages,
        ))

    return sort_jobs(jobs, UNIFIED_STAGE1_JOB_ORDER)


def parse_unified_stage1_jobs(
    *,
    jobs: List[JudgeJob],
    highlight_states: List[Dict[str, Any]],
    ck_states: List[Dict[str, Any]],
) -> None:
    for job in jobs:
        raw_text = safe_str(job.generated.get("raw_text", ""))
        model_input = safe_str(job.generated.get("messages_debug_text", ""))
        model_context = build_model_full_context_text(
            stage_label=f"Stage {job.stage} - {job.job_type}",
            messages_debug_text=model_input,
            raw_output=raw_text,
        )
        i = int(job.item_index)

        if job.job_type == "highlight_round1":
            status = normalize_round1_status(raw_text)
            round1_result = {
                "round1_status": status,
                "round1_model_output": raw_text,
                "_round1_model_input": model_input,
            }
            state = highlight_states[i]
            state["highlight_stage1_status"] = status
            state["_round1_result"] = round1_result

            if status == "PASS":
                state["highlight_entered_stage2"] = False
                state["highlight_model_input_rounds"] = 1
                state["highlight_model_input"] = model_context
                state["highlight_stage2_parse_ok"] = True
                state["highlight_stage2_parse_note"] = "not_entered_stage2"
                state["final_highlight"] = state["generated_highlight"]
                state["highlight_changed"] = False
            else:
                state["highlight_entered_stage2"] = True
                state["highlight_model_input_rounds"] = 2
                state["highlight_model_input"] = model_context
                state["highlight_stage2_parse_ok"] = False
                state["highlight_stage2_parse_note"] = "pending_stage2"
                state["final_highlight"] = ""
                state["highlight_changed"] = False

        elif job.job_type == "ck_stage1":
            parsed = parse_ck_stage1_output(raw_text)
            stage1_result = {
                **parsed,
                "stage1_model_output": raw_text,
                "_stage1_model_input": model_input,
            }
            route = route_from_stage1_result(stage1_result)
            state = ck_states[i]
            state["ck_stage1_status"] = safe_str(stage1_result.get("stage1_status", ""))
            state["ck_stage1_route"] = route
            state["ck_stage1_parse_ok"] = bool(stage1_result.get("stage1_parse_ok", True))
            state["ck_stage1_parse_note"] = safe_str(stage1_result.get("stage1_parse_note", ""))
            state["_stage1_result"] = stage1_result

            if route == "none":
                state["ck_entered_stage2"] = False
                state["ck_model_input_rounds"] = 1
                state["ck_model_input"] = model_context
                state["ck_stage2_parse_ok"] = True
                state["ck_stage2_parse_note"] = "not_entered_stage2"
                state["final_category"] = state["original_category"]
                state["final_keywords"] = state["original_keywords"]
                state["category_changed"] = False
                state["keywords_changed"] = False
            else:
                state["ck_entered_stage2"] = True
                state["ck_model_input_rounds"] = 2
                state["ck_model_input"] = model_context
                state["ck_stage2_parse_ok"] = False
                state["ck_stage2_parse_note"] = "pending_stage2"
                state["final_category"] = state["original_category"]
                state["final_keywords"] = state["original_keywords"]
                state["category_changed"] = False
                state["keywords_changed"] = False
        else:
            raise ValueError(f"Unsupported Stage 1 job_type: {job.job_type}")


def build_unified_stage2_jobs(
    *,
    highlight_states: List[Dict[str, Any]],
    ck_states: List[Dict[str, Any]],
    allowed_categories: List[str],
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
) -> List[JudgeJob]:
    jobs: List[JudgeJob] = []

    for i, state in enumerate(highlight_states):
        if not bool_value(state.get("highlight_entered_stage2", False)):
            continue
        messages = build_judge_round2_messages(
            state["_packet"],
            state["_round1_result"],
            top_n_per_claim=top_n_per_claim,
        )
        jobs.append(JudgeJob(
            job_id=f"highlight_round2:{i}",
            job_type="highlight_round2",
            stage=2,
            item_index=i,
            messages=messages,
        ))

    for i, state in enumerate(ck_states):
        if not bool_value(state.get("ck_entered_stage2", False)):
            continue
        route = route_from_stage1_result(state["_stage1_result"])
        if route == "none":
            continue
        messages = build_ck_stage2_messages(
            state["_sample"],
            state["_stage1_result"],
            route=route,
            allowed_categories=allowed_categories,
        )
        jobs.append(JudgeJob(
            job_id=f"ck_stage2_{route}:{i}",
            job_type=f"ck_stage2_{route}",
            stage=2,
            item_index=i,
            messages=messages,
            route=route,
        ))

    return sort_jobs(jobs, UNIFIED_STAGE2_JOB_ORDER)


def parse_unified_stage2_jobs(
    *,
    jobs: List[JudgeJob],
    highlight_states: List[Dict[str, Any]],
    ck_states: List[Dict[str, Any]],
    allowed_categories: List[str],
) -> None:
    for job in jobs:
        raw_text = safe_str(job.generated.get("raw_text", ""))
        model_input = safe_str(job.generated.get("messages_debug_text", ""))
        model_context = build_model_full_context_text(
            stage_label=f"Stage {job.stage} - {job.job_type}",
            messages_debug_text=model_input,
            raw_output=raw_text,
        )
        i = int(job.item_index)

        if job.job_type == "highlight_round2":
            final_answer, parse_ok, parse_note = extract_round2_final_answer(raw_text)
            state = highlight_states[i]
            state["highlight_model_input_rounds"] = 2
            state["highlight_model_input"] = append_model_full_context_round(
                state.get("highlight_model_input", ""),
                model_context,
            )
            state["highlight_stage2_parse_ok"] = bool(parse_ok)
            state["highlight_stage2_parse_note"] = parse_note
            if parse_ok:
                state["final_highlight"] = safe_str(final_answer)
                state["highlight_changed"] = highlight_changed_value(
                    state.get("generated_highlight", ""),
                    final_answer,
                )
            else:
                # Fallback: keep the original generated highlight when Stage 2 fails to parse,
                # including unclosed <think> blocks / likely truncation.
                # This prevents final_highlight from becoming empty while preserving the
                # parse-failure flags and notes for debugging.
                state["final_highlight"] = safe_str(state.get("generated_highlight", ""))
                state["highlight_changed"] = False

        elif job.job_type in {"ck_stage2_category", "ck_stage2_keywords", "ck_stage2_both"}:
            route = safe_str(job.route).strip().lower()
            state = ck_states[i]
            original_category = state.get("original_category", "")
            original_keywords = state.get("original_keywords", "")

            stage2_category = ""
            stage2_keywords = ""
            parse_ok = True
            parse_note = ""

            if route == "category":
                stage2_category, parse_ok, parse_note = extract_category_output(raw_text, allowed_categories)
            elif route == "keywords":
                stage2_keywords, parse_ok, parse_note = extract_keywords_output(raw_text)
            elif route == "both":
                stage2_category, stage2_keywords, parse_ok, parse_note = extract_combined_output(raw_text, allowed_categories)
            else:
                parse_ok = False
                parse_note = f"unsupported_route:{route}"

            final_category = original_category
            final_keywords = original_keywords
            if parse_ok:
                if route == "category" and stage2_category:
                    final_category = stage2_category
                elif route == "keywords" and stage2_keywords:
                    final_keywords = stage2_keywords
                elif route == "both":
                    if stage2_category:
                        final_category = stage2_category
                    if stage2_keywords:
                        final_keywords = stage2_keywords

            state["ck_model_input_rounds"] = 2
            state["ck_model_input"] = append_model_full_context_round(
                state.get("ck_model_input", ""),
                model_context,
            )
            state["ck_stage2_parse_ok"] = bool(parse_ok)
            state["ck_stage2_parse_note"] = parse_note
            state["final_category"] = safe_str(final_category)
            state["final_keywords"] = safe_str(final_keywords)
            state["category_changed"] = changed_value(original_category, final_category)
            state["keywords_changed"] = changed_value(original_keywords, final_keywords)
        else:
            raise ValueError(f"Unsupported Stage 2 job_type: {job.job_type}")


# ======================================================================================
# Unified article-level assembly and PostgreSQL saving
# ======================================================================================

def build_unique_source_id_index(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[str]]:
    source_id_to_indices: Dict[str, List[int]] = {}

    for idx, row in enumerate(rows):
        source_id = safe_str(row.get("source_id", "")).strip()
        if source_id:
            source_id_to_indices.setdefault(source_id, []).append(idx)

    unique: Dict[str, int] = {}
    duplicates: List[str] = []

    for source_id, indices in source_id_to_indices.items():
        if len(indices) == 1:
            unique[source_id] = indices[0]
        else:
            duplicates.append(source_id)

    return unique, duplicates


def empty_unified_row() -> Dict[str, Any]:
    return {col: "" for col in UNIFIED_OUTPUT_COLUMNS}


def fill_highlight_into_row(row: Dict[str, Any], state: Dict[str, Any]) -> None:
    for field_name in ["article_id", "source_id", "title"]:
        if not safe_str(row.get(field_name, "")) and safe_str(state.get(field_name, "")):
            row[field_name] = safe_str(state.get(field_name, ""))

    row["packet_index"] = state.get("packet_index", "")
    row["highlight_model_input_rounds"] = state.get("highlight_model_input_rounds", "")
    row["highlight_model_input"] = state.get("highlight_model_input", "")
    row["generated_highlight"] = state.get("generated_highlight", "")
    row["final_highlight"] = state.get("final_highlight", "")
    row["highlight_changed"] = bool(state.get("highlight_changed", False))
    row["highlight_stage1_status"] = state.get("highlight_stage1_status", "")
    row["highlight_entered_stage2"] = bool(state.get("highlight_entered_stage2", False))
    row["highlight_stage2_parse_ok"] = bool(state.get("highlight_stage2_parse_ok", True))
    row["highlight_stage2_parse_note"] = state.get("highlight_stage2_parse_note", "")


def fill_ck_into_row(row: Dict[str, Any], state: Dict[str, Any]) -> None:
    for field_name in ["article_id", "source_id", "title"]:
        if not safe_str(row.get(field_name, "")) and safe_str(state.get(field_name, "")):
            row[field_name] = safe_str(state.get(field_name, ""))

    row["ck_model_input_rounds"] = state.get("ck_model_input_rounds", "")
    row["ck_model_input"] = state.get("ck_model_input", "")
    row["original_category"] = state.get("original_category", "")
    row["final_category"] = state.get("final_category", "")
    row["category_changed"] = bool(state.get("category_changed", False))
    row["original_keywords"] = state.get("original_keywords", "")
    row["final_keywords"] = state.get("final_keywords", "")
    row["keywords_changed"] = bool(state.get("keywords_changed", False))
    row["ck_stage1_status"] = state.get("ck_stage1_status", "")
    row["ck_stage1_route"] = state.get("ck_stage1_route", "")
    row["ck_stage1_parse_ok"] = bool(state.get("ck_stage1_parse_ok", True))
    row["ck_stage1_parse_note"] = state.get("ck_stage1_parse_note", "")
    row["ck_entered_stage2"] = bool(state.get("ck_entered_stage2", False))
    row["ck_stage2_parse_ok"] = bool(state.get("ck_stage2_parse_ok", True))
    row["ck_stage2_parse_note"] = state.get("ck_stage2_parse_note", "")


def finalize_unified_row(row: Dict[str, Any]) -> None:
    changed_flags = [
        bool_value(row.get("highlight_changed", False)),
        bool_value(row.get("category_changed", False)),
        bool_value(row.get("keywords_changed", False)),
    ]

    parse_failed = False
    if row.get("highlight_stage2_parse_ok", "") != "":
        parse_failed = parse_failed or (not bool_value(row.get("highlight_stage2_parse_ok")))
    if row.get("ck_stage1_parse_ok", "") != "":
        parse_failed = parse_failed or (not bool_value(row.get("ck_stage1_parse_ok")))
    if row.get("ck_stage2_parse_ok", "") != "":
        parse_failed = parse_failed or (not bool_value(row.get("ck_stage2_parse_ok")))

    any_changed = any(changed_flags)
    row["any_changed"] = bool(any_changed)
    row["any_parse_failed"] = bool(parse_failed)
    if parse_failed:
        row["final_quality_status"] = "PARSE_FAILED"
    elif any_changed:
        row["final_quality_status"] = "REVISED"
    else:
        row["final_quality_status"] = "OK"


def assemble_unified_article_results(
    *,
    highlight_states: List[Dict[str, Any]],
    ck_states: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for state in highlight_states:
        row = empty_unified_row()
        fill_highlight_into_row(row, state)
        rows.append(row)

    unique_index, duplicate_source_ids = build_unique_source_id_index(rows)
    if duplicate_source_ids:
        print(
            "[WARN] Duplicate highlight source_id values found. "
            "They will not be used for automatic joining."
        )
        for source_id in duplicate_source_ids[:10]:
            print("duplicate source_id:", source_id)
        if len(duplicate_source_ids) > 10:
            print(
                "... additional duplicate source_id values:",
                len(duplicate_source_ids) - 10,
            )

    unmatched_ck = 0
    for state in ck_states:
        source_id = safe_str(state.get("source_id", "")).strip()
        row_idx = unique_index.get(source_id) if source_id else None

        if row_idx is None:
            if source_id in duplicate_source_ids:
                print(
                    "[WARN] CK row not joined because highlight source_id is duplicated:",
                    source_id,
                )
            row = empty_unified_row()
            fill_ck_into_row(row, state)
            rows.append(row)
            unmatched_ck += 1
        else:
            fill_ck_into_row(rows[row_idx], state)

    if unmatched_ck:
        print("[WARN] CK rows without a matched highlight source_id:", unmatched_ck)

    for row in rows:
        finalize_unified_row(row)
    return rows


def build_unified_guardian_judge_dataframe(
    rows: List[Dict[str, Any]],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in UNIFIED_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[UNIFIED_OUTPUT_COLUMNS]

    print("=" * 100)
    print("[UNIFIED GUARDIAN JUDGE RESULTS]")
    print("rows:", len(df))

    if len(df) > 0:
        print("final_quality_status:")
        print(df["final_quality_status"].value_counts(dropna=False))

        print("\nactual changed flags:")
        print("highlight_changed:")
        print(df["highlight_changed"].value_counts(dropna=False))
        print("category_changed:")
        print(df["category_changed"].value_counts(dropna=False))
        print("keywords_changed:")
        print(df["keywords_changed"].value_counts(dropna=False))

        print("\nhighlight stage1 status:")
        print(df["highlight_stage1_status"].value_counts(dropna=False))

        print("\nhighlight stage1 -> actual changed:")
        print(pd.crosstab(
            df["highlight_stage1_status"],
            df["highlight_changed"],
            dropna=False,
        ))

        # ------------------------------------------------------------------
        # Highlight Stage 2 parse / fallback report
        # ------------------------------------------------------------------
        def _bool_series(series: pd.Series) -> pd.Series:
            return (
                series
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1", "yes", "y"})
            )

        def _preview_text(text: Any, max_chars: int = 220) -> str:
            text = safe_str(text).replace("\n", " ").strip()
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."

        highlight_entered_stage2 = _bool_series(df["highlight_entered_stage2"])
        highlight_stage2_parse_ok = _bool_series(df["highlight_stage2_parse_ok"])

        generated_highlight = df["generated_highlight"].fillna("").astype(str).str.strip()
        final_highlight = df["final_highlight"].fillna("").astype(str).str.strip()

        highlight_stage2_parse_failed_mask = (
            highlight_entered_stage2
            & (~highlight_stage2_parse_ok)
        )

        highlight_fallback_to_original_mask = (
            highlight_stage2_parse_failed_mask
            & final_highlight.ne("")
            & final_highlight.eq(generated_highlight)
        )

        highlight_parse_failed_empty_mask = (
            highlight_stage2_parse_failed_mask
            & final_highlight.eq("")
        )

        generated_highlight_norm = generated_highlight.map(normalize_highlight_for_change)
        final_highlight_norm = final_highlight.map(normalize_highlight_for_change)

        highlight_stage2_revised_mask = (
            highlight_entered_stage2
            & highlight_stage2_parse_ok
            & final_highlight.ne("")
            & final_highlight_norm.ne(generated_highlight_norm)
        )

        print("\nhighlight stage2 summary:")
        print("entered_stage2:", int(highlight_entered_stage2.sum()))
        print("stage2_parse_ok:", int((highlight_entered_stage2 & highlight_stage2_parse_ok).sum()))
        print("stage2_parse_failed:", int(highlight_stage2_parse_failed_mask.sum()))
        print("stage2_revised:", int(highlight_stage2_revised_mask.sum()))
        print("stage2_failed_fallback_to_original:", int(highlight_fallback_to_original_mask.sum()))
        print("stage2_failed_empty_final_highlight:", int(highlight_parse_failed_empty_mask.sum()))

        if highlight_fallback_to_original_mask.any():
            print("\n[HIGHLIGHT STAGE2 FALLBACK TO ORIGINAL]")
            fallback_rows = df.loc[
                highlight_fallback_to_original_mask,
                [
                    "packet_index",
                    "title",
                    "highlight_stage2_parse_note",
                    "generated_highlight",
                    "final_highlight",
                ],
            ].copy()

            fallback_rows["generated_highlight"] = fallback_rows["generated_highlight"].map(_preview_text)
            fallback_rows["final_highlight"] = fallback_rows["final_highlight"].map(_preview_text)

            print(fallback_rows.to_string(index=False))

        if highlight_parse_failed_empty_mask.any():
            print("\n[WARN] HIGHLIGHT STAGE2 PARSE FAILED AND FINAL_HIGHLIGHT IS STILL EMPTY]")
            empty_rows = df.loc[
                highlight_parse_failed_empty_mask,
                [
                    "packet_index",
                    "title",
                    "highlight_stage2_parse_note",
                    "generated_highlight",
                    "final_highlight",
                ],
            ].copy()

            empty_rows["generated_highlight"] = empty_rows["generated_highlight"].map(_preview_text)
            empty_rows["final_highlight"] = empty_rows["final_highlight"].map(_preview_text)

            print(empty_rows.to_string(index=False))

        # ------------------------------------------------------------------
        # Category / keyword reports
        # ------------------------------------------------------------------
        print("\ncategory/keyword stage1 status:")
        print(df["ck_stage1_status"].value_counts(dropna=False))

        print("\ncategory/keyword route:")
        print(df["ck_stage1_route"].value_counts(dropna=False))

        print("\nck route -> category_changed:")
        print(pd.crosstab(
            df["ck_stage1_route"],
            df["category_changed"],
            dropna=False,
        ))

        print("\nck route -> keywords_changed:")
        print(pd.crosstab(
            df["ck_stage1_route"],
            df["keywords_changed"],
            dropna=False,
        ))

        ck_entered_stage2 = _bool_series(df["ck_entered_stage2"])
        ck_stage2_parse_ok = _bool_series(df["ck_stage2_parse_ok"])

        ck_stage2_parse_failed_mask = (
            ck_entered_stage2
            & (~ck_stage2_parse_ok)
        )

        print("\nck stage2 summary:")
        print("entered_stage2:", int(ck_entered_stage2.sum()))
        print("stage2_parse_ok:", int((ck_entered_stage2 & ck_stage2_parse_ok).sum()))
        print("stage2_parse_failed:", int(ck_stage2_parse_failed_mask.sum()))

        if ck_stage2_parse_failed_mask.any():
            print("\n[CK STAGE2 PARSE FAILED ROWS]")
            ck_failed_rows = df.loc[
                ck_stage2_parse_failed_mask,
                [
                                    "title",
                    "ck_stage1_route",
                    "ck_stage2_parse_note",
                    "original_category",
                    "final_category",
                    "original_keywords",
                    "final_keywords",
                ],
            ].copy()

            for col in [
                "original_category",
                "final_category",
                "original_keywords",
                "final_keywords",
            ]:
                ck_failed_rows[col] = ck_failed_rows[col].map(_preview_text)

            print(ck_failed_rows.to_string(index=False))

    return df


def save_confirmed_judge_results_to_postgres(
    df: pd.DataFrame,
) -> Dict[str, int]:
    initialize_judge_results_table()

    if df is None or df.empty:
        stats = {
            "received": 0,
            "confirmed": 0,
            "inserted": 0,
            "skipped": 0,
        }
        print("Judge result storage completed.")
        print(stats)
        return stats

    required_columns = [
        "source_id",
        "final_highlight",
        "final_category",
        "final_keywords",
        "highlight_changed",
        "category_changed",
        "keywords_changed",
        "any_changed",
        "highlight_stage1_status",
        "highlight_entered_stage2",
        "highlight_stage2_parse_ok",
        "highlight_stage2_parse_note",
        "ck_stage1_status",
        "ck_stage1_route",
        "ck_stage1_parse_ok",
        "ck_stage1_parse_note",
        "ck_entered_stage2",
        "ck_stage2_parse_ok",
        "ck_stage2_parse_note",
        "any_parse_failed",
        "final_quality_status",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Judge result DataFrame is missing columns: {missing_columns}"
        )

    confirmed_statuses = {"OK", "REVISED"}
    records: List[Dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        source_id = safe_str(row.get("source_id", "")).strip()
        final_highlight = safe_str(row.get("final_highlight", "")).strip()
        final_category = safe_str(row.get("final_category", "")).strip()
        final_keywords = safe_str(row.get("final_keywords", "")).strip()
        final_quality_status = safe_str(
            row.get("final_quality_status", "")
        ).strip().upper()

        if final_quality_status not in confirmed_statuses:
            continue

        ck_stage1_parse_ok = bool_value(
        row.get("ck_stage1_parse_ok", False)
        )
        ck_entered_stage2 = bool_value(
            row.get("ck_entered_stage2", False)
        )
        ck_stage2_parse_ok = bool_value(
            row.get("ck_stage2_parse_ok", False)
        )

        if not ck_stage1_parse_ok:
            continue

        if ck_entered_stage2 and not ck_stage2_parse_ok:
            continue
        if not source_id or not final_highlight or not final_category or not final_keywords:
            continue

        records.append({
            "source_id": source_id,
            "final_highlight": final_highlight,
            "final_category": final_category,
            "final_keywords": final_keywords,
            "highlight_changed": bool_value(row.get("highlight_changed", False)),
            "category_changed": bool_value(row.get("category_changed", False)),
            "keywords_changed": bool_value(row.get("keywords_changed", False)),
            "any_changed": bool_value(row.get("any_changed", False)),
            "highlight_stage1_status": safe_str(
                row.get("highlight_stage1_status", "")
            ).strip(),
            "highlight_entered_stage2": bool_value(
                row.get("highlight_entered_stage2", False)
            ),
            "highlight_stage2_parse_ok": bool_value(
                row.get("highlight_stage2_parse_ok", False)
            ),
            "highlight_stage2_parse_note": safe_str(
                row.get("highlight_stage2_parse_note", "")
            ).strip(),
            "ck_stage1_status": safe_str(
                row.get("ck_stage1_status", "")
            ).strip(),
            "ck_stage1_route": safe_str(
                row.get("ck_stage1_route", "")
            ).strip(),
            "ck_stage1_parse_ok": bool_value(
                row.get("ck_stage1_parse_ok", False)
            ),
            "ck_stage1_parse_note": safe_str(
                row.get("ck_stage1_parse_note", "")
            ).strip(),
            "ck_entered_stage2": bool_value(
                row.get("ck_entered_stage2", False)
            ),
            "ck_stage2_parse_ok": bool_value(
                row.get("ck_stage2_parse_ok", False)
            ),
            "ck_stage2_parse_note": safe_str(
                row.get("ck_stage2_parse_note", "")
            ).strip(),
            "any_parse_failed": bool_value(
                row.get("any_parse_failed", False)
            ),
            "final_quality_status": final_quality_status,
        })

    insert_sql = f"""
    INSERT INTO {JUDGE_RESULTS_TABLE} AS current_row (
        source_id,
        final_highlight,
        final_category,
        final_keywords,
        highlight_changed,
        category_changed,
        keywords_changed,
        any_changed,
        highlight_stage1_status,
        highlight_entered_stage2,
        highlight_stage2_parse_ok,
        highlight_stage2_parse_note,
        ck_stage1_status,
        ck_stage1_route,
        ck_stage1_parse_ok,
        ck_stage1_parse_note,
        ck_entered_stage2,
        ck_stage2_parse_ok,
        ck_stage2_parse_note,
        any_parse_failed,
        final_quality_status
    )
    VALUES (
        %(source_id)s,
        %(final_highlight)s,
        %(final_category)s,
        %(final_keywords)s,
        %(highlight_changed)s,
        %(category_changed)s,
        %(keywords_changed)s,
        %(any_changed)s,
        %(highlight_stage1_status)s,
        %(highlight_entered_stage2)s,
        %(highlight_stage2_parse_ok)s,
        %(highlight_stage2_parse_note)s,
        %(ck_stage1_status)s,
        %(ck_stage1_route)s,
        %(ck_stage1_parse_ok)s,
        %(ck_stage1_parse_note)s,
        %(ck_entered_stage2)s,
        %(ck_stage2_parse_ok)s,
        %(ck_stage2_parse_note)s,
        %(any_parse_failed)s,
        %(final_quality_status)s
    )
    ON CONFLICT (source_id) DO UPDATE SET
        final_highlight = EXCLUDED.final_highlight,
        final_category = EXCLUDED.final_category,
        final_keywords = EXCLUDED.final_keywords,

        highlight_changed = EXCLUDED.highlight_changed,
        category_changed = EXCLUDED.category_changed,
        keywords_changed = EXCLUDED.keywords_changed,
        any_changed = EXCLUDED.any_changed,

        highlight_stage1_status = EXCLUDED.highlight_stage1_status,
        highlight_entered_stage2 = EXCLUDED.highlight_entered_stage2,
        highlight_stage2_parse_ok = EXCLUDED.highlight_stage2_parse_ok,
        highlight_stage2_parse_note = EXCLUDED.highlight_stage2_parse_note,

        ck_stage1_status = EXCLUDED.ck_stage1_status,
        ck_stage1_route = EXCLUDED.ck_stage1_route,
        ck_stage1_parse_ok = EXCLUDED.ck_stage1_parse_ok,
        ck_stage1_parse_note = EXCLUDED.ck_stage1_parse_note,

        ck_entered_stage2 = EXCLUDED.ck_entered_stage2,
        ck_stage2_parse_ok = EXCLUDED.ck_stage2_parse_ok,
        ck_stage2_parse_note = EXCLUDED.ck_stage2_parse_note,

        any_parse_failed = EXCLUDED.any_parse_failed,
        final_quality_status = EXCLUDED.final_quality_status

    WHERE
        current_row.final_highlight
            IS DISTINCT FROM EXCLUDED.final_highlight
        OR current_row.final_category
            IS DISTINCT FROM EXCLUDED.final_category
        OR current_row.final_keywords
            IS DISTINCT FROM EXCLUDED.final_keywords;
    """

    inserted_count = 0
    if records:
        with psycopg.connect(get_database_url()) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, records)
                inserted_count = max(int(cursor.rowcount), 0)

    stats = {
        "received": len(df),
        "confirmed": len(records),
        "inserted": inserted_count,
        "skipped": len(df) - inserted_count,
    }
    print("=" * 100)
    print("[SAVED CONFIRMED JUDGE RESULTS TO POSTGRESQL]")
    print(stats)
    return stats


# ======================================================================================
# Public runner
# ======================================================================================

def run_unified_guardian_judge_with_loaded_model(
    *,
    judge_llm,
    judge_tokenizer,
    packets: Optional[List[Dict[str, Any]]] = None,
    ck_samples: Optional[List[Dict[str, Any]]] = None,
    packets_jsonl_path: str = RETRIEVAL_PACKETS_JSONL_PATH,
    max_packets: Optional[int] = MAX_RETRIEVAL_PACKETS,
    max_ck_samples: Optional[int] = MAX_CATEGORY_KEYWORD_SAMPLES,
    source_ids: Optional[List[str]] = None,
    allowed_categories: Any = None,
    top_n_per_claim: int = JUDGE_EVIDENCE_TOP_N_PER_CLAIM,
    prompt_batch_size: Optional[int] = JUDGE_VLLM_PROMPT_BATCH_SIZE,
    use_tqdm: bool = True,
) -> pd.DataFrame:
    """Run the unified two-stage judge using an already loaded model."""
    if packets is None:
        packets = load_saved_retrieval_packets_jsonl(
            packets_jsonl_path=packets_jsonl_path,
            max_packets=max_packets,
            source_ids=source_ids,
        )
    if ck_samples is None:
        ck_samples = load_category_keyword_samples_from_postgres(
            max_samples=max_ck_samples,
            source_ids=source_ids,
        )
    if allowed_categories is None:
        allowed_categories = flatten_allowed_categories()
    allowed_categories = flatten_allowed_categories(allowed_categories)

    print("=" * 100)
    print("[UNIFIED GUARDIAN JUDGE]")
    print("highlight packets:", len(packets))
    print("category/keyword samples:", len(ck_samples))
    print("stage1 enable_thinking:", UNIFIED_STAGE1_ENABLE_THINKING)
    print("stage1 max_tokens:", UNIFIED_STAGE1_MAX_TOKENS)
    print("stage1 repetition_penalty:", UNIFIED_STAGE1_REPETITION_PENALTY)
    print("stage1 frequency_penalty:", UNIFIED_STAGE1_FREQUENCY_PENALTY)
    print("stage2 enable_thinking:", UNIFIED_STAGE2_ENABLE_THINKING)
    print("stage2 max_tokens:", UNIFIED_STAGE2_MAX_TOKENS)
    print("stage2 repetition_penalty:", UNIFIED_STAGE2_REPETITION_PENALTY)
    print("stage2 frequency_penalty:", UNIFIED_STAGE2_FREQUENCY_PENALTY)
    print("prompt_batch_size:", prompt_batch_size)

    highlight_states = init_highlight_states(packets)
    ck_states = init_ck_states(ck_samples)

    # ------------------------------
    # Unified Stage 1
    # ------------------------------
    stage1_jobs = build_unified_stage1_jobs(
        packets=packets,
        ck_samples=ck_samples,
        top_n_per_claim=top_n_per_claim,
    )
    print("=" * 100)
    print("[UNIFIED STAGE 1]")
    print("total jobs:", len(stage1_jobs))
    print("highlight_round1:", sum(1 for j in stage1_jobs if j.job_type == "highlight_round1"))
    print("ck_stage1:", sum(1 for j in stage1_jobs if j.job_type == "ck_stage1"))

    t_stage1 = time.time()
    stage1_jobs = run_unified_jobs_generate(
        judge_llm=judge_llm,
        judge_tokenizer=judge_tokenizer,
        jobs=stage1_jobs,
        max_tokens=UNIFIED_STAGE1_MAX_TOKENS,
        enable_thinking=UNIFIED_STAGE1_ENABLE_THINKING,
        repetition_penalty=UNIFIED_STAGE1_REPETITION_PENALTY,
        frequency_penalty=UNIFIED_STAGE1_FREQUENCY_PENALTY,
        prompt_batch_size=prompt_batch_size,
        desc="unified stage1",
        use_tqdm=use_tqdm,
    )
    parse_unified_stage1_jobs(
        jobs=stage1_jobs,
        highlight_states=highlight_states,
        ck_states=ck_states,
    )
    print("stage1 elapsed_seconds:", round(time.time() - t_stage1, 3))
    print("highlight PASS:", sum(1 for s in highlight_states if s.get("highlight_stage1_status") == "PASS"))
    print("highlight VERIFY:", sum(1 for s in highlight_states if s.get("highlight_stage1_status") == "VERIFY"))
    print("ck PASS:", sum(1 for s in ck_states if s.get("ck_stage1_status") == "PASS"))
    print("ck VERIFY:", sum(1 for s in ck_states if s.get("ck_stage1_status") == "VERIFY"))
    print("ck route category:", sum(1 for s in ck_states if s.get("ck_stage1_route") == "category"))
    print("ck route keywords:", sum(1 for s in ck_states if s.get("ck_stage1_route") == "keywords"))
    print("ck route both:", sum(1 for s in ck_states if s.get("ck_stage1_route") == "both"))

    # ------------------------------
    # Unified Stage 2
    # ------------------------------
    stage2_jobs = build_unified_stage2_jobs(
        highlight_states=highlight_states,
        ck_states=ck_states,
        allowed_categories=allowed_categories,
        top_n_per_claim=top_n_per_claim,
    )
    print("=" * 100)
    print("[UNIFIED STAGE 2]")
    print("total jobs:", len(stage2_jobs))
    print("highlight_round2:", sum(1 for j in stage2_jobs if j.job_type == "highlight_round2"))
    print("ck_stage2_category:", sum(1 for j in stage2_jobs if j.job_type == "ck_stage2_category"))
    print("ck_stage2_keywords:", sum(1 for j in stage2_jobs if j.job_type == "ck_stage2_keywords"))
    print("ck_stage2_both:", sum(1 for j in stage2_jobs if j.job_type == "ck_stage2_both"))

    t_stage2 = time.time()
    if stage2_jobs:
        stage2_jobs = run_unified_jobs_generate(
            judge_llm=judge_llm,
            judge_tokenizer=judge_tokenizer,
            jobs=stage2_jobs,
            max_tokens=UNIFIED_STAGE2_MAX_TOKENS,
            enable_thinking=UNIFIED_STAGE2_ENABLE_THINKING,
            repetition_penalty=UNIFIED_STAGE2_REPETITION_PENALTY,
            frequency_penalty=UNIFIED_STAGE2_FREQUENCY_PENALTY,
            prompt_batch_size=prompt_batch_size,
            desc="unified stage2",
            use_tqdm=use_tqdm,
        )
        parse_unified_stage2_jobs(
            jobs=stage2_jobs,
            highlight_states=highlight_states,
            ck_states=ck_states,
            allowed_categories=allowed_categories,
        )
    print("stage2 elapsed_seconds:", round(time.time() - t_stage2, 3))

    rows = assemble_unified_article_results(
        highlight_states=highlight_states,
        ck_states=ck_states,
    )
    df = build_unified_guardian_judge_dataframe(rows)
    save_confirmed_judge_results_to_postgres(df)
    return df
