# -*- coding: utf-8 -*-
"""Guardian RAG batch generation and retrieval pipeline for Colab.

This is a regular top-to-bottom Python script.
"""

from __future__ import annotations

import gc
import glob
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import requests
import torch
from datasets import Dataset, DatasetDict
from openai import OpenAI
from safetensors.torch import safe_open
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from vllm import SamplingParams
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateQueryError
from weaviate.util import generate_uuid5

try:
    from .http_logging import redact_query_parameter
except ImportError:  # Preserve direct script execution from the rag directory.
    from http_logging import redact_query_parameter

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
except ImportError as exc:
    raise ImportError(
        "psycopg is required. Install the dependencies declared in "
        "requirements-rag.txt before running the RAG pipeline."
    ) from exc


EXTRACT_ADAPTER_ARCHIVE = False
ADAPTER_ARCHIVE_PATH = os.environ.get("ADAPTER_ARCHIVE_PATH", "")
ADAPTER_EXTRACT_DIR = os.environ.get("ADAPTER_EXTRACT_DIR", "")

BASE_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ARTIFACT_DIR = Path(
    os.environ.get("RAG_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts" / "rag"))
)
TEMP_ROOT = os.environ.get("RAG_TEMP_ROOT", str(RAG_ARTIFACT_DIR / "runtime"))
TEMP_RESIZED_BASE_DIR = f"{TEMP_ROOT}/resized_base"
TEMP_MERGED_MODEL_DIR = f"{TEMP_ROOT}/merged_adapter"
FORCE_REMERGE_MODEL = False
FORCE_RESIZE_BASE_MODEL = False

GUARDIAN_ORDER_BY = "newest"
GUARDIAN_FROM_DATE = None
GUARDIAN_TO_DATE = None

BATCH_N = 2500
RAW_ARTICLES_TABLE = "raw_articles"
MODEL_OUTPUTS_TABLE = "model_outputs"
GENERATION_TASK_STATUS_TABLE = "generation_task_status"
# Increment this value if the task input eligibility rules change. Status rows
# from an older version are deliberately ignored and evaluated again.
GENERATION_ELIGIBILITY_VERSION = "guardian-input-eligibility-v1"
GENERATION_TASKS = ("highlight", "both")
MODEL_OUTPUT_COLUMNS = [
    "task",
    "row_index",
    "source_id",
    "title",
    "url",
    "published_at",
    "guardian_section_id",
    "guardian_section_name",
    "expected_output_format",
    "prompt_raw",
    "model_input_text",
    "input_tokens",
    "generated_raw",
    "generated_clean",
    "output_tokens",
    "answer_repetition_penalty",
    "format_ok",
    "starts_with_highlight",
    "has_special_token",
    "rough_sentence_count",
    "repeated_3gram",
    "repeated_4gram",
    "maybe_repetition_loop",
    "has_categories",
    "has_keywords",
    "order_ok",
    "has_semicolon",
]
HIGHLIGHT_MAX_TOKENS = 256
BOTH_MAX_TOKENS = 128
ANSWER_REPETITION_PENALTY = 1.05
PRINT_EACH = True

WEAVIATE_INDEX_READY_WAIT_SECONDS = 20
WEAVIATE_HFRESH_QUERY_MAX_RETRIES = 8
WEAVIATE_HFRESH_QUERY_RETRY_SLEEP_SECONDS = 5

ANSWER_ONLY_REPETITION_PROCESSOR_PATH = str(
    Path(TEMP_ROOT) / "answer_only_repetition_processor_vllm.py"
)
ANSWER_ONLY_REPETITION_PROCESSOR_MODULE = r'''
from __future__ import annotations

import os
import re
from typing import List, Optional, Sequence

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


class AnswerOnlyRepetitionPenaltyProcessor:
    """
    Apply the repetition penalty only to tokens already generated in the answer.
    Prompt and article tokens are excluded.

    This prevents source terms such as Australia, World Cup, or 2026 from being
    penalized merely because they appeared in the input article.
    """

    def __init__(
        self,
        *,
        penalty: float,
        answer_prefix_ids: Sequence[int] = (),
        exclude_token_ids: Sequence[int] = (),
    ):
        penalty = float(penalty)
        if penalty <= 0:
            raise ValueError("penalty must be > 0")

        self.penalty = penalty
        self.answer_prefix_ids = [int(x) for x in answer_prefix_ids]
        self.exclude_token_ids = set(int(x) for x in exclude_token_ids)

    @staticmethod
    def _to_list(x) -> List[int]:
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            return [int(v) for v in x.detach().cpu().view(-1).tolist()]
        return [int(v) for v in list(x)]

    def _build_answer_ids(self, *args):
        """
        Support both vLLM processor signatures:
            1. (output_token_ids, logits)
            2. (prompt_token_ids, output_token_ids, logits)

        Only output_token_ids are used in either case.
        """
        if len(args) == 2:
            output_token_ids, logits = args
        elif len(args) == 3:
            _prompt_token_ids, output_token_ids, logits = args
        else:
            raise TypeError(
                "Unsupported vLLM logits processor signature. Expected "
                "(output_token_ids, logits) or "
                "(prompt_token_ids, output_token_ids, logits)."
            )

        output_ids = self._to_list(output_token_ids)
        answer_ids = list(self.answer_prefix_ids) + output_ids
        return answer_ids, logits

    def __call__(self, *args):
        answer_ids, logits = self._build_answer_ids(*args)

        if self.penalty == 1.0 or not answer_ids:
            return logits

        seen = sorted(set(int(x) for x in answer_ids))

        if self.exclude_token_ids:
            seen = [x for x in seen if x not in self.exclude_token_ids]

        if not seen:
            return logits

        idx = torch.tensor(seen, dtype=torch.long, device=logits.device)
        valid = (idx >= 0) & (idx < logits.shape[-1])
        idx = idx[valid]

        if idx.numel() == 0:
            return logits

        token_scores = logits[idx]

        # HF-style repetition penalty:
        # positive logit -> divide by penalty
        # negative logit -> multiply by penalty
        logits[idx] = torch.where(
            token_scores < 0,
            token_scores * self.penalty,
            token_scores / self.penalty,
        )

        return logits


class WrappedAnswerOnlyRepetitionPenaltyProcessor(AdapterLogitsProcessor):
    EXTRA_KEY = "answer_only_repetition_penalty"

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = getattr(params, "extra_args", None) or {}
        cfg = extra_args.get(cls.EXTRA_KEY)

        if cfg is None:
            return

        if "penalty" not in cfg:
            raise ValueError(f"Missing {cls.EXTRA_KEY}.penalty")

        penalty = float(cfg["penalty"])
        if penalty <= 0:
            raise ValueError("penalty must be > 0")

        for list_key in ["answer_prefix_ids", "exclude_token_ids"]:
            if list_key in cfg and cfg[list_key] is not None:
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

        return AnswerOnlyRepetitionPenaltyProcessor(
            penalty=float(cfg["penalty"]),
            answer_prefix_ids=[int(x) for x in cfg.get("answer_prefix_ids", [])],
            exclude_token_ids=[int(x) for x in cfg.get("exclude_token_ids", [])],
        )
'''

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = None
COLLECTION_NAME = "GuardianSentenceEvidenceOpenAISmallPOC"

RETRIEVAL_OVERLAP_RATIO = 0.1
RETRIEVAL_TOP_K = 8
RETRIEVAL_HYBRID_ALPHA = 0.7
RETRIEVAL_MIN_SCORE = 0.65
RETRIEVAL_TOP_N_TO_PRINT = 0
RETRIEVAL_PRINT_SEARCH_TEXT = False
RETRIEVAL_SAVE_DIR = str(RAG_ARTIFACT_DIR / "retrieval")
SEEN_IDS_PATH = os.path.join(TEMP_ROOT, "guardian_seen_ids.json")
# Batch retrieval settings.
# None selects all available highlight rows from PostgreSQL.
RETRIEVAL_BATCH_ROW_INDICES = None

# None = no cap. For debugging, set to a small integer such as 3.
RETRIEVAL_BATCH_MAX_SAMPLES = None

# True = if the expected packet json already exists, skip that row.
RETRIEVAL_BATCH_SKIP_EXISTING = False

# True = stop immediately on the first failed row.
RETRIEVAL_BATCH_FAIL_FAST = False

# Concurrent Weaviate hybrid search workers for claim-level evidence retrieval.
# 1 = serial. Start with 4 on Weaviate Cloud; increase only if stable.
RETRIEVAL_SEARCH_MAX_WORKERS = 4

RETRIEVAL_BATCH_SUMMARY_CSV_PATH = "guardian_retrieval_batch_summary.csv"
RETRIEVAL_BATCH_SUMMARY_JSON_PATH = "guardian_retrieval_batch_summary.json"
RETRIEVAL_BATCH_PACKETS_JSONL_PATH = "guardian_retrieval_packets.jsonl"
RETRIEVAL_BATCH_EVIDENCE_CSV_PATH = "guardian_retrieved_evidence_batch.csv"
RETRIEVAL_BATCH_CHUNKS_CSV_PATH = "guardian_source_chunks_batch.csv"


GUARDIAN_ALLOWED_SECTIONS = {
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
}

def run_shell(command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {command}")
    return subprocess.run(command, shell=True, check=check)


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:
        return None


def patch_vllm_stdout_for_colab() -> None:
    try:
        vllm_system_utils = importlib.import_module("vllm.utils.system_utils")
        vllm_system_utils.suppress_stdout = _noop_suppress_stdout
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.utils.system_utils":
            raise
        print("vllm.utils.system_utils is unavailable; stdout patch skipped.")
    try:
        vllm_parallel_state = importlib.import_module("vllm.distributed.parallel_state")
        vllm_parallel_state.suppress_stdout = _noop_suppress_stdout
    except Exception as exc:
        print("parallel_state patch skipped:", repr(exc))
    print("Patched vLLM suppress_stdout for Colab.")


def write_custom_logits_processor_module(path: str = ANSWER_ONLY_REPETITION_PROCESSOR_PATH) -> None:
    processor_path = Path(path)
    processor_path.parent.mkdir(parents=True, exist_ok=True)
    processor_path.write_text(ANSWER_ONLY_REPETITION_PROCESSOR_MODULE, encoding="utf-8")
    print(f"Wrote custom logits processor to {path}")


def ensure_content_on_pythonpath(
    processor_path: str = ANSWER_ONLY_REPETITION_PROCESSOR_PATH,
) -> None:
    module_dir = str(Path(processor_path).resolve().parent)
    existing = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    if module_dir not in existing:
        os.environ["PYTHONPATH"] = os.pathsep.join([module_dir, *existing])
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)




def fetch_guardian_new_since_last_seen(
    api_key: str,
    seen_ids: Set[str],
    order_by: str = "newest",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_new_articles: int = 200,
    timeout: int = 30,
    respect_rate_limit: bool = True,
) -> Tuple[List[Dict[str, Any]], Set[str]]:
    """
    Fetch the newest Guardian articles incrementally.

    Behavior:
    - Existing IDs cannot be excluded in the Guardian API request.
    - Results are processed page by page from page 1.
    - Fetching stops as soon as an ID from seen_ids is encountered.
    - At most max_new_articles new records are returned.
    """

    url = "https://content.guardianapis.com/search"

    page_size = 50

    base_params = {
        "api-key": api_key,
        "order-by": order_by,
        "page-size": page_size,

        "show-fields": "headline,trailText,bodyText,shortUrl",
        "show-tags": "keyword",

        "section": "|".join(sorted(GUARDIAN_ALLOWED_SECTIONS)),

        "query-fields": "headline",

        # Server-side query filter
        "q": 'NOT ("World Cup")',

        "type": "article",
    }

    if from_date:
        base_params["from-date"] = from_date

    if to_date:
        base_params["to-date"] = to_date

    records: List[Dict[str, Any]] = []
    updated_seen_ids = set(seen_ids)

    page = 1

    with requests.Session() as session:
        while len(records) < max_new_articles:
            params = dict(base_params)
            params["page"] = page

            response = session.get(url, params=params, timeout=timeout)

            print(
                f"Request page {page} URL:",
                redact_query_parameter(response.url, "api-key"),
            )
            print(f"HTTP status page {page}:", response.status_code)

            response.raise_for_status()

            data = response.json()
            api_response = data.get("response", {}) or {}
            results = api_response.get("results", []) or []

            print("API status:", api_response.get("status"))
            print("Total matched:", api_response.get("total"))
            print("Current page:", api_response.get("currentPage"))
            print("Pages available:", api_response.get("pages"))
            print("Returned:", len(results))

            if not results:
                break

            should_stop = False

            for item in results:
                article_id = item.get("id")

                if not article_id:
                    continue

                # Stop immediately when an already-seen article is reached.
                # Do not fetch a fixed batch and deduplicate afterward.
                if article_id in seen_ids:
                    should_stop = True
                    break

                fields = item.get("fields", {}) or {}
                tags = item.get("tags", []) or []

                record = {
                    "id": article_id,
                    "source": "the_guardian",
                    "type": item.get("type"),
                    "section_id": item.get("sectionId"),
                    "section_name": item.get("sectionName"),
                    "web_title": item.get("webTitle"),
                    "title": fields.get("headline") or item.get("webTitle"),
                    "summary": fields.get("trailText", ""),
                    "body_text": fields.get("bodyText", ""),
                    "url": item.get("webUrl"),
                    "short_url": fields.get("shortUrl"),
                    "published_at": item.get("webPublicationDate"),
                    "api_url": item.get("apiUrl"),
                    "guardian_keyword_tags": [
                        {
                            "id": tag.get("id"),
                            "web_title": tag.get("webTitle"),
                            "type": tag.get("type"),
                        }
                        for tag in tags
                    ],
                }

                records.append(record)
                updated_seen_ids.add(article_id)

                if len(records) >= max_new_articles:
                    should_stop = True
                    break

            if should_stop:
                break

            pages_available = api_response.get("pages") or 0
            if page >= pages_available:
                break

            page += 1

            if respect_rate_limit:
                time.sleep(1)

    return records, updated_seen_ids


def _extract_input_ids(x: Any) -> List[int]:
    """
    Compatible with tokenizer.apply_chat_template return types.
    """
    if isinstance(x, dict):
        return list(x["input_ids"])

    if hasattr(x, "data") and isinstance(getattr(x, "data"), dict) and "input_ids" in x.data:
        return list(x.data["input_ids"])

    return list(x)


def _stringify_text(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, str):
        return x.strip()

    if isinstance(x, (list, tuple)):
        return ", ".join(
            str(v).strip()
            for v in x
            if str(v).strip()
        ).strip()

    return str(x).strip()


def _stable_fallback_id(*parts: str, length: int = 16) -> str:
    payload = "|||".join(_stringify_text(p) for p in parts)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:length]


def _count_tokens(tokenizer, text: str) -> int:
    text = _stringify_text(text)
    if not text:
        return 0

    return len(
        tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
    )


def _truncate_text_by_tokens(
    tokenizer,
    text: str,
    max_tokens: Optional[int],
) -> str:
    text = _stringify_text(text)

    if not text:
        return ""

    if max_tokens is None or max_tokens <= 0:
        return text

    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    if len(ids) <= max_tokens:
        return text

    truncated_ids = ids[:max_tokens]

    return tokenizer.decode(
        truncated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()


def default_highlight_system_text() -> str:
    """
    CNN/DM-style highlight task.

    Expected output:
        highlight: ...
    """
    return (
        "You are a news editor.\n"
        "Write the key highlights of the article in 2-4 concise sentences.\n"
        "Use only information from the article.\n"
        "Be factual, neutral, and concise.\n"
        "Do not add outside facts, guesses, explanations, or commentary."
    )


def default_both_system_text(
    keyword_prefix_text: str = "keywords: ",
    keyword_example: Optional[List[str]] = None,
    keyword_separator: str = ", ",
) -> str:
    """
    KPTimes-style both task.

    Expected output:
        categories: cat1, cat2, ...
        keywords: kw1, kw2, kw3, kw4
    """
    keyword_example = keyword_example or ["kw1", "kw2", "kw3", "kw4"]

    keyword_example_text = keyword_separator.join(
        _stringify_text(x)
        for x in keyword_example
        if _stringify_text(x)
    )

    return (
        "You are a professional news editor.\n"
        "Task: Predict both the article categories and keywords from the available article information.\n"
        "The available information includes the title and the body.\n"
        "Output only two lines in the exact format below, in this exact order:\n"
        "categories: cat1, cat2, ...\n"
        f"{keyword_prefix_text}{keyword_example_text}\n"
        "Do not explain.\n"
        "Do not repeat the input.\n"
        "Do not generate any extra text."
    )


def make_highlight_prompt(article: str) -> str:
    """
    Training-compatible highlight prompt.

    Format:
        Article:
        {article}
    """
    article = _stringify_text(article)
    return f"Article:\n{article}"


def make_title_body_prompt(title: str, body: str) -> str:
    """
    Training-compatible both-task prompt.

    Format:
        Title: {title}
        Body: {body}
    """
    title = _stringify_text(title)
    body = _stringify_text(body)
    return f"Title: {title}\nBody: {body}\n"


def build_inference_chat_ids_and_texts(
    tokenizer,
    user_text: str,
    system_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build inference-only Qwen chat input.

    Important:
    - No assistant answer.
    - No labels.
    - No loss weights.
    - Use add_generation_prompt=True so generation starts at assistant.
    """
    user_text = _stringify_text(user_text)
    system_text = _stringify_text(system_text)

    messages = []

    if system_text:
        messages.append({
            "role": "system",
            "content": system_text,
        })

    messages.append({
        "role": "user",
        "content": user_text,
    })

    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    prompt_text = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return {
        "messages": messages,
        "prompt_ids": prompt_ids,
        "prompt_text": prompt_text,
        "input_ids": prompt_ids,
        "attention_mask": [1] * len(prompt_ids),
    }


def normalize_guardian_keyword_tags(
    guardian_keyword_tags: Any,
) -> Tuple[List[Dict[str, str]], List[str], str]:
    """
    Preserve Guardian keyword tags as valuable metadata.

    Returns:
        normalized_tags:
            [
                {"id": "...", "web_title": "...", "type": "..."},
                ...
            ]

        keyword_titles:
            ["World Cup 2026", "Portugal", ...]

        keywords_text:
            "World Cup 2026, Portugal, ..."
    """
    if not guardian_keyword_tags:
        return [], [], ""

    if isinstance(guardian_keyword_tags, list):
        raw_tags = guardian_keyword_tags
    else:
        raw_tags = [guardian_keyword_tags]

    normalized_tags: List[Dict[str, str]] = []
    keyword_titles: List[str] = []

    seen_tag_keys = set()
    seen_titles = set()

    for tag in raw_tags:
        if isinstance(tag, dict):
            tag_id = _stringify_text(tag.get("id"))
            web_title = _stringify_text(
                tag.get("web_title")
                or tag.get("webTitle")
                or tag.get("title")
            )
            tag_type = _stringify_text(tag.get("type"))
        else:
            tag_id = ""
            web_title = _stringify_text(tag)
            tag_type = ""

        if not tag_id and not web_title and not tag_type:
            continue

        tag_key = (tag_id, web_title, tag_type)

        if tag_key not in seen_tag_keys:
            seen_tag_keys.add(tag_key)
            normalized_tags.append({
                "id": tag_id,
                "web_title": web_title,
                "type": tag_type,
            })

        if web_title and web_title not in seen_titles:
            seen_titles.add(web_title)
            keyword_titles.append(web_title)

    keywords_text = ", ".join(keyword_titles)
    return normalized_tags, keyword_titles, keywords_text


def normalize_guardian_article(
    article: Dict[str, Any],
    *,
    allow_summary_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Normalize one Guardian raw article into a shared intermediate structure.

    Important:
    - Guardian section and keyword tags are saved as metadata.
    - They are not inserted into PromptRaw.
    """
    title = (
        _stringify_text(article.get("title"))
        or _stringify_text(article.get("web_title"))
        or _stringify_text(article.get("webTitle"))
        or _stringify_text(article.get("web_title"))
    )

    body_text = _stringify_text(article.get("body_text"))
    summary = _stringify_text(article.get("summary"))

    body_source = "body_text"
    body = body_text

    if not body and allow_summary_fallback and summary:
        body = summary
        body_source = "summary_fallback"

    guardian_keyword_tags, guardian_keyword_titles, guardian_keywords_text = normalize_guardian_keyword_tags(
        article.get("guardian_keyword_tags", [])
    )

    url = _stringify_text(article.get("url") or article.get("webUrl"))

    source_id = (
        _stringify_text(article.get("id"))
        or _stable_fallback_id(title, url)
    )

    guardian_section_id = _stringify_text(
        article.get("section_id")
        or article.get("sectionId")
    )

    guardian_section_name = _stringify_text(
        article.get("section_name")
        or article.get("sectionName")
    )

    return {
        "source_id": source_id,
        "source": _stringify_text(article.get("source")) or "the_guardian",

        "title": title,
        "body": body,
        "body_source": body_source,
        "summary": summary,

        "guardian_section_id": guardian_section_id,
        "guardian_section_name": guardian_section_name,

        "guardian_keyword_tags": guardian_keyword_tags,
        "guardian_keyword_titles": guardian_keyword_titles,
        "guardian_keywords_text": guardian_keywords_text,

        "url": url,
        "short_url": _stringify_text(article.get("short_url")),
        "published_at": _stringify_text(
            article.get("published_at")
            or article.get("webPublicationDate")
        ),
        "api_url": _stringify_text(article.get("api_url") or article.get("apiUrl")),
    }


def get_database_url() -> str:
    database_url = get_secret("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing from the environment or Colab Secrets.")
    return database_url


_POSTGRES_TABLES_INITIALIZED = False


def initialize_postgres_tables() -> None:
    global _POSTGRES_TABLES_INITIALIZED

    if _POSTGRES_TABLES_INITIALIZED:
        return
    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS {RAW_ARTICLES_TABLE} (
            source_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            article_type TEXT,
            section_id TEXT,
            section_name TEXT,
            web_title TEXT,
            title TEXT NOT NULL,
            summary TEXT,
            body_text TEXT,
            url TEXT,
            short_url TEXT,
            published_at TIMESTAMPTZ,
            api_url TEXT,
            guardian_keyword_tags JSONB NOT NULL DEFAULT '[]'::jsonb
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {MODEL_OUTPUTS_TABLE} (
            task TEXT NOT NULL,
            row_index INTEGER,
            source_id TEXT NOT NULL REFERENCES {RAW_ARTICLES_TABLE}(source_id) ON DELETE CASCADE,
            title TEXT,
            url TEXT,
            published_at TIMESTAMPTZ,
            guardian_section_id TEXT,
            guardian_section_name TEXT,
            expected_output_format TEXT,
            prompt_raw TEXT,
            model_input_text TEXT,
            input_tokens INTEGER,
            generated_raw TEXT,
            generated_clean TEXT,
            output_tokens INTEGER,
            answer_repetition_penalty DOUBLE PRECISION,
            format_ok BOOLEAN,
            starts_with_highlight BOOLEAN,
            has_special_token BOOLEAN,
            rough_sentence_count INTEGER,
            repeated_3gram BOOLEAN,
            repeated_4gram BOOLEAN,
            maybe_repetition_loop BOOLEAN,
            has_categories BOOLEAN,
            has_keywords BOOLEAN,
            order_ok BOOLEAN,
            has_semicolon BOOLEAN,
            PRIMARY KEY (source_id, task)
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {GENERATION_TASK_STATUS_TABLE} (
            source_id TEXT NOT NULL REFERENCES {RAW_ARTICLES_TABLE}(source_id) ON DELETE CASCADE,
            task TEXT NOT NULL CHECK (task IN ('highlight', 'both')),
            status TEXT NOT NULL CHECK (status = 'ineligible'),
            reason TEXT NOT NULL,
            eligibility_version TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (source_id, task)
        );
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_raw_articles_published_at
            ON {RAW_ARTICLES_TABLE} (published_at DESC);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_model_outputs_task
            ON {MODEL_OUTPUTS_TABLE} (task);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_model_outputs_published_at
            ON {MODEL_OUTPUTS_TABLE} (published_at DESC);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_generation_task_status_version
            ON {GENERATION_TASK_STATUS_TABLE} (status, eligibility_version);
        """,
    ]

    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

    _POSTGRES_TABLES_INITIALIZED = True
    print("PostgreSQL tables are ready.")


def _normalize_database_value(value: Any) -> Any:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def save_generation_task_statuses(
    statuses: Sequence[Dict[str, Any]],
) -> Dict[str, int]:
    """Persist deterministic task-input exclusions without creating model output."""

    if not statuses:
        stats = {"received": 0, "recorded": 0, "invalid": 0}
        print("Generation task status storage completed.")
        print(stats)
        return stats

    initialize_postgres_tables()
    upsert_sql = f"""
    INSERT INTO {GENERATION_TASK_STATUS_TABLE} (
        source_id,
        task,
        status,
        reason,
        eligibility_version
    )
    VALUES (
        %(source_id)s,
        %(task)s,
        'ineligible',
        %(reason)s,
        %(eligibility_version)s
    )
    ON CONFLICT (source_id, task) DO UPDATE SET
        status = EXCLUDED.status,
        reason = EXCLUDED.reason,
        eligibility_version = EXCLUDED.eligibility_version,
        updated_at = NOW();
    """

    records: List[Dict[str, str]] = []
    invalid_count = 0
    for item in statuses:
        source_id = _stringify_text(item.get("source_id"))
        task = _stringify_text(item.get("task"))
        reason = _stringify_text(item.get("reason"))
        eligibility_version = _stringify_text(
            item.get("eligibility_version") or GENERATION_ELIGIBILITY_VERSION
        )
        if (
            not source_id
            or task not in GENERATION_TASKS
            or not reason
            or not eligibility_version
        ):
            invalid_count += 1
            continue
        records.append({
            "source_id": source_id,
            "task": task,
            "reason": reason,
            "eligibility_version": eligibility_version,
        })

    if records:
        with psycopg.connect(get_database_url()) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(upsert_sql, records)

    stats = {
        "received": len(statuses),
        "recorded": len(records),
        "invalid": invalid_count,
    }
    print("Generation task status storage completed.")
    print(stats)
    return stats


def save_raw_articles_to_postgres(
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if not articles:
        stats = {
            "received": 0,
            "inserted": 0,
            "skipped": 0,
            "invalid": 0,
            "failed": 0,
            "inserted_source_ids": [],
            "accepted_source_ids": [],
            "failures": [],
        }
        print("Raw article storage completed.")
        print(stats)
        return stats

    insert_sql = f"""
    INSERT INTO {RAW_ARTICLES_TABLE} (
        source_id,
        source,
        article_type,
        section_id,
        section_name,
        web_title,
        title,
        summary,
        body_text,
        url,
        short_url,
        published_at,
        api_url,
        guardian_keyword_tags
    )
    VALUES (
        %(source_id)s,
        %(source)s,
        %(article_type)s,
        %(section_id)s,
        %(section_name)s,
        %(web_title)s,
        %(title)s,
        %(summary)s,
        %(body_text)s,
        %(url)s,
        %(short_url)s,
        %(published_at)s,
        %(api_url)s,
        %(guardian_keyword_tags)s
    )
    ON CONFLICT (source_id) DO NOTHING
    RETURNING source_id;
    """

    records: List[Dict[str, Any]] = []
    invalid_count = 0

    for article in articles:
        source_id = _stringify_text(article.get("id"))
        title = _stringify_text(
            article.get("title")
            or article.get("web_title")
            or article.get("webTitle")
        )

        if not source_id or not title:
            invalid_count += 1
            continue

        keyword_tags = article.get("guardian_keyword_tags", [])
        if not isinstance(keyword_tags, list):
            keyword_tags = []

        records.append({
            "source_id": source_id,
            "source": _stringify_text(article.get("source")) or "the_guardian",
            "article_type": _stringify_text(article.get("type")) or None,
            "section_id": _stringify_text(
                article.get("section_id") or article.get("sectionId")
            ) or None,
            "section_name": _stringify_text(
                article.get("section_name") or article.get("sectionName")
            ) or None,
            "web_title": _stringify_text(
                article.get("web_title") or article.get("webTitle")
            ) or None,
            "title": title,
            "summary": _stringify_text(article.get("summary")) or None,
            "body_text": _stringify_text(article.get("body_text")) or None,
            "url": _stringify_text(article.get("url") or article.get("webUrl")) or None,
            "short_url": _stringify_text(article.get("short_url")) or None,
            "published_at": _stringify_text(
                article.get("published_at") or article.get("webPublicationDate")
            ) or None,
            "api_url": _stringify_text(
                article.get("api_url") or article.get("apiUrl")
            ) or None,
            "guardian_keyword_tags": Jsonb(keyword_tags),
        })

    inserted_source_ids: List[str] = []
    accepted_source_ids: List[str] = []
    failures: List[Dict[str, str]] = []
    if records:
        with psycopg.connect(get_database_url()) as conn:
            with conn.cursor() as cursor:
                for record in records:
                    source_id = str(record["source_id"])
                    try:
                        with conn.transaction():
                            cursor.execute(insert_sql, record)
                            inserted_row = cursor.fetchone()
                        accepted_source_ids.append(source_id)
                        if inserted_row is not None:
                            inserted_source_ids.append(str(inserted_row[0]))
                    except Exception as exc:
                        failures.append({
                            "source_id": source_id,
                            "error": repr(exc),
                        })

    inserted_count = len(inserted_source_ids)
    skipped_count = len(articles) - inserted_count
    stats = {
        "received": len(articles),
        "inserted": inserted_count,
        "skipped": skipped_count,
        "invalid": invalid_count,
        "failed": len(failures),
        "inserted_source_ids": inserted_source_ids,
        "accepted_source_ids": accepted_source_ids,
        "failures": failures,
    }

    print("Raw article storage completed.")
    print(stats)
    return stats


def load_raw_articles_for_generation(
    limit: int = BATCH_N,
    source_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:

    initialize_postgres_tables()
    source_filter = ""
    params: List[Any] = [
        GENERATION_ELIGIBILITY_VERSION,
        GENERATION_ELIGIBILITY_VERSION,
    ]
    if source_ids is not None:
        cleaned_source_ids = [
            _stringify_text(source_id)
            for source_id in source_ids
            if _stringify_text(source_id)
        ]
        if not cleaned_source_ids:
            return []
        source_filter = "AND source_id = ANY(%s)"
        params.append(cleaned_source_ids)

    query = f"""
    WITH generation_state AS (
        SELECT
            raw.*,
            EXISTS (
                SELECT 1
                FROM {MODEL_OUTPUTS_TABLE} AS output
                WHERE output.source_id = raw.source_id
                  AND output.task = 'highlight'
            ) AS has_highlight_output,
            EXISTS (
                SELECT 1
                FROM {MODEL_OUTPUTS_TABLE} AS output
                WHERE output.source_id = raw.source_id
                  AND output.task = 'both'
            ) AS has_both_output,
            EXISTS (
                SELECT 1
                FROM {GENERATION_TASK_STATUS_TABLE} AS task_status
                WHERE task_status.source_id = raw.source_id
                  AND task_status.task = 'highlight'
                  AND task_status.status = 'ineligible'
                  AND task_status.eligibility_version = %s
            ) AS highlight_ineligible,
            EXISTS (
                SELECT 1
                FROM {GENERATION_TASK_STATUS_TABLE} AS task_status
                WHERE task_status.source_id = raw.source_id
                  AND task_status.task = 'both'
                  AND task_status.status = 'ineligible'
                  AND task_status.eligibility_version = %s
            ) AS both_ineligible
        FROM {RAW_ARTICLES_TABLE} AS raw
    )
    SELECT
        source_id AS id,
        source,
        article_type AS type,
        section_id,
        section_name,
        web_title,
        title,
        summary,
        body_text,
        url,
        short_url,
        published_at,
        api_url,
        guardian_keyword_tags,
        has_highlight_output,
        has_both_output,
        highlight_ineligible,
        both_ineligible
    FROM generation_state
    WHERE
        (
            (NOT has_highlight_output AND NOT highlight_ineligible)
            OR (NOT has_both_output AND NOT both_ineligible)
        )
        {source_filter}
    ORDER BY published_at DESC NULLS LAST, source_id
    LIMIT %s;
    """
    params.append(int(limit))

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    articles = [dict(row) for row in rows]
    print("Raw articles loaded for generation:", len(articles))
    return articles


def load_pending_generation_source_ids(
    *,
    limit: int,
    exclude_source_ids: Optional[List[str]] = None,
) -> List[str]:
    """Return recent raw rows with a task that still needs model generation."""

    initialize_postgres_tables()
    excluded = [
        _stringify_text(source_id)
        for source_id in (exclude_source_ids or [])
        if _stringify_text(source_id)
    ]
    exclude_filter = ""
    params: List[Any] = []
    if excluded:
        exclude_filter = "AND NOT (source_id = ANY(%s))"
        params.append(excluded)

    query = f"""
    WITH generation_state AS (
        SELECT
            raw.source_id,
            raw.published_at,
            EXISTS (
                SELECT 1
                FROM {MODEL_OUTPUTS_TABLE} AS output
                WHERE output.source_id = raw.source_id
                  AND output.task = 'highlight'
            ) AS has_highlight_output,
            EXISTS (
                SELECT 1
                FROM {MODEL_OUTPUTS_TABLE} AS output
                WHERE output.source_id = raw.source_id
                  AND output.task = 'both'
            ) AS has_both_output,
            EXISTS (
                SELECT 1
                FROM {GENERATION_TASK_STATUS_TABLE} AS task_status
                WHERE task_status.source_id = raw.source_id
                  AND task_status.task = 'highlight'
                  AND task_status.status = 'ineligible'
                  AND task_status.eligibility_version = %s
            ) AS highlight_ineligible,
            EXISTS (
                SELECT 1
                FROM {GENERATION_TASK_STATUS_TABLE} AS task_status
                WHERE task_status.source_id = raw.source_id
                  AND task_status.task = 'both'
                  AND task_status.status = 'ineligible'
                  AND task_status.eligibility_version = %s
            ) AS both_ineligible
        FROM {RAW_ARTICLES_TABLE} AS raw
    )
    SELECT source_id
    FROM generation_state
    WHERE
        (
            (NOT has_highlight_output AND NOT highlight_ineligible)
            OR (NOT has_both_output AND NOT both_ineligible)
        )
        {exclude_filter}
    ORDER BY published_at DESC NULLS LAST, source_id
    LIMIT %s;
    """
    params = [
        GENERATION_ELIGIBILITY_VERSION,
        GENERATION_ELIGIBILITY_VERSION,
        *params,
    ]
    params.append(int(limit))

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    return [str(row["source_id"]) for row in rows]


def load_generation_complete_source_ids(source_ids: List[str]) -> List[str]:
    """Return candidate IDs that have persisted output rows for both tasks."""

    cleaned_source_ids = list(
        dict.fromkeys(
            _stringify_text(source_id)
            for source_id in source_ids
            if _stringify_text(source_id)
        )
    )
    if not cleaned_source_ids:
        return []

    initialize_postgres_tables()
    query = f"""
    SELECT raw.source_id
    FROM {RAW_ARTICLES_TABLE} AS raw
    WHERE raw.source_id = ANY(%s)
      AND EXISTS (
          SELECT 1
          FROM {MODEL_OUTPUTS_TABLE} AS output
          WHERE output.source_id = raw.source_id
            AND output.task = 'highlight'
      )
      AND EXISTS (
          SELECT 1
          FROM {MODEL_OUTPUTS_TABLE} AS output
          WHERE output.source_id = raw.source_id
            AND output.task = 'both'
      );
    """
    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (cleaned_source_ids,))
            completed = {str(row["source_id"]) for row in cursor.fetchall()}

    return [source_id for source_id in cleaned_source_ids if source_id in completed]


def save_model_outputs_to_postgres(
    df: pd.DataFrame,
) -> Dict[str, int]:

    if df is None or df.empty:
        stats = {"received": 0, "inserted": 0, "skipped": 0}
        print("Model output storage completed.")
        print(stats)
        return stats

    missing_columns = [
        column
        for column in MODEL_OUTPUT_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Model output DataFrame is missing columns: {missing_columns}"
        )

    insert_sql = f"""
    INSERT INTO {MODEL_OUTPUTS_TABLE} (
        task,
        row_index,
        source_id,
        title,
        url,
        published_at,
        guardian_section_id,
        guardian_section_name,
        expected_output_format,
        prompt_raw,
        model_input_text,
        input_tokens,
        generated_raw,
        generated_clean,
        output_tokens,
        answer_repetition_penalty,
        format_ok,
        starts_with_highlight,
        has_special_token,
        rough_sentence_count,
        repeated_3gram,
        repeated_4gram,
        maybe_repetition_loop,
        has_categories,
        has_keywords,
        order_ok,
        has_semicolon
    )
    VALUES (
        %(task)s,
        %(row_index)s,
        %(source_id)s,
        %(title)s,
        %(url)s,
        %(published_at)s,
        %(guardian_section_id)s,
        %(guardian_section_name)s,
        %(expected_output_format)s,
        %(prompt_raw)s,
        %(model_input_text)s,
        %(input_tokens)s,
        %(generated_raw)s,
        %(generated_clean)s,
        %(output_tokens)s,
        %(answer_repetition_penalty)s,
        %(format_ok)s,
        %(starts_with_highlight)s,
        %(has_special_token)s,
        %(rough_sentence_count)s,
        %(repeated_3gram)s,
        %(repeated_4gram)s,
        %(maybe_repetition_loop)s,
        %(has_categories)s,
        %(has_keywords)s,
        %(order_ok)s,
        %(has_semicolon)s
    )
    ON CONFLICT (source_id, task) DO NOTHING;
    """

    records: List[Dict[str, Any]] = []
    invalid_count = 0

    for row in df.to_dict(orient="records"):
        source_id = _stringify_text(row.get("source_id"))
        task = _stringify_text(row.get("task"))

        if not source_id or not task:
            invalid_count += 1
            continue

        record = {
            column: _normalize_database_value(row.get(column))
            for column in MODEL_OUTPUT_COLUMNS
        }
        if record.get("published_at") == "":
            record["published_at"] = None
        record["source_id"] = source_id
        record["task"] = task
        records.append(record)

    inserted_count = 0
    if records:
        with psycopg.connect(get_database_url()) as conn:
            with conn.cursor() as cursor:
                cursor.executemany(insert_sql, records)
                inserted_count = max(int(cursor.rowcount), 0)

    skipped_count = len(df) - inserted_count
    stats = {
        "received": len(df),
        "inserted": inserted_count,
        "skipped": skipped_count,
        "invalid": invalid_count,
    }

    print("Model output storage completed.")
    print(stats)
    return stats


def load_model_outputs_from_postgres(
    *,
    task: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
) -> pd.DataFrame:

    select_columns = ",\n        ".join(MODEL_OUTPUT_COLUMNS)
    query = f"""
    SELECT
        {select_columns}
    FROM {MODEL_OUTPUTS_TABLE}
    """

    conditions: List[str] = []
    params: List[Any] = []

    if task is not None:
        conditions.append("task = %s")
        params.append(str(task))

    if source_ids is not None:
        cleaned_source_ids = [
            _stringify_text(source_id)
            for source_id in source_ids
            if _stringify_text(source_id)
        ]
        if not cleaned_source_ids:
            return pd.DataFrame(columns=MODEL_OUTPUT_COLUMNS)
        conditions.append("source_id = ANY(%s)")
        params.append(cleaned_source_ids)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY published_at DESC NULLS LAST, source_id, task;"

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    return pd.DataFrame([dict(row) for row in rows], columns=MODEL_OUTPUT_COLUMNS)


def fetch_and_store_guardian_articles(
    *,
    api_key: str,
    order_by: str = GUARDIAN_ORDER_BY,
    from_date: Optional[str] = GUARDIAN_FROM_DATE,
    to_date: Optional[str] = GUARDIAN_TO_DATE,
    max_new_articles: int = BATCH_N,
    seen_ids_path: str = SEEN_IDS_PATH,
    timeout: int = 30,
    respect_rate_limit: bool = True,
) -> Dict[str, Any]:
    initialize_postgres_tables()
    seen_ids = load_seen_ids(seen_ids_path)

    articles, updated_seen_ids = fetch_guardian_new_since_last_seen(
        api_key=api_key,
        seen_ids=seen_ids,
        order_by=order_by,
        from_date=from_date,
        to_date=to_date,
        max_new_articles=max_new_articles,
        timeout=timeout,
        respect_rate_limit=respect_rate_limit,
    )

    storage_stats = save_raw_articles_to_postgres(articles)
    accepted_source_ids = set(storage_stats.get("accepted_source_ids", []))
    save_seen_ids(seen_ids | accepted_source_ids, seen_ids_path)

    return {
        "articles": articles,
        "updated_seen_ids": seen_ids | accepted_source_ids,
        "inserted_source_ids": list(storage_stats.get("inserted_source_ids", [])),
        "storage_stats": storage_stats,
    }


def is_valid_guardian_normalized_article(
    norm: Dict[str, Any],
    *,
    allowed_sections: Optional[Set[str]] = None,
    require_allowed_section: bool = True,
) -> bool:
    title = _stringify_text(norm.get("title"))
    body = _stringify_text(norm.get("body"))
    section_id = _stringify_text(norm.get("guardian_section_id"))

    if not title:
        return False

    if not body:
        return False

    if allowed_sections is not None and require_allowed_section:
        if section_id not in allowed_sections:
            return False

    return True


def _build_shared_guardian_metadata(norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valuable metadata to keep in both inference datasets.

    These fields are never inserted into PromptRaw unless explicitly formatted
    by the task-specific prompt builder.
    """
    return {
        "source_id": norm["source_id"],
        "source": norm["source"],

        "title": norm["title"],
        "summary": norm["summary"],
        "body_source": norm["body_source"],

        "guardian_section_id": norm["guardian_section_id"],
        "guardian_section_name": norm["guardian_section_name"],

        "guardian_keyword_tags": norm["guardian_keyword_tags"],
        "guardian_keyword_titles": norm["guardian_keyword_titles"],
        "guardian_keywords_text": norm["guardian_keywords_text"],

        "url": norm["url"],
        "short_url": norm["short_url"],
        "published_at": norm["published_at"],
        "api_url": norm["api_url"],
    }


def _maybe_add_debug_fields(
    row: Dict[str, Any],
    *,
    built: Dict[str, Any],
    keep_debug_fields: bool,
) -> Dict[str, Any]:
    """
    Add duplicate/debug-heavy fields only when requested.

    Default dataset remains cleaner:
    - input_ids is enough for inference
    - PromptRaw is enough for human inspection
    - PromptText / PromptIds are reproducible from tokenizer
    """
    if keep_debug_fields:
        row["PromptText"] = built["prompt_text"]
        row["PromptIds"] = built["prompt_ids"]
        row["ChatMessages"] = built["messages"]

    return row


def build_guardian_highlight_inference_rows(
    articles: List[Dict[str, Any]],
    tokenizer,
    *,
    allowed_sections: Optional[Set[str]] = None,
    require_allowed_section: bool = True,
    allow_summary_fallback: bool = True,

    # Align with your CNN/DM highlight training call
    include_title_in_highlight_article: bool = True,
    article_max_tokens: Optional[int] = 2500,
    min_article_tokens: int = 30,

    system_text: Optional[str] = None,
    keep_system_text: bool = True,
    keep_debug_fields: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build Guardian highlight inference rows.

    PromptRaw:
        Article:
        {article}

    Expected output:
        highlight: ...
    """
    if system_text is None:
        system_text = default_highlight_system_text()

    rows: List[Dict[str, Any]] = []

    for article in articles:
        norm = normalize_guardian_article(
            article,
            allow_summary_fallback=allow_summary_fallback,
        )

        if not is_valid_guardian_normalized_article(
            norm,
            allowed_sections=allowed_sections,
            require_allowed_section=require_allowed_section,
        ):
            continue

        title = norm["title"]
        body = norm["body"]

        raw_article_text = f"{title}\n\n{body}".strip() if include_title_in_highlight_article else body.strip()

        raw_article_tokens = _count_tokens(tokenizer, raw_article_text)

        article_text = _truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=raw_article_text,
            max_tokens=article_max_tokens,
        )

        used_article_tokens = _count_tokens(tokenizer, article_text)

        if used_article_tokens < int(min_article_tokens):
            continue

        prompt_raw = make_highlight_prompt(article_text)

        # Format guards.
        assert prompt_raw.startswith("Article:\n"), repr(prompt_raw[:100])

        built = build_inference_chat_ids_and_texts(
            tokenizer=tokenizer,
            user_text=prompt_raw,
            system_text=system_text,
        )

        row = {
            **_build_shared_guardian_metadata(norm),

            "DatasetName": "guardian_highlight_infer",
            "TaskType": "article->highlight",

            "SystemPromptName": "cnn_dm_highlight_v4_default",
            "PromptRaw": prompt_raw,

            # Task-specific input text after truncation.
            "Article": article_text,

            # Useful truncation metadata.
            "raw_article_tokens": raw_article_tokens,
            "used_article_tokens": used_article_tokens,
            "article_max_tokens": article_max_tokens,
            "include_title_in_highlight_article": include_title_in_highlight_article,

            # Expected generation format.
            "ExpectedOutputPrefix": "highlight: ",
            "ExpectedOutputFormat": "highlight: ...",

            # Actual model input.
            "input_ids": built["input_ids"],
            "attention_mask": built["attention_mask"],
        }

        if keep_system_text:
            row["SystemText"] = system_text

        row = _maybe_add_debug_fields(
            row,
            built=built,
            keep_debug_fields=keep_debug_fields,
        )

        rows.append(row)

    return rows


def build_guardian_both_inference_rows(
    articles: List[Dict[str, Any]],
    tokenizer,
    *,
    allowed_sections: Optional[Set[str]] = None,
    require_allowed_section: bool = True,
    allow_summary_fallback: bool = True,

    # Align with your KPTimes both training setting
    body_max_tokens: Optional[int] = 2000,
    min_body_tokens: int = 30,

    system_text: Optional[str] = None,
    keyword_prefix_text: str = "keywords: ",
    keep_system_text: bool = True,
    keep_debug_fields: bool = False,
) -> List[Dict[str, Any]]:
    """
    Build Guardian both-task inference rows.

    PromptRaw:
        Title: {title}
        Body: {body}

    Expected output:
        categories: ...
        keywords: ...
    """
    if system_text is None:
        system_text = default_both_system_text(
            keyword_prefix_text=keyword_prefix_text,
        )

    rows: List[Dict[str, Any]] = []

    for article in articles:
        norm = normalize_guardian_article(
            article,
            allow_summary_fallback=allow_summary_fallback,
        )

        if not is_valid_guardian_normalized_article(
            norm,
            allowed_sections=allowed_sections,
            require_allowed_section=require_allowed_section,
        ):
            continue

        title = norm["title"]
        raw_body = norm["body"]

        raw_body_tokens = _count_tokens(tokenizer, raw_body)

        body = _truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=raw_body,
            max_tokens=body_max_tokens,
        )

        used_body_tokens = _count_tokens(tokenizer, body)

        if used_body_tokens < int(min_body_tokens):
            continue

        prompt_raw = make_title_body_prompt(
            title=title,
            body=body,
        )

        # Format guards.
        assert prompt_raw.startswith("Title: "), repr(prompt_raw[:100])
        assert "\nBody: " in prompt_raw, repr(prompt_raw[:160])

        # Do not include explicit Guardian metadata fields in prompt.
        lower_prompt = prompt_raw.lower()
        assert "guardian_section" not in lower_prompt
        assert "section_id" not in lower_prompt
        assert "section_name" not in lower_prompt

        built = build_inference_chat_ids_and_texts(
            tokenizer=tokenizer,
            user_text=prompt_raw,
            system_text=system_text,
        )

        row = {
            **_build_shared_guardian_metadata(norm),

            "DatasetName": "guardian_both_infer",
            "TaskType": "title+body->both",

            "SystemPromptName": "kptimes_both_default",
            "PromptRaw": prompt_raw,

            # Task-specific input text after truncation.
            "Title": title,
            "Body": body,

            # Useful truncation metadata.
            "raw_body_tokens": raw_body_tokens,
            "used_body_tokens": used_body_tokens,
            "body_max_tokens": body_max_tokens,

            # Expected generation format.
            "ExpectedOutputPrefix": "categories: ",
            "ExpectedOutputFormat": "categories: cat1, cat2, ...\nkeywords: kw1, kw2, kw3, kw4",

            # Actual model input.
            "input_ids": built["input_ids"],
            "attention_mask": built["attention_mask"],
        }

        if keep_system_text:
            row["SystemText"] = system_text

        row = _maybe_add_debug_fields(
            row,
            built=built,
            keep_debug_fields=keep_debug_fields,
        )

        rows.append(row)

    return rows


def build_guardian_inference_datasets(
    articles: List[Dict[str, Any]],
    tokenizer,
    *,
    highlight_articles: Optional[List[Dict[str, Any]]] = None,
    both_articles: Optional[List[Dict[str, Any]]] = None,
    allowed_sections: Optional[Set[str]] = None,
    require_allowed_section: bool = True,
    allow_summary_fallback: bool = True,

    # Highlight settings
    # Align with:
    # build_cnn_dm_highlight_trainer_dataset_v4(..., article_max_tokens=2500, ...)
    include_title_in_highlight_article: bool = True,
    article_max_tokens: Optional[int] = 2500,
    min_article_tokens: int = 30,
    highlight_system_text: Optional[str] = None,

    # Both-task settings
    # Align with KPTimes title/body/both training.
    body_max_tokens: Optional[int] = 2000,
    min_body_tokens: int = 30,
    both_system_text: Optional[str] = None,
    keyword_prefix_text: str = "keywords: ",

    # Storage settings
    keep_system_text: bool = True,
    keep_debug_fields: bool = False,
    save_root: Optional[str] = None,
) -> Dict[str, DatasetDict]:
    """
    Build two independent inference datasets from the requested Guardian rows.

    By default both tasks receive ``articles``. Recovery callers may provide a
    task-specific subset so an already persisted task is not generated again.

    Output:
        {
            "highlight": DatasetDict({"inference": ds_highlight}),
            "both": DatasetDict({"inference": ds_both}),
        }

    No AnswerPlain / labels / loss_weights / token_roles are created.
    """
    highlight_input_articles = (
        articles if highlight_articles is None else highlight_articles
    )
    both_input_articles = articles if both_articles is None else both_articles

    highlight_rows = build_guardian_highlight_inference_rows(
        articles=highlight_input_articles,
        tokenizer=tokenizer,
        allowed_sections=allowed_sections,
        require_allowed_section=require_allowed_section,
        allow_summary_fallback=allow_summary_fallback,
        include_title_in_highlight_article=include_title_in_highlight_article,
        article_max_tokens=article_max_tokens,
        min_article_tokens=min_article_tokens,
        system_text=highlight_system_text,
        keep_system_text=keep_system_text,
        keep_debug_fields=keep_debug_fields,
    )

    both_rows = build_guardian_both_inference_rows(
        articles=both_input_articles,
        tokenizer=tokenizer,
        allowed_sections=allowed_sections,
        require_allowed_section=require_allowed_section,
        allow_summary_fallback=allow_summary_fallback,
        body_max_tokens=body_max_tokens,
        min_body_tokens=min_body_tokens,
        system_text=both_system_text,
        keyword_prefix_text=keyword_prefix_text,
        keep_system_text=keep_system_text,
        keep_debug_fields=keep_debug_fields,
    )

    ds_highlight = DatasetDict({
        "inference": Dataset.from_list(highlight_rows)
    })

    ds_both = DatasetDict({
        "inference": Dataset.from_list(both_rows)
    })

    outputs = {
        "highlight": ds_highlight,
        "both": ds_both,
    }

    if save_root:
        save_root = Path(save_root)
        save_root.mkdir(parents=True, exist_ok=True)

        highlight_path = save_root / "guardian_highlight_infer"
        both_path = save_root / "guardian_both_infer"

        ds_highlight.save_to_disk(str(highlight_path))
        ds_both.save_to_disk(str(both_path))

        print(f"Saved highlight dataset to: {highlight_path}")
        print(f"Saved both dataset to: {both_path}")

    print("Guardian highlight inference rows:", len(highlight_rows))
    print("Guardian both inference rows:", len(both_rows))

    return outputs


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


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
    """
    Infer the saved embedding or lm_head vocabulary size from a PEFT adapter.
    This is required when the adapter contains resized embedding weights.
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
                if hasattr(v, "shape") and (
                    ("embed_tokens.weight" in lk) or ("lm_head.weight" in lk)
                ):
                    shape = tuple(v.shape)
                    if len(shape) >= 2:
                        return int(shape[0])
        except Exception:
            continue

    return None


def get_model_embedding_vocab_size(model) -> int:
    return int(model.get_input_embeddings().weight.shape[0])


def maybe_resize_model_embeddings_to_vocab(model, target_vocab_size: int, reason: str = ""):
    target_vocab_size = int(target_vocab_size)

    if target_vocab_size < 1000:
        raise ValueError(f"Suspicious target_vocab_size={target_vocab_size}")

    old_vocab_size = get_model_embedding_vocab_size(model)

    print("  base_embedding_vocab_size:", old_vocab_size)
    print("  target_vocab_size:", target_vocab_size)

    if reason:
        print("  resize_reason:", reason)

    if old_vocab_size != target_vocab_size:
        print(f"[RESIZE TOKEN EMBEDDINGS] {old_vocab_size} -> {target_vocab_size}")
        try:
            model.resize_token_embeddings(target_vocab_size, mean_resizing=False)
        except TypeError:
            model.resize_token_embeddings(target_vocab_size)

        model.config.vocab_size = target_vocab_size

    return model


def is_valid_hf_model_dir(path: str) -> bool:
    """Return whether a directory contains a loadable Hugging Face model."""
    path = str(path)
    if not os.path.isdir(path):
        return False

    if not os.path.exists(os.path.join(path, "config.json")):
        return False

    weight_patterns = [
        "*.safetensors",
        "*.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    return any(glob.glob(os.path.join(path, pattern)) for pattern in weight_patterns)


def read_model_config_vocab_size(path: str) -> Optional[int]:
    cfg_path = os.path.join(str(path), "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        vocab_size = cfg.get("vocab_size")
        return int(vocab_size) if vocab_size is not None else None
    except Exception:
        return None


def prepare_resized_base_model_if_needed(
    *,
    original_base_model_path: str,
    tokenizer,
    target_vocab_size: int,
    output_path: str,
    trust_remote_code: bool = True,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    max_shard_size: str = "2GB",
    force_resize: bool = FORCE_RESIZE_BASE_MODEL,
) -> str:
    """Create or reuse a base model with the requested vocabulary size."""
    output_path = str(output_path)
    target_vocab_size = int(target_vocab_size)

    if not force_resize and is_valid_hf_model_dir(output_path):
        cached_vocab_size = read_model_config_vocab_size(output_path)
        if cached_vocab_size == target_vocab_size:
            print("=" * 100)
            print("[USE EXISTING RESIZED BASE MODEL]")
            print("output_path:", output_path)
            print("cached_vocab_size:", cached_vocab_size)
            return output_path
        print(
            "[IGNORE CACHED RESIZED BASE] vocab mismatch:",
            cached_vocab_size,
            "!=",
            target_vocab_size,
        )

    print("=" * 100)
    print("[CHECK BASE VOCAB]")
    print("original_base_model_path:", original_base_model_path)
    print("tokenizer_len:", len(tokenizer))
    print("target_vocab_size:", target_vocab_size)

    if device == "cpu":
        device_map = {"": "cpu"}
    elif device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    model = None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            original_base_model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()

        old_vocab_size = get_model_embedding_vocab_size(model)
        print("original_base_embedding_vocab_size:", old_vocab_size)

        if old_vocab_size == target_vocab_size:
            print("[BASE VOCAB OK] no resized base needed")
            return original_base_model_path

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)

        model = maybe_resize_model_embeddings_to_vocab(
            model,
            target_vocab_size=target_vocab_size,
            reason="base/tokenizer/adapter vocab mismatch",
        )

        print("[SAVE TEMP RESIZED BASE]", output_path)
        model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(output_path)
        return output_path
    finally:
        try:
            del model
        except Exception:
            pass
        cleanup_cuda()


def merge_peft_adapter_to_temp_full_model(
    *,
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    tokenizer,
    trust_remote_code: bool = True,
    merge_dtype: torch.dtype = torch.float16,
    merge_device: str = "cpu",
    max_shard_size: str = "2GB",
    force_remerge: bool = FORCE_REMERGE_MODEL,
) -> str:
    """Create or reuse a full model produced by merging a PEFT adapter."""
    output_path = str(output_path)
    expected_vocab_size = infer_adapter_saved_vocab_size(adapter_path)
    if expected_vocab_size is None:
        expected_vocab_size = int(len(tokenizer))
    expected_vocab_size = int(expected_vocab_size)

    if not force_remerge and is_valid_hf_model_dir(output_path):
        cached_vocab_size = read_model_config_vocab_size(output_path)
        if cached_vocab_size == expected_vocab_size:
            print("=" * 100)
            print("[USE EXISTING MERGED MODEL]")
            print("output_path:", output_path)
            print("cached_vocab_size:", cached_vocab_size)
            return output_path

        print("=" * 100)
        print("[IGNORE EXISTING MERGED MODEL]")
        print("output_path:", output_path)
        print("cached_vocab_size:", cached_vocab_size)
        print("expected_vocab_size:", expected_vocab_size)
        print("Reason: cached merged model may be stale; rebuilding.")

    adapter_type = read_peft_adapter_type(adapter_path)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    if merge_device == "cpu":
        device_map = {"": "cpu"}
    elif merge_device == "auto":
        device_map = "auto"
    else:
        device_map = {"": merge_device}

    print("=" * 100)
    print("[TEMP MERGE PEFT ADAPTER]")
    print("adapter_type:", adapter_type)
    print("base_model_path:", base_model_path)
    print("adapter_path:", adapter_path)
    print("output_path:", output_path)
    print("merge_dtype:", merge_dtype)
    print("merge_device:", merge_device)

    base_model = None
    peft_model = None
    merged_model = None

    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=merge_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        base_model.eval()

        adapter_saved_vocab_size = infer_adapter_saved_vocab_size(adapter_path)
        tokenizer_vocab_size = int(len(tokenizer))

        print("tokenizer_vocab_size:", tokenizer_vocab_size)
        print("adapter_saved_vocab_size:", adapter_saved_vocab_size)

        if adapter_saved_vocab_size is not None:
            if tokenizer_vocab_size != int(adapter_saved_vocab_size):
                raise ValueError(
                    f"Tokenizer length ({tokenizer_vocab_size}) != adapter saved vocab size "
                    f"({adapter_saved_vocab_size}). Use the tokenizer saved with the adapter training run."
                )

            base_model = maybe_resize_model_embeddings_to_vocab(
                base_model,
                target_vocab_size=int(adapter_saved_vocab_size),
                reason="adapter checkpoint contains resized embedding/lm_head tensors",
            )

        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            local_files_only=False,
        )
        peft_model.eval()

        merged_model = peft_model.merge_and_unload()
        merged_model.eval()

        print("[SAVE MERGED FULL MODEL]", output_path)
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        tokenizer.save_pretrained(output_path)
    finally:
        try:
            del merged_model
        except Exception:
            pass
        try:
            del peft_model
        except Exception:
            pass
        try:
            del base_model
        except Exception:
            pass
        cleanup_cuda()

    print("[TEMP MERGE DONE]", output_path)
    return output_path



@contextlib.contextmanager
def _noop_suppress_stdout():
    yield


def clean_vllm_text(text: str) -> str:
    text = "" if text is None else str(text)
    for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>", "<|im_start|>"]:
        text = text.replace(tok, "")
    return text.strip()


def get_token_id_if_exists(tokenizer, token: str):
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is None:
            return None
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    return None


def build_exclude_token_ids(tokenizer):
    special_tokens = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|end|>",
        "<|im_start|>",
    ]

    ids = []

    for tok in special_tokens:
        tid = get_token_id_if_exists(tokenizer, tok)
        if tid is not None:
            ids.append(tid)

    if tokenizer.pad_token_id is not None:
        ids.append(int(tokenizer.pad_token_id))

    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))

    return sorted(set(ids))


def check_highlight_output(text: str) -> dict:
    clean = clean_vllm_text(text)
    lower = clean.lower()

    return {
        "format_ok": lower.startswith("highlight:"),
        "starts_with_highlight": lower.startswith("highlight:"),
        "has_special_token": any(tok in text for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>"]),
        "rough_sentence_count": clean.count("."),
    }


def check_both_output(text: str) -> dict:
    clean = clean_vllm_text(text)
    lower = clean.lower()

    has_categories = "categories:" in lower
    has_keywords = "keywords:" in lower

    cat_pos = lower.find("categories:")
    kw_pos = lower.find("keywords:")
    order_ok = cat_pos != -1 and kw_pos != -1 and cat_pos < kw_pos

    return {
        "format_ok": has_categories and has_keywords and order_ok,
        "has_categories": has_categories,
        "has_keywords": has_keywords,
        "order_ok": order_ok,
        "has_semicolon": ";" in clean,
        "has_special_token": any(tok in text for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>"]),
    }


def detect_repetition_issues(text: str) -> dict:
    """
    Detect basic repetition degeneration patterns.
    """
    clean = clean_vllm_text(text)
    words = re.findall(r"\b\w+\b", clean.lower())

    repeated_3gram = False
    repeated_4gram = False

    for n in [3, 4]:
        grams = [" ".join(words[i:i+n]) for i in range(max(0, len(words) - n + 1))]
        counts = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1

        if n == 3 and any(c >= 3 for c in counts.values()):
            repeated_3gram = True
        if n == 4 and any(c >= 2 for c in counts.values()):
            repeated_4gram = True

    return {
        "repeated_3gram": repeated_3gram,
        "repeated_4gram": repeated_4gram,
        "maybe_repetition_loop": repeated_3gram or repeated_4gram,
    }


def run_one_guardian_task_batch_vllm(
    *,
    task: str,
    ds,
    llm,
    tokenizer,
    n: int = 20,
    max_tokens: int | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    answer_repetition_penalty: float = 1.12,
    exclude_token_ids: list[int] | None = None,
    print_each: bool = True,
):
    """
    Run one Guardian inference task in batch.

    The implementation uses an answer-only repetition penalty, disables the
    built-in vLLM repetition penalty, and limits highlight generation length.
    """

    if exclude_token_ids is None:
        exclude_token_ids = []

    if len(ds) == 0:
        raise RuntimeError(f"Dataset for task={task!r} is empty.")

    n = min(int(n), len(ds))

    if max_tokens is None:
        max_tokens = 128 if task == "highlight" else 96

    prompts = []
    rows_meta = []

    for i in range(n):
        row = ds[i]
        input_ids = list(row["input_ids"])

        model_input_text = tokenizer.decode(
            input_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        prompts.append({
            "prompt_token_ids": input_ids
        })

        rows_meta.append({
            "task": task,
            "row_index": i,
            "source_id": row.get("source_id"),
            "title": row.get("title"),
            "url": row.get("url"),
            "published_at": row.get("published_at"),
            "guardian_section_id": row.get("guardian_section_id"),
            "guardian_section_name": row.get("guardian_section_name"),
            "expected_output_format": row.get("ExpectedOutputFormat"),
            "prompt_raw": row.get("PromptRaw"),
            "model_input_text": model_input_text,
            "input_tokens": len(input_ids),
        })

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,

        # Disable the built-in vLLM repetition penalty.
        # The built-in penalty also counts prompt and article tokens.
        repetition_penalty=1.0,

        stop=["<|im_end|>", "<|endoftext|>", "<|end|>"],
        skip_special_tokens=True,
        spaces_between_special_tokens=False,

        extra_args={
            "answer_only_repetition_penalty": {
                "penalty": float(answer_repetition_penalty),
                "answer_prefix_ids": [],
                "exclude_token_ids": exclude_token_ids,
            }
        },
    )

    vllm_outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    results = []

    for meta, out in zip(rows_meta, vllm_outputs):
        completion = out.outputs[0]
        generated_token_ids = list(completion.token_ids)

        generated_raw = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        generated_clean = tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        generated_clean = clean_vllm_text(generated_clean)

        if task == "highlight":
            check = check_highlight_output(generated_raw)
        elif task == "both":
            check = check_both_output(generated_raw)
        else:
            check = {"format_ok": None}

        repetition_check = detect_repetition_issues(generated_clean)

        result = {
            **meta,
            "generated_raw": generated_raw,
            "generated_clean": generated_clean,
            "output_tokens": len(generated_token_ids),
            "answer_repetition_penalty": answer_repetition_penalty,
            **check,
            **repetition_check,
        }

        results.append(result)

        if print_each:
            print("=" * 100)
            print(f"[{task.upper()}] row_index={meta['row_index']}")
            print("title:", meta["title"])
            print("source_id:", meta["source_id"])
            print("input_tokens:", meta["input_tokens"])
            print("output_tokens:", len(generated_token_ids))
            print("format_ok:", check.get("format_ok"))
            print("maybe_repetition_loop:", repetition_check.get("maybe_repetition_loop"))
            print("generated_clean:")
            print(generated_clean)

    return results


def _database_flag_is_true(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _article_source_id(article: Dict[str, Any]) -> str:
    return _stringify_text(article.get("id") or article.get("source_id"))


def _pending_articles_for_task(
    articles: Sequence[Dict[str, Any]],
    task: str,
) -> List[Dict[str, Any]]:
    if task not in GENERATION_TASKS:
        raise ValueError(f"Unsupported Guardian generation task: {task}")
    output_flag = f"has_{task}_output"
    ineligible_flag = f"{task}_ineligible"
    return [
        article
        for article in articles
        if not _database_flag_is_true(article.get(output_flag))
        and not _database_flag_is_true(article.get(ineligible_flag))
    ]


def _dataset_source_ids(dataset: Any) -> Set[str]:
    return {
        source_id
        for row in dataset
        if (source_id := _article_source_id(row))
    }


def _ineligible_task_statuses(
    *,
    task: str,
    requested_articles: Sequence[Dict[str, Any]],
    built_source_ids: Set[str],
) -> List[Dict[str, str]]:
    statuses: List[Dict[str, str]] = []
    seen: Set[str] = set()
    for article in requested_articles:
        source_id = _article_source_id(article)
        if not source_id or source_id in built_source_ids or source_id in seen:
            continue
        seen.add(source_id)
        statuses.append({
            "source_id": source_id,
            "task": task,
            "reason": "input_not_eligible_under_current_generation_rules",
            "eligibility_version": GENERATION_ELIGIBILITY_VERSION,
        })
    return statuses


def run_guardian_two_tasks_batch_vllm(
    *,
    articles,
    llm,
    tokenizer,
    n: int = 20,
    allowed_sections=None,
    require_allowed_section: bool = False,
    allow_summary_fallback: bool = True,
    article_max_tokens: int = 2500,
    body_max_tokens: int = 2000,
    min_article_tokens: int = 30,
    min_body_tokens: int = 30,
    highlight_max_tokens: int = 128,
    both_max_tokens: int = 96,
    answer_repetition_penalty: float = 1.12,
    exclude_token_ids: list[int] | None = None,
    print_each: bool = True,
    save_to_postgres: bool = True,
):
    """
    Run the pending Guardian tasks in one batch:
        1. highlight, when highlight rows are pending
        2. both, when category-and-keyword rows are pending

    The model is not reloaded. The existing llm instance and registered logits
    processor are reused.
    """

    if exclude_token_ids is None:
        exclude_token_ids = []

    if save_to_postgres:
        initialize_postgres_tables()

    if "build_guardian_inference_datasets" not in globals():
        raise NameError(
            "build_guardian_inference_datasets is unavailable. "
            "Run the Guardian inference dataset builder first."
        )

    if articles is None or len(articles) == 0:
        raise ValueError("articles is empty. Provide Guardian article records first.")

    article_rows = list(articles)
    highlight_articles = _pending_articles_for_task(article_rows, "highlight")
    both_articles = _pending_articles_for_task(article_rows, "both")

    outputs = build_guardian_inference_datasets(
        articles=article_rows,
        tokenizer=tokenizer,
        highlight_articles=highlight_articles,
        both_articles=both_articles,

        allowed_sections=allowed_sections,
        require_allowed_section=require_allowed_section,
        allow_summary_fallback=allow_summary_fallback,

        include_title_in_highlight_article=True,
        article_max_tokens=article_max_tokens,
        min_article_tokens=min_article_tokens,

        body_max_tokens=body_max_tokens,
        min_body_tokens=min_body_tokens,

        keep_system_text=True,
        keep_debug_fields=False,
        save_root=None,
    )

    ds_highlight = outputs["highlight"]["inference"]
    ds_both = outputs["both"]["inference"]

    ineligible_statuses = [
        *_ineligible_task_statuses(
            task="highlight",
            requested_articles=highlight_articles,
            built_source_ids=_dataset_source_ids(ds_highlight),
        ),
        *_ineligible_task_statuses(
            task="both",
            requested_articles=both_articles,
            built_source_ids=_dataset_source_ids(ds_both),
        ),
    ]
    if save_to_postgres and ineligible_statuses:
        save_generation_task_statuses(ineligible_statuses)

    print("=" * 120)
    print("[DATASET SIZE]")
    print("highlight rows:", len(ds_highlight))
    print("both rows:", len(ds_both))
    print("requested n:", n)

    if len(ds_highlight) < n:
        print(f"[WARN] Only {len(ds_highlight)} highlight samples are available.")

    if len(ds_both) < n:
        print(f"[WARN] Only {len(ds_both)} both-task samples are available.")

    all_results = []

    if len(ds_highlight) > 0:
        print("\n" + "#" * 120)
        print("[RUN HIGHLIGHT TASK]")
        print("#" * 120)

        highlight_results = run_one_guardian_task_batch_vllm(
            task="highlight",
            ds=ds_highlight,
            llm=llm,
            tokenizer=tokenizer,
            n=n,
            max_tokens=highlight_max_tokens,
            temperature=0.0,
            top_p=1.0,
            answer_repetition_penalty=answer_repetition_penalty,
            exclude_token_ids=exclude_token_ids,
            print_each=print_each,
        )

        all_results.extend(highlight_results)
    else:
        print("[SKIP HIGHLIGHT TASK] No pending highlight rows.")

    if len(ds_both) > 0:
        print("\n" + "#" * 120)
        print("[RUN BOTH TASK]")
        print("#" * 120)

        both_results = run_one_guardian_task_batch_vllm(
            task="both",
            ds=ds_both,
            llm=llm,
            tokenizer=tokenizer,
            n=n,
            max_tokens=both_max_tokens,
            temperature=0.0,
            top_p=1.0,
            answer_repetition_penalty=answer_repetition_penalty,
            exclude_token_ids=exclude_token_ids,
            print_each=print_each,
        )

        all_results.extend(both_results)
    else:
        print("[SKIP BOTH TASK] No pending category-and-keyword rows.")

    df = pd.DataFrame(all_results)
    if not df.empty:
        # A partial recovery omits the other task's validation fields; store
        # those unevaluated fields as NULL.
        for column in MODEL_OUTPUT_COLUMNS:
            if column not in df.columns:
                df[column] = None

    if df.empty:
        print("No pending Guardian generation tasks were produced.")
    else:
        print("\n" + "=" * 120)
        print("[SUMMARY]")
        print(df.groupby("task")["format_ok"].agg(["count", "sum", "mean"]))

        print("\n" + "=" * 120)
        print("[REPETITION SUMMARY]")
        print(df.groupby("task")["maybe_repetition_loop"].agg(["count", "sum", "mean"]))

    if save_to_postgres:
        save_model_outputs_to_postgres(df)

    return df, outputs


def run_guardian_generation_from_postgres(
    *,
    llm,
    tokenizer,
    limit: int = BATCH_N,
    source_ids: Optional[List[str]] = None,
    allowed_sections=None,
    require_allowed_section: bool = False,
    allow_summary_fallback: bool = True,
    article_max_tokens: int = 2500,
    body_max_tokens: int = 2000,
    min_article_tokens: int = 30,
    min_body_tokens: int = 30,
    highlight_max_tokens: int = HIGHLIGHT_MAX_TOKENS,
    both_max_tokens: int = BOTH_MAX_TOKENS,
    answer_repetition_penalty: float = ANSWER_REPETITION_PENALTY,
    exclude_token_ids: list[int] | None = None,
    print_each: bool = PRINT_EACH,
):
    """
    Load raw articles without complete model outputs, run both inference tasks,
    and insert only new output rows into PostgreSQL.
    """
    initialize_postgres_tables()
    articles = load_raw_articles_for_generation(limit=limit, source_ids=source_ids)

    if not articles:
        print("No raw articles require model generation.")
        return pd.DataFrame(columns=MODEL_OUTPUT_COLUMNS), None

    return run_guardian_two_tasks_batch_vllm(
        articles=articles,
        llm=llm,
        tokenizer=tokenizer,
        n=len(articles),
        allowed_sections=allowed_sections,
        require_allowed_section=require_allowed_section,
        allow_summary_fallback=allow_summary_fallback,
        article_max_tokens=article_max_tokens,
        body_max_tokens=body_max_tokens,
        min_article_tokens=min_article_tokens,
        min_body_tokens=min_body_tokens,
        highlight_max_tokens=highlight_max_tokens,
        both_max_tokens=both_max_tokens,
        answer_repetition_penalty=answer_repetition_penalty,
        exclude_token_ids=exclude_token_ids,
        print_each=print_each,
        save_to_postgres=True,
    )


class OpenAIEmbeddingClient:
    """Generate OpenAI embeddings in batches without a local file cache."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        batch_size: int = 128,
        max_retries: int = 5,
        sleep_base: float = 1.5,
    ):
        self.client = OpenAI()
        self.model = model
        self.dimensions = dimensions
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        self.sleep_base = float(sleep_base)

    def _normalize_text(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = text.replace("\n", " ")
        return re.sub(r"\s+", " ", text).strip()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts_norm = [self._normalize_text(text) for text in texts]
        if not texts_norm:
            return []

        print(f"[embedding] requesting: {len(texts_norm)}")
        output_vectors: List[List[float]] = []

        for start in tqdm(
            range(0, len(texts_norm), self.batch_size),
            desc="OpenAI embeddings",
        ):
            batch_texts = texts_norm[start:start + self.batch_size]
            kwargs = {
                "model": self.model,
                "input": batch_texts,
                "encoding_format": "float",
            }

            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions

            response = None
            for attempt in range(self.max_retries):
                try:
                    response = self.client.embeddings.create(**kwargs)
                    break
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        raise
                    sleep_seconds = self.sleep_base ** attempt
                    print(
                        f"[WARN] embedding failed; retry in "
                        f"{sleep_seconds:.1f}s: {repr(exc)}"
                    )
                    time.sleep(sleep_seconds)

            vectors = [item.embedding for item in response.data]
            if len(vectors) != len(batch_texts):
                raise RuntimeError(
                    f"Embedding count mismatch: got {len(vectors)}, "
                    f"expected {len(batch_texts)}"
                )

            output_vectors.extend(vectors)

        return output_vectors

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

def clean_generated_highlight(text: str) -> str:
    text = "" if text is None else str(text)

    for tok in ["<|im_end|>", "<|endoftext|>", "<|end|>", "<|im_start|>"]:
        text = text.replace(tok, "")

    text = text.strip()

    if text.lower().startswith("highlight:"):
        text = text[len("highlight:"):].strip()

    return text


def extract_article_text_from_prompt_raw(prompt_raw: str) -> str:
    """
    prompt_raw normally has this structure:
        Article:
        title

        body...
    """
    prompt_raw = "" if prompt_raw is None else str(prompt_raw)
    text = prompt_raw.strip()

    if text.startswith("Article:"):
        text = text[len("Article:"):].strip()

    return text


def split_highlight_into_claim_sentences(highlight_text: str) -> List[str]:
    text = clean_generated_highlight(highlight_text)

    # Lightweight English sentence splitting for the current proof of concept.
    parts = re.split(r"(?<=[.!?])\s+", text)

    claims = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        claims.append(p)

    return claims


def split_article_into_sentences(article_text: str) -> List[str]:
    """
    Lightweight sentence splitting for Guardian live blogs.
    Minute markers such as 45+3 mins: or 90 min: are treated as boundaries.
    """
    text = "" if article_text is None else str(article_text)

    # Normalize whitespace before sentence span processing.
    text = re.sub(r"\s+", " ", text).strip()

    # Insert a boundary before live-blog minute markers.
    text = re.sub(
        r"\s+(?=(?:\d{1,3}(?:\+\d+)?\s*mins?:|\d{1,3}\s*min(?:\s*\+\d+)?))",
        " <SPLIT_HERE> ",
        text,
        flags=re.IGNORECASE,
    )

    rough_blocks = [b.strip() for b in text.split("<SPLIT_HERE>") if b.strip()]

    sentences = []
    for block in rough_blocks:
        parts = re.split(r"(?<=[.!?])\s+", block)

        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p.split()) < 3:
                continue
            sentences.append(p)

    return sentences


def find_sentence_char_span(article_text: str, sentence: str, search_start: int = 0):
    article_text = "" if article_text is None else str(article_text)
    sentence = "" if sentence is None else str(sentence)

    idx = article_text.find(sentence, search_start)

    if idx == -1:
        # Some spans may be unavailable after whitespace normalization.
        return -1, -1

    return idx, idx + len(sentence)


def take_tail_words(text: str, ratio: float) -> str:
    if ratio <= 0:
        return ""

    words = str(text).split()
    if not words:
        return ""

    n = max(1, math.ceil(len(words) * ratio))
    return " ".join(words[-n:])


def take_head_words(text: str, ratio: float) -> str:
    if ratio <= 0:
        return ""

    words = str(text).split()
    if not words:
        return ""

    n = max(1, math.ceil(len(words) * ratio))
    return " ".join(words[:n])


def extract_minute_marker(sentence: str) -> str:
    sentence = "" if sentence is None else str(sentence).strip()

    m = re.match(
        r"^(\d{1,3}(?:\+\d+)?\s*mins?:|\d{1,3}\s*min(?:\s*\+\d+)?)",
        sentence,
        flags=re.IGNORECASE,
    )

    return m.group(1) if m else ""


def build_sentence_chunks(
    *,
    article_text: str,
    article_id: str,
    source_id: str,
    title: str,
    overlap_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Each chunk is centered on one source sentence.

    chunk_text:
        The core source sentence shown as evidence.

    search_text:
        Text used for Weaviate hybrid search.
        When overlap_ratio is 0, search_text equals chunk_text.
        When overlap_ratio is 0.1, neighboring sentence fragments are included.
    """
    assert 0.0 <= overlap_ratio <= 1.0

    sentences = split_article_into_sentences(article_text)

    chunks = []
    cursor = 0

    for i, sent in enumerate(sentences):
        prev_sent = sentences[i - 1] if i > 0 else ""
        next_sent = sentences[i + 1] if i + 1 < len(sentences) else ""

        before_context = take_tail_words(prev_sent, overlap_ratio)
        after_context = take_head_words(next_sent, overlap_ratio)

        search_parts = []
        if before_context:
            search_parts.append(before_context)
        search_parts.append(sent)
        if after_context:
            search_parts.append(after_context)

        search_text = " ".join(search_parts).strip()

        start_char, end_char = find_sentence_char_span(article_text, sent, cursor)
        if end_char != -1:
            cursor = end_char

        chunks.append({
            "article_id": article_id,
            "source_id": source_id,
            "title": title,
            "chunk_id": f"{article_id}::sent::{i}",
            "sentence_id": int(i),
            "chunk_text": sent,
            "search_text": search_text,
            "start_char": int(start_char),
            "end_char": int(end_char),
            "minute_marker": extract_minute_marker(sent),
            "overlap_ratio": float(overlap_ratio),
        })

    return chunks


def get_or_create_guardian_evidence_collection(
    client,
    collection_name: str = COLLECTION_NAME,
):
    if client.collections.exists(collection_name):
        print(f"Using existing collection: {collection_name}")
        return client.collections.use(collection_name)

    collection = client.collections.create(
        name=collection_name,

        # This Weaviate Cloud cluster supports hfresh rather than hnsw.
        # Embeddings are generated externally and supplied with each object.
        vector_config=Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hfresh(
                distance_metric=VectorDistances.COSINE
            )
        ),

        properties=[
            Property(
                name="article_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="source_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="title",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="chunk_id",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="sentence_id",
                data_type=DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="chunk_text",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(
                name="search_text",
                data_type=DataType.TEXT,
                tokenization=Tokenization.WORD,
                index_filterable=False,
                index_searchable=True,
            ),
            Property(
                name="start_char",
                data_type=DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="end_char",
                data_type=DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="minute_marker",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=True,
            ),
            Property(
                name="overlap_ratio",
                data_type=DataType.NUMBER,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="embedding_model",
                data_type=DataType.TEXT,
                tokenization=Tokenization.FIELD,
                index_filterable=True,
                index_searchable=False,
            ),
            Property(
                name="embedding_dimensions",
                data_type=DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
        ],
    )

    print(f"Created collection: {collection_name}")
    return collection


def _safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def hybrid_search_one_claim_with_vector(
    collection,
    *,
    claim_sentence: str,
    query_vector: List[float],
    article_id: Optional[str] = None,
    top_k: int = 8,
    hybrid_alpha: float = 0.5,
    query_properties: Optional[List[str]] = None,
    min_score: Optional[float] = None,
    max_retries: int = WEAVIATE_HFRESH_QUERY_MAX_RETRIES,
    sleep_seconds: int = WEAVIATE_HFRESH_QUERY_RETRY_SLEEP_SECONDS,
) -> List[Dict[str, Any]]:
    """
    Run one Weaviate hybrid search with a precomputed query vector.

    This function is designed for concurrent retrieval:
    - query embeddings are computed once in batch before threading;
    - each thread only performs the Weaviate hybrid query.
    """
    if query_properties is None:
        query_properties = ["search_text", "chunk_text", "title"]

    claim_sentence = _safe_str(claim_sentence)
    if not claim_sentence:
        return []

    if query_vector is None:
        raise ValueError("query_vector is None.")

    filters = None
    if article_id:
        filters = Filter.by_property("article_id").equal(article_id)

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = collection.query.hybrid(
                query=claim_sentence,
                vector=query_vector,
                alpha=hybrid_alpha,
                query_properties=query_properties,
                filters=filters,
                limit=top_k,
                return_metadata=MetadataQuery(score=True, explain_score=True),
            )

            rows = []

            for obj in response.objects:
                score = obj.metadata.score

                if min_score is not None and score is not None and score < min_score:
                    continue

                props = obj.properties

                rows.append(
                    {
                        "claim_sentence": claim_sentence,
                        "article_id": props.get("article_id"),
                        "source_id": props.get("source_id"),
                        "title": props.get("title"),
                        "chunk_id": props.get("chunk_id"),
                        "sentence_id": props.get("sentence_id"),
                        "chunk_text": props.get("chunk_text"),
                        "search_text": props.get("search_text"),
                        "start_char": props.get("start_char"),
                        "end_char": props.get("end_char"),
                        "minute_marker": props.get("minute_marker"),
                        "overlap_ratio": props.get("overlap_ratio"),
                        "score": score,
                        "explain_score": obj.metadata.explain_score,
                    }
                )

            return rows

        except WeaviateQueryError as error:
            last_error = error
            message = str(error)

            if "HFRESH distancer is not yet initialized" not in message:
                raise

            if attempt >= max_retries:
                break

            print(
                f"Weaviate HFresh index is not ready. "
                f"Retry {attempt}/{max_retries} after {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)

    raise last_error






def build_claim_search_jobs_for_batch(
    *,
    samples: List[Dict[str, Any]],
    embedder,
) -> List[Dict[str, Any]]:
    """
    Build all claim-level search jobs and batch-embed all claim queries.

    This removes per-claim OpenAI embedding calls from the threaded search path.
    """
    jobs: List[Dict[str, Any]] = []

    for sample in samples:
        sample_row_index = int(sample.get("sample_row_index", -1))
        article_id = _safe_str(sample.get("article_id", ""))
        source_id = _safe_str(sample.get("source_id", ""))
        title = _safe_str(sample.get("title", ""))

        for claim_idx, claim in enumerate(sample.get("claim_sentences", []) or []):
            claim_sentence = _safe_str(claim)
            if not claim_sentence:
                continue

            jobs.append({
                "job_id": len(jobs),
                "sample_row_index": sample_row_index,
                "article_id": article_id,
                "source_id": source_id,
                "title": title,
                "claim_idx": int(claim_idx),
                "claim_sentence": claim_sentence,
            })

    if not jobs:
        print("[CLAIM SEARCH JOBS] no non-empty claims found.")
        return jobs

    print("=" * 120)
    print("[BATCH QUERY EMBEDDINGS]")
    print("claim search jobs:", len(jobs))
    print("embedding model:", getattr(embedder, "model", ""))
    print("embedding batch_size:", getattr(embedder, "batch_size", ""))

    query_vectors = embedder.embed_texts([job["claim_sentence"] for job in jobs])

    if len(query_vectors) != len(jobs):
        raise RuntimeError(
            f"Query vector count mismatch: got {len(query_vectors)}, expected {len(jobs)}"
        )

    for job, query_vector in zip(jobs, query_vectors):
        job["query_vector"] = query_vector

    return jobs


def _evidence_rows_from_search_results(
    *,
    job: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []

    for rank, item in enumerate(results, start=1):
        row = dict(item)
        row["sample_row_index"] = int(job["sample_row_index"])
        row["claim_idx"] = int(job["claim_idx"])
        row["rank"] = int(rank)
        row["raw_rank"] = int(rank)
        rows.append(row)

    return rows


def run_concurrent_claim_hybrid_search(
    *,
    collection,
    jobs: List[Dict[str, Any]],
    top_k: int,
    hybrid_alpha: float,
    min_score: Optional[float],
    max_workers: int = 4,
    query_properties: Optional[List[str]] = None,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Run Weaviate hybrid search concurrently at claim-job granularity.

    Input jobs must already contain precomputed query_vector.
    Returns grouped evidence rows and per-claim failures.
    """
    evidence_rows_by_sample: Dict[int, List[Dict[str, Any]]] = {}
    failed_claim_rows: List[Dict[str, Any]] = []

    if not jobs:
        return {
            "evidence_rows_by_sample": evidence_rows_by_sample,
            "failed_claim_rows": failed_claim_rows,
        }

    max_workers = max(1, int(max_workers or 1))
    max_workers = min(max_workers, len(jobs))

    print("=" * 120)
    print("[CONCURRENT WEAVIATE CLAIM SEARCH]")
    print("jobs:", len(jobs))
    print("max_workers:", max_workers)
    print("top_k:", top_k)
    print("hybrid_alpha:", hybrid_alpha)
    print("min_score:", min_score)

    def run_one(job: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        results = hybrid_search_one_claim_with_vector(
            collection,
            claim_sentence=job["claim_sentence"],
            query_vector=job["query_vector"],
            article_id=job.get("article_id"),
            top_k=top_k,
            hybrid_alpha=hybrid_alpha,
            query_properties=query_properties,
            min_score=min_score,
        )
        rows = _evidence_rows_from_search_results(job=job, results=results)
        return job, rows

    if max_workers == 1:
        iterator = tqdm(jobs, desc="Weaviate claim search serial")
        for job in iterator:
            try:
                finished_job, rows = run_one(job)
                sample_row_index = int(finished_job["sample_row_index"])
                evidence_rows_by_sample.setdefault(sample_row_index, []).extend(rows)
            except Exception as exc:
                failure = {
                    "sample_row_index": int(job.get("sample_row_index", -1)),
                    "article_id": _safe_str(job.get("article_id", "")),
                    "claim_idx": int(job.get("claim_idx", -1)),
                    "claim_sentence": _safe_str(job.get("claim_sentence", "")),
                    "error": repr(exc),
                }
                failed_claim_rows.append(failure)
                print("[CLAIM SEARCH FAILED]", failure)
                if fail_fast:
                    raise
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_job = {
                executor.submit(run_one, job): job
                for job in jobs
            }

            for future in tqdm(
                as_completed(future_to_job),
                total=len(future_to_job),
                desc=f"Weaviate claim search x{max_workers}",
            ):
                job = future_to_job[future]

                try:
                    finished_job, rows = future.result()
                    sample_row_index = int(finished_job["sample_row_index"])
                    evidence_rows_by_sample.setdefault(sample_row_index, []).extend(rows)
                except Exception as exc:
                    failure = {
                        "sample_row_index": int(job.get("sample_row_index", -1)),
                        "article_id": _safe_str(job.get("article_id", "")),
                        "claim_idx": int(job.get("claim_idx", -1)),
                        "claim_sentence": _safe_str(job.get("claim_sentence", "")),
                        "error": repr(exc),
                    }
                    failed_claim_rows.append(failure)
                    print("[CLAIM SEARCH FAILED]", failure)
                    if fail_fast:
                        raise

    for sample_row_index, rows in evidence_rows_by_sample.items():
        rows.sort(key=lambda r: (int(r.get("claim_idx", 0)), int(r.get("rank", 0))))

    print("=" * 120)
    print("[CONCURRENT WEAVIATE CLAIM SEARCH DONE]")
    print("samples with evidence rows:", len(evidence_rows_by_sample))
    print("failed claim jobs:", len(failed_claim_rows))

    return {
        "evidence_rows_by_sample": evidence_rows_by_sample,
        "failed_claim_rows": failed_claim_rows,
    }


def evidence_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    evidence_df = pd.DataFrame(rows or [])

    if evidence_df.empty:
        return evidence_df

    sort_cols = [c for c in ["claim_idx", "rank"] if c in evidence_df.columns]
    if sort_cols:
        evidence_df = evidence_df.sort_values(sort_cols).reset_index(drop=True)

    return evidence_df


# Single-sample retrieval helpers were removed intentionally.
# Batch retrieval starts below and is the only retrieval path used by this script.

def print_retrieval_summary(
    packet: Dict[str, Any],
    top_n_per_claim: int = 3,
    print_search_text: bool = False,
) -> None:
    """
    Human-readable retrieval preview for one packet.
    """
    print("=" * 100)
    print("Retrieval summary")
    print("title:", packet.get("title", ""))
    print("source_id:", packet.get("source_id", ""))
    print("article_id:", packet.get("article_id", ""))

    for display_idx, claim in enumerate(packet.get("claims", []) or [], start=1):
        print("-" * 100)
        print(f"Claim {display_idx}: {claim.get('claim_sentence', '')}")
        evidence_items = (claim.get("evidence_candidates", []) or [])[:top_n_per_claim]
        if not evidence_items:
            print("No retrieved evidence after score filtering.")
            continue

        for evidence in evidence_items:
            print(
                f"  rank={evidence.get('rank')} "
                f"score={evidence.get('score')} "
                f"sentence_id={evidence.get('sentence_id')}"
            )
            print("     ", str(evidence.get("chunk_text", "")).strip())
            if print_search_text:
                print("      search_text:", str(evidence.get("search_text", "")).strip())

# ======================================================================================
# Batch highlight retrieval pipeline
# ======================================================================================

def build_retrieval_packet(
    *,
    sample: dict,
    evidence_df: pd.DataFrame,
    top_n_per_claim: int = 5,
):
    """
    Build a retrieval packet without requiring evidence_df to be non-empty.

    If a claim has no evidence after min_score filtering, its
    evidence_candidates list remains empty.
    """
    claims = []

    if evidence_df is None:
        evidence_df = pd.DataFrame()

    has_claim_idx = (not evidence_df.empty) and ("claim_idx" in evidence_df.columns)

    for claim_idx, claim_sentence in enumerate(sample.get("claim_sentences", []) or []):
        if has_claim_idx:
            sub = (
                evidence_df[evidence_df["claim_idx"] == claim_idx]
                .sort_values("rank")
                .head(top_n_per_claim)
            )
        else:
            sub = pd.DataFrame()

        evidence_candidates = []

        for _, row in sub.iterrows():
            evidence_candidates.append({
                "rank": int(row["rank"]) if pd.notna(row.get("rank")) else None,
                "raw_rank": int(row["raw_rank"]) if "raw_rank" in row and pd.notna(row.get("raw_rank")) else None,
                "score": float(row["score"]) if "score" in row and pd.notna(row.get("score")) else None,
                "sentence_id": int(row["sentence_id"]) if "sentence_id" in row and pd.notna(row.get("sentence_id")) else None,
                "minute_marker": _safe_str(row.get("minute_marker", "")),
                "chunk_text": _safe_str(row.get("chunk_text", "")),
                "search_text": _safe_str(row.get("search_text", "")),
                "start_char": int(row["start_char"]) if "start_char" in row and pd.notna(row.get("start_char")) else None,
                "end_char": int(row["end_char"]) if "end_char" in row and pd.notna(row.get("end_char")) else None,
            })

        claims.append({
            "claim_idx": int(claim_idx),
            "claim_sentence": _safe_str(claim_sentence),
            "evidence_candidates": evidence_candidates,
        })

    return {
        "title": _safe_str(sample.get("title", "")),
        "source_id": _safe_str(sample.get("source_id", "")),
        "article_id": _safe_str(sample.get("article_id", "")),
        "data_source": _safe_str(sample.get("data_source", "")),
        "sample_row_index": int(sample.get("sample_row_index", -1)),
        "original_article": _safe_str(sample.get("article_text", "")),
        "generated_highlight": _safe_str(sample.get("generated_highlight", "")),
        "claims": claims,
    }


def build_highlight_sample_from_output_row(
    *,
    row: Dict[str, Any],
    data_source: str,
    sample_row_index: int,
) -> Dict[str, Any]:
    """
    Build one highlight retrieval sample from a PostgreSQL model output row.
    """
    title = _safe_str(row.get("title", ""))
    source_id = _safe_str(row.get("source_id", ""))
    article_id = source_id if source_id else str(uuid.uuid4())

    prompt_raw = _safe_str(row.get("prompt_raw", ""))
    article_text = extract_article_text_from_prompt_raw(prompt_raw)

    generated_highlight = _safe_str(row.get("generated_clean", ""))
    claim_sentences = split_highlight_into_claim_sentences(generated_highlight)

    original_output_row_index_raw = row.get("original_output_row_index", -1)
    original_output_row_index_text = str(original_output_row_index_raw).strip()
    original_output_row_index = (
        int(original_output_row_index_raw)
        if original_output_row_index_text not in {"", "nan", "None"}
        else -1
    )

    return {
        "data_source": str(data_source),
        "sample_row_index": int(sample_row_index),
        "original_output_row_index": original_output_row_index,
        "row": row,
        "title": title,
        "source_id": source_id,
        "article_id": article_id,
        "prompt_raw": prompt_raw,
        "article_text": article_text,
        "generated_highlight": generated_highlight,
        "claim_sentences": claim_sentences,
    }


def _series_truthy_for_filter_ok(series: pd.Series) -> pd.Series:
    """
    Parse boolean-like values returned from PostgreSQL or pandas.
    """
    return series.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y"})


def load_highlight_samples_from_postgres(
    *,
    source_ids: Optional[List[str]] = None,
    row_indices: Optional[List[int]] = None,
    max_samples: Optional[int] = None,
    require_format_ok: bool = False,
) -> Dict[str, Any]:
    """
    Load selected highlight samples from the model_outputs PostgreSQL table.

    sample_row_index is assigned after ordering the selected highlight rows.
    """
    data_source = f"postgresql:{MODEL_OUTPUTS_TABLE}"
    df_outputs = load_model_outputs_from_postgres(
        task="highlight",
        source_ids=source_ids,
    )

    required_columns = ["task", "title", "source_id", "prompt_raw", "generated_clean"]
    missing_columns = [column for column in required_columns if column not in df_outputs.columns]
    if missing_columns:
        raise ValueError(
            f"PostgreSQL model outputs are missing columns: {missing_columns}. "
            f"Existing columns: {list(df_outputs.columns)}"
        )

    if df_outputs.empty:
        raise ValueError("PostgreSQL does not contain any selected highlight samples.")

    df_outputs = df_outputs.copy()
    df_outputs["original_output_row_index"] = list(range(len(df_outputs)))

    highlight_df = df_outputs[df_outputs["task"] == "highlight"].reset_index(drop=True)
    if highlight_df.empty:
        raise ValueError("PostgreSQL does not contain highlight samples.")

    highlight_df["sample_row_index"] = list(range(len(highlight_df)))

    if require_format_ok:
        if "format_ok" not in highlight_df.columns:
            raise ValueError(
                "require_format_ok=True, but model_outputs does not contain format_ok."
            )
        highlight_df = highlight_df[
            _series_truthy_for_filter_ok(highlight_df["format_ok"])
        ].reset_index(drop=True)

    available_indices = highlight_df["sample_row_index"].astype(int).tolist()

    if row_indices is not None:
        wanted = [int(value) for value in row_indices]
        available_set = set(available_indices)
        selected_indices = [value for value in wanted if value in available_set]
        missing_requested = [value for value in wanted if value not in available_set]
        if missing_requested:
            print(
                "[WARN] Requested sample_row_indices are unavailable and were skipped:",
                missing_requested,
            )
    else:
        selected_indices = available_indices

    if max_samples is not None:
        selected_indices = selected_indices[: int(max_samples)]

    selected_rows = []
    for sample_row_index in selected_indices:
        matched = highlight_df[
            highlight_df["sample_row_index"] == int(sample_row_index)
        ]
        if matched.empty:
            continue
        selected_rows.append(matched.iloc[0].to_dict())

    samples = [
        build_highlight_sample_from_output_row(
            row=row,
            data_source=data_source,
            sample_row_index=int(row["sample_row_index"]),
        )
        for row in selected_rows
    ]

    return {
        "data_source": data_source,
        "df_outputs": df_outputs,
        "highlight_df": highlight_df,
        "samples": samples,
        "selected_indices": selected_indices,
    }


def build_chunks_for_batch_samples(
    *,
    samples: List[Dict[str, Any]],
    overlap_ratio: float,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Build source chunks for selected samples.

    A bad article should not kill the whole batch unless fail_fast=True.
    """
    all_chunks: List[Dict[str, Any]] = []
    chunks_by_row_index: Dict[int, List[Dict[str, Any]]] = {}
    samples_with_chunks: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []

    for sample in samples:
        sample_row_index = int(sample["sample_row_index"])

        try:
            chunks = build_sentence_chunks(
                article_text=sample["article_text"],
                article_id=sample["article_id"],
                source_id=sample["source_id"],
                title=sample["title"],
                overlap_ratio=overlap_ratio,
            )

            if not chunks:
                raise RuntimeError("article_text did not produce any valid chunks.")

            enriched_chunks = []
            for chunk in chunks:
                c = dict(chunk)
                c["sample_row_index"] = sample_row_index
                enriched_chunks.append(c)
                all_chunks.append(c)

            chunks_by_row_index[sample_row_index] = enriched_chunks
            samples_with_chunks.append(sample)

        except Exception as exc:
            row = {
                "sample_row_index": sample_row_index,
                "status": "failed_chunk_build",
                "error": repr(exc),
                "title": _safe_str(sample.get("title", "")),
                "source_id": _safe_str(sample.get("source_id", "")),
                "article_id": _safe_str(sample.get("article_id", "")),
                "num_claims": len(sample.get("claim_sentences", []) or []),
                "num_evidence_rows": None,
                "num_chunks": 0,
                "collection_count": None,
                "packet_json": "",
                "evidence_csv": "",
                "chunks_csv": "",
            }
            failed_rows.append(row)
            print(f"[FAILED CHUNK BUILD] sample_row_index={sample_row_index}: {repr(exc)}")
            if fail_fast:
                raise

    return {
        "all_chunks": all_chunks,
        "chunks_by_row_index": chunks_by_row_index,
        "samples_with_chunks": samples_with_chunks,
        "failed_rows": failed_rows,
    }


def get_existing_weaviate_chunk_ids(collection) -> Set[str]:
    """Read all stored chunk identifiers without loading stored vectors."""
    chunk_ids: Set[str] = set()

    for obj in collection.iterator(include_vector=False):
        properties = obj.properties or {}
        chunk_id = _safe_str(properties.get("chunk_id"))
        if chunk_id:
            chunk_ids.add(chunk_id)

    print(f"[WEAVIATE] existing chunk ids: {len(chunk_ids)}")
    return chunk_ids


def ingest_batch_chunks_to_weaviate(
    *,
    client,
    collection_name: str,
    embedder,
    chunks: List[Dict[str, Any]],
    embedding_model: str | None = None,
):
    """Embed and insert only chunks that are not already stored in Weaviate."""
    if embedding_model is None:
        embedding_model = globals().get(
            "EMBEDDING_MODEL",
            "text-embedding-3-small",
        )

    if not chunks:
        raise ValueError("No chunks to ingest.")

    collection = get_or_create_guardian_evidence_collection(
        client,
        collection_name,
    )
    existing_chunk_ids = get_existing_weaviate_chunk_ids(collection)

    unique_chunks: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = _safe_str(chunk.get("chunk_id"))
        if not chunk_id:
            raise ValueError("A chunk is missing chunk_id.")
        if chunk_id not in unique_chunks:
            unique_chunks[chunk_id] = chunk

    missing_chunks = [
        chunk
        for chunk_id, chunk in unique_chunks.items()
        if chunk_id not in existing_chunk_ids
    ]

    print(f"[WEAVIATE] current unique chunks: {len(unique_chunks)}")
    print(f"[WEAVIATE] missing chunks: {len(missing_chunks)}")

    if missing_chunks:
        search_texts = [
            _safe_str(chunk.get("search_text", ""))
            for chunk in missing_chunks
        ]
        vectors = embedder.embed_texts(search_texts)

        if len(vectors) != len(missing_chunks):
            raise RuntimeError(
                f"Vector count mismatch: {len(vectors)} "
                f"vs {len(missing_chunks)}"
            )

        vector_dim = len(vectors[0]) if vectors else 0
        allowed_weaviate_props = {
            "article_id",
            "source_id",
            "title",
            "chunk_id",
            "sentence_id",
            "chunk_text",
            "search_text",
            "start_char",
            "end_char",
            "minute_marker",
            "overlap_ratio",
        }

        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(missing_chunks, vectors):
                properties = {
                    key: value
                    for key, value in chunk.items()
                    if key in allowed_weaviate_props
                }
                properties["embedding_model"] = embedding_model
                properties["embedding_dimensions"] = vector_dim

                chunk_id = _safe_str(chunk.get("chunk_id"))
                object_uuid = generate_uuid5({
                    "collection": collection_name,
                    "chunk_id": chunk_id,
                })

                batch.add_object(
                    properties=properties,
                    vector=vector,
                    uuid=object_uuid,
                )

        failed = collection.batch.failed_objects
        if failed:
            raise RuntimeError(
                f"Weaviate batch insert failed for {len(failed)} "
                f"objects: {failed[:3]}"
            )
    else:
        print("[WEAVIATE] No new source chunk embeddings are required.")

    count = collection.aggregate.over_all(total_count=True).total_count
    print(f"[WEAVIATE BATCH INGEST DONE] collection_count={count}")
    return collection, count


def unload_vllm_model(llm_obj=None, *, name: str = "llm"):
    """
    Best-effort vLLM cleanup for Colab.

    vLLM can hold GPU memory through engine/executor objects even after `del llm`.
    This function tries the known shutdown paths, removes the global reference,
    and then clears CUDA cache. It intentionally does NOT call
    destroy_distributed_environment(), which can poison later vLLM initialization
    in notebooks.
    """
    print("=" * 120)
    print("[UNLOAD VLLM MODEL]")

    if llm_obj is None:
        llm_obj = globals().get(name)

    if llm_obj is None:
        print("No vLLM object to unload.")
        cleanup_cuda()
        return None

    # Try common shutdown/close hooks across vLLM versions.
    candidates = [llm_obj]
    engine = getattr(llm_obj, "llm_engine", None)
    if engine is not None:
        candidates.append(engine)
        for attr in ["model_executor", "engine_core", "executor"]:
            child = getattr(engine, attr, None)
            if child is not None:
                candidates.append(child)

    for obj in candidates:
        for method_name in ["shutdown", "close"]:
            method = getattr(obj, method_name, None)
            if callable(method):
                try:
                    method()
                    print(f"called {type(obj).__name__}.{method_name}()")
                except Exception as exc:
                    print(f"[WARN] {type(obj).__name__}.{method_name}() failed: {repr(exc)}")

    # Clear vLLM model-parallel state only; avoid destroying the whole distributed env.
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        print("called destroy_model_parallel()")
    except Exception as exc:
        print("[WARN] destroy_model_parallel skipped:", repr(exc))

    try:
        if name in globals():
            del globals()[name]
    except Exception:
        pass

    try:
        del llm_obj
    except Exception:
        pass

    cleanup_cuda()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        cleanup_cuda()
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"CUDA memory after unload: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
        except Exception:
            pass

    return None


def _resolve_output_path(save_dir: str, configured_path: str) -> str:
    """
    Keep absolute configured paths as-is; resolve relative paths under save_dir.
    """
    configured_path = str(configured_path)
    if os.path.isabs(configured_path):
        return configured_path
    return os.path.join(str(save_dir), configured_path)


def get_batch_retrieval_output_paths(save_dir: str) -> Dict[str, str]:
    """
    Aggregate retrieval output paths: one file per output type.
    """
    return {
        "summary_csv": _resolve_output_path(save_dir, RETRIEVAL_BATCH_SUMMARY_CSV_PATH),
        "summary_json": _resolve_output_path(save_dir, RETRIEVAL_BATCH_SUMMARY_JSON_PATH),
        "packets_jsonl": _resolve_output_path(save_dir, RETRIEVAL_BATCH_PACKETS_JSONL_PATH),
        "evidence_csv": _resolve_output_path(save_dir, RETRIEVAL_BATCH_EVIDENCE_CSV_PATH),
        "chunks_csv": _resolve_output_path(save_dir, RETRIEVAL_BATCH_CHUNKS_CSV_PATH),
    }




def load_packets_jsonl(path: str) -> List[Dict[str, Any]]:
    packets = []
    if not os.path.exists(path):
        return packets
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    packets.append(obj)
            except Exception:
                continue
    return packets


def aggregate_retrieval_outputs_exist(paths: Dict[str, str]) -> bool:
    """
    Return True only if aggregate retrieval outputs exist and are readable.

    This avoids skipping retrieval because of zero-byte/corrupt files left by an
    interrupted run. JSONL may legitimately be empty only when no packets exist,
    but the summary/evidence/chunks CSVs must be parseable.
    """
    required = [
        paths["summary_csv"],
        paths["summary_json"],
        paths["packets_jsonl"],
        paths["evidence_csv"],
        paths["chunks_csv"],
    ]

    if any(not os.path.exists(p) for p in required):
        return False

    try:
        pd.read_csv(paths["summary_csv"])
        pd.read_csv(paths["evidence_csv"])
        pd.read_csv(paths["chunks_csv"])
    except Exception as exc:
        print("[IGNORE EXISTING RETRIEVAL OUTPUTS] unreadable CSV:", repr(exc))
        return False

    try:
        with open(paths["summary_json"], "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as exc:
        print("[IGNORE EXISTING RETRIEVAL OUTPUTS] unreadable summary JSON:", repr(exc))
        return False

    try:
        with open(paths["packets_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    json.loads(line)
    except Exception as exc:
        print("[IGNORE EXISTING RETRIEVAL OUTPUTS] unreadable packets JSONL:", repr(exc))
        return False

    return True


def save_single_batch_retrieval_outputs(
    *,
    summary_rows: List[Dict[str, Any]],
    packet_objects: List[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    paths: Dict[str, str],
) -> pd.DataFrame:
    """
    Save aggregate retrieval outputs.
    One file per output type:
    - summary CSV
    - summary JSON
    - packets JSONL
    - evidence CSV
    - chunks CSV
    """
    for path in paths.values():
        parent = os.path.dirname(str(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(paths["summary_csv"], index=False)

    with open(paths["summary_json"], "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    with open(paths["packets_jsonl"], "w", encoding="utf-8") as f:
        for packet in packet_objects:
            f.write(json.dumps(packet, ensure_ascii=False) + "\n")

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv(paths["evidence_csv"], index=False)

    chunks_df = pd.DataFrame(chunks)
    chunks_df.to_csv(paths["chunks_csv"], index=False)

    print("=" * 120)
    print("[SAVED AGGREGATE RETRIEVAL OUTPUTS]")
    print("summary_csv:", paths["summary_csv"])
    print("summary_json:", paths["summary_json"])
    print("packets_jsonl:", paths["packets_jsonl"])
    print("evidence_csv:", paths["evidence_csv"])
    print("chunks_csv:", paths["chunks_csv"])
    print("summary rows:", len(summary_rows))
    print("packets:", len(packet_objects))
    print("evidence rows:", len(evidence_rows))
    print("chunks:", len(chunks))

    return summary_df


def run_batch_highlight_retrieval_pipeline(
    *,
    client,
    embedder,
    collection_name: str = COLLECTION_NAME,
    source_ids: Optional[List[str]] = None,
    row_indices: Optional[List[int]] = RETRIEVAL_BATCH_ROW_INDICES,
    max_samples: Optional[int] = RETRIEVAL_BATCH_MAX_SAMPLES,
    require_format_ok: bool = False,
    overlap_ratio: float = RETRIEVAL_OVERLAP_RATIO,
    top_k: int = RETRIEVAL_TOP_K,
    hybrid_alpha: float = RETRIEVAL_HYBRID_ALPHA,
    min_score: float | None = RETRIEVAL_MIN_SCORE,
    index_ready_wait_seconds: int = WEAVIATE_INDEX_READY_WAIT_SECONDS,
    top_n_to_print: int = RETRIEVAL_TOP_N_TO_PRINT,
    print_search_text: bool = RETRIEVAL_PRINT_SEARCH_TEXT,
    save_dir: str = RETRIEVAL_SAVE_DIR,
    skip_existing: bool = RETRIEVAL_BATCH_SKIP_EXISTING,
    fail_fast: bool = RETRIEVAL_BATCH_FAIL_FAST,
    search_max_workers: int = RETRIEVAL_SEARCH_MAX_WORKERS,
    summary_csv_path: str = RETRIEVAL_BATCH_SUMMARY_CSV_PATH,
    summary_json_path: str = RETRIEVAL_BATCH_SUMMARY_JSON_PATH,
    packets_jsonl_path: str = RETRIEVAL_BATCH_PACKETS_JSONL_PATH,
) -> Dict[str, Any]:
    """
    Concurrent batch retrieval with aggregate outputs only.

    The pipeline writes:
    - guardian_retrieval_batch_summary.csv
    - guardian_retrieval_batch_summary.json
    - guardian_retrieval_packets.jsonl
    - guardian_retrieved_evidence_batch.csv
    - guardian_source_chunks_batch.csv
    """
    initialize_postgres_tables()
    os.makedirs(save_dir, exist_ok=True)
    paths = get_batch_retrieval_output_paths(save_dir)
    data_source = f"postgresql:{MODEL_OUTPUTS_TABLE}"

    # Respect explicit paths from the caller while still keeping aggregate behavior.
    paths["summary_csv"] = _resolve_output_path(save_dir, summary_csv_path)
    paths["summary_json"] = _resolve_output_path(save_dir, summary_json_path)
    paths["packets_jsonl"] = _resolve_output_path(save_dir, packets_jsonl_path)

    if skip_existing and aggregate_retrieval_outputs_exist(paths):
        print("=" * 120)
        print("[SKIP EXISTING AGGREGATE RETRIEVAL OUTPUTS]")
        print("summary_csv:", paths["summary_csv"])
        print("packets_jsonl:", paths["packets_jsonl"])
        summary_df = pd.read_csv(paths["summary_csv"])
        packet_objects = load_packets_jsonl(paths["packets_jsonl"])
        failed_rows = summary_df[summary_df.get("status", "") != "ok"].to_dict("records") if not summary_df.empty and "status" in summary_df.columns else []
        return {
            "summary_df": summary_df,
            "rows": summary_df.to_dict("records"),
            "packet_json_paths": [paths["packets_jsonl"]],
            "packet_objects": packet_objects,
            "failed_rows": failed_rows,
            "data_source": data_source,
            "save_dir": save_dir,
            "aggregate_paths": paths,
        }

    loaded = load_highlight_samples_from_postgres(
        source_ids=source_ids,
        row_indices=row_indices,
        max_samples=max_samples,
        require_format_ok=require_format_ok,
    )

    data_source = loaded["data_source"]
    samples = loaded["samples"]
    selected_indices = loaded["selected_indices"]

    print("=" * 120)
    print("[BATCH HIGHLIGHT RETRIEVAL - AGGREGATE OUTPUTS]")
    print("data_source:", data_source)
    print("selected samples:", len(samples))
    print("selected sample_row_indices:", selected_indices)
    print("save_dir:", save_dir)
    print("skip_existing_whole_batch:", skip_existing)
    print("search_max_workers:", search_max_workers)
    print("aggregate evidence_csv:", paths["evidence_csv"])
    print("aggregate chunks_csv:", paths["chunks_csv"])
    print("aggregate packets_jsonl:", paths["packets_jsonl"])
    print("=" * 120)

    summary_rows: List[Dict[str, Any]] = []
    packet_objects: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    all_evidence_rows: List[Dict[str, Any]] = []

    if not samples:
        print("[BATCH RETRIEVAL] No selected samples. Saving empty aggregate outputs.")
        summary_df = save_single_batch_retrieval_outputs(
            summary_rows=summary_rows,
            packet_objects=packet_objects,
            evidence_rows=all_evidence_rows,
            chunks=[],
            paths=paths,
        )
        return {
            "summary_df": summary_df,
            "rows": summary_rows,
            "packet_json_paths": [paths["packets_jsonl"]],
            "packet_objects": packet_objects,
            "failed_rows": failed_rows,
            "data_source": data_source,
            "save_dir": save_dir,
            "aggregate_paths": paths,
        }

    print(f"[PROCESS] {len(samples)} samples need retrieval.")

    chunk_build = build_chunks_for_batch_samples(
        samples=samples,
        overlap_ratio=overlap_ratio,
        fail_fast=fail_fast,
    )

    all_chunks = chunk_build["all_chunks"]
    chunks_by_row_index = chunk_build["chunks_by_row_index"]
    samples_ready = chunk_build["samples_with_chunks"]
    chunk_failed_rows = chunk_build["failed_rows"]

    for row in chunk_failed_rows:
        row = dict(row)
        row["packet_json"] = paths["packets_jsonl"]
        row["evidence_csv"] = paths["evidence_csv"]
        row["chunks_csv"] = paths["chunks_csv"]
        summary_rows.append(row)
        failed_rows.append(row)

    print("[CHUNKS]")
    print("total chunks:", len(all_chunks))
    print("samples with chunks:", len(samples_ready))
    print("samples failed chunk build:", len(chunk_failed_rows))

    if not samples_ready or not all_chunks:
        print("[BATCH RETRIEVAL] No valid chunks to ingest. Saving aggregate outputs and stopping retrieval stage.")
        summary_df = save_single_batch_retrieval_outputs(
            summary_rows=summary_rows,
            packet_objects=packet_objects,
            evidence_rows=all_evidence_rows,
            chunks=all_chunks,
            paths=paths,
        )
        return {
            "summary_df": summary_df,
            "rows": summary_rows,
            "packet_json_paths": [paths["packets_jsonl"]],
            "packet_objects": packet_objects,
            "failed_rows": failed_rows,
            "data_source": data_source,
            "save_dir": save_dir,
            "aggregate_paths": paths,
        }

    collection, collection_count = ingest_batch_chunks_to_weaviate(
        client=client,
        collection_name=collection_name,
        embedder=embedder,
        chunks=all_chunks,
        embedding_model=globals().get("EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    if index_ready_wait_seconds and int(index_ready_wait_seconds) > 0:
        print(f"Waiting {index_ready_wait_seconds}s for Weaviate vector index to become ready.")
        time.sleep(int(index_ready_wait_seconds))

    claim_jobs = build_claim_search_jobs_for_batch(
        samples=samples_ready,
        embedder=embedder,
    )

    search_result = run_concurrent_claim_hybrid_search(
        collection=collection,
        jobs=claim_jobs,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        min_score=min_score,
        max_workers=search_max_workers,
        fail_fast=fail_fast,
    )

    evidence_rows_by_sample = search_result["evidence_rows_by_sample"]
    failed_claim_rows = search_result["failed_claim_rows"]

    failed_claims_by_sample: Dict[int, List[Dict[str, Any]]] = {}
    for failed_claim in failed_claim_rows:
        sample_row_index = int(failed_claim.get("sample_row_index", -1))
        failed_claims_by_sample.setdefault(sample_row_index, []).append(failed_claim)

    for pos, sample in enumerate(samples_ready, start=1):
        sample_row_index = int(sample["sample_row_index"])

        print("\n" + "=" * 120)
        print(f"[BUILD RETRIEVAL PACKET] {pos}/{len(samples_ready)} | sample_row_index={sample_row_index}")
        print("title:", sample.get("title", ""))
        print("source_id:", sample.get("source_id", ""))
        print("claims:", len(sample.get("claim_sentences", []) or []))
        print("=" * 120)

        sample_failed_claims = failed_claims_by_sample.get(sample_row_index, [])

        if sample_failed_claims:
            error_text = json.dumps(sample_failed_claims, ensure_ascii=False)
            print(f"[FAILED RETRIEVAL CLAIMS] sample_row_index={sample_row_index}")
            print(error_text)

            row_summary = {
                "sample_row_index": sample_row_index,
                "status": "failed_retrieval",
                "error": error_text,
                "title": _safe_str(sample.get("title", "")),
                "source_id": _safe_str(sample.get("source_id", "")),
                "article_id": _safe_str(sample.get("article_id", "")),
                "num_claims": len(sample.get("claim_sentences", []) or []),
                "num_failed_claims": len(sample_failed_claims),
                "num_evidence_rows": None,
                "num_chunks": len(chunks_by_row_index.get(sample_row_index, []) or []),
                "collection_count": int(collection_count),
                "search_max_workers": int(search_max_workers or 1),
                "packet_json": paths["packets_jsonl"],
                "evidence_csv": paths["evidence_csv"],
                "chunks_csv": paths["chunks_csv"],
            }
            summary_rows.append(row_summary)
            failed_rows.append(row_summary)

            if fail_fast:
                raise RuntimeError(error_text)

            continue

        try:
            sample_evidence_rows = evidence_rows_by_sample.get(sample_row_index, []) or []
            all_evidence_rows.extend(sample_evidence_rows)
            evidence_df = evidence_rows_to_dataframe(sample_evidence_rows)

            packet = build_retrieval_packet(
                sample=sample,
                evidence_df=evidence_df,
                top_n_per_claim=top_k,
            )

            packet["_retrieval_meta"] = {
                "sample_row_index": sample_row_index,
                "overlap_ratio": float(overlap_ratio),
                "hybrid_alpha": float(hybrid_alpha),
                "top_k": int(top_k),
                "min_score": min_score,
                "search_max_workers": int(search_max_workers or 1),
                "packets_jsonl": paths["packets_jsonl"],
                "evidence_csv": paths["evidence_csv"],
                "chunks_csv": paths["chunks_csv"],
                "summary_csv": paths["summary_csv"],
            }

            if top_n_to_print and top_n_to_print > 0:
                print_retrieval_summary(
                    packet=packet,
                    top_n_per_claim=top_n_to_print,
                    print_search_text=print_search_text,
                )

            packet_objects.append(packet)

            chunks = chunks_by_row_index.get(sample_row_index, [])
            row_summary = {
                "sample_row_index": sample_row_index,
                "status": "ok",
                "error": "",
                "title": _safe_str(sample.get("title", "")),
                "source_id": _safe_str(sample.get("source_id", "")),
                "article_id": _safe_str(sample.get("article_id", "")),
                "num_claims": len(sample.get("claim_sentences", []) or []),
                "num_failed_claims": 0,
                "num_evidence_rows": int(len(evidence_df)) if evidence_df is not None else 0,
                "num_chunks": int(len(chunks)),
                "collection_count": int(collection_count),
                "search_max_workers": int(search_max_workers or 1),
                "packet_json": paths["packets_jsonl"],
                "evidence_csv": paths["evidence_csv"],
                "chunks_csv": paths["chunks_csv"],
            }
            summary_rows.append(row_summary)

        except Exception as exc:
            error_text = repr(exc)
            print(f"[FAILED PACKET BUILD] sample_row_index={sample_row_index}")
            print(error_text)

            row_summary = {
                "sample_row_index": sample_row_index,
                "status": "failed_packet_build",
                "error": error_text,
                "title": _safe_str(sample.get("title", "")),
                "source_id": _safe_str(sample.get("source_id", "")),
                "article_id": _safe_str(sample.get("article_id", "")),
                "num_claims": len(sample.get("claim_sentences", []) or []),
                "num_failed_claims": 0,
                "num_evidence_rows": len(evidence_rows_by_sample.get(sample_row_index, []) or []),
                "num_chunks": len(chunks_by_row_index.get(sample_row_index, []) or []),
                "collection_count": int(collection_count),
                "search_max_workers": int(search_max_workers or 1),
                "packet_json": paths["packets_jsonl"],
                "evidence_csv": paths["evidence_csv"],
                "chunks_csv": paths["chunks_csv"],
            }
            summary_rows.append(row_summary)
            failed_rows.append(row_summary)

            if fail_fast:
                raise

    summary_df = save_single_batch_retrieval_outputs(
        summary_rows=summary_rows,
        packet_objects=packet_objects,
        evidence_rows=all_evidence_rows,
        chunks=all_chunks,
        paths=paths,
    )

    print("\n" + "=" * 120)
    print("[BATCH RETRIEVAL DONE]")
    if not summary_df.empty and "status" in summary_df.columns:
        print(summary_df["status"].value_counts(dropna=False))
    print("packets_jsonl:", paths["packets_jsonl"])
    print("packet count:", len(packet_objects))
    print("failed_rows:", len(failed_rows))
    print("=" * 120)

    return {
        "summary_df": summary_df,
        "rows": summary_rows,
        "packet_json_paths": [paths["packets_jsonl"]],
        "packet_objects": packet_objects,
        "failed_rows": failed_rows,
        "data_source": data_source,
        "save_dir": save_dir,
        "aggregate_paths": paths,
    }

def load_seen_ids(path: str = SEEN_IDS_PATH) -> Set[str]:
    if not os.path.exists(path):
        return set()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return set(data)


def save_seen_ids(seen_ids: Set[str], path: str = SEEN_IDS_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(seen_ids), f, ensure_ascii=False, indent=2)
