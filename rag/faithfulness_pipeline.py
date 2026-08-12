# -*- coding: utf-8 -*-
"""
Guardian original-vs-final highlight faithfulness evaluation.

Input:
- PostgreSQL raw_articles
- PostgreSQL model_outputs, task='highlight'
- PostgreSQL judge_results

Scoring:
- article_text is the evidence context.
- original_response is the original generated highlight.
- final_response is the judge-approved/revised highlight.
- When original_response and final_response are equivalent after normalization,
  the final score reuses the original score and no second Gemini request is made.

Output:
- Complete successful scores are checkpointed to OUT_PATH.
- Failed score attempts are written only to ERROR_LOG_PATH.
- The joined input table stays in memory; no merged CSV is created.

Install the compatible RAG dependency set before execution with
``python scripts/install_dependencies.py --with-rag``.
This module never installs or changes packages at runtime.

Required Colab Secrets or environment variables:
    DATABASE_URL
    GOOGLE_API_KEY

Colab execution:
    merged, run_df, result_df = await run_faithfulness_pipeline()
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import instructor
import pandas as pd
import psycopg
from openai import AsyncOpenAI
from psycopg.rows import dict_row
from ragas.llms import InstructorLLM
from ragas.metrics.collections import Faithfulness


# ======================================================================================
# 1. Configuration
# ======================================================================================

RAW_ARTICLES_TABLE = "raw_articles"
MODEL_OUTPUTS_TABLE = "model_outputs"
JUDGE_RESULTS_TABLE = "judge_results"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get(
        "FAITHFULNESS_OUTPUT_DIR",
        str(PROJECT_ROOT / "artifacts" / "rag" / "faithfulness"),
    )
)
OUT_PATH = OUTPUT_DIR / "faithfulness_original_vs_final_results.csv"
ERROR_LOG_PATH = OUTPUT_DIR / "faithfulness_original_vs_final_errors.csv"

# None means all eligible rows.
RUN_N: Optional[int] = None

# Skip this many rows after sorting the current database result.
START_AT = 0

# Only evaluate rows where the judge changed the highlight.
ONLY_CHANGED_HIGHLIGHT = False

# None means all eligible source IDs. A tuple restricts the database query exactly.
SOURCE_IDS: Optional[Tuple[str, ...]] = None

# Only valid judge rows are loaded.
ALLOWED_QUALITY_STATUSES: Tuple[str, ...] = ("OK", "REVISED")
REQUIRE_NO_PARSE_FAILURE = True

GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_TOKENS = 8192

STOP_ON_QUOTA = True
SLEEP_SECONDS_BETWEEN_ROWS = 3.0
MAX_SCORE_RETRIES = 5
RETRY_BASE_SLEEP_SECONDS = 10.0
RETRY_MAX_SLEEP_SECONDS = 90.0
RETRY_JITTER_SECONDS = 2.0

# Complete rows already present in OUT_PATH are skipped.
RESUME_IF_OUTPUT_EXISTS = True

# If the normalized original and final responses are equal, score once and reuse it.
REUSE_SCORE_WHEN_RESPONSES_EQUAL = True
IGNORE_HIGHLIGHT_PREFIX_FOR_COMPARE = True


# ======================================================================================
# 2. Secrets and small utilities
# ======================================================================================


def get_secret(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value

    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    raise RuntimeError(
        f"Missing {name}. Set it in the environment or Colab Secrets."
    )


def get_database_url() -> str:
    return get_secret("DATABASE_URL")


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def strip_highlight_prefix(text: Any) -> str:
    text = safe_str(text).strip()
    return re.sub(
        r"^\s*highlight\s*:\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def normalize_response_for_compare(text: Any) -> str:
    """Normalize only for the decision whether final needs a separate score."""
    text = safe_str(text).strip()
    if IGNORE_HIGHLIGHT_PREFIX_FOR_COMPARE:
        text = strip_highlight_prefix(text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def is_quota_error(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "Quota exceeded" in message
        or "rate limit" in message.lower()
    )


def compact_error(exc: BaseException, max_len: int = 1500) -> str:
    return str(exc).replace("\n", " ")[:max_len]


def result_to_score_dict(result: Any) -> Dict[str, Any]:
    value = float(result.value) if result.value is not None else None
    return {
        "ok": True,
        "faithfulness": value,
        "hallucination_score": None if value is None else 1.0 - value,
        "error_type": "",
        "error_message": "",
        "raw_result": repr(result),
    }


def is_retryable_transient_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    markers = [
        "503",
        "500",
        "502",
        "504",
        "unavailable",
        "temporarily",
        "temporary",
        "high demand",
        "try again later",
        "timeout",
        "timed out",
        "connection",
        "connection reset",
        "server error",
        "internal error",
    ]
    return any(marker in message for marker in markers)


def is_hard_quota_error(exc: BaseException) -> bool:
    message = str(exc)
    markers = [
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
        "free_tier_requests",
        "quotaValue",
        "Quota exceeded for metric",
        "RESOURCE_EXHAUSTED",
    ]
    return any(marker in message for marker in markers)


def display_or_print(obj: Any) -> None:
    try:
        display(obj)  # type: ignore[name-defined]
    except NameError:
        print(obj)


# ======================================================================================
# 3. PostgreSQL join and in-memory preparation
# ======================================================================================


def build_merged_query(
    *,
    require_no_parse_failure: bool,
    filter_source_ids: bool = False,
) -> str:
    parse_filter = "AND judge.any_parse_failed = FALSE" if require_no_parse_failure else ""
    source_filter = "AND judge.source_id = ANY(%s)" if filter_source_ids else ""

    return f"""
    SELECT
        COALESCE(
            output.row_index::BIGINT,
            ROW_NUMBER() OVER (
                ORDER BY
                    COALESCE(article.published_at, output.published_at) DESC NULLS LAST,
                    judge.source_id
            )
        ) AS packet_index,

        judge.source_id,

        COALESCE(
            NULLIF(BTRIM(article.title), ''),
            NULLIF(BTRIM(output.title), ''),
            ''
        ) AS title_for_output,

        COALESCE(
            NULLIF(BTRIM(article.url), ''),
            NULLIF(BTRIM(output.url), ''),
            ''
        ) AS url,

        COALESCE(
            article.published_at,
            output.published_at
        ) AS published_at,

        output.prompt_raw,
        output.model_input_text,
        output.generated_raw,
        output.generated_clean,
        output.format_ok,
        output.output_tokens,
        output.answer_repetition_penalty,

        judge.highlight_changed,
        judge.any_parse_failed,
        judge.final_quality_status,
        judge.final_highlight,

        CASE
            WHEN NULLIF(BTRIM(article.body_text), '') IS NOT NULL
                THEN 'body_text'
            WHEN NULLIF(BTRIM(article.summary), '') IS NOT NULL
                THEN 'summary'
            WHEN NULLIF(BTRIM(output.prompt_raw), '') IS NOT NULL
                THEN 'prompt_raw_fallback'
            ELSE 'missing'
        END AS article_text_source,

        COALESCE(
            NULLIF(BTRIM(article.body_text), ''),
            NULLIF(BTRIM(article.summary), ''),
            NULLIF(BTRIM(output.prompt_raw), ''),
            ''
        ) AS article_text,

        COALESCE(
            NULLIF(BTRIM(output.generated_clean), ''),
            NULLIF(BTRIM(output.generated_raw), ''),
            ''
        ) AS original_response,

        COALESCE(
            NULLIF(BTRIM(judge.final_highlight), ''),
            NULLIF(BTRIM(output.generated_clean), ''),
            NULLIF(BTRIM(output.generated_raw), ''),
            ''
        ) AS final_response

    FROM {JUDGE_RESULTS_TABLE} AS judge

    LEFT JOIN {RAW_ARTICLES_TABLE} AS article
        ON article.source_id = judge.source_id

    LEFT JOIN {MODEL_OUTPUTS_TABLE} AS output
        ON output.source_id = judge.source_id
       AND output.task = 'highlight'

    WHERE judge.final_quality_status = ANY(%s)
      {parse_filter}
      {source_filter}

    ORDER BY
        packet_index,
        judge.source_id;
    """


def load_merged_from_postgres(
    *,
    database_url: Optional[str] = None,
    allowed_quality_statuses: Sequence[str] = ALLOWED_QUALITY_STATUSES,
    require_no_parse_failure: bool = REQUIRE_NO_PARSE_FAILURE,
    source_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    statuses = [safe_str(value).strip() for value in allowed_quality_statuses]
    statuses = [value for value in statuses if value]
    if not statuses:
        raise ValueError("allowed_quality_statuses must not be empty.")

    normalized_source_ids: Optional[List[str]] = None
    if source_ids is not None:
        normalized_source_ids = []
        seen_source_ids = set()
        for value in source_ids:
            source_id = safe_str(value).strip()
            if source_id and source_id not in seen_source_ids:
                seen_source_ids.add(source_id)
                normalized_source_ids.append(source_id)
        if not normalized_source_ids:
            raise ValueError("source_ids was provided but contained no valid IDs.")

    query = build_merged_query(
        require_no_parse_failure=require_no_parse_failure,
        filter_source_ids=normalized_source_ids is not None,
    )
    query_parameters: List[Any] = [statuses]
    if normalized_source_ids is not None:
        query_parameters.append(normalized_source_ids)

    with psycopg.connect(
        database_url or get_database_url(),
        row_factory=dict_row,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(query_parameters))
            rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        if normalized_source_ids is not None:
            print(
                "No eligible judge/model/article rows matched the requested source IDs. "
                "Faithfulness evaluation skipped."
            )
            return pd.DataFrame()
        raise ValueError(
            "No eligible judge/model/article rows were loaded from PostgreSQL."
        )

    merged = pd.DataFrame(rows)

    merged["packet_index"] = pd.to_numeric(
        merged["packet_index"],
        errors="raise",
    ).astype(int)

    text_columns = [
        "source_id",
        "title_for_output",
        "url",
        "prompt_raw",
        "model_input_text",
        "generated_raw",
        "generated_clean",
        "final_highlight",
        "final_quality_status",
        "article_text_source",
        "article_text",
        "original_response",
        "final_response",
    ]
    for column in text_columns:
        merged[column] = merged[column].map(safe_str)

    merged["highlight_changed"] = merged["highlight_changed"].map(parse_bool)
    merged["any_parse_failed"] = merged["any_parse_failed"].map(parse_bool)

    if merged["source_id"].eq("").any():
        raise ValueError("PostgreSQL result contains an empty source_id.")

    duplicate_source_ids = sorted(
        merged.loc[
            merged["source_id"].duplicated(keep=False),
            "source_id",
        ].unique()
    )
    if duplicate_source_ids:
        raise ValueError(
            "PostgreSQL join returned duplicate source_id rows. "
            "Check the model_outputs (source_id, task) uniqueness constraint. "
            f"Examples: {duplicate_source_ids[:10]}"
        )

    merged["original_norm_for_compare"] = merged["original_response"].map(
        normalize_response_for_compare
    )
    merged["final_norm_for_compare"] = merged["final_response"].map(
        normalize_response_for_compare
    )
    merged["responses_equal_normalized"] = merged[
        "original_norm_for_compare"
    ].eq(merged["final_norm_for_compare"])
    merged["needs_final_scoring"] = ~merged["responses_equal_normalized"]

    print("=" * 100)
    print("[POSTGRESQL JOINED INPUT]")
    print("merged shape:", merged.shape)
    print("missing article_text:", int(merged["article_text"].str.strip().eq("").sum()))
    print(
        "missing original_response:",
        int(merged["original_response"].str.strip().eq("").sum()),
    )
    print(
        "missing final_response:",
        int(merged["final_response"].str.strip().eq("").sum()),
    )
    print(
        "responses equal normalized:",
        int(merged["responses_equal_normalized"].sum()),
    )
    print(
        "responses need separate final scoring:",
        int(merged["needs_final_scoring"].sum()),
    )
    print("article_text sources:")
    print(merged["article_text_source"].value_counts(dropna=False).to_string())

    return merged


def select_run_rows(merged: pd.DataFrame) -> pd.DataFrame:
    run_df = merged.copy()

    if ONLY_CHANGED_HIGHLIGHT:
        run_df = run_df[run_df["highlight_changed"]].copy()

    run_df = run_df.sort_values(
        ["packet_index", "source_id"],
        kind="stable",
    ).reset_index(drop=True)

    if START_AT:
        run_df = run_df.iloc[int(START_AT):].copy()

    if RUN_N is not None:
        run_df = run_df.head(int(RUN_N)).copy()

    print("=" * 100)
    print("[ROWS SELECTED TO SCORE]")
    print("rows selected:", len(run_df))

    if not run_df.empty:
        print(
            "packet_index range:",
            int(run_df["packet_index"].min()),
            "to",
            int(run_df["packet_index"].max()),
        )
        planned_original_calls = len(run_df)
        planned_final_calls = (
            int(run_df["needs_final_scoring"].sum())
            if REUSE_SCORE_WHEN_RESPONSES_EQUAL
            else len(run_df)
        )
        print("planned original response scores:", planned_original_calls)
        print("planned separate final response scores:", planned_final_calls)
        print(
            "planned total response scores:",
            planned_original_calls + planned_final_calls,
        )

    return run_df


# ======================================================================================
# 4. Gemini + Ragas setup and scoring
# ======================================================================================


def build_faithfulness_scorer() -> Tuple[Faithfulness, AsyncOpenAI]:
    google_api_key = get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = google_api_key

    raw_client = AsyncOpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    client = instructor.from_openai(
        raw_client,
        mode=instructor.Mode.JSON,
    )

    try:
        from ragas.llms.base import InstructorModelArgs

        llm = InstructorLLM(
            client=client,
            model=GEMINI_MODEL,
            provider="openai",
            model_args=InstructorModelArgs(
                temperature=0,
                max_tokens=MAX_TOKENS,
            ),
        )
    except Exception:
        llm = InstructorLLM(
            client=client,
            model=GEMINI_MODEL,
            provider="openai",
            temperature=0,
            max_tokens=MAX_TOKENS,
        )

    print("=" * 100)
    print("[RAGAS FAITHFULNESS]")
    print("model:", GEMINI_MODEL)
    print("ragas llm is_async:", getattr(llm, "is_async", None))

    return Faithfulness(llm=llm), raw_client


async def score_one_response(
    faithfulness_scorer: Faithfulness,
    response: Any,
    article_text: Any,
) -> Dict[str, Any]:
    response = safe_str(response).strip()
    article_text = safe_str(article_text).strip()

    if not response:
        return {
            "ok": False,
            "faithfulness": None,
            "hallucination_score": None,
            "error_type": "empty_response",
            "error_message": "response is empty",
            "raw_result": None,
            "attempts": 0,
        }

    if not article_text:
        return {
            "ok": False,
            "faithfulness": None,
            "hallucination_score": None,
            "error_type": "empty_article",
            "error_message": "article text is empty",
            "raw_result": None,
            "attempts": 0,
        }

    case = {
        "user_input": "Article",
        "response": response,
        "retrieved_contexts": [article_text],
    }

    last_exc: Optional[BaseException] = None

    for attempt in range(1, MAX_SCORE_RETRIES + 1):
        try:
            try:
                result = await faithfulness_scorer.ascore(**case)
            except TypeError as exc:
                message = str(exc).lower()
                if "synchronous client" in message or "sync" in message:
                    result = await asyncio.to_thread(
                        faithfulness_scorer.score,
                        **case,
                    )
                else:
                    raise

            output = result_to_score_dict(result)
            output["attempts"] = attempt
            return output

        except Exception as exc:
            last_exc = exc

            if is_hard_quota_error(exc):
                return {
                    "ok": False,
                    "faithfulness": None,
                    "hallucination_score": None,
                    "error_type": "quota_or_rate_limit",
                    "error_message": compact_error(exc),
                    "raw_result": None,
                    "attempts": attempt,
                }

            if is_quota_error(exc) and not is_retryable_transient_error(exc):
                return {
                    "ok": False,
                    "faithfulness": None,
                    "hallucination_score": None,
                    "error_type": "quota_or_rate_limit",
                    "error_message": compact_error(exc),
                    "raw_result": None,
                    "attempts": attempt,
                }

            if is_retryable_transient_error(exc) and attempt < MAX_SCORE_RETRIES:
                sleep_seconds = min(
                    RETRY_MAX_SLEEP_SECONDS,
                    RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1)),
                )
                sleep_seconds += random.uniform(0.0, RETRY_JITTER_SECONDS)

                print(
                    f"retryable scoring error on attempt "
                    f"{attempt}/{MAX_SCORE_RETRIES}: "
                    f"{type(exc).__name__}: "
                    f"{compact_error(exc, max_len=250)}"
                )
                print(f"sleeping {sleep_seconds:.1f}s before retry...")
                await asyncio.sleep(sleep_seconds)
                continue

            if is_retryable_transient_error(exc):
                return {
                    "ok": False,
                    "faithfulness": None,
                    "hallucination_score": None,
                    "error_type": "transient_error_max_retries_exceeded",
                    "error_message": compact_error(exc),
                    "raw_result": None,
                    "attempts": attempt,
                }

            return {
                "ok": False,
                "faithfulness": None,
                "hallucination_score": None,
                "error_type": type(exc).__name__,
                "error_message": compact_error(exc),
                "raw_result": None,
                "attempts": attempt,
            }

    return {
        "ok": False,
        "faithfulness": None,
        "hallucination_score": None,
        "error_type": "max_retries_exceeded",
        "error_message": compact_error(last_exc) if last_exc else "",
        "raw_result": None,
        "attempts": MAX_SCORE_RETRIES,
    }


# ======================================================================================
# 5. Result and checkpoint helpers
# ======================================================================================


def make_row_result(
    row: pd.Series,
    *,
    title: str,
    article_text: str,
    original_response: str,
    final_response: str,
    original_score: Dict[str, Any],
    final_score: Dict[str, Any],
    responses_equal_normalized: bool,
    scored_final_separately: bool,
) -> Dict[str, Any]:
    original_faithfulness = original_score["faithfulness"]
    final_faithfulness = final_score["faithfulness"]

    if original_faithfulness is not None and final_faithfulness is not None:
        faithfulness_delta = final_faithfulness - original_faithfulness
        hallucination_delta = (
            (1.0 - final_faithfulness) - (1.0 - original_faithfulness)
        )
    else:
        faithfulness_delta = None
        hallucination_delta = None

    return {
        "packet_index": int(row["packet_index"]),
        "source_id": safe_str(row["source_id"]),
        "title": title,
        "article_text_source": safe_str(row.get("article_text_source", "")),
        "highlight_changed": parse_bool(row.get("highlight_changed", False)),
        "responses_equal_normalized": bool(responses_equal_normalized),
        "scored_final_separately": bool(scored_final_separately),
        "score_reused_for_final": not bool(scored_final_separately),
        "original_response": original_response,
        "final_response": final_response,
        "article_char_len": len(article_text),
        "original_ok": original_score["ok"],
        "original_faithfulness": original_score["faithfulness"],
        "original_hallucination_score": original_score["hallucination_score"],
        "original_error_type": original_score["error_type"],
        "original_error_message": original_score["error_message"],
        "original_raw_result": original_score["raw_result"],
        "final_ok": final_score["ok"],
        "final_faithfulness": final_score["faithfulness"],
        "final_hallucination_score": final_score["hallucination_score"],
        "final_error_type": final_score["error_type"],
        "final_error_message": final_score["error_message"],
        "final_raw_result": final_score["raw_result"],
        "faithfulness_delta": faithfulness_delta,
        "hallucination_delta": hallucination_delta,
    }


def result_row_is_complete(row: pd.Series) -> bool:
    try:
        return (
            pd.notna(row.get("original_faithfulness"))
            and pd.notna(row.get("final_faithfulness"))
            and safe_str(row.get("original_error_type", "")).strip() == ""
            and safe_str(row.get("final_error_type", "")).strip() == ""
        )
    except Exception:
        return False


def make_error_record(
    row: pd.Series,
    *,
    title: str,
    stage: str,
    score_result: Dict[str, Any],
    responses_equal_normalized: Optional[bool] = None,
    need_score_final: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "packet_index": int(row["packet_index"]),
        "source_id": safe_str(row["source_id"]),
        "title": title,
        "article_text_source": safe_str(row.get("article_text_source", "")),
        "stage": stage,
        "error_type": score_result.get("error_type"),
        "error_message": score_result.get("error_message"),
        "responses_equal_normalized": responses_equal_normalized,
        "need_score_final": need_score_final,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def append_error_log(error_record: Dict[str, Any]) -> None:
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([error_record])

    if ERROR_LOG_PATH.exists():
        old_rows = pd.read_csv(ERROR_LOG_PATH)
        new_row = pd.concat([old_rows, new_row], ignore_index=True)

    new_row.to_csv(ERROR_LOG_PATH, index=False)


def load_existing_complete_results() -> Tuple[List[Dict[str, Any]], set[Tuple[int, str]]]:
    if not RESUME_IF_OUTPUT_EXISTS or not OUT_PATH.exists():
        return [], set()

    old_df = pd.read_csv(OUT_PATH)
    required_columns = {
        "packet_index",
        "source_id",
        "original_faithfulness",
        "final_faithfulness",
    }

    if old_df.empty or not required_columns.issubset(old_df.columns):
        print("existing OUT_PATH has no complete-result schema; ignoring it:", OUT_PATH)
        return [], set()

    complete_mask = old_df.apply(result_row_is_complete, axis=1)
    complete_df = old_df[complete_mask].copy()
    incomplete_df = old_df[~complete_mask].copy()

    complete_df["packet_index"] = pd.to_numeric(
        complete_df["packet_index"],
        errors="coerce",
    )
    complete_df = complete_df.dropna(subset=["packet_index", "source_id"])
    complete_df["packet_index"] = complete_df["packet_index"].astype(int)
    complete_df["source_id"] = complete_df["source_id"].map(safe_str)
    complete_df = complete_df.drop_duplicates(
        subset=["packet_index", "source_id"],
        keep="last",
    )

    done_keys = set(
        zip(
            complete_df["packet_index"].astype(int),
            complete_df["source_id"].astype(str),
        )
    )

    print("=" * 100)
    print("[RESUME]")
    print("loaded existing result rows:", len(old_df))
    print("complete existing rows kept:", len(complete_df))
    print("incomplete/error existing rows ignored:", len(incomplete_df))

    return complete_df.to_dict("records"), done_keys


def save_complete_results(results: List[Dict[str, Any]]) -> pd.DataFrame:
    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df = result_df.drop_duplicates(
        subset=["packet_index", "source_id"],
        keep="last",
    )
    result_df = result_df.sort_values(
        ["packet_index", "source_id"],
        kind="stable",
    ).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUT_PATH, index=False)
    return result_df


# ======================================================================================
# 6. Summary
# ======================================================================================


def print_result_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 100)
    print("[SUMMARY]")
    print("result_df shape:", result_df.shape)

    if result_df.empty:
        print("No complete scored rows. Main result file was not written or updated.")
        return pd.DataFrame()

    valid_pair = result_df[
        result_df["original_faithfulness"].notna()
        & result_df["final_faithfulness"].notna()
    ].copy()

    print("complete result path:", OUT_PATH)
    print("total result rows:", len(result_df))
    print("valid original+final scored rows:", len(valid_pair))
    print(
        "original errors in main results:",
        int(result_df["original_error_type"].fillna("").ne("").sum()),
    )
    print(
        "final errors in main results:",
        int(result_df["final_error_type"].fillna("").ne("").sum()),
    )

    if "scored_final_separately" in result_df.columns:
        print(
            "final separately scored rows:",
            int(
                result_df["scored_final_separately"]
                .fillna(False)
                .infer_objects(copy=False)
                .astype(bool)
                .sum()
            ),
        )
        print(
            "final reused original score rows:",
            int(
                result_df["score_reused_for_final"]
                .fillna(False)
                .infer_objects(copy=False)
                .astype(bool)
                .sum()
            ),
        )

    if ERROR_LOG_PATH.exists():
        error_df = pd.read_csv(ERROR_LOG_PATH)
        print("error log rows, not included in main results:", len(error_df))
        print("error log path:", ERROR_LOG_PATH)

    if valid_pair.empty:
        return valid_pair

    print("mean original faithfulness:", valid_pair["original_faithfulness"].mean())
    print("mean final faithfulness:", valid_pair["final_faithfulness"].mean())
    print(
        "mean original hallucination:",
        valid_pair["original_hallucination_score"].mean(),
    )
    print(
        "mean final hallucination:",
        valid_pair["final_hallucination_score"].mean(),
    )
    print("mean faithfulness_delta:", valid_pair["faithfulness_delta"].mean())
    print("mean hallucination_delta:", valid_pair["hallucination_delta"].mean())

    print("improved count:", int((valid_pair["faithfulness_delta"] > 0).sum()))
    print("worsened count:", int((valid_pair["faithfulness_delta"] < 0).sum()))
    print("same count:", int((valid_pair["faithfulness_delta"] == 0).sum()))

    print(
        "original faithfulness < 1:",
        int((valid_pair["original_faithfulness"] < 1.0).sum()),
    )
    print(
        "final faithfulness < 1:",
        int((valid_pair["final_faithfulness"] < 1.0).sum()),
    )
    print(
        "original faithfulness < 0.99:",
        int((valid_pair["original_faithfulness"] < 0.99).sum()),
    )
    print(
        "final faithfulness < 0.99:",
        int((valid_pair["final_faithfulness"] < 0.99).sum()),
    )

    if "highlight_changed" in valid_pair.columns:
        by_changed = valid_pair.groupby(
            "highlight_changed",
            dropna=False,
        ).agg(
            n=("packet_index", "count"),
            separately_scored_final_n=("scored_final_separately", "sum"),
            reused_final_score_n=("score_reused_for_final", "sum"),
            original_faithfulness_mean=("original_faithfulness", "mean"),
            final_faithfulness_mean=("final_faithfulness", "mean"),
            faithfulness_delta_mean=("faithfulness_delta", "mean"),
            original_hallucination_mean=("original_hallucination_score", "mean"),
            final_hallucination_mean=("final_hallucination_score", "mean"),
        )
        print("\n===== By highlight_changed =====")
        display_or_print(by_changed)

    comparison_columns = [
        "packet_index",
        "source_id",
        "title",
        "highlight_changed",
        "scored_final_separately",
        "original_faithfulness",
        "final_faithfulness",
        "faithfulness_delta",
        "original_response",
        "final_response",
    ]

    print("\n===== Largest improvements =====")
    display_or_print(
        valid_pair.sort_values(
            "faithfulness_delta",
            ascending=False,
        )[comparison_columns].head(10)
    )

    print("\n===== Largest regressions =====")
    display_or_print(
        valid_pair.sort_values(
            "faithfulness_delta",
            ascending=True,
        )[comparison_columns].head(10)
    )

    return valid_pair


# ======================================================================================
# 7. End-to-end async runner
# ======================================================================================


async def run_faithfulness_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the joined database rows, evaluate faithfulness, and checkpoint results.

    Returns:
        merged: all eligible joined PostgreSQL rows
        run_df: rows selected by SOURCE_IDS / ONLY_CHANGED_HIGHLIGHT / START_AT / RUN_N
        result_df: all complete checkpointed score rows
    """
    if SOURCE_IDS is not None and not SOURCE_IDS:
        print("No source IDs were provided. Faithfulness evaluation skipped.")
        empty = pd.DataFrame()
        return empty, empty, empty

    merged = load_merged_from_postgres(source_ids=SOURCE_IDS)
    if merged.empty:
        empty = pd.DataFrame()
        return merged, empty, empty

    run_df = select_run_rows(merged)

    existing_results, done_keys = load_existing_complete_results()
    results: List[Dict[str, Any]] = list(existing_results)

    pending_keys = {
        (int(row["packet_index"]), safe_str(row["source_id"]))
        for _, row in run_df.iterrows()
    } - done_keys

    if not pending_keys:
        print("No pending rows require Gemini scoring.")
        result_df = save_complete_results(results) if results else pd.DataFrame()
        print_result_summary(result_df)
        return merged, run_df, result_df

    faithfulness_scorer, raw_client = build_faithfulness_scorer()

    try:
        for position, (_, row) in enumerate(run_df.iterrows(), start=1):
            packet_index = int(row["packet_index"])
            source_id = safe_str(row["source_id"])
            key = (packet_index, source_id)

            if key in done_keys:
                print(
                    f"[skip existing complete] "
                    f"packet_index={packet_index} | source_id={source_id}"
                )
                continue

            title = safe_str(row.get("title_for_output", ""))
            article_text = safe_str(row["article_text"])
            original_response = safe_str(row["original_response"])
            final_response = safe_str(row["final_response"])

            original_norm = normalize_response_for_compare(original_response)
            final_norm = normalize_response_for_compare(final_response)
            responses_equal_normalized = original_norm == final_norm

            need_score_final = (
                not responses_equal_normalized
                or not REUSE_SCORE_WHEN_RESPONSES_EQUAL
            )

            print(
                f"\n[{position}/{len(run_df)}] "
                f"packet_index={packet_index} | {title[:90]}"
            )
            print("source_id:", source_id)
            print("article_text_source:", row.get("article_text_source"))
            print("responses_equal_normalized:", responses_equal_normalized)
            print("need_score_final:", need_score_final)

            original_score = await score_one_response(
                faithfulness_scorer=faithfulness_scorer,
                response=original_response,
                article_text=article_text,
            )

            if not original_score["ok"]:
                print("original scoring failed; row not saved to the main result file.")
                print("original_error_type:", original_score["error_type"])

                append_error_log(
                    make_error_record(
                        row,
                        title=title,
                        stage="original",
                        score_result=original_score,
                        responses_equal_normalized=responses_equal_normalized,
                        need_score_final=need_score_final,
                    )
                )
                print("saved error log:", ERROR_LOG_PATH)

                if (
                    original_score["error_type"] == "quota_or_rate_limit"
                    and STOP_ON_QUOTA
                ):
                    print("STOP_ON_QUOTA=True; stopping now.")
                    break

                await asyncio.sleep(SLEEP_SECONDS_BETWEEN_ROWS)
                continue

            if need_score_final:
                final_score = await score_one_response(
                    faithfulness_scorer=faithfulness_scorer,
                    response=final_response,
                    article_text=article_text,
                )
                scored_final_separately = True

                if not final_score["ok"]:
                    print("final scoring failed; row not saved to the main result file.")
                    print("final_error_type:", final_score["error_type"])

                    append_error_log(
                        make_error_record(
                            row,
                            title=title,
                            stage="final",
                            score_result=final_score,
                            responses_equal_normalized=responses_equal_normalized,
                            need_score_final=need_score_final,
                        )
                    )
                    print("saved error log:", ERROR_LOG_PATH)

                    if (
                        final_score["error_type"] == "quota_or_rate_limit"
                        and STOP_ON_QUOTA
                    ):
                        print("STOP_ON_QUOTA=True; stopping now.")
                        break

                    await asyncio.sleep(SLEEP_SECONDS_BETWEEN_ROWS)
                    continue
            else:
                final_score = dict(original_score)
                scored_final_separately = False

            row_result = make_row_result(
                row,
                title=title,
                article_text=article_text,
                original_response=original_response,
                final_response=final_response,
                original_score=original_score,
                final_score=final_score,
                responses_equal_normalized=responses_equal_normalized,
                scored_final_separately=scored_final_separately,
            )

            results.append(row_result)
            done_keys.add(key)
            result_df = save_complete_results(results)

            print("original faithfulness:", original_score["faithfulness"])
            print("final faithfulness:", final_score["faithfulness"])
            print("faithfulness_delta:", row_result["faithfulness_delta"])
            print("scored_final_separately:", scored_final_separately)
            print("score_reused_for_final:", row_result["score_reused_for_final"])
            print("saved complete row:", OUT_PATH)

            await asyncio.sleep(SLEEP_SECONDS_BETWEEN_ROWS)

    finally:
        try:
            await raw_client.close()
        except Exception:
            pass

    result_df = save_complete_results(results) if results else pd.DataFrame()
    print_result_summary(result_df)

    return merged, run_df, result_df


# Run this in a Colab cell after importing or executing this file:
# merged, run_df, result_df = await run_faithfulness_pipeline()
