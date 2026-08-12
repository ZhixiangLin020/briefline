# -*- coding: utf-8 -*-
"""Guardian similar-articles pipeline backed by PostgreSQL and one Weaviate collection.

The pipeline:
1. Loads confirmed judge results from PostgreSQL by joining judge_results and raw_articles.
2. Builds an in-memory similarity base table. No CSV or JSONL files are read or written.
3. Reuses the existing GuardianSentenceEvidenceOpenAISmallPOC collection.
4. Stores one vector for each unique complete category combination.
5. Stores one semantic vector for each article.
6. Uses deterministic UUIDs and content hashes for incremental synchronization.
7. Calls OpenAI embeddings only for missing objects or changed vector text.
8. Reuses vectors already stored in Weaviate for all query searches.
9. Performs category search, category-filtered semantic recall, and optional ColBERT reranking.
10. Persists one ordered PostgreSQL TEXT[] recommendation list per processed article.

Install dependencies before execution from requirements-rag.txt. This module
never installs or changes packages at runtime.

Required Colab secrets or environment variables:
    DATABASE_URL
    OPENAI_API_KEY
    WEAVIATE_URL
    WEAVIATE_API_KEY
    HF_TOKEN
"""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import weaviate
from openai import OpenAI
from tqdm.auto import tqdm
from weaviate.auth import AuthApiKey
from weaviate.classes.config import DataType, Property, Tokenization
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import WeaviateQueryError
from weaviate.util import generate_uuid5

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:
    raise ImportError(
        "psycopg is required. Install the dependencies declared in "
        "requirements-rag.txt before running the similarity pipeline."
    ) from exc


# ======================================================================================
# 1. User-editable config
# ======================================================================================

RAW_ARTICLES_TABLE = "raw_articles"
JUDGE_RESULTS_TABLE = "judge_results"
RECOMMENDATIONS_TABLE = "article_recommendations"

# Persist only the ordered candidate IDs for each processed article.
SAVE_RECOMMENDATIONS_TO_POSTGRES = True

# Only these confirmed judge rows are used as article candidates.
ALLOWED_QUALITY_STATUSES = ("OK", "REVISED")

# Query only the first N rows after the full candidate library has been synchronized.
# None means query every eligible article.
PROCESS_N: Optional[int] = None

# Keep the current project setting. Change to 1000 if a longer body head is desired.
BODY_CHARS = 800

TOPIC_TOP_K = 5
COARSE_TOP_K = 50
FINAL_TOP_K = 10

MIN_TOPIC_SCORE = 0.52
MIN_SEMANTIC_SCORE = 0.30

TOPIC_HYBRID_ALPHA = 0.7
SEMANTIC_HYBRID_ALPHA = 0.7

EXCLUDE_SELF = True
MIN_COARSE_RESULTS_WARN = FINAL_TOP_K

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS: Optional[int] = None
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_MAX_RETRIES = 5
EMBEDDING_RETRY_SLEEP_BASE = 1.5

USE_COLBERT = True

# Hugging Face model download settings.
# Only the files required by the PyTorch/Transformers loader are downloaded.
# The ONNX checkpoint and duplicate pytorch_model.bin checkpoint are intentionally skipped.
COLBERT_MODEL_PATH = "colbert-ir/colbertv2.0"
COLBERT_MODEL_REVISION: Optional[str] = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAG_ARTIFACT_DIR = Path(
    os.environ.get("RAG_ARTIFACT_DIR", str(PROJECT_ROOT / "artifacts" / "rag"))
)
COLBERT_MODEL_LOCAL_DIR = os.environ.get(
    "COLBERT_MODEL_LOCAL_DIR",
    str(RAG_ARTIFACT_DIR / "models" / "colbertv2.0"),
)
COLBERT_WEIGHT_FILENAME = "model.safetensors"
COLBERT_REQUIRED_MODEL_FILES = (
    "artifact.metadata",
    "config.json",
    COLBERT_WEIGHT_FILENAME,
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
COLBERT_MIN_WEIGHT_FILE_BYTES = 400_000_000
COLBERT_DOWNLOAD_MAX_RETRIES = 4
COLBERT_DOWNLOAD_RETRY_SLEEP_SECONDS = 5.0
HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 300
HF_HUB_ETAG_TIMEOUT_SECONDS = 30
AUTO_INSTALL_HUGGINGFACE_HUB = False

AUTO_INSTALL_RAGATOUILLE = False
COLBERT_AUTO_INSTALL_COMPAT_DEPS = False
COLBERT_FAIL_OPEN = False

PRINT_QUERY_DETAILS = True
PRINT_SIMILAR_TOPICS = True
PRINT_NON_COLBERT_RESULTS = True
PRINT_COLBERT_RESULTS = True
PRINT_COMPARISON = True

SIMILARITY_COLLECTION_NAME = "GuardianSentenceEvidenceOpenAISmallPOC"
# Keep similarity article IDs outside the evidence article_id namespace.
# The existing evidence retriever filters article_id=source_id; this prefix prevents
# article-level similarity objects from appearing as sentence evidence.
SIMILARITY_ARTICLE_ID_PREFIX = "similarity::"
WEAVIATE_INDEX_READY_WAIT_SECONDS = 20
WEAVIATE_QUERY_MAX_RETRIES = 8
WEAVIATE_QUERY_RETRY_SLEEP_SECONDS = 5
WEAVIATE_SIMILARITY_OBJECT_FETCH_LIMIT = 10000
DELETE_STALE_SIMILARITY_OBJECTS = True

# Library modules never execute a pipeline at import time.
RUN_PIPELINE = False


# ======================================================================================
# 2. Generic helpers
# ======================================================================================


def stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def split_keywords(value: Any) -> List[str]:
    text = stringify(value)
    if not text:
        return []

    output: List[str] = []
    seen = set()
    for item in re.split(r"[,;|]\s*", text):
        item = re.sub(r"\s+", " ", item).strip()
        if not item:
            continue
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def normalize_category_combo(value: Any) -> str:
    """Normalize a complete category combination without splitting it."""
    text = stringify(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*\|\s*", "|", text)
    return text


def category_combo_list(value: Any) -> List[str]:
    combo = normalize_category_combo(value)
    return [combo] if combo else []


def truncate_chars(value: Any, n_chars: int) -> str:
    return stringify(value)[: max(0, int(n_chars))].strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256_text(encoded)


def build_vector_hash(text: str, *, model: str = EMBEDDING_MODEL) -> str:
    dimensions_marker = "default" if EMBEDDING_DIMENSIONS is None else str(EMBEDDING_DIMENSIONS)
    return sha256_text(f"{model}\n{dimensions_marker}\n{text}")


def get_secret(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value:
        return value
    try:
        from google.colab import userdata  # type: ignore

        value = userdata.get(name)
        if value:
            return value
    except Exception:
        pass
    return None


def get_database_url() -> str:
    value = get_secret("DATABASE_URL")
    if not value:
        raise RuntimeError("DATABASE_URL is missing from the environment or Colab Secrets.")
    return value


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def short(value: Any, n: int = 140) -> str:
    text = re.sub(r"\s+", " ", stringify(value))
    return text if len(text) <= n else text[:n] + "..."


# ======================================================================================
# 3. PostgreSQL input and in-memory base table
# ======================================================================================


def build_topic_text_from_category(category: Any) -> str:
    return f"category_combo: {normalize_category_combo(category)}".strip()


def build_semantic_text(row: pd.Series) -> str:
    return (
        f"title: {stringify(row.get('title'))}\n"
        f"keywords: {stringify(row.get('final_keywords'))}\n"
        f"highlight: {stringify(row.get('final_highlight'))}\n"
        f"body: {stringify(row.get('body_head'))}"
    ).strip()


def build_colbert_text(row: pd.Series) -> str:
    return (
        f"{stringify(row.get('title'))}\n\n"
        f"Keywords: {stringify(row.get('final_keywords'))}\n\n"
        f"Highlight: {stringify(row.get('final_highlight'))}\n\n"
        f"{stringify(row.get('body_head'))}"
    ).strip()


def load_similarity_base_table_from_postgres(
    *,
    body_chars: int = BODY_CHARS,
    allowed_quality_statuses: Sequence[str] = ALLOWED_QUALITY_STATUSES,
) -> pd.DataFrame:
    statuses = [stringify(x) for x in allowed_quality_statuses if stringify(x)]
    if not statuses:
        raise ValueError("allowed_quality_statuses must not be empty.")

    query = f"""
    SELECT
        judge.source_id,
        article.title,
        article.url,
        article.published_at,
        article.section_id AS guardian_section_id,
        article.section_name AS guardian_section_name,
        COALESCE(
            NULLIF(BTRIM(article.body_text), ''),
            NULLIF(BTRIM(article.summary), ''),
            ''
        ) AS body,
        judge.final_highlight,
        judge.final_category,
        judge.final_keywords,
        judge.final_quality_status,
        judge.any_parse_failed
    FROM {JUDGE_RESULTS_TABLE} AS judge
    INNER JOIN {RAW_ARTICLES_TABLE} AS article
        ON article.source_id = judge.source_id
    WHERE judge.final_quality_status = ANY(%s)
      AND judge.any_parse_failed = FALSE
    ORDER BY article.published_at DESC NULLS LAST, judge.source_id;
    """

    with psycopg.connect(
        get_database_url(),
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (statuses,))
            rows = [dict(row) for row in cursor.fetchall()]

    if not rows:
        raise ValueError("No confirmed judge rows were loaded from PostgreSQL.")

    base = pd.DataFrame(rows)
    base["source_id"] = base["source_id"].map(stringify)
    base["article_id"] = base["source_id"]
    base["title"] = base["title"].map(stringify)
    base["url"] = base["url"].map(stringify)
    base["published_at"] = base["published_at"].map(stringify)
    base["guardian_section_id"] = base["guardian_section_id"].map(stringify)
    base["guardian_section_name"] = base["guardian_section_name"].map(stringify)
    base["body"] = base["body"].map(stringify)
    base["body_head"] = base["body"].map(lambda x: truncate_chars(x, body_chars))
    base["final_highlight"] = base["final_highlight"].map(stringify)
    base["final_category"] = base["final_category"].map(stringify)
    base["final_keywords"] = base["final_keywords"].map(stringify)
    base["final_quality_status"] = base["final_quality_status"].map(stringify)
    base["any_parse_failed"] = base["any_parse_failed"].map(bool_value)

    base["final_category_norm"] = base["final_category"].map(normalize_category_combo)
    base["final_category_combo"] = base["final_category"].map(category_combo_list)
    base["final_keywords_list"] = base["final_keywords"].map(split_keywords)
    base["semantic_text"] = base.apply(build_semantic_text, axis=1)
    base["colbert_text"] = base.apply(build_colbert_text, axis=1)

    valid_mask = (
        base["source_id"].ne("")
        & base["title"].ne("")
        & base["final_category_norm"].ne("")
        & base["final_highlight"].ne("")
        & base["semantic_text"].ne("")
    )

    invalid = base.loc[~valid_mask].copy()
    base = base.loc[valid_mask].copy()
    base = base.drop_duplicates(subset=["source_id"], keep="first").reset_index(drop=True)

    if base.empty:
        raise ValueError("All PostgreSQL similarity rows were invalid after required-field checks.")

    preferred_columns = [
        "article_id",
        "source_id",
        "title",
        "url",
        "published_at",
        "guardian_section_id",
        "guardian_section_name",
        "body",
        "body_head",
        "final_highlight",
        "final_keywords",
        "final_keywords_list",
        "final_category",
        "final_category_norm",
        "final_category_combo",
        "final_quality_status",
        "any_parse_failed",
        "semantic_text",
        "colbert_text",
    ]
    base = base[[column for column in preferred_columns if column in base.columns]]

    print("=" * 100)
    print("[LOAD SIMILARITY BASE TABLE FROM POSTGRESQL]")
    print("loaded rows:", len(rows))
    print("valid unique rows:", len(base))
    print("invalid rows skipped:", len(invalid))
    print("body non-empty:", int(base["body"].map(bool).sum()), "/", len(base))
    print("unique complete category combinations:", int(base["final_category_norm"].nunique()))
    return base


# ======================================================================================
# 3A. PostgreSQL recommendation-result persistence
# ======================================================================================


def ensure_recommendations_table(conn) -> None:
    """Create the compact recommendation cache table when it does not exist.

    One source article occupies one row. The array order is the recommendation rank.
    Candidate metadata is intentionally not duplicated because the frontend can join the IDs
    back to raw_articles and judge_results.
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {RECOMMENDATIONS_TABLE} (
        source_id TEXT PRIMARY KEY
            REFERENCES {RAW_ARTICLES_TABLE}(source_id)
            ON DELETE CASCADE,
        recommended_source_ids TEXT[] NOT NULL DEFAULT '{{}}',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        CONSTRAINT {RECOMMENDATIONS_TABLE}_max_items_check
            CHECK (cardinality(recommended_source_ids) <= {int(FINAL_TOP_K)}),
        CONSTRAINT {RECOMMENDATIONS_TABLE}_no_self_check
            CHECK (NOT (source_id = ANY(recommended_source_ids)))
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(ddl)


def build_recommendation_records(
    *,
    query_table: pd.DataFrame,
    final_results: pd.DataFrame,
    max_items: int = FINAL_TOP_K,
) -> pd.DataFrame:
    """Build one ordered ID list for every processed query article.

    Every article in query_table is emitted, including articles with zero valid candidates.
    Saving an empty array is important because it clears recommendations from an earlier run.
    """
    limit = max(0, int(max_items))
    query_source_ids: List[str] = []
    seen_queries = set()
    for value in query_table.get("source_id", pd.Series(dtype=object)).tolist():
        source_id = stringify(value)
        if source_id and source_id not in seen_queries:
            seen_queries.add(source_id)
            query_source_ids.append(source_id)

    ordered_candidates: Dict[str, List[str]] = {source_id: [] for source_id in query_source_ids}
    seen_candidates: Dict[str, set] = {source_id: set() for source_id in query_source_ids}

    if not final_results.empty:
        ranked = final_results.copy()
        ranked["_query_source_id"] = ranked.get(
            "query_source_id", pd.Series(index=ranked.index, dtype=object)
        ).map(stringify)
        ranked["_candidate_source_id"] = ranked.get(
            "candidate_source_id", pd.Series(index=ranked.index, dtype=object)
        ).map(stringify)
        ranked["_status"] = ranked.get(
            "status", pd.Series(index=ranked.index, dtype=object)
        ).map(stringify)

        if "colbert_rank" in ranked.columns:
            ranked["_rank"] = pd.to_numeric(ranked["colbert_rank"], errors="coerce")
        else:
            ranked["_rank"] = np.nan
        if "non_colbert_rank" in ranked.columns:
            fallback_rank = pd.to_numeric(ranked["non_colbert_rank"], errors="coerce")
            ranked["_rank"] = ranked["_rank"].fillna(fallback_rank)
        ranked["_rank"] = ranked["_rank"].fillna(np.inf)
        ranked = ranked.sort_values(
            ["_query_source_id", "_rank"],
            ascending=[True, True],
            kind="stable",
        )

        for row in ranked[[
            "_query_source_id",
            "_candidate_source_id",
            "_status",
        ]].to_dict(orient="records"):
            query_source_id = stringify(row.get("_query_source_id"))
            candidate_source_id = stringify(row.get("_candidate_source_id"))
            status = stringify(row.get("_status")).lower()

            if query_source_id not in ordered_candidates:
                continue
            if status != "ok" or not candidate_source_id:
                continue
            if candidate_source_id == query_source_id:
                continue
            if candidate_source_id in seen_candidates[query_source_id]:
                continue
            if len(ordered_candidates[query_source_id]) >= limit:
                continue

            seen_candidates[query_source_id].add(candidate_source_id)
            ordered_candidates[query_source_id].append(candidate_source_id)

    records = pd.DataFrame([
        {
            "source_id": source_id,
            "recommended_source_ids": ordered_candidates[source_id],
            "recommendation_count": len(ordered_candidates[source_id]),
        }
        for source_id in query_source_ids
    ])
    return records


def save_recommendations_to_postgres(
    recommendation_records: pd.DataFrame,
) -> Dict[str, int]:
    """Upsert compact ordered recommendation lists into PostgreSQL."""
    if recommendation_records.empty:
        return {
            "processed": 0,
            "with_recommendations": 0,
            "empty_lists": 0,
            "total_candidate_ids": 0,
        }

    rows: List[Tuple[str, List[str]]] = []
    for record in recommendation_records.itertuples(index=False):
        source_id = stringify(record.source_id)
        candidate_ids = [
            stringify(value)
            for value in list(record.recommended_source_ids or [])
            if stringify(value)
        ]
        if source_id:
            rows.append((source_id, candidate_ids[: int(FINAL_TOP_K)]))

    upsert_sql = f"""
    INSERT INTO {RECOMMENDATIONS_TABLE} (
        source_id,
        recommended_source_ids,
        updated_at
    )
    VALUES (%s, %s::TEXT[], NOW())
    ON CONFLICT (source_id)
    DO UPDATE SET
        recommended_source_ids = EXCLUDED.recommended_source_ids,
        updated_at = NOW();
    """

    with psycopg.connect(get_database_url()) as conn:
        ensure_recommendations_table(conn)
        with conn.cursor() as cursor:
            cursor.executemany(upsert_sql, rows)
        conn.commit()

    counts = [len(candidate_ids) for _, candidate_ids in rows]
    stats = {
        "processed": len(rows),
        "with_recommendations": sum(count > 0 for count in counts),
        "empty_lists": sum(count == 0 for count in counts),
        "total_candidate_ids": sum(counts),
    }
    print("=" * 100)
    print("[POSTGRESQL RECOMMENDATION UPSERT DONE]")
    print("table:", RECOMMENDATIONS_TABLE)
    print(stats)
    return stats


def build_category_combo_table(base: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen = set()

    for _, article in base.iterrows():
        category = stringify(article.get("final_category"))
        category_norm = normalize_category_combo(category)
        if not category_norm or category_norm in seen:
            continue
        seen.add(category_norm)
        rows.append({
            "category_id": category_norm,
            "final_category": category,
            "final_category_norm": category_norm,
            "final_category_combo": [category_norm],
            "topic_text": build_topic_text_from_category(category_norm),
        })

    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values("final_category_norm").reset_index(drop=True)
    print("Unique category-combination objects:", len(output))
    return output


# ======================================================================================
# 4. OpenAI embeddings without a local cache
# ======================================================================================


class OpenAIEmbeddingClient:
    """Generate embeddings only for objects selected by incremental synchronization."""

    def __init__(
        self,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: Optional[int] = EMBEDDING_DIMENSIONS,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        max_retries: int = EMBEDDING_MAX_RETRIES,
        sleep_base: float = EMBEDDING_RETRY_SLEEP_BASE,
    ) -> None:
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from Colab Secrets or environment variables.")

        self.client = OpenAI(api_key=api_key)
        self.model = stringify(model)
        self.dimensions = dimensions
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(1, int(max_retries))
        self.sleep_base = max(0.1, float(sleep_base))

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = int(self.dimensions)

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.embeddings.create(**kwargs)
                ordered = sorted(response.data, key=lambda item: int(item.index))
                vectors = [[float(x) for x in item.embedding] for item in ordered]
                if len(vectors) != len(texts):
                    raise RuntimeError(
                        f"Embedding count mismatch: got {len(vectors)}, expected {len(texts)}."
                    )
                return vectors
            except BaseException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_seconds = self.sleep_base * (2 ** (attempt - 1))
                print(
                    f"[embedding retry] attempt={attempt}/{self.max_retries} "
                    f"sleep={sleep_seconds:.1f}s error={exc!r}"
                )
                time.sleep(sleep_seconds)

        raise RuntimeError(f"OpenAI embedding request failed: {last_error!r}") from last_error

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        clean_texts = [stringify(text) for text in texts]
        if not clean_texts:
            return []
        if any(not text for text in clean_texts):
            raise ValueError("Embedding input contains an empty text.")

        print("=" * 100)
        print("[OPENAI EMBEDDINGS]")
        print("model:", self.model)
        print("texts requiring embeddings:", len(clean_texts))

        output: List[List[float]] = []
        for start in tqdm(
            range(0, len(clean_texts), self.batch_size),
            desc=f"Embedding {self.model}",
        ):
            batch = clean_texts[start : start + self.batch_size]
            output.extend(self._embed_batch(batch))

        if len(output) != len(clean_texts):
            raise RuntimeError(
                f"Embedding output mismatch: got {len(output)}, expected {len(clean_texts)}."
            )
        return output


# ======================================================================================
# 5. Weaviate connection and schema extension
# ======================================================================================


def connect_weaviate_from_env():
    cluster_url = get_secret("WEAVIATE_URL")
    api_key = get_secret("WEAVIATE_API_KEY")

    if not cluster_url:
        raise RuntimeError("WEAVIATE_URL is missing from Colab Secrets or environment variables.")
    if not api_key:
        raise RuntimeError("WEAVIATE_API_KEY is missing from Colab Secrets or environment variables.")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=AuthApiKey(api_key),
    )
    if not client.is_ready():
        client.close()
        raise RuntimeError("Weaviate client connected, but the instance is not ready.")

    print("Weaviate ready: True")
    return client


def similarity_properties() -> List[Property]:
    """Properties added to the existing evidence collection for new object types.

    Existing evidence objects do not need these values. Category and article objects are
    inserted only after these properties are created, so their new inverted indexes are valid.
    """
    return [
        Property(
            name="object_type",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="category_id",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="url",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=False,
            index_searchable=False,
        ),
        Property(
            name="published_at",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="guardian_section_id",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="guardian_section_name",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(
            name="body_head",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="final_category",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(
            name="final_category_norm",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(
            name="final_category_combo",
            data_type=DataType.TEXT_ARRAY,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="final_keywords",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="final_keywords_list",
            data_type=DataType.TEXT_ARRAY,
            tokenization=Tokenization.WORD,
            index_filterable=True,
            index_searchable=True,
        ),
        Property(
            name="final_highlight",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="topic_text",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="semantic_text",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="colbert_text",
            data_type=DataType.TEXT,
            tokenization=Tokenization.WORD,
            index_filterable=False,
            index_searchable=True,
        ),
        Property(
            name="vector_hash",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
        Property(
            name="metadata_hash",
            data_type=DataType.TEXT,
            tokenization=Tokenization.FIELD,
            index_filterable=True,
            index_searchable=False,
        ),
    ]


def ensure_similarity_collection_schema(
    client,
    collection_name: str = SIMILARITY_COLLECTION_NAME,
):
    if not client.collections.exists(collection_name):
        raise RuntimeError(
            f"The existing evidence collection does not exist: {collection_name}. "
            "Run the retrieval ingestion pipeline first."
        )

    collection = client.collections.use(collection_name)
    config = collection.config.get()
    existing_names = {
        stringify(getattr(prop, "name", ""))
        for prop in (getattr(config, "properties", None) or [])
        if stringify(getattr(prop, "name", ""))
    }

    added: List[str] = []
    for prop in similarity_properties():
        if prop.name in existing_names:
            continue
        collection.config.add_property(prop)
        existing_names.add(prop.name)
        added.append(prop.name)

    print("=" * 100)
    print("[WEAVIATE COLLECTION SCHEMA]")
    print("collection:", collection_name)
    print("new properties added:", added if added else 0)
    return collection


# ======================================================================================
# 6. Incremental category/article synchronization
# ======================================================================================


def category_object_uuid(category_norm: str) -> str:
    return str(generate_uuid5({
        "guardian_similarity_object_type": "category",
        "category_id": normalize_category_combo(category_norm),
    }))


def article_object_uuid(source_id: str) -> str:
    return str(generate_uuid5({
        "guardian_similarity_object_type": "article",
        "source_id": stringify(source_id),
    }))


def article_weaviate_properties(row: pd.Series) -> Dict[str, Any]:
    category_norm = normalize_category_combo(row.get("final_category_norm") or row.get("final_category"))
    properties: Dict[str, Any] = {
        "object_type": "article",
        "category_id": "",
        "article_id": f"{SIMILARITY_ARTICLE_ID_PREFIX}{stringify(row.get('source_id'))}",
        "source_id": stringify(row.get("source_id")),
        "title": stringify(row.get("title")),
        "url": stringify(row.get("url")),
        "published_at": stringify(row.get("published_at")),
        "guardian_section_id": stringify(row.get("guardian_section_id")),
        "guardian_section_name": stringify(row.get("guardian_section_name")),
        "body_head": stringify(row.get("body_head")),
        "final_category": stringify(row.get("final_category")),
        "final_category_norm": category_norm,
        "final_category_combo": [category_norm] if category_norm else [],
        "final_keywords": stringify(row.get("final_keywords")),
        "final_keywords_list": [item.lower() for item in split_keywords(row.get("final_keywords"))],
        "final_highlight": stringify(row.get("final_highlight")),
        "topic_text": "",
        "semantic_text": stringify(row.get("semantic_text")),
        "colbert_text": stringify(row.get("colbert_text")),
        "embedding_model": EMBEDDING_MODEL,
    }
    properties["vector_hash"] = build_vector_hash(properties["semantic_text"])
    properties["metadata_hash"] = canonical_json_hash(properties)
    return properties


def category_weaviate_properties(row: pd.Series) -> Dict[str, Any]:
    category_norm = normalize_category_combo(row.get("final_category_norm") or row.get("final_category"))
    topic_text = stringify(row.get("topic_text")) or build_topic_text_from_category(category_norm)
    properties: Dict[str, Any] = {
        "object_type": "category",
        "category_id": category_norm,
        "article_id": "",
        "source_id": "",
        "title": "",
        "url": "",
        "published_at": "",
        "guardian_section_id": "",
        "guardian_section_name": "",
        "body_head": "",
        "final_category": stringify(row.get("final_category")),
        "final_category_norm": category_norm,
        "final_category_combo": [category_norm] if category_norm else [],
        "final_keywords": "",
        "final_keywords_list": [],
        "final_highlight": "",
        "topic_text": topic_text,
        "semantic_text": topic_text,
        "colbert_text": "",
        "embedding_model": EMBEDDING_MODEL,
    }
    properties["vector_hash"] = build_vector_hash(topic_text)
    properties["metadata_hash"] = canonical_json_hash(properties)
    return properties


@dataclass
class DesiredVectorObject:
    uuid: str
    object_type: str
    vector_text: str
    properties: Dict[str, Any]


def build_desired_similarity_objects(base: pd.DataFrame) -> Dict[str, DesiredVectorObject]:
    desired: Dict[str, DesiredVectorObject] = {}

    categories = build_category_combo_table(base)
    for _, row in categories.iterrows():
        properties = category_weaviate_properties(row)
        obj_uuid = category_object_uuid(properties["category_id"])
        desired[obj_uuid] = DesiredVectorObject(
            uuid=obj_uuid,
            object_type="category",
            vector_text=properties["topic_text"],
            properties=properties,
        )

    for _, row in base.iterrows():
        properties = article_weaviate_properties(row)
        obj_uuid = article_object_uuid(properties["source_id"])
        desired[obj_uuid] = DesiredVectorObject(
            uuid=obj_uuid,
            object_type="article",
            vector_text=properties["semantic_text"],
            properties=properties,
        )

    return desired


def fetch_existing_similarity_object_index(collection) -> Dict[str, Dict[str, Any]]:
    object_filter = Filter.any_of([
        Filter.by_property("object_type").equal("category"),
        Filter.by_property("object_type").equal("article"),
    ])

    response = collection.query.fetch_objects(
        filters=object_filter,
        limit=int(WEAVIATE_SIMILARITY_OBJECT_FETCH_LIMIT),
        return_properties=[
            "object_type",
            "category_id",
            "article_id",
            "source_id",
            "vector_hash",
            "metadata_hash",
            "embedding_model",
            "embedding_dimensions",
        ],
    )

    if len(response.objects) >= int(WEAVIATE_SIMILARITY_OBJECT_FETCH_LIMIT):
        raise RuntimeError(
            "Similarity-object fetch reached WEAVIATE_SIMILARITY_OBJECT_FETCH_LIMIT. "
            "Increase the limit before synchronizing to avoid incomplete stale-object detection."
        )

    output: Dict[str, Dict[str, Any]] = {}
    for obj in response.objects:
        output[str(obj.uuid)] = dict(obj.properties or {})
    return output


def synchronize_similarity_objects(
    *,
    collection,
    base: pd.DataFrame,
    embedder: OpenAIEmbeddingClient,
    delete_stale: bool = DELETE_STALE_SIMILARITY_OBJECTS,
) -> Dict[str, int]:
    desired = build_desired_similarity_objects(base)
    existing = fetch_existing_similarity_object_index(collection)

    to_insert: List[DesiredVectorObject] = []
    to_vector_update: List[DesiredVectorObject] = []
    to_metadata_update: List[DesiredVectorObject] = []
    unchanged: List[DesiredVectorObject] = []

    for obj_uuid, wanted in desired.items():
        current = existing.get(obj_uuid)
        if current is None:
            to_insert.append(wanted)
            continue

        current_vector_hash = stringify(current.get("vector_hash"))
        current_metadata_hash = stringify(current.get("metadata_hash"))
        wanted_vector_hash = stringify(wanted.properties.get("vector_hash"))
        wanted_metadata_hash = stringify(wanted.properties.get("metadata_hash"))

        if current_vector_hash != wanted_vector_hash:
            to_vector_update.append(wanted)
        elif current_metadata_hash != wanted_metadata_hash:
            to_metadata_update.append(wanted)
        else:
            unchanged.append(wanted)

    stale_uuids = sorted(set(existing) - set(desired)) if delete_stale else []

    vector_jobs = to_insert + to_vector_update
    print("=" * 100)
    print("[WEAVIATE INCREMENTAL SIMILARITY SYNC PLAN]")
    print("desired objects:", len(desired))
    print("existing similarity objects:", len(existing))
    print("new objects:", len(to_insert))
    print("vector-changing objects:", len(to_vector_update))
    print("metadata-only objects:", len(to_metadata_update))
    print("unchanged objects:", len(unchanged))
    print("stale objects to delete:", len(stale_uuids))
    print("OpenAI embeddings required:", len(vector_jobs))

    vectors = embedder.embed_texts([job.vector_text for job in vector_jobs]) if vector_jobs else []
    vector_by_uuid = {
        job.uuid: vector
        for job, vector in zip(vector_jobs, vectors)
    }

    if to_insert:
        with collection.batch.dynamic() as batch:
            for job in to_insert:
                vector = vector_by_uuid[job.uuid]
                properties = dict(job.properties)
                properties["embedding_dimensions"] = int(len(vector))
                batch.add_object(
                    uuid=job.uuid,
                    properties=properties,
                    vector=vector,
                )

        if collection.batch.failed_objects:
            raise RuntimeError(
                "Similarity batch insert failed: "
                f"{collection.batch.failed_objects[:3]}"
            )

    for job in tqdm(to_vector_update, desc="Updating changed Weaviate vectors"):
        vector = vector_by_uuid[job.uuid]
        properties = dict(job.properties)
        properties["embedding_dimensions"] = int(len(vector))
        collection.data.update(
            uuid=job.uuid,
            properties=properties,
            vector=vector,
        )

    for job in tqdm(to_metadata_update, desc="Updating changed Weaviate metadata"):
        collection.data.update(
            uuid=job.uuid,
            properties=job.properties,
        )

    for obj_uuid in tqdm(stale_uuids, desc="Deleting stale similarity objects"):
        collection.data.delete_by_id(obj_uuid)

    stats = {
        "desired": len(desired),
        "existing": len(existing),
        "inserted": len(to_insert),
        "vector_updated": len(to_vector_update),
        "metadata_updated": len(to_metadata_update),
        "unchanged": len(unchanged),
        "deleted_stale": len(stale_uuids),
        "openai_embeddings_requested": len(vector_jobs),
    }

    print("=" * 100)
    print("[WEAVIATE INCREMENTAL SIMILARITY SYNC DONE]")
    print(stats)

    if vector_jobs and WEAVIATE_INDEX_READY_WAIT_SECONDS > 0:
        print(
            f"Waiting {WEAVIATE_INDEX_READY_WAIT_SECONDS}s for updated Weaviate vectors "
            "to become query-ready."
        )
        time.sleep(WEAVIATE_INDEX_READY_WAIT_SECONDS)

    return stats


# ======================================================================================
# 7. Query helpers that reuse stored Weaviate vectors
# ======================================================================================


def extract_default_vector(data_object: Any) -> List[float]:
    vector = getattr(data_object, "vector", None)
    if isinstance(vector, dict):
        vector = vector.get("default")
    if vector is None:
        return []
    return [float(x) for x in vector]


def get_object_vector_by_uuid(
    collection,
    obj_uuid: str,
    *,
    vector_cache: Dict[str, List[float]],
) -> List[float]:
    if obj_uuid in vector_cache:
        return vector_cache[obj_uuid]

    data_object = collection.query.fetch_object_by_id(
        obj_uuid,
        include_vector=True,
    )
    if data_object is None:
        raise KeyError(f"Weaviate object was not found: {obj_uuid}")

    vector = extract_default_vector(data_object)
    if not vector:
        raise RuntimeError(f"Weaviate object has no default vector: {obj_uuid}")

    vector_cache[obj_uuid] = vector
    return vector


def query_weaviate_with_retries(callable_query, *, label: str):
    last_error: Optional[BaseException] = None
    for attempt in range(1, WEAVIATE_QUERY_MAX_RETRIES + 1):
        try:
            return callable_query()
        except WeaviateQueryError as exc:
            last_error = exc
            message = str(exc)
            if "HFRESH distancer is not yet initialized" not in message:
                raise
            if attempt >= WEAVIATE_QUERY_MAX_RETRIES:
                break
            print(
                f"Weaviate HFresh index not ready for {label}. "
                f"Retry {attempt}/{WEAVIATE_QUERY_MAX_RETRIES} after "
                f"{WEAVIATE_QUERY_RETRY_SLEEP_SECONDS}s."
            )
            time.sleep(WEAVIATE_QUERY_RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Weaviate query failed for {label}: {last_error!r}") from last_error


def metadata_score(obj: Any) -> float:
    try:
        value = obj.metadata.score
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def props_to_row(properties: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(properties or {})
    row.setdefault("object_type", "")
    row.setdefault("category_id", "")
    row.setdefault("article_id", "")
    row.setdefault("source_id", "")
    row.setdefault("title", "")
    row.setdefault("url", "")
    row.setdefault("final_category", "")
    row.setdefault("final_category_norm", normalize_category_combo(row.get("final_category")))
    row.setdefault("final_category_combo", category_combo_list(row.get("final_category")))
    row.setdefault("final_keywords", "")
    row.setdefault("final_highlight", "")
    row.setdefault("semantic_text", "")
    row.setdefault("colbert_text", "")
    return row


def search_similar_category_combinations(
    collection,
    query_row: pd.Series,
    *,
    vector_cache: Dict[str, List[float]],
    top_k: int = TOPIC_TOP_K,
) -> pd.DataFrame:
    query_category_norm = normalize_category_combo(
        query_row.get("final_category_norm") or query_row.get("final_category")
    )
    if not query_category_norm:
        return pd.DataFrame(
            columns=[
                "topic_rank",
                "topic_score",
                "category_id",
                "final_category",
                "final_category_norm",
                "final_category_combo",
            ]
        )

    query_uuid = category_object_uuid(query_category_norm)
    query_vector = get_object_vector_by_uuid(
        collection,
        query_uuid,
        vector_cache=vector_cache,
    )
    query_text = build_topic_text_from_category(query_category_norm)

    def do_query():
        return collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=float(TOPIC_HYBRID_ALPHA),
            query_properties=["topic_text", "final_category", "final_category_norm"],
            filters=Filter.by_property("object_type").equal("category"),
            limit=int(top_k),
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

    response = query_weaviate_with_retries(do_query, label="category-combination search")

    rows: List[Dict[str, Any]] = []
    seen = set()
    for obj in response.objects:
        props = props_to_row(obj.properties)
        category_norm = normalize_category_combo(
            props.get("final_category_norm") or props.get("final_category")
        )
        if not category_norm or category_norm in seen:
            continue
        seen.add(category_norm)
        props["topic_score"] = metadata_score(obj)
        props["final_category_norm"] = category_norm
        props["final_category_combo"] = [category_norm]
        rows.append(props)
        if len(rows) >= int(top_k):
            break

    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(
            columns=[
                "topic_rank",
                "topic_score",
                "category_id",
                "final_category",
                "final_category_norm",
                "final_category_combo",
            ]
        )
    output.insert(0, "topic_rank", range(1, len(output) + 1))
    return output


def filter_confident_topic_rows(
    similar_topics: pd.DataFrame,
    *,
    min_topic_score: float = MIN_TOPIC_SCORE,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "similar_topics_raw_count": int(len(similar_topics)),
        "similar_topics_after_score_filter_count": 0,
        "top_topic_score": 0.0,
        "min_topic_score": float(min_topic_score),
        "low_confidence_reason": "",
    }

    if similar_topics.empty or "topic_score" not in similar_topics.columns:
        output = similar_topics.iloc[0:0].copy()
        info["low_confidence_reason"] = "no_similar_topics"
        output.attrs["confidence_info"] = info
        return output, info

    scores = pd.to_numeric(similar_topics["topic_score"], errors="coerce").fillna(0.0)
    info["top_topic_score"] = float(scores.max()) if len(scores) else 0.0
    output = similar_topics.loc[scores >= float(min_topic_score)].copy()
    info["similar_topics_after_score_filter_count"] = int(len(output))
    if output.empty:
        info["low_confidence_reason"] = "no_topic_rows_above_min_topic_score"
    output.attrs["confidence_info"] = info
    return output, info


def category_filter_from_topic_rows_debug(
    similar_topics: pd.DataFrame,
) -> Tuple[List[str], Dict[str, Any]]:
    confident, info = filter_confident_topic_rows(similar_topics)
    combinations: List[str] = []
    seen = set()

    if not confident.empty:
        for value in confident["final_category_norm"].tolist():
            combo = normalize_category_combo(value)
            if combo and combo not in seen:
                seen.add(combo)
                combinations.append(combo)

    debug = dict(info)
    debug["allowed_category_combinations"] = combinations
    debug["category_filter_status"] = "ok" if combinations else "empty"
    if not combinations and not debug.get("low_confidence_reason"):
        debug["low_confidence_reason"] = "no_category_combinations_after_min_topic_score"

    print(
        "[TOPIC FILTER] "
        f"top_topic_score={float(debug.get('top_topic_score', 0.0)):.4f} | "
        f"kept={int(debug.get('similar_topics_after_score_filter_count', 0))}/"
        f"{int(debug.get('similar_topics_raw_count', 0))} | "
        f"min_topic_score={float(MIN_TOPIC_SCORE):.4f}"
    )
    print("[CATEGORY FILTER] allowed complete combinations:", combinations)
    if not combinations:
        print("[LOW CONFIDENCE]", debug.get("low_confidence_reason"))

    return combinations, debug


def empty_semantic_results(
    *,
    status: str,
    reason: str,
    extra: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    output = pd.DataFrame(
        columns=[
            "non_colbert_rank",
            "semantic_score",
            "article_id",
            "source_id",
            "title",
            "final_category",
            "final_keywords",
            "colbert_text",
        ]
    )
    output.attrs["status"] = status
    output.attrs["low_confidence_reason"] = reason
    if extra:
        output.attrs.update(extra)
    return output


def build_article_category_filter(combinations: Sequence[str]):
    normalized: List[str] = []
    seen = set()
    for value in combinations:
        combo = normalize_category_combo(value)
        if combo and combo not in seen:
            seen.add(combo)
            normalized.append(combo)

    if not normalized:
        return None

    return Filter.all_of([
        Filter.by_property("object_type").equal("article"),
        Filter.by_property("final_category_combo").contains_any(normalized),
    ])


def semantic_search_with_category_filter(
    collection,
    query_row: pd.Series,
    allowed_category_combinations: Sequence[str],
    *,
    vector_cache: Dict[str, List[float]],
    top_k: int = COARSE_TOP_K,
    exclude_self: bool = EXCLUDE_SELF,
    min_semantic_score: float = MIN_SEMANTIC_SCORE,
) -> pd.DataFrame:
    filters = build_article_category_filter(allowed_category_combinations)
    if filters is None:
        return empty_semantic_results(
            status="low_confidence",
            reason="no_category_combinations",
        )

    query_source_id = stringify(query_row.get("source_id"))
    query_uuid = article_object_uuid(query_source_id)
    query_vector = get_object_vector_by_uuid(
        collection,
        query_uuid,
        vector_cache=vector_cache,
    )
    query_text = stringify(query_row.get("semantic_text"))

    def do_query():
        return collection.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=float(SEMANTIC_HYBRID_ALPHA),
            query_properties=[
                "semantic_text",
                "title",
                "final_keywords",
                "final_highlight",
                "body_head",
            ],
            filters=filters,
            limit=int(top_k) + (1 if exclude_self else 0),
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )

    response = query_weaviate_with_retries(
        do_query,
        label="article semantic search with category filter",
    )

    rows: List[Dict[str, Any]] = []
    for obj in response.objects:
        props = props_to_row(obj.properties)
        if exclude_self and stringify(props.get("source_id")) == query_source_id:
            continue
        props["semantic_score"] = metadata_score(obj)
        try:
            props["explain_score"] = obj.metadata.explain_score
        except Exception:
            props["explain_score"] = None
        rows.append(props)
        if len(rows) >= int(top_k):
            break

    before_count = len(rows)
    best_score = max(
        [float(row.get("semantic_score") or 0.0) for row in rows],
        default=0.0,
    )
    output = pd.DataFrame(rows)

    if output.empty:
        return empty_semantic_results(
            status="low_confidence",
            reason="semantic_search_empty",
            extra={
                "semantic_results_before_filter": before_count,
                "semantic_results_after_filter": 0,
                "best_semantic_score": best_score,
                "min_semantic_score": float(min_semantic_score),
            },
        )

    output["semantic_score"] = pd.to_numeric(
        output["semantic_score"], errors="coerce"
    ).fillna(0.0)
    output = output.loc[
        output["semantic_score"] >= float(min_semantic_score)
    ].copy()
    after_count = len(output)

    if after_count < before_count:
        print(
            f"[SEMANTIC FILTER] kept {after_count}/{before_count} results with "
            f"semantic_score >= {float(min_semantic_score):.4f} "
            f"(best={best_score:.4f})"
        )

    if output.empty:
        return empty_semantic_results(
            status="low_confidence",
            reason="all_semantic_results_below_threshold",
            extra={
                "semantic_results_before_filter": before_count,
                "semantic_results_after_filter": 0,
                "best_semantic_score": best_score,
                "min_semantic_score": float(min_semantic_score),
            },
        )

    output = output.sort_values("semantic_score", ascending=False).reset_index(drop=True)
    output.insert(0, "non_colbert_rank", range(1, len(output) + 1))
    output.attrs.update({
        "status": "ok",
        "semantic_results_before_filter": before_count,
        "semantic_results_after_filter": after_count,
        "best_semantic_score": best_score,
        "min_semantic_score": float(min_semantic_score),
    })
    return output


# ======================================================================================
# 8. ColBERT helpers
# ======================================================================================


def pip_install_quiet(packages: Sequence[str]) -> None:
    raise RuntimeError(
        "Runtime package installation is disabled. Install optional ColBERT "
        "dependencies before starting the pipeline. Requested packages: "
        + ", ".join(packages)
    )


def ensure_huggingface_hub_download_helpers():
    """Return Hugging Face download helpers without changing the environment."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

        return hf_hub_download, snapshot_download
    except ImportError as first_error:
        if not AUTO_INSTALL_HUGGINGFACE_HUB:
            raise ImportError(
                "huggingface_hub is required to download the ColBERT model. "
                "Install the optional dependencies before starting the pipeline."
            ) from first_error

        print("[COLBERT] huggingface_hub is missing; installing it.")
        pip_install_quiet(["huggingface_hub"])
        invalidate_and_purge_modules(("huggingface_hub",))

        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

        return hf_hub_download, snapshot_download


def configure_huggingface_download_environment() -> Optional[str]:
    """Load HF_TOKEN from Colab Secrets without printing or persisting the token."""
    token = get_secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token

    os.environ.setdefault(
        "HF_HUB_DOWNLOAD_TIMEOUT",
        str(max(1, int(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS))),
    )
    os.environ.setdefault(
        "HF_HUB_ETAG_TIMEOUT",
        str(max(1, int(HF_HUB_ETAG_TIMEOUT_SECONDS))),
    )
    return token


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    """Return True when a tiny file is a Git LFS pointer instead of real model data."""
    try:
        if not path.is_file() or path.stat().st_size > 4096:
            return False
        prefix = path.read_text(encoding="utf-8", errors="ignore")[:256]
        return "git-lfs.github.com/spec/v1" in prefix
    except Exception:
        return False


def validate_colbert_model_directory(
    model_dir: Path,
) -> Tuple[bool, List[str]]:
    """Check that the local directory contains usable config, tokenizer, and weights."""
    model_dir = Path(model_dir)
    problems: List[str] = []

    if not model_dir.is_dir():
        return False, [f"directory does not exist: {model_dir}"]

    required_small_files = [
        "artifact.metadata",
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ]

    for filename in required_small_files:
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            problems.append(f"missing or empty: {filename}")
        elif _looks_like_git_lfs_pointer(path):
            problems.append(f"Git LFS pointer instead of file data: {filename}")

    weight_candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    valid_weight = None
    for path in weight_candidates:
        try:
            if (
                path.is_file()
                and path.stat().st_size >= int(COLBERT_MIN_WEIGHT_FILE_BYTES)
                and not _looks_like_git_lfs_pointer(path)
            ):
                valid_weight = path
                break
        except OSError:
            continue

    if valid_weight is None:
        found = {
            path.name: path.stat().st_size
            for path in weight_candidates
            if path.exists() and path.is_file()
        }
        problems.append(
            "no complete model weights found; expected model.safetensors or "
            f"pytorch_model.bin (found sizes={found})"
        )

    return not problems, problems


def _print_colbert_directory_status(model_dir: Path) -> None:
    """Print only filenames and sizes; never print secrets."""
    print("[COLBERT MODEL DIRECTORY STATUS]", model_dir)
    for filename in sorted(set(COLBERT_REQUIRED_MODEL_FILES) | {"pytorch_model.bin"}):
        path = model_dir / filename
        if path.is_file():
            print(f"  {filename}: {path.stat().st_size / (1024 ** 2):.1f} MiB")
        else:
            print(f"  {filename}: missing")


def ensure_colbert_model_downloaded(
    model_path: str = COLBERT_MODEL_PATH,
    *,
    local_dir: str = COLBERT_MODEL_LOCAL_DIR,
    revision: Optional[str] = COLBERT_MODEL_REVISION,
) -> str:
    """Return a verified local ColBERT directory and resume incomplete downloads.

    For a Hub repository ID, download only the files required for local PyTorch loading.
    Existing complete files are reused. Partial Hugging Face downloads are resumed by
    hf_hub_download through the metadata stored under local_dir/.cache/huggingface.
    """
    requested = stringify(model_path)
    if not requested:
        raise ValueError("COLBERT_MODEL_PATH must not be empty.")

    requested_path = Path(requested).expanduser()
    if requested_path.is_dir():
        valid, problems = validate_colbert_model_directory(requested_path)
        if not valid:
            _print_colbert_directory_status(requested_path)
            raise RuntimeError(
                "COLBERT_MODEL_PATH points to an incomplete local directory: "
                + "; ".join(problems)
            )
        resolved = str(requested_path.resolve())
        print("[COLBERT MODEL] using verified local directory:", resolved)
        return resolved

    destination = Path(local_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    token = configure_huggingface_download_environment()
    hf_hub_download, _snapshot_download = ensure_huggingface_hub_download_helpers()

    print("=" * 100)
    print("[PREPARE COLBERT MODEL]")
    print("repo_id:", requested)
    print("local_dir:", destination)
    print("revision:", stringify(revision) or "default")
    print("HF_TOKEN available:", bool(token))
    print("download policy: required files only; prefer model.safetensors")

    valid, problems = validate_colbert_model_directory(destination)
    if valid:
        print("[COLBERT MODEL CACHE HIT] verified local model:", destination)
        return str(destination.resolve())

    print("[COLBERT MODEL CACHE MISS OR INCOMPLETE]", "; ".join(problems))
    _print_colbert_directory_status(destination)

    common_kwargs: Dict[str, Any] = {
        "repo_id": requested,
        "repo_type": "model",
        "local_dir": str(destination),
        "token": token,
        "force_download": False,
    }
    if revision:
        common_kwargs["revision"] = stringify(revision)

    last_error: Optional[BaseException] = None
    max_attempts = max(1, int(COLBERT_DOWNLOAD_MAX_RETRIES))

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[COLBERT MODEL DOWNLOAD] attempt={attempt}/{max_attempts}")

            # Download files one by one so a missing checkpoint cannot be hidden by a
            # partially completed repository snapshot. hf_hub_download reuses complete
            # files and resumes partial data from local_dir/.cache/huggingface.
            for filename in COLBERT_REQUIRED_MODEL_FILES:
                target = destination / filename
                already_complete = False

                if filename == COLBERT_WEIGHT_FILENAME:
                    already_complete = (
                        target.is_file()
                        and target.stat().st_size >= int(COLBERT_MIN_WEIGHT_FILE_BYTES)
                        and not _looks_like_git_lfs_pointer(target)
                    )
                else:
                    already_complete = (
                        target.is_file()
                        and target.stat().st_size > 0
                        and not _looks_like_git_lfs_pointer(target)
                    )

                if already_complete:
                    print(f"[COLBERT FILE CACHE HIT] {filename}")
                    continue

                print(f"[COLBERT FILE DOWNLOAD] {filename}")
                hf_hub_download(
                    filename=filename,
                    **common_kwargs,
                )

            valid, problems = validate_colbert_model_directory(destination)
            if not valid:
                _print_colbert_directory_status(destination)
                raise RuntimeError(
                    "Hugging Face download returned, but the local ColBERT model is "
                    "still incomplete: " + "; ".join(problems)
                )

            resolved = str(destination.resolve())
            print("[COLBERT MODEL READY]", resolved)
            _print_colbert_directory_status(destination)
            return resolved

        except BaseException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            sleep_seconds = float(COLBERT_DOWNLOAD_RETRY_SLEEP_SECONDS) * (2 ** (attempt - 1))
            print(
                f"[COLBERT MODEL DOWNLOAD RETRY] sleep={sleep_seconds:.1f}s "
                f"error={type(exc).__name__}: {exc}"
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"Failed to prepare a complete ColBERT model {requested!r} after "
        f"{max_attempts} attempts: {last_error!r}"
    ) from last_error

def invalidate_and_purge_modules(prefixes: Tuple[str, ...]) -> None:
    importlib.invalidate_caches()
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(prefix + ".") for prefix in prefixes):
            del sys.modules[module_name]


def try_import_ragatouille_class() -> Tuple[Optional[Any], Optional[BaseException]]:
    try:
        from ragatouille import RAGPretrainedModel  # type: ignore

        return RAGPretrainedModel, None
    except BaseException as exc:
        return None, exc


def error_mentions_langchain_retrievers(exc: BaseException) -> bool:
    return "langchain.retrievers" in (repr(exc) + "\n" + str(exc))


def ensure_ragatouille_installed_for_colab():
    """Load RAGatouille and fail clearly instead of installing at runtime."""
    model_class, first_error = try_import_ragatouille_class()
    if model_class is not None:
        return model_class

    if not AUTO_INSTALL_RAGATOUILLE:
        raise ImportError(
            f"ragatouille could not be imported: {first_error!r}"
        ) from first_error

    print("[COLBERT] ragatouille import failed; attempting automatic repair.")
    print("[COLBERT] first import error:", repr(first_error))
    pip_install_quiet(["ragatouille==0.0.9.post2"])
    invalidate_and_purge_modules(("ragatouille",))

    model_class, second_error = try_import_ragatouille_class()
    if model_class is not None:
        return model_class

    if second_error is not None and error_mentions_langchain_retrievers(second_error):
        if not COLBERT_AUTO_INSTALL_COMPAT_DEPS:
            raise ImportError(
                "ragatouille import failed because langchain.retrievers is missing."
            ) from second_error

        print("[COLBERT] repairing LangChain compatibility for RAGatouille.")
        pip_install_quiet([
            "langchain==0.1.20",
            "langchain-community==0.0.38",
            "langchain-core==0.1.52",
            "langchain-text-splitters==0.0.2",
        ])
        invalidate_and_purge_modules((
            "ragatouille",
            "langchain",
            "langchain_core",
            "langchain_community",
            "langchain_text_splitters",
        ))
        model_class, third_error = try_import_ragatouille_class()
        if model_class is not None:
            print("[COLBERT] ragatouille import repaired successfully.")
            return model_class
        raise ImportError(
            "ragatouille import still failed after compatibility repair. "
            f"Restart Colab if necessary. Last error: {third_error!r}"
        ) from third_error

    raise ImportError(
        f"ragatouille was installed but could not be imported: {second_error!r}"
    ) from second_error


def patch_colbert_transformers_compat() -> None:
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception as exc:
        print("[COLBERT] Transformers compatibility patch skipped:", repr(exc))
        return

    if not getattr(PreTrainedModel, "_guardian_similarity_tied_weights_patch", False):
        original_move = getattr(PreTrainedModel, "_move_missing_keys_from_meta_to_device", None)
        if original_move is not None:
            def patched_move(self, missing_keys, *args, **kwargs):
                try:
                    tied = getattr(self, "all_tied_weights_keys")
                except AttributeError:
                    tied = None
                if tied is None or not hasattr(tied, "keys"):
                    try:
                        setattr(self, "all_tied_weights_keys", {})
                    except Exception:
                        pass
                return original_move(self, missing_keys, *args, **kwargs)

            PreTrainedModel._move_missing_keys_from_meta_to_device = patched_move
            PreTrainedModel._guardian_similarity_tied_weights_patch = True

    if not getattr(PreTrainedModel, "_guardian_similarity_ignore_keys_patch", False):
        original_adjust = getattr(PreTrainedModel, "_adjust_missing_and_unexpected_keys", None)
        if original_adjust is not None:
            def patched_adjust(self, loading_info, *args, **kwargs):
                for attr_name in [
                    "_keys_to_ignore_on_load_missing",
                    "_keys_to_ignore_on_load_unexpected",
                ]:
                    value = getattr(self, attr_name, None)
                    if value is not None and not isinstance(value, set):
                        try:
                            setattr(self, attr_name, set(value))
                        except Exception:
                            pass
                return original_adjust(self, loading_info, *args, **kwargs)

            PreTrainedModel._adjust_missing_and_unexpected_keys = patched_adjust
            PreTrainedModel._guardian_similarity_ignore_keys_patch = True


def load_colbert_reranker(model_path: str = COLBERT_MODEL_PATH):
    model_class = ensure_ragatouille_installed_for_colab()
    local_model_path = ensure_colbert_model_downloaded(model_path)

    print("=" * 100)
    print("[LOAD COLBERT RERANKER]")
    print("repo_or_input_path:", model_path)
    print("resolved_local_path:", local_model_path)
    patch_colbert_transformers_compat()

    try:
        return model_class.from_pretrained(local_model_path)
    except (AttributeError, TypeError):
        patch_colbert_transformers_compat()
        cleanup_cuda()
        return model_class.from_pretrained(local_model_path)


def close_colbert_reranker(reranker: Any) -> None:
    try:
        del reranker
    except Exception:
        pass
    cleanup_cuda()


def json_safe_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_colbert_results(results: Any, documents: List[str]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    used_indices = set()

    def find_unused_document(content: str) -> int:
        content = stringify(content)
        for index, document in enumerate(documents):
            if index not in used_indices and stringify(document) == content:
                used_indices.add(index)
                return index
        for index in range(len(documents)):
            if index not in used_indices:
                used_indices.add(index)
                return index
        return 0

    for rank, item in enumerate(list(results or []), start=1):
        document_index: Optional[int] = None
        score: Any = None
        content = ""

        if isinstance(item, dict):
            for key in ["result_index", "document_id", "doc_id", "passage_id", "index", "idx"]:
                if key in item:
                    try:
                        document_index = int(item[key])
                        break
                    except Exception:
                        pass
            for key in ["score", "colbert_score", "similarity"]:
                if key in item:
                    score = item.get(key)
                    break
            for key in ["content", "document", "text"]:
                if key in item:
                    content = stringify(item.get(key))
                    break
        elif isinstance(item, (tuple, list)):
            values = list(item)
            if values:
                if isinstance(values[0], int):
                    document_index = int(values[0])
                elif isinstance(values[0], str):
                    content = values[0]
            if len(values) >= 2:
                score = values[-1]

        if document_index is None or not (0 <= document_index < len(documents)):
            document_index = find_unused_document(content)
        else:
            used_indices.add(document_index)

        normalized.append({
            "rank": rank,
            "doc_index": document_index,
            "score": json_safe_scalar(score),
        })

    return normalized


def colbert_rerank(
    reranker: Any,
    query_text: str,
    coarse_results: pd.DataFrame,
    *,
    top_k: int = FINAL_TOP_K,
) -> pd.DataFrame:
    if coarse_results.empty:
        return coarse_results.copy()

    documents = [stringify(value) or " " for value in coarse_results["colbert_text"].tolist()]
    k = min(int(top_k), len(documents))

    try:
        raw_results = reranker.rerank(
            query=stringify(query_text),
            documents=documents,
            k=k,
        )
    except TypeError:
        raw_results = reranker.rerank(stringify(query_text), documents, k=k)
    except Exception:
        if not COLBERT_FAIL_OPEN:
            raise
        print("[COLBERT] rerank failed; returning semantic order because COLBERT_FAIL_OPEN=True.")
        output = coarse_results.head(k).copy()
        output["colbert_rank"] = range(1, len(output) + 1)
        output["colbert_score"] = np.nan
        return output

    normalized = normalize_colbert_results(raw_results, documents)
    selected_positions = [int(item["doc_index"]) for item in normalized[:k]]
    scores = [item.get("score") for item in normalized[:k]]

    if not selected_positions:
        output = coarse_results.head(k).copy()
        output["colbert_rank"] = range(1, len(output) + 1)
        output["colbert_score"] = np.nan
        return output

    output = coarse_results.iloc[selected_positions].copy()
    output["colbert_rank"] = range(1, len(output) + 1)
    output["colbert_score"] = scores
    return output


def compare_colbert_vs_non_colbert(
    coarse_results: pd.DataFrame,
    colbert_results: pd.DataFrame,
) -> pd.DataFrame:
    keep = [
        "article_id",
        "title",
        "final_category",
        "final_keywords",
        "semantic_score",
        "non_colbert_rank",
    ]
    required_right = ["article_id", "colbert_rank", "colbert_score"]
    if coarse_results.empty or colbert_results.empty:
        return pd.DataFrame(columns=keep + ["colbert_rank", "colbert_score", "rank_change"])
    if any(column not in colbert_results.columns for column in required_right):
        return pd.DataFrame(columns=keep + ["colbert_rank", "colbert_score", "rank_change"])

    left = coarse_results[[column for column in keep if column in coarse_results.columns]].copy()
    right = colbert_results[required_right].copy()
    output = left.merge(right, on="article_id", how="inner")
    if output.empty:
        return output
    output["rank_change"] = output["non_colbert_rank"] - output["colbert_rank"]
    return output.sort_values("colbert_rank")


# ======================================================================================
# 9. Console output helpers
# ======================================================================================


def print_query_article(row: pd.Series, index: int, total: int) -> None:
    if not PRINT_QUERY_DETAILS:
        return
    print("\n" + "=" * 100)
    print(f"Query Article {index}/{total}")
    print("article_id:", row.get("article_id"))
    print("title:", row.get("title"))
    print("category:", row.get("final_category"))
    print("keywords:", row.get("final_keywords"))
    print(f"body_head first {BODY_CHARS} chars:")
    print(stringify(row.get("body_head")))
    print("=" * 100)


def print_similar_topics_df(dataframe: pd.DataFrame) -> None:
    if not PRINT_SIMILAR_TOPICS:
        return
    print("\nTop Similar Complete Category Combinations:")
    if dataframe.empty:
        print("  <empty>")
        return
    for _, row in dataframe.iterrows():
        print(
            f"{int(row['topic_rank']):>2}. "
            f"score={float(row['topic_score']):.4f} | "
            f"category={row.get('final_category') or row.get('final_category_norm')}"
        )


def print_non_colbert_df(dataframe: pd.DataFrame, max_rows: int = 10) -> None:
    if not PRINT_NON_COLBERT_RESULTS:
        return
    print("\nNon-ColBERT Semantic Results:")
    if dataframe.empty:
        print("  <empty>")
        return
    for _, row in dataframe.head(max_rows).iterrows():
        print(
            f"{int(row['non_colbert_rank']):>2}. "
            f"semantic_score={float(row['semantic_score']):.4f} | "
            f"category={row.get('final_category')} | "
            f"keywords={short(row.get('final_keywords'), 80)} | "
            f"title={short(row.get('title'), 120)}"
        )


def print_colbert_df(dataframe: pd.DataFrame, max_rows: int = 10) -> None:
    if not PRINT_COLBERT_RESULTS:
        return
    print("\nColBERT Reranked Results:")
    if dataframe.empty:
        print("  <empty>")
        return
    for _, row in dataframe.head(max_rows).iterrows():
        colbert_score = row.get("colbert_score")
        semantic_score = row.get("semantic_score")
        colbert_text = "nan" if pd.isna(colbert_score) else f"{float(colbert_score):.4f}"
        semantic_text = "nan" if pd.isna(semantic_score) else f"{float(semantic_score):.4f}"
        print(
            f"{int(row['colbert_rank']):>2}. "
            f"colbert_score={colbert_text} | "
            f"semantic_score={semantic_text} | "
            f"category={row.get('final_category')} | "
            f"keywords={short(row.get('final_keywords'), 80)} | "
            f"title={short(row.get('title'), 120)}"
        )


def print_comparison_df(dataframe: pd.DataFrame) -> None:
    if not PRINT_COMPARISON:
        return
    print("\nColBERT vs Non-ColBERT Comparison:")
    if dataframe.empty:
        print("  <empty>")
        return
    columns = [
        "article_id",
        "title",
        "non_colbert_rank",
        "colbert_rank",
        "semantic_score",
        "colbert_score",
        "rank_change",
    ]
    view = dataframe[[column for column in columns if column in dataframe.columns]].copy()
    with pd.option_context("display.max_colwidth", 80, "display.width", 180):
        print(view.to_string(index=False))


# ======================================================================================
# 10. Pipeline execution
# ======================================================================================


def build_final_result_rows(
    *,
    query_index: int,
    query_row: pd.Series,
    topic_debug: Dict[str, Any],
    coarse_results: pd.DataFrame,
    final_ranked: pd.DataFrame,
) -> List[Dict[str, Any]]:
    status = stringify(coarse_results.attrs.get("status", "ok")) or "ok"
    reason = stringify(coarse_results.attrs.get("low_confidence_reason", ""))
    allowed_combinations = topic_debug.get("allowed_category_combinations", [])

    if final_ranked.empty:
        return [{
            "query_index": int(query_index),
            "query_article_id": stringify(query_row.get("article_id")),
            "query_source_id": stringify(query_row.get("source_id")),
            "query_title": stringify(query_row.get("title")),
            "query_final_category": stringify(query_row.get("final_category")),
            "query_final_keywords": stringify(query_row.get("final_keywords")),
            "allowed_category_combinations": allowed_combinations,
            "status": status if status != "ok" else "empty",
            "low_confidence_reason": reason or "no_final_results",
            "candidate_article_id": "",
            "candidate_source_id": "",
            "candidate_title": "",
            "candidate_url": "",
            "candidate_final_category": "",
            "candidate_final_keywords": "",
            "semantic_score": None,
            "non_colbert_rank": None,
            "colbert_score": None,
            "colbert_rank": None,
            "rank_change": None,
        }]

    rows: List[Dict[str, Any]] = []
    for _, candidate in final_ranked.iterrows():
        non_colbert_rank = candidate.get("non_colbert_rank")
        colbert_rank = candidate.get("colbert_rank")
        rank_change = None
        if pd.notna(non_colbert_rank) and pd.notna(colbert_rank):
            rank_change = int(non_colbert_rank) - int(colbert_rank)

        rows.append({
            "query_index": int(query_index),
            "query_article_id": stringify(query_row.get("article_id")),
            "query_source_id": stringify(query_row.get("source_id")),
            "query_title": stringify(query_row.get("title")),
            "query_final_category": stringify(query_row.get("final_category")),
            "query_final_keywords": stringify(query_row.get("final_keywords")),
            "allowed_category_combinations": allowed_combinations,
            "status": "ok",
            "low_confidence_reason": "",
            "candidate_article_id": stringify(candidate.get("source_id")),
            "candidate_source_id": stringify(candidate.get("source_id")),
            "candidate_title": stringify(candidate.get("title")),
            "candidate_url": stringify(candidate.get("url")),
            "candidate_final_category": stringify(candidate.get("final_category")),
            "candidate_final_keywords": stringify(candidate.get("final_keywords")),
            "semantic_score": json_safe_scalar(candidate.get("semantic_score")),
            "non_colbert_rank": json_safe_scalar(non_colbert_rank),
            "colbert_score": json_safe_scalar(candidate.get("colbert_score")),
            "colbert_rank": json_safe_scalar(colbert_rank),
            "rank_change": rank_change,
        })
    return rows


def run_guardian_similar_articles_pipeline(
    *,
    process_n: Optional[int] = PROCESS_N,
    use_colbert: bool = USE_COLBERT,
    source_ids: Optional[Sequence[str]] = None,
    collection_name: str = SIMILARITY_COLLECTION_NAME,
) -> Dict[str, Any]:
    base_table = load_similarity_base_table_from_postgres()

    current_source_ids = None
    if source_ids is not None:
        current_source_ids = {stringify(value) for value in source_ids if stringify(value)}
        if not current_source_ids:
            raise ValueError("source_ids was provided but contained no usable IDs.")

    if current_source_ids is None:
        sync_table = base_table
        query_table = base_table.copy()
        delete_stale = DELETE_STALE_SIMILARITY_OBJECTS
    else:
        sync_table = base_table[
            base_table["source_id"].isin(current_source_ids)
        ].copy()
        query_table = sync_table.copy()
        delete_stale = False
        if sync_table.empty:
            raise ValueError(
                "None of the requested source_ids have confirmed judge results."
            )

    client = None
    reranker = None
    try:
        client = connect_weaviate_from_env()
        collection = ensure_similarity_collection_schema(
            client,
            collection_name=collection_name,
        )
        embedder = OpenAIEmbeddingClient()

        sync_stats = synchronize_similarity_objects(
            collection=collection,
            base=sync_table,
            embedder=embedder,
            delete_stale=delete_stale,
        )

        if process_n is not None:
            query_table = query_table.head(max(0, int(process_n))).copy()
        query_table = query_table.reset_index(drop=True)

        if query_table.empty:
            raise ValueError("The query table is empty after applying PROCESS_N.")

        if use_colbert:
            reranker = load_colbert_reranker()

        vector_cache: Dict[str, List[float]] = {}
        all_result_rows: List[Dict[str, Any]] = []
        recommendation_frames: List[pd.DataFrame] = []
        recommendation_db_stats = {
            "processed": 0,
            "with_recommendations": 0,
            "empty_lists": 0,
            "total_candidate_ids": 0,
        }

        for query_position, (_, query_row) in enumerate(query_table.iterrows(), start=1):
            print_query_article(query_row, query_position, len(query_table))

            similar_topics = search_similar_category_combinations(
                collection,
                query_row,
                vector_cache=vector_cache,
            )
            print_similar_topics_df(similar_topics)

            allowed_combinations, topic_debug = category_filter_from_topic_rows_debug(
                similar_topics
            )

            coarse_results = semantic_search_with_category_filter(
                collection,
                query_row,
                allowed_combinations,
                vector_cache=vector_cache,
            )
            print_non_colbert_df(coarse_results)

            if len(coarse_results) < int(MIN_COARSE_RESULTS_WARN):
                print(
                    f"[WARN] Only {len(coarse_results)} semantic candidates remained; "
                    f"FINAL_TOP_K={FINAL_TOP_K}."
                )

            if use_colbert and reranker is not None and not coarse_results.empty:
                final_ranked = colbert_rerank(
                    reranker,
                    stringify(query_row.get("colbert_text")),
                    coarse_results,
                    top_k=FINAL_TOP_K,
                )
                print_colbert_df(final_ranked)
                comparison = compare_colbert_vs_non_colbert(
                    coarse_results,
                    final_ranked,
                )
                print_comparison_df(comparison)
            else:
                final_ranked = coarse_results.head(FINAL_TOP_K).copy()
                if not final_ranked.empty:
                    final_ranked["colbert_rank"] = final_ranked["non_colbert_rank"]
                    final_ranked["colbert_score"] = np.nan

            query_result_rows = build_final_result_rows(
                query_index=query_position - 1,
                query_row=query_row,
                topic_debug=topic_debug,
                coarse_results=coarse_results,
                final_ranked=final_ranked,
            )
            all_result_rows.extend(query_result_rows)

            query_recommendation_records = build_recommendation_records(
                query_table=query_row.to_frame().T,
                final_results=pd.DataFrame(query_result_rows),
                max_items=FINAL_TOP_K,
            )
            recommendation_frames.append(query_recommendation_records)
            if SAVE_RECOMMENDATIONS_TO_POSTGRES:
                query_db_stats = save_recommendations_to_postgres(
                    query_recommendation_records
                )
                for key in recommendation_db_stats:
                    recommendation_db_stats[key] += int(query_db_stats.get(key, 0))

        final_results = pd.DataFrame(all_result_rows)

        if recommendation_frames:
            recommendation_records = pd.concat(
                recommendation_frames,
                ignore_index=True,
            )
        else:
            recommendation_records = pd.DataFrame(
                columns=[
                    "source_id",
                    "recommended_source_ids",
                    "recommendation_count",
                ]
            )
        if not SAVE_RECOMMENDATIONS_TO_POSTGRES:
            print("[POSTGRESQL RECOMMENDATION UPSERT SKIPPED]")

        print("=" * 100)
        print("[SIMILAR ARTICLES PIPELINE DONE]")
        print("candidate library rows:", len(base_table))
        print("query articles processed:", len(query_table))
        print("final result rows:", len(final_results))
        print("stored vectors reused during queries:", len(vector_cache))
        print("No CSV or JSONL files were written.")
        print("recommendation rows persisted:", recommendation_db_stats.get("processed", 0))

        return {
            "base_table": base_table,
            "query_table": query_table,
            "final_results": final_results,
            "recommendation_records": recommendation_records,
            "recommendation_db_stats": recommendation_db_stats,
            "sync_stats": sync_stats,
            "vector_cache_size": len(vector_cache),
        }

    finally:
        if reranker is not None:
            close_colbert_reranker(reranker)
        if client is not None:
            client.close()
            print("Weaviate client closed.")
        cleanup_cuda()


# ======================================================================================
# 11. Colab direct execution block
# ======================================================================================
