"""PostgreSQL queries for bounded, stage-aware RAG recovery."""

from __future__ import annotations

from typing import Sequence


RAW_ARTICLES_TABLE = "raw_articles"
MODEL_OUTPUTS_TABLE = "model_outputs"
JUDGE_RESULTS_TABLE = "judge_results"
RECOMMENDATIONS_TABLE = "article_recommendations"
ALLOWED_QUALITY_STATUSES = ("OK", "REVISED")


def _clean_source_ids(source_ids: Sequence[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(source_id).strip()
            for source_id in source_ids
            if str(source_id).strip()
        )
    )


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute("SELECT to_regclass(%s) AS relation_name;", (table_name,))
    row = cursor.fetchone()
    return bool(row and row["relation_name"] is not None)


def load_pending_retrieval_source_ids(
    *,
    database_url: str,
    limit: int,
    exclude_source_ids: Sequence[str] = (),
) -> list[str]:
    """Return generation-complete IDs without a persisted Judge result.

    Retrieval has file outputs rather than a database completion table, so these
    rows are intentionally retrieved again before Judge resumes.
    """

    if limit <= 0:
        return []

    import psycopg
    from psycopg.rows import dict_row

    excluded = _clean_source_ids(exclude_source_ids)
    exclude_filter = ""
    params: list[object] = []
    if excluded:
        exclude_filter = "AND NOT (raw.source_id = ANY(%s))"
        params.append(excluded)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            judge_table_exists = _table_exists(cursor, JUDGE_RESULTS_TABLE)
            judge_filter = ""
            if judge_table_exists:
                judge_filter = f"""
                AND NOT EXISTS (
                    SELECT 1
                    FROM {JUDGE_RESULTS_TABLE} AS judge
                    WHERE judge.source_id = raw.source_id
                )
                """

            query = f"""
            SELECT raw.source_id
            FROM {RAW_ARTICLES_TABLE} AS raw
            WHERE EXISTS (
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
            )
              {judge_filter}
              {exclude_filter}
            ORDER BY raw.published_at DESC NULLS LAST, raw.source_id
            LIMIT %s;
            """
            params.append(int(limit))
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    return [str(row["source_id"]) for row in rows]


def load_pending_similarity_source_ids(
    *,
    database_url: str,
    limit: int,
    exclude_source_ids: Sequence[str] = (),
) -> list[str]:
    """Return similarity-eligible Judge rows without a recommendation row."""

    if limit <= 0:
        return []

    import psycopg
    from psycopg.rows import dict_row

    excluded = _clean_source_ids(exclude_source_ids)
    exclude_filter = ""
    params: list[object] = [list(ALLOWED_QUALITY_STATUSES)]
    if excluded:
        exclude_filter = "AND NOT (raw.source_id = ANY(%s))"
        params.append(excluded)

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            if not _table_exists(cursor, JUDGE_RESULTS_TABLE):
                return []
            recommendations_table_exists = _table_exists(
                cursor,
                RECOMMENDATIONS_TABLE,
            )
            recommendation_filter = ""
            if recommendations_table_exists:
                recommendation_filter = f"""
                AND NOT EXISTS (
                    SELECT 1
                    FROM {RECOMMENDATIONS_TABLE} AS recommendation
                    WHERE recommendation.source_id = judge.source_id
                )
                """

            query = f"""
            SELECT judge.source_id
            FROM {JUDGE_RESULTS_TABLE} AS judge
            INNER JOIN {RAW_ARTICLES_TABLE} AS raw
                ON raw.source_id = judge.source_id
            WHERE judge.final_quality_status = ANY(%s)
              AND judge.any_parse_failed = FALSE
              {recommendation_filter}
              {exclude_filter}
            ORDER BY raw.published_at DESC NULLS LAST, judge.source_id
            LIMIT %s;
            """
            params.append(int(limit))
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

    return [str(row["source_id"]) for row in rows]


def load_judge_complete_source_ids(
    *,
    database_url: str,
    source_ids: Sequence[str],
) -> list[str]:
    """Return requested IDs that have actually been persisted by Judge."""

    cleaned_source_ids = _clean_source_ids(source_ids)
    if not cleaned_source_ids:
        return []

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        with conn.cursor() as cursor:
            if not _table_exists(cursor, JUDGE_RESULTS_TABLE):
                return []
            cursor.execute(
                f"""
                SELECT source_id
                FROM {JUDGE_RESULTS_TABLE}
                WHERE source_id = ANY(%s);
                """,
                (cleaned_source_ids,),
            )
            completed = {str(row["source_id"]) for row in cursor.fetchall()}

    return [
        source_id for source_id in cleaned_source_ids if source_id in completed
    ]
