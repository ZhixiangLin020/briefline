from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import psycopg


CATEGORY_BROAD_MAPPING_TABLE = "category_broad_mapping"


def save_category_broad_mapping_to_postgres(
    mapping_df: pd.DataFrame,
    *,
    database_url: str,
    table_name: str = CATEGORY_BROAD_MAPPING_TABLE,
) -> Dict[str, int]:
    """
    Synchronize the current broad-category mapping into one PostgreSQL table.

    Database state after completion:
    - one row per source_category
    - no taxonomy versions
    - no taxonomy JSON
    - no model metadata
    - stale mappings are deleted
    - unchanged mappings are not updated
    """

    required_columns = {
        "source_category",
        "broad_category",
        "broad_description",
        "broad_category_rank",
        "article_count",
    }

    missing_columns = required_columns - set(mapping_df.columns)

    if missing_columns:
        raise ValueError(
            "mapping_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    mapping = mapping_df[
        [
            "source_category",
            "broad_category",
            "broad_description",
            "broad_category_rank",
            "article_count",
        ]
    ].copy()

    # Normalize text fields.
    for column in [
        "source_category",
        "broad_category",
        "broad_description",
    ]:
        mapping[column] = (
            mapping[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Normalize integer fields.
    mapping["broad_category_rank"] = (
        pd.to_numeric(
            mapping["broad_category_rank"],
            errors="raise",
        )
        .astype(int)
    )

    mapping["article_count"] = (
        pd.to_numeric(
            mapping["article_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # Reject invalid rows.
    if mapping["source_category"].eq("").any():
        raise ValueError(
            "mapping_df contains an empty source_category."
        )

    if mapping["broad_category"].eq("").any():
        raise ValueError(
            "mapping_df contains an empty broad_category."
        )

    duplicate_mask = mapping["source_category"].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicates = sorted(
            mapping.loc[
                duplicate_mask,
                "source_category",
            ].unique()
        )

        raise ValueError(
            "Each source_category must occur exactly once. "
            f"Duplicates: {duplicates}"
        )

    mapping = mapping.sort_values(
        [
            "broad_category_rank",
            "source_category",
        ]
    ).reset_index(drop=True)

    records = [
        (
            row.source_category,
            row.broad_category,
            row.broad_description,
            int(row.broad_category_rank),
            int(row.article_count),
        )
        for row in mapping.itertuples(index=False)
    ]

    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        source_category TEXT PRIMARY KEY,
        broad_category TEXT NOT NULL,
        broad_description TEXT NOT NULL,
        broad_category_rank INTEGER NOT NULL,
        article_count INTEGER NOT NULL DEFAULT 0
    );
    """

    create_stage_sql = """
    CREATE TEMP TABLE category_broad_mapping_stage (
        source_category TEXT PRIMARY KEY,
        broad_category TEXT NOT NULL,
        broad_description TEXT NOT NULL,
        broad_category_rank INTEGER NOT NULL,
        article_count INTEGER NOT NULL
    )
    ON COMMIT DROP;
    """

    insert_stage_sql = """
    INSERT INTO category_broad_mapping_stage (
        source_category,
        broad_category,
        broad_description,
        broad_category_rank,
        article_count
    )
    VALUES (
        %s,
        %s,
        %s,
        %s,
        %s
    );
    """

    # Delete rows that no longer exist in the current mapping.
    delete_stale_sql = f"""
    DELETE FROM {table_name} AS target
    WHERE NOT EXISTS (
        SELECT 1
        FROM category_broad_mapping_stage AS stage
        WHERE stage.source_category
            = target.source_category
    );
    """

    # Insert new rows and update only genuinely changed rows.
    upsert_sql = f"""
    INSERT INTO {table_name} (
        source_category,
        broad_category,
        broad_description,
        broad_category_rank,
        article_count
    )
    SELECT
        source_category,
        broad_category,
        broad_description,
        broad_category_rank,
        article_count
    FROM category_broad_mapping_stage
    ON CONFLICT (source_category)
    DO UPDATE SET
        broad_category =
            EXCLUDED.broad_category,

        broad_description =
            EXCLUDED.broad_description,

        broad_category_rank =
            EXCLUDED.broad_category_rank,

        article_count =
            EXCLUDED.article_count

    WHERE ROW(
        {table_name}.broad_category,
        {table_name}.broad_description,
        {table_name}.broad_category_rank,
        {table_name}.article_count
    )
    IS DISTINCT FROM
    ROW(
        EXCLUDED.broad_category,
        EXCLUDED.broad_description,
        EXCLUDED.broad_category_rank,
        EXCLUDED.article_count
    );
    """

    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(create_table_sql)
            cursor.execute(create_stage_sql)

            cursor.executemany(
                insert_stage_sql,
                records,
            )

            cursor.execute(delete_stale_sql)
            deleted = max(
                int(cursor.rowcount),
                0,
            )

            cursor.execute(upsert_sql)
            written = max(
                int(cursor.rowcount),
                0,
            )

            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM {table_name};
                """
            )

            final_count = int(
                cursor.fetchone()[0]
            )

    unchanged = max(
        len(records) - written,
        0,
    )

    stats = {
        "input_rows": len(records),
        "written": written,
        "unchanged": unchanged,
        "deleted": deleted,
        "database_rows": final_count,
    }

    print("=" * 100)
    print("[CATEGORY BROAD MAPPING SAVED]")
    print("table:", table_name)
    print("input rows:", stats["input_rows"])
    print("inserted or updated:", stats["written"])
    print("unchanged:", stats["unchanged"])
    print("deleted stale rows:", stats["deleted"])
    print("database rows:", stats["database_rows"])

    return stats
