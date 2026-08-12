"""Generate a broad taxonomy for confirmed Guardian categories."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg
from google import genai
from psycopg.rows import dict_row


# ======================================================================================
# Config
# ======================================================================================

GEMINI_MODEL = "gemini-3.5-flash"

RAW_ARTICLES_TABLE = "raw_articles"
JUDGE_RESULTS_TABLE = "judge_results"

MIN_BROAD_CATEGORIES = 8
MAX_BROAD_CATEGORIES = 14

# Low temperature makes taxonomy generation more stable.
GEMINI_TEMPERATURE = 0.1
MAX_GENERATION_ATTEMPTS = 3


# ======================================================================================
# Secrets
# ======================================================================================

def get_secret(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        try:
            from google.colab import userdata  # type: ignore

            value = userdata.get(name)
        except Exception:
            value = None

    if not value:
        raise RuntimeError(
            f"Missing {name}. Set it in the environment or Colab Secrets."
        )

    return str(value)


# ======================================================================================
# Load unique final categories from PostgreSQL
# ======================================================================================

def load_final_category_inventory(
    *,
    database_url: Optional[str] = None,
    judge_results_table: str = JUDGE_RESULTS_TABLE,
) -> pd.DataFrame:
    if not database_url:
        database_url = get_secret("DATABASE_URL")
    query = f"""
    SELECT
        BTRIM(final_category) AS source_category,
        COUNT(*)::INTEGER AS article_count
    FROM {judge_results_table}
    WHERE final_quality_status IN ('OK', 'REVISED')
      AND NULLIF(BTRIM(final_category), '') IS NOT NULL
    GROUP BY BTRIM(final_category)
    ORDER BY
        article_count DESC,
        source_category;
    """

    with psycopg.connect(
        database_url,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()

    category_df = pd.DataFrame(rows)

    if category_df.empty:
        raise ValueError(
            "No confirmed final_category values were found in judge_results."
        )

    category_df["source_category"] = (
        category_df["source_category"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    category_df["article_count"] = (
        pd.to_numeric(
            category_df["article_count"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    category_df = category_df[
        category_df["source_category"].ne("")
    ].copy()

    category_df = (
        category_df
        .drop_duplicates("source_category")
        .reset_index(drop=True)
    )

    print("=" * 100)
    print("[FINAL CATEGORY INVENTORY]")
    print("unique categories:", len(category_df))
    print("article count:", int(category_df["article_count"].sum()))

    return category_df


# ======================================================================================
# Dynamic structured-output schema
# ======================================================================================

def build_taxonomy_schema(
    source_categories: List[str],
) -> Dict[str, Any]:
    """Only allow exact category strings supplied by PostgreSQL."""

    return {
        "type": "object",
        "required": [
            "taxonomy_name",
            "groups",
        ],
        "properties": {
            "taxonomy_name": {
                "type": "string",
                "description": (
                    "A concise English name for the generated news taxonomy."
                ),
            },
            "groups": {
                "type": "array",
                "description": (
                    "Broad news groups that collectively contain every "
                    "source category exactly once."
                ),
                "items": {
                    "type": "object",
                    "required": [
                        "broad_category",
                        "description",
                        "source_categories",
                    ],
                    "properties": {
                        "broad_category": {
                            "type": "string",
                            "description": (
                                "A concise, stable, freely created broad "
                                "news category name in English."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "A short description of the scope of this "
                                "broad category."
                            ),
                        },
                        "source_categories": {
                            "type": "array",
                            "description": (
                                "Existing source categories assigned to this "
                                "broad category. Preserve every string exactly."
                            ),
                            "items": {
                                "type": "string",
                                "enum": source_categories,
                            },
                        },
                    },
                },
            },
        },
    }


# ======================================================================================
# Prompt
# ======================================================================================

def build_taxonomy_prompt(
    category_df: pd.DataFrame,
    *,
    min_groups: int,
    max_groups: int,
    previous_error: str = "",
) -> str:
    category_records = [
        {
            "category": str(row.source_category),
            "article_count": int(row.article_count),
        }
        for row in category_df.itertuples(index=False)
    ]

    repair_text = ""

    if previous_error:
        repair_text = f"""
The previous attempt failed local validation:
{previous_error}

Generate the complete taxonomy again from scratch and correct those problems.
"""

    return f"""Design a broad taxonomy for a Guardian news dataset.

The input contains existing category labels and their article frequencies.

Create between {min_groups} and {max_groups} broad news categories.
You are free to create suitable broad-category names.

Rules:
- Treat every input category string as one indivisible atomic label.
- A comma, semicolon, or other punctuation inside a category is part of that label.
- Do not split labels such as "politics, us".
- Assign every input category to exactly one broad category.
- Preserve each input category exactly, including spelling, punctuation, and case.
- Do not invent, rename, normalize, merge, or omit input category strings.
- Broad categories should be mutually distinct, general, stable, and useful.
- Avoid broad categories that contain only one source category unless necessary.
- Use article_count only to understand importance; rare categories must still be included.
- Place ambiguous categories in the single most reasonable broad category.
- Use concise English names for broad categories and descriptions.
{repair_text}
Input categories:
{json.dumps(category_records, ensure_ascii=False)}
"""


# ======================================================================================
# Strict local validation
# ======================================================================================

def validate_taxonomy(
    taxonomy: Dict[str, Any],
    source_categories: List[str],
    *,
    min_groups: int,
    max_groups: int,
) -> Tuple[bool, Dict[str, Any]]:
    expected = set(source_categories)

    groups = taxonomy.get("groups")

    if not isinstance(groups, list):
        return False, {
            "invalid_groups": "groups must be a list",
        }

    assigned: List[str] = []
    broad_category_names: List[str] = []
    empty_groups: List[str] = []

    for group in groups:
        if not isinstance(group, dict):
            continue

        broad_category = str(
            group.get("broad_category", "")
        ).strip()

        broad_category_names.append(broad_category)

        members = group.get("source_categories", [])

        if not isinstance(members, list):
            members = []

        cleaned_members = [
            str(member)
            for member in members
        ]

        if not cleaned_members:
            empty_groups.append(broad_category)

        assigned.extend(cleaned_members)

    assignment_counts = Counter(assigned)
    broad_name_counts = Counter(broad_category_names)

    missing = sorted(expected - set(assigned))
    extras = sorted(set(assigned) - expected)

    duplicate_assignments = sorted(
        category
        for category, count in assignment_counts.items()
        if count > 1
    )

    duplicate_broad_categories = sorted(
        name
        for name, count in broad_name_counts.items()
        if name and count > 1
    )

    blank_broad_categories = [
        index
        for index, name in enumerate(broad_category_names)
        if not name
    ]

    invalid_group_count = not (
        min_groups <= len(groups) <= max_groups
    )

    errors = {
        "group_count": len(groups),
        "required_group_range": [
            min_groups,
            max_groups,
        ],
        "invalid_group_count": invalid_group_count,
        "missing_categories": missing,
        "extra_categories": extras,
        "duplicate_assignments": duplicate_assignments,
        "duplicate_broad_categories": duplicate_broad_categories,
        "empty_groups": empty_groups,
        "blank_broad_category_indices": blank_broad_categories,
    }

    valid = not any([
        invalid_group_count,
        missing,
        extras,
        duplicate_assignments,
        duplicate_broad_categories,
        empty_groups,
        blank_broad_categories,
    ])

    return valid, errors


# ======================================================================================
# Gemini taxonomy generation
# ======================================================================================

def generate_broad_category_taxonomy(
    category_df: pd.DataFrame,
    *,
    api_key: Optional[str] = None,
    model: str = GEMINI_MODEL,
    min_broad_categories: int = MIN_BROAD_CATEGORIES,
    max_broad_categories: int = MAX_BROAD_CATEGORIES,
    max_attempts: int = MAX_GENERATION_ATTEMPTS,
) -> Dict[str, Any]:
    if not api_key:
        api_key = get_secret("GOOGLE_API_KEY")
    source_categories = (
        category_df["source_category"]
        .astype(str)
        .tolist()
    )

    category_count = len(source_categories)

    if category_count == 0:
        raise ValueError("source_categories must not be empty")

    # Avoid requesting more groups than source categories.
    min_groups = min(
        max(1, int(min_broad_categories)),
        category_count,
    )

    max_groups = min(
        max(min_groups, int(max_broad_categories)),
        category_count,
    )

    schema = build_taxonomy_schema(source_categories)
    client = genai.Client(api_key=api_key)

    previous_error = ""

    for attempt in range(1, int(max_attempts) + 1):
        print("=" * 100)
        print(
            f"[GENERATE BROAD CATEGORY TAXONOMY] "
            f"attempt={attempt}/{max_attempts}"
        )

        prompt = build_taxonomy_prompt(
            category_df,
            min_groups=min_groups,
            max_groups=max_groups,
            previous_error=previous_error,
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "temperature": float(GEMINI_TEMPERATURE),
                "response_mime_type": "application/json",
                "response_json_schema": schema,
            },
        )

        parsed = getattr(response, "parsed", None)

        if parsed is None:
            response_text = str(
                getattr(response, "text", "")
            ).strip()

            if not response_text:
                previous_error = "Gemini returned an empty response."
                continue

            try:
                taxonomy = json.loads(response_text)
            except json.JSONDecodeError as exc:
                previous_error = (
                    f"Invalid JSON response: {exc}"
                )
                continue

        elif hasattr(parsed, "model_dump"):
            taxonomy = parsed.model_dump()

        else:
            taxonomy = dict(parsed)

        valid, validation = validate_taxonomy(
            taxonomy,
            source_categories,
            min_groups=min_groups,
            max_groups=max_groups,
        )

        if valid:
            print("[TAXONOMY VALIDATION PASSED]")
            print("broad categories:", len(taxonomy["groups"]))
            print("assigned source categories:", category_count)

            return taxonomy

        previous_error = json.dumps(
            validation,
            ensure_ascii=False,
        )

        print("[TAXONOMY VALIDATION FAILED]")
        print(previous_error)

    raise RuntimeError(
        "Gemini did not produce a complete valid taxonomy after "
        f"{max_attempts} attempts. Last validation error: "
        f"{previous_error}"
    )


# ======================================================================================
# Convert taxonomy JSON into a mapping DataFrame
# ======================================================================================

def taxonomy_to_mapping_dataframe(
    taxonomy: Dict[str, Any],
    category_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    taxonomy_name = str(
        taxonomy.get("taxonomy_name", "")
    ).strip()

    for broad_rank, group in enumerate(
        taxonomy.get("groups", []),
        start=1,
    ):
        broad_category = str(
            group.get("broad_category", "")
        ).strip()

        broad_description = str(
            group.get("description", "")
        ).strip()

        for source_category in group.get(
            "source_categories",
            [],
        ):
            rows.append({
                "taxonomy_name": taxonomy_name,
                "broad_category_rank": broad_rank,
                "broad_category": broad_category,
                "broad_description": broad_description,
                "source_category": str(source_category),
            })

    mapping_df = pd.DataFrame(rows)

    mapping_df = mapping_df.merge(
        category_df,
        on="source_category",
        how="left",
        validate="one_to_one",
    )

    mapping_df = mapping_df.sort_values(
        [
            "broad_category_rank",
            "article_count",
            "source_category",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    return mapping_df
