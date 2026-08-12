"""Generate and persist the broad category mapping used by the frontend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--min-broad-categories", type=int, default=8)
    parser.add_argument("--max-broad-categories", type=int, default=14)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rag" / "taxonomy",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from rag.taxonomy_generation import (
        generate_broad_category_taxonomy,
        get_secret,
        load_final_category_inventory,
        taxonomy_to_mapping_dataframe,
    )
    from rag.taxonomy_storage import save_category_broad_mapping_to_postgres

    database_url = get_secret("DATABASE_URL")
    category_df = load_final_category_inventory(database_url=database_url)
    taxonomy = generate_broad_category_taxonomy(
        category_df,
        api_key=get_secret("GOOGLE_API_KEY"),
        model=args.model,
        min_broad_categories=args.min_broad_categories,
        max_broad_categories=args.max_broad_categories,
        max_attempts=args.max_attempts,
    )
    mapping_df = taxonomy_to_mapping_dataframe(taxonomy, category_df)
    stats = save_category_broad_mapping_to_postgres(
        mapping_df,
        database_url=database_url,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "taxonomy.json").write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mapping_df.to_csv(output_dir / "category_broad_mapping.csv", index=False)
    (output_dir / "storage_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

