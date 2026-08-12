"""Configuration and validation for the integrated Guardian RAG runner."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


STAGE_ORDER: Tuple[str, ...] = (
    "fetch",
    "generation",
    "retrieval",
    "judge",
    "similarity",
    "faithfulness",
)
DEFAULT_STAGES: Tuple[str, ...] = ("fetch", "generation", "retrieval")
STAGE_ALIASES = {
    "generate": "generation",
    "retrieve": "retrieval",
    "recommend": "similarity",
    "recommendation": "similarity",
}


def normalize_stages(value: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(value, str):
        raw_stages = [item.strip().lower() for item in value.split(",")]
    else:
        raw_stages = [str(item).strip().lower() for item in value]

    raw_stages = [item for item in raw_stages if item]
    if not raw_stages:
        raise ValueError("At least one RAG stage is required.")
    if "all" in raw_stages:
        return STAGE_ORDER

    normalized = {STAGE_ALIASES.get(item, item) for item in raw_stages}
    unknown = sorted(normalized - set(STAGE_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown RAG stages: {unknown}. Supported stages: {list(STAGE_ORDER)}"
        )
    return tuple(stage for stage in STAGE_ORDER if stage in normalized)


def load_source_ids(path: Optional[Path]) -> Tuple[str, ...]:
    if path is None:
        return ()
    if not path.is_file():
        raise FileNotFoundError(f"Source-ID file does not exist: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = None
        for key in (
            "downstream_source_ids",
            "generation_completed_source_ids",
            "inserted_source_ids",
        ):
            candidate_values = payload.get(key)
            if candidate_values:
                values = candidate_values
                break
            if values is None and key in payload:
                values = candidate_values
    else:
        values = payload

    if not isinstance(values, list):
        raise ValueError(
            "Source-ID input must be a JSON list or a run manifest containing "
            "downstream_source_ids, generation_completed_source_ids, or "
            "inserted_source_ids."
        )

    seen = set()
    source_ids = []
    for value in values:
        source_id = str(value or "").strip()
        if source_id and source_id not in seen:
            seen.add(source_id)
            source_ids.append(source_id)
    return tuple(source_ids)


@dataclass(frozen=True)
class RAGRunConfig:
    mode: str
    stages: Tuple[str, ...]
    max_new_articles: int
    only_current_run: bool
    adapter_path: Optional[Path]
    base_model_path: str
    judge_model_path: str
    artifact_dir: Path
    temp_root: Path
    retrieval_dir: Path
    run_manifest_path: Path
    source_ids: Tuple[str, ...]
    guardian_from_date: Optional[str]
    guardian_to_date: Optional[str]
    collection_name: str
    print_each: bool
    use_colbert: bool
    faithfulness_all_eligible: bool
    faithfulness_only_changed_highlight: bool
    faithfulness_run_n: Optional[int]
    cleanup_merged_model: bool
    preflight_only: bool
    recover_pending_generation: bool = True
    max_pending_articles: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "adapter_path",
            "artifact_dir",
            "temp_root",
            "retrieval_dir",
            "run_manifest_path",
        ):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        payload["stages"] = list(self.stages)
        payload["source_ids"] = list(self.source_ids)
        return payload

    def validate(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'.")
        if self.max_new_articles <= 0:
            raise ValueError("max_new_articles must be greater than zero.")
        if self.max_pending_articles is not None and self.max_pending_articles <= 0:
            raise ValueError("max_pending_articles must be greater than zero.")
        normalize_stages(self.stages)

        if "generation" in self.stages:
            if self.adapter_path is None:
                raise ValueError(
                    "An adapter is required for generation. Pass --adapter-path or "
                    "set ADAPTER_PATH."
                )
            if not self.adapter_path.is_dir():
                raise FileNotFoundError(
                    f"Adapter directory does not exist: {self.adapter_path}"
                )
            if not (self.adapter_path / "adapter_config.json").is_file():
                raise ValueError(
                    f"Adapter directory is missing adapter_config.json: {self.adapter_path}"
                )
            adapter_weight_files = [
                *self.adapter_path.glob("adapter_model*.safetensors"),
                *self.adapter_path.glob("adapter_model*.bin"),
            ]
            if not any(path.is_file() for path in adapter_weight_files):
                raise ValueError(
                    "Adapter directory is missing adapter model weights "
                    f"(adapter_model*.safetensors or adapter_model*.bin): {self.adapter_path}"
                )

        downstream = set(self.stages) - {"fetch"}
        scoped_downstream = set(downstream)
        if self.faithfulness_all_eligible:
            scoped_downstream.discard("faithfulness")
        if (
            self.only_current_run
            and scoped_downstream
            and "fetch" not in self.stages
            and not self.source_ids
        ):
            raise ValueError(
                "--only-current-run without the fetch stage requires "
                "--source-ids-file from a previous run manifest."
            )

        if (
            "faithfulness" in self.stages
            and not self.faithfulness_all_eligible
            and "fetch" not in self.stages
            and not self.source_ids
        ):
            raise ValueError(
                "The faithfulness stage defaults to an incremental source-ID scope. "
                "Include the fetch stage, pass --source-ids-file, or explicitly use "
                "--faithfulness-all-eligible."
            )

        if self.faithfulness_run_n is not None and self.faithfulness_run_n <= 0:
            raise ValueError("faithfulness_run_n must be greater than zero.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the incremental Guardian RAG backend without modifying the "
            "data-selection or training pipelines."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--stages",
        default=",".join(DEFAULT_STAGES),
        help=(
            "Comma-separated stages in pipeline order: fetch, generation, "
            "retrieval, judge, similarity, faithfulness. Use 'all' for every stage."
        ),
    )
    parser.add_argument("--max-new-articles", type=int, default=None)
    parser.add_argument(
        "--recover-pending-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "In fetch + current-run mode, also resume bounded historical rows "
            "at their first unfinished generation, retrieval/Judge, or similarity "
            "stage."
        ),
    )
    parser.add_argument(
        "--max-pending-articles",
        type=int,
        default=None,
        help=(
            "Shared maximum number of historical pending rows added across all "
            "recoverable stages; defaults to --max-new-articles."
        ),
    )
    parser.add_argument(
        "--only-current-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Restrict downstream stages to this run's new/recovered or supplied "
            "source-ID work set."
        ),
    )
    parser.add_argument("--adapter-path", default=os.environ.get("ADAPTER_PATH"))
    parser.add_argument(
        "--base-model-path",
        default=os.environ.get("BASE_MODEL_PATH", "Qwen/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--judge-model-path",
        default=os.environ.get("JUDGE_MODEL_PATH", "Qwen/Qwen3-14B"),
    )
    parser.add_argument("--artifact-dir", default=os.environ.get("RAG_ARTIFACT_DIR"))
    parser.add_argument("--temp-root", default=os.environ.get("RAG_TEMP_ROOT"))
    parser.add_argument("--retrieval-dir", default=None)
    parser.add_argument("--run-manifest", default=None)
    parser.add_argument("--source-ids-file", type=Path, default=None)
    parser.add_argument("--guardian-from-date", default=None)
    parser.add_argument("--guardian-to-date", default=None)
    parser.add_argument(
        "--collection-name",
        default=os.environ.get(
            "WEAVIATE_COLLECTION",
            "GuardianSentenceEvidenceOpenAISmallPOC",
        ),
    )
    parser.add_argument(
        "--print-each",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-colbert",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--faithfulness-all-eligible",
        action="store_true",
        help=(
            "Run faithfulness against every eligible historical database row. "
            "Without this explicit flag, faithfulness is restricted to the source "
            "IDs completed or supplied for the current pipeline run."
        ),
    )
    parser.add_argument(
        "--faithfulness-only-changed-highlight",
        action="store_true",
        help="Within the selected source-ID scope, score only revised highlights.",
    )
    parser.add_argument(
        "--faithfulness-run-n",
        type=int,
        default=None,
        help=(
            "Optional additional cap for faithfulness rows after applying the "
            "incremental or global scope."
        ),
    )
    parser.add_argument("--cleanup-merged-model", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths and required settings without external requests or models.",
    )
    return parser


def config_from_args(args: argparse.Namespace, project_root: Path) -> RAGRunConfig:
    mode = str(args.mode)
    stages = normalize_stages(args.stages)
    max_new_articles = args.max_new_articles
    if max_new_articles is None:
        max_new_articles = 500 if mode == "smoke" else 2500

    max_pending_articles = getattr(args, "max_pending_articles", None)
    if max_pending_articles is None:
        max_pending_articles = max_new_articles
    recover_pending_generation = getattr(args, "recover_pending_generation", True)

    only_current_run = args.only_current_run
    if only_current_run is None:
        only_current_run = True

    artifact_dir = Path(
        args.artifact_dir or project_root / "artifacts" / "rag"
    ).expanduser().resolve()
    temp_root = Path(
        args.temp_root or artifact_dir / "runtime"
    ).expanduser().resolve()
    retrieval_dir = Path(
        args.retrieval_dir or artifact_dir / "retrieval"
    ).expanduser().resolve()
    run_manifest_path = Path(
        args.run_manifest or artifact_dir / "last_run.json"
    ).expanduser().resolve()
    adapter_path = (
        Path(args.adapter_path).expanduser().resolve()
        if args.adapter_path
        else None
    )

    config = RAGRunConfig(
        mode=mode,
        stages=stages,
        max_new_articles=int(max_new_articles),
        only_current_run=bool(only_current_run),
        adapter_path=adapter_path,
        base_model_path=str(args.base_model_path),
        judge_model_path=str(args.judge_model_path),
        artifact_dir=artifact_dir,
        temp_root=temp_root,
        retrieval_dir=retrieval_dir,
        run_manifest_path=run_manifest_path,
        source_ids=load_source_ids(args.source_ids_file),
        guardian_from_date=args.guardian_from_date,
        guardian_to_date=args.guardian_to_date,
        collection_name=str(args.collection_name),
        print_each=(mode == "full") if args.print_each is None else bool(args.print_each),
        use_colbert=(mode == "full") if args.use_colbert is None else bool(args.use_colbert),
        faithfulness_all_eligible=bool(args.faithfulness_all_eligible),
        faithfulness_only_changed_highlight=bool(
            args.faithfulness_only_changed_highlight
        ),
        faithfulness_run_n=args.faithfulness_run_n,
        cleanup_merged_model=bool(args.cleanup_merged_model),
        preflight_only=bool(args.preflight_only),
        recover_pending_generation=bool(recover_pending_generation),
        max_pending_articles=int(max_pending_articles),
    )
    config.validate()
    return config
