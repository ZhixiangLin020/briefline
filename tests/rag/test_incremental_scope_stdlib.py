from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.config import RAGRunConfig
from rag.orchestrator import (
    _generation_batch_limit,
    _load_recoverable_downstream_source_ids,
    _load_recoverable_pending_source_ids,
    _run_similarity,
    run_pipeline,
)


def make_config(root: Path) -> RAGRunConfig:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    artifact_dir = root / "artifacts"
    return RAGRunConfig(
        mode="smoke",
        stages=("fetch", "generation", "retrieval"),
        max_new_articles=500,
        only_current_run=True,
        adapter_path=adapter,
        base_model_path="Qwen/Qwen2.5-3B-Instruct",
        judge_model_path="Qwen/Qwen3-14B",
        artifact_dir=artifact_dir,
        temp_root=artifact_dir / "runtime",
        retrieval_dir=artifact_dir / "retrieval",
        run_manifest_path=artifact_dir / "run.json",
        source_ids=(),
        guardian_from_date=None,
        guardian_to_date=None,
        collection_name="GuardianSentenceEvidenceOpenAISmallPOC",
        print_each=False,
        use_colbert=False,
        faithfulness_all_eligible=False,
        faithfulness_only_changed_highlight=False,
        faithfulness_run_n=None,
        cleanup_merged_model=False,
        preflight_only=False,
        recover_pending_generation=False,
        max_pending_articles=500,
    )


class IncrementalScopeTests(unittest.TestCase):
    def test_similarity_receives_configured_collection_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "collection_name", "CustomGuardianCollection")
            captured = {}
            fake_similarity = types.ModuleType("rag.similarity_pipeline")

            def run_similarity_pipeline(**kwargs):
                captured.update(kwargs)
                return {
                    "query_table": ["query"],
                    "recommendation_records": ["recommendation"],
                    "sync_stats": {},
                }

            fake_similarity.run_guardian_similar_articles_pipeline = (
                run_similarity_pipeline
            )

            with patch.dict(
                sys.modules,
                {"rag.similarity_pipeline": fake_similarity},
            ):
                result = _run_similarity(config, ["new-1"])

            self.assertEqual(captured["collection_name"], config.collection_name)
            self.assertEqual(captured["source_ids"], ["new-1"])
            self.assertEqual(result["query_articles"], 1)

    def test_inserted_ids_are_forwarded_to_every_downstream_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [{"id": "new-1"}],
                        "inserted_source_ids": ["new-1"],
                        "storage_stats": {"inserted": 1},
                    },
                ),
                patch(
                    "rag.orchestrator._run_generation",
                    return_value={"generated_rows": 2, "merged_model_path": ""},
                ) as generation,
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 1,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                    },
                ) as retrieval,
            ):
                result = run_pipeline(config)

            self.assertEqual(result["status"], "completed")
            generation.assert_called_once_with(config, ["new-1"])
            retrieval.assert_called_once_with(config, ["new-1"])

    def test_new_and_pending_ids_are_combined_then_completed_scope_is_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "recover_pending_generation", True)
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [{"id": "new-1"}],
                        "inserted_source_ids": ["new-1"],
                        "storage_stats": {"inserted": 1},
                    },
                ),
                patch(
                    "rag.orchestrator._load_recoverable_pending_source_ids",
                    return_value=["pending-1", "new-1"],
                ),
                patch(
                    "rag.orchestrator._run_generation",
                    return_value={
                        "generated_rows": 4,
                        "completed_source_ids": ["new-1", "pending-1"],
                        "merged_model_path": "",
                    },
                ) as generation,
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 2,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                    },
                ) as retrieval,
            ):
                result = run_pipeline(config)

            expected = ["new-1", "pending-1"]
            generation.assert_called_once_with(config, expected)
            retrieval.assert_called_once_with(config, expected)
            self.assertEqual(result["inserted_source_ids"], ["new-1"])
            self.assertEqual(
                result["recovered_pending_source_ids"],
                ["pending-1"],
            )
            self.assertEqual(result["downstream_source_ids"], expected)

    def test_pending_rows_run_even_when_fetch_inserts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "recover_pending_generation", True)
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [],
                        "inserted_source_ids": [],
                        "storage_stats": {"inserted": 0},
                    },
                ),
                patch(
                    "rag.orchestrator._load_recoverable_pending_source_ids",
                    return_value=["pending-1"],
                ),
                patch(
                    "rag.orchestrator._run_generation",
                    return_value={
                        "generated_rows": 2,
                        "completed_source_ids": ["pending-1"],
                        "merged_model_path": "",
                    },
                ) as generation,
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 1,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                    },
                ) as retrieval,
            ):
                result = run_pipeline(config)

            self.assertEqual(result["status"], "completed")
            generation.assert_called_once_with(config, ["pending-1"])
            retrieval.assert_called_once_with(config, ["pending-1"])

    def test_incomplete_generation_ids_are_not_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [{"id": "new-1"}, {"id": "new-2"}],
                        "inserted_source_ids": ["new-1", "new-2"],
                        "storage_stats": {"inserted": 2},
                    },
                ),
                patch(
                    "rag.orchestrator._run_generation",
                    return_value={
                        "generated_rows": 2,
                        "completed_source_ids": ["new-1"],
                        "merged_model_path": "",
                    },
                ),
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 1,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                    },
                ) as retrieval,
            ):
                result = run_pipeline(config)

            retrieval.assert_called_once_with(config, ["new-1"])
            self.assertEqual(result["generation_completed_source_ids"], ["new-1"])

    def test_generation_limit_covers_the_combined_explicit_work_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            self.assertEqual(_generation_batch_limit(config, None), 500)
            self.assertEqual(
                _generation_batch_limit(config, [f"source-{i}" for i in range(17)]),
                17,
            )

    def test_pending_loader_uses_its_own_bound_and_excludes_new_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "recover_pending_generation", True)
            object.__setattr__(config, "max_pending_articles", 10)
            captured = {}
            fake_guardian = types.ModuleType("rag.guardian_pipeline")

            def load_pending_generation_source_ids(**kwargs):
                captured.update(kwargs)
                return ["pending-1", "pending-1", "pending-2"]

            fake_guardian.load_pending_generation_source_ids = (
                load_pending_generation_source_ids
            )
            with patch.dict(
                sys.modules,
                {"rag.guardian_pipeline": fake_guardian},
            ):
                result = _load_recoverable_pending_source_ids(config, ["new-1"])

            self.assertEqual(result, ["pending-1", "pending-2"])
            self.assertEqual(captured["limit"], 10)
            self.assertEqual(captured["exclude_source_ids"], ["new-1"])

    def test_downstream_recovery_shares_one_pending_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "recover_pending_generation", True)
            object.__setattr__(
                config,
                "stages",
                ("fetch", "generation", "retrieval", "judge", "similarity"),
            )
            captured = {}

            def load_retrieval(**kwargs):
                captured["retrieval"] = kwargs
                return [f"retrieval-{index}" for index in range(7)]

            def load_similarity(**kwargs):
                captured["similarity"] = kwargs
                return ["similarity-1", "similarity-2", "similarity-3"]

            with (
                patch("rag.orchestrator._secret", return_value="postgresql://db"),
                patch(
                    "rag.stage_recovery.load_pending_retrieval_source_ids",
                    side_effect=load_retrieval,
                ),
                patch(
                    "rag.stage_recovery.load_pending_similarity_source_ids",
                    side_effect=load_similarity,
                ),
            ):
                retrieval_ids, similarity_ids = (
                    _load_recoverable_downstream_source_ids(
                        config,
                        exclude_source_ids=["new-1"],
                        remaining_limit=10,
                    )
                )

            self.assertEqual(len(retrieval_ids), 7)
            self.assertEqual(len(similarity_ids), 3)
            self.assertEqual(captured["retrieval"]["limit"], 10)
            self.assertEqual(captured["similarity"]["limit"], 3)
            self.assertEqual(
                captured["similarity"]["exclude_source_ids"],
                ["new-1", *retrieval_ids],
            )

    def test_unfinished_downstream_rows_resume_at_their_own_stage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(config, "recover_pending_generation", True)
            object.__setattr__(
                config,
                "stages",
                ("fetch", "generation", "retrieval", "judge", "similarity"),
            )
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [],
                        "inserted_source_ids": [],
                        "storage_stats": {"inserted": 0},
                    },
                ),
                patch(
                    "rag.orchestrator._load_recoverable_pending_source_ids",
                    return_value=[],
                ),
                patch(
                    "rag.orchestrator._load_recoverable_downstream_source_ids",
                    return_value=(["retrieval-1"], ["similarity-1"]),
                ),
                patch("rag.orchestrator._run_generation") as generation,
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 1,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                        "completed_source_ids": ["retrieval-1"],
                    },
                ) as retrieval,
                patch(
                    "rag.orchestrator._run_judge",
                    return_value={
                        "judged_rows": 1,
                        "completed_source_ids": ["retrieval-1"],
                    },
                ) as judge,
                patch(
                    "rag.orchestrator._run_similarity",
                    return_value={
                        "query_articles": 2,
                        "recommendation_rows": 2,
                        "completed_source_ids": ["retrieval-1", "similarity-1"],
                    },
                ) as similarity,
            ):
                result = run_pipeline(config)

            generation.assert_not_called()
            retrieval.assert_called_once_with(config, ["retrieval-1"])
            judge.assert_called_once_with(
                config,
                ["retrieval-1"],
                "packets.jsonl",
            )
            similarity.assert_called_once_with(
                config,
                ["retrieval-1", "similarity-1"],
            )
            self.assertEqual(
                result["recovered_pending_retrieval_source_ids"],
                ["retrieval-1"],
            )
            self.assertEqual(
                result["recovered_pending_similarity_source_ids"],
                ["similarity-1"],
            )

    def test_all_stages_use_current_inserted_ids_for_faithfulness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(
                config,
                "stages",
                ("fetch", "generation", "retrieval", "judge", "similarity", "faithfulness"),
            )
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [{"id": "new-1"}],
                        "inserted_source_ids": ["new-1"],
                        "storage_stats": {"inserted": 1},
                    },
                ),
                patch(
                    "rag.orchestrator._run_generation",
                    return_value={"generated_rows": 1, "merged_model_path": ""},
                ),
                patch(
                    "rag.orchestrator._run_retrieval",
                    return_value={
                        "summary_rows": 1,
                        "failed_rows": 0,
                        "packets_jsonl": "packets.jsonl",
                    },
                ),
                patch("rag.orchestrator._run_judge", return_value={"judged_rows": 1}),
                patch(
                    "rag.orchestrator._run_similarity",
                    return_value={"query_articles": 1, "recommendation_rows": 1},
                ),
                patch(
                    "rag.orchestrator._run_faithfulness",
                    return_value={"scope": "current_run", "selected_rows": 1},
                ) as faithfulness,
            ):
                result = run_pipeline(config)

            self.assertEqual(result["status"], "completed")
            faithfulness.assert_called_once_with(config, ["new-1"])
            self.assertEqual(
                result["stage_results"]["faithfulness"]["scope"],
                "current_run",
            )

    def test_explicit_global_faithfulness_runs_when_fetch_has_no_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            object.__setattr__(
                config,
                "stages",
                ("fetch", "generation", "retrieval", "judge", "similarity", "faithfulness"),
            )
            object.__setattr__(config, "faithfulness_all_eligible", True)
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [],
                        "inserted_source_ids": [],
                        "storage_stats": {"inserted": 0},
                    },
                ),
                patch("rag.orchestrator._run_generation") as generation,
                patch("rag.orchestrator._run_retrieval") as retrieval,
                patch("rag.orchestrator._run_judge") as judge,
                patch("rag.orchestrator._run_similarity") as similarity,
                patch(
                    "rag.orchestrator._run_faithfulness",
                    return_value={"scope": "all_eligible", "selected_rows": 100},
                ) as faithfulness,
            ):
                result = run_pipeline(config)

            self.assertEqual(result["status"], "completed")
            generation.assert_not_called()
            retrieval.assert_not_called()
            judge.assert_not_called()
            similarity.assert_not_called()
            faithfulness.assert_called_once_with(config, None)

    def test_zero_insertions_skip_all_downstream_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            with (
                patch("rag.orchestrator.preflight", return_value={"status": "ok"}),
                patch(
                    "rag.orchestrator._run_fetch",
                    return_value={
                        "articles": [],
                        "inserted_source_ids": [],
                        "storage_stats": {"inserted": 0},
                    },
                ),
                patch("rag.orchestrator._run_generation") as generation,
                patch("rag.orchestrator._run_retrieval") as retrieval,
            ):
                result = run_pipeline(config)

            self.assertEqual(result["status"], "no_new_records")
            generation.assert_not_called()
            retrieval.assert_not_called()


if __name__ == "__main__":
    unittest.main()
