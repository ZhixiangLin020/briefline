from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from rag.config import (
    RAGRunConfig,
    build_parser,
    config_from_args,
    load_source_ids,
    normalize_stages,
)


class RAGConfigTests(unittest.TestCase):
    def test_stage_aliases_are_ordered(self) -> None:
        self.assertEqual(
            normalize_stages("retrieve,fetch,generate"),
            ("fetch", "generation", "retrieval"),
        )

    def test_all_expands_to_full_pipeline(self) -> None:
        self.assertEqual(
            normalize_stages("all"),
            (
                "fetch",
                "generation",
                "retrieval",
                "judge",
                "similarity",
                "faithfulness",
            ),
        )

    def test_source_ids_load_from_run_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps({"inserted_source_ids": ["a", "b", "a", ""]}),
                encoding="utf-8",
            )
            self.assertEqual(load_source_ids(path), ("a", "b"))


    def test_source_ids_fall_back_to_downstream_ids_for_resume_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "inserted_source_ids": [],
                        "downstream_source_ids": ["resume-1", "resume-2"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_source_ids(path), ("resume-1", "resume-2"))

    def test_resume_manifest_prefers_the_latest_downstream_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            path.write_text(
                json.dumps(
                    {
                        "inserted_source_ids": ["new-1"],
                        "generation_completed_source_ids": ["new-1", "pending-1"],
                        "downstream_source_ids": ["pending-similarity-1"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_source_ids(path), ("pending-similarity-1",))

    def test_smoke_defaults_to_500_and_current_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = argparse.Namespace(
                mode="smoke",
                stages="fetch",
                max_new_articles=None,
                recover_pending_generation=True,
                max_pending_articles=None,
                only_current_run=None,
                adapter_path=None,
                base_model_path="Qwen/Qwen2.5-3B-Instruct",
                judge_model_path="Qwen/Qwen3-14B",
                artifact_dir=str(root / "artifacts"),
                temp_root=None,
                retrieval_dir=None,
                run_manifest=None,
                source_ids_file=None,
                guardian_from_date=None,
                guardian_to_date=None,
                collection_name="GuardianSentenceEvidenceOpenAISmallPOC",
                print_each=None,
                use_colbert=None,
                faithfulness_all_eligible=False,
                faithfulness_only_changed_highlight=False,
                faithfulness_run_n=None,
                cleanup_merged_model=False,
                preflight_only=True,
            )
            config = config_from_args(namespace, root)
            self.assertEqual(config.max_new_articles, 500)
            self.assertTrue(config.only_current_run)
            self.assertTrue(config.recover_pending_generation)
            self.assertEqual(config.max_pending_articles, 500)
            self.assertFalse(config.print_each)
            self.assertFalse(config.use_colbert)

    def test_full_mode_also_defaults_to_current_run_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = argparse.Namespace(
                mode="full",
                stages="fetch",
                max_new_articles=None,
                recover_pending_generation=True,
                max_pending_articles=None,
                only_current_run=None,
                adapter_path=None,
                base_model_path="Qwen/Qwen2.5-3B-Instruct",
                judge_model_path="Qwen/Qwen3-14B",
                artifact_dir=str(root / "artifacts"),
                temp_root=None,
                retrieval_dir=None,
                run_manifest=None,
                source_ids_file=None,
                guardian_from_date=None,
                guardian_to_date=None,
                collection_name="GuardianSentenceEvidenceOpenAISmallPOC",
                print_each=None,
                use_colbert=None,
                faithfulness_all_eligible=False,
                faithfulness_only_changed_highlight=False,
                faithfulness_run_n=None,
                cleanup_merged_model=False,
                preflight_only=True,
            )
            config = config_from_args(namespace, root)
            self.assertEqual(config.max_new_articles, 2500)
            self.assertEqual(config.max_pending_articles, 2500)
            self.assertTrue(config.only_current_run)

    def test_pending_recovery_can_be_disabled_explicitly(self) -> None:
        args = build_parser().parse_args(
            ["--stages", "fetch", "--no-recover-pending-generation"]
        )
        self.assertFalse(args.recover_pending_generation)

    def test_global_faithfulness_can_run_without_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = RAGRunConfig(
                mode="full",
                stages=("faithfulness",),
                max_new_articles=2500,
                only_current_run=True,
                adapter_path=None,
                base_model_path="Qwen/Qwen2.5-3B-Instruct",
                judge_model_path="Qwen/Qwen3-14B",
                artifact_dir=root / "artifacts",
                temp_root=root / "runtime",
                retrieval_dir=root / "retrieval",
                run_manifest_path=root / "run.json",
                source_ids=(),
                guardian_from_date=None,
                guardian_to_date=None,
                collection_name="GuardianSentenceEvidenceOpenAISmallPOC",
                print_each=False,
                use_colbert=False,
                faithfulness_all_eligible=True,
                faithfulness_only_changed_highlight=False,
                faithfulness_run_n=None,
                cleanup_merged_model=False,
                preflight_only=True,
            )
            config.validate()

    def test_generation_requires_adapter_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            config = RAGRunConfig(
                mode="smoke",
                stages=("generation",),
                max_new_articles=500,
                only_current_run=False,
                adapter_path=adapter,
                base_model_path="Qwen/Qwen2.5-3B-Instruct",
                judge_model_path="Qwen/Qwen3-14B",
                artifact_dir=root / "artifacts",
                temp_root=root / "runtime",
                retrieval_dir=root / "retrieval",
                run_manifest_path=root / "run.json",
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
                preflight_only=True,
            )
            with self.assertRaisesRegex(ValueError, "adapter model weights"):
                config.validate()



if __name__ == "__main__":
    unittest.main()
