from __future__ import annotations

import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.orchestrator import (
    COLBERT_REQUIRED_VERSIONS,
    FAITHFULNESS_REQUIRED_VERSIONS,
    _verify_colbert_environment,
    _verify_faithfulness_environment,
    preflight,
)


class ColbertPreflightTests(unittest.TestCase):
    def test_exact_text_only_versions_are_import_checked(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="RAGatouille import OK\n",
            stderr="",
        )

        with (
            patch(
                "rag.orchestrator.importlib.metadata.version",
                side_effect=lambda name: COLBERT_REQUIRED_VERSIONS[name],
            ),
            patch(
                "rag.orchestrator.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            versions = _verify_colbert_environment()

        self.assertEqual(versions, COLBERT_REQUIRED_VERSIONS)
        command = run.call_args.args[0]
        self.assertIn("from ragatouille import RAGPretrainedModel", command[-1])
        self.assertEqual(run.call_args.kwargs["env"]["USE_TF"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["USE_TORCH"], "1")

    def test_version_mismatch_fails_before_import(self) -> None:
        with (
            patch(
                "rag.orchestrator.importlib.metadata.version",
                side_effect=lambda name: (
                    "5.1.0"
                    if name == "sentence-transformers"
                    else COLBERT_REQUIRED_VERSIONS[name]
                ),
            ),
            patch("rag.orchestrator.subprocess.run") as run,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "sentence-transformers==3.4.1",
            ):
                _verify_colbert_environment()

        run.assert_not_called()

    def test_similarity_preflight_runs_colbert_check_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = types.SimpleNamespace(
                stages=("similarity",),
                use_colbert=True,
                artifact_dir=root / "artifacts",
                temp_root=root / "runtime",
                retrieval_dir=root / "retrieval",
                adapter_path=None,
                validate=lambda: None,
            )
            with (
                patch("rag.orchestrator._required_secrets", return_value=set()),
                patch(
                    "rag.orchestrator._verify_colbert_environment",
                    return_value=COLBERT_REQUIRED_VERSIONS,
                ) as verify,
            ):
                report = preflight(config)

        verify.assert_called_once_with()
        self.assertEqual(report["colbert_versions"], COLBERT_REQUIRED_VERSIONS)

    def test_exact_faithfulness_versions_and_imports_are_checked(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="Faithfulness imports OK\n",
            stderr="",
        )
        with (
            patch(
                "rag.orchestrator.importlib.metadata.version",
                side_effect=lambda name: FAITHFULNESS_REQUIRED_VERSIONS[name],
            ),
            patch(
                "rag.orchestrator.subprocess.run",
                return_value=completed,
            ) as run,
        ):
            versions = _verify_faithfulness_environment()

        self.assertEqual(versions, FAITHFULNESS_REQUIRED_VERSIONS)
        command = run.call_args.args[0]
        self.assertIn("from langchain_openai import OpenAIEmbeddings", command[-1])
        self.assertIn("from ragas.llms import InstructorLLM", command[-1])
        self.assertIn("from ragas.metrics.collections import Faithfulness", command[-1])
        self.assertEqual(run.call_args.kwargs["env"]["USE_TF"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["USE_TORCH"], "1")

    def test_faithfulness_preflight_runs_before_pipeline_stages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = types.SimpleNamespace(
                stages=("faithfulness",),
                use_colbert=False,
                artifact_dir=root / "artifacts",
                temp_root=root / "runtime",
                retrieval_dir=root / "retrieval",
                adapter_path=None,
                validate=lambda: None,
            )
            with (
                patch("rag.orchestrator._required_secrets", return_value=set()),
                patch(
                    "rag.orchestrator._verify_faithfulness_environment",
                    return_value=FAITHFULNESS_REQUIRED_VERSIONS,
                ) as verify,
            ):
                report = preflight(config)

        verify.assert_called_once_with()
        self.assertEqual(
            report["faithfulness_versions"],
            FAITHFULNESS_REQUIRED_VERSIONS,
        )


if __name__ == "__main__":
    unittest.main()
