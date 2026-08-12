import ast
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from briefline import runtime as environment_check
from scripts import install_dependencies


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class UnifiedDependencyTests(unittest.TestCase):
    def test_expected_requirements_files_are_shipped(self):
        names = sorted(path.name for path in PROJECT_ROOT.glob("requirements*.txt"))
        self.assertEqual(
            names,
            [
                "requirements-dev.txt",
                "requirements-frontend.txt",
                "requirements-rag-colbert.txt",
                "requirements-rag-evaluation.txt",
                "requirements-rag.txt",
                "requirements.txt",
            ],
        )

    def test_frontend_requirements_are_cpu_only_and_exactly_pinned(self):
        lines = {
            line.strip()
            for line in (PROJECT_ROOT / "requirements-frontend.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(
            lines,
            {
                "streamlit==1.61.1",
                "psycopg==3.3.4",
                "psycopg-binary==3.3.4",
                "psycopg-pool==3.3.1",
            },
        )

    def test_streamlit_cloud_requirements_match_frontend_requirements(self):
        def normalized_lines(path):
            return {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }

        self.assertEqual(
            normalized_lines(PROJECT_ROOT / "frontend/requirements.txt"),
            normalized_lines(PROJECT_ROOT / "requirements-frontend.txt"),
        )

    def test_shared_gpu_stack_is_pinned(self):
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for name, version in environment_check.REQUIRED_VERSIONS.items():
            self.assertIn(f"{name}=={version}", text)

    def test_vllm_transitive_dependencies_are_compatibly_pinned(self):
        lines = set(
            (PROJECT_ROOT / "requirements.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        self.assertTrue({
            "ray==2.49.2",
            "opencv-python-headless==4.11.0.86",
            "numpy==1.26.4",
            "packaging==23.2",
            "setuptools==79.0.1",
            "wheel==0.45.1",
        }.issubset(lines))

    def test_rag_overlays_preserve_the_stack_and_pin_compatible_apis(self):
        expected_lines = {
            "requirements-rag.txt": {
                "-c requirements.txt",
                "openai>=1.40,<2",
                "weaviate-client==4.22.0",
            },
            "requirements-rag-evaluation.txt": {
                "instructor==1.12.0",
                "ragas==0.4.3",
                "langchain==0.1.20",
                "langchain-community==0.0.38",
                "langchain-core==0.1.52",
                "langchain-openai==0.1.7",
                "langchain-text-splitters==0.0.2",
            },
            "requirements-rag-colbert.txt": {
                "-c requirements.txt",
                "ragatouille==0.0.9.post2",
                "llama-index==0.12.42",
                "llama-index-core==0.12.42",
                "sentence-transformers==3.4.1",
                "langchain==0.1.20",
                "langchain-community==0.0.38",
                "langchain-core==0.1.52",
                "langchain-text-splitters==0.0.2",
            },
        }
        for filename, expected in expected_lines.items():
            with self.subTest(filename=filename):
                lines = set(
                    (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
                )
                self.assertTrue(expected.issubset(lines))

    def test_guardian_does_not_require_vllm_system_utils_at_import_time(self):
        source = (PROJECT_ROOT / "rag/guardian_pipeline.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("vllm.utils.system_utils", top_level_imports)
        self.assertIn(
            'importlib.import_module("vllm.utils.system_utils")',
            source,
        )

    def test_optional_colbert_repair_cannot_request_an_unpinned_ragatouille(self):
        source = (PROJECT_ROOT / "rag/similarity_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('pip_install_quiet(["ragatouille==0.0.9.post2"])', source)
        self.assertNotIn('pip_install_quiet(["ragatouille"])', source)

    def test_exact_binary_install_sequence(self):
        phases, pip_check = install_dependencies.build_commands(
            PROJECT_ROOT / "requirements.txt"
        )
        commands = [command for _, command in phases]

        self.assertEqual(
            commands[0][-5:],
            [
                "torch==2.8.0",
                "torchvision==0.23.0",
                "torchaudio==2.8.0",
                "--index-url",
                install_dependencies.PYTORCH_INDEX_URL,
            ],
        )
        self.assertEqual(commands[1][-1], "vllm==0.10.2")
        self.assertIn("--constraint", commands[1])
        self.assertIn(str((PROJECT_ROOT / "requirements.txt").resolve()), commands[1])
        self.assertEqual(
            commands[2][-2:],
            ["--no-deps", install_dependencies.FLASH_ATTN_WHEEL_URL],
        )
        self.assertEqual(
            commands[3][-2:],
            ["--force-reinstall", "transformers==4.56.2"],
        )
        self.assertIn("--no-deps", commands[3])
        self.assertEqual(
            commands[4][-2:],
            ["--force-reinstall", "huggingface-hub==0.34.4"],
        )
        self.assertIn("--no-deps", commands[4])
        self.assertEqual(pip_check[-1], "check")

    def test_flashattention_command_can_never_fall_back_to_source(self):
        phases, _ = install_dependencies.build_commands(
            PROJECT_ROOT / "requirements.txt"
        )
        flash_command = phases[2][1]
        self.assertTrue(install_dependencies.FLASH_ATTN_WHEEL_URL.endswith(".whl"))
        self.assertIn("cp312-cp312-linux_x86_64.whl", flash_command[-1])
        self.assertNotIn("flash-attn==2.8.3.post1", flash_command)
        self.assertNotIn("--no-build-isolation", flash_command)

    def test_remaining_dependencies_are_constrained_by_the_stack(self):
        phases, _ = install_dependencies.build_commands(
            PROJECT_ROOT / "requirements.txt"
        )
        extras_command = phases[-1][1]
        self.assertIn("--constraint", extras_command)
        self.assertIn("datasets==5.0.0", extras_command)
        for distribution in environment_check.REQUIRED_VERSIONS:
            self.assertFalse(
                any(
                    item.startswith(distribution + "==")
                    for item in extras_command
                )
            )

    def test_complete_rag_is_resolved_in_one_pip_command(self):
        phases, _ = install_dependencies.build_commands(
            PROJECT_ROOT / "requirements.txt",
            include_rag=True,
        )
        install_commands = [command for _, command in phases]
        commands_with_overlays = [
            command
            for command in install_commands
            if any(
                str(path) in command
                for path in install_dependencies.COMPLETE_RAG_REQUIREMENTS
            )
        ]
        self.assertEqual(len(commands_with_overlays), 1)
        combined_command = commands_with_overlays[0]
        self.assertEqual(combined_command.count("--requirement"), 2)
        for overlay_path in install_dependencies.COMPLETE_RAG_REQUIREMENTS:
            self.assertIn(str(overlay_path), combined_command)

    def test_complete_rag_project_graph_includes_nested_rag_requirements(self):
        names = install_dependencies._overlay_distribution_names(
            install_dependencies.COMPLETE_RAG_REQUIREMENTS
        )
        self.assertIn("openai", names)
        self.assertIn("weaviate-client", names)
        self.assertIn("ragas", names)
        self.assertIn("ragatouille", names)
        self.assertIn("langchain-openai", names)

    def test_complete_rag_import_probe_checks_both_optional_stacks(self):
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Complete RAG imports OK\n", stderr=""
        )
        with mock.patch(
            "scripts.install_dependencies.subprocess.run",
            return_value=completed,
        ) as run:
            install_dependencies._verify_complete_rag_imports()

        command = run.call_args.args[0]
        self.assertIn("from ragatouille import RAGPretrainedModel", command[-1])
        self.assertIn("from ragas.llms import InstructorLLM", command[-1])
        self.assertIn(
            "from ragas.metrics.collections import Faithfulness",
            command[-1],
        )
        self.assertEqual(run.call_args.kwargs["env"]["USE_TF"], "0")
        self.assertEqual(run.call_args.kwargs["env"]["USE_TORCH"], "1")

    def test_requirements_must_contain_every_exact_stack_pin(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "requirements.txt"
            path.write_text("torch==2.8.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must pin"):
                install_dependencies.load_project_requirements(path)

            pins = "\n".join(
                f"{name}=={version}"
                for name, version in environment_check.REQUIRED_VERSIONS.items()
            )
            path.write_text(pins + "\nflash-attn==2.8.3.post1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                install_dependencies.load_project_requirements(path)

    def test_version_issues_report_missing_and_wrong_versions(self):
        actual = {"torch": "2.7.0", "vllm": None}
        with mock.patch.object(
            environment_check,
            "installed_version",
            side_effect=lambda name: actual[name],
        ):
            issues = environment_check.version_issues(
                {"torch": "2.8.0", "vllm": "0.10.2"}
            )
        self.assertEqual(issues["torch"]["actual"], "2.7.0")
        self.assertIsNone(issues["vllm"]["actual"])

    def test_cuda_local_version_suffix_is_accepted(self):
        with mock.patch.object(
            environment_check,
            "installed_version",
            return_value="2.8.0+cu128",
        ):
            issues = environment_check.version_issues({"torch": "2.8.0"})
        self.assertEqual(issues, {})

    def test_binary_probe_checks_kernel_abi_and_vllm_v1_api(self):
        source = environment_check._probe_source(require_cuda=True)
        self.assertIn("run_flashattention_kernel()", source)
        self.assertIn("_GLIBCXX_USE_CXX11_ABI", source)
        self.assertIn("AdapterLogitsProcessor", source)
        self.assertIn("RequestLogitsProcessor", source)

    def test_binary_probe_returns_the_subprocess_report(self):
        payload = {"torch": "2.8.0+cu128", "cuda_available": True}
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload) + "\n", stderr=""
        )
        with mock.patch("briefline.runtime.subprocess.run", return_value=completed):
            self.assertEqual(environment_check.binary_probe(), payload)

    def test_binary_probe_surfaces_abi_failures(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="flash_attn_2_cuda.so: undefined symbol",
        )
        with mock.patch("briefline.runtime.subprocess.run", return_value=completed):
            with self.assertRaisesRegex(RuntimeError, "undefined symbol"):
                environment_check.binary_probe()

    def test_dry_run_never_invokes_pip_or_platform_validation(self):
        with (
            mock.patch.object(install_dependencies, "_run") as run_mock,
            mock.patch.object(
                install_dependencies, "_validate_wheel_runtime"
            ) as runtime_mock,
        ):
            install_dependencies.install(
                requirements_path=PROJECT_ROOT / "requirements.txt",
                dry_run=True,
            )
        run_mock.assert_not_called()
        runtime_mock.assert_not_called()

    def test_pip_check_ignores_only_ambient_colab_conflicts(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=(
                "ipython 7.34.0 requires jedi, which is not installed.\n"
                "gradio 6.20.0 has requirement huggingface-hub<2.0,>=1.2.0, "
                "but you have huggingface-hub 0.34.4.\n"
            ),
            stderr="",
        )
        with (
            mock.patch(
                "scripts.install_dependencies.subprocess.run", return_value=completed
            ),
            mock.patch.object(
                install_dependencies,
                "_project_dependency_closure",
                return_value={"torch", "vllm", "flash-attn"},
            ),
        ):
            install_dependencies._run_project_pip_check(
                ["python", "-m", "pip", "check"],
                ["torch", "vllm", "flash-attn"],
            )

    def test_pip_check_still_fails_for_project_dependency_conflicts(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=(
                "vllm 0.10.2 has requirement torch==2.8.0, "
                "but you have torch 2.11.0.\n"
            ),
            stderr="",
        )
        with (
            mock.patch(
                "scripts.install_dependencies.subprocess.run", return_value=completed
            ),
            mock.patch.object(
                install_dependencies,
                "_project_dependency_closure",
                return_value={"torch", "vllm", "flash-attn"},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "vllm 0.10.2"):
                install_dependencies._run_project_pip_check(
                    ["python", "-m", "pip", "check"],
                    ["torch", "vllm", "flash-attn"],
                )

    def test_gpu_entrypoints_are_wired_to_the_preflight(self):
        for relative_path in (
            "briefline/commands/train.py",
            "briefline/commands/evaluate.py",
            "briefline/commands/all_stages.py",
        ):
            with self.subTest(relative_path=relative_path):
                source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("ensure_runtime_compatible", source)


if __name__ == "__main__":
    unittest.main()
