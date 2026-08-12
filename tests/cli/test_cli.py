from __future__ import annotations

import contextlib
import io
import os
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from briefline import cli


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BrieflineCliTests(unittest.TestCase):
    def test_help_lists_every_workflow_without_importing_them(self) -> None:
        output = io.StringIO()
        with mock.patch.object(cli.importlib, "import_module") as import_module:
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--help"]), 0)

        import_module.assert_not_called()
        rendered = output.getvalue()
        for command in cli.COMMANDS:
            self.assertIn(command, rendered)

    def test_command_dispatch_forwards_arguments(self) -> None:
        command_main = mock.Mock(return_value=0)
        module = SimpleNamespace(main=command_main)
        with mock.patch.object(cli.importlib, "import_module", return_value=module):
            self.assertEqual(cli.main(["data", "--dataset", "cnn_dm"]), 0)

        command_main.assert_called_once_with(["--dataset", "cnn_dm"])

    def test_command_selects_pytorch_before_importing_workflow(self) -> None:
        command_main = mock.Mock(return_value=0)
        module = SimpleNamespace(main=command_main)

        def import_module(_name):
            self.assertEqual(os.environ["USE_TF"], "0")
            self.assertEqual(os.environ["USE_TORCH"], "1")
            return module

        with (
            mock.patch.dict(os.environ, {"USE_TF": "1", "USE_TORCH": "0"}),
            mock.patch.object(cli.importlib, "import_module", side_effect=import_module),
        ):
            self.assertEqual(cli.main(["rag", "--mode", "smoke"]), 0)

        command_main.assert_called_once_with(["--mode", "smoke"])

    def test_unknown_command_returns_usage_error(self) -> None:
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(cli.main(["missing"]), 2)
        self.assertIn("Unknown command: missing", error.getvalue())

    def test_root_has_one_cli_instead_of_run_script_scatter(self) -> None:
        self.assertEqual(list(PROJECT_ROOT.glob("run_*.py")), [])
        payload = tomllib.loads(
            (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        self.assertEqual(
            payload["project"]["scripts"]["briefline"],
            "briefline.cli:main",
        )


if __name__ == "__main__":
    unittest.main()
