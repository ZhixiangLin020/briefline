import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def from_pretrained_calls(function):
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "from_pretrained":
            calls.append(node)
    return calls


class ModelDownloadPolicyTests(unittest.TestCase):
    def test_training_restores_original_pot_auto_install_fallback(self):
        source = (PROJECT_ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        pot_fallbacks = []
        for node in tree.body:
            if not isinstance(node, ast.Try):
                continue
            imports_ot = any(
                isinstance(statement, ast.Import)
                and any(alias.name == "ot" for alias in statement.names)
                for statement in node.body
            )
            if imports_ot:
                pot_fallbacks.append(node)

        self.assertEqual(len(pot_fallbacks), 1)
        fallback = pot_fallbacks[0]
        self.assertEqual(len(fallback.handlers), 1)

        handler = fallback.handlers[0]
        self.assertIsInstance(handler.type, ast.Name)
        self.assertEqual(handler.type.id, "Exception")

        pip_install_calls = [
            node
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
            and node.func.attr == "check_call"
        ]
        self.assertEqual(len(pip_install_calls), 1)

        command = pip_install_calls[0].args[0]
        self.assertIsInstance(command, ast.List)
        self.assertEqual(len(command.elts), 5)
        self.assertIsInstance(command.elts[0], ast.Attribute)
        self.assertEqual(command.elts[0].attr, "executable")
        self.assertEqual(
            [element.value for element in command.elts[1:]],
            ["-m", "pip", "install", "POT"],
        )

        reimports_ot = any(
            isinstance(node, ast.Import)
            and any(alias.name == "ot" for alias in node.names)
            for node in handler.body
        )
        self.assertTrue(reimports_ot)

    def test_training_mover_encoder_allows_cache_miss_download(self):
        source = (PROJECT_ROOT / "training" / "trainer.py").read_text(encoding="utf-8")
        function = find_function(ast.parse(source), "_get_mover_encoder")
        calls = from_pretrained_calls(function)
        self.assertEqual(len(calls), 3)
        for call in calls:
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            self.assertNotIn("local_files_only", keywords)

    def test_final_roberta_encoder_defaults_to_online_fallback(self):
        source = (PROJECT_ROOT / "evaluation" / "vllm_pipeline.py").read_text(
            encoding="utf-8"
        )
        function = find_function(ast.parse(source), "load_roberta_encoder")
        argument_names = [argument.arg for argument in function.args.kwonlyargs]
        default_index = argument_names.index("local_files_only")
        default = function.args.kw_defaults[default_index]
        self.assertIsInstance(default, ast.Constant)
        self.assertFalse(default.value)

    def test_final_evaluation_defaults_to_online_fallback(self):
        source = (PROJECT_ROOT / "evaluation" / "vllm_pipeline.py").read_text(
            encoding="utf-8"
        )
        function = find_function(
            ast.parse(source), "run_mixed_full_valid_test_eval_vllm"
        )
        argument_names = [argument.arg for argument in function.args.kwonlyargs]
        default_index = argument_names.index("local_files_only")
        default = function.args.kw_defaults[default_index]
        self.assertIsInstance(default, ast.Constant)
        self.assertFalse(default.value)

    def test_evaluation_entrypoint_explicitly_allows_hub_downloads(self):
        source = (PROJECT_ROOT / "evaluation" / "pipeline.py").read_text(
            encoding="utf-8"
        )
        function = find_function(ast.parse(source), "run_evaluation")
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_mixed_full_valid_test_eval_vllm"
        ]
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertIn("local_files_only", keywords)
        self.assertIsInstance(keywords["local_files_only"], ast.Constant)
        self.assertFalse(keywords["local_files_only"].value)


if __name__ == "__main__":
    unittest.main()
