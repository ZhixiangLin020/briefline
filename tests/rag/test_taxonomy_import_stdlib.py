from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_MODULE = PROJECT_ROOT / "rag" / "taxonomy_generation.py"
FORBIDDEN_IMPORT_TIME_CALLS = {
    "display",
    "generate_broad_category_taxonomy",
    "load_final_category_inventory",
    "print",
    "taxonomy_to_mapping_dataframe",
}


def call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


class TaxonomyImportTests(unittest.TestCase):
    def test_taxonomy_module_has_no_import_time_pipeline_calls(self) -> None:
        tree = ast.parse(TAXONOMY_MODULE.read_text(encoding="utf-8"))
        violations = []

        for node in tree.body:
            call = None
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                if isinstance(node.value, ast.Call):
                    call = node.value

            if call is not None and call_name(call) in FORBIDDEN_IMPORT_TIME_CALLS:
                violations.append((node.lineno, call_name(call)))

        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
