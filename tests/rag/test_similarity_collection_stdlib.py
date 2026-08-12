from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIMILARITY_MODULE = PROJECT_ROOT / "rag" / "similarity_pipeline.py"


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function not found: {name}")


class SimilarityCollectionTests(unittest.TestCase):
    def test_similarity_uses_the_configured_collection_name(self) -> None:
        tree = ast.parse(SIMILARITY_MODULE.read_text(encoding="utf-8"))
        schema_function = find_function(tree, "ensure_similarity_collection_schema")
        runner_function = find_function(
            tree,
            "run_guardian_similar_articles_pipeline",
        )

        schema_arguments = {argument.arg for argument in schema_function.args.args}
        self.assertIn("collection_name", schema_arguments)

        collection_calls = {
            node.func.attr: ast.unparse(node.args[0])
            for node in ast.walk(schema_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"exists", "use"}
            and node.args
        }
        self.assertEqual(collection_calls["exists"], "collection_name")
        self.assertEqual(collection_calls["use"], "collection_name")

        schema_calls = [
            node
            for node in ast.walk(runner_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ensure_similarity_collection_schema"
        ]
        self.assertEqual(len(schema_calls), 1)
        keyword_values = {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in schema_calls[0].keywords
        }
        self.assertEqual(keyword_values["collection_name"], "collection_name")

    def test_recommendations_are_checkpointed_inside_the_article_loop(self) -> None:
        tree = ast.parse(SIMILARITY_MODULE.read_text(encoding="utf-8"))
        runner_function = find_function(
            tree,
            "run_guardian_similar_articles_pipeline",
        )
        article_loops = [
            node
            for node in ast.walk(runner_function)
            if isinstance(node, ast.For)
            and "query_table.iterrows()" in ast.unparse(node.iter)
        ]
        self.assertEqual(len(article_loops), 1)
        checkpoint_calls = [
            node
            for node in ast.walk(article_loops[0])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "save_recommendations_to_postgres"
        ]
        self.assertEqual(len(checkpoint_calls), 1)


if __name__ == "__main__":
    unittest.main()
