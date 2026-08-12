from __future__ import annotations

import ast
import contextlib
import io
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARDIAN_MODULE = PROJECT_ROOT / "rag" / "guardian_pipeline.py"


def parse_guardian_module():
    source = GUARDIAN_MODULE.read_text(encoding="utf-8")
    return source, ast.parse(source)


def assignment_nodes(tree, names):
    return {
        target.id: node
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }


def function_nodes(tree, names):
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }


def future_import_node(tree):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )


class GenerationTaskStatusTests(unittest.TestCase):
    def test_schema_migration_is_additive_and_idempotent(self) -> None:
        source, tree = parse_guardian_module()
        initializer = function_nodes(tree, {"initialize_postgres_tables"})[
            "initialize_postgres_tables"
        ]
        initializer_source = ast.get_source_segment(source, initializer)

        self.assertIn(
            "CREATE TABLE IF NOT EXISTS {GENERATION_TASK_STATUS_TABLE}",
            initializer_source,
        )
        self.assertIn("PRIMARY KEY (source_id, task)", initializer_source)
        self.assertIn("ON DELETE CASCADE", initializer_source)
        self.assertIn("CHECK (task IN ('highlight', 'both'))", initializer_source)
        self.assertIn("CHECK (status = 'ineligible')", initializer_source)

    def test_status_upsert_is_idempotent_and_rejects_unknown_tasks(self) -> None:
        _, tree = parse_guardian_module()
        assignments = assignment_nodes(
            tree,
            {
                "GENERATION_TASK_STATUS_TABLE",
                "GENERATION_ELIGIBILITY_VERSION",
                "GENERATION_TASKS",
            },
        )
        functions = function_nodes(
            tree,
            {"_stringify_text", "save_generation_task_statuses"},
        )
        module = ast.Module(
            body=[
                future_import_node(tree),
                assignments["GENERATION_TASK_STATUS_TABLE"],
                assignments["GENERATION_ELIGIBILITY_VERSION"],
                assignments["GENERATION_TASKS"],
                functions["_stringify_text"],
                functions["save_generation_task_statuses"],
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(GUARDIAN_MODULE), "exec"), namespace)

        executed = {}

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def executemany(self, sql, records):
                executed["sql"] = sql
                executed["records"] = list(records)

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return FakeCursor()

        namespace["initialize_postgres_tables"] = lambda: None
        namespace["get_database_url"] = lambda: "postgresql://test"
        namespace["psycopg"] = types.SimpleNamespace(
            connect=lambda *_args, **_kwargs: FakeConnection()
        )

        with contextlib.redirect_stdout(io.StringIO()):
            stats = namespace["save_generation_task_statuses"]([
                {
                    "source_id": "article-1",
                    "task": "both",
                    "reason": "input_not_eligible_under_current_generation_rules",
                },
                {
                    "source_id": "article-2",
                    "task": "unknown",
                    "reason": "invalid",
                },
            ])

        self.assertEqual(stats, {"received": 2, "recorded": 1, "invalid": 1})
        self.assertIn("ON CONFLICT (source_id, task) DO UPDATE", executed["sql"])
        self.assertEqual(len(executed["records"]), 1)
        self.assertEqual(
            executed["records"][0]["eligibility_version"],
            namespace["GENERATION_ELIGIBILITY_VERSION"],
        )

    def test_pending_queries_ignore_only_current_version_ineligible_rows(self) -> None:
        _, tree = parse_guardian_module()
        names = {
            "BATCH_N",
            "RAW_ARTICLES_TABLE",
            "MODEL_OUTPUTS_TABLE",
            "GENERATION_TASK_STATUS_TABLE",
            "GENERATION_ELIGIBILITY_VERSION",
        }
        assignments = assignment_nodes(tree, names)
        functions = function_nodes(
            tree,
            {
                "_stringify_text",
                "load_raw_articles_for_generation",
                "load_pending_generation_source_ids",
            },
        )
        module = ast.Module(
            body=[
                future_import_node(tree),
                *(assignments[name] for name in names),
                functions["_stringify_text"],
                functions["load_raw_articles_for_generation"],
                functions["load_pending_generation_source_ids"],
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(GUARDIAN_MODULE), "exec"), namespace)

        executions = []

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, query, params):
                executions.append((query, params))

            def fetchall(self):
                return []

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return FakeCursor()

        namespace["initialize_postgres_tables"] = lambda: None
        namespace["get_database_url"] = lambda: "postgresql://test"
        namespace["dict_row"] = object()
        namespace["psycopg"] = types.SimpleNamespace(
            connect=lambda *_args, **_kwargs: FakeConnection()
        )

        with contextlib.redirect_stdout(io.StringIO()):
            namespace["load_raw_articles_for_generation"](
                limit=7,
                source_ids=["article-1"],
            )
            namespace["load_pending_generation_source_ids"](
                limit=5,
                exclude_source_ids=["article-2"],
            )

        version = namespace["GENERATION_ELIGIBILITY_VERSION"]
        raw_query, raw_params = executions[0]
        pending_query, pending_params = executions[1]
        for query in (raw_query, pending_query):
            self.assertIn("generation_task_status", query)
            self.assertIn("task_status.eligibility_version = %s", query)
            self.assertIn("NOT has_highlight_output AND NOT highlight_ineligible", query)
            self.assertIn("NOT has_both_output AND NOT both_ineligible", query)
        self.assertEqual(raw_params, (version, version, ["article-1"], 7))
        self.assertEqual(pending_params, (version, version, ["article-2"], 5))

    def test_completion_still_requires_two_real_model_outputs(self) -> None:
        source, tree = parse_guardian_module()
        completion = function_nodes(tree, {"load_generation_complete_source_ids"})[
            "load_generation_complete_source_ids"
        ]
        completion_source = ast.get_source_segment(source, completion)

        self.assertIn("output.task = 'highlight'", completion_source)
        self.assertIn("output.task = 'both'", completion_source)
        self.assertNotIn("GENERATION_TASK_STATUS_TABLE", completion_source)


if __name__ == "__main__":
    unittest.main()
