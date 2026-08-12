from __future__ import annotations

import ast
import contextlib
import io
import types
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GUARDIAN_MODULE = PROJECT_ROOT / "rag" / "guardian_pipeline.py"


def load_two_task_runner():
    tree = ast.parse(GUARDIAN_MODULE.read_text(encoding="utf-8"))
    future_import = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    )
    assignments = {
        target.id: node
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "MODEL_OUTPUTS_TABLE",
            "GENERATION_ELIGIBILITY_VERSION",
            "GENERATION_TASKS",
            "MODEL_OUTPUT_COLUMNS",
        }
    }
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_guardian_two_tasks_batch_vllm"
    )
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_stringify_text",
            "_normalize_database_value",
            "_database_flag_is_true",
            "_article_source_id",
            "_pending_articles_for_task",
            "_dataset_source_ids",
            "_ineligible_task_statuses",
            "save_model_outputs_to_postgres",
        }
    }
    module = ast.Module(
        body=[
            future_import,
            assignments["MODEL_OUTPUTS_TABLE"],
            assignments["GENERATION_ELIGIBILITY_VERSION"],
            assignments["GENERATION_TASKS"],
            assignments["MODEL_OUTPUT_COLUMNS"],
            functions["_stringify_text"],
            functions["_normalize_database_value"],
            functions["save_model_outputs_to_postgres"],
            functions["_database_flag_is_true"],
            functions["_article_source_id"],
            functions["_pending_articles_for_task"],
            functions["_dataset_source_ids"],
            functions["_ineligible_task_statuses"],
            runner,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {"pd": pd}
    exec(compile(module, str(GUARDIAN_MODULE), "exec"), namespace)
    return namespace, namespace["run_guardian_two_tasks_batch_vllm"]


class GuardianPartialGenerationTests(unittest.TestCase):
    def run_case(self, *, articles, highlight_rows, both_rows):
        namespace, runner = load_two_task_runner()
        task_calls = []
        task_inputs = {}
        database_records = []
        status_records = []
        initialization_calls = []

        class FakeCursor:
            rowcount = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def executemany(self, _sql, records):
                database_records.extend(records)
                self.rowcount = len(records)

        class FakeConnection:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return FakeCursor()

        namespace["initialize_postgres_tables"] = lambda: initialization_calls.append(
            True
        )
        def build_datasets(**kwargs):
            task_inputs["highlight"] = [
                row.get("id") or row.get("source_id")
                for row in kwargs["highlight_articles"]
            ]
            task_inputs["both"] = [
                row.get("id") or row.get("source_id")
                for row in kwargs["both_articles"]
            ]
            return {
                "highlight": {"inference": highlight_rows},
                "both": {"inference": both_rows},
            }

        namespace["build_guardian_inference_datasets"] = build_datasets
        namespace["save_generation_task_statuses"] = (
            lambda records: status_records.extend(records)
        )

        def run_one_task(**kwargs):
            self.assertGreater(len(kwargs["ds"]), 0)
            task = kwargs["task"]
            task_calls.append(task)
            task_specific_columns = {
                "highlight": {
                    "starts_with_highlight",
                    "rough_sentence_count",
                },
                "both": {
                    "has_categories",
                    "has_keywords",
                    "order_ok",
                    "has_semicolon",
                },
            }
            other_task = "both" if task == "highlight" else "highlight"
            row = {
                column: None
                for column in namespace["MODEL_OUTPUT_COLUMNS"]
                if column not in task_specific_columns[other_task]
            }
            row.update(
                {
                    "task": task,
                    "source_id": "article-1",
                    "format_ok": True,
                    "maybe_repetition_loop": False,
                }
            )
            return [row]

        namespace["run_one_guardian_task_batch_vllm"] = run_one_task
        namespace["psycopg"] = types.SimpleNamespace(
            connect=lambda *_args, **_kwargs: FakeConnection()
        )
        namespace["get_database_url"] = lambda: "postgresql://test"

        with contextlib.redirect_stdout(io.StringIO()):
            frame, outputs = runner(
                articles=articles,
                llm=object(),
                tokenizer=object(),
                n=1,
                save_to_postgres=True,
            )

        return (
            task_calls,
            task_inputs,
            database_records,
            status_records,
            initialization_calls,
            frame,
            outputs,
        )

    def test_highlight_only_recovery_skips_empty_both_task_and_saves(self) -> None:
        calls, inputs, records, statuses, initialized, frame, _ = self.run_case(
            articles=[{"source_id": "article-1", "has_both_output": True}],
            highlight_rows=[{"source_id": "article-1"}],
            both_rows=[],
        )

        self.assertEqual(calls, ["highlight"])
        self.assertEqual(inputs, {"highlight": ["article-1"], "both": []})
        self.assertEqual(statuses, [])
        self.assertEqual(initialized, [True])
        self.assertEqual(frame["task"].tolist(), ["highlight"])
        for column in (
            "has_categories",
            "has_keywords",
            "order_ok",
            "has_semicolon",
        ):
            self.assertIsNone(frame.iloc[0][column])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task"], "highlight")
        self.assertIsNone(records[0]["has_categories"])

    def test_both_only_recovery_skips_empty_highlight_task_and_saves(self) -> None:
        calls, inputs, records, statuses, initialized, frame, _ = self.run_case(
            articles=[{"source_id": "article-1", "has_highlight_output": True}],
            highlight_rows=[],
            both_rows=[{"source_id": "article-1"}],
        )

        self.assertEqual(calls, ["both"])
        self.assertEqual(inputs, {"highlight": [], "both": ["article-1"]})
        self.assertEqual(statuses, [])
        self.assertEqual(initialized, [True])
        self.assertEqual(frame["task"].tolist(), ["both"])
        for column in ("starts_with_highlight", "rough_sentence_count"):
            self.assertIsNone(frame.iloc[0][column])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["task"], "both")
        self.assertIsNone(records[0]["starts_with_highlight"])

    def test_complete_pair_keeps_the_original_task_order(self) -> None:
        calls, inputs, records, statuses, initialized, frame, _ = self.run_case(
            articles=[{"source_id": "article-1"}],
            highlight_rows=[{"source_id": "article-1"}],
            both_rows=[{"source_id": "article-1"}],
        )

        self.assertEqual(calls, ["highlight", "both"])
        self.assertEqual(
            inputs,
            {"highlight": ["article-1"], "both": ["article-1"]},
        )
        self.assertEqual(statuses, [])
        self.assertEqual(initialized, [True])
        self.assertEqual(frame["task"].tolist(), ["highlight", "both"])
        self.assertEqual(
            [record["task"] for record in records],
            ["highlight", "both"],
        )

    def test_no_pending_task_is_a_clean_no_op(self) -> None:
        calls, inputs, records, statuses, initialized, frame, _ = self.run_case(
            articles=[{
                "source_id": "article-1",
                "has_highlight_output": True,
                "has_both_output": True,
            }],
            highlight_rows=[],
            both_rows=[],
        )

        self.assertEqual(calls, [])
        self.assertEqual(inputs, {"highlight": [], "both": []})
        self.assertEqual(statuses, [])
        self.assertEqual(initialized, [True])
        self.assertTrue(frame.empty)
        self.assertEqual(records, [])

    def test_existing_highlight_is_not_regenerated_when_both_is_ineligible(self) -> None:
        calls, inputs, records, statuses, initialized, frame, _ = self.run_case(
            articles=[{"source_id": "article-1", "has_highlight_output": True}],
            highlight_rows=[],
            both_rows=[],
        )

        self.assertEqual(calls, [])
        self.assertEqual(inputs, {"highlight": [], "both": ["article-1"]})
        self.assertEqual(initialized, [True])
        self.assertTrue(frame.empty)
        self.assertEqual(records, [])
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["source_id"], "article-1")
        self.assertEqual(statuses[0]["task"], "both")
        self.assertEqual(
            statuses[0]["reason"],
            "input_not_eligible_under_current_generation_rules",
        )

    def test_current_ineligible_status_is_not_scheduled_again(self) -> None:
        calls, inputs, records, statuses, _, frame, _ = self.run_case(
            articles=[{
                "source_id": "article-1",
                "has_highlight_output": True,
                "both_ineligible": True,
            }],
            highlight_rows=[],
            both_rows=[],
        )

        self.assertEqual(calls, [])
        self.assertEqual(inputs, {"highlight": [], "both": []})
        self.assertEqual(records, [])
        self.assertEqual(statuses, [])
        self.assertTrue(frame.empty)


if __name__ == "__main__":
    unittest.main()
