import contextlib
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from briefline.commands import data as data_command
from data_processing.config import PipelineConfig
from data_processing.pipeline import (
    CNN_DM_ORIGINAL_PREPARE_PARAMETERS,
    KPTIMES_ORIGINAL_PREPARE_PARAMETERS,
    _prepare_kptimes,
    _require_packages,
    _select_cnn_dm,
    _select_kptimes,
    _tokenizer_contract,
    _validate_prepared_row,
    validate_data,
    run_pipeline,
)


ROOT = Path(__file__).resolve().parents[2]


class PipelineTests(unittest.TestCase):
    def test_validate_stage_checks_rows_after_the_old_20_row_cap(self):
        class FakeDataset:
            column_names = [
                "input_ids",
                "attention_mask",
                "labels",
                "loss_weights",
            ]

            def __init__(self):
                self.rows = [
                    {
                        "input_ids": [1, 2, 3],
                        "attention_mask": [1, 1, 1],
                        "labels": [-100, 2, 3],
                        "loss_weights": [0.0, 0.2, 1.0],
                    }
                    for _ in range(21)
                ]
                self.rows[20]["loss_weights"] = [0.0, 1.0]

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        class FakeDatasetDict(dict):
            pass

        class FakeTokenizer:
            model_max_length = 8

            def __len__(self):
                return 10

        fake_datasets = types.ModuleType("datasets")
        fake_datasets.DatasetDict = FakeDatasetDict
        fake_datasets.load_from_disk = lambda _path: FakeDatasetDict(
            train=FakeDataset(),
            validation=FakeDataset(),
            test=FakeDataset(),
        )
        previous = sys.modules.get("datasets")
        sys.modules["datasets"] = fake_datasets
        try:
            with tempfile.TemporaryDirectory() as temp:
                output_dir = Path(temp) / "output"
                (output_dir / "prepared").mkdir(parents=True)
                cfg = PipelineConfig(
                    dataset="cnn_dm",
                    stage="validate",
                    output_dir=output_dir,
                ).normalized()
                with patch("data_processing.pipeline._require_packages"), patch(
                    "data_processing.pipeline._load_tokenizer",
                    return_value=FakeTokenizer(),
                ):
                    with self.assertRaisesRegex(ValueError, "row 20"):
                        validate_data(cfg)
        finally:
            if previous is None:
                sys.modules.pop("datasets", None)
            else:
                sys.modules["datasets"] = previous

    def test_manifest_tokenizer_contract_is_content_based(self):
        class FakeTokenizer:
            model_max_length = 128
            chat_template = "template"
            bos_token_id = None
            eos_token_id = 2
            unk_token_id = None

            def __len__(self):
                return 3

            def get_vocab(self):
                return {"a": 0, "b": 1, "c": 2}

        contract = _tokenizer_contract(FakeTokenizer())
        self.assertEqual(contract["vocab_size"], 3)
        self.assertEqual(len(contract["vocab_sha256"]), 64)
        self.assertEqual(len(contract["chat_template_sha256"]), 64)
        self.assertNotIn("pad_token_id", contract)

    def test_prepared_row_contract_checks_actual_tokenizer_range(self):
        class FakeTokenizer:
            model_max_length = 8

            def __len__(self):
                return 10

        row = {
            "input_ids": [1, 2, 99],
            "attention_mask": [1, 1, 1],
            "labels": [-100, 2, 99],
            "loss_weights": [0.0, 0.2, 1.0],
        }
        with self.assertRaisesRegex(ValueError, "outside vocabulary"):
            _validate_prepared_row(
                row,
                dataset_name="train",
                row_idx=0,
                tokenizer=FakeTokenizer(),
            )

    def test_configuration_normalization(self):
        cfg = PipelineConfig(
            dataset="cnn-dm",
            stage="ALL",
            output_dir=Path("output"),
            limit=0,
            num_proc=0,
        ).normalized()
        self.assertEqual(cfg.dataset, "cnn_dm")
        self.assertEqual(cfg.stage, "all")
        self.assertEqual(cfg.limit, 1)
        self.assertEqual(cfg.num_proc, 1)
        self.assertEqual(cfg.cache_dir, Path("output/cache"))

    def test_cli_builds_config_without_running_external_work(self):
        output = io.StringIO()
        with patch(
            "briefline.commands.data.run_pipeline", return_value={"ok": True}
        ) as mocked:
            with contextlib.redirect_stdout(output):
                code = data_command.main(
                    [
                        "--dataset",
                        "kptimes",
                        "--stage",
                        "all",
                        "--output-dir",
                        "result",
                        "--limit",
                        "10",
                    ]
                )
        self.assertEqual(code, 0)
        self.assertEqual(mocked.call_args.args[0].limit, 10)
        self.assertIn('"ok": true', output.getvalue())

    def test_missing_dependency_fails_before_processing(self):
        with self.assertRaisesRegex(RuntimeError, "Missing packages"):
            _require_packages(["package_that_does_not_exist_12345"], stage="test")

    def test_cnn_selection_counts_a_dataset_without_assuming_a_train_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                dataset="cnn_dm",
                stage="select",
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
                limit=3,
                seed=17,
            ).normalized()
            fake_result = {"picked_raw": [object(), object(), object()]}
            with patch("data_processing.pipeline._require_packages"), patch(
                "data_processing.cnn_dm.run_cnn_dm_end2end",
                return_value=fake_result,
            ) as mocked:
                result = _select_cnn_dm(cfg)

        self.assertEqual(result["selected_rows"], 3)
        self.assertEqual(mocked.call_args.kwargs["cfg"].seed, 17)
        self.assertEqual(mocked.call_args.kwargs["seed"], 17)

    def test_kptimes_selection_uses_the_formal_experiment_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                dataset="kptimes",
                stage="select",
                output_dir=Path(tmp) / "output",
                cache_dir=Path(tmp) / "cache",
                seed=23,
            ).normalized()
            fake_result = ([object()], {"picked_len": 1}, {}, [0], None)
            with patch("data_processing.pipeline._require_packages"), patch(
                "data_processing.kptimes.load_kptimes_raw",
                return_value=object(),
            ), patch(
                "data_processing.kptimes.build_kptimes_dedup_dataset_v2",
                return_value=fake_result,
            ) as mocked:
                result = _select_kptimes(cfg)

        kwargs = mocked.call_args.kwargs
        self.assertEqual(result["selected_rows"], 1)
        self.assertEqual(kwargs["cfg"].seed, 23)
        self.assertEqual(kwargs["seed"], 23)
        self.assertEqual(kwargs["protect_n"], 40)
        self.assertEqual(kwargs["cluster_prefer_side"], "max")
        self.assertIs(kwargs["include_body"], True)
        self.assertEqual(kwargs["body_max_words"], 220)
        self.assertEqual(kwargs["body_max_chars"], 0)
        self.assertEqual(kwargs["sample_growth"], "sqrt")
        self.assertEqual(kwargs["sample_tau"], 1.0)
        self.assertEqual(kwargs["sample_cap"], 10_000)
        self.assertEqual(kwargs["gap_side"], "right")
        self.assertEqual(kwargs["log_base"], 4.0)
        self.assertIs(kwargs["strict_prepared_meta"], True)
        self.assertIs(kwargs["strict_emb_meta"], True)

    def test_original_prepare_parameters_are_frozen(self):
        self.assertEqual(CNN_DM_ORIGINAL_PREPARE_PARAMETERS["article_max_tokens"], 2_500)
        self.assertEqual(CNN_DM_ORIGINAL_PREPARE_PARAMETERS["highlight_prefix_weight"], 0.2)
        self.assertEqual(CNN_DM_ORIGINAL_PREPARE_PARAMETERS["terminal_loss_mode"], "final_only")

        self.assertEqual(KPTIMES_ORIGINAL_PREPARE_PARAMETERS["body_max_tokens"], 2_000)
        self.assertEqual(KPTIMES_ORIGINAL_PREPARE_PARAMETERS["keyword_separator_list"], [",", ";"])
        self.assertEqual(KPTIMES_ORIGINAL_PREPARE_PARAMETERS["separator_weight"], 0.8)
        self.assertEqual(KPTIMES_ORIGINAL_PREPARE_PARAMETERS["keyword_token_idf_temperature"], 0.15)
        self.assertEqual(KPTIMES_ORIGINAL_PREPARE_PARAMETERS["keyword_token_idf_cap"], 2)

    def test_kptimes_prepare_passes_the_original_experiment_parameters(self):
        class FakeDatasetDict(dict):
            def save_to_disk(self, path):
                self.saved_to = path

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            (output_dir / "selected").mkdir(parents=True)
            cfg = PipelineConfig(
                dataset="kptimes",
                stage="prepare",
                output_dir=output_dir,
                cache_dir=Path(tmp) / "cache",
                task_mode="both",
            ).normalized()
            fake_ds = FakeDatasetDict(
                train=[1, 2],
                validation=[3],
                test=[4],
            )
            with patch(
                "data_processing.kptimes.build_kptimes_title_cls_trainer_dataset_v3",
                return_value=(fake_ds, object()),
            ) as mocked, patch(
                "data_processing.pipeline._write_prepared_manifest",
                return_value=output_dir / "prepared" / "manifest.json",
            ):
                _prepare_kptimes(cfg, tokenizer=object())

        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["body_max_tokens"], 2_000)
        self.assertEqual(kwargs["keyword_separator_list"], [",", ";"])
        self.assertEqual(kwargs["keyword_token_idf_temperature"], 0.15)
        self.assertEqual(kwargs["keyword_token_idf_cap"], 2)
        self.assertEqual(kwargs["prefix_weight"], 0.2)
        self.assertEqual(kwargs["separator_weight"], 0.8)
        self.assertEqual(kwargs["terminal_active_weight"], 1.0)

    def test_stage_all_runs_select_prepare_and_validate_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PipelineConfig(
                dataset="cnn_dm",
                stage="all",
                output_dir=Path(tmp) / "output",
            )
            call_order = []

            def record(name):
                def inner(_cfg):
                    call_order.append(name)
                    return {"name": name}
                return inner

            with patch("data_processing.pipeline.select_data", side_effect=record("select")), patch(
                "data_processing.pipeline.prepare_data", side_effect=record("prepare")
            ), patch("data_processing.pipeline.validate_data", side_effect=record("validate")):
                result = run_pipeline(cfg)

        self.assertEqual(call_order, ["select", "prepare", "validate"])
        self.assertEqual(set(result), {"dataset", "stage", "select", "prepare", "validate"})

    def test_imports_create_no_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = (
                "import data_processing.core; "
                "import data_processing.cnn_dm; "
                "import data_processing.kptimes; "
                "import data_processing.pipeline"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp,
                env={"PYTHONPATH": str(ROOT)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
