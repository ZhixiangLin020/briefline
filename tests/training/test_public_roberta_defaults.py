import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from briefline.commands.all_stages import run_all
from briefline.commands.train import build_parser, config_from_args
from training.config import (
    DEFAULT_ROBERTA_MODEL,
    TrainingDataConfig,
    TrainingRunConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PublicRobertaDefaultTests(unittest.TestCase):
    def test_training_config_and_cli_use_the_public_model_id(self):
        cfg = TrainingRunConfig(
            data=TrainingDataConfig(Path("cnn"), Path("kpt"), smoke_test=True),
            output_dir=Path("out"),
        ).normalized()
        self.assertEqual(DEFAULT_ROBERTA_MODEL, "FacebookAI/roberta-large")
        self.assertEqual(cfg.roberta_path, DEFAULT_ROBERTA_MODEL)

        args = build_parser().parse_args(
            [
                "--cnn-dm-dataset",
                "cnn",
                "--kptimes-dataset",
                "kpt",
                "--output-dir",
                "out",
                "--smoke-test",
                "--dry-run",
            ]
        )
        self.assertEqual(config_from_args(args).roberta_path, DEFAULT_ROBERTA_MODEL)

    def test_all_stages_fallback_uses_the_public_model_id(self):
        payload = {
            "training": {
                "cnn_dm_dataset": "cnn",
                "kptimes_dataset": "kpt",
                "output_dir": "out",
                "smoke_test": True,
                "dry_run": True,
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            config_path = Path(temp) / "pipeline.yaml"
            config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with mock.patch(
                "training.pipeline.run_training",
                return_value={"status": "dry_run"},
            ) as run_training:
                run_all(config_path)

        training_cfg = run_training.call_args.args[0]
        self.assertEqual(training_cfg.roberta_path, DEFAULT_ROBERTA_MODEL)

    def test_public_yaml_configs_use_the_public_model_id(self):
        smoke = yaml.safe_load(
            (PROJECT_ROOT / "configs/smoke_test.yaml").read_text(encoding="utf-8")
        )
        pipeline = yaml.safe_load(
            (PROJECT_ROOT / "configs/pipeline_all.example.yaml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(smoke["training"]["roberta_path"], DEFAULT_ROBERTA_MODEL)
        self.assertEqual(
            pipeline["training"]["roberta_path"], DEFAULT_ROBERTA_MODEL
        )
        self.assertEqual(
            pipeline["evaluation"]["roberta_path"], DEFAULT_ROBERTA_MODEL
        )

    def test_trainer_fallback_uses_the_shared_public_default(self):
        tree = ast.parse(
            (PROJECT_ROOT / "training/trainer.py").read_text(encoding="utf-8")
        )
        sft_config = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SFTConfig"
        )
        assignment = next(
            node
            for node in sft_config.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "eval_mover_model_path"
        )
        self.assertIsInstance(assignment.value, ast.Name)
        self.assertEqual(assignment.value.id, "DEFAULT_ROBERTA_MODEL")

    def test_historical_absolute_path_is_isolated_to_original_config(self):
        historical_config = PROJECT_ROOT / "configs/original_experiment.yaml"
        historical = yaml.safe_load(
            historical_config.read_text(encoding="utf-8")
        )["training"]["roberta_path"]
        self.assertTrue(Path(historical).is_absolute())

        text_suffixes = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".example"}
        for path in PROJECT_ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in text_suffixes:
                continue
            if path == historical_config:
                continue
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                self.assertNotIn(historical, path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
