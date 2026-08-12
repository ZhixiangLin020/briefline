import unittest
from pathlib import Path

from training.config import (
    ORIGINAL_DATASET_COUNTS,
    ORIGINAL_MODEL_CONFIG_SIGNATURE,
    TrainingDataConfig,
    TrainingRunConfig,
    validate_original_model_config,
)
from training.data import TrainingDataBundle
from training.pipeline import (
    _apply_original_tokenizer_padding,
    _validate_resume_preflight,
    build_run_manifest,
)


def make_config(smoke=False):
    return TrainingRunConfig(
        data=TrainingDataConfig(
            Path("cnn"),
            Path("kpt"),
            smoke_test=smoke,
        ),
        output_dir=Path("out"),
    ).normalized()


class FrozenTrainingConfigTests(unittest.TestCase):
    def test_base_model_architecture_signature_is_frozen(self):
        class FakeConfig:
            quantization_config = None

        good = FakeConfig()
        for name, value in ORIGINAL_MODEL_CONFIG_SIGNATURE.items():
            setattr(good, name, value)
        validate_original_model_config(good)

        bad = FakeConfig()
        for name, value in ORIGINAL_MODEL_CONFIG_SIGNATURE.items():
            setattr(bad, name, value)
        bad.num_hidden_layers = 28
        with self.assertRaisesRegex(ValueError, "frozen Qwen"):
            validate_original_model_config(bad)

    def test_full_training_rejects_non_original_seed(self):
        cfg = TrainingDataConfig(Path("cnn"), Path("kpt"), seed=7)
        with self.assertRaisesRegex(ValueError, "freezes seed=42"):
            cfg.normalized()

    def test_smoke_seed_override_is_explicitly_non_reproduction(self):
        cfg = TrainingDataConfig(
            Path("cnn"),
            Path("kpt"),
            seed=7,
            smoke_test=True,
        ).normalized()
        self.assertEqual(cfg.seed, 7)
        self.assertEqual(cfg.run_mode, "smoke_test")

    def test_tokenizer_padding_matches_original_unconditional_assignment(self):
        class FakeTokenizer:
            eos_token = "<eos>"
            pad_token = "<old-pad>"

        tokenizer = FakeTokenizer()
        _apply_original_tokenizer_padding(tokenizer)
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)

    def test_resume_preflight_requires_checkpoint_and_top_k_history(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint-100"
            checkpoint.mkdir()
            cfg = TrainingRunConfig(
                data=TrainingDataConfig(Path("cnn"), Path("kpt")),
                output_dir=root / "out",
                best_model_dir=root / "best",
                resume_from_checkpoint=checkpoint,
            ).normalized()
            with self.assertRaisesRegex(FileNotFoundError, "top-k manifest"):
                _validate_resume_preflight(cfg)

            cfg.best_model_dir.mkdir()
            (cfg.best_model_dir / "best_k_metrics.json").write_text(
                "{}",
                encoding="utf-8",
            )
            _validate_resume_preflight(cfg)

    def test_formal_training_arguments_are_frozen(self):
        cfg = make_config()
        args = cfg.training_arguments_kwargs()
        self.assertEqual(args["per_device_train_batch_size"], 8)
        self.assertEqual(args["gradient_accumulation_steps"], 1)
        self.assertEqual(args["learning_rate"], 4e-4)
        self.assertEqual(args["num_train_epochs"], 6)
        self.assertEqual(args["save_steps"], 0.05)
        self.assertEqual(args["eval_steps"], 0.05)
        self.assertEqual(args["label_smoothing_factor"], 0.05)
        self.assertFalse(args["load_best_model_at_end"])
        self.assertFalse(args["greater_is_better"])

    def test_sft_and_adalora_values_match_recorded_run(self):
        cfg = make_config()
        sft = cfg.sft_config_kwargs(samples_per_epoch=37_739)
        self.assertEqual(sft["eval_metric_sample_size"], 300)
        self.assertEqual(sft["eval_metric_max_new_tokens"], 512)
        self.assertEqual(sft["prompt_loss_weight_start"], 0.04)
        self.assertEqual(sft["prompt_loss_weight_end"], 0.01)
        self.assertEqual(sft["prompt_loss_decay_anchor_value"], 0.8)
        self.assertEqual(sft["loss_normalization"], "sample_mean")
        self.assertEqual(
            sft["epoch_ratio_schedule"],
            [
                {"cnn_dm": 0.7, "kptime": 0.3},
                {"cnn_dm": 0.6, "kptime": 0.4},
                {"cnn_dm": 0.5, "kptime": 0.5},
            ],
        )
        peft = cfg.peft_spec_kwargs(total_steps=28_308)
        self.assertEqual(peft["init_r"], 128)
        self.assertEqual(peft["target_r"], 90)
        self.assertEqual(peft["tinit"], 11_323)
        self.assertEqual(peft["tfinal"], 5_661)
        self.assertEqual(peft["deltaT"], 250)
        self.assertEqual(peft["target_modules"], "all-linear")
        self.assertFalse(peft["use_rslora"])
        self.assertFalse(peft["use_dora"])

    def test_smoke_changes_scale_not_algorithm(self):
        formal = make_config()
        smoke = make_config(smoke=True)
        self.assertEqual(smoke.num_train_epochs, 3)
        formal_sft = formal.sft_config_kwargs(samples_per_epoch=37_739)
        smoke_sft = smoke.sft_config_kwargs(samples_per_epoch=500)
        differing = {
            key for key in formal_sft if formal_sft[key] != smoke_sft[key]
        }
        self.assertEqual(differing, {"samples_per_epoch"})
        self.assertEqual(
            formal.peft_spec_kwargs(total_steps=100),
            smoke.peft_spec_kwargs(total_steps=100),
        )

    def test_manifest_distinguishes_old_and_new_data_snapshots(self):
        cfg = make_config()
        bundle = TrainingDataBundle(
            train_pools={}, eval_pools={}, test_pools={},
            counts=dict(ORIGINAL_DATASET_COUNTS),
            samples_per_epoch=37_739,
            run_mode="reproduction",
            seed=42,
            source_paths={"cnn_dm": "cnn", "kptime": "kpt"},
            source_manifests={
                "cnn_dm": {
                    "prepare_config_hash": "cnn-hash",
                    "package_versions": {"datasets": "test"},
                    "tokenizer_contract": {"vocab_sha256": "test"},
                },
                "kptime": {
                    "prepare_config_hash": "kpt-hash",
                    "package_versions": {"datasets": "test"},
                    "tokenizer_contract": {"vocab_sha256": "test"},
                },
            },
        )
        manifest = build_run_manifest(cfg, bundle)
        self.assertTrue(manifest["experiment_reproduction"]["eligible"])
        bundle.counts["train_kptime"] = 37_334
        manifest = build_run_manifest(cfg, bundle)
        self.assertFalse(manifest["experiment_reproduction"]["eligible"])
        self.assertEqual(
            manifest["experiment_reproduction"]["mismatches"]["train_kptime"],
            {"expected": 34_287, "actual": 37_334},
        )

    def test_missing_manifest_prevents_reproduction_claim(self):
        cfg = make_config()
        bundle = TrainingDataBundle(
            train_pools={}, eval_pools={}, test_pools={},
            counts=dict(ORIGINAL_DATASET_COUNTS),
            samples_per_epoch=37_739,
            run_mode="reproduction",
            seed=42,
            source_paths={"cnn_dm": "cnn", "kptime": "kpt"},
            source_manifests={"cnn_dm": None, "kptime": None},
        )
        manifest = build_run_manifest(cfg, bundle)
        self.assertFalse(manifest["experiment_reproduction"]["eligible"])
        self.assertEqual(
            manifest["experiment_reproduction"]["provenance_issues"]
            ["missing_dataset_manifests"],
            ["cnn_dm", "kptime"],
        )


if __name__ == "__main__":
    unittest.main()
