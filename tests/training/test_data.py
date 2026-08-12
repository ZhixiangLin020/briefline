import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.config import TrainingDataConfig
from training.data import (
    build_training_data_bundle,
    validate_bundle_tokenizer_compatibility,
    validate_manifest_tokenizer_for_training,
    validate_trainer_split,
)


class FakeDataset:
    def __init__(self, rows, operations=None):
        self.rows = list(rows)
        self.operations = [] if operations is None else operations
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[int(index)]

    def select(self, indices):
        indices = [int(value) for value in indices]
        self.operations.append(("select", indices))
        return FakeDataset([self.rows[index] for index in indices], self.operations)

    def shuffle(self, seed):
        self.operations.append(("shuffle", int(seed)))
        order = np.random.default_rng(int(seed)).permutation(len(self.rows)).tolist()
        return FakeDataset([self.rows[index] for index in order], self.operations)

    def remove_columns(self, columns):
        remove = set(columns)
        return FakeDataset(
            [
                {key: value for key, value in row.items() if key not in remove}
                for row in self.rows
            ],
            self.operations,
        )


def make_split(size, prefix):
    rows = []
    for index in range(size):
        rows.append(
            {
                "input_ids": [1, 2, index + 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, index + 3],
                "loss_weights": [0.0, 0.2, 1.0],
                "row_id": f"{prefix}-{index}",
            }
        )
    return FakeDataset(rows)


def make_dataset_dict(prefix, train=10, validation=6, test=5):
    return {
        "train": make_split(train, f"{prefix}-train"),
        "validation": make_split(validation, f"{prefix}-validation"),
        "test": make_split(test, f"{prefix}-test"),
    }


def make_manifest(dataset_name, tokenizer_name, ds_dict):
    return {
        "schema_version": 2,
        "dataset": dataset_name,
        "tokenizer_name": tokenizer_name,
        "prepare_parameters": {"frozen": True},
        "prepare_config_hash": f"{dataset_name}-hash",
        "package_versions": {"datasets": "test"},
        "splits": {
            split: {"rows": len(ds_dict[split])}
            for split in ("train", "validation", "test")
        },
    }


class TrainingDataTests(unittest.TestCase):
    def test_full_mode_preserves_original_seed_offsets_and_epoch_size(self):
        cnn = make_dataset_dict("cnn", train=10)
        kpt = make_dataset_dict("kpt", train=8)
        config = TrainingDataConfig(Path("cnn"), Path("kpt"), seed=42)

        bundle = build_training_data_bundle(cnn, kpt, config)

        expected_cnn_selection = np.random.default_rng(143).permutation(10).tolist()
        expected_kpt_selection = np.random.default_rng(144).permutation(8).tolist()
        self.assertEqual(cnn["train"].operations[0], ("select", expected_cnn_selection))
        self.assertEqual(cnn["train"].operations[1], ("shuffle", 243))
        self.assertEqual(kpt["train"].operations[0], ("select", expected_kpt_selection))
        self.assertEqual(kpt["train"].operations[1], ("shuffle", 244))
        self.assertEqual(cnn["validation"].operations[0][1], np.random.default_rng(443).permutation(6).tolist())
        self.assertEqual(kpt["test"].operations[0][1], np.random.default_rng(544).permutation(5).tolist())
        self.assertEqual(bundle.samples_per_epoch, 10)
        self.assertEqual(bundle.run_mode, "reproduction")
        self.assertEqual(bundle.train_pools["cnn_dm"].column_names, [
            "input_ids", "attention_mask", "labels", "loss_weights"
        ])
        self.assertIn("row_id", cnn["train"].column_names)

    def test_smoke_mode_reduces_each_pool_without_changing_source_structure(self):
        cnn = make_dataset_dict("cnn", train=10, validation=6, test=5)
        kpt = make_dataset_dict("kpt", train=8, validation=6, test=5)
        config = TrainingDataConfig(
            Path("cnn"),
            Path("kpt"),
            smoke_test=True,
            max_train_samples_per_dataset=3,
            max_validation_samples_per_dataset=2,
            max_test_samples_per_dataset=2,
            samples_per_epoch=4,
        )

        bundle = build_training_data_bundle(cnn, kpt, config)

        self.assertEqual(bundle.run_mode, "smoke_test")
        self.assertEqual(bundle.samples_per_epoch, 4)
        self.assertEqual(bundle.counts["train_cnn_dm"], 3)
        self.assertEqual(bundle.counts["train_kptime"], 3)
        self.assertEqual(bundle.counts["validation_cnn_dm"], 2)
        self.assertEqual(bundle.counts["test_kptime"], 2)

    def test_reduced_sizes_are_rejected_outside_smoke_mode(self):
        config = TrainingDataConfig(
            Path("cnn"),
            Path("kpt"),
            max_train_samples_per_dataset=500,
        )
        with self.assertRaisesRegex(ValueError, "test-only"):
            config.normalized()

    def test_incompatible_manifest_tokenizers_are_rejected(self):
        config = TrainingDataConfig(Path("cnn"), Path("kpt"))
        cnn = make_dataset_dict("cnn")
        kpt = make_dataset_dict("kpt")
        manifests = {
            "cnn_dm": make_manifest("cnn_dm", "tokenizer-a", cnn),
            "kptime": make_manifest("kptimes", "tokenizer-b", kpt),
        }
        with self.assertRaisesRegex(ValueError, "incompatible tokenizers"):
            build_training_data_bundle(
                cnn,
                kpt,
                config,
                source_manifests=manifests,
            )

    def test_full_validation_checks_rows_after_the_old_1000_row_cap(self):
        split = make_split(1_001, "late-error")
        split.rows[-1]["loss_weights"] = [0.0, 1.0]
        with self.assertRaisesRegex(ValueError, "row=1000"):
            validate_trainer_split(split, dataset_name="cnn_dm/train")

    def test_token_types_vocab_range_and_sequence_limit_are_checked(self):
        split = make_split(1, "contract")
        split.rows[0]["input_ids"] = [1, 2, 99]
        split.rows[0]["labels"] = [-100, 2, 99]
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            validate_trainer_split(
                split,
                dataset_name="cnn_dm/train",
                vocab_size=10,
                max_sequence_length=8,
            )

        split = make_split(1, "contract")
        with self.assertRaisesRegex(ValueError, "exceeding model limit"):
            validate_trainer_split(
                split,
                dataset_name="cnn_dm/train",
                vocab_size=100,
                max_sequence_length=2,
            )

    def test_manifest_rows_and_dataset_identity_are_checked(self):
        cnn = make_dataset_dict("cnn")
        kpt = make_dataset_dict("kpt")
        manifest = make_manifest("wrong_dataset", "model", cnn)
        with self.assertRaisesRegex(ValueError, "expected 'cnn_dm'"):
            build_training_data_bundle(
                cnn,
                kpt,
                TrainingDataConfig(Path("cnn"), Path("kpt")),
                source_manifests={"cnn_dm": manifest, "kptime": None},
            )

        manifest = make_manifest("cnn_dm", "model", cnn)
        manifest["splits"]["train"]["rows"] += 1
        with self.assertRaisesRegex(ValueError, "manifest row count"):
            build_training_data_bundle(
                cnn,
                kpt,
                TrainingDataConfig(Path("cnn"), Path("kpt")),
                source_manifests={"cnn_dm": manifest, "kptime": None},
            )

    def test_manifest_tokenizer_must_match_training_model(self):
        with self.assertRaisesRegex(ValueError, "prepared with tokenizer"):
            validate_manifest_tokenizer_for_training(
                {
                    "cnn_dm": {"tokenizer_name": "org/model-a"},
                    "kptime": {"tokenizer_name": "org/model-a"},
                },
                model_name_or_path="org/model-b",
            )
        validate_manifest_tokenizer_for_training(
            {"cnn_dm": {"tokenizer_name": "Qwen/Qwen2.5-3B-Instruct"}},
            model_name_or_path="/models/Qwen2.5-3B-Instruct",
        )

    def test_selected_bundle_is_checked_against_actual_tokenizer(self):
        class FakeTokenizer:
            model_max_length = 8

            def __len__(self):
                return 10

        cnn = make_dataset_dict("cnn", train=2, validation=2, test=2)
        kpt = make_dataset_dict("kpt", train=2, validation=2, test=2)
        cnn["train"].rows[0]["input_ids"][-1] = 99
        cnn["train"].rows[0]["labels"][-1] = 99
        bundle = build_training_data_bundle(
            cnn,
            kpt,
            TrainingDataConfig(Path("cnn"), Path("kpt")),
        )
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            validate_bundle_tokenizer_compatibility(bundle, FakeTokenizer())

    def test_model_config_limit_is_checked_even_if_tokenizer_limit_is_larger(self):
        class FakeTokenizer:
            model_max_length = 128

            def __len__(self):
                return 100

        class FakeModelConfig:
            max_position_embeddings = 2

        cnn = make_dataset_dict("cnn", train=2, validation=2, test=2)
        kpt = make_dataset_dict("kpt", train=2, validation=2, test=2)
        bundle = build_training_data_bundle(
            cnn,
            kpt,
            TrainingDataConfig(Path("cnn"), Path("kpt")),
        )
        with self.assertRaisesRegex(ValueError, "exceeding model limit 2"):
            validate_bundle_tokenizer_compatibility(
                bundle,
                FakeTokenizer(),
                model_config=FakeModelConfig(),
            )

    def test_loaded_tokenizer_contract_detects_same_name_wrong_vocabulary(self):
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

        cnn = make_dataset_dict("cnn", train=1, validation=1, test=1)
        kpt = make_dataset_dict("kpt", train=1, validation=1, test=1)
        manifest = make_manifest("cnn_dm", "same-name", cnn)
        manifest["tokenizer_contract"] = {
            "vocab_size": 3,
            "vocab_sha256": "wrong-hash",
            "chat_template_sha256": None,
            "bos_token_id": None,
            "eos_token_id": 2,
            "unk_token_id": None,
        }
        bundle = build_training_data_bundle(
            cnn,
            kpt,
            TrainingDataConfig(Path("cnn"), Path("kpt")),
            source_manifests={"cnn_dm": manifest, "kptime": None},
        )
        with self.assertRaisesRegex(ValueError, "tokenizer contract"):
            validate_bundle_tokenizer_compatibility(bundle, FakeTokenizer())


if __name__ == "__main__":
    unittest.main()
