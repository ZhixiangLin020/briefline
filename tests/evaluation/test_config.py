import unittest
from pathlib import Path

from evaluation.config import EvaluationRunConfig
from evaluation.selection import select_best_finetuned_by_validation


def make_config(smoke=False, sample_n=None):
    return EvaluationRunConfig(
        cnn_dm_dataset=Path("cnn"),
        kptimes_dataset=Path("kpt"),
        base_model_path=Path("base"),
        roberta_path=Path("roberta"),
        output_dir=Path("out"),
        model_artifact_paths={"candidate": Path("adapter")},
        smoke_test=smoke,
        max_samples_per_split_per_dataset=sample_n,
    )


class EvaluationConfigTests(unittest.TestCase):
    def test_full_evaluation_rejects_non_original_seed(self):
        cfg = make_config()
        cfg = EvaluationRunConfig(**{**cfg.__dict__, "seed": 7})
        with self.assertRaisesRegex(ValueError, "freezes seed=42"):
            cfg.normalized()

    def test_smoke_evaluation_can_use_an_explicit_test_seed(self):
        cfg = make_config(smoke=True)
        cfg = EvaluationRunConfig(**{**cfg.__dict__, "seed": 7}).normalized()
        self.assertEqual(cfg.seed, 7)

    def test_decoding_protocol_is_frozen(self):
        self.assertEqual(
            EvaluationRunConfig.decoding_config(),
            {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": 1.0,
                "repetition_penalty": 1.02,
                "answer_prefix_len": 2,
                "request_chunk_size": 1024,
                "encoder_batch_size": 512,
            },
        )

    def test_smoke_subset_is_explicit_and_defaults_to_50(self):
        cfg = make_config(smoke=True).normalized()
        self.assertEqual(cfg.max_samples_per_split_per_dataset, 50)
        with self.assertRaisesRegex(ValueError, "test-only"):
            make_config(smoke=False, sample_n=50).normalized()

    def test_base_is_always_present(self):
        cfg = make_config().normalized()
        self.assertEqual(cfg.model_artifact_paths["base"], Path("base"))

    def test_selection_uses_validation_not_test(self):
        rows = [
            {"model_alias": "base", "validation_combo_mover_score": 0.7, "test_combo_mover_score": 0.7},
            {"model_alias": "step_a", "validation_combo_mover_score": 0.82, "test_combo_mover_score": 0.90},
            {"model_alias": "step_b", "validation_combo_mover_score": 0.83, "test_combo_mover_score": 0.80},
        ]
        selected = select_best_finetuned_by_validation(rows)
        self.assertEqual(selected["model_alias"], "step_b")


if __name__ == "__main__":
    unittest.main()
