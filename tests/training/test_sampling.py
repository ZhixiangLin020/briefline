import unittest

import numpy as np

from training.sampling import BalancedEpochRatioSampler, TwoSourceTrainerDataset


SCHEDULE = [
    {"cnn_dm": 0.7, "kptime": 0.3},
    {"cnn_dm": 0.6, "kptime": 0.4},
    {"cnn_dm": 0.5, "kptime": 0.5},
]


def make_sampler(samples_per_epoch=10):
    return BalancedEpochRatioSampler(
        source_lengths={"cnn_dm": 7, "kptime": 5},
        source_offsets={"cnn_dm": 0, "kptime": 7},
        source_names=["cnn_dm", "kptime"],
        epoch_ratio_schedule=SCHEDULE,
        samples_per_epoch=samples_per_epoch,
        seed=42,
        shuffle_epoch_indices=True,
        verbose=False,
    )


class SamplingTests(unittest.TestCase):
    def test_ratio_schedule_and_last_ratio_reuse(self):
        sampler = make_sampler()
        expected = [
            {"cnn_dm": 7, "kptime": 3},
            {"cnn_dm": 6, "kptime": 4},
            {"cnn_dm": 5, "kptime": 5},
            {"cnn_dm": 5, "kptime": 5},
        ]
        for epoch_counts in expected:
            indices = list(iter(sampler))
            self.assertEqual(len(indices), 10)
            self.assertEqual(sampler.last_epoch_info["requested_counts"], epoch_counts)

    def test_usage_balancing_never_skips_lower_usage_samples(self):
        sampler = make_sampler(samples_per_epoch=12)
        for _ in range(6):
            list(iter(sampler))
        for counts in sampler.usage_counts.values():
            self.assertLessEqual(int(counts.max()) - int(counts.min()), 1)

    def test_sampler_is_deterministic_for_fixed_seed(self):
        first = make_sampler()
        second = make_sampler()
        for _ in range(4):
            self.assertEqual(list(iter(first)), list(iter(second)))
            for name in ("cnn_dm", "kptime"):
                np.testing.assert_array_equal(first.usage_counts[name], second.usage_counts[name])

    def test_checkpoint_replays_exact_current_epoch_once(self):
        original = make_sampler()
        current = list(iter(original))
        state = original.state_dict()

        restored = make_sampler()
        restored.load_state_dict(state, replay_current_epoch=True)
        self.assertEqual(list(iter(restored)), current)
        self.assertEqual(restored.epoch, original.epoch)
        self.assertEqual(list(iter(restored)), list(iter(original)))

    def test_wrapper_adds_source_without_modifying_row(self):
        cnn_row = {"input_ids": [1]}
        kpt_row = {"input_ids": [2]}
        wrapper = TwoSourceTrainerDataset(
            {"cnn_dm": [cnn_row], "kptime": [kpt_row]},
            ["cnn_dm", "kptime"],
        )
        self.assertEqual(wrapper[0]["source"], "cnn_dm")
        self.assertEqual(wrapper[1]["source"], "kptime")
        self.assertNotIn("source", cnn_row)
        self.assertNotIn("source", kpt_row)


if __name__ == "__main__":
    unittest.main()
