import unittest

from training.losses import (
    resolve_loss_normalization,
    resolve_progress_from_trainer_state,
    resolve_schedule_point_progress,
    resolve_scheduled_prompt_loss_weight,
)


class LossScheduleTests(unittest.TestCase):
    def test_original_sample_mean_name_is_preserved(self):
        self.assertEqual(resolve_loss_normalization("sample_mean"), "sample_mean")
        self.assertEqual(resolve_loss_normalization("sequence_mean"), "sample_mean")

    def test_progress_is_based_on_global_step_over_max_steps(self):
        self.assertEqual(resolve_progress_from_trainer_state(global_step=25, max_steps=100), 0.25)
        self.assertEqual(resolve_progress_from_trainer_state(global_step=1, max_steps=0), 0.0)

    def test_epoch_schedule_points_convert_to_total_progress(self):
        self.assertEqual(
            resolve_schedule_point_progress(
                point_type="epoch", point_value=3, num_train_epochs=6
            ),
            0.5,
        )

    def test_original_dynamic_prompt_weight_boundaries_and_bins(self):
        kwargs = {
            "use_dynamic": True,
            "fixed_prompt_loss_weight": 0.0,
            "start_weight": 0.04,
            "end_weight": 0.01,
            "start_progress": 0.0,
            "anchor_progress": 0.8,
            "bin_ratio": 0.01,
        }
        self.assertEqual(
            resolve_scheduled_prompt_loss_weight(current_progress=0.0, **kwargs),
            0.04,
        )
        self.assertEqual(
            resolve_scheduled_prompt_loss_weight(current_progress=0.8, **kwargs),
            0.01,
        )
        expected_midpoint = 0.04 + (0.01 - 0.04) * (50 / 99)
        self.assertAlmostEqual(
            resolve_scheduled_prompt_loss_weight(current_progress=0.4, **kwargs),
            expected_midpoint,
            places=12,
        )

    def test_fixed_prompt_weight_path_is_unchanged(self):
        self.assertEqual(
            resolve_scheduled_prompt_loss_weight(
                use_dynamic=False,
                fixed_prompt_loss_weight=0.123,
                start_weight=0.04,
                end_weight=0.01,
                start_progress=0.0,
                anchor_progress=0.8,
                current_progress=0.5,
                bin_ratio=0.01,
            ),
            0.123,
        )


if __name__ == "__main__":
    unittest.main()
