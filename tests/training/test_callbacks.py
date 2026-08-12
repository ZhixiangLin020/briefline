import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


class DummyTrainerCallback:
    pass


class DummyState:
    is_world_process_zero = True
    epoch = 1.0
    log_history = []

    def __init__(self, step):
        self.global_step = step


class DummyModel:
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")


class CallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fake = types.ModuleType("transformers")
        fake.TrainerCallback = DummyTrainerCallback
        cls.previous = sys.modules.get("transformers")
        sys.modules["transformers"] = fake
        sys.modules.pop("training.callbacks", None)
        cls.module = importlib.import_module("training.callbacks")

    @classmethod
    def tearDownClass(cls):
        sys.modules.pop("training.callbacks", None)
        if cls.previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = cls.previous

    def test_save_best_peft_keeps_highest_top_k(self):
        with tempfile.TemporaryDirectory() as temp:
            callback = self.module.SaveBestPeftCallback(
                temp, metric_name="auto", greater_is_better=True, top_k=2
            )
            control = object()
            for step, score in [(1, 0.5), (2, 0.7), (3, 0.6)]:
                callback.on_evaluate(
                    None,
                    DummyState(step),
                    control,
                    metrics={"eval_combo_mover_score": score},
                    model=DummyModel(),
                )
            self.assertEqual([x["step"] for x in callback.best_records], [2, 3])
            payload = json.loads(
                (Path(temp) / "best_k_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual([x["step"] for x in payload["best_records"]], [2, 3])
            self.assertFalse(any("step-1-" in p.name for p in Path(temp).iterdir()))

    def test_resume_restores_previously_saved_top_k_history(self):
        with tempfile.TemporaryDirectory() as temp:
            first = self.module.SaveBestPeftCallback(
                temp,
                metric_name="auto",
                greater_is_better=True,
                top_k=2,
            )
            control = object()
            first.on_evaluate(
                None,
                DummyState(100),
                control,
                metrics={"eval_combo_mover_score": 0.9},
                model=DummyModel(),
            )

            resumed = self.module.SaveBestPeftCallback(
                temp,
                metric_name="auto",
                greater_is_better=True,
                top_k=2,
                resume_existing=True,
            )
            self.assertEqual([x["step"] for x in resumed.best_records], [100])
            resumed.on_evaluate(
                None,
                DummyState(200),
                control,
                metrics={"eval_combo_mover_score": 0.5},
                model=DummyModel(),
            )

            payload = json.loads(
                (Path(temp) / "best_k_metrics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [x["step"] for x in payload["best_records"]],
                [100, 200],
            )

    def test_resume_rejects_missing_historical_adapter(self):
        with tempfile.TemporaryDirectory() as temp:
            payload = {
                "metric_name": "auto",
                "greater_is_better": True,
                "top_k": 4,
                "best_records": [
                    {
                        "rank": 1,
                        "metric": 0.9,
                        "metric_name": "eval_combo_mover_score",
                        "step": 100,
                        "path": str(Path(temp) / "missing-adapter"),
                    }
                ],
            }
            (Path(temp) / "best_k_metrics.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                self.module.SaveBestPeftCallback(
                    temp,
                    metric_name="auto",
                    greater_is_better=True,
                    top_k=4,
                    resume_existing=True,
                )

    def test_resume_rejects_missing_top_k_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(FileNotFoundError, "manifest is missing"):
                self.module.SaveBestPeftCallback(
                    temp,
                    metric_name="auto",
                    greater_is_better=True,
                    top_k=4,
                    resume_existing=True,
                )


if __name__ == "__main__":
    unittest.main()
