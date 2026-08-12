import importlib
import sys
import types
import unittest


class TrainingModuleImportTests(unittest.TestCase):
    def test_warmup_has_no_notebook_import_side_effects(self):
        fake_torch = types.ModuleType("torch")
        previous = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        sys.modules.pop("training.warmup", None)
        try:
            module = importlib.import_module("training.warmup")
            self.assertTrue(callable(module.trainer_warmup))
        finally:
            sys.modules.pop("training.warmup", None)
            if previous is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous


if __name__ == "__main__":
    unittest.main()
