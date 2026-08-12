from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from rag.model_cache import build_merged_model_identity, cache_key


class ModelCacheIdentityTests(unittest.TestCase):
    def _make_adapter(self, root: Path, name: str, content: bytes) -> Path:
        adapter = root / name
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            '{"peft_type": "ADALORA"}',
            encoding="utf-8",
        )
        (adapter / "adapter_model.safetensors").write_bytes(content)
        return adapter

    def test_different_adapter_paths_have_different_cache_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_a = self._make_adapter(root, "adapter-a", b"same")
            adapter_b = self._make_adapter(root, "adapter-b", b"same")

            identity_a = build_merged_model_identity(
                original_base_model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=str(adapter_a),
                tokenizer_path=str(adapter_a),
                target_vocab_size=151936,
            )
            identity_b = build_merged_model_identity(
                original_base_model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=str(adapter_b),
                tokenizer_path=str(adapter_b),
                target_vocab_size=151936,
            )

            self.assertNotEqual(
                cache_key(identity_a, prefix="adapter"),
                cache_key(identity_b, prefix="adapter"),
            )

    def test_adapter_file_change_invalidates_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = self._make_adapter(root, "adapter", b"first")
            identity_before = build_merged_model_identity(
                original_base_model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=str(adapter),
                tokenizer_path=str(adapter),
                target_vocab_size=151936,
            )

            time.sleep(0.001)
            (adapter / "adapter_model.safetensors").write_bytes(b"second-version")
            identity_after = build_merged_model_identity(
                original_base_model_path="Qwen/Qwen2.5-3B-Instruct",
                adapter_path=str(adapter),
                tokenizer_path=str(adapter),
                target_vocab_size=151936,
            )

            self.assertNotEqual(
                cache_key(identity_before, prefix="adapter"),
                cache_key(identity_after, prefix="adapter"),
            )


if __name__ == "__main__":
    unittest.main()
