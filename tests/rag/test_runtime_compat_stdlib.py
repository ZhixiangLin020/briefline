from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from rag.http_logging import redact_query_parameter
from rag.orchestrator import _tokenizer_compatibility_kwargs


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RAGRuntimeCompatibilityTests(unittest.TestCase):
    def test_v5_extra_special_tokens_are_translated_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tokenizer_config.json"
            payload = {
                "additional_special_tokens": ["<existing>"],
                "extra_special_tokens": ["<new>", "<existing>"],
            }
            original = json.dumps(payload)
            config_path.write_text(original, encoding="utf-8")

            self.assertEqual(
                _tokenizer_compatibility_kwargs(Path(directory)),
                {
                    "extra_special_tokens": {},
                    "additional_special_tokens": ["<existing>", "<new>"],
                },
            )
            self.assertEqual(config_path.read_text(encoding="utf-8"), original)

    def test_v4_extra_special_tokens_mapping_is_left_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "tokenizer_config.json"
            config_path.write_text(
                json.dumps({"extra_special_tokens": {"image_token": "<image>"}}),
                encoding="utf-8",
            )
            self.assertEqual(_tokenizer_compatibility_kwargs(Path(directory)), {})

    def test_guardian_api_key_is_redacted_from_request_log_url(self) -> None:
        secret = "guardian-secret-value"
        redacted = redact_query_parameter(
            "https://content.guardianapis.com/search"
            f"?api-key={secret}&page=1&section=world",
            "api-key",
        )
        self.assertNotIn(secret, redacted)
        self.assertEqual(parse_qs(urlsplit(redacted).query)["api-key"], ["***"])
        self.assertEqual(parse_qs(urlsplit(redacted).query)["page"], ["1"])

        source = (PROJECT_ROOT / "rag/guardian_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("redact_query_parameter(response.url, \"api-key\")", source)


if __name__ == "__main__":
    unittest.main()
