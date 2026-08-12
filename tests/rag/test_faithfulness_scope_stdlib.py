from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from briefline.commands.faithfulness import resolve_source_ids


class FaithfulnessScopeTests(unittest.TestCase):
    def test_latest_run_manifest_limits_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "last_run.json"
            manifest.write_text(
                json.dumps({"inserted_source_ids": ["new-1", "new-2", "new-1"]}),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_source_ids(source_ids_file=manifest, all_eligible=False),
                ("new-1", "new-2"),
            )

    def test_empty_current_run_is_an_explicit_empty_scope(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "last_run.json"
            manifest.write_text(
                json.dumps({"inserted_source_ids": []}),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_source_ids(source_ids_file=manifest, all_eligible=False),
                (),
            )

    def test_all_eligible_requires_explicit_flag(self) -> None:
        self.assertIsNone(
            resolve_source_ids(source_ids_file=None, all_eligible=True)
        )


if __name__ == "__main__":
    unittest.main()
