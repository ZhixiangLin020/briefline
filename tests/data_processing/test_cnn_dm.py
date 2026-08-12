import ast
import hashlib
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from data_processing import cnn_dm
from data_processing.config import ClusterConfig


def _ast_hash(obj):
    node = ast.parse(inspect.getsource(obj)).body[0]
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode("utf-8")
    ).hexdigest()[:16]


class CNNDataTests(unittest.TestCase):
    def test_clean_highlights_matches_experiment(self):
        raw = "  VIDEO  NEW: First line.\n\nSecond   line. "
        self.assertEqual(cnn_dm.clean_highlights(raw), "First line. Second line.")

    def test_article_slice_uses_original_one_indexed_bounds(self):
        words = " ".join(f"w{i}" for i in range(1, 11))
        self.assertEqual(cnn_dm._slice_words_1idx(words, 3, 6), "w3 w4 w5 w6")

    def test_export_accepts_membership_for_a_limited_processing_view(self):
        class FakeDataset:
            def __init__(self, rows):
                self.rows = list(rows)

            def __len__(self):
                return len(self.rows)

            @property
            def column_names(self):
                return list(self.rows[0]) if self.rows else []

            def select(self, indices):
                return FakeDataset([self.rows[i] for i in indices])

            def remove_columns(self, columns):
                return FakeDataset(
                    [
                        {key: value for key, value in row.items() if key not in columns}
                        for row in self.rows
                    ]
                )

            def add_column(self, name, values):
                return FakeDataset(
                    [
                        {**row, name: value}
                        for row, value in zip(self.rows, values)
                    ]
                )

            def save_to_disk(self, _path):
                return None

        class FakeDatasetDict(dict):
            pass

        full_split = FakeDataset(
            [
                {"article": f"article-{i}", "highlights": f"highlight-{i}", "id": str(i)}
                for i in range(5)
            ]
        )
        processed_membership = np.array([7, 8, 9], dtype=np.int64)
        processed_to_raw = np.array([1, 3, 4], dtype=np.int64)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cnn_dm, "Dataset", FakeDataset), patch.object(
                cnn_dm, "DatasetDict", FakeDatasetDict
            ):
                selected, meta = cnn_dm.build_picked_cnn_dm_raw_dataset(
                    picked_idx=[2, 0],
                    membership=processed_membership,
                    source_indices=processed_to_raw,
                    ds_dict=FakeDatasetDict({"train": full_split}),
                    out_dir=str(Path(tmp) / "selected"),
                    reload_if_exists=False,
                )

        self.assertEqual([row["orig_idx"] for row in selected.rows], [4, 1])
        self.assertEqual([row["cluster_id"] for row in selected.rows], [9, 7])
        self.assertEqual(meta["orig_len"], 5)
        self.assertEqual(meta["processed_len"], 3)

    def test_cnn_clustering_forwards_config_seed_to_leiden(self):
        cfg = ClusterConfig(seed=31)
        emb = np.ones((2, 2), dtype=np.float32)
        scores = np.ones((2, 2), dtype=np.float32)
        nbrs = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
        pairs = np.asarray([[0, 1]], dtype=np.int32)
        weights = np.asarray([0.9], dtype=np.float32)
        membership = np.asarray([0, 0], dtype=np.int32)

        with patch.object(
            cnn_dm,
            "load_and_clean_cnn_dm_highlights",
            return_value=(["a", "b"], ["text a", "text b"], np.asarray([0, 1])),
        ), patch.object(
            cnn_dm,
            "embed_texts_e5",
            return_value=emb,
        ), patch.object(
            cnn_dm,
            "faiss_hnsw_knn",
            return_value=(scores, nbrs),
        ), patch.object(
            cnn_dm,
            "build_graph_edges",
            return_value=(pairs, weights, 0.9),
        ), patch.object(
            cnn_dm,
            "leiden_cluster",
            return_value=membership,
        ) as leiden, patch.object(
            cnn_dm,
            "attach_singletons_A",
            side_effect=lambda value, *_args: value,
        ), patch.object(
            cnn_dm,
            "attach_singletons_B",
            side_effect=lambda value, *_args: value,
        ), patch.object(
            cnn_dm,
            "print_cluster_stats",
            return_value={},
        ), patch.object(
            cnn_dm,
            "summarize_cluster_threshold_distribution",
            return_value={},
        ):
            cnn_dm.run_cnn_dm_highlights_clustering(
                cfg=cfg,
                save_emb_immediately=False,
                keep_emb=True,
            )

        self.assertEqual(leiden.call_args.kwargs["seed"], 31)

    def test_key_preparation_functions_match_original_ast(self):
        expected = {
            "clean_highlights": "706d2da9c4b8654c",
            "_build_highlight_token_roles_and_weights": "08f8edec084f2e1b",
            "build_cnn_dm_highlight_trainer_dataset_v4": "ca35087b93b1cf2b",
        }
        for name, expected_hash in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_ast_hash(getattr(cnn_dm, name)), expected_hash)


if __name__ == "__main__":
    unittest.main()
