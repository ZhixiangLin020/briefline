import ast
import hashlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data_processing import core
from data_processing.config import ClusterConfig, fit_cluster_config_to_size


def _ast_hash(obj):
    node = ast.parse(inspect.getsource(obj)).body[0]
    payload = ast.dump(node, include_attributes=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


class CoreRegressionTests(unittest.TestCase):
    def test_extracted_algorithms_match_original_ast(self):
        expected = {
            "faiss_hnsw_knn": "499c7380a68c8a46",
            "build_graph_edges": "7dbecf8adfc37676",
            "leiden_cluster": "a8ed21a5fda70a06",
            "sample_by_clusters_scheme2": "f0d11e1cee18aa4e",
            "save_all_cluster_artifacts": "3edaac2d53816fc0",
            "_extract_input_ids": "47ff3ef27ec1531f",
            "_safe_token_id": "a174c36187ae3027",
            "_unique_preserve_order": "69a73b73ba895cbc",
            "_resolve_default_terminal_sequences": "5160b59e3d003305",
            "_resolve_training_terminal_sequences": "b2ef3e0e5777fd58",
            "_build_answer_labels_with_terminal_mask": "9162061b3251138d",
        }
        for name, expected_hash in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_ast_hash(getattr(core, name)), expected_hash)

    def test_terminal_mask_modes_preserve_original_values(self):
        prompt = [1, 2]
        answer = [10, 20, 99, 100]
        terminals = [[99], [100]]

        labels, loss_answer_ids = core._build_answer_labels_with_terminal_mask(
            prompt, answer, terminals, terminal_loss_mode="final_only"
        )
        self.assertEqual(labels, [-100, -100, 10, 20, -100, 100])
        self.assertEqual(loss_answer_ids, [10, 20, 100])

        labels, _ = core._build_answer_labels_with_terminal_mask(
            prompt, answer, terminals, terminal_loss_mode="none"
        )
        self.assertEqual(labels, [-100, -100, 10, 20, -100, -100])

        labels, _ = core._build_answer_labels_with_terminal_mask(
            prompt, answer, terminals, terminal_loss_mode="all"
        )
        self.assertEqual(labels, [-100, -100, 10, 20, 99, 100])

    def test_cluster_allocation_is_seed_deterministic(self):
        sizes = np.asarray([0, 1, 4, 9, 16], dtype=np.int64)
        first = core.allocate_k_per_cluster(sizes, num=8, cap=20, seed=17)
        second = core.allocate_k_per_cluster(sizes, num=8, cap=20, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 8)

    def test_leiden_receives_the_requested_seed(self):
        seen = {}

        def find_partition(*_args, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(membership=[0, 0])

        fake_igraph = SimpleNamespace(
            Graph=lambda **_kwargs: SimpleNamespace(es={})
        )
        fake_leidenalg = SimpleNamespace(
            RBConfigurationVertexPartition=object(),
            find_partition=find_partition,
        )
        cfg = ClusterConfig(resolution_list=(1.8,))

        with patch.object(core, "ig", fake_igraph), patch.object(
            core, "leidenalg", fake_leidenalg
        ):
            membership = core.leiden_cluster(
                2,
                np.asarray([[0, 1]], dtype=np.int32),
                np.asarray([0.9], dtype=np.float32),
                cfg,
                seed=17,
            )

        self.assertEqual(seen["seed"], 17)
        np.testing.assert_array_equal(membership, [0, 0])

    def test_leiden_does_not_silently_drop_an_unsupported_seed(self):
        fake_igraph = SimpleNamespace(
            Graph=lambda **_kwargs: SimpleNamespace(es={})
        )
        fake_leidenalg = SimpleNamespace(
            RBConfigurationVertexPartition=object(),
            find_partition=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                TypeError("unexpected keyword argument 'seed'")
            ),
        )

        with patch.object(core, "ig", fake_igraph), patch.object(
            core, "leidenalg", fake_leidenalg
        ), self.assertRaisesRegex(RuntimeError, "without the requested seed"):
            core.leiden_cluster(
                2,
                np.asarray([[0, 1]], dtype=np.int32),
                np.asarray([0.9], dtype=np.float32),
                ClusterConfig(resolution_list=(1.8,)),
                seed=17,
            )

    def test_small_sample_config_does_not_change_full_run_defaults(self):
        cfg = ClusterConfig()
        full = fit_cluster_config_to_size(cfg, 300_000)
        self.assertEqual(full, cfg)

        small = fit_cluster_config_to_size(cfg, 10)
        self.assertEqual(small.top_k, 10)
        self.assertEqual(small.max_edges_per_node, 9)
        self.assertEqual(small.attach_neigh_k, 9)
        self.assertEqual(small.pair_k, 9)


if __name__ == "__main__":
    unittest.main()
