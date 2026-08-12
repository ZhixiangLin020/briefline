import ast
import hashlib
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from data_processing import kptimes
from data_processing.config import ClusterConfig


def _ast_hash(obj):
    node = ast.parse(inspect.getsource(obj)).body[0]
    return hashlib.sha256(
        ast.dump(node, include_attributes=False).encode("utf-8")
    ).hexdigest()[:16]


class _PrefixDecodeTokenizer:
    """Minimal tokenizer stub that reproduces a split UTF-8 character."""

    _decoded_prefixes = {
        (1,): "categories",
        (1, 2): "categories:",
        (1, 2, 3): "categories: caf",
        (1, 2, 3, 4): "categories: caf�",
        (1, 2, 3, 4, 5): "categories: café",
        (1, 2, 3, 4, 5, 6): "categories: café<end>",
    }

    def decode(self, ids, **_kwargs):
        return self._decoded_prefixes[tuple(ids)]


class KPTimesDataTests(unittest.TestCase):
    def test_string_normalization_preserves_original_behavior(self):
        self.assertEqual(kptimes._stringify_text([" alpha ", "beta"]), "alpha, beta")
        self.assertEqual(kptimes._normalize_categories([" politics ", "", None]), ["politics", "None"])

    def test_kptimes_spans_keep_weight_fields(self):
        specs = [
            {"role": "prefix", "group_key": "prefix", "budget": 0.2, "text": "categories: "},
            {"role": "label", "group_key": "label_0", "budget": 1.2, "text": "politics"},
        ]
        spans = kptimes._build_char_spans_from_specs(specs)
        self.assertEqual(spans[0]["group_key"], "prefix")
        self.assertEqual(spans[1]["budget"], 1.2)
        self.assertEqual(spans[1]["start"], len("categories: "))

    def test_unicode_byte_fragments_use_stable_final_decode(self):
        tokenizer = _PrefixDecodeTokenizer()

        text, segments = kptimes._decode_token_segments_from_ids(
            tokenizer,
            [1, 2, 3, 4, 5, 6],
        )

        self.assertEqual(text, "categories: café<end>")
        self.assertEqual(segments, ["categories", ":", " caf", "", "é", "<end>"])

    def test_unicode_byte_fragments_receive_the_content_role_and_weight(self):
        tokenizer = _PrefixDecodeTokenizer()
        prompt_ids = [99]
        answer_ids = [1, 2, 3, 4, 5, 6]
        labels = [-100, 1, 2, 3, 4, 5, 6]

        roles, weights = kptimes._build_full_token_roles_and_weights(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            labels=labels,
            categories=["café"],
            answer_plain="categories: café",
        )

        self.assertEqual(
            roles,
            [
                "prompt",
                "prefix",
                "prefix",
                "label_1",
                "label_1",
                "label_1",
                "terminal_active_only_label_first",
            ],
        )
        expected_label_weight = 1.3 / (3 ** 0.5)
        self.assertEqual(weights[:3], [0.0, 0.2, 0.2])
        for weight in weights[3:6]:
            self.assertAlmostEqual(weight, expected_label_weight)
        self.assertEqual(weights[6], 1.5)

    def test_repr_signature_includes_body_word_limit(self):
        cfg = ClusterConfig()
        first = kptimes._repr_signature(
            cfg,
            include_body=True,
            body_max_words=100,
            body_max_chars=0,
        )
        second = kptimes._repr_signature(
            cfg,
            include_body=True,
            body_max_words=220,
            body_max_chars=0,
        )
        self.assertNotEqual(first, second)

    def test_kptimes_clustering_uses_max_side_and_forwards_seed(self):
        cfg = ClusterConfig()
        emb = np.ones((3, 2), dtype=np.float32)
        scores = np.ones((3, 2), dtype=np.float32)
        nbrs = np.zeros((3, 2), dtype=np.int32)
        pairs = np.asarray([[0, 1], [1, 2]], dtype=np.int32)
        weights = np.asarray([0.9, 0.8], dtype=np.float32)

        with patch.object(
            kptimes,
            "adapt_cluster_cfg_by_n",
            return_value=cfg,
        ) as adapt, patch.object(
            kptimes,
            "faiss_hnsw_knn",
            return_value=(scores, nbrs),
        ), patch.object(
            kptimes,
            "build_graph_edges",
            return_value=(pairs, weights, 0.8),
        ), patch.object(
            kptimes,
            "leiden_cluster",
            return_value=np.asarray([0, 0, 1], dtype=np.int32),
        ) as leiden:
            membership = kptimes.cluster_embeddings_leiden(emb, cfg, seed=29)

        adapt.assert_called_once_with(cfg, 3, prefer_side="max")
        self.assertEqual(leiden.call_args.kwargs["seed"], 29)
        np.testing.assert_array_equal(membership, [0, 0, 1])

    def test_kptimes_formal_defaults_match_summary_parameters(self):
        parameters = inspect.signature(
            kptimes.build_kptimes_dedup_dataset_v2
        ).parameters

        self.assertEqual(parameters["protect_n"].default, 40)
        self.assertEqual(parameters["seed"].default, 42)
        self.assertEqual(parameters["cluster_prefer_side"].default, "max")
        self.assertIs(parameters["include_body"].default, True)
        self.assertEqual(parameters["body_max_words"].default, 220)
        self.assertEqual(parameters["body_max_chars"].default, 0)
        self.assertEqual(parameters["sample_growth"].default, "sqrt")
        self.assertEqual(parameters["sample_tau"].default, 1.0)
        self.assertEqual(parameters["sample_cap"].default, 10_000)
        self.assertEqual(parameters["gap_side"].default, "right")
        self.assertEqual(parameters["log_base"].default, 4.0)

    def test_old_cluster_cache_without_prefer_side_is_rejected(self):
        expected = {
            "version": "kptimes_cluster_cache_v1",
            "prefer_side": "max",
        }
        old_cache = {"version": "kptimes_cluster_cache_v1"}

        with self.assertRaisesRegex(ValueError, "prefer_side"):
            kptimes._validate_meta(expected, old_cache, strict=True, who="CLUSTER")

    def test_locked_alignment_and_dedup_functions_match_expected_ast(self):
        expected = {
            "_decode_token_segments_from_ids": "39e58c688829abce",
            "_build_char_spans_from_specs": "1cd41a936463eaf0",
            "build_kptimes_dedup_dataset_v2": "f3b68775d1c04a30",
        }
        for name, expected_hash in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_ast_hash(getattr(kptimes, name)), expected_hash)


if __name__ == "__main__":
    unittest.main()
