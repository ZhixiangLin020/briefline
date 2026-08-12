"""Python-only configuration for the data pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ClusterConfig:
    """Clustering defaults preserved from the original experiments."""

    split: str = "train"
    take_n: int = 287_113

    model_name: str = "intfloat/e5-base-v2"
    device: str = "cuda"
    batch_size: int = 512
    max_seq_length: int = 512
    prefix: str = "passage: "

    top_k: int = 8_192
    hnsw_m: int = 64
    ef_construction: int = 512
    ef_search: int = 16_384

    max_edges_per_node: int = 1_024
    q_sim: int = 95
    use_mutual: bool = True

    seed: int = 42
    resolution_list: Tuple[float, ...] = (1.8, 2.0, 2.2, 2.4)
    max_ok_cluster: int = 10_000

    attach_q: int = 97
    attach_neigh_k: int = 512

    pair_k: int = 2_048
    pair_q: int = 95


@dataclass
class PipelineConfig:
    """Runtime paths and choices shared by both dataset pipelines."""

    dataset: str
    stage: str
    output_dir: Path
    cache_dir: Optional[Path] = None
    limit: Optional[int] = None
    seed: int = 42
    device: str = "cuda"
    num_proc: int = 8
    batch_size: int = 512
    tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct"
    task_mode: str = "both"
    force_rebuild: bool = False

    def normalized(self) -> "PipelineConfig":
        dataset = self.dataset.strip().lower().replace("-", "_")
        if dataset in {"cnn", "cnn_dailymail", "cnn/dailymail"}:
            dataset = "cnn_dm"
        if dataset not in {"cnn_dm", "kptimes"}:
            raise ValueError("dataset must be one of: cnn_dm, kptimes")

        stage = self.stage.strip().lower()
        if stage not in {"select", "prepare", "all", "validate"}:
            raise ValueError("stage must be one of: select, prepare, all, validate")

        task_mode = self.task_mode.strip().lower()
        if task_mode not in {"category", "keywords", "both"}:
            raise ValueError("task_mode must be one of: category, keywords, both")

        limit = None if self.limit is None else max(1, int(self.limit))
        cache_dir = self.cache_dir or (self.output_dir / "cache")
        return replace(
            self,
            dataset=dataset,
            stage=stage,
            task_mode=task_mode,
            output_dir=Path(self.output_dir),
            cache_dir=Path(cache_dir),
            limit=limit,
            num_proc=max(1, int(self.num_proc)),
            batch_size=max(1, int(self.batch_size)),
        )


def fit_cluster_config_to_size(cfg: ClusterConfig, n: int) -> ClusterConfig:
    """Make the original clustering configuration legal for a small smoke test.

    Full experimental runs are unchanged because every value remains identical
    whenever ``n`` is larger than the configured neighbourhood sizes.
    """

    n = max(0, int(n))
    if n <= 1:
        return replace(
            cfg,
            top_k=1,
            max_edges_per_node=1,
            attach_neigh_k=1,
            pair_k=1,
            ef_search=1,
            ef_construction=2,
            use_mutual=False,
        )

    top_k = min(int(cfg.top_k), n)
    return replace(
        cfg,
        top_k=top_k,
        max_edges_per_node=min(int(cfg.max_edges_per_node), max(1, top_k - 1)),
        attach_neigh_k=min(int(cfg.attach_neigh_k), max(1, top_k - 1)),
        pair_k=min(int(cfg.pair_k), max(1, top_k - 1)),
        ef_search=max(top_k, min(int(cfg.ef_search), max(top_k, 4 * top_k))),
        ef_construction=max(2, min(int(cfg.ef_construction), max(2, n - 1))),
        use_mutual=bool(cfg.use_mutual and n > 2),
    )
