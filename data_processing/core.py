
"""Shared algorithms for data selection and trainer-dataset construction.

The implementations in this module are extracted from the original experiments.
They intentionally retain the original numerical behavior.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import ClusterConfig

try:
    import torch
except ImportError:  # optional until a training pipeline is executed
    torch = None

try:
    import faiss
except ImportError:  # optional until clustering is executed
    faiss = None

try:
    import igraph as ig
    import leidenalg
except ImportError:  # optional until clustering is executed
    ig = None
    leidenalg = None



def faiss_hnsw_knn(
    emb: np.ndarray,
    cfg: "ClusterConfig",
    scores_dtype=np.float16,   # Use float16 by default to reduce memory use.
    nbrs_dtype=np.int32,       # Use int32 by default to reduce memory use.
    log_every_sec: float = 30.0,  # Interval for progress messages.
    verbose: bool = True,
):


    # --- optional: RSS monitor ---
    try:
        import psutil
        _proc = psutil.Process(os.getpid())
        def _rss_gib():
            return _proc.memory_info().rss / 1024**3
    except Exception:
        _proc = None
        def _rss_gib():
            return float("nan")

    def _fmt_s(x: float) -> str:
        if x != x:  # nan
            return "NA"
        if x < 60:
            return f"{x:.1f}s"
        if x < 3600:
            return f"{x/60:.1f}m"
        return f"{x/3600:.2f}h"

    def _log(msg: str):
        if verbose:
            ts = time.strftime("%H:%M:%S")
            rss = _rss_gib()
            rss_str = f"{rss:.2f}GiB" if rss == rss else "NA"
            print(f"[{ts}] {msg} | rss={rss_str}")

    t_all = time.perf_counter()

    # --- basic info ---
    N, D = emb.shape
    _log(f"[HNSW] start: N={N:,} D={D} top_k={cfg.top_k} m={cfg.hnsw_m} "
         f"efC={cfg.ef_construction} efS={cfg.ef_search}")

    # --- build index ---
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(D, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(cfg.ef_construction)
    index.hnsw.efSearch = int(cfg.ef_search)
    _log(f"[HNSW] index init done ({_fmt_s(time.perf_counter()-t0)})")

    # --- add vectors ---
    t1 = time.perf_counter()
    _log("[HNSW] adding vectors...")
    index.add(emb)
    _log(f"[HNSW] add done ({_fmt_s(time.perf_counter()-t1)})")

    # --- search (this is usually the slowest) ---
    t2 = time.perf_counter()
    _log("[HNSW] searching... (this can take a while)")

    last_beat = time.perf_counter()
    scores, nbrs = index.search(emb, int(cfg.top_k))
    _log(f"[HNSW] search done ({_fmt_s(time.perf_counter()-t2)})")

    return scores, nbrs



def _percentile_threshold(x: np.ndarray, q: int, fallback: float) -> float:
    flat = x.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if len(flat) == 0:
        return float(fallback)
    return float(np.percentile(flat, q))



def build_graph_edges(
    nbrs: np.ndarray,
    scores: np.ndarray,
    cfg: ClusterConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return:
      pairs: [E,2] int32 undirected edges (u<v)
      weights: [E] float32 edge weights
      sim_th: used similarity threshold
    """
    N = nbrs.shape[0]
    m = min(cfg.max_edges_per_node, cfg.top_k - 1)

    # remove self at col 0
    nbrs2 = nbrs[:, 1:1 + m]
    sc2   = scores[:, 1:1 + m]

    sim_th = _percentile_threshold(sc2, cfg.q_sim, fallback=0.6)
    print("SIM_TH (percentile):", sim_th)

    mask = (sc2 >= sim_th)
    src = np.repeat(np.arange(N, dtype=np.int32), m)
    dst = nbrs2.reshape(-1).astype(np.int32)
    sim = sc2.reshape(-1).astype(np.float32)
    sel = mask.reshape(-1)

    src = src[sel]
    dst = dst[sel]
    sim = sim[sel]

    # mutual kNN
    if cfg.use_mutual and len(src) > 0:
        src64 = src.astype(np.int64)
        dst64 = dst.astype(np.int64)
        key   = src64 * N + dst64
        rkey  = dst64 * N + src64

        order = np.argsort(key)
        key_s = key[order]
        sim_s = sim[order]

        key_u, idx_u = np.unique(key_s, return_index=True)
        sim_u = sim_s[idx_u]

        rkey_u = np.unique(rkey)  # np.unique sorts its output.
        mutual = np.intersect1d(key_u, rkey_u, assume_unique=True)

        pos = np.searchsorted(key_u, mutual)
        mutual_sim = sim_u[pos]

        src = (mutual // N).astype(np.int32)
        dst = (mutual %  N).astype(np.int32)
        sim = mutual_sim.astype(np.float32)

    # undirected + dedup keep max weight
    a = np.minimum(src, dst).astype(np.int32)
    b = np.maximum(src, dst).astype(np.int32)
    undir_key = a.astype(np.int64) * N + b.astype(np.int64)

    order = np.argsort(undir_key)
    undir_key_s = undir_key[order]
    sim_s = sim[order]

    u_keys, first = np.unique(undir_key_s, return_index=True)
    boundaries = np.r_[first, len(sim_s)]

    max_sims = np.empty(len(first), dtype=np.float32)
    for i in range(len(first)):
        max_sims[i] = sim_s[boundaries[i]:boundaries[i + 1]].max()

    u = (u_keys // N).astype(np.int32)
    v = (u_keys %  N).astype(np.int32)

    pairs = np.stack([u, v], axis=1)
    weights = max_sims
    print("num edges:", len(pairs))
    return pairs, weights, sim_th



def leiden_cluster(
    n_nodes: int,
    pairs: np.ndarray,
    weights: np.ndarray,
    cfg: ClusterConfig,
    seed: Optional[int] = None,   # Optional random seed.
) -> np.ndarray:
    edges_list = list(map(tuple, pairs.tolist()))
    g = ig.Graph(n=n_nodes, edges=edges_list, directed=False)
    g.es["weight"] = weights.tolist()

    best_membership = None
    for res in cfg.resolution_list:
        partition_kwargs = {
            "weights": g.es["weight"],
            "resolution_parameter": float(res),
        }
        if seed is not None:
            partition_kwargs["seed"] = int(seed)

        try:
            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                **partition_kwargs,
            )
        except TypeError as exc:
            if seed is None or "seed" not in str(exc).lower():
                raise
            raise RuntimeError(
                "The installed leidenalg version does not accept a seed. "
                "Install a compatible version instead of running Leiden "
                "without the requested seed."
            ) from exc

        membership = np.array(part.membership, dtype=np.int32)
        sizes = np.bincount(membership)
        print(f"[Leiden] RES={res}  num_comms={len(sizes)}  max_size={int(sizes.max())}")
        best_membership = membership
        if int(sizes.max()) <= cfg.max_ok_cluster:
            break

    return best_membership



def pick_k_by_top_gaps(
    idx_sorted: np.ndarray,
    dist_sorted: np.ndarray,
    k: int,
    gap_side: str = "right",   # "right" maps to i+1; "left" maps to i.
) -> np.ndarray:
    """
    Cluster members already sorted by ascending distance:
      idx_sorted: (n,) global indices
      dist_sorted: (n,) ascending distances to the centroid
    Selection:
      - Always select positions 0 and n-1.
      - Compute gaps g[i] = d[i+1] - d[i].
      - Take the largest m=k-2 gaps and map them to boundary points
        (the right boundary i+1 by default).
      - Remove duplicates and return positions in ascending order.
    """
    idx_sorted = np.asarray(idx_sorted, dtype=np.int64)
    dist_sorted = np.asarray(dist_sorted, dtype=np.float32)
    n = len(idx_sorted)

    if k <= 0 or n == 0:
        return np.array([], dtype=np.int64)
    if k >= n:
        return idx_sorted.copy()
    if k == 1:
        return idx_sorted[:1].copy()

    # Fixed endpoints: nearest to the centroid and farthest outlier.
    pos_set = {0, n - 1}

    # Gap array.
    g = dist_sorted[1:] - dist_sorted[:-1]   # (n-1,)

    # Sort gaps from largest to smallest deterministically.
    gap_order = np.argsort(g, kind="mergesort")[::-1]

    need = k - len(pos_set)
    for gi in gap_order:
        if need <= 0:
            break
        if gap_side == "right":
            pos = int(gi) + 1
        elif gap_side == "left":
            pos = int(gi)
        else:
            raise ValueError("gap_side must be 'right' or 'left'")

        if pos not in pos_set:
            pos_set.add(pos)
            need -= 1

    # A shortfall can occur only for tiny clusters or duplicate endpoints.
    # Fill it with evenly spaced positions deterministically.
    if len(pos_set) < k:
        remain = k - len(pos_set)
        # Even spacing is a deterministic fallback, not an optimality claim.
        for t in range(1, n - 1):
            if t not in pos_set:
                pos_set.add(t)
                remain -= 1
                if remain <= 0:
                    break

    pos_list = sorted(pos_set)
    return idx_sorted[np.array(pos_list, dtype=np.int64)]



def k_from_size_num_none(
    s: int,
    growth: str = "log",     # "log" or "sqrt"
    cap: int = 5,
    log_base: float = 2.0,   # Logarithm base.
    tau: float = 50.0,       # Square-root scale; larger values are more conservative.
) -> int:
    if s <= 0:
        return 0
    if growth == "log":
        # 1 + floor(log_base(s))
        val = 1 + int(math.floor(math.log(max(s, 1), log_base)))
    elif growth == "sqrt":
        # 1 + floor(sqrt(s)/tau)
        val = 1 + int(math.floor(math.sqrt(s) / max(tau, 1e-6)))
    else:
        raise ValueError("growth must be 'log' or 'sqrt'")
    val = max(1, val)
    val = min(cap, val)
    val = min(s, val)
    return val



def allocate_k_per_cluster(
    sizes: np.ndarray,
    num: Optional[int] = None,
    growth: str = "log",
    cap: int = 5,
    log_base: float = 2.0,
    tau: float = 50.0,
    seed: int = 0,
) -> Dict[int, int]:
    sizes = np.asarray(sizes, dtype=np.int64)
    cluster_ids = np.where(sizes > 0)[0].astype(np.int64)
    K = len(cluster_ids)
    if K == 0:
        return {}

    # num=None: choose k(s) independently for each cluster.
    if num is None:
        out = {}
        for cid in cluster_ids:
            out[int(cid)] = k_from_size_num_none(
                int(sizes[cid]),
                growth=growth, cap=cap, log_base=log_base, tau=tau
            )
        return out

    # Fixed total: one base sample per cluster plus weighted extras.
    num = int(num)
    rng = np.random.default_rng(seed)

    # If the budget is smaller than the cluster count, sample num clusters.
    if num <= 0:
        return {}
    if num < K:
        chosen_clusters = rng.choice(cluster_ids, size=num, replace=False)
        return {int(cid): 1 for cid in chosen_clusters}

    # Start with one sample per cluster.
    k = np.ones(K, dtype=np.int64)
    extra_budget = num - K

    # Allocation weights.
    s = sizes[cluster_ids].astype(np.float64)
    if growth == "log":
        w = np.log1p(s) / np.log(log_base)
    elif growth == "sqrt":
        w = np.sqrt(s)
    else:
        raise ValueError("growth must be 'log' or 'sqrt'")
    w = np.maximum(w, 1e-12)
    w = w / w.sum()

    # Floor proportional allocations first.
    raw = extra_budget * w
    add = np.floor(raw).astype(np.int64)
    k += add
    remaining = extra_budget - int(add.sum())

    # Assign the remainder by descending fractional part deterministically.
    frac = raw - np.floor(raw)
    order = np.argsort(-frac, kind="mergesort")
    for idx in order[:remaining]:
        k[idx] += 1

    # Respect both the cap and cluster size.
    k = np.minimum(k, cap)
    k = np.minimum(k, sizes[cluster_ids])

    # A small cap may make the requested total unattainable.
    total = int(k.sum())
    if total < num:
        # Add to clusters that are still below both cap and size.
        room = np.minimum(cap, sizes[cluster_ids]) - k
        if room.sum() > 0:
            # Prioritize by weight deterministically.
            refill_order = np.argsort(-w, kind="mergesort")
            need = num - total
            for idx in refill_order:
                if need <= 0:
                    break
                if room[idx] > 0:
                    take = int(min(room[idx], need))
                    k[idx] += take
                    need -= take
            total = int(k.sum())
        # If a shortfall remains, hard cap/size limits prevent reaching num.
        # Report the shortfall instead of raising an exception.
    elif total > num:
        # If truncation overshoots num, reclaim from lower-weight clusters.
        over = total - num
        back_order = np.argsort(w, kind="mergesort")
        for idx in back_order:
            if over <= 0:
                break
            if k[idx] > 1:
                dec = int(min(k[idx] - 1, over))
                k[idx] -= dec
                over -= dec

    return {int(cid): int(kk) for cid, kk in zip(cluster_ids, k)}



def _build_sorted_indices_from_artifacts_with_dist(
    membership: np.ndarray,
    artifacts_dir: Optional[str] = None,
    arrays_npz_path: Optional[str] = None,
    strict_check_membership: bool = True,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Use the following arrays from arrays.npz:
      - dist_to_centroid
      - rank_in_cluster
    Reconstruct:
      out[cid]["idx_sorted"]  = global indices sorted by distance
      out[cid]["dist_sorted"] = corresponding ascending distances
    """
    if arrays_npz_path is None:
        if artifacts_dir is None:
            raise ValueError("emb=None requires arrays_npz_path or artifacts_dir.")
        arrays_npz_path = os.path.join(artifacts_dir, "arrays.npz")

    z = np.load(arrays_npz_path, allow_pickle=False)

    mem_saved = z["membership"].astype(np.int64)
    dist = z["dist_to_centroid"].astype(np.float32)
    rank = z["rank_in_cluster"].astype(np.int64)

    membership = np.asarray(membership, dtype=np.int64)
    if strict_check_membership:
        if mem_saved.shape != membership.shape or not np.array_equal(mem_saved, membership):
            raise ValueError("membership passed in != membership saved in arrays.npz (cannot guarantee reproducibility).")

    # Sort globally by cluster_id and then by distance-based rank.
    order = np.lexsort((rank, membership))  # primary=membership, secondary=rank
    mem_sorted = membership[order]

    # Split the sorted arrays into cluster segments.
    cuts = np.flatnonzero(mem_sorted[1:] != mem_sorted[:-1]) + 1
    bounds = np.r_[0, cuts, len(order)]

    out: Dict[int, Dict[str, np.ndarray]] = {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        cid = int(mem_sorted[s])
        idx_sorted = order[s:e].astype(np.int64)
        out[cid] = {
            "idx_sorted": idx_sorted,
            "dist_sorted": dist[idx_sorted],  # Keep distances aligned.
        }

    return out



def sample_by_clusters_scheme2(
    emb: Optional[np.ndarray],
    membership: np.ndarray,
    num: Optional[int] = None,
    growth: str = "log",
    cap: int = 5,
    log_base: float = 2.0,
    tau: float = 50.0,
    seed: int = 0,
    artifacts_dir: Optional[str] = None,
    arrays_npz_path: Optional[str] = None,
    gap_side: str = "right",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Scheme 2 allocates k_i per cluster and selects representatives using
    top distance gaps, always including both endpoints.

    emb != None:
      - Compute distance to centroid and sort into idx_sorted/dist_sorted.
    emb == None:
      - Reconstruct idx_sorted/dist_sorted from distance and rank in arrays.npz.
    """
    membership = np.asarray(membership, dtype=np.int64)
    K = int(membership.max()) + 1
    sizes = np.bincount(membership, minlength=K).astype(np.int64)

    k_map = allocate_k_per_cluster(
        sizes,
        num=num,
        growth=growth,
        cap=cap,
        log_base=log_base,
        tau=tau,
        seed=seed,
    )

    if emb is not None:
        # Compute distances and sort them when no artifacts are supplied.
        _, _, dists, _ = compute_centroid_sim_dist(emb, membership.astype(np.int32))

        # Sort globally by (cluster_id, distance), then split into segments.
        order = np.lexsort((dists.astype(np.float32), membership))
        mem_sorted = membership[order]
        dist_sorted_all = dists[order].astype(np.float32)

        cuts = np.flatnonzero(mem_sorted[1:] != mem_sorted[:-1]) + 1
        bounds = np.r_[0, cuts, len(order)]

        picked = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            cid = int(mem_sorted[s])
            k = int(k_map.get(cid, 0))
            if k <= 0:
                continue
            idx_sorted = order[s:e].astype(np.int64)
            dist_sorted = dist_sorted_all[s:e]
            picked.append(pick_k_by_top_gaps(idx_sorted, dist_sorted, k=k, gap_side=gap_side))

        sort_source = "emb"

    else:
        sorted_map = _build_sorted_indices_from_artifacts_with_dist(
            membership,
            artifacts_dir=artifacts_dir,
            arrays_npz_path=arrays_npz_path,
        )
        picked = []
        for cid, pack in sorted_map.items():
            k = int(k_map.get(int(cid), 0))
            if k <= 0:
                continue
            picked.append(
                pick_k_by_top_gaps(pack["idx_sorted"], pack["dist_sorted"], k=k, gap_side=gap_side)
            )
        sort_source = "artifacts"

    picked_idx = np.concatenate(picked) if picked else np.array([], dtype=np.int64)

    report = {
        "num_target": num,
        "num_picked": int(len(picked_idx)),
        "growth": growth,
        "cap": cap,
        "log_base": log_base,
        "tau": tau,
        "seed": seed,
        "sort_source": sort_source,
        "num_clusters_nonempty": int(np.sum(sizes > 0)),
        "num_clusters_sampled": int(sum(1 for v in k_map.values() if v > 0)),
        "gap_side": gap_side,
    }
    return picked_idx, report



def compute_centroid_sim_dist(emb: np.ndarray, membership: np.ndarray):
    emb = np.asarray(emb, dtype=np.float32)
    membership = np.asarray(membership, dtype=np.int32)
    N, D = emb.shape
    K = int(membership.max()) + 1

    sizes = np.bincount(membership, minlength=K).astype(np.int64)

    centroids = np.zeros((K, D), dtype=np.float32)
    # Accumulate each vector into its cluster.
    np.add.at(centroids, membership, emb)

    # Compute means.
    denom = np.maximum(sizes, 1)[:, None].astype(np.float32)
    centroids = centroids / denom

    # Normalize centroids to reduce numerical drift.
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(norms, 1e-12)

    # Similarity between each sample and its cluster centroid.
    sims = (emb * centroids[membership]).sum(axis=1).astype(np.float32)
    dists = (1.0 - sims).astype(np.float32)

    return centroids, sims, dists, sizes



def rank_within_cluster_by_dist(dists: np.ndarray, membership: np.ndarray):
    """
    Assign an in-cluster rank to each sample: 0 is nearest to the centroid,
    and larger values are farther away.
    """
    dists = np.asarray(dists, dtype=np.float32)
    membership = np.asarray(membership, dtype=np.int32)
    N = membership.shape[0]
    rank = np.empty(N, dtype=np.int32)

    # Sort by cluster for segmented processing.
    order = np.argsort(membership, kind="mergesort")
    mem_sorted = membership[order]

    # Find segment boundaries.
    cuts = np.flatnonzero(mem_sorted[1:] != mem_sorted[:-1]) + 1
    bounds = np.r_[0, cuts, N]

    for s, e in zip(bounds[:-1], bounds[1:]):
        idxs = order[s:e]
        sub = np.argsort(dists[idxs], kind="mergesort")  # Smaller means more central.
        rank[idxs[sub]] = np.arange(e - s, dtype=np.int32)

    return rank



def save_all_cluster_artifacts(
    out_dir: str,
    ids,
    texts,
    emb: np.ndarray,
    membership: np.ndarray,
    cfg=None,                       # Store configuration in metadata.
    chunk_size: int = 200_000,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Basic validation.
    ids = list(ids)
    texts = list(texts)
    emb = np.asarray(emb)  # Preserve the supplied embedding dtype.
    membership = np.asarray(membership, dtype=np.int32)

    N = len(texts)
    assert len(ids) == N, "ids and texts have different lengths."
    assert emb.shape[0] == N, "emb and texts have different row counts."
    assert membership.shape[0] == N, "membership and texts have different lengths."

    # Compute centroid similarity, distance, and cluster size.
    centroids, sims, dists, sizes = compute_centroid_sim_dist(emb, membership)
    rank = rank_within_cluster_by_dist(dists, membership)

    # Save arrays needed for later sampling without embeddings or KNN results.
    np.savez_compressed(
        out_dir / "arrays.npz",
        membership=membership,
        sizes=sizes,
        sim_to_centroid=sims,
        dist_to_centroid=dists,
        rank_in_cluster=rank,
    )
    np.save(out_dir / "centroids.npy", centroids)

    # Save the cluster-size table.
    import pandas as pd
    cluster_ids = np.arange(len(sizes), dtype=np.int32)
    df_sum = pd.DataFrame({"cluster_id": cluster_ids, "size": sizes})
    df_sum = df_sum[df_sum["size"] > 0].sort_values("size", ascending=False)
    df_sum.to_csv(out_dir / "cluster_summary.csv", index=False)

    # Parquet output requires pyarrow.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
        import pyarrow as pa
        import pyarrow.parquet as pq

    compression = "zstd"
    try:
        _ = pa.compress(pa.array([b"test"]), compression)
    except Exception:
        compression = "snappy"

    for start in range(0, N, chunk_size):
        end = min(N, start + chunk_size)
        part = {
            "idx": np.arange(start, end, dtype=np.int32),
            "id": ids[start:end],
            "text": texts[start:end],
            "cluster_id": membership[start:end],
            "sim_to_centroid": sims[start:end],
            "dist_to_centroid": dists[start:end],
            "rank_in_cluster": rank[start:end],
        }
        table = pa.Table.from_pydict(part)
        pq.write_table(
            table,
            out_dir / f"records_{start:09d}_{end:09d}.parquet",
            compression=compression,
        )

    # Prefer configuration values in metadata.
    meta = {
        "N": int(N),
        "D": int(emb.shape[1]),
        "embedding_dtype": str(emb.dtype),
        "notes": (
            "membership plus similarity, distance, and rank is sufficient for "
            "later sampling without rerunning FAISS or Leiden."
        ),
    }
    if cfg is not None:
        meta.update({
            "model_name": getattr(cfg, "model_name", None),
            "top_k": getattr(cfg, "top_k", None),
            "hnsw_m": getattr(cfg, "hnsw_m", None),
            "ef_construction": getattr(cfg, "ef_construction", None),
            "ef_search": getattr(cfg, "ef_search", None),
            "max_edges_per_node": getattr(cfg, "max_edges_per_node", None),
            "q_sim": getattr(cfg, "q_sim", None),
            "use_mutual": getattr(cfg, "use_mutual", None),
            "resolution_list": list(getattr(cfg, "resolution_list", [])),
            "max_ok_cluster": getattr(cfg, "max_ok_cluster", None),
            "attach_q": getattr(cfg, "attach_q", None),
            "attach_neigh_k": getattr(cfg, "attach_neigh_k", None),
            "pair_k": getattr(cfg, "pair_k", None),
            "pair_q": getattr(cfg, "pair_q", None),
        })

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved to: {out_dir.resolve()}")
    print("  - arrays.npz (membership/sizes/sim/dist/rank)")
    print("  - centroids.npy")
    print("  - cluster_summary.csv")
    print(f"  - records_*.parquet (chunk_size={chunk_size})")
    print("  - meta.json")



def _extract_input_ids(x: Any) -> List[int]:
    if isinstance(x, dict):
        return x["input_ids"]
    if hasattr(x, "data") and isinstance(getattr(x, "data"), dict) and "input_ids" in x.data:
        return x.data["input_ids"]
    return x



def _safe_token_id(tokenizer, token_text: Optional[str]) -> Optional[int]:
    if not token_text:
        return None

    tid = None

    try:
        tid = tokenizer.convert_tokens_to_ids(token_text)
    except Exception:
        tid = None

    unk_id = getattr(tokenizer, "unk_token_id", None)
    if tid is not None:
        try:
            tid = int(tid)
            if tid >= 0 and (unk_id is None or tid != unk_id):
                return tid
        except Exception:
            pass

    try:
        ids = tokenizer(token_text, add_special_tokens=False)["input_ids"]
        if isinstance(ids, list) and len(ids) == 1:
            tid = int(ids[0])
            decoded = tokenizer.decode([tid], skip_special_tokens=False)
            if decoded == token_text:
                return tid
    except Exception:
        pass

    return None



def _unique_preserve_order(seqs: List[List[int]]) -> List[List[int]]:
    out = []
    seen = set()
    for seq in seqs:
        key = tuple(int(x) for x in seq)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out



def _resolve_default_terminal_sequences(tokenizer) -> List[List[int]]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    end_turn_id = _safe_token_id(tokenizer, "<|end|>")
    eot_id = _safe_token_id(tokenizer, "<|endoftext|>")

    seqs: List[List[int]] = []

    if end_turn_id is not None and eot_id is not None and end_turn_id != eot_id:
        seqs.append([end_turn_id, eot_id])

    if eot_id is not None:
        seqs.append([eot_id])
    elif eos_id is not None:
        seqs.append([int(eos_id)])

    return _unique_preserve_order(seqs)



def _resolve_training_terminal_sequences(
    tokenizer,
    end_token_ids: Optional[List[int]] = None,
    terminal_sequences: Optional[List[List[int]]] = None,
) -> List[List[int]]:
    if terminal_sequences is not None:
        seqs = []
        for seq in terminal_sequences:
            seq = [int(x) for x in seq if x is not None]
            if seq:
                seqs.append(seq)
        return _unique_preserve_order(seqs)

    if end_token_ids is not None:
        seqs = []
        for tid in end_token_ids:
            if tid is not None:
                seqs.append([int(tid)])
        return _unique_preserve_order(seqs)

    return _resolve_default_terminal_sequences(tokenizer)



def _build_answer_labels_with_terminal_mask(
    prompt_ids: List[int],
    answer_ids: List[int],
    terminal_sequences: List[List[int]],
    terminal_loss_mode: str = "final_only",
) -> Tuple[List[int], List[int]]:
    if terminal_loss_mode not in {"all", "final_only", "none"}:
        raise ValueError("terminal_loss_mode must be one of: {'all', 'final_only', 'none'}")

    answer_ids = list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)

    terminal_token_set = {tid for seq in terminal_sequences for tid in seq}

    tail_positions = []
    i = len(answer_ids) - 1
    while i >= 0 and answer_ids[i] in terminal_token_set:
        tail_positions.append(i)
        i -= 1
    tail_positions.reverse()

    if terminal_loss_mode == "final_only" and tail_positions:
        for pos in tail_positions[:-1]:
            labels[len(prompt_ids) + pos] = -100

    elif terminal_loss_mode == "none" and tail_positions:
        for pos in tail_positions:
            labels[len(prompt_ids) + pos] = -100

    loss_answer_ids = [
        tid for tid, lab in zip(answer_ids, labels[len(prompt_ids):]) if lab != -100
    ]
    return labels, loss_answer_ids



class AnswerOnlyDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.pad_token_id is None:
            raise ValueError("Both tokenizer.pad_token_id and tokenizer.eos_token_id are None.")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)

        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of > 0:
            remainder = max_len % self.pad_to_multiple_of
            if remainder != 0:
                max_len = max_len + (self.pad_to_multiple_of - remainder)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_loss_weights = []

        has_loss_weights = all("loss_weights" in f for f in features)

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            input_ids = list(f["input_ids"]) + [self.pad_token_id] * pad_len
            attention_mask = list(f["attention_mask"]) + [0] * pad_len
            labels = list(f["labels"]) + [-100] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

            if has_loss_weights:
                loss_weights = [float(x) for x in f["loss_weights"]] + [0.0] * pad_len
                batch_loss_weights.append(loss_weights)

        out = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

        if has_loss_weights:
            out["loss_weights"] = torch.tensor(batch_loss_weights, dtype=torch.float32)

        return out
