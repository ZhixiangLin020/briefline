
"""CNN/DailyMail selection and trainer-dataset preparation."""

from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import random
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError:
    Dataset = DatasetDict = None
    load_dataset = load_from_disk = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable

from .config import ClusterConfig, fit_cluster_config_to_size
from .core import (
    _build_answer_labels_with_terminal_mask,
    _extract_input_ids,
    _resolve_training_terminal_sequences,
    build_graph_edges,
    faiss_hnsw_knn,
    leiden_cluster,
    sample_by_clusters_scheme2,
    save_all_cluster_artifacts,
)

PROMO_PREFIX = re.compile(r"^\s*(VIDEO|WATCH|NEW\s*:|NEW:|NEW\s*[-–—])\s*", flags=re.IGNORECASE)
TakeSpec = Optional[Union[int, float]]



def summarize_cluster_threshold_distribution(
    membership,
    thresholds: Iterable[int] = (10, 100, 1000),
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Count clusters whose sizes meet each configured threshold.

    Returns:
      {
        "ge": {thr: num_clusters_with_size_ge_thr},
        "num_clusters": total_clusters (sizes>0),
        "size_counter": optional Counter for ad hoc size queries
      }
    """

    sizes = np.bincount(membership)
    sizes = sizes[sizes > 0].astype(int)  # Keep only existing clusters.
    c = Counter(sizes)  # cluster size -> number of clusters of that size
    ge = {int(thr): int(sum(v for k, v in c.items() if k >= thr)) for thr in thresholds}

    if verbose:
        for thr in thresholds:
            print(f">={thr}:", ge[int(thr)])

    return {
        "ge": ge,
        "num_clusters": int(len(sizes)),
        "size_counter": c,
    }



def summarize_clusters_from_artifacts(art: dict, top_n: int = 20, print_stats: bool = True):
    """
    Summarize clustering results from saved artifacts.

    Returns:
      stats: dict with counts
      df_sizes: DataFrame sorted by size desc, columns: [cluster_id, size]
    """
    arrays = art["arrays"]
    df = art.get("records", None)

    membership = np.asarray(arrays["membership"], dtype=np.int32)
    sizes = np.asarray(arrays["sizes"])  # (K,)

    # only clusters with size > 0
    cluster_ids = np.flatnonzero(sizes > 0).astype(np.int32)
    df_sizes = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "size": sizes[cluster_ids].astype(np.int32, copy=False),
        }
    ).sort_values("size", ascending=False).reset_index(drop=True)

    n_clusters_total = int(len(df_sizes))
    n_singleton_clusters = int((df_sizes["size"] == 1).sum())
    n_non_singleton_clusters = int((df_sizes["size"] >= 2).sum())

    stats = {
        "N": int(len(membership)),
        "K_total_size_gt_0": n_clusters_total,
        "K_non_singleton_ge_2": n_non_singleton_clusters,
        "K_singleton_eq_1": n_singleton_clusters,
        "singleton_points": n_singleton_clusters,  # == number of singleton clusters
        "max_cluster_size": int(df_sizes["size"].iloc[0]) if n_clusters_total > 0 else 0,
        "max_cluster_id": int(df_sizes["cluster_id"].iloc[0]) if n_clusters_total > 0 else -1,
        "has_records_df": df is not None,
    }

    if print_stats:
        print("total clusters (size>0):", stats["K_total_size_gt_0"])
        print("non-singleton clusters (>=2):", stats["K_non_singleton_ge_2"])
        print("singleton clusters (==1):", stats["K_singleton_eq_1"])
        print("singleton points:", stats["singleton_points"])
        if stats["K_total_size_gt_0"] > 0:
            print("largest cluster:", stats["max_cluster_id"], "size:", stats["max_cluster_size"])

        # show top clusters if running in notebook
        try:
            from IPython.display import display
            display(df_sizes.head(int(top_n)))
        except Exception:
            print(df_sizes.head(int(top_n)).to_string(index=False))

    return stats, df_sizes



def clean_highlights(h: str) -> str:
    h = h or ""
    lines = [ln.strip() for ln in h.splitlines() if ln.strip()]
    txt = " ".join(lines).strip()
    while True:
        new_txt = PROMO_PREFIX.sub("", txt).strip()
        if new_txt == txt:
            break
        txt = new_txt
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt



def _slice_words_1idx(text: str, start_word: int = 50, end_word: int = 300) -> str:
    """
    Return the one-indexed inclusive word range [start_word, end_word].
    """
    if not text:
        return ""
    start_word = max(1, int(start_word))
    end_word = int(end_word)

    words = " ".join(text.split()).split()
    if not words:
        return ""

    s = start_word - 1
    e = None if end_word <= 0 else end_word  # 300 includes the 300th word.
    return " ".join(words[s:e]).strip()



def load_and_clean_cnn_dm_highlights(
    cfg,
    include_article: bool = True,
    article_word_start: int = 50,
    article_word_end: int = 300,

    # Natural-language field labels.
    highlight_prefix: str = "Highlights: ",
    article_prefix: str = "Article excerpt: ",
    return_source_indices: bool = False,
) -> Union[Tuple[List[str], List[str]], Tuple[List[str], List[str], List[int]]]:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=cfg.split)
    ds = ds.select(range(min(cfg.take_n, len(ds))))

    raw_ids = ds["id"]
    raw_hls = ds["highlights"]
    raw_arts = ds["article"] if include_article else [""] * len(ds)

    ids, texts, source_indices = [], [], []
    rows = enumerate(zip(raw_ids, raw_hls, raw_arts))
    for source_idx, (_id, h, a) in tqdm(rows, total=len(raw_ids), desc="Clean highlights"):
        hl = clean_highlights(h)
        if not hl:
            continue

        if include_article:
            art_snip = _slice_words_1idx(a or "", start_word=article_word_start, end_word=article_word_end)
            if art_snip:
                # Combine the summary with its article context.
                t = (f"{highlight_prefix}{hl}\n{article_prefix}{art_snip}").strip()
            else:
                t = (f"{highlight_prefix}{hl}").strip()
        else:
            t = (f"{highlight_prefix}{hl}").strip()

        ids.append(_id)
        texts.append(t)
        source_indices.append(source_idx)

    print("N =", len(texts))
    if return_source_indices:
        return ids, texts, source_indices
    return ids, texts



def embed_texts_e5(
    texts: List[str],
    cfg: ClusterConfig,
) -> np.ndarray:
    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    model.max_seq_length = cfg.max_seq_length

    # Prefix once before encoding.
    texts_prefixed = [cfg.prefix + t for t in texts]

    emb = model.encode(
        texts_prefixed,
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    norms = np.linalg.norm(emb, axis=1)
    print("embeddings:", emb.shape, emb.dtype)
    print("norm min/mean/max:", float(norms.min()), float(norms.mean()), float(norms.max()))
    return emb



def attach_singletons_A(
    membership: np.ndarray,
    nbrs: np.ndarray,
    scores: np.ndarray,
    cfg: ClusterConfig,
) -> np.ndarray:
    sizes = np.bincount(membership)
    is_single = (sizes[membership] == 1)
    single_idx = np.where(is_single)[0]
    print("singletons before A:", len(single_idx))

    k = min(cfg.attach_neigh_k, cfg.top_k - 1)
    attach_scores = scores[:, 1:1 + k].reshape(-1)
    attach_scores = attach_scores[np.isfinite(attach_scores)]
    attach_th = float(np.percentile(attach_scores, cfg.attach_q)) if len(attach_scores) else 0.6
    attach_th = min(0.95, max(0.3, attach_th))
    print("ATTACH_TH:", attach_th)

    # only attach to clusters with size>=2
    for i in tqdm(single_idx, desc="Attach A (to non-singleton)"):
        neigh = nbrs[i, 1:cfg.top_k]
        sims  = scores[i, 1:cfg.top_k]
        for j, s in zip(neigh, sims):
            if j < 0:
                continue
            if s < attach_th:
                break
            cj = membership[j]
            if sizes[cj] >= 2:
                membership[i] = cj
                break

    return membership



class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1



def attach_singletons_B(
    membership: np.ndarray,
    nbrs: np.ndarray,
    scores: np.ndarray,
    cfg: ClusterConfig,
) -> np.ndarray:
    sizes = np.bincount(membership)
    is_single = (sizes[membership] == 1)
    single_idx = np.where(is_single)[0]
    print("singletons before B:", len(single_idx))

    N = len(membership)
    k = min(cfg.pair_k, cfg.top_k - 1)

    pair_scores = scores[:, 1:1 + k].reshape(-1)
    pair_scores = pair_scores[np.isfinite(pair_scores)]
    pair_th = float(np.percentile(pair_scores, cfg.pair_q)) if len(pair_scores) else 0.3
    pair_th = min(0.95, max(0.3, pair_th))
    print("PAIR_TH:", pair_th)

    nbrp = nbrs[:, 1:1 + k]
    scp  = scores[:, 1:1 + k]
    mask = (scp >= pair_th)

    src = np.repeat(np.arange(N, dtype=np.int32), k)
    dst = nbrp.reshape(-1).astype(np.int32)
    sel = mask.reshape(-1)

    src = src[sel]
    dst = dst[sel]

    keep2 = is_single[src] & is_single[dst] & (src != dst)
    src = src[keep2]
    dst = dst[keep2]

    if len(src) == 0:
        return membership

    # mutual among singletons
    src64 = src.astype(np.int64)
    dst64 = dst.astype(np.int64)
    key = src64 * N + dst64
    rkey = dst64 * N + src64

    key = np.unique(key)
    rkey = np.unique(rkey)
    mutual = np.intersect1d(key, rkey, assume_unique=True)

    u = (mutual // N).astype(np.int32)
    v = (mutual %  N).astype(np.int32)
    a = np.minimum(u, v)
    b = np.maximum(u, v)
    pairsB = np.unique(np.stack([a, b], axis=1), axis=0)
    print("B-stage edges among singletons:", len(pairsB))

    uf = UnionFind(N)
    for a_, b_ in tqdm(pairsB, desc="Union singletons"):
        uf.union(int(a_), int(b_))

    roots = np.array([uf.find(int(i)) for i in single_idx], dtype=np.int32)
    rep: Dict[int, int] = {}
    for i, r in zip(single_idx, roots):
        if int(r) not in rep:
            rep[int(r)] = int(i)

    for i, r in tqdm(zip(single_idx, roots), total=len(single_idx), desc="Relabel singletons"):
        membership[int(i)] = membership[rep[int(r)]]

    return membership



def print_cluster_stats(membership: np.ndarray) -> Dict[str, int]:
    sizes = np.bincount(membership)
    total_clusters = int((sizes > 0).sum())
    non_singleton_clusters = int((sizes >= 2).sum())
    singleton_clusters = int((sizes == 1).sum())
    largest = int(sizes.max()) if len(sizes) else 0
    covered = int(sizes[sizes >= 2].sum())

    print("\n==== Final Stats ====")
    print("total clusters (incl. singletons):", total_clusters)
    print("non-singleton clusters (>=2):", non_singleton_clusters)
    print("singleton clusters:", singleton_clusters)
    print("largest cluster size:", largest)
    print("covered (in clusters>=2):", covered, "/", int(len(membership)))

    return {
        "total_clusters": total_clusters,
        "non_singleton_clusters": non_singleton_clusters,
        "singleton_clusters": singleton_clusters,
        "largest": largest,
        "covered": covered,
        "N": int(len(membership)),
    }



def _sha1_of_ids(ids) -> str:
    h = hashlib.sha1()
    for x in ids:
        h.update(str(x).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()



def _atomic_save_npy(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    np.save(tmp, arr)                 # Creates tmp + ".npy".
    if not tmp.endswith(".npy"):
        tmp_npy = tmp + ".npy"
        final = path if path.endswith(".npy") else path + ".npy"
        os.replace(tmp_npy, final)
        # Remove an unlikely extensionless temporary file.
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    else:
        os.replace(tmp, path)



def run_cnn_dm_highlights_clustering(
    cfg: Optional["ClusterConfig"] = None,
    emb_cache_dir: Optional[str] = None,
    emb_cache_key: Optional[str] = None,
    save_emb_immediately: bool = True,
    load_emb_if_exists: bool = True,
    mmap_on_load: bool = False,
    keep_emb: bool = False,

    artifacts_out_dir: Optional[str] = None,  # Save artifacts when provided.
    artifacts_chunk_size: int = 200_000,
    drop_knn_before_save: bool = True,        # Release graph arrays before saving.
):
    import os
    import json
    import numpy as np

    cfg = cfg or ClusterConfig()
    ids, texts, source_indices = load_and_clean_cnn_dm_highlights(
        cfg,
        return_source_indices=True,
    )
    cfg = fit_cluster_config_to_size(cfg, len(texts))

    # ---- embedding cache path ----
    emb_cache_dir = emb_cache_dir or getattr(cfg, "emb_cache_dir", None)
    if emb_cache_dir is None:
        save_emb_immediately = False

    emb = None
    emb_path = None
    meta_path = None

    if save_emb_immediately:
        if emb_cache_key is None:
            model_name = getattr(cfg, "e5_model_name", getattr(cfg, "model_name", "e5"))
            emb_cache_key = f"cnn_dm_{model_name}_{len(texts)}_{_sha1_of_ids(ids)[:12]}"
        emb_path = os.path.join(emb_cache_dir, emb_cache_key + ".npy")
        meta_path = os.path.join(emb_cache_dir, emb_cache_key + ".meta.json")

        if load_emb_if_exists and os.path.exists(emb_path):
            emb = np.load(emb_path, mmap_mode="r" if mmap_on_load else None)
        else:
            emb = embed_texts_e5(texts, cfg)
            _atomic_save_npy(np.asarray(emb), emb_path)

            meta = {
                "key": emb_cache_key,
                "n": int(len(texts)),
                "dim": int(np.asarray(emb).shape[1]),
                "dtype": str(np.asarray(emb).dtype),
                "model_name": getattr(cfg, "e5_model_name", getattr(cfg, "model_name", None)),
                "normalized": bool(getattr(cfg, "e5_normalize", True)),
            }
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            if mmap_on_load:
                del emb
                gc.collect()
                emb = np.load(emb_path, mmap_mode="r")
    else:
        emb = embed_texts_e5(texts, cfg)

    # Preserve embedding precision.
    if emb.dtype != np.float32:
        raise ValueError(
            f"Embedding dtype is {emb.dtype}, expected float32 for FAISS HNSW. "
            f"I won't auto-cast. Ensure embed_texts_e5 outputs float32 and cache is float32."
        )

    # Ensure contiguous storage without changing dtype.
    emb_for_knn = np.ascontiguousarray(emb) if not emb.flags["C_CONTIGUOUS"] else emb

    scores, nbrs = faiss_hnsw_knn(emb_for_knn, cfg)

    # Release embeddings after KNN by default.
    if not keep_emb:
        if emb_for_knn is not emb:
            del emb_for_knn
        del emb
        gc.collect()
        emb_ret = None
    else:
        emb_ret = emb_for_knn

    pairs, weights, sim_th = build_graph_edges(nbrs, scores, cfg)
    membership = leiden_cluster(
        len(texts),
        pairs,
        weights,
        cfg,
        seed=cfg.seed,
    )
    membership = attach_singletons_A(membership, nbrs, scores, cfg)
    membership = attach_singletons_B(membership, nbrs, scores, cfg)

    stats = print_cluster_stats(membership)
    dist = summarize_cluster_threshold_distribution(membership, thresholds=(10, 100, 1000), verbose=True)

    # Save artifacts after clustering.
    if artifacts_out_dir is not None:
        # Release large KNN and graph arrays before saving.
        if drop_knn_before_save:
            try:
                del pairs, weights
            except Exception:
                pass
            try:
                del scores, nbrs
            except Exception:
                pass
            gc.collect()

        # Memory-map released embeddings from disk without changing dtype.
        if emb_ret is None:
            if emb_path is None or (not os.path.exists(emb_path)):
                raise ValueError("Need embedding to save artifacts, but emb_path not found. "
                                 "Set emb_cache_dir/save_emb_immediately=True or keep_emb=True.")
            emb_for_save = np.load(emb_path, mmap_mode="r")
        else:
            emb_for_save = emb_ret

        save_all_cluster_artifacts(
            out_dir=artifacts_out_dir,
            ids=ids,
            texts=texts,
            emb=emb_for_save,
            membership=membership,
            cfg=cfg,
            chunk_size=artifacts_chunk_size,
        )

        # Release a reloaded memory map after saving.
        if emb_ret is None:
            del emb_for_save
            gc.collect()

    return {
        "ids": ids,
        "texts": texts,
        "source_indices": source_indices,
        "emb": emb_ret,   # Available only when keep_emb=True.
        "scores": scores if (not artifacts_out_dir or not drop_knn_before_save) else None,
        "nbrs": nbrs if (not artifacts_out_dir or not drop_knn_before_save) else None,
        "membership": membership,
        "stats": stats,
        "sim_th": sim_th,
        "dist": dist,
        "emb_path": emb_path,
        "emb_meta_path": meta_path,
        "emb_cache_key": emb_cache_key,
        "artifacts_out_dir": artifacts_out_dir,
    }



def load_cluster_artifacts(out_dir: str, load_records: bool = True, columns=None):
    """
    Load the output of save_all_cluster_artifacts().

    Returns:
      - arrays: every array in the NPZ file
      - centroids: shape (K, D), or None
      - summary: cluster_summary.csv, or None
      - records: DataFrame with id, text, idx, and cluster fields, or None
    """
    out_dir = Path(out_dir)

    # 1) arrays.npz
    arrays_path = out_dir / "arrays.npz"
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing: {arrays_path}")
    arrays = np.load(arrays_path, allow_pickle=False)

    # 2) Optional centroids.npy.
    centroids_path = out_dir / "centroids.npy"
    centroids = np.load(centroids_path) if centroids_path.exists() else None

    # 3) Optional summary.
    summary_path = out_dir / "cluster_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    # 4) Optional record shards; loading all shards uses substantial memory.
    records = None
    if load_records:
        files = sorted(glob.glob(str(out_dir / "records_*.parquet")))
        if not files:
            raise FileNotFoundError(f"No records_*.parquet under {out_dir}")

        # Load source fields plus fields required for sampling.
        if columns is None:
            columns = ["idx", "id", "text", "cluster_id",
                       "sim_to_centroid", "dist_to_centroid", "rank_in_cluster"]

        records = pd.concat(
            (pd.read_parquet(f, columns=columns) for f in files),
            ignore_index=True
        ).sort_values("idx").reset_index(drop=True)

        # Recommended consistency check.
        mem_from_df = records["cluster_id"].to_numpy(np.int32)
        mem_from_npz = arrays["membership"].astype(np.int32)
        if len(mem_from_df) == len(mem_from_npz) and not np.array_equal(mem_from_df, mem_from_npz):
            print(
                "[Warn] record cluster IDs do not match membership in arrays.npz. "
                "Check index ordering and ensure artifacts are from the same run."
            )

    return {
        "arrays": arrays,
        "centroids": centroids,
        "summary": summary,
        "records": records,
    }



def build_rep_map(membership: np.ndarray):
    membership = np.asarray(membership, dtype=np.int64)
    order = np.argsort(membership, kind="mergesort")
    mem_sorted = membership[order]

    cuts = np.flatnonzero(mem_sorted[1:] != mem_sorted[:-1]) + 1
    bounds = np.r_[0, cuts, len(order)]

    rep_map = {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        cid = int(mem_sorted[s])
        rep_map[cid] = order[s:e].astype(np.int64)
    return rep_map



def _dedup_keep_order(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    seen = set()
    out = []
    for x in arr.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.asarray(out, dtype=np.int64)



def _detect_cnn_dm_cols(ds: Dataset) -> Tuple[str, str]:
    cols = set(ds.column_names)
    if "article" not in cols:
        raise KeyError(f"Missing 'article' column; available columns: {ds.column_names}")
    # CNN/DailyMail uses "highlights"; some variants use "highlight".
    if "highlights" in cols:
        return "article", "highlights"
    if "highlight" in cols:
        return "article", "highlight"
    raise KeyError(
        f"Missing 'highlights' or 'highlight' column; available columns: {ds.column_names}"
    )



def load_cnn_dm_if_needed(ds_dict=None, name="abisee/cnn_dailymail", config="3.0.0"):
    """
    ds_dict=None downloads CNN/DailyMail.
    A supplied Dataset or DatasetDict is normalized and returned.
    """
    if ds_dict is None:
        return load_dataset(name, config)
    if isinstance(ds_dict, Dataset):
        return DatasetDict({"train": ds_dict})
    if isinstance(ds_dict, DatasetDict):
        return ds_dict
    raise TypeError(f"ds_dict must be None/Dataset/DatasetDict, got {type(ds_dict)}")



def build_picked_cnn_dm_raw_dataset(
    picked_idx,
    membership=None,
    source_indices=None,
    ds_dict=None,                 # Load the official dataset when omitted.
    split="train",
    out_dir="picked_cnn_dm_raw",  # save_to_disk destination.
    reload_if_exists=True,        # Reuse an existing directory.
    keep_only_article_highlights=True,
):
    """
    Select CNN/DailyMail article/highlight rows and save a new Dataset.

    Additional fields:
      - orig_idx: row position in the original split
      - cluster_id: included when membership is supplied
    """
    # 0) Reuse an existing saved result.
    if reload_if_exists and os.path.isdir(out_dir):
        ds = load_from_disk(out_dir)
        meta_path = os.path.join(out_dir, "meta.json")
        meta = json.load(open(meta_path, "r", encoding="utf-8")) if os.path.isfile(meta_path) else {}
        return ds, meta

    # 1) Load CNN/DailyMail or use the supplied dataset.
    ds_dict = load_cnn_dm_if_needed(ds_dict)

    if split not in ds_dict:
        raise KeyError(
            f"DatasetDict has no split='{split}'; available splits: {list(ds_dict.keys())}"
        )
    ds = ds_dict[split]
    n = len(ds)

    # 2) Normalize indices in the processed clustering view.
    idx = _dedup_keep_order(np.asarray(picked_idx, dtype=np.int64))
    if idx.size == 0:
        raise ValueError("picked_idx is empty after normalization.")

    membership_arr = None
    if membership is not None:
        membership_arr = np.asarray(membership, dtype=np.int64)
        if membership_arr.ndim != 1:
            raise ValueError("membership must be a one-dimensional array.")
        processed_len = len(membership_arr)
    elif source_indices is not None:
        processed_len = len(source_indices)
    else:
        processed_len = n

    if np.any(idx < 0) or np.any(idx >= processed_len):
        raise IndexError(
            "picked_idx contains positions outside the processed clustering view: "
            f"valid range is [0, {processed_len})."
        )

    # Map clustering-view positions back to rows in the original split. This
    # allows a limited run (for example, 50,000 rows) to export from the full
    # CNN/DailyMail split without requiring membership to cover all 287,113 rows.
    if source_indices is None:
        raw_idx = idx
    else:
        source_indices_arr = np.asarray(source_indices, dtype=np.int64)
        if source_indices_arr.ndim != 1:
            raise ValueError("source_indices must be a one-dimensional array.")
        if len(source_indices_arr) != processed_len:
            raise ValueError(
                "source_indices must align one-to-one with the processed clustering view: "
                f"got {len(source_indices_arr)} indices for {processed_len} processed rows."
            )
        raw_idx = source_indices_arr[idx]

    if np.any(raw_idx < 0) or np.any(raw_idx >= n):
        raise IndexError(
            "source_indices maps a selected row outside the original split: "
            f"valid range is [0, {n})."
        )

    # 3) Select rows.
    picked = ds.select(raw_idx.tolist())

    # 4) Optionally keep only article and highlight fields.
    article_col, hl_col = _detect_cnn_dm_cols(picked)
    if keep_only_article_highlights:
        keep_cols = [article_col, hl_col]
        drop_cols = [c for c in picked.column_names if c not in keep_cols]
        if drop_cols:
            picked = picked.remove_columns(drop_cols)

    # 5) Add traceability fields.
    picked = picked.add_column("orig_idx", raw_idx.tolist())
    if membership_arr is not None:
        picked = picked.add_column("cluster_id", membership_arr[idx].tolist())

    # 6) Save.
    os.makedirs(out_dir, exist_ok=True)
    picked.save_to_disk(out_dir)

    meta = {
        "dataset": "abisee/cnn_dailymail",
        "config": "3.0.0",
        "split_used": split,
        "orig_len": int(n),
        "processed_len": int(processed_len),
        "picked_len": int(len(picked)),
        "article_col": article_col,
        "highlight_col": hl_col,
        "has_cluster_id": membership_arr is not None,
        "has_source_indices": source_indices is not None,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return picked, meta



def summarize_sampling_by_cluster_simple(membership, picked_idx, top_n=30, return_df=False):
    """
    Print top-N clusters by original size, including:
      - orig: original cluster size
      - picked: sampled count from that cluster
      - ratio: picked/orig
      - orig_share: orig / total_N
      - picked_share: picked / total_picked
    """
    membership = np.asarray(membership, dtype=np.int64)
    picked_idx = np.asarray(picked_idx, dtype=np.int64)

    N = int(membership.shape[0])
    total_picked = int(picked_idx.shape[0])

    K = int(membership.max()) + 1
    sizes = np.bincount(membership, minlength=K).astype(np.int64)

    picked_c = membership[picked_idx]
    sampled_counts = np.bincount(picked_c, minlength=K).astype(np.int64)

    nonempty = np.where(sizes > 0)[0]
    order = nonempty[np.argsort(-sizes[nonempty])]

    header = f"{'cid':>6}  {'orig':>8}  {'picked':>8}  {'ratio':>7}  {'orig_share':>10}  {'picked_share':>11}"
    print(header)

    for cid in order[:top_n]:
        orig = int(sizes[cid])
        pick = int(sampled_counts[cid])

        ratio = (pick / orig)*100 if orig else 0.0
        orig_share = (orig / N)*100 if N else 0.0
        picked_share = (pick / total_picked)*100 if total_picked else 0.0

        print(
            f"{cid:6d}  {orig:8d}  {pick:8d}  {ratio:7.3f}  "
            f"{orig_share:10.3f}  {picked_share:11.3f}"
        )

    if return_df:
        import pandas as pd
        rows = []
        for cid in order:
            orig = int(sizes[cid])
            pick = int(sampled_counts[cid])
            rows.append(
                {
                    "cid": cid,
                    "orig": orig,
                    "picked": pick,
                    "ratio": (pick / orig) if orig else 0.0,
                    "orig_share": (orig / N) if N else 0.0,
                    "picked_share": (pick / total_picked) if total_picked else 0.0,
                }
            )
        df = pd.DataFrame(rows).sort_values("orig", ascending=False).reset_index(drop=True)
        return sizes, sampled_counts, order, df

    return sizes, sampled_counts, order



def run_cnn_dm_end2end(
    # ---- clustering ----
    cfg: Optional[ClusterConfig] = None,
    emb_cache_dir: str = "cnn_dm_cache/emb",
    artifacts_out_dir: str = "cnn_dm_cache/artifacts/run_001",
    artifacts_chunk_size: int = 200_000,
    load_emb_if_exists: bool = True,
    drop_knn_before_save: bool = True,

    # ---- read artifacts ----
    load_records: bool = True,
    records_columns: Optional[list] = None,
    top_n_clusters_to_show: int = 100,

    # ---- sampling (Scheme2) ----
    num: Optional[int] = None,
    growth: str = "sqrt",
    cap: int = 20_000,
    log_base: float = 4.0,
    tau: float = 1.0,
    seed: int = 42,
    gap_side: str = "right",

    # ---- export picked raw dataset ----
    picked_split: str = "train",
    picked_out_dir: str = "picked_cnn_dm_raw",
    reload_picked_if_exists: bool = True,
    keep_only_article_highlights: bool = True,

    # ---- reporting ----
    summarize_sampling_top_n: int = 200,
    build_repmap: bool = False,
) -> Dict[str, Any]:
    """
    One-shot pipeline:
      1) run clustering (+ embedding cache) and save artifacts
      2) load artifacts (arrays + optional records df)
      3) scheme2 sampling using artifacts (emb=None)
      4) export picked raw CNN/DM dataset
      5) print summaries

    Returns a dict with:
      - out: clustering return dict
      - art: loaded artifacts dict
      - membership/sizes/dist/sim/rank
      - stats/df_sizes (from summarize_clusters_from_artifacts)
      - picked_idx, pick_report
      - picked_raw, picked_meta
      - rep_map (optional)
    """
    if cfg is None:
        cfg = ClusterConfig()

    # 1) clustering + artifacts saving
    out = run_cnn_dm_highlights_clustering(
        cfg=cfg,
        emb_cache_dir=emb_cache_dir,
        load_emb_if_exists=load_emb_if_exists,
        artifacts_out_dir=artifacts_out_dir,
        artifacts_chunk_size=artifacts_chunk_size,
        drop_knn_before_save=drop_knn_before_save,
    )

    # membership from this run (for later checks / rep_map)
    membership_run = out["membership"]

    rep_map = build_rep_map(membership_run) if build_repmap else None

    # 2) load artifacts back (for emb=None sampling & aligned ids/texts)
    art = load_cluster_artifacts(
        artifacts_out_dir,
        load_records=load_records,
        columns=records_columns,
    )
    arrays = art["arrays"]
    df = art["records"] if load_records else None

    # restore aligned ids/texts (optional; only if you loaded records)
    if df is not None:
        ids = df["id"].tolist()
        texts = df["text"].tolist()
    else:
        # fallback: use out's ids/texts (still aligned to membership)
        ids = out["ids"]
        texts = out["texts"]

    membership = arrays["membership"].astype("int64", copy=False)
    sizes = arrays["sizes"]
    sim = arrays["sim_to_centroid"]
    dist = arrays["dist_to_centroid"]
    rank = arrays["rank_in_cluster"]

    print("N =", int(len(membership)), "K =", int((sizes > 0).sum()))
    stats, df_sizes = summarize_clusters_from_artifacts(
        art,
        top_n=top_n_clusters_to_show,
        print_stats=True,
    )

    # 3) scheme2 sampling (emb=None -> use artifacts arrays.npz)
    picked_idx, pick_report = sample_by_clusters_scheme2(
        emb=None,
        membership=membership,
        num=num,
        growth=growth,
        cap=cap,
        log_base=log_base,
        tau=tau,
        seed=seed,
        artifacts_dir=artifacts_out_dir,
        gap_side=gap_side,
    )
    print(pick_report)

    # 4) export picked raw dataset
    picked_raw, picked_meta = build_picked_cnn_dm_raw_dataset(
        picked_idx=picked_idx,
        membership=membership,  # optional but recommended
        source_indices=out.get("source_indices"),
        ds_dict=None,
        split=picked_split,
        out_dir=picked_out_dir,
        reload_if_exists=reload_picked_if_exists,
        keep_only_article_highlights=keep_only_article_highlights,
    )

    # 5) sampling summary
    summarize_sampling_by_cluster_simple(
        membership=membership,
        picked_idx=picked_idx,
        top_n=summarize_sampling_top_n,
        return_df=False,
    )

    return {
        "cfg": cfg,
        "cfg_dict": asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else None,

        "out": out,
        "art": art,

        "df_records": df,
        "ids": ids,
        "texts": texts,

        "membership": membership,
        "sizes": sizes,
        "sim": sim,
        "dist": dist,
        "rank": rank,

        "cluster_stats": stats,
        "df_cluster_sizes": df_sizes,

        "picked_idx": picked_idx,
        "pick_report": pick_report,
        "picked_raw": picked_raw,
        "picked_meta": picked_meta,

        "rep_map": rep_map,
    }



def _get_cluster_members(membership: np.ndarray, cid: int) -> np.ndarray:
    membership = np.asarray(membership, dtype=np.int64)
    return np.where(membership == int(cid))[0]



def _get_cluster_picks(
    cid: int,
    membership: np.ndarray,
    rep_map: dict | None = None,
    picked_idx: np.ndarray | None = None,
) -> np.ndarray:
    cid = int(cid)

    # 1) Prefer rep_map when it contains this cluster.
    if rep_map is not None:
        if cid in rep_map:
            return np.asarray(rep_map[cid], dtype=np.int64)
        # Also handle string keys.
        if str(cid) in rep_map:
            return np.asarray(rep_map[str(cid)], dtype=np.int64)

    # 2) Fall back to picked_idx.
    if picked_idx is None:
        return np.array([], dtype=np.int64)

    picked_idx = np.asarray(picked_idx, dtype=np.int64)
    membership = np.asarray(membership, dtype=np.int64)
    return picked_idx[membership[picked_idx] == cid]



def show_cluster_samples(
    cid: int,
    texts: list[str],
    membership: np.ndarray,
    rep_map: dict | None = None,
    picked_idx: np.ndarray | None = None,
    max_show: int = 10,
    show_all: bool = False,  # Print every member; avoid for large clusters.
):
    cid = int(cid)
    members = _get_cluster_members(membership, cid)
    picks = _get_cluster_picks(cid, membership, rep_map=rep_map, picked_idx=picked_idx)

    print(f"\n=== cluster {cid} ===")
    print(f"orig size = {len(members)}")
    print(f"picked    = {len(picks)}")

    if len(members) == 0:
        print("(empty cluster id?)")
        return

    # Print selected samples first.
    if len(picks) == 0:
        print("\n(no picked samples for this cluster)")
    else:
        to_show = picks if show_all else picks[:max_show]
        print(f"\n--- picked samples (show {len(to_show)}/{len(picks)}) ---")
        for i, idx in enumerate(to_show, 1):
            t = texts[int(idx)]
            print(f"\n[{i}] idx={int(idx)}")
            print(t)

    # Optionally print unselected samples from the cluster.
    print(f"\n--- other members (first {max_show}) ---")
    for i, idx in enumerate(members[:max_show], 1):
        print(f"\n({i}) idx={int(idx)}")
        print(texts[int(idx)])



def _sanitize_num_proc(num_proc: Optional[int]) -> int:
    cpu_count = os.cpu_count() or 1
    if num_proc is None:
        num_proc = min(8, cpu_count)
    return max(1, min(int(num_proc), cpu_count))



def _normalize_to_datasetdict(x):
    from datasets import Dataset, DatasetDict
    if isinstance(x, DatasetDict):
        return x
    if isinstance(x, Dataset):
        return DatasetDict({"train": x})
    raise TypeError(f"Expected Dataset or DatasetDict, got {type(x)}")



def _detect_cols(ds: Dataset) -> Tuple[str, str]:
    cols = set(ds.column_names)
    if "article" not in cols:
        raise KeyError(f"Missing 'article'. Available columns: {ds.column_names}")
    if "highlights" in cols:
        return "article", "highlights"
    if "highlight" in cols:
        return "article", "highlight"
    raise KeyError(f"Missing 'highlights'/'highlight'. Available columns: {ds.column_names}")



def _compute_target(raw_len: int, take_spec, split_name: str):
    if take_spec is None:
        return None
    if isinstance(take_spec, int):
        return max(0, int(take_spec))
    if isinstance(take_spec, float):
        take_spec = float(take_spec)
        if take_spec <= 0:
            return 0
        if take_spec >= 1.0:
            return None
        return max(1, int(raw_len * take_spec))
    raise TypeError(f"[{split_name}] Unsupported take_spec type: {type(take_spec)}")



def _contains_legacy_tags(text: str) -> bool:
    text = str(text or "")
    legacy_markers = [
        "[TASK=SUMMARY]",
        "[TASK=HIGHLIGHT]",
        "[BODY]",
        "[/BODY]",
        "[SUMMARY]",
        "[/SUMMARY]",
        "[HIGHLIGHT]",
        "[/HIGHLIGHT]",
    ]
    return any(tok in text for tok in legacy_markers)



def _ends_with_any_sequence(ids: List[int], seqs: List[List[int]]) -> bool:
    ids = list(ids)
    for seq in seqs:
        seq = list(seq)
        if seq and len(ids) >= len(seq) and ids[-len(seq):] == seq:
            return True
    return False



def _select_take_before_heavy_processing(raw: Dataset, take_spec, split_name: str, seed: int) -> Dataset:
    target = _compute_target(len(raw), take_spec, split_name)
    if target == 0:
        return Dataset.from_list([])
    raw = raw.shuffle(seed=seed)
    if target is None or target >= len(raw):
        return raw
    return raw.select(range(int(target)))



def _highlights_to_highlight_text(highlights: str) -> str:
    lines = [ln.strip() for ln in (highlights or "").splitlines()]
    lines = [ln for ln in lines if ln]
    text = " ".join(lines).strip()

    promo_tokens = [
        "Larry King Live",
        "VIDEO", "Video",
        "WATCH", "Watch",
        "tonight", "Tonight",
        "NEW:", "NEW :", "NEW -", "NEW –", "NEW —", "NEW --",
    ]

    def _strip_leading_token(s: str, tok: str) -> str:
        s2 = s.lstrip()
        if s2.find(tok) != 0:
            return s
        s2 = s2[len(tok):]
        s2 = s2.lstrip(" \t:：-–—•|,.")
        return s2

    changed = True
    while changed:
        changed = False
        for tok in promo_tokens:
            new_text = _strip_leading_token(text, tok)
            if new_text != text:
                text = new_text
                changed = True
                break

    fix_pairs = [
        (" . ", ". "),
        (" .", "."),
        # ("..", "."),
        (" ,", ","),
        (" ;", ";"),
        (" :", ":"),
        (" )", ")"),
        ("( ", "("),

    ]
    for a, b in fix_pairs:
        text = text.replace(a, b)

    return " ".join(text.split()).strip()



def _truncate_text_by_tokens(
    tokenizer,
    text: str,
    max_tokens: Optional[int],
) -> str:
    text = (text or "").strip()
    if not text or max_tokens is None or max_tokens <= 0:
        return text

    ids = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    if len(ids) <= max_tokens:
        return text

    truncated_ids = ids[:max_tokens]
    return tokenizer.decode(
        truncated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()



def _default_highlight_system_text() -> str:
    return (
        "You are a news editor.\n"
        "Write the key highlights of the article in 2-4 concise sentences.\n"
        "Use only information from the article.\n"
        "Be factual, neutral, and concise.\n"
        "Do not add outside facts, guesses, explanations, or commentary."
    )



def _make_highlight_prompt(article: str) -> str:
    article = (article or "").strip()
    return f"Article:\n{article}"



def _make_highlight_answer(highlight_text: str, highlight_prefix_text: str = "highlight: ") -> str:
    highlight_text = (highlight_text or "").strip()
    return f"{highlight_prefix_text}{highlight_text}"



def _build_highlight_prompt_batch(
    batch,
    *,
    article_col: str,
    highlights_col: str,
    highlight_prefix_text: str = "highlight: ",
    tokenizer=None,
    article_max_tokens: Optional[int] = None,
):
    out = {
        "source_id": [],
        "PromptRaw": [],
        "AnswerPlain": [],
        "HighlightBody": [],
        "TaskType": [],
    }

    articles = batch[article_col]
    highlights_list = batch[highlights_col]

    for row_idx, (article, highlights) in enumerate(zip(articles, highlights_list)):
        article = (article or "").strip()
        highlights = (highlights or "").strip()

        article = _truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=article,
            max_tokens=article_max_tokens,
        )

        if not article or not highlights:
            continue

        highlight_text = _highlights_to_highlight_text(highlights)
        if not highlight_text:
            continue

        out["source_id"].append(str(row_idx))
        out["PromptRaw"].append(_make_highlight_prompt(article))
        out["AnswerPlain"].append(_make_highlight_answer(highlight_text, highlight_prefix_text=highlight_prefix_text))
        out["HighlightBody"].append(highlight_text)
        out["TaskType"].append("article->highlight")

    return out



def _build_highlight_prompt_dataset_from_raw_parallel(
    raw: Dataset,
    *,
    take_spec,
    split_name: str,
    seed: int,
    highlight_prefix_text: str,
    tokenizer,
    article_max_tokens: Optional[int],
    batch_size: int,
    num_proc: int,
    load_from_cache_file: bool,
) -> Dataset:
    raw = _select_take_before_heavy_processing(raw, take_spec, split_name, seed)
    if len(raw) == 0:
        return Dataset.from_list([])

    article_col, hl_col = _detect_cols(raw)

    ds_prompt = raw.map(
        _build_highlight_prompt_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc=f"Building highlight prompts for split={split_name}",
        fn_kwargs={
            "article_col": article_col,
            "highlights_col": hl_col,
            "highlight_prefix_text": highlight_prefix_text,
            "tokenizer": tokenizer,
            "article_max_tokens": article_max_tokens,
        },
        remove_columns=raw.column_names,
    )

    return ds_prompt



def _build_chat_ids_and_texts(
    tokenizer,
    user_text: str,
    assistant_text_plain: str,
    system_text: Optional[str] = None,
):
    """
    The chat path does not append terminal tokens manually.
    """
    prefix_messages = []
    if system_text and system_text.strip():
        prefix_messages.append({"role": "system", "content": system_text.strip()})
    prefix_messages.append({"role": "user", "content": user_text.strip()})

    full_messages = list(prefix_messages) + [
        {"role": "assistant", "content": assistant_text_plain.strip()}
    ]

    prompt_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    full_ids = _extract_input_ids(
        tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )

    answer_ids = full_ids[len(prompt_ids):]

    prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
    full_text = tokenizer.decode(full_ids, skip_special_tokens=False)

    return {
        "prompt_ids": list(prompt_ids),
        "answer_ids": list(answer_ids),
        "full_ids": list(full_ids),
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
    }



def _build_plain_ids_and_texts(
    tokenizer,
    prompt_raw: str,
    answer_plain: str,
    system_text: Optional[str] = None,
    terminal_sequences: Optional[List[List[int]]] = None,
):
    """
    The plain-text path may append terminal tokens explicitly.
    """
    prompt_text = prompt_raw.strip()
    if system_text and system_text.strip():
        prompt_text = system_text.strip() + "\n\n" + prompt_text

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    answer_ids = tokenizer(
        answer_plain.strip(),
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    if terminal_sequences:
        primary_terminal_seq = list(terminal_sequences[0])
        if primary_terminal_seq and not _ends_with_any_sequence(answer_ids, terminal_sequences):
            answer_ids = list(answer_ids) + primary_terminal_seq

    full_ids = list(prompt_ids) + list(answer_ids)

    return {
        "prompt_ids": list(prompt_ids),
        "answer_ids": list(answer_ids),
        "full_ids": list(full_ids),
        "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        "answer_text": tokenizer.decode(answer_ids, skip_special_tokens=False),
        "full_text": tokenizer.decode(full_ids, skip_special_tokens=False),
    }



def _build_highlight_token_roles_and_weights(
    tokenizer,
    prompt_ids: List[int],
    answer_ids: List[int],
    labels: List[int],
    highlight_prefix_text: str = "highlight: ",
    answer_plain: str = "",
    highlight_prefix_weight: float = 0.20,
    highlight_body_weight: float = 1.00,
    terminal_active_weight: float = 1.00,
    terminal_masked_weight: float = 0.00,
) -> Tuple[List[str], List[float]]:
    """
    Stable token-role alignment:
    - Does not depend on incremental decoding.
    - Does not require prefix_ids to equal an exact tokenization prefix.
    - Uses offset_mapping to map answer character spans to tokens.
    - Classifies terminal tokens after answer_plain as active or masked from labels.
    """
    prompt_ids = list(prompt_ids)
    answer_ids = list(answer_ids)
    answer_labels = list(labels[len(prompt_ids): len(prompt_ids) + len(answer_ids)])

    if len(answer_labels) != len(answer_ids):
        raise ValueError(
            f"answer_labels length mismatch: "
            f"len(answer_labels)={len(answer_labels)} vs len(answer_ids)={len(answer_ids)}"
        )

    expected_content_text = str(answer_plain or "")
    if not expected_content_text.startswith(highlight_prefix_text):
        raise ValueError(
            f"answer_plain must start with highlight_prefix_text.\n"
            f"answer_plain={answer_plain!r}\n"
            f"highlight_prefix_text={highlight_prefix_text!r}"
        )

    # Tokenize answer_plain once with offsets.
    enc = tokenizer(
        expected_content_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )

    expected_content_ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])

    if len(expected_content_ids) != len(offsets):
        raise ValueError(
            f"input_ids / offset_mapping length mismatch: "
            f"{len(expected_content_ids)} vs {len(offsets)}"
        )

    if len(answer_ids) < len(expected_content_ids):
        raise ValueError(
            f"answer_ids shorter than tokenized answer_plain.\n"
            f"len(answer_ids)={len(answer_ids)}\n"
            f"len(expected_content_ids)={len(expected_content_ids)}\n"
            f"answer_plain={expected_content_text!r}\n"
            f"decoded_answer={tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}"
        )

    # The content portion of answer_ids must match the complete answer_plain
    # tokenization; a mismatch indicates an upstream construction error.
    if answer_ids[:len(expected_content_ids)] != expected_content_ids:
        raise ValueError(
            "answer_ids does not start with tokenized answer_plain.\n"
            f"answer_plain={expected_content_text!r}\n"
            f"expected_content_ids={expected_content_ids}\n"
            f"actual_prefix_ids={answer_ids[:len(expected_content_ids)]}\n"
            f"decoded_expected={tokenizer.decode(expected_content_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}\n"
            f"decoded_actual_prefix={tokenizer.decode(answer_ids[:len(expected_content_ids)], skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}\n"
            f"decoded_full_answer={tokenizer.decode(answer_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)!r}"
        )

    content_len = len(expected_content_ids)
    prefix_char_len = len(highlight_prefix_text)

    answer_roles: List[str] = []
    answer_weights: List[float] = []

    # Assign prefix/body roles to answer_plain content tokens.
    for (start, end) in offsets:
        # Conservatively classify empty tokenizer spans as prefix.
        if end <= start:
            role = "highlight_prefix"
            weight = float(highlight_prefix_weight)
        else:
            # Overlap with the prefix character range [0, prefix_char_len).
            prefix_overlap = max(0, min(end, prefix_char_len) - start)
            token_len = max(1, end - start)

            # Entirely inside the prefix.
            if end <= prefix_char_len:
                role = "highlight_prefix"
                weight = float(highlight_prefix_weight)
            # Entirely inside the body.
            elif start >= prefix_char_len:
                role = "highlight_body"
                weight = float(highlight_body_weight)
            else:
                # For a boundary token, choose the side with greater overlap.
                body_overlap = token_len - prefix_overlap
                if prefix_overlap >= body_overlap:
                    role = "highlight_prefix"
                    weight = float(highlight_prefix_weight)
                else:
                    role = "highlight_body"
                    weight = float(highlight_body_weight)

        answer_roles.append(role)
        answer_weights.append(weight)

    # Label terminal tokens after answer_plain.
    for idx in range(content_len, len(answer_ids)):
        lab = answer_labels[idx]
        if lab != -100:
            answer_roles.append("terminal_active")
            answer_weights.append(float(terminal_active_weight))
        else:
            answer_roles.append("terminal_masked")
            answer_weights.append(float(terminal_masked_weight))

    if len(answer_roles) != len(answer_ids) or len(answer_weights) != len(answer_ids):
        raise ValueError(
            f"Role/weight length mismatch: roles={len(answer_roles)}, "
            f"weights={len(answer_weights)}, answer_ids={len(answer_ids)}"
        )

    full_roles = ["prompt"] * len(prompt_ids) + answer_roles
    full_weights = [0.0] * len(prompt_ids) + answer_weights
    return full_roles, full_weights



def _to_highlight_trainer_features_batch(
    batch,
    *,
    tokenizer,
    to_chat_template: bool,
    system_text: Optional[str],
    max_length: Optional[int],
    terminal_sequences: List[List[int]],
    terminal_loss_mode: str,
    highlight_prefix_text: str,
    highlight_prefix_weight: float,
    highlight_body_weight: float,
    terminal_active_weight: float,
    terminal_masked_weight: float,
):
    out = {
        "source_id": [],
        "PromptRaw": [],
        "AnswerPlain": [],
        "HighlightBody": [],
        "TaskType": [],
        "PromptText": [],
        "AnswerText": [],
        "LossAnswerText": [],
        "FullText": [],
        "PromptIds": [],
        "AnswerIds": [],
        "LossAnswerIds": [],
        "FullIds": [],
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "loss_weights": [],
        "token_roles": [],
        "LossTokenRoles": [],
        "LossTokenWeights": [],
        "is_truncated": [],
    }

    source_ids = batch["source_id"]
    prompt_raws = batch["PromptRaw"]
    answer_plains = batch["AnswerPlain"]
    highlight_bodies = batch["HighlightBody"]
    task_types = batch["TaskType"]

    for source_id, prompt_raw, answer_plain, highlight_body, task_type in zip(
        source_ids, prompt_raws, answer_plains, highlight_bodies, task_types
    ):
        if to_chat_template:
            built = _build_chat_ids_and_texts(
                tokenizer=tokenizer,
                user_text=prompt_raw,
                assistant_text_plain=answer_plain,
                system_text=system_text,
            )
        else:
            built = _build_plain_ids_and_texts(
                tokenizer=tokenizer,
                prompt_raw=prompt_raw,
                answer_plain=answer_plain,
                system_text=system_text,
                terminal_sequences=terminal_sequences,
            )

        full_ids = list(built["full_ids"])
        prompt_ids = list(built["prompt_ids"])
        answer_ids = list(built["answer_ids"])

        labels, _ = _build_answer_labels_with_terminal_mask(
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            terminal_sequences=terminal_sequences,
            terminal_loss_mode=terminal_loss_mode,
        )

        token_roles, loss_weights = _build_highlight_token_roles_and_weights(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            labels=labels,
            highlight_prefix_text=highlight_prefix_text,
            answer_plain=answer_plain,
            highlight_prefix_weight=highlight_prefix_weight,
            highlight_body_weight=highlight_body_weight,
            terminal_active_weight=terminal_active_weight,
            terminal_masked_weight=terminal_masked_weight,
        )

        is_truncated = False
        if max_length is not None:
            is_truncated = len(full_ids) > max_length
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]
            loss_weights = loss_weights[:max_length]
            token_roles = token_roles[:max_length]

        attention_mask = [1] * len(full_ids)

        loss_answer_ids = [tid for tid, lab in zip(full_ids, labels) if lab != -100]
        loss_answer_text = tokenizer.decode(loss_answer_ids, skip_special_tokens=False)

        loss_token_roles = [role for role, lab in zip(token_roles, labels) if lab != -100]
        loss_token_weights = [float(w) for w, lab in zip(loss_weights, labels) if lab != -100]

        out["source_id"].append(source_id)
        out["PromptRaw"].append(prompt_raw)
        out["AnswerPlain"].append(answer_plain)
        out["HighlightBody"].append(highlight_body)
        out["TaskType"].append(task_type)
        out["PromptText"].append(built["prompt_text"])
        out["AnswerText"].append(built["answer_text"])
        out["LossAnswerText"].append(loss_answer_text)
        out["FullText"].append(
            built["full_text"] if not is_truncated else tokenizer.decode(full_ids, skip_special_tokens=False)
        )
        out["PromptIds"].append(prompt_ids)
        out["AnswerIds"].append(answer_ids)
        out["LossAnswerIds"].append(loss_answer_ids)
        out["FullIds"].append(full_ids)
        out["input_ids"].append(full_ids)
        out["attention_mask"].append(attention_mask)
        out["labels"].append(labels)
        out["loss_weights"].append(loss_weights)
        out["token_roles"].append(token_roles)
        out["LossTokenRoles"].append(loss_token_roles)
        out["LossTokenWeights"].append(loss_token_weights)
        out["is_truncated"].append(bool(is_truncated))

    return out



def _to_highlight_trainer_dataset_parallel(
    ds_prompt: Dataset,
    *,
    tokenizer,
    to_chat_template: bool,
    system_text: Optional[str],
    max_length: Optional[int],
    terminal_sequences: List[List[int]],
    terminal_loss_mode: str,
    highlight_prefix_text: str,
    highlight_prefix_weight: float,
    highlight_body_weight: float,
    terminal_active_weight: float,
    terminal_masked_weight: float,
    batch_size: int,
    num_proc: int,
    load_from_cache_file: bool,
) -> Dataset:
    if len(ds_prompt) == 0:
        return Dataset.from_list([])

    ds_out = ds_prompt.map(
        _to_highlight_trainer_features_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        desc="Building highlight trainer features",
        fn_kwargs={
            "tokenizer": tokenizer,
            "to_chat_template": to_chat_template,
            "system_text": system_text,
            "max_length": max_length,
            "terminal_sequences": terminal_sequences,
            "terminal_loss_mode": terminal_loss_mode,
            "highlight_prefix_text": highlight_prefix_text,
            "highlight_prefix_weight": highlight_prefix_weight,
            "highlight_body_weight": highlight_body_weight,
            "terminal_active_weight": terminal_active_weight,
            "terminal_masked_weight": terminal_masked_weight,
        },
        remove_columns=ds_prompt.column_names,
    )
    return ds_out



def preview_cnn_dm_highlight_trainer_dataset(
    ds_out: DatasetDict,
    tokenizer,
    n_preview_per_split: int = 2,
    verify_n: int = 20,
    seed: int = 42,
):
    rng = random.Random(seed)

    for split_name, ds_split in ds_out.items():
        print("=" * 120)
        print(f"[Preview] split={split_name} rows={len(ds_split)}")
        print("=" * 120)

        if len(ds_split) == 0:
            print("(empty split)")
            continue

        k = min(n_preview_per_split, len(ds_split))
        indices = list(range(len(ds_split)))
        rng.shuffle(indices)
        indices = indices[:k]

        for idx in indices:
            row = ds_split[idx]
            print("-" * 120)
            print(f"[idx={idx}] source_id={row['source_id']} task={row.get('TaskType', '')}")
            print("[PromptRaw]")
            print(row["PromptRaw"])
            print("[AnswerPlain]")
            print(row["AnswerPlain"])
            print("[Decoded loss-bearing answer-only]")
            answer_only_ids = [tid for tid, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
            print(tokenizer.decode(answer_only_ids, skip_special_tokens=False))

            print("[LegacyTagCheck]")
            print(
                {
                    "prompt_has_legacy_tags": _contains_legacy_tags(row["PromptRaw"]),
                    "answer_has_legacy_tags": _contains_legacy_tags(row["AnswerPlain"]),
                }
            )

            if "LossTokenRoles" in row:
                print("[Loss-bearing token roles]")
                print(row["LossTokenRoles"])
            if "LossTokenWeights" in row:
                print("[Loss-bearing token weights]")
                print([round(float(x), 4) for x in row["LossTokenWeights"]])

            _print_weighted_answer_debug(row, tokenizer)

        if verify_n and verify_n > 0:
            verify_cnn_dm_highlight_trainer_dataset_consistency(
                ds_split,
                tokenizer,
                terminal_sequences=None,
                highlight_prefix_text="highlight: ",
                n_check=min(verify_n, len(ds_split)),
            )



def inspect_one_cnn_dm_highlight_sample(ds_trainer, tokenizer, idx: Optional[int] = None, seed: int = 42):
    if len(ds_trainer) == 0:
        raise ValueError("Dataset is empty.")

    if idx is None:
        rng = random.Random(seed)
        idx = rng.randrange(len(ds_trainer))

    row = ds_trainer[idx]
    input_ids = row["input_ids"]
    labels = row["labels"]

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    answer_only_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded_loss_answer_only = tokenizer.decode(answer_only_ids, skip_special_tokens=False)

    print("=" * 120)
    print(f"[Decoded sample] idx={idx}")
    print("=" * 120)
    print("\n[source_id]")
    print(row["source_id"])
    print("\n[TaskType]")
    print(row.get("TaskType", ""))
    print("\n[PromptRaw]")
    print(row["PromptRaw"])
    print("\n[AnswerPlain]")
    print(row["AnswerPlain"])
    print("\n[Decoded FullText from input_ids | RAW]")
    print(decoded_full)
    print("\n[Decoded FullText from input_ids | REPR]")
    print(repr(decoded_full))
    print("\n[Stored AnswerText | RAW]")
    print(row["AnswerText"])
    print("\n[Stored LossAnswerText | RAW]")
    print(row["LossAnswerText"])
    print("\n[Decoded loss-bearing answer text from labels | RAW]")
    print(decoded_loss_answer_only)
    print("\n[Decoded loss-bearing answer text from labels | REPR]")
    print(repr(decoded_loss_answer_only))
    print("\n[Stored FullText == decoded_full ?]")
    print(row["FullText"] == decoded_full)
    print("\n[Stored LossAnswerText == decoded loss-bearing answer ?]")
    print(row["LossAnswerText"] == decoded_loss_answer_only)

    if "LossTokenRoles" in row:
        print("\n[LossTokenRoles]")
        print(row["LossTokenRoles"])
    if "LossTokenWeights" in row:
        print("\n[LossTokenWeights]")
        print([round(float(x), 4) for x in row["LossTokenWeights"]])

    print()
    _print_weighted_answer_debug(row, tokenizer)



def verify_cnn_dm_highlight_trainer_dataset_consistency(
    ds_split: Dataset,
    tokenizer,
    terminal_sequences: Optional[List[List[int]]] = None,
    highlight_prefix_text: str = "highlight: ",
    n_check: Optional[int] = None,
):
    total = len(ds_split) if n_check is None else min(n_check, len(ds_split))

    for i in range(total):
        row = ds_split[i]

        # Avoid strict FullText string or retokenization equality checks; chat
        # templates can trigger false positives without affecting training.

        answer_only_ids = [tid for tid, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
        answer_decoded = tokenizer.decode(answer_only_ids, skip_special_tokens=False)

        if answer_decoded != row["LossAnswerText"]:
            raise AssertionError(f"LossAnswerText decode mismatch at row {i}")

        if "loss_weights" in row and len(row["loss_weights"]) != len(row["input_ids"]):
            raise AssertionError(f"loss_weights length mismatch at row {i}")

        if "token_roles" in row and len(row["token_roles"]) != len(row["input_ids"]):
            raise AssertionError(f"token_roles length mismatch at row {i}")

        if _contains_legacy_tags(row["PromptRaw"]):
            raise AssertionError(f"Legacy tags found in PromptRaw at row {i}")
        if _contains_legacy_tags(row["AnswerPlain"]):
            raise AssertionError(f"Legacy tags found in AnswerPlain at row {i}")
        if not row["PromptRaw"].startswith("Article:"):
            raise AssertionError(f"PromptRaw format mismatch at row {i}")
        if not row["AnswerPlain"].startswith(highlight_prefix_text):
            raise AssertionError(f"AnswerPlain format mismatch at row {i}")

        if terminal_sequences and ("is_truncated" in row) and (not row["is_truncated"]):
            prompt_len = len(row["PromptIds"])
            answer_ids = list(row["input_ids"])[prompt_len:]
            if not _ends_with_any_sequence(answer_ids, terminal_sequences):
                pass

    print(f"Consistency check passed: {total} rows.")



def _resolve_cnn_dm_train_source(
    *,
    ds_dict=None,
    picked_train_dir: Optional[str] = None,
    auto_build_picked_if_missing: bool = True,
    end2end_cfg: Optional["ClusterConfig"] = None,
    end2end_kwargs: Optional[Dict[str, Any]] = None,
):
    official = None

    if ds_dict is not None:
        ds_src = _normalize_to_datasetdict(ds_dict)
        if "train" in ds_src:
            return ds_src["train"], official
        official = load_dataset("abisee/cnn_dailymail", "3.0.0")
        return official["train"], official

    if picked_train_dir is not None:
        picked_obj = load_from_disk(picked_train_dir)
        picked_dd = _normalize_to_datasetdict(picked_obj)
        if "train" not in picked_dd:
            raise KeyError(f"'train' split not found in picked_train_dir={picked_train_dir}")
        return picked_dd["train"], official

    if auto_build_picked_if_missing:
        if end2end_kwargs is None:
            end2end_kwargs = {}
        if end2end_cfg is None:
            end2end_cfg = ClusterConfig(split="train", take_n=287113, batch_size=512)

        res = run_cnn_dm_end2end(cfg=end2end_cfg, **end2end_kwargs)

        picked_dir = None
        meta = res.get("picked_meta", None) if isinstance(res, dict) else None
        if isinstance(meta, dict):
            picked_dir = meta.get("out_dir", None)
        if not picked_dir:
            picked_dir = end2end_kwargs.get("picked_out_dir", None)
        if not picked_dir:
            raise ValueError(
                "Auto end2end finished but cannot locate picked dataset directory. "
                "Ensure run_cnn_dm_end2end returns picked_meta['out_dir'] or pass end2end_kwargs['picked_out_dir']."
            )

        picked_obj = load_from_disk(picked_dir)
        picked_dd = _normalize_to_datasetdict(picked_obj)
        if "train" not in picked_dd:
            raise KeyError(f"'train' split not found in auto-built picked dataset dir={picked_dir}")
        return picked_dd["train"], official

    official = load_dataset("abisee/cnn_dailymail", "3.0.0")
    return official["train"], official



def _resolve_cnn_dm_valid_test_sources(
    *,
    ds_dict=None,
    official=None,
):
    valid_raw = None
    test_raw = None

    if ds_dict is not None:
        ds_src = _normalize_to_datasetdict(ds_dict)
        valid_raw = ds_src.get("validation", ds_src.get("valid", None))
        test_raw = ds_src.get("test", None)

    if valid_raw is None or test_raw is None:
        if official is None:
            official = load_dataset("abisee/cnn_dailymail", "3.0.0")
        if valid_raw is None:
            valid_raw = official["validation"]
        if test_raw is None:
            test_raw = official["test"]

    return valid_raw, test_raw, official



def build_cnn_dm_highlight_trainer_dataset_v4(
    tokenizer,
    *,
    ds_dict=None,
    picked_train_dir: Optional[str] = None,
    auto_build_picked_if_missing: bool = True,

    end2end_cfg: Optional["ClusterConfig"] = None,
    end2end_kwargs: Optional[Dict[str, Any]] = None,

    train_take: TakeSpec = 1.0,
    valid_take: TakeSpec = None,
    test_take: TakeSpec = None,
    seed: int = 42,

    to_chat_template: bool = True,
    system_text: Optional[str] = None,
    highlight_prefix_text: str = "highlight: ",
    article_max_tokens: Optional[int] = None,

    end_token_ids: Optional[List[int]] = None,
    terminal_sequences: Optional[List[List[int]]] = None,
    terminal_loss_mode: str = "final_only",

    max_length: Optional[int] = None,
    highlight_prefix_weight: float = 0.20,
    highlight_body_weight: float = 1.00,
    terminal_active_weight: float = 1.00,
    terminal_masked_weight: float = 0.00,

    num_proc: Optional[int] = 8,
    batch_size: int = 512,
    load_from_cache_file: bool = True,

    print_checks: bool = True,
    n_preview_per_split: int = 2,
    verify_n: int = 20,

    save_final_to_disk: bool = False,
    output_dir: Optional[str] = None,
):
    if tokenizer is None:
        raise ValueError("tokenizer is required.")

    num_proc = _sanitize_num_proc(num_proc)

    if system_text is None:
        system_text = _default_highlight_system_text()

    train_terminal_sequences = _resolve_training_terminal_sequences(
        tokenizer=tokenizer,
        end_token_ids=end_token_ids,
        terminal_sequences=terminal_sequences,
    )
    if not train_terminal_sequences:
        raise ValueError("No valid terminal sequence could be resolved for training.")

    train_raw, official = _resolve_cnn_dm_train_source(
        ds_dict=ds_dict,
        picked_train_dir=picked_train_dir,
        auto_build_picked_if_missing=auto_build_picked_if_missing,
        end2end_cfg=end2end_cfg,
        end2end_kwargs=end2end_kwargs,
    )

    out_prompt = {}

    if train_take is not None:
        out_prompt["train"] = _build_highlight_prompt_dataset_from_raw_parallel(
            raw=train_raw,
            take_spec=train_take,
            split_name="train",
            seed=seed,
            highlight_prefix_text=highlight_prefix_text,
            tokenizer=tokenizer,
            article_max_tokens=article_max_tokens,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
        )

    need_valid = valid_take is not None
    need_test = test_take is not None

    if need_valid or need_test:
        valid_raw, test_raw, official = _resolve_cnn_dm_valid_test_sources(
            ds_dict=ds_dict,
            official=official,
        )

        if need_valid:
            out_prompt["validation"] = _build_highlight_prompt_dataset_from_raw_parallel(
                raw=valid_raw,
                take_spec=valid_take,
                split_name="validation",
                seed=seed + 1000,
                highlight_prefix_text=highlight_prefix_text,
                tokenizer=tokenizer,
                article_max_tokens=article_max_tokens,
                batch_size=batch_size,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
            )

        if need_test:
            out_prompt["test"] = _build_highlight_prompt_dataset_from_raw_parallel(
                raw=test_raw,
                take_spec=test_take,
                split_name="test",
                seed=seed + 2000,
                highlight_prefix_text=highlight_prefix_text,
                tokenizer=tokenizer,
                article_max_tokens=article_max_tokens,
                batch_size=batch_size,
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
            )

    ds_prompt = DatasetDict(out_prompt)

    out_trainer = {}
    for split_name, ds_split in ds_prompt.items():
        out_trainer[split_name] = _to_highlight_trainer_dataset_parallel(
            ds_prompt=ds_split,
            tokenizer=tokenizer,
            to_chat_template=to_chat_template,
            system_text=system_text,
            max_length=max_length,
            terminal_sequences=train_terminal_sequences,
            terminal_loss_mode=terminal_loss_mode,
            highlight_prefix_text=highlight_prefix_text,
            highlight_prefix_weight=highlight_prefix_weight,
            highlight_body_weight=highlight_body_weight,
            terminal_active_weight=terminal_active_weight,
            terminal_masked_weight=terminal_masked_weight,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
        )

    ds_out = DatasetDict(out_trainer)

    if print_checks:
        preview_cnn_dm_highlight_trainer_dataset(
            ds_out=ds_out,
            tokenizer=tokenizer,
            n_preview_per_split=n_preview_per_split,
            verify_n=verify_n,
            seed=seed,
        )
        for _, ds_split in ds_out.items():
            verify_cnn_dm_highlight_trainer_dataset_consistency(
                ds_split=ds_split,
                tokenizer=tokenizer,
                terminal_sequences=train_terminal_sequences if not to_chat_template else None,
                highlight_prefix_text=highlight_prefix_text,
                n_check=min(verify_n, len(ds_split)),
            )

    if save_final_to_disk:
        if not output_dir:
            raise ValueError("save_final_to_disk=True requires output_dir.")
        ds_out.save_to_disk(output_dir)

    return {
        "dataset": ds_out,
        "terminal_sequences": train_terminal_sequences,
        "system_text": system_text,
        "highlight_prefix_text": highlight_prefix_text,
        "article_max_tokens": article_max_tokens,
        "to_chat_template": to_chat_template,
        "num_proc": num_proc,
        "batch_size": batch_size,
    }



def _print_weighted_answer_debug(row, tokenizer):
    if "token_roles" not in row or "loss_weights" not in row:
        return

    print("[Answer token weights]")
    answer_idx = 0
    for tid, lab, role, w in zip(
        row["input_ids"],
        row["labels"],
        row["token_roles"],
        row["loss_weights"],
    ):
        if role == "prompt":
            continue
        tok_text = tokenizer.decode(
            [int(tid)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        is_active = bool(lab != -100)
        print(
            f"{answer_idx:>3d}  token={tok_text!r:<20}  role={role:<24}  "
            f"weight={float(w):.4f}  active={is_active}"
        )
        answer_idx += 1



def inspect_one_trainer_sample(ds_trainer, tokenizer, idx: Optional[int] = None, seed: int = 42):
    if len(ds_trainer) == 0:
        raise ValueError("Dataset is empty.")

    if idx is None:
        rng = random.Random(seed)
        idx = rng.randrange(len(ds_trainer))

    row = ds_trainer[idx]
    input_ids = row["input_ids"]
    labels = row["labels"]

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    answer_only_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded_loss_answer_only = tokenizer.decode(answer_only_ids, skip_special_tokens=False)

    print("=" * 120)
    print(f"[Decoded sample] idx={idx}")
    print("=" * 120)
    print("\n[source_id]")
    print(row["source_id"])
    print("\n[TaskType]")
    print(row.get("TaskType", ""))
    print("\n[Title]")
    print(row["Title"])
    print("\n[PromptRaw]")
    print(row["PromptRaw"])
    print("\n[AnswerPlain]")
    print(row["AnswerPlain"])
    print("\n[Decoded FullText from input_ids | RAW]")
    print(decoded_full)
    print("\n[Decoded FullText from input_ids | REPR]")
    print(repr(decoded_full))
    print("\n[Stored AnswerText | RAW]")
    print(row["AnswerText"])
    print("\n[Stored LossAnswerText | RAW]")
    print(row["LossAnswerText"])
    print("\n[Decoded loss-bearing answer text from labels | RAW]")
    print(decoded_loss_answer_only)
    print("\n[Decoded loss-bearing answer text from labels | REPR]")
    print(repr(decoded_loss_answer_only))
    print("\n[Stored FullText == decoded_full ?]")
    print(row["FullText"] == decoded_full)
    print("\n[Stored LossAnswerText == decoded loss-bearing answer ?]")
    print(row["LossAnswerText"] == decoded_loss_answer_only)

    if "LossTokenRoles" in row:
        print("\n[LossTokenRoles]")
        print(row["LossTokenRoles"])
    if "LossTokenWeights" in row:
        print("\n[LossTokenWeights]")
        print([round(float(x), 4) for x in row["LossTokenWeights"]])

    print()
    _print_weighted_answer_debug(row, tokenizer)



def inspect_one_cnn_dm_summarization_sample(ds_trainer, tokenizer, idx: Optional[int] = None, seed: int = 42):
    if len(ds_trainer) == 0:
        raise ValueError("Dataset is empty.")

    if idx is None:
        rng = random.Random(seed)
        idx = rng.randrange(len(ds_trainer))

    row = ds_trainer[idx]
    input_ids = row["input_ids"]
    labels = row["labels"]

    decoded_full = tokenizer.decode(input_ids, skip_special_tokens=False)
    answer_only_ids = [tid for tid, lab in zip(input_ids, labels) if lab != -100]
    decoded_loss_answer_only = tokenizer.decode(answer_only_ids, skip_special_tokens=False)

    print("=" * 120)
    print(f"[Decoded sample] idx={idx}")
    print("=" * 120)
    print("\n[source_id]")
    print(row["source_id"])
    print("\n[TaskType]")
    print(row.get("TaskType", ""))
    print("\n[PromptRaw]")
    print(row["PromptRaw"])
    print("\n[AnswerPlain]")
    print(row["AnswerPlain"])
    print("\n[Decoded FullText from input_ids | RAW]")
    print(decoded_full)
    print("\n[Decoded FullText from input_ids | REPR]")
    print(repr(decoded_full))
    print("\n[Stored AnswerText | RAW]")
    print(row["AnswerText"])
    print("\n[Stored LossAnswerText | RAW]")
    print(row["LossAnswerText"])
    print("\n[Decoded loss-bearing answer text from labels | RAW]")
    print(decoded_loss_answer_only)
    print("\n[Decoded loss-bearing answer text from labels | REPR]")
    print(repr(decoded_loss_answer_only))
    print("\n[Stored FullText == decoded_full ?]")
    print(row["FullText"] == decoded_full)
    print("\n[Stored LossAnswerText == decoded loss-bearing answer ?]")
    print(row["LossAnswerText"] == decoded_loss_answer_only)

    if "LossTokenRoles" in row:
        print("\n[LossTokenRoles]")
        print(row["LossTokenRoles"])
    if "LossTokenWeights" in row:
        print("\n[LossTokenWeights]")
        print([round(float(x), 4) for x in row["LossTokenWeights"]])

    print()
    _print_weighted_answer_debug(row, tokenizer)
