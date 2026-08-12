
"""KPTimes selection and trainer-dataset preparation."""

from __future__ import annotations

import copy
import csv
import glob
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, fields, is_dataclass, replace as dc_replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
except ImportError:
    Dataset = DatasetDict = None
    concatenate_datasets = load_dataset = load_from_disk = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
except ImportError:
    AutoModelForCausalLM = AutoTokenizer = pipeline = None

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, **_kwargs):
        return iterable

from .core import (
    _build_answer_labels_with_terminal_mask,
    _extract_input_ids,
    _resolve_default_terminal_sequences,
    _resolve_training_terminal_sequences,
    _unique_preserve_order,
    build_graph_edges,
    faiss_hnsw_knn,
    leiden_cluster,
    sample_by_clusters_scheme2,
    save_all_cluster_artifacts,
)

_SPECIAL_NUM_RE = re.compile(r"^(.*special)\d+$")


def load_kptimes_raw(cache_dir: Optional[str] = None):
    """Load the exact KPTimes source used by the original processing notebook."""
    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is required to load KPTimes.")
    return load_dataset(
        "midas/kptimes",
        revision="refs/convert/parquet",
        data_dir="raw",
        cache_dir=cache_dir,
    )



def _normalize_one_label(lbl: str) -> str:
    m = _SPECIAL_NUM_RE.match(lbl)
    return m.group(1) if m else lbl



def _as_str_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if str(v).strip()]
    s = str(x).strip()
    return [s] if s else []



def _dedup_keep_order_list(xs: List[str]) -> List[str]:
    if not xs:
        return []
    return list(dict.fromkeys(xs))



def _dedup_keep_order(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int64)
    seen = set()
    out = []
    for x in arr.tolist():
        if x not in seen:
            seen.add(x)
            out.append(x)
    return np.asarray(out, dtype=np.int64)



def _prepare_kptimes_core_batch(
    batch: Dict[str, Any],
    missing_cols: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Build (optionally) these columns in ONE map pass:
      - categories_list_raw
      - categories_list_norm
      - keywords_list
      - title_text
      - body_text

    If missing_cols is provided, only those keys are returned (avoid overwriting existing).
    """
    metas = batch.get("other_metadata", []) or []
    bs = len(metas)

    def _col_list(key: str):
        v = batch.get(key, None)
        if v is None:
            return [None] * bs
        return v

    title_cols = [
        _col_list("title"),
        _col_list("headline"),
        _col_list("document_title"),
        _col_list("name"),
    ]
    body_cols = [
        _col_list("abstract"),
        _col_list("body"),
        _col_list("article"),
        _col_list("text"),
        _col_list("content"),
    ]

    kw_top = batch.get("keywords", None)

    raw_out, norm_out, kw_out, title_out, body_out = [], [], [], [], []

    for i in range(bs):
        meta = metas[i] or {}

        # categories
        cats = _dedup_keep_order_list(_as_str_list(meta.get("categories", [])))
        raw_out.append(cats)

        if cats:
            norm = [_normalize_one_label(c) for c in cats]
            norm = _dedup_keep_order_list(norm)
        else:
            norm = []
        norm_out.append(norm)

        # keywords
        if kw_top is not None:
            kws = kw_top[i] if isinstance(kw_top, list) else kw_top
        else:
            kws = meta.get("keywords", meta.get("keyword", meta.get("keyphrases", [])))
        kws = _dedup_keep_order_list(_as_str_list(kws))
        kw_out.append(kws)

        # title
        t = ""
        for col in title_cols:
            v = col[i]
            if v is not None and str(v).strip():
                t = str(v).strip()
                break
        if not t:
            for k in ("title", "headline", "document_title", "name"):
                v = meta.get(k, None)
                if v is not None and str(v).strip():
                    t = str(v).strip()
                    break
        title_out.append(t)

        # body
        b = ""
        for col in body_cols:
            v = col[i]
            if v is not None and str(v).strip():
                b = str(v).strip()
                break
        if not b:
            for k in ("abstract", "body", "article", "text", "content"):
                v = meta.get(k, None)
                if v is not None and str(v).strip():
                    b = str(v).strip()
                    break
        body_out.append(b)

    out = {
        "categories_list_raw": raw_out,
        "categories_list_norm": norm_out,
        "keywords_list": kw_out,
        "title_text": title_out,
        "body_text": body_out,
    }

    if missing_cols is None:
        return out
    return {k: v for k, v in out.items() if k in missing_cols}



def kptimes_prepare_core_columns(
    ds,
    num_proc: int = 8,
    batch_size: int = 4096,
    load_from_cache_file: bool = True,
    desc: str = "[KPTime] prepare core columns (cats/kw/title/body)",
    ensure_orig_idx: bool = True,
):
    """
    ONE-TIME preparation step.

    Adds (if missing):
      - categories_list_raw, categories_list_norm
      - keywords_list
      - title_text, body_text
      - orig_idx (optional): sequential index of THIS dataset view

    IMPORTANT:
      - If ds is NOT aligned with ds_raw[split] row indices (e.g. arbitrary filtering),
        ensure_orig_idx=True will produce view indices, not raw indices.
        In that case provide a correct orig_idx externally and set ensure_orig_idx=False.
    """
    need = {"categories_list_raw", "categories_list_norm", "keywords_list", "title_text", "body_text"}
    missing = need.difference(set(ds.column_names))

    if missing:
        ds = ds.map(
            lambda batch: _prepare_kptimes_core_batch(batch, missing_cols=missing),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=desc,
            load_from_cache_file=load_from_cache_file,
        )

    if ensure_orig_idx and ("orig_idx" not in ds.column_names):
        ds = ds.add_column("orig_idx", list(range(len(ds))))

    return ds



def _build_repr_text_batch(
    batch,
    include_body: bool = True,
    body_max_words: int = 220,   # Use the first 220 words by default.
    body_max_chars: int = 0,     # Optional character limit; 0 disables it.
):
    """
    Natural-language, field-aware representation for BERT/SBERT embeddings:
      Title: ...
      Categories: c1; c2; ...
      Keywords: k1; k2; ...
      Body: ...

    The fields are joined into one line with " | ". This resembles natural
    text more closely than custom tags while preserving explicit field meaning.
    """
    # Determine batch length for batched map.
    try:
        n = len(next(iter(batch.values())))
    except StopIteration:
        return {"repr_text": []}

    titles = batch.get("title_text", [""] * n)
    cats_all = batch.get("categories_list_norm", [[]] * n)
    kws_all = batch.get("keywords_list", [[]] * n)
    bodies = batch.get("body_text", [""] * n)

    _ws = re.compile(r"\s+")

    def _clean(s: str) -> str:
        s = (s or "").strip()
        return _ws.sub(" ", s)

    def _truncate_words(text: str, max_words: int) -> str:
        text = _clean(text)
        if not text or max_words is None or max_words <= 0:
            return text
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).strip()

    def _join_list(xs) -> str:
        # Semicolons make multi-label fields explicit.
        xs = [ _clean(x) for x in (xs or []) if _clean(x) ]
        return "; ".join(xs)

    out = []
    for t, cats, kws, b in zip(titles, cats_all, kws_all, bodies):
        title = _clean(t)
        cats_s = _join_list(cats)
        kws_s  = _join_list(kws)

        parts = []
        if title:
            parts.append(f"Title: {title}")
        if cats_s:
            parts.append(f"Categories: {cats_s}")
        if kws_s:
            parts.append(f"Keywords: {kws_s}")

        if include_body:
            bb = _truncate_words(b, body_max_words)
            if bb:
                if body_max_chars and body_max_chars > 0 and len(bb) > body_max_chars:
                    bb = bb[:body_max_chars].rstrip()
                if bb:
                    parts.append(f"Body: {bb}")

        # Join fields into one line for direct embedding.
        out.append(" | ".join(parts).strip())

    return {"repr_text": out}



def compute_small_bucketing(
    ds_prepared,
    protect_n: int = 10,
    cat_col: str = "categories_list_norm",
    orig_idx_col: str = "orig_idx",
    drop_empty: bool = False,
    uncategorized_label: str = "__uncategorized__",
    verbose: bool = True,
    show_tqdm: bool = True,
    chunk_size: int = 100_000,   # Retained for compatibility; unused in V2.
    # Hugging Face parallelism settings.
    num_proc: int = 16,
    batch_size: int = 4096,
    load_from_cache_file: bool = True,
):


    if cat_col not in ds_prepared.column_names:
        raise KeyError(f"Missing '{cat_col}' in ds_prepared columns: {ds_prepared.column_names}")
    if orig_idx_col not in ds_prepared.column_names:
        raise KeyError(
            f"Missing '{orig_idx_col}'. Please run kptimes_prepare_core_columns(..., ensure_orig_idx=True) "
            f"or provide a correct orig_idx column yourself."
        )

    N = int(len(ds_prepared))

    # ------------------------------------------------------------------
    # Pass 0: build cats_use and empty_mask in parallel.
    # ------------------------------------------------------------------
    tmp_cats_use_col = "_kpt_cats_use"
    tmp_empty_col = "_kpt_is_empty"

    need_cols = set(ds_prepared.column_names)
    if tmp_cats_use_col in need_cols or tmp_empty_col in need_cols:
        # Rebuild temporary columns to avoid stale values from an earlier run.
        rm = [c for c in (tmp_cats_use_col, tmp_empty_col) if c in ds_prepared.column_names]
        ds_prepared = ds_prepared.remove_columns(rm)

    def _build_cats_use_batch(batch):
        cats_all = batch.get(cat_col, []) or []
        out_cats_use = []
        out_empty = []
        for cats in cats_all:
            is_empty = (not cats) or (len(cats) == 0)
            out_empty.append(bool(is_empty))
            if is_empty:
                if drop_empty:
                    out_cats_use.append([])   # primary_label will be empty.
                else:
                    out_cats_use.append([uncategorized_label])
            else:
                out_cats_use.append(cats)
        return {tmp_cats_use_col: out_cats_use, tmp_empty_col: out_empty}

    ds_tmp = ds_prepared.map(
        _build_cats_use_batch,
        batched=True,
        batch_size=int(batch_size),
        num_proc=int(num_proc),
        desc="[KPTime] Pass0 build cats_use + empty_mask" if show_tqdm else None,
        load_from_cache_file=bool(load_from_cache_file),
    )

    empty_mask = np.asarray(ds_tmp[tmp_empty_col], dtype=bool)

    # ------------------------------------------------------------------
    # Pass 1: global raw counts via pandas explode and value_counts.
    # ------------------------------------------------------------------
    cats_use_all = ds_tmp[tmp_cats_use_col]  # Materialized once.
    # Empty lists become NaN after explode and are removed by dropna.
    ser_raw = pd.Series(cats_use_all, copy=False).explode()
    ser_raw = ser_raw.dropna()
    raw_counts = ser_raw.value_counts(dropna=False)
    # dict[label] -> int count
    raw_count_dict = raw_counts.astype("int64").to_dict()

    # ------------------------------------------------------------------
    # Pass 2: compute primary_label in parallel; choose the globally rarest
    # label within each sample for the small-category rule.
    # ------------------------------------------------------------------
    primary_col = "primary_label"
    if primary_col in ds_tmp.column_names:
        ds_tmp = ds_tmp.remove_columns([primary_col])

    # num_proc requires a picklable closure, so use a plain dictionary.
    def _assign_primary_batch(batch, raw_count_dict=raw_count_dict):
        cats_use = batch.get(tmp_cats_use_col, []) or []
        prim_out = []
        for cats in cats_use:
            if not cats:
                prim_out.append("")  # Empty category when drop_empty=True.
                continue
            # Choose the globally rarest label; break ties lexicographically.
            best = None
            best_key = None
            for lbl in cats:
                k = (int(raw_count_dict.get(lbl, 0)), str(lbl))
                if best_key is None or k < best_key:
                    best_key = k
                    best = lbl
            prim_out.append(str(best) if best is not None else "")
        return {primary_col: prim_out}

    ds_tmp = ds_tmp.map(
        _assign_primary_batch,
        batched=True,
        batch_size=int(batch_size),
        num_proc=int(num_proc),
        desc="[KPTime] Pass2 assign primary_label (HF map)" if show_tqdm else None,
        load_from_cache_file=bool(load_from_cache_file),
    )

    # ------------------------------------------------------------------
    # Compute statistics and buckets in one pandas pass.
    # ------------------------------------------------------------------
    prim = ds_tmp[primary_col]                # list[str]
    orig_idx = np.asarray(ds_tmp[orig_idx_col], dtype=np.int64)

    df = pd.DataFrame(
        {orig_idx_col: orig_idx, primary_col: prim},
        copy=False
    )
    df_valid = df[df[primary_col] != ""]

    # small_counter
    vc_small = df_valid[primary_col].value_counts()
    small_counter = Counter(vc_small.astype("int64").to_dict())

    # protected/big labels
    protected_labels = set(vc_small[vc_small < int(protect_n)].index.astype(str).tolist())
    big_labels = vc_small[vc_small >= int(protect_n)].index.astype(str).tolist()
    # Deterministic order: count descending, label ascending.
    big_labels.sort(key=lambda x: (-int(small_counter.get(x, 0)), str(x)))

    # protected_idx and big_idx_all preserve df_valid order.
    mask_prot = df_valid[primary_col].isin(protected_labels).to_numpy()
    protected_idx = df_valid.loc[mask_prot, orig_idx_col].to_numpy(dtype=np.int64)
    big_idx_all = df_valid.loc[~mask_prot, orig_idx_col].to_numpy(dtype=np.int64)

    # Map each primary label to original indices while preserving group order.
    # sort=False preserves first-seen label order.
    cat2idx_np = {
        str(k): g[orig_idx_col].to_numpy(dtype=np.int64)
        for k, g in df_valid.groupby(primary_col, sort=False)
    }

    # Return primary_label_of in dataset-view order with length N.
    primary_label_of = prim

    if verbose:
        assigned = sum(int(len(v)) for v in cat2idx_np.values())
        kept = int(np.sum(~empty_mask)) if drop_empty else int(N)
        print(f"[SANITY] N={N}  assigned_once={assigned}  "
              f"{'nonempty_only' if drop_empty else 'including_empty'}={kept}")
        print(f"[BUCKET] unique_primary_labels={len(cat2idx_np)}  "
              f"protected_labels(<{protect_n})={len(protected_labels)}  big_labels={len(big_labels)}")
        if drop_empty:
            print(f"[WARN] drop_empty=True -> dropped empty-category rows: {int(np.sum(empty_mask))}")
        else:
            if int(np.sum(empty_mask)) > 0:
                print(f"[INFO] empty-category rows assigned to '{uncategorized_label}': {int(np.sum(empty_mask))}")

    # Preserve the original Counter return type.
    raw_counter = Counter(raw_count_dict)

    return {
        "primary_label_of": primary_label_of,
        "raw_counter": raw_counter,
        "small_counter": small_counter,
        "cat2idx": cat2idx_np,
        "protected_labels": protected_labels,
        "big_labels": big_labels,
        "protected_idx": protected_idx,
        "big_idx_all": big_idx_all,
        "orig_idx": orig_idx,
        "empty_mask": empty_mask,
    }



def small_counter_to_df(small_counter: Counter) -> pd.DataFrame:
    if not small_counter:
        return pd.DataFrame(columns=["label", "count", "share", "cumulative_share"])
    df = (
        pd.DataFrame(list(small_counter.items()), columns=["label", "count"])
          .sort_values(["count", "label"], ascending=[False, True])
          .reset_index(drop=True)
    )
    total = int(df["count"].sum())
    df["share"] = df["count"] / max(total, 1)
    df["cumulative_share"] = df["share"].cumsum()
    return df



def _bucketing_from_primary_label(
    ds_prepared,
    protect_n: int,
    primary_col: str = "primary_label",
    orig_idx_col: str = "orig_idx",
    verbose: bool = True,
):
    import numpy as np
    import pandas as pd
    from collections import Counter

    if primary_col not in ds_prepared.column_names:
        raise KeyError(f"Missing '{primary_col}' in ds_prepared.")
    if orig_idx_col not in ds_prepared.column_names:
        raise KeyError(f"Missing '{orig_idx_col}' in ds_prepared.")

    prim = ds_prepared[primary_col]
    orig = np.asarray(ds_prepared[orig_idx_col], dtype=np.int64)

    df = pd.DataFrame({orig_idx_col: orig, primary_col: prim}, copy=False)
    df_valid = df[df[primary_col] != ""]

    vc = df_valid[primary_col].value_counts()
    small_counter = Counter(vc.astype("int64").to_dict())

    protected_labels = set(vc[vc < int(protect_n)].index.astype(str).tolist())
    big_labels = vc[vc >= int(protect_n)].index.astype(str).tolist()
    big_labels.sort(key=lambda x: (-int(small_counter.get(x, 0)), str(x)))

    mask_prot = df_valid[primary_col].isin(protected_labels).to_numpy()
    protected_idx = df_valid.loc[mask_prot, orig_idx_col].to_numpy(dtype=np.int64)
    big_idx_all = df_valid.loc[~mask_prot, orig_idx_col].to_numpy(dtype=np.int64)

    cat2idx = {
        str(k): g[orig_idx_col].to_numpy(dtype=np.int64)
        for k, g in df_valid.groupby(primary_col, sort=False)
    }

    if verbose:
        assigned = sum(len(v) for v in cat2idx.values())
        print(f"[BUCKET/FAST] N={len(ds_prepared)} assigned_once={assigned} "
              f"protected_labels(<{protect_n})={len(protected_labels)} big_labels={len(big_labels)}")

    return {
        "primary_label_of": prim,                # list[str], length N
        "raw_counter": Counter(),                # Compatibility placeholder.
        "small_counter": small_counter,
        "cat2idx": cat2idx,
        "protected_labels": protected_labels,
        "big_labels": big_labels,
        "protected_idx": protected_idx,
        "big_idx_all": big_idx_all,
        "orig_idx": orig,
        "empty_mask": None,                      # Optional; unused downstream.
    }



def adapt_cluster_cfg_by_n(
    cfg,
    n: int,
    quality_n_threshold: Optional[int] = 10_000,   # Default table threshold.
    prefer_side: str = "max",                      # Prefer maximum-side values.
    adapt_log: Optional[bool] = None,
    return_reco: bool = False,                     # True => (new_cfg, reco_dict)
):
    """
    Keep maximum-side output parameters aligned with the four-range table.

    Four ranges:
      - n < 200
      - 200 <= n < 1500
      - 1500 <= n < 10000
      - n >= 10000

    Notes:
      - The function first computes recommended (min, max) values.
      - It writes the side selected by prefer_side into cfg.
      - knn_mem_gib assumes float32 scores plus int64 neighbors: 12 bytes/entry.
    """
    n = int(n)
    if n <= 2:
        new = copy.deepcopy(cfg)
        # Keep tiny buckets legal.
        top_k = max(1, n - 1)
        _assign_cfg_fields(new, {
            "top_k": top_k,
            "max_edges_per_node": 1,
            "attach_neigh_k": 1,
            "pair_k": 1,
            "use_mutual": False,
            "q_sim": min(int(getattr(cfg, "q_sim", 90)), 90),
            "resolution_list": (1.0,),
            "ef_search": max(top_k, min(n - 1, int(getattr(cfg, "ef_search", top_k)))),
            "ef_construction": max(2, min(n - 1, int(getattr(cfg, "ef_construction", 128)))),
        })
        return (new, {}) if return_reco else new

    if adapt_log is None:
        adapt_log = bool(getattr(cfg, "adapt_log", True))

    prefer_side = (prefer_side or "max").strip().lower()
    if prefer_side not in ("min", "max"):
        prefer_side = "max"

    # ---------------------------
    # 1) Compute recommended ranges from n_range.
    # ---------------------------
    reco = _recommend_params_by_n(n)

    # Choose the requested side.
    pick = {}
    for k, (vmin, vmax) in reco["ranges"].items():
        pick[k] = vmax if prefer_side == "max" else vmin

    # Select one concrete resolution list from the tabulated range.
    pick["resolution_list"] = reco["resolution_max"] if prefer_side == "max" else reco["resolution_min"]

    # Estimated KNN memory in GiB: n * top_k * 12 / 2^30.
    top_k_for_mem = pick["top_k"]
    pick["knn_mem_gib_est"] = (n * top_k_for_mem * 12) / (1024 ** 3)

    # ---------------------------
    # 2) Write values into a dataclass or ordinary object.
    # ---------------------------
    new_cfg = _apply_updates_to_cfg(cfg, pick)

    # ---------------------------
    # 3) Print an audit summary.
    # ---------------------------
    if adapt_log:
        r = reco["n_range"]
        mode = reco["mode_default"]
        print(
            f"[ADAPT_TABLE_MATCH] n={n} n_range={r} mode={mode} prefer={prefer_side} "
            f"top_k={getattr(new_cfg,'top_k',None)} "
            f"edges={getattr(new_cfg,'max_edges_per_node',None)} "
            f"attach={getattr(new_cfg,'attach_neigh_k',None)} "
            f"pair={getattr(new_cfg,'pair_k',None)} "
            f"q={getattr(new_cfg,'q_sim',None)} mutual={getattr(new_cfg,'use_mutual',None)} "
            f"ef_s={getattr(new_cfg,'ef_search',None)} ef_c={getattr(new_cfg,'ef_construction',None)} "
            f"res={getattr(new_cfg,'resolution_list',None)} "
            f"knn_mem_gib≈{pick['knn_mem_gib_est']:.3f}"
        )

    return (new_cfg, reco) if return_reco else new_cfg



def _ceil(x: float) -> int:
    return int(math.ceil(float(x)))



def _round_int(x: float) -> int:
    # Python uses banker's rounding, for example 526.5 -> 526.
    return int(round(float(x)))



def _clip_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(int(x), hi))



def _recommend_params_by_n(n: int) -> Dict[str, Any]:
    """
    Apply the table's four-range rules and return (min, max) per field.
    """
    # ---- bucket split
    if n < 200:
        n_range = "n<200"
        mode_default = "precise"

        top_k_min = _ceil(0.6 * n)
        top_k_max = min(n - 1, 256)

        # Minimum values depend on top_k_min; maximum values use top_k_max.
        edges_min = max(8, _ceil(0.25 * top_k_min))
        edges_max = min(top_k_max - 1, 64)

        attach_min = max(8, _ceil(0.10 * top_k_min))
        attach_max = min(top_k_max - 1, 32)

        pair_min = max(16, _ceil(0.50 * top_k_min))
        pair_max = top_k_max - 1

        ef_s_min = top_k_min
        ef_s_max = min(n - 1, 2 * top_k_max)

        ef_c_min = min(top_k_min, n - 1)
        ef_c_max = min(n - 1, max(64, 4 * top_k_max))

        q_min, q_max = 75, 90
        use_mutual = False

        res_min = (0.8, 1.0, 1.2)
        res_max = (1.0, 1.2, 1.4)

    elif n < 1500:
        n_range = "200≤n<1500"
        mode_default = "precise"

        top_k_min = max(128, _ceil(0.5 * n))
        top_k_max = min(n - 1, 2048)

        edges_min = max(16, _ceil(0.15 * top_k_min))
        edges_max = min(top_k_max - 1, 256)

        attach_min = max(16, _ceil(0.08 * top_k_min))
        attach_max = min(top_k_max - 1, 128)

        pair_min = max(32, _ceil(0.20 * top_k_min))
        pair_max = min(top_k_max - 1, 512)

        ef_s_min = top_k_min
        ef_s_max = n - 1  # This range uses n-1, usually equal to top_k_max.

        # This range uses min=128 and caps round(0.40*top_k_max) at 512.
        ef_c_min = 128
        ef_c_max = min(n - 1, max(128, min(512, _round_int(0.40 * top_k_max))))

        q_min, q_max = 80, 92
        use_mutual = False

        res_min = (1.0, 1.2, 1.4)
        res_max = (1.2, 1.5, 1.8)

    elif n < 10000:
        n_range = "1500≤n<10000"
        mode_default = "precise"

        top_k_min = max(1024, _ceil(0.4 * n))
        top_k_max = min(n - 1, 6144)

        edges_min = max(64, _ceil(0.10 * top_k_min))
        edges_max = min(top_k_max - 1, min(1024, _round_int(0.30 * top_k_max)))

        attach_min = max(32, _ceil(0.05 * top_k_min))
        attach_max = min(top_k_max - 1, min(512, _round_int(0.15 * top_k_max)))

        pair_min = max(128, _ceil(0.15 * top_k_min))
        # Most rows use floor(0.35*top_k_max).
        pair_max = min(top_k_max - 1, min(2048, int(0.35 * top_k_max)))

        ef_s_min = top_k_min
        ef_s_max = n - 1

        # The maximum ef_construction approximates round(0.5*top_k_max),
        # capped at 2048; isolated rows may differ by one.
        ef_c_min = max(128, edges_min)
        ef_c_max = min(n - 1, min(2048, _round_int(0.50 * top_k_max)))

        q_min, q_max = 88, 95
        use_mutual = True

        res_min = (1.2, 1.5, 1.8)
        res_max = (1.6, 2.0, 2.4)

    else:
        n_range = "n≥10000"
        mode_default = "medium"

        top_k_min = max(2048, _ceil(0.12 * n))
        top_k_max = min(n - 1, 8192)

        edges_min = max(256, _ceil(0.08 * top_k_min))
        edges_max = min(top_k_max - 1, min(1024, _ceil(0.25 * top_k_max)))

        attach_min = max(64, _ceil(0.03 * top_k_min))
        attach_max = min(top_k_max - 1, min(512, _ceil(0.12 * top_k_max)))

        pair_min = max(512, _ceil(0.12 * top_k_min))
        pair_max = min(top_k_max - 1, min(2048, _ceil(0.30 * top_k_max)))

        ef_s_min = top_k_min
        ef_s_max = min(n - 1, 2 * top_k_max)

        ef_c_min = max(128, _ceil(0.04 * top_k_min))
        ef_c_max = min(n - 1, min(512, _ceil(0.15 * top_k_max)))

        q_min, q_max = 88, 95
        use_mutual = True

        res_min = (1.6, 1.8, 2.0)
        res_max = (1.8, 2.2, 2.6)

    # Clip values to legal ranges.
    top_k_min = _clip_int(top_k_min, 2, n - 1)
    top_k_max = _clip_int(top_k_max, 2, n - 1)
    if top_k_min > top_k_max:
        top_k_min, top_k_max = top_k_max, top_k_min

    def _cap_by_topk(v: int, topk: int) -> int:
        return _clip_int(v, 1, max(1, topk - 1))

    # Re-clip all top_k-dependent fields against their corresponding bounds.
    edges_min = _cap_by_topk(edges_min, top_k_min)
    attach_min = _cap_by_topk(attach_min, top_k_min)
    pair_min = _cap_by_topk(pair_min, top_k_min)
    ef_s_min = _clip_int(ef_s_min, top_k_min, n - 1)
    ef_c_min = _clip_int(ef_c_min, 1, n - 1)

    edges_max = _cap_by_topk(edges_max, top_k_max)
    attach_max = _cap_by_topk(attach_max, top_k_max)
    pair_max = _cap_by_topk(pair_max, top_k_max)
    ef_s_max = _clip_int(ef_s_max, top_k_max, n - 1)
    ef_c_max = _clip_int(ef_c_max, 1, n - 1)

    ranges = {
        "top_k": (top_k_min, top_k_max),
        "max_edges_per_node": (edges_min, edges_max),
        "attach_neigh_k": (attach_min, attach_max),
        "pair_k": (pair_min, pair_max),
        "ef_search": (ef_s_min, ef_s_max),
        "ef_construction": (ef_c_min, ef_c_max),
        "q_sim": (q_min, q_max),
        "use_mutual": (use_mutual, use_mutual),  # A recommendation, not a range.
    }

    return {
        "n_range": n_range,
        "mode_default": mode_default,
        "ranges": ranges,
        "resolution_min": res_min,
        "resolution_max": res_max,
    }



def _apply_updates_to_cfg(cfg, updates: Dict[str, Any]):
    """
    Support dataclasses and ordinary objects:
      - Replace declared dataclass fields and use setattr for extras.
      - Deep-copy ordinary objects and set attributes.
    """
    # Normalize alternate configuration field names.
    updates = _normalize_field_aliases(cfg, updates)

    if is_dataclass(cfg) and not isinstance(cfg, type):
        field_names = {f.name for f in fields(cfg)}
        kw = {k: v for k, v in updates.items() if k in field_names}
        new_cfg = dc_replace(cfg, **kw)
        # Attempt setattr for object attributes not declared as dataclass fields.
        _assign_cfg_fields(new_cfg, {k: v for k, v in updates.items() if k not in field_names})
        return new_cfg

    new_cfg = copy.deepcopy(cfg)
    _assign_cfg_fields(new_cfg, updates)
    return new_cfg



def _assign_cfg_fields(obj, kv: Dict[str, Any]):
    for k, v in kv.items():
        try:
            if hasattr(obj, k) or not is_dataclass(obj):
                setattr(obj, k, v)
        except Exception:
            pass



def _normalize_field_aliases(cfg, updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Common fields include use_mutual, max_edges_per_node, attach_neigh_k,
    pair_k, q_sim, ef_search, and ef_construction. Add aliases as needed.
    """
    out = dict(updates)

    # use_mutual may also be named use_mutual_reco or mutual.
    if "use_mutual" in out:
        if hasattr(cfg, "use_mutual_reco") and not hasattr(cfg, "use_mutual"):
            out["use_mutual_reco"] = out["use_mutual"]
        if hasattr(cfg, "mutual") and not hasattr(cfg, "use_mutual"):
            out["mutual"] = out["use_mutual"]

    return out



def embed_texts_e5_once(
    texts: List[str],
    cfg,
    model: Optional[SentenceTransformer] = None,
) -> Tuple[np.ndarray, SentenceTransformer]:
    if model is None:
        model = SentenceTransformer(cfg.model_name, device=cfg.device)
        if hasattr(cfg, "max_seq_length") and getattr(cfg, "max_seq_length", None):
            model.max_seq_length = cfg.max_seq_length

    pref = getattr(cfg, "prefix", "")
    if pref:
        texts = [pref + t for t in texts]

    emb = model.encode(
        texts,
        batch_size=int(getattr(cfg, "batch_size", 512)),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    return emb, model



def cluster_embeddings_leiden(
    emb: np.ndarray,
    cfg,
    seed: int = 42,
    prefer_side: str = "max",
) -> np.ndarray:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    prefer_side = str(prefer_side).strip().lower()
    if prefer_side not in {"min", "max"}:
        raise ValueError("prefer_side must be 'min' or 'max'")

    n = int(emb.shape[0])
    cfg_cat = adapt_cluster_cfg_by_n(cfg, n, prefer_side=prefer_side)

    scores, nbrs = faiss_hnsw_knn(emb, cfg_cat)
    pairs, weights, _sim_th = build_graph_edges(nbrs, scores, cfg_cat)
    membership = leiden_cluster(n, pairs, weights, cfg_cat, seed=seed)
    # membership = attach_singletons_A(membership, nbrs, scores, cfg_cat)
    # membership = attach_singletons_B(membership, nbrs, scores, cfg_cat)
    return np.asarray(membership, dtype=np.int32)



def _sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()



def _json_sha1(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha1_hex(s.encode("utf-8"))



def _repr_signature(
    cfg,
    include_body: bool,
    body_max_words: int,
    body_max_chars: int,
) -> str:
    """
    Signature for embedding cache dir (avoid mixing caches across repr/model settings).
    """
    key = {
        "model_name": str(getattr(cfg, "model_name", "")),
        "max_seq_length": int(getattr(cfg, "max_seq_length", 0)),
        "prefix": str(getattr(cfg, "prefix", "")),
        "include_body": bool(include_body),
        "body_max_words": int(body_max_words) if body_max_words is not None else 0,
        "body_max_chars": int(body_max_chars),
    }
    return _json_sha1(key)[:16]



def _auto_prepared_dir(cache_root: str, split: str) -> str:
    return os.path.join(cache_root, "prepared", split)



def _auto_emb_dir(cache_root: str, split: str, repr_sig: str) -> str:
    return os.path.join(cache_root, "emb", split, repr_sig)



def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def _validate_meta(expected: Dict[str, Any], cached: Optional[Dict[str, Any]], strict: bool, who: str) -> None:
    if cached is None:
        return
    mismatch = []
    for k, v in expected.items():
        if cached.get(k) != v:
            mismatch.append(k)
    if mismatch:
        msg = (
            f"[{who}/WARN] meta mismatch keys={mismatch}\n"
            f"  expected={ {k: expected[k] for k in mismatch} }\n"
            f"  cached  ={ {k: cached.get(k) for k in mismatch} }"
        )
        if strict:
            raise ValueError(msg + f"\n(set overwrite_{who.lower()}=True to rebuild)")
        else:
            print(msg)



def _big_idx_hash(big_idx_all: np.ndarray) -> str:
    big_idx_all = np.asarray(big_idx_all, dtype=np.int64)
    return _sha1_hex(big_idx_all.tobytes())



def _try_load_emb_cache(
    emb_dir: str,
    expected_meta: dict,
    big_idx_all: np.ndarray,
    strict_meta: bool = True,
):
    """
    Reuse a superset cache for a subset:
    - Ignore big_len and big_idx_sha1 during metadata checks.
    - Require big_idx_all to be a subset of idx_cached.
    - Return emb_big aligned with big_idx_all.
    """
    if not emb_dir or (not os.path.isdir(emb_dir)):
        return None

    meta_path = os.path.join(emb_dir, "meta.json")
    emb_path  = os.path.join(emb_dir, "emb_big.npy")
    idx_path  = os.path.join(emb_dir, "big_idx_all.npy")

    if (not os.path.isfile(meta_path)) or (not os.path.isfile(emb_path)) or (not os.path.isfile(idx_path)):
        return None

    try:
        cached_meta = _read_json(meta_path)

        # Validate representation/model fields only.
        ignore = {"big_len", "big_idx_sha1"}
        exp_core = {k: v for k, v in expected_meta.items() if k not in ignore}
        cch_core = {k: v for k, v in cached_meta.items() if k not in ignore}
        _validate_meta(exp_core, cch_core, strict_meta, who="EMB")

        idx_cached = np.load(idx_path)
        emb_cached = np.load(emb_path)

        idx_cached = np.asarray(idx_cached, dtype=np.int64)
        emb_cached = np.asarray(emb_cached, dtype=np.float32)

        if idx_cached.ndim != 1 or emb_cached.ndim != 2 or emb_cached.shape[0] != idx_cached.shape[0]:
            return None
        if len(np.unique(idx_cached)) != idx_cached.size:
            raise ValueError("[EMB] cached idx contains duplicates; cannot build stable mapping.")

        big_idx_all = np.asarray(big_idx_all, dtype=np.int64)
        if big_idx_all.size == 0:
            return emb_cached[:0]

        # Check the subset and gather with searchsorted in O(N log N).
        order = np.argsort(idx_cached)
        idx_sorted = idx_cached[order]

        pos = np.searchsorted(idx_sorted, big_idx_all)
        ok = (pos < idx_sorted.size) & (idx_sorted[pos] == big_idx_all)
        if not np.all(ok):
            # A non-subset cannot reuse this cache.
            return None

        rows = order[pos]               # rows in original emb_cached
        emb_sub = emb_cached[rows]      # same order as big_idx_all
        return emb_sub

    except Exception as e:
        # Treat any read or validation error as a cache miss.
        print(f"[EMB] cache miss at {emb_dir}: {type(e).__name__}: {e}")
        return None



def _save_emb_cache(
    emb_dir: str,
    emb_big: np.ndarray,
    big_idx_all: np.ndarray,
    meta: dict,
    overwrite: bool = False,
):
    """
    Protect superset caches from being overwritten by smaller subsets.
    """
    os.makedirs(emb_dir, exist_ok=True)

    meta_path = os.path.join(emb_dir, "meta.json")
    emb_path  = os.path.join(emb_dir, "emb_big.npy")
    idx_path  = os.path.join(emb_dir, "big_idx_all.npy")

    emb_big = np.asarray(emb_big, dtype=np.float32)
    big_idx_all = np.asarray(big_idx_all, dtype=np.int64)

    # Precompute large-set metadata for this run.
    meta = dict(meta)
    meta["big_len"] = int(big_idx_all.size)
    meta["big_idx_sha1"] = _big_idx_hash(big_idx_all)

    # Decide whether to retain or upgrade an existing cache.
    if (not overwrite) and os.path.isfile(meta_path) and os.path.isfile(emb_path) and os.path.isfile(idx_path):
        try:
            cached_meta = _read_json(meta_path)

            # Validate core metadata while ignoring set-size fields.
            ignore = {"big_len", "big_idx_sha1"}
            exp_core = {k: v for k, v in meta.items() if k not in ignore}
            cch_core = {k: v for k, v in cached_meta.items() if k not in ignore}
            _validate_meta(exp_core, cch_core, True, who="EMB")

            idx_cached = np.asarray(np.load(idx_path), dtype=np.int64)
            if len(np.unique(idx_cached)) != idx_cached.size:
                raise ValueError("[EMB] cached idx duplicates")

            # Use searchsorted to compare set inclusion.
            o = np.argsort(idx_cached)
            idx_sorted = idx_cached[o]

            def _is_subset(a: np.ndarray, sup_sorted: np.ndarray) -> bool:
                if a.size == 0:
                    return True
                pos = np.searchsorted(sup_sorted, a)
                ok = (pos < sup_sorted.size) & (sup_sorted[pos] == a)
                return bool(np.all(ok))

            cached_is_superset = _is_subset(big_idx_all, idx_sorted)   # new ⊆ cached ?
            if cached_is_superset:
                # Retain an existing superset cache.
                print(f"[EMB] skip save (cached is superset): {emb_dir}")
                return

            # Upgrade when the new cache is a larger superset.
            new_sorted = np.sort(big_idx_all)
            new_is_superset = _is_subset(idx_cached, new_sorted)       # cached ⊆ new ?
            if new_is_superset:
                print(f"[EMB] upgrade cache (new is superset): {emb_dir}")
            else:
                # Skip incomparable sets unless overwrite is requested.
                print(f"[EMB] skip save (idx not comparable, overwrite=False): {emb_dir}")
                return

        except Exception as e:
            # Preserve the cache on metadata mismatch or read errors.
            print(f"[EMB] skip save (existing cache incompatible): {type(e).__name__}: {e}")
            return

    # Write when overwriting or upgrading.
    np.save(emb_path, emb_big.astype(np.float32, copy=False))
    np.save(idx_path, big_idx_all.astype(np.int64, copy=False))
    _write_json(meta_path, meta)
    print(f"[EMB] saved embedding cache to: {emb_dir}")



def build_kptimes_dedup_dataset_v2(
    cfg,                       # ClusterConfig
    split: str = "train",
    protect_n: int = 40,
    seed: int = 42,
    cluster_prefer_side: str = "max",

    ds_raw=None,               # DatasetDict
    ds_prepared=None,          # prepared HF Dataset (one split)

    # preparation
    num_proc: int = 8,
    batch_size: int = 4096,
    cache_ok: bool = True,
    ensure_orig_idx: bool = True,
    drop_empty: bool = False,
    uncategorized_label: str = "__uncategorized__",

    # repr_text
    include_body: bool = True,
    body_max_words: int = 220,   # include body first N words (0/None disables)
    body_max_chars: int = 0,     # char cap (0 disables)

    # sampling
    sample_growth: str = "sqrt",
    sample_tau: float = 1.0,
    sample_cap: int = 10000,
    gap_side: str = "right",
    log_base: float = 4.0,

    # output
    save_to_disk: bool = True,
    out_ds_dir: str = "kptimes_dedup_picked_v2",

    # progress
    show_tqdm: bool = True,

    # cache
    cache_root: Optional[str] = "kptimes_cache",
    overwrite_prepared: bool = False,
    save_prepared_ds: bool = True,
    strict_prepared_meta: bool = False,

    save_emb: bool = True,
    overwrite_emb: bool = False,
    strict_emb_meta: bool = True,
):
    """
    Auto-cache behavior:
      - prepared cache dir: {cache_root}/prepared/{split}
      - emb cache dir:      {cache_root}/emb/{split}/{repr_sig}/

    Cluster artifacts (saved ONCE after all buckets done):
      - {out_ds_dir}/cluster_artifacts/{split}/{repr_sig}/
        - arrays.npz / centroids.npy / cluster_summary.csv / records_*.parquet / meta.json   (from save_all_cluster_artifacts)
        - cluster_summary_by_category.csv
        - cat_vocab.json
        - cat_cluster_offsets.json
        - kptimes_big_extra.npz
        - kptimes_cluster_meta.json
    """



    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    cluster_prefer_side = str(cluster_prefer_side).strip().lower()
    if cluster_prefer_side not in {"min", "max"}:
        raise ValueError("cluster_prefer_side must be 'min' or 'max'")

    # -------------------------
    # 0) load raw
    # -------------------------
    if ds_raw is None:
        ds_raw = load_kptimes_raw()
    if split not in ds_raw:
        raise KeyError(f"ds_raw has no split='{split}', available: {list(ds_raw.keys())}")

    ds_split_raw = ds_raw[split]
    raw_len = int(len(ds_split_raw))

    # -------------------------
    # 0.5) device safety
    # -------------------------
    try:
        import torch
        dev = str(getattr(cfg, "device", "")).lower()
        if dev.startswith("cuda") and (not torch.cuda.is_available()):
            print("[WARN] cfg.device='cuda' but CUDA not available -> fallback to CPU for embedding.")
            try:
                cfg = dc_replace(cfg, device="cpu")
            except Exception:
                try:
                    setattr(cfg, "device", "cpu")
                except Exception:
                    pass
    except Exception:
        pass

    # -------------------------
    # 1) auto paths + repr_sig
    # -------------------------
    prepared_ds_dir = None
    emb_dir = None

    repr_sig = _repr_signature(
        cfg,
        include_body=include_body,
        body_max_words=body_max_words,
        body_max_chars=body_max_chars,
    )

    if cache_root:
        prepared_ds_dir = _auto_prepared_dir(cache_root, split)
        emb_dir = _auto_emb_dir(cache_root, split, repr_sig)

    # -------------------------
    # 2) prepared cache load
    # -------------------------
    loaded_from_prepared_cache = False
    prepared_dirty = False

    prep_meta_expected = {
        "version": "kptimes_prepared_v1",
        "split": split,
        "raw_len": int(raw_len),
        "ensure_orig_idx": bool(ensure_orig_idx),
        "drop_empty": bool(drop_empty),
        "uncategorized_label": str(uncategorized_label),
        "include_body": bool(include_body),
        "body_max_words": int(body_max_words) if body_max_words is not None else 0,
        "body_max_chars": int(body_max_chars),
    }

    if ds_prepared is None and prepared_ds_dir:
        if os.path.isdir(prepared_ds_dir) and (not overwrite_prepared):
            ds_prepared = load_from_disk(prepared_ds_dir)
            loaded_from_prepared_cache = True
            print(f"[PREP] loaded prepared dataset from: {prepared_ds_dir}")

            cached_meta = _read_json(os.path.join(prepared_ds_dir, "_prep_meta.json"))
            _validate_meta(prep_meta_expected, cached_meta, strict_prepared_meta, who="PREP")
        else:
            if os.path.isdir(prepared_ds_dir) and overwrite_prepared:
                print(f"[PREP] overwrite_prepared=True -> will rebuild prepared cache at: {prepared_ds_dir}")
            else:
                print(f"[PREP] no prepared cache found at: {prepared_ds_dir} -> will build it")

    # -------------------------
    # 3) prepare core columns (ONLY if needed)
    # -------------------------
    if ds_prepared is None:
        ds_prepared = kptimes_prepare_core_columns(
            ds_split_raw,
            num_proc=num_proc,
            batch_size=batch_size,
            load_from_cache_file=cache_ok,
            desc=f"[KPTime] prepare core columns ({split})",
            ensure_orig_idx=ensure_orig_idx,
        )
        prepared_dirty = True
    else:
        before_cols = set(ds_prepared.column_names)
        ds_prepared = kptimes_prepare_core_columns(
            ds_prepared,
            num_proc=num_proc,
            batch_size=batch_size,
            load_from_cache_file=cache_ok,
            desc=f"[KPTime] prepare missing core columns ({split})",
            ensure_orig_idx=ensure_orig_idx,
        )
        after_cols = set(ds_prepared.column_names)
        if after_cols != before_cols:
            prepared_dirty = True

    # -------------------------
    # 4) primary_label + bucketing
    # -------------------------
    if "primary_label" in ds_prepared.column_names:
        buck = _bucketing_from_primary_label(
            ds_prepared,
            protect_n=protect_n,
            primary_col="primary_label",
            orig_idx_col="orig_idx",
            verbose=True,
        )
    else:
        buck = compute_small_bucketing(
            ds_prepared,
            protect_n=protect_n,
            cat_col="categories_list_norm",
            orig_idx_col="orig_idx",
            drop_empty=drop_empty,
            uncategorized_label=uncategorized_label,
            verbose=True,
            show_tqdm=show_tqdm,
            num_proc=num_proc,
            batch_size=batch_size,
            load_from_cache_file=cache_ok,
        )
        ds_prepared = ds_prepared.add_column("primary_label", buck["primary_label_of"])
        prepared_dirty = True

    small_counter: Counter = buck["small_counter"]
    cat2idx = buck["cat2idx"]
    protected_labels = buck["protected_labels"]
    big_labels = buck["big_labels"]
    protected_idx = buck["protected_idx"]
    big_idx_all = buck["big_idx_all"]

    if protected_idx.size and int(protected_idx.max()) >= raw_len:
        raise ValueError(f"protected_idx has values >= raw_len ({raw_len}). orig_idx likely misaligned.")
    if big_idx_all.size and int(big_idx_all.max()) >= raw_len:
        raise ValueError(f"big_idx_all has values >= raw_len ({raw_len}). orig_idx likely misaligned.")

    # -------------------------
    # 5) repr_text column
    # -------------------------
    if "repr_text" not in ds_prepared.column_names:
        ds_prepared = ds_prepared.map(
            lambda b: _build_repr_text_batch(
                b,
                include_body=include_body,
                body_max_words=body_max_words,
                body_max_chars=body_max_chars,
            ),
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc=f"[KPTime] build repr_text ({split})" if show_tqdm else None,
            load_from_cache_file=cache_ok,
        )
        prepared_dirty = True

    # -------------------------
    # 6) save prepared immediately
    # -------------------------
    if prepared_ds_dir and save_prepared_ds:
        need_save = prepared_dirty or overwrite_prepared or (not loaded_from_prepared_cache)
        if need_save:
            os.makedirs(prepared_ds_dir, exist_ok=True)
            ds_prepared.save_to_disk(prepared_ds_dir)
            _write_json(os.path.join(prepared_ds_dir, "_prep_meta.json"), prep_meta_expected)
            print(f"[PREP] saved prepared dataset to: {prepared_ds_dir}")
        else:
            print(f"[PREP] prepared cache unchanged, skip save: {prepared_ds_dir}")

    # -------------------------
    # 7) build big-only view
    # -------------------------
    orig_idx_view = np.asarray(ds_prepared["orig_idx"], dtype=np.int64)
    N_view = len(orig_idx_view)
    if len(np.unique(orig_idx_view)) != N_view:
        raise ValueError("[SANITY] ds_prepared['orig_idx'] contains duplicates -> idx->pos mapping breaks.")

    idx2pos = np.full(raw_len, -1, dtype=np.int32)
    idx2pos[orig_idx_view] = np.arange(N_view, dtype=np.int32)

    view_pos_big = idx2pos[big_idx_all]
    if np.any(view_pos_big < 0):
        bad = big_idx_all[view_pos_big < 0][:10]
        raise ValueError(f"[ALIGN] some big_idx_all not found in prepared view. e.g. {bad.tolist()}")

    ds_big_view = ds_prepared.select(view_pos_big.tolist())
    repr_texts_big = ds_big_view["repr_text"]
    print(f"[EMB] big-only samples = {len(repr_texts_big)} / {raw_len}")

    try:
      if len(repr_texts_big) > 0:
          t0 = repr_texts_big[0]
          # 1) repr_text from _build_repr_text_batch.
          print("\n" + "-" * 110)
          print("[DEBUG] repr_texts_big[0] (raw repr_text):")
          print(t0[:2000])
          print("[DEBUG] contains 'Body:' ?", ("Body:" in t0))

          # 2) Text passed to the embedding model, usually prefix + repr_text.
          pref = str(getattr(cfg, "prefix", "")) or ""
          print("\n[DEBUG] model_input[0] (prefix + repr_text):")
          print((pref + t0)[:2000])
          print("-" * 110 + "\n")
    except Exception as e:
        print(f"[DEBUG][WARN] failed to print first model input: {type(e).__name__}: {e}")
    # -------------------------
    # 8) embedding cache auto-load/save
    # -------------------------
    emb_big = None
    emb_meta_expected = {
        "version": "kptimes_emb_v1",
        "split": split,
        "repr_sig": repr_sig,
        "model_name": str(getattr(cfg, "model_name", "")),
        "max_seq_length": int(getattr(cfg, "max_seq_length", 0)),
        "prefix": str(getattr(cfg, "prefix", "")),
        "include_body": bool(include_body),
        "body_max_words": int(body_max_words) if body_max_words is not None else 0,
        "body_max_chars": int(body_max_chars),
        "big_len": int(len(big_idx_all)),
        "big_idx_sha1": _big_idx_hash(big_idx_all),
    }

    if emb_dir and (not overwrite_emb):
        emb_big = _try_load_emb_cache(
            emb_dir=emb_dir,
            expected_meta=emb_meta_expected,
            big_idx_all=big_idx_all,
            strict_meta=strict_emb_meta,
        )
        if emb_big is not None:
            print(f"[EMB] loaded cached embedding from: {emb_dir}")

    if emb_big is None:
        emb_big, _model = embed_texts_e5_once(repr_texts_big, cfg, model=None)
        if emb_dir and save_emb:
            _save_emb_cache(
                emb_dir=emb_dir,
                emb_big=np.asarray(emb_big, dtype=np.float32),
                big_idx_all=np.asarray(big_idx_all, dtype=np.int64),
                meta=emb_meta_expected,
                overwrite=overwrite_emb,
            )

    emb_big = np.asarray(emb_big, dtype=np.float32)

    # raw_idx -> position in emb_big
    pos_map = np.full(raw_len, -1, dtype=np.int32)
    pos_map[big_idx_all] = np.arange(len(big_idx_all), dtype=np.int32)

    # -------------------------
    # 9) cluster artifact cache (load if conditions match)
    # -------------------------
    cluster_dir = os.path.join(out_ds_dir, "cluster_artifacts", split, repr_sig)
    os.makedirs(cluster_dir, exist_ok=True)

    def _cluster_cfg_sig_dict(cfg_obj):
        # Include every parameter that can affect clustering to prevent stale reuse.
        keys = [
            "top_k","hnsw_m","ef_construction","ef_search",
            "max_edges_per_node","q_sim","use_mutual",
            "resolution_list","max_ok_cluster",
            "attach_q","attach_neigh_k","pair_k","pair_q",
        ]
        d = {}
        for k in keys:
            v = getattr(cfg_obj, k, None)
            if isinstance(v, (list, tuple)):
                v = tuple(v)
            d[k] = v
        return d

    def _stable_sha1_of_json(obj):
        s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    cluster_meta_expected = {
        "version": "kptimes_cluster_cache_v1",
        "split": split,
        "repr_sig": repr_sig,
        "seed": int(seed),
        "protect_n": int(protect_n),
        "prefer_side": cluster_prefer_side,
        "big_len": int(len(big_idx_all)),
        "big_idx_sha1": _big_idx_hash(big_idx_all),
        "cluster_cfg_sig": _stable_sha1_of_json(_cluster_cfg_sig_dict(cfg)),
    }
    cluster_meta_path = os.path.join(cluster_dir, "kptimes_cluster_meta.json")
    extra_npz_path = os.path.join(cluster_dir, "kptimes_big_extra.npz")
    cat_vocab_path = os.path.join(cluster_dir, "cat_vocab.json")
    offsets_path = os.path.join(cluster_dir, "cat_cluster_offsets.json")

    can_load_cluster_cache = (
        os.path.isfile(cluster_meta_path)
        and os.path.isfile(extra_npz_path)
        and os.path.isfile(cat_vocab_path)
        and os.path.isfile(offsets_path)
        and os.path.isfile(os.path.join(cluster_dir, "arrays.npz"))  # from save_all_cluster_artifacts
    )

    loaded_cluster_cache = False
    if can_load_cluster_cache:
        cached = _read_json(cluster_meta_path)
        try:
            _validate_meta(cluster_meta_expected, cached, strict=True, who="CLUSTER")
            loaded_cluster_cache = True
            print(f"[CLUSTER] loaded cluster cache at: {cluster_dir} (skip reclustering)")
        except Exception as e:
            print(f"[CLUSTER] cache exists but meta mismatch -> will recluster. ({type(e).__name__}: {e})")

    # -------------------------
    # 10) per-bucket clustering (or load membership) + sampling
    # -------------------------
    picked_big_global = []
    per_cat_manifest = []

    # these will be filled either by load or by clustering
    big_len = int(len(big_idx_all))
    big_cat_id = np.full(big_len, -1, dtype=np.int32)          # aligned to big_idx_all
    big_local_cluster = np.full(big_len, -1, dtype=np.int32)   # aligned to big_idx_all
    big_global_cluster = np.full(big_len, -1, dtype=np.int32)  # aligned
    big_is_picked = np.zeros(big_len, dtype=np.bool_)

    cluster_stats_by_cat = {}   # category -> {"sizes":..., "picked":..., "num_clusters":K}
    cat_vocab = list(big_labels)
    cat2id = {c:i for i,c in enumerate(cat_vocab)}

    if loaded_cluster_cache:
        ex = np.load(extra_npz_path, allow_pickle=False)
        big_cat_id = ex["cat_id"].astype(np.int32)
        big_local_cluster = ex["local_cluster_id"].astype(np.int32)
        big_global_cluster = ex["global_cluster_id"].astype(np.int32)
        # big_is_picked: may not exist if old cache; handle
        if "is_picked" in ex.files:
            big_is_picked = ex["is_picked"].astype(bool)
        cat_vocab = _read_json(cat_vocab_path)["cat_vocab"]
        cat2id = {c:i for i,c in enumerate(cat_vocab)}

        offsets_obj = _read_json(offsets_path)
        num_clusters_per_cat = offsets_obj["num_clusters_per_cat"]
        offsets = np.asarray(offsets_obj["offsets"], dtype=np.int64)

        # (re)do sampling per category from saved membership (NO recluster)
        it_cats = cat_vocab
        if show_tqdm:
            from tqdm.auto import tqdm
            it_cats = tqdm(it_cats, desc="[KPTime] sample buckets (from cached clusters)")

        for cat in it_cats:
            cid = cat2id[cat]
            mask = (big_cat_id == cid)
            if not np.any(mask):
                continue

            pos_i64 = np.where(mask)[0].astype(np.int64)
            emb_sub = emb_big[pos_i64]
            mem_local = big_local_cluster[pos_i64].astype(np.int64)

            # sample using existing function (no recluster)
            picked_local, rep = sample_by_clusters_scheme2(
                emb=emb_sub,
                membership=mem_local,
                num=None,
                growth=sample_growth,
                tau=float(sample_tau),
                cap=int(sample_cap),
                log_base=float(log_base),
                seed=seed,
                gap_side=gap_side,
            )
            picked_local = _dedup_keep_order(picked_local)

            if len(picked_local):
                picked_pos = pos_i64[np.asarray(picked_local, dtype=np.int64)]
                big_is_picked[picked_pos] = True

            # stats
            K = int(num_clusters_per_cat.get(cat, int(mem_local.max()) + 1))
            sizes64 = np.bincount(mem_local.astype(np.int32), minlength=K).astype(np.int64)
            if len(picked_local):
                picked_c = mem_local[np.asarray(picked_local, dtype=np.int64)].astype(np.int32)
                picked64 = np.bincount(picked_c, minlength=K).astype(np.int64)
            else:
                picked64 = np.zeros(K, dtype=np.int64)
            cluster_stats_by_cat[str(cat)] = {"sizes": sizes64, "picked": picked64, "num_clusters": int(K)}

            # picked global raw idx
            picked_global = big_idx_all[pos_i64[np.asarray(picked_local, dtype=np.int64)]]
            picked_big_global.append(picked_global)

            per_cat_manifest.append({
                "category": str(cat),
                "orig_count": int(mask.sum()),
                "picked_count": int(len(picked_global)),
                "num_clusters": int(K),
                "sort_source": rep.get("sort_source", "emb"),
                "cluster_source": "cache",
            })

    else:
        # fresh clustering
        it_cats = big_labels
        if show_tqdm:
            from tqdm.auto import tqdm
            it_cats = tqdm(it_cats, desc="[KPTime] cluster+sample buckets")

        num_clusters_per_cat = {}

        for cat in it_cats:
            global_idx = cat2idx.get(cat, None)
            if global_idx is None or global_idx.size == 0:
                continue

            pos = pos_map[global_idx]
            if np.any(pos < 0):
                bad = global_idx[pos < 0][:10]
                raise ValueError(f"[ALIGN] bucket '{cat}' has indices not in big_idx_all. e.g. {bad.tolist()}")

            pos_i64 = pos.astype(np.int64)
            emb_sub = emb_big[pos_i64]

            # ---- CLUSTER (expensive) ----
            membership = cluster_embeddings_leiden(
                emb_sub,
                cfg,
                seed=seed,
                prefer_side=cluster_prefer_side,
            ).astype(np.int64)
            mem_local = membership.astype(np.int32)
            K = int(mem_local.max()) + 1
            num_clusters_per_cat[str(cat)] = int(K)

            # fill arrays aligned to big_idx_all
            cid = cat2id[cat]
            big_cat_id[pos_i64] = int(cid)
            big_local_cluster[pos_i64] = mem_local

            # sample
            picked_local, rep = sample_by_clusters_scheme2(
                emb=emb_sub,
                membership=membership,
                num=None,
                growth=sample_growth,
                tau=float(sample_tau),
                cap=int(sample_cap),
                log_base=float(log_base),
                seed=seed,
                gap_side=gap_side,
            )
            picked_local = _dedup_keep_order(picked_local)

            if len(picked_local):
                picked_pos = pos_i64[np.asarray(picked_local, dtype=np.int64)]
                big_is_picked[picked_pos] = True

            # stats
            sizes64 = np.bincount(mem_local, minlength=K).astype(np.int64)
            if len(picked_local):
                picked_c = mem_local[np.asarray(picked_local, dtype=np.int64)]
                picked64 = np.bincount(picked_c, minlength=K).astype(np.int64)
            else:
                picked64 = np.zeros(K, dtype=np.int64)
            cluster_stats_by_cat[str(cat)] = {"sizes": sizes64, "picked": picked64, "num_clusters": int(K)}

            picked_global = global_idx[np.asarray(picked_local, dtype=np.int64)]
            picked_big_global.append(picked_global)

            per_cat_manifest.append({
                "category": str(cat),
                "orig_count": int(global_idx.size),
                "picked_count": int(picked_global.size),
                "num_clusters": int(K),
                "sort_source": rep.get("sort_source", "emb"),
                "cluster_source": "fresh",
            })

            orig_n = int(global_idx.size)
            picked_n = int(picked_global.size)
            ratio = (picked_n / orig_n) if orig_n > 0 else 0.0
            print(f"[BUCKET] {cat}  orig={orig_n}  picked={picked_n}  keep={ratio:.4f} ({ratio*100:.2f}%)")

        # after all cats: build global cluster ids (offset per category)
        # offsets over categories in cat_vocab order
        offsets = [0]
        for cat in cat_vocab:
            offsets.append(offsets[-1] + int(num_clusters_per_cat.get(str(cat), 0)))
        offsets = np.asarray(offsets, dtype=np.int64)  # len = num_cats+1

        # compute global cluster id for each big sample
        if np.any(big_cat_id < 0) or np.any(big_local_cluster < 0):
            # should not happen; big_idx_all should be fully covered by big_labels
            bad = np.where((big_cat_id < 0) | (big_local_cluster < 0))[0][:10]
            raise ValueError(f"[CLUSTER] some big samples have no (cat_id/local_cluster). e.g. positions={bad.tolist()}")

        big_global_cluster = (offsets[big_cat_id.astype(np.int64)] + big_local_cluster.astype(np.int64)).astype(np.int32)

        # ---- SAVE cluster artifacts ONCE ----
        if save_to_disk:
            # 1) reuse your canonical artifact saver
            # ids/texts aligned to emb_big rows: use orig_idx as id, repr_text as text
            save_all_cluster_artifacts(
                out_dir=cluster_dir,
                ids=big_idx_all.tolist(),
                texts=repr_texts_big,
                emb=emb_big,
                membership=big_global_cluster.astype(np.int64),
                chunk_size=200_000,
            )

            # 2) save kptimes-specific mapping + extras
            _write_json(cat_vocab_path, {"cat_vocab": cat_vocab})
            _write_json(offsets_path, {
                "cat_vocab": cat_vocab,
                "num_clusters_per_cat": num_clusters_per_cat,
                "offsets": offsets.tolist(),
                "notes": "global_cluster_id = offsets[cat_id] + local_cluster_id",
            })

            np.savez_compressed(
                extra_npz_path,
                big_idx_all=np.asarray(big_idx_all, dtype=np.int64),
                cat_id=np.asarray(big_cat_id, dtype=np.int32),
                local_cluster_id=np.asarray(big_local_cluster, dtype=np.int32),
                global_cluster_id=np.asarray(big_global_cluster, dtype=np.int32),
                is_picked=np.asarray(big_is_picked, dtype=np.bool_),
            )

            _write_json(cluster_meta_path, cluster_meta_expected)
            print(f"[CLUSTER] saved cluster artifacts ONCE to: {cluster_dir}")

    # unify picked_big_global
    picked_big_global = np.concatenate(picked_big_global) if picked_big_global else np.array([], dtype=np.int64)
    picked_big_global = _dedup_keep_order(picked_big_global)

    # -------------------------
    # 11) final merge + save picked
    # -------------------------
    final_idx = (
        np.concatenate([protected_idx, picked_big_global])
        if protected_idx.size or picked_big_global.size
        else np.array([], dtype=np.int64)
    )
    final_idx = _dedup_keep_order(final_idx)

    picked_ds = ds_split_raw.select(final_idx.tolist())
    picked_ds = picked_ds.add_column("orig_idx", final_idx.tolist())

    df_small = small_counter_to_df(small_counter)

    summary = {
        "version": "kptimes_dedup_v2",
        "split": split,
        "orig_len": int(raw_len),
        "prepared_view_len": int(len(ds_prepared)),
        "picked_len": int(len(picked_ds)),
        "protect_n": int(protect_n),
        "unique_primary_labels": int(len(cat2idx)),
        "num_protected_labels": int(len(protected_labels)),
        "protected_samples": int(protected_idx.size),
        "big_labels_processed": int(len(per_cat_manifest)),
        "picked_from_big_total": int(picked_big_global.size),
        "seed": int(seed),
        "drop_empty": bool(drop_empty),
        "uncategorized_label": str(uncategorized_label),
        "repr": {
            "include_body": bool(include_body),
            "body_max_words": int(body_max_words) if body_max_words is not None else 0,
            "body_max_chars": int(body_max_chars),
        },
        "sample": {
            "growth": sample_growth,
            "tau": float(sample_tau),
            "cap": int(sample_cap),
            "gap_side": gap_side,
            "log_base": float(log_base),
        },
        "cluster": {
            "prefer_side": cluster_prefer_side,
        },
        "cache": {
            "cache_root": cache_root,
            "prepared_ds_dir": prepared_ds_dir,
            "emb_dir": emb_dir,
            "repr_sig": repr_sig,
            "loaded_from_prepared_cache": bool(loaded_from_prepared_cache),
        },
        "cluster_artifacts_dir": cluster_dir if save_to_disk else None,
        "save_to_disk": bool(save_to_disk),
        "out_ds_dir": out_ds_dir if save_to_disk else None,
    }

    if save_to_disk:
        os.makedirs(out_ds_dir, exist_ok=True)
        picked_ds.save_to_disk(out_ds_dir)

        with open(os.path.join(out_ds_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(os.path.join(out_ds_dir, "manifest_by_category.json"), "w", encoding="utf-8") as f:
            json.dump(per_cat_manifest, f, ensure_ascii=False, indent=2)

        df_small.to_csv(os.path.join(out_ds_dir, "small_counter.csv"), index=False)
        np.save(os.path.join(out_ds_dir, "final_idx.npy"), final_idx)

        # write per-category cluster summary (size/picked/ratio) ONCE
        try:
            import pandas as pd
            rows = []
            for cat, st in cluster_stats_by_cat.items():
                sizes = st["sizes"]
                picked = st["picked"]
                nonempty = np.where(sizes > 0)[0]
                for cid in nonempty.tolist():
                    s = int(sizes[cid])
                    p = int(picked[cid]) if cid < len(picked) else 0
                    rows.append((cat, int(cid), s, p, (p / s) if s else 0.0))
            dfc = pd.DataFrame(rows, columns=["category", "local_cluster_id", "size", "picked", "ratio"])
            dfc.sort_values(["category", "size"], ascending=[True, False], inplace=True)
            dfc.to_csv(os.path.join(cluster_dir, "cluster_summary_by_category.csv"), index=False)
        except Exception as e:
            print(f"[CLUSTER][WARN] failed to write cluster_summary_by_category.csv: {type(e).__name__}: {e}")

    # -------------------------
    # 12) print summary
    # -------------------------
    print("\n" + "=" * 110)
    print("[KPTime] DONE (V2)")
    print("=" * 110)
    print(f"split={split}")
    print(f"orig_len={summary['orig_len']}")
    print(f"picked_len={summary['picked_len']}")
    if cache_root:
        print(f"cache_root={cache_root}")
        print(f"prepared_ds_dir={prepared_ds_dir}")
        print(f"emb_dir={emb_dir}")
    print(
        f"protected: labels<{protect_n} => {summary['num_protected_labels']} labels, "
        f"{summary['protected_samples']} samples kept"
    )
    print(
        f"big categories processed: {summary['big_labels_processed']}, "
        f"picked_from_big_total={summary['picked_from_big_total']}"
    )
    print(f"save_to_disk={save_to_disk}  out_ds_dir={out_ds_dir if save_to_disk else '(disabled)'}")
    if save_to_disk:
        print(f"cluster_artifacts_dir={cluster_dir}")
        print("  - see cluster_summary_by_category.csv (per category/cluster stats)")
        print("  - see records_*.parquet (members by global cluster_id)")
    print("=" * 110)

    return picked_ds, summary, per_cat_manifest, final_idx, df_small



def view_kptimes_members_in_cat_cluster(
    cluster_dir: str,
    category: str,
    local_cluster_id: int,
    *,
    limit: int | None = None,
    sort_by: str = "dist",          # "dist" / "sim" / "rank" / None
    ascending: bool = True,         # Smaller distance means closer to centroid.
    include_text: bool = True,
    include_picked: bool = True,    # Add is_picked from kptimes_big_extra.npz.
):
    """
    Return: (df, global_cluster_id)

    Example cluster_dir:
      {out_ds_dir}/cluster_artifacts/{split}/{repr_sig}/

    Required files in cluster_dir:
      - cat_vocab.json
      - cat_cluster_offsets.json
      - kptimes_big_extra.npz (optional; used when include_picked=True)
      - records_*.parquet
    """
    import os
    import json
    import numpy as np

    # ---- 1) category -> cat_id, local -> global_cluster_id ----
    vocab_path = os.path.join(cluster_dir, "cat_vocab.json")
    offsets_path = os.path.join(cluster_dir, "cat_cluster_offsets.json")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"missing: {vocab_path}")
    if not os.path.isfile(offsets_path):
        raise FileNotFoundError(f"missing: {offsets_path}")

    cat_vocab = json.load(open(vocab_path, "r", encoding="utf-8"))["cat_vocab"]
    cat2id = {c: i for i, c in enumerate(cat_vocab)}
    if category not in cat2id:
        # Provide approximate diagnostics.
        sample = ", ".join(cat_vocab[:20])
        raise KeyError(f"unknown category='{category}'. e.g. first cats: {sample}")

    offsets_obj = json.load(open(offsets_path, "r", encoding="utf-8"))
    offsets = np.asarray(offsets_obj["offsets"], dtype=np.int64)
    cat_id = int(cat2id[category])
    local_cluster_id = int(local_cluster_id)

    global_cluster_id = int(offsets[cat_id] + local_cluster_id)

    # ---- 2) query records_*.parquet for this global_cluster_id ----
    files = sorted(glob.glob(os.path.join(cluster_dir, "records_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no records_*.parquet under: {cluster_dir}")

    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise RuntimeError("pyarrow is required for efficient parquet filtering. `pip install pyarrow`") from e

    dset = ds.dataset(files, format="parquet")
    want_cols = ["id", "cluster_id", "dist", "sim", "rank"]
    if include_text:
        want_cols.append("text")

    cols = [c for c in want_cols if c in dset.schema.names]
    if "id" not in cols or "cluster_id" not in cols:
        raise ValueError(f"records parquet must contain 'id' and 'cluster_id'. got schema={dset.schema.names}")

    table = dset.to_table(
        filter=(ds.field("cluster_id") == global_cluster_id),
        columns=cols,
    )
    df = table.to_pandas()

    # ---- 3) optionally add is_picked from kptimes_big_extra.npz ----
    if include_picked:
        extra_path = os.path.join(cluster_dir, "kptimes_big_extra.npz")
        if os.path.isfile(extra_path) and len(df) > 0:
            ex = np.load(extra_path, allow_pickle=False)
            # big_idx_all aligns with is_picked and contains unique values.
            big_idx_all = ex["big_idx_all"].astype(np.int64)
            is_picked = ex["is_picked"].astype(bool) if "is_picked" in ex.files else None

            if is_picked is not None:
                # Map id to is_picked with sort/searchsorted to reduce memory use.
                order = np.argsort(big_idx_all)
                big_sorted = big_idx_all[order]

                ids = df["id"].to_numpy(dtype=np.int64)
                pos = np.searchsorted(big_sorted, ids)
                ok = (pos >= 0) & (pos < len(big_sorted)) & (big_sorted[pos] == ids)

                picked_flag = np.zeros(len(ids), dtype=bool)
                picked_flag[ok] = is_picked[order[pos[ok]]]
                df["is_picked"] = picked_flag
        else:
            # Omit the column when the extra file is unavailable.
            pass

    # ---- 4) sort + limit ----
    if sort_by and (sort_by in df.columns):
        df = df.sort_values(sort_by, ascending=ascending)

    if limit is not None:
        df = df.head(int(limit))

    return df, global_cluster_id



def _normalize_categories(raw_categories: Any) -> List[str]:
    if raw_categories is None:
        return []
    if not isinstance(raw_categories, list):
        raw_categories = [raw_categories]
    return [str(c).strip() for c in raw_categories if str(c).strip()]



def _sanitize_num_proc(num_proc: Optional[int]) -> int:
    cpu_count = os.cpu_count() or 1
    if num_proc is None:
        num_proc = min(4, cpu_count)
    return max(1, min(int(num_proc), cpu_count))



def _stringify_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, (list, tuple)):
        parts = [str(v).strip() for v in x if str(v).strip()]
        return ", ".join(parts).strip()
    return str(x).strip()



def _pick_first_text(
    meta: Optional[dict],
    meta_keys: List[str],
    batch: Optional[dict] = None,
    row_idx: Optional[int] = None,
    batch_keys: Optional[List[str]] = None,
) -> str:
    meta = meta or {}
    for k in meta_keys:
        if k in meta:
            s = _stringify_text(meta.get(k))
            if s:
                return s

    if batch is not None and row_idx is not None and batch_keys:
        for k in batch_keys:
            if k in batch:
                col = batch[k]
                if row_idx < len(col):
                    s = _stringify_text(col[row_idx])
                    if s:
                        return s
    return ""



def _normalize_keywords_for_prompt(x: Any) -> str:
    return _stringify_text(x)



def _truncate_text_by_tokens(
    tokenizer,
    text: str,
    max_tokens: Optional[int],
) -> str:
    text = _stringify_text(text)
    if not text or max_tokens is None or max_tokens <= 0:
        return text

    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text

    truncated_ids = ids[:max_tokens]
    return tokenizer.decode(
        truncated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()



def _build_chat_ids_and_texts(
    tokenizer,
    user_text: str,
    assistant_text_plain: str,
    system_text: Optional[str] = None,
):
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
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "full_ids": full_ids,
        "prompt_text": prompt_text,
        "answer_text": answer_text,
        "full_text": full_text,
    }



def _split_seed_offset(split_name: str) -> int:
    return 0 if split_name == "train" else (1000 if split_name == "validation" else 2000)



def _get_prompt_text(row: dict) -> str:
    candidate_keys = [
        "PromptRaw", "Prompt", "prompt",
        "input_text", "Input", "input",
        "source_text", "Source",
    ]
    for k in candidate_keys:
        if k in row and row[k] is not None:
            v = row[k]
            return v if isinstance(v, str) else str(v)
    return ""



def _get_answer_text(row: dict) -> str:
    candidate_keys = [
        "AnswerPlain", "Answer", "answer",
        "output_text", "Output", "output",
        "target_text", "Target", "target",
        "labels_text",
    ]
    for k in candidate_keys:
        if k in row and row[k] is not None:
            v = row[k]
            return v if isinstance(v, str) else str(v)
    return ""



def _normalize_unique_list_preserve_order(
    raw_values: Any,
    split_pattern: Optional[str] = None,
) -> List[str]:
    if raw_values is None:
        return []

    if isinstance(raw_values, (list, tuple)):
        vals = list(raw_values)
    else:
        txt = str(raw_values).strip()
        if not txt:
            return []
        vals = re.split(split_pattern, txt) if split_pattern else [txt]

    out = []
    seen = set()
    for v in vals:
        s = str(v).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out



def _build_prefixed_answer_plain(
    items: Any,
    prefix_text: str,
) -> str:
    if items is None:
        item_text = ""
    elif isinstance(items, str):
        item_text = items.strip()
    elif isinstance(items, (list, tuple)):
        item_text = ", ".join([str(x).strip() for x in items if str(x).strip()])
    else:
        item_text = str(items).strip()
    return f"{prefix_text}{item_text}"



def _build_prefixed_piece_specs_generic(
    items: List[str],
    *,
    prefix_text: str,
    item_role_prefix: str,
    prefix_weight: float = 0.2,
    separator_weight: float = 0.5,
    label_base_weight: float = 1.2,
    later_label_alpha: float = 0.3,
    later_label_power: float = 1.0,
    only_label_weight: float = 1.3,
):
    prefix_no_space = prefix_text[:-1] if prefix_text.endswith(" ") else prefix_text

    specs = [
        {
            "role": "prefix",
            "group_key": "prefix",
            "text": prefix_no_space,
            "budget": float(prefix_weight),
        }
    ]

    m = len(items)
    for idx, item in enumerate(items, start=1):
        if idx == 1:
            item_text = " " + str(item)
        else:
            specs.append(
                {
                    "role": "separator",
                    "group_key": "separator_all",
                    "text": ",",
                    "budget": float(separator_weight),
                }
            )
            item_text = " " + str(item)

        specs.append(
            {
                "role": f"{item_role_prefix}_{idx}",
                "group_key": f"{item_role_prefix}_{idx}",
                "text": item_text,
                "budget": _label_rank_weight(
                    rank_1based=idx,
                    total_labels=m,
                    base_weight=label_base_weight,
                    later_label_alpha=later_label_alpha,
                    later_label_power=later_label_power,
                    only_label_weight=only_label_weight,
                ),
            }
        )

    return specs



def _compute_group_per_token_weight_generic(
    group_key: str,
    budget: float,
    token_count: int,
    *,
    content_group_prefixes: Tuple[str, ...],
    label_token_weight_mode: str = "affine",
    label_token_affine_alpha: float = 0.5,
    label_token_power_gamma: float = 0.5,
) -> float:
    if token_count <= 0:
        raise ValueError(f"token_count must be > 0, got {token_count}")

    budget = float(budget)
    n = float(token_count)

    is_content_group = any(str(group_key).startswith(pref) for pref in content_group_prefixes)

    # Format tokens use a fixed per-instance weight instead of sharing group weight.
    if not is_content_group:
        return budget

    mode = str(label_token_weight_mode).strip().lower()

    if mode in {"strict", "linear", "divide_by_n"}:
        denom = n
    elif mode == "affine":
        alpha = float(label_token_affine_alpha)
        if alpha < 0:
            raise ValueError("label_token_affine_alpha must be >= 0.")
        denom = 1.0 + alpha * (n - 1.0)
    elif mode == "power":
        gamma = float(label_token_power_gamma)
        if gamma <= 0:
            raise ValueError("label_token_power_gamma must be > 0.")
        denom = n ** gamma
    else:
        raise ValueError(
            "Unknown label_token_weight_mode. "
            "Expected one of: strict / linear / divide_by_n / affine / power"
        )

    if denom <= 0:
        raise ValueError(f"Invalid denom={denom} for group_key={group_key!r}")

    return budget / denom



def _stable_prob_hit(
    seed: int,
    split_name: str,
    source_id: str,
    salt: str,
    prob: float,
) -> bool:
    if prob <= 0:
        return False
    if prob >= 1:
        return True
    key = f"{seed}|{split_name}|{source_id}|{salt}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    u = int.from_bytes(digest, byteorder="big") / float(2**64)
    return u < prob



def _tokenize_no_special(tokenizer, text: str) -> List[int]:
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]



def _label_rank_weight(
    rank_1based: int,
    total_labels: int,
    base_weight: float = 1.2,
    later_label_alpha: float = 0.3,
    later_label_power: float = 1.0,
    only_label_weight: float = 1.3,
) -> float:
    if later_label_power <= 0:
        raise ValueError("later_label_power must be > 0.")

    if total_labels < 1:
        frac = 0.0
    elif total_labels == 1:
        return float(only_label_weight)
    else:
        frac = float(rank_1based - 1) / float(total_labels - 1)

    frac = max(0.0, min(1.0, frac))
    return float(base_weight * (1.0 + later_label_alpha * (frac ** later_label_power)))



def _decode_token_segments_from_ids(tokenizer, ids: List[int]) -> Tuple[str, List[str]]:
    """Decode token IDs into stable segments of the final decoded text.

    Byte-level tokenizers can temporarily emit the replacement character while
    only part of a multi-byte Unicode character has been seen.  Decoding one
    more token then replaces that character, so consecutive prefix decodes are
    not guaranteed to be strict string extensions.  Use the complete decode as
    the source of truth and retain only the stable common prefix at each step.

    A token that contains only an incomplete byte fragment therefore receives
    an empty segment.  The span-alignment helper below assigns such a token to
    the span containing the character completed by a later token.
    """
    ids = list(ids)
    full_text = tokenizer.decode(
        ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    stable_cursor = 0
    cur_ids = []
    segs = []

    for tid in ids:
        cur_ids.append(int(tid))
        cur_text = tokenizer.decode(
            cur_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        common_prefix_len = 0
        common_limit = min(len(cur_text), len(full_text))
        while (
            common_prefix_len < common_limit
            and cur_text[common_prefix_len] == full_text[common_prefix_len]
        ):
            common_prefix_len += 1

        if common_prefix_len < stable_cursor:
            raise ValueError(
                "Decoded token prefix changed text that was already stable against "
                "the complete decode."
            )
        segs.append(full_text[stable_cursor:common_prefix_len])
        stable_cursor = common_prefix_len

    if stable_cursor != len(full_text):
        raise ValueError(
            "Failed to map all characters from the complete decode to token segments."
        )

    return full_text, segs


def _align_token_segments_to_spans(
    token_segments: List[str],
    spans: List[Dict[str, Any]],
    *,
    context: str,
) -> Tuple[List[str], List[str]]:
    """Map decoded token segments to answer spans, including byte fragments."""
    if not spans:
        raise ValueError(f"Cannot align {context} token segments without spans.")

    roles: List[str] = []
    group_keys: List[str] = []
    char_cursor = 0
    span_idx = 0

    for seg in token_segments:
        tok_start = char_cursor
        tok_end = tok_start + len(seg)
        char_cursor = tok_end

        best_span = None
        best_overlap = -1

        if tok_start == tok_end:
            # The token is an incomplete byte fragment.  Attribute it to the
            # character at the current cursor, which a later token completes.
            for sp in spans:
                if sp["start"] <= tok_start < sp["end"]:
                    best_span = sp
                    break
            if best_span is None and tok_start == spans[-1]["end"]:
                best_span = spans[-1]
        else:
            for j in range(span_idx, len(spans)):
                sp = spans[j]
                overlap = max(0, min(tok_end, sp["end"]) - max(tok_start, sp["start"]))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_span = sp
                if sp["start"] >= tok_end:
                    break

        if best_span is None or (tok_start != tok_end and best_overlap <= 0):
            raise ValueError(
                f"Failed to align {context} token segment to any span: "
                f"seg={seg!r}, tok_start={tok_start}, tok_end={tok_end}"
            )

        roles.append(best_span["role"])
        group_keys.append(best_span["group_key"])

        while span_idx < len(spans) and spans[span_idx]["end"] <= tok_end:
            span_idx += 1

    return roles, group_keys



def _build_char_spans_from_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans = []
    cursor = 0
    for spec in specs:
        text = spec["text"]
        start = cursor
        end = start + len(text)

        budget_raw = spec.get("budget", None)
        budget = None if budget_raw is None else float(budget_raw)

        spans.append(
            {
                "role": spec["role"],
                "group_key": spec["group_key"],
                "budget": budget,
                "text": text,
                "start": start,
                "end": end,
            }
        )
        cursor = end
    return spans



def _build_full_prefixed_token_roles_and_weights_generic(
    tokenizer,
    prompt_ids: List[int],
    answer_ids: List[int],
    labels: List[int],
    items: List[str],
    answer_plain: str,
    *,
    prefix_text: str,
    item_role_prefix: str,
    prefix_weight: float = 0.2,
    separator_weight: float = 0.5,
    label_base_weight: float = 1.2,
    later_label_alpha: float = 0.3,
    later_label_power: float = 1.0,
    only_label_weight: float = 1.3,
    terminal_active_weight: float = 1.0,
    terminal_masked_weight: float = 0.0,
    first_active_terminal_weight_for_single_item: Optional[float] = None,
    first_active_terminal_role_for_single_item: str = "terminal_active_only_label_first",
    label_token_weight_mode: str = "power",
    label_token_affine_alpha: float = 0.5,
    label_token_power_gamma: float = 0.5,
) -> Tuple[List[str], List[float]]:
    answer_ids = list(answer_ids)
    answer_labels = list(labels[len(prompt_ids): len(prompt_ids) + len(answer_ids)])

    plain_specs = _build_prefixed_piece_specs_generic(
        items=items,
        prefix_text=prefix_text,
        item_role_prefix=item_role_prefix,
        prefix_weight=prefix_weight,
        separator_weight=separator_weight,
        label_base_weight=label_base_weight,
        later_label_alpha=later_label_alpha,
        later_label_power=later_label_power,
        only_label_weight=only_label_weight,
    )
    plain_text_expected = "".join(spec["text"] for spec in plain_specs)

    full_answer_text, full_answer_token_segs = _decode_token_segments_from_ids(tokenizer, answer_ids)

    if plain_text_expected != answer_plain:
        plain_text_expected = answer_plain

    if not full_answer_text.startswith(plain_text_expected):
        raise ValueError(
            f"Decoded answer text does not start with answer_plain.\n"
            f"decoded={full_answer_text!r}\n"
            f"answer_plain={plain_text_expected!r}"
        )

    content_char_len = len(plain_text_expected)

    token_char_cursor = 0
    content_token_count = 0
    for seg in full_answer_token_segs:
        next_cursor = token_char_cursor + len(seg)
        if token_char_cursor < content_char_len:
            content_token_count += 1
        token_char_cursor = next_cursor

    content_ids = answer_ids[:content_token_count]
    terminal_labels = answer_labels[content_token_count:]

    content_text, content_token_segs = _decode_token_segments_from_ids(tokenizer, content_ids)
    if content_text != plain_text_expected:
        raise ValueError(
            f"Content text mismatch.\n"
            f"decoded_content={content_text!r}\n"
            f"expected={plain_text_expected!r}"
        )

    spans = _build_char_spans_from_specs(plain_specs)

    group_budgets: Dict[str, float] = {}
    for sp in spans:
        g = sp["group_key"]
        b = float(sp["budget"])
        if g not in group_budgets:
            group_budgets[g] = b
        elif abs(group_budgets[g] - b) > 1e-12:
            raise ValueError(
                f"Inconsistent budget for group_key={g!r}: "
                f"{group_budgets[g]} vs {b}"
            )

    answer_roles, content_group_keys = _align_token_segments_to_spans(
        content_token_segs,
        spans,
        context="content",
    )

    group_token_counts: Dict[str, int] = {}
    for g in content_group_keys:
        group_token_counts[g] = group_token_counts.get(g, 0) + 1

    group_per_token_weights: Dict[str, float] = {}
    for g, cnt in group_token_counts.items():
        budget = group_budgets.get(g, None)
        if budget is None:
            raise ValueError(f"Missing budget for group_key={g!r}")

        group_per_token_weights[g] = _compute_group_per_token_weight_generic(
            group_key=g,
            budget=budget,
            token_count=cnt,
            content_group_prefixes=(f"{item_role_prefix}_",),
            label_token_weight_mode=label_token_weight_mode,
            label_token_affine_alpha=label_token_affine_alpha,
            label_token_power_gamma=label_token_power_gamma,
        )

    answer_weights: List[float] = []
    for g in content_group_keys:
        if g not in group_per_token_weights:
            raise ValueError(f"Missing per-token weight for group_key={g!r}")
        answer_weights.append(float(group_per_token_weights[g]))

    is_single_item_sample = (len(items) == 1)
    first_active_terminal_used = False

    for lab in terminal_labels:
        is_active = bool(lab != -100)

        if is_active:
            if (
                is_single_item_sample
                and (first_active_terminal_weight_for_single_item is not None)
                and (not first_active_terminal_used)
            ):
                answer_roles.append(first_active_terminal_role_for_single_item)
                answer_weights.append(float(first_active_terminal_weight_for_single_item))
                first_active_terminal_used = True
            else:
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

    full_roles = (["prompt"] * len(prompt_ids)) + answer_roles
    full_weights = ([0.0] * len(prompt_ids)) + answer_weights
    return full_roles, full_weights



def _filter_keep_title_body_target_batch(
    batch,
    *,
    target_field: str,
):
    keep = []
    titles = batch["Title"]
    bodies = batch["Body"]
    targets = batch[target_field]

    for title, body, target in zip(titles, bodies, targets):
        ok = bool(_stringify_text(title) and _stringify_text(body) and target)
        keep.append(ok)

    return keep



def _to_trainer_features_batch_generic(
    batch,
    *,
    tokenizer,
    system_text: Optional[str],
    max_length: Optional[int],
    train_terminal_sequences: List[List[int]],
    terminal_loss_mode: str,
    prefix_weight: float,
    separator_weight: float,
    label_base_weight: float,
    later_label_alpha: float,
    later_label_power: float,
    only_label_weight: float,
    terminal_active_weight: float,
    target_field_name: str,
    extra_passthrough_fields: Optional[List[str]],
    role_weight_builder: Callable[..., Tuple[List[str], List[float]]],
):
    extra_passthrough_fields = list(extra_passthrough_fields or [])

    out = {
        "source_id": [],
        "Title": [],
        "PromptRaw": [],
        "AnswerPlain": [],
        "TaskType": [],
        target_field_name: [],
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
    }
    for f in extra_passthrough_fields:
        out[f] = []

    task_types = batch["TaskType"] if "TaskType" in batch else [""] * len(batch["PromptRaw"])

    n = len(batch["PromptRaw"])
    for i in range(n):
        source_id = batch["source_id"][i]
        title = batch["Title"][i]
        prompt_raw = batch["PromptRaw"][i]
        answer_plain = batch["AnswerPlain"][i]
        task_type = task_types[i]
        items = list(batch[target_field_name][i] or [])

        passthrough_payload = {
            f: list(batch[f][i] or []) if isinstance(batch[f][i], (list, tuple)) else batch[f][i]
            for f in extra_passthrough_fields
        }

        built = _build_chat_ids_and_texts(
            tokenizer=tokenizer,
            user_text=prompt_raw,
            assistant_text_plain=answer_plain,
            system_text=system_text,
        )

        full_ids = list(built["full_ids"])
        prompt_ids = list(built["prompt_ids"])
        answer_ids = list(built["answer_ids"])

        labels, _ = _build_answer_labels_with_terminal_mask(
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            terminal_sequences=train_terminal_sequences,
            terminal_loss_mode=terminal_loss_mode,
        )

        token_roles, loss_weights = role_weight_builder(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            labels=labels,
            items=items,
            answer_plain=answer_plain,
            prefix_weight=prefix_weight,
            separator_weight=separator_weight,
            label_base_weight=label_base_weight,
            later_label_alpha=later_label_alpha,
            later_label_power=later_label_power,
            only_label_weight=only_label_weight,
            terminal_active_weight=terminal_active_weight,
        )

        if max_length is not None:
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
        out["Title"].append(title)
        out["PromptRaw"].append(prompt_raw)
        out["AnswerPlain"].append(answer_plain)
        out["TaskType"].append(task_type)
        out[target_field_name].append(items)
        out["PromptText"].append(built["prompt_text"])
        out["AnswerText"].append(built["answer_text"])
        out["LossAnswerText"].append(loss_answer_text)
        out["FullText"].append(built["full_text"])
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

        for f in extra_passthrough_fields:
            out[f].append(passthrough_payload[f])

    return out



def _build_prompt_candidates_from_source_generic(
    ds_source,
    *,
    split_name: str,
    seed: int,
    batch_size: int,
    num_proc: int,
    load_from_cache_file: bool,
    print_checks: bool,
    extract_fields_fn: Callable,
    extract_fn_kwargs: Dict[str, Any],
    filter_keep_fn: Callable,
    build_prompt_fn: Callable,
    build_prompt_fn_kwargs: Dict[str, Any],
    extract_desc: str,
    filter_desc: str,
    prompt_desc: str,
):
    if ds_source is None:
        return None

    total_raw_n = len(ds_source)
    if total_raw_n == 0:
        return Dataset.from_list([])

    ds_split = ds_source.shuffle(seed=seed + _split_seed_offset(split_name))

    valid_chunks = []
    cursor = 0
    step = max(1, int(batch_size))

    while cursor < total_raw_n:
        end = min(total_raw_n, cursor + step)
        ds_chunk = ds_split.select(range(cursor, end))
        cursor = end

        ds_fields = ds_chunk.map(
            extract_fields_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=extract_desc.format(split_name=split_name),
            fn_kwargs=extract_fn_kwargs,
            remove_columns=ds_chunk.column_names,
        )

        ds_valid_chunk = ds_fields.filter(
            filter_keep_fn,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            desc=filter_desc.format(split_name=split_name),
        )

        if len(ds_valid_chunk) > 0:
            valid_chunks.append(ds_valid_chunk)

    if not valid_chunks:
        raise ValueError(f"No valid rows left for split='{split_name}' after filtering.")

    ds_base = valid_chunks[0] if len(valid_chunks) == 1 else concatenate_datasets(valid_chunks)

    if len(ds_base) == 0:
        raise ValueError(f"No valid base rows kept for split='{split_name}'.")

    ds_prompt = ds_base.map(
        build_prompt_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc=prompt_desc.format(split_name=split_name),
        fn_kwargs=build_prompt_fn_kwargs,
        remove_columns=ds_base.column_names,
    )

    if len(ds_prompt) == 0:
        raise ValueError(f"No prompt rows produced for split='{split_name}'.")

    sample0 = ds_prompt[0]
    prompt_text0 = _get_prompt_text(sample0)
    answer_text0 = _get_answer_text(sample0)

    if print_checks:
        print(f"[DEBUG {split_name}] prompt columns: {list(sample0.keys())}")
        print(
            f"[DEBUG {split_name}] "
            f"prompt_text_found={bool(prompt_text0)}  answer_text_found={bool(answer_text0)}"
        )

    if not prompt_text0:
        raise KeyError(
            f"[{split_name}] Could not find prompt text column in ds_prompt. "
            f"Available keys: {list(sample0.keys())}"
        )
    if not answer_text0:
        raise KeyError(
            f"[{split_name}] Could not find answer text column in ds_prompt. "
            f"Available keys: {list(sample0.keys())}"
        )

    return ds_prompt



def _resolve_split_name(ds_raw: DatasetDict, split_name: str) -> str:
    if split_name in ds_raw:
        return split_name
    if split_name == "validation" and "valid" in ds_raw:
        return "valid"
    raise KeyError(f"Split '{split_name}' not found. Available splits: {list(ds_raw.keys())}")



def _collect_kptimes_label_space(ds_split) -> List[str]:
    label_set = set()
    for meta in ds_split["other_metadata"]:
        meta = meta or {}
        cats = _normalize_categories(meta.get("categories", []))
        for c in cats:
            if c:
                label_set.add(c)
    return sorted(label_set)



def _build_label_space_safe(
    ds_raw: DatasetDict,
    constraint_source_split: str = "train",
) -> Tuple[List[str], Dict[str, int]]:
    split_name = _resolve_split_name(ds_raw, constraint_source_split)
    label_list = _collect_kptimes_label_space(ds_raw[split_name])
    if not label_list:
        raise ValueError(f"Empty label space from split='{split_name}'.")
    label_to_idx = {x: i for i, x in enumerate(label_list)}
    return label_list, label_to_idx



def _canonicalize_categories(
    categories: List[str],
    label_to_idx: Dict[str, int],
    raise_on_unknown: bool = True,
) -> List[str]:
    out = []
    seen = set()
    unknown = []

    for c in categories:
        c = str(c).strip()
        if not c:
            continue
        if c not in label_to_idx:
            unknown.append(c)
            continue
        if c not in seen:
            seen.add(c)
            out.append(c)

    if unknown and raise_on_unknown:
        raise ValueError(
            f"Found categories outside the constraint label space: {sorted(set(unknown))}"
        )

    out.sort(key=lambda x: label_to_idx[x])
    return out



@dataclass
class CategoryOutputFSM:
    """
    Token-trie version for category constrained decoding.

    Compatibility details:
    1) prefix_token_variants accepts token sequences that decode to the same
       "categories:" text even when their token IDs differ. For example,
       tokenizer("categories:") may yield [13997, 29901], while training labels
       contain [20683, 29901]. Both decode to the same valid prefix.

    2) _strip_prefix_ids supports answer_prefix_ids that already extend beyond
       the prefix. For example:
           categories: us
           categories: nyregion
       Matching only while len(ids) <= len(prefix) would reject the sequence as
       soon as it generated a label token after the prefix.
    """

    EXPECT_LABEL_START = "EXPECT_LABEL_START"
    IN_LABEL = "IN_LABEL"
    INVALID = "INVALID"

    def __init__(
        self,
        tokenizer,
        label_list: List[str],
        prefix_text: str = "categories: ",
        allow_multi_label: bool = True,
        end_token_ids: Optional[List[int]] = None,
        terminal_sequences: Optional[List[List[int]]] = None,
        debug_prefix_variants: bool = False,
    ):
        if not label_list:
            raise ValueError("label_list is empty.")

        self.tokenizer = tokenizer
        self.label_list = [str(x).strip() for x in label_list if str(x).strip()]
        if not self.label_list:
            raise ValueError("label_list is empty after stripping.")

        self.label_to_idx = {x: i for i, x in enumerate(self.label_list)}

        self.prefix_text = str(prefix_text or "")
        self.sep_text = ", "
        self.allow_multi_label = bool(allow_multi_label)

        if terminal_sequences is not None:
            seqs = []
            for seq in terminal_sequences:
                seq = [int(x) for x in seq if x is not None]
                if seq:
                    seqs.append(seq)
            self.terminal_sequences = _unique_preserve_order(seqs)

        elif end_token_ids is not None:
            seqs = []
            for tid in end_token_ids:
                if tid is not None:
                    seqs.append([int(tid)])
            self.terminal_sequences = _unique_preserve_order(seqs)

        else:
            self.terminal_sequences = _resolve_default_terminal_sequences(tokenizer)

        if not self.terminal_sequences:
            raise ValueError("No valid terminal sequences available.")

        self.end_token_ids = sorted(
            {int(tid) for seq in self.terminal_sequences for tid in seq}
        )
        self.terminal_start_token_ids = sorted(
            {int(seq[0]) for seq in self.terminal_sequences if seq}
        )
        self.final_end_token_ids = sorted(
            {int(seq[-1]) for seq in self.terminal_sequences if seq}
        )

        self.prefix_token_variants: List[List[int]] = self._build_prefix_token_variants(
            debug=bool(debug_prefix_variants)
        )
        if not self.prefix_token_variants:
            raise ValueError("No valid prefix token variants for CategoryOutputFSM.")

        self.separator_token_ids = self._scan_separator_token_ids()
        self.comma_token_ids = list(self.separator_token_ids)

        self.label_paths_by_idx: Dict[int, List[List[int]]] = self._build_label_paths_by_idx()
        self.trie_by_min_label_idx: Dict[int, Dict[str, Any]] = self._build_tries_by_min_label_idx()

        self._allowed_cache: Dict[Tuple[int, ...], List[int]] = {}
        self._parse_cache: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        self.initial_allowed_token_ids = self.allowed_next_tokens([])

    # =====================================================
    # Decode helpers
    # =====================================================

    def _decode_ids(self, ids: List[int]) -> str:
        ids = [int(x) for x in (ids or [])]
        try:
            return self.tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(
                ids,
                skip_special_tokens=False,
            )

    def _decode_appended_segment(
        self,
        answer_prefix_ids: List[int],
        next_tok: int,
        base_text: Optional[str] = None,
    ) -> str:
        answer_prefix_ids = [int(x) for x in (answer_prefix_ids or [])]
        next_tok = int(next_tok)

        if base_text is None:
            base_text = self._decode_ids(answer_prefix_ids)

        new_text = self._decode_ids(answer_prefix_ids + [next_tok])

        if new_text.startswith(base_text):
            return new_text[len(base_text):]

        return self._decode_ids([next_tok])

    # =====================================================
    # Prefix variants
    # =====================================================

    def _tokenize_no_special_local(self, text: str) -> List[int]:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

    def _encode_no_special_local(self, text: str) -> List[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return self._tokenize_no_special_local(text)

    def _build_prefix_token_variants(self, debug: bool = False) -> List[List[int]]:
        """
        Build token sequences whose decoded text is equivalent to prefix_text.

        The implementation does not rely on a single tokenizer output. It scans
        the vocabulary for equivalent tokens and combines additional variants.
        """
        target = str(self.prefix_text or "")
        target_no_trailing_space = target.rstrip()

        variants: List[List[int]] = []
        seen: Set[Tuple[int, ...]] = set()

        def _add_variant(ids: List[int]) -> None:
            key = tuple(int(x) for x in (ids or []) if x is not None)
            if not key or key in seen:
                return

            decoded = self._decode_ids(list(key))

            # Strip only trailing whitespace to accept "categories:" and
            # "categories: " without accepting a leading-space variant.
            if decoded.rstrip() != target_no_trailing_space:
                return

            seen.add(key)
            variants.append(list(key))

        raw_text_variants = [target]
        if target.endswith(" "):
            raw_text_variants.append(target.rstrip())

        # 1) Standard tokenizer and encode paths.
        for txt in raw_text_variants:
            if not txt:
                continue
            try:
                _add_variant(self._tokenize_no_special_local(txt))
            except Exception:
                pass
            try:
                _add_variant(self._encode_no_special_local(txt))
            except Exception:
                pass

        # 2) Scan the vocabulary for decode-equivalent categories, colon, and
        # optional-space combinations that ordinary tokenization may not return.
        if target_no_trailing_space.endswith(":"):
            head_word = target_no_trailing_space[:-1]

            word_token_ids: Set[int] = set()
            colon_token_ids: Set[int] = set()
            space_token_ids: Set[int] = set()

            # cheap seed: ":" and " "
            for s in [":", " "]:
                try:
                    ids = self._tokenize_no_special_local(s)
                    if len(ids) == 1:
                        if s == ":":
                            colon_token_ids.add(int(ids[0]))
                        else:
                            space_token_ids.add(int(ids[0]))
                except Exception:
                    pass

                try:
                    ids = self._encode_no_special_local(s)
                    if len(ids) == 1:
                        if s == ":":
                            colon_token_ids.add(int(ids[0]))
                        else:
                            space_token_ids.add(int(ids[0]))
                except Exception:
                    pass

            try:
                vocab_size = len(self.tokenizer)
            except Exception:
                vocab_size = 0

            special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])

            for tid in range(vocab_size):
                tid = int(tid)
                if tid in special_ids:
                    continue

                try:
                    txt = self._decode_ids([tid])
                except Exception:
                    continue

                if txt == head_word:
                    word_token_ids.add(tid)
                elif txt == ":":
                    colon_token_ids.add(tid)
                elif txt and txt.isspace():
                    space_token_ids.add(tid)

            for wid in sorted(word_token_ids):
                for cid in sorted(colon_token_ids):
                    _add_variant([wid, cid])
                    for sid in sorted(space_token_ids):
                        _add_variant([wid, cid, sid])

        # Put longer variants first so stripping prefers "categories: ".
        variants.sort(key=len, reverse=True)

        if debug:
            print("[CategoryOutputFSM prefix variants]")
            for v in variants:
                print(v, repr(self._decode_ids(v)))

        return variants

    # =====================================================
    # Separator scanning
    # =====================================================

    def _scan_separator_token_ids(self) -> List[int]:
        out = set()

        variants = [self.sep_text]
        sep_strip = self.sep_text.strip()

        if sep_strip and sep_strip != self.sep_text:
            variants.append(sep_strip)

        if sep_strip and not sep_strip.startswith(" "):
            variants.append(" " + sep_strip)

        seen_variants = set()
        for s in variants:
            s = str(s or "")
            if not s or s in seen_variants:
                continue
            seen_variants.add(s)

            try:
                ids = self.tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    out.add(int(ids[0]))
            except Exception:
                pass

            try:
                ids = self._tokenize_no_special_local(s)
                if len(ids) == 1:
                    out.add(int(ids[0]))
            except Exception:
                pass

        try:
            vocab_size = len(self.tokenizer)
        except Exception:
            vocab_size = 0

        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        compact_targets = {re.sub(r"\s+", "", str(v)) for v in variants if str(v)}

        for tid in range(vocab_size):
            tid = int(tid)
            if tid in special_ids:
                continue

            try:
                txt = self._decode_ids([tid])
            except Exception:
                continue

            if not txt:
                continue

            if re.sub(r"\s+", "", txt) in compact_targets:
                out.add(tid)

        if not out:
            raise ValueError(f"No separator token ids found for sep_text={self.sep_text!r}")

        return sorted(out)

    # =====================================================
    # Label trie
    # =====================================================

    def _build_label_paths_by_idx(self) -> Dict[int, List[List[int]]]:
        out: Dict[int, List[List[int]]] = {}

        for idx, label in enumerate(self.label_list):
            label = str(label).strip()
            variants_raw = [" " + label, label]

            seen = set()
            paths: List[List[int]] = []

            for txt in variants_raw:
                try:
                    ids = self._tokenize_no_special_local(txt)
                except Exception:
                    ids = []

                key = tuple(int(x) for x in ids)
                if key and key not in seen:
                    seen.add(key)
                    paths.append(list(key))

                try:
                    ids2 = self._encode_no_special_local(txt)
                except Exception:
                    ids2 = []

                key2 = tuple(int(x) for x in ids2)
                if key2 and key2 not in seen:
                    seen.add(key2)
                    paths.append(list(key2))

            if not paths:
                raise ValueError(f"No token paths built for category label={label!r}")

            out[int(idx)] = paths

        return out

    def _build_trie_with_label_indices(self, label_indices: List[int]) -> Dict[str, Any]:
        nodes = [{"children": {}, "terminal_label_indices": []}]

        paths: List[List[int]] = []
        path_label_indices: List[int] = []
        seen = set()

        for idx in label_indices:
            idx = int(idx)
            for path in self.label_paths_by_idx.get(idx, []):
                key = tuple(int(x) for x in path)
                if not key or key in seen:
                    continue

                seen.add(key)
                paths.append(list(key))
                path_label_indices.append(idx)

        for path, label_idx in zip(paths, path_label_indices):
            cur = 0

            for tid in path:
                tid = int(tid)
                children = nodes[cur]["children"]

                if tid not in children:
                    children[tid] = len(nodes)
                    nodes.append({"children": {}, "terminal_label_indices": []})

                cur = int(children[tid])

            term_list = nodes[cur]["terminal_label_indices"]

            if int(label_idx) not in term_list:
                term_list.append(int(label_idx))
                term_list.sort()

        return {
            "nodes": nodes,
            "root_id": 0,
            "paths": paths,
            "path_label_indices": path_label_indices,
        }

    def _build_tries_by_min_label_idx(self) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}

        n = len(self.label_list)
        for min_idx in range(n):
            out[int(min_idx)] = self._build_trie_with_label_indices(
                list(range(min_idx, n))
            )

        return out

    def _trie_root_children(self, min_label_idx: int) -> Set[int]:
        trie = self.trie_by_min_label_idx.get(int(min_label_idx))
        if trie is None:
            return set()

        nodes = trie["nodes"]
        root = int(trie["root_id"])

        return {int(tok) for tok in nodes[root]["children"].keys()}

    # =====================================================
    # Prefix parsing
    # =====================================================

    def _sample_key(self, answer_prefix_ids: List[int]) -> Tuple[int, ...]:
        return tuple(int(x) for x in (answer_prefix_ids or []))

    def _prefix_status(self, answer_prefix_ids: List[int]) -> Tuple[List[int], List[int]]:
        """
        Returns:
        - full_match_lens: lengths of fully matched prefix variants
        - next_tokens: allowed next tokens when ids is a partial prefix

        An ids sequence longer than a prefix still counts as a full match when
        ids[:len(prefix)] equals that prefix.
        """
        ids = [int(x) for x in (answer_prefix_ids or [])]

        full_match_lens: List[int] = []
        next_tokens = set()

        for pref in self.prefix_token_variants:
            pref = [int(x) for x in pref]
            if not pref:
                continue

            if len(ids) < len(pref):
                if ids == pref[:len(ids)]:
                    next_tokens.add(int(pref[len(ids)]))

            else:
                if ids[:len(pref)] == pref:
                    full_match_lens.append(len(pref))

        return full_match_lens, sorted(next_tokens)

    def _strip_prefix_ids(self, answer_prefix_ids: List[int]) -> Tuple[Optional[List[int]], List[int]]:
        """
        Return trimmed ids after a complete prefix match. For a valid partial
        prefix, return None plus next_prefix_tokens. For no match, return
        None plus an empty list.
        """
        ids = [int(x) for x in (answer_prefix_ids or [])]

        full_match_lens, next_tokens = self._prefix_status(ids)

        if full_match_lens:
            pref_len = max(full_match_lens)
            return ids[pref_len:], []

        return None, list(next_tokens)

    # =====================================================
    # Category state parser
    # =====================================================

    def _parse_trimmed_ids(self, trimmed_ids: List[int]) -> Dict[str, Any]:
        key = tuple(int(x) for x in (trimmed_ids or []))

        cached = self._parse_cache.get(key)
        if cached is not None:
            return cached

        state: Dict[str, Any] = {
            "mode": self.EXPECT_LABEL_START,
            "min_label_idx": 0,
            "node_id": None,
            "trie": None,
            "terminal_label_idx": None,
        }

        if len(trimmed_ids) == 0:
            self._parse_cache[key] = state
            return state

        for raw_tid in trimmed_ids:
            tid = int(raw_tid)
            mode = state["mode"]

            if mode == self.EXPECT_LABEL_START:
                trie = self.trie_by_min_label_idx.get(int(state["min_label_idx"]))

                if trie is None:
                    state = {
                        "mode": self.INVALID,
                        "min_label_idx": int(state["min_label_idx"]),
                        "node_id": None,
                        "trie": None,
                        "terminal_label_idx": None,
                    }
                    break

                nodes = trie["nodes"]
                root = int(trie["root_id"])
                child = nodes[root]["children"].get(tid)

                if child is None:
                    state = {
                        "mode": self.INVALID,
                        "min_label_idx": int(state["min_label_idx"]),
                        "node_id": None,
                        "trie": None,
                        "terminal_label_idx": None,
                    }
                    break

                node_id = int(child)
                term_list = list(nodes[node_id].get("terminal_label_indices", []) or [])

                state = {
                    "mode": self.IN_LABEL,
                    "min_label_idx": int(state["min_label_idx"]),
                    "node_id": node_id,
                    "trie": trie,
                    "terminal_label_idx": int(term_list[0]) if term_list else None,
                }
                continue

            if mode == self.IN_LABEL:
                trie = state["trie"]

                if trie is None:
                    state = {
                        "mode": self.INVALID,
                        "min_label_idx": int(state["min_label_idx"]),
                        "node_id": None,
                        "trie": None,
                        "terminal_label_idx": None,
                    }
                    break

                nodes = trie["nodes"]
                node_id = int(state["node_id"])
                node = nodes[node_id]

                child = node["children"].get(tid)

                if child is not None:
                    node_id = int(child)
                    term_list = list(nodes[node_id].get("terminal_label_indices", []) or [])

                    state = {
                        "mode": self.IN_LABEL,
                        "min_label_idx": int(state["min_label_idx"]),
                        "node_id": node_id,
                        "trie": trie,
                        "terminal_label_idx": int(term_list[0]) if term_list else None,
                    }
                    continue

                terminal_label_idx = state.get("terminal_label_idx", None)

                if (
                    terminal_label_idx is not None
                    and self.allow_multi_label
                    and tid in self.separator_token_ids
                ):
                    next_min_idx = int(terminal_label_idx) + 1

                    if next_min_idx >= len(self.label_list):
                        state = {
                            "mode": self.INVALID,
                            "min_label_idx": next_min_idx,
                            "node_id": None,
                            "trie": None,
                            "terminal_label_idx": None,
                        }
                        break

                    state = {
                        "mode": self.EXPECT_LABEL_START,
                        "min_label_idx": next_min_idx,
                        "node_id": None,
                        "trie": None,
                        "terminal_label_idx": None,
                    }
                    continue

                state = {
                    "mode": self.INVALID,
                    "min_label_idx": int(state["min_label_idx"]),
                    "node_id": None,
                    "trie": None,
                    "terminal_label_idx": None,
                }
                break

            state = {
                "mode": self.INVALID,
                "min_label_idx": int(state.get("min_label_idx", 0)),
                "node_id": None,
                "trie": None,
                "terminal_label_idx": None,
            }
            break

        self._parse_cache[key] = state
        return state

    def _can_end_from_state(self, state: Dict[str, Any]) -> bool:
        return bool(
            state.get("mode") == self.IN_LABEL
            and state.get("terminal_label_idx") is not None
        )

    # =====================================================
    # Terminal handling
    # =====================================================

    def _allowed_after_partial_terminal_sequence(
        self,
        answer_prefix_ids: List[int],
    ) -> List[int]:
        allowed = set()
        ids = [int(x) for x in (answer_prefix_ids or [])]

        for seq in self.terminal_sequences:
            seq = [int(x) for x in seq]
            if len(seq) <= 1:
                continue

            for consumed_len in range(1, len(seq)):
                if len(ids) < consumed_len:
                    continue

                if ids[-consumed_len:] != seq[:consumed_len]:
                    continue

                stem_ids = ids[:-consumed_len]
                trimmed, next_tokens = self._strip_prefix_ids(stem_ids)

                if trimmed is None:
                    continue

                state = self._parse_trimmed_ids(trimmed)

                if self._can_end_from_state(state):
                    allowed.add(int(seq[consumed_len]))

        return sorted(allowed)

    # =====================================================
    # Main allowed-token API
    # =====================================================

    def allowed_next_tokens(self, answer_prefix_ids: List[int]) -> List[int]:
        cache_key = self._sample_key(answer_prefix_ids)

        if cache_key in self._allowed_cache:
            return self._allowed_cache[cache_key]

        partial_terminal_allowed = self._allowed_after_partial_terminal_sequence(answer_prefix_ids)

        if partial_terminal_allowed:
            self._allowed_cache[cache_key] = partial_terminal_allowed
            return partial_terminal_allowed

        trimmed, next_prefix_tokens = self._strip_prefix_ids(answer_prefix_ids)

        # Continue enforcing an incomplete prefix.
        if trimmed is None:
            out = list(next_prefix_tokens) if next_prefix_tokens else list(self.final_end_token_ids)
            self._allowed_cache[cache_key] = out
            return out

        state = self._parse_trimmed_ids(trimmed)
        allowed: Set[int] = set()

        if state["mode"] == self.EXPECT_LABEL_START:
            allowed = self._trie_root_children(int(state["min_label_idx"]))
            out = sorted(allowed) if allowed else list(self.final_end_token_ids)
            self._allowed_cache[cache_key] = out
            return out

        if state["mode"] == self.INVALID:
            out = list(self.final_end_token_ids)
            self._allowed_cache[cache_key] = out
            return out

        trie = state.get("trie")

        if trie is None:
            out = list(self.final_end_token_ids)
            self._allowed_cache[cache_key] = out
            return out

        nodes = trie["nodes"]
        node_id = int(state["node_id"])
        node = nodes[node_id]

        for tok in node["children"].keys():
            allowed.add(int(tok))

        terminal_label_idx = state.get("terminal_label_idx", None)

        if terminal_label_idx is not None:
            allowed.update(int(x) for x in self.terminal_start_token_ids)

            if self.allow_multi_label and int(terminal_label_idx) + 1 < len(self.label_list):
                allowed.update(int(x) for x in self.separator_token_ids)

        out = sorted(allowed) if allowed else list(self.final_end_token_ids)

        self._allowed_cache[cache_key] = out
        return out

    def build_prefix_allowed_tokens_fn(
        self,
        prompt_lens: List[int],
        fallback_to_full_vocab: bool = False,
    ) -> Callable[[int, torch.Tensor], List[int]]:
        prompt_lens = list(prompt_lens)

        def _fn(batch_id: int, input_ids) -> List[int]:
            bid = int(batch_id)
            plen = int(prompt_lens[bid]) if 0 <= bid < len(prompt_lens) else 0

            full_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            answer_prefix_ids = full_ids[plen:]

            allowed = self.allowed_next_tokens(answer_prefix_ids)

            if allowed:
                return allowed

            if fallback_to_full_vocab:
                return list(range(len(self.tokenizer)))

            return list(self.final_end_token_ids)

        return _fn



def _categories_to_answer_plain(categories: Any) -> str:
    return _build_prefixed_answer_plain(categories, "categories: ")



def _build_full_token_roles_and_weights(
    tokenizer,
    prompt_ids: List[int],
    answer_ids: List[int],
    labels: List[int],
    categories: List[str],
    answer_plain: str,
    prefix_weight: float = 0.2,
    separator_weight: float = 0.5,
    label_base_weight: float = 1.2,
    later_label_alpha: float = 0.3,
    later_label_power: float = 1.0,
    only_label_weight: float = 1.3,
    terminal_active_weight: float = 1.0,
    terminal_masked_weight: float = 0.0,
    only_label_first_active_terminal_weight: float = 1.5,
    label_token_weight_mode: str = "power",
    label_token_affine_alpha: float = 0.5,
    label_token_power_gamma: float = 0.5,
) -> Tuple[List[str], List[float]]:
    return _build_full_prefixed_token_roles_and_weights_generic(
        tokenizer=tokenizer,
        prompt_ids=prompt_ids,
        answer_ids=answer_ids,
        labels=labels,
        items=categories,
        answer_plain=answer_plain,
        prefix_text="categories: ",
        item_role_prefix="label",
        prefix_weight=prefix_weight,
        separator_weight=separator_weight,
        label_base_weight=label_base_weight,
        later_label_alpha=later_label_alpha,
        later_label_power=later_label_power,
        only_label_weight=only_label_weight,
        terminal_active_weight=terminal_active_weight,
        terminal_masked_weight=terminal_masked_weight,
        first_active_terminal_weight_for_single_item=only_label_first_active_terminal_weight,
        first_active_terminal_role_for_single_item="terminal_active_only_label_first",
        label_token_weight_mode=label_token_weight_mode,
        label_token_affine_alpha=label_token_affine_alpha,
        label_token_power_gamma=label_token_power_gamma,
    )



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
            f"{answer_idx:>3d}  token={tok_text!r:<20}  role={role:<30}  "
            f"weight={float(w):.4f}  active={is_active}"
        )
        answer_idx += 1



def _extract_cls_fields_batch(
    batch,
    *,
    tokenizer,
    body_max_tokens: int,
    label_to_idx: Dict[str, int],
    drop_unknown_categories: bool,
):
    out = {
        "source_id": [],
        "Title": [],
        "Body": [],
        "Keywords": [],
        "Categories": [],
    }

    metas = batch["other_metadata"]
    for i, meta in enumerate(metas):
        meta = meta or {}

        source_id = _pick_first_text(
            meta=meta,
            meta_keys=["id"],
            batch=batch,
            row_idx=i,
            batch_keys=["id"],
        )

        title = _pick_first_text(
            meta=meta,
            meta_keys=["title"],
            batch=batch,
            row_idx=i,
            batch_keys=["title"],
        )

        body = _pick_first_text(
            meta=meta,
            meta_keys=["body", "abstract", "content", "document", "text"],
            batch=batch,
            row_idx=i,
            batch_keys=["body", "abstract", "content", "document", "text"],
        )
        body = _truncate_text_by_tokens(
            tokenizer=tokenizer,
            text=body,
            max_tokens=body_max_tokens,
        )

        keywords = (
            meta.get("keywords", None)
            if "keywords" in meta else
            meta.get("keyword", None)
            if "keyword" in meta else
            meta.get("keyphrases", None)
        )
        if keywords is None:
            keywords = _pick_first_text(
                meta=meta,
                meta_keys=["keywords", "keyword", "keyphrases"],
                batch=batch,
                row_idx=i,
                batch_keys=["keywords", "keyword", "keyphrases"],
            )
        keywords = _normalize_keywords_for_prompt(keywords)

        categories = _normalize_categories(meta.get("categories", []))
        if drop_unknown_categories:
            categories = [c for c in categories if c in label_to_idx]
            categories = _canonicalize_categories(categories, label_to_idx, raise_on_unknown=False)
        else:
            categories = _canonicalize_categories(categories, label_to_idx, raise_on_unknown=True)

        out["source_id"].append(source_id)
        out["Title"].append(title)
        out["Body"].append(body)
        out["Keywords"].append(keywords)
        out["Categories"].append(categories)

    return out



def _filter_keep_title_body_cls_batch(batch):
    return _filter_keep_title_body_target_batch(batch, target_field="Categories")



def _format_prompt_title_body(title: str, body: str) -> str:
    return f"Title: {title}\nBody: {body}\n"



def _format_prompt_title_keyword(title: str, keywords: str) -> str:
    return f"Title: {title}\nKeywords: {keywords}\n"



def _build_prompt_answer_batch_title_body_plus_optional_keyword(
    batch,
    *,
    split_name: str,
    seed: int,
    extra_title_keyword_prob: float,
):
    out = {
        "source_id": [],
        "Title": [],
        "PromptRaw": [],
        "AnswerPlain": [],
        "TaskType": [],
        "Categories": [],
    }

    source_ids = batch["source_id"]
    titles = batch["Title"]
    bodies = batch["Body"]
    keywords_all = batch["Keywords"]
    categories_all = batch["Categories"]

    for source_id, title, body, keywords, categories in zip(
        source_ids, titles, bodies, keywords_all, categories_all
    ):
        source_id = str(source_id or "").strip()
        title = str(title or "").strip()
        body = str(body or "").strip()
        keywords_text = _normalize_keywords_for_prompt(keywords)
        categories = list(categories)
        answer_plain = _categories_to_answer_plain(categories)

        if not source_id:
            source_id = hashlib.md5(f"{title}|||{answer_plain}".encode("utf-8")).hexdigest()[:16]

        out["source_id"].append(source_id)
        out["Title"].append(title)
        out["PromptRaw"].append(_format_prompt_title_body(title=title, body=body))
        out["AnswerPlain"].append(answer_plain)
        out["TaskType"].append("title+body")
        out["Categories"].append(categories)

        if keywords_text and _stable_prob_hit(
            seed=seed,
            split_name=split_name,
            source_id=source_id,
            salt="extra_title_keyword",
            prob=extra_title_keyword_prob,
        ):
            out["source_id"].append(source_id)
            out["Title"].append(title)
            out["PromptRaw"].append(_format_prompt_title_keyword(title=title, keywords=keywords_text))
            out["AnswerPlain"].append(answer_plain)
            out["TaskType"].append("title+keyword")
            out["Categories"].append(categories)

    return out



def _to_trainer_features_batch(
    batch,
    *,
    tokenizer,
    system_text: Optional[str],
    max_length: Optional[int],
    train_terminal_sequences: List[List[int]],
    terminal_loss_mode: str,
    prefix_weight: float,
    separator_weight: float,
    label_base_weight: float,
    later_label_alpha: float,
    later_label_power: float,
    only_label_weight: float,
    terminal_active_weight: float,
):
    def _builder(
        *,
        tokenizer,
        prompt_ids,
        answer_ids,
        labels,
        items,
        answer_plain,
        prefix_weight,
        separator_weight,
        label_base_weight,
        later_label_alpha,
        later_label_power,
        only_label_weight,
        terminal_active_weight,
    ):
        return _build_full_token_roles_and_weights(
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            answer_ids=answer_ids,
            labels=labels,
            categories=items,
            answer_plain=answer_plain,
            prefix_weight=prefix_weight,
            separator_weight=separator_weight,
            label_base_weight=label_base_weight,
            later_label_alpha=later_label_alpha,
            later_label_power=later_label_power,
            only_label_weight=only_label_weight,
            terminal_active_weight=terminal_active_weight,
            terminal_masked_weight=0.0,
        )

    return _to_trainer_features_batch_generic(
        batch=batch,
        tokenizer=tokenizer,
        system_text=system_text,
        max_length=max_length,
        train_terminal_sequences=train_terminal_sequences,
        terminal_loss_mode=terminal_loss_mode,
        prefix_weight=prefix_weight,
        separator_weight=separator_weight,
        label_base_weight=label_base_weight,
        later_label_alpha=later_label_alpha,
        later_label_power=later_label_power,
        only_label_weight=only_label_weight,
        terminal_active_weight=terminal_active_weight,
        target_field_name="Categories",
        extra_passthrough_fields=[],
        role_weight_builder=_builder,
    )



def _build_constraints_for_datasetdict(
    ds_out: DatasetDict,
    tokenizer,
    label_list: List[str],
    label_to_idx: Dict[str, int],
    allow_multi_label: bool = True,
    end_token_ids: Optional[List[int]] = None,
    terminal_sequences: Optional[List[List[int]]] = None,
    terminal_loss_mode: str = "final_only",
):
    fsm = CategoryOutputFSM(
        tokenizer=tokenizer,
        label_list=label_list,
        prefix_text="categories: ",
        allow_multi_label=allow_multi_label,
        end_token_ids=end_token_ids,
        terminal_sequences=terminal_sequences,
    )

    split_bundles = {}
    for split_name, ds_split in ds_out.items():
        prompt_lens = [len(x) for x in ds_split["PromptIds"]]
        split_bundles[split_name] = {
            "prompt_lens": prompt_lens,
            "prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn(
                prompt_lens=prompt_lens,
                fallback_to_full_vocab=False,
            ),
        }

    return {
        "label_list": label_list,
        "label_to_idx": label_to_idx,
        "initial_allowed_token_ids": fsm.initial_allowed_token_ids,
        "end_token_ids": fsm.end_token_ids,
        "terminal_sequences": fsm.terminal_sequences,
        "terminal_start_token_ids": fsm.terminal_start_token_ids,
        "final_end_token_ids": fsm.final_end_token_ids,
        "terminal_loss_mode": terminal_loss_mode,
        "fsm": fsm,
        "allowed_next_tokens": fsm.allowed_next_tokens,
        "build_prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn,
        "splits": split_bundles,
    }



def preview_kptimes_title_cls_dataset(
    ds_out: DatasetDict,
    tokenizer,
    n_preview_per_split: int = 2,
    verify_n: int = 20,
    seed: int = 42,
):
    rng = random.Random(seed)

    for split_name, ds_split in ds_out.items():
        print("=" * 120)
        print(f"[Preview] split={split_name}  rows={len(ds_split)}")
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
            print(f"[idx={idx}] source_id={row['source_id']}  task={row.get('TaskType', '')}")
            print("[PromptRaw]")
            print(row["PromptRaw"])
            print("[AnswerPlain]")
            print(row["AnswerPlain"])
            print("[Decoded loss-bearing answer-only]")
            answer_only_ids = [tid for tid, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
            print(tokenizer.decode(answer_only_ids, skip_special_tokens=False))

            if "LossTokenWeights" in row:
                print("[Loss-bearing token roles]")
                print(row["LossTokenRoles"])
                print("[Loss-bearing token weights]")
                print([round(float(x), 4) for x in row["LossTokenWeights"]])

            _print_weighted_answer_debug(row, tokenizer)

        if verify_n and verify_n > 0:
            verify_trainer_dataset_consistency(ds_split, tokenizer, n_check=min(verify_n, len(ds_split)))



def verify_trainer_dataset_consistency(ds_trainer, tokenizer, n_check: Optional[int] = None):
    total = len(ds_trainer) if n_check is None else min(n_check, len(ds_trainer))

    for i in range(total):
        row = ds_trainer[i]
        re_ids = tokenizer(
            row["FullText"],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        if re_ids != row["input_ids"]:
            raise AssertionError(f"FullText re-tokenize mismatch at row {i}")

        answer_only_ids = [tid for tid, lab in zip(row["input_ids"], row["labels"]) if lab != -100]
        answer_decoded = tokenizer.decode(answer_only_ids, skip_special_tokens=False)

        if answer_decoded != row["LossAnswerText"]:
            raise AssertionError(f"LossAnswerText decode mismatch at row {i}")

        if "loss_weights" in row and len(row["loss_weights"]) != len(row["input_ids"]):
            raise AssertionError(f"loss_weights length mismatch at row {i}")

        if "token_roles" in row and len(row["token_roles"]) != len(row["input_ids"]):
            raise AssertionError(f"token_roles length mismatch at row {i}")

    print(f"Consistency check passed: {total} rows.")



def _labels_union_from_prompt_ds_generic(ds_prompt, universe_set: set) -> set:
    out = set()
    if ds_prompt is None:
        return out

    for row in ds_prompt:
        if "Categories" in row and row["Categories"] is not None:
            vals = row["Categories"]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            labs = []
            for v in vals:
                if isinstance(v, str):
                    parts = re.split(r"[,\|;]", v)
                    for p in parts:
                        lab = str(p).strip().lower()
                        if lab:
                            labs.append(lab)
                else:
                    lab = str(v).strip().lower()
                    if lab:
                        labs.append(lab)
            out |= set(x for x in labs if x in universe_set)
            continue

        answer_text = _get_answer_text(row)
        txt = str(answer_text or "").strip()
        if not txt:
            continue
        m = re.search(r"\[CATEGORIES\](.*?)\[/CATEGORIES\]", txt, flags=re.I | re.S)
        if m:
            txt = m.group(1).strip()
        else:
            m = re.search(r"categories\s*:\s*(.*)", txt, flags=re.I | re.S)
            if m:
                txt = m.group(1).strip()
        txt = txt.splitlines()[0].strip()
        parts = re.split(r"[,\|;]", txt)
        labs = {str(p).strip().lower() for p in parts if str(p).strip()}
        out |= set(x for x in labs if x in universe_set)

    return out



def _random_select_prompt_rows(
    ds_prompt,
    target_n: Optional[int],
    split_name: str,
    universe_set: Optional[set] = None,
    *,
    seed: int = 42,
):
    """
    Randomly select the target number of prompt candidates.
    Returns:
      - ds_sel
      - seen_labels when universe_set is provided, otherwise an empty set
    """
    if ds_prompt is None:
        return None, set()

    n = len(ds_prompt)
    if target_n is not None and target_n < 0:
        raise ValueError(f"[{split_name}] target_n must be >= 0 or None.")

    if n == 0:
        return ds_prompt, set()

    if target_n == 0:
        return Dataset.from_list([]), set()

    if target_n is None or target_n >= n:
        ds_sel = ds_prompt
    else:
        ds_sel = ds_prompt.shuffle(
            seed=seed + _split_seed_offset(split_name)
        ).select(range(int(target_n)))

    if universe_set is None:
        return ds_sel, set()

    seen_labels = _labels_union_from_prompt_ds_generic(ds_sel, universe_set)
    return ds_sel, seen_labels



def _greedy_trim_cover_indices(
    selected_indices: List[int],
    label_sets_by_idx: List[set],
    need_labels: set,
    target_n: int,
    rng: random.Random,
) -> List[int]:
    """
    Trim selected samples toward target_n without breaking coverage requirements.
    """
    selected_indices = list(selected_indices)
    if target_n < 0:
        raise ValueError("target_n must be >= 0.")
    if len(selected_indices) <= target_n:
        return selected_indices

    need_labels = set(need_labels or set())
    cover_counter: Dict[str, int] = {}
    for idx in selected_indices:
        for lab in label_sets_by_idx[idx]:
            if lab in need_labels:
                cover_counter[lab] = cover_counter.get(lab, 0) + 1

    removable = list(selected_indices)
    rng.shuffle(removable)

    kept = list(selected_indices)

    for idx in removable:
        if len(kept) <= target_n:
            break

        row_labs = set(label_sets_by_idx[idx]) & need_labels
        can_remove = True
        for lab in row_labs:
            if cover_counter.get(lab, 0) <= 1:
                can_remove = False
                break

        if not can_remove:
            continue

        kept = [x for x in kept if x != idx]
        for lab in row_labs:
            cover_counter[lab] -= 1

    if len(kept) > target_n:
        rng.shuffle(kept)
        kept = kept[:target_n]

    return kept



def build_kptimes_title_cls_trainer_dataset_v3(
    tokenizer,
    train_samples: Optional[Union[int, float]] = 10000,
    valid_samples: Optional[Union[int, float]] = 200,
    test_samples: Optional[Union[int, float]] = None,
    seed: int = 42,
    body_max_tokens: int = 2000,
    cls_field_probs: Optional[Dict[int, float]] = None,
    extra_title_keyword_prob: float = 0.20,

    picked_train_dir: Optional[str] = None,
    auto_build_picked_if_missing: bool = True,
    dedup_cfg=None,
    dedup_kwargs: Optional[dict] = None,

    universe_csv: str = "small_counter.csv",
    print_category_coverage: bool = True,
    missing_print_limit: int = 50,

    save_root: str = "datasets_downloaded/kptimes_title_cls_v3",
    force_redownload: bool = False,
    force_rebuild: bool = False,
    num_proc: Optional[int] = 8,
    batch_size: int = 1024,
    load_from_cache_file: bool = True,
    save_final_to_disk: bool = False,
    max_length: Optional[int] = None,
    system_text: Optional[str] = None,
    return_constraints: bool = False,
    constraint_source_split: str = "train",
    allow_multi_label: bool = True,
    end_token_ids: Optional[List[int]] = None,
    terminal_sequences: Optional[List[List[int]]] = None,
    terminal_loss_mode: str = "final_only",
    drop_unknown_categories: bool = True,
    print_checks: bool = True,
    n_preview_per_split: int = 2,
    verify_n: int = 20,
    prefix_weight: float = 0.2,
    separator_weight: float = 0.5,
    label_base_weight: float = 1.2,
    later_label_alpha: float = 0.3,
    later_label_power: float = 1.0,
    terminal_active_weight: float = 1.0,
    only_label_weight: float = 1.3,

    task_mode: str = "category",

    keyword_prefix_text: str = "keywords: ",
    keyword_separator_list: Optional[List[str]] = None,
    keyword_universe_from_train_only: bool = False,
    keyword_universe_splits: Optional[List[str]] = None,

    # content token weight in keywords mode:
    #   1 + min(cap, T * log((N + 1) / (df(t) + 1)))
    keyword_token_idf_temperature: float = 0.0,
    keyword_token_idf_cap: float = 0.0,
    print_keyword_max_weight: bool = True,
):

    task_mode = str(task_mode).strip().lower()
    if task_mode in {"cate"}:
        task_mode = "category"
    if task_mode not in {"category", "keywords", "both"}:
        raise ValueError("task_mode must be one of: {'category', 'keywords', 'both'}")

    if cls_field_probs is not None:
        print("[INFO] cls_field_probs is deprecated and ignored in v3.")

    if terminal_loss_mode not in {"all", "final_only", "none"}:
        raise ValueError("terminal_loss_mode must be one of: {'all', 'final_only', 'none'}")

    if later_label_power <= 0:
        raise ValueError("later_label_power must be > 0.")

    def _normalize_keyword_separator_list_local(raw_separator_list: Optional[List[str]]) -> List[str]:
        if raw_separator_list is None:
            raw_separator_list = [","]

        if isinstance(raw_separator_list, str):
            raw_separator_list = [raw_separator_list]

        out: List[str] = []
        seen = set()
        for sep in raw_separator_list:
            s = str(sep or "").strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)

        if not out:
            raise ValueError("keyword_separator_list must contain at least one non-empty separator.")

        return out

    keyword_separator_list = _normalize_keyword_separator_list_local(
        keyword_separator_list if task_mode in {"keywords", "both"} else [","]
    )

    def _join_keywords_with_configured_separators(keywords: Any) -> str:
        if keywords is None:
            return ""
        if isinstance(keywords, str):
            items = [keywords.strip()] if keywords.strip() else []
        elif isinstance(keywords, (list, tuple)):
            items = [str(x).strip() for x in keywords if str(x).strip()]
        else:
            item = str(keywords).strip()
            items = [item] if item else []

        if not items:
            return ""

        pieces = [items[0]]
        for gap_idx, item in enumerate(items[1:]):
            sep = keyword_separator_list[gap_idx % len(keyword_separator_list)]
            pieces.append(f"{sep} {item}")
        return "".join(pieces)

    def _build_keyword_output_example_local(example_keywords: Optional[List[str]] = None) -> str:
        example_keywords = list(example_keywords or ["kw1", "kw2", "kw3", "kw4"])
        return f"{keyword_prefix_text}{_join_keywords_with_configured_separators(example_keywords)}"

    if not picked_train_dir:
        raise ValueError(
            "picked_train_dir is required now: train split is built from the picked dataset, "
            "not directly from raw_train."
        )

    def _default_system_text_for_task(task_mode_local: str) -> str:
        if task_mode_local == "category":
            return (
                "You are a professional news editor.\n"
                "Task: Predict the article categories from the available article information.\n"
                "The available information may include the title, the body, and keywords.\n"
                "Output only one line in the exact format below:\n"
                "categories: cat1, cat2, ...\n"
                "Do not explain.\n"
                "Do not repeat the input.\n"
                "Do not generate any extra text."
            )
        if task_mode_local == "keywords":
            return (
                "You are a professional news editor.\n"
                "Task: Predict the article keywords from the available article information.\n"
                "The available information may include the title and the body.\n"
                "Output only one line in the exact format below:\n"
                f"{_build_keyword_output_example_local()}\n"
                "Do not explain.\n"
                "Do not repeat the input.\n"
                "Do not generate any extra text."
            )
        return (
            "You are a professional news editor.\n"
            "Task: Predict both the article categories and keywords from the available article information.\n"
            "The available information includes the title and the body.\n"
            "Output only two lines in the exact format below, in this exact order:\n"
            "categories: cat1, cat2, ...\n"
            f"{_build_keyword_output_example_local()}\n"
            "Do not explain.\n"
            "Do not repeat the input.\n"
            "Do not generate any extra text."
        )

    if system_text is None:
        system_text = _default_system_text_for_task(task_mode)

    save_root = Path(save_root)
    raw_saved_dir = save_root / "hf_saved_raw"

    cache_payload = {
        "version": "title_cls_trainer_v3_taskmode_switch_v7_both_joint_constraints",
        "task_mode": task_mode,
        "train_samples": train_samples,
        "valid_samples": valid_samples,
        "test_samples": test_samples,
        "seed": seed,
        "body_max_tokens": body_max_tokens,
        "extra_title_keyword_prob": extra_title_keyword_prob if task_mode == "category" else 0.0,
        "picked_train_dir": str(picked_train_dir),
        "universe_csv": universe_csv if task_mode in {"category", "both"} else "",
        "system_text": system_text,
        "terminal_loss_mode": terminal_loss_mode,
        "prefix_weight": prefix_weight,
        "separator_weight": separator_weight,
        "label_base_weight": label_base_weight,
        "later_label_alpha": later_label_alpha,
        "later_label_power": later_label_power,
        "terminal_active_weight": terminal_active_weight,
        "only_label_weight": only_label_weight,
        "max_length": max_length,
        "drop_unknown_categories": drop_unknown_categories if task_mode in {"category", "both"} else False,
        "keyword_prefix_text": keyword_prefix_text if task_mode in {"keywords", "both"} else "",
        "keyword_separator_list": keyword_separator_list if task_mode in {"keywords", "both"} else None,
        "keyword_universe_from_train_only": keyword_universe_from_train_only if task_mode in {"keywords", "both"} else False,
        "keyword_universe_splits": keyword_universe_splits if task_mode in {"keywords", "both"} else None,
        "keyword_token_idf_temperature": keyword_token_idf_temperature if task_mode == "keywords" else 0.0,
        "keyword_token_idf_cap": keyword_token_idf_cap if task_mode == "keywords" else 0.0,
    }
    cache_tag = hashlib.md5(
        json.dumps(cache_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    if task_mode == "category":
        final_saved_dir = save_root / f"title_cls_trainer_v3_pickedcov__{cache_tag}"
    elif task_mode == "keywords":
        final_saved_dir = save_root / f"title_kw_trainer_v3__{cache_tag}"
    else:
        final_saved_dir = save_root / f"title_both_trainer_v3__{cache_tag}"

    num_proc = _sanitize_num_proc(num_proc)
    chat_num_proc = 1

    if raw_saved_dir.exists() and not force_redownload:
        ds_raw = load_from_disk(str(raw_saved_dir))
        if not isinstance(ds_raw, DatasetDict):
            ds_raw = DatasetDict({"train": ds_raw})
    else:
        ds_raw = load_dataset(
            "midas/kptimes",
            revision="refs/convert/parquet",
            data_dir="raw",
            cache_dir=str(save_root / "hf_cache" / "datasets"),
        )
        try:
            save_root.mkdir(parents=True, exist_ok=True)
            ds_raw.save_to_disk(str(raw_saved_dir))
        except Exception as e:
            print(f"[WARN] Failed to save raw dataset to disk: {e}")

    ds_out = None
    if final_saved_dir.exists() and not force_rebuild:
        ds_out = load_from_disk(str(final_saved_dir))
        if not isinstance(ds_out, DatasetDict):
            ds_out = DatasetDict({"train": ds_out})

    def _resolve_raw_split(ds_dict, requested_split: str):
        try:
            actual_split = _resolve_split_name(ds_dict, requested_split)
            return ds_dict[actual_split]
        except Exception:
            if requested_split in ds_dict:
                return ds_dict[requested_split]
            if requested_split == "validation" and "valid" in ds_dict:
                return ds_dict["valid"]
            return None

    def _looks_like_hf_dataset_dir(d: str) -> bool:
        if not os.path.isdir(d):
            return False
        if not os.path.isfile(os.path.join(d, "dataset_info.json")):
            return False
        if not os.path.isfile(os.path.join(d, "state.json")):
            return False
        return any(fn.startswith("data-") for fn in os.listdir(d))

    def _norm_label(x: str) -> str:
        x = str(x).strip()
        if not x:
            return ""
        return x.strip().lower()

    def _as_list_maybe(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        return [x]

    def _compute_target(source_len: int, take_spec, split_name: str, *, none_means_all: bool = True) -> Optional[int]:
        if take_spec is None:
            return None if none_means_all else None

        if isinstance(take_spec, bool):
            raise TypeError(f"[{split_name}] take_spec should not be bool.")

        if isinstance(take_spec, int):
            if take_spec < 0:
                raise ValueError(f"[{split_name}] int take_spec must be >= 0.")
            return take_spec

        if isinstance(take_spec, float):
            if take_spec <= 0:
                return 0
            if take_spec >= 1.0:
                return None
            return max(1, int(source_len * take_spec))

        raise TypeError(f"[{split_name}] Unsupported take_spec type: {type(take_spec)}")

    def _row_key_from_prompt_answer(row: dict) -> str:
        h = hashlib.md5()
        source_id = row.get("source_id", "")
        task_type = row.get("TaskType", "")
        prompt_text = _get_prompt_text(row)
        answer_text = _get_answer_text(row)

        h.update(str(source_id).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(str(task_type).encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(prompt_text.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(answer_text.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    # -----------------------------
    # category-specific helpers
    # -----------------------------
    def _extract_labels_from_raw_ex(raw_ex: dict) -> List[str]:
        out = []
        meta = raw_ex.get("other_metadata", {}) or {}
        cats = meta.get("categories", None)
        if cats is not None:
            vals = _as_list_maybe(cats)
            for v in vals:
                lab = _norm_label(v)
                if lab:
                    out.append(lab)
            if out:
                seen = set()
                keep = []
                for z in out:
                    if z not in seen:
                        seen.add(z)
                        keep.append(z)
                return keep

        candidate_keys = [
            "categories",
            "category_names",
            "label_names",
            "labels_text",
            "cats",
        ]
        for k in candidate_keys:
            if k not in raw_ex:
                continue
            vals = _as_list_maybe(raw_ex.get(k))
            tmp = []
            for v in vals:
                if isinstance(v, str):
                    parts = re.split(r"[,\|;]", v)
                    for p in parts:
                        lab = _norm_label(p)
                        if lab:
                            tmp.append(lab)
                else:
                    lab = _norm_label(v)
                    if lab:
                        tmp.append(lab)
            if tmp:
                seen = set()
                keep = []
                for z in tmp:
                    if z not in seen:
                        seen.add(z)
                        keep.append(z)
                return keep

        return []

    def _parse_labels_from_answer(answer_text: str) -> List[str]:
        txt = str(answer_text or "").strip()
        if not txt:
            return []

        m = re.search(r"\[CATEGORIES\](.*?)\[/CATEGORIES\]", txt, flags=re.I | re.S)
        if m:
            txt = m.group(1).strip()
        else:
            m = re.search(r"categories\s*:\s*(.*)", txt, flags=re.I | re.S)
            if m:
                txt = m.group(1).strip()

        txt = txt.splitlines()[0].strip()
        parts = re.split(r"[,\|;]", txt)

        labs = []
        seen = set()
        for p in parts:
            lab = _norm_label(p)
            if lab and lab not in seen:
                seen.add(lab)
                labs.append(lab)
        return labs

    def _labels_from_prompt_row(row: dict, universe_set: set) -> set:
        if "Categories" in row and row["Categories"] is not None:
            cats_raw = row["Categories"]
            vals = _as_list_maybe(cats_raw)
            labs = []
            for v in vals:
                if isinstance(v, str):
                    parts = re.split(r"[,\|;]", v)
                    for p in parts:
                        lab = _norm_label(p)
                        if lab:
                            labs.append(lab)
                else:
                    lab = _norm_label(v)
                    if lab:
                        labs.append(lab)
            return set([x for x in labs if x in universe_set])

        answer_text = _get_answer_text(row)
        labs = _parse_labels_from_answer(answer_text)
        return set([x for x in labs if x in universe_set])

    def _labels_union_from_prompt_ds_local(ds_prompt, universe_set: set) -> set:
        out = set()
        if ds_prompt is None:
            return out
        for row in ds_prompt:
            out |= _labels_from_prompt_row(row, universe_set)
        return out

    # -----------------------------
    # keywords-specific helpers
    # -----------------------------
    def _normalize_keywords_list(raw_keywords: Any) -> List[str]:
        if raw_keywords is None:
            return []

        if isinstance(raw_keywords, (list, tuple)):
            vals = list(raw_keywords)
        else:
            txt = str(raw_keywords).strip()
            if not txt:
                return []
            vals = re.split(r"[,\|;]", txt)

        out = []
        seen = set()
        for v in vals:
            s = str(v).strip()
            if not s:
                continue
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    def _keywords_to_answer_plain(keywords: Any) -> str:
        kw_text = _join_keywords_with_configured_separators(keywords)
        return f"{keyword_prefix_text}{kw_text}"

    def _extract_keywords_fields_batch(
        batch,
        *,
        tokenizer,
        body_max_tokens: int,
    ):
        out = {
            "source_id": [],
            "Title": [],
            "Body": [],
            "BodyTokenIds": [],
            "Keywords": [],
            "KeywordsList": [],
        }

        metas = batch["other_metadata"]
        for i, meta in enumerate(metas):
            meta = meta or {}

            source_id = _pick_first_text(
                meta=meta,
                meta_keys=["id"],
                batch=batch,
                row_idx=i,
                batch_keys=["id"],
            )

            title = _pick_first_text(
                meta=meta,
                meta_keys=["title"],
                batch=batch,
                row_idx=i,
                batch_keys=["title"],
            )

            body = _pick_first_text(
                meta=meta,
                meta_keys=["body", "abstract", "content", "document", "text"],
                batch=batch,
                row_idx=i,
                batch_keys=["body", "abstract", "content", "document", "text"],
            )
            body = _truncate_text_by_tokens(
                tokenizer=tokenizer,
                text=body,
                max_tokens=body_max_tokens,
            )

            keywords_raw = (
                meta.get("keywords", None)
                if "keywords" in meta else
                meta.get("keyword", None)
                if "keyword" in meta else
                meta.get("keyphrases", None)
            )
            if keywords_raw is None:
                keywords_raw = _pick_first_text(
                    meta=meta,
                    meta_keys=["keywords", "keyword", "keyphrases"],
                    batch=batch,
                    row_idx=i,
                    batch_keys=["keywords", "keyword", "keyphrases"],
                )

            keywords_list = _normalize_keywords_list(keywords_raw)
            keywords_text = ", ".join(keywords_list)
            body_token_ids = sorted(set(int(x) for x in _tokenize_no_special(tokenizer, body))) if body else []

            out["source_id"].append(source_id)
            out["Title"].append(title)
            out["Body"].append(body)
            out["BodyTokenIds"].append(body_token_ids)
            out["Keywords"].append(keywords_text)
            out["KeywordsList"].append(keywords_list)

        return out

    def _filter_keep_title_body_keywords_batch(batch):
        keep = []
        titles = batch["Title"]
        bodies = batch["Body"]
        keywords_all = batch["KeywordsList"]
        for title, body, keywords in zip(titles, bodies, keywords_all):
            ok = bool(_stringify_text(title) and _stringify_text(body) and keywords)
            keep.append(ok)
        return keep

    def _build_prompt_answer_batch_title_body_keywords(
        batch,
        *,
        split_name: str,
        seed: int,
    ):
        del split_name, seed

        out = {
            "source_id": [],
            "Title": [],
            "PromptRaw": [],
            "AnswerPlain": [],
            "TaskType": [],
            "KeywordsList": [],
            "BodyTokenIds": [],
        }

        for source_id, title, body, keywords_list, body_token_ids in zip(
            batch["source_id"],
            batch["Title"],
            batch["Body"],
            batch["KeywordsList"],
            batch["BodyTokenIds"],
        ):
            source_id = str(source_id or "").strip()
            title = str(title or "").strip()
            body = str(body or "").strip()
            keywords_list = list(keywords_list or [])
            answer_plain = _keywords_to_answer_plain(keywords_list)

            if not source_id:
                source_id = hashlib.md5(f"{title}|||{answer_plain}".encode("utf-8")).hexdigest()[:16]

            out["source_id"].append(source_id)
            out["Title"].append(title)
            out["PromptRaw"].append(_format_prompt_title_body(title=title, body=body))
            out["AnswerPlain"].append(answer_plain)
            out["TaskType"].append("title+body")
            out["KeywordsList"].append(keywords_list)
            out["BodyTokenIds"].append(list(body_token_ids or []))

        return out


    def _both_to_answer_plain(categories: Any, keywords: Any) -> str:
        cat_text = _categories_to_answer_plain(categories)
        kw_text = _keywords_to_answer_plain(keywords)
        if cat_text and kw_text:
            return f"{cat_text}\n{kw_text}"
        return cat_text or kw_text

    def _extract_both_fields_batch(
        batch,
        *,
        tokenizer,
        body_max_tokens: int,
        label_to_idx: Dict[str, int],
        drop_unknown_categories: bool,
    ):
        out = {
            "source_id": [],
            "Title": [],
            "Body": [],
            "Keywords": [],
            "KeywordsList": [],
            "Categories": [],
        }

        metas = batch["other_metadata"]
        for i, meta in enumerate(metas):
            meta = meta or {}

            source_id = _pick_first_text(
                meta=meta,
                meta_keys=["id"],
                batch=batch,
                row_idx=i,
                batch_keys=["id"],
            )

            title = _pick_first_text(
                meta=meta,
                meta_keys=["title"],
                batch=batch,
                row_idx=i,
                batch_keys=["title"],
            )

            body = _pick_first_text(
                meta=meta,
                meta_keys=["body", "abstract", "content", "document", "text"],
                batch=batch,
                row_idx=i,
                batch_keys=["body", "abstract", "content", "document", "text"],
            )
            body = _truncate_text_by_tokens(
                tokenizer=tokenizer,
                text=body,
                max_tokens=body_max_tokens,
            )

            keywords_raw = (
                meta.get("keywords", None)
                if "keywords" in meta else
                meta.get("keyword", None)
                if "keyword" in meta else
                meta.get("keyphrases", None)
            )
            if keywords_raw is None:
                keywords_raw = _pick_first_text(
                    meta=meta,
                    meta_keys=["keywords", "keyword", "keyphrases"],
                    batch=batch,
                    row_idx=i,
                    batch_keys=["keywords", "keyword", "keyphrases"],
                )

            keywords_list = _normalize_keywords_list(keywords_raw)
            keywords_text = _normalize_keywords_for_prompt(keywords_list)

            categories = _normalize_categories(meta.get("categories", []))
            if drop_unknown_categories:
                categories = [c for c in categories if c in label_to_idx]
                categories = _canonicalize_categories(categories, label_to_idx, raise_on_unknown=False)
            else:
                categories = _canonicalize_categories(categories, label_to_idx, raise_on_unknown=True)

            out["source_id"].append(source_id)
            out["Title"].append(title)
            out["Body"].append(body)
            out["Keywords"].append(keywords_text)
            out["KeywordsList"].append(keywords_list)
            out["Categories"].append(categories)

        return out

    def _filter_keep_title_body_both_batch(batch):
        keep = []
        titles = batch["Title"]
        bodies = batch["Body"]
        keywords_all = batch["KeywordsList"]
        categories_all = batch["Categories"]
        for title, body, keywords, categories in zip(titles, bodies, keywords_all, categories_all):
            ok = bool(_stringify_text(title) and _stringify_text(body) and keywords and categories)
            keep.append(ok)
        return keep

    def _build_prompt_answer_batch_title_body_both(
        batch,
        *,
        split_name: str,
        seed: int,
    ):
        del split_name, seed

        out = {
            "source_id": [],
            "Title": [],
            "PromptRaw": [],
            "AnswerPlain": [],
            "TaskType": [],
            "Categories": [],
            "KeywordsList": [],
        }

        for source_id, title, body, categories, keywords_list in zip(
            batch["source_id"],
            batch["Title"],
            batch["Body"],
            batch["Categories"],
            batch["KeywordsList"],
        ):
            source_id = str(source_id or "").strip()
            title = str(title or "").strip()
            body = str(body or "").strip()
            categories = list(categories or [])
            keywords_list = list(keywords_list or [])
            answer_plain = _both_to_answer_plain(categories, keywords_list)

            if not source_id:
                source_id = hashlib.md5(f"{title}|||{answer_plain}".encode("utf-8")).hexdigest()[:16]

            out["source_id"].append(source_id)
            out["Title"].append(title)
            out["PromptRaw"].append(_format_prompt_title_body(title=title, body=body))
            out["AnswerPlain"].append(answer_plain)
            out["TaskType"].append("title+body")
            out["Categories"].append(categories)
            out["KeywordsList"].append(keywords_list)

        return out

    def _get_prompt_candidate_task_spec(task_mode_local: str) -> Dict[str, Any]:
        if task_mode_local == "both":
            return {
                "extract_fields_fn": _extract_both_fields_batch,
                "extract_fn_kwargs": {
                    "tokenizer": tokenizer,
                    "body_max_tokens": body_max_tokens,
                    "label_to_idx": label_to_idx,
                    "drop_unknown_categories": drop_unknown_categories,
                },
                "filter_keep_fn": _filter_keep_title_body_both_batch,
                "build_prompt_fn": _build_prompt_answer_batch_title_body_both,
                "build_prompt_fn_kwargs": {},
                "extract_desc": "[{split_name}] extracting both-task fields + truncating body",
                "filter_desc": "[{split_name}] filtering valid title+body both-task rows",
                "prompt_desc": "[{split_name}] building title+body both-task prompts",
            }
        if task_mode_local == "keywords":
            return {
                "extract_fields_fn": _extract_keywords_fields_batch,
                "extract_fn_kwargs": {
                    "tokenizer": tokenizer,
                    "body_max_tokens": body_max_tokens,
                },
                "filter_keep_fn": _filter_keep_title_body_keywords_batch,
                "build_prompt_fn": _build_prompt_answer_batch_title_body_keywords,
                "build_prompt_fn_kwargs": {},
                "extract_desc": "[{split_name}] extracting keyword fields + truncating body",
                "filter_desc": "[{split_name}] filtering valid title+body keyword rows",
                "prompt_desc": "[{split_name}] building title+body keyword prompts",
            }
        if task_mode_local == "category":
            return {
                "extract_fields_fn": _extract_cls_fields_batch,
                "extract_fn_kwargs": {
                    "tokenizer": tokenizer,
                    "body_max_tokens": body_max_tokens,
                    "label_to_idx": label_to_idx,
                    "drop_unknown_categories": drop_unknown_categories,
                },
                "filter_keep_fn": _filter_keep_title_body_cls_batch,
                "build_prompt_fn": _build_prompt_answer_batch_title_body_plus_optional_keyword,
                "build_prompt_fn_kwargs": {
                    "extra_title_keyword_prob": extra_title_keyword_prob,
                },
                "extract_desc": "[{split_name}] extracting cls fields + truncating body",
                "filter_desc": "[{split_name}] filtering valid title+body cls rows",
                "prompt_desc": "[{split_name}] building title+body + optional title+keyword prompts",
            }
        raise ValueError(f"Unknown task_mode_local={task_mode_local!r}")

    def _build_prompt_candidates_from_source_task(
        ds_source,
        *,
        task_mode_local: str,
        split_name: str,
        target_prompt_rows: Optional[int] = None,
        early_stop_when_enough: bool = False,
        **_ignored,
    ):
        del target_prompt_rows, early_stop_when_enough, _ignored
        spec = _get_prompt_candidate_task_spec(task_mode_local)
        build_prompt_fn_kwargs = {"split_name": split_name, "seed": seed}
        build_prompt_fn_kwargs.update(spec.get("build_prompt_fn_kwargs", {}))
        return _build_prompt_candidates_from_source_generic(
            ds_source,
            split_name=split_name,
            seed=seed,
            batch_size=batch_size,
            num_proc=num_proc,
            load_from_cache_file=load_from_cache_file,
            print_checks=print_checks,
            extract_fields_fn=spec["extract_fields_fn"],
            extract_fn_kwargs=spec["extract_fn_kwargs"],
            filter_keep_fn=spec["filter_keep_fn"],
            build_prompt_fn=spec["build_prompt_fn"],
            build_prompt_fn_kwargs=build_prompt_fn_kwargs,
            extract_desc=spec["extract_desc"],
            filter_desc=spec["filter_desc"],
            prompt_desc=spec["prompt_desc"],
        )

    def _coverage_select_prompt_rows(
        ds_prompt,
        target_n: Optional[int],
        split_name: str,
        need_labels: set,
        *,
        universe_set: set,
        seed: int = 42,
    ):
        if ds_prompt is None:
            return None, set(), set(need_labels or set())

        n = len(ds_prompt)
        if target_n is not None and target_n < 0:
            raise ValueError(f"[{split_name}] target_n must be >= 0 or None.")
        if n == 0 or target_n == 0:
            empty = Dataset.from_list([])
            return empty, set(), set(need_labels or set())

        need_labels = set(need_labels or set())
        universe_set = set(universe_set or set())
        rng = random.Random(seed + _split_seed_offset(split_name))

        label_sets_by_idx = []
        for idx in range(n):
            row = ds_prompt[idx]
            labs = set()
            if "Categories" in row and row["Categories"] is not None:
                vals = row["Categories"]
                if not isinstance(vals, (list, tuple)):
                    vals = [vals]
                for v in vals:
                    if isinstance(v, str):
                        for p in re.split(r"[,|;]", v):
                            lab = str(p).strip().lower()
                            if lab and lab in universe_set:
                                labs.add(lab)
                    else:
                        lab = str(v).strip().lower()
                        if lab and lab in universe_set:
                            labs.add(lab)
            else:
                txt = _get_answer_text(row)
                m = re.search(r"categories\s*:\s*(.*)", str(txt or ""), flags=re.I | re.S)
                if m:
                    txt = m.group(1).strip()
                for p in re.split(r"[,|;]", str(txt or "")):
                    lab = str(p).strip().lower()
                    if lab and lab in universe_set:
                        labs.add(lab)
            label_sets_by_idx.append(labs)

        if target_n is None or target_n >= n:
            selected_indices = list(range(n))
        else:
            uncovered = set(need_labels)
            remaining = set(range(n))
            selected_indices = []
            while uncovered and remaining:
                best_idx = None
                best_gain = -1
                for idx in remaining:
                    gain = len(label_sets_by_idx[idx] & uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                if best_idx is None or best_gain <= 0:
                    break
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
                uncovered -= label_sets_by_idx[best_idx]

            need_more = int(target_n) - len(selected_indices)
            if need_more > 0:
                pool = list(remaining)
                rng.shuffle(pool)
                selected_indices.extend(pool[:need_more])

            selected_indices = _greedy_trim_cover_indices(
                selected_indices=selected_indices,
                label_sets_by_idx=label_sets_by_idx,
                need_labels=need_labels,
                target_n=int(target_n),
                rng=rng,
            )

        ds_sel = ds_prompt.select(selected_indices) if selected_indices else Dataset.from_list([])
        available_labels = _labels_union_from_prompt_ds_generic(ds_sel, universe_set)
        missing_need = set(need_labels) - set(available_labels)
        return ds_sel, available_labels, missing_need

    def _build_keyword_token_idf_boost_map(
        ds_raw: DatasetDict,
        tokenizer,
        *,
        keyword_universe_splits: Optional[List[str]] = None,
        keyword_universe_from_train_only: bool = False,
        temperature: float = 0.0,
        cap: float = 0.0,
    ) -> Tuple[Dict[int, float], int]:
        if temperature <= 0:
            return {}, 0

        if keyword_universe_splits is not None:
            split_names = list(keyword_universe_splits)
        elif keyword_universe_from_train_only:
            split_names = ["train"]
        else:
            split_names = list(ds_raw.keys())

        df_counter: Dict[int, int] = {}
        n_docs = 0

        for split_name in split_names:
            if split_name not in ds_raw:
                continue

            ds_split = ds_raw[split_name]
            if len(ds_split) == 0:
                continue

            for meta in ds_split["other_metadata"]:
                meta = meta or {}

                keywords_raw = (
                    meta.get("keywords", None)
                    if "keywords" in meta else
                    meta.get("keyword", None)
                    if "keyword" in meta else
                    meta.get("keyphrases", None)
                )

                keywords_list = _normalize_keywords_list(keywords_raw)
                if not keywords_list:
                    continue

                keywords_text = ", ".join(keywords_list)
                token_ids = _tokenize_no_special(tokenizer, keywords_text)
                uniq_token_ids = set(int(x) for x in token_ids)

                if not uniq_token_ids:
                    continue

                n_docs += 1
                for tid in uniq_token_ids:
                    df_counter[tid] = df_counter.get(tid, 0) + 1

        if n_docs <= 0:
            return {}, 0

        token_boost_map: Dict[int, float] = {}
        max_weight = 1.0

        for tid, df in df_counter.items():
            idf = math.log((n_docs + 1.0) / (float(df) + 1.0))
            raw = float(temperature) * idf
            if cap > 0:
                raw = min(float(cap), raw)

            # Content-token weights have a minimum of 1 instead of 1 + raw.
            weight = max(1.0, raw)

            token_boost_map[int(tid)] = float(weight)
            if weight > max_weight:
                max_weight = weight

        print(
            f"[keyword-token-idf] docs={n_docs}  unique_tokens={len(token_boost_map)}  "
            f"temperature={float(temperature):.4f}  cap={float(cap):.4f}  "
            f"max_weight={float(max_weight):.6f}"
        )

        return token_boost_map, n_docs

    def _build_keyword_answer_piece_specs(
        keywords: List[str],
        prefix_weight: float = 0.2,
        separator_weight: float = 0.5,
        keyword_separator_list: Optional[List[str]] = None,
    ):
        prefix_no_space = keyword_prefix_text[:-1] if keyword_prefix_text.endswith(" ") else keyword_prefix_text
        keyword_separator_list = _normalize_keyword_separator_list_local(keyword_separator_list)
        specs = [
            {
                "role": "prefix",
                "group_key": "prefix",
                "text": prefix_no_space,
                "budget": float(prefix_weight),
            }
        ]

        for idx, kw in enumerate(keywords, start=1):
            if idx > 1:
                sep = keyword_separator_list[(idx - 2) % len(keyword_separator_list)]
                specs.append(
                    {
                        "role": "separator",
                        "group_key": "separator_all",
                        "text": str(sep),
                        "budget": float(separator_weight),
                    }
                )
            specs.append(
                {
                    "role": f"keyword_{idx}",
                    "group_key": f"keyword_{idx}",
                    "text": " " + str(kw),
                    "budget": None,  # content no longer uses group budget
                }
            )
        return specs

    def _build_full_keyword_token_roles_and_weights(
        tokenizer,
        prompt_ids: List[int],
        answer_ids: List[int],
        labels: List[int],
        keywords: List[str],
        answer_plain: str,
        prefix_weight: float = 0.2,
        separator_weight: float = 0.5,
        terminal_active_weight: float = 1.0,
        terminal_masked_weight: float = 0.0,
        keyword_token_idf_boost_map: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[str], List[float]]:
        answer_ids = list(answer_ids)
        answer_labels = list(labels[len(prompt_ids): len(prompt_ids) + len(answer_ids)])

        plain_specs = _build_keyword_answer_piece_specs(
            keywords=keywords,
            prefix_weight=prefix_weight,
            separator_weight=separator_weight,
            keyword_separator_list=keyword_separator_list,
        )
        plain_text_expected = "".join(spec["text"] for spec in plain_specs)

        full_answer_text, full_answer_token_segs = _decode_token_segments_from_ids(tokenizer, answer_ids)

        if plain_text_expected != answer_plain:
            plain_text_expected = answer_plain

        if not full_answer_text.startswith(plain_text_expected):
            raise ValueError(
                f"Decoded keyword answer text does not start with answer_plain.\n"
                f"decoded={full_answer_text!r}\n"
                f"answer_plain={plain_text_expected!r}"
            )

        content_char_len = len(plain_text_expected)

        token_char_cursor = 0
        content_token_count = 0
        for seg in full_answer_token_segs:
            next_cursor = token_char_cursor + len(seg)
            if token_char_cursor < content_char_len:
                content_token_count += 1
            token_char_cursor = next_cursor

        content_ids = answer_ids[:content_token_count]
        terminal_labels = answer_labels[content_token_count:]

        content_text, content_token_segs = _decode_token_segments_from_ids(tokenizer, content_ids)
        if content_text != plain_text_expected:
            raise ValueError(
                f"Keyword content text mismatch.\n"
                f"decoded_content={content_text!r}\n"
                f"expected={plain_text_expected!r}"
            )

        spans = _build_char_spans_from_specs(plain_specs)

        group_budgets: Dict[str, float] = {}
        for sp in spans:
            g = sp["group_key"]
            b = sp["budget"]
            if b is None:
                continue
            b = float(b)
            if g not in group_budgets:
                group_budgets[g] = b
            else:
                if abs(group_budgets[g] - b) > 1e-12:
                    raise ValueError(
                        f"Inconsistent budget for group_key={g!r}: "
                        f"{group_budgets[g]} vs {b}"
                    )

        answer_roles, content_group_keys = _align_token_segments_to_spans(
            content_token_segs,
            spans,
            context="keyword",
        )

        keyword_token_idf_boost_map = keyword_token_idf_boost_map or {}

        answer_weights: List[float] = []
        for tid, role, g in zip(content_ids, answer_roles, content_group_keys):
            if g in ("prefix", "separator_all"):
                budget = group_budgets.get(g, None)
                if budget is None:
                    raise ValueError(f"Missing budget for group_key={g!r}")
                # Format tokens receive a fixed per-instance weight.
                final_w = float(budget)
            else:
                # Content tokens use precomputed weights with a 1.0 fallback.
                final_w = float(keyword_token_idf_boost_map.get(int(tid), 1.0))
            answer_weights.append(final_w)

        for lab in terminal_labels:
            if lab != -100:
                answer_roles.append("terminal_active")
                answer_weights.append(float(terminal_active_weight))
            else:
                answer_roles.append("terminal_masked")
                answer_weights.append(float(terminal_masked_weight))

        if len(answer_roles) != len(answer_ids) or len(answer_weights) != len(answer_ids):
            raise ValueError(
                f"Keyword role/weight length mismatch: roles={len(answer_roles)}, "
                f"weights={len(answer_weights)}, answer_ids={len(answer_ids)}"
            )

        full_roles = (["prompt"] * len(prompt_ids)) + answer_roles
        full_weights = ([0.0] * len(prompt_ids)) + answer_weights
        return full_roles, full_weights

    def _to_trainer_features_batch_keywords(
        batch,
        *,
        tokenizer,
        system_text: Optional[str],
        max_length: Optional[int],
        train_terminal_sequences: List[List[int]],
        terminal_loss_mode: str,
        prefix_weight: float,
        separator_weight: float,
        terminal_active_weight: float,
        keyword_token_idf_boost_map: Optional[Dict[int, float]] = None,
    ):
        out = {
            "source_id": [],
            "Title": [],
            "PromptRaw": [],
            "AnswerPlain": [],
            "TaskType": [],
            "KeywordsList": [],
            "BodyTokenIds": [],
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
        }

        task_types = batch["TaskType"] if "TaskType" in batch else [""] * len(batch["PromptRaw"])

        for source_id, title, prompt_raw, answer_plain, task_type, keywords_list, body_token_ids in zip(
            batch["source_id"],
            batch["Title"],
            batch["PromptRaw"],
            batch["AnswerPlain"],
            task_types,
            batch["KeywordsList"],
            batch["BodyTokenIds"],
        ):
            keywords_list = list(keywords_list or [])
            body_token_ids = list(body_token_ids or [])

            built = _build_chat_ids_and_texts(
                tokenizer=tokenizer,
                user_text=prompt_raw,
                assistant_text_plain=answer_plain,
                system_text=system_text,
            )

            full_ids = list(built["full_ids"])
            prompt_ids = list(built["prompt_ids"])
            answer_ids = list(built["answer_ids"])

            labels, _ = _build_answer_labels_with_terminal_mask(
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                terminal_sequences=train_terminal_sequences,
                terminal_loss_mode=terminal_loss_mode,
            )

            token_roles, loss_weights = _build_full_keyword_token_roles_and_weights(
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                labels=labels,
                keywords=keywords_list,
                answer_plain=answer_plain,
                prefix_weight=prefix_weight,
                separator_weight=separator_weight,
                terminal_active_weight=terminal_active_weight,
                terminal_masked_weight=0.0,
                keyword_token_idf_boost_map=keyword_token_idf_boost_map,
            )

            if max_length is not None:
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
            out["Title"].append(title)
            out["PromptRaw"].append(prompt_raw)
            out["AnswerPlain"].append(answer_plain)
            out["TaskType"].append(task_type)
            out["KeywordsList"].append(keywords_list)
            out["BodyTokenIds"].append(body_token_ids)
            out["PromptText"].append(built["prompt_text"])
            out["AnswerText"].append(built["answer_text"])
            out["LossAnswerText"].append(loss_answer_text)
            out["FullText"].append(built["full_text"])
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

        return out


    def _build_both_answer_piece_specs(
        categories: List[str],
        keywords: List[str],
        prefix_weight: float = 0.2,
        separator_weight: float = 0.5,
        keyword_separator_list: Optional[List[str]] = None,
    ):
        keyword_separator_list = _normalize_keyword_separator_list_local(keyword_separator_list)
        cat_prefix_text = "categories: "
        cat_prefix_no_space = cat_prefix_text[:-1] if cat_prefix_text.endswith(" ") else cat_prefix_text
        kw_prefix_no_space = keyword_prefix_text[:-1] if keyword_prefix_text.endswith(" ") else keyword_prefix_text

        specs = [
            {
                "role": "cat_prefix",
                "group_key": "cat_prefix",
                "text": cat_prefix_no_space,
                "budget": float(prefix_weight),
            }
        ]

        for idx, item in enumerate(categories, start=1):
            if idx > 1:
                specs.append(
                    {
                        "role": "cat_separator",
                        "group_key": "cat_separator_all",
                        "text": ",",
                        "budget": float(separator_weight),
                    }
                )
            specs.append(
                {
                    "role": f"cat_label_{idx}",
                    "group_key": f"cat_label_{idx}",
                    "text": " " + str(item),
                    "budget": None,
                }
            )

        specs.append(
            {
                "role": "bridge_newline",
                "group_key": "bridge_newline",
                "text": "\n",
                "budget": float(separator_weight),
            }
        )
        specs.append(
            {
                "role": "kw_prefix",
                "group_key": "kw_prefix",
                "text": kw_prefix_no_space,
                "budget": float(prefix_weight),
            }
        )

        for idx, kw in enumerate(keywords, start=1):
            if idx > 1:
                sep = keyword_separator_list[(idx - 2) % len(keyword_separator_list)]
                specs.append(
                    {
                        "role": "kw_separator",
                        "group_key": "kw_separator_all",
                        "text": str(sep),
                        "budget": float(separator_weight),
                    }
                )
            specs.append(
                {
                    "role": f"keyword_{idx}",
                    "group_key": f"keyword_{idx}",
                    "text": " " + str(kw),
                    "budget": None,
                }
            )

        return specs

    def _build_full_both_token_roles_and_weights(
        tokenizer,
        prompt_ids: List[int],
        answer_ids: List[int],
        labels: List[int],
        categories: List[str],
        keywords: List[str],
        answer_plain: str,
        prefix_weight: float = 0.2,
        separator_weight: float = 0.5,
        terminal_active_weight: float = 1.0,
        terminal_masked_weight: float = 0.0,
    ) -> Tuple[List[str], List[float]]:
        answer_ids = list(answer_ids)
        answer_labels = list(labels[len(prompt_ids): len(prompt_ids) + len(answer_ids)])

        plain_specs = _build_both_answer_piece_specs(
            categories=categories,
            keywords=keywords,
            prefix_weight=prefix_weight,
            separator_weight=separator_weight,
            keyword_separator_list=keyword_separator_list,
        )
        plain_text_expected = "".join(spec["text"] for spec in plain_specs)

        full_answer_text, full_answer_token_segs = _decode_token_segments_from_ids(tokenizer, answer_ids)

        if plain_text_expected != answer_plain:
            plain_text_expected = answer_plain

        if not full_answer_text.startswith(plain_text_expected):
            raise ValueError(
                f"Decoded BOTH answer text does not start with answer_plain.\n"
                f"decoded={full_answer_text!r}\n"
                f"answer_plain={plain_text_expected!r}"
            )

        content_char_len = len(plain_text_expected)

        token_char_cursor = 0
        content_token_count = 0
        for seg in full_answer_token_segs:
            next_cursor = token_char_cursor + len(seg)
            if token_char_cursor < content_char_len:
                content_token_count += 1
            token_char_cursor = next_cursor

        content_ids = answer_ids[:content_token_count]
        terminal_labels = answer_labels[content_token_count:]

        content_text, content_token_segs = _decode_token_segments_from_ids(tokenizer, content_ids)
        if content_text != plain_text_expected:
            raise ValueError(
                f"BOTH content text mismatch.\n"
                f"decoded_content={content_text!r}\n"
                f"expected={plain_text_expected!r}"
            )

        spans = _build_char_spans_from_specs(plain_specs)

        group_budgets: Dict[str, float] = {}
        for sp in spans:
            g = sp["group_key"]
            b = sp["budget"]
            if b is None:
                continue
            b = float(b)
            if g not in group_budgets:
                group_budgets[g] = b
            elif abs(group_budgets[g] - b) > 1e-12:
                raise ValueError(
                    f"Inconsistent budget for group_key={g!r}: "
                    f"{group_budgets[g]} vs {b}"
                )

        answer_roles, content_group_keys = _align_token_segments_to_spans(
            content_token_segs,
            spans,
            context="BOTH",
        )

        answer_weights: List[float] = []
        for tid, role, g in zip(content_ids, answer_roles, content_group_keys):
            del tid, role
            if g.startswith("cat_label_") or g.startswith("keyword_"):
                final_w = 1.0
            else:
                budget = group_budgets.get(g, None)
                if budget is None:
                    raise ValueError(f"Missing budget for group_key={g!r}")
                final_w = float(budget)
            answer_weights.append(final_w)

        for lab in terminal_labels:
            if lab != -100:
                answer_roles.append("terminal_active")
                answer_weights.append(float(terminal_active_weight))
            else:
                answer_roles.append("terminal_masked")
                answer_weights.append(float(terminal_masked_weight))

        if len(answer_roles) != len(answer_ids) or len(answer_weights) != len(answer_ids):
            raise ValueError(
                f"BOTH role/weight length mismatch: roles={len(answer_roles)}, "
                f"weights={len(answer_weights)}, answer_ids={len(answer_ids)}"
            )

        full_roles = (["prompt"] * len(prompt_ids)) + answer_roles
        full_weights = ([0.0] * len(prompt_ids)) + answer_weights
        return full_roles, full_weights

    def _to_trainer_features_batch_both(
        batch,
        *,
        tokenizer,
        system_text: Optional[str],
        max_length: Optional[int],
        train_terminal_sequences: List[List[int]],
        terminal_loss_mode: str,
        prefix_weight: float,
        separator_weight: float,
        terminal_active_weight: float,
    ):
        out = {
            "source_id": [],
            "Title": [],
            "PromptRaw": [],
            "AnswerPlain": [],
            "TaskType": [],
            "Categories": [],
            "KeywordsList": [],
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
        }

        task_types = batch["TaskType"] if "TaskType" in batch else [""] * len(batch["PromptRaw"])

        for source_id, title, prompt_raw, answer_plain, task_type, categories, keywords_list in zip(
            batch["source_id"],
            batch["Title"],
            batch["PromptRaw"],
            batch["AnswerPlain"],
            task_types,
            batch["Categories"],
            batch["KeywordsList"],
        ):
            categories = list(categories or [])
            keywords_list = list(keywords_list or [])

            built = _build_chat_ids_and_texts(
                tokenizer=tokenizer,
                user_text=prompt_raw,
                assistant_text_plain=answer_plain,
                system_text=system_text,
            )

            full_ids = list(built["full_ids"])
            prompt_ids = list(built["prompt_ids"])
            answer_ids = list(built["answer_ids"])

            labels, _ = _build_answer_labels_with_terminal_mask(
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                terminal_sequences=train_terminal_sequences,
                terminal_loss_mode=terminal_loss_mode,
            )

            token_roles, loss_weights = _build_full_both_token_roles_and_weights(
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                labels=labels,
                categories=categories,
                keywords=keywords_list,
                answer_plain=answer_plain,
                prefix_weight=prefix_weight,
                separator_weight=separator_weight,
                terminal_active_weight=terminal_active_weight,
                terminal_masked_weight=0.0,
            )

            if max_length is not None:
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
            out["Title"].append(title)
            out["PromptRaw"].append(prompt_raw)
            out["AnswerPlain"].append(answer_plain)
            out["TaskType"].append(task_type)
            out["Categories"].append(categories)
            out["KeywordsList"].append(keywords_list)
            out["PromptText"].append(built["prompt_text"])
            out["AnswerText"].append(built["answer_text"])
            out["LossAnswerText"].append(loss_answer_text)
            out["FullText"].append(built["full_text"])
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

        return out

    def _extract_keyword_token_ids_batch(
        batch,
        *,
        tokenizer,
    ):
        out = {"keyword_token_ids": []}

        metas = batch["other_metadata"]
        for i, meta in enumerate(metas):
            meta = meta or {}

            keywords_raw = (
                meta.get("keywords", None)
                if "keywords" in meta else
                meta.get("keyword", None)
                if "keyword" in meta else
                meta.get("keyphrases", None)
            )
            if keywords_raw is None:
                keywords_raw = _pick_first_text(
                    meta=meta,
                    meta_keys=["keywords", "keyword", "keyphrases"],
                    batch=batch,
                    row_idx=i,
                    batch_keys=["keywords", "keyword", "keyphrases"],
                )

            keywords_list = _normalize_keywords_list(keywords_raw)
            keywords_text = ", ".join(keywords_list)
            if keywords_text:
                ids = sorted(set(int(x) for x in _tokenize_no_special(tokenizer, keywords_text)))
            else:
                ids = []
            out["keyword_token_ids"].append(ids)

        return out

    def _build_global_keyword_token_universe(ds_raw: DatasetDict, tokenizer) -> List[int]:
        all_ids = set()

        if keyword_universe_splits is not None:
            split_names = list(keyword_universe_splits)
        elif keyword_universe_from_train_only:
            split_names = ["train"]
        else:
            split_names = list(ds_raw.keys())

        for split_name in split_names:
            if split_name not in ds_raw:
                continue
            ds_split = ds_raw[split_name]
            if len(ds_split) == 0:
                continue

            ds_kw = ds_split.map(
                _extract_keyword_token_ids_batch,
                batched=True,
                batch_size=max(64, min(batch_size, 1024)),
                num_proc=num_proc,
                load_from_cache_file=load_from_cache_file,
                desc=f"[{split_name}] collecting keyword token ids",
                fn_kwargs={"tokenizer": tokenizer},
                remove_columns=ds_split.column_names,
            )

            for ids in ds_kw["keyword_token_ids"]:
                all_ids.update(int(x) for x in ids)

        return sorted(all_ids)

    # =========================================================
    # Word-boundary helpers for keywords mode
    # =========================================================

    _WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[&'./-][A-Za-z0-9]+)*")

    def _extract_word_candidates(text: str) -> List[str]:
        text = str(text or "").strip()
        if not text:
            return []
        return [m.group(0).strip() for m in _WORD_RE.finditer(text)]

    def _normalize_word_key(word: str) -> str:
        return str(word or "").strip().lower()

    _WORD_VARIANT_PATH_CACHE: Dict[str, List[List[int]]] = {}

    def _batch_tokenize_word_variants(tokenizer, words: List[str]) -> Dict[str, List[List[int]]]:
        """
        Tokenize word variants in a batch to avoid repeated Python-level calls.
        Returns: {normalized_word_key: [path1, path2, ...]}
        """
        uniq_words: List[str] = []
        seen_word_keys = set()
        out: Dict[str, List[List[int]]] = {}

        for w in words:
            w = str(w or "").strip()
            if not w:
                continue
            wk = _normalize_word_key(w)
            if not wk or wk in seen_word_keys:
                continue
            seen_word_keys.add(wk)

            cached = _WORD_VARIANT_PATH_CACHE.get(w)
            if cached is not None:
                out[wk] = cached
            else:
                uniq_words.append(w)

        if not uniq_words:
            return out

        variant_texts: List[str] = []
        variant_owner_keys: List[str] = []
        variant_orders: List[int] = []
        for w in uniq_words:
            wk = _normalize_word_key(w)
            variant_texts.append(w)
            variant_owner_keys.append(wk)
            variant_orders.append(0)
            variant_texts.append(" " + w)
            variant_owner_keys.append(wk)
            variant_orders.append(1)

        encoded = tokenizer(
            variant_texts,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        temp: Dict[str, List[Tuple[int, List[int]]]] = {}
        for wk, ord_i, ids in zip(variant_owner_keys, variant_orders, encoded):
            ids = [int(x) for x in ids if x is not None]
            if not ids:
                continue
            temp.setdefault(wk, []).append((ord_i, ids))

        for w in uniq_words:
            wk = _normalize_word_key(w)
            variants = temp.get(wk, [])
            variants.sort(key=lambda x: x[0])
            paths: List[List[int]] = []
            seen_paths = set()
            for _, ids in variants:
                key = tuple(ids)
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                paths.append(ids)
            _WORD_VARIANT_PATH_CACHE[w] = paths
            out[wk] = paths

        return out

    def _tokenize_word_variants(tokenizer, word: str) -> List[List[int]]:
        word = str(word or "").strip()
        if not word:
            return []
        cached = _WORD_VARIANT_PATH_CACHE.get(word)
        if cached is not None:
            return cached
        wk = _normalize_word_key(word)
        return _batch_tokenize_word_variants(tokenizer, [word]).get(wk, [])

    def _build_global_keyword_word_paths(
        tokenizer,
        global_keyword_words: List[str],
    ) -> Dict[str, List[List[int]]]:
        return _batch_tokenize_word_variants(tokenizer, global_keyword_words)


    def _build_global_keyword_word_universe_train_only(ds_raw: DatasetDict) -> List[str]:
        """
        Build the word-level keyword vocabulary only from the complete raw
        training split. Validation and test are not included by default.
        """
        word_seen = set()
        out: List[str] = []

        if "train" not in ds_raw:
            raise KeyError("ds_raw does not contain 'train' split.")

        ds_split = ds_raw["train"]
        if len(ds_split) == 0:
            return out

        metas = ds_split["other_metadata"] if "other_metadata" in ds_split.column_names else []
        for meta in metas:
            meta = meta or {}
            kw_raw = (
                meta.get("keywords", None)
                if "keywords" in meta else
                meta.get("keyword", None)
                if "keyword" in meta else
                meta.get("keyphrases", None)
            )
            kw_text = _normalize_keywords_for_prompt(kw_raw)

            for w in _extract_word_candidates(kw_text):
                k = _normalize_word_key(w)
                if not k or k in word_seen:
                    continue
                word_seen.add(k)
                out.append(w)

        return out

    def _flatten_unique_paths_from_word_path_map(
        word_paths_map: Dict[str, List[List[int]]],
    ) -> List[List[int]]:
        out: List[List[int]] = []
        seen = set()
        for _, paths in word_paths_map.items():
            for path in (paths or []):
                key = tuple(int(x) for x in (path or []))
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(list(key))
        return out

    def _build_trie_from_paths(paths: List[List[int]]) -> Dict[str, Any]:
        nodes = [
            {
                "children": {},
                "is_terminal": False,
                "path_id": None,
            }
        ]

        unique_paths: List[List[int]] = []
        seen = set()
        for path in paths:
            key = tuple(int(x) for x in (path or []))
            if not key or key in seen:
                continue
            seen.add(key)
            unique_paths.append(list(key))

        for path_id, path in enumerate(unique_paths):
            cur = 0
            for tid in path:
                children = nodes[cur]["children"]
                if tid not in children:
                    children[tid] = len(nodes)
                    nodes.append(
                        {
                            "children": {},
                            "is_terminal": False,
                            "path_id": None,
                        }
                    )
                cur = children[tid]
            nodes[cur]["is_terminal"] = True
            nodes[cur]["path_id"] = int(path_id)

        return {
            "nodes": nodes,
            "root_id": 0,
            "paths": unique_paths,
        }

    def _extract_body_from_prompt_raw(prompt_raw: str) -> str:
        prompt_raw = str(prompt_raw or "")
        marker = "\nBody:"
        pos = prompt_raw.find(marker)
        if pos != -1:
            return prompt_raw[pos + len(marker):].strip()

        marker2 = "Body:"
        pos = prompt_raw.find(marker2)
        if pos != -1:
            return prompt_raw[pos + len(marker2):].strip()

        return ""

    def _build_local_sample_word_path_map(
        tokenizer,
        title: str,
        body: str,
        global_word_to_paths: Dict[str, List[List[int]]],
    ) -> Dict[str, List[List[int]]]:
        text = f"{title or ''}\n{body or ''}".strip()
        cand_words = _extract_word_candidates(text)

        local_words: List[str] = []
        seen_local = set()
        for w in cand_words:
            wk = _normalize_word_key(w)
            if not wk:
                continue
            if wk in global_word_to_paths:
                continue
            if wk in seen_local:
                continue
            seen_local.add(wk)
            local_words.append(w)

        return _batch_tokenize_word_variants(tokenizer, local_words)

    @dataclass
    class _KeywordSampleRef:
        split_name: str
        sample_idx: int

    class _SimpleLRUCache:
        def __init__(self, max_size: int = 256):
            self.max_size = max(1, int(max_size))
            self._data: OrderedDict[Any, Any] = OrderedDict()

        def get(self, key):
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

        def set(self, key, value):
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    class KeywordOutputFSM:
        """
        Low-memory implementation:
        - Store global training-word paths in one shared trie.
        - Do not store complete sample_allowed_word_paths per sample.
        - Build a local trie for new body words on demand with a small LRU cache.
        - Legal paths are the union of global training words and local body words.

        Semantics:
        - Each constraint unit is a complete word token path.
        - Separators are not allowed inside a word.
        - A separator, terminal, or next word is allowed only after a full word.
        """

        EXPECT_WORD_START = "EXPECT_WORD_START"
        IN_WORD = "IN_WORD"
        AFTER_WORD = "AFTER_WORD"
        INVALID = "INVALID"

        def __init__(
            self,
            tokenizer,
            global_word_to_paths: Dict[str, List[List[int]]],
            split_to_titles: Dict[str, List[str]],
            split_to_prompt_raws: Dict[str, List[str]],
            prefix_text: str = "keywords: ",
            keyword_separator_list: Optional[List[str]] = None,
            end_token_ids: Optional[List[int]] = None,
            terminal_sequences: Optional[List[List[int]]] = None,
            local_cache_size: int = 256,
        ):
            self.tokenizer = tokenizer
            self.prefix_text = str(prefix_text or "")
            self.keyword_separator_list = _normalize_keyword_separator_list_local(keyword_separator_list)
            self.global_word_to_paths = dict(global_word_to_paths)
            self.split_to_titles = split_to_titles
            self.split_to_prompt_raws = split_to_prompt_raws

            global_paths = _flatten_unique_paths_from_word_path_map(self.global_word_to_paths)
            self.global_trie = _build_trie_from_paths(global_paths)

            if terminal_sequences is not None:
                seqs = []
                for seq in terminal_sequences:
                    seq = [int(x) for x in seq if x is not None]
                    if seq:
                        seqs.append(seq)
                self.terminal_sequences = _unique_preserve_order(seqs)
            elif end_token_ids is not None:
                seqs = []
                for tid in end_token_ids:
                    if tid is not None:
                        seqs.append([int(tid)])
                self.terminal_sequences = _unique_preserve_order(seqs)
            else:
                self.terminal_sequences = _resolve_default_terminal_sequences(tokenizer)

            if not self.terminal_sequences:
                raise ValueError("No valid terminal sequences available for KeywordOutputFSM.")

            self.end_token_ids = sorted({int(tid) for seq in self.terminal_sequences for tid in seq})
            self.terminal_start_token_ids = sorted({int(seq[0]) for seq in self.terminal_sequences if seq})
            self.final_end_token_ids = sorted({int(seq[-1]) for seq in self.terminal_sequences if seq})

            self.prefix_token_variants: List[List[int]] = []
            prefix_variants_raw = [self.prefix_text]
            if self.prefix_text.endswith(" "):
                prefix_variants_raw.append(self.prefix_text.rstrip())

            seen_variants = set()
            for txt in prefix_variants_raw:
                ids = tokenizer(txt, add_special_tokens=False)["input_ids"]
                key = tuple(int(x) for x in ids)
                if key and key not in seen_variants:
                    seen_variants.add(key)
                    self.prefix_token_variants.append(list(key))
            self.prefix_token_variants.sort(key=len, reverse=True)

            self.separator_token_ids = self._scan_separator_token_ids()
            self.comma_token_ids = list(self.separator_token_ids)

            self._allowed_cache: Dict[Tuple[Any, Tuple[int, ...]], List[int]] = {}
            self._local_info_cache = _SimpleLRUCache(max_size=local_cache_size)

            self.initial_allowed_token_ids = self.allowed_next_tokens(
                answer_prefix_ids=[],
                sample_ref=None,
            )

        def _scan_separator_token_ids(self) -> List[int]:
            out = set()
            separator_compacts = {re.sub(r"\s+", "", str(sep)) for sep in self.keyword_separator_list if str(sep).strip()}

            for raw_sep in self.keyword_separator_list:
                sep = str(raw_sep or "").strip()
                if not sep:
                    continue
                variants = [sep]
                if not sep.startswith(" "):
                    variants.append(" " + sep)
                for s in variants:
                    try:
                        ids = self.tokenizer.encode(s, add_special_tokens=False)
                        if len(ids) == 1:
                            out.add(int(ids[0]))
                    except Exception:
                        pass

            try:
                vocab_size = len(self.tokenizer)
            except Exception:
                vocab_size = 0

            special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
            for tid in range(vocab_size):
                if tid in special_ids:
                    continue
                try:
                    txt = self.tokenizer.decode([tid], skip_special_tokens=False)
                except Exception:
                    continue
                if not txt:
                    continue
                compact = re.sub(r"\s+", "", txt)
                if compact in separator_compacts:
                    out.add(int(tid))

            if not out:
                raise ValueError(
                    f"No separator token ids found for keyword_separator_list={self.keyword_separator_list!r}"
                )

            return sorted(out)

        def _strip_prefix_ids(self, answer_prefix_ids: List[int]) -> List[int]:
            ids = list(int(x) for x in (answer_prefix_ids or []))
            for pref in self.prefix_token_variants:
                if len(ids) >= len(pref) and ids[:len(pref)] == pref:
                    return ids[len(pref):]
            return ids

        def _sample_key(self, sample_ref: Optional[_KeywordSampleRef]) -> Any:
            if sample_ref is None:
                return ("GLOBAL", -1)
            return (sample_ref.split_name, int(sample_ref.sample_idx))

        def _build_local_sample_info(self, sample_ref: _KeywordSampleRef) -> Dict[str, Any]:
            split_name = str(sample_ref.split_name)
            sample_idx = int(sample_ref.sample_idx)

            titles = self.split_to_titles.get(split_name, [])
            prompt_raws = self.split_to_prompt_raws.get(split_name, [])
            title = titles[sample_idx] if 0 <= sample_idx < len(titles) else ""
            prompt_raw = prompt_raws[sample_idx] if 0 <= sample_idx < len(prompt_raws) else ""
            body = _extract_body_from_prompt_raw(prompt_raw)

            local_word_to_paths = _build_local_sample_word_path_map(
                tokenizer=self.tokenizer,
                title=title,
                body=body,
                global_word_to_paths=self.global_word_to_paths,
            )
            local_paths = _flatten_unique_paths_from_word_path_map(local_word_to_paths)
            local_trie = _build_trie_from_paths(local_paths)

            return {
                "title": title,
                "body": body,
                "local_word_to_paths": local_word_to_paths,
                "local_trie": local_trie,
            }

        def _get_local_sample_info(self, sample_ref: Optional[_KeywordSampleRef]) -> Optional[Dict[str, Any]]:
            if sample_ref is None:
                return None
            key = self._sample_key(sample_ref)
            cached = self._local_info_cache.get(key)
            if cached is not None:
                return cached
            info = self._build_local_sample_info(sample_ref)
            self._local_info_cache.set(key, info)
            return info

        def _traverse_trie(self, trie: Dict[str, Any], prefix_tokens: List[int]) -> List[int]:
            if trie is None:
                return []
            nodes = trie.get("nodes", [])
            if len(nodes) == 0:
                return []

            cur = int(trie.get("root_id", 0))
            for tid in prefix_tokens:
                children = nodes[cur]["children"]
                if int(tid) not in children:
                    return []
                cur = int(children[int(tid)])
            return [cur]

        def _token_text(self, tid: int) -> str:
            try:
                tok = self.tokenizer.convert_ids_to_tokens(int(tid))
                if isinstance(tok, (list, tuple)):
                    tok = tok[0]
                return "" if tok is None else str(tok)
            except Exception:
                return ""

        def _is_whitespace_only_token(self, tid: int) -> bool:
            try:
                txt = self.tokenizer.decode(
                    [int(tid)],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except Exception:
                return False
            return bool(txt) and txt.isspace()

        def _is_word_start_token(self, tid: int) -> bool:
            tok = self._token_text(tid)
            if not tok:
                return False

            # Common word starts for SentencePiece, Metaspace, and byte-level BPE.
            if tok.startswith("▁") or tok.startswith("Ġ"):
                return True

            # A WordPiece continuation is not a word start.
            if tok.startswith("##"):
                return False

            return False

        def _current_word_prefix(self, trimmed_ids: List[int]) -> List[int]:
            if len(trimmed_ids) == 0:
                return []

            # 1) Keep the suffix after the final separator.
            last_sep = -1
            for i, tid in enumerate(trimmed_ids):
                if int(tid) in self.separator_token_ids:
                    last_sep = i
            suffix = list(trimmed_ids[last_sep + 1:])

            if len(suffix) == 0:
                return []

            # 2) Remove leading whitespace-only tokens from the suffix.
            while len(suffix) > 0 and self._is_whitespace_only_token(suffix[0]):
                suffix = suffix[1:]

            if len(suffix) == 0:
                return []

            # 3) Search backward for the nearest word start.
            for i in range(len(suffix) - 1, -1, -1):
                if self._is_word_start_token(suffix[i]):
                    return suffix[i:]

            # 4) Fall back to the complete suffix when no word start is found.
            return suffix

        def _matched_nodes(self, current_prefix: List[int], sample_ref: Optional[_KeywordSampleRef]) -> Tuple[List[int], List[int]]:
            global_nodes = self._traverse_trie(self.global_trie, current_prefix)
            local_info = self._get_local_sample_info(sample_ref)
            local_trie = local_info["local_trie"] if local_info is not None else None
            local_nodes = self._traverse_trie(local_trie, current_prefix) if local_trie is not None else []
            return global_nodes, local_nodes

        def _is_terminal_match(self, current_prefix: List[int], sample_ref: Optional[_KeywordSampleRef]) -> bool:
            if len(current_prefix) == 0:
                return False
            global_nodes, local_nodes = self._matched_nodes(current_prefix, sample_ref)
            gnodes = self.global_trie["nodes"]
            for nid in global_nodes:
                if gnodes[nid]["is_terminal"]:
                    return True
            local_info = self._get_local_sample_info(sample_ref)
            if local_info is not None:
                lnodes = local_info["local_trie"]["nodes"]
                for nid in local_nodes:
                    if lnodes[nid]["is_terminal"]:
                        return True
            return False

        def _allowed_after_partial_terminal_sequence(
            self,
            answer_prefix_ids: List[int],
            sample_ref: Optional[_KeywordSampleRef],
        ) -> List[int]:
            allowed = set()
            ids = list(answer_prefix_ids)

            for seq in self.terminal_sequences:
                if len(seq) <= 1:
                    continue

                for consumed_len in range(1, len(seq)):
                    if len(ids) < consumed_len:
                        continue
                    if ids[-consumed_len:] != seq[:consumed_len]:
                        continue

                    stem_ids = ids[:-consumed_len]
                    trimmed = self._strip_prefix_ids(stem_ids)
                    cur_prefix = self._current_word_prefix(trimmed)
                    if self._is_terminal_match(cur_prefix, sample_ref):
                        allowed.add(int(seq[consumed_len]))

            return sorted(allowed)

        def _allowed_root_start_tokens(self, sample_ref: Optional[_KeywordSampleRef]) -> Set[int]:
            allowed: Set[int] = set()
            gnodes = self.global_trie["nodes"]
            root = int(self.global_trie["root_id"])
            for tok in gnodes[root]["children"].keys():
                allowed.add(int(tok))

            local_info = self._get_local_sample_info(sample_ref)
            if local_info is not None:
                local_trie = local_info["local_trie"]
                lnodes = local_trie["nodes"]
                lroot = int(local_trie["root_id"])
                for tok in lnodes[lroot]["children"].keys():
                    allowed.add(int(tok))
            return allowed

        def allowed_next_tokens(
            self,
            answer_prefix_ids: List[int],
            sample_ref: Optional[_KeywordSampleRef] = None,
        ) -> List[int]:
            cache_key = (self._sample_key(sample_ref), tuple(int(x) for x in (answer_prefix_ids or [])))
            if cache_key in self._allowed_cache:
                return self._allowed_cache[cache_key]

            partial_terminal_allowed = self._allowed_after_partial_terminal_sequence(
                answer_prefix_ids=answer_prefix_ids,
                sample_ref=sample_ref,
            )
            if partial_terminal_allowed:
                self._allowed_cache[cache_key] = partial_terminal_allowed
                return partial_terminal_allowed

            trimmed = self._strip_prefix_ids(answer_prefix_ids)
            allowed: Set[int] = set()

            if len(trimmed) == 0:
                allowed = self._allowed_root_start_tokens(sample_ref)
                out = sorted(allowed) if allowed else list(self.final_end_token_ids)
                self._allowed_cache[cache_key] = out
                return out

            last_tok = int(trimmed[-1])
            if last_tok in self.separator_token_ids:
                allowed = self._allowed_root_start_tokens(sample_ref)
                out = sorted(allowed) if allowed else list(self.final_end_token_ids)
                self._allowed_cache[cache_key] = out
                return out

            current_prefix = self._current_word_prefix(trimmed)
            global_nodes, local_nodes = self._matched_nodes(current_prefix, sample_ref)

            if len(global_nodes) == 0 and len(local_nodes) == 0:
                out = list(self.final_end_token_ids)
                self._allowed_cache[cache_key] = out
                return out

            gnodes = self.global_trie["nodes"]
            for nid in global_nodes:
                node = gnodes[nid]
                if node["is_terminal"]:
                    allowed.update(int(x) for x in self.separator_token_ids)
                    allowed.update(int(x) for x in self.terminal_start_token_ids)
                    allowed.update(self._allowed_root_start_tokens(sample_ref))
                for tok in node["children"].keys():
                    allowed.add(int(tok))

            local_info = self._get_local_sample_info(sample_ref)
            if local_info is not None:
                lnodes = local_info["local_trie"]["nodes"]
                for nid in local_nodes:
                    node = lnodes[nid]
                    if node["is_terminal"]:
                        allowed.update(int(x) for x in self.separator_token_ids)
                        allowed.update(int(x) for x in self.terminal_start_token_ids)
                        allowed.update(self._allowed_root_start_tokens(sample_ref))
                    for tok in node["children"].keys():
                        allowed.add(int(tok))

            out = sorted(allowed) if allowed else list(self.final_end_token_ids)
            self._allowed_cache[cache_key] = out
            return out

        def build_prefix_allowed_tokens_fn(
            self,
            prompt_lens: List[int],
            sample_constraint_refs: List[_KeywordSampleRef],
            fallback_to_full_vocab: bool = False,
        ) -> Callable[[int, torch.Tensor], List[int]]:
            prompt_lens = list(prompt_lens)
            sample_constraint_refs = list(sample_constraint_refs)
            vocab_size = len(self.tokenizer)

            def _fn(batch_id: int, input_ids) -> List[int]:
                plen = int(prompt_lens[int(batch_id)]) if 0 <= int(batch_id) < len(prompt_lens) else 0
                full_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
                answer_prefix_ids = full_ids[plen:]

                sample_ref = None
                if 0 <= int(batch_id) < len(sample_constraint_refs):
                    sample_ref = sample_constraint_refs[int(batch_id)]

                allowed = self.allowed_next_tokens(
                    answer_prefix_ids=answer_prefix_ids,
                    sample_ref=sample_ref,
                )
                if allowed:
                    return allowed
                if fallback_to_full_vocab:
                    return list(range(vocab_size))
                return list(self.final_end_token_ids)

            return _fn

        # ------------------------
        # debug helpers
        # ------------------------
        def get_global_paths_for_word(self, word: str) -> List[List[int]]:
            return self.global_word_to_paths.get(_normalize_word_key(word), [])

        def get_local_paths_for_sample_word(self, sample_ref: _KeywordSampleRef, word: str) -> List[List[int]]:
            info = self._get_local_sample_info(sample_ref)
            if info is None:
                return []
            return info["local_word_to_paths"].get(_normalize_word_key(word), [])

    def _build_keyword_constraints_for_datasetdict(
        ds_out: DatasetDict,
        tokenizer,
        end_token_ids: Optional[List[int]] = None,
        terminal_sequences: Optional[List[List[int]]] = None,
        terminal_loss_mode: str = "final_only",
        keyword_separator_list: Optional[List[str]] = None,
    ):
        keyword_separator_list = _normalize_keyword_separator_list_local(keyword_separator_list)
        """
        Low-memory shared-trie architecture:
        - The shared vocabulary contains keyword words from raw train only.
        - Constraint units are words, not phrases.
        - Each sample allows global training words plus new words from its body.
        - Complete sample_allowed_word_paths lists are not stored per sample.
        """
        global_keyword_words = _build_global_keyword_word_universe_train_only(ds_raw=ds_raw)
        global_word_to_paths = _build_global_keyword_word_paths(
            tokenizer=tokenizer,
            global_keyword_words=global_keyword_words,
        )

        split_to_titles: Dict[str, List[str]] = {}
        split_to_prompt_raws: Dict[str, List[str]] = {}
        split_bundles = {}

        for split_name, ds_split in ds_out.items():
            split_to_titles[split_name] = list(ds_split["Title"] if "Title" in ds_split.column_names else [""] * len(ds_split))
            split_to_prompt_raws[split_name] = list(ds_split["PromptRaw"] if "PromptRaw" in ds_split.column_names else [""] * len(ds_split))

        fsm = KeywordOutputFSM(
            tokenizer=tokenizer,
            global_word_to_paths=global_word_to_paths,
            split_to_titles=split_to_titles,
            split_to_prompt_raws=split_to_prompt_raws,
            prefix_text=keyword_prefix_text,
            keyword_separator_list=keyword_separator_list,
            end_token_ids=end_token_ids,
            terminal_sequences=terminal_sequences,
        )

        all_initial_allowed = set(int(x) for x in fsm.initial_allowed_token_ids)

        for split_name, ds_split in ds_out.items():
            prompt_lens = [len(x) for x in ds_split["PromptIds"]]
            sample_constraint_refs = [
                _KeywordSampleRef(split_name=split_name, sample_idx=i)
                for i in range(len(ds_split))
            ]

            split_bundles[split_name] = {
                "prompt_lens": prompt_lens,
                "sample_constraint_refs": sample_constraint_refs,
                "separator_token_ids": list(fsm.separator_token_ids),
                "comma_token_ids": list(fsm.comma_token_ids),
                "terminal_start_token_ids": list(fsm.terminal_start_token_ids),
                "final_end_token_ids": list(fsm.final_end_token_ids),
                "prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn(
                    prompt_lens=prompt_lens,
                    sample_constraint_refs=sample_constraint_refs,
                    fallback_to_full_vocab=False,
                ),
            }

        return {
            "task_mode": "keywords_word_level_shared_trie",
            "keyword_prefix_text": keyword_prefix_text,
            "keyword_separator_list": list(keyword_separator_list),
            "global_words": sorted(global_word_to_paths.keys()),
            "global_word_to_paths": global_word_to_paths,
            "global_word_trie": fsm.global_trie,
            "separator_token_ids": list(fsm.separator_token_ids),
            "comma_token_ids": list(fsm.comma_token_ids),
            "initial_allowed_token_ids": sorted(all_initial_allowed),
            "end_token_ids": fsm.end_token_ids,
            "terminal_sequences": fsm.terminal_sequences,
            "terminal_start_token_ids": fsm.terminal_start_token_ids,
            "final_end_token_ids": fsm.final_end_token_ids,
            "terminal_loss_mode": terminal_loss_mode,
            "fsm": fsm,
            "allowed_next_tokens": fsm.allowed_next_tokens,
            "build_prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn,
            "splits": split_bundles,
        }

    class BothOutputFSM:
        """
        Loose BOTH FSM.

        Design:
        1) Use CategoryOutputFSM where possible during category generation.
        2) Allow the model to enter keywords at any time.
        3) Enter a permissive transition zone after a complete category.
        4) Switch to KeywordOutputFSM once keywords: appears in the prefix.
        5) Support spacing and newline variants of the keywords marker.
        6) Disallow terminal output before keywords: to prevent early completion.
        """

        def __init__(
            self,
            tokenizer,
            category_fsm,
            keyword_fsm,
            max_category_answer_ids_before_force_bridge: int = 20,
        ):
            self.tokenizer = tokenizer
            self.category_fsm = category_fsm
            self.keyword_fsm = keyword_fsm

            self.end_token_ids = list(keyword_fsm.end_token_ids)
            self.terminal_sequences = keyword_fsm.terminal_sequences
            self.terminal_start_token_ids = list(keyword_fsm.terminal_start_token_ids)
            self.final_end_token_ids = list(keyword_fsm.final_end_token_ids)
            self.initial_allowed_token_ids = list(category_fsm.initial_allowed_token_ids)

            # Retained for compatibility with older constraint output.
            self.max_category_answer_ids_before_force_bridge = max(
                1,
                int(max_category_answer_ids_before_force_bridge),
            )

            self.category_prefix_text = str(
                getattr(category_fsm, "prefix_text", "categories: ") or "categories: "
            )
            self.keyword_prefix_text = str(
                getattr(keyword_fsm, "prefix_text", "") or ""
            )

            self.category_prefix_ids = _tokenize_no_special(
                self.tokenizer,
                self.category_prefix_text,
            )

            # Canonical prefix recognized by keyword_fsm.
            self.canonical_bridge_ids = self._resolve_canonical_bridge_ids()
            if not self.canonical_bridge_ids:
                raise ValueError(
                    "BothOutputFSM canonical_bridge_ids is empty. "
                    "Check keyword_fsm.prefix_text / keyword_fsm.prefix_token_variants."
                )

            # Permissive marker paths detect or guide keywords: without fixed formatting.
            self.keyword_marker_variants = self._build_keyword_marker_variants()
            if not self.keyword_marker_variants:
                raise ValueError("BothOutputFSM keyword_marker_variants is empty.")

            self.keyword_marker_first_token_ids = sorted(
                {int(p[0]) for p in self.keyword_marker_variants if p}
            )

            # Compatibility fields.
            self.bridge_ids = list(self.canonical_bridge_ids)
            self.bridge_first_token_id = int(self.bridge_ids[0])

            # Broad vocabulary before keywords, excluding early BOTH termination.
            vocab_size = len(self.tokenizer)
            blocked_before_keywords = set(int(x) for x in self.terminal_start_token_ids)
            blocked_before_keywords.update(int(x) for x in self.final_end_token_ids)

            self._full_vocab_before_keywords = [
                i for i in range(vocab_size) if int(i) not in blocked_before_keywords
            ]

        # =====================================================
        # Basic helpers
        # =====================================================

        def _decode_ids(self, ids: List[int]) -> str:
            ids = [int(x) for x in (ids or [])]
            try:
                return self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                return self.tokenizer.decode(
                    ids,
                    skip_special_tokens=False,
                )

        def _tokenize_no_special_local(self, text: str) -> List[int]:
            return self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

        def _encode_no_special_local(self, text: str) -> List[int]:
            try:
                return self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                return self._tokenize_no_special_local(text)

        def _dedup_paths(self, paths: List[List[int]]) -> List[List[int]]:
            out = []
            seen = set()

            for p in paths:
                key = tuple(int(x) for x in (p or []) if x is not None)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(list(key))

            # Longer paths first for debugging; runtime behavior does not depend on order.
            out.sort(key=len, reverse=True)
            return out

        def _resolve_canonical_bridge_ids(self) -> List[int]:
            """
            Return the canonical prefix used by keyword_fsm, preferring the
            longest entry in keyword_fsm.prefix_token_variants.
            """
            variants = getattr(self.keyword_fsm, "prefix_token_variants", None)

            if variants:
                clean_variants = []
                for v in variants:
                    v = [int(x) for x in (v or [])]
                    if v:
                        clean_variants.append(v)

                if clean_variants:
                    clean_variants.sort(key=len, reverse=True)
                    return list(clean_variants[0])

            ids = _tokenize_no_special(self.tokenizer, self.keyword_prefix_text)
            return [int(x) for x in ids]

        # =====================================================
        # Keyword marker variants
        # =====================================================

        def _build_keyword_marker_variants(self) -> List[List[int]]:
            """
            Build permissive token paths for the keywords: marker.

            Allow direct, space-prefixed, newline-prefixed, and space-newline
            forms, native keyword_fsm variants, and decode-equivalent tokenizer
            paths.
            """
            raw_prefix = str(self.keyword_prefix_text or "")
            core = raw_prefix.strip()

            if not core:
                core = "keywords:"

            # If core contains additional text, retain the keywords: portion.
            m = re.search(r"keywords\s*:", core, flags=re.IGNORECASE)
            if m:
                core = core[m.start():m.end()]
            else:
                core = "keywords:"

            text_variants = [
                core,
                core + " ",
                " " + core,
                " " + core + " ",
                "\n" + core,
                "\n" + core + " ",
                " \n" + core,
                " \n" + core + " ",
                "\n " + core,
                "\n " + core + " ",
            ]

            if raw_prefix:
                text_variants.append(raw_prefix)
                text_variants.append(raw_prefix.rstrip())

            paths: List[List[int]] = []

            # Add native keyword_fsm prefix variants.
            native_variants = getattr(self.keyword_fsm, "prefix_token_variants", None)
            if native_variants:
                for v in native_variants:
                    v = [int(x) for x in (v or [])]
                    if v:
                        paths.append(v)

            # Tokenize textual variants normally.
            for txt in text_variants:
                txt = str(txt or "")
                if not txt:
                    continue

                try:
                    ids = self._tokenize_no_special_local(txt)
                    if ids:
                        paths.append([int(x) for x in ids])
                except Exception:
                    pass

                try:
                    ids = self._encode_no_special_local(txt)
                    if ids:
                        paths.append([int(x) for x in ids])
                except Exception:
                    pass

            # Compose paths manually to match stepwise generation.
            core_ids_candidates = []
            for txt in [core, core + " "]:
                try:
                    ids = self._tokenize_no_special_local(txt)
                    if ids:
                        core_ids_candidates.append([int(x) for x in ids])
                except Exception:
                    pass
                try:
                    ids = self._encode_no_special_local(txt)
                    if ids:
                        core_ids_candidates.append([int(x) for x in ids])
                except Exception:
                    pass

            lead_texts = [" ", "\n", " \n", "\n "]
            lead_ids_candidates = [[]]

            for lead in lead_texts:
                try:
                    ids = self._tokenize_no_special_local(lead)
                    if ids:
                        lead_ids_candidates.append([int(x) for x in ids])
                except Exception:
                    pass
                try:
                    ids = self._encode_no_special_local(lead)
                    if ids:
                        lead_ids_candidates.append([int(x) for x in ids])
                except Exception:
                    pass

            for lead_ids in lead_ids_candidates:
                for core_ids in core_ids_candidates:
                    paths.append(list(lead_ids) + list(core_ids))

            paths = self._dedup_paths(paths)

            # Keep only paths whose decoded text contains keywords:.
            filtered = []
            for p in paths:
                decoded = self._decode_ids(p)
                compact = re.sub(r"\s+", "", decoded).lower()
                if "keywords:" in compact:
                    filtered.append(p)

            return self._dedup_paths(filtered)

        def _partial_keyword_marker_next_tokens(self, ids: List[int]) -> List[int]:
            """
            If the current suffix is a marker prefix, additionally allow tokens
            that complete it. This expands choices without forcing the marker.
            """
            ids = [int(x) for x in (ids or [])]
            allowed = set()

            if not ids:
                return []

            for path in self.keyword_marker_variants:
                path = [int(x) for x in path]
                n = len(path)
                if n <= 1:
                    continue

                max_k = min(len(ids), n - 1)

                for k in range(max_k, 0, -1):
                    if ids[-k:] == path[:k]:
                        allowed.add(int(path[k]))

            return sorted(allowed)

        def _split_after_keywords_marker_loose(
            self,
            answer_prefix_ids: List[int],
        ) -> Optional[List[int]]:
            """
            Detect the keyword phase whenever decoded answer_prefix_ids contains
            keywords:. Return the token suffix after that marker, or None when
            the marker is absent.
            """
            ids = [int(x) for x in (answer_prefix_ids or [])]

            if not ids:
                return None

            text = self._decode_ids(ids)
            if re.search(r"keywords\s*:", text, flags=re.IGNORECASE) is None:
                return None

            # Find the shortest prefix containing keywords: and treat following
            # tokens as keyword content.
            for cut in range(1, len(ids) + 1):
                prefix_text = self._decode_ids(ids[:cut])
                if re.search(r"keywords\s*:", prefix_text, flags=re.IGNORECASE):
                    return ids[cut:]

            return []

        # =====================================================
        # Category helpers
        # =====================================================

        def _category_state(self, answer_prefix_ids: List[int]) -> Optional[Dict[str, Any]]:
            ids = [int(x) for x in (answer_prefix_ids or [])]

            trimmed, next_prefix_tokens = self.category_fsm._strip_prefix_ids(ids)

            if trimmed is None:
                return None

            return self.category_fsm._parse_trimmed_ids(trimmed)

        def _category_can_transition_to_keywords(self, answer_prefix_ids: List[int]) -> bool:
            state = self._category_state(answer_prefix_ids)

            if state is None:
                return False

            return bool(self.category_fsm._can_end_from_state(state))

        def _find_completed_category_prefix_end(self, ids: List[int]) -> Optional[int]:
            """
            Find a prefix end where a category has completed legally. This
            enables entry into the permissive transition zone. For example:
            ids = categories: business blah blah
            Once categories: business is valid and complete, following text is
            temporarily treated as a loose transition.
            """
            ids = [int(x) for x in (ids or [])]

            for cut in range(len(ids), -1, -1):
                if self._category_can_transition_to_keywords(ids[:cut]):
                    return int(cut)

            return None

        # =====================================================
        # Main API
        # =====================================================

        def allowed_next_tokens(
            self,
            answer_prefix_ids: List[int],
            sample_ref: Optional[_KeywordSampleRef] = None,
        ) -> List[int]:
            answer_prefix_ids = [int(x) for x in (answer_prefix_ids or [])]

            # =====================================================
            # 1) Switch to KeywordOutputFSM once keywords: appears anywhere.
            # =====================================================
            suffix_after_keywords = self._split_after_keywords_marker_loose(answer_prefix_ids)

            if suffix_after_keywords is not None:
                keyword_prefix_ids = list(self.canonical_bridge_ids) + list(suffix_after_keywords)

                allowed = self.keyword_fsm.allowed_next_tokens(
                    answer_prefix_ids=keyword_prefix_ids,
                    sample_ref=sample_ref,
                )

                return list(allowed) if allowed else list(self.final_end_token_ids)

            # =====================================================
            # 2) If a marker is partial, additionally allow its completion.
            # =====================================================
            marker_next_tokens = set(self._partial_keyword_marker_next_tokens(answer_prefix_ids))

            # =====================================================
            # 3) Try category_fsm, excluding terminal before keywords appears.
            # =====================================================
            cat_allowed = set()

            try:
                cat_allowed = set(
                    int(x) for x in self.category_fsm.allowed_next_tokens(answer_prefix_ids)
                )
            except Exception:
                cat_allowed = set()

            terminal_block = set(int(x) for x in self.terminal_start_token_ids)
            terminal_block.update(int(x) for x in self.final_end_token_ids)

            cat_allowed -= terminal_block

            # =====================================================
            # 4) Always allow direct, space, or newline transitions to keywords.
            # =====================================================
            cat_allowed.update(int(x) for x in self.keyword_marker_first_token_ids)
            cat_allowed.update(marker_next_tokens)

            # =====================================================
            # 5) After one valid category, permit nearly the full vocabulary so
            # the model can choose when to emit keywords, while excluding terminal.
            # =====================================================
            completed_category_end = self._find_completed_category_prefix_end(answer_prefix_ids)

            if completed_category_end is not None:
                # The full vocabulary already includes spaces, newlines, commas, and keywords.
                return list(self._full_vocab_before_keywords)

            if cat_allowed:
                return sorted(cat_allowed)

            # =====================================================
            # 6) Final fallback: an invalid category prefix may still transition
            # through a keywords marker instead of terminating immediately.
            # =====================================================
            if self.keyword_marker_first_token_ids:
                return list(self.keyword_marker_first_token_ids)

            return list(self.final_end_token_ids)

        def build_prefix_allowed_tokens_fn(
            self,
            prompt_lens: List[int],
            sample_constraint_refs: List[_KeywordSampleRef],
            fallback_to_full_vocab: bool = False,
        ) -> Callable[[int, torch.Tensor], List[int]]:
            prompt_lens = list(prompt_lens)
            sample_constraint_refs = list(sample_constraint_refs)
            vocab_size = len(self.tokenizer)

            def _fn(batch_id: int, input_ids) -> List[int]:
                bid = int(batch_id)
                plen = int(prompt_lens[bid]) if 0 <= bid < len(prompt_lens) else 0

                full_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
                answer_prefix_ids = full_ids[plen:]

                sample_ref = (
                    sample_constraint_refs[bid]
                    if 0 <= bid < len(sample_constraint_refs)
                    else None
                )

                allowed = self.allowed_next_tokens(
                    answer_prefix_ids=answer_prefix_ids,
                    sample_ref=sample_ref,
                )

                if allowed:
                    return allowed

                if fallback_to_full_vocab:
                    return list(range(vocab_size))

                return list(self.final_end_token_ids)

            return _fn


    def _build_both_constraints_for_datasetdict(
        ds_out: DatasetDict,
        tokenizer,
        label_list: List[str],
        label_to_idx: Dict[str, int],
        allow_multi_label: bool = True,
        end_token_ids: Optional[List[int]] = None,
        terminal_sequences: Optional[List[List[int]]] = None,
        terminal_loss_mode: str = "final_only",
        keyword_separator_list: Optional[List[str]] = None,
    ):
        del label_to_idx
        keyword_separator_list = _normalize_keyword_separator_list_local(keyword_separator_list)

        category_fsm = CategoryOutputFSM(
            tokenizer=tokenizer,
            label_list=label_list,
            prefix_text="categories: ",
            allow_multi_label=allow_multi_label,
            end_token_ids=end_token_ids,
            terminal_sequences=terminal_sequences,
        )

        global_keyword_words = _build_global_keyword_word_universe_train_only(ds_raw=ds_raw)
        global_word_to_paths = _build_global_keyword_word_paths(
            tokenizer=tokenizer,
            global_keyword_words=global_keyword_words,
        )

        split_to_titles: Dict[str, List[str]] = {}
        split_to_prompt_raws: Dict[str, List[str]] = {}
        split_bundles = {}

        for split_name, ds_split in ds_out.items():
            split_to_titles[split_name] = list(ds_split["Title"] if "Title" in ds_split.column_names else [""] * len(ds_split))
            split_to_prompt_raws[split_name] = list(ds_split["PromptRaw"] if "PromptRaw" in ds_split.column_names else [""] * len(ds_split))

        keyword_fsm = KeywordOutputFSM(
            tokenizer=tokenizer,
            global_word_to_paths=global_word_to_paths,
            split_to_titles=split_to_titles,
            split_to_prompt_raws=split_to_prompt_raws,
            prefix_text="\n" + keyword_prefix_text,
            keyword_separator_list=keyword_separator_list,
            end_token_ids=end_token_ids,
            terminal_sequences=terminal_sequences,
        )

        fsm = BothOutputFSM(
            tokenizer=tokenizer,
            category_fsm=category_fsm,
            keyword_fsm=keyword_fsm,
            max_category_answer_ids_before_force_bridge=20,
        )

        for split_name, ds_split in ds_out.items():
            prompt_lens = [len(x) for x in ds_split["PromptIds"]]
            sample_constraint_refs = [
                _KeywordSampleRef(split_name=split_name, sample_idx=i)
                for i in range(len(ds_split))
            ]

            split_bundles[split_name] = {
                "prompt_lens": prompt_lens,
                "sample_constraint_refs": sample_constraint_refs,
                "separator_token_ids": list(keyword_fsm.separator_token_ids),
                "terminal_start_token_ids": list(fsm.terminal_start_token_ids),
                "final_end_token_ids": list(fsm.final_end_token_ids),
                "prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn(
                    prompt_lens=prompt_lens,
                    sample_constraint_refs=sample_constraint_refs,
                    fallback_to_full_vocab=False,
                ),
            }

        return {
            "task_mode": "both_joint_category_keyword",
            "label_list": label_list,
            "keyword_prefix_text": keyword_prefix_text,
            "keyword_separator_list": list(keyword_separator_list),
            "initial_allowed_token_ids": list(fsm.initial_allowed_token_ids),
            "end_token_ids": list(fsm.end_token_ids),
            "terminal_sequences": fsm.terminal_sequences,
            "terminal_start_token_ids": list(fsm.terminal_start_token_ids),
            "final_end_token_ids": list(fsm.final_end_token_ids),
            "terminal_loss_mode": terminal_loss_mode,
            "bridge_ids": list(fsm.bridge_ids),
            "max_category_answer_ids_before_force_bridge": int(fsm.max_category_answer_ids_before_force_bridge),
            "fsm": fsm,
            "category_fsm": category_fsm,
            "keyword_fsm": keyword_fsm,
            "allowed_next_tokens": fsm.allowed_next_tokens,
            "build_prefix_allowed_tokens_fn": fsm.build_prefix_allowed_tokens_fn,
            "splits": split_bundles,
        }

    def debug_print_one_sample_keyword_path(
        ds_out: DatasetDict,
        constraints: Dict[str, Any],
        tokenizer,
        split_name: str = "train",
        sample_idx: int = 0,
        keyword: str = "machine",
    ):
        if split_name not in ds_out:
            raise KeyError(f"split '{split_name}' not found in ds_out.")
        if split_name not in constraints["splits"]:
            raise KeyError(f"split '{split_name}' not found in constraints['splits'].")

        ds_split = ds_out[split_name]
        bundle = constraints["splits"][split_name]
        if not (0 <= int(sample_idx) < len(ds_split)):
            raise IndexError(f"sample_idx out of range: {sample_idx}")

        row = ds_split[int(sample_idx)]
        sample_ref = bundle["sample_constraint_refs"][int(sample_idx)]
        fsm = constraints["fsm"]
        keyword_norm = _normalize_word_key(keyword)

        print("=" * 120)
        print(f"[DEBUG SAMPLE] split={split_name} idx={sample_idx}")
        print("=" * 120)
        print("[Title]")
        print(row.get("Title", ""))

        print("\n[PromptRaw]")
        print(row.get("PromptRaw", ""))

        print("\n[Keyword Query]")
        print(f"keyword={keyword!r}  normalized={keyword_norm!r}")

        global_paths = fsm.get_global_paths_for_word(keyword)
        if global_paths:
            print("\n[Found in GLOBAL train keyword table]")
            for i, p in enumerate(global_paths):
                print(f"#{i}: ids={p}  decoded={tokenizer.decode(p, skip_special_tokens=False)!r}")
        else:
            print("\n[Not found in GLOBAL train keyword table]")

        local_paths = fsm.get_local_paths_for_sample_word(sample_ref, keyword)
        if local_paths:
            print("\n[Found in SAMPLE local-new words]")
            for i, p in enumerate(local_paths):
                print(f"#{i}: ids={p}  decoded={tokenizer.decode(p, skip_special_tokens=False)!r}")
        else:
            print("\n[Not found in SAMPLE local-new words]")

        print("\n[Allowed tokens at answer start for this sample]")
        allowed0 = fsm.allowed_next_tokens(answer_prefix_ids=[], sample_ref=sample_ref)
        print(allowed0[:100])

        print("\n[Keyword path legal?]")
        print(bool(global_paths or local_paths))

    if ds_out is None:
        train_terminal_sequences = _resolve_training_terminal_sequences(
            tokenizer=tokenizer,
            end_token_ids=end_token_ids,
            terminal_sequences=terminal_sequences,
        )

        raw_train = _resolve_raw_split(ds_raw, "train")
        raw_valid = _resolve_raw_split(ds_raw, "validation")
        raw_test = _resolve_raw_split(ds_raw, "test")

        if raw_train is None:
            raise KeyError(f"raw dataset has no train split. available={sorted(list(ds_raw.keys()))}")

        picked_train_dir = str(picked_train_dir)
        picked_ds = None

        if _looks_like_hf_dataset_dir(picked_train_dir):
            picked_ds = load_from_disk(picked_train_dir)
            if isinstance(picked_ds, DatasetDict):
                picked_ds = picked_ds["train"] if "train" in picked_ds else picked_ds[sorted(picked_ds.keys())[0]]
        else:
            if not auto_build_picked_if_missing:
                raise FileNotFoundError(f"picked_train_dir not found or not a HF dataset dir: {picked_train_dir}")
            if dedup_cfg is None:
                raise ValueError(
                    "picked_train_dir missing -> auto_build_picked_if_missing=True requires dedup_cfg."
                )

            dd_kwargs = dict(dedup_kwargs or {})
            dd_kwargs.setdefault("split", "train")
            dd_kwargs.setdefault("seed", seed)
            dd_kwargs.setdefault("ds_raw", ds_raw)
            dd_kwargs.setdefault("out_ds_dir", picked_train_dir)
            dd_kwargs.setdefault("save_to_disk", True)

            print(f"[PICKED] not found -> building via build_kptimes_dedup_dataset_v2(.) into: {picked_train_dir}")
            _picked_ds, _summary, _manifest, _final_idx, _df_small = build_kptimes_dedup_dataset_v2(
                cfg=dedup_cfg,
                **dd_kwargs,
            )
            del _picked_ds, _summary, _manifest, _final_idx, _df_small

            picked_ds = load_from_disk(picked_train_dir)
            if isinstance(picked_ds, DatasetDict):
                picked_ds = picked_ds["train"] if "train" in picked_ds else picked_ds[sorted(picked_ds.keys())[0]]

        if picked_ds is None:
            raise RuntimeError("Failed to load/build picked train dataset.")

        # compute target_n first for efficient early-stop candidate building
        train_target_n = _compute_target(len(picked_ds), train_samples, "train", none_means_all=True)
        valid_target_n = _compute_target(len(raw_valid), valid_samples, "validation", none_means_all=True) if raw_valid is not None else None
        test_target_n = _compute_target(len(raw_test), test_samples, "test", none_means_all=True) if (raw_test is not None and test_samples is not None) else None

        if task_mode == "category":
            label_list, label_to_idx = _build_label_space_safe(
                ds_raw=ds_raw,
                constraint_source_split=constraint_source_split,
            )

            universe_path = Path(picked_train_dir) / universe_csv
            if not universe_path.exists():
                raise FileNotFoundError(f"universe csv not found: {universe_path}")

            universe_labels = []
            with universe_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lab = _norm_label(row.get("label", ""))
                    if not lab:
                        continue
                    universe_labels.append(lab)

            seen_uni = set()
            uni_keep = []
            for z in universe_labels:
                if z not in seen_uni:
                    seen_uni.add(z)
                    uni_keep.append(z)
            universe_labels = uni_keep
            universe_set = set(universe_labels)

            train_prompt_candidates = _build_prompt_candidates_from_source_task(
                picked_ds,
                task_mode_local="category",
                split_name="train",
                label_to_idx=label_to_idx,
                target_prompt_rows=train_target_n,
                early_stop_when_enough=(train_target_n is not None),
            )

            valid_prompt_candidates = None
            if raw_valid is not None:
                valid_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_valid,
                    task_mode_local="category",
                    split_name="validation",
                    label_to_idx=label_to_idx,
                    target_prompt_rows=None,
                    early_stop_when_enough=False,
                )

            test_prompt_candidates = None
            if test_samples is not None:
                if raw_test is None:
                    raise KeyError(
                        f"test_samples is set but raw dataset has no test split. available={sorted(list(ds_raw.keys()))}"
                    )
                test_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_test,
                    task_mode_local="category",
                    split_name="test",
                    label_to_idx=label_to_idx,
                    target_prompt_rows=None,
                    early_stop_when_enough=False,
                )

            need_valid = _labels_union_from_prompt_ds_local(valid_prompt_candidates, universe_set) if valid_prompt_candidates is not None else set()

            need_test = set()
            if raw_test is not None and test_samples is not None:
                for ex in raw_test:
                    for lab in _extract_labels_from_raw_ex(ex):
                        if lab in universe_set:
                            need_test.add(lab)
                if not need_test and test_prompt_candidates is not None:
                    need_test = _labels_union_from_prompt_ds_local(test_prompt_candidates, universe_set)

            out_prompt_splits = {}
            coverage_stats = {}

            train_selected_prompt, train_seen_labels = _random_select_prompt_rows(
                train_prompt_candidates,
                target_n=train_target_n,
                split_name="train",
                universe_set=universe_set,
            )
            out_prompt_splits["train"] = train_selected_prompt
            coverage_stats["train"] = {
                "seen": train_seen_labels,
                "need": None,
                "missing_need": None,
                "available": _labels_union_from_prompt_ds_local(train_prompt_candidates, universe_set),
                "n_candidates": len(train_prompt_candidates),
                "n_selected": len(train_selected_prompt),
                "target_n": train_target_n,
            }

            if valid_prompt_candidates is not None:
                valid_selected_prompt, valid_available_labels, valid_missing_need = _coverage_select_prompt_rows(
                    valid_prompt_candidates,
                    target_n=valid_target_n,
                    split_name="validation",
                    need_labels=need_valid,
                    universe_set=universe_set,
                )
                out_prompt_splits["validation"] = valid_selected_prompt
                coverage_stats["validation"] = {
                    "seen": _labels_union_from_prompt_ds_local(valid_selected_prompt, universe_set),
                    "need": need_valid,
                    "missing_need": valid_missing_need,
                    "available": valid_available_labels,
                    "n_candidates": len(valid_prompt_candidates),
                    "n_selected": len(valid_selected_prompt),
                    "target_n": valid_target_n,
                }

            if test_prompt_candidates is not None:
                test_selected_prompt, test_available_labels, test_missing_need = _coverage_select_prompt_rows(
                    test_prompt_candidates,
                    target_n=test_target_n,
                    split_name="test",
                    need_labels=need_test,
                    universe_set=universe_set,
                )
                out_prompt_splits["test"] = test_selected_prompt
                coverage_stats["test"] = {
                    "seen": _labels_union_from_prompt_ds_local(test_selected_prompt, universe_set),
                    "need": need_test,
                    "missing_need": test_missing_need,
                    "available": test_available_labels,
                    "n_candidates": len(test_prompt_candidates),
                    "n_selected": len(test_selected_prompt),
                    "target_n": test_target_n,
                }

            out_splits = {}
            for split_name, ds_prompt_selected in out_prompt_splits.items():
                ds_trainer = ds_prompt_selected.map(
                    _to_trainer_features_batch,
                    batched=True,
                    batch_size=max(64, min(batch_size, 512)),
                    num_proc=chat_num_proc,
                    load_from_cache_file=False,
                    desc=f"[{split_name}] building trainer-ready answer-only features",
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "system_text": system_text,
                        "max_length": max_length,
                        "train_terminal_sequences": train_terminal_sequences,
                        "terminal_loss_mode": terminal_loss_mode,
                        "prefix_weight": prefix_weight,
                        "separator_weight": separator_weight,
                        "label_base_weight": label_base_weight,
                        "later_label_alpha": later_label_alpha,
                        "later_label_power": later_label_power,
                        "only_label_weight": only_label_weight,
                        "terminal_active_weight": terminal_active_weight,
                    },
                    remove_columns=ds_prompt_selected.column_names,
                )
                out_splits[split_name] = ds_trainer

            ds_out = DatasetDict(out_splits)

            if print_category_coverage and universe_set:
                def _report_coverage(split_name: str, stat: dict):
                    seen_labels = set(stat["seen"] or set())
                    need_labels = set(stat["need"] or set()) if stat.get("need", None) is not None else None
                    missing_need = set(stat["missing_need"] or set()) if stat.get("missing_need", None) is not None else None
                    available_labels = set(stat["available"] or set())
                    covered = len(seen_labels & universe_set)
                    ratio = (covered / len(universe_set)) if universe_set else 0.0
                    miss_universe = universe_set - seen_labels

                    print("\n" + "=" * 90)
                    print(
                        f"[CATEGORY COVERAGE] split={split_name}  "
                        f"candidate_rows={stat['n_candidates']}  selected_rows={stat['n_selected']}  "
                        f"target_n={'ALL' if stat['target_n'] is None else stat['target_n']}"
                    )
                    print(
                        f"seen={len(seen_labels)}  covered={covered}/{len(universe_set)} ({ratio:.2%})  "
                        f"available_under_constraints={len(available_labels)}"
                    )

                    if need_labels is not None:
                        covers_need = len(need_labels - missing_need)
                        print(
                            f"need_labels={len(need_labels)}  selected_covers_need={covers_need}  "
                            f"missing_need={len(missing_need)}"
                        )
                        if missing_need and missing_print_limit > 0:
                            show = sorted(list(missing_need))[: int(missing_print_limit)]
                            print(f"missing_need examples (show {len(show)}/{len(missing_need)}): {show}")

                    if miss_universe and missing_print_limit > 0:
                        show = sorted(list(miss_universe))[: int(missing_print_limit)]
                        print(f"missing_universe examples (show {len(show)}/{len(miss_universe)}): {show}")

                    print("=" * 90)

                for split_name in ["train", "validation", "test"]:
                    if split_name in coverage_stats:
                        _report_coverage(split_name, coverage_stats[split_name])

        elif task_mode == "both":
            label_list, label_to_idx = _build_label_space_safe(
                ds_raw=ds_raw,
                constraint_source_split=constraint_source_split,
            )

            universe_path = Path(picked_train_dir) / universe_csv
            if not universe_path.exists():
                raise FileNotFoundError(f"universe csv not found: {universe_path}")

            universe_labels = []
            with universe_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    lab = _norm_label(row.get("label", ""))
                    if not lab:
                        continue
                    universe_labels.append(lab)

            seen_uni = set()
            uni_keep = []
            for z in universe_labels:
                if z not in seen_uni:
                    seen_uni.add(z)
                    uni_keep.append(z)
            universe_labels = uni_keep
            universe_set = set(universe_labels)

            train_prompt_candidates = _build_prompt_candidates_from_source_task(
                picked_ds,
                task_mode_local="both",
                split_name="train",
                target_prompt_rows=train_target_n,
                early_stop_when_enough=(train_target_n is not None),
            )

            valid_prompt_candidates = None
            if raw_valid is not None:
                valid_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_valid,
                    task_mode_local="both",
                    split_name="validation",
                    target_prompt_rows=None,
                    early_stop_when_enough=False,
                )

            test_prompt_candidates = None
            if test_samples is not None:
                if raw_test is None:
                    raise KeyError(
                        f"test_samples is set but raw dataset has no test split. available={sorted(list(ds_raw.keys()))}"
                    )
                test_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_test,
                    task_mode_local="both",
                    split_name="test",
                    target_prompt_rows=None,
                    early_stop_when_enough=False,
                )

            need_valid = _labels_union_from_prompt_ds_local(valid_prompt_candidates, universe_set) if valid_prompt_candidates is not None else set()

            need_test = set()
            if raw_test is not None and test_samples is not None:
                for ex in raw_test:
                    for lab in _extract_labels_from_raw_ex(ex):
                        if lab in universe_set:
                            need_test.add(lab)
                if not need_test and test_prompt_candidates is not None:
                    need_test = _labels_union_from_prompt_ds_local(test_prompt_candidates, universe_set)

            out_prompt_splits = {}
            coverage_stats = {}

            train_selected_prompt, train_seen_labels = _random_select_prompt_rows(
                train_prompt_candidates,
                target_n=train_target_n,
                split_name="train",
                universe_set=universe_set,
            )
            out_prompt_splits["train"] = train_selected_prompt
            coverage_stats["train"] = {
                "seen": train_seen_labels,
                "need": None,
                "missing_need": None,
                "available": _labels_union_from_prompt_ds_local(train_prompt_candidates, universe_set),
                "n_candidates": len(train_prompt_candidates),
                "n_selected": len(train_selected_prompt),
                "target_n": train_target_n,
            }

            if valid_prompt_candidates is not None:
                valid_selected_prompt, valid_available_labels, valid_missing_need = _coverage_select_prompt_rows(
                    valid_prompt_candidates,
                    target_n=valid_target_n,
                    split_name="validation",
                    need_labels=need_valid,
                    universe_set=universe_set,
                )
                out_prompt_splits["validation"] = valid_selected_prompt
                coverage_stats["validation"] = {
                    "seen": _labels_union_from_prompt_ds_local(valid_selected_prompt, universe_set),
                    "need": need_valid,
                    "missing_need": valid_missing_need,
                    "available": valid_available_labels,
                    "n_candidates": len(valid_prompt_candidates),
                    "n_selected": len(valid_selected_prompt),
                    "target_n": valid_target_n,
                }

            if test_prompt_candidates is not None:
                test_selected_prompt, test_available_labels, test_missing_need = _coverage_select_prompt_rows(
                    test_prompt_candidates,
                    target_n=test_target_n,
                    split_name="test",
                    need_labels=need_test,
                    universe_set=universe_set,
                )
                out_prompt_splits["test"] = test_selected_prompt
                coverage_stats["test"] = {
                    "seen": _labels_union_from_prompt_ds_local(test_selected_prompt, universe_set),
                    "need": need_test,
                    "missing_need": test_missing_need,
                    "available": test_available_labels,
                    "n_candidates": len(test_prompt_candidates),
                    "n_selected": len(test_selected_prompt),
                    "target_n": test_target_n,
                }

            out_splits = {}
            for split_name, ds_prompt_selected in out_prompt_splits.items():
                ds_trainer = ds_prompt_selected.map(
                    _to_trainer_features_batch_both,
                    batched=True,
                    batch_size=max(64, min(batch_size, 512)),
                    num_proc=chat_num_proc,
                    load_from_cache_file=False,
                    desc=f"[{split_name}] building BOTH trainer-ready answer-only features",
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "system_text": system_text,
                        "max_length": max_length,
                        "train_terminal_sequences": train_terminal_sequences,
                        "terminal_loss_mode": terminal_loss_mode,
                        "prefix_weight": prefix_weight,
                        "separator_weight": separator_weight,
                        "terminal_active_weight": terminal_active_weight,
                    },
                    remove_columns=ds_prompt_selected.column_names,
                )
                out_splits[split_name] = ds_trainer

            ds_out = DatasetDict(out_splits)

            if print_category_coverage and universe_set:
                def _report_coverage_both(split_name: str, stat: dict):
                    seen_labels = set(stat["seen"] or set())
                    need_labels = set(stat["need"] or set()) if stat.get("need", None) is not None else None
                    missing_need = set(stat["missing_need"] or set()) if stat.get("missing_need", None) is not None else None
                    available_labels = set(stat["available"] or set())
                    covered = len(seen_labels & universe_set)
                    ratio = (covered / len(universe_set)) if universe_set else 0.0
                    miss_universe = universe_set - seen_labels

                    print("\n" + "=" * 90)
                    print(
                        f"[BOTH CATEGORY COVERAGE] split={split_name}  "
                        f"candidate_rows={stat['n_candidates']}  selected_rows={stat['n_selected']}  "
                        f"target_n={'ALL' if stat['target_n'] is None else stat['target_n']}"
                    )
                    print(
                        f"seen={len(seen_labels)}  covered={covered}/{len(universe_set)} ({ratio:.2%})  "
                        f"available_under_constraints={len(available_labels)}"
                    )

                    if need_labels is not None:
                        covers_need = len(need_labels - missing_need)
                        print(
                            f"need_labels={len(need_labels)}  selected_covers_need={covers_need}  "
                            f"missing_need={len(missing_need)}"
                        )
                        if missing_need and missing_print_limit > 0:
                            show = sorted(list(missing_need))[: int(missing_print_limit)]
                            print(f"missing_need examples (show {len(show)}/{len(missing_need)}): {show}")

                    if miss_universe and missing_print_limit > 0:
                        show = sorted(list(miss_universe))[: int(missing_print_limit)]
                        print(f"missing_universe examples (show {len(show)}/{len(miss_universe)}): {show}")

                    print("=" * 90)

                for split_name in ["train", "validation", "test"]:
                    if split_name in coverage_stats:
                        _report_coverage_both(split_name, coverage_stats[split_name])

        else:
            if extra_title_keyword_prob not in (None, 0, 0.0):
                print("[INFO] extra_title_keyword_prob is ignored in keywords mode.")

            keyword_token_idf_boost_map = {}
            if keyword_token_idf_temperature > 0 and keyword_token_idf_cap > 0:
                keyword_token_idf_boost_map, _ = _build_keyword_token_idf_boost_map(
                    ds_raw=ds_raw,
                    tokenizer=tokenizer,
                    keyword_universe_splits=keyword_universe_splits,
                    keyword_universe_from_train_only=keyword_universe_from_train_only,
                    temperature=keyword_token_idf_temperature,
                    cap=keyword_token_idf_cap,
                )

            train_prompt_candidates = _build_prompt_candidates_from_source_task(
                picked_ds,
                task_mode_local="keywords",
                split_name="train",
                target_prompt_rows=train_target_n,
                early_stop_when_enough=(train_target_n is not None),
            )

            valid_prompt_candidates = None
            if raw_valid is not None:
                valid_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_valid,
                    task_mode_local="keywords",
                    split_name="validation",
                    target_prompt_rows=valid_target_n,
                    early_stop_when_enough=(valid_target_n is not None),
                )

            test_prompt_candidates = None
            if test_samples is not None:
                if raw_test is None:
                    raise KeyError(
                        f"test_samples is set but raw dataset has no test split. available={sorted(list(ds_raw.keys()))}"
                    )
                test_prompt_candidates = _build_prompt_candidates_from_source_task(
                    raw_test,
                    task_mode_local="keywords",
                    split_name="test",
                    target_prompt_rows=test_target_n,
                    early_stop_when_enough=(test_target_n is not None),
                )

            out_prompt_splits = {}
            train_selected_prompt, _ = _random_select_prompt_rows(
                train_prompt_candidates,
                target_n=train_target_n,
                split_name="train",
                universe_set=None,
            )
            out_prompt_splits["train"] = train_selected_prompt

            if valid_prompt_candidates is not None:
                valid_selected_prompt, _ = _random_select_prompt_rows(
                    valid_prompt_candidates,
                    target_n=valid_target_n,
                    split_name="validation",
                    universe_set=None,
                )
                out_prompt_splits["validation"] = valid_selected_prompt

            if test_prompt_candidates is not None:
                test_selected_prompt, _ = _random_select_prompt_rows(
                    test_prompt_candidates,
                    target_n=test_target_n,
                    split_name="test",
                    universe_set=None,
                )
                out_prompt_splits["test"] = test_selected_prompt

            out_splits = {}
            for split_name, ds_prompt_selected in out_prompt_splits.items():
                ds_trainer = ds_prompt_selected.map(
                    _to_trainer_features_batch_keywords,
                    batched=True,
                    batch_size=max(64, min(batch_size, 512)),
                    num_proc=chat_num_proc,
                    load_from_cache_file=False,
                    desc=f"[{split_name}] building keyword trainer-ready answer-only features",
                    fn_kwargs={
                        "tokenizer": tokenizer,
                        "system_text": system_text,
                        "max_length": max_length,
                        "train_terminal_sequences": train_terminal_sequences,
                        "terminal_loss_mode": terminal_loss_mode,
                        "prefix_weight": prefix_weight,
                        "separator_weight": separator_weight,
                        "terminal_active_weight": terminal_active_weight,
                        "keyword_token_idf_boost_map": keyword_token_idf_boost_map,
                    },
                    remove_columns=ds_prompt_selected.column_names,
                )
                out_splits[split_name] = ds_trainer

            ds_out = DatasetDict(out_splits)

            if print_keyword_max_weight:
                global_max_weight = 0.0
                global_max_split = None
                global_max_row = None

                for split_name, ds_split in ds_out.items():
                    if "loss_weights" not in ds_split.column_names:
                        continue
                    for idx in range(len(ds_split)):
                        row = ds_split[idx]
                        ws = row["loss_weights"]
                        if not ws:
                            continue
                        row_max = max(float(x) for x in ws)
                        if row_max > global_max_weight:
                            global_max_weight = row_max
                            global_max_split = split_name
                            global_max_row = idx

                print(
                    f"[keyword-max-loss-weight] max_weight={global_max_weight:.6f}  "
                    f"split={global_max_split}  row={global_max_row}"
                )

        if save_final_to_disk:
            try:
                save_root.mkdir(parents=True, exist_ok=True)
                ds_out.save_to_disk(str(final_saved_dir))
            except Exception as e:
                print(f"[WARN] Failed to save processed dataset to disk: {e}")

    if print_checks:
        preview_kptimes_title_cls_dataset(
            ds_out=ds_out,
            tokenizer=tokenizer,
            n_preview_per_split=n_preview_per_split,
            verify_n=verify_n,
            seed=seed,
        )

    def _build_constraints_for_task(task_mode_local: str):
        if task_mode_local in {"category", "both"}:
            label_list_local, label_to_idx_local = _build_label_space_safe(
                ds_raw=ds_raw,
                constraint_source_split=constraint_source_split,
            )
            if task_mode_local == "category":
                return _build_constraints_for_datasetdict(
                    ds_out=ds_out,
                    tokenizer=tokenizer,
                    label_list=label_list_local,
                    label_to_idx=label_to_idx_local,
                    allow_multi_label=allow_multi_label,
                    end_token_ids=end_token_ids,
                    terminal_sequences=terminal_sequences,
                    terminal_loss_mode=terminal_loss_mode,
                )
            return _build_both_constraints_for_datasetdict(
                ds_out=ds_out,
                tokenizer=tokenizer,
                label_list=label_list_local,
                label_to_idx=label_to_idx_local,
                allow_multi_label=allow_multi_label,
                end_token_ids=end_token_ids,
                terminal_sequences=terminal_sequences,
                terminal_loss_mode=terminal_loss_mode,
                keyword_separator_list=keyword_separator_list,
            )
        return _build_keyword_constraints_for_datasetdict(
            ds_out=ds_out,
            tokenizer=tokenizer,
            end_token_ids=end_token_ids,
            terminal_sequences=terminal_sequences,
            terminal_loss_mode=terminal_loss_mode,
            keyword_separator_list=keyword_separator_list,
        )

    if return_constraints:
        constraint_bundle = _build_constraints_for_task(task_mode)
        return ds_out, constraint_bundle

    return ds_out
