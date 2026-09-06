"""Build A/B/C imbalance datasets from raw HIGGS.csv.

Two passes over the raw file:
  pass 1 -> read label column only, cache as labels.npy
  pass 2 -> extract only the sampled rows

Design: one shared pool, one stratified split, imbalance created by
truncating the signal class. Background rows are identical across versions,
and version C's split sets are subsets of B's, which are subsets of A's.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_FEATURES = 28
COLUMNS = ["label"] + [f"feature_{i}" for i in range(1, N_FEATURES + 1)]


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def scan_labels(csv_path: Path, cache_path: Path, chunksize: int) -> np.ndarray:
    """Pass 1: label column only, cached to disk."""
    if cache_path.exists():
        labels = np.load(cache_path)
        print(f"[pass1] cached labels loaded: {labels.shape[0]:,} rows")
        return labels

    print("[pass1] scanning label column (slow, one time)...")
    t0 = time.time()
    parts = []
    reader = pd.read_csv(csv_path, header=None, usecols=[0],
                         dtype=np.float32, chunksize=chunksize)
    for i, chunk in enumerate(reader, 1):
        parts.append(chunk.iloc[:, 0].to_numpy(dtype=np.int8))
        if i % 5 == 0:
            print(f"  chunk {i}, rows so far {sum(p.size for p in parts):,}")
    labels = np.concatenate(parts)
    np.save(cache_path, labels)
    print(f"[pass1] done in {time.time() - t0:.0f}s, {labels.shape[0]:,} rows")
    return labels


def choose_pool(labels: np.ndarray, n_bg: int, n_sig: int, seed: int):
    """Sample background/signal row indices without replacement."""
    uniq = np.unique(labels)
    assert set(uniq.tolist()) <= {0, 1}, f"unexpected labels: {uniq}"

    bg_all = np.flatnonzero(labels == 0)
    sig_all = np.flatnonzero(labels == 1)
    print(f"[pool] available bg={bg_all.size:,} sig={sig_all.size:,}")
    assert bg_all.size >= n_bg, "not enough background rows"
    assert sig_all.size >= n_sig, "not enough signal rows"

    rng = np.random.default_rng(seed)
    bg = rng.choice(bg_all, size=n_bg, replace=False)
    sig = rng.choice(sig_all, size=n_sig, replace=False)
    rng.shuffle(bg)
    rng.shuffle(sig)
    return np.sort(bg), np.sort(sig), bg, sig


def extract_rows(csv_path: Path, wanted: np.ndarray, chunksize: int) -> pd.DataFrame:
    """Pass 2: pull only the wanted global row indices."""
    print(f"[pass2] extracting {wanted.size:,} rows...")
    t0 = time.time()
    wanted_set = np.sort(wanted)
    out, offset, ptr = [], 0, 0
    reader = pd.read_csv(csv_path, header=None, names=COLUMNS,
                         dtype=np.float32, chunksize=chunksize)
    for chunk in reader:
        n = len(chunk)
        hi = ptr
        while hi < wanted_set.size and wanted_set[hi] < offset + n:
            hi += 1
        if hi > ptr:
            local = wanted_set[ptr:hi] - offset
            sel = chunk.iloc[local].copy()
            sel.index = wanted_set[ptr:hi]
            out.append(sel)
            ptr = hi
        offset += n
        if ptr >= wanted_set.size:
            break
    df = pd.concat(out)
    print(f"[pass2] done in {time.time() - t0:.0f}s, shape={df.shape}")
    assert len(df) == wanted.size, f"expected {wanted.size}, got {len(df)}"
    return df


def three_way(idx: np.ndarray, frac_train: float, frac_val: float):
    n = idx.size
    n_tr = int(round(n * frac_train))
    n_va = int(round(n * frac_val))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def sanity_check(df: pd.DataFrame, name: str, expect_bg: int, expect_sig: int):
    problems = []
    if len(df) != expect_bg + expect_sig:
        problems.append(f"row count {len(df)} != {expect_bg + expect_sig}")
    if df.shape[1] != N_FEATURES + 1:
        problems.append(f"col count {df.shape[1]} != {N_FEATURES + 1}")
    n_null = int(df.isna().sum().sum())
    if n_null:
        problems.append(f"{n_null} nulls")
    if not np.isfinite(df.to_numpy()).all():
        problems.append("non-finite values present")
    vc = df["label"].value_counts().to_dict()
    got_bg, got_sig = int(vc.get(0.0, 0)), int(vc.get(1.0, 0))
    if (got_bg, got_sig) != (expect_bg, expect_sig):
        problems.append(f"class counts {got_bg}/{got_sig} != {expect_bg}/{expect_sig}")
    if df.index.duplicated().any():
        problems.append("duplicate source row indices")

    prev = got_sig / max(len(df), 1)
    if not 0.0 < prev < 1.0:
        problems.append(f"prevalence out of (0,1): {prev}")

    status = "OK " if not problems else "FAIL"
    print(f"  [{status}] {name:16s} rows={len(df):>7,} bg={got_bg:>7,} "
          f"sig={got_sig:>7,} prevalence={prev:.4f}")
    for p in problems:
        print(f"        -> {p}")
    return problems


def build(config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    raw = Path(cfg["paths"]["raw_csv"])
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    assert raw.exists(), f"raw csv not found: {raw}"

    d = cfg["data"]
    seed, chunksize = d["seed"], d["chunksize"]
    n_bg, n_sig = d["n_background"], d["n_signal"]
    f_tr, f_va = d["split"]["train"], d["split"]["val"]

    labels = scan_labels(raw, data_dir / "labels.npy", chunksize)
    bg_sorted, sig_sorted, bg_shuf, sig_shuf = choose_pool(labels, n_bg, n_sig, seed)

    wanted = np.sort(np.concatenate([bg_sorted, sig_sorted]))
    assert np.unique(wanted).size == wanted.size, "duplicate indices in pool"
    pool = extract_rows(raw, wanted, chunksize)

    bg_tr, bg_va, bg_te = three_way(bg_shuf, f_tr, f_va)
    sig_tr, sig_va, sig_te = three_way(sig_shuf, f_tr, f_va)

    for a, b in [(bg_tr, bg_va), (bg_tr, bg_te), (bg_va, bg_te),
                 (sig_tr, sig_va), (sig_tr, sig_te), (sig_va, sig_te)]:
        assert np.intersect1d(a, b).size == 0, "split overlap detected"

    manifest = {"seed": seed, "raw_csv": str(raw), "raw_rows": int(labels.size),
                "n_background": n_bg, "n_signal": n_sig, "versions": {}}

    print("\n=== building versions ===")
    for version, ratio in d["versions"].items():
        splits = {"train": (bg_tr, sig_tr), "val": (bg_va, sig_va), "test": (bg_te, sig_te)}
        manifest["versions"][version] = {"ratio": ratio, "splits": {}}
        print(f"\nVersion {version} (1:{ratio})")
        for split_name, (bg_idx, sig_idx) in splits.items():
            n_take = bg_idx.size // ratio
            assert n_take > 0, f"ratio {ratio} leaves 0 signal in {split_name}"
            take = sig_idx[:n_take]
            rows = np.concatenate([bg_idx, take])
            rng = np.random.default_rng(seed + hash(version + split_name) % 10_000)
            rng.shuffle(rows)

            df = pool.loc[rows].reset_index(drop=True)
            problems = sanity_check(df, f"{version}_{split_name}", bg_idx.size, n_take)
            if problems:
                raise RuntimeError(f"sanity check failed for {version}_{split_name}")

            out = data_dir / f"version_{version}_{split_name}.parquet"
            df.to_parquet(out, index=False, compression="snappy")
            manifest["versions"][version]["splits"][split_name] = {
                "file": out.name, "rows": int(len(df)),
                "background": int(bg_idx.size), "signal": int(n_take),
                "prevalence": float(n_take / len(df)),
                "size_mb": round(out.stat().st_size / 1e6, 2),
            }

    with open(data_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest written -> {data_dir / 'manifest.json'}")
    print("STEP 3 BUILD COMPLETE")

def load_split(version: str, split: str, cfg: dict):
    """Load one prepared split -> (X float32, y int8)."""
    data_dir = Path(cfg["paths"]["data_dir"])
    path = data_dir / f"version_{version}_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m higgs_bench.data` first")
    df = pd.read_parquet(path)
    if df.shape[1] != N_FEATURES + 1:
        raise RuntimeError(f"{path.name}: expected 29 cols, got {df.shape[1]}")
    if df.isna().any().any():
        raise RuntimeError(f"{path.name}: contains nulls")
    y = df["label"].to_numpy(dtype=np.int8)
    X = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(f"{path.name}: labels not binary 0/1")
    return X, y


if __name__ == "__main__":
    build()