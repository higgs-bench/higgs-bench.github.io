"""Precompute JSON for the static web frontend.

Everything the site needs is computed here, so the browser only reads
small JSON files. Raw predictions are never shipped - only threshold
sweeps derived from them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from higgs_bench.data import load_config

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)
R = 4  # decimal places


def sweep(y: np.ndarray, p: np.ndarray) -> dict:
    """Precision/recall/F1/Z across the threshold grid, vectorised."""
    order = np.argsort(-p)
    ys = y[order]
    ps = p[order]
    tp_cum = np.cumsum(ys)
    fp_cum = np.cumsum(1 - ys)
    n_pos = int(y.sum())

    # for each threshold, how many predictions have p >= t
    k = np.searchsorted(-ps, -THRESHOLDS, side="right")
    tp = np.where(k > 0, tp_cum[np.clip(k - 1, 0, None)], 0).astype(float)
    fp = np.where(k > 0, fp_cum[np.clip(k - 1, 0, None)], 0).astype(float)

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / n_pos if n_pos else np.zeros_like(tp)
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(tp), where=(prec + rec) > 0)
    z = np.divide(tp, np.sqrt(tp + fp), out=np.zeros_like(tp), where=(tp + fp) > 0)

    for name, arr in [("precision", prec), ("recall", rec), ("f1", f1)]:
        if arr.min() < 0 or arr.max() > 1:
            raise RuntimeError(f"{name} out of [0,1]: [{arr.min()}, {arr.max()}]")

    return {
        "precision": [round(float(v), R) for v in prec],
        "recall": [round(float(v), R) for v in rec],
        "f1": [round(float(v), R) for v in f1],
        "z": [round(float(v), 2) for v in z],
        "tp": [int(v) for v in tp],
        "fp": [int(v) for v in fp],
    }


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    out = Path("web/public/data")
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(res / "all_runs.csv")
    summary = pd.read_csv(res / "summary_by_cell.csv")
    boot = pd.read_csv(res / "bootstrap_auprc.csv")
    deltas = pd.read_csv(res / "bootstrap_deltas.csv")
    signs = pd.read_csv(res / "sign_tests.csv")

    # --- threshold sweeps from cached predictions -----------------------
    preds_dir = res / "preds"
    files = sorted(preds_dir.glob("*_s0.npz"))
    if not files:
        raise FileNotFoundError("no prediction files in results/preds")

    cells = {}
    for f in files:
        stem = f.stem[:-3]              # strip _s0
        version, rest = stem.split("_", 1)
        d = np.load(f)
        y = d["y_test"].astype(int)
        p = d["p_test"].astype(np.float64)
        if y.shape != p.shape:
            raise RuntimeError(f"{f.name}: shape mismatch")

        row = runs[(runs["key"] == f.stem)]
        if len(row) != 1:
            raise RuntimeError(f"{f.stem}: expected 1 run row, found {len(row)}")
        row = row.iloc[0]

        cells[stem] = {
            **sweep(y, p),
            "tuned_threshold": float(row["tuned_threshold"]),
            "auprc": round(float(row["test_auprc"]), R),
            "auc": round(float(row["test_auc_roc"]), R),
            "prevalence": round(float(row["test_prevalence"]), R),
            "n_test": int(row["test_n_test"]),
            "model": rest.rsplit("_", 1)[0],
            "technique": rest.rsplit("_", 1)[1],
            "version": version,
        }
    print(f"threshold sweeps: {len(cells)} cells")

    with open(out / "threshold_curves.json", "w") as fh:
        json.dump({"thresholds": [float(t) for t in THRESHOLDS],
                   "cells": cells}, fh, separators=(",", ":"))

    # --- ranking: bootstrap CI vs seed-only CI --------------------------
    s = summary[summary["seed" if "seed" in summary else "n"].notna()] if False else summary
    merged = boot.merge(
        s[["version", "model", "technique", "auprc_std", "auprc_ci95", "fit_s"]],
        on=["version", "model", "technique"], how="left")
    merged["seed_ci_lo"] = merged["auprc"] - merged["auprc_ci95"]
    merged["seed_ci_hi"] = merged["auprc"] + merged["auprc_ci95"]
    keep = ["version", "model", "technique", "auprc", "auprc_ci_lo", "auprc_ci_hi",
            "seed_ci_lo", "seed_ci_hi", "ci_width", "auprc_ci95", "prevalence", "fit_s"]
    ranking = merged[keep].round(6)
    if ranking.isna().any().any():
        raise RuntimeError("NaNs in ranking export:\n" + str(ranking[ranking.isna().any(axis=1)]))
    ranking.to_json(out / "ranking.json", orient="records")

    # --- technique deltas ----------------------------------------------
    deltas.round(6).to_json(out / "deltas.json", orient="records")
    signs.round(6).to_json(out / "sign_tests.json", orient="records")

    # --- all runs table -------------------------------------------------
    cols = ["key", "version", "model", "technique", "seed", "test_auprc",
            "test_auc_roc", "test_f1", "fixed_f1", "test_precision",
            "test_recall", "test_brier", "test_mcc", "test_signal_z",
            "tuned_threshold", "test_prevalence", "test_n_test", "fit_seconds"]
    runs[cols].round(6).to_json(out / "runs.json", orient="records")

    # --- meta -----------------------------------------------------------
    meta = {
        "n_runs": int(len(runs)),
        "n_cells": int(len(summary)),
        "seeds": sorted(runs["seed"].unique().tolist()),
        "versions": [
            {"id": v, "label": lbl,
             "prevalence": float(runs[runs.version == v]["test_prevalence"].iloc[0]),
             "n_test": int(runs[runs.version == v]["test_n_test"].iloc[0])}
            for v, lbl in [("A", "1:1"), ("B", "1:10"), ("C", "1:50")]
        ],
        "models": sorted(runs["model"].unique().tolist()),
        "techniques": sorted(runs["technique"].unique().tolist()),
        "median_seed_ci_width": round(float((2 * summary["auprc_ci95"]).median()), 6),
        "median_boot_ci_width": round(float(boot["ci_width"].median()), 6),
        "n_sig_help": int((deltas["direction"] == "helps").sum()),
        "n_sig_hurt": int((deltas["direction"] == "hurts").sum()),
        "n_cells_imbalanced": int(len(deltas[deltas.version != "A"])),
    }
    meta["uncertainty_ratio"] = round(
        meta["median_boot_ci_width"] / meta["median_seed_ci_width"], 1)
    with open(out / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\n=== exported ===")
    total = 0
    for f in sorted(out.glob("*.json")):
        kb = f.stat().st_size / 1024
        total += kb
        print(f"  {f.name:24s} {kb:8.1f} KB")
    print(f"  {'TOTAL':24s} {total:8.1f} KB")
    print(f"\nheadline: {meta['n_runs']} runs, {meta['n_sig_help']} significant "
          f"improvements, uncertainty ratio {meta['uncertainty_ratio']}x")


if __name__ == "__main__":
    main()