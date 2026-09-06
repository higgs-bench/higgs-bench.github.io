"""Aggregate run JSONs -> tidy table + seed-variance and paired stats."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from higgs_bench.data import load_config

PRIMARY = "test_auprc"


def load_runs(cfg) -> pd.DataFrame:
    runs_dir = Path(cfg["paths"]["results_dir"]) / "runs"
    rows = []
    for p in sorted(runs_dir.glob("*.json")):
        with open(p, "r", encoding="utf-8") as fh:
            rows.append(json.load(fh))
    df = pd.DataFrame(rows)
    bad = df[df["status"] != "ok"]
    if len(bad):
        raise RuntimeError(f"{len(bad)} failed runs present: {bad['key'].tolist()[:5]}")
    return df


def sanity(df: pd.DataFrame):
    print("=== sanity ===")
    print(f"rows: {len(df)}  (expect 450)")
    print(f"nulls in {PRIMARY}: {int(df[PRIMARY].isna().sum())}")
    for col in ["test_auprc", "test_auc_roc", "test_f1", "test_brier", "tuned_threshold"]:
        lo, hi = df[col].min(), df[col].max()
        flag = "" if 0 <= lo and hi <= 1 else "  <-- OUT OF BOUNDS"
        print(f"  {col:18s} [{lo:.4f}, {hi:.4f}]{flag}")
    counts = df.groupby(["version", "model", "technique"]).size()
    if not (counts == 5).all():
        print("  !! cells without exactly 5 seeds:")
        print(counts[counts != 5])
    else:
        print("  all cells have 5 seeds")
    below = df[df["test_auprc"] < df["test_prevalence"]]
    print(f"  runs worse than random: {len(below)}")
    if len(below):
        print(below[["key", "test_auprc", "test_prevalence"]].to_string(index=False))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["version", "model", "technique"])
    out = g.agg(
        auprc_mean=(PRIMARY, "mean"),
        auprc_std=(PRIMARY, "std"),
        auc_mean=("test_auc_roc", "mean"),
        auc_std=("test_auc_roc", "std"),
        f1_mean=("test_f1", "mean"),
        f1_std=("test_f1", "std"),
        f1_fixed_mean=("fixed_f1", "mean"),
        brier_mean=("test_brier", "mean"),
        thr_mean=("tuned_threshold", "mean"),
        thr_std=("tuned_threshold", "std"),
        z_mean=("test_signal_z", "mean"),
        prevalence=("test_prevalence", "first"),
        fit_s=("fit_seconds", "mean"),
        n=(PRIMARY, "size"),
    ).reset_index()
    # 95% CI of the mean, t-distribution, n=5
    tcrit = stats.t.ppf(0.975, df=out["n"] - 1)
    out["auprc_ci95"] = tcrit * out["auprc_std"] / np.sqrt(out["n"])
    out["auprc_lift"] = out["auprc_mean"] / out["prevalence"]
    return out.sort_values(["version", "auprc_mean"], ascending=[True, False])


def paired_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (v, m), sub in df.groupby(["version", "model"]):
        base = sub[sub["technique"] == "baseline"].set_index("seed")[PRIMARY]
        for tech, tsub in sub.groupby("technique"):
            if tech == "baseline":
                continue
            t = tsub.set_index("seed")[PRIMARY]
            common = base.index.intersection(t.index)
            if len(common) < 5:
                continue
            b, x = base.loc[common].to_numpy(), t.loc[common].to_numpy()
            d = x - b
            sd = d.std(ddof=1)
            try:
                _, p = stats.wilcoxon(x, b)
            except ValueError:
                p = np.nan
            rows.append({
                "version": v, "model": m, "technique": tech,
                "baseline_auprc": b.mean(), "technique_auprc": x.mean(),
                "delta": d.mean(), "delta_sd": sd,
                "cohens_d": d.mean() / sd if sd > 0 else np.nan,
                "wilcoxon_p": p,
                "beats_baseline": bool(d.mean() > 0),
            })
    return pd.DataFrame(rows)


def threshold_contrast(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["f1_gain_from_tuning"] = d["test_f1"] - d["fixed_f1"]
    return (d.groupby(["version", "technique"])
             .agg(f1_tuned=("test_f1", "mean"),
                  f1_fixed=("fixed_f1", "mean"),
                  gain=("f1_gain_from_tuning", "mean"),
                  gain_sd=("f1_gain_from_tuning", "std"))
             .reset_index())


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    df = load_runs(cfg)
    sanity(df)

    df.to_csv(res / "all_runs.csv", index=False)
    summary = summarise(df)
    summary.to_csv(res / "summary_by_cell.csv", index=False)
    paired = paired_vs_baseline(df)
    paired.to_csv(res / "paired_vs_baseline.csv", index=False)
    thr = threshold_contrast(df)
    thr.to_csv(res / "threshold_contrast.csv", index=False)

    print("\n=== seed variance (spread across 5 seeds) ===")
    print(f"median AUPRC std within a cell: {summary['auprc_std'].median():.5f}")
    print(f"median 95% CI half-width      : {summary['auprc_ci95'].median():.5f}")
    for v in ["A", "B", "C"]:
        s = summary[summary["version"] == v]
        rng = s["auprc_mean"].max() - s["auprc_mean"].min()
        print(f"  version {v}: best-worst cell gap = {rng:.5f}, "
              f"median within-cell CI = {s['auprc_ci95'].median():.5f}")

    print("\n=== top 5 cells per version ===")
    for v in ["A", "B", "C"]:
        s = summary[summary["version"] == v].head(5)
        print(f"\nversion {v} (prevalence {s['prevalence'].iloc[0]:.4f})")
        print(s[["model", "technique", "auprc_mean", "auprc_ci95",
                 "auc_mean", "f1_mean", "thr_mean"]].to_string(index=False))

    print("\n=== techniques vs baseline (paired, 5 seeds) ===")
    agg = (paired.groupby(["version", "technique"])
                 .agg(mean_delta=("delta", "mean"),
                      n_models_better=("beats_baseline", "sum"),
                      n_models=("beats_baseline", "size"),
                      median_p=("wilcoxon_p", "median"))
                 .reset_index())
    print(agg.to_string(index=False))

    print("\n=== F1 gain from threshold tuning ===")
    print(thr.to_string(index=False))

    print(f"\nwritten: {res/'all_runs.csv'}, summary_by_cell.csv, "
          f"paired_vs_baseline.csv, threshold_contrast.csv")


if __name__ == "__main__":
    main()