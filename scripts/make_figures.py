"""Publication figures. All read from results/*.csv - no refitting."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from higgs_bench.data import load_config

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

TECH_ORDER = ["baseline", "class_weight", "smote", "undersample", "focal"]
VER_LABEL = {"A": "A  (1:1)", "B": "B  (1:10)", "C": "C  (1:50)"}


def fig1_uncertainty(res, fig_dir):
    """THE key figure: seed-only spread vs bootstrap CI width."""
    summ = pd.read_csv(res / "summary_by_cell.csv")
    boot = pd.read_csv(res / "bootstrap_auprc.csv")
    seed_w = (2 * summ["auprc_ci95"]).dropna()
    boot_w = boot["ci_width"].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-6, -0.5, 45)
    ax.hist(seed_w, bins=bins, alpha=0.75, label=f"model-seed only (n=5)\nmedian {seed_w.median():.5f}")
    ax.hist(boot_w, bins=bins, alpha=0.75, label=f"test-set bootstrap\nmedian {boot_w.median():.5f}")
    ax.set_xscale("log")
    ax.set_xlabel("width of 95% interval on AUPRC")
    ax.set_ylabel("number of cells")
    ax.set_title("Seed variance understates uncertainty by ~60x")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "fig1_uncertainty_sources.png")
    plt.close(fig)


def fig2_deltas(res, fig_dir):
    """Per-cell technique effect with paired bootstrap CI."""
    d = pd.read_csv(res / "bootstrap_deltas.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharex=True)
    for ax, v in zip(axes, ["A", "B", "C"]):
        sub = d[d["version"] == v].sort_values(["technique", "model"]).reset_index(drop=True)
        y = np.arange(len(sub))
        colors = ["tab:red" if r == "hurts" else "tab:green" if r == "helps" else "0.6"
                  for r in sub["direction"]]
        ax.errorbar(sub["delta"], y,
                    xerr=[sub["delta"] - sub["delta_ci_lo"],
                          sub["delta_ci_hi"] - sub["delta"]],
                    fmt="none", ecolor=colors, elinewidth=1.4)
        ax.scatter(sub["delta"], y, c=colors, s=18, zorder=3)
        ax.axvline(0, color="k", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r.model}  {r.technique}" for r in sub.itertuples()],
                           fontsize=6)
        ax.set_title(f"Version {VER_LABEL[v]}")
        ax.set_xlabel("AUPRC change vs baseline")
    fig.suptitle("No technique significantly helps; many significantly hurt", y=1.01)
    fig.savefig(fig_dir / "fig2_technique_deltas.png")
    plt.close(fig)


def fig3_auprc_vs_auc(res, fig_dir):
    """AUC looks fine while AUPRC collapses with prevalence."""
    s = pd.read_csv(res / "summary_by_cell.csv")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(3)
    w = 0.35
    auc = [s[s.version == v]["auc_mean"].mean() for v in "ABC"]
    apr = [s[s.version == v]["auprc_mean"].mean() for v in "ABC"]
    prev = [s[s.version == v]["prevalence"].iloc[0] for v in "ABC"]
    ax.bar(x - w/2, auc, w, label="AUC-ROC")
    ax.bar(x + w/2, apr, w, label="AUPRC")
    ax.plot(x + w/2, prev, "k^--", ms=7, label="AUPRC random baseline")
    for i, (a, p) in enumerate(zip(apr, prev)):
        ax.annotate(f"{a/p:.1f}x", (i + w/2, a), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([VER_LABEL[v] for v in "ABC"])
    ax.set_ylabel("mean score across all cells")
    ax.set_title("AUC-ROC stays flat while AUPRC tracks prevalence")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(fig_dir / "fig3_auc_vs_auprc.png")
    plt.close(fig)


def fig4_threshold(res, fig_dir):
    """Threshold tuning explains the apparent value of these techniques."""
    t = pd.read_csv(res / "threshold_contrast.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, v in zip(axes, ["A", "B", "C"]):
        sub = t[t.version == v].set_index("technique").reindex(TECH_ORDER).dropna()
        x = np.arange(len(sub)); w = 0.38
        ax.bar(x - w/2, sub["f1_fixed"], w, label="threshold = 0.5")
        ax.bar(x + w/2, sub["f1_tuned"], w, label="threshold tuned on val")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Version {VER_LABEL[v]}")
        ax.set_ylabel("test F1")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fixed-threshold F1 rewards rebalancing; tuned F1 does not", y=1.02)
    fig.savefig(fig_dir / "fig4_threshold_effect.png")
    plt.close(fig)


def fig5_ranking(res, fig_dir):
    """Model ranking with CIs - overlapping intervals mean no winner."""
    b = pd.read_csv(res / "bootstrap_auprc.csv")
    b = b[b["technique"] == "baseline"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, v in zip(axes, ["A", "B", "C"]):
        sub = b[b.version == v].sort_values("auprc")
        y = np.arange(len(sub))
        ax.errorbar(sub["auprc"], y,
                    xerr=[sub["auprc"] - sub["auprc_ci_lo"],
                          sub["auprc_ci_hi"] - sub["auprc"]],
                    fmt="o", ms=5, capsize=3)
        ax.set_yticks(y); ax.set_yticklabels(sub["model"], fontsize=8)
        ax.set_xlabel("AUPRC")
        ax.set_title(f"Version {VER_LABEL[v]}")
    fig.suptitle("Baseline models with 95% bootstrap CIs", y=1.03)
    fig.savefig(fig_dir / "fig5_model_ranking.png")
    plt.close(fig)


def fig6_cost(res, fig_dir):
    """Compute cost - techniques cost time and deliver nothing."""
    s = pd.read_csv(res / "summary_by_cell.csv")
    g = s.groupby("technique")["fit_s"].mean().reindex(TECH_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(g.index, g.values, color="tab:gray")
    ax.set_ylabel("mean fit time (s)")
    ax.set_title("Training cost by technique")
    plt.xticks(rotation=30, ha="right")
    fig.savefig(fig_dir / "fig6_compute_cost.png")
    plt.close(fig)


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    for fn in (fig1_uncertainty, fig2_deltas, fig3_auprc_vs_auc,
               fig4_threshold, fig5_ranking, fig6_cost):
        fn(res, fig_dir)
        print(f"  {fn.__name__} ok")

    files = sorted(fig_dir.glob("fig*.png"))
    print(f"\n{len(files)} figures written to {fig_dir}")
    for f in files:
        print(f"  {f.name}  {f.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()