"""Bootstrap CIs and sign tests from cached predictions (seed 0).

Uncertainty here comes from the TEST SET, not from model seeds.
Model-seed variance is reported separately in compile_results.py and is
near-zero for deterministic learners - which is why it must not be
presented as a confidence interval.

Paired bootstrap: the same resample index is applied to a technique and
to its baseline, so the CI is on the difference, not on two independent
quantities.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, f1_score

from higgs_bench.data import load_config

N_BOOT = 2000
BOOT_SEED = 12345
VERSIONS = ["A", "B", "C"]
MODELS = ["xgboost", "lightgbm", "random_forest", "catboost",
          "adaboost", "voting_3", "voting_4"]
TECHNIQUES = ["class_weight", "smote", "undersample", "focal"]


def load_pred(preds_dir: Path, version, model, technique, seed=0):
    p = preds_dir / f"{version}_{model}_{technique}_s{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_test"].astype(int), d["p_test"].astype(np.float64)


def boot_indices(y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Stratified bootstrap: resample positives and negatives separately
    so every replicate keeps the original prevalence and never degenerates
    to a single class."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out = np.empty((n_boot, y.size), dtype=np.int32)
    for b in range(n_boot):
        out[b] = np.concatenate([
            rng.choice(pos, pos.size, replace=True),
            rng.choice(neg, neg.size, replace=True),
        ])
    return out


def ci(values, alpha=0.05):
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main():
    cfg = load_config()
    res = Path(cfg["paths"]["results_dir"])
    preds_dir = res / "preds"

    single_rows, delta_rows = [], []

    for version in VERSIONS:
        ref = load_pred(preds_dir, version, "xgboost", "baseline")
        if ref is None:
            raise FileNotFoundError(f"missing baseline preds for version {version}")
        y_ref = ref[0]
        idx = boot_indices(y_ref, N_BOOT, BOOT_SEED)
        prevalence = float(y_ref.mean())
        print(f"\nversion {version}: n_test={y_ref.size:,} prevalence={prevalence:.4f} "
              f"bootstrapping {N_BOOT} replicates")

        cache = {}
        for model in MODELS:
            for tech in ["baseline"] + TECHNIQUES:
                d = load_pred(preds_dir, version, model, tech)
                if d is None:
                    continue
                y, p = d
                if not np.array_equal(y, y_ref):
                    raise RuntimeError(f"{version}/{model}/{tech}: test labels differ")
                cache[(model, tech)] = p

                boot = np.array([average_precision_score(y[i], p[i]) for i in idx])
                lo, hi = ci(boot)
                single_rows.append({
                    "version": version, "model": model, "technique": tech,
                    "auprc": float(average_precision_score(y, p)),
                    "auprc_boot_mean": float(boot.mean()),
                    "auprc_ci_lo": lo, "auprc_ci_hi": hi,
                    "ci_width": hi - lo, "prevalence": prevalence,
                })
            print(f"  {model} done")

        # paired deltas vs baseline
        for model in MODELS:
            if (model, "baseline") not in cache:
                continue
            pb = cache[(model, "baseline")]
            base_boot = np.array([average_precision_score(y_ref[i], pb[i]) for i in idx])
            for tech in TECHNIQUES:
                if (model, tech) not in cache:
                    continue
                pt = cache[(model, tech)]
                tech_boot = np.array([average_precision_score(y_ref[i], pt[i]) for i in idx])
                diff = tech_boot - base_boot
                lo, hi = ci(diff)
                delta_rows.append({
                    "version": version, "model": model, "technique": tech,
                    "baseline_auprc": float(average_precision_score(y_ref, pb)),
                    "technique_auprc": float(average_precision_score(y_ref, pt)),
                    "delta": float(average_precision_score(y_ref, pt)
                                   - average_precision_score(y_ref, pb)),
                    "delta_ci_lo": lo, "delta_ci_hi": hi,
                    "significant": bool(lo > 0 or hi < 0),
                    "direction": "helps" if lo > 0 else ("hurts" if hi < 0 else "null"),
                })

    single = pd.DataFrame(single_rows)
    delta = pd.DataFrame(delta_rows)
    single.to_csv(res / "bootstrap_auprc.csv", index=False)
    delta.to_csv(res / "bootstrap_deltas.csv", index=False)

    # sign test across models
    sign_rows = []
    for (v, t), sub in delta.groupby(["version", "technique"]):
        n = len(sub)
        TIE = 1e-9  # deltas below this are numerically identical, not effects
        eff = sub[sub["delta"].abs() > TIE]
        n_eff = len(eff)
        n_pos = int((eff["delta"] > 0).sum())
        p = (float(stats.binomtest(n_pos, n_eff, 0.5).pvalue)
             if n_eff > 0 else float("nan"))
        sign_rows.append({
            "version": v, "technique": t, "n_models": n, "n_effective": n_eff,
            "n_helped": n_pos, "n_hurt": n - n_pos,
            "mean_delta": float(sub["delta"].mean()),
            "sign_test_p": p,
            "n_sig_hurt": int((sub["direction"] == "hurts").sum()),
            "n_sig_help": int((sub["direction"] == "helps").sum()),
        })
    sign = pd.DataFrame(sign_rows)
    sign.to_csv(res / "sign_tests.csv", index=False)

    print("\n=== bootstrap CI widths vs seed-only spread ===")
    print(f"median bootstrap 95% CI width: {single['ci_width'].median():.5f}")
    print("(compare: median seed-only CI half-width was ~0.0004 - "
          "seed variance understates uncertainty)")

    print("\n=== best cell per version, with honest CI ===")
    for v in VERSIONS:
        s = single[single["version"] == v].sort_values("auprc", ascending=False)
        print(f"\nversion {v}")
        print(s.head(6)[["model", "technique", "auprc",
                         "auprc_ci_lo", "auprc_ci_hi"]].to_string(index=False))
        top = s.iloc[0]
        overlap = s[(s["auprc_ci_hi"] >= top["auprc_ci_lo"])]
        print(f"  cells whose CI overlaps the best: {len(overlap)} of {len(s)}")

    print("\n=== sign test: does the technique beat baseline across models? ===")
    print(sign.to_string(index=False))

    print("\n=== per-cell significant deltas ===")
    for v in VERSIONS:
        d = delta[delta["version"] == v]
        print(f"version {v}: sig hurt={int((d['direction']=='hurts').sum())}, "
              f"sig help={int((d['direction']=='helps').sum())}, "
              f"null={int((d['direction']=='null').sum())} (of {len(d)})")

    print(f"\nwritten: bootstrap_auprc.csv, bootstrap_deltas.csv, sign_tests.csv")


if __name__ == "__main__":
    main()