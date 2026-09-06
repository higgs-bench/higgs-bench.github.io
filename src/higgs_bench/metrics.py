"""Evaluation metrics.

Primary metric is AUPRC (average precision), reported with the
prevalence baseline so lift is visible.

WARNING on signal significance Z:
    Z = TP / sqrt(TP + FP)
This is the s/sqrt(s+b) approximation computed on raw test-set counts.
It scales with sqrt(dataset size): doubling the test set inflates Z by
~1.41 with no change in classifier quality. It is therefore NOT a
physics discovery significance and must never be compared against the
Z>=5 threshold without scaling to a stated luminosity and cross-section.
Here it is used only as a relative ranking statistic between models
evaluated on the SAME test set. Always report n_test alongside it.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

FIXED_THRESHOLD = 0.5


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("empty input")
    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains inf")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError(f"y_prob outside [0,1]: [{y_prob.min()}, {y_prob.max()}]")
    uniq = set(np.unique(y_true).astype(int).tolist())
    if not uniq <= {0, 1}:
        raise ValueError(f"y_true not binary: {uniq}")
    if len(uniq) < 2:
        raise ValueError("y_true has only one class; metrics undefined")
    return y_true.astype(int), y_prob


def significance_z(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """s/sqrt(s+b) on raw counts. SCALE-DEPENDENT - see module docstring."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    denom = tp + fp
    return 0.0 if denom == 0 else tp / np.sqrt(denom)


def tune_threshold(y_true, y_prob, metric: str = "f1",
                   grid: np.ndarray | None = None) -> tuple[float, float]:
    """Pick threshold maximising `metric`. MUST be called on validation only."""
    y_true, y_prob = _validate(y_true, y_prob)
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 4)

    best_t, best_v = 0.5, -np.inf
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            v = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        elif metric == "mcc":
            v = matthews_corrcoef(y_true, y_pred)
        elif metric == "z":
            v = significance_z(y_true, y_pred)
        else:
            raise ValueError(f"unknown metric: {metric}")
        if v > best_v:
            best_t, best_v = float(t), float(v)
    return best_t, best_v


def compute_metrics(y_true, y_prob, threshold: float = FIXED_THRESHOLD,
                    prefix: str = "") -> dict:
    """Threshold-free + thresholded metrics. Threshold must come from val."""
    y_true, y_prob = _validate(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    n = int(y_true.size)
    n_sig = int(y_true.sum())
    prevalence = n_sig / n

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    auprc = float(average_precision_score(y_true, y_prob))

    m = {
        "n_test": n,
        "n_signal": n_sig,
        "prevalence": prevalence,
        "threshold": float(threshold),
        "auprc": auprc,
        "auprc_baseline": prevalence,
        "auprc_lift": auprc / prevalence if prevalence > 0 else np.nan,
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "signal_z": float(significance_z(y_true, y_pred)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }

    if tp + fp + fn + tn != n:
        raise RuntimeError("confusion matrix does not sum to n")
    for k in ("auprc", "auc_roc", "f1", "precision", "recall", "brier", "prevalence"):
        v = m[k]
        if not (0.0 <= v <= 1.0):
            raise RuntimeError(f"metric {k} out of [0,1]: {v}")
    if not (-1.0 <= m["mcc"] <= 1.0):
        raise RuntimeError(f"mcc out of [-1,1]: {m['mcc']}")

    return {f"{prefix}{k}": v for k, v in m.items()} if prefix else m