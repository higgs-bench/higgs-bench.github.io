"""Class-imbalance handling techniques.

Each technique returns (X_train, y_train, sample_weight, model_kwargs).
Resampling touches TRAINING data only - never val, never test.

Threshold tuning is NOT a technique here. It is applied uniformly to
every technique's output probabilities using the validation set, so
that technique effects and threshold effects can be separated.
"""
from __future__ import annotations

import numpy as np

TECHNIQUES = ["baseline", "class_weight", "smote", "undersample", "focal"]

# techniques requiring a model that supports a custom objective
NEEDS_FOCAL_SUPPORT = {"focal"}


def _imbalance_ratio(y: np.ndarray) -> float:
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0:
        raise ValueError("no positive samples in training data")
    return n_neg / n_pos


def make_focal_objective(gamma: float = 2.0, alpha: float = 0.5):
    """Binary focal loss (Lin et al. 2017) as a custom objective.

    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Returns exact grad and hess w.r.t. the raw margin z.

    Deviation: alpha defaults to 0.5 (neutral), not 0.25. With a rare
    positive class, alpha=0.25 down-weights positives and mixes class
    weighting into the focal arm. Neutral alpha isolates the focal
    modulation, which is what this technique is meant to test.
    """
    def objective(y_true, y_pred_raw):
        y = np.asarray(y_true, dtype=np.float64).ravel()
        z = np.clip(np.asarray(y_pred_raw, dtype=np.float64).ravel(), -30.0, 30.0)
        p = np.clip(1.0 / (1.0 + np.exp(-z)), 1e-7, 1.0 - 1e-7)

        s = np.where(y == 1, 1.0, -1.0)          # dp_t/dz sign
        pt = np.where(y == 1, p, 1.0 - p)
        at = np.where(y == 1, alpha, 1.0 - alpha)
        q = 1.0 - pt
        L = np.log(pt)
        w = q ** gamma

        # u = dFL/dp_t * (-1/at) ... assembled directly:
        u = w * (q - gamma * pt * L)
        grad = -s * at * u

        du_dpt = (-(gamma + 1.0) * w
                  - gamma * w * L
                  + gamma ** 2 * pt * q ** (gamma - 1.0) * L
                  - gamma * w)
        hess = -at * pt * q * du_dpt
        hess = np.maximum(hess, 1e-6)

        if not np.isfinite(grad).all() or not np.isfinite(hess).all():
            raise RuntimeError("focal objective produced non-finite grad/hess")
        return grad, hess

    return objective


def apply_technique(technique: str, X_train, y_train, seed: int,
                    model_name: str, gamma: float = 2.0):
    """Return (X, y, sample_weight, model_kwargs) for the given technique."""
    if technique not in TECHNIQUES:
        raise ValueError(f"unknown technique {technique!r}")

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int8).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y row mismatch: {X.shape[0]} vs {y.shape[0]}")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("training labels must contain both classes")

    n_before = X.shape[0]
    kwargs: dict = {}

    if technique == "baseline":
        pass

    elif technique == "class_weight":
        kwargs["scale_pos_weight"] = _imbalance_ratio(y)

    elif technique == "smote":
        from imblearn.over_sampling import SMOTE
        k = min(5, int(np.sum(y == 1)) - 1)
        if k < 1:
            raise ValueError("too few positives for SMOTE")
        X, y = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X, y)
        X = X.astype(np.float32)
        y = y.astype(np.int8)

    elif technique == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        X, y = RandomUnderSampler(random_state=seed).fit_resample(X, y)
        X = X.astype(np.float32)
        y = y.astype(np.int8)

    elif technique == "focal":
        from higgs_bench.models import SUPPORTS_FOCAL
        if model_name not in SUPPORTS_FOCAL:
            raise ValueError(f"focal loss unsupported for model {model_name!r}")
        kwargs["objective"] = make_focal_objective(gamma=gamma)

    # post-conditions
    if X.shape[0] != y.shape[0]:
        raise RuntimeError("resampling desynchronised X and y")
    if np.isnan(X).any():
        raise RuntimeError("resampling introduced NaNs")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError("resampling removed a class")
    if technique in ("smote", "undersample"):
        r = _imbalance_ratio(y)
        if abs(r - 1.0) > 0.01:
            raise RuntimeError(f"{technique} did not balance: ratio={r:.4f}")
    if technique == "baseline" and X.shape[0] != n_before:
        raise RuntimeError("baseline must not change row count")

    return X, y, None, kwargs


def valid_cells(model_names, technique_names, focal_supported) -> list[tuple[str, str]]:
    """Enumerate legal (model, technique) pairs - the grid is ragged."""
    cells = []
    for m in model_names:
        for t in technique_names:
            if t == "focal" and m not in focal_supported:
                continue
            cells.append((m, t))
    return cells