"""Model factory.

Design: matched capacity, not identical kwargs. Every tree ensemble gets
500 estimators / depth 6 / lr 0.1 where the hyperparameter exists.
Passing one shared kwargs dict to all libraries is what breaks -
sklearn estimators reject eval_metric/verbosity, so each family has
its own builder.

Deviation: AdaBoost uses 200 estimators, not 500. With depth-6 base
trees it trains sequentially and is ~10x slower than XGBoost on 6 cores.
This is a compute-budget decision and is reported in the paper.
"""
from __future__ import annotations

import numpy as np

N_ESTIMATORS = 500
MAX_DEPTH = 6
LEARNING_RATE = 0.1
N_JOBS = 2  # 3 parallel worker processes x 2 threads = 6 cores

MODEL_NAMES = [
    "xgboost", "lightgbm", "random_forest", "catboost",
    "adaboost", "voting_3", "voting_4",
]

# models that accept a custom objective (focal loss)
SUPPORTS_FOCAL = {"xgboost", "lightgbm"}
# models with a native scale_pos_weight-style knob
SUPPORTS_CLASS_WEIGHT = {
    "xgboost", "lightgbm", "random_forest", "catboost", "adaboost",
    "voting_3", "voting_4",
}


def _xgb(seed, scale_pos_weight=None, objective=None):
    from xgboost import XGBClassifier
    kw = dict(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, random_state=seed,
        n_jobs=N_JOBS, tree_method="hist", verbosity=0,
        eval_metric="logloss",
    )
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = float(scale_pos_weight)
    if objective is not None:
        kw["objective"] = objective
    return XGBClassifier(**kw)


def _lgb(seed, scale_pos_weight=None, objective=None):
    from lightgbm import LGBMClassifier
    kw = dict(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, random_state=seed,
        n_jobs=N_JOBS, verbose=-1,
    )
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = float(scale_pos_weight)
    if objective is not None:
        kw["objective"] = objective
    return LGBMClassifier(**kw)


def _rf(seed, class_weight=None):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        random_state=seed, n_jobs=N_JOBS, class_weight=class_weight,
    )


def _cat(seed, scale_pos_weight=None):
    from catboost import CatBoostClassifier
    kw = dict(
        iterations=N_ESTIMATORS, depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE, random_seed=seed,
        thread_count=N_JOBS, verbose=0, allow_writing_files=False,
    )
    if scale_pos_weight is not None:
        kw["scale_pos_weight"] = float(scale_pos_weight)
    return CatBoostClassifier(**kw)


def _ada(seed):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    return AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=seed),
        n_estimators=200, learning_rate=LEARNING_RATE, random_state=seed,
        algorithm="SAMME",
    )


def _voting(seed, n_models, scale_pos_weight=None):
    from sklearn.ensemble import VotingClassifier
    est = [
        ("xgb", _xgb(seed, scale_pos_weight)),
        ("lgb", _lgb(seed, scale_pos_weight)),
        ("rf", _rf(seed, "balanced" if scale_pos_weight else None)),
    ]
    if n_models == 4:
        est.append(("cat", _cat(seed, scale_pos_weight)))
    return VotingClassifier(estimators=est, voting="soft", n_jobs=1)


def build_model(name: str, seed: int, scale_pos_weight=None, objective=None):
    """Return an unfitted estimator with a predict_proba interface."""
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")
    if objective is not None and name not in SUPPORTS_FOCAL:
        raise ValueError(f"{name} does not support a custom objective")

    if name == "xgboost":
        return _xgb(seed, scale_pos_weight, objective)
    if name == "lightgbm":
        return _lgb(seed, scale_pos_weight, objective)
    if name == "random_forest":
        return _rf(seed, "balanced" if scale_pos_weight else None)
    if name == "catboost":
        return _cat(seed, scale_pos_weight)
    if name == "adaboost":
        return _ada(seed)
    if name == "voting_3":
        return _voting(seed, 3, scale_pos_weight)
    if name == "voting_4":
        return _voting(seed, 4, scale_pos_weight)
    raise AssertionError("unreachable")


def predict_proba_pos(model, X) -> np.ndarray:
    """Positive-class probability, shape-checked."""
    p = model.predict_proba(X)
    if p.ndim != 2 or p.shape[1] != 2:
        raise RuntimeError(f"unexpected predict_proba shape {p.shape}")
    out = np.asarray(p[:, 1], dtype=np.float64)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite probabilities")
    if out.min() < 0.0 or out.max() > 1.0:
        raise RuntimeError(f"probabilities outside [0,1]: [{out.min()}, {out.max()}]")
    return out