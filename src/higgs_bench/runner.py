"""Single-cell experiment runner.

Protocol (identical for every cell):
  1. fit on TRAIN (after technique applied to train only)
  2. tune decision threshold on VAL
  3. evaluate on TEST at the tuned threshold AND at 0.5

Every run is cached as JSON keyed by (version, model, technique, seed),
so an interrupted sweep resumes without repeating finished work.
Optionally the val/test probability vectors are cached as .npz so that
bootstrap confidence intervals can be computed without refitting.
"""
from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np

from higgs_bench.data import load_config, load_split
from higgs_bench.metrics import compute_metrics, tune_threshold
from higgs_bench.models import build_model, predict_proba_pos
from higgs_bench.techniques import apply_technique


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _proba(model, X) -> tuple[np.ndarray, bool]:
    """Positive-class probability. Returns (p, sigmoid_was_applied).

    Custom objectives make some libraries emit raw margins - LightGBM
    returns a 1-D array, XGBoost a 2-D array outside [0,1]. Both are
    converted, and the conversion is recorded in the results row.
    """
    try:
        return predict_proba_pos(model, X), False
    except RuntimeError as err:
        if "outside [0,1]" not in str(err) and "unexpected predict_proba shape" not in str(err):
            raise
        raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        raw = raw.ravel() if raw.ndim == 1 else raw[:, -1]
        if raw.shape[0] != X.shape[0]:
            raise RuntimeError(f"raw score length {raw.shape[0]} != n rows {X.shape[0]}")
        p = _sigmoid(raw)
        if not np.isfinite(p).all():
            raise RuntimeError("sigmoid of raw margin still non-finite")
        return p, True


def run_key(version, model, technique, seed) -> str:
    return f"{version}_{model}_{technique}_s{seed}"


def run_cell(version: str, model_name: str, technique: str, seed: int,
             cfg: dict, force: bool = False,
             save_predictions: bool = False) -> dict:
    """Run one experiment cell. Returns the result row."""
    runs_dir = Path(cfg["paths"]["results_dir"]) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    key = run_key(version, model_name, technique, seed)
    cache = runs_dir / f"{key}.json"

    preds_dir = Path(cfg["paths"]["results_dir"]) / "preds"
    npz_path = preds_dir / f"{key}.npz"
    need_preds = save_predictions and not npz_path.exists()

    if cache.exists() and not force and not need_preds:
        with open(cache, "r", encoding="utf-8") as fh:
            row = json.load(fh)
        print(f"  [cached] {key}")
        return row

    row = {
        "key": key, "version": version, "model": model_name,
        "technique": technique, "seed": seed, "status": "running",
        "host": platform.node(),
    }

    try:
        X_tr, y_tr = load_split(version, "train", cfg)
        X_va, y_va = load_split(version, "val", cfg)
        X_te, y_te = load_split(version, "test", cfg)

        row["n_train_raw"] = int(X_tr.shape[0])
        row["train_prevalence_raw"] = float(y_tr.mean())

        Xt, yt, _, kwargs = apply_technique(technique, X_tr, y_tr, seed, model_name)
        row["n_train_used"] = int(Xt.shape[0])
        row["train_prevalence_used"] = float(yt.mean())

        # AdaBoost has no scale_pos_weight; express it as sample weights
        sample_weight = None
        if model_name == "adaboost" and "scale_pos_weight" in kwargs:
            spw = kwargs.pop("scale_pos_weight")
            sample_weight = np.where(yt == 1, spw, 1.0).astype(np.float64)
            row["adaboost_sample_weight_used"] = True

        model = build_model(model_name, seed=seed, **kwargs)

        t0 = time.time()
        if sample_weight is not None:
            model.fit(Xt, yt, sample_weight=sample_weight)
        else:
            model.fit(Xt, yt)
        row["fit_seconds"] = round(time.time() - t0, 2)

        t0 = time.time()
        p_va, sig_va = _proba(model, X_va)
        p_te, sig_te = _proba(model, X_te)
        row["predict_seconds"] = round(time.time() - t0, 2)
        row["sigmoid_applied"] = bool(sig_va or sig_te)

        if save_predictions:
            preds_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path,
                                y_test=y_te.astype(np.int8),
                                p_test=p_te.astype(np.float32),
                                y_val=y_va.astype(np.int8),
                                p_val=p_va.astype(np.float32))
            row["preds_file"] = npz_path.name

        thr, val_f1 = tune_threshold(y_va, p_va, metric="f1")
        row["tuned_threshold"] = thr
        row["val_f1_at_tuned"] = float(val_f1)

        row.update(compute_metrics(y_te, p_te, threshold=thr, prefix="test_"))
        row.update(compute_metrics(y_te, p_te, threshold=0.5, prefix="fixed_"))
        row.update(compute_metrics(y_va, p_va, threshold=thr, prefix="val_"))

        if row["test_auprc"] < row["test_prevalence"]:
            row["warning"] = "AUPRC below prevalence baseline - worse than random"

        row["status"] = "ok"

    except Exception as err:
        row["status"] = "failed"
        row["error"] = f"{type(err).__name__}: {err}"
        row["traceback"] = traceback.format_exc()
        print(f"  [FAILED] {key}: {row['error']}")

    with open(cache, "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)

    if row["status"] == "ok":
        print(f"  [ok] {key:44s} AUPRC={row['test_auprc']:.4f} "
              f"(base {row['test_prevalence']:.4f}) F1={row['test_f1']:.4f} "
              f"thr={row['tuned_threshold']:.2f} {row['fit_seconds']:.0f}s")
    return row


if __name__ == "__main__":
    cfg = load_config()
    for tech in ["baseline", "class_weight"]:
        run_cell("C", "xgboost", tech, seed=0, cfg=cfg)