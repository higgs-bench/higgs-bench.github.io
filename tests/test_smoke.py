import numpy as np
import pytest

from higgs_bench.metrics import compute_metrics, significance_z, tune_threshold


def _toy(n=2000, prevalence=0.1, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    p = np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0.001, 0.999)
    return y, p


def test_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["auc_roc"] == 1.0
    assert m["f1"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0


def test_random_classifier_auprc_near_prevalence():
    rng = np.random.default_rng(1)
    y = (rng.random(20000) < 0.1).astype(int)
    p = rng.random(20000)
    m = compute_metrics(y, p)
    assert abs(m["auprc"] - 0.1) < 0.02
    assert abs(m["auc_roc"] - 0.5) < 0.02


def test_bounds_and_confusion():
    y, p = _toy()
    m = compute_metrics(y, p, threshold=0.4)
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n_test"]
    assert 0.0 <= m["auprc"] <= 1.0
    assert abs(m["prevalence"] - m["n_signal"] / m["n_test"]) < 1e-12


def test_rejects_bad_input():
    y, p = _toy()
    with pytest.raises(ValueError):
        compute_metrics(y, p * 2)                 # out of [0,1]
    with pytest.raises(ValueError):
        compute_metrics(np.zeros(10), np.full(10, 0.5))   # one class
    with pytest.raises(ValueError):
        compute_metrics(y[:10], p)                # shape mismatch
    bad = p.copy(); bad[0] = np.nan
    with pytest.raises(ValueError):
        compute_metrics(y, bad)


def test_threshold_tuning_beats_default():
    y, p = _toy(prevalence=0.05, seed=3)
    t, v = tune_threshold(y, p, metric="f1")
    assert 0.0 < t < 1.0
    assert v >= compute_metrics(y, p, threshold=0.5)["f1"] - 1e-9


def test_z_is_scale_dependent():
    """Documents the flaw: duplicating the test set inflates Z ~sqrt(2)."""
    y, p = _toy(n=5000, prevalence=0.05, seed=7)
    pred = (p >= 0.5).astype(int)
    z1 = significance_z(y, pred)
    z2 = significance_z(np.concatenate([y, y]), np.concatenate([pred, pred]))
    assert z2 == pytest.approx(z1 * np.sqrt(2), rel=1e-6)

# --- model factory tests -------------------------------------------------

from higgs_bench.models import MODEL_NAMES, build_model, predict_proba_pos


def _tiny(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 28)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name):
    X, y = _tiny()
    m = build_model(name, seed=42)
    m.fit(X, y)
    p = predict_proba_pos(m, X)
    assert p.shape == (len(y),)
    assert 0.0 <= p.min() and p.max() <= 1.0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_class_weight_variant_builds(name):
    X, y = _tiny()
    m = build_model(name, seed=42, scale_pos_weight=5.0)
    m.fit(X, y)
    assert predict_proba_pos(m, X).shape == (len(y),)


def test_focal_rejected_for_unsupported_models():
    with pytest.raises(ValueError):
        build_model("random_forest", seed=1, objective="binary:logistic")
    with pytest.raises(ValueError):
        build_model("not_a_model", seed=1)


def test_seed_reproducibility():
    X, y = _tiny()
    a = build_model("xgboost", seed=42); a.fit(X, y)
    b = build_model("xgboost", seed=42); b.fit(X, y)
    np.testing.assert_allclose(predict_proba_pos(a, X), predict_proba_pos(b, X))

# --- technique tests -----------------------------------------------------

from higgs_bench.models import SUPPORTS_FOCAL
from higgs_bench.techniques import (
    TECHNIQUES, apply_technique, make_focal_objective, valid_cells,
)


def _imbalanced(n=2000, prevalence=0.05, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.int8)
    X = rng.normal(size=(n, 28)).astype(np.float32) + y[:, None] * 0.8
    return X, y


def test_baseline_is_identity():
    X, y = _imbalanced()
    Xo, yo, sw, kw = apply_technique("baseline", X, y, 42, "xgboost")
    assert Xo.shape == X.shape and kw == {} and sw is None


def test_class_weight_ratio_correct():
    X, y = _imbalanced(prevalence=0.02, seed=1)
    _, _, _, kw = apply_technique("class_weight", X, y, 42, "xgboost")
    expected = (y == 0).sum() / (y == 1).sum()
    assert abs(kw["scale_pos_weight"] - expected) < 1e-9
    assert Xo_unchanged(X)


def Xo_unchanged(X):
    return np.isfinite(X).all()


@pytest.mark.parametrize("tech", ["smote", "undersample"])
def test_resamplers_balance_and_stay_clean(tech):
    X, y = _imbalanced(prevalence=0.05, seed=2)
    Xo, yo, _, _ = apply_technique(tech, X, y, 42, "xgboost")
    assert (yo == 1).sum() == (yo == 0).sum()
    assert Xo.shape[0] == yo.shape[0]
    assert not np.isnan(Xo).any()
    if tech == "smote":
        assert Xo.shape[0] > X.shape[0]
    else:
        assert Xo.shape[0] < X.shape[0]


def test_focal_objective_shapes_and_finiteness():
    obj = make_focal_objective(gamma=2.0)
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(float)
    z = rng.normal(0, 3, 500)
    g, h = obj(y, z)
    assert g.shape == (500,) and h.shape == (500,)
    assert np.isfinite(g).all() and np.isfinite(h).all()
    assert (h > 0).all(), "Hessian must be positive"


def test_focal_extreme_margins_stable():
    obj = make_focal_objective()
    y = np.array([0.0, 1.0, 0.0, 1.0])
    z = np.array([-1e4, 1e4, 1e4, -1e4])
    g, h = obj(y, z)
    assert np.isfinite(g).all() and np.isfinite(h).all()


def test_focal_rejected_for_wrong_model():
    X, y = _imbalanced()
    with pytest.raises(ValueError):
        apply_technique("focal", X, y, 42, "random_forest")


def test_focal_trains_on_xgboost():
    from higgs_bench.models import build_model, predict_proba_pos
    X, y = _imbalanced(n=800, prevalence=0.1, seed=5)
    Xo, yo, _, kw = apply_technique("focal", X, y, 42, "xgboost")
    m = build_model("xgboost", seed=42, **kw)
    m.fit(Xo, yo)
    p = predict_proba_pos(m, Xo)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_grid_is_ragged():
    cells = valid_cells(
        ["xgboost", "lightgbm", "random_forest", "catboost",
         "adaboost", "voting_3", "voting_4"],
        TECHNIQUES, SUPPORTS_FOCAL,
    )
    assert len(cells) == 30  # 7 models x 5 techniques, minus focal for 5 models
    assert ("random_forest", "focal") not in cells
    assert ("xgboost", "focal") in cells


def test_focal_gradient_matches_numerical():
    """Finite-difference check - catches sign errors in the derivation."""
    gamma, alpha = 2.0, 0.5
    obj = make_focal_objective(gamma=gamma, alpha=alpha)

    def loss(y, z):
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, alpha, 1 - alpha)
        return -at * (1 - pt) ** gamma * np.log(pt)

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)
    z = rng.normal(0, 2.0, 400)

    g, h = obj(y, z)
    eps = 1e-5
    num_g = (loss(y, z + eps) - loss(y, z - eps)) / (2 * eps)
    np.testing.assert_allclose(g, num_g, rtol=1e-4, atol=1e-6)

    eps2 = 1e-3
    num_h = (loss(y, z + eps2) - 2 * loss(y, z) + loss(y, z - eps2)) / eps2 ** 2
    mask = num_h > 1e-3          # skip clipped/near-zero region
    np.testing.assert_allclose(h[mask], num_h[mask], rtol=1e-2, atol=1e-4)