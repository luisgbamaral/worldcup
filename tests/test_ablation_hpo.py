"""Tests for the per-window HPO in the training-size ablation.

Covered: inner-CV is leakage-safe; foundation models are never tuned (zero cost); the
selected hyper-parameters actually vary with window size (guards a no-op); the degenerate-CV
fallback triggers on a tiny window; and the survival table is well-formed.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import train_size_ablation as A  # noqa: E402
from worldcup import modeling as M  # noqa: E402


def _synth(n, p=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logits = X[:, 0] * 1.2 - X[:, 1] * 0.7
    y = np.where(logits > 0.4, 0, np.where(logits < -0.4, 2, 1))
    dates = np.array([dt.date(2015, 1, 1) + dt.timedelta(days=int(i)) for i in range(n)])
    return X, y, dates


def test_inner_folds_are_temporally_clean():
    _, _, dates = _synth(400)
    folds, ok = A._inner_folds(dates)
    assert ok and len(folds) >= 2
    for tr, va in folds:                              # every val date strictly after all train dates
        assert dates[tr].max() < dates[va].min()


def test_foundation_models_are_disjoint_from_tuned_classical():
    assert set(A.CLASSICAL).isdisjoint(set(A._fms()))
    assert all(m in A.STRONG_REG for m in A.CLASSICAL)


def test_foundation_model_branch_records_zero_tuning(monkeypatch):
    """The FM branch of _eval_cell must fit once and record zero tuning cost."""
    from sklearn.dummy import DummyClassifier
    from sklearn.pipeline import Pipeline

    def fake_build(name, params=None, seed=0):
        return Pipeline([("m", DummyClassifier(strategy="prior"))])

    monkeypatch.setattr(A.M, "build_classifier", fake_build)
    monkeypatch.setattr(A, "CLASSICAL", [])           # isolate the FM branch
    monkeypatch.setattr(A, "_fms", lambda: ["FakeFM"])
    Xtr, ytr, _ = _synth(120)
    Xte, yte, _ = _synth(60, seed=1)
    cell = A._eval_cell(Xtr, ytr, _synth(120)[2], Xte, yte)
    assert cell["FakeFM"]["tuning_fits"] == 0
    assert cell["FakeFM"]["params"] == "{}"
    assert len(cell["FakeFM"]["pm"]) == len(yte)


def test_degenerate_tiny_window_falls_back():
    X, y, dates = _synth(20)                           # below MIN_FOLD -> degenerate
    params, degen, nfit = A._hpo("LogReg", X, y, dates)
    assert degen is True and nfit == 0
    assert params == A.STRONG_REG["LogReg"]


def test_hpo_selects_valid_and_varying_hyperparameters():
    # non-degenerate windows of different sizes/noise -> valid grid choices, and not all identical
    chosen = []
    for n, seed in [(300, 1), (1500, 2), (4000, 3)]:
        X, y, dates = _synth(n, seed=seed)
        params, degen, nfit = A._hpo("LogReg", X, y, dates)
        assert not degen and nfit > 0
        assert params["C"] in A.LOGREG_C_GRID and params["penalty"] in A.LOGREG_PENALTY
        chosen.append(params["C"])
    assert len(set(chosen)) > 1                         # guards a no-op (HPO actually varies)


def test_survival_table_is_well_formed():
    windows = [w for w, _ in A.WINDOWS]
    n = 200
    rng = np.random.default_rng(0)
    rows, store = [], {}
    clusters = rng.integers(2010, 2014, size=n)         # pretend "years"
    for w in windows:
        for model, base in [("TabPFN", 0.18), ("LogReg", 0.19), ("CatBoost", 0.20)]:
            rows.append({"scope": "onestep", "features": "best", "window": w, "days": 1,
                         "model": model, "rps": base, "rps_lo": base, "rps_hi": base,
                         "hpo_degenerate": False, "tuning_fits": 0})
            store[("best", w, model)] = np.full(n, base) + rng.normal(0, 0.01, n)
    table = pl.DataFrame(rows)
    surv = A._survival(table, store, clusters, "onestep")
    assert surv.height == len(windows)                  # one row per window
    for col in ("best_fm", "best_classical", "delta_fm_minus_cl", "CI95", "p_fm_better"):
        assert col in surv.columns and surv[col].null_count() == 0
