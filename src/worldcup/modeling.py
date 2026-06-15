"""Reusable modelling machinery for the forecasting experiments (Arms A & B).

Everything here is phase-agnostic: temporal cross-validation, ordinal metrics,
a model/HPO factory, probability calibration, the independent-Poisson scoreline,
and append-only prediction-log helpers. The three phase scripts in ``experiments/``
compose these pieces — this module never reads the raw data or picks a phase.

Hard rules baked in: temporal (never shuffled) CV, fixed seeds, deterministic
sorting, and a prediction log that is never overwritten.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import poisson
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42
CLASSES = ["H", "D", "A"]          # fixed ordinal order
_CLS_IDX = {c: i for i, c in enumerate(CLASSES)}

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # noqa: BLE001
    HAS_XGB = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # noqa: BLE001
    HAS_CATBOOST = False
import os

HAS_TABPFN = False
try:  # cloud TabPFN (no GPU needed); token cached by tabpfn_client.set_access_token
    import tabpfn_client as _tc
    _tok = os.environ.get("TABPFN_TOKEN") or os.environ.get("TABPFN_API_TOKEN")
    if _tok:
        _tc.set_access_token(_tok)
    from tabpfn_client import TabPFNClassifier
    HAS_TABPFN = bool(_tc.get_access_token())
except Exception:  # noqa: BLE001 — package missing or no token
    HAS_TABPFN = False

# columns that are never predictors
DROP_COLS = {
    "match_id", "date", "home_team", "away_team", "tournament", "stage",
    "country", "venue", "kickoff", "group", "match_date",
    "result", "home_score", "away_score", "total_goals", "goal_diff",
    "is_2026", "played", "home_elo_post", "away_elo_post",
    "team", "opponent", "gf",  # team_match grain
}


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def feature_columns(df: pl.DataFrame) -> list[str]:
    return [c for c, t in zip(df.columns, df.dtypes)
            if c not in DROP_COLS and (t.is_numeric() or t == pl.Boolean)]


def to_numpy(df: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return (df.select([pl.col(c).cast(pl.Float64) for c in cols])
            .to_numpy().astype(np.float64))


def encode_result(df: pl.DataFrame) -> np.ndarray:
    return df["result"].replace_strict(_CLS_IDX, default=None).to_numpy()


def select_top_k(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k features by mutual information (fit on train only).

    Useful for linear models and TabPFN (attention-based, feature-capped); trees
    do implicit selection so it matters less for them.
    """
    from sklearn.feature_selection import mutual_info_classif
    if X.shape[1] <= k:
        return np.arange(X.shape[1])
    mi = mutual_info_classif(np.nan_to_num(X), y, random_state=SEED)
    return np.sort(np.argsort(mi)[::-1][:k])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rps(probs: np.ndarray, y: np.ndarray) -> float:
    """Ranked Probability Score for ordinal classes (0 best, 1 worst)."""
    probs = np.asarray(probs, float)
    onehot = np.eye(probs.shape[1])[y]
    cp = np.cumsum(probs, axis=1)[:, :-1]
    co = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(((cp - co) ** 2).sum(axis=1) / (probs.shape[1] - 1)))


def log_loss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, float), 1e-12, 1)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(np.asarray(probs).shape[1])[y]
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == y))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-label expected calibration error."""
    conf = probs.max(axis=1)
    correct = (np.argmax(probs, axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += abs(correct[m].mean() - conf[m].mean()) * m.mean()
    return float(out)


# --------------------------------------------------------------------------- #
# temporal cross-validation (never shuffled)
# --------------------------------------------------------------------------- #
def walk_forward_folds(dates, n_folds: int = 4, min_train_frac: float = 0.5):
    """Expanding-window folds: train ``date <= cut``, validate the next slice.

    Cuts fall on date *values*, so every validation date is strictly after every
    training date in its fold (no boundary leakage).
    """
    d = np.asarray(dates)
    uniq = np.array(sorted(set(d.tolist())))
    pos = np.linspace(int(len(uniq) * min_train_frac), len(uniq), n_folds + 1)
    cuts = sorted(set(uniq[np.clip(pos.astype(int), 0, len(uniq) - 1)].tolist()))
    folds = []
    for i in range(len(cuts) - 1):
        tr = np.where(d <= cuts[i])[0]
        va = np.where((d > cuts[i]) & (d <= cuts[i + 1]))[0]
        if len(tr) and len(va):
            folds.append((tr, va))
    return folds


# --------------------------------------------------------------------------- #
# model factory (classifiers, Arm A) + Poisson regressors (Arm B)
# --------------------------------------------------------------------------- #
def _impute(*steps):
    # keep_empty_features: all-null columns (e.g. squad features on historical rows)
    # become a constant instead of erroring; feature count stays stable.
    return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                     *steps])


def build_classifier(name: str, params: dict | None = None, seed: int = SEED) -> Pipeline:
    p = dict(params or {})
    if name == "LogReg":
        return _impute(("scale", StandardScaler()),
                       ("m", LogisticRegression(max_iter=1000, C=p.get("C", 1.0), random_state=seed)))
    if name == "SVM":
        return _impute(("scale", StandardScaler()),
                       ("m", SVC(probability=True, C=p.get("C", 1.0),
                                 gamma=p.get("gamma", "scale"), random_state=seed)))
    if name == "RandomForest":
        return _impute(("m", RandomForestClassifier(
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", None),
            max_features=p.get("max_features", "sqrt"), random_state=seed, n_jobs=-1)))
    if name == "ExtraTrees":
        return _impute(("m", ExtraTreesClassifier(
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", None),
            max_features=p.get("max_features", "sqrt"), random_state=seed, n_jobs=-1)))
    if name == "XGBoost" and HAS_XGB:
        return _impute(("m", XGBClassifier(
            objective="multi:softprob", num_class=3, random_state=seed, n_jobs=-1,
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05), reg_lambda=p.get("reg_lambda", 1.0),
            eval_metric="mlogloss")))
    if name == "CatBoost" and HAS_CATBOOST:
        return _impute(("m", CatBoostClassifier(
            loss_function="MultiClass", random_seed=seed, verbose=False,
            iterations=p.get("iterations", 400), depth=p.get("depth", 5),
            learning_rate=p.get("learning_rate", 0.05), l2_leaf_reg=p.get("l2_leaf_reg", 3.0))))
    if name == "TabPFN" and HAS_TABPFN:
        return _impute(("m", TabPFNClassifier()))
    raise ValueError(f"unknown/unavailable classifier: {name}")


def available_classifiers() -> list[str]:
    out = ["LogReg", "SVM", "RandomForest", "ExtraTrees"]
    if HAS_XGB:
        out.append("XGBoost")
    if HAS_CATBOOST:
        out.append("CatBoost")
    if HAS_TABPFN:
        out.append("TabPFN")
    return out


def build_poisson_regressor(name: str, params: dict | None = None, seed: int = SEED):
    p = dict(params or {})
    if name == "PoissonGLM":
        return _impute(("scale", StandardScaler()),
                       ("m", PoissonRegressor(alpha=p.get("alpha", 1.0), max_iter=500)))
    if name == "XGBoostPoisson" and HAS_XGB:
        return _impute(("m", XGBRegressor(
            objective="count:poisson", random_state=seed, n_jobs=-1,
            n_estimators=p.get("n_estimators", 400), max_depth=p.get("max_depth", 4),
            learning_rate=p.get("learning_rate", 0.05))))
    if name == "CatBoostPoisson" and HAS_CATBOOST:
        return _impute(("m", CatBoostRegressor(
            loss_function="Poisson", random_seed=seed, verbose=False,
            iterations=p.get("iterations", 400), depth=p.get("depth", 5),
            learning_rate=p.get("learning_rate", 0.05))))
    if name == "GrollRF":  # Groll et al. (2019) "forest half": RF regression on goals
        return _impute(("m", RandomForestRegressor(
            n_estimators=p.get("n_estimators", 500), max_depth=p.get("max_depth", None),
            max_features=p.get("max_features", "sqrt"), random_state=seed, n_jobs=-1)))
    raise ValueError(f"unknown/unavailable regressor: {name}")


def proba_hda(pipe: Pipeline, X: np.ndarray) -> np.ndarray:
    """predict_proba reordered to fixed [H, D, A] columns."""
    p = np.asarray(pipe.predict_proba(X))
    classes = list(pipe.classes_)
    order = [classes.index(i) for i in range(3)]
    return p[:, order]


# --------------------------------------------------------------------------- #
# Optuna HPO (temporal, minimise RPS) — never shuffled
# --------------------------------------------------------------------------- #
def _suggest(trial, name):
    if name == "LogReg":
        return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True)}
    if name == "SVM":
        return {"C": trial.suggest_float("C", 1e-1, 1e2, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])}
    if name in ("RandomForest", "ExtraTrees"):
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=200),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5])}
    if name == "XGBoost":
        return {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=200),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True)}
    if name == "CatBoost":
        return {"iterations": trial.suggest_int("iterations", 200, 800, step=200),
                "depth": trial.suggest_int("depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True)}
    return {}


def tune_classifier(name, X, y, dates, n_trials=20, n_folds=3, seed=SEED):
    """Optuna search inside walk-forward CV; returns best params (min mean RPS)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = walk_forward_folds(dates, n_folds=n_folds)
    if not folds or name == "TabPFN":
        return {}

    def objective(trial):
        params = _suggest(trial, name)
        scores = []
        for tr, va in folds:
            pipe = build_classifier(name, params, seed).fit(X[tr], y[tr])
            scores.append(rps(proba_hda(pipe, X[va]), y[va]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# --------------------------------------------------------------------------- #
# calibration (fit on a temporal validation slice)
# --------------------------------------------------------------------------- #
def fit_calibrators(probs_val: np.ndarray, y_val: np.ndarray):
    return [IsotonicRegression(out_of_bounds="clip").fit(probs_val[:, k], (y_val == k).astype(float))
            for k in range(probs_val.shape[1])]


def apply_calibrators(probs: np.ndarray, cals) -> np.ndarray:
    out = np.column_stack([cals[k].predict(probs[:, k]) for k in range(len(cals))])
    out = np.clip(out, 1e-9, None)
    return out / out.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# scoreline from two Poisson lambdas (independent / conditional)
# --------------------------------------------------------------------------- #
def poisson_joint(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, max(lam_home, 1e-6)), poisson.pmf(g, max(lam_away, 1e-6)))
    return m / m.sum()


def scoreline_readoffs(m: np.ndarray) -> dict:
    n = m.shape[0]
    totals = np.zeros(2 * n - 1)
    for i in range(n):
        for j in range(n):
            totals[i + j] += m[i, j]
    i, j = np.unravel_index(int(np.argmax(m)), m.shape)
    g = np.arange(n)
    return {
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "exp_home": float((g[:, None] * m).sum()),
        "exp_away": float((g[None, :] * m).sum()),
        "p_over25": float(totals[3:].sum()),
        "ml_home": int(i), "ml_away": int(j),
    }


def hda_from_joint(m: np.ndarray) -> np.ndarray:
    return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])


# --------------------------------------------------------------------------- #
# append-only prediction log
# --------------------------------------------------------------------------- #
def append_prediction_log(path, new: pl.DataFrame, keys=("match_id", "run_date")) -> pl.DataFrame:
    """Append rows whose (match_id, run_date) are new; never overwrite existing ones."""
    keys = list(keys)
    if path.exists():
        old = pl.read_parquet(path)
        add = new.join(old.select(keys), on=keys, how="anti")
        out = pl.concat([old, add], how="diagonal_relaxed")
    else:
        out = new
    out.write_parquet(path)
    return out
