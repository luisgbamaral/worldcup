"""Training-size ablation with PER-WINDOW hyperparameter tuning.

For growing train windows (1mo / 6mo / 1y / 2y / 5y / 8y) and two feature sets
(``few`` = top-8 greedy, ``best`` = 107) we re-fit every model and measure RPS, in two
scopes (one-step on 2023+, multi-step per Cup). The classical comparators (LogReg, CatBoost,
XGBoost) are **re-tuned at every (window, feature_set[, Cup]) cell** with an inner *temporal*
cross-validation strictly inside the training window; the foundation models (TabPFN, TabICL)
are left **tune-free** -- that configuration-free property is exactly what is under test.

We then ask whether the foundation-model advantage **survives** a fully-tuned classical
baseline (``ablation_survival``), record the tuning-cost asymmetry, and put 95% CIs on every
cell (>=5 seeds for one-step; per-Cup spread for multi-step).

Leakage rule: HPO and fitting see only rows strictly inside the training window; the test set
(and the one-step 2022 calibration slice) are never touched by the search. Temporal CV only.

    python experiments/train_size_ablation.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldcup import config, elo as E, features, modeling as M, viz  # noqa: E402
import worldcup_multistep as MS  # noqa: E402

WINDOWS = [("1mo", 30), ("6mo", 182), ("1y", 365), ("2y", 730), ("5y", 1826), ("8y", 2922)]
CLASSICAL = ["LogReg", "CatBoost", "XGBoost"]
SEEDS = [42, 43, 44, 45, 46]            # >=5 seeds for variability bands
HPO_SEED = 42
OPTUNA_TRIALS = 10                       # identical budget across windows (GBDTs)
INNER_FOLDS = 3
MIN_FOLD = 25                            # below this, inner CV is degenerate -> fallback
FEW_K = 8
LOGREG_C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]
LOGREG_PENALTY = ["l2", "l1"]
# strong-regularization fallbacks for the degenerate (tiny-window) corner
STRONG_REG = {
    "LogReg": {"C": 0.01, "penalty": "l2", "max_iter": 2000},
    "CatBoost": {"depth": 3, "l2_leaf_reg": 10.0, "iterations": 200, "learning_rate": 0.05},
    "XGBoost": {"max_depth": 2, "reg_lambda": 10.0, "n_estimators": 200, "learning_rate": 0.05,
                "min_child_weight": 5, "subsample": 0.8}}
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _fms():
    return [m for m in ("TabPFN", "TabICL") if m in M.available_classifiers()]


def _feature_sets():
    opt = json.loads((config.PROCESSED / "optimal_features.json").read_text("utf-8"))
    return {"few": opt["order"][:FEW_K], "best": opt["features"]}


def _rps_pm(p, y):
    """Per-match RPS (mean of this == modeling.rps)."""
    K = p.shape[1]
    Y = np.eye(K)[y]
    return (((np.cumsum(p, 1) - np.cumsum(Y, 1)) ** 2).sum(1) / (K - 1))


def _ci(vals):
    """95% t-interval (two-sided) of a small sample of cell means."""
    v = np.asarray(vals, float)
    if len(v) < 2 or np.allclose(v, v[0]):
        return float(v.mean()), float(v.mean()), float(v.mean())
    from scipy import stats
    m, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
    h = float(stats.t.ppf(0.975, len(v) - 1) * se)
    return m, m - h, m + h


# --------------------------------------------------------------------------- #
# per-window HPO (classical only); foundation models are NOT tuned
# --------------------------------------------------------------------------- #
def _inner_folds(dates):
    folds = M.walk_forward_folds(np.asarray(dates), n_folds=INNER_FOLDS)
    ok = (len(folds) >= 2 and
          min(min(len(tr), len(va)) for tr, va in folds) >= MIN_FOLD)
    return folds, ok


def _suggest_full(trial, name):
    if name == "XGBoost":
        return {"n_estimators": trial.suggest_categorical("n_estimators", [200, 400, 600, 800]),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)}
    return {"iterations": trial.suggest_categorical("iterations", [200, 400, 600, 800]),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True)}


def _hpo(name, X, y, dates):
    """Return (best_params, hpo_degenerate, n_fits). Inner temporal CV inside the window."""
    folds, ok = _inner_folds(dates)
    if not ok:
        return dict(STRONG_REG[name]), True, 0
    if name == "LogReg":
        best, n = None, 0
        for pen in LOGREG_PENALTY:
            for C in LOGREG_C_GRID:
                pr = {"C": C, "penalty": pen, "max_iter": 2000}
                sc = []
                for tr, va in folds:
                    pipe = M.build_classifier("LogReg", pr, HPO_SEED).fit(X[tr], y[tr]); n += 1
                    sc.append(M.rps(M.proba_hda(pipe, X[va]), y[va]))
                m = float(np.mean(sc))
                if best is None or m < best[0]:
                    best = (m, pr)
        return best[1], False, n
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n = [0]

    def objective(trial):
        pr = _suggest_full(trial, name)
        sc = []
        for tr, va in folds:
            pipe = M.build_classifier(name, pr, HPO_SEED).fit(X[tr], y[tr]); n[0] += 1
            sc.append(M.rps(M.proba_hda(pipe, X[va]), y[va]))
        return float(np.mean(sc))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=HPO_SEED))
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    return study.best_params, False, n[0]


# --------------------------------------------------------------------------- #
# evaluate one cell: tune classical once, fit each model, return per-match RPS
# --------------------------------------------------------------------------- #
def _eval_cell(Xtr, ytr, dtr, Xte, yte, Xva=None, yva=None):
    """Returns dict model -> {pm: per-match RPS (seed-ensemble), seed_means, n_train,
    hpo_degenerate, params, tuning_fits}. Calibrates iff a validation slice is given."""
    out = {}
    for name in CLASSICAL:
        params, degen, nfit = _hpo(name, Xtr, ytr, dtr)
        seed_probs, seed_means = [], []
        for s in SEEDS:
            pipe = M.build_classifier(name, params, s).fit(Xtr, ytr)
            p = M.proba_hda(pipe, Xte)
            if Xva is not None:
                cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
                p = M.apply_calibrators(p, cals)
            seed_probs.append(p)
            seed_means.append(float(_rps_pm(p, yte).mean()))
        pm = _rps_pm(np.mean(seed_probs, axis=0), yte)
        out[name] = {"pm": pm, "seed_means": seed_means, "n_train": len(ytr),
                     "hpo_degenerate": degen, "params": json.dumps(params), "tuning_fits": nfit}
    for name in _fms():                                  # tune-free: one fit, zero tuning cost
        Xf, yf = Xtr, ytr
        if len(ytr) > 10000:
            sub = np.random.RandomState(HPO_SEED).choice(len(ytr), 10000, replace=False)
            Xf, yf = Xtr[sub], ytr[sub]
        pipe = M.build_classifier(name).fit(Xf, yf)
        p = M.proba_hda(pipe, Xte)
        if Xva is not None:
            cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
            p = M.apply_calibrators(p, cals)
        pm = _rps_pm(p, yte)
        out[name] = {"pm": pm, "seed_means": [float(pm.mean())], "n_train": len(yf),
                     "hpo_degenerate": False, "params": "{}", "tuning_fits": 0}
    return out


# --------------------------------------------------------------------------- #
# one-step: fixed test 2023+, fixed calibration 2022, growing train window
# --------------------------------------------------------------------------- #
def onestep_ablation(mf):
    cut = dt.date(2022, 1, 1)
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1)).sort("date")
    yva, yte = _y(va), _y(te)
    clusters = te["date"].dt.year().to_numpy()
    rows, store = [], {}
    for fs, cols in _feature_sets().items():
        med = np.nan_to_num(np.nanmedian(M.to_numpy(mf, cols), axis=0), nan=0.0)
        fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
        Xva, Xte = fill(va), fill(te)
        for wname, days in WINDOWS:
            tr = mf.filter((pl.col("date") >= cut - dt.timedelta(days=days)) & (pl.col("date") < cut)).sort("date")
            if tr.height < MIN_FOLD:
                continue
            cell = _eval_cell(fill(tr), _y(tr), tr["date"].to_numpy(), Xte, yte, Xva, yva)
            for name, r in cell.items():
                m, lo, hi = _ci(r["seed_means"])
                rows.append({"scope": "onestep", "features": fs, "window": wname, "days": days,
                             "model": name, "n_train": r["n_train"], "rps": m, "rps_lo": lo, "rps_hi": hi,
                             "hpo_degenerate": r["hpo_degenerate"], "tuning_fits": r["tuning_fits"],
                             "params": r["params"]})
                store[(fs, wname, name)] = r["pm"]
                print(f"  [1step/{fs}/{wname}] {name:9} n={r['n_train']:5} RPS={m:.4f} "
                      f"[{lo:.4f},{hi:.4f}] fits={r['tuning_fits']} degen={r['hpo_degenerate']}", flush=True)
    return pl.DataFrame(rows), store, (yte, clusters)


# --------------------------------------------------------------------------- #
# multi-step: per Cup, growing pre-kickoff window -> Cup-match RPS (no podium sim)
# --------------------------------------------------------------------------- #
def multistep_ablation(mf):
    rows, store, cl_store, y_store = [], {}, {}, {}
    for fs, cols in _feature_sets().items():
        med = np.nan_to_num(np.nanmedian(M.to_numpy(mf, cols), axis=0), nan=0.0)
        fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
        cups = {}
        for year in MS.WORLD_CUPS:
            cup = mf.filter((pl.col("tournament") == "FIFA World Cup")
                            & (pl.col("date").dt.year() == year)).sort("date")
            cups[year] = (cup, cup["date"].min())
        for wname, days in WINDOWS:
            pm, cupmean, ntr, degen, fits = {}, {}, {}, {}, {}
            cl_concat, y_concat = [], []
            for year, (cup, start) in cups.items():
                tr = mf.filter((pl.col("date") >= start - dt.timedelta(days=days))
                               & (pl.col("date") < start)).sort("date")
                if tr.height < MIN_FOLD or cup.is_empty():
                    continue
                ycup = _y(cup)
                cell = _eval_cell(fill(tr), _y(tr), tr["date"].to_numpy(), fill(cup), ycup)
                for name, r in cell.items():
                    pm.setdefault(name, []).append(r["pm"])
                    cupmean.setdefault(name, []).append(float(r["pm"].mean()))
                    ntr.setdefault(name, []).append(r["n_train"])
                    degen[name] = degen.get(name, False) or r["hpo_degenerate"]
                    fits[name] = fits.get(name, 0) + r["tuning_fits"]
                    store.setdefault((fs, wname, name), []).append(r["pm"])
                cl_concat.append(np.full(cup.height, year)); y_concat.append(ycup)
            if not pm:
                continue
            cl_store[(fs, wname)] = np.concatenate(cl_concat)
            y_store[(fs, wname)] = np.concatenate(y_concat)
            for name in pm:
                m, lo, hi = _ci(cupmean[name])
                rows.append({"scope": "multistep", "features": fs, "window": wname, "days": days,
                             "model": name, "n_train": int(np.mean(ntr[name])), "rps": m,
                             "rps_lo": lo, "rps_hi": hi, "hpo_degenerate": degen[name],
                             "tuning_fits": fits[name]})
                print(f"  [multi/{fs}/{wname}] {name:9} n~{int(np.mean(ntr[name])):5} RPS={m:.4f} "
                      f"[{lo:.4f},{hi:.4f}] fits={fits[name]} degen={degen[name]} "
                      f"({len(cupmean[name])} Cups)", flush=True)
    store = {k: np.concatenate(v) for k, v in store.items()}
    return pl.DataFrame(rows), store, cl_store, y_store


# --------------------------------------------------------------------------- #
# survival: does the best FM beat the best tuned classical, after per-window HPO?
# --------------------------------------------------------------------------- #
def _survival(table, store, clusters_by_cell, scope):
    rng = np.random.default_rng(HPO_SEED)
    rows = []
    fcol = table.filter(pl.col("scope") == scope)
    for fs in ("few", "best"):
        for wname, days in WINDOWS:
            sub = fcol.filter((pl.col("features") == fs) & (pl.col("window") == wname))
            if sub.is_empty():
                continue
            fm = sub.filter(pl.col("model").is_in(_fms())).sort("rps")
            cl = sub.filter(pl.col("model").is_in(CLASSICAL)).sort("rps")
            if fm.is_empty() or cl.is_empty():
                continue
            best_fm, best_cl = fm["model"][0], cl["model"][0]
            pm_fm, pm_cl = store.get((fs, wname, best_fm)), store.get((fs, wname, best_cl))
            if pm_fm is None or pm_cl is None:
                continue
            d = pm_cl - pm_fm                       # >0 => FM better
            cl_ids = clusters_by_cell[(fs, wname)] if scope == "multistep" else clusters_by_cell
            uniq = np.unique(cl_ids)
            boot = np.array([np.concatenate([d[cl_ids == rng.choice(uniq)] for _ in uniq]).mean()
                             for _ in range(2000)])
            degen = bool(sub.filter(pl.col("model") == best_cl)["hpo_degenerate"][0]) \
                if "hpo_degenerate" in sub.columns else False
            rows.append({"scope": scope, "features": fs, "window": wname, "days": days,
                         "best_fm": best_fm, "rps_fm": round(float(fm["rps"][0]), 4),
                         "best_classical": best_cl, "rps_classical": round(float(cl["rps"][0]), 4),
                         "delta_fm_minus_cl": round(float(d.mean()), 4),
                         "CI95": f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]",
                         "p_fm_better": round(float(np.mean(boot <= 0)), 4),
                         "hpo_degenerate": degen})
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# charts with error bands (FMs vs best-tuned classical curves)
# --------------------------------------------------------------------------- #
def _curve_chart(df, title, fname):
    viz.set_style()
    import matplotlib.pyplot as plt
    order = [w for w, _ in WINDOWS]
    fig, axes = plt.subplots(1, 2, figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.45), sharey=True)
    for ax, fs in zip(axes, ("few", "best")):
        sub = df.filter(pl.col("features") == fs)
        for i, name in enumerate(sorted(sub["model"].unique().to_list())):
            s = sub.filter(pl.col("model") == name)
            s = s.with_columns(pl.col("window").cast(pl.Enum(order))).sort("window")
            x = [order.index(w) for w in s["window"]]
            ax.plot(x, s["rps"], label=name, **viz.style_for(i))
            ax.fill_between(x, s["rps_lo"], s["rps_hi"], alpha=0.15, lw=0)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, fontsize=6)
        ax.set_title(f"{fs} features", fontsize=7); ax.set_xlabel("train window")
    axes[0].set_ylabel("test RPS (mean ± 95% CI)")
    axes[1].legend(ncol=2, fontsize=5)
    fig.suptitle(title, fontsize=8)
    viz.save_fig(fig, fname)


def _tex(df, name, caption, label):
    viz.save_table(viz.df_to_neurips_latex(df, label=label, float_format="%.4f", caption=caption), name)


def main():
    t0 = time.time()
    mf = (features.build_match_features_cached(train_cut=dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    print(f"config: seeds={SEEDS} optuna_trials={OPTUNA_TRIALS} inner_folds={INNER_FOLDS} "
          f"logreg_grid={len(LOGREG_C_GRID)}x{len(LOGREG_PENALTY)}", flush=True)

    print("=== ONE-STEP ablation (per-window HPO) ===", flush=True)
    one, one_store, (one_y, one_cl) = onestep_ablation(mf)
    one.write_parquet(config.PROCESSED / "ablation_onestep.parquet")
    one.sort("features", "model", "days").write_csv(config.FIGURES / "ablation_onestep.csv")
    _curve_chart(one, "Train-size ablation — one-step (tuned classical vs tune-free FM)",
                 "15_trainsize_onestep")
    _tex(one.with_columns(pl.col("rps").round(4)).select(
        "features", "window", "model", "n_train", "rps", "rps_lo", "rps_hi",
        "tuning_fits", "hpo_degenerate"),
        "ablation_onestep", "One-step training-size ablation with per-window HPO for the classical "
        "models (foundation models tune-free). RPS mean with 95\\% CI over 5 seeds; tuning\\_fits is "
        "the inner-CV fit count (0 for foundation models).", "tab:ablation_onestep")

    print("\n=== MULTI-STEP ablation (per-window-per-Cup HPO, Cup-match RPS) ===", flush=True)
    multi, multi_store, multi_cl, multi_yy = multistep_ablation(mf)
    multi.write_parquet(config.PROCESSED / "ablation_multistep.parquet")
    multi.sort("features", "model", "days").write_csv(config.FIGURES / "ablation_multistep.csv")
    _curve_chart(multi, "Train-size ablation — multi-step Cup-match RPS (tuned classical vs FM)",
                 "17_trainsize_multistep_rps")
    _tex(multi.with_columns(pl.col("rps").round(4)).select(
        "features", "window", "model", "n_train", "rps", "rps_lo", "rps_hi",
        "tuning_fits", "hpo_degenerate"),
        "ablation_multistep", "Multi-step training-size ablation: mean Cup-match RPS per window with "
        "95\\% CI across the five World Cups; classical models re-tuned per window-per-Cup, foundation "
        "models tune-free.", "tab:ablation_multistep")

    print("\n=== SURVIVAL: best FM vs best tuned classical ===", flush=True)
    surv1 = _survival(one, one_store, one_cl, "onestep")
    surv2 = _survival(multi, multi_store, multi_cl, "multistep")
    surv = pl.concat([surv1, surv2], how="diagonal_relaxed")
    surv.write_parquet(config.PROCESSED / "ablation_survival.parquet")
    surv.write_csv(config.FIGURES / "ablation_survival.csv")
    _tex(surv, "ablation_survival",
         "Does the foundation-model advantage survive a per-window-tuned classical baseline? "
         "Paired per-match RPS difference (classical $-$ FM; positive favours the FM), clustered "
         "block bootstrap (by year for one-step, by Cup for multi-step). p\\_fm\\_better is the "
         "one-sided bootstrap probability the FM is not better.", "tab:ablation_survival")
    print(surv)
    print(f"\nsaved: ablation_{{onestep,multistep,survival}}.{{parquet,csv,tex}}, "
          f"15/17 charts | wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
