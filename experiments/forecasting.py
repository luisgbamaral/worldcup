"""Forecasting experiments in three phases (benchmark / tournament test / production).

    python experiments/forecasting.py --phase 1 [--trials N]
    python experiments/forecasting.py --phase 2
    python experiments/forecasting.py --phase 3

Phase 1 — academic benchmark on pre-World-Cup data only (temporal CV + held-out
test), every model vs. the Elo / Poisson-GLM / Groll baselines.
Phase 2 — predict the 2026 games already played, as of each game's own date
(out-of-sample), scorecard on winners and goals.
Phase 3 — daily: retrain on everything known now (incl. played WC games) and
predict the scheduled fixtures; freeze predictions in an append-only log.

Composes :mod:`worldcup.modeling`; never modifies the feature base.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from worldcup import config, modeling as M, viz  # noqa: E402

PRE_WC = dt.date(2026, 6, 10)
ARM_A = ["LogReg", "RandomForest", "ExtraTrees", "XGBoost", "CatBoost"]
ARM_B = ["PoissonGLM", "XGBoostPoisson", "CatBoostPoisson"]
BENCH = ["Elo", "GrollRF"]            # trivial Elo + Groll SOTA (PoissonGLM already in ARM_B)
SEED = M.SEED


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _load():
    # match_id must be a unique join key; a few fixtures are duplicated in the raw
    # martj42 source (e.g. Gibraltar v Cayman Islands 2026-06-06 ×16), which would
    # fan out the goals/metrics joins. Drop exact duplicates here (experiment layer).
    mf = (pl.read_parquet(config.MATCH_FEATURES).sort("date", "match_id")
          .unique(subset="match_id", keep="first", maintain_order=True))
    tmf = (pl.read_parquet(config.TEAM_MATCH_FEATURES).sort("date", "match_id", "is_home")
           .unique(subset=["match_id", "is_home"], keep="first", maintain_order=True))
    return mf, tmf


def _tmf_for(tmf, match_ids):
    return tmf.filter(pl.col("match_id").is_in(list(match_ids)))


# --------------------------------------------------------------------------- #
# predictors — every model returns a frame keyed by match_id with HDA (+goals)
# --------------------------------------------------------------------------- #
def _predict_classifier(name, params, train_mf, pred_mf):
    cols = M.feature_columns(train_mf)
    pipe = M.build_classifier(name, params, SEED).fit(
        M.to_numpy(train_mf, cols), M.encode_result(train_mf))
    p = M.proba_hda(pipe, M.to_numpy(pred_mf, cols))
    return pred_mf.select("match_id").with_columns(
        p_home=p[:, 0], p_draw=p[:, 1], p_away=p[:, 2])


def _elo(train_mf, pred_mf):
    from sklearn.linear_model import LogisticRegression
    Xtr = train_mf.select("elo_diff_eff").to_numpy()
    clf = LogisticRegression(max_iter=1000).fit(Xtr, M.encode_result(train_mf))
    p = clf.predict_proba(pred_mf.select("elo_diff_eff").to_numpy())
    order = [list(clf.classes_).index(i) for i in range(3)]
    p = p[:, order]
    return pred_mf.select("match_id").with_columns(p_home=p[:, 0], p_draw=p[:, 1], p_away=p[:, 2])


def _assemble(pred_tmf, lam):
    """Independent-Poisson scoreline per match from per-team λ."""
    lf = pred_tmf.select("match_id", "is_home").with_columns(lam=np.clip(lam, 0.05, None))
    j = (lf.filter(pl.col("is_home")).select("match_id", lam_home="lam")
         .join(lf.filter(~pl.col("is_home")).select("match_id", lam_away="lam"), on="match_id"))
    rows = []
    for r in j.iter_rows(named=True):
        ro = M.scoreline_readoffs(M.poisson_joint(r["lam_home"], r["lam_away"]))
        rows.append({"match_id": r["match_id"], **{k: ro[k] for k in
                     ("p_home", "p_draw", "p_away", "exp_home", "exp_away", "p_over25",
                      "ml_home", "ml_away")}})
    return pl.DataFrame(rows)


def _predict_goals(name, params, train_tmf, pred_tmf):
    cols = M.feature_columns(train_tmf)
    reg = M.build_poisson_regressor(name, params, SEED).fit(
        M.to_numpy(train_tmf, cols), train_tmf["gf"].to_numpy().astype(float))
    return _assemble(pred_tmf, reg.predict(M.to_numpy(pred_tmf, cols)))


def _struct_resid(train_tmf, pred_tmf):
    """Pure Zhang residual hybrid: Poisson-GLM structural base as a log offset,
    XGBoost(Poisson) learns the residual (native ``base_margin``)."""
    import xgboost as xgb
    cols = M.feature_columns(train_tmf)
    Xtr, Xpr = M.to_numpy(train_tmf, cols), M.to_numpy(pred_tmf, cols)
    med = np.nanmedian(Xtr, axis=0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr)
    Xpr = np.where(np.isnan(Xpr), med, Xpr)
    base_tr = M.build_poisson_regressor("PoissonGLM", {}, SEED).fit(
        Xtr, train_tmf["gf"].to_numpy().astype(float))
    btr = np.log(np.clip(base_tr.predict(Xtr), 0.05, None))
    bpr = np.log(np.clip(base_tr.predict(Xpr), 0.05, None))
    dtr = xgb.DMatrix(Xtr, label=train_tmf["gf"].to_numpy().astype(float), base_margin=btr)
    dpr = xgb.DMatrix(Xpr, base_margin=bpr)
    bst = xgb.train({"objective": "count:poisson", "max_depth": 4, "eta": 0.05, "seed": SEED},
                    dtr, num_boost_round=400)
    return _assemble(pred_tmf, bst.predict(dpr))


def predict(kind, name, params, train_mf, train_tmf, pred_mf, pred_tmf):
    if kind == "A":
        return _predict_classifier(name, params, train_mf, pred_mf)
    if name == "Elo":
        return _elo(train_mf, pred_mf)
    if name == "StructResid":
        return _struct_resid(train_tmf, pred_tmf)
    return _predict_goals(name, params, train_tmf, pred_tmf)


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def _metrics(pred, actual_mf):
    a = actual_mf.select("match_id", "result", "total_goals").join(pred, on="match_id")
    y = M.encode_result(a)
    probs = a.select("p_home", "p_draw", "p_away").to_numpy()
    out = {"RPS": M.rps(probs, y), "log_loss": M.log_loss(probs, y),
           "Brier": M.brier(probs, y), "accuracy": M.accuracy(probs, y), "ECE": M.ece(probs, y)}
    return out


def _kind(name):
    return "A" if name in ARM_A else "B"


def _roster(arm_a, arm_b):
    models = []
    for n in ARM_A:
        if arm_a and n in M.available_classifiers():
            models.append(("A", n))
    if arm_b:
        for n in ARM_B + BENCH + ["StructResid"]:
            models.append(("B", n))
    return models


# --------------------------------------------------------------------------- #
# Phase 1 — benchmark
# --------------------------------------------------------------------------- #
def phase1(trials=0):
    mf, tmf = _load()
    data = mf.filter((~pl.col("is_2026")) & (pl.col("date") <= PRE_WC))
    dates = data["date"].to_numpy()
    folds = M.walk_forward_folds(dates, n_folds=4)
    test_cut = sorted(set(dates.tolist()))[int(len(set(dates.tolist())) * 0.85)]
    is_test = data["date"].to_numpy() > test_cut

    models = [("A", n) for n in ARM_A if n in M.available_classifiers()] \
        + [("B", n) for n in ["Elo", "PoissonGLM", "GrollRF", "XGBoostPoisson", "CatBoostPoisson", "StructResid"]]

    tuned = {}
    if trials:
        Xtr, ytr = M.to_numpy(data, M.feature_columns(data)), M.encode_result(data)
        for kind, n in models:
            if kind == "A":                      # tune every Arm A classifier with a search space
                tuned[n] = M.tune_classifier(n, Xtr, ytr, dates, n_trials=trials, n_folds=3)
                print(f"[tuned] {n}: {tuned[n]}")

    fold_rows, summary = [], []
    for kind, n in models:
        rps_folds = []
        for fi, (tr, va) in enumerate(folds):
            tr_mf, va_mf = data[tr], data[va]
            pred = predict(kind, n, tuned.get(n, {}), tr_mf, _tmf_for(tmf, tr_mf["match_id"]),
                           va_mf, _tmf_for(tmf, va_mf["match_id"]))
            m = _metrics(pred, va_mf)
            m |= {"model": n, "fold": fi}
            fold_rows.append(m)
            rps_folds.append(m["RPS"])
        # held-out test (train on everything up to test_cut)
        tr_mf = data.filter(~pl.Series(is_test))
        te_mf = data.filter(pl.Series(is_test))
        pred = predict(kind, n, tuned.get(n, {}), tr_mf, _tmf_for(tmf, tr_mf["match_id"]),
                       te_mf, _tmf_for(tmf, te_mf["match_id"]))
        mt = _metrics(pred, te_mf)
        summary.append({"model": n, "arm": kind,
                        "RPS_cv_mean": float(np.mean(rps_folds)),
                        "RPS_cv_std": float(np.std(rps_folds)),
                        "RPS_test": mt["RPS"], "logloss_test": mt["log_loss"],
                        "Brier_test": mt["Brier"], "acc_test": mt["accuracy"],
                        "ECE_test": mt["ECE"]})

    res = pl.DataFrame(summary).sort("RPS_test")
    pl.DataFrame(fold_rows).write_parquet(config.PROCESSED / "benchmark_results.parquet")
    res.write_parquet(config.PROCESSED / "benchmark_summary.parquet")
    _report_phase1(res)
    return res


def _report_phase1(res):
    show = res.select(
        pl.col("model"), pl.col("arm"),
        (pl.col("RPS_cv_mean").round(4).cast(str) + " ± " + pl.col("RPS_cv_std").round(4).cast(str)).alias("RPS (CV)"),
        pl.col("RPS_test").round(4), pl.col("logloss_test").round(4),
        pl.col("acc_test").round(3), pl.col("ECE_test").round(4))
    tex = viz.df_to_neurips_latex(
        show, label="tab:benchmark", float_format="%.4f", bold_best="RPS_test",
        lower_is_better=True,
        caption=("Pre-World-Cup benchmark (1X2): walk-forward RPS (mean ± std) and held-out "
                 "test metrics. RPS / log-loss: lower is better. The headline question is "
                 "whether the boosting/hybrid models beat the Elo and Groll baselines."))
    viz.save_table(tex, "benchmark")
    viz.set_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.6))
    r = res.sort("RPS_test")
    ax.barh(r["model"], r["RPS_test"], color=viz.style_for(0)["color"])
    ax.set(xlabel="held-out test RPS (lower is better)", title="1X2 benchmark by model")
    ax.invert_yaxis()
    viz.save_fig(fig, "10_benchmark_rps")
    print(show)


# --------------------------------------------------------------------------- #
# Phase 2 — tournament test (played WC games, as of each game's date)
# --------------------------------------------------------------------------- #
def _selected_models(mf):
    """Best model from the Phase-1 summary (if present) + Groll + the residual hybrid."""
    best = ("A", "CatBoost" if "CatBoost" in M.available_classifiers() else "XGBoost")
    path = config.PROCESSED / "benchmark_summary.parquet"
    if path.exists():  # best *learner* (exclude the trivial Elo / GLM baselines)
        top = (pl.read_parquet(path).filter(~pl.col("model").is_in(["Elo", "PoissonGLM"]))
               .sort("RPS_test").row(0, named=True))
        best = (top["arm"], top["model"])
    sel = [best]
    for m in (("B", "GrollRF"), ("B", "StructResid")):
        if m != best:
            sel.append(m)
    return sel


def phase2():
    mf, tmf = _load()
    played = mf.filter(pl.col("is_2026") & pl.col("played")).sort("date", "match_id")
    rows = []
    for kind, n in _selected_models(mf):
        for g in played.iter_rows(named=True):
            train = mf.filter((pl.col("date") < g["date"]) & pl.col("played"))
            pm = mf.filter(pl.col("match_id") == g["match_id"])
            pr = predict(kind, n, {}, train, _tmf_for(tmf, train["match_id"]),
                         pm, _tmf_for(tmf, [g["match_id"]])).row(0, named=True)
            probs = [pr["p_home"], pr["p_draw"], pr["p_away"]]
            pwin = ["H", "D", "A"][int(np.argmax(probs))]
            eh, ea = pr.get("exp_home"), pr.get("exp_away")
            rows.append({
                "model": n, "date": g["date"], "match": f'{g["home_team"]} vs {g["away_team"]}',
                "p_home": round(probs[0], 3), "p_draw": round(probs[1], 3), "p_away": round(probs[2], 3),
                "pred_winner": pwin, "actual_winner": g["result"], "winner_hit": pwin == g["result"],
                "pred_goals": round(eh + ea, 2) if eh is not None else None,
                "actual_goals": g["total_goals"],
                "pred_over25": pr.get("p_over25")})
    sc = pl.DataFrame(rows)
    sc.write_parquet(config.PROCESSED / "wc2026_scorecard.parquet")
    summ = sc.group_by("model").agg(
        games=pl.len(), winner_acc=pl.col("winner_hit").mean().round(3),
        goals_mae=(pl.col("pred_goals") - pl.col("actual_goals")).abs().mean().round(2))
    tex = viz.df_to_neurips_latex(
        summ.sort("winner_acc", descending=True), label="tab:wc2026_scorecard",
        float_format="%.3f", bold_best="winner_acc", lower_is_better=False,
        caption=("2026 World Cup played-game scorecard (out-of-sample, each game predicted as of "
                 "its own date): winner accuracy (higher is better) and total-goals MAE."))
    viz.save_table(tex, "wc2026_scorecard")
    print(summ.sort("winner_acc", descending=True))
    print(sc.filter(pl.col("model") == _selected_models(mf)[0][1]))
    return sc


# --------------------------------------------------------------------------- #
# Phase 3 — production (retrain on all known, predict scheduled fixtures)
# --------------------------------------------------------------------------- #
def phase3(run_date=None):
    run_date = run_date or dt.date.today()
    mf, tmf = _load()
    train = mf.filter(pl.col("played"))                       # all history incl. played WC games
    upcoming = mf.filter(pl.col("is_2026") & ~pl.col("played"))
    if upcoming.height == 0:
        print("no scheduled fixtures to predict.")
        return None
    kind, name = _selected_models(mf)[0]
    cls = predict(kind, name, {}, train, _tmf_for(tmf, train["match_id"]),
                  upcoming, _tmf_for(tmf, upcoming["match_id"]))
    goals = _predict_goals("CatBoostPoisson", {}, _tmf_for(tmf, train["match_id"]),
                           _tmf_for(tmf, upcoming["match_id"]))
    out = (upcoming.select("match_id", "date", "home_team", "away_team")
           .join(cls, on="match_id")
           .join(goals.select("match_id", "exp_home", "exp_away", "p_over25",
                              "ml_home", "ml_away"), on="match_id"))
    idx = np.argmax(out.select("p_home", "p_draw", "p_away").to_numpy(), axis=1)
    out = out.with_columns(pred_winner=pl.Series(np.array(["H", "D", "A"])[idx]),
                           run_date=pl.lit(run_date))
    out.write_parquet(config.PROCESSED / "wc2026_predictions.parquet")
    M.append_prediction_log(config.PROCESSED / "wc2026_prediction_log.parquet", out)
    rnd = ("p_home", "p_draw", "p_away", "exp_home", "exp_away", "p_over25")
    show = out.select("date", "home_team", "away_team", *rnd, "pred_winner").head(20).with_columns(
        [pl.col(c).round(3) for c in rnd])
    tex = viz.df_to_neurips_latex(
        show, label="tab:wc2026_predictions", float_format="%.3f",
        caption=(f"Scheduled-fixture predictions (run {run_date}): P(home/draw/away), expected "
                 "goals and P(over 2.5), retrained on all results known to date."))
    viz.save_table(tex, "wc2026_predictions")
    print(out.select("date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                     "pred_winner", "p_over25").head(12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--trials", type=int, default=0, help="Optuna trials (phase 1 GBDTs)")
    a = ap.parse_args()
    {1: lambda: phase1(a.trials), 2: phase2, 3: phase3}[a.phase]()


if __name__ == "__main__":
    main()
