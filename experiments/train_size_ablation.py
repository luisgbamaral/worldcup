"""Training-size ablation: does RPS change with how much history each model sees?

For growing train windows (1mo / 6mo / 1y / 2y / 5y / 8y) we re-fit every model and
measure performance, in two scopes and two feature sets:

* **one-step** (per-match 1X2): test fixed on 2023+, calibration fixed on 2022, the
  train window ends at 2022-01-01 and grows backwards -> RPS on the held-out test.
* **multi-step** (tournament): per Cup, train on the window before kick-off (frozen),
  re-simulate the bracket (podium points) and also score per-match RPS on that Cup.

Feature sets: ``few`` = top-8 of the greedy order, ``best`` = optimal k* set.
The question: do the tune-free foundation models (TabPFN/TabICL) hold up — or win — in
the small-training regime, where GBDTs usually struggle?

    python experiments/train_size_ablation.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from worldcup import config, elo as E, features, modeling as M, viz  # noqa: E402
import worldcup_multistep as MS  # noqa: E402

WINDOWS = [("1mo", 30), ("6mo", 182), ("1y", 365), ("2y", 730), ("5y", 1826), ("8y", 2922)]
MODELS = ["LogReg", "CatBoost", "XGBoost"]
MS_N_SIMS = 4000                       # lighter than the headline run; podium arg-max is stable
SEED = M.SEED
FEW_K = 8
_y = lambda d: np.where(d["home_score"].to_numpy() > d["away_score"].to_numpy(), 0,
                        np.where(d["home_score"].to_numpy() == d["away_score"].to_numpy(), 1, 2))


def _fms():
    return [m for m in ("TabPFN", "TabICL") if m in M.available_classifiers()]


def _feature_sets():
    opt = json.loads((config.PROCESSED / "optimal_features.json").read_text("utf-8"))
    return {"few": opt["order"][:FEW_K], "best": opt["features"]}


def _fit(name, Xtr, ytr):
    """Fit, capping the foundation models at their 10k-row training limit."""
    if name in ("TabPFN", "TabICL") and len(ytr) > 10000:
        sub = np.random.RandomState(SEED).choice(len(ytr), 10000, replace=False)
        Xtr, ytr = Xtr[sub], ytr[sub]
    return M.build_classifier(name).fit(Xtr, ytr), len(ytr)


# --------------------------------------------------------------------------- #
# one-step: fixed test 2023+, fixed calibration 2022, growing train window
# --------------------------------------------------------------------------- #
def onestep_ablation(mf):
    cut = dt.date(2022, 1, 1)
    va = mf.filter(pl.col("date").dt.year() == 2022)
    te = mf.filter(pl.col("date") >= dt.date(2023, 1, 1))
    yva, yte = _y(va), _y(te)
    rows = []
    for fs, cols in _feature_sets().items():
        med = np.nan_to_num(np.nanmedian(M.to_numpy(mf, cols), axis=0), nan=0.0)
        fill = lambda d: np.where(np.isnan(M.to_numpy(d, cols)), med, M.to_numpy(d, cols))
        Xva, Xte = fill(va), fill(te)
        for wname, days in WINDOWS:
            tr = mf.filter((pl.col("date") >= cut - dt.timedelta(days=days)) & (pl.col("date") < cut))
            if tr.height < 30:
                continue
            Xtr, ytr = fill(tr), _y(tr)
            for name in MODELS + _fms():
                try:
                    pipe, n = _fit(name, Xtr, ytr)
                    cals = M.fit_calibrators(M.proba_hda(pipe, Xva), yva)
                    p = M.apply_calibrators(M.proba_hda(pipe, Xte), cals)
                    r = float(M.rps(p, yte))
                    rows.append({"scope": "onestep", "features": fs, "window": wname, "days": days,
                                 "model": name, "n_train": n, "rps": r,
                                 "acc": float((p.argmax(1) == yte).mean())})
                    print(f"  [1step/{fs}/{wname}] {name:9} n={n:5} RPS={r:.4f}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [skip 1step/{fs}/{wname}] {name}: {type(exc).__name__} {str(exc)[:50]}")
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# multi-step: per Cup, growing pre-kickoff train window -> podium + Cup RPS
# --------------------------------------------------------------------------- #
def multistep_ablation(mf, elo):
    rows = []
    for fs, cols in _feature_sets().items():
        med = {c: float(np.nan_to_num(np.nanmedian(M.to_numpy(mf, [c])))) for c in cols}
        for year in MS.WORLD_CUPS:
            groups, fixtures = MS._groups(year)
            teams = [t for g in groups.values() for t in g]
            start = mf.filter((pl.col("tournament") == "FIFA World Cup")
                              & (pl.col("date").dt.year() == year))["date"].min()
            snap = MS._snapshots(mf, teams, start)
            cup = mf.filter((pl.col("tournament") == "FIFA World Cup")
                            & (pl.col("date").dt.year() == year)).sort("date")
            ycup = _y(cup)
            Xcup = np.nan_to_num(M.to_numpy(cup, cols))
            for wname, days in WINDOWS:
                tr = mf.filter((pl.col("date") >= start - dt.timedelta(days=days)) & (pl.col("date") < start))
                if tr.height < 30:
                    continue
                Xtr, ytr = np.nan_to_num(M.to_numpy(tr, cols)), _y(tr)
                for name in MODELS + _fms():
                    try:
                        pipe, n = _fit(name, Xtr, ytr)
                        P = MS._pair_matrix(pipe, snap, teams, cols, year, med)
                        rng = np.random.default_rng(SEED)
                        c = {"champ": {}, "runner": {}, "third": {}}
                        for _ in range(MS_N_SIMS):
                            ch, ru, th = MS._simulate(P, groups, fixtures, elo, rng)
                            for key, t in (("champ", ch), ("runner", ru), ("third", th)):
                                c[key][t] = c[key].get(t, 0) + 1
                        amax = lambda d, ex=(): max((t for t in d if t not in ex), key=lambda t: d[t], default=None)
                        p1 = amax(c["champ"]); p2 = amax(c["runner"], (p1,)); p3 = amax(c["third"], (p1, p2))
                        act = MS.PODIUM[year]
                        pts = int(p1 == act[0]) + int(p2 == act[1]) + int(p3 == act[2])
                        rps_cup = float(M.rps(M.proba_hda(pipe, Xcup), ycup))
                        rows.append({"scope": "multistep", "features": fs, "window": wname, "days": days,
                                     "wc": year, "model": name, "n_train": n, "pred_champion": p1,
                                     "actual_champion": act[0], "podium_points": pts, "rps_cup": rps_cup})
                        print(f"  [multi/{fs}/{wname}/{year}] {name:9} n={n:5} champ={p1} pts={pts} "
                              f"RPS={rps_cup:.4f}", flush=True)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [skip multi/{fs}/{wname}/{year}] {name}: {type(exc).__name__} {str(exc)[:50]}")
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def _curve_chart(df, yname, ylabel, title, fname, agg="mean"):
    viz.set_style()
    import matplotlib.pyplot as plt
    order = [w for w, _ in WINDOWS]
    fig, axes = plt.subplots(1, 2, figsize=(viz.TEXTWIDTH_IN, viz.TEXTWIDTH_IN * 0.42), sharey=True)
    for ax, fs in zip(axes, ("few", "best")):
        sub = df.filter(pl.col("features") == fs)
        for i, name in enumerate(sorted(sub["model"].unique().to_list())):
            s = (sub.filter(pl.col("model") == name)
                 .group_by("window").agg(getattr(pl.col(yname), agg)().alias("v")))
            s = s.with_columns(pl.col("window").cast(pl.Enum(order)).alias("o")).sort("o")
            ax.plot([order.index(w) for w in s["window"]], s["v"], label=name, **viz.style_for(i))
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=45, fontsize=6)
        ax.set_title(f"{fs} features", fontsize=7); ax.set_xlabel("train window")
    axes[0].set_ylabel(ylabel)
    axes[1].legend(ncol=2, fontsize=5)
    fig.suptitle(title, fontsize=8)
    viz.save_fig(fig, fname)


def main():
    mf = (features.build_match_features_cached(train_cut=dt.date(2000, 1, 1))
          .unique(subset="match_id", keep="first").filter(~pl.col("is_2026")))
    elo = {r["team"]: r["rating"] for r in
           E.latest_ratings().with_columns(MS.canon_team("team").alias("team"))
           .select("team", "rating").iter_rows(named=True)}

    print("=== ONE-STEP ablation ===", flush=True)
    one = onestep_ablation(mf)
    one.write_parquet(config.PROCESSED / "ablation_onestep.parquet")
    one.sort("features", "model", "days").write_csv(config.FIGURES / "ablation_onestep.csv")
    _curve_chart(one, "rps", "test RPS (2023+)", "Train-size ablation — one-step (RPS vs history)",
                 "15_trainsize_onestep")
    print(one.sort("features", "model", "days"))

    print("\n=== MULTI-STEP ablation ===", flush=True)
    multi = multistep_ablation(mf, elo)
    multi.write_parquet(config.PROCESSED / "ablation_multistep.parquet")
    multi.sort("features", "model", "wc", "days").write_csv(config.FIGURES / "ablation_multistep.csv")
    pod = multi.group_by("features", "window", "model").agg(
        podium=pl.col("podium_points").sum(), rps=pl.col("rps_cup").mean())
    _curve_chart(multi, "podium_points", "podium points (sum 5 Cups)",
                 "Train-size ablation — multi-step (podium vs history)", "16_trainsize_multistep_podium", agg="sum")
    _curve_chart(multi, "rps_cup", "Cup-match RPS",
                 "Train-size ablation — multi-step (Cup RPS vs history)", "17_trainsize_multistep_rps")
    print(pod.sort("features", "model", "window"))
    print("\nsaved: ablation_{onestep,multistep}.{parquet,csv}, 15/16/17 charts")


if __name__ == "__main__":
    main()
